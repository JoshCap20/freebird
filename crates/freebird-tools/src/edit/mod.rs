//! Search/replace edit tool for surgical code modifications.
//!
//! The LLM provides `path`, `old_string` (exact text to find), and `new_string`
//! (replacement text). No line numbers — the model specifies literal code blocks,
//! which avoids off-by-one errors that plague diff-based formats.
//!
//! Research shows search/replace format has a 23–27 percentage point improvement
//! over unified diff and line-based diff formats (Meta agentic repair paper).

mod apply;
mod validate;

use std::collections::HashMap;
use std::fmt::Write;
use std::path::Path;
use std::sync::Arc;

use async_trait::async_trait;
use tokio::io::AsyncReadExt;

use crate::edit_history::EditHistory;
use freebird_security::taint::TaintedToolInput;
use freebird_traits::tool::{
    Capability, RiskLevel, SideEffects, Tool, ToolContext, ToolError, ToolInfo, ToolOutcome,
    ToolOutput,
};
use freebird_types::config::{EditConfig, LargeEditAction};

use self::apply::find_and_replace;
use self::validate::{validate_checkpoint_name, validate_syntax};

/// Maximum file size the edit tool will read (10 MiB).
///
/// Matches `read_file`'s limit. Prevents OOM on huge files — the LLM
/// context window can't usefully represent files larger than this anyway.
const MAX_EDIT_FILE_BYTES: usize = 10 * 1024 * 1024;

/// Returns the edit, undo, and checkpoint tools as trait objects.
///
/// All tools share an `Arc<EditHistory>` for session-scoped undo and
/// checkpoint state.
#[must_use]
pub fn edit_tools(config: &EditConfig) -> Vec<Box<dyn Tool>> {
    let history = Arc::new(EditHistory::new());
    vec![
        Box::new(SearchReplaceEditTool::new(config, Arc::clone(&history))),
        Box::new(UndoEditTool::new(Arc::clone(&history))),
        Box::new(CreateCheckpointTool::new(Arc::clone(&history))),
        Box::new(RollbackToCheckpointTool::new(Arc::clone(&history))),
    ]
}

// ── SearchReplaceEditTool ──────────────────────────────────────────

struct SearchReplaceEditTool {
    info: ToolInfo,
    diff_preview: bool,
    diff_context_lines: usize,
    syntax_validation: bool,
    large_edit_threshold: f64,
    large_edit_action: LargeEditAction,
    history: Arc<EditHistory>,
}

impl SearchReplaceEditTool {
    const NAME: &str = "search_replace_edit";

    fn new(config: &EditConfig, history: Arc<EditHistory>) -> Self {
        Self {
            diff_preview: config.diff_preview,
            diff_context_lines: config.diff_context_lines,
            syntax_validation: config.syntax_validation,
            large_edit_threshold: config.large_edit_threshold,
            large_edit_action: config.large_edit_action,
            history,
            info: ToolInfo {
                name: Self::NAME.into(),
                description: "Replace an exact string in a file with new content. \
                    Provide the literal text to find (old_string) and its replacement (new_string). \
                    The old_string must be unique within the file. Use empty new_string to delete text."
                    .into(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Relative path to the file to edit within the sandbox"
                        },
                        "old_string": {
                            "type": "string",
                            "description": "The exact text to find in the file. Must be unique within the file. Include enough surrounding context to ensure uniqueness."
                        },
                        "new_string": {
                            "type": "string",
                            "description": "The text to replace old_string with. Use empty string to delete the matched text."
                        }
                    },
                    "required": ["path", "old_string", "new_string"]
                }),
                required_capability: Capability::FileWrite,
                risk_level: RiskLevel::Medium,
                side_effects: SideEffects::HasSideEffects,
            },
        }
    }
}

#[async_trait]
impl Tool for SearchReplaceEditTool {
    fn info(&self) -> &ToolInfo {
        &self.info
    }

    async fn execute(
        &self,
        input: serde_json::Value,
        context: &ToolContext<'_>,
    ) -> Result<ToolOutput, ToolError> {
        let tainted = TaintedToolInput::new(input);

        // Validate path — file must exist (read variant)
        let safe_path = tainted
            .extract_path_multi_root("path", context.sandbox_root, context.allowed_directories)
            .map_err(|e| ToolError::InvalidInput {
                tool: Self::NAME.into(),
                reason: e.to_string(),
            })?;

        // Extract old_string and new_string via file content bridge
        let old_content =
            tainted
                .extract_file_content("old_string")
                .map_err(|e| ToolError::InvalidInput {
                    tool: Self::NAME.into(),
                    reason: e.to_string(),
                })?;
        let new_content =
            tainted
                .extract_file_content("new_string")
                .map_err(|e| ToolError::InvalidInput {
                    tool: Self::NAME.into(),
                    reason: e.to_string(),
                })?;

        let old_str = old_content.as_str();
        let new_str = new_content.as_str();

        // Reject no-op edits
        if old_str == new_str {
            return Err(ToolError::InvalidInput {
                tool: Self::NAME.into(),
                reason: "old_string and new_string are identical".into(),
            });
        }

        let file_content = read_file_content(safe_path.as_path(), Self::NAME).await?;

        let relative = relative_path_display(&safe_path);
        let result = find_and_replace(&file_content, old_str, new_str, &relative)?;

        // Large edit guardrail — detect edits that change a large fraction of the file.
        let large_edit_metrics =
            compute_large_edit_metrics(&file_content, old_str, new_str, self.large_edit_threshold);
        if let Some(ref metrics) = large_edit_metrics {
            if let Some(output) = check_large_edit_guard(metrics, &relative, self.large_edit_action)
            {
                return Ok(output);
            }
        }

        // Syntax validation before write — original file untouched on failure.
        if self.syntax_validation {
            validate_syntax(safe_path.as_path(), &result.content)?;
        }

        atomic_write(safe_path.as_path(), &result.content, Self::NAME).await?;

        let mut message = format!(
            "Edited {relative}: replaced {} lines starting at line {}",
            result.replaced_lines, result.start_line
        );

        if self.diff_preview {
            let diff = format_diff_preview(
                &file_content,
                &result.matched_text,
                &result.adjusted_new,
                result.start_line,
                result.replaced_lines,
                self.diff_context_lines,
            );
            message.push_str("\n\n");
            message.push_str(&diff);
        }

        if let Some(ref metrics) = large_edit_metrics {
            if matches!(self.large_edit_action, LargeEditAction::Warn) {
                message.push_str(&large_edit_warning_text(metrics));
            }
        }

        // Record pre-edit content for undo support — AFTER successful write,
        // so failed writes don't create phantom undo entries.
        self.history.record_pre_edit(
            context.session_id,
            safe_path.as_path().to_path_buf(),
            file_content,
        );

        Ok(ToolOutput {
            content: message,
            outcome: ToolOutcome::Success,
            metadata: None,
        })
    }
}

/// Read file content with size and encoding validation.
async fn read_file_content(path: &Path, tool_name: &str) -> Result<String, ToolError> {
    let file = tokio::fs::File::open(path)
        .await
        .map_err(|e| ToolError::ExecutionFailed {
            tool: tool_name.into(),
            reason: e.to_string(),
        })?;

    let cap = MAX_EDIT_FILE_BYTES + 1;
    let mut buf = Vec::with_capacity(cap.min(8 * 1024));
    file.take(cap as u64)
        .read_to_end(&mut buf)
        .await
        .map_err(|e| ToolError::ExecutionFailed {
            tool: tool_name.into(),
            reason: e.to_string(),
        })?;

    if buf.len() > MAX_EDIT_FILE_BYTES {
        return Err(ToolError::ExecutionFailed {
            tool: tool_name.into(),
            reason: format!("file exceeds {MAX_EDIT_FILE_BYTES} byte limit"),
        });
    }

    String::from_utf8(buf).map_err(|_| ToolError::ExecutionFailed {
        tool: tool_name.into(),
        reason: "file is not valid UTF-8".into(),
    })
}

/// Write content to a file atomically via temp file + rename.
async fn atomic_write(path: &Path, content: &str, tool_name: &str) -> Result<(), ToolError> {
    let file_name = path.file_name().and_then(|n| n.to_str()).unwrap_or("file");
    let tmp_path = path.with_file_name(format!(".{file_name}.{}.tmp", std::process::id()));

    tokio::fs::write(&tmp_path, content)
        .await
        .map_err(|e| ToolError::ExecutionFailed {
            tool: tool_name.into(),
            reason: e.to_string(),
        })?;

    if let Err(e) = tokio::fs::rename(&tmp_path, path).await {
        let _ = tokio::fs::remove_file(&tmp_path).await;
        return Err(ToolError::ExecutionFailed {
            tool: tool_name.into(),
            reason: e.to_string(),
        });
    }

    Ok(())
}

/// Compute relative path display string, stripping the sandbox root.
fn relative_path_display(safe_path: &freebird_security::safe_types::SafeFilePath) -> String {
    safe_path
        .as_path()
        .strip_prefix(safe_path.root())
        .unwrap_or(safe_path.as_path())
        .display()
        .to_string()
}

/// Computed metrics for a large-edit check.
struct LargeEditMetrics {
    pct: u64,
    threshold_pct: u64,
    old_lines: usize,
    new_lines: usize,
}

/// Compute large-edit metrics if the ratio exceeds the threshold.
///
/// Returns `None` when the edit is under the threshold or the file is empty
/// (avoids division by zero).
fn compute_large_edit_metrics(
    file_content: &str,
    old_str: &str,
    new_str: &str,
    threshold: f64,
) -> Option<LargeEditMetrics> {
    if file_content.is_empty() {
        return None;
    }
    let max_span = old_str.len().max(new_str.len());
    #[expect(
        clippy::cast_precision_loss,
        reason = "line counts are small; f64 mantissa overflow is not a concern"
    )]
    let change_ratio = max_span as f64 / file_content.len() as f64;

    if change_ratio < threshold {
        return None;
    }

    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "value is non-negative"
    )]
    let pct = (change_ratio * 100.0).round() as u64;
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "value is non-negative"
    )]
    let threshold_pct = (threshold * 100.0).round() as u64;

    Some(LargeEditMetrics {
        pct,
        threshold_pct,
        old_lines: old_str.lines().count(),
        new_lines: new_str.lines().count(),
    })
}

/// Evaluate the large-edit guardrail.
///
/// Returns `Some(ToolOutput)` for `Block`/`Consent` when the threshold is
/// exceeded. Returns `None` when the edit is allowed (including `Warn` mode,
/// whose warning text is handled separately via `large_edit_warning_text`).
fn check_large_edit_guard(
    metrics: &LargeEditMetrics,
    relative: &str,
    action: LargeEditAction,
) -> Option<ToolOutput> {
    let LargeEditMetrics {
        pct,
        threshold_pct,
        old_lines,
        new_lines,
    } = *metrics;

    match action {
        LargeEditAction::Block => Some(ToolOutput {
            content: format!(
                "Edit rejected: this edit changes {pct}% of {relative} \
                 ({old_lines} \u{2192} {new_lines} lines), which exceeds the \
                 {threshold_pct}% large-edit threshold. \
                 Break the edit into smaller, targeted replacements."
            ),
            outcome: ToolOutcome::Error,
            metadata: None,
        }),
        LargeEditAction::Consent => Some(ToolOutput {
            content: format!(
                "Edit rejected (requires confirmation): this edit changes \
                 {pct}% of {relative} ({old_lines} \u{2192} {new_lines} lines), \
                 which exceeds the {threshold_pct}% large-edit threshold. \
                 Please break this into smaller edits."
            ),
            outcome: ToolOutcome::Error,
            metadata: None,
        }),
        LargeEditAction::Warn => None,
    }
}

/// Return the warning text for a large edit in `Warn` mode, or `None`.
fn large_edit_warning_text(metrics: &LargeEditMetrics) -> String {
    let LargeEditMetrics {
        pct,
        threshold_pct,
        old_lines,
        new_lines,
    } = *metrics;
    format!(
        "\n\nLarge edit warning: this edit changes {pct}% of the file \
         ({old_lines} \u{2192} {new_lines} lines, threshold: {threshold_pct}%). \
         Consider smaller, targeted edits."
    )
}

/// Format a compact diff preview showing what changed with context lines.
///
/// Produces git-style output: ` ` prefix for context, `-` for removed, `+` for
/// added, with line numbers for orientation.
fn format_diff_preview(
    file_content: &str,
    matched_text: &str,
    adjusted_new: &str,
    start_line: usize,
    replaced_lines: usize,
    context_lines: usize,
) -> String {
    let file_lines: Vec<&str> = file_content.lines().collect();
    let total_lines = file_lines.len();

    let old_lines: Vec<&str> = if matched_text.is_empty() {
        vec![]
    } else {
        matched_text.lines().collect()
    };
    let new_lines: Vec<&str> = if adjusted_new.is_empty() {
        vec![]
    } else {
        adjusted_new.lines().collect()
    };

    // Context window (1-indexed → 0-indexed for vec access)
    let ctx_start = start_line.saturating_sub(context_lines).max(1);
    let change_end = start_line + replaced_lines.saturating_sub(1);
    let ctx_end = (change_end + context_lines).min(total_lines);

    // Line number width for alignment — use the highest line number we might display
    let last_possible = ctx_end
        .max(start_line + new_lines.len().saturating_sub(1))
        .max(1);
    let width = last_possible.to_string().len();

    let mut out = String::new();

    // Context lines before the change
    for line_num in ctx_start..start_line {
        if let Some(text) = file_lines.get(line_num - 1) {
            let _ = writeln!(out, "  {line_num:>width$}\u{2502} {text}");
        }
    }

    // Removed lines
    for (i, line) in old_lines.iter().enumerate() {
        let line_num = start_line + i;
        let _ = writeln!(out, "- {line_num:>width$}\u{2502} {line}");
    }

    // Added lines
    for (i, line) in new_lines.iter().enumerate() {
        let line_num = start_line + i;
        let _ = writeln!(out, "+ {line_num:>width$}\u{2502} {line}");
    }

    // Context lines after the change
    for line_num in (change_end + 1)..=ctx_end {
        if let Some(text) = file_lines.get(line_num - 1) {
            let _ = writeln!(out, "  {line_num:>width$}\u{2502} {text}");
        }
    }

    // Remove trailing newline
    if out.ends_with('\n') {
        out.pop();
    }

    out
}

// ── UndoEditTool ────────────────────────────────────────────────

struct UndoEditTool {
    info: ToolInfo,
    history: Arc<EditHistory>,
}

impl UndoEditTool {
    const NAME: &str = "undo_edit";

    fn new(history: Arc<EditHistory>) -> Self {
        Self {
            history,
            info: ToolInfo {
                name: Self::NAME.into(),
                description: "Undo the last edit made to a file by search_replace_edit. \
                    Restores the file to its state before the most recent edit. \
                    Can be called multiple times to undo up to 10 edits per file."
                    .into(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path to the file to undo the last edit on"
                        }
                    },
                    "required": ["path"]
                }),
                required_capability: Capability::FileWrite,
                risk_level: RiskLevel::Medium,
                side_effects: SideEffects::HasSideEffects,
            },
        }
    }
}

#[async_trait]
impl Tool for UndoEditTool {
    fn info(&self) -> &ToolInfo {
        &self.info
    }

    async fn execute(
        &self,
        input: serde_json::Value,
        context: &ToolContext<'_>,
    ) -> Result<ToolOutput, ToolError> {
        let tainted = TaintedToolInput::new(input);
        let safe_path = tainted
            .extract_path_multi_root("path", context.sandbox_root, context.allowed_directories)
            .map_err(|e| ToolError::InvalidInput {
                tool: Self::NAME.into(),
                reason: e.to_string(),
            })?;

        let canonical = safe_path.as_path().to_path_buf();
        let previous = self
            .history
            .pop_last_version(context.session_id, &canonical)
            .ok_or_else(|| ToolError::ExecutionFailed {
                tool: Self::NAME.into(),
                reason: format!("no edit history for {}", relative_path_display(&safe_path)),
            })?;

        atomic_write(safe_path.as_path(), &previous, Self::NAME).await?;

        let remaining = self.history.version_count(context.session_id, &canonical);
        let relative = relative_path_display(&safe_path);
        Ok(ToolOutput {
            content: format!(
                "Restored {relative} to previous version ({remaining} undo steps remaining)"
            ),
            outcome: ToolOutcome::Success,
            metadata: None,
        })
    }
}

// ── CreateCheckpointTool ────────────────────────────────────────

struct CreateCheckpointTool {
    info: ToolInfo,
    history: Arc<EditHistory>,
}

impl CreateCheckpointTool {
    const NAME: &str = "create_checkpoint";

    fn new(history: Arc<EditHistory>) -> Self {
        Self {
            history,
            info: ToolInfo {
                name: Self::NAME.into(),
                description: "Create a named checkpoint that snapshots all files modified \
                    by search_replace_edit in this session. Use rollback_to_checkpoint \
                    to restore files to this state later. Max 5 checkpoints per session; \
                    checkpoints expire after 1 hour."
                    .into(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Checkpoint name (e.g., 'before-refactor'). Alphanumeric, hyphens, underscores, 1–64 chars."
                        }
                    },
                    "required": ["name"]
                }),
                required_capability: Capability::FileRead,
                risk_level: RiskLevel::Low,
                side_effects: SideEffects::None,
            },
        }
    }
}

#[async_trait]
impl Tool for CreateCheckpointTool {
    fn info(&self) -> &ToolInfo {
        &self.info
    }

    async fn execute(
        &self,
        input: serde_json::Value,
        context: &ToolContext<'_>,
    ) -> Result<ToolOutput, ToolError> {
        let tainted = TaintedToolInput::new(input);

        let name_content =
            tainted
                .extract_file_content("name")
                .map_err(|e| ToolError::InvalidInput {
                    tool: Self::NAME.into(),
                    reason: e.to_string(),
                })?;
        let name = name_content.as_str();
        validate_checkpoint_name(name, Self::NAME)?;

        let modified = self.history.modified_files(context.session_id);
        if modified.is_empty() {
            return Err(ToolError::ExecutionFailed {
                tool: Self::NAME.into(),
                reason: "no files have been modified in this session".into(),
            });
        }

        // Read current content of each modified file
        let mut files = HashMap::new();
        for file_path in &modified {
            match tokio::fs::read_to_string(file_path).await {
                Ok(data) => {
                    files.insert(file_path.clone(), data);
                }
                // File was deleted since edit — skip
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                Err(e) => {
                    return Err(ToolError::ExecutionFailed {
                        tool: Self::NAME.into(),
                        reason: format!("failed to read {}: {e}", file_path.display()),
                    });
                }
            }
        }

        let file_list: Vec<String> = files.keys().map(|p| p.display().to_string()).collect();

        self.history
            .create_checkpoint(context.session_id, name.to_string(), files)
            .map_err(|reason| ToolError::ExecutionFailed {
                tool: Self::NAME.into(),
                reason: reason.into(),
            })?;
        Ok(ToolOutput {
            content: format!(
                "Checkpoint '{}' created with {} files: {}",
                name,
                file_list.len(),
                file_list.join(", ")
            ),
            outcome: ToolOutcome::Success,
            metadata: None,
        })
    }
}

// ── RollbackToCheckpointTool ────────────────────────────────────

struct RollbackToCheckpointTool {
    info: ToolInfo,
    history: Arc<EditHistory>,
}

impl RollbackToCheckpointTool {
    const NAME: &str = "rollback_to_checkpoint";

    fn new(history: Arc<EditHistory>) -> Self {
        Self {
            history,
            info: ToolInfo {
                name: Self::NAME.into(),
                description: "Restore all files to the state captured by a named checkpoint. \
                    The checkpoint is consumed (removed) after rollback. Continues restoring \
                    remaining files on partial failure."
                    .into(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Name of the checkpoint to restore"
                        }
                    },
                    "required": ["name"]
                }),
                required_capability: Capability::FileWrite,
                risk_level: RiskLevel::High,
                side_effects: SideEffects::HasSideEffects,
            },
        }
    }
}

#[async_trait]
impl Tool for RollbackToCheckpointTool {
    fn info(&self) -> &ToolInfo {
        &self.info
    }

    async fn execute(
        &self,
        input: serde_json::Value,
        context: &ToolContext<'_>,
    ) -> Result<ToolOutput, ToolError> {
        let tainted = TaintedToolInput::new(input);

        let name_content =
            tainted
                .extract_file_content("name")
                .map_err(|e| ToolError::InvalidInput {
                    tool: Self::NAME.into(),
                    reason: e.to_string(),
                })?;
        let name = name_content.as_str();
        validate_checkpoint_name(name, Self::NAME)?;

        let files = self
            .history
            .take_checkpoint(context.session_id, name)
            .map_err(|reason| ToolError::ExecutionFailed {
                tool: Self::NAME.into(),
                reason: reason.into(),
            })?;

        let mut restored = Vec::with_capacity(files.len());
        let mut failed = Vec::new();

        for (path, content) in &files {
            match atomic_write(path, content, Self::NAME).await {
                Ok(()) => restored.push(path.display().to_string()),
                Err(e) => failed.push(format!("{}: {e}", path.display())),
            }
        }

        if restored.is_empty() && !failed.is_empty() {
            return Err(ToolError::ExecutionFailed {
                tool: Self::NAME.into(),
                reason: format!("rollback failed for all files: {}", failed.join("; ")),
            });
        }

        let mut message = format!(
            "Rolled back to checkpoint '{}': restored {} files",
            name,
            restored.len()
        );
        if !failed.is_empty() {
            let _ = write!(message, " (failed: {})", failed.join("; "));
        }

        Ok(ToolOutput {
            content: message,
            outcome: ToolOutcome::Success,
            metadata: None,
        })
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic, clippy::indexing_slicing)]
mod tests {
    use freebird_traits::tool::{Capability, RiskLevel, SideEffects, Tool, ToolError};

    use super::*;
    use crate::test_utils::TestHarness;

    /// Config for tests that don't exercise syntax validation.
    /// Validation is off so edits to Rust snippets don't need to be full programs.
    fn test_config() -> EditConfig {
        EditConfig {
            syntax_validation: false,
            ..EditConfig::default()
        }
    }

    // ── Factory test ─────────────────────────────────────────────

    #[test]
    fn test_edit_tool_info() {
        let tool = SearchReplaceEditTool::new(&test_config(), Arc::new(EditHistory::new()));
        let info = tool.info();
        assert_eq!(info.name, "search_replace_edit");
        assert_eq!(info.required_capability, Capability::FileWrite);
        assert_eq!(info.risk_level, RiskLevel::Medium);
        assert!(matches!(info.side_effects, SideEffects::HasSideEffects));
    }

    // ── Core edit tests ──────────────────────────────────────────

    #[tokio::test]
    async fn test_exact_match_replaces() {
        let h = TestHarness::with_capabilities(vec![Capability::FileRead, Capability::FileWrite]);
        std::fs::write(
            h.path().join("file.rs"),
            "fn hello() {\n    println!(\"hi\");\n}\n",
        )
        .unwrap();

        let tool = SearchReplaceEditTool::new(&test_config(), Arc::new(EditHistory::new()));
        let output = tool
            .execute(
                serde_json::json!({
                    "path": "file.rs",
                    "old_string": "println!(\"hi\");",
                    "new_string": "println!(\"hello world\");"
                }),
                &h.context(),
            )
            .await
            .unwrap();

        assert!(matches!(output.outcome, ToolOutcome::Success));
        let content = std::fs::read_to_string(h.path().join("file.rs")).unwrap();
        assert!(content.contains("println!(\"hello world\");"));
        assert!(!content.contains("println!(\"hi\");"));
    }

    #[tokio::test]
    async fn test_no_match_returns_error() {
        let h = TestHarness::with_capabilities(vec![Capability::FileRead, Capability::FileWrite]);
        std::fs::write(h.path().join("file.rs"), "fn main() {}\n").unwrap();

        let tool = SearchReplaceEditTool::new(&test_config(), Arc::new(EditHistory::new()));
        let err = tool
            .execute(
                serde_json::json!({
                    "path": "file.rs",
                    "old_string": "nonexistent text",
                    "new_string": "replacement"
                }),
                &h.context(),
            )
            .await
            .unwrap_err();

        match err {
            ToolError::ExecutionFailed { reason, .. } => {
                assert!(reason.contains("not found"), "error: {reason}");
            }
            other => panic!("expected ExecutionFailed, got: {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_multiple_matches_returns_error() {
        let h = TestHarness::with_capabilities(vec![Capability::FileRead, Capability::FileWrite]);
        std::fs::write(
            h.path().join("file.rs"),
            "let x = 1;\nlet y = 1;\nlet z = 1;\n",
        )
        .unwrap();

        let tool = SearchReplaceEditTool::new(&test_config(), Arc::new(EditHistory::new()));
        let err = tool
            .execute(
                serde_json::json!({
                    "path": "file.rs",
                    "old_string": "= 1;",
                    "new_string": "= 2;"
                }),
                &h.context(),
            )
            .await
            .unwrap_err();

        match err {
            ToolError::ExecutionFailed { reason, .. } => {
                assert!(reason.contains("3 locations"), "error: {reason}");
            }
            other => panic!("expected ExecutionFailed, got: {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_empty_new_string_deletes() {
        let h = TestHarness::with_capabilities(vec![Capability::FileRead, Capability::FileWrite]);
        std::fs::write(h.path().join("file.rs"), "line1\nDELETE_ME\nline3\n").unwrap();

        let tool = SearchReplaceEditTool::new(&test_config(), Arc::new(EditHistory::new()));
        let output = tool
            .execute(
                serde_json::json!({
                    "path": "file.rs",
                    "old_string": "DELETE_ME\n",
                    "new_string": ""
                }),
                &h.context(),
            )
            .await
            .unwrap();

        assert!(matches!(output.outcome, ToolOutcome::Success));
        let content = std::fs::read_to_string(h.path().join("file.rs")).unwrap();
        assert_eq!(content, "line1\nline3\n");
    }

    #[tokio::test]
    async fn test_identical_strings_returns_error() {
        let h = TestHarness::with_capabilities(vec![Capability::FileRead, Capability::FileWrite]);
        std::fs::write(h.path().join("file.rs"), "content\n").unwrap();

        let tool = SearchReplaceEditTool::new(&test_config(), Arc::new(EditHistory::new()));
        let err = tool
            .execute(
                serde_json::json!({
                    "path": "file.rs",
                    "old_string": "content",
                    "new_string": "content"
                }),
                &h.context(),
            )
            .await
            .unwrap_err();

        match err {
            ToolError::InvalidInput { reason, .. } => {
                assert!(reason.contains("identical"), "error: {reason}");
            }
            other => panic!("expected InvalidInput, got: {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_multiline_replace() {
        let h = TestHarness::with_capabilities(vec![Capability::FileRead, Capability::FileWrite]);
        let original = "fn main() {\n    let a = 1;\n    let b = 2;\n    let c = 3;\n    let d = 4;\n    let e = 5;\n}\n";
        std::fs::write(h.path().join("file.rs"), original).unwrap();

        let tool = SearchReplaceEditTool::new(&test_config(), Arc::new(EditHistory::new()));
        let output = tool
            .execute(
                serde_json::json!({
                    "path": "file.rs",
                    "old_string": "    let a = 1;\n    let b = 2;\n    let c = 3;\n    let d = 4;\n    let e = 5;",
                    "new_string": "    let sum = 15;\n    let count = 5;\n    let avg = 3;"
                }),
                &h.context(),
            )
            .await
            .unwrap();

        assert!(matches!(output.outcome, ToolOutcome::Success));
        let content = std::fs::read_to_string(h.path().join("file.rs")).unwrap();
        assert!(content.contains("let sum = 15;"));
        assert!(content.contains("let count = 5;"));
        assert!(content.contains("let avg = 3;"));
        assert!(!content.contains("let a = 1;"));
    }

    #[tokio::test]
    async fn test_replace_preserves_surrounding_content() {
        let h = TestHarness::with_capabilities(vec![Capability::FileRead, Capability::FileWrite]);
        let original = "BEFORE\nTARGET\nAFTER\n";
        std::fs::write(h.path().join("file.txt"), original).unwrap();

        let tool = SearchReplaceEditTool::new(&test_config(), Arc::new(EditHistory::new()));
        tool.execute(
            serde_json::json!({
                "path": "file.txt",
                "old_string": "TARGET",
                "new_string": "REPLACED"
            }),
            &h.context(),
        )
        .await
        .unwrap();

        let content = std::fs::read_to_string(h.path().join("file.txt")).unwrap();
        assert!(content.starts_with("BEFORE\n"));
        assert!(content.ends_with("AFTER\n"));
        assert!(content.contains("REPLACED"));
    }

    // ── Whitespace normalization tests ───────────────────────────

    #[tokio::test]
    async fn test_whitespace_normalized_fallback() {
        let h = TestHarness::with_capabilities(vec![Capability::FileRead, Capability::FileWrite]);
        // File has single spaces
        std::fs::write(h.path().join("file.rs"), "fn main() {\n    let x = 1;\n}\n").unwrap();

        let tool = SearchReplaceEditTool::new(&test_config(), Arc::new(EditHistory::new()));
        // old_string has extra spaces — exact match fails, normalized match succeeds
        let output = tool
            .execute(
                serde_json::json!({
                    "path": "file.rs",
                    "old_string": "let  x  =  1;",
                    "new_string": "let x = 2;"
                }),
                &h.context(),
            )
            .await
            .unwrap();

        assert!(matches!(output.outcome, ToolOutcome::Success));
        let content = std::fs::read_to_string(h.path().join("file.rs")).unwrap();
        assert!(content.contains("let x = 2;"));
    }

    #[tokio::test]
    async fn test_indentation_mismatch_handled() {
        let h = TestHarness::with_capabilities(vec![Capability::FileRead, Capability::FileWrite]);
        // File has 2-space indent
        std::fs::write(h.path().join("file.rs"), "fn main() {\n  let x = 1;\n}\n").unwrap();

        let tool = SearchReplaceEditTool::new(&test_config(), Arc::new(EditHistory::new()));
        // old_string has 4-space indent — normalized match should work
        let output = tool
            .execute(
                serde_json::json!({
                    "path": "file.rs",
                    "old_string": "    let x = 1;",
                    "new_string": "    let x = 2;"
                }),
                &h.context(),
            )
            .await
            .unwrap();

        assert!(matches!(output.outcome, ToolOutcome::Success));
        let content = std::fs::read_to_string(h.path().join("file.rs")).unwrap();
        // Indentation should be preserved from the original file (2-space)
        assert!(content.contains("  let x = 2;"), "content: {content}");
    }

    #[tokio::test]
    async fn test_normalized_multiple_matches_returns_error() {
        let h = TestHarness::with_capabilities(vec![Capability::FileRead, Capability::FileWrite]);
        std::fs::write(
            h.path().join("file.rs"),
            "let x = 1;\nlet  x  =  1;\nlet x=1;\n",
        )
        .unwrap();

        let tool = SearchReplaceEditTool::new(&test_config(), Arc::new(EditHistory::new()));
        // All three lines normalize to "let x = 1;"
        let err = tool
            .execute(
                serde_json::json!({
                    "path": "file.rs",
                    "old_string": "let   x   =   1;",
                    "new_string": "let x = 2;"
                }),
                &h.context(),
            )
            .await
            .unwrap_err();

        match err {
            ToolError::ExecutionFailed { reason, .. } => {
                assert!(
                    reason.contains("locations") && reason.contains("normalization"),
                    "error: {reason}"
                );
            }
            other => panic!("expected ExecutionFailed, got: {other:?}"),
        }
    }

    // ── Indentation preservation tests ───────────────────────────

    #[tokio::test]
    async fn test_indentation_preserved_on_replace() {
        let h = TestHarness::with_capabilities(vec![Capability::FileRead, Capability::FileWrite]);
        let original = "fn main() {\n    if true {\n        let x = 1;\n    }\n}\n";
        std::fs::write(h.path().join("file.rs"), original).unwrap();

        let tool = SearchReplaceEditTool::new(&test_config(), Arc::new(EditHistory::new()));
        let output = tool
            .execute(
                serde_json::json!({
                    "path": "file.rs",
                    "old_string": "        let x = 1;",
                    "new_string": "        let x = 2;\n        let y = 3;"
                }),
                &h.context(),
            )
            .await
            .unwrap();

        assert!(matches!(output.outcome, ToolOutcome::Success));
        let content = std::fs::read_to_string(h.path().join("file.rs")).unwrap();
        assert!(content.contains("        let x = 2;\n        let y = 3;"));
    }

    #[tokio::test]
    async fn test_indentation_delta_applied_to_all_lines() {
        let h = TestHarness::with_capabilities(vec![Capability::FileRead, Capability::FileWrite]);
        // File has 8-space indented block
        let original =
            "fn main() {\n    if true {\n        old_line_1\n        old_line_2\n    }\n}\n";
        std::fs::write(h.path().join("file.rs"), original).unwrap();

        let tool = SearchReplaceEditTool::new(&test_config(), Arc::new(EditHistory::new()));
        // new_string has no indentation — indentation preservation should add 8 spaces
        let output = tool
            .execute(
                serde_json::json!({
                    "path": "file.rs",
                    "old_string": "        old_line_1\n        old_line_2",
                    "new_string": "new_line_1\nnew_line_2\nnew_line_3"
                }),
                &h.context(),
            )
            .await
            .unwrap();

        assert!(matches!(output.outcome, ToolOutcome::Success));
        let content = std::fs::read_to_string(h.path().join("file.rs")).unwrap();
        // All lines should have 8-space indentation
        assert!(
            content.contains("        new_line_1\n        new_line_2\n        new_line_3"),
            "content: {content}"
        );
    }

    // ── Security tests ───────────────────────────────────────────

    #[tokio::test]
    async fn test_path_traversal_rejected() {
        let h = TestHarness::with_capabilities(vec![Capability::FileRead, Capability::FileWrite]);
        let tool = SearchReplaceEditTool::new(&test_config(), Arc::new(EditHistory::new()));

        let err = tool
            .execute(
                serde_json::json!({
                    "path": "../../etc/passwd",
                    "old_string": "root",
                    "new_string": "hacked"
                }),
                &h.context(),
            )
            .await
            .unwrap_err();

        match err {
            ToolError::InvalidInput { tool, .. } => assert_eq!(tool, "search_replace_edit"),
            other => panic!("expected InvalidInput, got: {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_nonexistent_file_returns_error() {
        let h = TestHarness::with_capabilities(vec![Capability::FileRead, Capability::FileWrite]);
        let tool = SearchReplaceEditTool::new(&test_config(), Arc::new(EditHistory::new()));

        let err = tool
            .execute(
                serde_json::json!({
                    "path": "nope.rs",
                    "old_string": "x",
                    "new_string": "y"
                }),
                &h.context(),
            )
            .await
            .unwrap_err();

        match err {
            ToolError::InvalidInput { tool, .. } => assert_eq!(tool, "search_replace_edit"),
            other => panic!("expected InvalidInput, got: {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_output_reports_relative_path() {
        let h = TestHarness::with_capabilities(vec![Capability::FileRead, Capability::FileWrite]);
        std::fs::write(h.path().join("src.rs"), "old\n").unwrap();

        let tool = SearchReplaceEditTool::new(&test_config(), Arc::new(EditHistory::new()));
        let output = tool
            .execute(
                serde_json::json!({
                    "path": "src.rs",
                    "old_string": "old",
                    "new_string": "new"
                }),
                &h.context(),
            )
            .await
            .unwrap();

        assert!(
            output.content.contains("src.rs"),
            "output: {}",
            output.content
        );
        let sandbox_str = h.path().to_string_lossy();
        assert!(
            !output.content.contains(sandbox_str.as_ref()),
            "output leaked sandbox root: {}",
            output.content
        );
    }

    // ── Atomic write tests ───────────────────────────────────────

    #[tokio::test]
    async fn test_no_orphaned_temp_on_success() {
        let h = TestHarness::with_capabilities(vec![Capability::FileRead, Capability::FileWrite]);
        std::fs::write(h.path().join("clean.rs"), "old_text\n").unwrap();

        let tool = SearchReplaceEditTool::new(&test_config(), Arc::new(EditHistory::new()));
        tool.execute(
            serde_json::json!({
                "path": "clean.rs",
                "old_string": "old_text",
                "new_string": "new_text"
            }),
            &h.context(),
        )
        .await
        .unwrap();

        let entries: Vec<_> = std::fs::read_dir(h.path())
            .unwrap()
            .filter_map(Result::ok)
            .filter(|e| e.file_name().to_string_lossy().ends_with(".tmp"))
            .collect();
        assert!(
            entries.is_empty(),
            "no .tmp files should remain after successful edit"
        );
    }

    #[tokio::test]
    async fn test_original_unchanged_on_match_failure() {
        let h = TestHarness::with_capabilities(vec![Capability::FileRead, Capability::FileWrite]);
        let original = "unchanged content\n";
        std::fs::write(h.path().join("file.rs"), original).unwrap();

        let tool = SearchReplaceEditTool::new(&test_config(), Arc::new(EditHistory::new()));
        let _ = tool
            .execute(
                serde_json::json!({
                    "path": "file.rs",
                    "old_string": "nonexistent",
                    "new_string": "replacement"
                }),
                &h.context(),
            )
            .await;

        let content = std::fs::read_to_string(h.path().join("file.rs")).unwrap();
        assert_eq!(content, original);
    }

    // ── Edge case tests ──────────────────────────────────────────

    #[tokio::test]
    async fn test_crlf_line_endings_normalized_fallback() {
        let h = TestHarness::with_capabilities(vec![Capability::FileRead, Capability::FileWrite]);
        // Write a file with \r\n line endings
        std::fs::write(
            h.path().join("crlf.rs"),
            "fn main() {\r\n    let x = 1;\r\n}\r\n",
        )
        .unwrap();

        let tool = SearchReplaceEditTool::new(&test_config(), Arc::new(EditHistory::new()));
        // Use normalized fallback (extra spaces)
        let output = tool
            .execute(
                serde_json::json!({
                    "path": "crlf.rs",
                    "old_string": "let  x  =  1;",
                    "new_string": "let x = 2;"
                }),
                &h.context(),
            )
            .await
            .unwrap();

        assert!(matches!(output.outcome, ToolOutcome::Success));
        let content = std::fs::read_to_string(h.path().join("crlf.rs")).unwrap();
        assert!(content.contains("let x = 2;"), "content: {content:?}");
        // Verify surrounding content isn't corrupted
        assert!(content.contains("fn main()"), "content: {content:?}");
        assert!(content.contains('}'), "content: {content:?}");
    }

    #[tokio::test]
    async fn test_empty_old_string_returns_error() {
        let h = TestHarness::with_capabilities(vec![Capability::FileRead, Capability::FileWrite]);
        std::fs::write(h.path().join("file.rs"), "content\n").unwrap();

        let tool = SearchReplaceEditTool::new(&test_config(), Arc::new(EditHistory::new()));
        let err = tool
            .execute(
                serde_json::json!({
                    "path": "file.rs",
                    "old_string": "",
                    "new_string": "inserted"
                }),
                &h.context(),
            )
            .await
            .unwrap_err();

        // Empty string matches at every position, so >1 matches → error
        match err {
            ToolError::ExecutionFailed { reason, .. } => {
                assert!(reason.contains("locations"), "error: {reason}");
            }
            other => panic!("expected ExecutionFailed, got: {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_edit_tools_factory() {
        let tools = edit_tools(&test_config());
        assert_eq!(tools.len(), 4);
        assert_eq!(tools[0].info().name, "search_replace_edit");
        assert_eq!(tools[1].info().name, "undo_edit");
        assert_eq!(tools[2].info().name, "create_checkpoint");
        assert_eq!(tools[3].info().name, "rollback_to_checkpoint");

        // Verify to_definition() produces a valid tool definition
        for tool in &tools {
            let def = tool.to_definition();
            assert!(!def.name.is_empty());
            assert!(!def.description.is_empty());
            assert!(def.input_schema.is_object());
        }
    }

    // ── File size limit tests ────────────────────────────────────

    #[tokio::test]
    async fn test_file_exceeding_size_limit_rejected() {
        let h = TestHarness::with_capabilities(vec![Capability::FileRead, Capability::FileWrite]);
        let file_path = h.path().join("huge.rs");
        {
            let f = std::fs::File::create(&file_path).unwrap();
            // Set file size to just over 10 MiB without writing all bytes
            f.set_len((MAX_EDIT_FILE_BYTES + 1) as u64).unwrap();
        }

        let tool = SearchReplaceEditTool::new(&test_config(), Arc::new(EditHistory::new()));
        let err = tool
            .execute(
                serde_json::json!({
                    "path": "huge.rs",
                    "old_string": "x",
                    "new_string": "y"
                }),
                &h.context(),
            )
            .await
            .unwrap_err();

        match &err {
            ToolError::ExecutionFailed { reason, .. } => {
                assert!(
                    reason.contains("exceeds"),
                    "error should mention limit: {reason}"
                );
            }
            other => panic!("expected ExecutionFailed, got: {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_non_utf8_file_rejected() {
        use std::io::Write as _;
        let h = TestHarness::with_capabilities(vec![Capability::FileRead, Capability::FileWrite]);
        let file_path = h.path().join("binary.bin");
        {
            let mut f = std::fs::File::create(&file_path).unwrap();
            f.write_all(&[0xFF, 0xFE, 0x00, 0x01]).unwrap();
        }

        let tool = SearchReplaceEditTool::new(&test_config(), Arc::new(EditHistory::new()));
        let err = tool
            .execute(
                serde_json::json!({
                    "path": "binary.bin",
                    "old_string": "x",
                    "new_string": "y"
                }),
                &h.context(),
            )
            .await
            .unwrap_err();

        match &err {
            ToolError::ExecutionFailed { reason, .. } => {
                assert!(
                    reason.contains("UTF-8"),
                    "error should mention UTF-8: {reason}"
                );
            }
            other => panic!("expected ExecutionFailed, got: {other:?}"),
        }
    }

    // ── Tab indentation tests ────────────────────────────────────

    #[tokio::test]
    async fn test_tab_indentation_preserved() {
        let h = TestHarness::with_capabilities(vec![Capability::FileRead, Capability::FileWrite]);
        // File uses tab indentation
        std::fs::write(
            h.path().join("file.go"),
            "func main() {\n\toldLine1\n\toldLine2\n}\n",
        )
        .unwrap();

        let tool = SearchReplaceEditTool::new(&test_config(), Arc::new(EditHistory::new()));
        let output = tool
            .execute(
                serde_json::json!({
                    "path": "file.go",
                    "old_string": "\toldLine1\n\toldLine2",
                    "new_string": "newLine1\nnewLine2\nnewLine3"
                }),
                &h.context(),
            )
            .await
            .unwrap();

        assert!(matches!(output.outcome, ToolOutcome::Success));
        let content = std::fs::read_to_string(h.path().join("file.go")).unwrap();
        // All replacement lines should have tab indentation, not spaces
        assert!(
            content.contains("\tnewLine1\n\tnewLine2\n\tnewLine3"),
            "content: {content:?}"
        );
    }

    // ── Fuzzy matching tests ─────────────────────────────────────

    #[tokio::test]
    async fn test_fuzzy_match_one_line_wrong() {
        let h = TestHarness::with_capabilities(vec![Capability::FileRead, Capability::FileWrite]);
        let original =
            "fn main() {\n    let x = 1;\n    let y = 2;\n    let z = 3;\n    let w = 4;\n}\n";
        std::fs::write(h.path().join("file.rs"), original).unwrap();

        let tool = SearchReplaceEditTool::new(&test_config(), Arc::new(EditHistory::new()));
        // old_string has one line wrong ("let y = 999" instead of "let y = 2")
        // 4 out of 5 lines match (80%) — should fuzzy match
        let output = tool
            .execute(
                serde_json::json!({
                    "path": "file.rs",
                    "old_string": "    let x = 1;\n    let y = 999;\n    let z = 3;\n    let w = 4;",
                    "new_string": "    let sum = 10;"
                }),
                &h.context(),
            )
            .await
            .unwrap();

        assert!(matches!(output.outcome, ToolOutcome::Success));
        let content = std::fs::read_to_string(h.path().join("file.rs")).unwrap();
        assert!(content.contains("let sum = 10;"));
        assert!(!content.contains("let x = 1;"));
    }

    #[tokio::test]
    async fn test_fuzzy_match_too_many_lines_wrong_fails() {
        let h = TestHarness::with_capabilities(vec![Capability::FileRead, Capability::FileWrite]);
        let original = "fn main() {\n    let x = 1;\n    let y = 2;\n    let z = 3;\n}\n";
        std::fs::write(h.path().join("file.rs"), original).unwrap();

        let tool = SearchReplaceEditTool::new(&test_config(), Arc::new(EditHistory::new()));
        // old_string has 2 out of 3 lines wrong — below 60% threshold
        let err = tool
            .execute(
                serde_json::json!({
                    "path": "file.rs",
                    "old_string": "    let a = 99;\n    let b = 88;\n    let z = 3;",
                    "new_string": "    let sum = 6;"
                }),
                &h.context(),
            )
            .await
            .unwrap_err();

        match err {
            ToolError::ExecutionFailed { reason, .. } => {
                assert!(reason.contains("not found"), "error: {reason}");
            }
            other => panic!("expected ExecutionFailed, got: {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_fuzzy_match_ambiguous_returns_error() {
        let h = TestHarness::with_capabilities(vec![Capability::FileRead, Capability::FileWrite]);
        // Two identical blocks — fuzzy match should be ambiguous
        let original = "fn a() {\n    let x = 1;\n    let y = 2;\n}\nfn b() {\n    let x = 1;\n    let y = 2;\n}\n";
        std::fs::write(h.path().join("file.rs"), original).unwrap();

        let tool = SearchReplaceEditTool::new(&test_config(), Arc::new(EditHistory::new()));
        // Fuzzy search with one wrong line — matches both blocks equally
        let err = tool
            .execute(
                serde_json::json!({
                    "path": "file.rs",
                    "old_string": "    let x = 1;\n    let y = 999;",
                    "new_string": "    let z = 3;"
                }),
                &h.context(),
            )
            .await
            .unwrap_err();

        match err {
            ToolError::ExecutionFailed { reason, .. } => {
                assert!(reason.contains("not found"), "error: {reason}");
            }
            other => panic!("expected ExecutionFailed, got: {other:?}"),
        }
    }

    // ── Diff preview tests ────────────────────────────────────────

    #[tokio::test]
    async fn test_diff_preview_in_success_output() {
        let h = TestHarness::with_capabilities(vec![Capability::FileRead, Capability::FileWrite]);
        std::fs::write(
            h.path().join("src.rs"),
            "aaa\nbbb\nccc\nddd\neee\nfff\nggg\n",
        )
        .unwrap();

        let tool = SearchReplaceEditTool::new(&test_config(), Arc::new(EditHistory::new()));
        let output = tool
            .execute(
                serde_json::json!({
                    "path": "src.rs",
                    "old_string": "ddd",
                    "new_string": "DDD"
                }),
                &h.context(),
            )
            .await
            .unwrap();

        // Output must contain removed and added lines
        assert!(output.content.contains("- "), "missing removed line marker");
        assert!(output.content.contains("+ "), "missing added line marker");
        assert!(
            output.content.contains("\u{2502} ddd"),
            "missing removed line content"
        );
        assert!(
            output.content.contains("\u{2502} DDD"),
            "missing added line content"
        );
    }

    #[tokio::test]
    async fn test_context_lines_around_diff() {
        let h = TestHarness::with_capabilities(vec![Capability::FileRead, Capability::FileWrite]);
        std::fs::write(
            h.path().join("src.rs"),
            "line1\nline2\nline3\nline4\nline5\nline6\nline7\nline8\nline9\n",
        )
        .unwrap();

        let tool = SearchReplaceEditTool::new(
            &EditConfig {
                diff_preview: true,
                diff_context_lines: 3,
                syntax_validation: false,
                ..EditConfig::default()
            },
            Arc::new(EditHistory::new()),
        );
        let output = tool
            .execute(
                serde_json::json!({
                    "path": "src.rs",
                    "old_string": "line5",
                    "new_string": "LINE5"
                }),
                &h.context(),
            )
            .await
            .unwrap();

        // 3 context lines before (line2, line3, line4) and after (line6, line7, line8)
        assert!(output.content.contains(" line2"), "missing context line2");
        assert!(output.content.contains(" line3"), "missing context line3");
        assert!(output.content.contains(" line4"), "missing context line4");
        assert!(output.content.contains(" line6"), "missing context line6");
        assert!(output.content.contains(" line7"), "missing context line7");
        assert!(output.content.contains(" line8"), "missing context line8");
        // line1 is 4 lines away — should NOT be shown with 3 context lines
        assert!(
            !output.content.contains(" line1"),
            "line1 should not appear with context=3"
        );
        assert!(
            !output.content.contains(" line9"),
            "line9 should not appear with context=3"
        );
    }

    #[tokio::test]
    async fn test_multiline_diff() {
        let h = TestHarness::with_capabilities(vec![Capability::FileRead, Capability::FileWrite]);
        std::fs::write(
            h.path().join("src.rs"),
            "aaa\nbbb\nccc\nddd\neee\nfff\nggg\n",
        )
        .unwrap();

        let tool = SearchReplaceEditTool::new(&test_config(), Arc::new(EditHistory::new()));
        let output = tool
            .execute(
                serde_json::json!({
                    "path": "src.rs",
                    "old_string": "ccc\nddd\neee",
                    "new_string": "CCC\nDDD\nEEE"
                }),
                &h.context(),
            )
            .await
            .unwrap();

        assert!(output.content.contains("\u{2502} ccc"), "missing -ccc");
        assert!(output.content.contains("\u{2502} ddd"), "missing -ddd");
        assert!(output.content.contains("\u{2502} eee"), "missing -eee");
        assert!(output.content.contains("\u{2502} CCC"), "missing +CCC");
        assert!(output.content.contains("\u{2502} DDD"), "missing +DDD");
        assert!(output.content.contains("\u{2502} EEE"), "missing +EEE");
    }

    #[tokio::test]
    async fn test_single_line_diff() {
        let h = TestHarness::with_capabilities(vec![Capability::FileRead, Capability::FileWrite]);
        std::fs::write(h.path().join("src.rs"), "aaa\nbbb\nccc\nddd\neee\n").unwrap();

        let tool = SearchReplaceEditTool::new(&test_config(), Arc::new(EditHistory::new()));
        let output = tool
            .execute(
                serde_json::json!({
                    "path": "src.rs",
                    "old_string": "ccc",
                    "new_string": "CCC"
                }),
                &h.context(),
            )
            .await
            .unwrap();

        // Single-line change: exactly 1 removed, 1 added
        let lines: Vec<&str> = output.content.lines().collect();
        let minus_count = lines.iter().filter(|l| l.contains("- ")).count();
        let plus_count = lines.iter().filter(|l| l.contains("+ ")).count();
        assert_eq!(minus_count, 1, "single line edit should have 1 minus line");
        assert_eq!(plus_count, 1, "single line edit should have 1 plus line");
    }

    #[tokio::test]
    async fn test_deletion_shows_minus_only() {
        let h = TestHarness::with_capabilities(vec![Capability::FileRead, Capability::FileWrite]);
        std::fs::write(h.path().join("src.rs"), "aaa\nbbb\nccc\nddd\neee\n").unwrap();

        let tool = SearchReplaceEditTool::new(&test_config(), Arc::new(EditHistory::new()));
        let output = tool
            .execute(
                serde_json::json!({
                    "path": "src.rs",
                    "old_string": "bbb\nccc",
                    "new_string": ""
                }),
                &h.context(),
            )
            .await
            .unwrap();

        let lines: Vec<&str> = output.content.lines().collect();
        let minus_count = lines.iter().filter(|l| l.contains("- ")).count();
        let plus_count = lines.iter().filter(|l| l.contains("+ ")).count();
        assert_eq!(minus_count, 2, "deletion should show 2 minus lines");
        assert_eq!(plus_count, 0, "deletion should show no plus lines");
    }

    #[tokio::test]
    async fn test_insertion_shows_plus_only() {
        let h = TestHarness::with_capabilities(vec![Capability::FileRead, Capability::FileWrite]);
        std::fs::write(h.path().join("src.rs"), "aaa\nbbb\nccc\nddd\neee\n").unwrap();

        let tool = SearchReplaceEditTool::new(&test_config(), Arc::new(EditHistory::new()));
        let output = tool
            .execute(
                serde_json::json!({
                    "path": "src.rs",
                    "old_string": "bbb",
                    "new_string": "bbb\nnew1\nnew2"
                }),
                &h.context(),
            )
            .await
            .unwrap();

        // old "bbb" is replaced by "bbb\nnew1\nnew2", so 1 minus and 3 plus
        let lines: Vec<&str> = output.content.lines().collect();
        let minus_count = lines.iter().filter(|l| l.contains("- ")).count();
        let plus_count = lines.iter().filter(|l| l.contains("+ ")).count();
        assert_eq!(
            minus_count, 1,
            "insertion should show 1 minus line (old bbb)"
        );
        assert_eq!(plus_count, 3, "insertion should show 3 plus lines");
    }

    #[tokio::test]
    async fn test_line_numbers_correct() {
        let h = TestHarness::with_capabilities(vec![Capability::FileRead, Capability::FileWrite]);
        std::fs::write(
            h.path().join("src.rs"),
            "line1\nline2\nline3\nline4\nline5\n",
        )
        .unwrap();

        let tool = SearchReplaceEditTool::new(&test_config(), Arc::new(EditHistory::new()));
        let output = tool
            .execute(
                serde_json::json!({
                    "path": "src.rs",
                    "old_string": "line3",
                    "new_string": "LINE3"
                }),
                &h.context(),
            )
            .await
            .unwrap();

        // The diff should show the removed line at line 3
        assert!(
            output.content.contains("- 3\u{2502} line3"),
            "line number 3 should prefix removed line"
        );
        assert!(
            output.content.contains("+ 3\u{2502} LINE3"),
            "line number 3 should prefix added line"
        );
    }

    #[tokio::test]
    async fn test_disabled_no_diff() {
        let h = TestHarness::with_capabilities(vec![Capability::FileRead, Capability::FileWrite]);
        std::fs::write(h.path().join("src.rs"), "aaa\nbbb\nccc\n").unwrap();

        let tool = SearchReplaceEditTool::new(
            &EditConfig {
                diff_preview: false,
                diff_context_lines: 3,
                syntax_validation: false,
                ..EditConfig::default()
            },
            Arc::new(EditHistory::new()),
        );
        let output = tool
            .execute(
                serde_json::json!({
                    "path": "src.rs",
                    "old_string": "bbb",
                    "new_string": "BBB"
                }),
                &h.context(),
            )
            .await
            .unwrap();

        // Should just have the summary line, no diff markers
        assert!(
            !output.content.contains("- "),
            "disabled should have no diff markers"
        );
        assert!(
            !output.content.contains("+ "),
            "disabled should have no diff markers"
        );
        assert!(
            output.content.starts_with("Edited"),
            "should have summary line"
        );
    }

    #[test]
    fn test_format_diff_preview_exact_output() {
        let content = "aaa\nbbb\nccc\nddd\neee\nfff\nggg\n";
        let result = format_diff_preview(content, "ddd", "DDD", 4, 1, 2);
        let expected = "  2\u{2502} bbb\n  3\u{2502} ccc\n- 4\u{2502} ddd\n+ 4\u{2502} DDD\n  5\u{2502} eee\n  6\u{2502} fff";
        assert_eq!(result, expected);
    }

    #[tokio::test]
    async fn test_diff_preview_at_file_start() {
        let h = TestHarness::with_capabilities(vec![Capability::FileRead, Capability::FileWrite]);
        std::fs::write(
            h.path().join("src.rs"),
            "first\nsecond\nthird\nfourth\nfifth\n",
        )
        .unwrap();

        let tool = SearchReplaceEditTool::new(&test_config(), Arc::new(EditHistory::new()));
        let output = tool
            .execute(
                serde_json::json!({
                    "path": "src.rs",
                    "old_string": "first",
                    "new_string": "FIRST"
                }),
                &h.context(),
            )
            .await
            .unwrap();

        // No context before line 1 — only after-context should appear
        assert!(output.content.contains("- "), "should have removed marker");
        assert!(output.content.contains("+ "), "should have added marker");
        assert!(
            output.content.contains("\u{2502} second"),
            "should have context after change"
        );
    }

    #[tokio::test]
    async fn test_diff_preview_at_file_end() {
        let h = TestHarness::with_capabilities(vec![Capability::FileRead, Capability::FileWrite]);
        std::fs::write(
            h.path().join("src.rs"),
            "first\nsecond\nthird\nfourth\nlast",
        )
        .unwrap();

        let tool = SearchReplaceEditTool::new(&test_config(), Arc::new(EditHistory::new()));
        let output = tool
            .execute(
                serde_json::json!({
                    "path": "src.rs",
                    "old_string": "last",
                    "new_string": "LAST"
                }),
                &h.context(),
            )
            .await
            .unwrap();

        // Context before the last line, no context after
        assert!(
            output.content.contains("\u{2502} fourth"),
            "should have context before change"
        );
        assert!(output.content.contains("- "), "should have removed marker");
        assert!(output.content.contains("+ "), "should have added marker");
    }

    #[tokio::test]
    async fn test_diff_preview_zero_context_lines() {
        let h = TestHarness::with_capabilities(vec![Capability::FileRead, Capability::FileWrite]);
        std::fs::write(h.path().join("src.rs"), "aaa\nbbb\nccc\nddd\neee\n").unwrap();

        let tool = SearchReplaceEditTool::new(
            &EditConfig {
                diff_preview: true,
                diff_context_lines: 0,
                syntax_validation: false,
                ..EditConfig::default()
            },
            Arc::new(EditHistory::new()),
        );
        let output = tool
            .execute(
                serde_json::json!({
                    "path": "src.rs",
                    "old_string": "ccc",
                    "new_string": "CCC"
                }),
                &h.context(),
            )
            .await
            .unwrap();

        let diff_lines: Vec<&str> = output
            .content
            .lines()
            .skip_while(|l| !l.contains("\u{2502}"))
            .collect();
        // With 0 context: only the changed lines, no surrounding context
        assert_eq!(
            diff_lines.len(),
            2,
            "should have exactly 2 lines (1 removed + 1 added)"
        );
        assert!(
            diff_lines.iter().any(|l| l.starts_with("- ")),
            "should have removed"
        );
        assert!(
            diff_lines.iter().any(|l| l.starts_with("+ ")),
            "should have added"
        );
    }

    // ── Property-based tests ─────────────────────────────────────

    mod proptests {
        use super::*;
        use proptest::prelude::*;

        /// Strategy that generates safe filenames.
        fn safe_filename() -> impl Strategy<Value = String> {
            "[a-zA-Z0-9_-]{1,64}\\.(rs|txt|go|py)"
        }

        /// Strategy that generates lines of visible ASCII.
        fn visible_line() -> impl Strategy<Value = String> {
            // At least one non-space printable char so indentation preservation
            // doesn't trim the entire string (pure-whitespace "code" is not a
            // realistic edit scenario).
            "[\\x21-\\x7E][\\x20-\\x7E]{0,79}"
        }

        proptest! {
            /// Path traversal never succeeds, regardless of depth or suffix.
            #[test]
            fn path_traversal_always_rejected(
                depth in 1usize..20,
                suffix in "[a-z]{1,10}",
            ) {
                let rt = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .unwrap();
                rt.block_on(async {
                    let h = TestHarness::with_capabilities(vec![Capability::FileRead, Capability::FileWrite]);
                    let tool = SearchReplaceEditTool::new(&test_config(), Arc::new(EditHistory::new()));

                    let traversal = format!("{}{}", "../".repeat(depth), suffix);
                    let result = tool
                        .execute(
                            serde_json::json!({
                                "path": &traversal,
                                "old_string": "x",
                                "new_string": "y"
                            }),
                            &h.context(),
                        )
                        .await;
                    prop_assert!(result.is_err());
                    Ok(())
                })?;
            }

            /// Editing a file then reading it back always yields content
            /// that contains new_string and does not contain old_string.
            #[test]
            fn edit_roundtrip_replaces_correctly(
                name in safe_filename(),
                prefix in visible_line(),
                old_text in visible_line(),
                suffix in visible_line(),
                new_text in visible_line(),
            ) {
                // Skip if old_text == new_text (identity edit rejected)
                // or old_text appears in prefix/suffix (ambiguous match)
                if old_text == new_text
                    || prefix.contains(&old_text)
                    || suffix.contains(&old_text)
                {
                    return Ok(());
                }
                let rt = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .unwrap();
                rt.block_on(async {
                    let h = TestHarness::with_capabilities(vec![Capability::FileRead, Capability::FileWrite]);
                    let content = format!("{prefix}\n{old_text}\n{suffix}\n");
                    std::fs::write(h.path().join(&name), &content).unwrap();

                    let tool = SearchReplaceEditTool::new(&test_config(), Arc::new(EditHistory::new()));
                    let result = tool
                        .execute(
                            serde_json::json!({
                                "path": &name,
                                "old_string": &old_text,
                                "new_string": &new_text
                            }),
                            &h.context(),
                        )
                        .await;
                    prop_assert!(result.is_ok(), "edit failed: {:?}", result.err());

                    let after = std::fs::read_to_string(h.path().join(&name)).unwrap();
                    prop_assert!(after.contains(&new_text), "new_text not in result");
                    // Only assert old_text is gone when it's not a substring of new_text,
                    // because replacement naturally re-introduces it in that case.
                    if !new_text.contains(&old_text) {
                        prop_assert!(!after.contains(&old_text), "old_text still in result");
                    }
                    Ok(())
                })?;
            }

            /// Output never leaks the sandbox root path.
            #[test]
            fn output_never_leaks_sandbox_root(
                name in safe_filename(),
            ) {
                let rt = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .unwrap();
                rt.block_on(async {
                    let h = TestHarness::with_capabilities(vec![Capability::FileRead, Capability::FileWrite]);
                    std::fs::write(h.path().join(&name), "old_unique_sentinel\n").unwrap();

                    let tool = SearchReplaceEditTool::new(&test_config(), Arc::new(EditHistory::new()));
                    let output = tool
                        .execute(
                            serde_json::json!({
                                "path": &name,
                                "old_string": "old_unique_sentinel",
                                "new_string": "new_sentinel"
                            }),
                            &h.context(),
                        )
                        .await
                        .unwrap();
                    let sandbox_str = h.path().to_string_lossy();
                    prop_assert!(
                        !output.content.contains(sandbox_str.as_ref()),
                        "output leaked sandbox root: {}",
                        output.content
                    );
                    Ok(())
                })?;
            }
        }
    }

    // ── Syntax validation integration tests ──────────────────────

    #[tokio::test]
    async fn test_edit_rejects_syntax_breaking_change() {
        let h = TestHarness::with_capabilities(vec![Capability::FileRead, Capability::FileWrite]);
        let original = "fn hello() {\n    println!(\"hi\");\n}\n";
        std::fs::write(h.path().join("code.rs"), original).unwrap();

        let tool = SearchReplaceEditTool::new(
            &EditConfig {
                syntax_validation: true,
                ..EditConfig::default()
            },
            Arc::new(EditHistory::new()),
        );

        // Remove the closing brace — breaks syntax
        let result = tool
            .execute(
                serde_json::json!({
                    "path": "code.rs",
                    "old_string": "}\n",
                    "new_string": ""
                }),
                &h.context(),
            )
            .await;

        assert!(result.is_err(), "edit should be rejected");
        // Original file must be untouched
        let on_disk = std::fs::read_to_string(h.path().join("code.rs")).unwrap();
        assert_eq!(on_disk, original, "original file must be preserved exactly");
    }

    #[tokio::test]
    async fn test_edit_allows_valid_syntax_change() {
        let h = TestHarness::with_capabilities(vec![Capability::FileRead, Capability::FileWrite]);
        std::fs::write(
            h.path().join("valid.rs"),
            "fn hello() {\n    println!(\"hi\");\n}\n",
        )
        .unwrap();

        let tool = SearchReplaceEditTool::new(
            &EditConfig {
                syntax_validation: true,
                ..EditConfig::default()
            },
            Arc::new(EditHistory::new()),
        );

        let output = tool
            .execute(
                serde_json::json!({
                    "path": "valid.rs",
                    "old_string": "fn hello()",
                    "new_string": "fn greet()"
                }),
                &h.context(),
            )
            .await
            .unwrap();

        assert!(matches!(output.outcome, ToolOutcome::Success));
        let content = std::fs::read_to_string(h.path().join("valid.rs")).unwrap();
        assert!(content.contains("fn greet()"));
    }

    #[tokio::test]
    async fn test_edit_skips_validation_for_non_rust() {
        let h = TestHarness::with_capabilities(vec![Capability::FileRead, Capability::FileWrite]);
        std::fs::write(h.path().join("notes.txt"), "hello world").unwrap();

        let tool = SearchReplaceEditTool::new(
            &EditConfig {
                syntax_validation: true,
                ..EditConfig::default()
            },
            Arc::new(EditHistory::new()),
        );

        let output = tool
            .execute(
                serde_json::json!({
                    "path": "notes.txt",
                    "old_string": "hello world",
                    "new_string": "fn broken( {{{"
                }),
                &h.context(),
            )
            .await
            .unwrap();

        assert!(matches!(output.outcome, ToolOutcome::Success));
    }

    #[tokio::test]
    async fn test_edit_skips_validation_when_disabled() {
        let h = TestHarness::with_capabilities(vec![Capability::FileRead, Capability::FileWrite]);
        let original = "fn hello() {\n    println!(\"hi\");\n}\n";
        std::fs::write(h.path().join("code.rs"), original).unwrap();

        let tool = SearchReplaceEditTool::new(
            &EditConfig {
                syntax_validation: false,
                ..EditConfig::default()
            },
            Arc::new(EditHistory::new()),
        );

        // Remove closing brace — breaks syntax, but validation is off
        let output = tool
            .execute(
                serde_json::json!({
                    "path": "code.rs",
                    "old_string": "}\n",
                    "new_string": ""
                }),
                &h.context(),
            )
            .await
            .unwrap();

        assert!(matches!(output.outcome, ToolOutcome::Success));
    }

    // ── Undo tool tests ─────────────────────────────────────────

    /// Create an edit tool and undo tool sharing the same history.
    fn edit_undo_pair() -> (SearchReplaceEditTool, UndoEditTool) {
        let history = Arc::new(EditHistory::new());
        let edit = SearchReplaceEditTool::new(&test_config(), Arc::clone(&history));
        let undo = UndoEditTool::new(history);
        (edit, undo)
    }

    #[tokio::test]
    async fn test_undo_restores_previous() {
        let h = TestHarness::with_capabilities(vec![Capability::FileRead, Capability::FileWrite]);
        let original = "line 1\nline 2\nline 3\n";
        std::fs::write(h.path().join("file.rs"), original).unwrap();

        let (edit, undo) = edit_undo_pair();
        let ctx = h.context();

        // Edit the file
        edit.execute(
            serde_json::json!({
                "path": "file.rs",
                "old_string": "line 2",
                "new_string": "CHANGED"
            }),
            &ctx,
        )
        .await
        .unwrap();

        // Verify edit took effect
        let content = std::fs::read_to_string(h.path().join("file.rs")).unwrap();
        assert!(content.contains("CHANGED"));

        // Undo
        let output = undo
            .execute(serde_json::json!({"path": "file.rs"}), &ctx)
            .await
            .unwrap();

        assert!(matches!(output.outcome, ToolOutcome::Success));
        assert!(output.content.contains("Restored"));

        // Verify undo restored original content
        let content = std::fs::read_to_string(h.path().join("file.rs")).unwrap();
        assert_eq!(content, original);
    }

    #[tokio::test]
    async fn test_multiple_undos() {
        let h = TestHarness::with_capabilities(vec![Capability::FileRead, Capability::FileWrite]);
        let v0 = "original\n";
        std::fs::write(h.path().join("f.rs"), v0).unwrap();

        let (edit, undo) = edit_undo_pair();
        let ctx = h.context();

        // Edit 3 times
        edit.execute(
            serde_json::json!({"path": "f.rs", "old_string": "original", "new_string": "v1"}),
            &ctx,
        )
        .await
        .unwrap();
        edit.execute(
            serde_json::json!({"path": "f.rs", "old_string": "v1", "new_string": "v2"}),
            &ctx,
        )
        .await
        .unwrap();
        edit.execute(
            serde_json::json!({"path": "f.rs", "old_string": "v2", "new_string": "v3"}),
            &ctx,
        )
        .await
        .unwrap();

        // Undo 3 times, each should restore the prior version
        undo.execute(serde_json::json!({"path": "f.rs"}), &ctx)
            .await
            .unwrap();
        assert_eq!(
            std::fs::read_to_string(h.path().join("f.rs")).unwrap(),
            "v2\n"
        );

        undo.execute(serde_json::json!({"path": "f.rs"}), &ctx)
            .await
            .unwrap();
        assert_eq!(
            std::fs::read_to_string(h.path().join("f.rs")).unwrap(),
            "v1\n"
        );

        undo.execute(serde_json::json!({"path": "f.rs"}), &ctx)
            .await
            .unwrap();
        assert_eq!(std::fs::read_to_string(h.path().join("f.rs")).unwrap(), v0);
    }

    #[tokio::test]
    async fn test_undo_unedited_file_error() {
        let h = TestHarness::with_capabilities(vec![Capability::FileRead, Capability::FileWrite]);
        std::fs::write(h.path().join("untouched.rs"), "content").unwrap();

        let undo = UndoEditTool::new(Arc::new(EditHistory::new()));
        let result = undo
            .execute(serde_json::json!({"path": "untouched.rs"}), &h.context())
            .await;

        assert!(result.is_err());
        match result.unwrap_err() {
            ToolError::ExecutionFailed { reason, .. } => {
                assert!(reason.contains("no edit history"), "got: {reason}");
            }
            other => panic!("expected ExecutionFailed, got: {other:?}"),
        }
    }

    // ── Checkpoint tool tests ───────────────────────────────────

    /// Create edit, checkpoint, and rollback tools sharing the same history.
    fn edit_checkpoint_triple() -> (
        SearchReplaceEditTool,
        CreateCheckpointTool,
        RollbackToCheckpointTool,
    ) {
        let history = Arc::new(EditHistory::new());
        let edit = SearchReplaceEditTool::new(&test_config(), Arc::clone(&history));
        let checkpoint = CreateCheckpointTool::new(Arc::clone(&history));
        let rollback = RollbackToCheckpointTool::new(history);
        (edit, checkpoint, rollback)
    }

    #[tokio::test]
    async fn test_checkpoint_captures_state() {
        let h = TestHarness::with_capabilities(vec![Capability::FileRead, Capability::FileWrite]);
        let original = "before\n";
        std::fs::write(h.path().join("file.rs"), original).unwrap();

        let (edit, checkpoint, rollback) = edit_checkpoint_triple();
        let ctx = h.context();

        // Edit to trigger tracking
        edit.execute(
            serde_json::json!({"path": "file.rs", "old_string": "before", "new_string": "after-edit"}),
            &ctx,
        )
        .await
        .unwrap();

        // Checkpoint
        checkpoint
            .execute(serde_json::json!({"name": "cp1"}), &ctx)
            .await
            .unwrap();

        // Edit again
        edit.execute(
            serde_json::json!({"path": "file.rs", "old_string": "after-edit", "new_string": "further-edit"}),
            &ctx,
        )
        .await
        .unwrap();
        assert!(
            std::fs::read_to_string(h.path().join("file.rs"))
                .unwrap()
                .contains("further-edit")
        );

        // Rollback to checkpoint
        let output = rollback
            .execute(serde_json::json!({"name": "cp1"}), &ctx)
            .await
            .unwrap();

        assert!(matches!(output.outcome, ToolOutcome::Success));
        assert!(output.content.contains("Rolled back"));

        // File should be at checkpoint state (after first edit, before second)
        let content = std::fs::read_to_string(h.path().join("file.rs")).unwrap();
        assert!(content.contains("after-edit"));
    }

    #[tokio::test]
    async fn test_checkpoint_restores_multiple_files() {
        let h = TestHarness::with_capabilities(vec![Capability::FileRead, Capability::FileWrite]);
        std::fs::write(h.path().join("a.rs"), "a-original\n").unwrap();
        std::fs::write(h.path().join("b.rs"), "b-original\n").unwrap();
        std::fs::write(h.path().join("c.rs"), "c-original\n").unwrap();

        let (edit, checkpoint, rollback) = edit_checkpoint_triple();
        let ctx = h.context();

        // Edit all 3 files
        for name in ["a.rs", "b.rs", "c.rs"] {
            let original = format!("{}-original", name.strip_suffix(".rs").unwrap_or(name));
            let edited = format!("{}-edited", name.strip_suffix(".rs").unwrap_or(name));
            edit.execute(
                serde_json::json!({"path": name, "old_string": original, "new_string": edited}),
                &ctx,
            )
            .await
            .unwrap();
        }

        // Checkpoint
        checkpoint
            .execute(serde_json::json!({"name": "multi"}), &ctx)
            .await
            .unwrap();

        // Edit again
        for name in ["a.rs", "b.rs", "c.rs"] {
            let edited = format!("{}-edited", name.strip_suffix(".rs").unwrap_or(name));
            let further = format!("{}-further", name.strip_suffix(".rs").unwrap_or(name));
            edit.execute(
                serde_json::json!({"path": name, "old_string": edited, "new_string": further}),
                &ctx,
            )
            .await
            .unwrap();
        }

        // Rollback
        rollback
            .execute(serde_json::json!({"name": "multi"}), &ctx)
            .await
            .unwrap();

        // All 3 files should be at checkpoint state
        assert!(
            std::fs::read_to_string(h.path().join("a.rs"))
                .unwrap()
                .contains("a-edited")
        );
        assert!(
            std::fs::read_to_string(h.path().join("b.rs"))
                .unwrap()
                .contains("b-edited")
        );
        assert!(
            std::fs::read_to_string(h.path().join("c.rs"))
                .unwrap()
                .contains("c-edited")
        );
    }

    #[tokio::test]
    async fn test_checkpoint_not_found_error() {
        let h = TestHarness::with_capabilities(vec![Capability::FileRead, Capability::FileWrite]);
        let rollback = RollbackToCheckpointTool::new(Arc::new(EditHistory::new()));

        let result = rollback
            .execute(serde_json::json!({"name": "nonexistent"}), &h.context())
            .await;

        assert!(result.is_err());
        match result.unwrap_err() {
            ToolError::ExecutionFailed { reason, .. } => {
                assert!(
                    reason.contains("no checkpoints") || reason.contains("not found"),
                    "got: {reason}"
                );
            }
            other => panic!("expected ExecutionFailed, got: {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_max_checkpoints_enforced() {
        let h = TestHarness::with_capabilities(vec![Capability::FileRead, Capability::FileWrite]);
        std::fs::write(h.path().join("f.rs"), "content\n").unwrap();

        let (edit, checkpoint, rollback) = edit_checkpoint_triple();
        let ctx = h.context();

        // Edit to track the file
        edit.execute(
            serde_json::json!({"path": "f.rs", "old_string": "content", "new_string": "modified"}),
            &ctx,
        )
        .await
        .unwrap();

        // Create 6 checkpoints — 1st should be evicted
        for i in 0..6 {
            // Need to update file content so checkpoint tool can read it
            let prev = if i == 0 {
                "modified".to_string()
            } else {
                format!("v{}", i - 1)
            };
            edit.execute(
                serde_json::json!({"path": "f.rs", "old_string": prev, "new_string": format!("v{i}")}),
                &ctx,
            )
            .await
            .unwrap();

            checkpoint
                .execute(serde_json::json!({"name": format!("cp{i}")}), &ctx)
                .await
                .unwrap();
        }

        // cp0 should be evicted
        let result = rollback
            .execute(serde_json::json!({"name": "cp0"}), &ctx)
            .await;
        assert!(result.is_err());

        // cp5 (latest) should still work
        let output = rollback
            .execute(serde_json::json!({"name": "cp5"}), &ctx)
            .await
            .unwrap();
        assert!(matches!(output.outcome, ToolOutcome::Success));
    }

    #[tokio::test]
    async fn test_checkpoint_expiry() {
        use std::time::{Duration, Instant};

        let h = TestHarness::with_capabilities(vec![Capability::FileRead, Capability::FileWrite]);
        std::fs::write(h.path().join("f.rs"), "content\n").unwrap();

        let history = Arc::new(EditHistory::new());
        let rollback = RollbackToCheckpointTool::new(Arc::clone(&history));
        let ctx = h.context();

        // Insert an expired checkpoint via test helper
        let mut files = HashMap::new();
        files.insert(h.path().join("f.rs"), "old-content".to_string());
        history.insert_checkpoint_at(
            ctx.session_id,
            "expired-cp".to_string(),
            files,
            Instant::now()
                .checked_sub(Duration::from_secs(7200))
                .unwrap(),
        );

        let result = rollback
            .execute(serde_json::json!({"name": "expired-cp"}), &ctx)
            .await;

        assert!(result.is_err());
        match result.unwrap_err() {
            ToolError::ExecutionFailed { reason, .. } => {
                assert!(reason.contains("expired"), "got: {reason}");
            }
            other => panic!("expected ExecutionFailed, got: {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_checkpoint_name_validation() {
        let h = TestHarness::with_capabilities(vec![Capability::FileRead, Capability::FileWrite]);
        let checkpoint = CreateCheckpointTool::new(Arc::new(EditHistory::new()));
        let ctx = h.context();

        // Invalid names
        for name in ["", "has spaces", "has;semicolons", "has/slash"] {
            let result = checkpoint
                .execute(serde_json::json!({"name": name}), &ctx)
                .await;
            assert!(result.is_err(), "expected error for name: {name:?}");
        }
    }

    // ── Large edit guardrail tests ──────────────────────────────

    fn large_edit_config(threshold: f64, action: LargeEditAction) -> EditConfig {
        EditConfig {
            syntax_validation: false,
            large_edit_threshold: threshold,
            large_edit_action: action,
            ..EditConfig::default()
        }
    }

    #[tokio::test]
    async fn test_large_edit_small_edit_no_warning() {
        let h = TestHarness::with_capabilities(vec![Capability::FileRead, Capability::FileWrite]);
        // 100 bytes of content, replace 5 bytes → 5% ratio
        let content = "a".repeat(95) + "XXXXX";
        std::fs::write(h.path().join("file.txt"), &content).unwrap();

        let tool = SearchReplaceEditTool::new(
            &large_edit_config(0.5, LargeEditAction::Warn),
            Arc::new(EditHistory::new()),
        );
        let ctx = h.context();
        let out = tool
            .execute(
                serde_json::json!({
                    "path": "file.txt",
                    "old_string": "XXXXX",
                    "new_string": "YYYYY",
                }),
                &ctx,
            )
            .await
            .unwrap();

        assert_eq!(out.outcome, ToolOutcome::Success);
        assert!(
            !out.content.to_lowercase().contains("warning"),
            "small edit should not have warning, got: {}",
            out.content
        );
    }

    #[tokio::test]
    async fn test_large_edit_warns() {
        let h = TestHarness::with_capabilities(vec![Capability::FileRead, Capability::FileWrite]);
        // 100 bytes, replace 60 bytes → 60% ratio (>= 50%)
        let content = "a".repeat(40) + &"b".repeat(60);
        std::fs::write(h.path().join("file.txt"), &content).unwrap();

        let tool = SearchReplaceEditTool::new(
            &large_edit_config(0.5, LargeEditAction::Warn),
            Arc::new(EditHistory::new()),
        );
        let ctx = h.context();
        let out = tool
            .execute(
                serde_json::json!({
                    "path": "file.txt",
                    "old_string": "b".repeat(60),
                    "new_string": "c".repeat(60),
                }),
                &ctx,
            )
            .await
            .unwrap();

        assert_eq!(out.outcome, ToolOutcome::Success);
        assert!(
            out.content.contains("Large edit warning"),
            "expected warning, got: {}",
            out.content
        );
        // File should be modified
        let modified = std::fs::read_to_string(h.path().join("file.txt")).unwrap();
        assert!(modified.contains(&"c".repeat(60)));
    }

    #[tokio::test]
    async fn test_large_edit_blocked() {
        let h = TestHarness::with_capabilities(vec![Capability::FileRead, Capability::FileWrite]);
        let content = "a".repeat(40) + &"b".repeat(60);
        std::fs::write(h.path().join("file.txt"), &content).unwrap();

        let tool = SearchReplaceEditTool::new(
            &large_edit_config(0.5, LargeEditAction::Block),
            Arc::new(EditHistory::new()),
        );
        let ctx = h.context();
        let out = tool
            .execute(
                serde_json::json!({
                    "path": "file.txt",
                    "old_string": "b".repeat(60),
                    "new_string": "c".repeat(60),
                }),
                &ctx,
            )
            .await
            .unwrap();

        assert_eq!(out.outcome, ToolOutcome::Error);
        assert!(
            out.content.contains("Edit rejected"),
            "expected rejection, got: {}",
            out.content
        );
        // File should NOT be modified
        let unchanged = std::fs::read_to_string(h.path().join("file.txt")).unwrap();
        assert_eq!(unchanged, content);
    }

    #[tokio::test]
    async fn test_large_edit_consent_rejected() {
        let h = TestHarness::with_capabilities(vec![Capability::FileRead, Capability::FileWrite]);
        let content = "a".repeat(40) + &"b".repeat(60);
        std::fs::write(h.path().join("file.txt"), &content).unwrap();

        let tool = SearchReplaceEditTool::new(
            &large_edit_config(0.5, LargeEditAction::Consent),
            Arc::new(EditHistory::new()),
        );
        let ctx = h.context();
        let out = tool
            .execute(
                serde_json::json!({
                    "path": "file.txt",
                    "old_string": "b".repeat(60),
                    "new_string": "c".repeat(60),
                }),
                &ctx,
            )
            .await
            .unwrap();

        assert_eq!(out.outcome, ToolOutcome::Error);
        assert!(
            out.content.contains("smaller edits"),
            "expected consent guidance, got: {}",
            out.content
        );
        // File should NOT be modified
        let unchanged = std::fs::read_to_string(h.path().join("file.txt")).unwrap();
        assert_eq!(unchanged, content);
    }

    #[tokio::test]
    async fn test_large_edit_threshold_configurable() {
        let h = TestHarness::with_capabilities(vec![Capability::FileRead, Capability::FileWrite]);
        // 100 bytes, replace 60 → 60% ratio, but threshold is 80%
        let content = "a".repeat(40) + &"b".repeat(60);
        std::fs::write(h.path().join("file.txt"), &content).unwrap();

        let tool = SearchReplaceEditTool::new(
            &large_edit_config(0.8, LargeEditAction::Warn),
            Arc::new(EditHistory::new()),
        );
        let ctx = h.context();
        let out = tool
            .execute(
                serde_json::json!({
                    "path": "file.txt",
                    "old_string": "b".repeat(60),
                    "new_string": "c".repeat(60),
                }),
                &ctx,
            )
            .await
            .unwrap();

        assert_eq!(out.outcome, ToolOutcome::Success);
        assert!(
            !out.content.to_lowercase().contains("warning"),
            "60% under 80% threshold should not warn, got: {}",
            out.content
        );
    }

    #[tokio::test]
    async fn test_large_edit_warning_includes_ratio() {
        let h = TestHarness::with_capabilities(vec![Capability::FileRead, Capability::FileWrite]);
        let content = "a".repeat(40) + &"b".repeat(60);
        std::fs::write(h.path().join("file.txt"), &content).unwrap();

        let tool = SearchReplaceEditTool::new(
            &large_edit_config(0.5, LargeEditAction::Warn),
            Arc::new(EditHistory::new()),
        );
        let ctx = h.context();
        let out = tool
            .execute(
                serde_json::json!({
                    "path": "file.txt",
                    "old_string": "b".repeat(60),
                    "new_string": "c".repeat(60),
                }),
                &ctx,
            )
            .await
            .unwrap();

        assert!(
            out.content.contains("60%"),
            "warning should include the percentage, got: {}",
            out.content
        );
    }

    #[tokio::test]
    async fn test_large_edit_exact_threshold_boundary() {
        let h = TestHarness::with_capabilities(vec![Capability::FileRead, Capability::FileWrite]);
        // 100 bytes, replace exactly 50 → 50% ratio, threshold 0.5 → should flag (>=)
        let content = "a".repeat(50) + &"b".repeat(50);
        std::fs::write(h.path().join("file.txt"), &content).unwrap();

        let tool = SearchReplaceEditTool::new(
            &large_edit_config(0.5, LargeEditAction::Block),
            Arc::new(EditHistory::new()),
        );
        let ctx = h.context();
        let out = tool
            .execute(
                serde_json::json!({
                    "path": "file.txt",
                    "old_string": "b".repeat(50),
                    "new_string": "c".repeat(50),
                }),
                &ctx,
            )
            .await
            .unwrap();

        assert_eq!(
            out.outcome,
            ToolOutcome::Error,
            "exactly at threshold should be flagged"
        );
    }

    #[tokio::test]
    async fn test_large_edit_empty_file_bypasses_check() {
        let h = TestHarness::with_capabilities(vec![Capability::FileRead, Capability::FileWrite]);
        // Empty file — guardrail should not trigger (no division by zero)
        std::fs::write(h.path().join("file.txt"), "").unwrap();

        let tool = SearchReplaceEditTool::new(
            &large_edit_config(0.5, LargeEditAction::Block),
            Arc::new(EditHistory::new()),
        );
        let ctx = h.context();
        // Inserting into empty file: old_string="" matches the empty file content
        let out = tool
            .execute(
                serde_json::json!({
                    "path": "file.txt",
                    "old_string": "",
                    "new_string": "new content here",
                }),
                &ctx,
            )
            .await
            .unwrap();

        // Should succeed without hitting the guardrail
        assert_eq!(
            out.outcome,
            ToolOutcome::Success,
            "empty file should bypass guardrail, got: {}",
            out.content
        );
    }
}
