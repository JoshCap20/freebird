//! Search/replace edit tool for surgical code modifications.
//!
//! The LLM provides `path`, `old_string` (exact text to find), and `new_string`
//! (replacement text). No line numbers — the model specifies literal code blocks,
//! which avoids off-by-one errors that plague diff-based formats.
//!
//! Research shows search/replace format has a 23–27 percentage point improvement
//! over unified diff and line-based diff formats (Meta agentic repair paper).

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

/// Count the line number (1-indexed) where `needle` first appears in `haystack`.
fn line_number_of_match(haystack: &str, byte_offset: usize) -> usize {
    haystack
        .get(..byte_offset)
        .unwrap_or(haystack)
        .chars()
        .filter(|&c| c == '\n')
        .count()
        + 1
}

/// Count how many lines `s` spans (at least 1 for non-empty, 0 for empty).
fn line_count(s: &str) -> usize {
    if s.is_empty() {
        return 0;
    }
    s.chars().filter(|&c| c == '\n').count() + 1
}

/// Normalize a string for whitespace-tolerant matching:
/// - Trim leading/trailing whitespace per line
/// - Collapse runs of inline whitespace to a single space
/// - Preserve line boundaries
fn normalize_whitespace(s: &str) -> String {
    let mut result: String = s
        .lines()
        .map(|line| line.split_whitespace().collect::<Vec<_>>().join(" "))
        .collect::<Vec<_>>()
        .join("\n");
    // Preserve trailing newline so normalization is idempotent
    if s.ends_with('\n') {
        result.push('\n');
    }
    result
}

/// A line from the original haystack, remembering its byte offset.
struct SourceLine<'a> {
    text: &'a str,
    byte_offset: usize,
}

/// Split `haystack` into lines that remember their byte offsets.
/// Handles both `\n` and `\r\n`. Does NOT use `str::lines()` because that
/// erases terminator length information needed for correct byte-offset math.
///
/// `split_inclusive('\n')` returns the last segment even without a trailing
/// newline, so no special end-of-input handling is needed.
fn split_source_lines(haystack: &str) -> Vec<SourceLine<'_>> {
    if haystack.is_empty() {
        return vec![];
    }
    let mut lines = Vec::new();
    let mut offset = 0;
    for line_with_term in haystack.split_inclusive('\n') {
        let text = line_with_term.strip_suffix('\n').unwrap_or(line_with_term);
        let text = text.strip_suffix('\r').unwrap_or(text);
        lines.push(SourceLine {
            text,
            byte_offset: offset,
        });
        offset += line_with_term.len();
    }
    lines
}

/// Find all byte-offset positions where `needle` appears in `haystack` using
/// normalized (whitespace-collapsed) comparison. Returns the byte offsets and
/// byte lengths in the *original* haystack.
///
/// Correctly handles `\r\n` line endings by tracking actual terminator byte
/// lengths instead of assuming 1-byte `\n`.
fn find_normalized_matches(haystack: &str, needle: &str) -> Vec<(usize, usize)> {
    let norm_needle = normalize_whitespace(needle);
    if norm_needle.is_empty() {
        return vec![];
    }

    let needle_line_count = norm_needle.lines().count();
    let source_lines = split_source_lines(haystack);

    let mut matches = Vec::new();

    if source_lines.len() < needle_line_count {
        return matches;
    }

    for start_idx in 0..=source_lines.len() - needle_line_count {
        let window_slice = source_lines
            .get(start_idx..start_idx + needle_line_count)
            .unwrap_or(&[]);
        let window_texts: Vec<&str> = window_slice.iter().map(|sl| sl.text).collect();
        let window = window_texts.join("\n");
        let norm_window = normalize_whitespace(&window);

        if norm_window == norm_needle {
            let byte_start = window_slice.first().map_or(0, |sl| sl.byte_offset);
            // Total bytes from start of first matched line to end of last matched line's text
            // (excluding the last line's terminator, since we're replacing the text content).
            let byte_len: usize =
                if let (Some(first), Some(last)) = (window_slice.first(), window_slice.last()) {
                    (last.byte_offset + last.text.len()) - first.byte_offset
                } else {
                    0
                };
            matches.push((byte_start, byte_len));
        }
    }

    matches
}

/// Detect the indentation character used in a whitespace prefix.
/// Returns `'\t'` if tabs dominate, `' '` otherwise.
fn detect_indent_char(prefix: &str) -> char {
    let tabs = prefix.chars().filter(|&c| c == '\t').count();
    let spaces = prefix.chars().filter(|&c| c == ' ').count();
    if tabs > spaces { '\t' } else { ' ' }
}

/// Apply indentation preservation: detect the indentation delta between the
/// matched block and the replacement, then shift all replacement lines.
///
/// Tab-aware: detects whether the matched region uses tabs or spaces and
/// replicates the same character for relative indentation offsets.
fn apply_indentation(matched_first_line: &str, new_string: &str) -> String {
    if new_string.is_empty() {
        return String::new();
    }

    let match_indent = matched_first_line.len() - matched_first_line.trim_start().len();
    let new_lines: Vec<&str> = new_string.lines().collect();

    // Detect indentation of the first line of new_string
    let new_indent = new_lines
        .first()
        .map_or(0, |l| l.len() - l.trim_start().len());

    if match_indent == new_indent {
        return new_string.to_string();
    }

    let match_prefix = matched_first_line.get(..match_indent).unwrap_or("");
    let indent_char = detect_indent_char(match_prefix);

    new_lines
        .iter()
        .enumerate()
        .map(|(i, line)| {
            if i == 0 {
                // First line: replace its indentation with match indentation
                format!("{}{}", match_prefix, line.trim_start())
            } else if line.trim().is_empty() {
                // Preserve blank lines as-is
                String::new()
            } else {
                // Other lines: compute relative indent from first line and apply
                // using the same indent character as the matched region
                let line_indent = line.len() - line.trim_start().len();
                let relative = line_indent.saturating_sub(new_indent);
                let relative_str: String = std::iter::repeat_n(indent_char, relative).collect();
                format!("{}{}{}", match_prefix, relative_str, line.trim_start())
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
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
            let _ = writeln!(out, "  {line_num:>width$}│ {text}");
        }
    }

    // Removed lines
    for (i, line) in old_lines.iter().enumerate() {
        let line_num = start_line + i;
        let _ = writeln!(out, "- {line_num:>width$}│ {line}");
    }

    // Added lines
    for (i, line) in new_lines.iter().enumerate() {
        let line_num = start_line + i;
        let _ = writeln!(out, "+ {line_num:>width$}│ {line}");
    }

    // Context lines after the change
    for line_num in (change_end + 1)..=ctx_end {
        if let Some(text) = file_lines.get(line_num - 1) {
            let _ = writeln!(out, "  {line_num:>width$}│ {text}");
        }
    }

    // Remove trailing newline
    if out.ends_with('\n') {
        out.pop();
    }

    out
}

/// Result of a successful match+replace operation.
struct ReplaceResult {
    content: String,
    start_line: usize,
    replaced_lines: usize,
    matched_text: String,
    adjusted_new: String,
}

/// Build a replaced string from the file content given a byte range and replacement.
fn build_replacement(
    file_content: &str,
    byte_start: usize,
    byte_len: usize,
    new_str: &str,
) -> ReplaceResult {
    let start_line = line_number_of_match(file_content, byte_start);
    let matched_region = file_content
        .get(byte_start..byte_start + byte_len)
        .unwrap_or("");
    let replaced_lines = line_count(matched_region);
    let first_matched_line = matched_region.lines().next().unwrap_or("");
    let adjusted_new = apply_indentation(first_matched_line, new_str);

    let mut result = String::with_capacity(file_content.len());
    result.push_str(file_content.get(..byte_start).unwrap_or(""));
    result.push_str(&adjusted_new);
    result.push_str(file_content.get(byte_start + byte_len..).unwrap_or(""));
    ReplaceResult {
        content: result,
        start_line,
        replaced_lines,
        matched_text: matched_region.to_string(),
        adjusted_new,
    }
}

/// Minimum fraction of lines that must match (after normalization) for a
/// fuzzy window to be considered a candidate. 0.6 allows up to 40% of lines
/// to differ, which handles common LLM errors (wrong variable names, slight
/// reformatting) while still being selective enough to avoid false positives.
const FUZZY_MATCH_THRESHOLD: f64 = 0.6;

/// Compute the fraction of lines in `needle_lines` that match `window_lines`
/// after whitespace normalization. Both slices must have the same length.
fn line_similarity(window_lines: &[&str], needle_lines: &[&str]) -> f64 {
    if window_lines.is_empty() {
        return 0.0;
    }
    let matching = window_lines
        .iter()
        .zip(needle_lines.iter())
        .filter(|(w, n)| normalize_whitespace(w) == normalize_whitespace(n))
        .count();
    #[expect(
        clippy::cast_precision_loss,
        reason = "line counts are small; f64 mantissa overflow is not a concern"
    )]
    {
        matching as f64 / window_lines.len() as f64
    }
}

/// Find the best fuzzy match: a window of N lines where at least
/// `FUZZY_MATCH_THRESHOLD` of lines match after normalization.
///
/// Returns `Some((byte_start, byte_len, score))` if exactly one window
/// achieves the best score above the threshold. Returns `None` if no window
/// passes the threshold, or if multiple windows tie for the best score
/// (ambiguous).
fn find_fuzzy_match(haystack: &str, needle: &str) -> Option<(usize, usize)> {
    let needle_lines: Vec<&str> = needle.lines().collect();
    if needle_lines.is_empty() {
        return None;
    }

    let source_lines = split_source_lines(haystack);
    if source_lines.len() < needle_lines.len() {
        return None;
    }

    let mut best_score: f64 = 0.0;
    let mut best_match: Option<(usize, usize)> = None;
    let mut best_is_unique = true;

    for start_idx in 0..=source_lines.len() - needle_lines.len() {
        let window_slice = source_lines
            .get(start_idx..start_idx + needle_lines.len())
            .unwrap_or(&[]);
        let window_texts: Vec<&str> = window_slice.iter().map(|sl| sl.text).collect();
        let score = line_similarity(&window_texts, &needle_lines);

        if score >= FUZZY_MATCH_THRESHOLD {
            if score > best_score {
                best_score = score;
                let byte_start = window_slice.first().map_or(0, |sl| sl.byte_offset);
                let byte_len = if let (Some(first), Some(last)) =
                    (window_slice.first(), window_slice.last())
                {
                    (last.byte_offset + last.text.len()) - first.byte_offset
                } else {
                    0
                };
                best_match = Some((byte_start, byte_len));
                best_is_unique = true;
            } else if (score - best_score).abs() < f64::EPSILON {
                // Tie — ambiguous, can't pick one
                best_is_unique = false;
            }
        }
    }

    if best_is_unique { best_match } else { None }
}

/// Find the replacement for the given old/new strings in the file content.
/// Returns the replaced file content and metadata, or a `ToolError`.
///
/// Matching strategy (layered, most precise first):
/// 1. **Exact match** — literal string comparison
/// 2. **Whitespace-normalized match** — collapse whitespace per line, compare
/// 3. **Fuzzy line-level match** — per-line similarity scoring with threshold
///
/// Each layer requires a unique match. Ambiguous matches (>1 candidate)
/// always return an error asking the agent to provide more context.
fn find_and_replace(
    file_content: &str,
    old_str: &str,
    new_str: &str,
    relative_path: &str,
) -> Result<ReplaceResult, ToolError> {
    let exact_count = file_content.matches(old_str).count();
    match exact_count {
        1 => {
            let byte_offset =
                file_content
                    .find(old_str)
                    .ok_or_else(|| ToolError::ExecutionFailed {
                        tool: SearchReplaceEditTool::NAME.into(),
                        reason: "internal: exact match count was 1 but find returned None".into(),
                    })?;
            Ok(build_replacement(
                file_content,
                byte_offset,
                old_str.len(),
                new_str,
            ))
        }
        0 => {
            // Layer 2: whitespace-normalized match
            let norm_matches = find_normalized_matches(file_content, old_str);
            match norm_matches.len() {
                1 => {
                    if let Some(&(byte_start, byte_len)) = norm_matches.first() {
                        return Ok(build_replacement(
                            file_content,
                            byte_start,
                            byte_len,
                            new_str,
                        ));
                    }
                }
                n if n > 1 => {
                    return Err(ToolError::ExecutionFailed {
                        tool: SearchReplaceEditTool::NAME.into(),
                        reason: format!(
                            "old_string matches {n} locations in {relative_path} after whitespace normalization, provide more surrounding context"
                        ),
                    });
                }
                _ => {} // 0 normalized matches — fall through to fuzzy
            }

            // Layer 3: fuzzy line-level match
            if let Some((byte_start, byte_len)) = find_fuzzy_match(file_content, old_str) {
                return Ok(build_replacement(
                    file_content,
                    byte_start,
                    byte_len,
                    new_str,
                ));
            }

            Err(ToolError::ExecutionFailed {
                tool: SearchReplaceEditTool::NAME.into(),
                reason: format!("old_string not found in {relative_path}"),
            })
        }
        n => Err(ToolError::ExecutionFailed {
            tool: SearchReplaceEditTool::NAME.into(),
            reason: format!(
                "old_string matches {n} locations in {relative_path}, provide more surrounding context"
            ),
        }),
    }
}

// ── Syntax Validation ────────────────────────────────────────────

/// A syntax error found by tree-sitter.
struct SyntaxError {
    /// 1-indexed line number.
    line: usize,
    /// 1-indexed column number.
    column: usize,
    /// Node kind: `"ERROR"` or `"MISSING"`.
    kind: &'static str,
}

/// Maximum syntax errors to collect before stopping the walk.
const MAX_SYNTAX_ERRORS: usize = 5;

/// Resolve a file extension to a tree-sitter language.
///
/// Returns `None` for unsupported extensions — validation is skipped.
fn language_for_path(path: &Path) -> Option<tree_sitter::Language> {
    let ext = path.extension().and_then(|e| e.to_str())?;
    match ext {
        "rs" => Some(tree_sitter_rust::LANGUAGE.into()),
        _ => None,
    }
}

/// Walk the tree collecting ERROR and MISSING nodes via depth-first traversal.
fn collect_syntax_errors(tree: &tree_sitter::Tree) -> Vec<SyntaxError> {
    let mut errors = Vec::with_capacity(MAX_SYNTAX_ERRORS);
    let mut cursor = tree.walk();

    loop {
        let node = cursor.node();
        if node.is_error() || node.is_missing() {
            let pos = node.start_position();
            errors.push(SyntaxError {
                line: pos.row + 1,
                column: pos.column + 1,
                kind: if node.is_error() { "ERROR" } else { "MISSING" },
            });
            if errors.len() >= MAX_SYNTAX_ERRORS {
                break;
            }
        }

        // Depth-first: try child → sibling → parent's sibling
        if cursor.goto_first_child() {
            continue;
        }
        while !cursor.goto_next_sibling() {
            if !cursor.goto_parent() {
                return errors;
            }
        }
    }

    errors
}

/// Validate that edited content has no syntax errors for supported languages.
///
/// Returns `Ok(())` for unsupported file types (validation skipped).
fn validate_syntax(path: &Path, content: &str) -> Result<(), ToolError> {
    let Some(language) = language_for_path(path) else {
        return Ok(());
    };

    let mut parser = tree_sitter::Parser::new();
    parser
        .set_language(&language)
        .map_err(|_| ToolError::ExecutionFailed {
            tool: SearchReplaceEditTool::NAME.into(),
            reason: "failed to initialize syntax parser".into(),
        })?;

    let Some(tree) = parser.parse(content.as_bytes(), None) else {
        // parse() returns None on cancellation — treat as skip
        return Ok(());
    };

    let errors = collect_syntax_errors(&tree);
    if errors.is_empty() {
        return Ok(());
    }

    let error_count = errors.len();
    let mut msg = String::from("Edit rejected — syntax errors detected (file not modified):\n");
    for err in &errors {
        let _ = writeln!(msg, "  line {}:{} — {}", err.line, err.column, err.kind);
    }

    tracing::debug!(
        path = %path.display(),
        error_count,
        "syntax validation rejected edit"
    );

    Err(ToolError::ExecutionFailed {
        tool: SearchReplaceEditTool::NAME.into(),
        reason: msg,
    })
}

// ── Checkpoint Name Validation ──────────────────────────────────

/// Validate a checkpoint name: alphanumeric, hyphens, underscores, 1–64 chars.
fn validate_checkpoint_name(name: &str, tool_name: &str) -> Result<(), ToolError> {
    if name.is_empty() || name.len() > 64 {
        return Err(ToolError::InvalidInput {
            tool: tool_name.into(),
            reason: "checkpoint name must be 1–64 characters".into(),
        });
    }
    if !name
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
    {
        return Err(ToolError::InvalidInput {
            tool: tool_name.into(),
            reason: "checkpoint name must contain only alphanumeric characters, hyphens, and underscores".into(),
        });
    }
    Ok(())
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
            f.set_len((super::MAX_EDIT_FILE_BYTES + 1) as u64).unwrap();
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

    // ── Unit tests for internal helpers ──────────────────────────

    #[test]
    fn test_split_source_lines_lf() {
        let lines = split_source_lines("abc\ndef\n");
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0].text, "abc");
        assert_eq!(lines[0].byte_offset, 0);
        assert_eq!(lines[1].text, "def");
        assert_eq!(lines[1].byte_offset, 4);
    }

    #[test]
    fn test_split_source_lines_crlf() {
        let lines = split_source_lines("abc\r\ndef\r\n");
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0].text, "abc");
        assert_eq!(lines[0].byte_offset, 0);
        assert_eq!(lines[1].text, "def");
        assert_eq!(lines[1].byte_offset, 5); // "abc\r\n" = 5 bytes
    }

    #[test]
    fn test_split_source_lines_no_trailing_newline() {
        let lines = split_source_lines("abc\ndef");
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0].text, "abc");
        assert_eq!(lines[1].text, "def");
        assert_eq!(lines[1].byte_offset, 4);
    }

    #[test]
    fn test_split_source_lines_empty() {
        let lines = split_source_lines("");
        assert!(lines.is_empty());
    }

    #[test]
    fn test_line_similarity_perfect() {
        let score = line_similarity(&["let x = 1;", "let y = 2;"], &["let x = 1;", "let y = 2;"]);
        assert!((score - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_line_similarity_partial() {
        let score = line_similarity(
            &["let x = 1;", "let y = 2;", "let z = 3;"],
            &["let x = 1;", "let WRONG = 2;", "let z = 3;"],
        );
        // 2 out of 3 match
        assert!((score - 2.0 / 3.0).abs() < 0.01);
    }

    #[test]
    fn test_line_similarity_none() {
        let score = line_similarity(&["aaa", "bbb"], &["xxx", "yyy"]);
        assert!((score - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_detect_indent_char_spaces() {
        assert_eq!(detect_indent_char("    "), ' ');
        assert_eq!(detect_indent_char("  "), ' ');
    }

    #[test]
    fn test_detect_indent_char_tabs() {
        assert_eq!(detect_indent_char("\t\t"), '\t');
        assert_eq!(detect_indent_char("\t"), '\t');
    }

    #[test]
    fn test_detect_indent_char_mixed_prefers_majority() {
        // More tabs than spaces → tab
        assert_eq!(detect_indent_char("\t\t "), '\t');
        // More spaces than tabs → space
        assert_eq!(detect_indent_char("\t  "), ' ');
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
            output.content.contains("│ ddd"),
            "missing removed line content"
        );
        assert!(
            output.content.contains("│ DDD"),
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

        assert!(output.content.contains("│ ccc"), "missing -ccc");
        assert!(output.content.contains("│ ddd"), "missing -ddd");
        assert!(output.content.contains("│ eee"), "missing -eee");
        assert!(output.content.contains("│ CCC"), "missing +CCC");
        assert!(output.content.contains("│ DDD"), "missing +DDD");
        assert!(output.content.contains("│ EEE"), "missing +EEE");
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
            output.content.contains("- 3│ line3"),
            "line number 3 should prefix removed line"
        );
        assert!(
            output.content.contains("+ 3│ LINE3"),
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
        let expected = "  2│ bbb\n  3│ ccc\n- 4│ ddd\n+ 4│ DDD\n  5│ eee\n  6│ fff";
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
            output.content.contains("│ second"),
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
            output.content.contains("│ fourth"),
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
            .skip_while(|l| !l.contains("│"))
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

            /// Normalize whitespace is idempotent — normalizing twice
            /// gives the same result as normalizing once.
            #[test]
            fn normalize_whitespace_idempotent(input in "[ \\t\\n\\x20-\\x7E]{0,200}") {
                let once = normalize_whitespace(&input);
                let twice = normalize_whitespace(&once);
                prop_assert_eq!(once, twice);
            }

            /// split_source_lines byte offsets reconstruct original text.
            #[test]
            fn source_lines_cover_text(input in "[\\x20-\\x7E\\n]{1,200}") {
                let lines = split_source_lines(&input);
                for sl in &lines {
                    // Each line's text must be at its stated byte offset
                    let actual = input.get(sl.byte_offset..sl.byte_offset + sl.text.len());
                    prop_assert_eq!(actual, Some(sl.text),
                        "offset {} len {} doesn't match", sl.byte_offset, sl.text.len());
                }
            }

            /// validate_syntax never panics on arbitrary input.
            #[test]
            fn validate_syntax_never_panics(content in "\\PC{0,2000}") {
                let path = Path::new("test.rs");
                // Must not panic — may return Ok or Err, both are fine
                let _ = validate_syntax(path, &content);
            }
        }
    }

    // ── Syntax validation unit tests ─────────────────────────────

    #[test]
    fn test_language_for_path_rs() {
        let path = Path::new("src/main.rs");
        assert!(language_for_path(path).is_some());
    }

    #[test]
    fn test_language_for_path_unknown() {
        assert!(language_for_path(Path::new("file.txt")).is_none());
        assert!(language_for_path(Path::new("file.py")).is_none());
        assert!(language_for_path(Path::new("file.js")).is_none());
        assert!(language_for_path(Path::new("no_extension")).is_none());
    }

    #[test]
    fn test_validate_syntax_valid_rust() {
        let path = Path::new("test.rs");
        let content = "fn main() {\n    let x = 42;\n}\n";
        assert!(validate_syntax(path, content).is_ok());
    }

    #[test]
    fn test_validate_syntax_invalid_rust() {
        let path = Path::new("test.rs");
        let content = "fn main( {\n    let x = 42;\n}\n";
        let err = validate_syntax(path, content).unwrap_err();
        match err {
            ToolError::ExecutionFailed { reason, .. } => {
                assert!(reason.contains("syntax errors detected"), "got: {reason}");
                assert!(reason.contains("file not modified"), "got: {reason}");
            }
            other => panic!("expected ExecutionFailed, got: {other:?}"),
        }
    }

    #[test]
    fn test_validate_syntax_unknown_extension() {
        let path = Path::new("readme.txt");
        let content = "this is not valid rust at all fn {{{";
        assert!(validate_syntax(path, content).is_ok());
    }

    #[test]
    fn test_validate_syntax_empty_content() {
        let path = Path::new("empty.rs");
        assert!(validate_syntax(path, "").is_ok());
    }

    #[test]
    fn test_collect_syntax_errors_multiple() {
        let mut parser = tree_sitter::Parser::new();
        parser
            .set_language(&tree_sitter_rust::LANGUAGE.into())
            .unwrap();
        // Multiple syntax errors: broken function signatures
        let content = "fn a( { } fn b( { } fn c( { } fn d( { } fn e( { } fn f( { }";
        let tree = parser.parse(content.as_bytes(), None).unwrap();
        let errors = collect_syntax_errors(&tree);
        assert!(
            !errors.is_empty(),
            "should detect syntax errors in broken code"
        );
        assert!(
            errors.len() <= MAX_SYNTAX_ERRORS,
            "should cap at {MAX_SYNTAX_ERRORS}, got {}",
            errors.len()
        );
    }

    #[test]
    fn test_error_message_includes_line_column() {
        let path = Path::new("test.rs");
        // Missing closing paren on line 2
        let content = "fn main() {\n    let x = foo(;\n}\n";
        let err = validate_syntax(path, content).unwrap_err();
        match err {
            ToolError::ExecutionFailed { reason, .. } => {
                // Should contain "line N:M" format
                assert!(
                    reason.contains("line "),
                    "error should include line numbers, got: {reason}"
                );
            }
            other => panic!("expected ExecutionFailed, got: {other:?}"),
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

    #[test]
    fn test_syntax_validation_latency() {
        // Generate a large valid Rust file
        let mut content = String::with_capacity(300_000);
        for i in 0..10_000 {
            use std::fmt::Write;
            let _ = writeln!(content, "fn func_{i}() {{ let _x = {i}; }}");
        }

        let path = Path::new("big.rs");
        let start = std::time::Instant::now();
        let result = validate_syntax(path, &content);
        let elapsed = start.elapsed();

        assert!(result.is_ok(), "valid large file should pass");
        // In release mode tree-sitter parses this in <10ms.
        // Debug builds are ~20x slower, and CI/parallel workloads add more.
        assert!(
            elapsed.as_millis() < 2000,
            "validation took {}ms, expected <2000ms (debug)",
            elapsed.as_millis()
        );
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
