//! Search/replace edit tool for surgical code modifications.
//!
//! The LLM provides `path`, `old_string` (exact text to find), and `new_string`
//! (replacement text). No line numbers — the model specifies literal code blocks,
//! which avoids off-by-one errors that plague diff-based formats.
//!
//! Research shows search/replace format has a 23–27 percentage point improvement
//! over unified diff and line-based diff formats (Meta agentic repair paper).

use std::fmt::Write;

use async_trait::async_trait;
use tokio::io::AsyncReadExt;

use freebird_security::taint::TaintedToolInput;
use freebird_traits::tool::{
    Capability, RiskLevel, SideEffects, Tool, ToolContext, ToolError, ToolInfo, ToolOutcome,
    ToolOutput,
};
use freebird_types::config::EditConfig;

/// Maximum file size the edit tool will read (10 MiB).
///
/// Matches `read_file`'s limit. Prevents OOM on huge files — the LLM
/// context window can't usefully represent files larger than this anyway.
const MAX_EDIT_FILE_BYTES: usize = 10 * 1024 * 1024;

/// Returns the search/replace edit tool as a trait object.
#[must_use]
pub fn edit_tools(config: &EditConfig) -> Vec<Box<dyn Tool>> {
    vec![Box::new(SearchReplaceEditTool::new(config))]
}

// ── SearchReplaceEditTool ──────────────────────────────────────────

struct SearchReplaceEditTool {
    info: ToolInfo,
    diff_preview: bool,
    diff_context_lines: usize,
}

impl SearchReplaceEditTool {
    const NAME: &str = "search_replace_edit";

    fn new(config: &EditConfig) -> Self {
        Self {
            diff_preview: config.diff_preview,
            diff_context_lines: config.diff_context_lines,
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

        // Read file content with size cap (same pattern as read_file)
        let file = tokio::fs::File::open(safe_path.as_path())
            .await
            .map_err(|e| ToolError::ExecutionFailed {
                tool: Self::NAME.into(),
                reason: e.to_string(),
            })?;

        let cap = MAX_EDIT_FILE_BYTES + 1;
        let mut buf = Vec::with_capacity(cap.min(8 * 1024));
        file.take(cap as u64)
            .read_to_end(&mut buf)
            .await
            .map_err(|e| ToolError::ExecutionFailed {
                tool: Self::NAME.into(),
                reason: e.to_string(),
            })?;

        if buf.len() > MAX_EDIT_FILE_BYTES {
            return Err(ToolError::ExecutionFailed {
                tool: Self::NAME.into(),
                reason: format!("file exceeds {MAX_EDIT_FILE_BYTES} byte limit"),
            });
        }

        let file_content = String::from_utf8(buf).map_err(|_| ToolError::ExecutionFailed {
            tool: Self::NAME.into(),
            reason: "file is not valid UTF-8".into(),
        })?;

        let relative = relative_path_display(&safe_path);
        let result = find_and_replace(&file_content, old_str, new_str, &relative)?;

        // Atomic write: temp file + rename
        let file_name = safe_path
            .as_path()
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("file");
        let tmp_path = safe_path.as_path().with_file_name(format!(
            ".{}.{}.tmp",
            file_name,
            std::process::id()
        ));

        tokio::fs::write(&tmp_path, &result.content)
            .await
            .map_err(|e| ToolError::ExecutionFailed {
                tool: Self::NAME.into(),
                reason: e.to_string(),
            })?;

        if let Err(e) = tokio::fs::rename(&tmp_path, safe_path.as_path()).await {
            let _ = tokio::fs::remove_file(&tmp_path).await;
            return Err(ToolError::ExecutionFailed {
                tool: Self::NAME.into(),
                reason: e.to_string(),
            });
        }

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

        Ok(ToolOutput {
            content: message,
            outcome: ToolOutcome::Success,
            metadata: None,
        })
    }
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
    #[allow(clippy::cast_precision_loss)]
    // line counts are small; f64 mantissa overflow is not a concern
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

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic, clippy::indexing_slicing)]
mod tests {
    use std::path::PathBuf;

    use freebird_traits::id::SessionId;
    use freebird_traits::tool::{Capability, RiskLevel, SideEffects, Tool, ToolContext, ToolError};

    use super::*;

    /// Test harness — same pattern as filesystem.rs.
    struct TestHarness {
        _tmp: tempfile::TempDir,
        sandbox: PathBuf,
        session_id: SessionId,
        capabilities: Vec<Capability>,
        allowed_directories: Vec<PathBuf>,
    }

    impl TestHarness {
        fn new() -> Self {
            let tmp = tempfile::tempdir().unwrap();
            let sandbox = tmp.path().to_path_buf();
            Self {
                _tmp: tmp,
                sandbox,
                session_id: SessionId::from_string("test-session"),
                capabilities: vec![Capability::FileRead, Capability::FileWrite],
                allowed_directories: vec![],
            }
        }

        fn path(&self) -> &std::path::Path {
            &self.sandbox
        }

        fn context(&self) -> ToolContext<'_> {
            ToolContext {
                session_id: &self.session_id,
                sandbox_root: &self.sandbox,
                granted_capabilities: &self.capabilities,
                allowed_directories: &self.allowed_directories,
                knowledge_store: None,
            }
        }
    }

    // ── Factory test ─────────────────────────────────────────────

    #[test]
    fn test_edit_tool_info() {
        let tool = SearchReplaceEditTool::new(&EditConfig::default());
        let info = tool.info();
        assert_eq!(info.name, "search_replace_edit");
        assert_eq!(info.required_capability, Capability::FileWrite);
        assert_eq!(info.risk_level, RiskLevel::Medium);
        assert!(matches!(info.side_effects, SideEffects::HasSideEffects));
    }

    // ── Core edit tests ──────────────────────────────────────────

    #[tokio::test]
    async fn test_exact_match_replaces() {
        let h = TestHarness::new();
        std::fs::write(
            h.path().join("file.rs"),
            "fn hello() {\n    println!(\"hi\");\n}\n",
        )
        .unwrap();

        let tool = SearchReplaceEditTool::new(&EditConfig::default());
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
        let h = TestHarness::new();
        std::fs::write(h.path().join("file.rs"), "fn main() {}\n").unwrap();

        let tool = SearchReplaceEditTool::new(&EditConfig::default());
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
        let h = TestHarness::new();
        std::fs::write(
            h.path().join("file.rs"),
            "let x = 1;\nlet y = 1;\nlet z = 1;\n",
        )
        .unwrap();

        let tool = SearchReplaceEditTool::new(&EditConfig::default());
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
        let h = TestHarness::new();
        std::fs::write(h.path().join("file.rs"), "line1\nDELETE_ME\nline3\n").unwrap();

        let tool = SearchReplaceEditTool::new(&EditConfig::default());
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
        let h = TestHarness::new();
        std::fs::write(h.path().join("file.rs"), "content\n").unwrap();

        let tool = SearchReplaceEditTool::new(&EditConfig::default());
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
        let h = TestHarness::new();
        let original = "fn main() {\n    let a = 1;\n    let b = 2;\n    let c = 3;\n    let d = 4;\n    let e = 5;\n}\n";
        std::fs::write(h.path().join("file.rs"), original).unwrap();

        let tool = SearchReplaceEditTool::new(&EditConfig::default());
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
        let h = TestHarness::new();
        let original = "BEFORE\nTARGET\nAFTER\n";
        std::fs::write(h.path().join("file.txt"), original).unwrap();

        let tool = SearchReplaceEditTool::new(&EditConfig::default());
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
        let h = TestHarness::new();
        // File has single spaces
        std::fs::write(h.path().join("file.rs"), "fn main() {\n    let x = 1;\n}\n").unwrap();

        let tool = SearchReplaceEditTool::new(&EditConfig::default());
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
        let h = TestHarness::new();
        // File has 2-space indent
        std::fs::write(h.path().join("file.rs"), "fn main() {\n  let x = 1;\n}\n").unwrap();

        let tool = SearchReplaceEditTool::new(&EditConfig::default());
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
        let h = TestHarness::new();
        std::fs::write(
            h.path().join("file.rs"),
            "let x = 1;\nlet  x  =  1;\nlet x=1;\n",
        )
        .unwrap();

        let tool = SearchReplaceEditTool::new(&EditConfig::default());
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
        let h = TestHarness::new();
        let original = "fn main() {\n    if true {\n        let x = 1;\n    }\n}\n";
        std::fs::write(h.path().join("file.rs"), original).unwrap();

        let tool = SearchReplaceEditTool::new(&EditConfig::default());
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
        let h = TestHarness::new();
        // File has 8-space indented block
        let original =
            "fn main() {\n    if true {\n        old_line_1\n        old_line_2\n    }\n}\n";
        std::fs::write(h.path().join("file.rs"), original).unwrap();

        let tool = SearchReplaceEditTool::new(&EditConfig::default());
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
        let h = TestHarness::new();
        let tool = SearchReplaceEditTool::new(&EditConfig::default());

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
        let h = TestHarness::new();
        let tool = SearchReplaceEditTool::new(&EditConfig::default());

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
        let h = TestHarness::new();
        std::fs::write(h.path().join("src.rs"), "old\n").unwrap();

        let tool = SearchReplaceEditTool::new(&EditConfig::default());
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
        let h = TestHarness::new();
        std::fs::write(h.path().join("clean.rs"), "old_text\n").unwrap();

        let tool = SearchReplaceEditTool::new(&EditConfig::default());
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
        let h = TestHarness::new();
        let original = "unchanged content\n";
        std::fs::write(h.path().join("file.rs"), original).unwrap();

        let tool = SearchReplaceEditTool::new(&EditConfig::default());
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
        let h = TestHarness::new();
        // Write a file with \r\n line endings
        std::fs::write(
            h.path().join("crlf.rs"),
            "fn main() {\r\n    let x = 1;\r\n}\r\n",
        )
        .unwrap();

        let tool = SearchReplaceEditTool::new(&EditConfig::default());
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
        let h = TestHarness::new();
        std::fs::write(h.path().join("file.rs"), "content\n").unwrap();

        let tool = SearchReplaceEditTool::new(&EditConfig::default());
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
        let tools = edit_tools(&EditConfig::default());
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].info().name, "search_replace_edit");

        // Verify to_definition() produces a valid tool definition
        let def = tools[0].to_definition();
        assert_eq!(def.name, "search_replace_edit");
        assert!(!def.description.is_empty());
        assert!(def.input_schema.is_object());
    }

    // ── File size limit tests ────────────────────────────────────

    #[tokio::test]
    async fn test_file_exceeding_size_limit_rejected() {
        let h = TestHarness::new();
        let file_path = h.path().join("huge.rs");
        {
            let f = std::fs::File::create(&file_path).unwrap();
            // Set file size to just over 10 MiB without writing all bytes
            f.set_len((super::MAX_EDIT_FILE_BYTES + 1) as u64).unwrap();
        }

        let tool = SearchReplaceEditTool::new(&EditConfig::default());
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
        let h = TestHarness::new();
        let file_path = h.path().join("binary.bin");
        {
            let mut f = std::fs::File::create(&file_path).unwrap();
            f.write_all(&[0xFF, 0xFE, 0x00, 0x01]).unwrap();
        }

        let tool = SearchReplaceEditTool::new(&EditConfig::default());
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
        let h = TestHarness::new();
        // File uses tab indentation
        std::fs::write(
            h.path().join("file.go"),
            "func main() {\n\toldLine1\n\toldLine2\n}\n",
        )
        .unwrap();

        let tool = SearchReplaceEditTool::new(&EditConfig::default());
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
        let h = TestHarness::new();
        let original =
            "fn main() {\n    let x = 1;\n    let y = 2;\n    let z = 3;\n    let w = 4;\n}\n";
        std::fs::write(h.path().join("file.rs"), original).unwrap();

        let tool = SearchReplaceEditTool::new(&EditConfig::default());
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
        let h = TestHarness::new();
        let original = "fn main() {\n    let x = 1;\n    let y = 2;\n    let z = 3;\n}\n";
        std::fs::write(h.path().join("file.rs"), original).unwrap();

        let tool = SearchReplaceEditTool::new(&EditConfig::default());
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
        let h = TestHarness::new();
        // Two identical blocks — fuzzy match should be ambiguous
        let original = "fn a() {\n    let x = 1;\n    let y = 2;\n}\nfn b() {\n    let x = 1;\n    let y = 2;\n}\n";
        std::fs::write(h.path().join("file.rs"), original).unwrap();

        let tool = SearchReplaceEditTool::new(&EditConfig::default());
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
        let h = TestHarness::new();
        std::fs::write(
            h.path().join("src.rs"),
            "aaa\nbbb\nccc\nddd\neee\nfff\nggg\n",
        )
        .unwrap();

        let tool = SearchReplaceEditTool::new(&EditConfig::default());
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
        let h = TestHarness::new();
        std::fs::write(
            h.path().join("src.rs"),
            "line1\nline2\nline3\nline4\nline5\nline6\nline7\nline8\nline9\n",
        )
        .unwrap();

        let tool = SearchReplaceEditTool::new(&EditConfig {
            diff_preview: true,
            diff_context_lines: 3,
        });
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
        let h = TestHarness::new();
        std::fs::write(
            h.path().join("src.rs"),
            "aaa\nbbb\nccc\nddd\neee\nfff\nggg\n",
        )
        .unwrap();

        let tool = SearchReplaceEditTool::new(&EditConfig::default());
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
        let h = TestHarness::new();
        std::fs::write(h.path().join("src.rs"), "aaa\nbbb\nccc\nddd\neee\n").unwrap();

        let tool = SearchReplaceEditTool::new(&EditConfig::default());
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
        let h = TestHarness::new();
        std::fs::write(h.path().join("src.rs"), "aaa\nbbb\nccc\nddd\neee\n").unwrap();

        let tool = SearchReplaceEditTool::new(&EditConfig::default());
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
        let h = TestHarness::new();
        std::fs::write(h.path().join("src.rs"), "aaa\nbbb\nccc\nddd\neee\n").unwrap();

        let tool = SearchReplaceEditTool::new(&EditConfig::default());
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
        let h = TestHarness::new();
        std::fs::write(
            h.path().join("src.rs"),
            "line1\nline2\nline3\nline4\nline5\n",
        )
        .unwrap();

        let tool = SearchReplaceEditTool::new(&EditConfig::default());
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
        let h = TestHarness::new();
        std::fs::write(h.path().join("src.rs"), "aaa\nbbb\nccc\n").unwrap();

        let tool = SearchReplaceEditTool::new(&EditConfig {
            diff_preview: false,
            diff_context_lines: 3,
        });
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
        let h = TestHarness::new();
        std::fs::write(
            h.path().join("src.rs"),
            "first\nsecond\nthird\nfourth\nfifth\n",
        )
        .unwrap();

        let tool = SearchReplaceEditTool::new(&EditConfig::default());
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
        let h = TestHarness::new();
        std::fs::write(
            h.path().join("src.rs"),
            "first\nsecond\nthird\nfourth\nlast",
        )
        .unwrap();

        let tool = SearchReplaceEditTool::new(&EditConfig::default());
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
        let h = TestHarness::new();
        std::fs::write(h.path().join("src.rs"), "aaa\nbbb\nccc\nddd\neee\n").unwrap();

        let tool = SearchReplaceEditTool::new(&EditConfig {
            diff_preview: true,
            diff_context_lines: 0,
        });
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
                    let h = TestHarness::new();
                    let tool = SearchReplaceEditTool::new(&EditConfig::default());

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
                    let h = TestHarness::new();
                    let content = format!("{prefix}\n{old_text}\n{suffix}\n");
                    std::fs::write(h.path().join(&name), &content).unwrap();

                    let tool = SearchReplaceEditTool::new(&EditConfig::default());
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
                    let h = TestHarness::new();
                    std::fs::write(h.path().join(&name), "old_unique_sentinel\n").unwrap();

                    let tool = SearchReplaceEditTool::new(&EditConfig::default());
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
        }
    }
}
