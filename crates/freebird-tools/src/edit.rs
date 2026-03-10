//! Search/replace edit tool for surgical code modifications.
//!
//! The LLM provides `path`, `old_string` (exact text to find), and `new_string`
//! (replacement text). No line numbers — the model specifies literal code blocks,
//! which avoids off-by-one errors that plague diff-based formats.
//!
//! Research shows search/replace format has a 23–27 percentage point improvement
//! over unified diff and line-based diff formats (Meta agentic repair paper).

use async_trait::async_trait;

use freebird_security::taint::TaintedToolInput;
use freebird_traits::tool::{
    Capability, RiskLevel, SideEffects, Tool, ToolContext, ToolError, ToolInfo, ToolOutcome,
    ToolOutput,
};

/// Returns the search/replace edit tool as a trait object.
#[must_use]
pub fn edit_tools() -> Vec<Box<dyn Tool>> {
    vec![Box::new(SearchReplaceEditTool::new())]
}

// ── SearchReplaceEditTool ──────────────────────────────────────────

struct SearchReplaceEditTool {
    info: ToolInfo,
}

impl SearchReplaceEditTool {
    const NAME: &str = "search_replace_edit";

    fn new() -> Self {
        Self {
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
    haystack[..byte_offset]
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
    s.lines()
        .map(|line| line.split_whitespace().collect::<Vec<_>>().join(" "))
        .collect::<Vec<_>>()
        .join("\n")
}

/// A line from the original haystack, remembering its byte offset.
struct SourceLine<'a> {
    text: &'a str,
    byte_offset: usize,
}

/// Split `haystack` into lines that remember their byte offsets.
/// Handles both `\n` and `\r\n`. Does NOT use `str::lines()` because that
/// erases terminator length information needed for correct byte-offset math.
fn split_source_lines(haystack: &str) -> Vec<SourceLine<'_>> {
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
    // If file has no trailing newline, split_inclusive won't have caught the
    // last segment. Check if we've consumed all bytes.
    if offset < haystack.len() {
        let remaining = haystack.get(offset..).unwrap_or("");
        if !remaining.is_empty() {
            lines.push(SourceLine {
                text: remaining,
                byte_offset: offset,
            });
        }
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

/// Apply indentation preservation: detect the indentation delta between the
/// matched block and the replacement, then shift all replacement lines.
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

    let match_prefix = &matched_first_line[..match_indent];

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
                let line_indent = line.len() - line.trim_start().len();
                let relative = line_indent.saturating_sub(new_indent);
                format!(
                    "{}{}{}",
                    match_prefix,
                    " ".repeat(relative),
                    line.trim_start()
                )
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

        // Read file content
        let file_content = tokio::fs::read_to_string(safe_path.as_path())
            .await
            .map_err(|e| ToolError::ExecutionFailed {
                tool: Self::NAME.into(),
                reason: e.to_string(),
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

        Ok(ToolOutput {
            content: format!(
                "Edited {relative}: replaced {} lines starting at line {}",
                result.replaced_lines, result.start_line
            ),
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

/// Result of a successful match+replace operation.
struct ReplaceResult {
    content: String,
    start_line: usize,
    replaced_lines: usize,
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
    }
}

/// Find the replacement for the given old/new strings in the file content.
/// Returns the replaced file content and metadata, or a `ToolError`.
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
            // Try whitespace-normalized fallback
            let norm_matches = find_normalized_matches(file_content, old_str);
            match norm_matches.first() {
                None => Err(ToolError::ExecutionFailed {
                    tool: SearchReplaceEditTool::NAME.into(),
                    reason: format!("old_string not found in {relative_path}"),
                }),
                Some(&(byte_start, byte_len)) if norm_matches.len() == 1 => Ok(build_replacement(
                    file_content,
                    byte_start,
                    byte_len,
                    new_str,
                )),
                Some(_) => Err(ToolError::ExecutionFailed {
                    tool: SearchReplaceEditTool::NAME.into(),
                    reason: format!(
                        "old_string matches {} locations in {relative_path} after whitespace normalization, provide more surrounding context",
                        norm_matches.len()
                    ),
                }),
            }
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
            }
        }
    }

    // ── Factory test ─────────────────────────────────────────────

    #[test]
    fn test_edit_tool_info() {
        let tool = SearchReplaceEditTool::new();
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

        let tool = SearchReplaceEditTool::new();
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

        let tool = SearchReplaceEditTool::new();
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

        let tool = SearchReplaceEditTool::new();
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

        let tool = SearchReplaceEditTool::new();
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

        let tool = SearchReplaceEditTool::new();
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

        let tool = SearchReplaceEditTool::new();
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

        let tool = SearchReplaceEditTool::new();
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

        let tool = SearchReplaceEditTool::new();
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

        let tool = SearchReplaceEditTool::new();
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

        let tool = SearchReplaceEditTool::new();
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

        let tool = SearchReplaceEditTool::new();
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

        let tool = SearchReplaceEditTool::new();
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
        let tool = SearchReplaceEditTool::new();

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
        let tool = SearchReplaceEditTool::new();

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

        let tool = SearchReplaceEditTool::new();
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

        let tool = SearchReplaceEditTool::new();
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

        let tool = SearchReplaceEditTool::new();
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

        let tool = SearchReplaceEditTool::new();
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

        let tool = SearchReplaceEditTool::new();
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
        let tools = edit_tools();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].info().name, "search_replace_edit");

        // Verify to_definition() produces a valid tool definition
        let def = tools[0].to_definition();
        assert_eq!(def.name, "search_replace_edit");
        assert!(!def.description.is_empty());
        assert!(def.input_schema.is_object());
    }
}
