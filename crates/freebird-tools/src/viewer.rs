//! Windowed file viewer with line numbers and scroll/goto navigation.
//!
//! Returns a slice of a file (default 100 lines, max 300) with structural
//! metadata (path, total lines, current window position) that helps the
//! agent navigate codebases. Based on SWE-agent's ablation studies
//! (`NeurIPS` 2024) showing windowed viewing outperforms raw `cat` by 10.7pp.

use std::fmt::Write as _;

use async_trait::async_trait;
use tokio::io::AsyncReadExt;

use freebird_security::error::SecurityError;
use freebird_security::taint::TaintedToolInput;
use freebird_traits::tool::{
    Capability, RiskLevel, SideEffects, Tool, ToolContext, ToolError, ToolInfo, ToolOutcome,
    ToolOutput,
};

/// Maximum file size the viewer will read (10 MiB).
const MAX_READ_FILE_BYTES: usize = 10 * 1024 * 1024;

/// Default number of lines to display per window.
const DEFAULT_LIMIT: usize = 100;

/// Maximum number of lines a single view can return.
const MAX_LIMIT: usize = 300;

/// Number of context lines to show above a pattern match.
const PATTERN_CONTEXT_LINES: usize = 5;

/// Returns all viewer tools as trait objects.
#[must_use]
pub fn viewer_tools() -> Vec<Box<dyn Tool>> {
    vec![Box::new(FileViewerTool::new())]
}

// ── FileViewerTool ──────────────────────────────────────────────────

struct FileViewerTool {
    info: ToolInfo,
}

impl FileViewerTool {
    const NAME: &str = "file_viewer";

    fn new() -> Self {
        Self {
            info: ToolInfo {
                name: Self::NAME.into(),
                description: "View a window of lines from a file with line numbers. \
                    Returns a slice of the file (default 100 lines, max 300) with \
                    navigation metadata. Optionally jump to a pattern match."
                    .into(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Relative path within the sandbox, or absolute path within an allowed directory"
                        },
                        "offset": {
                            "type": "integer",
                            "description": "1-indexed line number to start viewing from (default: 1)",
                            "minimum": 1
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Number of lines to display (default: 100, max: 300)",
                            "minimum": 1,
                            "maximum": 300
                        },
                        "pattern": {
                            "type": "string",
                            "description": "String pattern to jump to. If found, offset is adjusted to show the first match with context."
                        }
                    },
                    "required": ["path"]
                }),
                required_capability: Capability::FileRead,
                risk_level: RiskLevel::Low,
                side_effects: SideEffects::None,
            },
        }
    }
}

#[async_trait]
impl Tool for FileViewerTool {
    fn info(&self) -> &ToolInfo {
        &self.info
    }

    async fn execute(
        &self,
        input: serde_json::Value,
        context: &ToolContext<'_>,
    ) -> Result<ToolOutput, ToolError> {
        let tainted = TaintedToolInput::new(input);
        let params = extract_params(&tainted, context)?;
        let file_content = read_file_content(&params.safe_path).await?;
        Ok(render_view(&file_content, &params))
    }
}

/// Validated parameters for a viewer invocation.
struct ViewerParams {
    safe_path: freebird_security::safe_types::SafeFilePath,
    offset: usize,
    limit: usize,
    pattern: Option<freebird_security::safe_types::SafeFileContent>,
}

fn extract_params(
    tainted: &TaintedToolInput,
    ctx: &ToolContext<'_>,
) -> Result<ViewerParams, ToolError> {
    let safe_path = tainted
        .extract_path_multi_root("path", ctx.sandbox_root, ctx.allowed_directories)
        .map_err(|e| ToolError::InvalidInput {
            tool: FileViewerTool::NAME.into(),
            reason: e.to_string(),
        })?;

    let raw_offset =
        tainted
            .extract_u64_optional("offset")
            .map_err(|e| ToolError::InvalidInput {
                tool: FileViewerTool::NAME.into(),
                reason: e.to_string(),
            })?;

    let raw_limit = tainted
        .extract_u64_optional("limit")
        .map_err(|e| ToolError::InvalidInput {
            tool: FileViewerTool::NAME.into(),
            reason: e.to_string(),
        })?;

    // Pattern is optional — MissingField means "no pattern provided".
    // Empty strings are treated as "no pattern" since `str::contains("")`
    // matches every line, which is never the intended behavior.
    let pattern = match tainted.extract_file_content("pattern") {
        Ok(pat) if !pat.as_str().is_empty() => Some(pat),
        Ok(_) | Err(SecurityError::MissingField { .. }) => None,
        Err(e) => {
            return Err(ToolError::InvalidInput {
                tool: FileViewerTool::NAME.into(),
                reason: e.to_string(),
            });
        }
    };

    let offset: usize = match raw_offset {
        Some(0) => {
            return Err(ToolError::InvalidInput {
                tool: FileViewerTool::NAME.into(),
                reason: "offset must be >= 1 (1-indexed)".into(),
            });
        }
        Some(n) => usize::try_from(n).map_err(|_| ToolError::InvalidInput {
            tool: FileViewerTool::NAME.into(),
            reason: "offset value out of range".into(),
        })?,
        None => 1,
    };

    let limit: usize = match raw_limit {
        Some(0) => {
            return Err(ToolError::InvalidInput {
                tool: FileViewerTool::NAME.into(),
                reason: "limit must be >= 1".into(),
            });
        }
        Some(n) => {
            let val = usize::try_from(n).map_err(|_| ToolError::InvalidInput {
                tool: FileViewerTool::NAME.into(),
                reason: "limit value out of range".into(),
            })?;
            val.min(MAX_LIMIT)
        }
        None => DEFAULT_LIMIT,
    };

    Ok(ViewerParams {
        safe_path,
        offset,
        limit,
        pattern,
    })
}

async fn read_file_content(
    safe_path: &freebird_security::safe_types::SafeFilePath,
) -> Result<String, ToolError> {
    let file = tokio::fs::File::open(safe_path.as_path())
        .await
        .map_err(|e| ToolError::ExecutionFailed {
            tool: FileViewerTool::NAME.into(),
            reason: e.to_string(),
        })?;

    let cap = MAX_READ_FILE_BYTES + 1;
    let mut buf = Vec::with_capacity(cap.min(8 * 1024));
    file.take(cap as u64)
        .read_to_end(&mut buf)
        .await
        .map_err(|e| ToolError::ExecutionFailed {
            tool: FileViewerTool::NAME.into(),
            reason: e.to_string(),
        })?;

    if buf.len() > MAX_READ_FILE_BYTES {
        return Err(ToolError::ExecutionFailed {
            tool: FileViewerTool::NAME.into(),
            reason: format!("file exceeds {MAX_READ_FILE_BYTES} byte limit"),
        });
    }

    String::from_utf8(buf).map_err(|_| ToolError::ExecutionFailed {
        tool: FileViewerTool::NAME.into(),
        reason: "file is not valid UTF-8".into(),
    })
}

fn render_view(file_content: &str, params: &ViewerParams) -> ToolOutput {
    let lines: Vec<&str> = file_content.lines().collect();
    let total_lines = lines.len();
    let mut offset = params.offset;

    // Relative path for output — never leak sandbox root
    let relative = params
        .safe_path
        .as_path()
        .strip_prefix(params.safe_path.root())
        .unwrap_or(params.safe_path.as_path());
    let rel_display = relative.display().to_string();

    // Handle pattern search
    let mut pattern_match_line: Option<usize> = None;
    if let Some(ref pat) = params.pattern {
        let pat_str = pat.as_str();
        let found = lines
            .iter()
            .enumerate()
            .find(|(_, line)| line.contains(pat_str));
        match found {
            Some((idx, _)) => {
                let match_line = idx + 1; // 1-indexed
                pattern_match_line = Some(match_line);
                offset = match_line.saturating_sub(PATTERN_CONTEXT_LINES).max(1);
            }
            None => {
                return ToolOutput {
                    content: format!(
                        "Pattern '{pat_str}' not found in {rel_display} ({total_lines} lines)"
                    ),
                    outcome: ToolOutcome::Success,
                    metadata: None,
                };
            }
        }
    }

    // Handle empty file
    if total_lines == 0 {
        let header = format_header(&rel_display, 0, 0, 0, None);
        let footer = format_footer(0, 0);
        return ToolOutput {
            content: format!("{header}\n\n{footer}"),
            outcome: ToolOutcome::Success,
            metadata: None,
        };
    }

    // Clamp offset to valid range
    if offset > total_lines {
        offset = total_lines.saturating_sub(params.limit).max(1);
    }

    // Compute window slice (0-indexed for Vec access)
    let start_idx = offset - 1;
    let end_idx = (start_idx + params.limit).min(total_lines);
    let window = lines.get(start_idx..end_idx).unwrap_or_default();

    let start_line = offset;
    // window.len() == end_idx - start_idx, so start_idx + window.len() gives
    // the 1-indexed inclusive end line (by coincidence of 0-indexed math).
    let end_line_inclusive = start_idx + window.len();

    let lines_above = start_idx;
    let lines_below = total_lines.saturating_sub(end_idx);

    let header = format_header(
        &rel_display,
        start_line,
        end_line_inclusive,
        total_lines,
        pattern_match_line,
    );
    let body = format_lines(window, start_line);
    let footer = format_footer(lines_above, lines_below);

    ToolOutput {
        content: format!("{header}\n\n{body}\n{footer}"),
        outcome: ToolOutcome::Success,
        metadata: None,
    }
}

// ── Output formatting helpers ───────────────────────────────────────

fn format_header(
    path: &str,
    start: usize,
    end: usize,
    total: usize,
    pattern_match: Option<usize>,
) -> String {
    let range_info = pattern_match.map_or_else(
        || format!("lines {start}-{end} of {total}"),
        |line| format!("lines {start}-{end} of {total}, jumped to pattern match at line {line}"),
    );
    format!("\u{2500}\u{2500}\u{2500} {path} ({range_info}) \u{2500}\u{2500}\u{2500}")
}

fn format_footer(above: usize, below: usize) -> String {
    let above_label = if above == 1 {
        "1 line above".to_string()
    } else {
        format!("{above} lines above")
    };
    let below_label = if below == 1 {
        "1 line below".to_string()
    } else {
        format!("{below} lines below")
    };
    format!(
        "\u{2500}\u{2500}\u{2500} {above_label} \u{00b7} {below_label} \u{2500}\u{2500}\u{2500}"
    )
}

fn format_lines(lines: &[&str], start_line: usize) -> String {
    if lines.is_empty() {
        return String::new();
    }

    let max_line_num = start_line + lines.len() - 1;
    let width = max_line_num.to_string().len();

    let mut out = String::new();
    for (i, line) in lines.iter().enumerate() {
        let line_num = start_line + i;
        let _ = writeln!(out, "{line_num:>width$}\u{2502} {line}");
    }

    // Remove trailing newline
    if out.ends_with('\n') {
        out.pop();
    }

    out
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic, clippy::indexing_slicing)]
mod tests {
    use std::path::PathBuf;

    use freebird_traits::id::SessionId;
    use freebird_traits::tool::{Capability, RiskLevel, SideEffects, Tool, ToolContext, ToolError};

    use super::*;

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
                capabilities: vec![Capability::FileRead],
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

    /// Generate a file with `n` numbered lines: "Line 1\nLine 2\n..."
    fn generate_numbered_file(h: &TestHarness, name: &str, n: usize) {
        let mut content = String::new();
        for i in 1..=n {
            let _ = writeln!(content, "Line {i}");
        }
        std::fs::write(h.path().join(name), content).unwrap();
    }

    // ── Core functionality ──────────────────────────────────────────

    #[tokio::test]
    async fn test_default_window_100_lines() {
        let h = TestHarness::new();
        generate_numbered_file(&h, "big.txt", 200);

        let tool = FileViewerTool::new();
        let output = tool
            .execute(serde_json::json!({"path": "big.txt"}), &h.context())
            .await
            .unwrap();

        assert_eq!(output.outcome, ToolOutcome::Success);
        assert!(output.content.contains("lines 1-100 of 200"));
        assert!(output.content.contains("Line 1"));
        assert!(output.content.contains("Line 100"));
        assert!(!output.content.contains("Line 101"));
    }

    #[tokio::test]
    async fn test_offset_starts_at_specified_line() {
        let h = TestHarness::new();
        generate_numbered_file(&h, "big.txt", 200);

        let tool = FileViewerTool::new();
        let output = tool
            .execute(
                serde_json::json!({"path": "big.txt", "offset": 50}),
                &h.context(),
            )
            .await
            .unwrap();

        assert!(output.content.contains("lines 50-149 of 200"));
        assert!(output.content.contains("Line 50"));
        assert!(output.content.contains("Line 149"));
        assert!(!output.content.contains("\n 49\u{2502}"));
    }

    #[tokio::test]
    async fn test_custom_limit() {
        let h = TestHarness::new();
        generate_numbered_file(&h, "big.txt", 100);

        let tool = FileViewerTool::new();
        let output = tool
            .execute(
                serde_json::json!({"path": "big.txt", "limit": 20}),
                &h.context(),
            )
            .await
            .unwrap();

        assert!(output.content.contains("lines 1-20 of 100"));
        assert!(output.content.contains("Line 20"));
        assert!(!output.content.contains("Line 21"));
    }

    #[tokio::test]
    async fn test_limit_clamped_at_300() {
        let h = TestHarness::new();
        generate_numbered_file(&h, "big.txt", 500);

        let tool = FileViewerTool::new();
        let output = tool
            .execute(
                serde_json::json!({"path": "big.txt", "limit": 500}),
                &h.context(),
            )
            .await
            .unwrap();

        assert!(output.content.contains("lines 1-300 of 500"));
        assert!(output.content.contains("Line 300"));
        assert!(!output.content.contains("Line 301"));
    }

    #[tokio::test]
    async fn test_offset_beyond_end() {
        let h = TestHarness::new();
        generate_numbered_file(&h, "small.txt", 10);

        let tool = FileViewerTool::new();
        let output = tool
            .execute(
                serde_json::json!({"path": "small.txt", "offset": 9999}),
                &h.context(),
            )
            .await
            .unwrap();

        assert_eq!(output.outcome, ToolOutcome::Success);
        // Should show the last available window
        assert!(output.content.contains("of 10"));
        assert!(output.content.contains("Line 10"));
    }

    // ── Header/footer format ────────────────────────────────────────

    #[tokio::test]
    async fn test_header_format() {
        let h = TestHarness::new();
        generate_numbered_file(&h, "test.rs", 50);

        let tool = FileViewerTool::new();
        let output = tool
            .execute(serde_json::json!({"path": "test.rs"}), &h.context())
            .await
            .unwrap();

        let first_line = output.content.lines().next().unwrap();
        assert!(first_line.starts_with('\u{2500}'));
        assert!(first_line.contains("test.rs"));
        assert!(first_line.contains("lines 1-50 of 50"));
    }

    #[tokio::test]
    async fn test_footer_shows_remaining() {
        let h = TestHarness::new();
        generate_numbered_file(&h, "big.txt", 200);

        let tool = FileViewerTool::new();
        let output = tool
            .execute(
                serde_json::json!({"path": "big.txt", "offset": 51, "limit": 50}),
                &h.context(),
            )
            .await
            .unwrap();

        let last_line = output.content.lines().last().unwrap();
        assert!(last_line.contains("50 lines above"));
        assert!(last_line.contains("100 lines below"));
    }

    #[tokio::test]
    async fn test_line_numbers_right_aligned() {
        let h = TestHarness::new();
        generate_numbered_file(&h, "aligned.txt", 200);

        let tool = FileViewerTool::new();
        let output = tool
            .execute(
                serde_json::json!({"path": "aligned.txt", "offset": 1, "limit": 10}),
                &h.context(),
            )
            .await
            .unwrap();

        // Lines 1-10, so max width is 2 digits. Line 1 should be " 1│"
        assert!(output.content.contains(" 1\u{2502} Line 1"));
        assert!(output.content.contains("10\u{2502} Line 10"));
    }

    // ── Pattern search ──────────────────────────────────────────────

    #[tokio::test]
    async fn test_pattern_jump() {
        let h = TestHarness::new();
        let mut content = String::new();
        for i in 1..=100 {
            if i == 50 {
                content.push_str("fn main() {\n");
            } else {
                let _ = writeln!(content, "Line {i}");
            }
        }
        std::fs::write(h.path().join("code.rs"), &content).unwrap();

        let tool = FileViewerTool::new();
        let output = tool
            .execute(
                serde_json::json!({"path": "code.rs", "pattern": "fn main"}),
                &h.context(),
            )
            .await
            .unwrap();

        assert!(
            output
                .content
                .contains("jumped to pattern match at line 50")
        );
        assert!(output.content.contains("fn main"));
    }

    #[tokio::test]
    async fn test_pattern_not_found() {
        let h = TestHarness::new();
        generate_numbered_file(&h, "code.rs", 100);

        let tool = FileViewerTool::new();
        let output = tool
            .execute(
                serde_json::json!({"path": "code.rs", "pattern": "nonexistent_function"}),
                &h.context(),
            )
            .await
            .unwrap();

        assert_eq!(output.outcome, ToolOutcome::Success);
        assert!(
            output
                .content
                .contains("Pattern 'nonexistent_function' not found")
        );
        assert!(output.content.contains("100 lines"));
    }

    #[tokio::test]
    async fn test_pattern_on_first_line() {
        let h = TestHarness::new();
        std::fs::write(
            h.path().join("first.txt"),
            "target_pattern\nline 2\nline 3\n",
        )
        .unwrap();

        let tool = FileViewerTool::new();
        let output = tool
            .execute(
                serde_json::json!({"path": "first.txt", "pattern": "target_pattern"}),
                &h.context(),
            )
            .await
            .unwrap();

        assert!(output.content.contains("jumped to pattern match at line 1"));
        assert!(output.content.contains("target_pattern"));
    }

    // ── Edge cases ──────────────────────────────────────────────────

    #[tokio::test]
    async fn test_small_file_shows_all() {
        let h = TestHarness::new();
        generate_numbered_file(&h, "tiny.txt", 10);

        let tool = FileViewerTool::new();
        let output = tool
            .execute(serde_json::json!({"path": "tiny.txt"}), &h.context())
            .await
            .unwrap();

        assert!(output.content.contains("lines 1-10 of 10"));
        assert!(output.content.contains("0 lines above"));
        assert!(output.content.contains("0 lines below"));
    }

    #[tokio::test]
    async fn test_empty_file() {
        let h = TestHarness::new();
        std::fs::write(h.path().join("empty.txt"), "").unwrap();

        let tool = FileViewerTool::new();
        let output = tool
            .execute(serde_json::json!({"path": "empty.txt"}), &h.context())
            .await
            .unwrap();

        assert_eq!(output.outcome, ToolOutcome::Success);
        assert!(output.content.contains("0-0 of 0"));
    }

    #[tokio::test]
    async fn test_path_traversal_rejected() {
        let h = TestHarness::new();

        let tool = FileViewerTool::new();
        let result = tool
            .execute(
                serde_json::json!({"path": "../../etc/passwd"}),
                &h.context(),
            )
            .await;

        match result {
            Err(ToolError::InvalidInput { tool, .. }) => {
                assert_eq!(tool, "file_viewer");
            }
            other => panic!("expected InvalidInput, got: {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_output_uses_relative_paths() {
        let h = TestHarness::new();
        generate_numbered_file(&h, "rel.txt", 5);

        let tool = FileViewerTool::new();
        let output = tool
            .execute(serde_json::json!({"path": "rel.txt"}), &h.context())
            .await
            .unwrap();

        // Output must contain "rel.txt" but NOT the absolute sandbox path
        assert!(output.content.contains("rel.txt"));
        let sandbox_str = h.path().to_str().unwrap();
        assert!(
            !output.content.contains(sandbox_str),
            "output must not contain absolute sandbox path"
        );
    }

    #[tokio::test]
    async fn test_viewer_tool_info() {
        let tool = FileViewerTool::new();
        let info = tool.info();

        assert_eq!(info.name, "file_viewer");
        assert_eq!(info.required_capability, Capability::FileRead);
        assert_eq!(info.risk_level, RiskLevel::Low);
        assert_eq!(info.side_effects, SideEffects::None);
    }

    #[tokio::test]
    async fn test_non_utf8_returns_error() {
        let h = TestHarness::new();
        std::fs::write(h.path().join("binary.bin"), [0xFF, 0xFE, 0x00, 0x01]).unwrap();

        let tool = FileViewerTool::new();
        let result = tool
            .execute(serde_json::json!({"path": "binary.bin"}), &h.context())
            .await;

        match result {
            Err(ToolError::ExecutionFailed { tool, reason }) => {
                assert_eq!(tool, "file_viewer");
                assert!(reason.contains("UTF-8"));
            }
            other => panic!("expected ExecutionFailed, got: {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_offset_zero_rejected() {
        let h = TestHarness::new();
        generate_numbered_file(&h, "file.txt", 10);

        let tool = FileViewerTool::new();
        let result = tool
            .execute(
                serde_json::json!({"path": "file.txt", "offset": 0}),
                &h.context(),
            )
            .await;

        match result {
            Err(ToolError::InvalidInput { tool, reason }) => {
                assert_eq!(tool, "file_viewer");
                assert!(reason.contains("offset"));
            }
            other => panic!("expected InvalidInput, got: {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_limit_zero_rejected() {
        let h = TestHarness::new();
        generate_numbered_file(&h, "file.txt", 10);

        let tool = FileViewerTool::new();
        let result = tool
            .execute(
                serde_json::json!({"path": "file.txt", "limit": 0}),
                &h.context(),
            )
            .await;

        match result {
            Err(ToolError::InvalidInput { tool, reason }) => {
                assert_eq!(tool, "file_viewer");
                assert!(reason.contains("limit"));
            }
            other => panic!("expected InvalidInput, got: {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_pattern_overrides_offset() {
        let h = TestHarness::new();
        let mut content = String::new();
        for i in 1..=100 {
            if i == 80 {
                content.push_str("MARKER_LINE\n");
            } else {
                let _ = writeln!(content, "Line {i}");
            }
        }
        std::fs::write(h.path().join("mixed.txt"), &content).unwrap();

        let tool = FileViewerTool::new();
        let output = tool
            .execute(
                serde_json::json!({"path": "mixed.txt", "offset": 1, "pattern": "MARKER_LINE"}),
                &h.context(),
            )
            .await
            .unwrap();

        // Pattern should override the offset=1
        assert!(
            output
                .content
                .contains("jumped to pattern match at line 80")
        );
        assert!(output.content.contains("MARKER_LINE"));
    }

    #[tokio::test]
    async fn test_file_exactly_default_limit_lines() {
        let h = TestHarness::new();
        generate_numbered_file(&h, "exact.txt", 100);

        let tool = FileViewerTool::new();
        let output = tool
            .execute(serde_json::json!({"path": "exact.txt"}), &h.context())
            .await
            .unwrap();

        assert!(output.content.contains("lines 1-100 of 100"));
        assert!(output.content.contains("0 lines above"));
        assert!(output.content.contains("0 lines below"));
        assert!(output.content.contains("Line 1"));
        assert!(output.content.contains("Line 100"));
    }

    #[tokio::test]
    async fn test_non_integer_offset_returns_error() {
        let h = TestHarness::new();
        generate_numbered_file(&h, "file.txt", 10);

        let tool = FileViewerTool::new();
        let result = tool
            .execute(
                serde_json::json!({"path": "file.txt", "offset": "abc"}),
                &h.context(),
            )
            .await;

        match result {
            Err(ToolError::InvalidInput { tool, .. }) => {
                assert_eq!(tool, "file_viewer");
            }
            other => panic!("expected InvalidInput, got: {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_empty_pattern_treated_as_no_pattern() {
        let h = TestHarness::new();
        generate_numbered_file(&h, "file.txt", 50);

        let tool = FileViewerTool::new();
        let output = tool
            .execute(
                serde_json::json!({"path": "file.txt", "pattern": ""}),
                &h.context(),
            )
            .await
            .unwrap();

        // Empty pattern should be ignored — show default view from line 1
        assert!(output.content.contains("lines 1-50 of 50"));
        assert!(!output.content.contains("jumped to pattern match"));
    }
}
