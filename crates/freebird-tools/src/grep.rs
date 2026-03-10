//! Regex-based code search tool with configurable context lines and result capping.
//!
//! Foundational search primitive for agentic coding workflows. Recursively
//! searches files within the sandbox, matching lines against a regex pattern.
//! Skips binary files, hidden directories, and common non-code directories.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use async_trait::async_trait;
use regex::RegexBuilder;

use freebird_security::taint::TaintedToolInput;
use freebird_traits::tool::{
    Capability, RiskLevel, SideEffects, Tool, ToolContext, ToolError, ToolInfo, ToolOutcome,
    ToolOutput,
};

/// Default number of context lines before and after each match.
const DEFAULT_CONTEXT_LINES: usize = 3;

/// Maximum context lines allowed.
const MAX_CONTEXT_LINES: usize = 10;

/// Default maximum number of matches to return.
const DEFAULT_MAX_RESULTS: usize = 50;

/// Hard cap on maximum results to prevent context window flooding.
const MAX_RESULTS_HARD_CAP: usize = 200;

/// Number of bytes to check for null bytes when detecting binary files.
const BINARY_CHECK_BYTES: usize = 8 * 1024;

/// Maximum compiled regex size (bytes). Defense-in-depth against patterns
/// that cause excessive NFA construction. The `regex` crate already prevents
/// catastrophic backtracking, but this limits resource usage during compilation.
const REGEX_SIZE_LIMIT: usize = 256 * 1024;

/// Directories to always skip during recursive search.
const SKIP_DIRS: &[&str] = &[
    ".git",
    ".hg",
    ".svn",
    "node_modules",
    "target",
    "__pycache__",
    ".build",
];

/// Returns the grep search tool as a trait object.
#[must_use]
pub fn grep_tools() -> Vec<Box<dyn Tool>> {
    vec![Box::new(GrepSearchTool::new())]
}

// ── GrepSearchTool ─────────────────────────────────────────────────

struct GrepSearchTool {
    info: ToolInfo,
}

impl GrepSearchTool {
    const NAME: &str = "grep_search";

    fn new() -> Self {
        Self {
            info: ToolInfo {
                name: Self::NAME.into(),
                description: "Search file contents using a regex pattern. Returns matching lines \
                    with surrounding context. Searches recursively within the sandbox, skipping \
                    binary files and common non-code directories (.git, node_modules, target)."
                    .into(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "pattern": {
                            "type": "string",
                            "description": "Regex pattern to search for in file contents"
                        },
                        "path": {
                            "type": "string",
                            "description": "Relative directory or file path to search within. Defaults to sandbox root."
                        },
                        "file_glob": {
                            "type": "string",
                            "description": "Optional glob pattern to filter files (e.g. '*.rs', '*.py'). Searches all files if omitted."
                        },
                        "context_lines": {
                            "type": "integer",
                            "description": "Number of lines to show before and after each match. Default: 3. Max: 10."
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum number of matches to return. Default: 50. Max: 200."
                        },
                        "case_insensitive": {
                            "type": "boolean",
                            "description": "Whether to perform case-insensitive matching. Default: false."
                        }
                    },
                    "required": ["pattern"]
                }),
                required_capability: Capability::FileRead,
                risk_level: RiskLevel::Low,
                side_effects: SideEffects::None,
            },
        }
    }
}

/// Parsed and validated grep search parameters.
struct GrepParams {
    pattern: String,
    search_path: PathBuf,
    file_glob: Option<glob::Pattern>,
    context_lines: usize,
    max_results: usize,
    case_insensitive: bool,
    /// The root used for relativizing output paths.
    display_root: PathBuf,
}

/// A single match result with its context.
struct MatchResult {
    /// Relative path from display root.
    relative_path: String,
    /// 1-indexed line number of the match.
    line_number: usize,
    /// The matching line content (trimmed trailing newline).
    matching_line: String,
    /// Context lines before the match: `(line_number, content)`.
    before: Vec<(usize, String)>,
    /// Context lines after the match: `(line_number, content)`.
    after: Vec<(usize, String)>,
}

impl MatchResult {
    fn format(&self, context_lines: usize) -> String {
        let mut out = String::new();

        if context_lines == 0 {
            let _ = write!(
                out,
                "{}:{}: {}",
                self.relative_path, self.line_number, self.matching_line
            );
            return out;
        }

        // Context before
        let _ = write!(out, "{}:{}", self.relative_path, self.line_number);
        for (num, line) in &self.before {
            let _ = write!(out, "\n    {num}: {line}");
        }

        // The matching line with `>` prefix
        let _ = write!(out, "\n  > {}: {}", self.line_number, self.matching_line);

        // Context after
        for (num, line) in &self.after {
            let _ = write!(out, "\n    {num}: {line}");
        }

        out
    }
}

/// Extract optional parameters from JSON input (non-tainted primitives).
fn extract_optional_usize(input: &serde_json::Value, key: &str) -> Option<usize> {
    input
        .get(key)
        .and_then(serde_json::Value::as_u64)
        .and_then(|v| usize::try_from(v).ok())
}

fn extract_optional_bool(input: &serde_json::Value, key: &str) -> Option<bool> {
    input.get(key).and_then(serde_json::Value::as_bool)
}

fn extract_optional_str<'a>(input: &'a serde_json::Value, key: &str) -> Option<&'a str> {
    input.get(key).and_then(serde_json::Value::as_str)
}

/// Check if a file appears to be binary by looking for null bytes in the first 8KB.
fn is_binary(content: &[u8]) -> bool {
    let check_len = content.len().min(BINARY_CHECK_BYTES);
    content
        .get(..check_len)
        .is_some_and(|slice| slice.contains(&0))
}

/// Check if a directory name should be skipped.
fn should_skip_dir(name: &str) -> bool {
    SKIP_DIRS.contains(&name)
}

/// Recursively collect files in a directory, respecting skip rules and glob filter.
fn collect_files_recursive(
    dir: &Path,
    glob_filter: Option<&glob::Pattern>,
    files: &mut Vec<PathBuf>,
) -> Result<(), std::io::Error> {
    let entries = std::fs::read_dir(dir)?;

    let mut sorted_entries: Vec<std::fs::DirEntry> = entries
        .filter_map(|entry| match entry {
            Ok(e) => Some(e),
            Err(e) => {
                tracing::debug!(dir = %dir.display(), error = %e, "skipping unreadable dir entry");
                None
            }
        })
        .collect();
    sorted_entries.sort_by_key(std::fs::DirEntry::file_name);

    for entry in sorted_entries {
        let Ok(file_type) = entry.file_type() else {
            continue;
        };
        let name = entry.file_name();
        let name_str = name.to_string_lossy();

        if file_type.is_dir() {
            if should_skip_dir(&name_str) {
                continue;
            }
            collect_files_recursive(&entry.path(), glob_filter, files)?;
        } else if file_type.is_file() {
            if let Some(pattern) = glob_filter {
                if !pattern.matches(&name_str) {
                    continue;
                }
            }
            files.push(entry.path());
        }
    }

    Ok(())
}

/// Result of searching a single file: capped results plus total match count.
struct FileSearchResult {
    matches: Vec<MatchResult>,
    total_count: usize,
}

/// Search a single file for regex matches, returning match results with context.
///
/// Returns up to `max_results_to_collect` detailed results, plus the total
/// count of matches in the file (for accurate truncation messages).
fn search_file(
    file_path: &Path,
    regex: &regex::Regex,
    context_lines: usize,
    display_root: &Path,
    max_results_to_collect: usize,
) -> Result<FileSearchResult, std::io::Error> {
    let raw_bytes = std::fs::read(file_path)?;

    if is_binary(&raw_bytes) {
        return Ok(FileSearchResult {
            matches: Vec::new(),
            total_count: 0,
        });
    }

    let Ok(content) = std::str::from_utf8(&raw_bytes) else {
        return Ok(FileSearchResult {
            matches: Vec::new(),
            total_count: 0,
        });
    };

    let lines: Vec<&str> = content.lines().collect();
    let relative = file_path
        .strip_prefix(display_root)
        .unwrap_or(file_path)
        .to_string_lossy()
        .into_owned();

    let mut results = Vec::new();
    let mut total_count: usize = 0;

    for (idx, line) in lines.iter().enumerate() {
        if !regex.is_match(line) {
            continue;
        }

        total_count += 1;

        if results.len() >= max_results_to_collect {
            continue; // Keep counting but don't collect
        }

        let line_number = idx + 1;

        let before: Vec<(usize, String)> = if context_lines > 0 {
            let start = idx.saturating_sub(context_lines);
            (start..idx)
                .map(|i| (i + 1, lines.get(i).unwrap_or(&"").to_string()))
                .collect()
        } else {
            Vec::new()
        };

        let after: Vec<(usize, String)> = if context_lines > 0 {
            let end = (idx + 1 + context_lines).min(lines.len());
            ((idx + 1)..end)
                .map(|i| (i + 1, lines.get(i).unwrap_or(&"").to_string()))
                .collect()
        } else {
            Vec::new()
        };

        results.push(MatchResult {
            relative_path: relative.clone(),
            line_number,
            matching_line: (*line).to_string(),
            before,
            after,
        });
    }

    Ok(FileSearchResult {
        matches: results,
        total_count,
    })
}

#[async_trait]
impl Tool for GrepSearchTool {
    fn info(&self) -> &ToolInfo {
        &self.info
    }

    async fn execute(
        &self,
        input: serde_json::Value,
        context: &ToolContext<'_>,
    ) -> Result<ToolOutput, ToolError> {
        let params = parse_grep_params(&input, context)?;

        // Compile regex
        let regex = RegexBuilder::new(&params.pattern)
            .case_insensitive(params.case_insensitive)
            .size_limit(REGEX_SIZE_LIMIT)
            .build()
            .map_err(|e| ToolError::InvalidInput {
                tool: Self::NAME.into(),
                reason: format!("invalid regex pattern: {e}"),
            })?;

        // Collect files to search
        let files = if params.search_path.is_file() {
            vec![params.search_path.clone()]
        } else {
            let mut files = Vec::new();
            collect_files_recursive(&params.search_path, params.file_glob.as_ref(), &mut files)
                .map_err(|e| ToolError::ExecutionFailed {
                    tool: Self::NAME.into(),
                    reason: format!("failed to walk directory: {e}"),
                })?;
            files
        };

        // Search files and collect results
        let mut all_results: Vec<MatchResult> = Vec::with_capacity(params.max_results);
        let mut total_match_count: usize = 0;
        let mut files_with_matches: usize = 0;

        for file in &files {
            let remaining = params.max_results.saturating_sub(all_results.len());
            let file_result = search_file(
                file,
                &regex,
                params.context_lines,
                &params.display_root,
                remaining,
            )
            .unwrap_or_else(|e| {
                tracing::debug!(file = %file.display(), error = %e, "skipping unreadable file");
                FileSearchResult {
                    matches: Vec::new(),
                    total_count: 0,
                }
            });

            if file_result.total_count > 0 {
                files_with_matches += 1;
                total_match_count += file_result.total_count;
            }

            all_results.extend(file_result.matches);
        }

        // Format output
        if all_results.is_empty() && total_match_count == 0 {
            return Ok(ToolOutput {
                content: format!("No matches found for pattern: {}", params.pattern),
                outcome: ToolOutcome::Success,
                metadata: None,
            });
        }

        let mut output = String::new();
        for (i, result) in all_results.iter().enumerate() {
            if i > 0 {
                output.push_str("\n\n");
            }
            output.push_str(&result.format(params.context_lines));
        }

        // Summary line
        let truncated = total_match_count > all_results.len();
        if truncated {
            let remaining = total_match_count - all_results.len();
            let _ = write!(
                output,
                "\n\n... and {remaining} more matches in {files_with_matches} files (truncated at {})",
                params.max_results
            );
        } else {
            let _ = write!(
                output,
                "\n\nFound {} matches in {} files",
                all_results.len(),
                files_with_matches
            );
        }

        Ok(ToolOutput {
            content: output,
            outcome: ToolOutcome::Success,
            metadata: None,
        })
    }
}

/// Parse and validate grep parameters from raw JSON input.
fn parse_grep_params(
    input: &serde_json::Value,
    context: &ToolContext<'_>,
) -> Result<GrepParams, ToolError> {
    let tainted = TaintedToolInput::new(input.clone());

    // Required: pattern — validate through taint boundary
    let _pattern_tainted =
        tainted
            .extract_string("pattern")
            .map_err(|e| ToolError::InvalidInput {
                tool: GrepSearchTool::NAME.into(),
                reason: e.to_string(),
            })?;
    // Re-extract from raw JSON since Tainted::inner() is pub(crate)
    let pattern = input
        .get("pattern")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| ToolError::InvalidInput {
            tool: GrepSearchTool::NAME.into(),
            reason: "missing or non-string 'pattern' field".into(),
        })?
        .to_string();

    // Optional: path (validated through SafeFilePath)
    let search_path = if input
        .get("path")
        .and_then(serde_json::Value::as_str)
        .is_some()
    {
        let safe_path = tainted
            .extract_path_multi_root("path", context.sandbox_root, context.allowed_directories)
            .map_err(|e| ToolError::InvalidInput {
                tool: GrepSearchTool::NAME.into(),
                reason: e.to_string(),
            })?;
        safe_path.as_path().to_path_buf()
    } else {
        context.sandbox_root.to_path_buf()
    };

    // Optional: file_glob
    let file_glob = match extract_optional_str(input, "file_glob") {
        Some(g) => Some(glob::Pattern::new(g).map_err(|e| ToolError::InvalidInput {
            tool: GrepSearchTool::NAME.into(),
            reason: format!("invalid file glob: {e}"),
        })?),
        None => None,
    };

    // Optional: context_lines (default 3, max 10)
    let context_lines = extract_optional_usize(input, "context_lines")
        .unwrap_or(DEFAULT_CONTEXT_LINES)
        .min(MAX_CONTEXT_LINES);

    // Optional: max_results (default 50, max 200)
    let max_results = extract_optional_usize(input, "max_results")
        .unwrap_or(DEFAULT_MAX_RESULTS)
        .min(MAX_RESULTS_HARD_CAP);

    // Optional: case_insensitive (default false)
    let case_insensitive = extract_optional_bool(input, "case_insensitive").unwrap_or(false);

    Ok(GrepParams {
        pattern,
        search_path,
        file_glob,
        context_lines,
        max_results,
        case_insensitive,
        display_root: context.sandbox_root.to_path_buf(),
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic, clippy::indexing_slicing)]
mod tests {
    use std::io::Write as _;
    use std::path::PathBuf;

    use freebird_traits::id::SessionId;
    use freebird_traits::tool::{Capability, RiskLevel, SideEffects, Tool, ToolContext, ToolError};

    use super::*;

    /// Test harness matching the pattern from filesystem.rs.
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
            let sandbox = tmp.path().canonicalize().unwrap();
            Self {
                _tmp: tmp,
                sandbox,
                session_id: SessionId::from_string("test-session"),
                capabilities: vec![Capability::FileRead],
                allowed_directories: vec![],
            }
        }

        fn path(&self) -> &Path {
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

    fn tool() -> GrepSearchTool {
        GrepSearchTool::new()
    }

    // ── Tool Info Test ─────────────────────────────────────────────

    #[test]
    fn test_grep_tool_info() {
        let t = tool();
        let info = t.info();
        assert_eq!(info.name, "grep_search");
        assert_eq!(info.required_capability, Capability::FileRead);
        assert_eq!(info.risk_level, RiskLevel::Low);
        assert_eq!(info.side_effects, SideEffects::None);
    }

    // ── Core Search Tests ──────────────────────────────────────────

    #[tokio::test]
    async fn test_simple_pattern_match() {
        let h = TestHarness::new();
        std::fs::write(
            h.path().join("hello.rs"),
            "fn main() {\n    println!(\"hello\");\n}\n",
        )
        .unwrap();

        let output = tool()
            .execute(
                serde_json::json!({"pattern": "hello", "context_lines": 0}),
                &h.context(),
            )
            .await
            .unwrap();

        assert!(output.content.contains("hello.rs:2:"));
        assert!(output.content.contains("println!"));
        assert!(output.content.contains("Found 1 matches in 1 files"));
    }

    #[tokio::test]
    async fn test_regex_pattern_match() {
        let h = TestHarness::new();
        std::fs::write(
            h.path().join("code.rs"),
            "fn process_input() {\n}\n\nfn validate_output() {\n}\n",
        )
        .unwrap();

        let output = tool()
            .execute(
                serde_json::json!({"pattern": r"fn\s+\w+", "context_lines": 0}),
                &h.context(),
            )
            .await
            .unwrap();

        assert!(output.content.contains("fn process_input"));
        assert!(output.content.contains("fn validate_output"));
        assert!(output.content.contains("Found 2 matches in 1 files"));
    }

    #[tokio::test]
    async fn test_no_matches_returns_informative_message() {
        let h = TestHarness::new();
        std::fs::write(h.path().join("empty.rs"), "fn main() {}\n").unwrap();

        let output = tool()
            .execute(
                serde_json::json!({"pattern": "nonexistent_symbol"}),
                &h.context(),
            )
            .await
            .unwrap();

        assert!(
            output.content.contains("No matches found"),
            "expected informative message, got: {}",
            output.content
        );
        assert!(output.content.contains("nonexistent_symbol"));
    }

    #[tokio::test]
    async fn test_context_lines_shown() {
        let h = TestHarness::new();
        let content = "line1\nline2\nline3\nTARGET\nline5\nline6\nline7\n";
        std::fs::write(h.path().join("ctx.txt"), content).unwrap();

        let output = tool()
            .execute(
                serde_json::json!({"pattern": "TARGET", "context_lines": 3}),
                &h.context(),
            )
            .await
            .unwrap();

        // Header with file:line
        assert!(output.content.contains("ctx.txt:4"), "missing header");
        // Context before
        assert!(
            output.content.contains("1: line1"),
            "missing context before"
        );
        assert!(
            output.content.contains("2: line2"),
            "missing context before"
        );
        assert!(
            output.content.contains("3: line3"),
            "missing context before"
        );
        // Match line with > prefix
        assert!(output.content.contains("> 4: TARGET"), "missing match line");
        // Context after
        assert!(output.content.contains("5: line5"), "missing context after");
        assert!(output.content.contains("6: line6"), "missing context after");
        assert!(output.content.contains("7: line7"), "missing context after");
    }

    #[tokio::test]
    async fn test_context_lines_zero_compact() {
        let h = TestHarness::new();
        let content = "line1\nline2\nTARGET\nline4\nline5\n";
        std::fs::write(h.path().join("compact.txt"), content).unwrap();

        let output = tool()
            .execute(
                serde_json::json!({"pattern": "TARGET", "context_lines": 0}),
                &h.context(),
            )
            .await
            .unwrap();

        // Compact: just file:line:content, no context
        assert!(output.content.contains("compact.txt:3: TARGET"));
        // Should NOT contain the `>` prefix format
        assert!(
            !output.content.contains("> 3:"),
            "compact mode should not use context format"
        );
    }

    #[tokio::test]
    async fn test_multiple_matches_same_file() {
        let h = TestHarness::new();
        let content = "first match\nno match\nsecond match\n";
        std::fs::write(h.path().join("multi.txt"), content).unwrap();

        let output = tool()
            .execute(
                serde_json::json!({"pattern": "match", "context_lines": 0}),
                &h.context(),
            )
            .await
            .unwrap();

        assert!(output.content.contains("multi.txt:1: first match"));
        assert!(output.content.contains("multi.txt:3: second match"));
        // "no match" also contains "match"
        assert!(output.content.contains("multi.txt:2: no match"));
        assert!(output.content.contains("Found 3 matches in 1 files"));
    }

    #[tokio::test]
    async fn test_matches_across_files() {
        let h = TestHarness::new();
        std::fs::write(h.path().join("a.txt"), "target line\n").unwrap();
        std::fs::write(h.path().join("b.txt"), "another target\n").unwrap();
        std::fs::write(h.path().join("c.txt"), "no match here\n").unwrap();

        let output = tool()
            .execute(
                serde_json::json!({"pattern": "target", "context_lines": 0}),
                &h.context(),
            )
            .await
            .unwrap();

        assert!(output.content.contains("a.txt:1:"));
        assert!(output.content.contains("b.txt:1:"));
        assert!(!output.content.contains("c.txt"));
        assert!(output.content.contains("Found 2 matches in 2 files"));
    }

    // ── Result Capping Tests ───────────────────────────────────────

    fn make_match_lines(count: usize) -> String {
        (0..count).fold(String::new(), |mut acc, i| {
            let _ = writeln!(acc, "match_line_{i}");
            acc
        })
    }

    #[tokio::test]
    async fn test_max_results_caps_output() {
        let h = TestHarness::new();
        // Create a file with 100 matching lines
        std::fs::write(h.path().join("many.txt"), make_match_lines(100)).unwrap();

        let output = tool()
            .execute(
                serde_json::json!({"pattern": "match_line", "max_results": 10, "context_lines": 0}),
                &h.context(),
            )
            .await
            .unwrap();

        // Count the number of match lines (file:line: format)
        let match_count = output
            .content
            .lines()
            .filter(|l| l.starts_with("many.txt:"))
            .count();
        assert_eq!(match_count, 10, "should cap at 10 results");
        assert!(output.content.contains("truncated at 10"));
    }

    #[tokio::test]
    async fn test_truncation_message_shows_remaining_count() {
        let h = TestHarness::new();
        std::fs::write(h.path().join("many.txt"), make_match_lines(100)).unwrap();

        let output = tool()
            .execute(
                serde_json::json!({"pattern": "match_line", "max_results": 10, "context_lines": 0}),
                &h.context(),
            )
            .await
            .unwrap();

        assert!(
            output.content.contains("90 more matches"),
            "should show remaining count, got: {}",
            output.content
        );
    }

    #[tokio::test]
    async fn test_default_max_results_is_50() {
        let h = TestHarness::new();
        // Create 60 matching lines
        std::fs::write(h.path().join("lots.txt"), make_match_lines(60)).unwrap();

        let output = tool()
            .execute(
                serde_json::json!({"pattern": "match_line", "context_lines": 0}),
                &h.context(),
            )
            .await
            .unwrap();

        let match_count = output
            .content
            .lines()
            .filter(|l| l.starts_with("lots.txt:"))
            .count();
        assert_eq!(match_count, 50, "default cap should be 50");
        assert!(output.content.contains("truncated at 50"));
    }

    // ── Filtering Tests ────────────────────────────────────────────

    #[tokio::test]
    async fn test_file_glob_filters_by_extension() {
        let h = TestHarness::new();
        std::fs::write(h.path().join("code.rs"), "fn search_target() {}\n").unwrap();
        std::fs::write(h.path().join("notes.txt"), "search_target in notes\n").unwrap();

        let output = tool()
            .execute(
                serde_json::json!({"pattern": "search_target", "file_glob": "*.rs", "context_lines": 0}),
                &h.context(),
            )
            .await
            .unwrap();

        assert!(output.content.contains("code.rs"));
        assert!(
            !output.content.contains("notes.txt"),
            "should not search .txt files when glob is *.rs"
        );
    }

    #[tokio::test]
    async fn test_case_insensitive_search() {
        let h = TestHarness::new();
        std::fs::write(h.path().join("case.txt"), "Foo\nfoo\nFOO\nbar\n").unwrap();

        let output = tool()
            .execute(
                serde_json::json!({"pattern": "foo", "case_insensitive": true, "context_lines": 0}),
                &h.context(),
            )
            .await
            .unwrap();

        assert!(output.content.contains("case.txt:1: Foo"));
        assert!(output.content.contains("case.txt:2: foo"));
        assert!(output.content.contains("case.txt:3: FOO"));
        assert!(output.content.contains("Found 3 matches"));
    }

    #[tokio::test]
    async fn test_skips_binary_files() {
        let h = TestHarness::new();
        // Create a binary file with null bytes
        let binary_path = h.path().join("binary.dat");
        {
            let mut f = std::fs::File::create(&binary_path).unwrap();
            f.write_all(b"match_this\x00binary data").unwrap();
        }
        // Create a text file with the same pattern
        std::fs::write(h.path().join("text.txt"), "match_this in text\n").unwrap();

        let output = tool()
            .execute(
                serde_json::json!({"pattern": "match_this", "context_lines": 0}),
                &h.context(),
            )
            .await
            .unwrap();

        assert!(output.content.contains("text.txt"));
        assert!(
            !output.content.contains("binary.dat"),
            "should skip binary files"
        );
    }

    #[tokio::test]
    async fn test_skips_git_directory() {
        let h = TestHarness::new();
        let git_dir = h.path().join(".git");
        std::fs::create_dir(&git_dir).unwrap();
        std::fs::write(git_dir.join("config"), "match_this\n").unwrap();
        std::fs::write(h.path().join("real.txt"), "match_this\n").unwrap();

        let output = tool()
            .execute(
                serde_json::json!({"pattern": "match_this", "context_lines": 0}),
                &h.context(),
            )
            .await
            .unwrap();

        assert!(output.content.contains("real.txt"));
        assert!(
            !output.content.contains(".git"),
            "should skip .git directory"
        );
    }

    #[tokio::test]
    async fn test_skips_node_modules() {
        let h = TestHarness::new();
        let nm_dir = h.path().join("node_modules");
        std::fs::create_dir(&nm_dir).unwrap();
        std::fs::write(nm_dir.join("dep.js"), "match_this\n").unwrap();
        std::fs::write(h.path().join("app.js"), "match_this\n").unwrap();

        let output = tool()
            .execute(
                serde_json::json!({"pattern": "match_this", "context_lines": 0}),
                &h.context(),
            )
            .await
            .unwrap();

        assert!(output.content.contains("app.js"));
        assert!(
            !output.content.contains("node_modules"),
            "should skip node_modules"
        );
    }

    // ── Path & Security Tests ──────────────────────────────────────

    #[tokio::test]
    async fn test_path_traversal_rejected() {
        let h = TestHarness::new();

        let err = tool()
            .execute(
                serde_json::json!({"pattern": "anything", "path": "../../"}),
                &h.context(),
            )
            .await
            .unwrap_err();

        match err {
            ToolError::InvalidInput { tool, .. } => assert_eq!(tool, "grep_search"),
            other => panic!("expected InvalidInput, got: {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_search_subdirectory() {
        let h = TestHarness::new();
        let src = h.path().join("src");
        std::fs::create_dir(&src).unwrap();
        std::fs::write(src.join("lib.rs"), "fn target_fn() {}\n").unwrap();
        std::fs::write(h.path().join("root.rs"), "fn target_fn() {}\n").unwrap();

        let output = tool()
            .execute(
                serde_json::json!({"pattern": "target_fn", "path": "src", "context_lines": 0}),
                &h.context(),
            )
            .await
            .unwrap();

        assert!(
            output.content.contains("src/lib.rs") || output.content.contains("src\\lib.rs"),
            "should find match in src/: {}",
            output.content
        );
        assert!(
            !output.content.contains("root.rs"),
            "should not search outside specified path"
        );
    }

    #[tokio::test]
    async fn test_search_single_file() {
        let h = TestHarness::new();
        let src = h.path().join("src");
        std::fs::create_dir(&src).unwrap();
        std::fs::write(src.join("main.rs"), "fn target_fn() {}\n").unwrap();
        std::fs::write(src.join("lib.rs"), "fn target_fn() {}\n").unwrap();

        let output = tool()
            .execute(
                serde_json::json!({"pattern": "target_fn", "path": "src/main.rs", "context_lines": 0}),
                &h.context(),
            )
            .await
            .unwrap();

        assert!(output.content.contains("main.rs"));
        assert!(
            !output.content.contains("lib.rs"),
            "should only search the specified file"
        );
        assert!(output.content.contains("Found 1 matches in 1 files"));
    }

    #[tokio::test]
    async fn test_output_uses_relative_paths() {
        let h = TestHarness::new();
        std::fs::write(h.path().join("test.rs"), "fn target() {}\n").unwrap();

        let output = tool()
            .execute(
                serde_json::json!({"pattern": "target", "context_lines": 0}),
                &h.context(),
            )
            .await
            .unwrap();

        let sandbox_str = h.path().to_string_lossy();
        assert!(
            !output.content.contains(sandbox_str.as_ref()),
            "output should not contain absolute sandbox path: {}",
            output.content
        );
    }

    // ── Error Tests ────────────────────────────────────────────────

    #[tokio::test]
    async fn test_invalid_regex_returns_descriptive_error() {
        let h = TestHarness::new();

        let err = tool()
            .execute(serde_json::json!({"pattern": "["}), &h.context())
            .await
            .unwrap_err();

        match &err {
            ToolError::InvalidInput { tool, reason } => {
                assert_eq!(tool, "grep_search");
                assert!(
                    reason.contains("invalid regex"),
                    "error should describe regex issue: {reason}"
                );
            }
            other => panic!("expected InvalidInput, got: {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_nonexistent_path_returns_error() {
        let h = TestHarness::new();

        let err = tool()
            .execute(
                serde_json::json!({"pattern": "anything", "path": "nonexistent_dir"}),
                &h.context(),
            )
            .await
            .unwrap_err();

        match err {
            ToolError::InvalidInput { tool, .. } | ToolError::ExecutionFailed { tool, .. } => {
                assert_eq!(tool, "grep_search");
            }
            other => panic!("expected InvalidInput or ExecutionFailed, got: {other:?}"),
        }
    }

    // ── Edge Case Tests ─────────────────────────────────────────────

    #[tokio::test]
    async fn test_empty_sandbox_returns_no_matches() {
        let h = TestHarness::new();
        // No files created — empty directory

        let output = tool()
            .execute(serde_json::json!({"pattern": "anything"}), &h.context())
            .await
            .unwrap();

        assert!(
            output.content.contains("No matches found"),
            "empty sandbox should return no-match message, got: {}",
            output.content
        );
    }

    #[tokio::test]
    async fn test_context_lines_clamped_to_max() {
        let h = TestHarness::new();
        let content = "line1\nline2\nTARGET\nline4\nline5\n";
        std::fs::write(h.path().join("clamp.txt"), content).unwrap();

        // Request 100 context lines — should be clamped to MAX_CONTEXT_LINES (10)
        let output = tool()
            .execute(
                serde_json::json!({"pattern": "TARGET", "context_lines": 100}),
                &h.context(),
            )
            .await
            .unwrap();

        // Should succeed, not error — and show context (clamped)
        assert!(output.content.contains("TARGET"));
        assert!(output.content.contains("Found 1 matches"));
    }

    #[tokio::test]
    async fn test_max_results_clamped_to_hard_cap() {
        let h = TestHarness::new();
        std::fs::write(h.path().join("cap.txt"), "match\n").unwrap();

        // Request 999 max_results — should be clamped to MAX_RESULTS_HARD_CAP (200)
        let output = tool()
            .execute(
                serde_json::json!({"pattern": "match", "max_results": 999, "context_lines": 0}),
                &h.context(),
            )
            .await
            .unwrap();

        // Should succeed, not error
        assert!(output.content.contains("Found 1 matches"));
    }

    #[tokio::test]
    async fn test_regex_size_limit_rejects_huge_pattern() {
        let h = TestHarness::new();
        std::fs::write(h.path().join("a.txt"), "x\n").unwrap();

        // A pattern that would produce a large NFA
        let huge_pattern = format!("({})", "a?".repeat(10_000));
        let err = tool()
            .execute(serde_json::json!({"pattern": huge_pattern}), &h.context())
            .await
            .unwrap_err();

        match err {
            ToolError::InvalidInput { tool, reason } => {
                assert_eq!(tool, "grep_search");
                assert!(
                    reason.contains("invalid regex"),
                    "should mention regex issue: {reason}"
                );
            }
            other => panic!("expected InvalidInput, got: {other:?}"),
        }
    }

    // ── Factory Test ───────────────────────────────────────────────

    #[test]
    fn test_grep_tools_factory() {
        let tools = grep_tools();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].info().name, "grep_search");
    }
}
