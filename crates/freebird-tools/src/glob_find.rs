//! Glob-based file discovery tool.
//!
//! Finds files and directories matching glob patterns within the sandbox.
//! Complementary to grep (content search) — this searches file *names*.
//! Uses the `glob` crate for pattern expansion with security enforcement
//! through `SafeFilePath` and sandbox containment.

use std::fmt::Write as _;
use std::path::{Component, Path, PathBuf};

use async_trait::async_trait;

use freebird_security::taint::TaintedToolInput;
use freebird_traits::tool::{
    Capability, RiskLevel, SideEffects, Tool, ToolContext, ToolError, ToolInfo, ToolOutcome,
    ToolOutput,
};

use crate::common;

/// Default maximum number of results to return.
const DEFAULT_MAX_RESULTS: usize = 100;

/// Hard cap on maximum results to prevent context window flooding.
const MAX_RESULTS_HARD_CAP: usize = 500;

/// Returns the glob find tool as a trait object.
#[must_use]
pub fn glob_find_tools() -> Vec<Box<dyn Tool>> {
    vec![Box::new(GlobFindTool::new())]
}

// ── GlobFindTool ──────────────────────────────────────────────────

struct GlobFindTool {
    info: ToolInfo,
}

impl GlobFindTool {
    const NAME: &str = "glob_find";

    fn new() -> Self {
        Self {
            info: ToolInfo {
                name: Self::NAME.into(),
                description: "Find files and directories matching a glob pattern. Returns \
                    matching paths sorted alphabetically. Skips common non-code directories \
                    (.git, node_modules, target)."
                    .into(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "pattern": {
                            "type": "string",
                            "description": "Glob pattern to match file paths (e.g. '**/*.rs', 'src/**/*.toml', '**/test*')"
                        },
                        "path": {
                            "type": "string",
                            "description": "Relative directory to search within. Defaults to sandbox root."
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum number of files to return. Default: 100. Max: 500."
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

/// Parsed and validated glob find parameters.
struct GlobFindParams {
    pattern: String,
    search_root: PathBuf,
    max_results: usize,
    /// The root used for relativizing output paths.
    display_root: PathBuf,
}

/// Check if any component of a relative path is a skipped directory.
fn path_contains_skip_dir(path: &Path, root: &Path) -> bool {
    let relative = path.strip_prefix(root).unwrap_or(path);
    for component in relative.components() {
        if let Component::Normal(name) = component {
            let name_str = name.to_string_lossy();
            if common::should_skip_dir(&name_str) {
                return true;
            }
        }
    }
    false
}

/// Parse and validate glob find parameters from raw JSON input.
fn parse_glob_find_params(
    input: &serde_json::Value,
    context: &ToolContext<'_>,
) -> Result<GlobFindParams, ToolError> {
    let tainted = TaintedToolInput::new(input.clone());

    // Required: pattern — validate through taint boundary
    let _pattern_tainted =
        tainted
            .extract_string("pattern")
            .map_err(|e| ToolError::InvalidInput {
                tool: GlobFindTool::NAME.into(),
                reason: e.to_string(),
            })?;
    // Re-extract from raw JSON since Tainted::inner() is pub(crate)
    let pattern = input
        .get("pattern")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| ToolError::InvalidInput {
            tool: GlobFindTool::NAME.into(),
            reason: "missing or non-string 'pattern' field".into(),
        })?
        .to_string();

    // Optional: path (validated through SafeFilePath)
    let search_root = if input
        .get("path")
        .and_then(serde_json::Value::as_str)
        .is_some()
    {
        let safe_path = tainted
            .extract_path_multi_root("path", context.sandbox_root, context.allowed_directories)
            .map_err(|e| ToolError::InvalidInput {
                tool: GlobFindTool::NAME.into(),
                reason: e.to_string(),
            })?;
        safe_path.as_path().to_path_buf()
    } else {
        context.sandbox_root.to_path_buf()
    };

    // Optional: max_results (default 100, hard cap 500)
    let max_results = common::extract_optional_usize(input, "max_results")
        .unwrap_or(DEFAULT_MAX_RESULTS)
        .min(MAX_RESULTS_HARD_CAP);

    Ok(GlobFindParams {
        pattern,
        search_root,
        max_results,
        display_root: context.sandbox_root.to_path_buf(),
    })
}

#[async_trait]
impl Tool for GlobFindTool {
    fn info(&self) -> &ToolInfo {
        &self.info
    }

    async fn execute(
        &self,
        input: serde_json::Value,
        context: &ToolContext<'_>,
    ) -> Result<ToolOutput, ToolError> {
        let params = parse_glob_find_params(&input, context)?;

        // Build absolute glob pattern
        let absolute_pattern = params
            .search_root
            .join(&params.pattern)
            .to_string_lossy()
            .into_owned();

        // Execute glob
        let paths = glob::glob(&absolute_pattern).map_err(|e| ToolError::InvalidInput {
            tool: Self::NAME.into(),
            reason: format!("invalid glob pattern: {e}"),
        })?;

        // Collect, filter, classify
        let mut entries: Vec<(String, bool)> = Vec::with_capacity(params.max_results.min(256));
        let mut total_found: usize = 0;

        for entry in paths {
            let path = match entry {
                Ok(p) => p,
                Err(e) => {
                    tracing::debug!(error = %e, "skipping unreadable glob entry");
                    continue;
                }
            };

            // Skip entries in excluded directories (pre-canonicalization check
            // avoids unnecessary I/O for obviously-excluded paths).
            if path_contains_skip_dir(&path, &params.display_root) {
                continue;
            }

            // Canonicalize to resolve symlinks before sandbox containment check.
            // This prevents symlinks inside the sandbox from escaping to external paths.
            let canonical = match path.canonicalize() {
                Ok(p) => p,
                Err(e) => {
                    tracing::debug!(path = %path.display(), error = %e, "skipping non-canonicalizable path");
                    continue;
                }
            };

            // Defense-in-depth: verify canonical path is within sandbox
            if !canonical.starts_with(&params.display_root) {
                tracing::debug!(
                    path = %path.display(),
                    canonical = %canonical.display(),
                    "skipping glob result outside sandbox"
                );
                continue;
            }

            let is_dir = canonical.is_dir();
            let relative = path
                .strip_prefix(&params.display_root)
                .unwrap_or(&path)
                .to_string_lossy()
                .into_owned();

            total_found += 1;
            entries.push((relative, is_dir));
        }

        // Sort alphabetically
        entries.sort_by(|a, b| a.0.cmp(&b.0));

        // No matches
        if entries.is_empty() {
            return Ok(ToolOutput {
                content: format!("No files found matching '{}'", params.pattern),
                outcome: ToolOutcome::Success,
                metadata: None,
            });
        }

        // Cap at max_results
        let truncated = entries.len() > params.max_results;
        let showing = entries.len().min(params.max_results);
        entries.truncate(params.max_results);

        // Format output
        let mut output = String::new();
        for (relative, is_dir) in &entries {
            if !output.is_empty() {
                output.push('\n');
            }
            if *is_dir {
                let _ = write!(output, "dir   {relative}/");
            } else {
                let _ = write!(output, "file  {relative}");
            }
        }

        // Summary line
        if truncated {
            let _ = write!(
                output,
                "\n\nFound {showing} of {total_found} results matching '{}' (truncated at {showing})",
                params.pattern
            );
        } else {
            let _ = write!(
                output,
                "\n\nFound {} results matching '{}'",
                entries.len(),
                params.pattern
            );
        }

        Ok(ToolOutput {
            content: output,
            outcome: ToolOutcome::Success,
            metadata: None,
        })
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic, clippy::indexing_slicing)]
mod tests {
    use std::path::PathBuf;

    use freebird_traits::id::SessionId;
    use freebird_traits::tool::{Capability, RiskLevel, SideEffects, Tool, ToolContext, ToolError};

    use super::*;

    /// Test harness matching the pattern from filesystem.rs / grep.rs.
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

    fn tool() -> GlobFindTool {
        GlobFindTool::new()
    }

    // ── Tool Info Test ──────────────────────────────────────────────

    #[test]
    fn test_glob_tool_info() {
        let t = tool();
        let info = t.info();
        assert_eq!(info.name, "glob_find");
        assert_eq!(info.required_capability, Capability::FileRead);
        assert_eq!(info.risk_level, RiskLevel::Low);
        assert_eq!(info.side_effects, SideEffects::None);
    }

    // ── Core Glob Tests ─────────────────────────────────────────────

    #[tokio::test]
    async fn test_simple_glob_matches() {
        let h = TestHarness::new();
        std::fs::write(h.path().join("main.rs"), "fn main() {}\n").unwrap();
        std::fs::write(h.path().join("lib.rs"), "pub fn lib() {}\n").unwrap();
        std::fs::write(h.path().join("readme.md"), "# readme\n").unwrap();

        let output = tool()
            .execute(serde_json::json!({"pattern": "*.rs"}), &h.context())
            .await
            .unwrap();

        assert!(
            output.content.contains("file  lib.rs"),
            "should find lib.rs: {}",
            output.content
        );
        assert!(
            output.content.contains("file  main.rs"),
            "should find main.rs: {}",
            output.content
        );
        assert!(
            !output.content.contains("readme.md"),
            "should not match .md files"
        );
        assert!(output.content.contains("Found 2 results matching '*.rs'"));
    }

    #[tokio::test]
    async fn test_recursive_glob() {
        let h = TestHarness::new();
        let src = h.path().join("src");
        let nested = src.join("tools");
        std::fs::create_dir_all(&nested).unwrap();
        std::fs::write(h.path().join("top.rs"), "").unwrap();
        std::fs::write(src.join("lib.rs"), "").unwrap();
        std::fs::write(nested.join("edit.rs"), "").unwrap();

        let output = tool()
            .execute(serde_json::json!({"pattern": "**/*.rs"}), &h.context())
            .await
            .unwrap();

        assert!(output.content.contains("top.rs"), "should find top-level");
        assert!(
            output.content.contains("src/lib.rs") || output.content.contains("src\\lib.rs"),
            "should find nested: {}",
            output.content
        );
        assert!(
            output.content.contains("src/tools/edit.rs")
                || output.content.contains("src\\tools\\edit.rs"),
            "should find deeply nested: {}",
            output.content
        );
        assert!(output.content.contains("Found 3 results"));
    }

    #[tokio::test]
    async fn test_no_matches_informative() {
        let h = TestHarness::new();
        std::fs::write(h.path().join("file.txt"), "content").unwrap();

        let output = tool()
            .execute(
                serde_json::json!({"pattern": "*.nonexistent"}),
                &h.context(),
            )
            .await
            .unwrap();

        assert!(
            output.content.contains("No files found matching"),
            "should report no matches: {}",
            output.content
        );
        assert!(output.content.contains("*.nonexistent"));
    }

    #[tokio::test]
    async fn test_directory_entries_included() {
        let h = TestHarness::new();
        let subdir = h.path().join("mydir");
        std::fs::create_dir(&subdir).unwrap();
        std::fs::write(h.path().join("myfile"), "").unwrap();

        let output = tool()
            .execute(serde_json::json!({"pattern": "my*"}), &h.context())
            .await
            .unwrap();

        assert!(
            output.content.contains("dir   mydir/"),
            "should show directory with dir type: {}",
            output.content
        );
        assert!(
            output.content.contains("file  myfile"),
            "should show file with file type: {}",
            output.content
        );
    }

    // ── Result Capping Tests ────────────────────────────────────────

    #[tokio::test]
    async fn test_max_results_caps() {
        let h = TestHarness::new();
        // Create 200 files
        for i in 0..200 {
            std::fs::write(h.path().join(format!("file_{i:03}.txt")), "").unwrap();
        }

        let output = tool()
            .execute(
                serde_json::json!({"pattern": "*.txt", "max_results": 10}),
                &h.context(),
            )
            .await
            .unwrap();

        let file_lines = output
            .content
            .lines()
            .filter(|l| l.starts_with("file  "))
            .count();
        assert_eq!(file_lines, 10, "should cap at 10 results");
        assert!(
            output.content.contains("truncated at 10"),
            "should show truncation message: {}",
            output.content
        );
    }

    // ── Sorting Test ────────────────────────────────────────────────

    #[tokio::test]
    async fn test_results_sorted_alphabetically() {
        let h = TestHarness::new();
        std::fs::write(h.path().join("charlie.rs"), "").unwrap();
        std::fs::write(h.path().join("alpha.rs"), "").unwrap();
        std::fs::write(h.path().join("bravo.rs"), "").unwrap();

        let output = tool()
            .execute(serde_json::json!({"pattern": "*.rs"}), &h.context())
            .await
            .unwrap();

        let file_lines: Vec<&str> = output
            .content
            .lines()
            .filter(|l| l.starts_with("file  "))
            .collect();

        assert_eq!(file_lines.len(), 3);
        assert_eq!(file_lines[0], "file  alpha.rs");
        assert_eq!(file_lines[1], "file  bravo.rs");
        assert_eq!(file_lines[2], "file  charlie.rs");
    }

    // ── Skip Directory Tests ────────────────────────────────────────

    #[tokio::test]
    async fn test_skips_git_directory() {
        let h = TestHarness::new();
        let git_dir = h.path().join(".git");
        std::fs::create_dir(&git_dir).unwrap();
        std::fs::write(git_dir.join("config"), "").unwrap();
        std::fs::write(h.path().join("real.rs"), "").unwrap();

        let output = tool()
            .execute(serde_json::json!({"pattern": "**/*"}), &h.context())
            .await
            .unwrap();

        assert!(output.content.contains("real.rs"), "should find real file");
        assert!(
            !output.content.contains(".git"),
            "should skip .git directory: {}",
            output.content
        );
    }

    #[tokio::test]
    async fn test_skips_node_modules() {
        let h = TestHarness::new();
        let nm_dir = h.path().join("node_modules");
        std::fs::create_dir(&nm_dir).unwrap();
        std::fs::write(nm_dir.join("dep.js"), "").unwrap();
        std::fs::write(h.path().join("app.js"), "").unwrap();

        let output = tool()
            .execute(serde_json::json!({"pattern": "**/*.js"}), &h.context())
            .await
            .unwrap();

        assert!(output.content.contains("app.js"));
        assert!(
            !output.content.contains("node_modules"),
            "should skip node_modules: {}",
            output.content
        );
    }

    #[tokio::test]
    async fn test_skips_target_directory() {
        let h = TestHarness::new();
        let target_dir = h.path().join("target");
        let debug_dir = target_dir.join("debug");
        std::fs::create_dir_all(&debug_dir).unwrap();
        std::fs::write(debug_dir.join("build_artifact"), "").unwrap();
        std::fs::write(h.path().join("src.rs"), "").unwrap();

        let output = tool()
            .execute(serde_json::json!({"pattern": "**/*"}), &h.context())
            .await
            .unwrap();

        assert!(output.content.contains("src.rs"), "should find source file");
        assert!(
            !output.content.contains("target"),
            "should skip target directory: {}",
            output.content
        );
    }

    #[tokio::test]
    async fn test_default_max_results_is_100() {
        let h = TestHarness::new();
        // Create 120 files
        for i in 0..120 {
            std::fs::write(h.path().join(format!("file_{i:03}.txt")), "").unwrap();
        }

        let output = tool()
            .execute(serde_json::json!({"pattern": "*.txt"}), &h.context())
            .await
            .unwrap();

        let file_lines = output
            .content
            .lines()
            .filter(|l| l.starts_with("file  "))
            .count();
        assert_eq!(file_lines, 100, "default cap should be 100");
        assert!(
            output.content.contains("truncated at 100"),
            "should truncate at default: {}",
            output.content
        );
    }

    #[tokio::test]
    async fn test_max_results_clamped_to_hard_cap() {
        let h = TestHarness::new();
        std::fs::write(h.path().join("only.txt"), "").unwrap();

        // Request 999 max_results — should be clamped to 500
        let output = tool()
            .execute(
                serde_json::json!({"pattern": "*.txt", "max_results": 999}),
                &h.context(),
            )
            .await
            .unwrap();

        // Should succeed, not error
        assert!(output.content.contains("Found 1 results"));
    }

    // ── Path & Security Tests ───────────────────────────────────────

    #[tokio::test]
    async fn test_path_traversal_rejected() {
        let h = TestHarness::new();

        let err = tool()
            .execute(
                serde_json::json!({"pattern": "*.rs", "path": "../../"}),
                &h.context(),
            )
            .await
            .unwrap_err();

        match err {
            ToolError::InvalidInput { tool, .. } => assert_eq!(tool, "glob_find"),
            other => panic!("expected InvalidInput, got: {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_search_subdirectory() {
        let h = TestHarness::new();
        let src = h.path().join("src");
        std::fs::create_dir(&src).unwrap();
        std::fs::write(src.join("lib.rs"), "").unwrap();
        std::fs::write(h.path().join("root.rs"), "").unwrap();

        let output = tool()
            .execute(
                serde_json::json!({"pattern": "*.rs", "path": "src"}),
                &h.context(),
            )
            .await
            .unwrap();

        assert!(
            output.content.contains("lib.rs"),
            "should find file in src/: {}",
            output.content
        );
        assert!(
            !output.content.contains("root.rs"),
            "should not find file outside specified path"
        );
    }

    #[tokio::test]
    async fn test_output_uses_relative_paths() {
        let h = TestHarness::new();
        std::fs::write(h.path().join("test.rs"), "").unwrap();

        let output = tool()
            .execute(serde_json::json!({"pattern": "*.rs"}), &h.context())
            .await
            .unwrap();

        let sandbox_str = h.path().to_string_lossy();
        assert!(
            !output.content.contains(sandbox_str.as_ref()),
            "output should not contain absolute sandbox path: {}",
            output.content
        );
    }

    // ── Error Tests ─────────────────────────────────────────────────

    #[tokio::test]
    async fn test_invalid_glob_returns_error() {
        let h = TestHarness::new();

        let err = tool()
            .execute(serde_json::json!({"pattern": "[invalid"}), &h.context())
            .await
            .unwrap_err();

        match &err {
            ToolError::InvalidInput { tool, reason } => {
                assert_eq!(tool, "glob_find");
                assert!(
                    reason.contains("invalid glob"),
                    "error should describe glob issue: {reason}"
                );
            }
            other => panic!("expected InvalidInput, got: {other:?}"),
        }
    }

    // ── Symlink Security Tests ────────────────────────────────────

    #[cfg(unix)]
    #[tokio::test]
    async fn test_symlink_escape_rejected() {
        let h = TestHarness::new();

        // Create a file inside the sandbox
        std::fs::write(h.path().join("legit.txt"), "safe").unwrap();

        // Create a symlink that points outside the sandbox
        let outside = tempfile::tempdir().unwrap();
        let outside_file = outside.path().join("secret.txt");
        std::fs::write(&outside_file, "secret data").unwrap();

        std::os::unix::fs::symlink(outside.path(), h.path().join("escape_link")).unwrap();

        let output = tool()
            .execute(serde_json::json!({"pattern": "**/*.txt"}), &h.context())
            .await
            .unwrap();

        assert!(
            output.content.contains("legit.txt"),
            "should find legitimate file"
        );
        assert!(
            !output.content.contains("secret.txt"),
            "should not expose files via symlink escape: {}",
            output.content
        );
    }

    // ── Factory Test ────────────────────────────────────────────────

    #[test]
    fn test_glob_find_tools_factory() {
        let tools = glob_find_tools();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].info().name, "glob_find");
    }
}
