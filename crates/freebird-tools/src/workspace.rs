//! Workspace status tool — on-demand git workspace inspection.
//!
//! Provides the `workspace_status` tool that the agent can invoke to
//! orient itself in the working directory. Returns compact git status
//! (branch, modified/staged/untracked files).
//!
//! Git commands are executed directly via `tokio::process::Command`
//! (never through a shell) with environment isolation: specific
//! `GIT_*` env vars removed to prevent discovery interference,
//! `stdin(null)`, `kill_on_drop(true)`.

use std::fmt::Write as _;
use std::path::Path;
use std::time::Duration;

use async_trait::async_trait;

use freebird_traits::tool::{
    Capability, RiskLevel, SideEffects, Tool, ToolContext, ToolError, ToolInfo, ToolOutcome,
    ToolOutput,
};

// ── Constants ────────────────────────────────────────────────────────

/// Maximum files listed per category before truncation.
const MAX_FILES_SHOWN: usize = 5;

// ── Tool Implementation ─────────────────────────────────────────────

/// Read-only git workspace inspector.
///
/// Returns branch name, modified files, staged files, and untracked
/// file count in a compact format (<10 lines).
struct WorkspaceStatusTool {
    info: ToolInfo,
    timeout: Duration,
}

impl WorkspaceStatusTool {
    const NAME: &str = "workspace_status";

    fn new(git_timeout_secs: u64) -> Self {
        Self {
            info: ToolInfo {
                name: Self::NAME.into(),
                description: "Get the current git workspace status including branch, \
                              modified files, staged files, and untracked files. Use this \
                              to orient yourself in the workspace before making changes."
                    .into(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {},
                    "required": []
                }),
                required_capability: Capability::ShellExecute,
                risk_level: RiskLevel::Low,
                side_effects: SideEffects::None,
            },
            timeout: Duration::from_secs(git_timeout_secs),
        }
    }
}

#[async_trait]
impl Tool for WorkspaceStatusTool {
    fn info(&self) -> &ToolInfo {
        &self.info
    }

    async fn execute(
        &self,
        _input: serde_json::Value,
        context: &ToolContext<'_>,
    ) -> Result<ToolOutput, ToolError> {
        let work_dir = resolve_work_dir(context.sandbox_root);

        // Collect branch and file status concurrently.
        let (branch_result, status_result) = tokio::join!(
            git_command(
                &work_dir,
                &["rev-parse", "--abbrev-ref", "HEAD"],
                self.timeout
            ),
            git_command(&work_dir, &["status", "--porcelain"], self.timeout),
        );

        // If branch fails, likely not a git repo.
        let branch = match branch_result {
            Ok(output) => output.trim().to_owned(),
            Err(_) => {
                return Ok(ToolOutput {
                    content: "Not a git repository (or git is not installed).".into(),
                    outcome: ToolOutcome::Success,
                    metadata: None,
                });
            }
        };

        let status_output = match status_result {
            Ok(s) => s,
            Err(e) => {
                tracing::warn!(error = %e, "git status --porcelain failed; reporting empty status");
                String::new()
            }
        };
        let (modified, staged, untracked) = parse_porcelain_status(&status_output);

        let mut output = String::with_capacity(256);
        let _unused = writeln!(output, "Branch: {branch}");
        let _unused = writeln!(output, "Modified: {}", format_file_list(&modified));
        let _unused = writeln!(output, "Staged: {}", format_file_list(&staged));
        let _unused = write!(
            output,
            "Untracked: {}",
            if untracked == 0 {
                "(none)".into()
            } else {
                format!("{untracked} file{}", if untracked == 1 { "" } else { "s" })
            }
        );

        Ok(ToolOutput {
            content: output,
            outcome: ToolOutcome::Success,
            metadata: None,
        })
    }
}

// ── Git Subprocess Helpers ──────────────────────────────────────────

/// Resolve the working directory, handling macOS `/var` → `/private/var`
/// symlink so that git operations use the canonical path.
fn resolve_work_dir(sandbox_root: &Path) -> std::path::PathBuf {
    std::fs::canonicalize(sandbox_root).unwrap_or_else(|_| sandbox_root.to_path_buf())
}

/// Run a git command with full environment isolation.
///
/// Returns the trimmed stdout on success, or an error string on failure.
async fn git_command(work_dir: &Path, args: &[&str], timeout: Duration) -> Result<String, String> {
    let result = tokio::time::timeout(timeout, async {
        let mut cmd = tokio::process::Command::new("git");
        cmd.args(args)
            .current_dir(work_dir)
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .kill_on_drop(true);

        // Remove git env vars that could interfere with discovery,
        // but keep PATH, HOME, TMPDIR etc. needed by the OS.
        for var in &[
            "GIT_DIR",
            "GIT_WORK_TREE",
            "GIT_INDEX_FILE",
            "GIT_OBJECT_DIRECTORY",
            "GIT_ALTERNATE_OBJECT_DIRECTORIES",
            "GIT_CEILING_DIRECTORIES",
            "GIT_COMMON_DIR",
            "GIT_NAMESPACE",
            "GIT_DISCOVERY_ACROSS_FILESYSTEM",
        ] {
            cmd.env_remove(var);
        }

        cmd.output().await
    })
    .await;

    match result {
        Ok(Ok(output)) if output.status.success() => String::from_utf8(output.stdout)
            .map_err(|e| format!("git output was not valid UTF-8: {e}")),
        Ok(Ok(output)) => {
            let stderr = String::from_utf8_lossy(&output.stderr);
            Err(format!(
                "git exited with {}: {}",
                output.status,
                stderr.trim()
            ))
        }
        Ok(Err(e)) => Err(format!("failed to spawn git: {e}")),
        Err(_) => Err("git command timed out".into()),
    }
}

// ── Status Parsing ──────────────────────────────────────────────────

/// Parse porcelain status output into (modified, staged, untracked).
fn parse_porcelain_status(output: &str) -> (Vec<String>, Vec<String>, usize) {
    let line_count = output.lines().count();
    let mut modified = Vec::with_capacity(line_count);
    let mut staged = Vec::with_capacity(line_count);
    let mut untracked: usize = 0;

    for line in output.lines() {
        if line.len() < 3 {
            continue;
        }
        // Porcelain format: XY filename
        // X = index status, Y = worktree status
        let bytes = line.as_bytes();
        let index_status = bytes.first().copied().unwrap_or(b' ');
        let worktree_status = bytes.get(1).copied().unwrap_or(b' ');
        let filename = line.get(3..).unwrap_or("").to_owned();

        if index_status == b'?' {
            untracked += 1;
            continue;
        }

        // Staged: any non-space, non-? in the index column
        if index_status != b' ' {
            staged.push(filename.clone());
        }

        // Modified: any non-space in the worktree column
        if worktree_status != b' ' && worktree_status != b'?' {
            modified.push(filename);
        }
    }

    (modified, staged, untracked)
}

/// Format a file list with truncation beyond `MAX_FILES_SHOWN`.
fn format_file_list(files: &[String]) -> String {
    if files.is_empty() {
        return "(none)".into();
    }

    let mut result: String = files
        .iter()
        .take(MAX_FILES_SHOWN)
        .map(String::as_str)
        .collect::<Vec<_>>()
        .join(", ");

    let remaining = files.len().saturating_sub(MAX_FILES_SHOWN);
    if remaining > 0 {
        let _unused = write!(result, " (+{remaining} more)");
    }

    result
}

// ── Factory ─────────────────────────────────────────────────────────

/// Create the `workspace_status` tool for registration.
#[must_use]
pub fn workspace_tools(git_timeout_secs: u64) -> Vec<Box<dyn Tool>> {
    vec![Box::new(WorkspaceStatusTool::new(git_timeout_secs))]
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use freebird_traits::id::SessionId;
    use tempfile::TempDir;

    #[test]
    fn test_tool_info() {
        let tool = WorkspaceStatusTool::new(5);
        let info = tool.info();
        assert_eq!(info.name, "workspace_status");
        assert_eq!(info.required_capability, Capability::ShellExecute);
        assert_eq!(info.risk_level, RiskLevel::Low);
        assert_eq!(info.side_effects, SideEffects::None);
    }

    #[tokio::test]
    async fn test_non_git_directory() {
        let tmp = TempDir::new().unwrap();
        let tool = WorkspaceStatusTool::new(5);
        let ctx = make_tool_context(tmp.path());

        let result = tool.execute(serde_json::json!({}), &ctx).await.unwrap();
        assert_eq!(result.outcome, ToolOutcome::Success);
        assert!(
            result.content.contains("Not a git repository"),
            "expected 'Not a git repository', got: {}",
            result.content
        );
    }

    #[tokio::test]
    async fn test_includes_branch() {
        let tmp = TempDir::new().unwrap();
        init_git_repo(tmp.path());

        let tool = WorkspaceStatusTool::new(5);
        let ctx = make_tool_context(tmp.path());

        let result = tool.execute(serde_json::json!({}), &ctx).await.unwrap();
        assert!(
            result.content.contains("Branch:"),
            "output should contain branch: {}",
            result.content
        );
    }

    #[tokio::test]
    async fn test_shows_clean_repo() {
        let tmp = TempDir::new().unwrap();
        init_git_repo(tmp.path());

        let tool = WorkspaceStatusTool::new(5);
        let ctx = make_tool_context(tmp.path());

        let result = tool.execute(serde_json::json!({}), &ctx).await.unwrap();
        assert!(
            result.content.contains("Modified: (none)"),
            "output: {}",
            result.content
        );
        assert!(
            result.content.contains("Staged: (none)"),
            "output: {}",
            result.content
        );
        assert!(
            result.content.contains("Untracked: (none)"),
            "output: {}",
            result.content
        );
    }

    #[tokio::test]
    async fn test_includes_modified_files() {
        let tmp = TempDir::new().unwrap();
        init_git_repo(tmp.path());

        // Create and commit a file, then modify it
        std::fs::write(tmp.path().join("foo.rs"), "fn main() {}").unwrap();
        git_in(tmp.path(), &["add", "foo.rs"]);
        git_in(
            tmp.path(),
            &[
                "-c",
                "user.email=test@test.com",
                "-c",
                "user.name=Test",
                "commit",
                "--no-verify",
                "-m",
                "initial",
            ],
        );
        std::fs::write(tmp.path().join("foo.rs"), "fn main() { changed }").unwrap();

        let tool = WorkspaceStatusTool::new(5);
        let ctx = make_tool_context(tmp.path());

        let result = tool.execute(serde_json::json!({}), &ctx).await.unwrap();
        assert!(
            result.content.contains("foo.rs"),
            "output should contain modified file: {}",
            result.content
        );
    }

    #[tokio::test]
    async fn test_includes_staged_files() {
        let tmp = TempDir::new().unwrap();
        init_git_repo(tmp.path());

        std::fs::write(tmp.path().join("staged.rs"), "// staged").unwrap();
        git_in(tmp.path(), &["add", "staged.rs"]);

        let tool = WorkspaceStatusTool::new(5);
        let ctx = make_tool_context(tmp.path());

        let result = tool.execute(serde_json::json!({}), &ctx).await.unwrap();
        assert!(
            result.content.contains("Staged:") && result.content.contains("staged.rs"),
            "output should show staged file: {}",
            result.content
        );
    }

    #[tokio::test]
    async fn test_compact_output() {
        let tmp = TempDir::new().unwrap();
        init_git_repo(tmp.path());

        let tool = WorkspaceStatusTool::new(5);
        let ctx = make_tool_context(tmp.path());

        let result = tool.execute(serde_json::json!({}), &ctx).await.unwrap();
        let line_count = result.content.lines().count();
        assert!(
            line_count <= 10,
            "output should be <=10 lines, got {line_count}: {}",
            result.content
        );
    }

    #[test]
    fn test_file_list_truncation() {
        let files: Vec<String> = (0..8).map(|i| format!("file{i}.rs")).collect();
        let formatted = format_file_list(&files);
        assert!(
            formatted.contains("+3 more"),
            "should truncate: {formatted}"
        );
    }

    #[test]
    fn test_file_list_empty() {
        assert_eq!(format_file_list(&[]), "(none)");
    }

    #[test]
    fn test_parse_porcelain_modified() {
        let output = " M src/main.rs\n M src/lib.rs\n";
        let (modified, staged, untracked) = parse_porcelain_status(output);
        assert_eq!(modified, vec!["src/main.rs", "src/lib.rs"]);
        assert!(staged.is_empty());
        assert_eq!(untracked, 0);
    }

    #[test]
    fn test_parse_porcelain_staged() {
        let output = "A  new_file.rs\nM  changed.rs\n";
        let (modified, staged, untracked) = parse_porcelain_status(output);
        assert!(modified.is_empty());
        assert_eq!(staged, vec!["new_file.rs", "changed.rs"]);
        assert_eq!(untracked, 0);
    }

    #[test]
    fn test_parse_porcelain_untracked() {
        let output = "?? untracked1.rs\n?? untracked2.rs\n";
        let (modified, staged, untracked) = parse_porcelain_status(output);
        assert!(modified.is_empty());
        assert!(staged.is_empty());
        assert_eq!(untracked, 2);
    }

    #[test]
    fn test_parse_porcelain_mixed() {
        let output =
            "M  staged_and_modified.rs\n M modified_only.rs\n?? untracked.rs\nA  added.rs\n";
        let (modified, staged, untracked) = parse_porcelain_status(output);
        assert_eq!(modified, vec!["modified_only.rs"]);
        assert_eq!(staged, vec!["staged_and_modified.rs", "added.rs"]);
        assert_eq!(untracked, 1);
    }

    #[test]
    fn test_parse_porcelain_both_staged_and_modified() {
        // MM = modified in index AND working tree — file appears in both lists
        let output = "MM both.rs\n";
        let (modified, staged, untracked) = parse_porcelain_status(output);
        assert_eq!(modified, vec!["both.rs"]);
        assert_eq!(staged, vec!["both.rs"]);
        assert_eq!(untracked, 0);
    }

    #[test]
    fn test_parse_porcelain_renamed() {
        // R = renamed in index; porcelain shows "R  old -> new"
        let output = "R  old.rs -> new.rs\n";
        let (modified, staged, untracked) = parse_porcelain_status(output);
        assert!(modified.is_empty());
        assert_eq!(staged, vec!["old.rs -> new.rs"]);
        assert_eq!(untracked, 0);
    }

    #[test]
    fn test_parse_porcelain_deleted() {
        let output = " D removed.rs\nD  staged_delete.rs\n";
        let (modified, staged, untracked) = parse_porcelain_status(output);
        assert_eq!(modified, vec!["removed.rs"]);
        assert_eq!(staged, vec!["staged_delete.rs"]);
        assert_eq!(untracked, 0);
    }

    #[test]
    fn test_parse_porcelain_empty_input() {
        let (modified, staged, untracked) = parse_porcelain_status("");
        assert!(modified.is_empty());
        assert!(staged.is_empty());
        assert_eq!(untracked, 0);
    }

    #[test]
    fn test_config_git_timeout_default() {
        let tool = WorkspaceStatusTool::new(5);
        assert_eq!(tool.timeout, Duration::from_secs(5));
    }

    // ── Test Helpers ────────────────────────────────────────────────

    fn make_tool_context(path: &Path) -> ToolContext<'_> {
        // Leak session_id and capabilities so the context can borrow them.
        // This is fine for tests — small fixed allocations.
        let session_id: &'static SessionId =
            Box::leak(Box::new(SessionId::from_string("test-session")));
        let capabilities: &'static [Capability] =
            Box::leak(vec![Capability::ShellExecute].into_boxed_slice());
        ToolContext {
            session_id,
            sandbox_root: path,
            granted_capabilities: capabilities,
            allowed_directories: &[],
            knowledge_store: None,
            memory: None,
        }
    }

    fn init_git_repo(path: &Path) {
        git_in(path, &["init", "--initial-branch=main"]);
        git_in(
            path,
            &[
                "-c",
                "user.email=test@test.com",
                "-c",
                "user.name=Test",
                "commit",
                "--allow-empty",
                "--no-verify",
                "-m",
                "init",
            ],
        );
    }

    fn git_in(path: &Path, args: &[&str]) {
        let output = std::process::Command::new("git")
            .args(args)
            .current_dir(path)
            .env("GIT_CONFIG_NOSYSTEM", "1")
            .env("GIT_CONFIG_GLOBAL", "/dev/null")
            // Clear GIT env vars to prevent pre-commit hook contamination
            // (git sets GIT_DIR during hooks, which would point to the
            // project repo instead of our test temp directory).
            .env_remove("GIT_DIR")
            .env_remove("GIT_WORK_TREE")
            .env_remove("GIT_INDEX_FILE")
            .env_remove("GIT_OBJECT_DIRECTORY")
            .env_remove("GIT_ALTERNATE_OBJECT_DIRECTORIES")
            .env_remove("GIT_CEILING_DIRECTORIES")
            .env_remove("GIT_COMMON_DIR")
            .output()
            .expect("git command failed to execute");
        assert!(
            output.status.success(),
            "git {:?} failed: {}",
            args,
            String::from_utf8_lossy(&output.stderr)
        );
    }
}
