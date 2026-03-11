//! Raw bash command executor for agentic coding feedback loops.
//!
//! Executes arbitrary bash commands via `bash -c` with interleaved
//! stdout+stderr, exit code reporting, working directory support, and
//! tail-preserving output truncation. Designed for the agent to run
//! linters, test suites, and build commands after edits, then self-correct
//! on failures.
//!
//! Unlike [`super::shell::ShellTool`], this tool intentionally allows shell
//! metacharacters (pipes, redirects, compound commands). Security relies on
//! the capability system, approval gate, sandbox root, and clean environment
//! — not argument-level validation.

use std::path::Path;
use std::time::Duration;

use async_trait::async_trait;
use tokio::io::AsyncReadExt;

use freebird_security::taint::TaintedToolInput;
use freebird_traits::tool::{
    Capability, RiskLevel, SideEffects, Tool, ToolContext, ToolError, ToolInfo, ToolOutcome,
    ToolOutput,
};

use crate::common::SANDBOXED_PATH;

/// Default timeout for bash commands (30 seconds).
const DEFAULT_TIMEOUT_SECS: u64 = 30;

/// Maximum timeout for bash commands (5 minutes).
const MAX_TIMEOUT_SECS: u64 = 300;

/// Default maximum output lines before tail truncation.
const DEFAULT_MAX_OUTPUT_LINES: usize = 200;

/// Raw bash command executor.
///
/// Runs commands via `bash -c` with process isolation (clean environment,
/// no stdin, `kill_on_drop`). Output includes a command echo header,
/// interleaved stdout+stderr, and an exit code footer.
///
/// Non-zero exit codes are reported in the output but are NOT treated as
/// tool errors — the agent decides whether to treat them as failures.
struct BashExecTool {
    info: ToolInfo,
}

impl BashExecTool {
    const NAME: &str = "bash_exec";

    fn new() -> Self {
        Self {
            info: ToolInfo {
                name: Self::NAME.into(),
                description: "Execute a bash command with pipes, redirects, and compound \
                              expressions. Returns interleaved stdout+stderr with exit code. \
                              Use for running linters, tests, builds, and other dev tools."
                    .into(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "The bash command to execute. Supports pipes, redirects, compound commands."
                        },
                        "working_directory": {
                            "type": "string",
                            "description": "Working directory relative to sandbox root. Defaults to sandbox root."
                        },
                        "timeout": {
                            "type": "integer",
                            "description": "Timeout in seconds. Default: 30. Max: 300 (5 minutes)."
                        }
                    },
                    "required": ["command"]
                }),
                required_capability: Capability::ShellExecute,
                risk_level: RiskLevel::Critical,
                side_effects: SideEffects::HasSideEffects,
            },
        }
    }
}

/// Result of spawning and waiting for a bash process.
struct RunResult {
    output: String,
    exit_code: Option<i32>,
    timed_out: bool,
}

/// Spawn a bash process with process isolation and wait for completion.
///
/// The command is wrapped in `{ <cmd>; } 2>&1` to interleave stderr into
/// stdout. On timeout, the process is killed and partial output is returned.
async fn run_bash(
    bash_cmd: &str,
    working_dir: &Path,
    sandbox_root: &Path,
    timeout: Duration,
) -> Result<RunResult, ToolError> {
    let wrapped = format!("{{ {bash_cmd}; }} 2>&1");

    let mut child = tokio::process::Command::new("bash")
        .arg("-c")
        .arg(&wrapped)
        .current_dir(working_dir)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .env_clear()
        .env("PATH", SANDBOXED_PATH)
        .env("HOME", sandbox_root)
        .env("LC_ALL", "C.UTF-8")
        .kill_on_drop(true)
        .spawn()
        .map_err(|e| ToolError::ExecutionFailed {
            tool: BashExecTool::NAME.into(),
            reason: format!("failed to spawn bash: {e}"),
        })?;

    let stdout = child.stdout.take();
    let reader_handle = tokio::spawn(async move {
        let mut buf = String::new();
        if let Some(mut reader) = stdout {
            let _ = reader.read_to_string(&mut buf).await;
        }
        buf
    });

    match tokio::time::timeout(timeout, child.wait()).await {
        Ok(Ok(status)) => {
            let output = reader_handle.await.unwrap_or_default();
            Ok(RunResult {
                output,
                exit_code: status.code(),
                timed_out: false,
            })
        }
        Ok(Err(e)) => Err(ToolError::ExecutionFailed {
            tool: BashExecTool::NAME.into(),
            reason: format!("error waiting for bash process: {e}"),
        }),
        Err(_elapsed) => {
            let _ = child.kill().await;
            let partial = match tokio::time::timeout(Duration::from_secs(2), reader_handle).await {
                Ok(Ok(s)) => s,
                _ => String::new(),
            };
            Ok(RunResult {
                output: partial,
                exit_code: None,
                timed_out: true,
            })
        }
    }
}

/// Apply tail-preserving truncation to output.
///
/// When output exceeds `max_lines`, keeps the last `max_lines` lines
/// and prepends a truncation header. Error messages, test summaries,
/// and compiler diagnostics are typically at the bottom.
///
/// Returns `(output, truncated, lines_shown, total_lines)`.
#[must_use]
fn truncate_output(raw: &str, max_lines: usize) -> (String, bool, usize, usize) {
    let lines: Vec<&str> = raw.lines().collect();
    let total = lines.len();

    if total <= max_lines {
        return (raw.to_string(), false, total, total);
    }

    let skip = total.saturating_sub(max_lines);
    let kept = lines.get(skip..).unwrap_or_default();
    let header = format!("[{skip} lines truncated, showing last {max_lines} of {total}]\n\n");
    let body = kept.join("\n");
    (format!("{header}{body}"), true, max_lines, total)
}

#[async_trait]
impl Tool for BashExecTool {
    fn info(&self) -> &ToolInfo {
        &self.info
    }

    async fn execute(
        &self,
        input: serde_json::Value,
        ctx: &ToolContext<'_>,
    ) -> Result<ToolOutput, ToolError> {
        let tainted = TaintedToolInput::new(input);

        // 1. Extract and validate command
        let command =
            tainted
                .extract_bash_command("command")
                .map_err(|e| ToolError::InvalidInput {
                    tool: Self::NAME.into(),
                    reason: e.to_string(),
                })?;

        // 2. Extract optional working directory (validated against sandbox + allowed dirs)
        let working_dir = match tainted.extract_path_multi_root(
            "working_directory",
            ctx.sandbox_root,
            ctx.allowed_directories,
        ) {
            Ok(safe_path) => safe_path.as_path().to_path_buf(),
            Err(freebird_security::error::SecurityError::MissingField { .. }) => {
                ctx.sandbox_root.to_path_buf()
            }
            Err(e) => {
                return Err(ToolError::InvalidInput {
                    tool: Self::NAME.into(),
                    reason: e.to_string(),
                });
            }
        };

        // 3. Extract and clamp timeout
        let timeout_secs = tainted
            .extract_u64_optional("timeout")
            .map_err(|e| ToolError::InvalidInput {
                tool: Self::NAME.into(),
                reason: e.to_string(),
            })?
            .unwrap_or(DEFAULT_TIMEOUT_SECS)
            .clamp(1, MAX_TIMEOUT_SECS);

        // 4. Spawn and wait for bash process
        let result = run_bash(
            command.as_str(),
            &working_dir,
            ctx.sandbox_root,
            Duration::from_secs(timeout_secs),
        )
        .await?;

        // 5. Format output: command echo + truncated output + footer
        let cmd_str = command.as_str();
        let (body, truncated, lines_shown, total_lines) =
            truncate_output(&result.output, DEFAULT_MAX_OUTPUT_LINES);

        let output_text = if result.timed_out {
            format!("$ {cmd_str}\n{body}\n\n[timed out after {timeout_secs}s]")
        } else {
            let code = result.exit_code.unwrap_or(-1);
            format!("$ {cmd_str}\n{body}\n\nexit code: {code}")
        };

        let metadata = serde_json::json!({
            "exit_code": result.exit_code,
            "timed_out": result.timed_out,
            "truncated": truncated,
            "lines_shown": lines_shown,
            "total_lines": total_lines,
        });

        let outcome = if result.timed_out {
            ToolOutcome::Error
        } else {
            ToolOutcome::Success
        };

        tracing::info!(
            command = cmd_str,
            exit_code = ?result.exit_code,
            timed_out = result.timed_out,
            truncated,
            "bash_exec: command executed"
        );

        Ok(ToolOutput {
            content: output_text,
            outcome,
            metadata: Some(metadata),
        })
    }
}

/// Factory: create a bash exec tool for registration.
#[must_use]
pub fn bash_exec_tool() -> Box<dyn Tool> {
    Box::new(BashExecTool::new())
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::needless_pass_by_value
)]
mod tests {
    use std::path::PathBuf;

    use freebird_traits::id::SessionId;
    use freebird_traits::tool::{Capability, RiskLevel, SideEffects, Tool, ToolContext, ToolError};

    use super::*;

    /// Test harness that owns the temp directory, session ID, and capabilities,
    /// providing a zero-boilerplate `context()` method for tool tests.
    struct TestHarness {
        _tmp: tempfile::TempDir,
        sandbox: PathBuf,
        session_id: SessionId,
        capabilities: Vec<Capability>,
    }

    impl TestHarness {
        fn new() -> Self {
            let tmp = tempfile::tempdir().unwrap();
            let sandbox = tmp.path().to_path_buf();
            Self {
                _tmp: tmp,
                sandbox,
                session_id: SessionId::from_string("test-session"),
                capabilities: vec![Capability::ShellExecute],
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
                allowed_directories: &[],
                knowledge_store: None,
            }
        }
    }

    fn tool() -> BashExecTool {
        BashExecTool::new()
    }

    // ── Input Validation ──────────────────────────────────────────

    #[tokio::test]
    async fn test_missing_command_field() {
        let h = TestHarness::new();
        let t = tool();
        let result = t.execute(serde_json::json!({}), &h.context()).await;
        assert!(matches!(result, Err(ToolError::InvalidInput { .. })));
    }

    #[tokio::test]
    async fn test_empty_command() {
        let h = TestHarness::new();
        let t = tool();
        let result = t
            .execute(serde_json::json!({"command": ""}), &h.context())
            .await;
        assert!(matches!(result, Err(ToolError::InvalidInput { .. })));
    }

    #[tokio::test]
    async fn test_whitespace_only_command() {
        let h = TestHarness::new();
        let t = tool();
        let result = t
            .execute(serde_json::json!({"command": "   "}), &h.context())
            .await;
        assert!(matches!(result, Err(ToolError::InvalidInput { .. })));
    }

    #[tokio::test]
    async fn test_command_too_long() {
        let h = TestHarness::new();
        let t = tool();
        let long = "a".repeat(33_000);
        let result = t
            .execute(serde_json::json!({"command": long}), &h.context())
            .await;
        assert!(matches!(result, Err(ToolError::InvalidInput { .. })));
    }

    #[tokio::test]
    async fn test_null_byte_in_command() {
        let h = TestHarness::new();
        let t = tool();
        let result = t
            .execute(serde_json::json!({"command": "ls\0-la"}), &h.context())
            .await;
        assert!(matches!(result, Err(ToolError::InvalidInput { .. })));
    }

    // ── Basic Execution ───────────────────────────────────────────

    #[tokio::test]
    async fn test_simple_command() {
        let h = TestHarness::new();
        let t = tool();
        let output = t
            .execute(serde_json::json!({"command": "echo hello"}), &h.context())
            .await
            .unwrap();

        assert_eq!(output.outcome, ToolOutcome::Success);
        assert!(output.content.contains("hello"));
        assert!(output.content.contains("exit code: 0"));
    }

    #[tokio::test]
    async fn test_command_echo() {
        let h = TestHarness::new();
        let t = tool();
        let output = t
            .execute(serde_json::json!({"command": "echo hello"}), &h.context())
            .await
            .unwrap();

        assert!(
            output.content.starts_with("$ echo hello"),
            "output should start with command echo, got: {}",
            &output.content[..output.content.len().min(80)]
        );
    }

    #[tokio::test]
    async fn test_exit_code_in_output() {
        let h = TestHarness::new();
        let t = tool();
        let output = t
            .execute(serde_json::json!({"command": "echo ok"}), &h.context())
            .await
            .unwrap();

        assert!(
            output.content.ends_with("exit code: 0"),
            "output should end with exit code, got: {}",
            output.content
        );
    }

    #[tokio::test]
    async fn test_non_zero_exit_is_success() {
        let h = TestHarness::new();
        let t = tool();
        let output = t
            .execute(serde_json::json!({"command": "false"}), &h.context())
            .await
            .unwrap();

        assert_eq!(output.outcome, ToolOutcome::Success);
        assert!(output.content.contains("exit code: 1"));
    }

    #[tokio::test]
    async fn test_command_not_found() {
        let h = TestHarness::new();
        let t = tool();
        let output = t
            .execute(
                serde_json::json!({"command": "nonexistent_cmd_xyz_42"}),
                &h.context(),
            )
            .await
            .unwrap();

        // bash returns 127 for command not found
        assert_eq!(output.outcome, ToolOutcome::Success);
        assert!(output.content.contains("exit code: 127"));
    }

    // ── Pipes & Compound Commands ─────────────────────────────────

    #[tokio::test]
    async fn test_pipe_command() {
        let h = TestHarness::new();
        let t = tool();
        let output = t
            .execute(
                serde_json::json!({"command": "echo hello | tr a-z A-Z"}),
                &h.context(),
            )
            .await
            .unwrap();

        assert!(output.content.contains("HELLO"));
    }

    #[tokio::test]
    async fn test_semicolon_compound() {
        let h = TestHarness::new();
        let t = tool();
        let output = t
            .execute(
                serde_json::json!({"command": "echo one; echo two"}),
                &h.context(),
            )
            .await
            .unwrap();

        assert!(output.content.contains("one"));
        assert!(output.content.contains("two"));
    }

    #[tokio::test]
    async fn test_and_operator() {
        let h = TestHarness::new();
        let t = tool();
        let output = t
            .execute(
                serde_json::json!({"command": "true && echo yes"}),
                &h.context(),
            )
            .await
            .unwrap();

        assert!(output.content.contains("yes"));
    }

    // ── Stderr Interleaving ───────────────────────────────────────

    #[tokio::test]
    async fn test_stderr_in_output() {
        let h = TestHarness::new();
        let t = tool();
        let output = t
            .execute(
                serde_json::json!({"command": "echo error_msg >&2"}),
                &h.context(),
            )
            .await
            .unwrap();

        assert!(
            output.content.contains("error_msg"),
            "stderr should appear in output"
        );
        // Should NOT have the ShellTool's separator
        assert!(!output.content.contains("--- stderr ---"));
    }

    #[tokio::test]
    async fn test_stdout_and_stderr_interleaved() {
        let h = TestHarness::new();
        let t = tool();
        let output = t
            .execute(
                serde_json::json!({"command": "echo stdout_line; echo stderr_line >&2"}),
                &h.context(),
            )
            .await
            .unwrap();

        assert!(output.content.contains("stdout_line"));
        assert!(output.content.contains("stderr_line"));
    }

    // ── Working Directory ─────────────────────────────────────────

    #[tokio::test]
    async fn test_default_cwd_is_sandbox() {
        let h = TestHarness::new();
        let t = tool();
        let output = t
            .execute(serde_json::json!({"command": "pwd"}), &h.context())
            .await
            .unwrap();

        let sandbox_str = h.path().to_str().unwrap();
        assert!(
            output.content.contains(sandbox_str),
            "pwd should be sandbox root: {}",
            output.content
        );
    }

    #[tokio::test]
    async fn test_custom_working_directory() {
        let h = TestHarness::new();
        let t = tool();

        // Create a subdirectory in the sandbox
        let subdir = h.path().join("mysubdir");
        std::fs::create_dir(&subdir).unwrap();

        let output = t
            .execute(
                serde_json::json!({
                    "command": "pwd",
                    "working_directory": "mysubdir"
                }),
                &h.context(),
            )
            .await
            .unwrap();

        assert!(
            output.content.contains("mysubdir"),
            "pwd should be in subdirectory: {}",
            output.content
        );
    }

    #[tokio::test]
    async fn test_working_directory_traversal_rejected() {
        let h = TestHarness::new();
        let t = tool();
        let result = t
            .execute(
                serde_json::json!({
                    "command": "ls",
                    "working_directory": "../../"
                }),
                &h.context(),
            )
            .await;

        assert!(matches!(result, Err(ToolError::InvalidInput { .. })));
    }

    // ── Timeout ───────────────────────────────────────────────────

    #[tokio::test]
    async fn test_timeout_kills_process() {
        let h = TestHarness::new();
        let t = tool();
        let output = t
            .execute(
                serde_json::json!({
                    "command": "sleep 60",
                    "timeout": 1
                }),
                &h.context(),
            )
            .await
            .unwrap();

        assert_eq!(output.outcome, ToolOutcome::Error);
        assert!(output.content.contains("[timed out"));
        let meta = output.metadata.unwrap();
        assert_eq!(meta["timed_out"], true);
    }

    #[tokio::test]
    async fn test_timeout_returns_partial_output() {
        let h = TestHarness::new();
        let t = tool();
        let output = t
            .execute(
                serde_json::json!({
                    "command": "echo partial_before_sleep; sleep 60",
                    "timeout": 2
                }),
                &h.context(),
            )
            .await
            .unwrap();

        assert_eq!(output.outcome, ToolOutcome::Error);
        assert!(
            output.content.contains("partial_before_sleep"),
            "partial output should be preserved: {}",
            output.content
        );
    }

    #[tokio::test]
    async fn test_timeout_max_clamped() {
        // We can't easily test that 999 gets clamped to 300 without a long wait,
        // but we can verify the metadata for a quick command with a large timeout.
        let h = TestHarness::new();
        let t = tool();
        let output = t
            .execute(
                serde_json::json!({
                    "command": "echo ok",
                    "timeout": 999
                }),
                &h.context(),
            )
            .await
            .unwrap();

        // Command completes successfully (clamp doesn't block execution)
        assert_eq!(output.outcome, ToolOutcome::Success);
    }

    #[tokio::test]
    async fn test_timeout_zero_clamped() {
        let h = TestHarness::new();
        let t = tool();
        // timeout: 0 should be clamped to 1, so a quick command succeeds
        let output = t
            .execute(
                serde_json::json!({
                    "command": "echo ok",
                    "timeout": 0
                }),
                &h.context(),
            )
            .await
            .unwrap();

        assert_eq!(output.outcome, ToolOutcome::Success);
    }

    // ── Truncation ────────────────────────────────────────────────

    #[test]
    fn test_no_truncation_under_limit() {
        let input = "line1\nline2\nline3";
        let (output, truncated, shown, total) = truncate_output(input, 10);
        assert!(!truncated);
        assert_eq!(shown, 3);
        assert_eq!(total, 3);
        assert_eq!(output, input);
    }

    #[test]
    fn test_truncation_keeps_tail() {
        let lines: Vec<String> = (1..=500).map(|i| format!("line {i}")).collect();
        let input = lines.join("\n");

        let (output, truncated, shown, total) = truncate_output(&input, 10);

        assert!(truncated);
        assert_eq!(shown, 10);
        assert_eq!(total, 500);
        // Should contain the last 10 lines
        assert!(output.contains("line 491"));
        assert!(output.contains("line 500"));
        // Should NOT contain early lines
        assert!(!output.contains("line 1\n"));
    }

    #[test]
    fn test_truncation_header() {
        let lines: Vec<String> = (1..=500).map(|i| format!("line {i}")).collect();
        let input = lines.join("\n");

        let (output, _, _, _) = truncate_output(&input, 200);

        assert!(
            output.starts_with("[300 lines truncated, showing last 200 of 500]"),
            "truncation header mismatch: {}",
            &output[..output.len().min(80)]
        );
    }

    // ── Security ──────────────────────────────────────────────────

    #[tokio::test]
    async fn test_clean_environment() {
        let h = TestHarness::new();
        let t = tool();
        let output = t
            .execute(serde_json::json!({"command": "env"}), &h.context())
            .await
            .unwrap();

        // Should only have PATH, HOME, LC_ALL (plus any bash internals)
        let env_lines: Vec<&str> = output
            .content
            .lines()
            .filter(|l| l.contains('=') && !l.starts_with('$') && !l.starts_with('['))
            .filter(|l| !l.starts_with("exit code"))
            .collect();

        for line in &env_lines {
            assert!(
                line.starts_with("PATH=")
                    || line.starts_with("HOME=")
                    || line.starts_with("LC_ALL=")
                    || line.starts_with("PWD=")
                    || line.starts_with("SHLVL=")
                    || line.starts_with("_="),
                "unexpected env var: {line}"
            );
        }
    }

    #[tokio::test]
    async fn test_path_restricted() {
        let h = TestHarness::new();
        let t = tool();
        let output = t
            .execute(
                serde_json::json!({"command": "printenv PATH"}),
                &h.context(),
            )
            .await
            .unwrap();

        assert!(
            output.content.contains(SANDBOXED_PATH),
            "PATH should be sandboxed: {}",
            output.content
        );
    }

    #[tokio::test]
    async fn test_stdin_closed() {
        let h = TestHarness::new();
        let t = tool();
        // `read` with no stdin should exit immediately
        let output = t
            .execute(
                serde_json::json!({
                    "command": "read -t 1 line; echo done",
                    "timeout": 5
                }),
                &h.context(),
            )
            .await
            .unwrap();

        assert!(output.content.contains("done"));
    }

    // ── Metadata ──────────────────────────────────────────────────

    #[tokio::test]
    async fn test_metadata_exit_code() {
        let h = TestHarness::new();
        let t = tool();
        let output = t
            .execute(serde_json::json!({"command": "exit 42"}), &h.context())
            .await
            .unwrap();

        let meta = output.metadata.unwrap();
        assert_eq!(meta["exit_code"], 42);
        assert_eq!(meta["timed_out"], false);
    }

    #[tokio::test]
    async fn test_metadata_truncation_info() {
        let h = TestHarness::new();
        let t = tool();
        let output = t
            .execute(serde_json::json!({"command": "echo one"}), &h.context())
            .await
            .unwrap();

        let meta = output.metadata.unwrap();
        assert_eq!(meta["truncated"], false);
        assert_eq!(meta["total_lines"], meta["lines_shown"]);
    }

    // ── Info / Factory ────────────────────────────────────────────

    #[test]
    fn test_tool_info() {
        let t = tool();
        let info = t.info();
        assert_eq!(info.name, "bash_exec");
        assert_eq!(info.required_capability, Capability::ShellExecute);
        assert_eq!(info.risk_level, RiskLevel::Critical);
        assert_eq!(info.side_effects, SideEffects::HasSideEffects);
    }

    #[test]
    fn test_factory() {
        let t = bash_exec_tool();
        assert_eq!(t.info().name, "bash_exec");
    }
}
