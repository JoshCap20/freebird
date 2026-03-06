//! Sandboxed shell command executor.
//!
//! Executes allowlisted commands with argument validation through
//! `SafeShellArg`, process isolation (`env_clear()`, `kill_on_drop`,
//! `stdin(null)`), output size limiting, and zero shell expansion.
//! Commands run via `tokio::process::Command` — never `sh -c`.

use std::collections::BTreeSet;

use async_trait::async_trait;

use freebird_security::safe_types::SafeShellArg;
use freebird_security::taint::TaintedToolInput;
use freebird_traits::tool::{
    Capability, RiskLevel, SideEffects, Tool, ToolContext, ToolError, ToolInfo, ToolOutcome,
    ToolOutput,
};

/// Minimal PATH for sandboxed command execution.
///
/// Only standard system directories. Prevents PATH hijacking where an
/// attacker places a malicious binary earlier in PATH.
const SANDBOXED_PATH: &str = "/usr/local/bin:/usr/bin:/bin";

/// Sandboxed shell command executor.
///
/// Invariants enforced by construction:
/// - `allowed_commands` is never empty in production (empty = deny all).
/// - All execution goes through `tokio::process::Command` (no shell expansion).
/// - Sandbox root comes from `ToolContext` (the verified `CapabilityGrant`),
///   not from this struct — the struct does not store `sandbox_root`.
struct ShellTool {
    allowed_commands: BTreeSet<String>,
    max_output_bytes: usize,
    info: ToolInfo,
}

impl ShellTool {
    const NAME: &str = "shell";

    /// Create a new shell tool.
    ///
    /// # Arguments
    ///
    /// * `allowed_commands` — Commands the LLM may invoke. Empty set means
    ///   all commands are rejected (deny-by-default).
    /// * `max_output_bytes` — Maximum stdout+stderr bytes returned. Output
    ///   beyond this limit is truncated with a marker.
    fn new(allowed_commands: impl IntoIterator<Item = String>, max_output_bytes: usize) -> Self {
        let allowed: BTreeSet<String> = allowed_commands.into_iter().collect();
        Self {
            info: ToolInfo {
                name: Self::NAME.into(),
                description: format!(
                    "Execute a shell command. Only these commands are permitted: {}. \
                     No shell expansion, pipes, or redirection.",
                    allowed
                        .iter()
                        .map(String::as_str)
                        .collect::<Vec<_>>()
                        .join(", ")
                ),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "The command to execute (must be allowlisted)"
                        },
                        "args": {
                            "type": "array",
                            "items": { "type": "string" },
                            "description": "Command arguments (no shell metacharacters)"
                        }
                    },
                    "required": ["command"]
                }),
                required_capability: Capability::ShellExecute,
                risk_level: RiskLevel::High,
                side_effects: SideEffects::HasSideEffects,
            },
            allowed_commands: allowed,
            max_output_bytes,
        }
    }

    /// Extract and validate the optional `args` array from tainted tool input.
    ///
    /// Returns `Ok(vec![])` if the field is absent (args are optional).
    /// Returns `Err` if the field is present but not an array of strings,
    /// or if any argument fails `SafeShellArg` validation.
    fn extract_validated_args(tainted: &TaintedToolInput) -> Result<Vec<SafeShellArg>, ToolError> {
        let tainted_args =
            tainted
                .extract_string_array_optional("args")
                .map_err(|e| ToolError::InvalidInput {
                    tool: Self::NAME.into(),
                    reason: e.to_string(),
                })?;

        tainted_args
            .iter()
            .enumerate()
            .map(|(i, t)| {
                SafeShellArg::from_tainted(t).map_err(|e| ToolError::InvalidInput {
                    tool: Self::NAME.into(),
                    reason: format!("args[{i}]: {e}"),
                })
            })
            .collect()
    }
}

#[async_trait]
impl Tool for ShellTool {
    fn info(&self) -> &ToolInfo {
        &self.info
    }

    async fn execute(
        &self,
        input: serde_json::Value,
        context: &ToolContext<'_>,
    ) -> Result<ToolOutput, ToolError> {
        let tainted = TaintedToolInput::new(input);

        // 1. Extract and validate command name
        let command =
            tainted
                .extract_shell_arg("command")
                .map_err(|e| ToolError::InvalidInput {
                    tool: Self::NAME.into(),
                    reason: e.to_string(),
                })?;

        // 2. Check allowlist — error does NOT reveal which commands are allowed
        if !self.allowed_commands.contains(command.as_str()) {
            tracing::warn!(
                command = command.as_str(),
                "shell tool: rejected non-allowlisted command"
            );
            return Err(ToolError::SecurityViolation {
                tool: Self::NAME.into(),
                reason: format!("command `{}` is not permitted", command.as_str()),
            });
        }

        // 3. Extract and validate arguments (optional field)
        let args = Self::extract_validated_args(&tainted)?;

        // 4. Execute with process isolation
        let output = tokio::process::Command::new(command.as_str())
            .args(args.iter().map(SafeShellArg::as_str))
            .current_dir(context.sandbox_root)
            .stdin(std::process::Stdio::null())
            .env_clear()
            .env("PATH", SANDBOXED_PATH)
            .env("HOME", context.sandbox_root)
            .env("LC_ALL", "C.UTF-8")
            .kill_on_drop(true)
            .output()
            .await
            .map_err(|e| ToolError::ExecutionFailed {
                tool: Self::NAME.into(),
                reason: e.to_string(),
            })?;

        // 5. Build result with output size limiting
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);

        let mut combined = if stderr.is_empty() {
            stdout.into_owned()
        } else {
            format!("{stdout}\n--- stderr ---\n{stderr}")
        };

        // Truncate if over limit — find nearest char boundary to avoid panic
        let truncated = combined.len() > self.max_output_bytes;
        if truncated {
            let mut truncate_at = self.max_output_bytes;
            while truncate_at > 0 && !combined.is_char_boundary(truncate_at) {
                truncate_at -= 1;
            }
            combined.truncate(truncate_at);
            combined.push_str("\n\n[output truncated]");
            tracing::warn!(
                command = command.as_str(),
                limit = self.max_output_bytes,
                "shell tool: output truncated"
            );
        }

        // 6. Build metadata with exit status details
        let exit_code = output.status.code(); // None if killed by signal
        let metadata = serde_json::json!({
            "exit_code": exit_code,
            "signal_killed": exit_code.is_none(),
            "truncated": truncated,
        });

        let outcome = if output.status.success() {
            ToolOutcome::Success
        } else {
            ToolOutcome::Error
        };

        tracing::info!(
            command = command.as_str(),
            exit_code = ?exit_code,
            truncated,
            "shell tool: command executed"
        );

        Ok(ToolOutput {
            content: combined,
            outcome,
            metadata: Some(metadata),
        })
    }
}

/// Factory: create a shell tool from config.
///
/// Returns the tool as a boxed trait object for registration with `ToolExecutor`.
#[must_use]
pub fn shell_tool(allowed_commands: Vec<String>, max_output_bytes: usize) -> Box<dyn Tool> {
    Box::new(ShellTool::new(allowed_commands, max_output_bytes))
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::needless_pass_by_value
)]
mod tests {
    use freebird_traits::id::SessionId;
    use freebird_traits::tool::{Capability, Tool, ToolContext, ToolError, ToolOutcome};

    use super::*;

    const TEST_MAX_OUTPUT: usize = 1_048_576;

    fn make_context() -> (SessionId, Vec<Capability>) {
        let session_id = SessionId::from_string("test-session");
        let caps = vec![Capability::ShellExecute];
        (session_id, caps)
    }

    fn echo_tool() -> ShellTool {
        ShellTool::new(["echo".to_string()], TEST_MAX_OUTPUT)
    }

    fn ls_tool() -> ShellTool {
        ShellTool::new(["ls".to_string()], TEST_MAX_OUTPUT)
    }

    // ── Input Validation ──────────────────────────────────────────

    #[tokio::test]
    async fn test_allowlisted_command_succeeds() {
        let tmp = tempfile::tempdir().unwrap();
        let tool = echo_tool();
        let (sid, caps) = make_context();
        let ctx = ToolContext {
            session_id: &sid,
            sandbox_root: tmp.path(),
            granted_capabilities: &caps,
        };

        let output = tool
            .execute(
                serde_json::json!({"command": "echo", "args": ["hello"]}),
                &ctx,
            )
            .await
            .unwrap();
        assert!(matches!(output.outcome, ToolOutcome::Success));
        assert!(
            output.content.contains("hello"),
            "output should contain 'hello': {}",
            output.content
        );
    }

    #[tokio::test]
    async fn test_non_allowlisted_command_returns_security_violation() {
        let tmp = tempfile::tempdir().unwrap();
        let tool = ls_tool();
        let (sid, caps) = make_context();
        let ctx = ToolContext {
            session_id: &sid,
            sandbox_root: tmp.path(),
            granted_capabilities: &caps,
        };

        let err = tool
            .execute(
                serde_json::json!({"command": "rm", "args": ["-rf", "/"]}),
                &ctx,
            )
            .await
            .unwrap_err();
        match &err {
            ToolError::SecurityViolation { tool, reason } => {
                assert_eq!(tool, "shell");
                assert!(reason.contains("not permitted"), "reason: {reason}");
                // Must NOT reveal what IS permitted
                assert!(
                    !reason.contains("ls"),
                    "error should not reveal allowlist: {reason}"
                );
            }
            other => panic!("expected SecurityViolation, got: {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_empty_allowlist_rejects_all() {
        let tmp = tempfile::tempdir().unwrap();
        let tool = ShellTool::new(Vec::<String>::new(), TEST_MAX_OUTPUT);
        let (sid, caps) = make_context();
        let ctx = ToolContext {
            session_id: &sid,
            sandbox_root: tmp.path(),
            granted_capabilities: &caps,
        };

        let err = tool
            .execute(serde_json::json!({"command": "ls"}), &ctx)
            .await
            .unwrap_err();
        assert!(matches!(err, ToolError::SecurityViolation { .. }));
    }

    #[tokio::test]
    async fn test_missing_command_field_returns_invalid_input() {
        let tmp = tempfile::tempdir().unwrap();
        let tool = echo_tool();
        let (sid, caps) = make_context();
        let ctx = ToolContext {
            session_id: &sid,
            sandbox_root: tmp.path(),
            granted_capabilities: &caps,
        };

        let err = tool
            .execute(serde_json::json!({"args": ["hello"]}), &ctx)
            .await
            .unwrap_err();
        match &err {
            ToolError::InvalidInput { tool, reason } => {
                assert_eq!(tool, "shell");
                assert!(reason.contains("command"), "reason: {reason}");
            }
            other => panic!("expected InvalidInput, got: {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_empty_command_name_rejected() {
        let tmp = tempfile::tempdir().unwrap();
        let tool = echo_tool();
        let (sid, caps) = make_context();
        let ctx = ToolContext {
            session_id: &sid,
            sandbox_root: tmp.path(),
            granted_capabilities: &caps,
        };

        // Empty string passes SafeShellArg (no forbidden chars) but fails allowlist
        let err = tool
            .execute(serde_json::json!({"command": ""}), &ctx)
            .await
            .unwrap_err();
        assert!(matches!(err, ToolError::SecurityViolation { .. }));
    }

    #[tokio::test]
    async fn test_command_with_pipe_returns_invalid_input() {
        let tmp = tempfile::tempdir().unwrap();
        let tool = echo_tool();
        let (sid, caps) = make_context();
        let ctx = ToolContext {
            session_id: &sid,
            sandbox_root: tmp.path(),
            granted_capabilities: &caps,
        };

        let err = tool
            .execute(serde_json::json!({"command": "ls | cat"}), &ctx)
            .await
            .unwrap_err();
        assert!(matches!(err, ToolError::InvalidInput { .. }));
    }

    #[tokio::test]
    async fn test_command_with_semicolon_returns_invalid_input() {
        let tmp = tempfile::tempdir().unwrap();
        let tool = echo_tool();
        let (sid, caps) = make_context();
        let ctx = ToolContext {
            session_id: &sid,
            sandbox_root: tmp.path(),
            granted_capabilities: &caps,
        };

        let err = tool
            .execute(serde_json::json!({"command": "ls; rm"}), &ctx)
            .await
            .unwrap_err();
        assert!(matches!(err, ToolError::InvalidInput { .. }));
    }

    #[tokio::test]
    async fn test_arg_with_pipe_returns_invalid_input() {
        let tmp = tempfile::tempdir().unwrap();
        let tool = echo_tool();
        let (sid, caps) = make_context();
        let ctx = ToolContext {
            session_id: &sid,
            sandbox_root: tmp.path(),
            granted_capabilities: &caps,
        };

        let err = tool
            .execute(
                serde_json::json!({"command": "echo", "args": ["hello | rm -rf /"]}),
                &ctx,
            )
            .await
            .unwrap_err();
        match &err {
            ToolError::InvalidInput { reason, .. } => {
                assert!(reason.contains("args[0]"), "reason: {reason}");
            }
            other => panic!("expected InvalidInput, got: {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_arg_with_semicolon_returns_invalid_input() {
        let tmp = tempfile::tempdir().unwrap();
        let tool = echo_tool();
        let (sid, caps) = make_context();
        let ctx = ToolContext {
            session_id: &sid,
            sandbox_root: tmp.path(),
            granted_capabilities: &caps,
        };

        let err = tool
            .execute(
                serde_json::json!({"command": "echo", "args": ["hello; rm"]}),
                &ctx,
            )
            .await
            .unwrap_err();
        assert!(matches!(err, ToolError::InvalidInput { .. }));
    }

    #[tokio::test]
    async fn test_arg_with_backtick_returns_invalid_input() {
        let tmp = tempfile::tempdir().unwrap();
        let tool = echo_tool();
        let (sid, caps) = make_context();
        let ctx = ToolContext {
            session_id: &sid,
            sandbox_root: tmp.path(),
            granted_capabilities: &caps,
        };

        let err = tool
            .execute(
                serde_json::json!({"command": "echo", "args": ["`whoami`"]}),
                &ctx,
            )
            .await
            .unwrap_err();
        assert!(matches!(err, ToolError::InvalidInput { .. }));
    }

    #[tokio::test]
    async fn test_arg_with_dollar_sign_returns_invalid_input() {
        let tmp = tempfile::tempdir().unwrap();
        let tool = echo_tool();
        let (sid, caps) = make_context();
        let ctx = ToolContext {
            session_id: &sid,
            sandbox_root: tmp.path(),
            granted_capabilities: &caps,
        };

        let err = tool
            .execute(
                serde_json::json!({"command": "echo", "args": ["$HOME"]}),
                &ctx,
            )
            .await
            .unwrap_err();
        assert!(matches!(err, ToolError::InvalidInput { .. }));
    }

    #[tokio::test]
    async fn test_non_array_args_returns_invalid_input() {
        let tmp = tempfile::tempdir().unwrap();
        let tool = echo_tool();
        let (sid, caps) = make_context();
        let ctx = ToolContext {
            session_id: &sid,
            sandbox_root: tmp.path(),
            granted_capabilities: &caps,
        };

        let err = tool
            .execute(
                serde_json::json!({"command": "echo", "args": "hello"}),
                &ctx,
            )
            .await
            .unwrap_err();
        assert!(matches!(err, ToolError::InvalidInput { .. }));
    }

    #[tokio::test]
    async fn test_non_string_array_element_returns_invalid_input() {
        let tmp = tempfile::tempdir().unwrap();
        let tool = echo_tool();
        let (sid, caps) = make_context();
        let ctx = ToolContext {
            session_id: &sid,
            sandbox_root: tmp.path(),
            granted_capabilities: &caps,
        };

        let err = tool
            .execute(serde_json::json!({"command": "echo", "args": [42]}), &ctx)
            .await
            .unwrap_err();
        match &err {
            ToolError::InvalidInput { reason, .. } => {
                assert!(reason.contains("args[0]"), "reason: {reason}");
            }
            other => panic!("expected InvalidInput, got: {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_missing_args_defaults_to_empty() {
        let tmp = tempfile::tempdir().unwrap();
        let tool = echo_tool();
        let (sid, caps) = make_context();
        let ctx = ToolContext {
            session_id: &sid,
            sandbox_root: tmp.path(),
            granted_capabilities: &caps,
        };

        let output = tool
            .execute(serde_json::json!({"command": "echo"}), &ctx)
            .await
            .unwrap();
        assert!(matches!(output.outcome, ToolOutcome::Success));
    }

    #[tokio::test]
    async fn test_empty_args_array_is_valid() {
        let tmp = tempfile::tempdir().unwrap();
        let tool = echo_tool();
        let (sid, caps) = make_context();
        let ctx = ToolContext {
            session_id: &sid,
            sandbox_root: tmp.path(),
            granted_capabilities: &caps,
        };

        let output = tool
            .execute(serde_json::json!({"command": "echo", "args": []}), &ctx)
            .await
            .unwrap();
        assert!(matches!(output.outcome, ToolOutcome::Success));
    }

    // ── Execution Behavior ────────────────────────────────────────

    #[tokio::test]
    async fn test_successful_command_returns_stdout() {
        let tmp = tempfile::tempdir().unwrap();
        let tool = echo_tool();
        let (sid, caps) = make_context();
        let ctx = ToolContext {
            session_id: &sid,
            sandbox_root: tmp.path(),
            granted_capabilities: &caps,
        };

        let output = tool
            .execute(
                serde_json::json!({"command": "echo", "args": ["hello"]}),
                &ctx,
            )
            .await
            .unwrap();
        assert!(matches!(output.outcome, ToolOutcome::Success));
        assert!(output.content.contains("hello"));
    }

    #[tokio::test]
    async fn test_failed_command_returns_stderr_and_is_error() {
        let tmp = tempfile::tempdir().unwrap();
        let tool = ls_tool();
        let (sid, caps) = make_context();
        let ctx = ToolContext {
            session_id: &sid,
            sandbox_root: tmp.path(),
            granted_capabilities: &caps,
        };

        let output = tool
            .execute(
                serde_json::json!({"command": "ls", "args": ["nonexistent_dir_xxxxx"]}),
                &ctx,
            )
            .await
            .unwrap();
        assert!(matches!(output.outcome, ToolOutcome::Error));
        assert!(
            output.content.contains("stderr"),
            "should contain stderr separator: {}",
            output.content
        );
    }

    #[tokio::test]
    async fn test_exit_code_in_metadata() {
        let tmp = tempfile::tempdir().unwrap();
        let tool = echo_tool();
        let (sid, caps) = make_context();
        let ctx = ToolContext {
            session_id: &sid,
            sandbox_root: tmp.path(),
            granted_capabilities: &caps,
        };

        let output = tool
            .execute(
                serde_json::json!({"command": "echo", "args": ["hello"]}),
                &ctx,
            )
            .await
            .unwrap();
        let meta = output.metadata.unwrap();
        assert_eq!(meta["exit_code"], 0);
        assert_eq!(meta["signal_killed"], false);
    }

    #[tokio::test]
    async fn test_stderr_separated_by_marker() {
        let tmp = tempfile::tempdir().unwrap();
        let tool = ls_tool();
        let (sid, caps) = make_context();
        let ctx = ToolContext {
            session_id: &sid,
            sandbox_root: tmp.path(),
            granted_capabilities: &caps,
        };

        // ls on a nonexistent path produces stderr
        let output = tool
            .execute(
                serde_json::json!({"command": "ls", "args": ["nonexistent_dir_xxxxx"]}),
                &ctx,
            )
            .await
            .unwrap();
        assert!(
            output.content.contains("--- stderr ---"),
            "output should contain stderr separator: {}",
            output.content
        );
    }

    #[tokio::test]
    async fn test_runs_in_sandbox_root_cwd() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join("test_marker.txt"), "marker").unwrap();
        let tool = ls_tool();
        let (sid, caps) = make_context();
        let ctx = ToolContext {
            session_id: &sid,
            sandbox_root: tmp.path(),
            granted_capabilities: &caps,
        };

        let output = tool
            .execute(serde_json::json!({"command": "ls"}), &ctx)
            .await
            .unwrap();
        assert!(matches!(output.outcome, ToolOutcome::Success));
        assert!(
            output.content.contains("test_marker.txt"),
            "ls should list files in sandbox: {}",
            output.content
        );
    }

    #[tokio::test]
    async fn test_command_not_found_returns_execution_failed() {
        let tmp = tempfile::tempdir().unwrap();
        let tool = ShellTool::new(["nonexistent_cmd_xyz_12345".to_string()], TEST_MAX_OUTPUT);
        let (sid, caps) = make_context();
        let ctx = ToolContext {
            session_id: &sid,
            sandbox_root: tmp.path(),
            granted_capabilities: &caps,
        };

        let err = tool
            .execute(
                serde_json::json!({"command": "nonexistent_cmd_xyz_12345"}),
                &ctx,
            )
            .await
            .unwrap_err();
        assert!(matches!(err, ToolError::ExecutionFailed { .. }));
    }

    // ── Security ──────────────────────────────────────────────────

    #[tokio::test]
    async fn test_environment_is_clean() {
        let tmp = tempfile::tempdir().unwrap();
        let tool = ShellTool::new(["env".to_string()], TEST_MAX_OUTPUT);
        let (sid, caps) = make_context();
        let ctx = ToolContext {
            session_id: &sid,
            sandbox_root: tmp.path(),
            granted_capabilities: &caps,
        };

        let output = tool
            .execute(serde_json::json!({"command": "env"}), &ctx)
            .await
            .unwrap();

        // Should only contain PATH, HOME, LC_ALL
        assert!(output.content.contains("PATH="));
        assert!(output.content.contains("HOME="));
        assert!(output.content.contains("LC_ALL="));

        // Should NOT contain inherited env vars
        assert!(
            !output.content.contains("SHELL="),
            "env should not leak SHELL: {}",
            output.content
        );
    }

    #[tokio::test]
    async fn test_path_is_restricted() {
        let tmp = tempfile::tempdir().unwrap();
        let tool = ShellTool::new(["printenv".to_string()], TEST_MAX_OUTPUT);
        let (sid, caps) = make_context();
        let ctx = ToolContext {
            session_id: &sid,
            sandbox_root: tmp.path(),
            granted_capabilities: &caps,
        };

        let output = tool
            .execute(
                serde_json::json!({"command": "printenv", "args": ["PATH"]}),
                &ctx,
            )
            .await
            .unwrap();
        assert!(matches!(output.outcome, ToolOutcome::Success));
        assert_eq!(output.content.trim(), SANDBOXED_PATH);
    }

    #[tokio::test]
    async fn test_output_truncated_at_limit() {
        let tmp = tempfile::tempdir().unwrap();
        // Create a file with known content > 100 bytes
        let content = "x".repeat(200);
        std::fs::write(tmp.path().join("big.txt"), &content).unwrap();

        let tool = ShellTool::new(["cat".to_string()], 100);
        let (sid, caps) = make_context();
        let ctx = ToolContext {
            session_id: &sid,
            sandbox_root: tmp.path(),
            granted_capabilities: &caps,
        };

        let output = tool
            .execute(
                serde_json::json!({"command": "cat", "args": ["big.txt"]}),
                &ctx,
            )
            .await
            .unwrap();
        assert!(
            output.content.contains("[output truncated]"),
            "should have truncation marker: {}",
            output.content
        );
        let meta = output.metadata.unwrap();
        assert_eq!(meta["truncated"], true);
    }

    #[tokio::test]
    async fn test_output_not_truncated_when_under_limit() {
        let tmp = tempfile::tempdir().unwrap();
        let tool = ShellTool::new(["echo".to_string()], 1000);
        let (sid, caps) = make_context();
        let ctx = ToolContext {
            session_id: &sid,
            sandbox_root: tmp.path(),
            granted_capabilities: &caps,
        };

        let output = tool
            .execute(
                serde_json::json!({"command": "echo", "args": ["short"]}),
                &ctx,
            )
            .await
            .unwrap();
        assert!(
            !output.content.contains("[output truncated]"),
            "should not have truncation marker"
        );
        let meta = output.metadata.unwrap();
        assert_eq!(meta["truncated"], false);
    }

    #[tokio::test]
    async fn test_stdin_is_closed() {
        let tmp = tempfile::tempdir().unwrap();
        let tool = ShellTool::new(["cat".to_string()], TEST_MAX_OUTPUT);
        let (sid, caps) = make_context();
        let ctx = ToolContext {
            session_id: &sid,
            sandbox_root: tmp.path(),
            granted_capabilities: &caps,
        };

        // cat with no args reads from stdin — should exit immediately
        // because stdin is /dev/null
        let output = tool
            .execute(serde_json::json!({"command": "cat"}), &ctx)
            .await
            .unwrap();
        assert!(matches!(output.outcome, ToolOutcome::Success));
        assert!(
            output.content.is_empty(),
            "cat from /dev/null should be empty"
        );
    }

    // ── Additional Tests ──────────────────────────────────────────

    #[tokio::test]
    async fn test_concurrent_shell_calls_dont_interfere() {
        let tmp1 = tempfile::tempdir().unwrap();
        let tmp2 = tempfile::tempdir().unwrap();
        std::fs::write(tmp1.path().join("file1.txt"), "content1").unwrap();
        std::fs::write(tmp2.path().join("file2.txt"), "content2").unwrap();

        let tool = ls_tool();
        let (sid, caps) = make_context();

        let ctx1 = ToolContext {
            session_id: &sid,
            sandbox_root: tmp1.path(),
            granted_capabilities: &caps,
        };
        let ctx2 = ToolContext {
            session_id: &sid,
            sandbox_root: tmp2.path(),
            granted_capabilities: &caps,
        };

        let (out1, out2) = tokio::join!(
            tool.execute(serde_json::json!({"command": "ls"}), &ctx1),
            tool.execute(serde_json::json!({"command": "ls"}), &ctx2),
        );

        let out1 = out1.unwrap();
        let out2 = out2.unwrap();
        assert!(out1.content.contains("file1.txt"));
        assert!(out2.content.contains("file2.txt"));
        assert!(!out1.content.contains("file2.txt"));
        assert!(!out2.content.contains("file1.txt"));
    }

    #[tokio::test]
    async fn test_very_long_arg_rejected() {
        let tmp = tempfile::tempdir().unwrap();
        let tool = echo_tool();
        let (sid, caps) = make_context();
        let ctx = ToolContext {
            session_id: &sid,
            sandbox_root: tmp.path(),
            granted_capabilities: &caps,
        };

        let long_arg = "a".repeat(5000); // > 4096 SafeShellArg limit
        let err = tool
            .execute(
                serde_json::json!({"command": "echo", "args": [long_arg]}),
                &ctx,
            )
            .await
            .unwrap_err();
        assert!(matches!(err, ToolError::InvalidInput { .. }));
    }

    #[tokio::test]
    async fn test_unicode_command_name() {
        let tmp = tempfile::tempdir().unwrap();
        let tool = echo_tool();
        let (sid, caps) = make_context();
        let ctx = ToolContext {
            session_id: &sid,
            sandbox_root: tmp.path(),
            granted_capabilities: &caps,
        };

        // Unicode passes SafeShellArg (no forbidden chars) but fails allowlist
        let err = tool
            .execute(serde_json::json!({"command": "\u{1F600}"}), &ctx)
            .await
            .unwrap_err();
        assert!(matches!(err, ToolError::SecurityViolation { .. }));
    }

    #[tokio::test]
    async fn test_unicode_args_pass_validation() {
        let tmp = tempfile::tempdir().unwrap();
        let tool = echo_tool();
        let (sid, caps) = make_context();
        let ctx = ToolContext {
            session_id: &sid,
            sandbox_root: tmp.path(),
            granted_capabilities: &caps,
        };

        let output = tool
            .execute(
                serde_json::json!({"command": "echo", "args": ["\u{00E9}\u{00F1}\u{00FC}"]}),
                &ctx,
            )
            .await
            .unwrap();
        assert!(matches!(output.outcome, ToolOutcome::Success));
    }

    #[test]
    fn test_info_returns_correct_metadata() {
        let tool = echo_tool();
        let info = tool.info();
        assert_eq!(info.name, "shell");
        assert_eq!(info.required_capability, Capability::ShellExecute);
        assert_eq!(info.risk_level, RiskLevel::High);
        assert!(matches!(info.side_effects, SideEffects::HasSideEffects));
    }

    #[test]
    fn test_to_definition_matches_info() {
        let tool = echo_tool();
        let def = tool.to_definition();
        assert_eq!(def.name, tool.info().name);
        assert_eq!(def.description, tool.info().description);
    }

    #[test]
    fn test_shell_tool_factory() {
        let tool = shell_tool(vec!["ls".to_string(), "cat".to_string()], 1024);
        assert_eq!(tool.info().name, "shell");
        assert!(tool.info().description.contains("ls"));
        assert!(tool.info().description.contains("cat"));
    }

    #[tokio::test]
    async fn test_truncation_at_multibyte_char_boundary() {
        let tmp = tempfile::tempdir().unwrap();
        // "日" is 3 bytes in UTF-8 (E6 97 A5). Write 10 of them = 30 bytes.
        let content = "日".repeat(10);
        std::fs::write(tmp.path().join("mb.txt"), &content).unwrap();

        // Set limit to 8 — falls mid-character (3rd char starts at byte 6, ends at 9).
        let tool = ShellTool::new(["cat".to_string()], 8);
        let (sid, caps) = make_context();
        let ctx = ToolContext {
            session_id: &sid,
            sandbox_root: tmp.path(),
            granted_capabilities: &caps,
        };

        let output = tool
            .execute(
                serde_json::json!({"command": "cat", "args": ["mb.txt"]}),
                &ctx,
            )
            .await
            .unwrap();
        assert!(
            output.content.contains("[output truncated]"),
            "should have truncation marker"
        );
        // Content before marker should be valid UTF-8 (truncated to char boundary)
        let before_marker = output
            .content
            .split("\n\n[output truncated]")
            .next()
            .unwrap();
        assert!(
            before_marker.len() <= 8,
            "truncated content should be at most max_output_bytes: {}",
            before_marker.len()
        );
        // Should have truncated to byte 6 (2 full 3-byte chars)
        assert_eq!(before_marker, "日日");
    }

    #[tokio::test]
    async fn test_path_traversal_in_args_constrained_by_sandbox_cwd() {
        let tmp = tempfile::tempdir().unwrap();
        let tool = ls_tool();
        let (sid, caps) = make_context();
        let ctx = ToolContext {
            session_id: &sid,
            sandbox_root: tmp.path(),
            granted_capabilities: &caps,
        };

        // ".." doesn't contain shell metacharacters, so SafeShellArg allows it.
        // But the command runs with cwd=sandbox_root, so ls sees sandbox parent.
        // This documents that path traversal via args is limited by the sandbox cwd,
        // not by SafeShellArg. Full containment is the ToolExecutor's responsibility.
        let output = tool
            .execute(serde_json::json!({"command": "ls", "args": [".."] }), &ctx)
            .await
            .unwrap();
        // Command succeeds — SafeShellArg does not block ".."
        assert!(matches!(output.outcome, ToolOutcome::Success));
    }
}
