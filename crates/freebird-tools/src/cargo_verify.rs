//! Cargo verification pipeline tool.
//!
//! Runs Rust verification commands (`cargo check`, `clippy`, `test`, `fmt`,
//! `deny`, `doc`, `build`) and returns **structured, parsed results** with
//! file paths, line numbers, and compiler suggestions. This is the agent's
//! primary feedback loop for self-improvement.
//!
//! Commands are executed directly via `tokio::process::Command` (never through
//! a shell) with process isolation: `env_clear()`, selective env vars,
//! `kill_on_drop(true)`, and `stdin(null)`.

use std::fmt::Write as _;
use std::path::Path;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use freebird_security::taint::TaintedToolInput;
use freebird_traits::tool::{
    Capability, RiskLevel, SideEffects, Tool, ToolContext, ToolError, ToolInfo, ToolOutcome,
    ToolOutput,
};

use crate::common::{extract_optional_bool, extract_optional_str};

// ── Constants ────────────────────────────────────────────────────────

/// Maximum issues reported per step to prevent context window flooding.
const MAX_ISSUES_PER_STEP: usize = 50;

/// Maximum raw output bytes captured per cargo invocation.
const MAX_OUTPUT_BYTES: usize = 512 * 1024; // 512 KB

/// Per-step timeouts.
const TIMEOUT_CHECK: Duration = Duration::from_secs(120);
const TIMEOUT_CLIPPY: Duration = Duration::from_secs(120);
const TIMEOUT_TEST: Duration = Duration::from_secs(300);
const TIMEOUT_FMT: Duration = Duration::from_secs(30);
const TIMEOUT_DENY: Duration = Duration::from_secs(60);
const TIMEOUT_DOC: Duration = Duration::from_secs(120);
const TIMEOUT_BUILD: Duration = Duration::from_secs(300);

// ── Structured Output Types ──────────────────────────────────────────

/// Which cargo verification step to execute.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum VerifyStep {
    Check,
    Clippy,
    Test,
    Fmt,
    Deny,
    Doc,
    Build,
}

impl VerifyStep {
    fn from_str(s: &str) -> Option<Self> {
        match s {
            "check" => Some(Self::Check),
            "clippy" => Some(Self::Clippy),
            "test" => Some(Self::Test),
            "fmt" => Some(Self::Fmt),
            "deny" => Some(Self::Deny),
            "doc" => Some(Self::Doc),
            "build" => Some(Self::Build),
            _ => None,
        }
    }

    const fn label(self) -> &'static str {
        match self {
            Self::Check => "cargo check",
            Self::Clippy => "cargo clippy",
            Self::Test => "cargo test",
            Self::Fmt => "cargo fmt",
            Self::Deny => "cargo deny",
            Self::Doc => "cargo doc",
            Self::Build => "cargo build",
        }
    }

    const fn timeout(self) -> Duration {
        match self {
            Self::Check => TIMEOUT_CHECK,
            Self::Clippy => TIMEOUT_CLIPPY,
            Self::Test => TIMEOUT_TEST,
            Self::Fmt => TIMEOUT_FMT,
            Self::Deny => TIMEOUT_DENY,
            Self::Doc => TIMEOUT_DOC,
            Self::Build => TIMEOUT_BUILD,
        }
    }
}

/// Status of a single verification step.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum StepStatus {
    Pass,
    Fail,
    Skipped,
}

/// Overall verdict across all steps.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum Verdict {
    Pass,
    Fail,
    Partial,
}

/// Severity of a diagnostic issue.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum IssueLevel {
    Error,
    Warning,
}

/// A single diagnostic from cargo.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Issue {
    level: IssueLevel,
    message: String,
    file: Option<String>,
    line: Option<u32>,
    column: Option<u32>,
    help: Option<String>,
}

/// Result of a single verification step.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct StepResult {
    step: VerifyStep,
    status: StepStatus,
    duration_ms: u64,
    issues: Vec<Issue>,
    summary: Option<String>,
}

/// Overall verification result.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct VerificationResult {
    verdict: Verdict,
    steps: Vec<StepResult>,
    total_duration_ms: u64,
}

// ── Parsed Parameters ────────────────────────────────────────────────

/// Validated parameters extracted from tool input.
struct VerifyParams {
    steps: Vec<VerifyStep>,
    package: Option<String>,
    test_filter: Option<String>,
    release: bool,
    /// Whether the pipeline should stop on first failure.
    stop_on_failure: bool,
}

// ── Tool Struct ──────────────────────────────────────────────────────

/// Cargo verification pipeline tool.
///
/// Runs Rust build/test/lint/fmt commands and returns structured results
/// with parsed diagnostics (file, line, column, message, help).
struct CargoVerifyTool {
    info: ToolInfo,
    /// Real HOME directory captured at construction time.
    /// Cargo needs this for `~/.cargo` and `~/.rustup`.
    home_dir: String,
}

impl CargoVerifyTool {
    const NAME: &str = "cargo_verify";

    fn new() -> Self {
        let home_dir = std::env::var("HOME").unwrap_or_else(|_| "/root".into());
        Self {
            info: ToolInfo {
                name: Self::NAME.into(),
                description: "Run Rust verification commands (check, clippy, test, fmt, deny, \
                              doc, build) and return structured results with parsed diagnostics. \
                              Use checks: [\"all\"] for the standard pipeline (check → clippy → \
                              test → fmt), or specify individual steps."
                    .into(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "checks": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "enum": ["all", "check", "clippy", "test", "fmt", "deny", "doc", "build"]
                            },
                            "description": "Which verification steps to run. 'all' runs check → clippy → test → fmt and stops on first failure. Default: ['all']."
                        },
                        "package": {
                            "type": "string",
                            "description": "Specific crate to verify (e.g. 'freebird-tools'). Omit to verify the entire workspace."
                        },
                        "test_filter": {
                            "type": "string",
                            "description": "Only run tests matching this name/pattern (passed to cargo test as filter)."
                        },
                        "release": {
                            "type": "boolean",
                            "description": "Build in release mode. Default: false."
                        }
                    }
                }),
                required_capability: Capability::ShellExecute,
                risk_level: RiskLevel::Medium,
                side_effects: SideEffects::HasSideEffects,
            },
            home_dir,
        }
    }

    /// Build the PATH string including cargo/rustup bin directories.
    fn cargo_path(&self) -> String {
        format!("{}/.cargo/bin:/usr/local/bin:/usr/bin:/bin", self.home_dir)
    }

    /// Parse and validate tool input into `VerifyParams`.
    fn parse_params(input: &serde_json::Value) -> Result<VerifyParams, ToolError> {
        let tainted = TaintedToolInput::new(input.clone());

        // Extract checks array (optional, defaults to ["all"])
        let check_strings = tainted
            .extract_string_array_optional("checks")
            .map_err(|e| ToolError::InvalidInput {
                tool: Self::NAME.into(),
                reason: e.to_string(),
            })?;

        // Determine steps and whether to stop on failure.
        //
        // The `extract_string_array_optional` call above validates the field is a
        // string array (rejecting non-string elements). We re-read from raw JSON
        // for the values because `Tainted::inner()` is `pub(crate)`.
        let (steps, stop_on_failure) = if check_strings.is_empty() {
            // Default: "all" pipeline
            (
                vec![
                    VerifyStep::Check,
                    VerifyStep::Clippy,
                    VerifyStep::Test,
                    VerifyStep::Fmt,
                ],
                true,
            )
        } else {
            let mut steps = Vec::new();
            let mut has_all = false;

            let checks_arr = input
                .get("checks")
                .and_then(serde_json::Value::as_array)
                .map_or_else(Vec::new, |arr| {
                    arr.iter()
                        .filter_map(serde_json::Value::as_str)
                        .map(str::to_string)
                        .collect()
                });

            for check in &checks_arr {
                if check == "all" {
                    has_all = true;
                    steps.extend([
                        VerifyStep::Check,
                        VerifyStep::Clippy,
                        VerifyStep::Test,
                        VerifyStep::Fmt,
                    ]);
                } else {
                    let step =
                        VerifyStep::from_str(check).ok_or_else(|| ToolError::InvalidInput {
                            tool: Self::NAME.into(),
                            reason: format!(
                                "unknown check: `{check}`. Valid values: all, check, clippy, \
                                 test, fmt, deny, doc, build"
                            ),
                        })?;
                    steps.push(step);
                }
            }

            (steps, has_all)
        };

        // Extract optional package (validate via SafeShellArg for injection prevention)
        let package = if extract_optional_str(input, "package").is_some() {
            let safe_arg =
                tainted
                    .extract_shell_arg("package")
                    .map_err(|e| ToolError::InvalidInput {
                        tool: Self::NAME.into(),
                        reason: format!("package: {e}"),
                    })?;
            Some(safe_arg.as_str().to_string())
        } else {
            None
        };

        // Extract optional test_filter (validate via SafeShellArg)
        let test_filter = if extract_optional_str(input, "test_filter").is_some() {
            let safe_arg =
                tainted
                    .extract_shell_arg("test_filter")
                    .map_err(|e| ToolError::InvalidInput {
                        tool: Self::NAME.into(),
                        reason: format!("test_filter: {e}"),
                    })?;
            Some(safe_arg.as_str().to_string())
        } else {
            None
        };

        let release = extract_optional_bool(input, "release").unwrap_or(false);

        Ok(VerifyParams {
            steps,
            package,
            test_filter,
            release,
            stop_on_failure,
        })
    }

    /// Build a `tokio::process::Command` for the given step.
    fn build_command(
        &self,
        step: VerifyStep,
        params: &VerifyParams,
        sandbox_root: &Path,
    ) -> tokio::process::Command {
        let mut cmd = tokio::process::Command::new("cargo");
        cmd.current_dir(sandbox_root)
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .kill_on_drop(true)
            .env_clear()
            .env("PATH", self.cargo_path())
            .env("HOME", &self.home_dir)
            .env("CARGO_TERM_COLOR", "never")
            .env("LC_ALL", "C.UTF-8");

        // Pass through RUSTUP_HOME and CARGO_HOME if they were set
        if let Ok(v) = std::env::var("RUSTUP_HOME") {
            cmd.env("RUSTUP_HOME", v);
        }
        if let Ok(v) = std::env::var("CARGO_HOME") {
            cmd.env("CARGO_HOME", v);
        }

        match step {
            VerifyStep::Check => {
                cmd.args(["check", "--all-targets", "--message-format=json"]);
            }
            VerifyStep::Clippy => {
                cmd.args([
                    "clippy",
                    "--all-targets",
                    "--message-format=json",
                    "--",
                    "-D",
                    "warnings",
                ]);
            }
            VerifyStep::Test => {
                cmd.arg("test");
            }
            VerifyStep::Fmt => {
                cmd.args(["fmt", "--check"]);
            }
            VerifyStep::Deny => {
                cmd.args(["deny", "check"]);
            }
            VerifyStep::Doc => {
                cmd.args([
                    "doc",
                    "--no-deps",
                    "--document-private-items",
                    "--message-format=json",
                ]);
            }
            VerifyStep::Build => {
                cmd.args(["build", "--all-targets", "--message-format=json"]);
            }
        }

        // --release flag (not applicable to fmt or deny)
        if params.release && !matches!(step, VerifyStep::Fmt | VerifyStep::Deny) {
            cmd.arg("--release");
        }

        // Package scoping
        if let Some(pkg) = &params.package {
            // fmt is always workspace-wide; deny doesn't support -p
            if !matches!(step, VerifyStep::Fmt | VerifyStep::Deny) {
                cmd.args(["--package", pkg]);
            }
        } else {
            // Workspace-wide for check/clippy/test/build/doc
            if matches!(
                step,
                VerifyStep::Check
                    | VerifyStep::Clippy
                    | VerifyStep::Test
                    | VerifyStep::Build
                    | VerifyStep::Doc
            ) {
                cmd.arg("--workspace");
            }
        }

        // Test filter
        if step == VerifyStep::Test {
            if let Some(filter) = &params.test_filter {
                // cargo test <filter> -- separator already implied
                cmd.arg(filter);
            }
        }

        cmd
    }

    /// Run the verification pipeline.
    async fn run_pipeline(
        &self,
        params: &VerifyParams,
        sandbox_root: &Path,
    ) -> Result<VerificationResult, ToolError> {
        let start = Instant::now();
        let mut results = Vec::with_capacity(params.steps.len());
        let mut had_failure = false;

        for &step in &params.steps {
            // Skip remaining steps if pipeline mode and a prior step failed
            if had_failure && params.stop_on_failure {
                results.push(StepResult {
                    step,
                    status: StepStatus::Skipped,
                    duration_ms: 0,
                    issues: Vec::new(),
                    summary: Some("skipped (prior step failed)".into()),
                });
                continue;
            }

            let step_result = self.run_step(step, params, sandbox_root).await;
            if step_result.status == StepStatus::Fail {
                had_failure = true;
            }
            results.push(step_result);
        }

        let verdict = if had_failure {
            if results.iter().any(|r| r.status == StepStatus::Pass) {
                Verdict::Partial
            } else {
                Verdict::Fail
            }
        } else {
            Verdict::Pass
        };

        Ok(VerificationResult {
            verdict,
            steps: results,
            total_duration_ms: start.elapsed().as_millis().try_into().unwrap_or(u64::MAX),
        })
    }

    /// Run a single verification step with timeout.
    async fn run_step(
        &self,
        step: VerifyStep,
        params: &VerifyParams,
        sandbox_root: &Path,
    ) -> StepResult {
        let step_start = Instant::now();
        let mut cmd = self.build_command(step, params, sandbox_root);

        let output = match tokio::time::timeout(step.timeout(), cmd.output()).await {
            Ok(Ok(output)) => output,
            Ok(Err(e)) => {
                tracing::warn!(step = step.label(), error = %e, "cargo command failed to execute");
                return fail_step(
                    step,
                    &step_start,
                    format!("failed to execute {}: {e}", step.label()),
                );
            }
            Err(_elapsed) => {
                tracing::warn!(step = step.label(), "cargo command timed out");
                return fail_step(
                    step,
                    &step_start,
                    format!(
                        "{} timed out after {}s",
                        step.label(),
                        step.timeout().as_secs()
                    ),
                );
            }
        };

        let stdout = truncate_output(&output.stdout);
        let stderr = truncate_output(&output.stderr);

        tracing::debug!(
            step = step.label(),
            stdout_len = stdout.len(),
            stderr_len = stderr.len(),
            exit_code = ?output.status.code(),
            "cargo output captured"
        );

        let (issues, summary) = match step {
            VerifyStep::Check | VerifyStep::Clippy | VerifyStep::Build | VerifyStep::Doc => {
                // Cargo outputs JSON diagnostics to stdout, but check stderr too
                // in case some environments route them there.
                let mut diags = parse_cargo_json_diagnostics(&stdout, sandbox_root);
                if diags.is_empty() {
                    diags = parse_cargo_json_diagnostics(&stderr, sandbox_root);
                }
                // If still no parsed issues but the step failed, include stderr
                // as a single error issue so the agent gets actionable feedback.
                if diags.is_empty() && !output.status.success() {
                    diags = parse_rendered_diagnostics(&stderr, sandbox_root);
                }
                (diags, None)
            }
            VerifyStep::Test => parse_test_output(&stdout, &stderr, sandbox_root),
            VerifyStep::Fmt => (parse_fmt_output(&stdout, sandbox_root), None),
            VerifyStep::Deny => (parse_deny_output(&stderr), None),
        };

        let status = if output.status.success() {
            StepStatus::Pass
        } else {
            StepStatus::Fail
        };

        let duration_ms = step_start
            .elapsed()
            .as_millis()
            .try_into()
            .unwrap_or(u64::MAX);

        tracing::info!(
            step = step.label(),
            status = ?status,
            issue_count = issues.len(),
            duration_ms,
            "verification step completed"
        );

        StepResult {
            step,
            status,
            duration_ms,
            issues,
            summary,
        }
    }
}

/// Build a failed `StepResult` with a single error message.
fn fail_step(step: VerifyStep, start: &Instant, message: String) -> StepResult {
    StepResult {
        step,
        status: StepStatus::Fail,
        duration_ms: start.elapsed().as_millis().try_into().unwrap_or(u64::MAX),
        issues: vec![Issue {
            level: IssueLevel::Error,
            message,
            file: None,
            line: None,
            column: None,
            help: None,
        }],
        summary: None,
    }
}

#[async_trait]
impl Tool for CargoVerifyTool {
    fn info(&self) -> &ToolInfo {
        &self.info
    }

    async fn execute(
        &self,
        input: serde_json::Value,
        context: &ToolContext<'_>,
    ) -> Result<ToolOutput, ToolError> {
        let params = Self::parse_params(&input)?;
        let result = self.run_pipeline(&params, context.sandbox_root).await?;

        let formatted = format_result(&result);
        let metadata = serde_json::to_value(&result).ok();

        let outcome = if result.verdict == Verdict::Pass {
            ToolOutcome::Success
        } else {
            ToolOutcome::Error
        };

        Ok(ToolOutput {
            content: formatted,
            outcome,
            metadata,
        })
    }
}

// ── Output Parsing ───────────────────────────────────────────────────

/// Truncate raw output bytes to `MAX_OUTPUT_BYTES`, producing a lossy UTF-8 string.
fn truncate_output(raw: &[u8]) -> String {
    if raw.len() <= MAX_OUTPUT_BYTES {
        String::from_utf8_lossy(raw).into_owned()
    } else {
        let truncated = raw.get(..MAX_OUTPUT_BYTES).unwrap_or(raw);
        let mut s = String::from_utf8_lossy(truncated).into_owned();
        s.push_str("\n[output truncated]");
        s
    }
}

/// Parse `--message-format=json` output from cargo check/clippy/build/doc.
///
/// Each line of stdout is a JSON object. We extract compiler diagnostics
/// from objects where `reason == "compiler-message"`.
fn parse_cargo_json_diagnostics(stdout: &str, sandbox_root: &Path) -> Vec<Issue> {
    let sandbox_str = sandbox_root.to_string_lossy();
    let mut issues = Vec::new();

    for line in stdout.lines() {
        if issues.len() >= MAX_ISSUES_PER_STEP {
            break;
        }

        let Ok(msg) = serde_json::from_str::<serde_json::Value>(line) else {
            continue;
        };

        if msg.get("reason").and_then(serde_json::Value::as_str) != Some("compiler-message") {
            continue;
        }

        let Some(message) = msg.get("message") else {
            continue;
        };

        let Some(level_str) = message.get("level").and_then(serde_json::Value::as_str) else {
            continue;
        };

        let level = match level_str {
            "error" => IssueLevel::Error,
            "warning" => IssueLevel::Warning,
            _ => continue, // skip "note", "help", "ice" at top level
        };

        let text = message
            .get("message")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("(unknown)")
            .to_string();

        // Extract primary span
        let span = message
            .get("spans")
            .and_then(serde_json::Value::as_array)
            .and_then(|arr| {
                arr.iter()
                    .find(|s| {
                        s.get("is_primary")
                            .and_then(serde_json::Value::as_bool)
                            .unwrap_or(false)
                    })
                    .or_else(|| arr.first())
            });

        let file = span
            .and_then(|s| s.get("file_name"))
            .and_then(serde_json::Value::as_str)
            .map(|f| make_relative(f, &sandbox_str));

        let line_num = span
            .and_then(|s| s.get("line_start"))
            .and_then(serde_json::Value::as_u64)
            .and_then(|v| u32::try_from(v).ok());

        let column = span
            .and_then(|s| s.get("column_start"))
            .and_then(serde_json::Value::as_u64)
            .and_then(|v| u32::try_from(v).ok());

        // Collect first help message from children
        let help = message
            .get("children")
            .and_then(serde_json::Value::as_array)
            .and_then(|children| {
                children
                    .iter()
                    .filter(|c| c.get("level").and_then(serde_json::Value::as_str) == Some("help"))
                    .find_map(|c| c.get("message").and_then(serde_json::Value::as_str))
            })
            .map(String::from);

        issues.push(Issue {
            level,
            message: text,
            file,
            line: line_num,
            column,
            help,
        });
    }

    if issues.len() >= MAX_ISSUES_PER_STEP {
        issues.push(Issue {
            level: IssueLevel::Warning,
            message: format!("... output capped at {MAX_ISSUES_PER_STEP} issues"),
            file: None,
            line: None,
            column: None,
            help: None,
        });
    }

    issues
}

/// Fallback parser for rendered cargo diagnostics when JSON parsing yields nothing.
///
/// Parses lines like `error[E0308]: mismatched types` or `error: ...`
/// with optional `  --> file:line:col` location lines.
fn parse_rendered_diagnostics(text: &str, sandbox_root: &Path) -> Vec<Issue> {
    let sandbox_str = sandbox_root.to_string_lossy();
    let mut issues = Vec::new();

    let mut lines = text.lines().peekable();
    while let Some(line) = lines.next() {
        if issues.len() >= MAX_ISSUES_PER_STEP {
            break;
        }

        let Some((level, msg)) = parse_rendered_line(line) else {
            continue;
        };

        // Try to find a --> location line following this diagnostic
        let (file, line_num, column) = match lines
            .peek()
            .and_then(|next| parse_location_line(next, &sandbox_str))
        {
            Some((f, l, c)) => {
                let _ = lines.next(); // consume the location line
                (f, l, c)
            }
            None => (None, None, None),
        };

        issues.push(Issue {
            level,
            message: msg,
            file,
            line: line_num,
            column,
            help: None,
        });
    }

    issues
}

/// Try to parse a single rendered diagnostic line (`error: ...` or `warning: ...`).
fn parse_rendered_line(line: &str) -> Option<(IssueLevel, String)> {
    let trimmed = line.trim();

    let (level, rest) = if let Some(rest) = trimmed.strip_prefix("error") {
        (IssueLevel::Error, rest)
    } else if let Some(rest) = trimmed.strip_prefix("warning") {
        (IssueLevel::Warning, rest)
    } else {
        return None;
    };

    let msg = rest
        .trim_start_matches(|c: char| c == '[' || c.is_alphanumeric() || c == ']')
        .trim_start_matches(':')
        .trim();

    if msg.is_empty()
        || msg.starts_with("could not compile")
        || msg.starts_with("aborting due to")
        || msg.starts_with("build failed")
    {
        return None;
    }

    Some((level, msg.to_string()))
}

/// Parse a `  --> file:line:col` location line.
fn parse_location_line(
    line: &str,
    sandbox_str: &str,
) -> Option<(Option<String>, Option<u32>, Option<u32>)> {
    let loc = line.trim().strip_prefix("--> ")?;
    let parts: Vec<&str> = loc.splitn(3, ':').collect();
    let f = parts.first().map(|s| make_relative(s, sandbox_str));
    let l = parts.get(1).and_then(|s| s.parse::<u32>().ok());
    let c = parts.get(2).and_then(|s| s.parse::<u32>().ok());
    Some((f, l, c))
}

/// Parse `cargo test` text output for failures.
fn parse_test_output(
    stdout: &str,
    stderr: &str,
    sandbox_root: &Path,
) -> (Vec<Issue>, Option<String>) {
    let sandbox_str = sandbox_root.to_string_lossy();
    let combined = format!("{stdout}\n{stderr}");
    let mut issues = Vec::new();

    // Extract test summary line
    let summary = combined
        .lines()
        .find(|l| l.starts_with("test result:"))
        .map(String::from);

    // Parse failure sections: look for "---- test_name stdout ----"
    // followed by panic messages
    let mut current_test: Option<String> = None;
    let mut panic_msg: Option<String> = None;
    let mut panic_location: Option<String> = None;

    for line in combined.lines() {
        if let Some(rest) = line.strip_prefix("---- ") {
            if let Some(name) = rest.strip_suffix(" stdout ----") {
                // Flush previous test if any
                if let Some(test_name) = current_test.take() {
                    if issues.len() < MAX_ISSUES_PER_STEP {
                        issues.push(build_test_issue(
                            &test_name,
                            panic_msg.as_deref(),
                            panic_location.as_deref(),
                            &sandbox_str,
                        ));
                    }
                }
                current_test = Some(name.to_string());
                panic_msg = None;
                panic_location = None;
            }
        } else if current_test.is_some() {
            if let Some(rest) = line.strip_prefix("thread '") {
                if let Some(msg_start) = rest.find("panicked at ") {
                    let after = &rest[msg_start + "panicked at ".len()..];
                    // Format: 'message', file:line:col
                    // Or: file:line:col:\nmessage (newer Rust)
                    let cleaned = after.trim_matches('\'').trim_matches(',').trim();
                    panic_msg = Some(cleaned.to_string());
                }
            } else if line.trim().starts_with("--> ") || line.contains("src/") {
                // Try to capture file:line location
                let trimmed = line.trim().trim_start_matches("--> ");
                if trimmed.contains(':') {
                    panic_location = Some(trimmed.to_string());
                }
            }
        }
    }

    // Flush last test
    if let Some(test_name) = current_test.take() {
        if issues.len() < MAX_ISSUES_PER_STEP {
            issues.push(build_test_issue(
                &test_name,
                panic_msg.as_deref(),
                panic_location.as_deref(),
                &sandbox_str,
            ));
        }
    }

    // If we found failures in the summary but didn't parse individual tests,
    // add a generic failure issue
    if issues.is_empty() {
        if let Some(ref sum) = summary {
            if sum.contains("FAILED") || sum.contains("failed") {
                issues.push(Issue {
                    level: IssueLevel::Error,
                    message: sum.clone(),
                    file: None,
                    line: None,
                    column: None,
                    help: None,
                });
            }
        }
    }

    (issues, summary)
}

/// Build an `Issue` for a failed test.
fn build_test_issue(
    test_name: &str,
    panic_msg: Option<&str>,
    panic_location: Option<&str>,
    sandbox_str: &str,
) -> Issue {
    let message = panic_msg.map_or_else(
        || format!("{test_name}: test failed"),
        |msg| format!("{test_name}: {msg}"),
    );

    let (file, line) = panic_location.map_or((None, None), |loc| parse_file_line(loc, sandbox_str));

    Issue {
        level: IssueLevel::Error,
        message,
        file,
        line,
        column: None,
        help: None,
    }
}

/// Parse a `file:line` or `file:line:col` string into components.
fn parse_file_line(loc: &str, sandbox_str: &str) -> (Option<String>, Option<u32>) {
    let parts: Vec<&str> = loc.splitn(3, ':').collect();
    if parts.len() >= 2 {
        let file = make_relative(parts.first().unwrap_or(&""), sandbox_str);
        let line = parts.get(1).and_then(|s| s.parse::<u32>().ok());
        (Some(file), line)
    } else {
        (None, None)
    }
}

/// Parse `cargo fmt --check` output.
fn parse_fmt_output(stdout: &str, sandbox_root: &Path) -> Vec<Issue> {
    let sandbox_str = sandbox_root.to_string_lossy();
    let mut issues = Vec::new();

    for line in stdout.lines() {
        if issues.len() >= MAX_ISSUES_PER_STEP {
            break;
        }

        // cargo fmt --check outputs "Diff in <path>:" lines
        if let Some(rest) = line.strip_prefix("Diff in ") {
            let file_path = rest.trim_end_matches(':').trim();
            issues.push(Issue {
                level: IssueLevel::Warning,
                message: format!("unformatted: {}", make_relative(file_path, &sandbox_str)),
                file: Some(make_relative(file_path, &sandbox_str)),
                line: None,
                column: None,
                help: Some("run `cargo fmt` to fix".into()),
            });
        }
    }

    issues
}

/// Parse `cargo deny check` output.
fn parse_deny_output(stderr: &str) -> Vec<Issue> {
    let mut issues = Vec::new();

    for line in stderr.lines() {
        if issues.len() >= MAX_ISSUES_PER_STEP {
            break;
        }

        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        // cargo deny outputs lines like "error[A001]: ..." or "warning[L002]: ..."
        let level = if trimmed.starts_with("error") {
            Some(IssueLevel::Error)
        } else if trimmed.starts_with("warning") {
            Some(IssueLevel::Warning)
        } else {
            None
        };

        if let Some(level) = level {
            issues.push(Issue {
                level,
                message: trimmed.to_string(),
                file: None,
                line: None,
                column: None,
                help: None,
            });
        }
    }

    issues
}

/// Strip sandbox root prefix from a path to produce a relative path.
fn make_relative(path: &str, sandbox_str: &str) -> String {
    path.strip_prefix(sandbox_str)
        .unwrap_or(path)
        .trim_start_matches('/')
        .to_string()
}

// ── Output Formatting ────────────────────────────────────────────────

/// Format `VerificationResult` as human-readable text.
fn format_result(result: &VerificationResult) -> String {
    let mut out = String::with_capacity(2048);

    for step in &result.steps {
        let status_label = match step.status {
            StepStatus::Pass => "PASS",
            StepStatus::Fail => "FAIL",
            StepStatus::Skipped => "SKIP",
        };

        let _ = writeln!(
            out,
            "── {} ── {status_label} ({}.{}s)",
            step.step.label(),
            step.duration_ms / 1000,
            (step.duration_ms % 1000) / 100,
        );

        if let Some(summary) = &step.summary {
            let _ = writeln!(out, "  {summary}");
        }

        for issue in &step.issues {
            let level = match issue.level {
                IssueLevel::Error => "error",
                IssueLevel::Warning => "warning",
            };

            let location = match (&issue.file, issue.line, issue.column) {
                (Some(f), Some(l), Some(c)) => format!("\n    → {f}:{l}:{c}"),
                (Some(f), Some(l), None) => format!("\n    → {f}:{l}"),
                (Some(f), None, None) => format!("\n    → {f}"),
                _ => String::new(),
            };

            let _ = writeln!(out, "  {level}: {}{location}", issue.message);

            if let Some(help) = &issue.help {
                let _ = writeln!(out, "    = help: {help}");
            }
        }

        if !step.issues.is_empty() {
            let errors = step
                .issues
                .iter()
                .filter(|i| i.level == IssueLevel::Error)
                .count();
            let warnings = step
                .issues
                .iter()
                .filter(|i| i.level == IssueLevel::Warning)
                .count();
            let _ = writeln!(out, "  {errors} error(s), {warnings} warning(s)");
        }

        let _ = writeln!(out);
    }

    let verdict_label = match result.verdict {
        Verdict::Pass => "PASS",
        Verdict::Fail => "FAIL",
        Verdict::Partial => "PARTIAL",
    };
    let _ = writeln!(
        out,
        "VERDICT: {verdict_label} ({}.{}s total)",
        result.total_duration_ms / 1000,
        (result.total_duration_ms % 1000) / 100,
    );

    out
}

// ── Factory ──────────────────────────────────────────────────────────

/// Factory: create the `cargo_verify` tool for registration.
#[must_use]
pub fn cargo_verify_tools() -> Vec<Box<dyn Tool>> {
    vec![Box::new(CargoVerifyTool::new())]
}

// ── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic, clippy::indexing_slicing)]
mod tests {
    use std::path::PathBuf;

    use freebird_traits::id::SessionId;
    use freebird_traits::tool::{
        Capability, RiskLevel, SideEffects, Tool, ToolContext, ToolError, ToolOutcome,
    };

    use super::*;

    // ── Test Harness ─────────────────────────────────────────────────

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

        fn context(&self) -> ToolContext<'_> {
            ToolContext {
                session_id: &self.session_id,
                sandbox_root: &self.sandbox,
                granted_capabilities: &self.capabilities,
                allowed_directories: &[],
                knowledge_store: None,
            }
        }

        fn path(&self) -> &Path {
            &self.sandbox
        }
    }

    /// Create a minimal Cargo project in the given directory.
    fn create_cargo_project(dir: &Path, src: &str) {
        let cargo_toml = r#"[package]
name = "test-crate"
version = "0.1.0"
edition = "2021"
"#;
        std::fs::write(dir.join("Cargo.toml"), cargo_toml).unwrap();
        std::fs::create_dir_all(dir.join("src")).unwrap();
        std::fs::write(dir.join("src/lib.rs"), src).unwrap();
    }

    /// Create a minimal Cargo project with a binary target.
    fn create_cargo_project_bin(dir: &Path, src: &str) {
        let cargo_toml = r#"[package]
name = "test-crate"
version = "0.1.0"
edition = "2021"
"#;
        std::fs::write(dir.join("Cargo.toml"), cargo_toml).unwrap();
        std::fs::create_dir_all(dir.join("src")).unwrap();
        std::fs::write(dir.join("src/main.rs"), src).unwrap();
    }

    fn make_tool() -> CargoVerifyTool {
        CargoVerifyTool::new()
    }

    // ── Info / Metadata Tests ────────────────────────────────────────

    #[test]
    fn test_info_returns_correct_metadata() {
        let tool = make_tool();
        let info = tool.info();
        assert_eq!(info.name, "cargo_verify");
        assert_eq!(info.required_capability, Capability::ShellExecute);
        assert_eq!(info.risk_level, RiskLevel::Medium);
        assert!(matches!(info.side_effects, SideEffects::HasSideEffects));
    }

    #[test]
    fn test_to_definition_matches_info() {
        let tool = make_tool();
        let def = tool.to_definition();
        assert_eq!(def.name, tool.info().name);
        assert_eq!(def.description, tool.info().description);
    }

    // ── Input Validation Tests ───────────────────────────────────────

    #[tokio::test]
    async fn test_unknown_check_rejected() {
        let h = TestHarness::new();
        let tool = make_tool();

        let err = tool
            .execute(serde_json::json!({"checks": ["banana"]}), &h.context())
            .await
            .unwrap_err();

        match &err {
            ToolError::InvalidInput { tool, reason } => {
                assert_eq!(tool, "cargo_verify");
                assert!(reason.contains("banana"), "reason: {reason}");
            }
            other => panic!("expected InvalidInput, got: {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_package_with_metacharacters_rejected() {
        let h = TestHarness::new();
        let tool = make_tool();

        let err = tool
            .execute(
                serde_json::json!({"checks": ["check"], "package": "bad;name"}),
                &h.context(),
            )
            .await
            .unwrap_err();

        assert!(matches!(err, ToolError::InvalidInput { .. }));
    }

    #[tokio::test]
    async fn test_test_filter_with_metacharacters_rejected() {
        let h = TestHarness::new();
        let tool = make_tool();

        let err = tool
            .execute(
                serde_json::json!({"checks": ["test"], "test_filter": "test | rm -rf /"}),
                &h.context(),
            )
            .await
            .unwrap_err();

        assert!(matches!(err, ToolError::InvalidInput { .. }));
    }

    // ── Pipeline Tests ───────────────────────────────────────────────

    #[tokio::test]
    async fn test_clean_project_passes_check() {
        let h = TestHarness::new();
        create_cargo_project(h.path(), "pub fn add(a: i32, b: i32) -> i32 { a + b }");
        let tool = make_tool();

        let output = tool
            .execute(serde_json::json!({"checks": ["check"]}), &h.context())
            .await
            .unwrap();

        assert!(
            matches!(output.outcome, ToolOutcome::Success),
            "expected success, got: {}",
            output.content
        );
        assert!(
            output.content.contains("PASS"),
            "expected PASS in output: {}",
            output.content
        );

        // Verify metadata is parseable
        let meta = output.metadata.unwrap();
        let result: VerificationResult = serde_json::from_value(meta).unwrap();
        assert_eq!(result.verdict, Verdict::Pass);
        assert_eq!(result.steps.len(), 1);
        assert_eq!(result.steps[0].status, StepStatus::Pass);
    }

    #[tokio::test]
    async fn test_syntax_error_fails_check() {
        let h = TestHarness::new();
        create_cargo_project(h.path(), "fn broken( {");
        let tool = make_tool();

        let output = tool
            .execute(serde_json::json!({"checks": ["check"]}), &h.context())
            .await
            .unwrap();

        assert!(matches!(output.outcome, ToolOutcome::Error));
        assert!(
            output.content.contains("FAIL"),
            "output: {}",
            output.content
        );

        let meta = output.metadata.unwrap();
        let result: VerificationResult = serde_json::from_value(meta).unwrap();
        assert_eq!(result.verdict, Verdict::Fail);
        assert!(!result.steps[0].issues.is_empty(), "should have issues");
        assert_eq!(result.steps[0].issues[0].level, IssueLevel::Error);
    }

    #[tokio::test]
    async fn test_clippy_warning_fails_with_deny() {
        let h = TestHarness::new();
        // `len() == 0` triggers clippy::len_zero warning
        create_cargo_project(
            h.path(),
            r"
pub fn check_empty(v: &Vec<i32>) -> bool {
    v.len() == 0
}
",
        );
        let tool = make_tool();

        let output = tool
            .execute(serde_json::json!({"checks": ["clippy"]}), &h.context())
            .await
            .unwrap();

        assert!(matches!(output.outcome, ToolOutcome::Error));

        let meta = output.metadata.unwrap();
        let result: VerificationResult = serde_json::from_value(meta).unwrap();
        assert_eq!(result.steps[0].step, VerifyStep::Clippy);
        assert_eq!(result.steps[0].status, StepStatus::Fail);
        // Should have at least one warning/error about len_zero or similar
        assert!(
            !result.steps[0].issues.is_empty(),
            "should have clippy issues"
        );
    }

    #[tokio::test]
    async fn test_failing_test_detected() {
        let h = TestHarness::new();
        create_cargo_project(
            h.path(),
            r#"
#[cfg(test)]
mod tests {
    #[test]
    fn it_fails() {
        assert_eq!(1, 2, "one is not two");
    }
}
"#,
        );
        let tool = make_tool();

        let output = tool
            .execute(serde_json::json!({"checks": ["test"]}), &h.context())
            .await
            .unwrap();

        assert!(matches!(output.outcome, ToolOutcome::Error));
        assert!(
            output.content.contains("FAIL"),
            "output: {}",
            output.content
        );

        let meta = output.metadata.unwrap();
        let result: VerificationResult = serde_json::from_value(meta).unwrap();
        assert_eq!(result.steps[0].status, StepStatus::Fail);
        assert!(result.steps[0].summary.is_some());
    }

    #[tokio::test]
    async fn test_fmt_unformatted_detected() {
        let h = TestHarness::new();
        // Intentionally bad formatting
        create_cargo_project_bin(h.path(), "fn main(){let x=1;println!(\"{}\",x);}");
        let tool = make_tool();

        let output = tool
            .execute(serde_json::json!({"checks": ["fmt"]}), &h.context())
            .await
            .unwrap();

        assert!(matches!(output.outcome, ToolOutcome::Error));
        assert!(
            output.content.contains("FAIL"),
            "output: {}",
            output.content
        );
    }

    #[tokio::test]
    async fn test_all_stops_on_first_failure() {
        let h = TestHarness::new();
        // Syntax error will fail check, so clippy/test/fmt should be skipped
        create_cargo_project(h.path(), "fn broken( {");
        let tool = make_tool();

        let output = tool
            .execute(serde_json::json!({"checks": ["all"]}), &h.context())
            .await
            .unwrap();

        let meta = output.metadata.unwrap();
        let result: VerificationResult = serde_json::from_value(meta).unwrap();
        assert_eq!(result.steps.len(), 4);
        assert_eq!(result.steps[0].step, VerifyStep::Check);
        assert_eq!(result.steps[0].status, StepStatus::Fail);
        // Remaining steps should be skipped
        assert_eq!(result.steps[1].status, StepStatus::Skipped);
        assert_eq!(result.steps[2].status, StepStatus::Skipped);
        assert_eq!(result.steps[3].status, StepStatus::Skipped);
    }

    #[tokio::test]
    async fn test_all_expands_to_four_steps() {
        let h = TestHarness::new();
        create_cargo_project(h.path(), "pub fn ok() {}");
        let tool = make_tool();

        let output = tool
            .execute(serde_json::json!({"checks": ["all"]}), &h.context())
            .await
            .unwrap();

        let meta = output.metadata.unwrap();
        let result: VerificationResult = serde_json::from_value(meta).unwrap();
        assert_eq!(result.steps.len(), 4);
        assert_eq!(result.steps[0].step, VerifyStep::Check);
        assert_eq!(result.steps[1].step, VerifyStep::Clippy);
        assert_eq!(result.steps[2].step, VerifyStep::Test);
        assert_eq!(result.steps[3].step, VerifyStep::Fmt);
    }

    #[tokio::test]
    async fn test_individual_step_runs_only_that_step() {
        let h = TestHarness::new();
        create_cargo_project(h.path(), "pub fn ok() {}");
        let tool = make_tool();

        let output = tool
            .execute(serde_json::json!({"checks": ["test"]}), &h.context())
            .await
            .unwrap();

        let meta = output.metadata.unwrap();
        let result: VerificationResult = serde_json::from_value(meta).unwrap();
        assert_eq!(result.steps.len(), 1);
        assert_eq!(result.steps[0].step, VerifyStep::Test);
    }

    #[tokio::test]
    async fn test_empty_checks_defaults_to_all() {
        let h = TestHarness::new();
        create_cargo_project(h.path(), "pub fn ok() {}");
        let tool = make_tool();

        // No checks field at all
        let output = tool
            .execute(serde_json::json!({}), &h.context())
            .await
            .unwrap();

        let meta = output.metadata.unwrap();
        let result: VerificationResult = serde_json::from_value(meta).unwrap();
        assert_eq!(result.steps.len(), 4, "should default to all (4 steps)");
    }

    // ── Output Parsing Unit Tests ────────────────────────────────────

    #[test]
    fn test_parse_cargo_json_diagnostics() {
        let sandbox = PathBuf::from("/tmp/sandbox");
        let json_line = r#"{"reason":"compiler-message","package_id":"test","manifest_path":"","target":{},"message":{"message":"unused variable: `x`","code":{"code":"unused_variables"},"level":"warning","spans":[{"file_name":"/tmp/sandbox/src/lib.rs","byte_start":10,"byte_end":11,"line_start":3,"line_end":3,"column_start":9,"column_end":10,"is_primary":true}],"children":[{"message":"if this is intentional, prefix it with an underscore: `_x`","level":"help","spans":[],"children":[]}],"rendered":"warning: unused variable"}}"#;

        let issues = parse_cargo_json_diagnostics(json_line, &sandbox);
        assert_eq!(issues.len(), 1);
        assert_eq!(issues[0].level, IssueLevel::Warning);
        assert!(issues[0].message.contains("unused variable"));
        assert_eq!(issues[0].file.as_deref(), Some("src/lib.rs"));
        assert_eq!(issues[0].line, Some(3));
        assert_eq!(issues[0].column, Some(9));
        assert!(issues[0].help.as_ref().unwrap().contains("_x"));
    }

    #[test]
    fn test_parse_cargo_json_diagnostics_error() {
        let sandbox = PathBuf::from("/tmp/sandbox");
        let json_line = r#"{"reason":"compiler-message","message":{"message":"expected one of `!`, `(`, `)`, `,`","level":"error","spans":[{"file_name":"src/lib.rs","line_start":1,"column_start":5,"is_primary":true}],"children":[]}}"#;

        let issues = parse_cargo_json_diagnostics(json_line, &sandbox);
        assert_eq!(issues.len(), 1);
        assert_eq!(issues[0].level, IssueLevel::Error);
        assert_eq!(issues[0].file.as_deref(), Some("src/lib.rs"));
    }

    #[test]
    fn test_parse_cargo_json_skips_non_diagnostic_lines() {
        let sandbox = PathBuf::from("/tmp/sandbox");
        let lines = r#"{"reason":"compiler-artifact","package_id":"test","target":{}}
not json at all
{"reason":"build-finished","success":false}"#;

        let issues = parse_cargo_json_diagnostics(lines, &sandbox);
        assert!(issues.is_empty());
    }

    #[test]
    fn test_parse_test_output_failures() {
        let sandbox = PathBuf::from("/tmp/sandbox");
        let stdout = r"
running 2 tests
test tests::it_passes ... ok
test tests::it_fails ... FAILED

failures:

---- tests::it_fails stdout ----
thread 'tests::it_fails' panicked at 'assertion failed: one is not two'

failures:
    tests::it_fails

test result: FAILED. 1 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out
";

        let (issues, summary) = parse_test_output(stdout, "", &sandbox);
        assert!(!issues.is_empty(), "should have failure issues");
        assert!(issues[0].message.contains("it_fails"));
        assert!(summary.is_some());
        assert!(summary.unwrap().contains("1 failed"));
    }

    #[test]
    fn test_parse_fmt_output() {
        let sandbox = PathBuf::from("/tmp/sandbox");
        let stdout = "Diff in /tmp/sandbox/src/main.rs:\nDiff in /tmp/sandbox/src/lib.rs:\n";

        let issues = parse_fmt_output(stdout, &sandbox);
        assert_eq!(issues.len(), 2);
        assert_eq!(issues[0].file.as_deref(), Some("src/main.rs"));
        assert_eq!(issues[1].file.as_deref(), Some("src/lib.rs"));
        assert!(issues[0].help.is_some());
    }

    #[test]
    fn test_make_relative_strips_prefix() {
        assert_eq!(
            make_relative("/tmp/sandbox/src/lib.rs", "/tmp/sandbox"),
            "src/lib.rs"
        );
        assert_eq!(make_relative("src/lib.rs", "/tmp/sandbox"), "src/lib.rs");
    }

    #[test]
    fn test_metadata_json_parseable() {
        let result = VerificationResult {
            verdict: Verdict::Pass,
            steps: vec![StepResult {
                step: VerifyStep::Check,
                status: StepStatus::Pass,
                duration_ms: 100,
                issues: Vec::new(),
                summary: None,
            }],
            total_duration_ms: 100,
        };

        let json = serde_json::to_value(&result).unwrap();
        let parsed: VerificationResult = serde_json::from_value(json).unwrap();
        assert_eq!(parsed.verdict, Verdict::Pass);
    }

    #[test]
    fn test_no_ansi_codes_in_format_output() {
        let result = VerificationResult {
            verdict: Verdict::Fail,
            steps: vec![StepResult {
                step: VerifyStep::Check,
                status: StepStatus::Fail,
                duration_ms: 500,
                issues: vec![Issue {
                    level: IssueLevel::Error,
                    message: "something broke".into(),
                    file: Some("src/lib.rs".into()),
                    line: Some(42),
                    column: Some(5),
                    help: Some("try fixing it".into()),
                }],
                summary: None,
            }],
            total_duration_ms: 500,
        };

        let formatted = format_result(&result);
        // No ANSI escape codes
        assert!(
            !formatted.contains('\x1b'),
            "output should not contain ANSI codes: {formatted}"
        );
    }

    #[tokio::test]
    async fn test_cargo_not_found_returns_error() {
        let h = TestHarness::new();
        create_cargo_project(h.path(), "pub fn ok() {}");

        // Create a tool with an empty PATH so cargo can't be found
        let mut tool = make_tool();
        tool.home_dir = "/nonexistent".into();

        let output = tool
            .execute(serde_json::json!({"checks": ["check"]}), &h.context())
            .await
            .unwrap();

        // This returns a result (not a ToolError) because the step itself
        // reports the failure in structured output
        assert!(matches!(output.outcome, ToolOutcome::Error));
        assert!(
            output.content.contains("FAIL"),
            "output: {}",
            output.content
        );
    }

    #[test]
    fn test_factory_function() {
        let tools = cargo_verify_tools();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].info().name, "cargo_verify");
    }

    #[test]
    fn test_verify_step_from_str() {
        assert_eq!(VerifyStep::from_str("check"), Some(VerifyStep::Check));
        assert_eq!(VerifyStep::from_str("clippy"), Some(VerifyStep::Clippy));
        assert_eq!(VerifyStep::from_str("test"), Some(VerifyStep::Test));
        assert_eq!(VerifyStep::from_str("fmt"), Some(VerifyStep::Fmt));
        assert_eq!(VerifyStep::from_str("deny"), Some(VerifyStep::Deny));
        assert_eq!(VerifyStep::from_str("doc"), Some(VerifyStep::Doc));
        assert_eq!(VerifyStep::from_str("build"), Some(VerifyStep::Build));
        assert_eq!(VerifyStep::from_str("banana"), None);
    }

    #[test]
    fn test_parse_rendered_diagnostics() {
        let sandbox = PathBuf::from("/tmp/sandbox");
        let text = "\
error[E0308]: mismatched types
  --> src/lib.rs:5:10
warning: unused variable `x`
  --> src/lib.rs:3:9
error: aborting due to 3 previous errors
warning: build failed, waiting for other jobs to finish...
error: could not compile `test-crate`
";
        let issues = parse_rendered_diagnostics(text, &sandbox);
        // Should only capture the real diagnostics, not meta-messages
        assert_eq!(issues.len(), 2, "issues: {issues:?}");
        assert_eq!(issues[0].level, IssueLevel::Error);
        assert!(issues[0].message.contains("mismatched types"));
        assert_eq!(issues[0].file.as_deref(), Some("src/lib.rs"));
        assert_eq!(issues[0].line, Some(5));
        assert_eq!(issues[1].level, IssueLevel::Warning);
        assert!(issues[1].message.contains("unused variable"));
    }

    #[test]
    fn test_parse_deny_output() {
        let stderr =
            "error[A001]: serde 1.0.0 is vulnerable\nwarning[L003]: MIT license detected\n";
        let issues = parse_deny_output(stderr);
        assert_eq!(issues.len(), 2);
        assert_eq!(issues[0].level, IssueLevel::Error);
        assert_eq!(issues[1].level, IssueLevel::Warning);
    }
}
