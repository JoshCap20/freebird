//! Secret-aware tool guardrail — detects and gates tool invocations that
//! access sensitive files, run secret-revealing commands, or produce output
//! containing credentials.
//!
//! Sits in the [`ToolExecutor`] pipeline between capability check and consent
//! gate. Inspects tool inputs for sensitive file/command patterns and
//! optionally redacts secrets in tool outputs.
//!
//! See issue #125 and CLAUDE.md §19.

use std::path::Path;

use freebird_types::config::{SecretGuardAction, SecretGuardConfig};
use regex::Regex;

use crate::sensitive::redact_sensitive_content;

/// Result of checking a tool input for secret access patterns.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SecretCheckResult {
    /// No sensitive patterns detected — proceed normally.
    Safe,
    /// Sensitive pattern detected — escalate to consent gate with
    /// `RiskLevel::Critical` regardless of the tool's declared risk level.
    RequiresConsent {
        /// Human-readable reason shown in the consent prompt.
        reason: String,
    },
    /// Sensitive pattern detected and configured to block outright.
    Blocked {
        /// Human-readable reason returned as tool error.
        reason: String,
    },
}

/// Guards against tool invocations that access or expose secrets.
///
/// Constructed once at startup from [`SecretGuardConfig`] and shared
/// immutably for the lifetime of the runtime.
pub struct SecretGuard {
    /// Compiled regexes for sensitive shell command detection.
    command_patterns: Vec<Regex>,
    /// Action to take on detection (consent or block).
    action: SecretGuardAction,
    /// Whether to redact secrets in tool output.
    redact_output: bool,
    /// Extra file patterns from config (stored for matching).
    extra_file_patterns: Vec<String>,
}

impl std::fmt::Debug for SecretGuard {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SecretGuard")
            .field("action", &self.action)
            .field("redact_output", &self.redact_output)
            .field("command_pattern_count", &self.command_patterns.len())
            .field("extra_file_patterns", &self.extra_file_patterns)
            .finish()
    }
}

// ── Built-in sensitive file patterns ────────────────────────────────────

/// File patterns that always require consent/blocking when accessed.
/// Matched against the filename component (last segment of the path).
const SENSITIVE_EXACT_FILENAMES: &[&str] = &[
    ".env",
    ".netrc",
    ".pgpass",
    ".npmrc",
    ".pypirc",
    "keys.json",
    "credentials",
    "config.json", // Docker config with auth
];

/// File prefixes — if the filename starts with this, it's sensitive.
const SENSITIVE_FILE_PREFIXES: &[&str] = &[
    ".env.", // .env.local, .env.production, .env.development, etc.
    "secrets.",
    "credentials.",
];

/// File suffixes — if the filename ends with this, it's sensitive.
const SENSITIVE_FILE_SUFFIXES: &[&str] =
    &[".env", ".key", ".pem", ".p12", ".pfx", ".jks", ".secret"];

/// Directory components that make any file under them sensitive.
const SENSITIVE_DIRECTORIES: &[&str] = &[".ssh", ".aws", ".gnupg", ".docker"];

// ── Built-in sensitive command patterns ─────────────────────────────────

/// Regex patterns for shell commands that access secrets.
/// Compiled once at startup via `SecretGuard::from_config`.
const BUILTIN_COMMAND_PATTERNS: &[&str] = &[
    // Direct env access
    r"^\s*env\b",
    r"^\s*printenv\b",
    r"\benv\s",
    r"\bprintenv\b",
    r"echo\s+\$\w+",
    r"printf\s.*\$\w+",
    // Reading known secret files
    r"cat\s+.*\.env\b",
    r"less\s+.*\.env\b",
    r"more\s+.*\.env\b",
    r"head\s+.*\.env\b",
    r"tail\s+.*\.env\b",
    r"cat\s+.*credentials\b",
    r"cat\s+.*/\.ssh/",
    r"cat\s+.*\.key\b",
    r"cat\s+.*\.pem\b",
    // Proc environ
    r"cat\s+/proc/.*/environ",
    r"strings\s+/proc/.*/environ",
    // Exporting/setting secrets
    r"export\s+\w+=",
    // Curl/wget with auth headers
    r"curl\s.*(-H|--header)\s.*[Aa]uth",
    r"curl\s.*(-u|--user)\s",
    r"wget\s.*--header.*[Aa]uth",
    // Git credential exposure
    r"git\s+config\s.*credential",
    r"git\s+credential\b",
];

/// Tools that operate on file paths and should have their path inputs checked.
const FILE_TOOLS: &[&str] = &[
    "read_file",
    "write_file",
    "file_viewer",
    "search_replace_edit",
    "grep_search",
    "list_directory",
];

/// Tools that execute shell commands and should have their command inputs checked.
const SHELL_TOOLS: &[&str] = &["shell"];

/// JSON keys that commonly hold file paths in tool inputs.
const PATH_KEYS: &[&str] = &["path", "file_path", "file", "target", "directory"];

/// JSON keys that commonly hold shell commands in tool inputs.
const COMMAND_KEYS: &[&str] = &["command", "cmd"];

// ── File pattern matching ───────────────────────────────────────────────

#[allow(clippy::option_if_let_else)] // if-let chain is clearer than map_or_else here
fn matches_extra_pattern(filename: &str, pattern: &str) -> bool {
    if let Some(suffix) = pattern.strip_prefix('*') {
        // *.ext pattern
        filename.ends_with(suffix)
    } else if let Some(prefix) = pattern.strip_suffix('*') {
        // prefix* pattern
        filename.starts_with(prefix)
    } else {
        // Exact match
        filename == pattern
    }
}

/// Check if a file path matches any sensitive pattern.
fn is_sensitive_path(path_str: &str, extra_patterns: &[String]) -> Option<String> {
    let path = Path::new(path_str);

    // Check directory components
    for component in path.components() {
        let comp_str = component.as_os_str().to_string_lossy();
        for dir in SENSITIVE_DIRECTORIES {
            if comp_str == *dir {
                return Some(format!("path contains sensitive directory `{dir}`"));
            }
        }
    }

    // Check filename
    let filename = path
        .file_name()
        .map(|f| f.to_string_lossy())
        .unwrap_or_default();

    if filename.is_empty() {
        return None;
    }

    // Exact filename match
    for name in SENSITIVE_EXACT_FILENAMES {
        if filename == *name {
            return Some(format!("sensitive file `{filename}`"));
        }
    }

    // Prefix match
    for prefix in SENSITIVE_FILE_PREFIXES {
        if filename.starts_with(prefix) {
            return Some(format!(
                "file `{filename}` matches sensitive prefix `{prefix}`"
            ));
        }
    }

    // Suffix match
    for suffix in SENSITIVE_FILE_SUFFIXES {
        if filename.ends_with(suffix) {
            return Some(format!(
                "file `{filename}` matches sensitive suffix `{suffix}`"
            ));
        }
    }

    // Extra user-defined patterns
    for pattern in extra_patterns {
        if matches_extra_pattern(&filename, pattern) {
            return Some(format!(
                "file `{filename}` matches custom pattern `{pattern}`"
            ));
        }
    }

    None
}

// ── Implementation ──────────────────────────────────────────────────────

impl SecretGuard {
    /// Construct a `SecretGuard` from configuration.
    ///
    /// Compiles all regex patterns at construction time so that per-call
    /// overhead is minimal.
    ///
    /// # Errors
    ///
    /// Returns an error if any user-provided regex pattern is invalid.
    pub fn from_config(config: &SecretGuardConfig) -> Result<Self, crate::error::SecurityError> {
        let mut patterns = Vec::with_capacity(
            BUILTIN_COMMAND_PATTERNS.len() + config.extra_sensitive_command_patterns.len(),
        );

        for pattern_str in BUILTIN_COMMAND_PATTERNS {
            let re = Regex::new(pattern_str).map_err(|e| {
                crate::error::SecurityError::SecretGuardConfigError {
                    reason: format!("invalid built-in command pattern `{pattern_str}`: {e}"),
                }
            })?;
            patterns.push(re);
        }

        for pattern_str in &config.extra_sensitive_command_patterns {
            let re = Regex::new(pattern_str).map_err(|e| {
                crate::error::SecurityError::SecretGuardConfigError {
                    reason: format!("invalid custom command pattern `{pattern_str}`: {e}"),
                }
            })?;
            patterns.push(re);
        }

        Ok(Self {
            command_patterns: patterns,
            action: config.action.clone(),
            redact_output: config.redact_output,
            extra_file_patterns: config.extra_sensitive_file_patterns.clone(),
        })
    }

    /// Check a tool invocation for sensitive patterns in its input.
    ///
    /// Inspects the tool name and input JSON to determine if the invocation
    /// targets sensitive files or runs secret-revealing commands.
    #[must_use]
    pub fn check_tool_input(
        &self,
        tool_name: &str,
        input: &serde_json::Value,
    ) -> SecretCheckResult {
        // 1. Check file tools for sensitive paths
        if FILE_TOOLS.contains(&tool_name) {
            if let Some(reason) = self.check_file_input(input) {
                return self.make_result(&reason);
            }
        }

        // 2. Check shell tools for sensitive commands
        if SHELL_TOOLS.contains(&tool_name) {
            if let Some(reason) = self.check_command_input(input) {
                return self.make_result(&reason);
            }
        }

        SecretCheckResult::Safe
    }

    /// Redact secrets in tool output text.
    ///
    /// Returns `(redacted_content, was_redacted)`. Only called when
    /// `redact_output` is configured.
    #[must_use]
    pub const fn should_redact_output(&self) -> bool {
        self.redact_output
    }

    /// Redact secrets from tool output content.
    ///
    /// Delegates to [`redact_sensitive_content`].
    #[must_use]
    pub fn redact_output(content: &str) -> (String, bool) {
        redact_sensitive_content(content)
    }

    // ── Private helpers ─────────────────────────────────────────────

    fn check_file_input(&self, input: &serde_json::Value) -> Option<String> {
        for key in PATH_KEYS {
            if let Some(path_str) = input.get(*key).and_then(serde_json::Value::as_str) {
                if let Some(reason) = is_sensitive_path(path_str, &self.extra_file_patterns) {
                    return Some(reason);
                }
            }
        }
        None
    }

    fn check_command_input(&self, input: &serde_json::Value) -> Option<String> {
        for key in COMMAND_KEYS {
            if let Some(cmd) = input.get(*key).and_then(serde_json::Value::as_str) {
                for pattern in &self.command_patterns {
                    if pattern.is_match(cmd) {
                        return Some(format!(
                            "command matches sensitive pattern `{}`",
                            pattern.as_str()
                        ));
                    }
                }
            }
        }
        None
    }

    fn make_result(&self, reason: &str) -> SecretCheckResult {
        match self.action {
            SecretGuardAction::Consent => SecretCheckResult::RequiresConsent {
                reason: reason.to_owned(),
            },
            SecretGuardAction::Block => SecretCheckResult::Blocked {
                reason: reason.to_owned(),
            },
        }
    }
}

// ── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn default_guard() -> SecretGuard {
        SecretGuard::from_config(&SecretGuardConfig::default()).unwrap()
    }

    fn block_guard() -> SecretGuard {
        SecretGuard::from_config(&SecretGuardConfig {
            action: SecretGuardAction::Block,
            ..SecretGuardConfig::default()
        })
        .unwrap()
    }

    fn guard_with_extra_file_patterns(patterns: Vec<String>) -> SecretGuard {
        SecretGuard::from_config(&SecretGuardConfig {
            extra_sensitive_file_patterns: patterns,
            ..SecretGuardConfig::default()
        })
        .unwrap()
    }

    fn guard_with_extra_command_patterns(patterns: Vec<String>) -> SecretGuard {
        SecretGuard::from_config(&SecretGuardConfig {
            extra_sensitive_command_patterns: patterns,
            ..SecretGuardConfig::default()
        })
        .unwrap()
    }

    fn file_input(path: &str) -> serde_json::Value {
        serde_json::json!({"path": path})
    }

    fn shell_input(cmd: &str) -> serde_json::Value {
        serde_json::json!({"command": cmd})
    }

    fn is_consent(result: &SecretCheckResult) -> bool {
        matches!(result, SecretCheckResult::RequiresConsent { .. })
    }

    fn is_blocked(result: &SecretCheckResult) -> bool {
        matches!(result, SecretCheckResult::Blocked { .. })
    }

    // ── File access tests ──

    #[test]
    fn test_read_env_requires_consent() {
        let guard = default_guard();
        let result = guard.check_tool_input("read_file", &file_input(".env"));
        assert!(is_consent(&result), "expected consent, got {result:?}");
    }

    #[test]
    fn test_read_env_local_requires_consent() {
        let guard = default_guard();
        let result = guard.check_tool_input("read_file", &file_input(".env.local"));
        assert!(is_consent(&result), "expected consent, got {result:?}");
    }

    #[test]
    fn test_read_env_production_requires_consent() {
        let guard = default_guard();
        let result = guard.check_tool_input("read_file", &file_input(".env.production"));
        assert!(is_consent(&result), "expected consent, got {result:?}");
    }

    #[test]
    fn test_read_ssh_key_requires_consent() {
        let guard = default_guard();
        let result = guard.check_tool_input("read_file", &file_input("/home/user/.ssh/id_rsa"));
        assert!(is_consent(&result), "expected consent, got {result:?}");
    }

    #[test]
    fn test_read_aws_credentials_requires_consent() {
        let guard = default_guard();
        let result =
            guard.check_tool_input("read_file", &file_input("/home/user/.aws/credentials"));
        assert!(is_consent(&result), "expected consent, got {result:?}");
    }

    #[test]
    fn test_read_pem_requires_consent() {
        let guard = default_guard();
        let result = guard.check_tool_input("read_file", &file_input("server.pem"));
        assert!(is_consent(&result), "expected consent, got {result:?}");
    }

    #[test]
    fn test_read_key_file_requires_consent() {
        let guard = default_guard();
        let result = guard.check_tool_input("read_file", &file_input("tls.key"));
        assert!(is_consent(&result), "expected consent, got {result:?}");
    }

    #[test]
    fn test_read_keys_json_requires_consent() {
        let guard = default_guard();
        let result = guard.check_tool_input("read_file", &file_input("keys.json"));
        assert!(is_consent(&result), "expected consent, got {result:?}");
    }

    #[test]
    fn test_read_npmrc_requires_consent() {
        let guard = default_guard();
        let result = guard.check_tool_input("read_file", &file_input("/home/user/.npmrc"));
        assert!(is_consent(&result), "expected consent, got {result:?}");
    }

    #[test]
    fn test_read_normal_file_no_consent() {
        let guard = default_guard();
        let result = guard.check_tool_input("read_file", &file_input("src/main.rs"));
        assert_eq!(result, SecretCheckResult::Safe);
    }

    #[test]
    fn test_read_readme_no_consent() {
        let guard = default_guard();
        let result = guard.check_tool_input("read_file", &file_input("README.md"));
        assert_eq!(result, SecretCheckResult::Safe);
    }

    #[test]
    fn test_read_data_csv_no_consent() {
        let guard = default_guard();
        let result = guard.check_tool_input("read_file", &file_input("data.csv"));
        assert_eq!(result, SecretCheckResult::Safe);
    }

    #[test]
    fn test_env_substring_no_false_positive() {
        let guard = default_guard();
        let result = guard.check_tool_input("read_file", &file_input("environment.rs"));
        assert_eq!(result, SecretCheckResult::Safe);
    }

    #[test]
    fn test_write_env_requires_consent() {
        let guard = default_guard();
        let result = guard.check_tool_input("write_file", &file_input(".env"));
        assert!(is_consent(&result), "expected consent, got {result:?}");
    }

    #[test]
    fn test_viewer_env_requires_consent() {
        let guard = default_guard();
        let result = guard.check_tool_input("file_viewer", &file_input(".env"));
        assert!(is_consent(&result), "expected consent, got {result:?}");
    }

    #[test]
    fn test_grep_env_requires_consent() {
        let guard = default_guard();
        let result = guard.check_tool_input("grep_search", &file_input(".env"));
        assert!(is_consent(&result), "expected consent, got {result:?}");
    }

    #[test]
    fn test_edit_env_requires_consent() {
        let guard = default_guard();
        let result = guard.check_tool_input("search_replace_edit", &file_input(".env"));
        assert!(is_consent(&result), "expected consent, got {result:?}");
    }

    #[test]
    fn test_secrets_toml_requires_consent() {
        let guard = default_guard();
        let result = guard.check_tool_input("read_file", &file_input("secrets.toml"));
        assert!(is_consent(&result), "expected consent, got {result:?}");
    }

    #[test]
    fn test_p12_requires_consent() {
        let guard = default_guard();
        let result = guard.check_tool_input("read_file", &file_input("cert.p12"));
        assert!(is_consent(&result), "expected consent, got {result:?}");
    }

    #[test]
    fn test_pfx_requires_consent() {
        let guard = default_guard();
        let result = guard.check_tool_input("read_file", &file_input("cert.pfx"));
        assert!(is_consent(&result), "expected consent, got {result:?}");
    }

    #[test]
    fn test_path_with_directory() {
        let guard = default_guard();
        let result = guard.check_tool_input("read_file", &file_input("/project/config/.env.local"));
        assert!(is_consent(&result), "expected consent, got {result:?}");
    }

    #[test]
    fn test_gnupg_directory_requires_consent() {
        let guard = default_guard();
        let result =
            guard.check_tool_input("read_file", &file_input("/home/user/.gnupg/pubring.kbx"));
        assert!(is_consent(&result), "expected consent, got {result:?}");
    }

    #[test]
    fn test_docker_config_requires_consent() {
        let guard = default_guard();
        let result =
            guard.check_tool_input("read_file", &file_input("/home/user/.docker/config.json"));
        assert!(is_consent(&result), "expected consent, got {result:?}");
    }

    // ── Shell command tests ──

    #[test]
    fn test_env_command_requires_consent() {
        let guard = default_guard();
        let result = guard.check_tool_input("shell", &shell_input("env"));
        assert!(is_consent(&result), "expected consent, got {result:?}");
    }

    #[test]
    fn test_printenv_requires_consent() {
        let guard = default_guard();
        let result = guard.check_tool_input("shell", &shell_input("printenv"));
        assert!(is_consent(&result), "expected consent, got {result:?}");
    }

    #[test]
    fn test_echo_var_requires_consent() {
        let guard = default_guard();
        let result = guard.check_tool_input("shell", &shell_input("echo $API_KEY"));
        assert!(is_consent(&result), "expected consent, got {result:?}");
    }

    #[test]
    fn test_cat_env_requires_consent() {
        let guard = default_guard();
        let result = guard.check_tool_input("shell", &shell_input("cat .env"));
        assert!(is_consent(&result), "expected consent, got {result:?}");
    }

    #[test]
    fn test_cat_credentials_requires_consent() {
        let guard = default_guard();
        let result = guard.check_tool_input("shell", &shell_input("cat ~/.aws/credentials"));
        assert!(is_consent(&result), "expected consent, got {result:?}");
    }

    #[test]
    fn test_curl_auth_requires_consent() {
        let guard = default_guard();
        let result = guard.check_tool_input(
            "shell",
            &shell_input("curl -H 'Authorization: Bearer token123' https://api.example.com"),
        );
        assert!(is_consent(&result), "expected consent, got {result:?}");
    }

    #[test]
    fn test_export_requires_consent() {
        let guard = default_guard();
        let result = guard.check_tool_input("shell", &shell_input("export SECRET=foo"));
        assert!(is_consent(&result), "expected consent, got {result:?}");
    }

    #[test]
    fn test_proc_environ_requires_consent() {
        let guard = default_guard();
        let result = guard.check_tool_input("shell", &shell_input("cat /proc/1/environ"));
        assert!(is_consent(&result), "expected consent, got {result:?}");
    }

    #[test]
    fn test_head_env_requires_consent() {
        let guard = default_guard();
        let result = guard.check_tool_input("shell", &shell_input("head .env"));
        assert!(is_consent(&result), "expected consent, got {result:?}");
    }

    #[test]
    fn test_cat_pem_requires_consent() {
        let guard = default_guard();
        let result = guard.check_tool_input("shell", &shell_input("cat server.pem"));
        assert!(is_consent(&result), "expected consent, got {result:?}");
    }

    #[test]
    fn test_curl_user_requires_consent() {
        let guard = default_guard();
        let result = guard.check_tool_input(
            "shell",
            &shell_input("curl -u admin:password https://api.example.com"),
        );
        assert!(is_consent(&result), "expected consent, got {result:?}");
    }

    #[test]
    fn test_git_credential_requires_consent() {
        let guard = default_guard();
        let result = guard.check_tool_input("shell", &shell_input("git credential fill"));
        assert!(is_consent(&result), "expected consent, got {result:?}");
    }

    #[test]
    fn test_cargo_test_no_consent() {
        let guard = default_guard();
        let result = guard.check_tool_input("shell", &shell_input("cargo test"));
        assert_eq!(result, SecretCheckResult::Safe);
    }

    #[test]
    fn test_ls_no_consent() {
        let guard = default_guard();
        let result = guard.check_tool_input("shell", &shell_input("ls -la"));
        assert_eq!(result, SecretCheckResult::Safe);
    }

    #[test]
    fn test_git_status_no_consent() {
        let guard = default_guard();
        let result = guard.check_tool_input("shell", &shell_input("git status"));
        assert_eq!(result, SecretCheckResult::Safe);
    }

    #[test]
    fn test_grep_pattern_no_consent() {
        let guard = default_guard();
        let result = guard.check_tool_input("shell", &shell_input("grep -r 'TODO' src/"));
        assert_eq!(result, SecretCheckResult::Safe);
    }

    // ── Action mode tests ──

    #[test]
    fn test_block_mode_blocks() {
        let guard = block_guard();
        let result = guard.check_tool_input("read_file", &file_input(".env"));
        assert!(is_blocked(&result), "expected blocked, got {result:?}");
    }

    #[test]
    fn test_consent_mode_requires_consent() {
        let guard = default_guard();
        let result = guard.check_tool_input("read_file", &file_input(".env"));
        assert!(is_consent(&result), "expected consent, got {result:?}");
    }

    #[test]
    fn test_block_mode_safe_passes() {
        let guard = block_guard();
        let result = guard.check_tool_input("read_file", &file_input("src/main.rs"));
        assert_eq!(result, SecretCheckResult::Safe);
    }

    // ── Custom pattern tests ──

    #[test]
    fn test_custom_file_pattern_star_suffix() {
        let guard = guard_with_extra_file_patterns(vec!["*.secret".into()]);
        let result = guard.check_tool_input("read_file", &file_input("app.secret"));
        assert!(is_consent(&result), "expected consent, got {result:?}");
    }

    #[test]
    fn test_custom_file_pattern_no_match() {
        let guard = guard_with_extra_file_patterns(vec!["*.secret".into()]);
        let result = guard.check_tool_input("read_file", &file_input("app.txt"));
        assert_eq!(result, SecretCheckResult::Safe);
    }

    #[test]
    fn test_custom_command_pattern() {
        let guard = guard_with_extra_command_patterns(vec![r"^myutil\b".into()]);
        let result = guard.check_tool_input("shell", &shell_input("myutil --dump"));
        assert!(is_consent(&result), "expected consent, got {result:?}");
    }

    #[test]
    fn test_custom_command_pattern_no_match() {
        let guard = guard_with_extra_command_patterns(vec![r"^myutil\b".into()]);
        let result = guard.check_tool_input("shell", &shell_input("cargo test"));
        assert_eq!(result, SecretCheckResult::Safe);
    }

    // ── Edge cases ──

    #[test]
    fn test_unknown_tool_returns_safe() {
        let guard = default_guard();
        let result = guard.check_tool_input("unknown_tool", &file_input(".env"));
        assert_eq!(result, SecretCheckResult::Safe);
    }

    #[test]
    fn test_missing_path_key_returns_safe() {
        let guard = default_guard();
        let result = guard.check_tool_input("read_file", &serde_json::json!({"content": "hello"}));
        assert_eq!(result, SecretCheckResult::Safe);
    }

    #[test]
    fn test_missing_command_key_returns_safe() {
        let guard = default_guard();
        let result = guard.check_tool_input("shell", &serde_json::json!({"args": ["ls"]}));
        assert_eq!(result, SecretCheckResult::Safe);
    }

    #[test]
    fn test_empty_path_returns_safe() {
        let guard = default_guard();
        let result = guard.check_tool_input("read_file", &file_input(""));
        assert_eq!(result, SecretCheckResult::Safe);
    }

    #[test]
    fn test_path_via_file_path_key() {
        let guard = default_guard();
        let input = serde_json::json!({"file_path": ".env"});
        let result = guard.check_tool_input("read_file", &input);
        assert!(is_consent(&result), "expected consent, got {result:?}");
    }

    #[test]
    fn test_invalid_regex_in_config_errors() {
        let result = SecretGuard::from_config(&SecretGuardConfig {
            extra_sensitive_command_patterns: vec!["[invalid".into()],
            ..SecretGuardConfig::default()
        });
        assert!(result.is_err());
    }

    // ── Output redaction ──

    #[test]
    fn test_redact_output_delegates() {
        let (result, redacted) = SecretGuard::redact_output("my key is sk-abc123def456");
        assert!(redacted);
        assert!(result.contains("[REDACTED]"));
        assert!(!result.contains("sk-abc123def456"));
    }

    #[test]
    fn test_should_redact_output_config() {
        let guard = default_guard();
        assert!(guard.should_redact_output());

        let guard_no_redact = SecretGuard::from_config(&SecretGuardConfig {
            redact_output: false,
            ..SecretGuardConfig::default()
        })
        .unwrap();
        assert!(!guard_no_redact.should_redact_output());
    }

    // ── Glob matching helper ──

    #[test]
    fn test_matches_extra_pattern_star_prefix() {
        assert!(matches_extra_pattern("config.secret", "*.secret"));
        assert!(!matches_extra_pattern("config.txt", "*.secret"));
    }

    #[test]
    fn test_matches_extra_pattern_star_suffix() {
        assert!(matches_extra_pattern(".env.local", ".env.*"));
    }

    #[test]
    fn test_matches_extra_pattern_exact() {
        assert!(matches_extra_pattern("keys.json", "keys.json"));
        assert!(!matches_extra_pattern("other.json", "keys.json"));
    }
}
