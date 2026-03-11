//! Shared utilities for tool implementations.
//!
//! Constants and helpers used across multiple tools (grep, glob, etc.)
//! to maintain a single source of truth.

/// Minimal PATH for sandboxed command execution.
///
/// Only standard system directories. Prevents PATH hijacking where an
/// attacker places a malicious binary earlier in PATH.
///
/// Shared by `shell` and `bash` tools.
pub const SANDBOXED_PATH: &str = "/usr/local/bin:/usr/bin:/bin";

/// Directories to always skip during recursive search and glob results.
///
/// Shared by `grep_search` and `glob_find` to ensure consistent behavior.
pub const SKIP_DIRS: &[&str] = &[
    ".git",
    ".hg",
    ".svn",
    "node_modules",
    "target",
    "__pycache__",
    ".build",
];

/// Check if a directory name should be skipped.
pub fn should_skip_dir(name: &str) -> bool {
    SKIP_DIRS.contains(&name)
}

/// Extract an optional `usize` from a JSON value by key.
///
/// Returns `None` if the key is absent, not a number, or out of `usize` range.
pub fn extract_optional_usize(input: &serde_json::Value, key: &str) -> Option<usize> {
    input
        .get(key)
        .and_then(serde_json::Value::as_u64)
        .and_then(|v| usize::try_from(v).ok())
}

/// Extract an optional `bool` from a JSON value by key.
pub fn extract_optional_bool(input: &serde_json::Value, key: &str) -> Option<bool> {
    input.get(key).and_then(serde_json::Value::as_bool)
}

/// Extract an optional string reference from a JSON value by key.
pub fn extract_optional_str<'a>(input: &'a serde_json::Value, key: &str) -> Option<&'a str> {
    input.get(key).and_then(serde_json::Value::as_str)
}
