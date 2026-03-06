/*
SECURITY INVARIANT
-------------------

The types in this module are the primary compile-time security boundary
for untrusted input. DO NOT weaken these invariants without a formal
security review.

- `Tainted` and `TaintedToolInput` MUST NOT implement `Display`, `Deref`,
    `AsRef<str>`, `Into<String>`, or any other convenience that exposes the
    contained raw string/JSON to downstream crates.
- The inner value accessor is intentionally `pub(crate)` (or private).
    Changing its visibility to `pub` or exposing an API that returns the raw
    value is a security regression.
- Any removal or relaxation of these rules requires: (1) a code review
    explicitly approving the change, and (2) an entry in the security
    changelog documenting the rationale.

ENFORCEMENT: These invariants are enforced by code review, not compile-fail
tests. trybuild-based compile-fail tests were evaluated and deferred: they
are brittle across compiler versions, produce opaque failure messages, and
add a heavy dev-dependency for checks that are trivially caught in review.
The `pub(crate)` boundary is the primary enforcement mechanism — any attempt
to call `inner()` from outside this crate is a hard compiler error.
*/

//! Opaque wrappers for untrusted external input.
//!
//! `Tainted` wraps raw string input from channels. `TaintedToolInput` wraps
//! raw JSON from LLM tool calls. The **only** way to extract the inner value
//! is through a safe type factory in this crate — `pub(crate)` on `inner()`
//! is the entire security boundary.
//!
//! No `Display`, `Deref`, `AsRef<str>`, or `Into<String>` implementations.
//! No escape hatches. Adding a new way to detaint requires modifying this
//! crate (which is security-reviewed).

use std::path::{Path, PathBuf};

use crate::egress::EgressPolicy;
use crate::error::SecurityError;
use crate::safe_types::{SafeFileContent, SafeFilePath, SafeShellArg, SafeUrl};

/// Opaque wrapper for unvalidated external string input.
///
/// The ONLY way to extract the value is through a safe type factory
/// in this crate. `pub(crate)` on `inner()` enforces this at compile time.
pub struct Tainted(String);

impl Tainted {
    /// Wrap raw external input. This is the entry point for ALL untrusted
    /// string data entering the system.
    #[must_use]
    pub fn new(raw: impl Into<String>) -> Self {
        Self(raw.into())
    }

    /// Access the raw inner value. Only callable within `freebird-security`.
    ///
    /// Every path from untrusted input to usable value flows through
    /// a safe type factory that calls this method.
    pub(crate) fn inner(&self) -> &str {
        &self.0
    }
}

// No Display. No Deref. No AsRef<str>. No Into<String>.
// Debug does NOT reveal contents.
impl std::fmt::Debug for Tainted {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Tainted([{} bytes])", self.0.len())
    }
}

/// Opaque wrapper for LLM-generated tool input (JSON).
///
/// Tools extract validated fields through typed methods —
/// they never touch raw JSON directly.
pub struct TaintedToolInput(serde_json::Value);

impl TaintedToolInput {
    /// Wrap raw tool input from the LLM.
    #[must_use]
    pub const fn new(raw: serde_json::Value) -> Self {
        Self(raw)
    }

    /// Access the raw inner JSON. Only callable within `freebird-security`.
    #[allow(dead_code)] // Will be used when tool input validation needs raw JSON access
    pub(crate) const fn inner(&self) -> &serde_json::Value {
        &self.0
    }

    /// Extract a string field as a `Tainted` value for further validation.
    ///
    /// # Errors
    ///
    /// Returns `SecurityError::MissingField` if the key doesn't exist or
    /// the value is not a string.
    pub fn extract_string(&self, key: &str) -> Result<Tainted, SecurityError> {
        let val = self
            .0
            .get(key)
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| SecurityError::MissingField {
                field: key.into(),
                context: "tool input".into(),
            })?;
        Ok(Tainted::new(val))
    }

    /// Extract a string field and validate it as a file path in one step.
    ///
    /// # Errors
    ///
    /// Returns `SecurityError::MissingField` if the key doesn't exist or
    /// isn't a string. Returns path-related errors if validation fails.
    pub fn extract_path(&self, key: &str, sandbox: &Path) -> Result<SafeFilePath, SecurityError> {
        let tainted = self.extract_string(key)?;
        SafeFilePath::from_tainted(&tainted, sandbox)
    }

    /// Extract a path that may be absolute (within an allowed directory)
    /// or relative (within the sandbox).
    ///
    /// # Errors
    ///
    /// Returns `SecurityError::MissingField` if the key doesn't exist or
    /// isn't a string. Returns path-related errors if validation fails.
    pub fn extract_path_multi_root(
        &self,
        key: &str,
        sandbox: &Path,
        allowed_dirs: &[PathBuf],
    ) -> Result<SafeFilePath, SecurityError> {
        let tainted = self.extract_string(key)?;
        SafeFilePath::from_tainted_multi_root(&tainted, sandbox, allowed_dirs)
    }

    /// Extract a string field and validate it as a path for file creation.
    ///
    /// # Errors
    ///
    /// Returns `SecurityError::MissingField` if the key doesn't exist or
    /// isn't a string. Returns path-related errors if validation fails.
    pub fn extract_path_for_creation(
        &self,
        key: &str,
        sandbox: &Path,
    ) -> Result<SafeFilePath, SecurityError> {
        let tainted = self.extract_string(key)?;
        SafeFilePath::from_tainted_for_creation(&tainted, sandbox)
    }

    /// Extract a path for file creation that may be absolute (within an
    /// allowed directory) or relative (within the sandbox).
    ///
    /// # Errors
    ///
    /// Returns `SecurityError::MissingField` if the key doesn't exist or
    /// isn't a string. Returns path-related errors if validation fails.
    pub fn extract_path_for_creation_multi_root(
        &self,
        key: &str,
        sandbox: &Path,
        allowed_dirs: &[PathBuf],
    ) -> Result<SafeFilePath, SecurityError> {
        let tainted = self.extract_string(key)?;
        SafeFilePath::from_tainted_for_creation_multi_root(&tainted, sandbox, allowed_dirs)
    }

    /// Extract a string field and validate it as a shell argument.
    ///
    /// # Errors
    ///
    /// Returns `SecurityError::MissingField` if the key doesn't exist or
    /// isn't a string. Returns `ForbiddenCharacter` if validation fails.
    pub fn extract_shell_arg(&self, key: &str) -> Result<SafeShellArg, SecurityError> {
        let tainted = self.extract_string(key)?;
        SafeShellArg::from_tainted(&tainted)
    }

    /// Extract a string field as file content for writing.
    ///
    /// No content validation — file content is arbitrary text. The safe type
    /// exists solely to bridge the `pub(crate)` taint boundary.
    ///
    /// # Errors
    ///
    /// Returns `SecurityError::MissingField` if the key doesn't exist or
    /// the value is not a string.
    pub fn extract_file_content(&self, key: &str) -> Result<SafeFileContent, SecurityError> {
        let tainted = self.extract_string(key)?;
        Ok(SafeFileContent::new(tainted.inner().to_string()))
    }

    /// Extract a string field and validate it as a URL.
    ///
    /// # Errors
    ///
    /// Returns `SecurityError::MissingField` if the key doesn't exist or
    /// isn't a string. Returns URL/egress errors if validation fails.
    pub fn extract_url(
        &self,
        key: &str,
        egress_policy: &EgressPolicy,
    ) -> Result<SafeUrl, SecurityError> {
        let tainted = self.extract_string(key)?;
        SafeUrl::from_tainted(&tainted, egress_policy)
    }
}

impl std::fmt::Debug for TaintedToolInput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "TaintedToolInput([JSON object])")
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;

    #[test]
    fn test_tainted_debug_hides_contents() {
        let t = Tainted::new("super secret password");
        let debug = format!("{t:?}");
        assert!(
            !debug.contains("super secret password"),
            "Debug must not reveal contents"
        );
    }

    #[test]
    fn test_tainted_debug_shows_length() {
        let t = Tainted::new("hello"); // 5 bytes
        let debug = format!("{t:?}");
        assert!(debug.contains("5 bytes"), "Debug should show byte length");
    }

    #[test]
    fn test_tainted_inner_accessible_in_crate() {
        let t = Tainted::new("test value");
        // This compiles because we're inside freebird-security
        assert_eq!(t.inner(), "test value");
    }

    #[test]
    fn test_tainted_tool_input_extract_string() {
        let input = TaintedToolInput::new(serde_json::json!({
            "path": "/home/user/file.txt",
            "mode": "read"
        }));
        let tainted = input.extract_string("path").unwrap();
        assert_eq!(tainted.inner(), "/home/user/file.txt");
    }

    #[test]
    fn test_tainted_tool_input_extract_missing_field() {
        let input = TaintedToolInput::new(serde_json::json!({"path": "/tmp"}));
        let result = input.extract_string("nonexistent");
        assert!(result.is_err());
        match result.unwrap_err() {
            SecurityError::MissingField { field, context } => {
                assert_eq!(field, "nonexistent");
                assert_eq!(context, "tool input");
            }
            other => panic!("expected MissingField, got: {other:?}"),
        }
    }

    #[test]
    fn test_tainted_tool_input_extract_non_string_field() {
        let input = TaintedToolInput::new(serde_json::json!({
            "count": 42,
            "items": [1, 2, 3]
        }));
        // Integer field
        let result = input.extract_string("count");
        assert!(result.is_err());
        match result.unwrap_err() {
            SecurityError::MissingField { field, .. } => {
                assert_eq!(field, "count");
            }
            other => panic!("expected MissingField, got: {other:?}"),
        }

        // Array field
        let result = input.extract_string("items");
        assert!(result.is_err());
    }

    #[test]
    fn test_tainted_tool_input_debug_hides_contents() {
        let input = TaintedToolInput::new(serde_json::json!({"secret": "password123"}));
        let debug = format!("{input:?}");
        assert!(
            !debug.contains("password123"),
            "Debug must not reveal JSON contents"
        );
        assert!(debug.contains("TaintedToolInput"));
    }

    #[test]
    fn test_extract_path_for_creation_valid() {
        let tmp = tempfile::tempdir().unwrap();
        let input = TaintedToolInput::new(serde_json::json!({
            "path": "newfile.txt"
        }));
        let safe = input.extract_path_for_creation("path", tmp.path()).unwrap();
        let expected = tmp.path().canonicalize().unwrap().join("newfile.txt");
        assert_eq!(safe.as_path(), expected);
    }

    #[test]
    fn test_extract_path_for_creation_missing_field() {
        let tmp = tempfile::tempdir().unwrap();
        let input = TaintedToolInput::new(serde_json::json!({"other": "value"}));
        let result = input.extract_path_for_creation("path", tmp.path());
        assert!(result.is_err());
        match result.unwrap_err() {
            SecurityError::MissingField { field, .. } => {
                assert_eq!(field, "path");
            }
            other => panic!("expected MissingField, got: {other:?}"),
        }
    }

    #[test]
    fn test_extract_path_for_creation_non_string() {
        let tmp = tempfile::tempdir().unwrap();
        let input = TaintedToolInput::new(serde_json::json!({"path": 42}));
        let result = input.extract_path_for_creation("path", tmp.path());
        assert!(result.is_err());
        match result.unwrap_err() {
            SecurityError::MissingField { field, .. } => {
                assert_eq!(field, "path");
            }
            other => panic!("expected MissingField, got: {other:?}"),
        }
    }
}
