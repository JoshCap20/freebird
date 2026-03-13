//! Security error types for the freebird-security crate.

use std::path::PathBuf;

use chrono::{DateTime, Utc};
use freebird_traits::tool::Capability;
use serde::{Deserialize, Serialize};

use crate::budget::BudgetResource;

/// Severity classification for security events.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Severity {
    Low,
    Medium,
    High,
    Critical,
}

/// Unified error type for all security-related failures in `freebird-security`.
#[derive(Debug, thiserror::Error)]
pub enum SecurityError {
    // ── Taint / input validation ─────────────────────────────────
    #[error("input too long: {actual} bytes exceeds {max} byte limit")]
    InputTooLong { max: usize, actual: usize },

    #[error("missing required field `{field}` in {context}")]
    MissingField { field: String, context: String },

    #[error("forbidden character '{character}' in {context}")]
    ForbiddenCharacter { character: char, context: String },

    // ── Prompt injection ─────────────────────────────────────────
    #[error("potential prompt injection detected: pattern `{pattern}`")]
    PotentialInjection { pattern: String, severity: Severity },

    // ── Path safety ──────────────────────────────────────────────
    #[error("path traversal: `{}` escapes sandbox `{}`", attempted.display(), sandbox.display())]
    PathTraversal {
        attempted: PathBuf,
        sandbox: PathBuf,
    },

    #[error("path resolution failed for `{}`: {source}", path.display())]
    PathResolution {
        path: PathBuf,
        source: std::io::Error,
    },

    // ── URL / egress ─────────────────────────────────────────────
    #[error("invalid URL `{url}`: {reason}")]
    InvalidUrl { url: String, reason: String },

    #[error("egress blocked: {reason}")]
    EgressBlocked { reason: String },

    #[error("egress rate limited: {limit_per_minute} requests/minute exceeded")]
    EgressRateLimited { limit_per_minute: u32 },

    #[error("egress request body too large: {actual} bytes exceeds {max} byte limit")]
    EgressBodyTooLarge { actual: usize, max: usize },

    // ── Context poisoning ────────────────────────────────────────
    #[error("context poisoning attempt detected: pattern `{pattern}`")]
    ContextPoisoningAttempt { pattern: String },

    // ── Capability grants ──────────────────────────────────────────
    #[error("capability `{capability:?}` not granted")]
    CapabilityDenied { capability: Capability },

    #[error("capability grant expired at {expired_at}")]
    GrantExpired { expired_at: DateTime<Utc> },

    #[error("sub-grant exceeds parent: capabilities {denied:?} not in parent")]
    SubGrantExceedsParent { denied: Vec<Capability> },

    #[error("sub-grant sandbox `{}` escapes parent sandbox `{}`", child.display(), parent.display())]
    SubGrantSandboxEscape { child: PathBuf, parent: PathBuf },

    // ── Audit logging ─────────────────────────────────────────────
    #[error("audit log corrupted at line {line}: {reason}")]
    AuditCorruption { line: usize, reason: String },

    #[error("failed to write audit log: {reason}")]
    AuditWriteFailed { reason: String },

    // ── Session authentication ──────────────────────────────────
    #[error("session key `{key_id}` has expired")]
    SessionExpired { key_id: String },

    #[error("invalid session key for `{key_id}`")]
    InvalidSessionKey { key_id: String },

    #[error("TTL too large to represent: {seconds}s exceeds maximum")]
    InvalidTtl { seconds: u64 },

    #[error("invalid session credential: {reason}")]
    InvalidCredential { reason: String },

    // ── Database encryption ──────────────────────────────────────
    #[error("no database encryption key found: {message}")]
    NoEncryptionKey { message: String },

    #[error("insecure keyfile permissions on `{}`: mode {actual_mode:o}, required {required_mode:o}", path.display())]
    InsecureKeyfile {
        path: PathBuf,
        actual_mode: u32,
        required_mode: u32,
    },

    #[error("keyfile error: {0}")]
    KeyfileError(String),

    // ── Secret guard ──────────────────────────────────────────────
    #[error("secret guard configuration error: {reason}")]
    SecretGuardConfigError { reason: String },

    #[error("secret access blocked: {reason}")]
    SecretAccessBlocked { reason: String },

    // ── Budget enforcement (ASI08) ───────────────────────────────
    #[error("budget exceeded for `{resource}`: used {used}, limit {limit}")]
    BudgetExceeded {
        resource: BudgetResource,
        used: u64,
        limit: u64,
    },
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn test_security_error_input_too_long_display() {
        let err = SecurityError::InputTooLong {
            max: 100,
            actual: 200,
        };
        assert_eq!(
            err.to_string(),
            "input too long: 200 bytes exceeds 100 byte limit"
        );
    }

    #[test]
    fn test_security_error_missing_field_display() {
        let err = SecurityError::MissingField {
            field: "path".into(),
            context: "tool input".into(),
        };
        assert_eq!(
            err.to_string(),
            "missing required field `path` in tool input"
        );
    }

    #[test]
    fn test_security_error_forbidden_char_display() {
        let err = SecurityError::ForbiddenCharacter {
            character: '|',
            context: "shell argument".into(),
        };
        assert_eq!(err.to_string(), "forbidden character '|' in shell argument");
    }

    #[test]
    fn test_security_error_injection_display() {
        let err = SecurityError::PotentialInjection {
            pattern: "ignore previous instructions".into(),
            severity: Severity::High,
        };
        assert!(err.to_string().contains("ignore previous instructions"));
    }

    #[test]
    fn test_security_error_path_traversal_display() {
        let err = SecurityError::PathTraversal {
            attempted: PathBuf::from("/etc/passwd"),
            sandbox: PathBuf::from("/home/user/sandbox"),
        };
        let msg = err.to_string();
        assert!(msg.contains("/etc/passwd"));
        assert!(msg.contains("/home/user/sandbox"));
    }

    #[test]
    fn test_security_error_egress_display() {
        let err = SecurityError::EgressBlocked {
            reason: "host not in allowlist".into(),
        };
        assert_eq!(err.to_string(), "egress blocked: host not in allowlist");
    }

    #[test]
    fn test_severity_debug_and_clone() {
        let s = Severity::High;
        let s2 = s.clone();
        assert_eq!(s, s2);
        assert_eq!(format!("{s:?}"), "High");
    }

    #[test]
    fn test_security_error_capability_denied_display() {
        let err = SecurityError::CapabilityDenied {
            capability: Capability::ShellExecute,
        };
        assert_eq!(err.to_string(), "capability `ShellExecute` not granted");
    }

    #[test]
    fn test_security_error_grant_expired_display() {
        let expired = Utc::now();
        let err = SecurityError::GrantExpired {
            expired_at: expired,
        };
        let msg = err.to_string();
        assert!(msg.starts_with("capability grant expired at "));
    }

    #[test]
    fn test_security_error_sub_grant_exceeds_parent_display() {
        let err = SecurityError::SubGrantExceedsParent {
            denied: vec![Capability::ShellExecute, Capability::NetworkOutbound],
        };
        let msg = err.to_string();
        assert!(msg.contains("ShellExecute"));
        assert!(msg.contains("NetworkOutbound"));
    }

    #[test]
    fn test_security_error_sub_grant_sandbox_escape_display() {
        let err = SecurityError::SubGrantSandboxEscape {
            child: PathBuf::from("/tmp/x/b"),
            parent: PathBuf::from("/tmp/x/a"),
        };
        let msg = err.to_string();
        assert!(msg.contains("/tmp/x/b"));
        assert!(msg.contains("/tmp/x/a"));
    }

    #[test]
    fn test_security_error_audit_corruption_display() {
        let err = SecurityError::AuditCorruption {
            line: 42,
            reason: "hash chain broken".into(),
        };
        assert_eq!(
            err.to_string(),
            "audit log corrupted at line 42: hash chain broken"
        );
    }

    #[test]
    fn test_security_error_audit_write_failed_display() {
        let err = SecurityError::AuditWriteFailed {
            reason: "disk full".into(),
        };
        assert_eq!(err.to_string(), "failed to write audit log: disk full");
    }

    #[test]
    fn test_severity_serde_roundtrip() {
        let severity = Severity::High;
        let json = serde_json::to_string(&severity).unwrap();
        assert_eq!(json, r#""high""#);
        let deserialized: Severity = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized, severity);
    }

    #[test]
    fn test_security_error_session_expired_display() {
        let err = SecurityError::SessionExpired {
            key_id: "freebird_a1b2c3d4e5f6".into(),
        };
        assert_eq!(
            err.to_string(),
            "session key `freebird_a1b2c3d4e5f6` has expired"
        );
    }

    #[test]
    fn test_security_error_invalid_session_key_display() {
        let err = SecurityError::InvalidSessionKey {
            key_id: "freebird_a1b2c3d4e5f6".into(),
        };
        assert_eq!(
            err.to_string(),
            "invalid session key for `freebird_a1b2c3d4e5f6`"
        );
    }

    #[test]
    fn test_security_error_invalid_ttl_display() {
        let err = SecurityError::InvalidTtl {
            seconds: 18_446_744_073_709_551_615,
        };
        let msg = err.to_string();
        assert!(msg.contains("TTL too large"));
        assert!(msg.contains("18446744073709551615"));
    }

    #[test]
    fn test_security_error_invalid_credential_display() {
        let err = SecurityError::InvalidCredential {
            reason: "key_hash must be 64 hex characters".into(),
        };
        assert_eq!(
            err.to_string(),
            "invalid session credential: key_hash must be 64 hex characters"
        );
    }
}
