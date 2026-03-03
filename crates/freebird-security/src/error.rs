//! Security error types for the freebird-security crate.

use std::path::PathBuf;

/// Severity classification for security events.
#[derive(Debug, Clone, PartialEq, Eq)]
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

    // ── Context poisoning ────────────────────────────────────────
    #[error("context poisoning attempt detected: pattern `{pattern}`")]
    ContextPoisoningAttempt { pattern: String },
}

#[cfg(test)]
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
}
