//! Safe URL type with egress policy validation.

use crate::egress::EgressPolicy;
use crate::error::SecurityError;
use crate::taint::Tainted;

// ── SafeUrl ──────────────────────────────────────────────────────

/// A URL that has been validated against the egress policy.
///
/// Produced by: tool input extraction.
/// Consumed by: network request tool.
#[derive(Debug)]
pub struct SafeUrl(url::Url);

impl SafeUrl {
    /// Validate an untrusted URL against the egress policy.
    ///
    /// - Parses and validates the URL
    /// - Enforces HTTPS-only (HTTP rejected)
    /// - Checks host against allowlist in the egress policy
    /// - Checks port against allowed ports
    ///
    /// # Errors
    ///
    /// Returns `SecurityError::InvalidUrl` if the URL cannot be parsed.
    /// Returns `SecurityError::EgressBlocked` if the URL violates the egress policy.
    pub fn from_tainted(t: &Tainted, policy: &EgressPolicy) -> Result<Self, SecurityError> {
        let raw = t.inner();

        let parsed = url::Url::parse(raw).map_err(|e| SecurityError::InvalidUrl {
            url: raw.to_string(),
            reason: e.to_string(),
        })?;

        // HTTPS only
        if parsed.scheme() != "https" {
            return Err(SecurityError::EgressBlocked {
                reason: format!("only HTTPS is allowed, got: {}", parsed.scheme()),
            });
        }

        // Delegate host + port validation to the canonical EgressPolicy check.
        policy.check_url(&parsed)?;

        Ok(Self(parsed))
    }

    /// Access the validated URL.
    #[must_use]
    pub const fn as_url(&self) -> &url::Url {
        &self.0
    }

    /// Access the URL as a string.
    #[must_use]
    pub fn as_str(&self) -> &str {
        self.0.as_str()
    }
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic
)]
mod tests {
    use super::*;
    use crate::taint::Tainted;

    fn test_policy() -> EgressPolicy {
        EgressPolicy::new(
            ["api.anthropic.com".into(), "api.openai.com".into()]
                .into_iter()
                .collect(),
            std::iter::once(443).collect(),
            1_048_576,
        )
    }

    #[test]
    fn test_url_accepts_valid_https() {
        let t = Tainted::new("https://api.anthropic.com/v1/messages");
        let url = SafeUrl::from_tainted(&t, &test_policy()).unwrap();
        assert_eq!(url.as_url().host_str(), Some("api.anthropic.com"));
    }

    #[test]
    fn test_url_rejects_http() {
        let t = Tainted::new("http://api.anthropic.com/v1/messages");
        let err = SafeUrl::from_tainted(&t, &test_policy()).unwrap_err();
        assert!(matches!(err, SecurityError::EgressBlocked { .. }));
    }

    #[test]
    fn test_url_rejects_unlisted_host() {
        let t = Tainted::new("https://evil.com/exfiltrate");
        let err = SafeUrl::from_tainted(&t, &test_policy()).unwrap_err();
        assert!(matches!(err, SecurityError::EgressBlocked { .. }));
    }

    #[test]
    fn test_url_rejects_non_443_port() {
        let t = Tainted::new("https://api.anthropic.com:8080/v1/messages");
        let err = SafeUrl::from_tainted(&t, &test_policy()).unwrap_err();
        assert!(matches!(err, SecurityError::EgressBlocked { .. }));
    }

    #[test]
    fn test_url_rejects_invalid_url() {
        let t = Tainted::new("not a url at all");
        let err = SafeUrl::from_tainted(&t, &test_policy()).unwrap_err();
        assert!(matches!(err, SecurityError::InvalidUrl { .. }));
    }
}
