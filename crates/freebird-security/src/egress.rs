//! Network egress control — host/port allowlisting for outbound requests.
//!
//! Provides `EgressPolicy` which validates URLs against an allowlist of
//! permitted hosts and ports. Used by `SafeUrl::from_tainted()` to ensure
//! the agent can only reach explicitly authorized endpoints.

use std::collections::BTreeSet;

use crate::error::SecurityError;

/// Controls which hosts and ports the agent is permitted to contact.
///
/// Default policy is deny-all: only hosts explicitly added to the allowlist
/// are reachable. The agent cannot modify its own egress policy.
///
/// Uses `BTreeSet` (not `HashSet`) for deterministic iteration order,
/// which ensures stable debug output and predictable test assertions.
///
/// # Future: DNS rebinding prevention
///
/// This policy validates hostnames but does not verify resolved IP addresses.
/// A DNS rebinding attack could cause an allowlisted hostname to resolve to
/// a private/loopback IP (`10.x`, `172.16-31.x`, `192.168.x`, `127.x`, `::1`),
/// enabling SSRF to internal services. IP validation should be added at the
/// HTTP client layer before outbound connections are established.
pub struct EgressPolicy {
    allowed_hosts: BTreeSet<String>,
    allowed_ports: BTreeSet<u16>,
}

impl EgressPolicy {
    /// Create a new egress policy with the given allowed hosts and ports.
    ///
    /// All hosts are stored lowercase for case-insensitive matching.
    #[must_use]
    pub fn new(allowed_hosts: BTreeSet<String>, allowed_ports: BTreeSet<u16>) -> Self {
        let normalized_hosts = allowed_hosts
            .into_iter()
            .map(|h| h.to_lowercase())
            .collect();
        Self {
            allowed_hosts: normalized_hosts,
            allowed_ports,
        }
    }

    /// Validate a parsed URL against this policy.
    ///
    /// Checks that the host is in the allowlist and the port is permitted.
    /// When no explicit port is present in the URL, defaults to 443 (HTTPS).
    ///
    /// **Assumption**: This method is designed to be called after scheme
    /// validation (e.g., by `SafeUrl::from_tainted()` which enforces HTTPS).
    /// The port default of 443 is incorrect for non-HTTPS schemes.
    ///
    /// # Errors
    ///
    /// Returns `SecurityError::EgressBlocked` if the host is not allowlisted,
    /// the URL has no host, or the port is not permitted.
    #[must_use = "egress check result must not be silently discarded"]
    pub fn check_url(&self, url: &url::Url) -> Result<(), SecurityError> {
        let host = url.host_str().ok_or_else(|| SecurityError::EgressBlocked {
            reason: "URL has no host".into(),
        })?;

        let host_lower = host.to_lowercase();
        if !self.allowed_hosts.contains(&host_lower) {
            return Err(SecurityError::EgressBlocked {
                reason: format!("host `{host}` not in egress allowlist"),
            });
        }

        let port = url.port().unwrap_or(443);
        if !self.allowed_ports.contains(&port) {
            return Err(SecurityError::EgressBlocked {
                reason: format!("port {port} not allowed"),
            });
        }

        Ok(())
    }
}

/// Default egress policy per CLAUDE.md section 12:
/// - `allowed_hosts`: `["api.anthropic.com", "api.openai.com"]`
/// - `allowed_ports`: `[443]` (HTTPS only)
impl Default for EgressPolicy {
    fn default() -> Self {
        let allowed_hosts: BTreeSet<String> =
            ["api.anthropic.com".to_owned(), "api.openai.com".to_owned()]
                .into_iter()
                .collect();

        let allowed_ports: BTreeSet<u16> = std::iter::once(443).collect();

        Self {
            allowed_hosts,
            allowed_ports,
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn test_policy() -> EgressPolicy {
        let mut hosts = BTreeSet::new();
        hosts.insert("api.anthropic.com".into());
        hosts.insert("api.openai.com".into());

        let mut ports = BTreeSet::new();
        ports.insert(443);

        EgressPolicy::new(hosts, ports)
    }

    #[test]
    fn test_egress_allows_allowlisted_host() {
        let policy = test_policy();
        let url: url::Url = "https://api.anthropic.com/v1/messages".parse().unwrap();
        assert!(policy.check_url(&url).is_ok());
    }

    #[test]
    fn test_egress_blocks_unknown_host() {
        let policy = test_policy();
        let url: url::Url = "https://evil.com/exfiltrate".parse().unwrap();
        let result = policy.check_url(&url);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("not in egress allowlist")
        );
    }

    #[test]
    fn test_egress_blocks_non_allowed_port() {
        let policy = test_policy();
        let url: url::Url = "https://api.anthropic.com:8080/v1/messages"
            .parse()
            .unwrap();
        let result = policy.check_url(&url);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("port 8080 not allowed")
        );
    }

    #[test]
    fn test_egress_case_insensitive_host() {
        let policy = test_policy();
        let url: url::Url = "https://API.ANTHROPIC.COM/v1/messages".parse().unwrap();
        assert!(policy.check_url(&url).is_ok());
    }

    #[test]
    fn test_egress_default_policy() {
        let policy = EgressPolicy::default();
        let anthropic: url::Url = "https://api.anthropic.com/v1/messages".parse().unwrap();
        let openai: url::Url = "https://api.openai.com/v1/chat".parse().unwrap();
        let evil: url::Url = "https://evil.com/exfiltrate".parse().unwrap();

        assert!(policy.check_url(&anthropic).is_ok());
        assert!(policy.check_url(&openai).is_ok());
        assert!(policy.check_url(&evil).is_err());
    }

    #[test]
    fn test_egress_default_blocks_non_443_port() {
        let policy = EgressPolicy::default();
        let url: url::Url = "https://api.anthropic.com:8080/v1/messages"
            .parse()
            .unwrap();
        assert!(policy.check_url(&url).is_err());
    }
}
