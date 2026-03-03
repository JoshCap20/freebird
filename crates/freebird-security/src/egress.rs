//! Network egress control — host/port allowlisting for outbound requests.
//!
//! Provides `EgressPolicy` which validates URLs against an allowlist of
//! permitted hosts and ports. Used by `SafeUrl::from_tainted()` to ensure
//! the agent can only reach explicitly authorized endpoints.

use std::collections::HashSet;

use crate::error::SecurityError;

/// Controls which hosts and ports the agent is permitted to contact.
///
/// Default policy is deny-all: only hosts explicitly added to the allowlist
/// are reachable. The agent cannot modify its own egress policy.
pub struct EgressPolicy {
    allowed_hosts: HashSet<String>,
    allowed_ports: HashSet<u16>,
}

impl EgressPolicy {
    /// Create a new egress policy with the given allowed hosts and ports.
    ///
    /// All hosts are stored lowercase for case-insensitive matching.
    #[must_use]
    pub fn new(allowed_hosts: HashSet<String>, allowed_ports: HashSet<u16>) -> Self {
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
    ///
    /// # Errors
    ///
    /// Returns `SecurityError::EgressBlocked` if the host is not allowlisted,
    /// the URL has no host, or the port is not permitted.
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

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn test_policy() -> EgressPolicy {
        let mut hosts = HashSet::new();
        hosts.insert("api.anthropic.com".into());
        hosts.insert("api.openai.com".into());

        let mut ports = HashSet::new();
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
}
