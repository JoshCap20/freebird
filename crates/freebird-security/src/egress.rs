//! Network egress control — host/port allowlisting for outbound requests.
//!
//! Provides `EgressPolicy` which validates URLs against an allowlist of
//! permitted hosts and ports. Used by `SafeUrl::from_tainted()` to ensure
//! the agent can only reach explicitly authorized endpoints.
//!
//! Also provides DNS rebinding prevention via [`is_private_ip`] and
//! [`EgressPolicy::check_resolved_ip`], which reject resolved IP addresses
//! in private/loopback/link-local ranges to prevent SSRF to internal services.

use std::collections::BTreeSet;
use std::net::IpAddr;

use crate::error::SecurityError;

/// Check whether an IP address is private, loopback, link-local, or otherwise
/// non-routable.
///
/// Covers:
/// - RFC 1918 private ranges: `10.0.0.0/8`, `172.16.0.0/12`, `192.168.0.0/16`
/// - Loopback: `127.0.0.0/8` (IPv4), `::1` (IPv6)
/// - Link-local: `169.254.0.0/16` (IPv4), `fe80::/10` (IPv6)
/// - IPv4-mapped IPv6: `::ffff:0:0/96` — delegates to the embedded IPv4 check
/// - Documentation/example ranges: `192.0.2.0/24`, `198.51.100.0/24`, `203.0.113.0/24`
/// - Broadcast: `255.255.255.255`
/// - Unspecified: `0.0.0.0`, `::`
#[must_use]
pub fn is_private_ip(ip: &IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => {
            let octets = v4.octets();
            // Loopback: 127.0.0.0/8
            octets[0] == 127
            // RFC 1918: 10.0.0.0/8
            || octets[0] == 10
            // RFC 1918: 172.16.0.0/12
            || (octets[0] == 172 && (16..=31).contains(&octets[1]))
            // RFC 1918: 192.168.0.0/16
            || (octets[0] == 192 && octets[1] == 168)
            // Link-local: 169.254.0.0/16
            || (octets[0] == 169 && octets[1] == 254)
            // Documentation: 192.0.2.0/24, 198.51.100.0/24, 203.0.113.0/24
            || (octets[0] == 192 && octets[1] == 0 && octets[2] == 2)
            || (octets[0] == 198 && octets[1] == 51 && octets[2] == 100)
            || (octets[0] == 203 && octets[1] == 0 && octets[2] == 113)
            // Broadcast
            || *v4 == std::net::Ipv4Addr::BROADCAST
            // Unspecified
            || v4.is_unspecified()
        }
        IpAddr::V6(v6) => {
            // Loopback: ::1
            v6.is_loopback()
            // Unspecified: ::
            || v6.is_unspecified()
            // Link-local: fe80::/10
            || (v6.segments()[0] & 0xffc0) == 0xfe80
            // IPv4-mapped IPv6: ::ffff:x.x.x.x — check the embedded IPv4
            || v6.to_ipv4_mapped().is_some_and(|v4| is_private_ip(&IpAddr::V4(v4)))
        }
    }
}

/// Controls which hosts and ports the agent is permitted to contact.
///
/// Default policy is deny-all: only hosts explicitly added to the allowlist
/// are reachable. The agent cannot modify its own egress policy.
///
/// Uses `BTreeSet` (not `HashSet`) for deterministic iteration order,
/// which ensures stable debug output and predictable test assertions.
///
/// Also provides DNS rebinding prevention via [`check_resolved_ip`](EgressPolicy::check_resolved_ip),
/// which rejects resolved IP addresses in private/loopback/link-local ranges.
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

    /// Return the set of allowed hosts.
    #[must_use]
    pub const fn allowed_hosts(&self) -> &BTreeSet<String> {
        &self.allowed_hosts
    }

    /// Return the set of allowed ports.
    #[must_use]
    pub const fn allowed_ports(&self) -> &BTreeSet<u16> {
        &self.allowed_ports
    }

    /// Validate that a resolved IP address is not private/loopback/link-local.
    ///
    /// Call this after DNS resolution and before establishing a connection
    /// to prevent DNS rebinding attacks where an allowlisted hostname resolves
    /// to a private IP (SSRF to internal services).
    ///
    /// # Errors
    ///
    /// Returns `SecurityError::EgressBlocked` if the IP is private, loopback,
    /// link-local, or otherwise non-routable.
    #[must_use = "egress check result must not be silently discarded"]
    pub fn check_resolved_ip(&self, ip: &IpAddr) -> Result<(), SecurityError> {
        if is_private_ip(ip) {
            return Err(SecurityError::EgressBlocked {
                reason: format!(
                    "resolved IP `{ip}` is in a private/loopback/link-local range \
                     (DNS rebinding prevention)"
                ),
            });
        }
        Ok(())
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

    // ── DNS rebinding prevention tests ──────────────────────────────

    #[test]
    fn test_private_ip_rfc1918_10() {
        let ip: IpAddr = "10.0.0.1".parse().unwrap();
        assert!(is_private_ip(&ip));
    }

    #[test]
    fn test_private_ip_rfc1918_172() {
        let ip: IpAddr = "172.16.0.1".parse().unwrap();
        assert!(is_private_ip(&ip));
    }

    #[test]
    fn test_private_ip_rfc1918_192() {
        let ip: IpAddr = "192.168.1.1".parse().unwrap();
        assert!(is_private_ip(&ip));
    }

    #[test]
    fn test_private_ip_loopback_v4() {
        let ip: IpAddr = "127.0.0.1".parse().unwrap();
        assert!(is_private_ip(&ip));
    }

    #[test]
    fn test_private_ip_loopback_v6() {
        let ip: IpAddr = "::1".parse().unwrap();
        assert!(is_private_ip(&ip));
    }

    #[test]
    fn test_private_ip_link_local_v4() {
        let ip: IpAddr = "169.254.1.1".parse().unwrap();
        assert!(is_private_ip(&ip));
    }

    #[test]
    fn test_public_ip_8888() {
        let ip: IpAddr = "8.8.8.8".parse().unwrap();
        assert!(!is_private_ip(&ip));
    }

    #[test]
    fn test_public_ip_1111() {
        let ip: IpAddr = "1.1.1.1".parse().unwrap();
        assert!(!is_private_ip(&ip));
    }

    #[test]
    fn test_check_resolved_ip_rejects_private() {
        let policy = test_policy();
        let ip: IpAddr = "10.0.0.1".parse().unwrap();
        let result = policy.check_resolved_ip(&ip);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("DNS rebinding prevention")
        );
    }

    #[test]
    fn test_check_resolved_ip_allows_public() {
        let policy = test_policy();
        let ip: IpAddr = "8.8.8.8".parse().unwrap();
        assert!(policy.check_resolved_ip(&ip).is_ok());
    }

    #[test]
    fn test_check_resolved_ip_rejects_loopback_v6() {
        let policy = test_policy();
        let ip: IpAddr = "::1".parse().unwrap();
        assert!(policy.check_resolved_ip(&ip).is_err());
    }

    #[test]
    fn test_check_resolved_ip_rejects_link_local() {
        let policy = test_policy();
        let ip: IpAddr = "169.254.1.1".parse().unwrap();
        assert!(policy.check_resolved_ip(&ip).is_err());
    }
}
