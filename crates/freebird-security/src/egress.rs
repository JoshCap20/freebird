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
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

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
    max_request_body_bytes: usize,
}

impl EgressPolicy {
    /// Create a new egress policy with the given allowed hosts, ports, and body size limit.
    ///
    /// All hosts are stored lowercase for case-insensitive matching.
    #[must_use]
    pub fn new(
        allowed_hosts: BTreeSet<String>,
        allowed_ports: BTreeSet<u16>,
        max_request_body_bytes: usize,
    ) -> Self {
        let normalized_hosts = allowed_hosts
            .into_iter()
            .map(|h| h.to_lowercase())
            .collect();
        Self {
            allowed_hosts: normalized_hosts,
            allowed_ports,
            max_request_body_bytes,
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

    /// Return the maximum allowed request body size in bytes.
    #[must_use]
    pub const fn max_request_body_bytes(&self) -> usize {
        self.max_request_body_bytes
    }

    /// Validate that a request body size does not exceed the configured limit.
    ///
    /// # Errors
    ///
    /// Returns `SecurityError::EgressBodyTooLarge` if `len` exceeds
    /// `max_request_body_bytes`.
    #[must_use = "egress check result must not be silently discarded"]
    pub const fn check_request_body_size(&self, len: usize) -> Result<(), SecurityError> {
        if len > self.max_request_body_bytes {
            return Err(SecurityError::EgressBodyTooLarge {
                actual: len,
                max: self.max_request_body_bytes,
            });
        }
        Ok(())
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
/// - `max_request_body_bytes`: `1_048_576` (1 MiB)
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
            max_request_body_bytes: 1_048_576,
        }
    }
}

/// Thread-safe sliding-window rate limiter for egress requests.
///
/// Uses a circular buffer of timestamps (Unix epoch milliseconds) with atomic
/// operations. Each call to [`check_and_record`](EgressRateLimiter::check_and_record)
/// inspects the oldest entry in the window; if it falls within the last 60 seconds,
/// the rate limit has been exceeded.
pub struct EgressRateLimiter {
    /// Circular buffer of request timestamps (Unix epoch millis).
    timestamps: Vec<AtomicU64>,
    /// Maximum requests per 60-second window.
    limit: u32,
    /// Atomic cursor for the next slot in the circular buffer.
    cursor: AtomicU64,
}

impl EgressRateLimiter {
    /// Create a new rate limiter allowing `limit` requests per 60-second window.
    ///
    /// A `limit` of `0` will reject every request. A `limit` of `u32::MAX` is
    /// effectively unlimited.
    #[must_use]
    pub fn new(limit: u32) -> Self {
        // u32→usize is always safe: usize ≥ 32 bits on all supported platforms.
        #[allow(clippy::cast_possible_truncation)]
        let size = limit as usize;
        let mut timestamps = Vec::with_capacity(size);
        for _ in 0..size {
            timestamps.push(AtomicU64::new(0));
        }
        Self {
            timestamps,
            limit,
            cursor: AtomicU64::new(0),
        }
    }

    /// Check whether the rate limit allows another request, and if so, record it.
    ///
    /// Uses a 60-second sliding window. The oldest timestamp in the circular buffer
    /// is compared against the current time; if it is less than 60 seconds old, the
    /// window is full and the request is rejected.
    ///
    /// **Concurrency note**: Under high contention, a small number of extra
    /// requests (up to the number of concurrent callers) may pass the check
    /// before the window is fully recorded. This is an inherent trade-off of
    /// the lock-free atomic design — acceptable for egress rate limiting where
    /// occasional overshoot is tolerable.
    ///
    /// # Errors
    ///
    /// Returns `SecurityError::EgressRateLimited` if the limit has been exceeded.
    #[must_use = "rate limit check result must not be silently discarded"]
    pub fn check_and_record(&self) -> Result<(), SecurityError> {
        if self.limit == 0 {
            return Err(SecurityError::EgressRateLimited {
                limit_per_minute: self.limit,
            });
        }

        // Truncation from u128→u64 is safe: millis since epoch fit in u64 until year 584M.
        #[allow(clippy::cast_possible_truncation)]
        let now_millis = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        let slot = self.cursor.fetch_add(1, Ordering::Relaxed) % u64::from(self.limit);

        // `slot` is always < self.limit (which fits in u32), so truncation to usize
        // is safe on all platforms (usize ≥ 32 bits).
        #[allow(clippy::cast_possible_truncation)]
        let slot_idx = slot as usize;
        let Some(entry) = self.timestamps.get(slot_idx) else {
            return Err(SecurityError::EgressRateLimited {
                limit_per_minute: self.limit,
            });
        };

        let oldest = entry.load(Ordering::Relaxed);
        let window_millis: u64 = 60_000;

        if oldest != 0 && now_millis.saturating_sub(oldest) < window_millis {
            return Err(SecurityError::EgressRateLimited {
                limit_per_minute: self.limit,
            });
        }

        entry.store(now_millis, Ordering::Relaxed);
        Ok(())
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

        EgressPolicy::new(hosts, ports, 1_048_576)
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

    // ── Body size check tests ──────────────────────────────────────

    #[test]
    fn test_body_size_within_limit_passes() {
        let policy = test_policy();
        assert!(policy.check_request_body_size(512).is_ok());
        assert!(policy.check_request_body_size(1_048_576).is_ok());
    }

    #[test]
    fn test_body_size_exceeds_limit_returns_error() {
        let policy = test_policy();
        let result = policy.check_request_body_size(1_048_577);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("too large"));
        assert!(err.contains("1048577"));
    }

    #[test]
    fn test_body_size_zero_byte_limit_rejects_all() {
        let hosts = BTreeSet::new();
        let ports = BTreeSet::new();
        let policy = EgressPolicy::new(hosts, ports, 0);
        assert!(policy.check_request_body_size(0).is_ok());
        assert!(policy.check_request_body_size(1).is_err());
    }

    // ── Rate limiter tests ─────────────────────────────────────────

    #[test]
    fn test_rate_limiter_within_limit_passes() {
        let limiter = EgressRateLimiter::new(5);
        for _ in 0..5 {
            assert!(limiter.check_and_record().is_ok());
        }
    }

    #[test]
    fn test_rate_limiter_exceeded_returns_error() {
        let limiter = EgressRateLimiter::new(3);
        for _ in 0..3 {
            assert!(limiter.check_and_record().is_ok());
        }
        let result = limiter.check_and_record();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("rate limited"));
    }

    #[test]
    fn test_rate_limiter_zero_limit_rejects_all() {
        let limiter = EgressRateLimiter::new(0);
        let result = limiter.check_and_record();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("rate limited"));
    }
}
