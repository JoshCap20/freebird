//! Adversarial tests for egress policy bypass attempts.
//!
//! The egress module has only 6 inline tests. This suite adds 12+ tests
//! targeting URL parsing edge cases, subdomain matching, scheme validation,
//! and port boundary conditions.

#![allow(
    clippy::all,
    clippy::pedantic,
    clippy::nursery,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

mod helpers;

use freebird_security::safe_types::SafeUrl;
use freebird_security::taint::Tainted;
use helpers::test_egress_policy;

// ---------------------------------------------------------------------------
// Host validation bypass attempts
// ---------------------------------------------------------------------------

#[test]
fn ip_address_url_blocked() {
    let policy = test_egress_policy();
    let t = Tainted::new("https://93.184.216.34/exfiltrate");
    let result = SafeUrl::from_tainted(&t, &policy);
    assert!(result.is_err(), "IP address URLs should be blocked");
}

#[test]
fn subdomain_of_allowed_host_blocked() {
    let policy = test_egress_policy();
    let t = Tainted::new("https://evil.api.anthropic.com/exfiltrate");
    let result = SafeUrl::from_tainted(&t, &policy);
    assert!(
        result.is_err(),
        "subdomain of allowed host should be blocked (exact match, not suffix)"
    );
}

#[test]
fn userinfo_in_url_passes_when_host_matches() {
    let policy = test_egress_policy();
    // url::Url separates userinfo from host — host still matches allowlist
    let t = Tainted::new("https://admin:password@api.anthropic.com/v1/messages");
    let result = SafeUrl::from_tainted(&t, &policy);
    assert!(
        result.is_ok(),
        "userinfo is separate from host; host matches allowlist"
    );
}

#[test]
fn percent_encoded_hostname_decoded_to_match() {
    let policy = test_egress_policy();
    // url::Url decodes %2E to '.' during parsing, so host = "api.anthropic.com"
    let t = Tainted::new("https://api%2Eanthropic%2Ecom/v1/messages");
    let result = SafeUrl::from_tainted(&t, &policy);
    assert!(
        result.is_ok(),
        "percent-encoded dots decoded by url crate; host matches allowlist"
    );
}

#[test]
fn idn_punycode_domain_blocked() {
    let policy = test_egress_policy();
    let t = Tainted::new("https://xn--n3h.com/");
    let result = SafeUrl::from_tainted(&t, &policy);
    assert!(
        result.is_err(),
        "punycode domain not in allowlist should be blocked"
    );
}

// ---------------------------------------------------------------------------
// Scheme validation
// ---------------------------------------------------------------------------

#[test]
fn http_scheme_blocked() {
    let policy = test_egress_policy();
    let t = Tainted::new("http://api.anthropic.com/v1/messages");
    let result = SafeUrl::from_tainted(&t, &policy);
    assert!(result.is_err(), "HTTP (not HTTPS) should be blocked");
}

#[test]
fn ftp_scheme_blocked() {
    let policy = test_egress_policy();
    let t = Tainted::new("ftp://api.anthropic.com/file");
    let result = SafeUrl::from_tainted(&t, &policy);
    assert!(result.is_err(), "FTP scheme should be blocked");
}

#[test]
fn data_uri_scheme_blocked() {
    let policy = test_egress_policy();
    let t = Tainted::new("data:text/html,<script>alert(1)</script>");
    let result = SafeUrl::from_tainted(&t, &policy);
    assert!(result.is_err(), "data: URI should be blocked");
}

// ---------------------------------------------------------------------------
// Port validation
// ---------------------------------------------------------------------------

#[test]
fn port_zero_blocked() {
    let policy = test_egress_policy();
    let t = Tainted::new("https://api.anthropic.com:0/v1/messages");
    let result = SafeUrl::from_tainted(&t, &policy);
    assert!(result.is_err(), "port 0 should be blocked");
}

#[test]
fn port_65535_blocked() {
    let policy = test_egress_policy();
    let t = Tainted::new("https://api.anthropic.com:65535/v1/messages");
    let result = SafeUrl::from_tainted(&t, &policy);
    assert!(
        result.is_err(),
        "port 65535 should be blocked (only 443 allowed)"
    );
}

#[test]
fn default_port_443_passes() {
    let policy = test_egress_policy();
    let t = Tainted::new("https://api.anthropic.com/v1/messages");
    let result = SafeUrl::from_tainted(&t, &policy);
    assert!(result.is_ok(), "default HTTPS port (443) should pass");
}

#[test]
fn explicit_port_443_passes() {
    let policy = test_egress_policy();
    let t = Tainted::new("https://api.anthropic.com:443/v1/messages");
    let result = SafeUrl::from_tainted(&t, &policy);
    assert!(result.is_ok(), "explicit port 443 should pass");
}

// ---------------------------------------------------------------------------
// Localhost variants (SSRF prevention)
// ---------------------------------------------------------------------------

#[test]
fn localhost_blocked() {
    let policy = test_egress_policy();
    let t = Tainted::new("https://localhost/api");
    let result = SafeUrl::from_tainted(&t, &policy);
    assert!(result.is_err(), "localhost should be blocked");
}

#[test]
fn loopback_ipv4_blocked() {
    let policy = test_egress_policy();
    let t = Tainted::new("https://127.0.0.1/api");
    let result = SafeUrl::from_tainted(&t, &policy);
    assert!(result.is_err(), "127.0.0.1 should be blocked");
}

#[test]
fn loopback_ipv6_blocked() {
    let policy = test_egress_policy();
    let t = Tainted::new("https://[::1]/api");
    let result = SafeUrl::from_tainted(&t, &policy);
    assert!(result.is_err(), "::1 should be blocked");
}

// ---------------------------------------------------------------------------
// URL edge cases
// ---------------------------------------------------------------------------

#[test]
fn url_with_fragment_and_query_passes() {
    let policy = test_egress_policy();
    let t = Tainted::new("https://api.anthropic.com/v1/messages?key=val#fragment");
    let result = SafeUrl::from_tainted(&t, &policy);
    assert!(
        result.is_ok(),
        "URL with query and fragment should pass if host/port are valid"
    );
}

#[test]
fn empty_string_url_blocked() {
    let policy = test_egress_policy();
    let t = Tainted::new("");
    let result = SafeUrl::from_tainted(&t, &policy);
    assert!(result.is_err(), "empty URL should fail");
}

#[test]
fn url_without_host_blocked() {
    let policy = test_egress_policy();
    let t = Tainted::new("https:///path");
    let result = SafeUrl::from_tainted(&t, &policy);
    assert!(result.is_err(), "URL without host should fail");
}

// ---------------------------------------------------------------------------
// EgressPolicy direct tests
// ---------------------------------------------------------------------------

#[test]
fn check_url_no_host_returns_error() {
    let policy = test_egress_policy();
    // Construct a URL with no host component
    let url: url::Url = "data:text/plain,hello".parse().unwrap();
    let result = policy.check_url(&url);
    assert!(result.is_err());
}
