//! Sensitive content detection and redaction.
//!
//! Scans text for patterns that resemble API keys, passwords, private keys,
//! and other credentials. Two entry points:
//!
//! - [`contains_sensitive_content`] — detection only (used by knowledge store writes)
//! - [`redact_sensitive_content`] — replaces detected secrets with `[REDACTED]`
//!   (used by the secret guard's output pipeline)
//!
//! Runs on ALL knowledge writes, regardless of kind or consent status,
//! BEFORE the consent gate — sensitive content is never even presented for approval.

use std::sync::LazyLock;

use regex::Regex;

/// Check if content contains patterns that resemble sensitive credentials.
///
/// Returns `Some(reason)` describing what was detected, or `None` if clean.
///
/// # Detected patterns
///
/// - API keys: `sk-`, `ghp_`, `gho_`, `ghs_`, `ghr_`, `xoxb-`, `xoxp-`, `xoxs-`
/// - Bearer tokens: `Bearer ` followed by long token
/// - AWS credentials: `AKIA` prefix, `aws_secret_access_key`
/// - PEM private keys: `-----BEGIN ... PRIVATE KEY-----`
/// - Password assignments: `password=`, `passwd=`, `secret_key=`, `private_key=`
/// - Generic high-entropy strings (base64 > 40 chars on a single line)
#[must_use]
pub fn contains_sensitive_content(content: &str) -> Option<&'static str> {
    if check_api_key_prefixes(content) {
        return Some("contains API key pattern");
    }

    if check_aws_credentials(content) {
        return Some("contains AWS credential pattern");
    }

    if content.contains("-----BEGIN") && content.contains("PRIVATE KEY-----") {
        return Some("contains PEM private key");
    }

    if check_password_assignments(content) {
        return Some("contains password or secret assignment");
    }

    if check_bearer_token(content) {
        return Some("contains Bearer token");
    }

    if check_high_entropy_lines(content) {
        return Some("contains high-entropy string resembling encoded secret");
    }

    None
}

/// Check for common API key prefixes.
fn check_api_key_prefixes(content: &str) -> bool {
    const PREFIXES: &[&str] = &[
        "sk-",      // OpenAI, Stripe
        "ghp_",     // GitHub personal access token
        "gho_",     // GitHub OAuth
        "ghs_",     // GitHub server-to-server
        "ghr_",     // GitHub refresh token
        "xoxb-",    // Slack bot token
        "xoxp-",    // Slack user token
        "xoxs-",    // Slack legacy token
        "sk_live_", // Stripe live key
        "pk_live_", // Stripe publishable live key
        "rk_live_", // Stripe restricted live key
    ];

    PREFIXES.iter().any(|prefix| content.contains(prefix))
}

/// Check for AWS access key IDs and secret patterns.
fn check_aws_credentials(content: &str) -> bool {
    let content_lower = content.to_lowercase();
    if content_lower.contains("aws_secret_access_key")
        || content_lower.contains("aws_session_token")
    {
        return true;
    }

    // AWS access key ID: starts with AKIA, 20 alphanumeric chars
    for (i, _) in content.match_indices("AKIA") {
        let remainder = &content[i..];
        if remainder.len() >= 20
            && remainder
                .chars()
                .take(20)
                .all(|c| c.is_ascii_alphanumeric())
        {
            return true;
        }
    }

    false
}

/// Check for password/secret assignment patterns with actual values.
fn check_password_assignments(content: &str) -> bool {
    const PATTERNS: &[&str] = &[
        "password=",
        "password:",
        "passwd=",
        "passwd:",
        "secret_key=",
        "secret_key:",
        "private_key=",
        "private_key:",
        "api_key=",
        "api_key:",
        "apikey=",
        "apikey:",
        "access_token=",
        "access_token:",
    ];

    let content_lower = content.to_lowercase();
    for pattern in PATTERNS {
        if let Some(pos) = content_lower.find(pattern) {
            let after = &content[pos + pattern.len()..];
            let value = after.trim();
            // Only flag if there's an actual value (not empty or a placeholder)
            if !value.is_empty()
                && !value.starts_with('<')
                && !value.starts_with('[')
                && !value.starts_with('{')
                && value != "\"\""
                && value != "''"
            {
                return true;
            }
        }
    }

    false
}

/// Check for Bearer tokens with sufficient length.
fn check_bearer_token(content: &str) -> bool {
    for (i, _) in content.match_indices("Bearer ") {
        let after = &content[i + 7..];
        let token_len = after
            .chars()
            .take_while(|c| c.is_ascii_alphanumeric() || *c == '-' || *c == '_' || *c == '.')
            .count();
        if token_len >= 20 {
            return true;
        }
    }
    false
}

/// Check for high-entropy base64-like lines.
fn check_high_entropy_lines(content: &str) -> bool {
    content.lines().any(|line| {
        let trimmed = line.trim();
        trimmed.len() >= 40 && is_high_entropy_base64(trimmed)
    })
}

/// Check if a string looks like a high-entropy base64-encoded value.
///
/// Returns true if >85% of characters are base64-alphabet and the string
/// has mixed case + digits (indicating encoded data, not normal prose).
fn is_high_entropy_base64(s: &str) -> bool {
    if s.len() < 40 {
        return false;
    }

    let total = s.len();
    let base64_chars = s
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || *c == '+' || *c == '/' || *c == '=')
        .count();

    // Must be >85% base64 characters
    #[allow(clippy::cast_precision_loss)] // Precision loss is acceptable for a ratio check
    if (base64_chars as f64 / total as f64) < 0.85 {
        return false;
    }

    // Must have mixed case + digits (normal words won't have this)
    let has_upper = s.chars().any(char::is_uppercase);
    let has_lower = s.chars().any(char::is_lowercase);
    let has_digit = s.chars().any(|c| c.is_ascii_digit());

    has_upper && has_lower && has_digit
}

// ── Compiled redaction regexes ───────────────────────────────────────

/// API key prefixes followed by token-like characters.
static RE_API_KEY: LazyLock<Regex> = LazyLock::new(|| {
    // nb: `unwrap` safety — the pattern is a compile-time constant.
    #[allow(clippy::unwrap_used)]
    Regex::new(
        r"(?:sk-|ghp_|gho_|ghs_|ghr_|xoxb-|xoxp-|xoxs-|sk_live_|pk_live_|rk_live_)[A-Za-z0-9_\-]{4,}",
    )
    .unwrap()
});

/// AWS access key IDs (AKIA + 16 alphanumeric).
static RE_AWS_KEY: LazyLock<Regex> = LazyLock::new(|| {
    #[allow(clippy::unwrap_used)]
    Regex::new(r"AKIA[A-Za-z0-9]{16}").unwrap()
});

/// AWS secret / session token assignments.
static RE_AWS_SECRET: LazyLock<Regex> = LazyLock::new(|| {
    #[allow(clippy::unwrap_used)]
    Regex::new(r"(?i)(aws_secret_access_key|aws_session_token)\s*[=:]\s*\S+").unwrap()
});

/// PEM private key blocks.
static RE_PEM: LazyLock<Regex> = LazyLock::new(|| {
    #[allow(clippy::unwrap_used)]
    Regex::new(r"-----BEGIN[A-Z \r\n]*PRIVATE KEY-----[\s\S]*?-----END[A-Z \r\n]*PRIVATE KEY-----")
        .unwrap()
});

/// Password / secret assignment with a value.
static RE_PASSWORD: LazyLock<Regex> = LazyLock::new(|| {
    #[allow(clippy::unwrap_used)]
    Regex::new(
        r"(?i)(password|passwd|secret_key|private_key|api_key|apikey|access_token)\s*[=:]\s*(\S+)",
    )
    .unwrap()
});

/// Bearer tokens with sufficient length (≥20 token chars).
static RE_BEARER: LazyLock<Regex> = LazyLock::new(|| {
    #[allow(clippy::unwrap_used)]
    Regex::new(r"Bearer [A-Za-z0-9_\-\.]{20,}").unwrap()
});

/// Redact sensitive content in text, replacing detected secrets with `[REDACTED]`.
///
/// Returns `(redacted_text, was_redacted)`. When `was_redacted` is `false` the
/// returned string is identical to the input (no allocation).
#[must_use]
pub fn redact_sensitive_content(content: &str) -> (String, bool) {
    use std::borrow::Cow;

    let mut result: Cow<'_, str> = Cow::Borrowed(content);

    // Order matters: PEM first (multi-line), then more specific patterns,
    // then generic password assignments last.
    for (re, replacement) in [
        (&*RE_PEM, "[REDACTED PEM KEY]"),
        (&*RE_API_KEY, "[REDACTED]"),
        (&*RE_AWS_KEY, "[REDACTED]"),
        (&*RE_AWS_SECRET, "$1=[REDACTED]"),
        (&*RE_BEARER, "Bearer [REDACTED]"),
    ] {
        let replaced = re.replace_all(&result, replacement);
        if let Cow::Owned(new) = replaced {
            result = Cow::Owned(new);
        }
    }

    // Password assignments: keep the key name, redact the value.
    let replaced = RE_PASSWORD.replace_all(&result, "$1=[REDACTED]");
    if let Cow::Owned(new) = replaced {
        result = Cow::Owned(new);
    }

    let redacted = matches!(result, Cow::Owned(_));
    (result.into_owned(), redacted)
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    // ── Positive detections (should flag) ──

    #[test]
    fn test_detects_openai_key() {
        assert!(contains_sensitive_content("My key is sk-abc123def456").is_some());
    }

    #[test]
    fn test_detects_github_pat() {
        assert!(contains_sensitive_content("ghp_ABCDEFghijklmnopqrstuvwxyz1234").is_some());
    }

    #[test]
    fn test_detects_slack_token() {
        assert!(contains_sensitive_content("token: xoxb-123456-abcdef").is_some());
    }

    #[test]
    fn test_detects_aws_access_key() {
        assert!(contains_sensitive_content("AKIAIOSFODNN7EXAMPLE").is_some());
    }

    #[test]
    fn test_detects_aws_secret_pattern() {
        assert!(
            contains_sensitive_content("aws_secret_access_key = wJalrXUtnFEMI/K7MDENG/bPxRfiCY")
                .is_some()
        );
    }

    #[test]
    fn test_detects_pem_private_key() {
        let pem = "-----BEGIN RSA PRIVATE KEY-----\nMIIEpAIBAAK...\n-----END RSA PRIVATE KEY-----";
        assert!(contains_sensitive_content(pem).is_some());
    }

    #[test]
    fn test_detects_ec_private_key() {
        let pem = "-----BEGIN EC PRIVATE KEY-----\ndata\n-----END EC PRIVATE KEY-----";
        assert!(contains_sensitive_content(pem).is_some());
    }

    #[test]
    fn test_detects_password_assignment() {
        assert!(contains_sensitive_content("password=hunter2").is_some());
        assert!(contains_sensitive_content("secret_key: myS3cretV4lue").is_some());
    }

    #[test]
    fn test_detects_bearer_token() {
        assert!(
            contains_sensitive_content(
                "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
            )
            .is_some()
        );
    }

    #[test]
    fn test_detects_stripe_live_key() {
        assert!(contains_sensitive_content("sk_live_4eC39HqLyjWDarjtT1zdp7dc").is_some());
    }

    // ── Negative detections (should NOT flag) ──

    #[test]
    fn test_allows_normal_technical_content() {
        assert!(
            contains_sensitive_content("The filesystem tool requires FileRead capability")
                .is_none()
        );
    }

    #[test]
    fn test_allows_code_discussion() {
        assert!(
            contains_sensitive_content("Use `cargo test -p freebird-security` to run tests")
                .is_none()
        );
    }

    #[test]
    fn test_allows_error_messages() {
        assert!(
            contains_sensitive_content("Error: connection refused to api.anthropic.com:443")
                .is_none()
        );
    }

    #[test]
    fn test_allows_password_label_without_value() {
        assert!(contains_sensitive_content("The password field is required").is_none());
    }

    #[test]
    fn test_allows_password_placeholder() {
        assert!(contains_sensitive_content("password=<your-password-here>").is_none());
        assert!(contains_sensitive_content("password=[REDACTED]").is_none());
    }

    #[test]
    fn test_allows_short_strings() {
        assert!(contains_sensitive_content("hello world").is_none());
    }

    #[test]
    fn test_allows_paths_and_urls() {
        assert!(contains_sensitive_content("/home/user/.freebird/freebird.db").is_none());
        assert!(contains_sensitive_content("https://api.anthropic.com/v1/messages").is_none());
    }

    #[test]
    fn test_allows_rust_code_snippets() {
        let code = r#"
            fn main() {
                let config = AppConfig::load();
                println!("Loaded {} providers", config.providers.len());
            }
        "#;
        assert!(contains_sensitive_content(code).is_none());
    }

    #[test]
    fn test_allows_empty_password_assignment() {
        assert!(contains_sensitive_content("password=\"\"").is_none());
        assert!(contains_sensitive_content("password=''").is_none());
    }

    #[test]
    fn test_allows_json_with_password_field_placeholder() {
        assert!(contains_sensitive_content("password={env.DB_PASSWORD}").is_none());
    }

    // ── Redaction tests ──

    #[test]
    fn test_redact_api_key_openai() {
        let (result, redacted) = redact_sensitive_content("My key is sk-ant-abc123def456");
        assert!(redacted);
        assert_eq!(result, "My key is [REDACTED]");
    }

    #[test]
    fn test_redact_github_pat() {
        let (result, redacted) =
            redact_sensitive_content("token: ghp_ABCDEFghijklmnopqrstuvwxyz1234");
        assert!(redacted);
        assert_eq!(result, "token: [REDACTED]");
    }

    #[test]
    fn test_redact_bearer_token() {
        let (result, redacted) =
            redact_sensitive_content("Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9");
        assert!(redacted);
        assert_eq!(result, "Authorization: Bearer [REDACTED]");
    }

    #[test]
    fn test_redact_pem_key() {
        let pem = "before\n-----BEGIN RSA PRIVATE KEY-----\nMIIEpAIBAAK...\n-----END RSA PRIVATE KEY-----\nafter";
        let (result, redacted) = redact_sensitive_content(pem);
        assert!(redacted);
        assert!(result.contains("[REDACTED PEM KEY]"));
        assert!(result.contains("before"));
        assert!(result.contains("after"));
        assert!(!result.contains("MIIEpAIBAAK"));
    }

    #[test]
    fn test_redact_password_assignment() {
        let (result, redacted) = redact_sensitive_content("password=hunter2");
        assert!(redacted);
        assert_eq!(result, "password=[REDACTED]");
    }

    #[test]
    fn test_redact_api_key_assignment() {
        let (result, redacted) = redact_sensitive_content("API_KEY=secret123abc");
        assert!(redacted);
        assert_eq!(result, "API_KEY=[REDACTED]");
    }

    #[test]
    fn test_redact_aws_key() {
        let (result, redacted) = redact_sensitive_content("key: AKIAIOSFODNN7EXAMPLE");
        assert!(redacted);
        assert_eq!(result, "key: [REDACTED]");
    }

    #[test]
    fn test_redact_aws_secret() {
        let (result, redacted) =
            redact_sensitive_content("aws_secret_access_key = wJalrXUtnFEMI/K7MDENG/bPxRfiCY");
        assert!(redacted);
        assert!(result.contains("[REDACTED]"));
        assert!(!result.contains("wJalrXUtnFEMI"));
    }

    #[test]
    fn test_no_redaction_normal_output() {
        let (result, redacted) =
            redact_sensitive_content("cargo test passed: 42 tests, 0 failures");
        assert!(!redacted);
        assert_eq!(result, "cargo test passed: 42 tests, 0 failures");
    }

    #[test]
    fn test_redaction_preserves_surrounding_text() {
        let input = "Start\npassword=hunter2\nEnd";
        let (result, redacted) = redact_sensitive_content(input);
        assert!(redacted);
        assert!(result.contains("Start"));
        assert!(result.contains("End"));
        assert!(!result.contains("hunter2"));
    }

    #[test]
    fn test_redact_multiple_secrets() {
        let input = "key1=sk-abcdef1234 and key2=ghp_ABCDEFghijklmnopqrstuvwxyz1234";
        let (result, redacted) = redact_sensitive_content(input);
        assert!(redacted);
        assert!(!result.contains("sk-abcdef1234"));
        assert!(!result.contains("ghp_ABCDEF"));
    }

    #[test]
    fn test_redact_stripe_live_key() {
        let (result, redacted) = redact_sensitive_content("key: sk_live_4eC39HqLyjWDarjtT1zdp7dc");
        assert!(redacted);
        assert_eq!(result, "key: [REDACTED]");
    }
}
