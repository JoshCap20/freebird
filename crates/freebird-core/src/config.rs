//! Configuration loading, validation, and security enforcement.

use anyhow::{Context, Result, bail};
use figment::Figment;
use figment::providers::{Env, Format, Toml};

use freebird_types::config::AppConfig;

/// Load configuration from TOML file with environment variable overrides.
///
/// Sources (in priority order):
/// 1. `FREEBIRD_CONFIG` env var (path to TOML file, default: `config/default.toml`)
/// 2. Environment variables with `FREEBIRD_` prefix and `__` separator for nesting
pub fn load_config() -> Result<AppConfig> {
    let config_path =
        std::env::var("FREEBIRD_CONFIG").unwrap_or_else(|_| "config/default.toml".into());

    Figment::new()
        .merge(Toml::file(&config_path))
        .merge(Env::prefixed("FREEBIRD_").split("__"))
        .extract()
        .with_context(|| format!("failed to load configuration from `{config_path}`"))
}

/// Validate configuration invariants that cannot be expressed in types.
pub fn validate_config(config: &AppConfig) -> Result<()> {
    if config.providers.is_empty() {
        bail!("at least one provider must be configured in [[providers]]");
    }

    // Budget limits must be positive
    if config.security.budgets.max_tokens_per_session == 0 {
        bail!("security.budgets.max_tokens_per_session must be > 0");
    }
    if config.security.budgets.max_tokens_per_request == 0 {
        bail!("security.budgets.max_tokens_per_request must be > 0");
    }
    if config.security.budgets.max_tool_rounds_per_turn == 0 {
        bail!("security.budgets.max_tool_rounds_per_turn must be > 0");
    }
    if config.runtime.max_tool_rounds == 0 {
        bail!("runtime.max_tool_rounds must be > 0");
    }

    // Egress allowed_hosts must be valid domain-like strings
    for host in &config.security.egress.allowed_hosts {
        if host.is_empty() || host.contains(' ') {
            bail!(
                "security.egress.allowed_hosts contains invalid entry: `{host}` \
                 (must be non-empty and contain no spaces)"
            );
        }
    }

    // PBKDF2 iterations minimum
    if config.memory.pbkdf2_iterations < 1000 {
        bail!(
            "memory.pbkdf2_iterations must be >= 1000 (got {})",
            config.memory.pbkdf2_iterations
        );
    }

    // Session TTL must be positive
    if config.runtime.session.session_ttl_secs == 0 {
        bail!("runtime.session.session_ttl_secs must be > 0");
    }

    Ok(())
}

/// Enforce minimum security thresholds regardless of env var overrides.
///
/// Prevents misconfiguration (accidental or adversarial) from weakening
/// security below safe minimums. Values that exceed limits are clamped
/// with a warning.
pub fn enforce_security_minimums(mut config: AppConfig) -> AppConfig {
    use freebird_traits::tool::RiskLevel;

    // require_consent_above must be at most High (Critical would disable all consent)
    if config.security.require_consent_above > RiskLevel::High {
        tracing::warn!(
            original = ?config.security.require_consent_above,
            clamped = ?RiskLevel::High,
            "require_consent_above exceeds maximum — clamping to High"
        );
        config.security.require_consent_above = RiskLevel::High;
    }

    // max_tokens_per_session must be > 0
    if config.security.budgets.max_tokens_per_session == 0 {
        tracing::warn!("max_tokens_per_session is 0 — clamping to 1");
        config.security.budgets.max_tokens_per_session = 1;
    }

    // max_tool_rounds_per_turn must be > 0 and <= 50
    if config.security.budgets.max_tool_rounds_per_turn == 0 {
        tracing::warn!("max_tool_rounds_per_turn is 0 — clamping to 1");
        config.security.budgets.max_tool_rounds_per_turn = 1;
    }
    if config.security.budgets.max_tool_rounds_per_turn > 50 {
        tracing::warn!(
            original = config.security.budgets.max_tool_rounds_per_turn,
            clamped = 50,
            "max_tool_rounds_per_turn exceeds maximum — clamping to 50"
        );
        config.security.budgets.max_tool_rounds_per_turn = 50;
    }

    config
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::indexing_slicing, clippy::unwrap_used)]
mod tests {
    use super::*;

    fn valid_config() -> AppConfig {
        use figment::providers::{Format, Toml};

        let toml = r#"
[runtime]
default_model = "test"
default_provider = "test"
max_output_tokens = 1024
max_tool_rounds = 5
max_turns_per_session = 10
drain_timeout_secs = 5

[[providers]]
id = "test"
kind = "anthropic"

[[channels]]
id = "cli"
kind = "cli"

[tools]
sandbox_root = "/tmp"
default_timeout_secs = 10

[memory]

[security]
max_tool_calls_per_turn = 10
require_consent_above = "high"

[logging]
level = "info"
format = "pretty"
"#;
        Figment::new()
            .merge(Toml::string(toml))
            .extract()
            .expect("test config should deserialize")
    }

    #[test]
    fn test_validate_config_valid() {
        let config = valid_config();
        assert!(validate_config(&config).is_ok());
    }

    #[test]
    fn test_validate_config_empty_providers_errors() {
        let mut config = valid_config();
        config.providers.clear();
        let err = validate_config(&config).unwrap_err();
        assert!(
            err.to_string().contains("at least one provider"),
            "expected provider error, got: {err}"
        );
    }

    #[test]
    fn test_validate_config_no_channels_still_valid() {
        let mut config = valid_config();
        config.channels.clear();
        assert!(
            validate_config(&config).is_ok(),
            "channels are optional — TcpChannel uses daemon config, not [[channels]]"
        );
    }

    #[test]
    fn test_daemon_config_defaults_used_when_absent() {
        let config = valid_config();
        assert_eq!(
            config.daemon.host,
            std::net::IpAddr::V4(std::net::Ipv4Addr::LOCALHOST)
        );
        assert_eq!(config.daemon.port, 7531);
    }

    #[test]
    fn test_validate_config_zero_max_tool_rounds_errors() {
        let mut config = valid_config();
        config.runtime.max_tool_rounds = 0;
        let err = validate_config(&config).unwrap_err();
        assert!(err.to_string().contains("max_tool_rounds"), "got: {err}");
    }

    #[test]
    fn test_validate_config_zero_budget_tokens_errors() {
        let mut config = valid_config();
        config.security.budgets.max_tokens_per_session = 0;
        let err = validate_config(&config).unwrap_err();
        assert!(
            err.to_string().contains("max_tokens_per_session"),
            "got: {err}"
        );
    }

    #[test]
    fn test_validate_config_invalid_egress_host_errors() {
        let mut config = valid_config();
        config.security.egress.allowed_hosts = vec!["valid.com".into(), String::new()];
        let err = validate_config(&config).unwrap_err();
        assert!(err.to_string().contains("allowed_hosts"), "got: {err}");
    }

    #[test]
    fn test_validate_config_low_pbkdf2_errors() {
        let mut config = valid_config();
        config.memory.pbkdf2_iterations = 500;
        let err = validate_config(&config).unwrap_err();
        assert!(err.to_string().contains("pbkdf2_iterations"), "got: {err}");
    }

    #[test]
    fn test_enforce_security_minimums_clamps_consent() {
        let mut config = valid_config();
        config.security.require_consent_above = freebird_traits::tool::RiskLevel::Critical;
        let config = enforce_security_minimums(config);
        assert_eq!(
            config.security.require_consent_above,
            freebird_traits::tool::RiskLevel::High
        );
    }

    #[test]
    fn test_enforce_security_minimums_clamps_tool_rounds() {
        let mut config = valid_config();
        config.security.budgets.max_tool_rounds_per_turn = 100;
        let config = enforce_security_minimums(config);
        assert_eq!(config.security.budgets.max_tool_rounds_per_turn, 50);
    }
}
