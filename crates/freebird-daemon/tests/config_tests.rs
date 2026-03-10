//! Config deserialization tests for the daemon composition root.
//!
//! These test that `AppConfig` deserializes correctly from TOML via figment,
//! including merge overrides, error messages for missing fields, and
//! validation of the shipped `config/default.toml`.

#![allow(clippy::expect_used, clippy::indexing_slicing, clippy::unwrap_used)]

use figment::Figment;
use figment::providers::{Format, Toml};
use freebird_traits::id::ProviderId;
use freebird_types::config::{AppConfig, ChannelKind, ProviderKind};

/// A complete, valid TOML config string for testing.
const fn complete_toml() -> &'static str {
    r#"
[runtime]
default_model = "claude-opus-4-6"
default_provider = "anthropic"
system_prompt = "You are a test assistant."
max_output_tokens = 8192
max_tool_rounds = 10
temperature = 0.7
max_turns_per_session = 50
drain_timeout_secs = 30

[[providers]]
id = "anthropic"
kind = "anthropic"

[[channels]]
id = "cli"
kind = "cli"
prompt = "test> "

[tools]
sandbox_root = "/tmp/sandbox"
default_timeout_secs = 30

[memory]
db_path = "/tmp/freebird.db"

[security]
max_tool_calls_per_turn = 25
require_consent_above = "high"

[logging]
level = "info"
format = "pretty"
"#
}

#[test]
fn test_config_deserializes_from_toml() {
    let config: AppConfig = Figment::new()
        .merge(Toml::string(complete_toml()))
        .extract()
        .expect("complete TOML should deserialize");

    assert_eq!(config.runtime.default_model.as_str(), "claude-opus-4-6");
    assert_eq!(config.runtime.default_provider.as_str(), "anthropic");
    assert_eq!(
        config.runtime.system_prompt.as_deref(),
        Some("You are a test assistant.")
    );
    assert_eq!(config.runtime.max_output_tokens, 8192);
    assert_eq!(config.runtime.max_tool_rounds, 10);
    assert_eq!(config.runtime.temperature, Some(0.7));
    assert_eq!(config.runtime.max_turns_per_session, 50);
    assert_eq!(config.runtime.drain_timeout_secs, 30);

    assert_eq!(config.providers.len(), 1);
    assert_eq!(config.providers[0].id, ProviderId::from("anthropic"));
    assert!(matches!(config.providers[0].kind, ProviderKind::Anthropic));

    assert_eq!(config.channels.len(), 1);
    assert!(matches!(config.channels[0].kind, ChannelKind::Cli));
    assert_eq!(config.channels[0].prompt.as_deref(), Some("test> "));
}

#[test]
fn test_config_figment_merge_override() {
    let config: AppConfig = Figment::new()
        .merge(Toml::string(complete_toml()))
        .merge(("runtime.default_model", "override-model"))
        .extract()
        .expect("merge override should work");

    assert_eq!(config.runtime.default_model.as_str(), "override-model");
    // Other fields unchanged
    assert_eq!(config.runtime.max_output_tokens, 8192);
}

#[test]
fn test_config_missing_required_field_errors() {
    let incomplete_toml = r#"
[runtime]
default_provider = "anthropic"
max_output_tokens = 8192
max_tool_rounds = 10
max_turns_per_session = 50
drain_timeout_secs = 30
"#;

    let result: Result<AppConfig, _> = Figment::new()
        .merge(Toml::string(incomplete_toml))
        .extract();

    assert!(result.is_err());
    let err_msg = format!("{}", result.unwrap_err());
    // Should mention the missing field
    assert!(
        err_msg.contains("default_model")
            || err_msg.contains("providers")
            || err_msg.contains("missing"),
        "error should mention missing field, got: {err_msg}"
    );
}

#[test]
fn test_config_invalid_provider_kind_errors() {
    let bad_toml = r#"
[runtime]
default_model = "test"
default_provider = "anthropic"
max_output_tokens = 8192
max_tool_rounds = 10
max_turns_per_session = 50
drain_timeout_secs = 30

[[providers]]
id = "bedrock"
kind = "bedrock"

[[channels]]
id = "cli"
kind = "cli"

[tools]
sandbox_root = "/tmp"
default_timeout_secs = 30

[memory]

[security]
max_tool_calls_per_turn = 25
require_consent_above = "high"

[logging]
level = "info"
format = "pretty"
"#;

    let result: Result<AppConfig, _> = Figment::new().merge(Toml::string(bad_toml)).extract();

    assert!(result.is_err());
    let err_msg = format!("{}", result.unwrap_err());
    assert!(
        err_msg.contains("bedrock") || err_msg.contains("unknown variant"),
        "error should mention invalid variant, got: {err_msg}"
    );
}

#[test]
fn test_config_optional_fields_absent() {
    let minimal_toml = r#"
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

    let config: AppConfig = Figment::new()
        .merge(Toml::string(minimal_toml))
        .extract()
        .expect("minimal TOML with no optional fields should deserialize");

    assert!(config.runtime.system_prompt.is_none());
    assert!(config.runtime.temperature.is_none());
    assert!(config.memory.db_path.is_none());
    assert!(config.channels[0].prompt.is_none());
}

#[test]
fn test_default_toml_deserializes() {
    let default_toml = include_str!("../../../config/default.toml");

    let config: AppConfig = Figment::new()
        .merge(Toml::string(default_toml))
        .extract()
        .expect("config/default.toml should deserialize without error");

    // Spot-check key fields
    assert_eq!(config.runtime.default_model.as_str(), "claude-sonnet-4-6");
    assert_eq!(config.runtime.default_provider.as_str(), "anthropic");
    assert!(!config.providers.is_empty());
    assert!(!config.channels.is_empty());
}
