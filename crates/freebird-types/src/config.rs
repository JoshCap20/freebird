//! Typed configuration structs, loaded from TOML/env via figment.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Top-level application configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppConfig {
    pub runtime: RuntimeConfig,
    pub providers: Vec<ProviderConfig>,
    pub channels: Vec<ChannelConfig>,
    pub tools: ToolsConfig,
    pub memory: MemoryConfig,
    pub security: SecurityConfig,
    pub logging: LoggingConfig,
}

/// Runtime behavior configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeConfig {
    /// Default LLM model ID (e.g., "claude-opus-4-6-20250929").
    pub default_model: String,
    /// Default provider ID — must match a `ProviderConfig::id` in `providers`.
    pub default_provider: String,
    /// System prompt prepended to every new conversation. None = no system prompt.
    pub system_prompt: Option<String>,
    /// Maximum tokens per provider response.
    pub max_output_tokens: u32,
    /// Maximum provider round-trips in a single agentic turn.
    pub max_tool_rounds: usize,
    /// Sampling temperature. `None` lets the provider use its default.
    pub temperature: Option<f32>,
    /// Maximum turns (user-assistant exchanges) before requiring a new session.
    pub max_turns_per_session: usize,
    /// Seconds to wait for in-flight work during graceful shutdown.
    pub drain_timeout_secs: u64,
}

/// Which LLM provider backend to use.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProviderKind {
    Anthropic,
    OpenAi,
    Ollama,
}

/// Provider-specific configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderConfig {
    pub id: String,
    pub kind: ProviderKind,
    pub default_model: Option<String>,
    pub base_url: Option<String>,
}

/// Which transport channel to use.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ChannelKind {
    Cli,
    Signal,
    WebSocket,
}

/// Channel-specific configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChannelConfig {
    pub id: String,
    pub kind: ChannelKind,
    /// Optional prompt string for interactive channels (e.g., "you> " for CLI).
    /// Consumed by channel implementations. None = channel uses its own default.
    pub prompt: Option<String>,
}

/// Tool sandbox configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolsConfig {
    pub sandbox_root: PathBuf,
    pub default_timeout_secs: u64,
}

/// Which memory storage backend to use.
/// TODO: Extend this with in-memory, Redis, Postgres, vector DBs, etc. as needed.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryKind {
    File,
    Sqlite,
}

/// Memory backend configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryConfig {
    pub kind: MemoryKind,
    pub base_dir: Option<PathBuf>,
}

/// Security policy configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityConfig {
    pub max_tool_calls_per_turn: usize,
    pub require_consent_above: String,
}

/// Logging and audit configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfig {
    pub level: String,
    pub format: String,
    pub audit_dir: Option<PathBuf>,
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::indexing_slicing)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config_deserializes() {
        let toml_str = include_str!("../../../config/default.toml");
        let config: AppConfig =
            toml::from_str(toml_str).expect("default.toml should deserialize into AppConfig");

        assert_eq!(config.runtime.default_model, "claude-opus-4-6-20250929");
        assert_eq!(config.runtime.default_provider, "anthropic");
        assert_eq!(config.runtime.max_output_tokens, 8192);
        assert_eq!(config.runtime.max_tool_rounds, 10);
        assert_eq!(config.runtime.max_turns_per_session, 50);
        assert_eq!(config.runtime.drain_timeout_secs, 30);
        assert!(config.runtime.system_prompt.is_some());
        assert_eq!(config.runtime.temperature, Some(0.7));
        assert_eq!(config.channels[0].prompt, Some("you> ".to_string()));
    }

    #[test]
    fn test_optional_fields_absent() {
        let toml_str = r#"
[runtime]
default_model = "claude-opus-4-6-20250929"
default_provider = "anthropic"
max_output_tokens = 8192
max_tool_rounds = 10
max_turns_per_session = 50
drain_timeout_secs = 30

[[providers]]
id = "anthropic"
kind = "anthropic"

[[channels]]
id = "cli"
kind = "cli"

[tools]
sandbox_root = "~/.freebird/sandbox"
default_timeout_secs = 30

[memory]
kind = "file"

[security]
max_tool_calls_per_turn = 25
require_consent_above = "high"

[logging]
level = "info"
format = "pretty"
"#;
        let config: AppConfig = toml::from_str(toml_str).expect("minimal TOML should deserialize");

        assert!(config.runtime.system_prompt.is_none());
        assert!(config.runtime.temperature.is_none());
        assert!(config.channels[0].prompt.is_none());
    }

    #[test]
    fn test_missing_required_field_errors() {
        let toml_str = r#"
[runtime]
default_model = "claude-opus-4-6-20250929"
max_output_tokens = 8192
max_tool_rounds = 10
max_turns_per_session = 50
drain_timeout_secs = 30

[[providers]]
id = "anthropic"
kind = "anthropic"

[[channels]]
id = "cli"
kind = "cli"

[tools]
sandbox_root = "~/.freebird/sandbox"
default_timeout_secs = 30

[memory]
kind = "file"

[security]
max_tool_calls_per_turn = 25
require_consent_above = "high"

[logging]
level = "info"
format = "pretty"
"#;
        let result = toml::from_str::<AppConfig>(toml_str);
        assert!(
            result.is_err(),
            "missing default_provider should cause deserialization error"
        );
    }

    #[test]
    fn test_temperature_none_when_absent() {
        let toml_str = r#"
[runtime]
default_model = "claude-opus-4-6-20250929"
default_provider = "anthropic"
max_output_tokens = 8192
max_tool_rounds = 10
max_turns_per_session = 50
drain_timeout_secs = 30

[[providers]]
id = "anthropic"
kind = "anthropic"

[[channels]]
id = "cli"
kind = "cli"

[tools]
sandbox_root = "~/.freebird/sandbox"
default_timeout_secs = 30

[memory]
kind = "file"

[security]
max_tool_calls_per_turn = 25
require_consent_above = "high"

[logging]
level = "info"
format = "pretty"
"#;
        let config: AppConfig =
            toml::from_str(toml_str).expect("TOML without temperature should deserialize");

        assert!(
            config.runtime.temperature.is_none(),
            "temperature should be None when absent, not a default value"
        );
    }
}
