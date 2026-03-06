//! Typed configuration structs, loaded from TOML/env via figment.

use std::net::{IpAddr, Ipv4Addr};
use std::path::PathBuf;

use freebird_traits::id::{ChannelId, ModelId, ProviderId};
use freebird_traits::tool::RiskLevel;
use serde::{Deserialize, Serialize};

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
    #[serde(default)]
    pub daemon: DaemonConfig,
}

/// Daemon TCP listener configuration.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DaemonConfig {
    /// Bind address for the TCP listener (parsed at deserialization per §3.3).
    pub host: IpAddr,
    /// Port for the TCP listener. Use `0` to let the OS assign an ephemeral
    /// port (useful in tests; the actual port is available via
    /// `TcpListener::local_addr` after binding).
    pub port: u16,
}

impl Default for DaemonConfig {
    fn default() -> Self {
        Self {
            host: IpAddr::V4(Ipv4Addr::LOCALHOST),
            port: 7531,
        }
    }
}

/// Runtime behavior configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeConfig {
    /// Default LLM model ID (e.g., "claude-sonnet-4-6").
    pub default_model: ModelId,
    /// Default provider ID — must match a `ProviderConfig::id` in `providers`.
    pub default_provider: ProviderId,
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
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProviderKind {
    Anthropic,
    OpenAi,
    Ollama,
}

/// Provider-specific configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderConfig {
    pub id: ProviderId,
    pub kind: ProviderKind,
    pub default_model: Option<ModelId>,
    pub base_url: Option<String>,
}

/// Which transport channel to use.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ChannelKind {
    Cli,
    Signal,
    WebSocket,
}

/// Channel-specific configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChannelConfig {
    pub id: ChannelId,
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
    /// Additional directories the agent is allowed to access beyond the
    /// sandbox root. Typically set via the `--allow-dir` CLI flag.
    #[serde(default)]
    pub allowed_directories: Vec<PathBuf>,
}

/// Which memory storage backend to use.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
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
    /// Minimum risk level that requires explicit human consent before tool
    /// execution. Tools at this level or above trigger the consent gate.
    pub require_consent_above: RiskLevel,
}

/// Log severity level.
///
/// Replaces raw `String` per CLAUDE.md §3.2 ("make illegal states
/// unrepresentable") and §30 ("magic strings → enums").
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LogLevel {
    Trace,
    Debug,
    Info,
    Warn,
    Error,
}

/// Log output format.
///
/// Replaces raw `String` per CLAUDE.md §3.2 and §30.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LogFormat {
    Pretty,
    Json,
    Compact,
}

/// Logging and audit configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfig {
    pub level: LogLevel,
    pub format: LogFormat,
    pub audit_dir: Option<PathBuf>,
}

#[cfg(test)]
// `indexing_slicing`: tests index into known-length config arrays (e.g., `channels[0]`).
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::indexing_slicing)]
mod tests {
    use super::*;
    use std::net::{IpAddr, Ipv4Addr};

    /// Build a minimal valid TOML config with customizable sections.
    ///
    /// All sections use fixed valid defaults so tests can focus on one
    /// section at a time without duplicating boilerplate. Pass section
    /// overrides as `(section_name, toml_body)` pairs — they replace
    /// the corresponding default section entirely.
    fn config_toml(overrides: &[(&str, &str)]) -> String {
        let runtime = overrides.iter().find(|(k, _)| *k == "runtime").map_or(
            r#"default_model = "m"
default_provider = "p"
max_output_tokens = 1
max_tool_rounds = 1
max_turns_per_session = 1
drain_timeout_secs = 1"#,
            |(_, v)| v,
        );

        let security = overrides.iter().find(|(k, _)| *k == "security").map_or(
            r#"max_tool_calls_per_turn = 25
require_consent_above = "high""#,
            |(_, v)| v,
        );

        let logging = overrides.iter().find(|(k, _)| *k == "logging").map_or(
            r#"level = "info"
format = "pretty""#,
            |(_, v)| v,
        );

        let daemon = overrides.iter().find(|(k, _)| *k == "daemon");

        let daemon_section = daemon.map_or(String::new(), |(_, v)| format!("\n[daemon]\n{v}\n"));

        format!(
            r#"
[runtime]
{runtime}

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
{security}
{daemon_section}
[logging]
{logging}
"#
        )
    }

    /// Convenience: build a config with only a runtime override.
    fn config_toml_with_runtime(runtime_block: &str) -> String {
        config_toml(&[("runtime", runtime_block)])
    }

    #[test]
    fn test_default_config_deserializes() {
        let toml_str = include_str!("../../../config/default.toml");
        let config: AppConfig =
            toml::from_str(toml_str).expect("default.toml should deserialize into AppConfig");

        assert_eq!(config.runtime.default_model.as_str(), "claude-sonnet-4-6");
        assert_eq!(config.runtime.default_provider.as_str(), "anthropic");
        assert_eq!(
            config.runtime.system_prompt.as_deref(),
            Some("You are FreeBird, a helpful AI assistant.")
        );
        assert_eq!(config.runtime.max_output_tokens, 8192);
        assert_eq!(config.runtime.max_tool_rounds, 10);
        assert_eq!(config.runtime.temperature, Some(0.7));
        assert_eq!(config.runtime.max_turns_per_session, 50);
        assert_eq!(config.runtime.drain_timeout_secs, 30);
        assert_eq!(config.channels[0].prompt, Some("you> ".to_string()));
    }

    #[test]
    fn test_optional_fields_absent() {
        let toml_str = config_toml_with_runtime(
            r#"default_model = "claude-opus-4-6"
default_provider = "anthropic"
max_output_tokens = 8192
max_tool_rounds = 10
max_turns_per_session = 50
drain_timeout_secs = 30"#,
        );
        let config: AppConfig = toml::from_str(&toml_str).expect("minimal TOML should deserialize");

        assert!(config.runtime.system_prompt.is_none());
        assert!(config.runtime.temperature.is_none());
        assert!(config.channels[0].prompt.is_none());
    }

    #[test]
    fn test_missing_required_field_errors() {
        let toml_str = config_toml_with_runtime(
            r#"default_model = "claude-opus-4-6"
max_output_tokens = 8192
max_tool_rounds = 10
max_turns_per_session = 50
drain_timeout_secs = 30"#,
        );
        let result = toml::from_str::<AppConfig>(&toml_str);
        assert!(
            result.is_err(),
            "missing default_provider should cause deserialization error"
        );
    }

    #[test]
    fn test_temperature_none_when_absent() {
        let toml_str = config_toml_with_runtime(
            r#"default_model = "claude-opus-4-6"
default_provider = "anthropic"
max_output_tokens = 8192
max_tool_rounds = 10
max_turns_per_session = 50
drain_timeout_secs = 30"#,
        );
        let config: AppConfig =
            toml::from_str(&toml_str).expect("TOML without temperature should deserialize");

        assert!(
            config.runtime.temperature.is_none(),
            "temperature should be None when absent, not a default value"
        );
    }

    // ── New tests for enum config fields ──────────────────────────────

    #[test]
    fn test_security_config_require_consent_above_is_risk_level() {
        let toml_str = config_toml_with_runtime(
            r#"default_model = "m"
default_provider = "p"
max_output_tokens = 1
max_tool_rounds = 1
max_turns_per_session = 1
drain_timeout_secs = 1"#,
        );
        let config: AppConfig = toml::from_str(&toml_str).unwrap();
        assert_eq!(config.security.require_consent_above, RiskLevel::High);
    }

    #[test]
    fn test_security_config_all_risk_levels() {
        for (value, expected) in [
            ("low", RiskLevel::Low),
            ("medium", RiskLevel::Medium),
            ("high", RiskLevel::High),
            ("critical", RiskLevel::Critical),
        ] {
            let security_block =
                format!("max_tool_calls_per_turn = 1\nrequire_consent_above = \"{value}\"");
            let toml_str = config_toml(&[("security", &security_block)]);
            let config: AppConfig = toml::from_str(&toml_str).unwrap();
            assert_eq!(config.security.require_consent_above, expected);
        }
    }

    #[test]
    fn test_security_config_invalid_risk_level_errors() {
        let toml_str = config_toml(&[(
            "security",
            "max_tool_calls_per_turn = 1\nrequire_consent_above = \"banana\"",
        )]);
        let result = toml::from_str::<AppConfig>(&toml_str);
        assert!(
            result.is_err(),
            "invalid risk level should fail deserialization"
        );
    }

    #[test]
    fn test_log_level_serde_roundtrip() {
        for (level, expected_json) in [
            (LogLevel::Trace, "\"trace\""),
            (LogLevel::Debug, "\"debug\""),
            (LogLevel::Info, "\"info\""),
            (LogLevel::Warn, "\"warn\""),
            (LogLevel::Error, "\"error\""),
        ] {
            let json = serde_json::to_string(&level).unwrap();
            assert_eq!(json, expected_json);
            let back: LogLevel = serde_json::from_str(&json).unwrap();
            assert_eq!(back, level);
        }
    }

    #[test]
    fn test_log_format_serde_roundtrip() {
        for (fmt, expected_json) in [
            (LogFormat::Pretty, "\"pretty\""),
            (LogFormat::Json, "\"json\""),
            (LogFormat::Compact, "\"compact\""),
        ] {
            let json = serde_json::to_string(&fmt).unwrap();
            assert_eq!(json, expected_json);
            let back: LogFormat = serde_json::from_str(&json).unwrap();
            assert_eq!(back, fmt);
        }
    }

    #[test]
    fn test_logging_config_deserialized_from_default_toml() {
        let toml_str = include_str!("../../../config/default.toml");
        let config: AppConfig = toml::from_str(toml_str).unwrap();

        assert_eq!(config.logging.level, LogLevel::Info);
        assert_eq!(config.logging.format, LogFormat::Pretty);
    }

    #[test]
    fn test_invalid_log_level_errors() {
        let toml_str = config_toml(&[("logging", "level = \"verbose\"\nformat = \"pretty\"")]);
        let result = toml::from_str::<AppConfig>(&toml_str);
        assert!(
            result.is_err(),
            "invalid log level should fail deserialization"
        );
    }

    #[test]
    fn test_invalid_log_format_errors() {
        let toml_str = config_toml(&[("logging", "level = \"info\"\nformat = \"xml\"")]);
        let result = toml::from_str::<AppConfig>(&toml_str);
        assert!(
            result.is_err(),
            "invalid log format should fail deserialization"
        );
    }

    // ── Daemon config tests ──────────────────────────────────────────

    #[test]
    fn test_daemon_config_defaults_when_absent() {
        let toml_str = config_toml(&[]);
        let config: AppConfig = toml::from_str(&toml_str).unwrap();
        assert_eq!(config.daemon.host, IpAddr::V4(Ipv4Addr::LOCALHOST));
        assert_eq!(config.daemon.port, 7531);
    }

    #[test]
    fn test_daemon_config_explicit_values() {
        let toml_str = config_toml(&[("daemon", "host = \"0.0.0.0\"\nport = 9000")]);
        let config: AppConfig = toml::from_str(&toml_str).unwrap();
        assert_eq!(config.daemon.host, IpAddr::V4(Ipv4Addr::UNSPECIFIED));
        assert_eq!(config.daemon.port, 9000);
    }

    #[test]
    fn test_daemon_config_from_default_toml() {
        let toml_str = include_str!("../../../config/default.toml");
        let config: AppConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(config.daemon.host, IpAddr::V4(Ipv4Addr::LOCALHOST));
        assert_eq!(config.daemon.port, 7531);
    }

    #[test]
    fn test_daemon_config_invalid_host_errors() {
        let toml_str = config_toml(&[("daemon", "host = \"not-an-ip\"\nport = 7531")]);
        let result = toml::from_str::<AppConfig>(&toml_str);
        assert!(
            result.is_err(),
            "invalid IP address should fail deserialization"
        );
    }

    #[test]
    fn test_daemon_config_serde_roundtrip() {
        let daemon = DaemonConfig {
            host: IpAddr::V4(Ipv4Addr::new(10, 0, 0, 1)),
            port: 8080,
        };
        let json = serde_json::to_string(&daemon).unwrap();
        let back: DaemonConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(daemon, back);
    }
}
