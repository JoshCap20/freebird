//! Typed configuration structs, loaded from TOML/env via figment.

mod memory;
mod runtime;
mod security;

// Re-export all public types so `freebird_types::config::*` continues to work.
pub use memory::{KnowledgeConfig, MemoryConfig};
pub use runtime::{
    ContextConfig, ConversationSummary, DaemonConfig, EditConfig, LargeEditAction, RuntimeConfig,
    SessionConfig, SummarizationConfig, ToolsConfig,
};
pub use security::{
    BudgetConfig, EgressConfig, InjectionConfig, InjectionResponse, SecretGuardAction,
    SecretGuardConfig, SecurityConfig,
};

use freebird_traits::id::{ChannelId, ModelId, ProviderId};
use serde::{Deserialize, Serialize};

/// Top-level application configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppConfig {
    pub runtime: RuntimeConfig,
    pub providers: Vec<ProviderConfig>,
    pub channels: Vec<ChannelConfig>,
    pub tools: ToolsConfig,
    pub memory: MemoryConfig,
    #[serde(default)]
    pub knowledge: KnowledgeConfig,
    #[serde(default)]
    pub summarization: SummarizationConfig,
    pub security: SecurityConfig,
    pub logging: LoggingConfig,
    #[serde(default)]
    pub daemon: DaemonConfig,
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

/// Log severity level.
///
/// Replaces raw `String` per CLAUDE.md \u{00a7}3.2 ("make illegal states
/// unrepresentable") and \u{00a7}30 ("magic strings \u{2192} enums").
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
/// Replaces raw `String` per CLAUDE.md \u{00a7}3.2 and \u{00a7}30.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LogFormat {
    Pretty,
    Json,
    Compact,
}

/// Logging configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfig {
    pub level: LogLevel,
    pub format: LogFormat,
}

#[cfg(test)]
// `indexing_slicing`: tests index into known-length config arrays (e.g., `channels[0]`).
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::indexing_slicing)]
mod tests {
    use super::*;
    use chrono::Utc;
    use freebird_traits::id::SessionId;
    use freebird_traits::tool::RiskLevel;
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

        let tools = overrides.iter().find(|(k, _)| *k == "tools").map_or(
            r#"sandbox_root = "~/.freebird/sandbox"
default_timeout_secs = 30
allowed_shell_commands = ["ls", "cat", "grep", "head", "tail", "wc"]
max_shell_output_bytes = 1048576"#,
            |(_, v)| v,
        );

        let daemon = overrides.iter().find(|(k, _)| *k == "daemon");
        let knowledge = overrides.iter().find(|(k, _)| *k == "knowledge");

        let daemon_section = daemon.map_or(String::new(), |(_, v)| format!("\n[daemon]\n{v}\n"));
        let knowledge_section =
            knowledge.map_or(String::new(), |(_, v)| format!("\n[knowledge]\n{v}\n"));

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
{tools}

[memory]
db_path = "~/.freebird/freebird.db"
{knowledge_section}
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
        let toml_str = include_str!("../../../../config/default.toml");
        let config: AppConfig =
            toml::from_str(toml_str).expect("default.toml should deserialize into AppConfig");

        assert_eq!(config.runtime.default_model.as_str(), "claude-sonnet-4-6");
        assert_eq!(config.runtime.default_provider.as_str(), "anthropic");
        assert_eq!(
            config.runtime.system_prompt.as_deref(),
            Some("You are Freebird, a helpful AI assistant.")
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

    // -- New tests for enum config fields --

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
        let toml_str = include_str!("../../../../config/default.toml");
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

    // -- Daemon config tests --

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
        let toml_str = include_str!("../../../../config/default.toml");
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

    // -- Shell config tests --

    #[test]
    fn test_shell_config_from_default_toml() {
        let toml_str = include_str!("../../../../config/default.toml");
        let config: AppConfig = toml::from_str(toml_str).unwrap();
        let cmds = &config.tools.allowed_shell_commands;
        // Must include original read-only commands plus build/vcs/file-mgmt commands
        for expected in &[
            "ls", "cat", "grep", "head", "tail", "wc", "cargo", "rustfmt", "git", "find", "diff",
            "mkdir", "rm", "cp", "mv", "touch", "sort", "uniq", "sed", "awk", "tr", "cut",
        ] {
            assert!(
                cmds.contains(&expected.to_string()),
                "missing command: {expected}"
            );
        }
        assert_eq!(config.tools.max_shell_output_bytes, 1_048_576);
    }

    #[test]
    fn test_shell_config_serde_defaults_when_absent() {
        // Use a tools override that omits shell fields to verify serde defaults apply
        let toml_str = config_toml(&[(
            "tools",
            "sandbox_root = \"/tmp\"\ndefault_timeout_secs = 10",
        )]);
        let config: AppConfig = toml::from_str(&toml_str).unwrap();
        assert_eq!(
            config.tools.allowed_shell_commands,
            vec!["ls", "cat", "grep", "head", "tail", "wc"]
        );
        assert_eq!(config.tools.max_shell_output_bytes, 1_048_576);
    }

    // -- Edit config tests --

    #[test]
    fn test_edit_config_defaults_when_absent() {
        let toml_str = config_toml(&[]);
        let config: AppConfig = toml::from_str(&toml_str).unwrap();
        assert!(config.tools.edit.diff_preview);
        assert_eq!(config.tools.edit.diff_context_lines, 3);
    }

    #[test]
    fn test_edit_config_explicit_values() {
        let toml_str = config_toml(&[(
            "tools",
            "sandbox_root = \"/tmp\"\ndefault_timeout_secs = 10\n\n[tools.edit]\ndiff_preview = false\ndiff_context_lines = 5",
        )]);
        let config: AppConfig = toml::from_str(&toml_str).unwrap();
        assert!(!config.tools.edit.diff_preview);
        assert_eq!(config.tools.edit.diff_context_lines, 5);
    }

    // -- Consent config tests --

    #[test]
    fn test_security_config_consent_defaults() {
        // Config TOML without consent fields — serde defaults should apply.
        let toml_str = config_toml(&[(
            "security",
            "max_tool_calls_per_turn = 25\nrequire_consent_above = \"high\"",
        )]);
        let config: AppConfig = toml::from_str(&toml_str).unwrap();
        assert_eq!(config.security.consent_timeout_secs, 60);
        assert_eq!(config.security.max_pending_consent_requests, 5);
    }

    #[test]
    fn test_security_config_consent_explicit() {
        let toml_str = config_toml(&[(
            "security",
            "max_tool_calls_per_turn = 25\nrequire_consent_above = \"high\"\nconsent_timeout_secs = 120\nmax_pending_consent_requests = 10",
        )]);
        let config: AppConfig = toml::from_str(&toml_str).unwrap();
        assert_eq!(config.security.consent_timeout_secs, 120);
        assert_eq!(config.security.max_pending_consent_requests, 10);
    }

    #[test]
    fn test_default_toml_deserializes_with_consent() {
        let toml_str = include_str!("../../../../config/default.toml");
        let config: AppConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(config.security.consent_timeout_secs, 60);
        assert_eq!(config.security.max_pending_consent_requests, 5);
    }

    // -- Egress config tests --

    #[test]
    fn test_egress_config_defaults_when_absent() {
        // Config TOML without egress section — serde defaults should apply.
        let toml_str = config_toml(&[(
            "security",
            "max_tool_calls_per_turn = 25\nrequire_consent_above = \"high\"",
        )]);
        let config: AppConfig = toml::from_str(&toml_str).unwrap();
        assert_eq!(
            config.security.egress.allowed_hosts,
            vec!["api.anthropic.com", "api.openai.com"]
        );
        assert_eq!(config.security.egress.allowed_ports, vec![443]);
        assert_eq!(config.security.egress.max_response_bytes, 102_400);
        assert_eq!(config.security.egress.request_timeout_secs, 30);
    }

    #[test]
    fn test_egress_config_explicit_values() {
        let toml_str = config_toml(&[(
            "security",
            "max_tool_calls_per_turn = 25\nrequire_consent_above = \"high\"\n\n[security.egress]\nallowed_hosts = [\"custom.api.com\"]\nallowed_ports = [443, 8443]\nmax_response_bytes = 2097152\nrequest_timeout_secs = 60",
        )]);
        let config: AppConfig = toml::from_str(&toml_str).unwrap();
        assert_eq!(config.security.egress.allowed_hosts, vec!["custom.api.com"]);
        assert_eq!(config.security.egress.allowed_ports, vec![443, 8443]);
        assert_eq!(config.security.egress.max_response_bytes, 2_097_152);
        assert_eq!(config.security.egress.request_timeout_secs, 60);
    }

    #[test]
    fn test_default_toml_deserializes_egress() {
        let toml_str = include_str!("../../../../config/default.toml");
        let config: AppConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(
            config.security.egress.allowed_hosts,
            vec![
                "api.anthropic.com",
                "api.openai.com",
                "api.open-meteo.com",
                "api.github.com",
                "api.exchangerate-api.com",
                "timeapi.io",
                "en.wikipedia.org",
                "api.stackexchange.com",
            ]
        );
        assert_eq!(config.security.egress.allowed_ports, vec![443]);
        assert_eq!(config.security.egress.max_response_bytes, 102_400);
        assert_eq!(config.security.egress.request_timeout_secs, 30);
    }

    #[test]
    fn test_egress_config_serde_roundtrip() {
        let egress = EgressConfig {
            allowed_hosts: vec!["example.com".into()],
            allowed_ports: vec![443, 8080],
            max_response_bytes: 512_000,
            request_timeout_secs: 15,
            rate_limit_per_minute: 30,
            max_request_body_bytes: 2_097_152,
        };
        let json = serde_json::to_string(&egress).unwrap();
        let back: EgressConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(back.allowed_hosts, vec!["example.com"]);
        assert_eq!(back.allowed_ports, vec![443, 8080]);
        assert_eq!(back.max_response_bytes, 512_000);
        assert_eq!(back.request_timeout_secs, 15);
        assert_eq!(back.rate_limit_per_minute, 30);
        assert_eq!(back.max_request_body_bytes, 2_097_152);
    }

    // -- Knowledge config tests --

    #[test]
    fn test_knowledge_config_defaults_when_absent() {
        let toml_str = config_toml(&[]);
        let config: AppConfig = toml::from_str(&toml_str).unwrap();
        assert!(config.knowledge.auto_retrieve);
        assert_eq!(config.knowledge.max_context_entries, 5);
        assert!((config.knowledge.relevance_threshold - (-0.5)).abs() < f64::EPSILON);
        assert_eq!(config.knowledge.max_context_tokens, 2000);
    }

    #[test]
    fn test_knowledge_config_explicit_values() {
        let toml_str = config_toml(&[(
            "knowledge",
            "auto_retrieve = false\nmax_context_entries = 10\nrelevance_threshold = -1.0\nmax_context_tokens = 4000",
        )]);
        let config: AppConfig = toml::from_str(&toml_str).unwrap();
        assert!(!config.knowledge.auto_retrieve);
        assert_eq!(config.knowledge.max_context_entries, 10);
        assert!((config.knowledge.relevance_threshold - (-1.0)).abs() < f64::EPSILON);
        assert_eq!(config.knowledge.max_context_tokens, 4000);
    }

    #[test]
    fn test_memory_config_pbkdf2_default() {
        let toml_str = config_toml(&[]);
        let config: AppConfig = toml::from_str(&toml_str).unwrap();
        assert_eq!(config.memory.pbkdf2_iterations, 100_000);
    }

    #[test]
    fn test_memory_config_db_path_from_default_toml() {
        let toml_str = include_str!("../../../../config/default.toml");
        let config: AppConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(
            config.memory.db_path.as_ref().map(|p| p.to_str().unwrap()),
            Some("~/.freebird/freebird.db")
        );
    }

    #[test]
    fn test_knowledge_config_from_default_toml() {
        let toml_str = include_str!("../../../../config/default.toml");
        let config: AppConfig = toml::from_str(toml_str).unwrap();
        assert!(config.knowledge.auto_retrieve);
        assert_eq!(config.knowledge.max_context_entries, 5);
        assert_eq!(config.knowledge.max_context_tokens, 2000);
    }

    // -- Secret guard config tests --

    #[test]
    fn test_secret_guard_defaults_when_absent() {
        let toml_str = config_toml(&[]);
        let config: AppConfig = toml::from_str(&toml_str).unwrap();
        assert!(config.security.secret_guard.enabled);
        assert_eq!(
            config.security.secret_guard.action,
            SecretGuardAction::Consent
        );
        assert!(config.security.secret_guard.redact_output);
        assert!(
            config
                .security
                .secret_guard
                .extra_sensitive_file_patterns
                .is_empty()
        );
        assert!(
            config
                .security
                .secret_guard
                .extra_sensitive_command_patterns
                .is_empty()
        );
    }

    #[test]
    fn test_secret_guard_explicit_values() {
        let toml_str = config_toml(&[(
            "security",
            "max_tool_calls_per_turn = 25\nrequire_consent_above = \"high\"\n\n[security.secret_guard]\nenabled = false\naction = \"block\"\nredact_output = false\nextra_sensitive_file_patterns = [\"*.secret\"]\nextra_sensitive_command_patterns = [\"^myutil\"]",
        )]);
        let config: AppConfig = toml::from_str(&toml_str).unwrap();
        assert!(!config.security.secret_guard.enabled);
        assert_eq!(
            config.security.secret_guard.action,
            SecretGuardAction::Block
        );
        assert!(!config.security.secret_guard.redact_output);
        assert_eq!(
            config.security.secret_guard.extra_sensitive_file_patterns,
            vec!["*.secret"]
        );
        assert_eq!(
            config
                .security
                .secret_guard
                .extra_sensitive_command_patterns,
            vec!["^myutil"]
        );
    }

    #[test]
    fn test_secret_guard_from_default_toml() {
        let toml_str = include_str!("../../../../config/default.toml");
        let config: AppConfig = toml::from_str(toml_str).unwrap();
        assert!(config.security.secret_guard.enabled);
        assert_eq!(
            config.security.secret_guard.action,
            SecretGuardAction::Consent
        );
        assert!(config.security.secret_guard.redact_output);
    }

    #[test]
    fn test_secret_guard_action_serde_roundtrip() {
        for (action, expected_json) in [
            (SecretGuardAction::Consent, "\"consent\""),
            (SecretGuardAction::Block, "\"block\""),
        ] {
            let json = serde_json::to_string(&action).unwrap();
            assert_eq!(json, expected_json);
            let back: SecretGuardAction = serde_json::from_str(&json).unwrap();
            assert_eq!(back, action);
        }
    }

    #[test]
    fn test_secret_guard_invalid_action_errors() {
        let toml_str = config_toml(&[(
            "security",
            "max_tool_calls_per_turn = 25\nrequire_consent_above = \"high\"\n\n[security.secret_guard]\naction = \"ignore\"",
        )]);
        let result = toml::from_str::<AppConfig>(&toml_str);
        assert!(
            result.is_err(),
            "invalid secret guard action should fail deserialization"
        );
    }

    // -- LargeEditAction tests --

    #[test]
    fn test_large_edit_action_serde_roundtrip() {
        for (action, expected_json) in [
            (LargeEditAction::Warn, "\"warn\""),
            (LargeEditAction::Block, "\"block\""),
            (LargeEditAction::Consent, "\"consent\""),
        ] {
            let json = serde_json::to_string(&action).unwrap();
            assert_eq!(json, expected_json);
            let back: LargeEditAction = serde_json::from_str(&json).unwrap();
            assert_eq!(back, action);
        }
    }

    #[test]
    fn test_large_edit_action_invalid_value_rejected() {
        let result = serde_json::from_str::<LargeEditAction>("\"ignore\"");
        assert!(
            result.is_err(),
            "invalid large-edit action should fail deserialization"
        );
    }

    #[test]
    fn test_edit_config_large_edit_defaults() {
        // Parse EditConfig with only required fields — large-edit fields use defaults
        let toml_str = config_toml(&[(
            "tools",
            "sandbox_root = \"/tmp\"\ndefault_timeout_secs = 10\n\n[tools.edit]\ndiff_preview = true\ndiff_context_lines = 3\nsyntax_validation = true",
        )]);
        let config: AppConfig = toml::from_str(&toml_str).unwrap();
        assert!(
            (config.tools.edit.large_edit_threshold - 0.5).abs() < f64::EPSILON,
            "default threshold should be 0.5"
        );
        assert_eq!(config.tools.edit.large_edit_action, LargeEditAction::Warn);
    }

    // -- Summarization config tests --

    #[test]
    fn test_summarization_config_defaults() {
        let config = SummarizationConfig::default();
        assert!(config.enabled);
        assert!((config.trigger_threshold - 0.75).abs() < f64::EPSILON);
        assert_eq!(config.preserve_recent_turns, 5);
        assert_eq!(config.max_summary_tokens, 1024);
        assert_eq!(config.min_turns_before_summarize, 8);
    }

    #[test]
    fn test_summarization_config_from_default_toml() {
        let toml_str = include_str!("../../../../config/default.toml");
        let config: AppConfig =
            toml::from_str(toml_str).expect("default.toml should deserialize into AppConfig");

        assert!(config.summarization.enabled);
        assert!((config.summarization.trigger_threshold - 0.75).abs() < f64::EPSILON);
        assert_eq!(config.summarization.preserve_recent_turns, 5);
        assert_eq!(config.summarization.max_summary_tokens, 1024);
        assert_eq!(config.summarization.min_turns_before_summarize, 8);
    }

    #[test]
    fn test_summarization_config_absent_uses_defaults() {
        // When [summarization] is absent, serde(default) kicks in
        let toml_str = config_toml(&[]);
        let config: AppConfig = toml::from_str(&toml_str).unwrap();
        assert!(config.summarization.enabled);
        assert_eq!(config.summarization.preserve_recent_turns, 5);
    }

    #[test]
    fn test_conversation_summary_serde_roundtrip() {
        let summary = ConversationSummary {
            session_id: SessionId::from("test-session"),
            text: "User asked about Rust. We discussed ownership and borrowing.".into(),
            summarized_through_turn: 4,
            original_token_estimate: 2500,
            generated_at: Utc::now(),
        };

        let json = serde_json::to_string(&summary).unwrap();
        let back: ConversationSummary = serde_json::from_str(&json).unwrap();
        assert_eq!(back, summary);
    }
}
