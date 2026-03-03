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
    pub default_model: String,
    pub max_turns_per_session: usize,
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
