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
    #[serde(default)]
    pub knowledge: KnowledgeConfig,
    #[serde(default)]
    pub summarization: SummarizationConfig,
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
    /// Maximum concurrent message-handling tasks (ASI08). Default: 8.
    /// When the limit is reached, new messages get an immediate error response.
    #[serde(default = "default_max_concurrent_tasks")]
    pub max_concurrent_tasks: usize,
    /// In-memory session manager limits.
    #[serde(default)]
    pub session: SessionConfig,
    /// Context window management — observation collapsing.
    #[serde(default)]
    pub context: ContextConfig,
}

/// In-memory session manager configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionConfig {
    /// Maximum number of concurrent in-memory sessions. Default: 100.
    /// When exceeded, the least-recently-used session is evicted.
    #[serde(default = "default_max_sessions")]
    pub max_sessions: usize,
    /// Session time-to-live in seconds. Default: 86400 (24 hours).
    /// Sessions idle longer than this are evicted on the next operation.
    #[serde(default = "default_session_ttl_secs")]
    pub session_ttl_secs: u64,
}

impl Default for SessionConfig {
    fn default() -> Self {
        Self {
            max_sessions: default_max_sessions(),
            session_ttl_secs: default_session_ttl_secs(),
        }
    }
}

const fn default_max_concurrent_tasks() -> usize {
    8
}

const fn default_max_sessions() -> usize {
    100
}

const fn default_session_ttl_secs() -> u64 {
    86_400 // 24 hours
}

/// Context window management configuration.
///
/// Controls observation collapsing — compressing tool outputs from older
/// turns into compact one-line summaries to prevent context window
/// exhaustion in long sessions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextConfig {
    /// Number of recent turns whose tool outputs are kept intact.
    /// Turns older than this threshold have their tool outputs replaced
    /// with compact summaries. Default: 5.
    #[serde(default = "default_collapse_after_turns")]
    pub collapse_after_turns: usize,
    /// Whether observation collapsing is enabled. Default: true.
    #[serde(default = "default_collapse_tool_outputs")]
    pub collapse_tool_outputs: bool,
}

impl Default for ContextConfig {
    fn default() -> Self {
        Self {
            collapse_after_turns: default_collapse_after_turns(),
            collapse_tool_outputs: default_collapse_tool_outputs(),
        }
    }
}

const fn default_collapse_after_turns() -> usize {
    5
}

const fn default_collapse_tool_outputs() -> bool {
    true
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
    /// Commands the shell tool is permitted to execute. Empty = deny all.
    #[serde(default = "default_allowed_shell_commands")]
    pub allowed_shell_commands: Vec<String>,
    /// Maximum stdout+stderr bytes the shell tool returns. Output beyond
    /// this limit is truncated with a `[output truncated]` marker.
    #[serde(default = "default_max_shell_output_bytes")]
    pub max_shell_output_bytes: usize,
    /// Edit tool configuration (diff preview, context lines).
    #[serde(default)]
    pub edit: EditConfig,
    /// Timeout in seconds for git subprocess calls used by `workspace_status`
    /// tool. Default: 5.
    #[serde(default = "default_git_timeout_secs")]
    pub git_timeout_secs: u64,
}

const fn default_git_timeout_secs() -> u64 {
    5
}

/// Action to take when an edit exceeds the large-edit threshold.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LargeEditAction {
    /// Apply the edit but include a warning in the output.
    Warn,
    /// Reject the edit entirely — the file is not modified.
    Block,
    /// Reject the edit with guidance to break it into smaller edits.
    Consent,
}

/// Configuration for the search/replace edit tool.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EditConfig {
    /// Whether to append a compact diff preview to the edit tool output.
    #[serde(default = "default_diff_preview")]
    pub diff_preview: bool,
    /// Number of unchanged context lines shown before/after the change.
    #[serde(default = "default_diff_context_lines")]
    pub diff_context_lines: usize,
    /// Whether to run tree-sitter syntax validation after each edit.
    /// Invalid edits are rejected and the original file is preserved.
    #[serde(default = "default_syntax_validation")]
    pub syntax_validation: bool,
    /// Flag edits that change more than this fraction of the file (0.0–1.0).
    #[serde(default = "default_large_edit_threshold")]
    pub large_edit_threshold: f64,
    /// Action when the large-edit threshold is exceeded.
    #[serde(default = "default_large_edit_action")]
    pub large_edit_action: LargeEditAction,
}

impl Default for EditConfig {
    fn default() -> Self {
        Self {
            diff_preview: default_diff_preview(),
            diff_context_lines: default_diff_context_lines(),
            syntax_validation: default_syntax_validation(),
            large_edit_threshold: default_large_edit_threshold(),
            large_edit_action: default_large_edit_action(),
        }
    }
}

const fn default_diff_preview() -> bool {
    true
}

const fn default_diff_context_lines() -> usize {
    3
}

const fn default_syntax_validation() -> bool {
    true
}

const fn default_large_edit_threshold() -> f64 {
    0.5
}

const fn default_large_edit_action() -> LargeEditAction {
    LargeEditAction::Warn
}

fn default_allowed_shell_commands() -> Vec<String> {
    // Only read-only, single-file commands. Notably excluded:
    // - `git`: can make network connections, bypassing EgressPolicy (CLAUDE.md §12)
    // - `find`: `-delete` flag enables filesystem destruction outside sandbox
    // Admins can add these via config if they accept the risks.
    ["ls", "cat", "grep", "head", "tail", "wc"]
        .iter()
        .map(ToString::to_string)
        .collect()
}

const fn default_max_shell_output_bytes() -> usize {
    1_048_576 // 1 MiB
}

/// Memory backend configuration (`SQLCipher` encrypted `SQLite`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryConfig {
    /// Path to the `SQLite` database file. Default: `~/.freebird/freebird.db`.
    pub db_path: Option<PathBuf>,
    /// Path to the encryption keyfile. If not set, falls back to
    /// `FREEBIRD_DB_KEY` env var or interactive prompt.
    pub keyfile_path: Option<PathBuf>,
    /// PBKDF2 iteration count for key derivation. Default: 100,000.
    #[serde(default = "default_pbkdf2_iterations")]
    pub pbkdf2_iterations: u32,
    /// Verify HMAC chain integrity when loading conversations. Default: true.
    ///
    /// When enabled, every `Memory::load()` call verifies the per-session HMAC
    /// chain before replaying events. Tampered or corrupted events produce
    /// `MemoryError::IntegrityViolation`. Disable only for recovery/debugging.
    #[serde(default = "default_verify_on_load")]
    pub verify_on_load: bool,
}

const fn default_pbkdf2_iterations() -> u32 {
    100_000
}

const fn default_verify_on_load() -> bool {
    true
}

/// Knowledge retrieval behavior configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeConfig {
    /// Whether to auto-retrieve knowledge on every user message.
    #[serde(default = "default_auto_retrieve")]
    pub auto_retrieve: bool,
    /// Max entries injected into context per message.
    #[serde(default = "default_max_context_entries")]
    pub max_context_entries: usize,
    /// BM25 rank threshold. Entries with rank worse (higher) than this are excluded.
    /// FTS5 BM25: lower (more negative) = more relevant. Default: -0.5.
    #[serde(default = "default_relevance_threshold")]
    pub relevance_threshold: f64,
    /// Max approximate tokens for injected knowledge context.
    #[serde(default = "default_max_context_tokens")]
    pub max_context_tokens: usize,
}

impl Default for KnowledgeConfig {
    fn default() -> Self {
        Self {
            auto_retrieve: default_auto_retrieve(),
            max_context_entries: default_max_context_entries(),
            relevance_threshold: default_relevance_threshold(),
            max_context_tokens: default_max_context_tokens(),
        }
    }
}

const fn default_auto_retrieve() -> bool {
    true
}

const fn default_max_context_entries() -> usize {
    5
}

const fn default_relevance_threshold() -> f64 {
    -0.5
}

const fn default_max_context_tokens() -> usize {
    2000
}

// ---------------------------------------------------------------------------
// Conversation Summarization
// ---------------------------------------------------------------------------

/// Re-export from `freebird-traits` where the canonical definition lives
/// (required by the `SummarySink` trait).
pub use freebird_traits::summary::ConversationSummary;

/// Conversation summarization configuration.
///
/// Controls when and how the agent compresses older conversation turns
/// into summaries to stay within context window limits.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SummarizationConfig {
    /// Whether conversation summarization is enabled.
    #[serde(default = "default_summarization_enabled")]
    pub enabled: bool,
    /// Trigger threshold as a fraction (0.0–1.0) of the model's
    /// `max_context_tokens`. When estimated token count of the message
    /// history exceeds this fraction, summarization triggers.
    /// Default: 0.75 (75%).
    #[serde(default = "default_trigger_threshold")]
    pub trigger_threshold: f64,
    /// Number of recent turns to preserve intact (never summarized).
    /// These provide the model with immediate working context.
    /// Default: 5.
    #[serde(default = "default_preserve_recent_turns")]
    pub preserve_recent_turns: usize,
    /// Maximum tokens for the generated summary. Passed as `max_tokens`
    /// to the summarization provider request.
    /// Default: 1024.
    #[serde(default = "default_max_summary_tokens")]
    pub max_summary_tokens: u32,
    /// Minimum total turns in a conversation before summarization can
    /// trigger. Prevents thrashing on short conversations.
    /// Default: 8.
    #[serde(default = "default_min_turns_before_summarize")]
    pub min_turns_before_summarize: usize,
}

impl Default for SummarizationConfig {
    fn default() -> Self {
        Self {
            enabled: default_summarization_enabled(),
            trigger_threshold: default_trigger_threshold(),
            preserve_recent_turns: default_preserve_recent_turns(),
            max_summary_tokens: default_max_summary_tokens(),
            min_turns_before_summarize: default_min_turns_before_summarize(),
        }
    }
}

const fn default_summarization_enabled() -> bool {
    true
}

const fn default_trigger_threshold() -> f64 {
    0.75
}

const fn default_preserve_recent_turns() -> usize {
    5
}

const fn default_max_summary_tokens() -> u32 {
    1024
}

const fn default_min_turns_before_summarize() -> usize {
    8
}

/// Token and tool-round budget limits (CLAUDE.md §13 — ASI08).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BudgetConfig {
    /// Maximum total tokens (input + output) per session.
    pub max_tokens_per_session: u64,
    /// Maximum tokens (input + output) per single provider request.
    pub max_tokens_per_request: u64,
    /// Maximum tool rounds in a single agentic turn.
    pub max_tool_rounds_per_turn: u32,
    /// Maximum cost per session in microdollars (1 microdollar = $0.000001).
    /// Default: 5,000,000 = $5.00.
    #[serde(default = "default_max_cost_microdollars")]
    pub max_cost_microdollars: u64,
}

const fn default_max_cost_microdollars() -> u64 {
    5_000_000 // $5.00
}

impl Default for BudgetConfig {
    fn default() -> Self {
        Self {
            max_tokens_per_session: 500_000,
            max_tokens_per_request: 32_768,
            max_tool_rounds_per_turn: 10,
            max_cost_microdollars: default_max_cost_microdollars(),
        }
    }
}

/// Security policy configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityConfig {
    pub max_tool_calls_per_turn: usize,
    /// Minimum risk level that requires explicit human consent before tool
    /// execution. Tools at this level or above trigger the consent gate.
    pub require_consent_above: RiskLevel,
    /// How long (in seconds) to wait for a user to respond to a consent
    /// request before auto-denying. Default: 60.
    #[serde(default = "default_consent_timeout_secs")]
    pub consent_timeout_secs: u64,
    /// Maximum number of simultaneous pending consent requests.
    /// Prevents LLM flooding attacks. Default: 5.
    #[serde(default = "default_max_pending_consent")]
    pub max_pending_consent_requests: usize,
    /// Default session TTL in hours. Sessions expire after this duration
    /// unless a more specific TTL is provided. Default: 24 hours.
    #[serde(default = "default_session_ttl_hours")]
    pub default_session_ttl_hours: u64,
    /// Network egress policy. Controls which hosts the agent can contact.
    #[serde(default)]
    pub egress: EgressConfig,
    /// Secret guard policy. Controls detection and action for tool
    /// invocations that may access or expose secrets.
    #[serde(default)]
    pub secret_guard: SecretGuardConfig,
    /// Token and tool-round budget limits.
    #[serde(default)]
    pub budgets: BudgetConfig,
    /// Injection detection response configuration.
    #[serde(default)]
    pub injection: InjectionConfig,
}

/// Network egress allowlist configuration (CLAUDE.md §12 — ASI01).
///
/// Default is deny-all with only provider API hosts permitted.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EgressConfig {
    /// Hosts the agent is allowed to contact (e.g., `["api.anthropic.com"]`).
    #[serde(default = "default_egress_allowed_hosts")]
    pub allowed_hosts: Vec<String>,
    /// Ports the agent is allowed to contact. Default: `[443]`.
    #[serde(default = "default_egress_allowed_ports")]
    pub allowed_ports: Vec<u16>,
    /// Maximum response body bytes the network tool will read. Default: 1 MiB.
    #[serde(default = "default_egress_max_response_bytes")]
    pub max_response_bytes: usize,
    /// Per-request timeout in seconds for outbound HTTP. Default: 30.
    #[serde(default = "default_egress_request_timeout_secs")]
    pub request_timeout_secs: u64,
    /// Maximum requests per 60-second sliding window. Default: 60.
    /// Prevents rapid-fire exfiltration bursts (CLAUDE.md §12 — ASI01).
    #[serde(default = "default_egress_rate_limit_per_minute")]
    pub rate_limit_per_minute: u32,
    /// Maximum request body bytes the network tool will send. Default: 1 MiB.
    /// Prevents large data exfiltration to allowlisted hosts (CLAUDE.md §12).
    #[serde(default = "default_egress_max_request_body_bytes")]
    pub max_request_body_bytes: usize,
}

impl Default for EgressConfig {
    fn default() -> Self {
        Self {
            allowed_hosts: default_egress_allowed_hosts(),
            allowed_ports: default_egress_allowed_ports(),
            max_response_bytes: default_egress_max_response_bytes(),
            request_timeout_secs: default_egress_request_timeout_secs(),
            rate_limit_per_minute: default_egress_rate_limit_per_minute(),
            max_request_body_bytes: default_egress_max_request_body_bytes(),
        }
    }
}

fn default_egress_allowed_hosts() -> Vec<String> {
    vec!["api.anthropic.com".into(), "api.openai.com".into()]
}

fn default_egress_allowed_ports() -> Vec<u16> {
    vec![443]
}

const fn default_egress_max_response_bytes() -> usize {
    102_400 // 100 KiB — ~25k tokens, fits comfortably in LLM context
}

const fn default_egress_request_timeout_secs() -> u64 {
    30
}

const fn default_egress_rate_limit_per_minute() -> u32 {
    60
}

const fn default_egress_max_request_body_bytes() -> usize {
    1_048_576 // 1 MiB — prevents data exfiltration
}

/// What action the secret guard takes when a sensitive input is detected.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SecretGuardAction {
    /// Escalate to consent gate with `RiskLevel::Critical` — the user must
    /// explicitly approve the access.
    Consent,
    /// Deny outright without prompting. Use in headless/automated deployments.
    Block,
}

/// Secret guard configuration — detects and gates tool invocations that
/// access sensitive files, run secret-revealing commands, or produce output
/// containing credentials.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecretGuardConfig {
    /// Whether the secret guard is active. Default: `true`.
    #[serde(default = "default_secret_guard_enabled")]
    pub enabled: bool,
    /// Action on detection. Default: `consent`.
    #[serde(default = "default_secret_guard_action")]
    pub action: SecretGuardAction,
    /// Whether to redact detected secrets in tool output before returning
    /// to the LLM context. Default: `true`.
    #[serde(default = "default_secret_guard_redact_output")]
    pub redact_output: bool,
    /// Additional file patterns to treat as sensitive (glob syntax).
    /// Merged with built-in patterns.
    #[serde(default)]
    pub extra_sensitive_file_patterns: Vec<String>,
    /// Additional shell command patterns to treat as sensitive (regex syntax).
    /// Merged with built-in patterns.
    #[serde(default)]
    pub extra_sensitive_command_patterns: Vec<String>,
}

impl Default for SecretGuardConfig {
    fn default() -> Self {
        Self {
            enabled: default_secret_guard_enabled(),
            action: default_secret_guard_action(),
            redact_output: default_secret_guard_redact_output(),
            extra_sensitive_file_patterns: Vec::new(),
            extra_sensitive_command_patterns: Vec::new(),
        }
    }
}

const fn default_secret_guard_enabled() -> bool {
    true
}

const fn default_secret_guard_action() -> SecretGuardAction {
    SecretGuardAction::Consent
}

const fn default_secret_guard_redact_output() -> bool {
    true
}

/// How injection detection responds when a pattern is found.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InjectionResponse {
    /// Block the content outright — no user choice.
    Block,
    /// Warn the user and ask whether to proceed (default for input/tool output).
    Prompt,
    /// Allow the content through without warning.
    Allow,
}

/// Injection detection response configuration.
///
/// Controls how each injection detection layer responds when a pattern
/// is found. Model output and loaded context are always blocked
/// (non-configurable) because they represent compromised trust boundaries.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InjectionConfig {
    /// Response for injection patterns in user input. Default: `prompt`.
    #[serde(default = "default_injection_input_response")]
    pub input_response: InjectionResponse,
    /// Response for injection patterns in tool output. Default: `prompt`.
    #[serde(default = "default_injection_tool_output_response")]
    pub tool_output_response: InjectionResponse,
    /// Timeout in seconds for security prompts. Default: 60.
    #[serde(default = "default_injection_prompt_timeout_secs")]
    pub prompt_timeout_secs: u64,
}

impl Default for InjectionConfig {
    fn default() -> Self {
        Self {
            input_response: default_injection_input_response(),
            tool_output_response: default_injection_tool_output_response(),
            prompt_timeout_secs: default_injection_prompt_timeout_secs(),
        }
    }
}

const fn default_injection_input_response() -> InjectionResponse {
    InjectionResponse::Prompt
}

const fn default_injection_tool_output_response() -> InjectionResponse {
    InjectionResponse::Prompt
}

const fn default_injection_prompt_timeout_secs() -> u64 {
    60
}

const fn default_session_ttl_hours() -> u64 {
    24
}

const fn default_consent_timeout_secs() -> u64 {
    60
}

const fn default_max_pending_consent() -> usize {
    5
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
        let toml_str = include_str!("../../../config/default.toml");
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

    // ── Shell config tests ──────────────────────────────────────────

    #[test]
    fn test_shell_config_from_default_toml() {
        let toml_str = include_str!("../../../config/default.toml");
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

    // ── Edit config tests ────────────────────────────────────────

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

    // ── Consent config tests ───────────────────────────────────────

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
        let toml_str = include_str!("../../../config/default.toml");
        let config: AppConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(config.security.consent_timeout_secs, 60);
        assert_eq!(config.security.max_pending_consent_requests, 5);
    }

    // ── Egress config tests ───────────────────────────────────────

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
        let toml_str = include_str!("../../../config/default.toml");
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

    // ── Knowledge config tests ─────────────────────────────────

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
        let toml_str = include_str!("../../../config/default.toml");
        let config: AppConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(
            config.memory.db_path.as_ref().map(|p| p.to_str().unwrap()),
            Some("~/.freebird/freebird.db")
        );
    }

    #[test]
    fn test_knowledge_config_from_default_toml() {
        let toml_str = include_str!("../../../config/default.toml");
        let config: AppConfig = toml::from_str(toml_str).unwrap();
        assert!(config.knowledge.auto_retrieve);
        assert_eq!(config.knowledge.max_context_entries, 5);
        assert_eq!(config.knowledge.max_context_tokens, 2000);
    }

    // ── Secret guard config tests ─────────────────────────────────

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
        let toml_str = include_str!("../../../config/default.toml");
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

    // ── LargeEditAction tests ───────────────────────────────────

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

    // ── Summarization config tests ──────────────────────────────

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
        let toml_str = include_str!("../../../config/default.toml");
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
