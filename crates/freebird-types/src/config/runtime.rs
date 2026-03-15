//! Runtime, daemon, budget, tools, and edit configuration structs.

use std::net::{IpAddr, Ipv4Addr};
use std::path::PathBuf;

use freebird_traits::id::{ModelId, ProviderId};
use serde::{Deserialize, Serialize};

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

impl Default for ToolsConfig {
    fn default() -> Self {
        Self {
            sandbox_root: std::env::temp_dir(),
            default_timeout_secs: 30,
            allowed_directories: Vec::new(),
            allowed_shell_commands: default_allowed_shell_commands(),
            max_shell_output_bytes: default_max_shell_output_bytes(),
            edit: EditConfig::default(),
            git_timeout_secs: default_git_timeout_secs(),
        }
    }
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

/// Default TCP port for the `FreeBird` daemon.
///
/// Chosen to avoid conflicts with common services (HTTP 8080, Node 3000, etc.)
/// and low-numbered well-known ports. Configurable via `[daemon] port`.
const DEFAULT_DAEMON_PORT: u16 = 7531;

impl Default for DaemonConfig {
    fn default() -> Self {
        Self {
            host: IpAddr::V4(Ipv4Addr::LOCALHOST),
            port: DEFAULT_DAEMON_PORT,
        }
    }
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
