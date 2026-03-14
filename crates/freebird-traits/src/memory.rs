//! Memory trait — abstracts over conversation persistence backends.

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::id::{ModelId, ProviderId, SessionId};
use crate::provider::Message;
use crate::tool::ToolOutcome;

/// A complete conversation turn: user message + assistant responses + tool calls.
///
/// `assistant_messages` stores all assistant messages in order: intermediate
/// `ToolUse` responses followed by the final text response. Empty when the
/// turn is still in progress.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Turn {
    pub user_message: Message,
    pub assistant_messages: Vec<Message>,
    pub tool_invocations: Vec<ToolInvocation>,
    pub started_at: DateTime<Utc>,
    pub completed_at: Option<DateTime<Utc>>,
}

/// Record of a single tool invocation within a turn.
#[expect(
    clippy::derive_partial_eq_without_eq,
    reason = "serde_json::Value does not impl Eq"
)]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolInvocation {
    pub tool_use_id: String,
    pub tool_name: String,
    pub input: serde_json::Value,
    pub output: Option<String>,
    pub outcome: ToolOutcome,
    pub duration_ms: Option<u64>,
}

/// A complete conversation (ordered list of turns + metadata).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Conversation {
    pub session_id: SessionId,
    pub system_prompt: Option<String>,
    pub turns: Vec<Turn>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub model_id: ModelId,
    pub provider_id: ProviderId,
}

/// Summary of a stored session for listing/search results.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionSummary {
    pub session_id: SessionId,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub turn_count: usize,
    pub model_id: ModelId,
    pub preview: String,
}

/// The core memory trait for persisting conversations.
#[async_trait]
pub trait Memory: Send + Sync + 'static {
    async fn load(&self, session_id: &SessionId) -> Result<Option<Conversation>, MemoryError>;
    async fn save(&self, conversation: &Conversation) -> Result<(), MemoryError>;
    async fn list_sessions(&self, limit: usize) -> Result<Vec<SessionSummary>, MemoryError>;
    async fn delete(&self, session_id: &SessionId) -> Result<(), MemoryError>;
    async fn search(&self, query: &str, limit: usize) -> Result<Vec<SessionSummary>, MemoryError>;
}

/// Memory-specific errors.
#[derive(Debug, thiserror::Error)]
pub enum MemoryError {
    #[error("session `{session_id}` not found")]
    NotFound { session_id: SessionId },

    #[error("storage I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("serialization error: {0}")]
    Serialization(String),

    #[error("storage is read-only")]
    ReadOnly,

    #[error("integrity violation: {reason}")]
    IntegrityViolation { reason: String },
}
