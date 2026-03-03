//! Memory trait — abstracts over conversation persistence backends.

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// The core memory trait for persisting conversations.
#[async_trait]
pub trait Memory: Send + Sync + 'static {
    /// Load a conversation by session ID. Returns `None` if not found.
    async fn load(&self, session_id: &str) -> Result<Option<serde_json::Value>, MemoryError>;

    /// Save/update a conversation.
    async fn save(&self, session_id: &str, conversation: &serde_json::Value) -> Result<(), MemoryError>;

    /// List all session IDs, most recent first.
    async fn list_sessions(&self, limit: usize) -> Result<Vec<SessionSummary>, MemoryError>;

    /// Delete a conversation by session ID.
    async fn delete(&self, session_id: &str) -> Result<(), MemoryError>;

    /// Search conversations by content.
    async fn search(&self, query: &str, limit: usize) -> Result<Vec<SessionSummary>, MemoryError>;
}

/// Summary of a stored session for listing/search results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionSummary {
    pub session_id: String,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub turn_count: usize,
    pub model_id: String,
    /// First ~100 chars of the first user message, for display.
    pub preview: String,
}

/// Memory-specific errors.
#[derive(Debug, thiserror::Error)]
pub enum MemoryError {
    #[error("session `{session_id}` not found")]
    NotFound { session_id: String },

    #[error("storage I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("serialization error: {0}")]
    Serialization(String),

    #[error("storage is read-only")]
    ReadOnly,
}
