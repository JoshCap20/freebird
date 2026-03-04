//! In-memory conversation store.
//!
//! Thread-safe via `tokio::sync::RwLock`. All data is lost on drop.
//! This backend is ideal for:
//! - Unit and integration testing (no filesystem side effects)
//! - Development/prototyping
//! - Short-lived sessions that don't need persistence

use std::collections::HashMap;

use async_trait::async_trait;
use tokio::sync::RwLock;

use freebird_traits::id::SessionId;
use freebird_traits::memory::{Conversation, Memory, MemoryError, SessionSummary};
use freebird_traits::provider::{ContentBlock, Message};

/// In-memory conversation store.
///
/// Thread-safe via `tokio::sync::RwLock`. All data is lost on drop.
pub struct InMemoryMemory {
    store: RwLock<HashMap<SessionId, Conversation>>,
}

impl InMemoryMemory {
    /// Create an empty in-memory store.
    #[must_use]
    pub fn new() -> Self {
        Self {
            store: RwLock::new(HashMap::new()),
        }
    }
}

impl Default for InMemoryMemory {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Memory for InMemoryMemory {
    async fn load(&self, session_id: &SessionId) -> Result<Option<Conversation>, MemoryError> {
        let store = self.store.read().await;
        Ok(store.get(session_id).cloned())
    }

    async fn save(&self, conversation: &Conversation) -> Result<(), MemoryError> {
        self.store
            .write()
            .await
            .insert(conversation.session_id.clone(), conversation.clone());
        Ok(())
    }

    async fn list_sessions(&self, limit: usize) -> Result<Vec<SessionSummary>, MemoryError> {
        let mut summaries: Vec<SessionSummary> = self
            .store
            .read()
            .await
            .values()
            .map(conversation_to_summary)
            .collect();
        summaries.sort_by(|a, b| b.updated_at.cmp(&a.updated_at));
        summaries.truncate(limit);
        Ok(summaries)
    }

    async fn delete(&self, session_id: &SessionId) -> Result<(), MemoryError> {
        self.store
            .write()
            .await
            .remove(session_id)
            .ok_or_else(|| MemoryError::NotFound {
                session_id: session_id.clone(),
            })?;
        Ok(())
    }

    async fn search(&self, query: &str, limit: usize) -> Result<Vec<SessionSummary>, MemoryError> {
        if query.is_empty() {
            return Ok(vec![]);
        }

        let query_lower = query.to_lowercase();
        let mut results: Vec<SessionSummary> = self
            .store
            .read()
            .await
            .values()
            .filter(|conv| conversation_contains(conv, &query_lower))
            .map(conversation_to_summary)
            .collect();
        results.sort_by(|a, b| b.updated_at.cmp(&a.updated_at));
        results.truncate(limit);
        Ok(results)
    }
}

/// Build a `SessionSummary` from a conversation.
///
/// The preview is the first 100 characters of the first text content block
/// in the first user message. Non-text blocks produce an empty preview.
fn conversation_to_summary(conv: &Conversation) -> SessionSummary {
    let preview = conv
        .turns
        .first()
        .and_then(|t| t.user_message.content.first())
        .map(|block| match block {
            ContentBlock::Text { text } => text.chars().take(100).collect(),
            _ => String::new(),
        })
        .unwrap_or_default();

    SessionSummary {
        session_id: conv.session_id.clone(),
        created_at: conv.created_at,
        updated_at: conv.updated_at,
        turn_count: conv.turns.len(),
        model_id: conv.model_id.clone(),
        preview,
    }
}

/// Check if a conversation contains the query string in any text content block.
///
/// Searches both user messages and assistant responses. Does NOT search
/// tool invocation input/output (JSON, not useful for text search).
fn conversation_contains(conv: &Conversation, query_lower: &str) -> bool {
    conv.turns.iter().any(|turn| {
        message_contains(&turn.user_message, query_lower)
            || turn
                .assistant_response
                .as_ref()
                .is_some_and(|msg| message_contains(msg, query_lower))
    })
}

/// Check if any text content block in a message contains the query.
fn message_contains(msg: &Message, query_lower: &str) -> bool {
    msg.content.iter().any(|block| {
        matches!(block, ContentBlock::Text { text } if text.to_lowercase().contains(query_lower))
    })
}
