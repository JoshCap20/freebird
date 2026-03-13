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

use crate::helpers::{conversation_contains, conversation_to_summary};
use freebird_traits::id::SessionId;
use freebird_traits::memory::{Conversation, Memory, MemoryError, SessionSummary};

/// In-memory conversation store.
///
/// Thread-safe via `tokio::sync::RwLock`. All data is lost on drop.
pub struct InMemoryMemory {
    store: RwLock<HashMap<SessionId, Conversation>>,
}

impl std::fmt::Debug for InMemoryMemory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InMemoryMemory").finish_non_exhaustive()
    }
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
        summaries.sort_unstable_by_key(|s| std::cmp::Reverse(s.updated_at));
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
        results.sort_unstable_by_key(|s| std::cmp::Reverse(s.updated_at));
        results.truncate(limit);
        Ok(results)
    }
}
