//! File-based conversation storage.
//!
//! Each conversation is stored as a JSON file named `{session_id}.json`.
//! Writes are atomic: data is written to a `.json.tmp` file first, then
//! renamed to `.json`. This prevents corruption if the process crashes mid-write.

use std::path::PathBuf;

use async_trait::async_trait;

use freebird_traits::id::SessionId;
use freebird_traits::memory::{Conversation, Memory, MemoryError, SessionSummary};
use freebird_traits::provider::{ContentBlock, Message};

/// File-based conversation store.
///
/// Each conversation is stored as a JSON file named `{session_id}.json`.
/// Writes are atomic: data is written to a `.json.tmp` file first, then
/// renamed to `.json`. This prevents corruption if the process crashes mid-write.
///
/// No internal locking is needed — atomic rename provides consistency
/// for concurrent access from the tokio runtime.
pub struct FileMemory {
    base_dir: PathBuf,
}

impl FileMemory {
    /// Create a new file-based memory backend.
    ///
    /// Creates `base_dir` if it doesn't exist. Canonicalizes the path
    /// to ensure consistent absolute path comparisons. Cleans up any
    /// orphaned `.json.tmp` files from prior crashes.
    ///
    /// # Errors
    ///
    /// Returns `MemoryError::Io` if the directory cannot be created or
    /// the path cannot be canonicalized.
    pub fn new(base_dir: impl Into<PathBuf>) -> Result<Self, MemoryError> {
        let base_dir = base_dir.into();
        std::fs::create_dir_all(&base_dir)?;
        let base_dir = base_dir.canonicalize()?;
        let mem = Self { base_dir };
        mem.cleanup_orphaned_tmp_files();
        Ok(mem)
    }

    /// Build the file path for a session, with path traversal defense.
    ///
    /// Validates the session ID contains only filesystem-safe characters
    /// (`[a-zA-Z0-9_-]`). Returns `MemoryError::Io` with `InvalidInput`
    /// if the ID contains path separators, dots, null bytes, or other
    /// unsafe characters.
    fn session_path(&self, session_id: &SessionId) -> Result<PathBuf, MemoryError> {
        let id_str = session_id.as_str();
        if id_str.is_empty()
            || !id_str
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
        {
            return Err(MemoryError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "session ID contains characters unsafe for filesystem storage",
            )));
        }
        Ok(self.base_dir.join(format!("{id_str}.json")))
    }

    /// Remove orphaned `.json.tmp` files left from prior crashes.
    fn cleanup_orphaned_tmp_files(&self) {
        if let Ok(entries) = std::fs::read_dir(&self.base_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().is_some_and(|e| e == "tmp") {
                    let _ = std::fs::remove_file(&path);
                    tracing::debug!(?path, "cleaned up orphaned tmp file");
                }
            }
        }
    }
}

#[async_trait]
impl Memory for FileMemory {
    async fn load(&self, session_id: &SessionId) -> Result<Option<Conversation>, MemoryError> {
        let path = self.session_path(session_id)?;
        let data = match tokio::fs::read_to_string(&path).await {
            Ok(data) => data,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(e) => return Err(MemoryError::Io(e)),
        };
        let conversation: Conversation =
            serde_json::from_str(&data).map_err(|e| MemoryError::Serialization(e.to_string()))?;
        Ok(Some(conversation))
    }

    async fn save(&self, conversation: &Conversation) -> Result<(), MemoryError> {
        let path = self.session_path(&conversation.session_id)?;
        let tmp_path = path.with_extension("json.tmp");
        let data = serde_json::to_string_pretty(conversation)
            .map_err(|e| MemoryError::Serialization(e.to_string()))?;
        // Atomic write: tmp then rename (atomic on same filesystem).
        // If rename fails, clean up the tmp file to avoid accumulating orphans.
        tokio::fs::write(&tmp_path, data).await?;
        if let Err(e) = tokio::fs::rename(&tmp_path, &path).await {
            let _ = tokio::fs::remove_file(&tmp_path).await;
            return Err(MemoryError::Io(e));
        }
        Ok(())
    }

    async fn list_sessions(&self, limit: usize) -> Result<Vec<SessionSummary>, MemoryError> {
        let mut summaries = Vec::new();
        let mut entries = tokio::fs::read_dir(&self.base_dir).await?;

        while let Some(entry) = entries.next_entry().await? {
            let path = entry.path();
            if path.extension().is_none_or(|e| e != "json") {
                continue;
            }
            match tokio::fs::read_to_string(&path).await {
                Ok(data) => match serde_json::from_str::<Conversation>(&data) {
                    Ok(conv) => summaries.push(conversation_to_summary(&conv)),
                    Err(e) => {
                        tracing::warn!(?path, error = %e, "skipping corrupt conversation file");
                    }
                },
                Err(e) => {
                    tracing::warn!(?path, error = %e, "failed to read conversation file");
                }
            }
        }

        summaries.sort_unstable_by(|a, b| b.updated_at.cmp(&a.updated_at));
        summaries.truncate(limit);
        Ok(summaries)
    }

    async fn delete(&self, session_id: &SessionId) -> Result<(), MemoryError> {
        let path = self.session_path(session_id)?;
        match tokio::fs::remove_file(&path).await {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Err(MemoryError::NotFound {
                session_id: session_id.clone(),
            }),
            Err(e) => Err(MemoryError::Io(e)),
        }
    }

    async fn search(&self, query: &str, limit: usize) -> Result<Vec<SessionSummary>, MemoryError> {
        if query.is_empty() {
            return Ok(vec![]);
        }
        let query_lower = query.to_lowercase();
        let mut results = Vec::new();
        let mut entries = tokio::fs::read_dir(&self.base_dir).await?;

        while let Some(entry) = entries.next_entry().await? {
            let path = entry.path();
            if path.extension().is_none_or(|e| e != "json") {
                continue;
            }
            match tokio::fs::read_to_string(&path).await {
                Ok(data) => match serde_json::from_str::<Conversation>(&data) {
                    Ok(conv) => {
                        if conversation_contains(&conv, &query_lower) {
                            results.push(conversation_to_summary(&conv));
                        }
                    }
                    Err(e) => {
                        tracing::warn!(?path, error = %e, "skipping corrupt conversation file");
                    }
                },
                Err(e) => {
                    tracing::warn!(?path, error = %e, "failed to read conversation file");
                }
            }
        }

        results.sort_unstable_by(|a, b| b.updated_at.cmp(&a.updated_at));
        results.truncate(limit);
        Ok(results)
    }
}

/// Build a `SessionSummary` from a conversation.
///
/// Preview is the first 100 characters of the first text content block
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
/// Searches user messages and assistant responses. Does NOT search tool
/// invocation input/output (JSON, not useful for text search).
fn conversation_contains(conv: &Conversation, query_lower: &str) -> bool {
    conv.turns.iter().any(|turn| {
        message_contains(&turn.user_message, query_lower)
            || turn
                .assistant_response
                .as_ref()
                .is_some_and(|msg| message_contains(msg, query_lower))
    })
}

fn message_contains(msg: &Message, query_lower: &str) -> bool {
    msg.content.iter().any(|block| {
        matches!(block, ContentBlock::Text { text } if text.to_lowercase().contains(query_lower))
    })
}
