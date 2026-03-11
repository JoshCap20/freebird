//! File-based conversation storage.
//!
//! Each conversation is stored as a JSON file named `{session_id}.json`.
//! Writes are atomic: data is written to a `.json.tmp` file first, then
//! renamed to `.json`. This prevents corruption if the process crashes mid-write.

use std::path::PathBuf;

use async_trait::async_trait;

use freebird_traits::id::SessionId;
use freebird_traits::memory::{Conversation, Memory, MemoryError, SessionSummary};

use crate::helpers::{conversation_contains, conversation_to_summary};

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

    /// Iterate all conversation files, apply a filter, sort by recency, and truncate.
    ///
    /// Shared implementation for `list_sessions` (filter = `|_| true`) and
    /// `search` (filter = content match). Skips corrupt or unreadable files
    /// with a warning log rather than failing the entire scan.
    async fn scan_conversations(
        &self,
        filter: impl Fn(&Conversation) -> bool,
        limit: usize,
    ) -> Result<Vec<SessionSummary>, MemoryError> {
        let mut summaries = Vec::new();
        let mut entries = tokio::fs::read_dir(&self.base_dir).await?;

        while let Some(entry) = entries.next_entry().await? {
            let path = entry.path();
            if path.extension().is_none_or(|e| e != "json") {
                continue;
            }
            match tokio::fs::read_to_string(&path).await {
                Ok(data) => match serde_json::from_str::<Conversation>(&data) {
                    Ok(conv) => {
                        if filter(&conv) {
                            summaries.push(conversation_to_summary(&conv));
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

        summaries.sort_unstable_by_key(|s| std::cmp::Reverse(s.updated_at));
        summaries.truncate(limit);
        Ok(summaries)
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
        self.scan_conversations(|_| true, limit).await
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
        self.scan_conversations(|conv| conversation_contains(conv, &query_lower), limit)
            .await
    }
}
