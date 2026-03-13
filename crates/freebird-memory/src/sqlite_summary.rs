//! Persistent storage for conversation summaries.
//!
//! One summary per session — re-summarization replaces the previous summary.
//! Shares the encrypted `SqliteDb` connection with other memory backends.

use std::sync::Arc;

use async_trait::async_trait;
use chrono::DateTime;
use freebird_traits::id::SessionId;
use freebird_traits::memory::MemoryError;
use freebird_traits::summary::{ConversationSummary, SummarySink};

use crate::sqlite::SqliteDb;

/// Persistent storage for conversation summaries.
///
/// One summary per session. Re-summarization replaces the previous summary.
pub struct SummaryStore {
    db: Arc<SqliteDb>,
}

impl SummaryStore {
    /// Create a new `SummaryStore` sharing the given database connection.
    #[must_use]
    pub const fn new(db: Arc<SqliteDb>) -> Self {
        Self { db }
    }

    /// Load the summary for a session, if one exists.
    ///
    /// # Errors
    ///
    /// Returns `MemoryError::Io` on database errors.
    pub async fn load(
        &self,
        session_id: &SessionId,
    ) -> Result<Option<ConversationSummary>, MemoryError> {
        let sid = session_id.as_str().to_owned();

        let result = self.db.conn().await.query_row(
            "SELECT summary_text, summarized_through_turn, original_token_estimate, generated_at
                 FROM conversation_summaries WHERE session_id = ?1",
            rusqlite::params![sid],
            |row| {
                let text: String = row.get(0)?;
                let through_turn: i64 = row.get(1)?;
                let token_est: i64 = row.get(2)?;
                let generated_at_str: String = row.get(3)?;
                Ok((text, through_turn, token_est, generated_at_str))
            },
        );

        match result {
            Ok((text, through_turn, token_est, generated_at_str)) => {
                let generated_at = DateTime::parse_from_rfc3339(&generated_at_str)
                    .map(|dt| dt.with_timezone(&chrono::Utc))
                    .map_err(|e| {
                        MemoryError::Io(std::io::Error::other(format!(
                            "invalid generated_at timestamp: {e}"
                        )))
                    })?;

                Ok(Some(ConversationSummary {
                    session_id: session_id.clone(),
                    text,
                    summarized_through_turn: usize::try_from(through_turn).unwrap_or(0),
                    original_token_estimate: usize::try_from(token_est).unwrap_or(0),
                    generated_at,
                }))
            }
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(to_io("load summary", &e)),
        }
    }

    /// Save (upsert) a summary for a session.
    ///
    /// # Errors
    ///
    /// Returns `MemoryError::Io` on database errors.
    pub async fn save(&self, summary: &ConversationSummary) -> Result<(), MemoryError> {
        self.db
            .conn()
            .await
            .execute(
                "INSERT OR REPLACE INTO conversation_summaries
                 (session_id, summary_text, summarized_through_turn, original_token_estimate, generated_at)
                 VALUES (?1, ?2, ?3, ?4, ?5)",
                rusqlite::params![
                    summary.session_id.as_str(),
                    summary.text,
                    i64::try_from(summary.summarized_through_turn).unwrap_or(i64::MAX),
                    i64::try_from(summary.original_token_estimate).unwrap_or(i64::MAX),
                    summary.generated_at.to_rfc3339(),
                ],
            )
            .map_err(|e| to_io("save summary", &e))?;

        Ok(())
    }

    /// Delete the summary for a session (e.g., on session delete).
    ///
    /// # Errors
    ///
    /// Returns `MemoryError::Io` on database errors.
    pub async fn delete(&self, session_id: &SessionId) -> Result<(), MemoryError> {
        self.db
            .conn()
            .await
            .execute(
                "DELETE FROM conversation_summaries WHERE session_id = ?1",
                rusqlite::params![session_id.as_str()],
            )
            .map_err(|e| to_io("delete summary", &e))?;

        Ok(())
    }
}

#[async_trait]
impl SummarySink for SummaryStore {
    async fn load(
        &self,
        session_id: &SessionId,
    ) -> Result<Option<ConversationSummary>, MemoryError> {
        Self::load(self, session_id).await
    }

    async fn save(&self, summary: &ConversationSummary) -> Result<(), MemoryError> {
        Self::save(self, summary).await
    }

    async fn delete(&self, session_id: &SessionId) -> Result<(), MemoryError> {
        Self::delete(self, session_id).await
    }
}

/// Convert a `rusqlite` error into a `MemoryError::Io`.
fn to_io(context: &str, e: &rusqlite::Error) -> MemoryError {
    MemoryError::Io(std::io::Error::other(format!("{context}: {e}")))
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::significant_drop_tightening
)]
mod tests {
    use super::*;
    use chrono::Utc;
    use secrecy::SecretString;

    fn test_db() -> (tempfile::TempDir, Arc<SqliteDb>) {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("test.db");
        let key = SecretString::from("a".repeat(64));
        let signing_key = ring::hmac::Key::new(ring::hmac::HMAC_SHA256, b"test-signing-key");
        let db = Arc::new(SqliteDb::open(&db_path, &key, signing_key).unwrap());
        (dir, db)
    }

    fn make_summary(session_id: &str) -> ConversationSummary {
        ConversationSummary {
            session_id: SessionId::from(session_id),
            text: "User asked about Rust ownership. We discussed borrowing.".into(),
            summarized_through_turn: 4,
            original_token_estimate: 2500,
            generated_at: Utc::now(),
        }
    }

    #[tokio::test]
    async fn test_summary_store_load_empty() {
        let (_dir, db) = test_db();
        let store = SummaryStore::new(db);

        let result = store.load(&SessionId::from("nonexistent")).await.unwrap();
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn test_summary_store_save_load_roundtrip() {
        let (_dir, db) = test_db();
        let store = SummaryStore::new(db);
        let summary = make_summary("sess-1");

        store.save(&summary).await.unwrap();
        let loaded = store
            .load(&SessionId::from("sess-1"))
            .await
            .unwrap()
            .unwrap();

        assert_eq!(loaded.session_id, summary.session_id);
        assert_eq!(loaded.text, summary.text);
        assert_eq!(
            loaded.summarized_through_turn,
            summary.summarized_through_turn
        );
        assert_eq!(
            loaded.original_token_estimate,
            summary.original_token_estimate
        );
        // Timestamp comparison: RFC3339 roundtrip loses sub-nanosecond precision
        assert_eq!(
            loaded.generated_at.timestamp(),
            summary.generated_at.timestamp()
        );
    }

    #[tokio::test]
    async fn test_summary_store_upsert() {
        let (_dir, db) = test_db();
        let store = SummaryStore::new(db);

        let mut summary = make_summary("sess-1");
        store.save(&summary).await.unwrap();

        // Update with new content
        summary.text = "Updated summary with more details.".into();
        summary.summarized_through_turn = 7;
        summary.original_token_estimate = 5000;
        store.save(&summary).await.unwrap();

        let loaded = store
            .load(&SessionId::from("sess-1"))
            .await
            .unwrap()
            .unwrap();
        assert_eq!(loaded.text, "Updated summary with more details.");
        assert_eq!(loaded.summarized_through_turn, 7);
        assert_eq!(loaded.original_token_estimate, 5000);
    }

    #[tokio::test]
    async fn test_summary_store_delete() {
        let (_dir, db) = test_db();
        let store = SummaryStore::new(db);
        let summary = make_summary("sess-1");

        store.save(&summary).await.unwrap();
        assert!(
            store
                .load(&SessionId::from("sess-1"))
                .await
                .unwrap()
                .is_some()
        );

        store.delete(&SessionId::from("sess-1")).await.unwrap();
        assert!(
            store
                .load(&SessionId::from("sess-1"))
                .await
                .unwrap()
                .is_none()
        );
    }

    #[tokio::test]
    async fn test_summary_store_delete_nonexistent() {
        let (_dir, db) = test_db();
        let store = SummaryStore::new(db);

        // Deleting a non-existent summary should not error
        store.delete(&SessionId::from("nonexistent")).await.unwrap();
    }
}
