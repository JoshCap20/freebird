//! `SQLite`-backed implementation of the [`Memory`] trait.
//!
//! Stores conversations in a `SQLCipher`-encrypted database, sharing the
//! [`SqliteDb`] connection with the knowledge store. Turns are serialized
//! as a JSON blob in the `data` column for simplicity.

#![allow(clippy::significant_drop_tightening)]

use std::sync::Arc;

use async_trait::async_trait;
use freebird_traits::id::{ModelId, ProviderId, SessionId};
use freebird_traits::memory::{Conversation, Memory, MemoryError, SessionSummary, Turn};

use crate::helpers::conversation_to_summary;
use crate::sqlite::SqliteDb;

/// `SQLite`-backed memory backend.
///
/// Thread-safe via [`Arc<SqliteDb>`] (async `Mutex` inside).
pub struct SqliteMemory {
    db: Arc<SqliteDb>,
}

impl SqliteMemory {
    /// Create a new [`SqliteMemory`] sharing the given database connection.
    #[must_use]
    pub const fn new(db: Arc<SqliteDb>) -> Self {
        Self { db }
    }
}

/// Saturating conversion from `usize` to `i64` for SQL LIMIT parameters.
#[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
const fn limit_i64(limit: usize) -> i64 {
    if limit > i64::MAX as usize {
        i64::MAX
    } else {
        limit as i64
    }
}

#[async_trait]
impl Memory for SqliteMemory {
    async fn load(&self, session_id: &SessionId) -> Result<Option<Conversation>, MemoryError> {
        let sid = session_id.as_str().to_owned();
        let conn = self.db.conn().await;
        let mut stmt = conn
            .prepare(
                "SELECT session_id, system_prompt, model_id, provider_id, \
                 created_at, updated_at, data FROM conversations WHERE session_id = ?1",
            )
            .map_err(|e| io_err("prepare", &e))?;

        let conv_row = stmt
            .query_row(rusqlite::params![sid], |row| {
                Ok(ConversationRow {
                    session_id: row.get(0)?,
                    system_prompt: row.get(1)?,
                    model_id: row.get(2)?,
                    provider_id: row.get(3)?,
                    created_at: row.get(4)?,
                    updated_at: row.get(5)?,
                    data: row.get(6)?,
                })
            })
            .optional()
            .map_err(|e| io_err("query", &e))?;

        match conv_row {
            None => Ok(None),
            Some(r) => Ok(Some(row_to_conversation(r)?)),
        }
    }

    async fn save(&self, conversation: &Conversation) -> Result<(), MemoryError> {
        let data = serde_json::to_string(&conversation.turns)
            .map_err(|e| MemoryError::Serialization(e.to_string()))?;

        let sid = conversation.session_id.as_str().to_owned();
        let system_prompt = conversation.system_prompt.clone();
        let mid = conversation.model_id.as_str().to_owned();
        let pid = conversation.provider_id.as_str().to_owned();
        let cat = conversation.created_at.to_rfc3339();
        let uat = conversation.updated_at.to_rfc3339();

        let conn = self.db.conn().await;
        conn.execute(
            "INSERT OR REPLACE INTO conversations \
             (session_id, system_prompt, model_id, provider_id, created_at, updated_at, data) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            rusqlite::params![sid, system_prompt, mid, pid, cat, uat, data],
        )
        .map_err(|e| io_err("insert", &e))?;

        Ok(())
    }

    async fn list_sessions(&self, limit: usize) -> Result<Vec<SessionSummary>, MemoryError> {
        let conn = self.db.conn().await;
        let rows = query_conversations(
            &conn,
            "SELECT session_id, system_prompt, model_id, provider_id, \
             created_at, updated_at, data FROM conversations \
             ORDER BY updated_at DESC LIMIT ?1",
            rusqlite::params![limit_i64(limit)],
        )?;

        rows.into_iter()
            .map(|r| row_to_conversation(r).map(|c| conversation_to_summary(&c)))
            .collect()
    }

    async fn delete(&self, session_id: &SessionId) -> Result<(), MemoryError> {
        let sid = session_id.as_str().to_owned();
        let conn = self.db.conn().await;
        let affected = conn
            .execute(
                "DELETE FROM conversations WHERE session_id = ?1",
                rusqlite::params![sid],
            )
            .map_err(|e| io_err("delete", &e))?;

        if affected == 0 {
            return Err(MemoryError::NotFound {
                session_id: session_id.clone(),
            });
        }
        Ok(())
    }

    async fn search(&self, query: &str, limit: usize) -> Result<Vec<SessionSummary>, MemoryError> {
        if query.is_empty() {
            return Ok(Vec::new());
        }

        let pattern = format!("%{}%", query.to_lowercase());
        let conn = self.db.conn().await;
        let rows = query_conversations(
            &conn,
            "SELECT session_id, system_prompt, model_id, provider_id, \
             created_at, updated_at, data FROM conversations \
             WHERE LOWER(data) LIKE ?1 \
             ORDER BY updated_at DESC LIMIT ?2",
            rusqlite::params![pattern, limit_i64(limit)],
        )?;

        rows.into_iter()
            .map(|r| row_to_conversation(r).map(|c| conversation_to_summary(&c)))
            .collect()
    }
}

/// Intermediate row representation for conversations.
struct ConversationRow {
    session_id: String,
    system_prompt: Option<String>,
    model_id: String,
    provider_id: String,
    created_at: String,
    updated_at: String,
    data: String,
}

/// Execute a parameterized query and collect all rows as `ConversationRow`.
fn query_conversations(
    conn: &rusqlite::Connection,
    sql: &str,
    params: impl rusqlite::Params,
) -> Result<Vec<ConversationRow>, MemoryError> {
    let mut stmt = conn.prepare(sql).map_err(|e| io_err("prepare", &e))?;
    let rows = stmt
        .query_map(params, |row| {
            Ok(ConversationRow {
                session_id: row.get(0)?,
                system_prompt: row.get(1)?,
                model_id: row.get(2)?,
                provider_id: row.get(3)?,
                created_at: row.get(4)?,
                updated_at: row.get(5)?,
                data: row.get(6)?,
            })
        })
        .map_err(|e| io_err("query", &e))?;

    let mut result = Vec::new();
    for row_result in rows {
        result.push(row_result.map_err(|e| io_err("row", &e))?);
    }
    Ok(result)
}

/// Convert a database row into a [`Conversation`].
fn row_to_conversation(row: ConversationRow) -> Result<Conversation, MemoryError> {
    let turns: Vec<Turn> = serde_json::from_str(&row.data)
        .map_err(|e| MemoryError::Serialization(format!("turns JSON: {e}")))?;

    let created_at = chrono::DateTime::parse_from_rfc3339(&row.created_at)
        .map_err(|e| MemoryError::Serialization(format!("created_at: {e}")))?
        .to_utc();

    let updated_at = chrono::DateTime::parse_from_rfc3339(&row.updated_at)
        .map_err(|e| MemoryError::Serialization(format!("updated_at: {e}")))?
        .to_utc();

    Ok(Conversation {
        session_id: SessionId::from_string(row.session_id),
        system_prompt: row.system_prompt,
        turns,
        created_at,
        updated_at,
        model_id: ModelId::from_string(row.model_id),
        provider_id: ProviderId::from_string(row.provider_id),
    })
}

/// Convert a `rusqlite::Error` to `MemoryError::Io`.
fn io_err(context: &str, e: &rusqlite::Error) -> MemoryError {
    MemoryError::Io(std::io::Error::other(format!("{context}: {e}")))
}

use crate::helpers::OptionalExt as _;

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::significant_drop_tightening,
    clippy::panic
)]
mod tests {
    use chrono::Utc;
    use freebird_traits::id::{ModelId, ProviderId, SessionId};
    use freebird_traits::memory::{Conversation, Memory, Turn};
    use freebird_traits::provider::{ContentBlock, Message, Role};
    use secrecy::SecretString;

    use super::*;
    use crate::sqlite::SqliteDb;

    fn test_memory() -> (tempfile::TempDir, SqliteMemory) {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("test.db");
        let key = SecretString::from("a".repeat(64));
        let db = Arc::new(SqliteDb::open(&db_path, &key).unwrap());
        (dir, SqliteMemory::new(db))
    }

    fn make_conversation(session_id: &str, user_text: &str) -> Conversation {
        Conversation {
            session_id: SessionId::from_string(session_id),
            system_prompt: Some("test".into()),
            turns: vec![Turn {
                user_message: Message {
                    role: Role::User,
                    content: vec![ContentBlock::Text {
                        text: user_text.into(),
                    }],
                    timestamp: Utc::now(),
                },
                assistant_messages: vec![Message {
                    role: Role::Assistant,
                    content: vec![ContentBlock::Text {
                        text: "response".into(),
                    }],
                    timestamp: Utc::now(),
                }],
                tool_invocations: vec![],
                started_at: Utc::now(),
                completed_at: Some(Utc::now()),
            }],
            created_at: Utc::now(),
            updated_at: Utc::now(),
            model_id: ModelId::from_string("test-model"),
            provider_id: ProviderId::from_string("test-provider"),
        }
    }

    #[tokio::test]
    async fn test_save_and_load_roundtrip() {
        let (_dir, mem) = test_memory();
        let conv = make_conversation("session-1", "hello world");
        mem.save(&conv).await.unwrap();

        let loaded = mem
            .load(&SessionId::from_string("session-1"))
            .await
            .unwrap()
            .expect("should find session");
        assert_eq!(loaded.session_id.as_str(), "session-1");
        assert_eq!(loaded.turns.len(), 1);
    }

    #[tokio::test]
    async fn test_load_nonexistent_returns_none() {
        let (_dir, mem) = test_memory();
        let result = mem
            .load(&SessionId::from_string("nonexistent"))
            .await
            .unwrap();
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn test_save_overwrites() {
        let (_dir, mem) = test_memory();
        let conv1 = make_conversation("session-1", "first");
        mem.save(&conv1).await.unwrap();

        let conv2 = make_conversation("session-1", "second");
        mem.save(&conv2).await.unwrap();

        let loaded = mem
            .load(&SessionId::from_string("session-1"))
            .await
            .unwrap()
            .unwrap();
        if let ContentBlock::Text { text } = &loaded.turns[0].user_message.content[0] {
            assert_eq!(text, "second");
        } else {
            panic!("expected text block");
        }
    }

    #[tokio::test]
    async fn test_delete_existing() {
        let (_dir, mem) = test_memory();
        let conv = make_conversation("session-1", "hello");
        mem.save(&conv).await.unwrap();
        mem.delete(&SessionId::from_string("session-1"))
            .await
            .unwrap();

        let result = mem
            .load(&SessionId::from_string("session-1"))
            .await
            .unwrap();
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn test_delete_nonexistent_returns_not_found() {
        let (_dir, mem) = test_memory();
        let result = mem.delete(&SessionId::from_string("nonexistent")).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_list_sessions_ordered_by_updated_at() {
        let (_dir, mem) = test_memory();
        let mut conv1 = make_conversation("session-1", "first");
        conv1.updated_at = chrono::DateTime::parse_from_rfc3339("2025-01-01T00:00:00Z")
            .unwrap()
            .to_utc();
        let mut conv2 = make_conversation("session-2", "second");
        conv2.updated_at = chrono::DateTime::parse_from_rfc3339("2025-06-01T00:00:00Z")
            .unwrap()
            .to_utc();

        mem.save(&conv1).await.unwrap();
        mem.save(&conv2).await.unwrap();

        let sessions = mem.list_sessions(10).await.unwrap();
        assert_eq!(sessions.len(), 2);
        assert_eq!(sessions[0].session_id.as_str(), "session-2");
        assert_eq!(sessions[1].session_id.as_str(), "session-1");
    }

    #[tokio::test]
    async fn test_list_sessions_limit() {
        let (_dir, mem) = test_memory();
        for i in 0..5 {
            let conv = make_conversation(&format!("session-{i}"), &format!("msg {i}"));
            mem.save(&conv).await.unwrap();
        }

        let sessions = mem.list_sessions(3).await.unwrap();
        assert_eq!(sessions.len(), 3);
    }

    #[tokio::test]
    async fn test_search_finds_matching_content() {
        let (_dir, mem) = test_memory();
        let conv = make_conversation("session-1", "the quick brown fox");
        mem.save(&conv).await.unwrap();
        let conv2 = make_conversation("session-2", "lazy dog");
        mem.save(&conv2).await.unwrap();

        let results = mem.search("quick", 10).await.unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].session_id.as_str(), "session-1");
    }

    #[tokio::test]
    async fn test_search_case_insensitive() {
        let (_dir, mem) = test_memory();
        let conv = make_conversation("session-1", "Hello World");
        mem.save(&conv).await.unwrap();

        let results = mem.search("hello", 10).await.unwrap();
        assert_eq!(results.len(), 1);
    }

    #[tokio::test]
    async fn test_search_empty_query() {
        let (_dir, mem) = test_memory();
        let conv = make_conversation("session-1", "hello");
        mem.save(&conv).await.unwrap();

        let results = mem.search("", 10).await.unwrap();
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn test_search_no_match() {
        let (_dir, mem) = test_memory();
        let conv = make_conversation("session-1", "hello");
        mem.save(&conv).await.unwrap();

        let results = mem.search("zebra", 10).await.unwrap();
        assert!(results.is_empty());
    }
}
