//! `SQLite`-backed implementation of the [`Memory`] trait.
//!
//! Supports two storage modes:
//! - **Event-sourced** (new): loads from `conversation_events` table via event
//!   replay, lists/searches via `session_metadata` and `conversation_fts`.
//! - **Blob** (legacy): loads from `conversations` table as a JSON blob.
//!
//! On `load()`, events are tried first. If no events exist for a session,
//! the legacy blob table is checked as a fallback. `save()` writes to both
//! stores for backward compatibility during migration.

#![allow(clippy::significant_drop_tightening)]

use std::sync::Arc;

use async_trait::async_trait;
use freebird_traits::id::{ModelId, ProviderId, SessionId};
use freebird_traits::memory::{Conversation, Memory, MemoryError, SessionSummary, Turn};

use crate::event::{StoredEvent, replay_events_to_conversation};
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

        // Try event-sourced load first
        let events = load_events(&conn, &sid)?;
        if !events.is_empty() {
            return replay_events_to_conversation(session_id, &events);
        }

        // Fall back to legacy blob storage
        load_from_blob(&conn, &sid)
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

        // Write to legacy blob table (backward compat)
        conn.execute(
            "INSERT OR REPLACE INTO conversations \
             (session_id, system_prompt, model_id, provider_id, created_at, updated_at, data) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            rusqlite::params![sid, system_prompt, mid, pid, cat, uat, data],
        )
        .map_err(|e| io_err("insert conversation", &e))?;

        // Upsert session_metadata for event-sourced queries
        let preview = conversation
            .turns
            .first()
            .and_then(|t| t.user_message.content.first())
            .map(|block| match block {
                freebird_traits::provider::ContentBlock::Text { text } => {
                    text.chars().take(100).collect()
                }
                _ => String::new(),
            })
            .unwrap_or_default();

        #[allow(clippy::cast_possible_wrap)]
        let turn_count = conversation.turns.len() as i64;

        conn.execute(
            "INSERT OR REPLACE INTO session_metadata \
             (session_id, system_prompt, model_id, provider_id, \
              created_at, updated_at, turn_count, preview) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            rusqlite::params![sid, system_prompt, mid, pid, cat, uat, turn_count, preview],
        )
        .map_err(|e| io_err("upsert session_metadata", &e))?;

        Ok(())
    }

    async fn list_sessions(&self, limit: usize) -> Result<Vec<SessionSummary>, MemoryError> {
        let conn = self.db.conn().await;

        // Try session_metadata first (covers both event-sourced and save()-synced sessions)
        let summaries = list_from_metadata(&conn, limit)?;
        if !summaries.is_empty() {
            return Ok(summaries);
        }

        // Fall back to legacy blob table
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

        // Delete from all tables
        let blob_affected = conn
            .execute(
                "DELETE FROM conversations WHERE session_id = ?1",
                rusqlite::params![sid],
            )
            .map_err(|e| io_err("delete conversation", &e))?;

        let events_affected = conn
            .execute(
                "DELETE FROM conversation_events WHERE session_id = ?1",
                rusqlite::params![sid],
            )
            .map_err(|e| io_err("delete events", &e))?;

        conn.execute(
            "DELETE FROM session_metadata WHERE session_id = ?1",
            rusqlite::params![sid],
        )
        .map_err(|e| io_err("delete session_metadata", &e))?;

        if blob_affected == 0 && events_affected == 0 {
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

        let conn = self.db.conn().await;

        // Try FTS5 search first
        let fts_results = search_fts(&conn, query, limit)?;
        if !fts_results.is_empty() {
            return Ok(fts_results);
        }

        // Fall back to LIKE search on legacy blob table
        let pattern = format!("%{}%", query.to_lowercase());
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

// ---------------------------------------------------------------------------
// Event-sourced helpers
// ---------------------------------------------------------------------------

/// Load all events for a session, ordered by sequence.
fn load_events(
    conn: &rusqlite::Connection,
    session_id: &str,
) -> Result<Vec<StoredEvent>, MemoryError> {
    let mut stmt = conn
        .prepare(
            "SELECT session_id, sequence, event_data, timestamp, previous_hmac, hmac \
             FROM conversation_events WHERE session_id = ?1 ORDER BY sequence ASC",
        )
        .map_err(|e| io_err("prepare load events", &e))?;

    let rows = stmt
        .query_map(rusqlite::params![session_id], |row| {
            let event_json: String = row.get(2)?;
            let ts_str: String = row.get(3)?;
            Ok(EventRow {
                session_id: row.get(0)?,
                sequence: row.get(1)?,
                event_json,
                timestamp_str: ts_str,
                previous_hmac: row.get(4)?,
                hmac: row.get(5)?,
            })
        })
        .map_err(|e| io_err("query events", &e))?;

    let mut events = Vec::new();
    for row_result in rows {
        let row = row_result.map_err(|e| io_err("read event row", &e))?;
        let event = serde_json::from_str(&row.event_json)
            .map_err(|e| MemoryError::Serialization(format!("event JSON: {e}")))?;
        let timestamp = chrono::DateTime::parse_from_rfc3339(&row.timestamp_str)
            .map_err(|e| MemoryError::Serialization(format!("event timestamp: {e}")))?
            .to_utc();
        events.push(StoredEvent {
            session_id: row.session_id,
            sequence: row.sequence,
            event,
            timestamp,
            previous_hmac: row.previous_hmac,
            hmac: row.hmac,
        });
    }
    Ok(events)
}

/// List sessions from the `session_metadata` table.
fn list_from_metadata(
    conn: &rusqlite::Connection,
    limit: usize,
) -> Result<Vec<SessionSummary>, MemoryError> {
    let mut stmt = conn
        .prepare(
            "SELECT session_id, model_id, created_at, updated_at, turn_count, preview \
             FROM session_metadata ORDER BY updated_at DESC LIMIT ?1",
        )
        .map_err(|e| io_err("prepare list metadata", &e))?;

    let rows = stmt
        .query_map(rusqlite::params![limit_i64(limit)], |row| {
            Ok(MetadataRow {
                session_id: row.get(0)?,
                model_id: row.get(1)?,
                created_at: row.get(2)?,
                updated_at: row.get(3)?,
                turn_count: row.get(4)?,
                preview: row.get(5)?,
            })
        })
        .map_err(|e| io_err("query metadata", &e))?;

    let mut summaries = Vec::new();
    for row_result in rows {
        let row = row_result.map_err(|e| io_err("read metadata row", &e))?;
        let created_at = chrono::DateTime::parse_from_rfc3339(&row.created_at)
            .map_err(|e| MemoryError::Serialization(format!("created_at: {e}")))?
            .to_utc();
        let updated_at = chrono::DateTime::parse_from_rfc3339(&row.updated_at)
            .map_err(|e| MemoryError::Serialization(format!("updated_at: {e}")))?
            .to_utc();

        #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
        let turn_count = row.turn_count as usize;

        summaries.push(SessionSummary {
            session_id: SessionId::from_string(row.session_id),
            created_at,
            updated_at,
            turn_count,
            model_id: ModelId::from_string(row.model_id),
            preview: row.preview,
        });
    }
    Ok(summaries)
}

/// Search using FTS5 on `conversation_fts`, joining to `session_metadata`.
fn search_fts(
    conn: &rusqlite::Connection,
    query: &str,
    limit: usize,
) -> Result<Vec<SessionSummary>, MemoryError> {
    // Escape FTS5 special characters by wrapping in double quotes (phrase query).
    // Internal double quotes are doubled per FTS5 syntax.
    let escaped = query.replace('"', "\"\"");
    let fts_query = format!("\"{escaped}\"*");

    let mut stmt = conn
        .prepare(
            "SELECT DISTINCT m.session_id, m.model_id, m.created_at, m.updated_at, \
                    m.turn_count, m.preview \
             FROM conversation_fts f \
             JOIN session_metadata m ON f.session_id = m.session_id \
             WHERE conversation_fts MATCH ?1 \
             ORDER BY m.updated_at DESC LIMIT ?2",
        )
        .map_err(|e| io_err("prepare FTS search", &e))?;

    let rows = stmt
        .query_map(rusqlite::params![fts_query, limit_i64(limit)], |row| {
            Ok(MetadataRow {
                session_id: row.get(0)?,
                model_id: row.get(1)?,
                created_at: row.get(2)?,
                updated_at: row.get(3)?,
                turn_count: row.get(4)?,
                preview: row.get(5)?,
            })
        })
        .map_err(|e| io_err("FTS query", &e))?;

    let mut summaries = Vec::new();
    for row_result in rows {
        let row = row_result.map_err(|e| io_err("read FTS row", &e))?;
        let created_at = chrono::DateTime::parse_from_rfc3339(&row.created_at)
            .map_err(|e| MemoryError::Serialization(format!("created_at: {e}")))?
            .to_utc();
        let updated_at = chrono::DateTime::parse_from_rfc3339(&row.updated_at)
            .map_err(|e| MemoryError::Serialization(format!("updated_at: {e}")))?
            .to_utc();

        #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
        let turn_count = row.turn_count as usize;

        summaries.push(SessionSummary {
            session_id: SessionId::from_string(row.session_id),
            created_at,
            updated_at,
            turn_count,
            model_id: ModelId::from_string(row.model_id),
            preview: row.preview,
        });
    }
    Ok(summaries)
}

// ---------------------------------------------------------------------------
// Legacy blob helpers
// ---------------------------------------------------------------------------

/// Load a conversation from the legacy `conversations` blob table.
fn load_from_blob(
    conn: &rusqlite::Connection,
    session_id: &str,
) -> Result<Option<Conversation>, MemoryError> {
    let mut stmt = conn
        .prepare(
            "SELECT session_id, system_prompt, model_id, provider_id, \
             created_at, updated_at, data FROM conversations WHERE session_id = ?1",
        )
        .map_err(|e| io_err("prepare blob load", &e))?;

    let conv_row = stmt
        .query_row(rusqlite::params![session_id], |row| {
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
        .map_err(|e| io_err("query blob", &e))?;

    match conv_row {
        None => Ok(None),
        Some(r) => Ok(Some(row_to_conversation(r)?)),
    }
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// Intermediate row for events loaded from `conversation_events`.
struct EventRow {
    session_id: String,
    sequence: i64,
    event_json: String,
    timestamp_str: String,
    previous_hmac: String,
    hmac: String,
}

/// Intermediate row for session metadata.
struct MetadataRow {
    session_id: String,
    model_id: String,
    created_at: String,
    updated_at: String,
    turn_count: i64,
    preview: String,
}

/// Intermediate row representation for legacy conversations.
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
        let signing_key = ring::hmac::Key::new(ring::hmac::HMAC_SHA256, b"test-signing-key");
        let db = Arc::new(SqliteDb::open(&db_path, &key, signing_key).unwrap());
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

    #[tokio::test]
    async fn test_search_fts5_special_characters_escaped() {
        let (_dir, mem) = test_memory();
        let conv = make_conversation("session-1", "hello world");
        mem.save(&conv).await.unwrap();

        // FTS5 metacharacters should not cause errors or injection
        let results = mem.search("hello\" OR session_id:*", 10).await.unwrap();
        // Should not match (the injection attempt is treated as a literal phrase)
        assert!(results.is_empty());

        // Parens, AND/OR operators treated as literal
        let results = mem.search("(hello) AND secret", 10).await.unwrap();
        assert!(results.is_empty());
    }

    // --- Event-sourced load tests ---

    #[tokio::test]
    async fn test_load_from_events() {
        use freebird_traits::event::{ConversationEvent, EventSink};

        use crate::sqlite_event::SqliteEventSink;

        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("test.db");
        let key = SecretString::from("a".repeat(64));
        let signing_key = ring::hmac::Key::new(ring::hmac::HMAC_SHA256, b"test-signing-key");
        let db = Arc::new(SqliteDb::open(&db_path, &key, signing_key).unwrap());

        let event_sink = SqliteEventSink::new(Arc::clone(&db));
        let memory = SqliteMemory::new(Arc::clone(&db));

        let sid = SessionId::from_string("evt-session");

        // Append events
        event_sink
            .append(
                &sid,
                ConversationEvent::SessionCreated {
                    system_prompt: Some("test system".into()),
                    model_id: "claude".into(),
                    provider_id: "anthropic".into(),
                },
            )
            .await
            .unwrap();
        event_sink
            .append(
                &sid,
                ConversationEvent::TurnStarted {
                    turn_index: 0,
                    user_message: Message {
                        role: Role::User,
                        content: vec![ContentBlock::Text {
                            text: "hello from events".into(),
                        }],
                        timestamp: Utc::now(),
                    },
                },
            )
            .await
            .unwrap();
        event_sink
            .append(
                &sid,
                ConversationEvent::AssistantMessage {
                    turn_index: 0,
                    message_index: 0,
                    message: Message {
                        role: Role::Assistant,
                        content: vec![ContentBlock::Text {
                            text: "hi from assistant".into(),
                        }],
                        timestamp: Utc::now(),
                    },
                },
            )
            .await
            .unwrap();

        // Load via Memory trait — should replay from events
        let loaded = memory.load(&sid).await.unwrap().unwrap();
        assert_eq!(loaded.session_id.as_str(), "evt-session");
        assert_eq!(loaded.system_prompt.as_deref(), Some("test system"));
        assert_eq!(loaded.model_id.as_str(), "claude");
        assert_eq!(loaded.turns.len(), 1);
        assert_eq!(loaded.turns[0].assistant_messages.len(), 1);
    }

    #[tokio::test]
    async fn test_delete_removes_events_and_metadata() {
        use freebird_traits::event::{ConversationEvent, EventSink};

        use crate::sqlite_event::SqliteEventSink;

        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("test.db");
        let key = SecretString::from("a".repeat(64));
        let signing_key = ring::hmac::Key::new(ring::hmac::HMAC_SHA256, b"test-signing-key");
        let db = Arc::new(SqliteDb::open(&db_path, &key, signing_key).unwrap());

        let event_sink = SqliteEventSink::new(Arc::clone(&db));
        let memory = SqliteMemory::new(Arc::clone(&db));

        let sid = SessionId::from_string("del-session");
        event_sink
            .append(
                &sid,
                ConversationEvent::SessionCreated {
                    system_prompt: None,
                    model_id: "m".into(),
                    provider_id: "p".into(),
                },
            )
            .await
            .unwrap();

        // Should exist
        assert!(memory.load(&sid).await.unwrap().is_some());

        // Delete
        memory.delete(&sid).await.unwrap();

        // Should be gone
        assert!(memory.load(&sid).await.unwrap().is_none());

        // Metadata should be gone too
        let conn = db.conn().await;
        let count: i64 = conn
            .query_row(
                "SELECT count(*) FROM session_metadata WHERE session_id = 'del-session'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(count, 0);
    }
}
