//! `SQLite`-backed implementation of the [`Memory`] trait.
//!
//! All conversation data is stored as immutable events in `conversation_events`
//! and loaded via event replay. Session metadata in `session_metadata` supports
//! efficient listing and search. FTS5 powers full-text search.

#![allow(clippy::significant_drop_tightening)]

use std::sync::Arc;

use async_trait::async_trait;
use chrono::Utc;
use freebird_traits::event::ConversationEvent;
use freebird_traits::id::{ModelId, SessionId};
use freebird_traits::memory::{Conversation, Memory, MemoryError, SessionSummary};
use freebird_traits::provider::ContentBlock;

use crate::event::{StoredEvent, compute_event_hmac, replay_events_to_conversation};
use crate::helpers::rusqlite_to_io;
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

        let events = load_events(&conn, &sid)?;
        if events.is_empty() {
            return Ok(None);
        }

        replay_events_to_conversation(session_id, &events)
    }

    async fn save(&self, conversation: &Conversation) -> Result<(), MemoryError> {
        let sid = conversation.session_id.as_str().to_owned();
        let conn = self.db.conn().await;

        // Check if events already exist for this session (written by EventSink).
        // If not, emit events from the conversation to preserve the Memory trait
        // save→load roundtrip contract for standalone usage and tests.
        let event_count: i64 = conn
            .query_row(
                "SELECT count(*) FROM conversation_events WHERE session_id = ?1",
                rusqlite::params![sid],
                |row| row.get(0),
            )
            .map_err(|e| rusqlite_to_io("count events", &e))?;

        if event_count == 0 {
            emit_events_from_conversation(&conn, conversation, self.db.signing_key())?;
        }

        // Always upsert session_metadata
        upsert_session_metadata(&conn, conversation)?;

        Ok(())
    }

    async fn list_sessions(&self, limit: usize) -> Result<Vec<SessionSummary>, MemoryError> {
        let conn = self.db.conn().await;
        list_from_metadata(&conn, limit)
    }

    async fn delete(&self, session_id: &SessionId) -> Result<(), MemoryError> {
        let sid = session_id.as_str().to_owned();
        let conn = self.db.conn().await;

        let events_affected = conn
            .execute(
                "DELETE FROM conversation_events WHERE session_id = ?1",
                rusqlite::params![sid],
            )
            .map_err(|e| rusqlite_to_io("delete events", &e))?;

        conn.execute(
            "DELETE FROM session_metadata WHERE session_id = ?1",
            rusqlite::params![sid],
        )
        .map_err(|e| rusqlite_to_io("delete session_metadata", &e))?;

        if events_affected == 0 {
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
        search_fts(&conn, query, limit)
    }
}

// ---------------------------------------------------------------------------
// Event loading
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
        .map_err(|e| rusqlite_to_io("prepare load events", &e))?;

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
        .map_err(|e| rusqlite_to_io("query events", &e))?;

    let mut events = Vec::new();
    for row_result in rows {
        let row = row_result.map_err(|e| rusqlite_to_io("read event row", &e))?;
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

// ---------------------------------------------------------------------------
// Event emission from Conversation (for save() without EventSink)
// ---------------------------------------------------------------------------

/// Emit events from a [`Conversation`] into `conversation_events`.
///
/// Used by `save()` when no events exist yet — ensures the Memory trait
/// save→load roundtrip works without an active `EventSink`.
fn emit_events_from_conversation(
    db_conn: &rusqlite::Connection,
    conv: &Conversation,
    signing_key: &ring::hmac::Key,
) -> Result<(), MemoryError> {
    let sid = conv.session_id.as_str();
    let mut sequence: i64 = 0;
    let mut prev_hmac = String::new();

    // Helper: insert one event and advance the chain
    let mut emit = |event: &ConversationEvent| -> Result<(), MemoryError> {
        let event_type = event.event_type().to_owned();
        let event_json = serde_json::to_string(event)
            .map_err(|e| MemoryError::Serialization(format!("event serialization: {e}")))?;
        let timestamp = Utc::now().to_rfc3339();

        let hmac_hex =
            compute_event_hmac(sid, sequence, event, &timestamp, &prev_hmac, signing_key)?;

        db_conn
            .execute(
                "INSERT INTO conversation_events \
                 (session_id, sequence, event_type, event_data, timestamp, previous_hmac, hmac) \
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
                rusqlite::params![
                    sid, sequence, event_type, event_json, timestamp, prev_hmac, hmac_hex
                ],
            )
            .map_err(|e| rusqlite_to_io("insert event from save", &e))?;

        prev_hmac = hmac_hex;
        sequence += 1;
        Ok(())
    };

    // SessionCreated
    emit(&ConversationEvent::SessionCreated {
        system_prompt: conv.system_prompt.clone(),
        model_id: conv.model_id.as_str().to_owned(),
        provider_id: conv.provider_id.as_str().to_owned(),
    })?;

    // Per-turn events
    for (turn_idx, turn) in conv.turns.iter().enumerate() {
        emit(&ConversationEvent::TurnStarted {
            turn_index: turn_idx,
            user_message: turn.user_message.clone(),
        })?;

        for (msg_idx, msg) in turn.assistant_messages.iter().enumerate() {
            emit(&ConversationEvent::AssistantMessage {
                turn_index: turn_idx,
                message_index: msg_idx,
                message: msg.clone(),
            })?;
        }

        for (inv_idx, inv) in turn.tool_invocations.iter().enumerate() {
            emit(&ConversationEvent::ToolInvoked {
                turn_index: turn_idx,
                invocation_index: inv_idx,
                invocation: inv.clone(),
            })?;
        }

        if let Some(completed_at) = turn.completed_at {
            emit(&ConversationEvent::TurnCompleted {
                turn_index: turn_idx,
                completed_at,
            })?;
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Session metadata
// ---------------------------------------------------------------------------

/// Upsert session metadata from a [`Conversation`].
fn upsert_session_metadata(
    db_conn: &rusqlite::Connection,
    conv: &Conversation,
) -> Result<(), MemoryError> {
    let sid = conv.session_id.as_str();
    let cat = conv.created_at.to_rfc3339();
    let uat = conv.updated_at.to_rfc3339();

    let preview = conv
        .turns
        .first()
        .and_then(|t| t.user_message.content.first())
        .map(|block| match block {
            ContentBlock::Text { text } => text.chars().take(100).collect(),
            _ => String::new(),
        })
        .unwrap_or_default();

    #[allow(clippy::cast_possible_wrap)]
    let turn_count = conv.turns.len() as i64;

    db_conn
        .execute(
            "INSERT OR REPLACE INTO session_metadata \
             (session_id, system_prompt, model_id, provider_id, \
              created_at, updated_at, turn_count, preview) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            rusqlite::params![
                sid,
                conv.system_prompt,
                conv.model_id.as_str(),
                conv.provider_id.as_str(),
                cat,
                uat,
                turn_count,
                preview
            ],
        )
        .map_err(|e| rusqlite_to_io("upsert session_metadata", &e))?;

    Ok(())
}

// ---------------------------------------------------------------------------
// Metadata listing
// ---------------------------------------------------------------------------

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
        .map_err(|e| rusqlite_to_io("prepare list metadata", &e))?;

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
        .map_err(|e| rusqlite_to_io("query metadata", &e))?;

    let mut summaries = Vec::new();
    for row_result in rows {
        let row = row_result.map_err(|e| rusqlite_to_io("read metadata row", &e))?;
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
// FTS5 search
// ---------------------------------------------------------------------------

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

    // Use a subquery to avoid reading columns from the FTS5 virtual table,
    // which would try to fetch from the content table with mismatched column names.
    let mut stmt = conn
        .prepare(
            "SELECT m.session_id, m.model_id, m.created_at, m.updated_at, \
                    m.turn_count, m.preview \
             FROM session_metadata m \
             WHERE m.session_id IN ( \
                 SELECT ce.session_id FROM conversation_events ce \
                 WHERE ce.event_id IN ( \
                     SELECT rowid FROM conversation_fts WHERE conversation_fts MATCH ?1 \
                 ) \
             ) \
             ORDER BY m.updated_at DESC LIMIT ?2",
        )
        .map_err(|e| rusqlite_to_io("prepare FTS search", &e))?;

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
        .map_err(|e| rusqlite_to_io("FTS query", &e))?;

    let mut summaries = Vec::new();
    for row_result in rows {
        let row = row_result.map_err(|e| rusqlite_to_io("read FTS row", &e))?;
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
// Row types
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

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::significant_drop_tightening,
    clippy::panic,
    clippy::similar_names
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
        assert_eq!(loaded.system_prompt.as_deref(), Some("test"));
        assert_eq!(loaded.model_id.as_str(), "test-model");
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

        // Second save with same session should update metadata and not duplicate events
        let conv2 = make_conversation("session-1", "first");
        mem.save(&conv2).await.unwrap();

        let loaded = mem
            .load(&SessionId::from_string("session-1"))
            .await
            .unwrap()
            .unwrap();
        assert_eq!(loaded.turns.len(), 1);
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

    #[tokio::test]
    async fn test_save_emits_events_on_first_save() {
        let (_dir, mem) = test_memory();
        let conv = make_conversation("session-ev", "event test");
        mem.save(&conv).await.unwrap();

        // Verify events were emitted
        let loaded = mem
            .load(&SessionId::from_string("session-ev"))
            .await
            .unwrap()
            .unwrap();
        assert_eq!(loaded.turns.len(), 1);
        assert_eq!(loaded.system_prompt.as_deref(), Some("test"));

        // Verify assistant messages roundtrip
        if let ContentBlock::Text { text } = &loaded.turns[0].assistant_messages[0].content[0] {
            assert_eq!(text, "response");
        } else {
            panic!("expected text block");
        }
    }

    #[tokio::test]
    async fn test_save_does_not_duplicate_events() {
        use freebird_traits::event::{ConversationEvent, EventSink};

        use crate::sqlite_event::SqliteEventSink;

        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("test.db");
        let key = SecretString::from("a".repeat(64));
        let signing_key = ring::hmac::Key::new(ring::hmac::HMAC_SHA256, b"test-signing-key");
        let db = Arc::new(SqliteDb::open(&db_path, &key, signing_key).unwrap());

        let event_sink = SqliteEventSink::new(Arc::clone(&db));
        let memory = SqliteMemory::new(Arc::clone(&db));

        let sid = SessionId::from_string("no-dup");

        // Write events via EventSink first
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

        // Count events before save
        let conn = db.conn().await;
        let before: i64 = conn
            .query_row(
                "SELECT count(*) FROM conversation_events WHERE session_id = 'no-dup'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        drop(conn);

        // save() should NOT emit more events since they already exist
        let conv = Conversation {
            session_id: sid.clone(),
            system_prompt: None,
            turns: vec![],
            created_at: Utc::now(),
            updated_at: Utc::now(),
            model_id: ModelId::from_string("m"),
            provider_id: ProviderId::from_string("p"),
        };
        memory.save(&conv).await.unwrap();

        let conn = db.conn().await;
        let after: i64 = conn
            .query_row(
                "SELECT count(*) FROM conversation_events WHERE session_id = 'no-dup'",
                [],
                |row| row.get(0),
            )
            .unwrap();

        assert_eq!(before, after, "save() should not duplicate events");
    }
}
