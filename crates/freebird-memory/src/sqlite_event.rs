//! `SQLite`-backed implementation of the [`EventSink`] trait.
//!
//! Persists conversation events as immutable rows in the `conversation_events`
//! table with per-session HMAC chain integrity. Also maintains the
//! `session_metadata` denormalized table for efficient listing/search.

#![allow(clippy::significant_drop_tightening)]

use std::sync::Arc;

use async_trait::async_trait;
use chrono::Utc;
use freebird_traits::event::{ConversationEvent, EventSink};
use freebird_traits::id::SessionId;
use freebird_traits::memory::MemoryError;

use crate::event::compute_event_hmac;
use crate::sqlite::SqliteDb;

/// `SQLite`-backed event sink.
///
/// Thread-safe via [`Arc<SqliteDb>`] (async `Mutex` inside).
pub struct SqliteEventSink {
    db: Arc<SqliteDb>,
}

impl SqliteEventSink {
    /// Create a new [`SqliteEventSink`] sharing the given database connection.
    #[must_use]
    pub const fn new(db: Arc<SqliteDb>) -> Self {
        Self { db }
    }
}

#[async_trait]
impl EventSink for SqliteEventSink {
    async fn append(
        &self,
        session_id: &SessionId,
        event: ConversationEvent,
    ) -> Result<(), MemoryError> {
        let sid = session_id.as_str().to_owned();
        let event_type = event.event_type().to_owned();
        let event_json = serde_json::to_string(&event)
            .map_err(|e| MemoryError::Serialization(format!("event serialization: {e}")))?;
        let timestamp = Utc::now().to_rfc3339();

        let conn = self.db.conn().await;

        // Get the next sequence number and previous HMAC for this session
        let (next_seq, prev_hmac) = get_next_sequence_and_hmac(&conn, &sid)?;

        // Compute HMAC for this event
        let hmac_hex = compute_event_hmac(
            &sid,
            next_seq,
            &event,
            &timestamp,
            &prev_hmac,
            self.db.signing_key(),
        )?;

        // Insert the event
        conn.execute(
            "INSERT INTO conversation_events \
             (session_id, sequence, event_type, event_data, timestamp, previous_hmac, hmac) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            rusqlite::params![
                sid, next_seq, event_type, event_json, timestamp, prev_hmac, hmac_hex
            ],
        )
        .map_err(|e| io_err("insert event", &e))?;

        // Update session metadata
        update_session_metadata(&conn, &sid, &event, &timestamp)?;

        Ok(())
    }
}

/// Get the next sequence number and the HMAC of the last event for a session.
///
/// Returns `(0, "")` if no events exist yet.
fn get_next_sequence_and_hmac(
    conn: &rusqlite::Connection,
    session_id: &str,
) -> Result<(i64, String), MemoryError> {
    let result: Option<(i64, String)> = conn
        .query_row(
            "SELECT sequence, hmac FROM conversation_events \
             WHERE session_id = ?1 ORDER BY sequence DESC LIMIT 1",
            rusqlite::params![session_id],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .optional()
        .map_err(|e| io_err("query last event", &e))?;

    match result {
        Some((last_seq, last_hmac)) => Ok((last_seq + 1, last_hmac)),
        None => Ok((0, String::new())),
    }
}

/// Update the `session_metadata` table based on the event type.
fn update_session_metadata(
    conn: &rusqlite::Connection,
    session_id: &str,
    event: &ConversationEvent,
    timestamp: &str,
) -> Result<(), MemoryError> {
    match event {
        ConversationEvent::SessionCreated {
            system_prompt,
            model_id,
            provider_id,
        } => {
            conn.execute(
                "INSERT OR REPLACE INTO session_metadata \
                 (session_id, system_prompt, model_id, provider_id, created_at, updated_at, turn_count, preview) \
                 VALUES (?1, ?2, ?3, ?4, ?5, ?5, 0, '')",
                rusqlite::params![session_id, system_prompt, model_id, provider_id, timestamp],
            )
            .map_err(|e| io_err("insert session metadata", &e))?;
        }
        ConversationEvent::SessionMetadataUpdated {
            system_prompt,
            model_id,
            provider_id,
        } => {
            conn.execute(
                "UPDATE session_metadata \
                 SET system_prompt = ?2, model_id = ?3, provider_id = ?4, updated_at = ?5 \
                 WHERE session_id = ?1",
                rusqlite::params![session_id, system_prompt, model_id, provider_id, timestamp],
            )
            .map_err(|e| io_err("update session metadata", &e))?;
        }
        ConversationEvent::TurnStarted { user_message, .. } => {
            // Extract preview from first user message
            let preview = extract_preview(user_message);
            conn.execute(
                "UPDATE session_metadata \
                 SET turn_count = turn_count + 1, updated_at = ?2, \
                     preview = CASE WHEN preview = '' THEN ?3 ELSE preview END \
                 WHERE session_id = ?1",
                rusqlite::params![session_id, timestamp, preview],
            )
            .map_err(|e| io_err("update turn count", &e))?;
        }
        ConversationEvent::TurnCompleted { .. }
        | ConversationEvent::AssistantMessage { .. }
        | ConversationEvent::ToolInvoked { .. } => {
            // Just update the timestamp
            conn.execute(
                "UPDATE session_metadata SET updated_at = ?2 WHERE session_id = ?1",
                rusqlite::params![session_id, timestamp],
            )
            .map_err(|e| io_err("update timestamp", &e))?;
        }
    }

    Ok(())
}

/// Extract a preview string (up to 100 chars) from a message.
fn extract_preview(message: &freebird_traits::provider::Message) -> String {
    use freebird_traits::provider::ContentBlock;
    message
        .content
        .first()
        .and_then(|block| match block {
            ContentBlock::Text { text } => Some(text.chars().take(100).collect()),
            _ => None,
        })
        .unwrap_or_default()
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
    use std::sync::Arc;

    use chrono::Utc;
    use freebird_traits::event::{ConversationEvent, EventSink};
    use freebird_traits::id::SessionId;
    use freebird_traits::provider::{ContentBlock, Message, Role};
    use secrecy::SecretString;

    use super::*;
    use crate::event::{StoredEvent, replay_events_to_conversation, verify_event_chain};
    use crate::sqlite::SqliteDb;

    fn test_sink() -> (tempfile::TempDir, SqliteEventSink) {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("test.db");
        let key = SecretString::from("a".repeat(64));
        let signing_key = ring::hmac::Key::new(ring::hmac::HMAC_SHA256, b"test-signing-key");
        let db = Arc::new(SqliteDb::open(&db_path, &key, signing_key).unwrap());
        (dir, SqliteEventSink::new(db))
    }

    fn make_message(role: Role, text: &str) -> Message {
        Message {
            role,
            content: vec![ContentBlock::Text { text: text.into() }],
            timestamp: Utc::now(),
        }
    }

    #[tokio::test]
    async fn test_append_session_created() {
        let (_dir, sink) = test_sink();
        let sid = SessionId::from_string("s1");

        sink.append(
            &sid,
            ConversationEvent::SessionCreated {
                system_prompt: Some("test prompt".into()),
                model_id: "claude".into(),
                provider_id: "anthropic".into(),
            },
        )
        .await
        .unwrap();

        // Verify event was stored
        let conn = sink.db.conn().await;
        let count: i64 = conn
            .query_row(
                "SELECT count(*) FROM conversation_events WHERE session_id = 's1'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(count, 1);

        // Verify metadata was created
        let model: String = conn
            .query_row(
                "SELECT model_id FROM session_metadata WHERE session_id = 's1'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(model, "claude");
    }

    #[tokio::test]
    async fn test_append_increments_sequence() {
        let (_dir, sink) = test_sink();
        let sid = SessionId::from_string("s1");

        sink.append(
            &sid,
            ConversationEvent::SessionCreated {
                system_prompt: None,
                model_id: "m1".into(),
                provider_id: "p1".into(),
            },
        )
        .await
        .unwrap();

        sink.append(
            &sid,
            ConversationEvent::TurnStarted {
                turn_index: 0,
                user_message: make_message(Role::User, "hello"),
            },
        )
        .await
        .unwrap();

        let conn = sink.db.conn().await;
        let max_seq: i64 = conn
            .query_row(
                "SELECT MAX(sequence) FROM conversation_events WHERE session_id = 's1'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(max_seq, 1);
    }

    #[tokio::test]
    async fn test_hmac_chain_integrity() {
        let (_dir, sink) = test_sink();
        let sid = SessionId::from_string("s1");

        sink.append(
            &sid,
            ConversationEvent::SessionCreated {
                system_prompt: None,
                model_id: "m1".into(),
                provider_id: "p1".into(),
            },
        )
        .await
        .unwrap();

        sink.append(
            &sid,
            ConversationEvent::TurnStarted {
                turn_index: 0,
                user_message: make_message(Role::User, "hello"),
            },
        )
        .await
        .unwrap();

        // Load events and verify chain
        let conn = sink.db.conn().await;
        let mut stmt = conn
            .prepare(
                "SELECT session_id, sequence, event_data, timestamp, previous_hmac, hmac \
                 FROM conversation_events WHERE session_id = 's1' ORDER BY sequence",
            )
            .unwrap();

        let events: Vec<StoredEvent> = stmt
            .query_map([], |row| {
                let event_json: String = row.get(2)?;
                let ts_str: String = row.get(3)?;
                Ok(StoredEvent {
                    session_id: row.get(0)?,
                    sequence: row.get(1)?,
                    event: serde_json::from_str(&event_json).unwrap(),
                    timestamp: chrono::DateTime::parse_from_rfc3339(&ts_str)
                        .unwrap()
                        .to_utc(),
                    previous_hmac: row.get(4)?,
                    hmac: row.get(5)?,
                })
            })
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();

        assert_eq!(events.len(), 2);
        assert!(verify_event_chain(&events, sink.db.signing_key()).is_ok());
    }

    #[tokio::test]
    async fn test_full_turn_roundtrip_via_events() {
        let (_dir, sink) = test_sink();
        let sid = SessionId::from_string("s1");
        let now = Utc::now();

        // Emit a full session + turn
        sink.append(
            &sid,
            ConversationEvent::SessionCreated {
                system_prompt: Some("system".into()),
                model_id: "claude".into(),
                provider_id: "anthropic".into(),
            },
        )
        .await
        .unwrap();

        sink.append(
            &sid,
            ConversationEvent::TurnStarted {
                turn_index: 0,
                user_message: make_message(Role::User, "hello world"),
            },
        )
        .await
        .unwrap();

        sink.append(
            &sid,
            ConversationEvent::AssistantMessage {
                turn_index: 0,
                message_index: 0,
                message: make_message(Role::Assistant, "hi there"),
            },
        )
        .await
        .unwrap();

        sink.append(
            &sid,
            ConversationEvent::TurnCompleted {
                turn_index: 0,
                completed_at: now,
            },
        )
        .await
        .unwrap();

        // Load events and replay into Conversation
        let conn = sink.db.conn().await;
        let mut stmt = conn
            .prepare(
                "SELECT session_id, sequence, event_data, timestamp, previous_hmac, hmac \
                 FROM conversation_events WHERE session_id = 's1' ORDER BY sequence",
            )
            .unwrap();

        let events: Vec<StoredEvent> = stmt
            .query_map([], |row| {
                let event_json: String = row.get(2)?;
                let ts_str: String = row.get(3)?;
                Ok(StoredEvent {
                    session_id: row.get(0)?,
                    sequence: row.get(1)?,
                    event: serde_json::from_str(&event_json).unwrap(),
                    timestamp: chrono::DateTime::parse_from_rfc3339(&ts_str)
                        .unwrap()
                        .to_utc(),
                    previous_hmac: row.get(4)?,
                    hmac: row.get(5)?,
                })
            })
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();

        let conversation = replay_events_to_conversation(&sid, &events)
            .unwrap()
            .unwrap();

        assert_eq!(conversation.session_id.as_str(), "s1");
        assert_eq!(conversation.system_prompt.as_deref(), Some("system"));
        assert_eq!(conversation.model_id.as_str(), "claude");
        assert_eq!(conversation.turns.len(), 1);
        assert_eq!(conversation.turns[0].assistant_messages.len(), 1);
        assert!(conversation.turns[0].completed_at.is_some());
    }

    #[tokio::test]
    async fn test_metadata_preview_set_on_first_turn() {
        let (_dir, sink) = test_sink();
        let sid = SessionId::from_string("s1");

        sink.append(
            &sid,
            ConversationEvent::SessionCreated {
                system_prompt: None,
                model_id: "m1".into(),
                provider_id: "p1".into(),
            },
        )
        .await
        .unwrap();

        sink.append(
            &sid,
            ConversationEvent::TurnStarted {
                turn_index: 0,
                user_message: make_message(Role::User, "first message preview"),
            },
        )
        .await
        .unwrap();

        let conn = sink.db.conn().await;
        let preview: String = conn
            .query_row(
                "SELECT preview FROM session_metadata WHERE session_id = 's1'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(preview, "first message preview");

        let turn_count: i64 = conn
            .query_row(
                "SELECT turn_count FROM session_metadata WHERE session_id = 's1'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(turn_count, 1);
    }

    #[tokio::test]
    async fn test_independent_sessions() {
        let (_dir, sink) = test_sink();
        let s1 = SessionId::from_string("s1");
        let s2 = SessionId::from_string("s2");

        sink.append(
            &s1,
            ConversationEvent::SessionCreated {
                system_prompt: None,
                model_id: "m1".into(),
                provider_id: "p1".into(),
            },
        )
        .await
        .unwrap();

        sink.append(
            &s2,
            ConversationEvent::SessionCreated {
                system_prompt: None,
                model_id: "m2".into(),
                provider_id: "p2".into(),
            },
        )
        .await
        .unwrap();

        // Both start at sequence 0
        let conn = sink.db.conn().await;
        let count: i64 = conn
            .query_row(
                "SELECT count(*) FROM conversation_events WHERE sequence = 0",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(count, 2);
    }
}
