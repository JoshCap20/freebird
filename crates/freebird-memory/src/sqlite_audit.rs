//! `SQLite`-backed implementation of the [`AuditSink`] trait.
//!
//! Persists security audit events as immutable rows in the `audit_events`
//! table with a global HMAC chain. Replaces the file-based JSONL audit logger.

#![allow(clippy::significant_drop_tightening)]

use std::sync::Arc;

use async_trait::async_trait;
use chrono::Utc;
use freebird_traits::audit::AuditSink;
use freebird_traits::memory::MemoryError;

use crate::event::compute_audit_hmac;
use crate::sqlite::SqliteDb;

/// `SQLite`-backed audit event sink.
///
/// Thread-safe via [`Arc<SqliteDb>`] (async `Mutex` inside).
pub struct SqliteAuditSink {
    db: Arc<SqliteDb>,
}

impl SqliteAuditSink {
    /// Create a new [`SqliteAuditSink`] sharing the given database connection.
    #[must_use]
    pub const fn new(db: Arc<SqliteDb>) -> Self {
        Self { db }
    }
}

use crate::helpers::rusqlite_to_io;
use rusqlite::OptionalExtension as _;

#[async_trait]
impl AuditSink for SqliteAuditSink {
    async fn record(
        &self,
        session_id: Option<&str>,
        event_type: &str,
        event_json: &str,
    ) -> Result<(), MemoryError> {
        let sid = session_id.map(ToOwned::to_owned);
        let etype = event_type.to_owned();
        let ejson = event_json.to_owned();
        let timestamp = Utc::now().to_rfc3339();

        let conn = self.db.conn().await;

        // Get the next global sequence number and previous HMAC
        let (next_seq, prev_hmac) = get_next_audit_sequence(&conn)?;

        // Compute HMAC
        let hmac_hex = compute_audit_hmac(
            next_seq,
            sid.as_deref(),
            &etype,
            &ejson,
            &timestamp,
            &prev_hmac,
            self.db.signing_key(),
        );

        // Insert event + update tail metadata atomically. If the process
        // crashes between the two, verify_chain() would false-positive on
        // tail truncation. A transaction ensures both succeed or neither does.
        let tx = conn
            .unchecked_transaction()
            .map_err(|e| rusqlite_to_io("begin audit transaction", &e))?;

        tx.execute(
            "INSERT INTO audit_events \
             (sequence, session_id, event_type, event_data, timestamp, previous_hmac, hmac) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            rusqlite::params![next_seq, sid, etype, ejson, timestamp, prev_hmac, hmac_hex],
        )
        .map_err(|e| rusqlite_to_io("insert audit event", &e))?;

        tx.execute(
            "UPDATE audit_metadata SET expected_next_sequence = ?1, last_hmac = ?2 WHERE id = 1",
            rusqlite::params![next_seq + 1, hmac_hex],
        )
        .map_err(|e| rusqlite_to_io("update audit metadata", &e))?;

        tx.commit()
            .map_err(|e| rusqlite_to_io("commit audit transaction", &e))?;

        Ok(())
    }

    async fn verify_chain(&self) -> Result<(), MemoryError> {
        let conn = self.db.conn().await;

        let mut stmt = conn
            .prepare(
                "SELECT sequence, session_id, event_type, event_data, \
                 timestamp, previous_hmac, hmac \
                 FROM audit_events ORDER BY sequence ASC",
            )
            .map_err(|e| rusqlite_to_io("prepare verify", &e))?;

        let mut expected_previous = String::new();
        let mut last_sequence: Option<i64> = None;

        let rows = stmt
            .query_map([], |row| {
                Ok(AuditRow {
                    sequence: row.get(0)?,
                    session_id: row.get(1)?,
                    event_type: row.get(2)?,
                    event_data: row.get(3)?,
                    timestamp: row.get(4)?,
                    previous_hmac: row.get(5)?,
                    hmac: row.get(6)?,
                })
            })
            .map_err(|e| rusqlite_to_io("query audit events", &e))?;

        for row_result in rows {
            let row = row_result.map_err(|e| rusqlite_to_io("read audit row", &e))?;

            if row.previous_hmac != expected_previous {
                return Err(MemoryError::IntegrityViolation {
                    reason: format!(
                        "audit chain broken at sequence {}: expected `{}`, got `{}`",
                        row.sequence, expected_previous, row.previous_hmac
                    ),
                });
            }

            let computed = compute_audit_hmac(
                row.sequence,
                row.session_id.as_deref(),
                &row.event_type,
                &row.event_data,
                &row.timestamp,
                &row.previous_hmac,
                self.db.signing_key(),
            );

            if computed != row.hmac {
                return Err(MemoryError::IntegrityViolation {
                    reason: format!(
                        "audit HMAC mismatch at sequence {}: event may have been tampered with",
                        row.sequence
                    ),
                });
            }

            last_sequence = Some(row.sequence);
            expected_previous = row.hmac;
        }

        // Tail truncation detection: compare actual last row against metadata
        verify_tail_metadata(&conn, last_sequence, &expected_previous)?;

        Ok(())
    }
}

/// Verify that the actual last row matches the metadata table.
///
/// If an attacker deletes the last N rows from `audit_events`, the remaining
/// chain is internally valid, but the metadata will disagree with the actual
/// last row's sequence and HMAC.
fn verify_tail_metadata(
    conn: &rusqlite::Connection,
    last_sequence: Option<i64>,
    last_hmac: &str,
) -> Result<(), MemoryError> {
    // Check if the audit_metadata table exists (legacy DB migration compat)
    let table_exists: bool = conn
        .query_row(
            "SELECT COUNT(*) > 0 FROM sqlite_master WHERE type='table' AND name='audit_metadata'",
            [],
            |row| row.get(0),
        )
        .map_err(|e| rusqlite_to_io("check audit_metadata table", &e))?;

    if !table_exists {
        // Legacy database without metadata — skip tail check
        tracing::warn!("audit_metadata table missing; skipping tail truncation check");
        return Ok(());
    }

    let meta: Option<(i64, String)> = conn
        .query_row(
            "SELECT expected_next_sequence, last_hmac FROM audit_metadata WHERE id = 1",
            [],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .optional()
        .map_err(|e| rusqlite_to_io("query audit metadata", &e))?;

    let Some((expected_next, expected_hmac)) = meta else {
        // No metadata row — skip (shouldn't happen with schema, but be defensive)
        return Ok(());
    };

    match last_sequence {
        Some(seq) => {
            if seq + 1 != expected_next {
                return Err(MemoryError::IntegrityViolation {
                    reason: format!(
                        "audit log tail truncation detected: \
                         last sequence is {seq} but metadata expects next sequence {expected_next}"
                    ),
                });
            }
            if last_hmac != expected_hmac {
                return Err(MemoryError::IntegrityViolation {
                    reason: "audit log tail truncation detected: \
                             last HMAC does not match metadata"
                        .into(),
                });
            }
        }
        None => {
            // No rows in audit_events — metadata should also be empty
            if expected_next != 0 || !expected_hmac.is_empty() {
                return Err(MemoryError::IntegrityViolation {
                    reason: format!(
                        "audit log tail truncation detected: \
                         no audit events but metadata expects next sequence {expected_next}"
                    ),
                });
            }
        }
    }

    Ok(())
}

/// Get the next global sequence number and the HMAC of the last audit event.
///
/// Reads from `audit_metadata` (the authoritative counter) rather than
/// scanning `audit_events ORDER BY sequence DESC`, which is O(n) and could
/// diverge if an events INSERT failed within a retried transaction.
fn get_next_audit_sequence(conn: &rusqlite::Connection) -> Result<(i64, String), MemoryError> {
    let (next_seq, last_hmac): (i64, String) = conn
        .query_row(
            "SELECT expected_next_sequence, last_hmac FROM audit_metadata WHERE id = 1",
            [],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .map_err(|e| rusqlite_to_io("query audit_metadata", &e))?;

    Ok((next_seq, last_hmac))
}

/// Intermediate row for audit event verification.
struct AuditRow {
    sequence: i64,
    session_id: Option<String>,
    event_type: String,
    event_data: String,
    timestamp: String,
    previous_hmac: String,
    hmac: String,
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::significant_drop_tightening,
    clippy::panic
)]
mod tests {
    use std::sync::Arc;

    use freebird_traits::audit::AuditSink;
    use secrecy::SecretString;

    use super::*;
    use crate::sqlite::SqliteDb;

    fn test_audit_sink() -> (tempfile::TempDir, SqliteAuditSink) {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("test.db");
        let key = SecretString::from("a".repeat(64));
        let signing_key = ring::hmac::Key::new(ring::hmac::HMAC_SHA256, b"test-signing-key");
        let db = Arc::new(SqliteDb::open(&db_path, &key, signing_key).unwrap());
        (dir, SqliteAuditSink::new(db))
    }

    #[tokio::test]
    async fn test_record_and_verify() {
        let (_dir, sink) = test_audit_sink();

        sink.record(Some("s1"), "test_event", r#"{"data": "hello"}"#)
            .await
            .unwrap();
        sink.record(Some("s1"), "test_event_2", r#"{"data": "world"}"#)
            .await
            .unwrap();
        sink.record(None, "auth_failed", r#"{"key": "xxx"}"#)
            .await
            .unwrap();

        // Verify chain integrity
        sink.verify_chain().await.unwrap();
    }

    #[tokio::test]
    async fn test_verify_empty_chain() {
        let (_dir, sink) = test_audit_sink();
        sink.verify_chain().await.unwrap();
    }

    #[tokio::test]
    async fn test_tamper_detection() {
        let (_dir, sink) = test_audit_sink();

        sink.record(Some("s1"), "event1", r#"{"a": 1}"#)
            .await
            .unwrap();
        sink.record(Some("s1"), "event2", r#"{"a": 2}"#)
            .await
            .unwrap();

        // Tamper with an event
        let conn = sink.db.conn().await;
        conn.execute(
            "UPDATE audit_events SET event_data = '{\"a\": 999}' WHERE sequence = 0",
            [],
        )
        .unwrap();
        drop(conn);

        let result = sink.verify_chain().await;
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("HMAC mismatch"));
    }

    #[tokio::test]
    async fn test_tail_truncation_detected_via_sequence_counter() {
        let (_dir, sink) = test_audit_sink();

        // Append 5 events
        for i in 0..5 {
            sink.record(Some("s1"), &format!("event_{i}"), "{}")
                .await
                .unwrap();
        }

        // Delete the last 2 events (sequences 3 and 4)
        let conn = sink.db.conn().await;
        conn.execute("DELETE FROM audit_events WHERE sequence >= 3", [])
            .unwrap();
        drop(conn);

        // The remaining chain (0,1,2) is internally valid, but metadata
        // says expected_next=5, while actual last is 2
        let result = sink.verify_chain().await;
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("tail truncation"),
            "expected tail truncation error, got: {err}"
        );
    }

    #[tokio::test]
    async fn test_no_tail_truncation_on_intact_log() {
        let (_dir, sink) = test_audit_sink();

        for i in 0..5 {
            sink.record(Some("s1"), &format!("event_{i}"), "{}")
                .await
                .unwrap();
        }

        // Should pass — log is intact
        sink.verify_chain().await.unwrap();
    }

    #[tokio::test]
    async fn test_tail_truncation_on_empty_with_stale_metadata() {
        let (_dir, sink) = test_audit_sink();

        // Append events, then delete ALL of them but leave metadata
        sink.record(Some("s1"), "e1", "{}").await.unwrap();
        sink.record(Some("s1"), "e2", "{}").await.unwrap();

        let conn = sink.db.conn().await;
        conn.execute("DELETE FROM audit_events", []).unwrap();
        drop(conn);

        // Metadata says expected_next=2, but no events exist
        let result = sink.verify_chain().await;
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("tail truncation"),
            "expected tail truncation error, got: {err}"
        );
    }

    #[tokio::test]
    async fn test_sequence_increments_globally() {
        let (_dir, sink) = test_audit_sink();

        sink.record(Some("s1"), "e1", "{}").await.unwrap();
        sink.record(Some("s2"), "e2", "{}").await.unwrap();
        sink.record(None, "e3", "{}").await.unwrap();

        let conn = sink.db.conn().await;
        let max_seq: i64 = conn
            .query_row("SELECT MAX(sequence) FROM audit_events", [], |row| {
                row.get(0)
            })
            .unwrap();
        assert_eq!(max_seq, 2);
    }
}
