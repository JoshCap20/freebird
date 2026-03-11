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

/// Convert a `rusqlite::Error` to `MemoryError::Io`.
fn io_err(context: &str, e: &rusqlite::Error) -> MemoryError {
    MemoryError::Io(std::io::Error::other(format!("{context}: {e}")))
}

use crate::helpers::OptionalExt as _;

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

        conn.execute(
            "INSERT INTO audit_events \
             (sequence, session_id, event_type, event_data, timestamp, previous_hmac, hmac) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            rusqlite::params![next_seq, sid, etype, ejson, timestamp, prev_hmac, hmac_hex],
        )
        .map_err(|e| io_err("insert audit event", &e))?;

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
            .map_err(|e| io_err("prepare verify", &e))?;

        let mut expected_previous = String::new();

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
            .map_err(|e| io_err("query audit events", &e))?;

        for row_result in rows {
            let row = row_result.map_err(|e| io_err("read audit row", &e))?;

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

            expected_previous = row.hmac;
        }

        Ok(())
    }
}

/// Get the next global sequence number and the HMAC of the last audit event.
fn get_next_audit_sequence(conn: &rusqlite::Connection) -> Result<(i64, String), MemoryError> {
    let result: Option<(i64, String)> = conn
        .query_row(
            "SELECT sequence, hmac FROM audit_events ORDER BY sequence DESC LIMIT 1",
            [],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .optional()
        .map_err(|e| io_err("query last audit event", &e))?;

    match result {
        Some((last_seq, last_hmac)) => Ok((last_seq + 1, last_hmac)),
        None => Ok((0, String::new())),
    }
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
