//! Shared `SQLCipher` database connection for memory and knowledge backends.
//!
//! Manages encrypted database lifecycle: key application, schema migrations,
//! and connection access via `tokio::sync::Mutex` for async safety.

use std::path::Path;

use freebird_traits::memory::MemoryError;
use ring::hmac;
use secrecy::{ExposeSecret, SecretString};

/// Shared encrypted `SQLite` database connection.
///
/// All access goes through the async mutex. Single connection is sufficient
/// because `FreeBird` handles one user at a time. `SQLCipher` key is applied
/// once at open time and is transparent for all subsequent operations.
///
/// The `signing_key` is used for HMAC chain integrity on conversation events
/// and audit events.
pub struct SqliteDb {
    conn: tokio::sync::Mutex<rusqlite::Connection>,
    signing_key: hmac::Key,
}

impl std::fmt::Debug for SqliteDb {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SqliteDb").finish_non_exhaustive()
    }
}

/// Convert a `rusqlite` error into a `MemoryError::Io`.
fn rusqlite_io(context: &str, e: &rusqlite::Error) -> MemoryError {
    MemoryError::Io(std::io::Error::other(format!("{context}: {e}")))
}

impl SqliteDb {
    /// Open (or create) an encrypted `SQLite` database.
    ///
    /// Applies the `SQLCipher` key, verifies it's correct, and runs
    /// any pending schema migrations.
    ///
    /// # Errors
    ///
    /// - `MemoryError::IntegrityViolation` if the key is wrong
    /// - `MemoryError::Io` if the database cannot be opened or migrated
    pub fn open(
        path: &Path,
        key: &SecretString,
        signing_key: hmac::Key,
    ) -> Result<Self, MemoryError> {
        // Create parent directory if needed
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| {
                MemoryError::Io(std::io::Error::other(format!(
                    "failed to create database directory: {e}"
                )))
            })?;
        }

        let conn = rusqlite::Connection::open(path)
            .map_err(|e| rusqlite_io("failed to open database", &e))?;

        // Apply encryption key — MUST be the first statement after open
        let pragma_key = format!("x'{}'", key.expose_secret());
        conn.pragma_update(None, "key", &pragma_key)
            .map_err(|e| rusqlite_io("failed to apply database key", &e))?;

        // Verify the key is correct by reading sqlite_master
        conn.query_row("SELECT count(*) FROM sqlite_master", [], |_| Ok(()))
            .map_err(|_| MemoryError::IntegrityViolation {
                reason: "database key is incorrect or database is corrupted".into(),
            })?;

        // Enable WAL mode for better concurrent read performance
        conn.pragma_update(None, "journal_mode", "WAL")
            .map_err(|e| rusqlite_io("failed to enable WAL mode", &e))?;

        // Run migrations
        Self::migrate(&conn)?;

        Ok(Self {
            conn: tokio::sync::Mutex::new(conn),
            signing_key,
        })
    }

    /// Acquire the database connection lock.
    ///
    /// Callers MUST NOT hold this lock across `.await` points.
    pub async fn conn(&self) -> tokio::sync::MutexGuard<'_, rusqlite::Connection> {
        self.conn.lock().await
    }

    /// The HMAC signing key for event and audit chain integrity.
    #[must_use]
    pub const fn signing_key(&self) -> &hmac::Key {
        &self.signing_key
    }

    /// Run pending schema migrations.
    fn migrate(conn: &rusqlite::Connection) -> Result<(), MemoryError> {
        let current_version = Self::get_schema_version(conn);

        let migrations: &[(i64, &str)] = &[(1, include_str!("migrations/001_initial.sql"))];

        for &(version, sql) in migrations {
            if version > current_version {
                conn.execute_batch(sql)
                    .map_err(|e| rusqlite_io(&format!("migration {version} failed"), &e))?;
                conn.execute(
                    "INSERT OR REPLACE INTO schema_version (version, applied_at) VALUES (?1, ?2)",
                    rusqlite::params![version, chrono::Utc::now().to_rfc3339()],
                )
                .map_err(|e| rusqlite_io(&format!("failed to record migration {version}"), &e))?;
                tracing::info!(version, "applied database migration");
            }
        }

        Ok(())
    }

    /// Get the current schema version (0 if no migrations applied yet).
    fn get_schema_version(conn: &rusqlite::Connection) -> i64 {
        conn.query_row("SELECT MAX(version) FROM schema_version", [], |row| {
            row.get(0)
        })
        .unwrap_or(0)
    }
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::significant_drop_tightening
)]
mod tests {
    use super::*;

    /// Create a test database with a known key.
    fn test_db() -> (tempfile::TempDir, SqliteDb) {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("test.db");
        let key = SecretString::from("a".repeat(64));
        let signing_key = ring::hmac::Key::new(ring::hmac::HMAC_SHA256, b"test-signing-key");
        let db = SqliteDb::open(&db_path, &key, signing_key).unwrap();
        (dir, db)
    }

    #[tokio::test]
    async fn test_open_creates_database() {
        let (dir, _db) = test_db();
        assert!(dir.path().join("test.db").exists());
    }

    #[tokio::test]
    async fn test_migration_creates_tables() {
        let (_dir, db) = test_db();
        let conn = db.conn().await;
        // conversations table dropped by migration 003
        let count: i64 = conn
            .query_row("SELECT count(*) FROM knowledge", [], |row| row.get(0))
            .unwrap();
        assert_eq!(count, 0);
        let count: i64 = conn
            .query_row("SELECT count(*) FROM conversation_events", [], |row| {
                row.get(0)
            })
            .unwrap();
        assert_eq!(count, 0);
    }

    #[tokio::test]
    async fn test_schema_version_recorded() {
        let (_dir, db) = test_db();
        let conn = db.conn().await;
        let version: i64 = conn
            .query_row("SELECT MAX(version) FROM schema_version", [], |row| {
                row.get(0)
            })
            .unwrap();
        assert_eq!(version, 1);
    }

    fn test_signing_key() -> ring::hmac::Key {
        ring::hmac::Key::new(ring::hmac::HMAC_SHA256, b"test-signing-key")
    }

    #[test]
    fn test_wrong_key_returns_integrity_violation() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("test.db");
        let key = SecretString::from("a".repeat(64));

        // Create DB with key, then drop
        drop(SqliteDb::open(&db_path, &key, test_signing_key()).unwrap());

        // Try to open with wrong key
        let wrong_key = SecretString::from("b".repeat(64));
        let result = SqliteDb::open(&db_path, &wrong_key, test_signing_key());
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            MemoryError::IntegrityViolation { .. }
        ));
    }

    #[tokio::test]
    async fn test_migration_is_idempotent() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("test.db");
        let key = SecretString::from("a".repeat(64));

        // Open twice — migrations should not fail on second open
        drop(SqliteDb::open(&db_path, &key, test_signing_key()).unwrap());
        drop(SqliteDb::open(&db_path, &key, test_signing_key()).unwrap());
    }

    #[tokio::test]
    async fn test_fts5_table_exists() {
        let (_dir, db) = test_db();
        let conn = db.conn().await;
        let count: i64 = conn
            .query_row(
                "SELECT count(*) FROM sqlite_master WHERE name = 'knowledge_fts'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(count, 1);
    }

    #[tokio::test]
    async fn test_migration_creates_event_tables() {
        let (_dir, db) = test_db();
        let conn = db.conn().await;

        // conversation_events table exists
        let count: i64 = conn
            .query_row("SELECT count(*) FROM conversation_events", [], |row| {
                row.get(0)
            })
            .unwrap();
        assert_eq!(count, 0);

        // session_metadata table exists
        let count: i64 = conn
            .query_row("SELECT count(*) FROM session_metadata", [], |row| {
                row.get(0)
            })
            .unwrap();
        assert_eq!(count, 0);

        // audit_events table exists
        let count: i64 = conn
            .query_row("SELECT count(*) FROM audit_events", [], |row| row.get(0))
            .unwrap();
        assert_eq!(count, 0);

        // conversation_fts virtual table exists
        let count: i64 = conn
            .query_row(
                "SELECT count(*) FROM sqlite_master WHERE name = 'conversation_fts'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(count, 1);
    }

    #[tokio::test]
    async fn test_signing_key_accessible() {
        let (_dir, db) = test_db();
        let _key = db.signing_key();
    }
}
