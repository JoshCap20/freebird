//! `SQLite`-backed implementation of the [`KnowledgeStore`] trait.
//!
//! Uses FTS5 with BM25 ranking for text search and `SQLCipher` encryption
//! via the shared [`SqliteDb`] connection.

#![allow(clippy::significant_drop_tightening)]

use std::collections::BTreeSet;
use std::sync::Arc;

use async_trait::async_trait;
use freebird_traits::id::KnowledgeId;
use freebird_traits::knowledge::{
    KnowledgeEntry, KnowledgeError, KnowledgeKind, KnowledgeMatch, KnowledgeSource, KnowledgeStore,
};

use crate::sqlite::SqliteDb;

/// `SQLite`-backed knowledge store.
///
/// Thread-safe via [`Arc<SqliteDb>`] (async `Mutex` inside).
pub struct SqliteKnowledgeStore {
    db: Arc<SqliteDb>,
}

impl SqliteKnowledgeStore {
    /// Create a new [`SqliteKnowledgeStore`] sharing the given database connection.
    #[must_use]
    pub const fn new(db: Arc<SqliteDb>) -> Self {
        Self { db }
    }
}

/// Convert a `rusqlite::Error` into a [`KnowledgeError::Database`].
fn db_err(context: &str, e: &rusqlite::Error) -> KnowledgeError {
    KnowledgeError::Database(format!("{context}: {e}"))
}

/// Parse a tags JSON array string back into a `BTreeSet<String>`.
fn parse_tags(json: &str) -> Result<BTreeSet<String>, KnowledgeError> {
    serde_json::from_str(json).map_err(|e| KnowledgeError::Serialization(format!("tags: {e}")))
}

/// Parse a [`KnowledgeKind`] from its serde string representation.
fn parse_kind(s: &str) -> Result<KnowledgeKind, KnowledgeError> {
    serde_json::from_str(&format!("\"{s}\""))
        .map_err(|e| KnowledgeError::Serialization(format!("kind: {e}")))
}

/// Parse a [`KnowledgeSource`] from its serde string representation.
fn parse_source(s: &str) -> Result<KnowledgeSource, KnowledgeError> {
    serde_json::from_str(&format!("\"{s}\""))
        .map_err(|e| KnowledgeError::Serialization(format!("source: {e}")))
}

/// Parse an RFC 3339 datetime string.
fn parse_datetime(s: &str) -> Result<chrono::DateTime<chrono::Utc>, KnowledgeError> {
    chrono::DateTime::parse_from_rfc3339(s)
        .map(|dt| dt.to_utc())
        .map_err(|e| KnowledgeError::Serialization(format!("datetime: {e}")))
}

/// Convert a `rusqlite` deserialization error for use in `FromSql` conversions.
fn from_sql_err(col: usize, e: KnowledgeError) -> rusqlite::Error {
    rusqlite::Error::FromSqlConversionFailure(col, rusqlite::types::Type::Text, Box::new(e))
}

/// Extract a [`KnowledgeEntry`] from a `rusqlite::Row`.
///
/// Column order: id, kind, content, tags, source, confidence,
/// `session_id`, `created_at`, `updated_at`, `access_count`, `last_accessed`
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn row_to_entry(row: &rusqlite::Row<'_>) -> Result<KnowledgeEntry, rusqlite::Error> {
    Ok(KnowledgeEntry {
        id: KnowledgeId::from_string(row.get::<_, String>(0)?),
        kind: parse_kind(&row.get::<_, String>(1)?).map_err(|e| from_sql_err(1, e))?,
        content: row.get(2)?,
        tags: parse_tags(&row.get::<_, String>(3)?).map_err(|e| from_sql_err(3, e))?,
        source: parse_source(&row.get::<_, String>(4)?).map_err(|e| from_sql_err(4, e))?,
        confidence: row.get::<_, f64>(5)? as f32,
        session_id: row
            .get::<_, Option<String>>(6)?
            .map(freebird_traits::id::SessionId::from_string),
        created_at: parse_datetime(&row.get::<_, String>(7)?).map_err(|e| from_sql_err(7, e))?,
        updated_at: parse_datetime(&row.get::<_, String>(8)?).map_err(|e| from_sql_err(8, e))?,
        access_count: row.get::<_, i64>(9)? as u64,
        last_accessed: row
            .get::<_, Option<String>>(10)?
            .map(|s| parse_datetime(&s))
            .transpose()
            .map_err(|e| from_sql_err(10, e))?,
    })
}

/// Serialize a [`KnowledgeKind`] to its serde string (e.g. `"system_config"`).
fn kind_str(kind: &KnowledgeKind) -> Result<String, KnowledgeError> {
    let json = serde_json::to_string(kind)
        .map_err(|e| KnowledgeError::Serialization(format!("kind: {e}")))?;
    Ok(json.trim_matches('"').to_owned())
}

/// Serialize a [`KnowledgeSource`] to its serde string.
fn source_str(source: &KnowledgeSource) -> Result<String, KnowledgeError> {
    let json = serde_json::to_string(source)
        .map_err(|e| KnowledgeError::Serialization(format!("source: {e}")))?;
    Ok(json.trim_matches('"').to_owned())
}

/// All columns for SELECT queries.
const SELECT_COLS: &str = "id, kind, content, tags, source, confidence, \
                           session_id, created_at, updated_at, access_count, last_accessed";

/// Saturating conversion from `usize` to `i64` for SQL LIMIT parameters.
#[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
const fn limit_i64(limit: usize) -> i64 {
    if limit > i64::MAX as usize {
        i64::MAX
    } else {
        limit as i64
    }
}

#[allow(clippy::cast_possible_wrap)]
#[async_trait]
impl KnowledgeStore for SqliteKnowledgeStore {
    async fn store(&self, entry: KnowledgeEntry) -> Result<KnowledgeId, KnowledgeError> {
        let id_str = entry.id.as_str().to_owned();
        let kind = kind_str(&entry.kind)?;
        let tags_json = serde_json::to_string(&entry.tags)
            .map_err(|e| KnowledgeError::Serialization(e.to_string()))?;
        let source = source_str(&entry.source)?;
        let sid = entry.session_id.as_ref().map(|s| s.as_str().to_owned());
        let cat = entry.created_at.to_rfc3339();
        let uat = entry.updated_at.to_rfc3339();
        let la = entry.last_accessed.map(|dt| dt.to_rfc3339());

        let conn = self.db.conn().await;
        conn.execute(
            &format!(
                "INSERT INTO knowledge ({SELECT_COLS}) \
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)"
            ),
            rusqlite::params![
                id_str,
                kind,
                entry.content,
                tags_json,
                source,
                f64::from(entry.confidence),
                sid,
                cat,
                uat,
                entry.access_count as i64,
                la,
            ],
        )
        .map_err(|e| db_err("insert knowledge", &e))?;

        Ok(entry.id)
    }

    async fn update(&self, entry: &KnowledgeEntry) -> Result<(), KnowledgeError> {
        let tags_json = serde_json::to_string(&entry.tags)
            .map_err(|e| KnowledgeError::Serialization(e.to_string()))?;
        let uat = entry.updated_at.to_rfc3339();

        let conn = self.db.conn().await;
        let affected = conn
            .execute(
                "UPDATE knowledge SET content = ?1, tags = ?2, confidence = ?3, \
                 updated_at = ?4 WHERE id = ?5",
                rusqlite::params![
                    entry.content,
                    tags_json,
                    f64::from(entry.confidence),
                    uat,
                    entry.id.as_str(),
                ],
            )
            .map_err(|e| db_err("update knowledge", &e))?;

        if affected == 0 {
            return Err(KnowledgeError::NotFound {
                id: entry.id.clone(),
            });
        }
        Ok(())
    }

    async fn get(&self, id: &KnowledgeId) -> Result<Option<KnowledgeEntry>, KnowledgeError> {
        let conn = self.db.conn().await;
        let mut stmt = conn
            .prepare(&format!(
                "SELECT {SELECT_COLS} FROM knowledge WHERE id = ?1"
            ))
            .map_err(|e| db_err("prepare get", &e))?;

        let result = stmt
            .query_row(rusqlite::params![id.as_str()], row_to_entry)
            .optional()
            .map_err(|e| db_err("get knowledge", &e))?;

        Ok(result)
    }

    async fn delete(&self, id: &KnowledgeId) -> Result<(), KnowledgeError> {
        let conn = self.db.conn().await;
        let affected = conn
            .execute(
                "DELETE FROM knowledge WHERE id = ?1",
                rusqlite::params![id.as_str()],
            )
            .map_err(|e| db_err("delete knowledge", &e))?;

        if affected == 0 {
            return Err(KnowledgeError::NotFound { id: id.clone() });
        }
        Ok(())
    }

    async fn search(
        &self,
        query: &str,
        limit: usize,
    ) -> Result<Vec<KnowledgeMatch>, KnowledgeError> {
        if query.is_empty() {
            return Ok(Vec::new());
        }

        // Sanitize for FTS5: wrap in double-quotes to treat as phrase query,
        // escaping any embedded double-quotes. This prevents FTS5 operator
        // injection (e.g., unbalanced parens, NEAR, AND/OR operators).
        let sanitized = format!("\"{}\"", query.replace('"', "\"\""));

        let conn = self.db.conn().await;
        let qualified_cols = SELECT_COLS.replace(", ", ", k.");
        let mut stmt = conn
            .prepare(&format!(
                "SELECT k.{qualified_cols}, bm25(knowledge_fts) as rank \
                 FROM knowledge_fts f \
                 JOIN knowledge k ON k.rowid = f.rowid \
                 WHERE knowledge_fts MATCH ?1 \
                 ORDER BY rank \
                 LIMIT ?2",
            ))
            .map_err(|e| db_err("prepare search", &e))?;

        let rows = stmt
            .query_map(rusqlite::params![sanitized, limit_i64(limit)], |row| {
                let entry = row_to_entry(row)?;
                let rank: f64 = row.get(11)?;
                Ok(KnowledgeMatch { entry, rank })
            })
            .map_err(|e| db_err("search knowledge", &e))?;

        let mut results = Vec::new();
        for row_result in rows {
            results.push(row_result.map_err(|e| db_err("search row", &e))?);
        }
        Ok(results)
    }

    async fn list_by_kind(
        &self,
        kind: &KnowledgeKind,
        limit: usize,
    ) -> Result<Vec<KnowledgeEntry>, KnowledgeError> {
        let conn = self.db.conn().await;
        let mut stmt = conn
            .prepare(&format!(
                "SELECT {SELECT_COLS} FROM knowledge WHERE kind = ?1 \
                 ORDER BY updated_at DESC LIMIT ?2"
            ))
            .map_err(|e| db_err("prepare list_by_kind", &e))?;

        let rows = stmt
            .query_map(
                rusqlite::params![kind_str(kind)?, limit_i64(limit)],
                row_to_entry,
            )
            .map_err(|e| db_err("list_by_kind", &e))?;

        let mut entries = Vec::new();
        for row_result in rows {
            entries.push(row_result.map_err(|e| db_err("list_by_kind row", &e))?);
        }
        Ok(entries)
    }

    async fn list_by_tag(
        &self,
        tag: &str,
        limit: usize,
    ) -> Result<Vec<KnowledgeEntry>, KnowledgeError> {
        let pattern = format!("%\"{tag}\"%");
        let conn = self.db.conn().await;
        let mut stmt = conn
            .prepare(&format!(
                "SELECT {SELECT_COLS} FROM knowledge WHERE tags LIKE ?1 \
                 ORDER BY updated_at DESC LIMIT ?2"
            ))
            .map_err(|e| db_err("prepare list_by_tag", &e))?;

        let rows = stmt
            .query_map(rusqlite::params![pattern, limit_i64(limit)], row_to_entry)
            .map_err(|e| db_err("list_by_tag", &e))?;

        let mut entries = Vec::new();
        for row_result in rows {
            entries.push(row_result.map_err(|e| db_err("list_by_tag row", &e))?);
        }
        Ok(entries)
    }

    async fn replace_kind(
        &self,
        kind: &KnowledgeKind,
        entries: Vec<KnowledgeEntry>,
    ) -> Result<(), KnowledgeError> {
        let kind_s = kind_str(kind)?;
        let conn = self.db.conn().await;

        // Use unchecked_transaction because we already hold the async mutex.
        // Auto-rolls-back on drop if commit() is never reached.
        let tx = conn
            .unchecked_transaction()
            .map_err(|e| db_err("begin transaction", &e))?;

        tx.execute(
            "DELETE FROM knowledge WHERE kind = ?1",
            rusqlite::params![kind_s],
        )
        .map_err(|e| db_err("delete kind", &e))?;

        for entry in &entries {
            let tags_json = serde_json::to_string(&entry.tags)
                .map_err(|e| KnowledgeError::Serialization(e.to_string()))?;
            tx.execute(
                &format!(
                    "INSERT INTO knowledge ({SELECT_COLS}) \
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)"
                ),
                rusqlite::params![
                    entry.id.as_str(),
                    kind_s,
                    entry.content,
                    tags_json,
                    source_str(&entry.source)?,
                    f64::from(entry.confidence),
                    entry.session_id.as_ref().map(|s| s.as_str().to_owned()),
                    entry.created_at.to_rfc3339(),
                    entry.updated_at.to_rfc3339(),
                    entry.access_count as i64,
                    entry.last_accessed.map(|dt| dt.to_rfc3339()),
                ],
            )
            .map_err(|e| db_err("insert replacement", &e))?;
        }

        tx.commit().map_err(|e| db_err("commit replace_kind", &e))?;

        Ok(())
    }

    async fn record_access(&self, ids: &[KnowledgeId]) -> Result<(), KnowledgeError> {
        if ids.is_empty() {
            return Ok(());
        }

        let now = chrono::Utc::now().to_rfc3339();
        let conn = self.db.conn().await;
        let mut stmt = conn
            .prepare(
                "UPDATE knowledge SET access_count = access_count + 1, \
                 last_accessed = ?1 WHERE id = ?2",
            )
            .map_err(|e| db_err("prepare record_access", &e))?;

        for id in ids {
            stmt.execute(rusqlite::params![now, id.as_str()])
                .map_err(|e| db_err("record_access", &e))?;
        }
        Ok(())
    }
}

use crate::helpers::OptionalExt as _;

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::significant_drop_tightening
)]
mod tests {
    use chrono::Utc;
    use freebird_traits::id::KnowledgeId;
    use freebird_traits::knowledge::{
        KnowledgeEntry, KnowledgeKind, KnowledgeSource, KnowledgeStore,
    };
    use secrecy::SecretString;

    use super::*;
    use crate::sqlite::SqliteDb;

    fn test_store() -> (tempfile::TempDir, SqliteKnowledgeStore) {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("test.db");
        let key = SecretString::from("a".repeat(64));
        let signing_key = ring::hmac::Key::new(ring::hmac::HMAC_SHA256, b"test-signing-key");
        let db = Arc::new(SqliteDb::open(&db_path, &key, signing_key).unwrap());
        (dir, SqliteKnowledgeStore::new(db))
    }

    fn make_entry(content: &str, kind: KnowledgeKind) -> KnowledgeEntry {
        KnowledgeEntry {
            id: KnowledgeId::from_string(uuid::Uuid::new_v4().to_string()),
            kind,
            content: content.into(),
            tags: BTreeSet::from(["test".to_owned()]),
            source: KnowledgeSource::Agent,
            confidence: 0.9,
            session_id: None,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            access_count: 0,
            last_accessed: None,
        }
    }

    #[tokio::test]
    async fn test_store_and_get_roundtrip() {
        let (_dir, store) = test_store();
        let entry = make_entry("test content", KnowledgeKind::LearnedPattern);
        let id = entry.id.clone();
        store.store(entry).await.unwrap();

        let loaded = store.get(&id).await.unwrap().expect("should exist");
        assert_eq!(loaded.content, "test content");
        assert_eq!(loaded.kind, KnowledgeKind::LearnedPattern);
    }

    #[tokio::test]
    async fn test_get_nonexistent_returns_none() {
        let (_dir, store) = test_store();
        let result = store
            .get(&KnowledgeId::from_string("nonexistent"))
            .await
            .unwrap();
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn test_update_modifies_content() {
        let (_dir, store) = test_store();
        let mut entry = make_entry("original", KnowledgeKind::LearnedPattern);
        let id = entry.id.clone();
        store.store(entry.clone()).await.unwrap();

        entry.content = "updated".into();
        entry.updated_at = Utc::now();
        store.update(&entry).await.unwrap();

        let loaded = store.get(&id).await.unwrap().unwrap();
        assert_eq!(loaded.content, "updated");
    }

    #[tokio::test]
    async fn test_update_nonexistent_returns_not_found() {
        let (_dir, store) = test_store();
        let entry = make_entry("content", KnowledgeKind::LearnedPattern);
        let result = store.update(&entry).await;
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            KnowledgeError::NotFound { .. }
        ));
    }

    #[tokio::test]
    async fn test_delete_removes_entry() {
        let (_dir, store) = test_store();
        let entry = make_entry("to delete", KnowledgeKind::LearnedPattern);
        let id = entry.id.clone();
        store.store(entry).await.unwrap();
        store.delete(&id).await.unwrap();

        let result = store.get(&id).await.unwrap();
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn test_delete_nonexistent_returns_not_found() {
        let (_dir, store) = test_store();
        let result = store.delete(&KnowledgeId::from_string("nonexistent")).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_fts5_search_returns_ranked_results() {
        let (_dir, store) = test_store();
        store
            .store(make_entry(
                "Rust programming language is fast",
                KnowledgeKind::LearnedPattern,
            ))
            .await
            .unwrap();
        store
            .store(make_entry(
                "Python is a popular language",
                KnowledgeKind::LearnedPattern,
            ))
            .await
            .unwrap();
        store
            .store(make_entry(
                "unrelated entry about cooking",
                KnowledgeKind::SessionInsight,
            ))
            .await
            .unwrap();

        let results = store.search("programming language", 10).await.unwrap();
        assert!(!results.is_empty());
        assert!(results[0].entry.content.contains("Rust"));
    }

    #[tokio::test]
    async fn test_search_empty_query_returns_empty() {
        let (_dir, store) = test_store();
        store
            .store(make_entry("content", KnowledgeKind::LearnedPattern))
            .await
            .unwrap();
        let results = store.search("", 10).await.unwrap();
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn test_list_by_kind() {
        let (_dir, store) = test_store();
        store
            .store(make_entry("pattern 1", KnowledgeKind::LearnedPattern))
            .await
            .unwrap();
        store
            .store(make_entry("pattern 2", KnowledgeKind::LearnedPattern))
            .await
            .unwrap();
        store
            .store(make_entry("config", KnowledgeKind::SystemConfig))
            .await
            .unwrap();

        let results = store
            .list_by_kind(&KnowledgeKind::LearnedPattern, 10)
            .await
            .unwrap();
        assert_eq!(results.len(), 2);
        assert!(
            results
                .iter()
                .all(|e| e.kind == KnowledgeKind::LearnedPattern)
        );
    }

    #[tokio::test]
    async fn test_list_by_tag() {
        let (_dir, store) = test_store();
        let mut entry = make_entry("tagged entry", KnowledgeKind::LearnedPattern);
        entry.tags = BTreeSet::from(["rust".to_owned(), "performance".to_owned()]);
        store.store(entry).await.unwrap();

        let mut entry2 = make_entry("other entry", KnowledgeKind::LearnedPattern);
        entry2.tags = BTreeSet::from(["python".to_owned()]);
        store.store(entry2).await.unwrap();

        let results = store.list_by_tag("rust", 10).await.unwrap();
        assert_eq!(results.len(), 1);
        assert!(results[0].tags.contains("rust"));
    }

    #[tokio::test]
    async fn test_replace_kind_is_atomic() {
        let (_dir, store) = test_store();
        store
            .store(make_entry("old config 1", KnowledgeKind::SystemConfig))
            .await
            .unwrap();
        store
            .store(make_entry("old config 2", KnowledgeKind::SystemConfig))
            .await
            .unwrap();
        store
            .store(make_entry("pattern", KnowledgeKind::LearnedPattern))
            .await
            .unwrap();

        let new_entries = vec![make_entry("new config", KnowledgeKind::SystemConfig)];
        store
            .replace_kind(&KnowledgeKind::SystemConfig, new_entries)
            .await
            .unwrap();

        let configs = store
            .list_by_kind(&KnowledgeKind::SystemConfig, 10)
            .await
            .unwrap();
        assert_eq!(configs.len(), 1);
        assert_eq!(configs[0].content, "new config");

        let patterns = store
            .list_by_kind(&KnowledgeKind::LearnedPattern, 10)
            .await
            .unwrap();
        assert_eq!(patterns.len(), 1);
    }

    #[tokio::test]
    async fn test_record_access_increments_count() {
        let (_dir, store) = test_store();
        let entry = make_entry("content", KnowledgeKind::LearnedPattern);
        let id = entry.id.clone();
        store.store(entry).await.unwrap();

        store
            .record_access(std::slice::from_ref(&id))
            .await
            .unwrap();
        store
            .record_access(std::slice::from_ref(&id))
            .await
            .unwrap();

        let loaded = store.get(&id).await.unwrap().unwrap();
        assert_eq!(loaded.access_count, 2);
        assert!(loaded.last_accessed.is_some());
    }

    #[tokio::test]
    async fn test_record_access_empty_ids_is_noop() {
        let (_dir, store) = test_store();
        store.record_access(&[]).await.unwrap();
    }
}
