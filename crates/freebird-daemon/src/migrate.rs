//! One-time migration from `FileMemory` JSON files to `SQLite`.
//!
//! Reads `*.json` conversation files from the legacy directory, inserts them
//! into the encrypted `SQLite` database, and moves originals to a `.bak/`
//! subdirectory. Safe to run multiple times (idempotent).

use std::path::Path;

use anyhow::{Context, Result};

use freebird_memory::sqlite::SqliteDb;
use freebird_traits::memory::Conversation;

/// Summary of a migration run.
#[derive(Debug)]
pub struct MigrationReport {
    /// Number of conversations successfully migrated.
    pub migrated: usize,
    /// Number of conversations skipped (already in database).
    pub skipped: usize,
    /// Files that failed to migrate: `(filename, error_message)`.
    pub failed: Vec<(String, String)>,
}

impl MigrationReport {
    const fn empty() -> Self {
        Self {
            migrated: 0,
            skipped: 0,
            failed: Vec::new(),
        }
    }
}

/// Migrate legacy `FileMemory` JSON conversations into the `SQLite` database.
///
/// For each `*.json` file in `legacy_dir`:
/// 1. Deserialize as [`Conversation`]
/// 2. Skip if the session already exists in the database
/// 3. Insert into the `conversations` table
/// 4. Move the original file to `{legacy_dir}/.bak/`
///
/// Returns a [`MigrationReport`] summarizing the results.
pub async fn migrate_file_conversations(
    db: &SqliteDb,
    legacy_dir: &Path,
) -> Result<MigrationReport> {
    if !legacy_dir.is_dir() {
        return Ok(MigrationReport::empty());
    }

    let mut entries = Vec::new();
    for dir_entry in std::fs::read_dir(legacy_dir)
        .with_context(|| format!("reading legacy directory {}", legacy_dir.display()))?
    {
        match dir_entry {
            Ok(e) if e.path().extension().is_some_and(|ext| ext == "json") => {
                entries.push(e);
            }
            Ok(_) => {} // non-JSON file, skip
            Err(e) => {
                tracing::warn!(error = %e, "failed to read directory entry in legacy dir");
            }
        }
    }

    if entries.is_empty() {
        return Ok(MigrationReport::empty());
    }

    let bak_dir = legacy_dir.join(".bak");
    let mut report = MigrationReport::empty();

    for entry in &entries {
        let path = entry.path();
        let filename = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("<invalid>")
            .to_owned();

        // Read and parse
        let data = match std::fs::read_to_string(&path) {
            Ok(d) => d,
            Err(e) => {
                tracing::warn!(%filename, error = %e, "failed to read legacy conversation file");
                report.failed.push((filename, e.to_string()));
                continue;
            }
        };

        let conversation: Conversation = match serde_json::from_str(&data) {
            Ok(c) => c,
            Err(e) => {
                tracing::warn!(%filename, error = %e, "failed to parse legacy conversation file");
                report.failed.push((filename, e.to_string()));
                continue;
            }
        };

        let sid = conversation.session_id.as_str().to_owned();

        // Check if already in DB
        let exists = {
            let conn = db.conn().await;
            conn.query_row(
                "SELECT EXISTS(SELECT 1 FROM conversations WHERE session_id = ?1)",
                rusqlite::params![sid],
                |row| row.get::<_, bool>(0),
            )
            .with_context(|| format!("checking existence of session {sid}"))?
        };

        if exists {
            tracing::debug!(%sid, "session already in database, skipping");
            report.skipped += 1;
            continue;
        }

        // Serialize turns for the data column
        let turns_json = serde_json::to_string(&conversation.turns)
            .with_context(|| format!("serializing turns for session {sid}"))?;

        // Insert into DB
        {
            let conn = db.conn().await;
            conn.execute(
                "INSERT INTO conversations \
                 (session_id, system_prompt, model_id, provider_id, created_at, updated_at, data) \
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
                rusqlite::params![
                    sid,
                    conversation.system_prompt,
                    conversation.model_id.as_str(),
                    conversation.provider_id.as_str(),
                    conversation.created_at.to_rfc3339(),
                    conversation.updated_at.to_rfc3339(),
                    turns_json,
                ],
            )
            .with_context(|| format!("inserting session {sid}"))?;
        }

        // Move original to backup
        std::fs::create_dir_all(&bak_dir)
            .with_context(|| format!("creating backup directory {}", bak_dir.display()))?;
        let bak_path = bak_dir.join(&filename);
        std::fs::rename(&path, &bak_path)
            .with_context(|| format!("moving {filename} to backup"))?;

        tracing::info!(%sid, %filename, "migrated legacy conversation to SQLite");
        report.migrated += 1;
    }

    Ok(report)
}

/// Migrate existing blob conversations to event-sourced format.
///
/// For each session in the `conversations` table that has no events in
/// `conversation_events`, emits `SessionCreated`, `TurnStarted`,
/// `AssistantMessage`, `ToolInvoked`, and `TurnCompleted` events with
/// HMAC chain integrity. Also upserts `session_metadata`.
///
/// Safe to run multiple times (idempotent).
#[allow(clippy::too_many_lines)]
pub async fn migrate_blob_to_events(db: &SqliteDb) -> Result<BlobMigrationReport> {
    let conn = db.conn().await;

    // Find sessions that have blob data but no events yet
    let sessions: Vec<String> = {
        let mut stmt = conn
            .prepare(
                "SELECT c.session_id FROM conversations c \
                 LEFT JOIN conversation_events e ON c.session_id = e.session_id \
                 WHERE e.session_id IS NULL \
                 GROUP BY c.session_id",
            )
            .context("prepare blob migration query")?;

        let rows = stmt
            .query_map([], |row| row.get(0))
            .context("query sessions for blob migration")?;

        let mut sids = Vec::new();
        for row in rows {
            sids.push(row.context("read session_id")?);
        }
        sids
    };

    if sessions.is_empty() {
        return Ok(BlobMigrationReport { migrated: 0 });
    }

    let mut migrated = 0;

    for sid in &sessions {
        // Load the blob
        let row: Option<BlobRow> = conn
            .query_row(
                "SELECT session_id, system_prompt, model_id, provider_id, \
                 created_at, updated_at, data FROM conversations WHERE session_id = ?1",
                rusqlite::params![sid],
                |row| {
                    Ok(BlobRow {
                        session_id: row.get(0)?,
                        system_prompt: row.get(1)?,
                        model_id: row.get(2)?,
                        provider_id: row.get(3)?,
                        created_at: row.get(4)?,
                        updated_at: row.get(5)?,
                        data: row.get(6)?,
                    })
                },
            )
            .optional()
            .context("query blob conversation")?;

        let Some(row) = row else { continue };

        let turns: Vec<freebird_traits::memory::Turn> = match serde_json::from_str(&row.data) {
            Ok(t) => t,
            Err(e) => {
                tracing::warn!(session_id = %row.session_id, error = %e, "failed to parse blob turns, skipping");
                continue;
            }
        };

        // Emit events for this session
        let mut sequence: i64 = 0;
        let mut prev_hmac = String::new();

        // SessionCreated
        let event = freebird_traits::event::ConversationEvent::SessionCreated {
            system_prompt: row.system_prompt.clone(),
            model_id: row.model_id.clone(),
            provider_id: row.provider_id.clone(),
        };
        prev_hmac = insert_migration_event(
            &conn,
            &row.session_id,
            sequence,
            &event,
            &row.created_at,
            &prev_hmac,
            db.signing_key(),
        )?;
        sequence += 1;

        for (turn_idx, turn) in turns.iter().enumerate() {
            // TurnStarted
            let event = freebird_traits::event::ConversationEvent::TurnStarted {
                turn_index: turn_idx,
                user_message: turn.user_message.clone(),
            };
            prev_hmac = insert_migration_event(
                &conn,
                &row.session_id,
                sequence,
                &event,
                &turn.started_at.to_rfc3339(),
                &prev_hmac,
                db.signing_key(),
            )?;
            sequence += 1;

            // AssistantMessages
            for (msg_idx, msg) in turn.assistant_messages.iter().enumerate() {
                let event = freebird_traits::event::ConversationEvent::AssistantMessage {
                    turn_index: turn_idx,
                    message_index: msg_idx,
                    message: msg.clone(),
                };
                prev_hmac = insert_migration_event(
                    &conn,
                    &row.session_id,
                    sequence,
                    &event,
                    &msg.timestamp.to_rfc3339(),
                    &prev_hmac,
                    db.signing_key(),
                )?;
                sequence += 1;
            }

            // ToolInvocations
            for (inv_idx, inv) in turn.tool_invocations.iter().enumerate() {
                let event = freebird_traits::event::ConversationEvent::ToolInvoked {
                    turn_index: turn_idx,
                    invocation_index: inv_idx,
                    invocation: inv.clone(),
                };
                let ts = turn.completed_at.unwrap_or(turn.started_at).to_rfc3339();
                prev_hmac = insert_migration_event(
                    &conn,
                    &row.session_id,
                    sequence,
                    &event,
                    &ts,
                    &prev_hmac,
                    db.signing_key(),
                )?;
                sequence += 1;
            }

            // TurnCompleted
            if let Some(completed_at) = turn.completed_at {
                let event = freebird_traits::event::ConversationEvent::TurnCompleted {
                    turn_index: turn_idx,
                    completed_at,
                };
                prev_hmac = insert_migration_event(
                    &conn,
                    &row.session_id,
                    sequence,
                    &event,
                    &completed_at.to_rfc3339(),
                    &prev_hmac,
                    db.signing_key(),
                )?;
                sequence += 1;
            }
        }

        // Upsert session_metadata
        let preview = turns
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
        let turn_count = turns.len() as i64;

        conn.execute(
            "INSERT OR REPLACE INTO session_metadata \
             (session_id, system_prompt, model_id, provider_id, \
              created_at, updated_at, turn_count, preview) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            rusqlite::params![
                row.session_id,
                row.system_prompt,
                row.model_id,
                row.provider_id,
                row.created_at,
                row.updated_at,
                turn_count,
                preview,
            ],
        )
        .with_context(|| format!("upsert session_metadata for {}", row.session_id))?;

        tracing::info!(session_id = %row.session_id, events = sequence, "migrated blob to events");
        migrated += 1;
    }
    drop(conn);

    Ok(BlobMigrationReport { migrated })
}

/// Insert a single migration event with HMAC chain.
///
/// Returns the computed HMAC for chaining.
fn insert_migration_event(
    conn: &rusqlite::Connection,
    session_id: &str,
    sequence: i64,
    event: &freebird_traits::event::ConversationEvent,
    timestamp: &str,
    prev_hmac: &str,
    signing_key: &ring::hmac::Key,
) -> Result<String> {
    let event_type = event.event_type().to_owned();
    let event_json = serde_json::to_string(event).context("serialize migration event")?;

    let hmac_hex = freebird_memory::event::compute_event_hmac(
        session_id,
        sequence,
        event,
        timestamp,
        prev_hmac,
        signing_key,
    )
    .context("compute event HMAC")?;

    conn.execute(
        "INSERT INTO conversation_events \
         (session_id, sequence, event_type, event_data, timestamp, previous_hmac, hmac) \
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
        rusqlite::params![
            session_id, sequence, event_type, event_json, timestamp, prev_hmac, hmac_hex
        ],
    )
    .with_context(|| format!("insert migration event seq={sequence} for {session_id}"))?;

    Ok(hmac_hex)
}

/// Summary of a blob-to-event migration run.
#[derive(Debug)]
pub struct BlobMigrationReport {
    /// Number of sessions converted from blob to event format.
    pub migrated: usize,
}

/// Intermediate row for blob migration.
struct BlobRow {
    session_id: String,
    system_prompt: Option<String>,
    model_id: String,
    provider_id: String,
    created_at: String,
    updated_at: String,
    data: String,
}

use rusqlite::OptionalExtension as _;

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::similar_names,
    clippy::significant_drop_tightening
)]
mod tests {
    use super::*;

    use chrono::Utc;
    use secrecy::SecretString;

    use freebird_traits::id::{ModelId, ProviderId, SessionId};
    use freebird_traits::memory::{Conversation, Turn};
    use freebird_traits::provider::{ContentBlock, Message, Role};

    /// Create a test database with a known key.
    fn test_db(dir: &Path) -> SqliteDb {
        let db_path = dir.join("test.db");
        let key = SecretString::from("a".repeat(64));
        let signing_key = ring::hmac::Key::new(ring::hmac::HMAC_SHA256, b"test-signing-key");
        SqliteDb::open(&db_path, &key, signing_key).unwrap()
    }

    /// Create a minimal valid `Conversation` with the given session ID.
    fn make_conversation(session_id: &str) -> Conversation {
        Conversation {
            session_id: SessionId::from(session_id),
            system_prompt: Some("test prompt".into()),
            turns: vec![Turn {
                user_message: Message {
                    role: Role::User,
                    content: vec![ContentBlock::Text {
                        text: "hello".into(),
                    }],
                    timestamp: Utc::now(),
                },
                assistant_messages: vec![Message {
                    role: Role::Assistant,
                    content: vec![ContentBlock::Text {
                        text: "hi there".into(),
                    }],
                    timestamp: Utc::now(),
                }],
                tool_invocations: vec![],
                started_at: Utc::now(),
                completed_at: Some(Utc::now()),
            }],
            created_at: Utc::now(),
            updated_at: Utc::now(),
            model_id: ModelId::from("test-model"),
            provider_id: ProviderId::from("test-provider"),
        }
    }

    /// Write a conversation as JSON to the given directory.
    fn write_conversation(dir: &Path, session_id: &str) {
        let conv = make_conversation(session_id);
        let json = serde_json::to_string_pretty(&conv).unwrap();
        std::fs::write(dir.join(format!("{session_id}.json")), json).unwrap();
    }

    #[tokio::test]
    async fn test_nonexistent_directory_returns_empty_report() {
        let dir = tempfile::tempdir().unwrap();
        let db = test_db(dir.path());
        let fake_path = dir.path().join("nonexistent");

        let report = migrate_file_conversations(&db, &fake_path).await.unwrap();

        assert_eq!(report.migrated, 0);
        assert_eq!(report.skipped, 0);
        assert!(report.failed.is_empty());
    }

    #[tokio::test]
    async fn test_empty_directory_returns_empty_report() {
        let dir = tempfile::tempdir().unwrap();
        let db = test_db(dir.path());
        let legacy_dir = dir.path().join("conversations");
        std::fs::create_dir(&legacy_dir).unwrap();

        let report = migrate_file_conversations(&db, &legacy_dir).await.unwrap();

        assert_eq!(report.migrated, 0);
        assert_eq!(report.skipped, 0);
        assert!(report.failed.is_empty());
    }

    #[tokio::test]
    async fn test_single_conversation_migrated() {
        let dir = tempfile::tempdir().unwrap();
        let db = test_db(dir.path());
        let legacy_dir = dir.path().join("conversations");
        std::fs::create_dir(&legacy_dir).unwrap();
        write_conversation(&legacy_dir, "sess-1");

        let report = migrate_file_conversations(&db, &legacy_dir).await.unwrap();

        assert_eq!(report.migrated, 1);
        assert_eq!(report.skipped, 0);
        assert!(report.failed.is_empty());

        // Verify row exists in DB
        let conn = db.conn().await;
        let count: i64 = conn
            .query_row(
                "SELECT count(*) FROM conversations WHERE session_id = 'sess-1'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(count, 1);
        drop(conn);

        // Verify original moved to .bak/
        assert!(!legacy_dir.join("sess-1.json").exists());
        assert!(legacy_dir.join(".bak/sess-1.json").exists());
    }

    #[tokio::test]
    async fn test_multiple_conversations_migrated() {
        let dir = tempfile::tempdir().unwrap();
        let db = test_db(dir.path());
        let legacy_dir = dir.path().join("conversations");
        std::fs::create_dir(&legacy_dir).unwrap();
        write_conversation(&legacy_dir, "sess-a");
        write_conversation(&legacy_dir, "sess-b");
        write_conversation(&legacy_dir, "sess-c");

        let report = migrate_file_conversations(&db, &legacy_dir).await.unwrap();

        assert_eq!(report.migrated, 3);
        assert_eq!(report.skipped, 0);
        assert!(report.failed.is_empty());
    }

    #[tokio::test]
    async fn test_malformed_json_counted_as_failed() {
        let dir = tempfile::tempdir().unwrap();
        let db = test_db(dir.path());
        let legacy_dir = dir.path().join("conversations");
        std::fs::create_dir(&legacy_dir).unwrap();
        write_conversation(&legacy_dir, "good");
        std::fs::write(legacy_dir.join("bad.json"), "not valid json {{{").unwrap();

        let report = migrate_file_conversations(&db, &legacy_dir).await.unwrap();

        assert_eq!(report.migrated, 1);
        assert_eq!(report.failed.len(), 1);
        assert_eq!(report.failed[0].0, "bad.json");
    }

    #[tokio::test]
    async fn test_already_existing_session_skipped() {
        let dir = tempfile::tempdir().unwrap();
        let db = test_db(dir.path());
        let legacy_dir = dir.path().join("conversations");
        std::fs::create_dir(&legacy_dir).unwrap();

        // Pre-insert session into DB
        let conv = make_conversation("existing");
        let turns_json = serde_json::to_string(&conv.turns).unwrap();
        {
            let conn = db.conn().await;
            conn.execute(
                "INSERT INTO conversations \
                 (session_id, system_prompt, model_id, provider_id, created_at, updated_at, data) \
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
                rusqlite::params![
                    "existing",
                    conv.system_prompt,
                    conv.model_id.as_str(),
                    conv.provider_id.as_str(),
                    conv.created_at.to_rfc3339(),
                    conv.updated_at.to_rfc3339(),
                    turns_json,
                ],
            )
            .unwrap();
        }

        // Write JSON file with same session ID
        write_conversation(&legacy_dir, "existing");

        let report = migrate_file_conversations(&db, &legacy_dir).await.unwrap();

        assert_eq!(report.migrated, 0);
        assert_eq!(report.skipped, 1);
        // File should NOT be moved
        assert!(legacy_dir.join("existing.json").exists());
    }

    #[tokio::test]
    async fn test_idempotent_second_run() {
        let dir = tempfile::tempdir().unwrap();
        let db = test_db(dir.path());
        let legacy_dir = dir.path().join("conversations");
        std::fs::create_dir(&legacy_dir).unwrap();
        write_conversation(&legacy_dir, "sess-1");
        write_conversation(&legacy_dir, "sess-2");

        // First run
        let r1 = migrate_file_conversations(&db, &legacy_dir).await.unwrap();
        assert_eq!(r1.migrated, 2);

        // Add one more file
        write_conversation(&legacy_dir, "sess-3");

        // Second run — previous files already moved to .bak
        let r2 = migrate_file_conversations(&db, &legacy_dir).await.unwrap();
        assert_eq!(r2.migrated, 1);
        assert_eq!(r2.skipped, 0);
    }

    #[tokio::test]
    async fn test_non_json_files_ignored() {
        let dir = tempfile::tempdir().unwrap();
        let db = test_db(dir.path());
        let legacy_dir = dir.path().join("conversations");
        std::fs::create_dir(&legacy_dir).unwrap();
        write_conversation(&legacy_dir, "real");
        std::fs::write(legacy_dir.join("readme.txt"), "not a conversation").unwrap();
        std::fs::write(legacy_dir.join("data.csv"), "a,b,c").unwrap();

        let report = migrate_file_conversations(&db, &legacy_dir).await.unwrap();

        assert_eq!(report.migrated, 1);
        assert!(report.failed.is_empty());
        // Non-JSON files should still be there
        assert!(legacy_dir.join("readme.txt").exists());
        assert!(legacy_dir.join("data.csv").exists());
    }

    #[tokio::test]
    async fn test_backup_directory_created() {
        let dir = tempfile::tempdir().unwrap();
        let db = test_db(dir.path());
        let legacy_dir = dir.path().join("conversations");
        std::fs::create_dir(&legacy_dir).unwrap();
        write_conversation(&legacy_dir, "sess-1");

        let bak_dir = legacy_dir.join(".bak");
        assert!(!bak_dir.exists());

        migrate_file_conversations(&db, &legacy_dir).await.unwrap();

        assert!(bak_dir.is_dir());
    }

    #[tokio::test]
    async fn test_data_integrity_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let db = test_db(dir.path());
        let legacy_dir = dir.path().join("conversations");
        std::fs::create_dir(&legacy_dir).unwrap();

        // Create a specific conversation
        let original = make_conversation("integrity-check");
        let json = serde_json::to_string_pretty(&original).unwrap();
        std::fs::write(legacy_dir.join("integrity-check.json"), json).unwrap();

        migrate_file_conversations(&db, &legacy_dir).await.unwrap();

        // Load from DB and verify fields
        let conn = db.conn().await;
        let row = conn
            .query_row(
                "SELECT session_id, system_prompt, model_id, provider_id, \
                 created_at, updated_at, data FROM conversations \
                 WHERE session_id = 'integrity-check'",
                [],
                |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, Option<String>>(1)?,
                        row.get::<_, String>(2)?,
                        row.get::<_, String>(3)?,
                        row.get::<_, String>(4)?,
                        row.get::<_, String>(5)?,
                        row.get::<_, String>(6)?,
                    ))
                },
            )
            .unwrap();

        assert_eq!(row.0, "integrity-check");
        assert_eq!(row.1.as_deref(), original.system_prompt.as_deref());
        assert_eq!(row.2, original.model_id.as_str());
        assert_eq!(row.3, original.provider_id.as_str());
        assert_eq!(row.4, original.created_at.to_rfc3339());
        assert_eq!(row.5, original.updated_at.to_rfc3339());

        // Verify turns roundtrip
        let turns: Vec<Turn> = serde_json::from_str(&row.6).unwrap();
        assert_eq!(turns.len(), original.turns.len());
    }

    // --- Blob-to-event migration tests ---

    /// Insert a blob conversation directly into the `conversations` table.
    async fn insert_blob(db: &SqliteDb, conversation: &Conversation) {
        let conn = db.conn().await;
        let turns_json = serde_json::to_string(&conversation.turns).unwrap();
        conn.execute(
            "INSERT INTO conversations \
             (session_id, system_prompt, model_id, provider_id, created_at, updated_at, data) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            rusqlite::params![
                conversation.session_id.as_str(),
                conversation.system_prompt,
                conversation.model_id.as_str(),
                conversation.provider_id.as_str(),
                conversation.created_at.to_rfc3339(),
                conversation.updated_at.to_rfc3339(),
                turns_json,
            ],
        )
        .unwrap();
    }

    #[tokio::test]
    async fn test_blob_to_events_migrates_single_session() {
        let dir = tempfile::tempdir().unwrap();
        let db = test_db(dir.path());
        let conversation = make_conversation("blob-sess-1");
        insert_blob(&db, &conversation).await;

        let report = migrate_blob_to_events(&db).await.unwrap();
        assert_eq!(report.migrated, 1);

        // Verify events were created
        let conn = db.conn().await;
        let count: i64 = conn
            .query_row(
                "SELECT count(*) FROM conversation_events WHERE session_id = 'blob-sess-1'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        // SessionCreated + TurnStarted + AssistantMessage + TurnCompleted = 4
        assert_eq!(count, 4);

        // Verify session_metadata was created
        let preview: String = conn
            .query_row(
                "SELECT preview FROM session_metadata WHERE session_id = 'blob-sess-1'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(preview, "hello");
    }

    #[tokio::test]
    async fn test_blob_to_events_idempotent() {
        let dir = tempfile::tempdir().unwrap();
        let db = test_db(dir.path());
        let conversation = make_conversation("blob-idem");
        insert_blob(&db, &conversation).await;

        let r1 = migrate_blob_to_events(&db).await.unwrap();
        assert_eq!(r1.migrated, 1);

        // Second run should find no sessions to migrate (events already exist)
        let r2 = migrate_blob_to_events(&db).await.unwrap();
        assert_eq!(r2.migrated, 0);
    }

    #[tokio::test]
    async fn test_blob_to_events_hmac_chain_valid() {
        let dir = tempfile::tempdir().unwrap();
        let db = test_db(dir.path());
        let conversation = make_conversation("blob-hmac");
        insert_blob(&db, &conversation).await;

        migrate_blob_to_events(&db).await.unwrap();

        // Verify the HMAC chain using the event module's verification
        let conn = db.conn().await;
        let mut stmt = conn
            .prepare(
                "SELECT session_id, sequence, event_data, timestamp, previous_hmac, hmac \
                 FROM conversation_events WHERE session_id = 'blob-hmac' ORDER BY sequence",
            )
            .unwrap();
        let rows: Vec<(String, i64, String, String, String, String)> = stmt
            .query_map([], |row| {
                Ok((
                    row.get(0)?,
                    row.get(1)?,
                    row.get(2)?,
                    row.get(3)?,
                    row.get(4)?,
                    row.get(5)?,
                ))
            })
            .unwrap()
            .map(|r| r.unwrap())
            .collect();

        // Verify chain linkage
        for i in 1..rows.len() {
            assert_eq!(rows[i].4, rows[i - 1].5, "HMAC chain broken at seq {i}");
        }
        assert_eq!(rows[0].4, "", "First event should have empty previous_hmac");
    }

    #[tokio::test]
    async fn test_blob_to_events_no_blobs_returns_zero() {
        let dir = tempfile::tempdir().unwrap();
        let db = test_db(dir.path());

        let report = migrate_blob_to_events(&db).await.unwrap();
        assert_eq!(report.migrated, 0);
    }
}
