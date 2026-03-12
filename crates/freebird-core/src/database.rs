//! Database initialization with `SQLCipher` encryption and key derivation.

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result};
use secrecy::{ExposeSecret as _, SecretString};

use freebird_memory::sqlite::SqliteDb;
use freebird_memory::sqlite_knowledge::SqliteKnowledgeStore;
use freebird_memory::sqlite_memory::SqliteMemory;
use freebird_types::config::AppConfig;

use crate::util::expand_tilde;

/// Strategy for resolving the database encryption passphrase.
pub enum PassphraseStrategy {
    /// Try env var → keyfile → interactive prompt (production default).
    AutoResolve {
        /// Whether to prompt interactively if env var and keyfile are absent.
        allow_prompt: bool,
    },
    /// Use a specific passphrase (for tests).
    Provided(SecretString),
}

/// All database-backed subsystem handles returned from initialization.
pub struct DatabaseComponents {
    /// Conversation persistence (Memory trait).
    pub memory: Arc<dyn freebird_traits::memory::Memory>,
    /// Knowledge store for tool capabilities and agent learning.
    pub knowledge_store: Option<Arc<dyn freebird_traits::knowledge::KnowledgeStore>>,
    /// Event-sourced conversation persistence with HMAC chains.
    pub event_sink: Option<Arc<dyn freebird_traits::event::EventSink>>,
    /// Security audit log with HMAC chain integrity.
    pub audit_sink: Option<Arc<dyn freebird_traits::audit::AuditSink>>,
    /// Raw database handle (for constructing additional components via `SqliteDb`).
    pub db: Arc<SqliteDb>,
}

/// Initialize the SQLite-backed memory, knowledge, event, and audit subsystems.
///
/// Opens (or creates) an encrypted `SQLCipher` database, derives the key via
/// PBKDF2-HMAC-SHA256, and returns all database-backed components.
pub fn init_database(
    config: &AppConfig,
    passphrase_strategy: &PassphraseStrategy,
) -> Result<DatabaseComponents> {
    let db_path = config
        .memory
        .db_path
        .as_ref()
        .map(|p| expand_tilde(p))
        .transpose()?
        .unwrap_or_else(|| {
            home::home_dir().map_or_else(
                || PathBuf::from(".freebird/freebird.db"),
                |h| h.join(".freebird/freebird.db"),
            )
        });

    let salt_path = db_path.with_extension("salt");
    let salt = freebird_security::db_key::load_or_create_salt(&salt_path)
        .context("failed to load or create database salt")?;

    let passphrase = match passphrase_strategy {
        PassphraseStrategy::AutoResolve { allow_prompt } => {
            freebird_security::db_key::resolve_passphrase(
                config.memory.keyfile_path.as_deref(),
                *allow_prompt,
            )
            .context("failed to resolve database encryption key")?
        }
        PassphraseStrategy::Provided(secret) => secret.clone(),
    };

    let key =
        freebird_security::db_key::derive_key(&passphrase, &salt, config.memory.pbkdf2_iterations)
            .context("failed to derive database encryption key")?;

    // Derive HMAC signing key from passphrase + salt for event and audit chain integrity.
    let signing_tag = ring::hmac::sign(
        &ring::hmac::Key::new(
            ring::hmac::HMAC_SHA256,
            passphrase.expose_secret().as_bytes(),
        ),
        format!("freebird-event-signing|{}", hex::encode(&salt)).as_bytes(),
    );
    let signing_key = ring::hmac::Key::new(ring::hmac::HMAC_SHA256, signing_tag.as_ref());

    let db =
        SqliteDb::open(&db_path, &key, signing_key).context("failed to open encrypted database")?;
    let db = Arc::new(db);

    tracing::info!(path = %db_path.display(), "encrypted database opened");

    let memory: Arc<dyn freebird_traits::memory::Memory> = Arc::new(SqliteMemory::new(
        Arc::clone(&db),
        config.memory.verify_on_load,
    ));
    let knowledge_store: Arc<dyn freebird_traits::knowledge::KnowledgeStore> =
        Arc::new(SqliteKnowledgeStore::new(Arc::clone(&db)));

    let event_sink: Arc<dyn freebird_traits::event::EventSink> = Arc::new(
        freebird_memory::sqlite_event::SqliteEventSink::new(Arc::clone(&db)),
    );
    let audit_sink: Arc<dyn freebird_traits::audit::AuditSink> = Arc::new(
        freebird_memory::sqlite_audit::SqliteAuditSink::new(Arc::clone(&db)),
    );

    Ok(DatabaseComponents {
        memory,
        knowledge_store: Some(knowledge_store),
        event_sink: Some(event_sink),
        audit_sink: Some(audit_sink),
        db,
    })
}
