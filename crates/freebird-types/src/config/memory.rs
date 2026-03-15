//! Memory and knowledge configuration structs.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

/// Memory backend configuration (`SQLCipher` encrypted `SQLite`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryConfig {
    /// Path to the `SQLite` database file. Default: `~/.freebird/freebird.db`.
    pub db_path: Option<PathBuf>,
    /// Path to the encryption keyfile. If not set, falls back to
    /// `FREEBIRD_DB_KEY` env var or interactive prompt.
    pub keyfile_path: Option<PathBuf>,
    /// PBKDF2 iteration count for key derivation. Default: 100,000.
    #[serde(default = "default_pbkdf2_iterations")]
    pub pbkdf2_iterations: u32,
    /// Verify HMAC chain integrity when loading conversations. Default: true.
    ///
    /// When enabled, every `Memory::load()` call verifies the per-session HMAC
    /// chain before replaying events. Tampered or corrupted events produce
    /// `MemoryError::IntegrityViolation`. Disable only for recovery/debugging.
    #[serde(default = "default_verify_on_load")]
    pub verify_on_load: bool,
}

const fn default_pbkdf2_iterations() -> u32 {
    100_000
}

const fn default_verify_on_load() -> bool {
    true
}

/// Knowledge retrieval behavior configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeConfig {
    /// Whether to auto-retrieve knowledge on every user message.
    #[serde(default = "default_auto_retrieve")]
    pub auto_retrieve: bool,
    /// Max entries injected into context per message.
    #[serde(default = "default_max_context_entries")]
    pub max_context_entries: usize,
    /// BM25 rank threshold. Entries with rank worse (higher) than this are excluded.
    /// FTS5 BM25: lower (more negative) = more relevant. Default: -0.5.
    #[serde(default = "default_relevance_threshold")]
    pub relevance_threshold: f64,
    /// Max approximate tokens for injected knowledge context.
    #[serde(default = "default_max_context_tokens")]
    pub max_context_tokens: usize,
}

impl Default for KnowledgeConfig {
    fn default() -> Self {
        Self {
            auto_retrieve: default_auto_retrieve(),
            max_context_entries: default_max_context_entries(),
            relevance_threshold: default_relevance_threshold(),
            max_context_tokens: default_max_context_tokens(),
        }
    }
}

const fn default_auto_retrieve() -> bool {
    true
}

const fn default_max_context_entries() -> usize {
    5
}

const fn default_relevance_threshold() -> f64 {
    -0.5
}

const fn default_max_context_tokens() -> usize {
    2000
}
