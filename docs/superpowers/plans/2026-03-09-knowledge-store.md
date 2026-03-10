# Knowledge Store Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace FileMemory with an encrypted SQLite backend and add an FTS5-powered knowledge store for cross-session agent learning.

**Architecture:** Single SQLCipher-encrypted SQLite database replaces JSON file storage. New `KnowledgeStore` trait in `freebird-traits`, implemented over SQLite in `freebird-memory`. Hybrid retrieval: auto-inject on every user message + explicit search tool. Sensitive content filter + consent-gated writes for protected knowledge kinds.

**Tech Stack:** `rusqlite` with `bundled-sqlcipher`, `ring` (PBKDF2), `secrecy` (key zeroization), FTS5 with Porter stemming.

**Design Spec:** `docs/superpowers/specs/2026-03-09-knowledge-store-design.md`
**GitHub Issue:** #68

---

## Chunk 1: Foundation — Traits, Types, Config (Tasks 1-2)

### Task 1: KnowledgeStore Trait + Types

**Files:**
- Create: `crates/freebird-traits/src/knowledge.rs`
- Modify: `crates/freebird-traits/src/id.rs:46-69` (add `KnowledgeId`)
- Modify: `crates/freebird-traits/src/lib.rs:12-16` (add `pub mod knowledge`)

**Context:** The `freebird-traits` crate has ZERO `freebird-*` dependencies. It uses `async-trait`, `serde`, `serde_json`, `thiserror`, `chrono`. The existing pattern for IDs is the `define_id!` macro in `id.rs`. The existing trait pattern is in `memory.rs` (see `Memory` trait).

- [ ] **Step 1: Add `KnowledgeId` to `id.rs`**

After the existing `ModelId` definition (line 69), add:

```rust
define_id!(
    /// Identifies a knowledge entry in the knowledge store.
    KnowledgeId
);
```

- [ ] **Step 2: Create `knowledge.rs` with types and trait**

Create `crates/freebird-traits/src/knowledge.rs`:

```rust
//! Knowledge store trait — abstracts over persistent agent knowledge backends.

use std::collections::BTreeSet;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::id::{KnowledgeId, SessionId};

/// Categories of knowledge the agent can store and retrieve.
///
/// # Consent rules
///
/// `SystemConfig`, `ToolCapability`, and `UserPreference` require human consent
/// for any write, update, or delete. Agent-owned kinds (`LearnedPattern`,
/// `ErrorResolution`, `SessionInsight`) can be modified autonomously.
///
/// # Variant ordering contract
///
/// `Ord` is derived — do not reorder existing variants. Append new variants at the end.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KnowledgeKind {
    SystemConfig,
    ToolCapability,
    UserPreference,
    LearnedPattern,
    ErrorResolution,
    SessionInsight,
}

impl KnowledgeKind {
    /// Whether modifying entries of this kind requires human consent.
    #[must_use]
    pub fn requires_consent(&self) -> bool {
        matches!(
            self,
            Self::SystemConfig | Self::ToolCapability | Self::UserPreference
        )
    }

    /// Whether the agent owns entries of this kind (can write without consent).
    #[must_use]
    pub fn agent_owned(&self) -> bool {
        !self.requires_consent()
    }
}

/// Who created a knowledge entry.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KnowledgeSource {
    /// Auto-populated by the system at startup.
    System,
    /// Explicitly declared by the user (via consent-gated tool call).
    User,
    /// Inferred by the agent during conversation.
    Agent,
}

/// A single knowledge entry in the knowledge store.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeEntry {
    pub id: KnowledgeId,
    pub kind: KnowledgeKind,
    pub content: String,
    pub tags: BTreeSet<String>,
    pub source: KnowledgeSource,
    /// Confidence score (0.0–1.0). System entries default to 1.0.
    pub confidence: f32,
    /// Session that created this entry. `None` for system-populated entries.
    pub session_id: Option<SessionId>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    /// How many times this entry was retrieved for context injection.
    pub access_count: u64,
    /// Last time this entry was retrieved for context injection.
    pub last_accessed: Option<DateTime<Utc>>,
}

/// A ranked search result from the knowledge store.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeMatch {
    pub entry: KnowledgeEntry,
    /// BM25 relevance score. Lower (more negative) = more relevant.
    pub rank: f64,
}

/// The core knowledge store trait.
#[async_trait]
pub trait KnowledgeStore: Send + Sync + 'static {
    /// Store a new knowledge entry. Returns the assigned ID.
    async fn store(&self, entry: KnowledgeEntry) -> Result<KnowledgeId, KnowledgeError>;

    /// Update an existing entry's content, tags, or confidence.
    async fn update(&self, entry: &KnowledgeEntry) -> Result<(), KnowledgeError>;

    /// Retrieve a single entry by ID.
    async fn get(&self, id: &KnowledgeId) -> Result<Option<KnowledgeEntry>, KnowledgeError>;

    /// Delete an entry by ID.
    async fn delete(&self, id: &KnowledgeId) -> Result<(), KnowledgeError>;

    /// FTS5 ranked search. Returns entries ordered by BM25 relevance.
    async fn search(
        &self,
        query: &str,
        limit: usize,
    ) -> Result<Vec<KnowledgeMatch>, KnowledgeError>;

    /// List entries filtered by kind, ordered by `updated_at` descending.
    async fn list_by_kind(
        &self,
        kind: &KnowledgeKind,
        limit: usize,
    ) -> Result<Vec<KnowledgeEntry>, KnowledgeError>;

    /// List entries that contain a specific tag.
    async fn list_by_tag(
        &self,
        tag: &str,
        limit: usize,
    ) -> Result<Vec<KnowledgeEntry>, KnowledgeError>;

    /// Replace all entries of a given kind (for system auto-population).
    ///
    /// Deletes all existing entries with `kind`, then inserts `entries`.
    /// Runs in a single transaction for atomicity.
    async fn replace_kind(
        &self,
        kind: &KnowledgeKind,
        entries: Vec<KnowledgeEntry>,
    ) -> Result<(), KnowledgeError>;

    /// Record that entries were accessed (bumps `access_count`, sets `last_accessed`).
    async fn record_access(&self, ids: &[KnowledgeId]) -> Result<(), KnowledgeError>;
}

/// Knowledge store errors.
#[derive(Debug, thiserror::Error)]
pub enum KnowledgeError {
    #[error("knowledge entry `{id}` not found")]
    NotFound { id: KnowledgeId },

    #[error("storage I/O error: {0}")]
    Io(String),

    #[error("serialization error: {0}")]
    Serialization(String),

    #[error("database error: {0}")]
    Database(String),
}
```

- [ ] **Step 3: Register the module in `lib.rs`**

Add `pub mod knowledge;` to `crates/freebird-traits/src/lib.rs` (after line 14, before `pub mod provider`).

- [ ] **Step 4: Write tests**

Add a `#[cfg(test)]` module at the bottom of `knowledge.rs`:

```rust
#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn test_knowledge_kind_requires_consent() {
        assert!(KnowledgeKind::SystemConfig.requires_consent());
        assert!(KnowledgeKind::ToolCapability.requires_consent());
        assert!(KnowledgeKind::UserPreference.requires_consent());
        assert!(!KnowledgeKind::LearnedPattern.requires_consent());
        assert!(!KnowledgeKind::ErrorResolution.requires_consent());
        assert!(!KnowledgeKind::SessionInsight.requires_consent());
    }

    #[test]
    fn test_knowledge_kind_agent_owned_inverse_of_requires_consent() {
        for kind in [
            KnowledgeKind::SystemConfig,
            KnowledgeKind::ToolCapability,
            KnowledgeKind::UserPreference,
            KnowledgeKind::LearnedPattern,
            KnowledgeKind::ErrorResolution,
            KnowledgeKind::SessionInsight,
        ] {
            assert_eq!(kind.agent_owned(), !kind.requires_consent());
        }
    }

    #[test]
    fn test_knowledge_kind_serde_roundtrip() {
        for (kind, expected) in [
            (KnowledgeKind::SystemConfig, "\"system_config\""),
            (KnowledgeKind::ToolCapability, "\"tool_capability\""),
            (KnowledgeKind::UserPreference, "\"user_preference\""),
            (KnowledgeKind::LearnedPattern, "\"learned_pattern\""),
            (KnowledgeKind::ErrorResolution, "\"error_resolution\""),
            (KnowledgeKind::SessionInsight, "\"session_insight\""),
        ] {
            let json = serde_json::to_string(&kind).unwrap();
            assert_eq!(json, expected);
            let back: KnowledgeKind = serde_json::from_str(&json).unwrap();
            assert_eq!(back, kind);
        }
    }

    #[test]
    fn test_knowledge_source_serde_roundtrip() {
        for (source, expected) in [
            (KnowledgeSource::System, "\"system\""),
            (KnowledgeSource::User, "\"user\""),
            (KnowledgeSource::Agent, "\"agent\""),
        ] {
            let json = serde_json::to_string(&source).unwrap();
            assert_eq!(json, expected);
            let back: KnowledgeSource = serde_json::from_str(&json).unwrap();
            assert_eq!(back, source);
        }
    }

    #[test]
    fn test_knowledge_kind_ordering() {
        assert!(KnowledgeKind::SystemConfig < KnowledgeKind::ToolCapability);
        assert!(KnowledgeKind::ToolCapability < KnowledgeKind::UserPreference);
        assert!(KnowledgeKind::UserPreference < KnowledgeKind::LearnedPattern);
    }

    #[test]
    fn test_knowledge_error_display() {
        let err = KnowledgeError::NotFound {
            id: KnowledgeId::from_string("test-id"),
        };
        assert_eq!(err.to_string(), "knowledge entry `test-id` not found");

        let err = KnowledgeError::Database("connection failed".into());
        assert_eq!(err.to_string(), "database error: connection failed");
    }
}
```

- [ ] **Step 5: Verify**

Run: `cargo test -p freebird-traits`
Run: `cargo clippy -p freebird-traits`

- [ ] **Step 6: Commit**

```bash
git add crates/freebird-traits/src/knowledge.rs crates/freebird-traits/src/id.rs crates/freebird-traits/src/lib.rs
git commit -m "feat(traits): add KnowledgeStore trait, types, and KnowledgeId"
```

---

### Task 2: Config Changes

**Files:**
- Modify: `crates/freebird-types/src/config.rs:135-148` (replace `MemoryConfig`, remove `MemoryKind`)
- Modify: `crates/freebird-types/src/config.rs:10-21` (add `knowledge` field to `AppConfig`)
- Modify: `config/default.toml:29-31` (update `[memory]` section, add `[knowledge]` section)

**Context:** `MemoryKind` enum at line 138 has `File` and `Sqlite` variants. `MemoryConfig` at line 144 has `kind` and `base_dir`. We're removing `MemoryKind` entirely (SQLite is the only backend) and restructuring `MemoryConfig` for SQLCipher. Tests start at line 257.

- [ ] **Step 1: Replace `MemoryKind` and `MemoryConfig`**

Replace lines 135-148 in `config.rs`:

Old:
```rust
/// Which memory storage backend to use.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryKind {
    File,
    Sqlite,
}

/// Memory backend configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryConfig {
    pub kind: MemoryKind,
    pub base_dir: Option<PathBuf>,
}
```

New:
```rust
/// Memory backend configuration (SQLCipher encrypted SQLite).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryConfig {
    /// Path to the SQLite database file. Default: `~/.freebird/freebird.db`.
    pub db_path: Option<PathBuf>,
    /// Path to the encryption keyfile. If not set, falls back to
    /// `FREEBIRD_DB_KEY` env var or interactive prompt.
    pub keyfile_path: Option<PathBuf>,
    /// PBKDF2 iteration count for key derivation. Default: 100,000.
    #[serde(default = "default_pbkdf2_iterations")]
    pub pbkdf2_iterations: u32,
}

const fn default_pbkdf2_iterations() -> u32 {
    100_000
}
```

- [ ] **Step 2: Add `KnowledgeConfig` struct**

Add after the new `MemoryConfig` (and its default fn):

```rust
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
```

- [ ] **Step 3: Add `knowledge` field to `AppConfig`**

In the `AppConfig` struct (line 11-22), add after `memory`:

```rust
#[serde(default)]
pub knowledge: KnowledgeConfig,
```

- [ ] **Step 4: Update `default.toml`**

Replace the `[memory]` section (lines 29-31):

Old:
```toml
[memory]
kind = "file"
base_dir = "~/.freebird/conversations"
```

New:
```toml
[memory]
db_path = "~/.freebird/freebird.db"
# keyfile_path = "~/.freebird/db.key"
# pbkdf2_iterations = 100000

[knowledge]
auto_retrieve = true
max_context_entries = 5
relevance_threshold = -0.5
max_context_tokens = 2000
```

- [ ] **Step 5: Fix all existing tests**

The existing tests reference `MemoryKind`, `kind = "file"`, and `base_dir`. Update:

1. `config_toml()` helper (line ~322): change the memory block from `kind = "file"` to `db_path = "~/.freebird/freebird.db"`
2. `test_default_config_deserializes` (line ~340): remove assertions about `MemoryKind`, add assertions for new memory fields
3. Any test that references `config.memory.kind` or `config.memory.base_dir` — update to `config.memory.db_path`

Add new tests:

```rust
#[test]
fn test_knowledge_config_defaults_when_absent() {
    let toml_str = config_toml(&[]);
    let config: AppConfig = toml::from_str(&toml_str).unwrap();
    assert!(config.knowledge.auto_retrieve);
    assert_eq!(config.knowledge.max_context_entries, 5);
    assert!((config.knowledge.relevance_threshold - (-0.5)).abs() < f64::EPSILON);
    assert_eq!(config.knowledge.max_context_tokens, 2000);
}

#[test]
fn test_knowledge_config_explicit_values() {
    let toml_str = config_toml(&[(
        "knowledge",
        "auto_retrieve = false\nmax_context_entries = 10\nrelevance_threshold = -1.0\nmax_context_tokens = 4000",
    )]);
    let config: AppConfig = toml::from_str(&toml_str).unwrap();
    assert!(!config.knowledge.auto_retrieve);
    assert_eq!(config.knowledge.max_context_entries, 10);
    assert!((config.knowledge.relevance_threshold - (-1.0)).abs() < f64::EPSILON);
    assert_eq!(config.knowledge.max_context_tokens, 4000);
}

#[test]
fn test_memory_config_pbkdf2_default() {
    let toml_str = config_toml(&[]);
    let config: AppConfig = toml::from_str(&toml_str).unwrap();
    assert_eq!(config.memory.pbkdf2_iterations, 100_000);
}
```

- [ ] **Step 6: Update `config_toml()` helper to support `knowledge` overrides**

The `config_toml()` helper needs a new section for `knowledge`. Follow the same pattern as `daemon`:

```rust
let knowledge = overrides.iter().find(|(k, _)| *k == "knowledge");
let knowledge_section = knowledge.map_or(String::new(), |(_, v)| format!("\n[knowledge]\n{v}\n"));
```

And include `{knowledge_section}` in the format string.

- [ ] **Step 7: Verify**

Run: `cargo test -p freebird-types`
Run: `cargo clippy -p freebird-types`

Note: `freebird-daemon` will fail to compile until Task 5 (SqliteMemory) replaces the `FileMemory` usage. That's expected — downstream crates are updated in later tasks.

- [ ] **Step 8: Commit**

```bash
git add crates/freebird-types/src/config.rs config/default.toml
git commit -m "feat(types): replace MemoryKind with SQLCipher config, add KnowledgeConfig"
```

---

## Chunk 2: Security Infrastructure (Tasks 3-4)

### Task 3: Encryption Key Derivation

**Files:**
- Create: `crates/freebird-security/src/db_key.rs`
- Modify: `crates/freebird-security/src/lib.rs:8-16` (add `pub mod db_key`)
- Modify: `crates/freebird-security/src/error.rs:21-92` (add new error variants)

**Context:** `freebird-security` depends on `ring`, `secrecy`, `hex` (already in Cargo.toml). The `SecurityError` enum is in `error.rs`. Ring's PBKDF2 API: `ring::pbkdf2::derive(algorithm, iterations, salt, secret, out)`.

- [ ] **Step 1: Add error variants to `SecurityError`**

Add after the `InvalidCredential` variant (line 91) in `error.rs`:

```rust
    // ── Database encryption ──────────────────────────────────────
    #[error("no database encryption key found: {message}")]
    NoEncryptionKey { message: String },

    #[error("insecure keyfile permissions on `{}`: mode {actual_mode:o}, required {required_mode:o}", path.display())]
    InsecureKeyfile {
        path: PathBuf,
        actual_mode: u32,
        required_mode: u32,
    },

    #[error("keyfile error: {0}")]
    KeyfileError(String),
```

- [ ] **Step 2: Create `db_key.rs`**

Create `crates/freebird-security/src/db_key.rs`:

```rust
//! Database encryption key derivation and source resolution.
//!
//! Derives a SQLCipher-compatible encryption key from a user-provided
//! passphrase using PBKDF2-HMAC-SHA256 with a persistent random salt.
//!
//! Key sources are tried in order: environment variable → keyfile → interactive prompt.
//! The agent has no tool or code path to access any of these sources.

use std::path::{Path, PathBuf};

use ring::pbkdf2;
use ring::rand::{SecureRandom, SystemRandom};
use secrecy::{ExposeSecret, SecretString};

use crate::error::SecurityError;

/// PBKDF2 algorithm: HMAC-SHA256.
static PBKDF2_ALG: pbkdf2::Algorithm = pbkdf2::PBKDF2_HMAC_SHA256;

/// Length of the derived key in bytes (256-bit for SQLCipher).
const KEY_LEN: usize = 32;

/// Length of the salt in bytes.
const SALT_LEN: usize = 32;

/// Environment variable name for the database encryption key.
const ENV_VAR_NAME: &str = "FREEBIRD_DB_KEY";

/// Derives a SQLCipher-compatible hex key from a passphrase and salt.
///
/// Returns a 64-character hex string suitable for `PRAGMA key = 'x"..."'`.
pub fn derive_key(
    passphrase: &SecretString,
    salt: &[u8],
    iterations: u32,
) -> SecretString {
    let mut key_bytes = [0u8; KEY_LEN];
    pbkdf2::derive(
        PBKDF2_ALG,
        std::num::NonZeroU32::new(iterations).unwrap_or(
            // SAFETY: 100_000 is non-zero. This fallback is unreachable
            // in practice because config validation ensures iterations > 0.
            std::num::NonZeroU32::MIN,
        ),
        salt,
        passphrase.expose_secret().as_bytes(),
        &mut key_bytes,
    );
    let hex_key = hex::encode(key_bytes);
    // Zeroize the intermediate key bytes
    key_bytes.fill(0);
    SecretString::from(hex_key)
}

/// Load or create the salt file for key derivation.
///
/// On first run, generates a cryptographically random salt and writes it
/// to `salt_path` with 0600 permissions. On subsequent runs, reads the existing salt.
///
/// # Errors
///
/// Returns `SecurityError::KeyfileError` if the salt file cannot be read or created.
pub fn load_or_create_salt(salt_path: &Path) -> Result<Vec<u8>, SecurityError> {
    if salt_path.exists() {
        std::fs::read(salt_path).map_err(|e| SecurityError::KeyfileError(format!(
            "failed to read salt file `{}`: {e}",
            salt_path.display()
        )))
    } else {
        let rng = SystemRandom::new();
        let mut salt = vec![0u8; SALT_LEN];
        rng.fill(&mut salt).map_err(|_| {
            SecurityError::KeyfileError("failed to generate random salt".into())
        })?;

        // Create parent directory if needed
        if let Some(parent) = salt_path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| {
                SecurityError::KeyfileError(format!(
                    "failed to create salt directory `{}`: {e}",
                    parent.display()
                ))
            })?;
        }

        std::fs::write(salt_path, &salt).map_err(|e| {
            SecurityError::KeyfileError(format!(
                "failed to write salt file `{}`: {e}",
                salt_path.display()
            ))
        })?;

        // Set permissions to 0600 on Unix
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let perms = std::fs::Permissions::from_mode(0o600);
            std::fs::set_permissions(salt_path, perms).map_err(|e| {
                SecurityError::KeyfileError(format!(
                    "failed to set salt file permissions: {e}"
                ))
            })?;
        }

        Ok(salt)
    }
}

/// Resolve the database passphrase from layered sources.
///
/// Priority: environment variable → keyfile → interactive prompt.
///
/// # Errors
///
/// Returns `SecurityError::NoEncryptionKey` if no source provides a key.
pub fn resolve_passphrase(
    keyfile_path: Option<&Path>,
    allow_prompt: bool,
) -> Result<SecretString, SecurityError> {
    // 1. Environment variable
    if let Ok(val) = std::env::var(ENV_VAR_NAME) {
        // Clear from process environment immediately
        // SAFETY: This is intentional — we don't want the key lingering in env.
        #[allow(unused_unsafe)]
        unsafe {
            std::env::remove_var(ENV_VAR_NAME);
        }
        return Ok(SecretString::from(val));
    }

    // 2. Keyfile
    if let Some(path) = keyfile_path {
        if path.exists() {
            validate_keyfile_permissions(path)?;
            let contents = std::fs::read_to_string(path).map_err(|e| {
                SecurityError::KeyfileError(format!(
                    "failed to read keyfile `{}`: {e}",
                    path.display()
                ))
            })?;
            let trimmed = contents.trim();
            if trimmed.is_empty() {
                return Err(SecurityError::KeyfileError(format!(
                    "keyfile `{}` is empty",
                    path.display()
                )));
            }
            return Ok(SecretString::from(trimmed.to_owned()));
        }
    }

    // 3. Interactive prompt
    if allow_prompt && std::io::IsTerminal::is_terminal(&std::io::stdin()) {
        return prompt_passphrase();
    }

    Err(SecurityError::NoEncryptionKey {
        message: format!(
            "No database key found. Set {ENV_VAR_NAME} env var, \
             create a keyfile, or run interactively."
        ),
    })
}

/// Validate that a keyfile has restrictive permissions (0600 on Unix).
#[cfg(unix)]
fn validate_keyfile_permissions(path: &Path) -> Result<(), SecurityError> {
    use std::os::unix::fs::PermissionsExt;
    let metadata = std::fs::metadata(path).map_err(|e| {
        SecurityError::KeyfileError(format!(
            "failed to stat keyfile `{}`: {e}",
            path.display()
        ))
    })?;
    let mode = metadata.permissions().mode() & 0o777;
    if mode != 0o600 {
        return Err(SecurityError::InsecureKeyfile {
            path: path.to_path_buf(),
            actual_mode: mode,
            required_mode: 0o600,
        });
    }
    Ok(())
}

#[cfg(not(unix))]
fn validate_keyfile_permissions(_path: &Path) -> Result<(), SecurityError> {
    // On non-Unix platforms, skip permission check with a warning.
    tracing::warn!("keyfile permission check is not supported on this platform");
    Ok(())
}

/// Prompt the user for a passphrase via stdin.
fn prompt_passphrase() -> Result<SecretString, SecurityError> {
    use std::io::Write;
    eprint!("Enter database encryption passphrase: ");
    std::io::stderr().flush().map_err(|e| {
        SecurityError::KeyfileError(format!("failed to flush stderr: {e}"))
    })?;

    let mut input = String::new();
    std::io::stdin().read_line(&mut input).map_err(|e| {
        SecurityError::KeyfileError(format!("failed to read passphrase: {e}"))
    })?;

    let trimmed = input.trim();
    if trimmed.is_empty() {
        return Err(SecurityError::NoEncryptionKey {
            message: "empty passphrase provided".into(),
        });
    }

    let secret = SecretString::from(trimmed.to_owned());
    // Zeroize the input buffer
    input.clear();
    input.shrink_to_fit();
    Ok(secret)
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn test_derive_key_deterministic_with_same_salt() {
        let passphrase = SecretString::from("test-passphrase".to_owned());
        let salt = b"fixed-salt-for-testing-1234567890";
        let key1 = derive_key(&passphrase, salt, 1000);
        let key2 = derive_key(&passphrase, salt, 1000);
        assert_eq!(key1.expose_secret(), key2.expose_secret());
    }

    #[test]
    fn test_derive_key_different_salt_different_key() {
        let passphrase = SecretString::from("test-passphrase".to_owned());
        let salt1 = b"salt-aaaaaaaaaaaaaaaaaaaaaaaaaaaa";
        let salt2 = b"salt-bbbbbbbbbbbbbbbbbbbbbbbbbbbb";
        let key1 = derive_key(&passphrase, salt1, 1000);
        let key2 = derive_key(&passphrase, salt2, 1000);
        assert_ne!(key1.expose_secret(), key2.expose_secret());
    }

    #[test]
    fn test_derive_key_produces_64_char_hex() {
        let passphrase = SecretString::from("test".to_owned());
        let salt = b"12345678901234567890123456789012";
        let key = derive_key(&passphrase, salt, 1000);
        assert_eq!(key.expose_secret().len(), 64);
        assert!(key.expose_secret().chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn test_load_or_create_salt_creates_new() {
        let dir = tempfile::tempdir().unwrap();
        let salt_path = dir.path().join("db.salt");
        let salt = load_or_create_salt(&salt_path).unwrap();
        assert_eq!(salt.len(), SALT_LEN);
        assert!(salt_path.exists());
    }

    #[test]
    fn test_load_or_create_salt_reads_existing() {
        let dir = tempfile::tempdir().unwrap();
        let salt_path = dir.path().join("db.salt");
        let salt1 = load_or_create_salt(&salt_path).unwrap();
        let salt2 = load_or_create_salt(&salt_path).unwrap();
        assert_eq!(salt1, salt2);
    }

    #[cfg(unix)]
    #[test]
    fn test_validate_keyfile_permissions_rejects_world_readable() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().unwrap();
        let keyfile = dir.path().join("db.key");
        std::fs::write(&keyfile, "secret").unwrap();
        std::fs::set_permissions(&keyfile, std::fs::Permissions::from_mode(0o644)).unwrap();
        let result = validate_keyfile_permissions(&keyfile);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, SecurityError::InsecureKeyfile { .. }));
    }

    #[cfg(unix)]
    #[test]
    fn test_validate_keyfile_permissions_accepts_0600() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().unwrap();
        let keyfile = dir.path().join("db.key");
        std::fs::write(&keyfile, "secret").unwrap();
        std::fs::set_permissions(&keyfile, std::fs::Permissions::from_mode(0o600)).unwrap();
        assert!(validate_keyfile_permissions(&keyfile).is_ok());
    }

    #[test]
    fn test_resolve_passphrase_returns_no_key_when_nothing_available() {
        // Ensure env var is not set
        std::env::remove_var(ENV_VAR_NAME);
        let result = resolve_passphrase(None, false);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            SecurityError::NoEncryptionKey { .. }
        ));
    }
}
```

- [ ] **Step 3: Register module in `lib.rs`**

Add `pub mod db_key;` to `crates/freebird-security/src/lib.rs`.

- [ ] **Step 4: Add error variant tests**

Add to `error.rs` test module:

```rust
#[test]
fn test_security_error_no_encryption_key_display() {
    let err = SecurityError::NoEncryptionKey {
        message: "Set FREEBIRD_DB_KEY".into(),
    };
    assert!(err.to_string().contains("Set FREEBIRD_DB_KEY"));
}

#[test]
fn test_security_error_insecure_keyfile_display() {
    let err = SecurityError::InsecureKeyfile {
        path: PathBuf::from("/tmp/db.key"),
        actual_mode: 0o644,
        required_mode: 0o600,
    };
    let msg = err.to_string();
    assert!(msg.contains("/tmp/db.key"));
    assert!(msg.contains("644"));
    assert!(msg.contains("600"));
}
```

- [ ] **Step 5: Verify**

Run: `cargo test -p freebird-security`
Run: `cargo clippy -p freebird-security`

- [ ] **Step 6: Commit**

```bash
git add crates/freebird-security/src/db_key.rs crates/freebird-security/src/lib.rs crates/freebird-security/src/error.rs
git commit -m "feat(security): add database key derivation and source resolution"
```

---

### Task 4: Sensitive Content Filter

**Files:**
- Create: `crates/freebird-security/src/sensitive.rs`
- Modify: `crates/freebird-security/src/lib.rs` (add `pub mod sensitive`)

**Context:** This is a pure function with no dependencies beyond `std`. It scans text for patterns that look like secrets. Must have zero false positives on normal technical content.

- [ ] **Step 1: Create `sensitive.rs`**

Create `crates/freebird-security/src/sensitive.rs`:

```rust
//! Sensitive content detection for knowledge store writes.
//!
//! Scans text for patterns that resemble API keys, passwords, private keys,
//! and other credentials. Used to prevent the agent from accidentally storing
//! secrets in the knowledge store.
//!
//! Runs on ALL knowledge writes, regardless of kind or consent status,
//! BEFORE the consent gate — sensitive content is never even presented for approval.

/// Check if content contains patterns that resemble sensitive credentials.
///
/// Returns `Some(reason)` describing what was detected, or `None` if clean.
///
/// # Detected patterns
///
/// - API keys: `sk-`, `ghp_`, `gho_`, `ghs_`, `ghr_`, `xoxb-`, `xoxp-`, `xoxs-`
/// - Bearer tokens: `Bearer ` followed by long base64
/// - AWS credentials: `AKIA` prefix, `aws_secret_access_key`
/// - PEM private keys: `-----BEGIN ... PRIVATE KEY-----`
/// - Password assignments: `password=`, `passwd=`, `secret_key=`, `private_key=`
/// - Generic high-entropy strings (base64 > 40 chars on a single line)
#[must_use]
pub fn contains_sensitive_content(content: &str) -> Option<&'static str> {
    // API key prefixes (common providers)
    let api_key_prefixes = [
        "sk-",    // OpenAI, Stripe
        "ghp_",   // GitHub personal access token
        "gho_",   // GitHub OAuth
        "ghs_",   // GitHub server-to-server
        "ghr_",   // GitHub refresh token
        "xoxb-",  // Slack bot token
        "xoxp-",  // Slack user token
        "xoxs-",  // Slack legacy token
        "sk_live_", // Stripe live key
        "pk_live_", // Stripe publishable live key
        "rk_live_", // Stripe restricted live key
    ];

    let content_lower = content.to_lowercase();

    for prefix in &api_key_prefixes {
        if content.contains(prefix) {
            return Some("contains API key pattern");
        }
    }

    // AWS access key ID (starts with AKIA, 20 chars)
    if content.contains("AKIA") {
        // Verify it looks like an actual key (followed by alphanumeric chars)
        for (i, _) in content.match_indices("AKIA") {
            let remainder = &content[i..];
            if remainder.len() >= 20
                && remainder[..20].chars().all(|c| c.is_ascii_alphanumeric())
            {
                return Some("contains AWS access key pattern");
            }
        }
    }

    // AWS secret patterns
    if content_lower.contains("aws_secret_access_key")
        || content_lower.contains("aws_session_token")
    {
        return Some("contains AWS credential pattern");
    }

    // PEM private key blocks
    if content.contains("-----BEGIN") && content.contains("PRIVATE KEY-----") {
        return Some("contains PEM private key");
    }

    // Password/secret assignment patterns
    let password_patterns = [
        "password=",
        "password:",
        "passwd=",
        "passwd:",
        "secret_key=",
        "secret_key:",
        "private_key=",
        "private_key:",
        "api_key=",
        "api_key:",
        "apikey=",
        "apikey:",
        "access_token=",
        "access_token:",
    ];

    for pattern in &password_patterns {
        if content_lower.contains(pattern) {
            // Check that something follows the assignment (not just the label)
            if let Some(pos) = content_lower.find(pattern) {
                let after = &content[pos + pattern.len()..];
                let value = after.trim();
                // Only flag if there's an actual value (not empty, not a placeholder)
                if !value.is_empty()
                    && !value.starts_with('<')
                    && !value.starts_with('[')
                    && !value.starts_with('{')
                    && value != "\"\"" && value != "''"
                {
                    return Some("contains password or secret assignment");
                }
            }
        }
    }

    // Bearer token
    if content.contains("Bearer ") {
        for (i, _) in content.match_indices("Bearer ") {
            let after = &content[i + 7..];
            let token: String = after.chars().take_while(|c| c.is_ascii_alphanumeric() || *c == '-' || *c == '_' || *c == '.').collect();
            if token.len() >= 20 {
                return Some("contains Bearer token");
            }
        }
    }

    // High-entropy base64 strings (likely encoded secrets)
    // Look for lines that are mostly base64 characters and longer than 40 chars
    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.len() >= 40 && is_high_entropy_base64(trimmed) {
            return Some("contains high-entropy string resembling encoded secret");
        }
    }

    None
}

/// Check if a string looks like a high-entropy base64-encoded value.
///
/// Returns true if >80% of characters are base64-alphabet and the string
/// has reasonable entropy distribution.
fn is_high_entropy_base64(s: &str) -> bool {
    if s.len() < 40 {
        return false;
    }

    let base64_chars = s
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || *c == '+' || *c == '/' || *c == '=')
        .count();

    let ratio = base64_chars as f64 / s.len() as f64;
    if ratio < 0.85 {
        return false;
    }

    // Check it's not just a normal word or path by requiring mixed case + digits
    let has_upper = s.chars().any(|c| c.is_ascii_uppercase());
    let has_lower = s.chars().any(|c| c.is_ascii_lowercase());
    let has_digit = s.chars().any(|c| c.is_ascii_digit());

    has_upper && has_lower && has_digit
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    // ── Positive detections (should flag) ──

    #[test]
    fn test_detects_openai_key() {
        assert!(contains_sensitive_content("My key is sk-abc123def456").is_some());
    }

    #[test]
    fn test_detects_github_pat() {
        assert!(contains_sensitive_content("ghp_ABCDEFghijklmnopqrstuvwxyz1234").is_some());
    }

    #[test]
    fn test_detects_slack_token() {
        assert!(contains_sensitive_content("token: xoxb-123456-abcdef").is_some());
    }

    #[test]
    fn test_detects_aws_access_key() {
        assert!(contains_sensitive_content("AKIAIOSFODNN7EXAMPLE").is_some());
    }

    #[test]
    fn test_detects_aws_secret_pattern() {
        assert!(
            contains_sensitive_content("aws_secret_access_key = wJalrXUtnFEMI/K7MDENG/bPxRfiCY")
                .is_some()
        );
    }

    #[test]
    fn test_detects_pem_private_key() {
        let pem = "-----BEGIN RSA PRIVATE KEY-----\nMIIEpAIBAAK...\n-----END RSA PRIVATE KEY-----";
        assert!(contains_sensitive_content(pem).is_some());
    }

    #[test]
    fn test_detects_password_assignment() {
        assert!(contains_sensitive_content("password=hunter2").is_some());
        assert!(contains_sensitive_content("secret_key: myS3cretV4lue").is_some());
    }

    #[test]
    fn test_detects_bearer_token() {
        assert!(
            contains_sensitive_content("Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9")
                .is_some()
        );
    }

    // ── Negative detections (should NOT flag) ──

    #[test]
    fn test_allows_normal_technical_content() {
        assert!(contains_sensitive_content("The filesystem tool requires FileRead capability").is_none());
    }

    #[test]
    fn test_allows_code_discussion() {
        assert!(contains_sensitive_content("Use `cargo test -p freebird-security` to run tests").is_none());
    }

    #[test]
    fn test_allows_error_messages() {
        assert!(contains_sensitive_content("Error: connection refused to api.anthropic.com:443").is_none());
    }

    #[test]
    fn test_allows_password_label_without_value() {
        assert!(contains_sensitive_content("The password field is required").is_none());
    }

    #[test]
    fn test_allows_password_placeholder() {
        assert!(contains_sensitive_content("password=<your-password-here>").is_none());
        assert!(contains_sensitive_content("password=[REDACTED]").is_none());
    }

    #[test]
    fn test_allows_short_strings() {
        assert!(contains_sensitive_content("hello world").is_none());
    }

    #[test]
    fn test_allows_paths_and_urls() {
        assert!(contains_sensitive_content("/home/user/.freebird/freebird.db").is_none());
        assert!(contains_sensitive_content("https://api.anthropic.com/v1/messages").is_none());
    }

    #[test]
    fn test_allows_rust_code_snippets() {
        let code = r#"
            fn main() {
                let config = AppConfig::load()?;
                println!("Loaded {} providers", config.providers.len());
            }
        "#;
        assert!(contains_sensitive_content(code).is_none());
    }
}
```

- [ ] **Step 2: Register module**

Add `pub mod sensitive;` to `crates/freebird-security/src/lib.rs`.

- [ ] **Step 3: Verify**

Run: `cargo test -p freebird-security`
Run: `cargo clippy -p freebird-security`

- [ ] **Step 4: Commit**

```bash
git add crates/freebird-security/src/sensitive.rs crates/freebird-security/src/lib.rs
git commit -m "feat(security): add sensitive content filter for knowledge store"
```

---

## Chunk 3: SQLite Infrastructure (Task 5)

### Task 5: SQLite Database Foundation

**Files:**
- Modify: `Cargo.toml` (workspace root, line ~48 area — add `rusqlite`)
- Modify: `crates/freebird-memory/Cargo.toml` (add `rusqlite`, `chrono`, `tracing` deps)
- Create: `crates/freebird-memory/src/sqlite.rs` (shared `SqliteDb`)
- Create: `crates/freebird-memory/src/migrations/001_initial.sql`
- Modify: `crates/freebird-memory/src/lib.rs` (add `pub mod sqlite`)

**Context:** The workspace `Cargo.toml` defines all shared deps. `freebird-memory` currently depends on `freebird-traits`, `freebird-types`, `tokio`, `serde`, `serde_json`, `tracing`, `async-trait`. `rusqlite` with `bundled-sqlcipher` bundles the SQLCipher library.

- [ ] **Step 1: Add `rusqlite` to workspace dependencies**

In root `Cargo.toml`, add to `[workspace.dependencies]`:

```toml
rusqlite = { version = "0.32", features = ["bundled-sqlcipher", "column_decltype"] }
```

- [ ] **Step 2: Add `rusqlite` and `chrono` to `freebird-memory/Cargo.toml`**

Add to `[dependencies]`:

```toml
rusqlite = { workspace = true }
chrono = { workspace = true }
secrecy = { workspace = true }
```

Add to `[dev-dependencies]`:

```toml
ring = { workspace = true }
hex = { workspace = true }
```

(Dev-deps are for test key derivation.)

- [ ] **Step 3: Create migration SQL**

Create directory and file `crates/freebird-memory/src/migrations/001_initial.sql`:

```sql
-- FreeBird database schema v1: conversations + knowledge with FTS5

-- Conversations (replaces file-based JSON storage)
CREATE TABLE IF NOT EXISTS conversations (
    session_id    TEXT PRIMARY KEY,
    system_prompt TEXT,
    model_id      TEXT NOT NULL,
    provider_id   TEXT NOT NULL,
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL,
    data          TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_conversations_updated ON conversations(updated_at DESC);

-- Knowledge entries
CREATE TABLE IF NOT EXISTS knowledge (
    id            TEXT PRIMARY KEY,
    kind          TEXT NOT NULL,
    content       TEXT NOT NULL,
    tags          TEXT NOT NULL,
    source        TEXT NOT NULL,
    confidence    REAL NOT NULL DEFAULT 1.0,
    session_id    TEXT,
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL,
    access_count  INTEGER NOT NULL DEFAULT 0,
    last_accessed TEXT
);

CREATE INDEX IF NOT EXISTS idx_knowledge_kind ON knowledge(kind);
CREATE INDEX IF NOT EXISTS idx_knowledge_updated ON knowledge(updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_knowledge_confidence ON knowledge(confidence DESC);

-- FTS5 virtual table for ranked text search over knowledge
CREATE VIRTUAL TABLE IF NOT EXISTS knowledge_fts USING fts5(
    content,
    tags,
    kind,
    content=knowledge,
    content_rowid=rowid,
    tokenize='porter unicode61'
);

-- Triggers to keep FTS5 index in sync with knowledge table
CREATE TRIGGER IF NOT EXISTS knowledge_ai AFTER INSERT ON knowledge BEGIN
    INSERT INTO knowledge_fts(rowid, content, tags, kind)
    VALUES (new.rowid, new.content, new.tags, new.kind);
END;

CREATE TRIGGER IF NOT EXISTS knowledge_ad AFTER DELETE ON knowledge BEGIN
    INSERT INTO knowledge_fts(knowledge_fts, rowid, content, tags, kind)
    VALUES ('delete', old.rowid, old.content, old.tags, old.kind);
END;

CREATE TRIGGER IF NOT EXISTS knowledge_au AFTER UPDATE ON knowledge BEGIN
    INSERT INTO knowledge_fts(knowledge_fts, rowid, content, tags, kind)
    VALUES ('delete', old.rowid, old.content, old.tags, old.kind);
    INSERT INTO knowledge_fts(rowid, content, tags, kind)
    VALUES (new.rowid, new.content, new.tags, new.kind);
END;

-- Schema version tracking
CREATE TABLE IF NOT EXISTS schema_version (
    version    INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL
);
```

- [ ] **Step 4: Create `sqlite.rs`**

Create `crates/freebird-memory/src/sqlite.rs`:

```rust
//! Shared SQLCipher database connection for memory and knowledge backends.
//!
//! Manages encrypted database lifecycle: key application, schema migrations,
//! and connection access via `tokio::sync::Mutex` for async safety.

use std::path::Path;

use freebird_traits::memory::MemoryError;
use secrecy::{ExposeSecret, SecretString};

/// Shared encrypted SQLite database connection.
///
/// All access goes through the async mutex. Single connection is sufficient
/// because FreeBird handles one user at a time. SQLCipher key is applied
/// once at open time and is transparent for all subsequent operations.
pub struct SqliteDb {
    conn: tokio::sync::Mutex<rusqlite::Connection>,
}

impl SqliteDb {
    /// Open (or create) an encrypted SQLite database.
    ///
    /// Applies the SQLCipher key, verifies it's correct, and runs
    /// any pending schema migrations.
    ///
    /// # Errors
    ///
    /// - `MemoryError::IntegrityViolation` if the key is wrong
    /// - `MemoryError::Io` if the database cannot be opened or migrated
    pub fn open(path: &Path, key: &SecretString) -> Result<Self, MemoryError> {
        // Create parent directory if needed
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| {
                MemoryError::Io(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    format!("failed to create database directory: {e}"),
                ))
            })?;
        }

        let conn = rusqlite::Connection::open(path).map_err(|e| {
            MemoryError::Io(std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("failed to open database: {e}"),
            ))
        })?;

        // Apply encryption key — MUST be the first statement after open
        let pragma_key = format!("x'{}'", key.expose_secret());
        conn.pragma_update(None, "key", &pragma_key).map_err(|e| {
            MemoryError::Io(std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("failed to apply database key: {e}"),
            ))
        })?;

        // Verify the key is correct by reading sqlite_master
        conn.query_row("SELECT count(*) FROM sqlite_master", [], |_| Ok(()))
            .map_err(|_| MemoryError::IntegrityViolation {
                reason: "database key is incorrect or database is corrupted".into(),
            })?;

        // Enable WAL mode for better concurrent read performance
        conn.pragma_update(None, "journal_mode", "WAL")
            .map_err(|e| {
                MemoryError::Io(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    format!("failed to enable WAL mode: {e}"),
                ))
            })?;

        // Run migrations
        Self::migrate(&conn)?;

        Ok(Self {
            conn: tokio::sync::Mutex::new(conn),
        })
    }

    /// Acquire the database connection lock.
    ///
    /// Callers MUST NOT hold this lock across `.await` points.
    pub async fn conn(
        &self,
    ) -> tokio::sync::MutexGuard<'_, rusqlite::Connection> {
        self.conn.lock().await
    }

    /// Run pending schema migrations.
    fn migrate(conn: &rusqlite::Connection) -> Result<(), MemoryError> {
        let current_version = Self::get_schema_version(conn);

        let migrations: &[(i64, &str)] = &[
            (1, include_str!("migrations/001_initial.sql")),
        ];

        for &(version, sql) in migrations {
            if version > current_version {
                conn.execute_batch(sql).map_err(|e| {
                    MemoryError::Io(std::io::Error::new(
                        std::io::ErrorKind::Other,
                        format!("migration {version} failed: {e}"),
                    ))
                })?;
                conn.execute(
                    "INSERT OR REPLACE INTO schema_version (version, applied_at) VALUES (?1, ?2)",
                    rusqlite::params![version, chrono::Utc::now().to_rfc3339()],
                )
                .map_err(|e| {
                    MemoryError::Io(std::io::Error::new(
                        std::io::ErrorKind::Other,
                        format!("failed to record migration {version}: {e}"),
                    ))
                })?;
                tracing::info!(version, "applied database migration");
            }
        }

        Ok(())
    }

    /// Get the current schema version (0 if no migrations applied yet).
    fn get_schema_version(conn: &rusqlite::Connection) -> i64 {
        conn.query_row(
            "SELECT MAX(version) FROM schema_version",
            [],
            |row| row.get(0),
        )
        .unwrap_or(0)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    /// Create a test database with a known key.
    fn test_db() -> (tempfile::TempDir, SqliteDb) {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("test.db");
        let key = SecretString::from("a".repeat(64));
        let db = SqliteDb::open(&db_path, &key).unwrap();
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
        // Verify conversations table exists
        let count: i64 = conn
            .query_row("SELECT count(*) FROM conversations", [], |row| row.get(0))
            .unwrap();
        assert_eq!(count, 0);
        // Verify knowledge table exists
        let count: i64 = conn
            .query_row("SELECT count(*) FROM knowledge", [], |row| row.get(0))
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

    #[test]
    fn test_wrong_key_returns_integrity_violation() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("test.db");
        let key = SecretString::from("a".repeat(64));

        // Create DB with key
        let _db = SqliteDb::open(&db_path, &key).unwrap();
        drop(_db);

        // Try to open with wrong key
        let wrong_key = SecretString::from("b".repeat(64));
        let result = SqliteDb::open(&db_path, &wrong_key);
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
        let _db1 = SqliteDb::open(&db_path, &key).unwrap();
        drop(_db1);
        let _db2 = SqliteDb::open(&db_path, &key).unwrap();
    }

    #[tokio::test]
    async fn test_fts5_table_exists() {
        let (_dir, db) = test_db();
        let conn = db.conn().await;
        // FTS5 tables show up in sqlite_master
        let count: i64 = conn
            .query_row(
                "SELECT count(*) FROM sqlite_master WHERE name = 'knowledge_fts'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(count, 1);
    }
}
```

- [ ] **Step 5: Register module**

Add `pub mod sqlite;` to `crates/freebird-memory/src/lib.rs`.

- [ ] **Step 6: Verify**

Run: `cargo test -p freebird-memory`
Run: `cargo clippy -p freebird-memory`

Note: First build will take a while — `bundled-sqlcipher` compiles SQLCipher from source.

- [ ] **Step 7: Commit**

```bash
git add Cargo.toml crates/freebird-memory/Cargo.toml crates/freebird-memory/src/sqlite.rs crates/freebird-memory/src/migrations/ crates/freebird-memory/src/lib.rs
git commit -m "feat(memory): add SQLCipher database infrastructure with FTS5 schema"
```

---

## Chunk 4: Storage Implementations (Tasks 6-7)

### Task 6: SqliteMemory — Memory Trait Implementation

**Files:**
- Create: `crates/freebird-memory/src/sqlite_memory.rs`
- Modify: `crates/freebird-memory/src/lib.rs` (add `pub mod sqlite_memory`)

**Context:** Implements `Memory` trait (from `freebird-traits/src/memory.rs`) over the `SqliteDb` from Task 5. Conversations are stored with turns as a JSON blob in the `data` column. Must match the behavior of `FileMemory` for backward compatibility (search is case-insensitive substring, list is ordered by `updated_at` DESC).

- [ ] **Step 1: Create `sqlite_memory.rs`**

The implementation wraps `Arc<SqliteDb>` and implements the `Memory` trait. Key methods:
- `load()` — SELECT by session_id, deserialize JSON `data` into `Vec<Turn>`, reconstruct `Conversation`
- `save()` — serialize turns to JSON, INSERT OR REPLACE
- `list_sessions()` — SELECT all, build `SessionSummary` from columns + first turn preview
- `delete()` — DELETE by session_id
- `search()` — SELECT WHERE data LIKE (case-insensitive substring, matching FileMemory behavior)

Use `Arc<SqliteDb>` so both `SqliteMemory` and `SqliteKnowledgeStore` share the same connection.

Include comprehensive tests: save/load roundtrip, list ordering, search matching, delete, empty results, concurrent access.

- [ ] **Step 2: Register module, verify, commit**

---

### Task 7: SqliteKnowledgeStore — KnowledgeStore Trait Implementation

**Files:**
- Create: `crates/freebird-memory/src/sqlite_knowledge.rs`
- Modify: `crates/freebird-memory/src/lib.rs` (add `pub mod sqlite_knowledge`)

**Context:** Implements `KnowledgeStore` trait over the same `Arc<SqliteDb>`. The FTS5 search uses `MATCH` with `bm25()` ranking. Tags are stored as JSON arrays. `replace_kind()` does DELETE + INSERT in a single transaction.

- [ ] **Step 1: Create `sqlite_knowledge.rs`**

Key implementation details:
- `store()` — INSERT into knowledge table (FTS5 trigger auto-updates index)
- `search()` — `SELECT k.*, bm25(knowledge_fts) as rank FROM knowledge_fts f JOIN knowledge k ON k.rowid = f.rowid WHERE knowledge_fts MATCH ?1 ORDER BY rank LIMIT ?2`
- `list_by_tag()` — `SELECT * FROM knowledge WHERE tags LIKE '%"' || ?1 || '"%'` (JSON array contains)
- `replace_kind()` — `BEGIN; DELETE FROM knowledge WHERE kind = ?; INSERT ...; COMMIT;`
- `record_access()` — `UPDATE knowledge SET access_count = access_count + 1, last_accessed = ?1 WHERE id IN (...)`

Include tests: CRUD cycle, FTS5 search returns ranked results, replace_kind is atomic, tag filtering, access count tracking, empty query handling.

- [ ] **Step 2: Register module, verify, commit**

---

## Chunk 5: Tools & Agent Integration (Tasks 8-10)

### Task 8: Knowledge Tools

**Files:**
- Create: `crates/freebird-tools/src/knowledge.rs`
- Modify: `crates/freebird-tools/src/lib.rs` (add `pub mod knowledge`)
- Modify: `crates/freebird-traits/src/tool.rs:96-104` (add `knowledge_store` and `consent_gate` to `ToolContext`)

**Context:** Four tools: `store_knowledge`, `search_knowledge`, `update_knowledge`, `delete_knowledge`. Uses `freebird_security::sensitive::contains_sensitive_content` for pre-write scanning. Uses dynamic consent via `ToolContext.consent_gate` for protected kinds. The `ToolContext` struct is in `freebird-traits/src/tool.rs` at line 96.

Key implementation details:
- `ToolContext` gains `knowledge_store: Option<&'a dyn KnowledgeStore>` and `consent_gate: Option<&'a dyn ConsentGate>` (import `ConsentGate` from `freebird-security::consent`)
- `store_knowledge`: parse kind from input → sensitive scan → consent if `kind.requires_consent()` → store
- `search_knowledge`: FTS5 search with optional kind filter → format results
- `update_knowledge`: get existing → sensitive scan on new content → consent if protected kind → update
- `delete_knowledge`: get existing → consent if protected kind → delete (reason is audit-logged)
- All tools return `ToolError::ExecutionFailed` if `knowledge_store` is `None`

Note: Adding fields to `ToolContext` is a breaking change — all tool implementations and all call sites in `tool_executor.rs` need updating. The new fields are `Option` so existing code can pass `None`.

- [ ] **Step 1: Update `ToolContext` in `freebird-traits/src/tool.rs`**
- [ ] **Step 2: Update all `ToolContext` construction sites** (in `freebird-runtime/src/tool_executor.rs`)
- [ ] **Step 3: Create knowledge tools**
- [ ] **Step 4: Register in `freebird-tools/src/lib.rs`**
- [ ] **Step 5: Verify, commit**

---

### Task 9: Agent Loop Integration

**Files:**
- Modify: `crates/freebird-runtime/src/agent.rs:70-82` (add `knowledge` and `knowledge_config` fields to `AgentRuntime`)
- Modify: `crates/freebird-runtime/src/agent.rs:84-112` (update `new()` constructor)
- Modify: `crates/freebird-runtime/src/agent.rs:462-521` (add auto-retrieval to `handle_message_inner`)
- Create: `crates/freebird-runtime/src/knowledge_context.rs` (context block formatting + retrieval logic)
- Modify: `crates/freebird-runtime/src/lib.rs` (add `pub mod knowledge_context`)
- Modify: `crates/freebird-runtime/Cargo.toml` (add `freebird-memory` as optional dep for tests if needed)

**Context:** `AgentRuntime` at line 70 holds `memory: Box<dyn Memory>`. We add `knowledge: Option<Box<dyn KnowledgeStore>>` and `knowledge_config: KnowledgeConfig`. The auto-retrieval happens between step 2 (load conversation) and step 3 (build CompletionRequest) in `handle_message_inner` (line 462).

Key implementation:
- `knowledge_context.rs` exports `retrieve_relevant_knowledge()` and `format_knowledge_block()`
- `retrieve_relevant_knowledge()` calls `knowledge.search(query, limit)`, filters by threshold, caps tokens
- `format_knowledge_block()` produces the `[RELEVANT CONTEXT]` block
- In `handle_message_inner()`, call retrieve after load, inject block into the messages sent to provider
- Thread `knowledge_store` reference from `AgentRuntime` into `ToolContext` during agentic loop

- [ ] **Steps: Add fields, update constructor, create retrieval module, inject into agent loop, verify, commit**

---

### Task 10: Daemon Composition & Startup Bootstrap

**Files:**
- Modify: `crates/freebird-daemon/src/main.rs:24` (replace `freebird_memory::file::FileMemory` import)
- Modify: `crates/freebird-daemon/src/main.rs:73-186` (`cmd_serve` — replace memory init, add key resolution, add knowledge bootstrap)
- Modify: `crates/freebird-daemon/src/main.rs:266-278` (replace `init_memory` with `init_database`)
- Modify: `crates/freebird-daemon/src/tools.rs` (register knowledge tools)
- Create: `crates/freebird-daemon/src/bootstrap.rs` (system knowledge population)

**Context:** `cmd_serve()` currently calls `init_memory()` at line 100 which creates a `FileMemory`. We replace this with: resolve key → derive key → open `SqliteDb` → create `SqliteMemory` + `SqliteKnowledgeStore` → run migration check → populate system knowledge → wire into `AgentRuntime`.

Key changes:
- Import `freebird_security::db_key::{resolve_passphrase, derive_key, load_or_create_salt}`
- Import `freebird_memory::sqlite::SqliteDb`, `sqlite_memory::SqliteMemory`, `sqlite_knowledge::SqliteKnowledgeStore`
- `init_database()` replaces `init_memory()`: does key resolution → derivation → SqliteDb::open
- `bootstrap.rs`: `populate_system_knowledge()` generates `SystemConfig` + `ToolCapability` entries from `AppConfig` and registered tools
- `AgentRuntime::new()` call updated with new knowledge params
- Register `store_knowledge`, `search_knowledge`, `update_knowledge`, `delete_knowledge` in tool registry

- [ ] **Steps: Replace memory init, add key resolution, add bootstrap, wire knowledge tools, update runtime construction, verify, commit**

---

## Chunk 6: Migration & Documentation (Tasks 11-12)

### Task 11: FileMemory Migration

**Files:**
- Create: `crates/freebird-daemon/src/migrate.rs`

**Context:** One-time migration from JSON files to SQLite. Runs on first startup if `~/.freebird/conversations/*.json` exists.

- [ ] **Step 1: Create `migrate.rs`**

```rust
/// Migrate conversations from FileMemory JSON files to SQLite.
///
/// Checks if the legacy conversations directory exists. If so, reads each
/// JSON file, inserts into the SQLite conversations table, and moves originals
/// to a `.bak` directory.
pub async fn migrate_file_conversations(
    db: &SqliteDb,
    legacy_dir: &Path,
) -> Result<usize>
```

- [ ] **Step 2: Wire into `cmd_serve()` after database init**
- [ ] **Step 3: Test with fixtures, commit**

---

### Task 12: Documentation

**Files:**
- Create or modify: `README.md` (add Security section)
- Modify: `CLAUDE.md` (sections 1, 3, 5, 6, 14, 19, 24)

- [ ] **Step 1: Add Security section to README** (features table, encryption, knowledge security, threat model)
- [ ] **Step 2: Update CLAUDE.md** (dependency table, message flow, agent loop, knowledge security, config, checklist)
- [ ] **Step 3: Commit**

---

## Review Checkpoints

After each chunk, the orchestrating agent should:

1. Run `cargo test --workspace` — all tests must pass
2. Run `cargo clippy --workspace` — zero warnings
3. Run `cargo build --workspace` — confirms cross-crate compilation
4. Review code for: DRY violations, unused code, missing error handling, style consistency with CLAUDE.md §22-23
5. Check that no `.unwrap()` or `.expect()` exists in production code (only in `#[cfg(test)]` modules)

## Parallelism Map

```
Wave 1 (parallel):
  Agent A: Task 1 (traits + types)
  Agent B: Task 4 (sensitive filter) — no deps on Task 1

Wave 2 (parallel, after Wave 1):
  Agent C: Task 2 (config) — depends on Task 1
  Agent D: Task 3 (encryption) — depends on nothing new

Wave 3 (sequential):
  Agent E: Task 5 (SQLite infra) — depends on Tasks 2, 3

Wave 4 (parallel):
  Agent F: Task 6 (SqliteMemory) — depends on Task 5
  Agent G: Task 7 (SqliteKnowledgeStore) — depends on Task 5

Wave 5 (sequential):
  Agent H: Task 8 (knowledge tools) — depends on Tasks 6, 7

Wave 6 (sequential):
  Agent I: Task 9 (agent loop integration) — depends on Task 8
  Agent J: Task 10 (daemon composition) — depends on Task 9

Wave 7 (parallel):
  Agent K: Task 11 (migration) — depends on Task 10
  Agent L: Task 12 (documentation) — can start after Task 10
```
