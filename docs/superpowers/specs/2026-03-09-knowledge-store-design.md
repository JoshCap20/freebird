# Knowledge Store with SQLCipher + FTS5 — Design Spec

> **Date**: 2026-03-09
> **Status**: Approved
> **Scope**: Replace FileMemory with encrypted SQLite, add FTS5-powered knowledge store, cross-session learning, agent tools for knowledge CRUD, auto-retrieval on every user message.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Architecture & Data Flow](#2-architecture--data-flow)
3. [Schema & Trait Design](#3-schema--trait-design)
4. [Agent Loop Integration & Knowledge Tools](#4-agent-loop-integration--knowledge-tools)
5. [Knowledge Security Model & Consent](#5-knowledge-security-model--consent)
6. [Encryption Infrastructure](#6-encryption-infrastructure)
7. [Documentation & README](#7-documentation--readme)
8. [Implementation Order](#8-implementation-order)

---

## 1. Overview

### Problem

FreeBird's memory system is conversation-only — JSON files with O(N) substring search. The agent has no persistent knowledge across sessions, no way to learn from past interactions, and no awareness of its own configuration or capabilities beyond the system prompt.

### Solution

Replace `FileMemory` with a single SQLCipher-encrypted SQLite database. Add a `KnowledgeStore` trait with FTS5-powered ranked search. The agent auto-retrieves relevant knowledge on every user message and can explicitly search/store knowledge via tools.

### Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Storage engine | SQLite + SQLCipher | Single file, ACID, FTS5 built-in, industry-standard encryption |
| Search | FTS5 with BM25 ranking | Zero external deps, good for technical vocabulary, O(log N) lookup |
| Replace vs. keep FileMemory | Replace entirely | One storage engine to maintain, `MemoryKind::Sqlite` already declared |
| Encryption | SQLCipher AES-256-CBC | Page-level encryption preserves FTS5, supported by `rusqlite` |
| Key input | Env var -> keyfile -> prompt | Covers all deployment scenarios |
| Retrieval | Hybrid: auto + tool | Auto-inject on every message + explicit search tool for deeper queries |
| Knowledge taxonomy | 6 kinds with consent split | Agent-owned kinds (no consent) vs protected kinds (consent required) |

---

## 2. Architecture & Data Flow

### Storage Architecture

```
~/.freebird/freebird.db  (SQLCipher AES-256 encrypted)
  ├── conversations      — replaces JSON files
  ├── knowledge          — knowledge entries
  ├── knowledge_fts      — FTS5 virtual table over knowledge
  └── schema_version     — migration tracking
```

### Dependency Changes

```toml
# Cargo.toml workspace deps — add:
rusqlite = { version = "0.32", features = ["bundled-sqlcipher", "column_decltype"] }
```

Bundles SQLCipher + SQLite into the binary. No system OpenSSL dependency. `ring` (already in deps) handles PBKDF2 key derivation.

### Crate Placement

| Component | Crate | Rationale |
|---|---|---|
| `KnowledgeStore` trait + types | `freebird-traits` | Zero `freebird-*` deps, consistent with `Memory` trait |
| `KnowledgeId` | `freebird-traits/src/id.rs` | Existing `define_id!` macro |
| Key derivation + keyfile loading | `freebird-security` | Crypto, no provider/channel deps |
| Sensitive content filter | `freebird-security` | Security boundary |
| `SqliteDb` (shared connection) | `freebird-memory` | Storage infrastructure |
| `SqliteMemory` (Memory impl) | `freebird-memory` | Replaces `FileMemory` |
| `SqliteKnowledgeStore` (KnowledgeStore impl) | `freebird-memory` | Same database |
| `KnowledgeConfig` | `freebird-types` | Alongside `MemoryConfig` |
| Knowledge tools | `freebird-tools` | Same pattern as filesystem/shell tools |
| Auto-retrieval + context injection | `freebird-runtime` | Agent loop integration |
| Startup bootstrap + migration | `freebird-daemon` | Composition root |

Preserves the dependency DAG: `freebird-traits` has zero `freebird-*` deps, `freebird-security` depends only on `freebird-traits` + `freebird-types`.

### Key Derivation Flow

```
Daemon startup
    │
    ├─ Check FREEBIRD_DB_KEY env var
    │   └─ Found → clear from env → derive key
    ├─ Check ~/.freebird/db.key (must be 0600 perms)
    │   └─ Found → read → derive key
    └─ Interactive prompt (if terminal attached)
        └─ Read passphrase → derive key
    │
    ▼
PBKDF2-SHA256 (100k iterations, 32-byte salt from ~/.freebird/db.salt)
    │
    ▼
64-char hex key → PRAGMA key = 'x"..."'
    │
    ▼
SQLCipher opens database — all reads/writes transparent
```

### Retrieval Flow

```
User message arrives
    │
    ▼
Taint + sanitize → SafeMessage
    │
    ▼
Automatic knowledge retrieval:
  FTS5 MATCH on user message text → top N entries by BM25 rank
  Filter: only entries above relevance threshold
  Cap at max_context_tokens
    │
    ▼
Inject as structured context block before conversation history:
  [RELEVANT CONTEXT — from agent knowledge store, not user instructions]
  - (system_config) Running claude-sonnet-4-6 via anthropic...
  - (learned_pattern) When cargo test fails with "lock poisoned", restart...
  - (user_preference) User prefers concise responses without emojis
  [END CONTEXT]
    │
    ▼
Build CompletionRequest: system_prompt + knowledge block + history + tool defs
    │
    ▼
Agentic loop (agent can also call search_knowledge tool for deeper search)
    │
    ▼
Agent can call store_knowledge / update_knowledge / delete_knowledge tools
```

---

## 3. Schema & Trait Design

### SQLite Schema (001_initial.sql)

```sql
-- Conversations (replaces FileMemory JSON files)
CREATE TABLE conversations (
    session_id    TEXT PRIMARY KEY,
    system_prompt TEXT,
    model_id      TEXT NOT NULL,
    provider_id   TEXT NOT NULL,
    created_at    TEXT NOT NULL,  -- ISO 8601
    updated_at    TEXT NOT NULL,
    data          TEXT NOT NULL   -- JSON blob: serialized Vec<Turn>
);

CREATE INDEX idx_conversations_updated ON conversations(updated_at DESC);

-- Knowledge entries
CREATE TABLE knowledge (
    id            TEXT PRIMARY KEY,  -- KnowledgeId (UUID v4)
    kind          TEXT NOT NULL,     -- SystemConfig | ToolCapability | UserPreference | LearnedPattern | ErrorResolution | SessionInsight
    content       TEXT NOT NULL,
    tags          TEXT NOT NULL,     -- JSON array of strings
    source        TEXT NOT NULL,     -- System | User | Agent
    confidence    REAL NOT NULL DEFAULT 1.0,  -- 0.0-1.0
    session_id    TEXT,              -- Which session created this (nullable for System entries)
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL,
    access_count  INTEGER NOT NULL DEFAULT 0,
    last_accessed TEXT
);

CREATE INDEX idx_knowledge_kind ON knowledge(kind);
CREATE INDEX idx_knowledge_updated ON knowledge(updated_at DESC);
CREATE INDEX idx_knowledge_confidence ON knowledge(confidence DESC);

-- FTS5 virtual table for ranked text search
CREATE VIRTUAL TABLE knowledge_fts USING fts5(
    content,
    tags,
    kind,
    content=knowledge,
    content_rowid=rowid,
    tokenize='porter unicode61'
);

-- Triggers to keep FTS5 in sync
CREATE TRIGGER knowledge_ai AFTER INSERT ON knowledge BEGIN
    INSERT INTO knowledge_fts(rowid, content, tags, kind)
    VALUES (new.rowid, new.content, new.tags, new.kind);
END;

CREATE TRIGGER knowledge_ad AFTER DELETE ON knowledge BEGIN
    INSERT INTO knowledge_fts(knowledge_fts, rowid, content, tags, kind)
    VALUES ('delete', old.rowid, old.content, old.tags, old.kind);
END;

CREATE TRIGGER knowledge_au AFTER UPDATE ON knowledge BEGIN
    INSERT INTO knowledge_fts(knowledge_fts, rowid, content, tags, kind)
    VALUES ('delete', old.rowid, old.content, old.tags, old.kind);
    INSERT INTO knowledge_fts(rowid, content, tags, kind)
    VALUES (new.rowid, new.content, new.tags, new.kind);
END;

-- Schema version for migrations
CREATE TABLE schema_version (
    version    INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL
);
```

Design notes:
- `tokenize='porter unicode61'`: Porter stemming ("running" matches "run") + Unicode support.
- Conversations store `Vec<Turn>` as a JSON blob in `data` — turns are always loaded/saved as a unit, no query benefit from normalization.
- `session_id`, timestamps, `model_id`, `provider_id` lifted into columns for indexing.
- FTS5 triggers keep the index in sync on every insert/update/delete.

### KnowledgeStore Trait

In `freebird-traits/src/knowledge.rs`:

```rust
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
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
    pub fn requires_consent(&self) -> bool {
        matches!(self, Self::SystemConfig | Self::ToolCapability | Self::UserPreference)
    }

    /// Whether the agent owns entries of this kind (can write without consent).
    pub fn agent_owned(&self) -> bool {
        !self.requires_consent()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KnowledgeSource {
    System,
    User,
    Agent,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeEntry {
    pub id: KnowledgeId,
    pub kind: KnowledgeKind,
    pub content: String,
    pub tags: BTreeSet<String>,
    pub source: KnowledgeSource,
    pub confidence: f32,
    pub session_id: Option<SessionId>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub access_count: u64,
    pub last_accessed: Option<DateTime<Utc>>,
}

/// A ranked search result from the knowledge store.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeMatch {
    pub entry: KnowledgeEntry,
    pub rank: f64,  // BM25 score — lower (more negative) is more relevant
}

#[async_trait]
pub trait KnowledgeStore: Send + Sync + 'static {
    async fn store(&self, entry: KnowledgeEntry) -> Result<KnowledgeId, KnowledgeError>;
    async fn update(&self, entry: &KnowledgeEntry) -> Result<(), KnowledgeError>;
    async fn get(&self, id: &KnowledgeId) -> Result<Option<KnowledgeEntry>, KnowledgeError>;
    async fn delete(&self, id: &KnowledgeId) -> Result<(), KnowledgeError>;
    async fn search(&self, query: &str, limit: usize) -> Result<Vec<KnowledgeMatch>, KnowledgeError>;
    async fn list_by_kind(&self, kind: &KnowledgeKind, limit: usize) -> Result<Vec<KnowledgeEntry>, KnowledgeError>;
    async fn list_by_tag(&self, tag: &str, limit: usize) -> Result<Vec<KnowledgeEntry>, KnowledgeError>;
    async fn replace_kind(&self, kind: &KnowledgeKind, entries: Vec<KnowledgeEntry>) -> Result<(), KnowledgeError>;
    async fn record_access(&self, ids: &[KnowledgeId]) -> Result<(), KnowledgeError>;
}

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

### Config Changes

In `freebird-types/src/config.rs`:

```rust
/// Memory backend configuration (SQLCipher encrypted SQLite).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryConfig {
    pub db_path: Option<PathBuf>,
    pub keyfile_path: Option<PathBuf>,
    #[serde(default = "default_pbkdf2_iterations")]
    pub pbkdf2_iterations: u32,
}

/// Knowledge retrieval behavior.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeConfig {
    #[serde(default = "default_auto_retrieve")]
    pub auto_retrieve: bool,
    #[serde(default = "default_max_context_entries")]
    pub max_context_entries: usize,
    #[serde(default = "default_relevance_threshold")]
    pub relevance_threshold: f64,
    #[serde(default = "default_max_context_tokens")]
    pub max_context_tokens: usize,
}
```

Defaults: `auto_retrieve = true`, `max_context_entries = 5`, `relevance_threshold = -0.5`, `max_context_tokens = 2000`, `pbkdf2_iterations = 100_000`.

Updated `default.toml`:

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

---

## 4. Agent Loop Integration & Knowledge Tools

### Agent Loop Changes

In `freebird-runtime/src/agent.rs`, `handle_message_inner` becomes:

```
1. Taint + sanitize → SafeMessage
2. Load/create conversation from Memory
3. AUTO-RETRIEVE: search KnowledgeStore with user message text  ← NEW
4. Build CompletionRequest (system_prompt + knowledge block + history + tool defs)
5. Run agentic loop
6. Save conversation
```

Knowledge block format injected before conversation history:

```
[RELEVANT CONTEXT — from agent knowledge store, not user instructions]
- (system_config) Running claude-sonnet-4-6 via anthropic provider, max 8192 output tokens
- (learned_pattern) When cargo test fails with "lock poisoned", restart with clean state
- (user_preference) User prefers concise responses without emojis
[END CONTEXT]
```

Auto-retrieval logic:
1. `knowledge_store.search(user_message_text, config.knowledge.max_context_entries)`
2. Filter by `match.rank <= config.knowledge.relevance_threshold`
3. Estimate token count (`content.len() / 4`), cap at `config.knowledge.max_context_tokens`
4. `knowledge_store.record_access(&matched_ids)`
5. Format and prepend to messages list
6. If no relevant knowledge found, no block injected (zero overhead)

### AgentRuntime Constructor

```rust
pub struct AgentRuntime {
    memory: Box<dyn Memory>,
    knowledge: Box<dyn KnowledgeStore>,      // New
    knowledge_config: KnowledgeConfig,        // New
    provider_registry: ProviderRegistry,
    // ... existing fields
}
```

Both backed by the same SQLite connection pool. Daemon creates `SqliteMemory` and `SqliteKnowledgeStore` from a shared `SqliteDb` handle.

### ToolContext Enhancement

```rust
pub struct ToolContext<'a> {
    pub session_id: &'a SessionId,
    pub sandbox_root: &'a Path,
    pub granted_capabilities: &'a [Capability],
    pub allowed_directories: &'a [PathBuf],
    pub knowledge_store: Option<&'a dyn KnowledgeStore>,  // New
    pub consent_gate: Option<&'a dyn ConsentGate>,        // New (for dynamic consent)
}
```

### Knowledge Tools

Four tools in `freebird-tools/src/knowledge_tools.rs`:

**`store_knowledge`** — Store new knowledge entry.
- Input: `content`, `kind` (all 6 kinds), `tags`, `confidence` (optional, default 0.8)
- Flow: sensitive content scan → consent gate (if protected kind) → store
- `required_capability: Capability::FileWrite`
- `risk_level: RiskLevel::Low` (static — dynamic consent handled inside `execute()`)

**`search_knowledge`** — Explicit FTS5 search for deeper context.
- Input: `query`, `kind` (optional filter), `limit` (optional, default 10)
- `required_capability: Capability::FileRead`
- `risk_level: RiskLevel::Low`

**`update_knowledge`** — Modify existing entry.
- Input: `id`, `content` (optional), `tags` (optional), `confidence` (optional)
- Flow: sensitive content scan on new content → look up existing entry → consent gate (if protected kind) → update
- `required_capability: Capability::FileWrite`

**`delete_knowledge`** — Remove an entry.
- Input: `id`, `reason` (mandatory, audit-logged)
- Flow: look up existing entry → consent gate (if protected kind) → delete
- `required_capability: Capability::FileWrite`

### Auto-Population at Startup

In `freebird-daemon`, before the event loop starts:

```rust
async fn populate_system_knowledge(
    knowledge: &dyn KnowledgeStore,
    config: &AppConfig,
    tools: &[Box<dyn Tool>],
) -> Result<()> {
    // SystemConfig entries from AppConfig
    let config_entries = vec![
        // Runtime: model, provider, max_output_tokens, max_tool_rounds, temperature
        // Security: consent threshold, egress policy, allowed hosts
        // Memory: db_path, knowledge config
    ];
    knowledge.replace_kind(&KnowledgeKind::SystemConfig, config_entries).await?;

    // ToolCapability entries from registered tools
    let tool_entries: Vec<_> = tools.iter().map(|t| {
        // name, description, required_capability, risk_level, side_effects
    }).collect();
    knowledge.replace_kind(&KnowledgeKind::ToolCapability, tool_entries).await?;

    Ok(())
}
```

`replace_kind` does `DELETE WHERE kind = ? + INSERT` in a single transaction. Idempotent across restarts.

---

## 5. Knowledge Security Model & Consent

### Permission Matrix

| Kind | Agent read | Agent write | Consent required | Agent delete | Consent required |
|---|---|---|---|---|---|
| `SystemConfig` | Yes | Yes | **Yes** | Yes | **Yes** |
| `ToolCapability` | Yes | Yes | **Yes** | Yes | **Yes** |
| `UserPreference` | Yes | Yes | **Yes** | Yes | **Yes** |
| `LearnedPattern` | Yes | Yes | No | Yes | No |
| `ErrorResolution` | Yes | Yes | No | Yes | No |
| `SessionInsight` | Yes | Yes | No | Yes | No |

Principle: anything affecting agent identity (config), capabilities (tools), or user-declared preferences requires human approval. Agent's own observations are its private scratchpad.

### Sensitive Content Filtering

In `freebird-security/src/sensitive.rs`:

Runs on ALL knowledge writes (regardless of kind or consent), BEFORE the consent gate:

```rust
fn contains_sensitive_content(content: &str) -> Option<&'static str>
```

Patterns detected:
- API keys / tokens: `sk-...`, `ghp_...`, `xoxb-...`, `Bearer ...`
- Passwords: `password=`, `passwd:`, `secret_key`, `private_key`
- PEM blocks: `-----BEGIN (RSA |EC |)PRIVATE KEY-----`
- AWS / cloud credentials: `AKIA...`, `aws_secret_access_key`
- Generic high-entropy strings resembling secrets (base64 > 32 chars)

On detection: `ToolError::SecurityViolation` with explanation. Entry is never stored or presented for consent.

### Dynamic Consent in Knowledge Tools

Knowledge tools use dynamic `RiskLevel` based on the `kind` field:

```rust
async fn execute(&self, input: Value, context: &ToolContext<'_>) -> Result<ToolOutput, ToolError> {
    let kind: KnowledgeKind = /* parse from input */;

    // 1. Sensitive content scan (always, before consent)
    if let Some(reason) = contains_sensitive_content(&content) {
        return Err(ToolError::SecurityViolation { ... });
    }

    // 2. Consent gate for protected kinds
    if kind.requires_consent() {
        context.consent_gate()?.request_consent(ConsentRequest { ... }).await?;
    }

    // 3. Execute operation
    // ...
}
```

### Consent Request Format

```
┌─ Consent Required ─────────────────────────────────────┐
│ store_knowledge wants to store a UserPreference:        │
│                                                         │
│ Content: "User prefers concise responses without emojis"│
│ Tags: [communication, style]                            │
│ Confidence: 0.9                                         │
│                                                         │
│ [y] Approve  [n] Deny                                   │
└─────────────────────────────────────────────────────────┘
```

---

## 6. Encryption Infrastructure

### Key Derivation Module

In `freebird-security/src/db_key.rs`:

```rust
pub struct DbKeyDeriver {
    salt_path: PathBuf,
    iterations: u32,
}

impl DbKeyDeriver {
    pub fn derive(passphrase: SecretString) -> Result<SecretString, SecurityError>
}
```

Flow:
1. Load or create salt file (`~/.freebird/db.salt`, 32 bytes from `ring::rand::SystemRandom`, 0600 perms)
2. `ring::pbkdf2::derive(PBKDF2_HMAC_SHA256, iterations, &salt, passphrase, &mut key)`
3. Encode 32-byte key as 64-char hex string
4. Wrap in `SecretString`, zeroize passphrase + intermediates

### Key Source Resolution

In `freebird-security/src/db_key.rs`:

```rust
pub struct KeySource {
    env_var: &'static str,          // "FREEBIRD_DB_KEY"
    keyfile_path: Option<PathBuf>,
}

impl KeySource {
    pub fn resolve(&self, allow_prompt: bool) -> Result<SecretString, SecurityError>
}
```

Priority: env var (cleared after read) → keyfile (0600 enforced) → interactive prompt (if terminal attached).

### Keyfile Permission Enforcement

```rust
fn validate_keyfile_permissions(path: &Path) -> Result<(), SecurityError>
// Checks mode & 0o777 == 0o600, rejects otherwise
// #[cfg(unix)] — returns descriptive error on Windows
```

### Database Connection

In `freebird-memory/src/sqlite.rs`:

```rust
pub struct SqliteDb {
    conn: tokio::sync::Mutex<rusqlite::Connection>,
}

impl SqliteDb {
    pub fn open(path: &Path, key: &SecretString) -> Result<Self, MemoryError>
}
```

Open flow:
1. `Connection::open(path)`
2. `PRAGMA key = 'x\"...\"'` — MUST be first statement
3. `SELECT count(*) FROM sqlite_master` — verifies key is correct
4. Run migrations
5. Wrap in `tokio::sync::Mutex`

Single mutex'd connection (not a pool) — SQLCipher key is per-connection, FreeBird handles one user at a time. Pool can be added later if needed.

### Migration System

Embedded SQL files via `include_str!`. Schema version tracked in `schema_version` table. Each migration runs in `execute_batch`. Forward-only (no rollback).

### First-Run Setup

```
freebird serve (first time)
  → No freebird.db → resolve key → create salt → derive key
  → Create DB → apply key → run 001_initial.sql
  → Check for ~/.freebird/conversations/*.json → migrate to DB
  → Populate system knowledge → ready

freebird serve (subsequent)
  → Resolve key → load salt → derive key
  → Open DB → verify key → run pending migrations
  → Populate system knowledge (idempotent) → ready
```

### FileMemory Migration

On first startup with SQLite:
1. Check if `~/.freebird/conversations/*.json` exists
2. Read each JSON file, insert into `conversations` table
3. Move originals to `~/.freebird/conversations.bak/`
4. Log migration progress

---

## 7. Documentation & README

### README Security Section

Add comprehensive security features table to README with implementation status for every feature:
- Taint tracking, injection scanning, capability system, consent gates, egress control, output scanning: **Implemented**
- Encrypted database, knowledge store security: **Implemented** (after this work)
- Session key auth, channel pairing, token budgets, memory HMAC, supply chain CI: **Planned**

Include subsections for:
- Encryption (SQLCipher, key sources, zeroization)
- Knowledge store security (consent split, sensitive content filter)
- Threat model (semi-trusted LLM, untrusted tool output, trusted-but-sandboxed filesystem, hostile network)

### CLAUDE.md Updates

- Section 1: Add `rusqlite` to dependency table
- Section 3: Update message flow diagram with knowledge retrieval step
- Section 5: Update agent runtime loop description
- Section 6: Add knowledge security (sensitive filter, consent split)
- Section 14: Update memory integrity for SQLite/SQLCipher
- Section 19: Add `[knowledge]` config section, encryption key config
- Section 24: Add knowledge-specific security checklist items

### Inline Module Documentation

Every new module gets a module-level doc comment explaining its purpose and security role.

---

## 8. Implementation Order

Sub-stories are ordered by dependency — each builds on the previous:

1. **KnowledgeStore trait + types** (`freebird-traits`) — foundation, no deps
2. **Config changes** (`freebird-types`) — `MemoryConfig` update, `KnowledgeConfig` addition
3. **Encryption infrastructure** (`freebird-security`) — key derivation, source resolution, sensitive content filter
4. **SQLite infrastructure** (`freebird-memory`) — `SqliteDb`, migrations, connection management
5. **SqliteMemory** (`freebird-memory`) — `Memory` trait impl over SQLite, replaces `FileMemory`
6. **SqliteKnowledgeStore** (`freebird-memory`) — `KnowledgeStore` trait impl with FTS5
7. **FileMemory migration** (`freebird-memory` / `freebird-daemon`) — JSON → SQLite one-time migration
8. **Knowledge tools** (`freebird-tools`) — store, search, update, delete with consent + sensitive filter
9. **Agent loop integration** (`freebird-runtime`) — auto-retrieval, context injection, `AgentRuntime` changes
10. **Startup bootstrap** (`freebird-daemon`) — key resolution, DB init, system knowledge population, composition
11. **Documentation** — README security section, CLAUDE.md updates, inline docs
12. **Tests** — unit tests per module, integration test for full flow, property tests for sensitive filter
