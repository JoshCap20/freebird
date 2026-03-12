-- FreeBird database schema v1: event-sourced conversations, knowledge, audit

-- Schema version tracking
CREATE TABLE IF NOT EXISTS schema_version (
    version    INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL
);

-- ---------------------------------------------------------------------------
-- Knowledge
-- ---------------------------------------------------------------------------

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

-- ---------------------------------------------------------------------------
-- Conversation Events (immutable event log with per-session HMAC chain)
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS conversation_events (
    event_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id     TEXT NOT NULL,
    sequence       INTEGER NOT NULL,
    event_type     TEXT NOT NULL,
    event_data     TEXT NOT NULL,
    timestamp      TEXT NOT NULL,
    previous_hmac  TEXT NOT NULL DEFAULT '',
    hmac           TEXT NOT NULL,
    UNIQUE(session_id, sequence)
);

CREATE INDEX IF NOT EXISTS idx_events_session ON conversation_events(session_id, sequence);
CREATE INDEX IF NOT EXISTS idx_events_session_type ON conversation_events(session_id, event_type);

-- ---------------------------------------------------------------------------
-- Session Metadata (denormalized for list/search without full replay)
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS session_metadata (
    session_id    TEXT PRIMARY KEY,
    system_prompt TEXT,
    model_id      TEXT NOT NULL,
    provider_id   TEXT NOT NULL,
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL,
    turn_count    INTEGER NOT NULL DEFAULT 0,
    preview       TEXT NOT NULL DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_session_meta_updated ON session_metadata(updated_at DESC);

-- ---------------------------------------------------------------------------
-- Conversation FTS5 (text search over event data)
-- ---------------------------------------------------------------------------

CREATE VIRTUAL TABLE IF NOT EXISTS conversation_fts USING fts5(
    session_id,
    content,
    content=conversation_events,
    content_rowid=event_id,
    tokenize='porter unicode61'
);

CREATE TRIGGER IF NOT EXISTS events_fts_ai AFTER INSERT ON conversation_events BEGIN
    INSERT INTO conversation_fts(rowid, session_id, content)
    VALUES (new.event_id, new.session_id, new.event_data);
END;

CREATE TRIGGER IF NOT EXISTS events_fts_ad AFTER DELETE ON conversation_events BEGIN
    INSERT INTO conversation_fts(conversation_fts, rowid, session_id, content)
    VALUES ('delete', old.event_id, old.session_id, old.event_data);
END;

-- ---------------------------------------------------------------------------
-- Audit Events (security audit log with global HMAC chain)
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS audit_events (
    event_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    sequence       INTEGER NOT NULL UNIQUE,
    session_id     TEXT,
    event_type     TEXT NOT NULL,
    event_data     TEXT NOT NULL,
    timestamp      TEXT NOT NULL,
    previous_hmac  TEXT NOT NULL DEFAULT '',
    hmac           TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_audit_sequence ON audit_events(sequence);
CREATE INDEX IF NOT EXISTS idx_audit_session ON audit_events(session_id);
CREATE INDEX IF NOT EXISTS idx_audit_type ON audit_events(event_type);
