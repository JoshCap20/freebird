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
