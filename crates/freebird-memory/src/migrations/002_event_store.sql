-- FreeBird database schema v2: event-sourced conversation persistence
--
-- Replaces the JSON blob in `conversations.data` with an immutable event log.
-- Also adds audit_events table to replace the file-based JSONL audit logger.

-- Immutable conversation events with per-session HMAC chain
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

-- Denormalized session metadata for list_sessions/search without full replay
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

-- FTS5 for text search over conversation events
CREATE VIRTUAL TABLE IF NOT EXISTS conversation_fts USING fts5(
    session_id,
    content,
    content=conversation_events,
    content_rowid=event_id,
    tokenize='porter unicode61'
);

-- Keep FTS5 in sync with conversation_events
CREATE TRIGGER IF NOT EXISTS events_fts_ai AFTER INSERT ON conversation_events BEGIN
    INSERT INTO conversation_fts(rowid, session_id, content)
    VALUES (new.event_id, new.session_id, new.event_data);
END;

CREATE TRIGGER IF NOT EXISTS events_fts_ad AFTER DELETE ON conversation_events BEGIN
    INSERT INTO conversation_fts(conversation_fts, rowid, session_id, content)
    VALUES ('delete', old.event_id, old.session_id, old.event_data);
END;

-- Security audit events with global HMAC chain (replaces JSONL file)
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
