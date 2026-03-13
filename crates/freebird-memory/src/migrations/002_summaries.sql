-- Conversation summaries for context window compression.
-- One summary per session (upsert on re-summarization).

CREATE TABLE IF NOT EXISTS conversation_summaries (
    session_id                TEXT PRIMARY KEY,
    summary_text              TEXT NOT NULL,
    summarized_through_turn   INTEGER NOT NULL,
    original_token_estimate   INTEGER NOT NULL,
    generated_at              TEXT NOT NULL
);
