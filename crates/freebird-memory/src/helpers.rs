//! Shared helper functions for memory backends.

use freebird_traits::memory::{Conversation, SessionSummary};
use freebird_traits::provider::{ContentBlock, Message};

/// Build a [`SessionSummary`] from a [`Conversation`].
///
/// Extracts a preview from the first user message (up to 100 chars).
pub fn conversation_to_summary(conv: &Conversation) -> SessionSummary {
    let preview = conv
        .turns
        .first()
        .and_then(|t| t.user_message.content.first())
        .map(|block| match block {
            ContentBlock::Text { text } => text.chars().take(100).collect(),
            _ => String::new(),
        })
        .unwrap_or_default();

    SessionSummary {
        session_id: conv.session_id.clone(),
        created_at: conv.created_at,
        updated_at: conv.updated_at,
        turn_count: conv.turns.len(),
        model_id: conv.model_id.clone(),
        preview,
    }
}

/// Check if a conversation contains the query string in any text content block.
///
/// Searches user messages and assistant responses. Does NOT search tool
/// invocation input/output (JSON, not useful for text search).
pub fn conversation_contains(conv: &Conversation, query_lower: &str) -> bool {
    conv.turns.iter().any(|turn| {
        message_contains(&turn.user_message, query_lower)
            || turn
                .assistant_messages
                .iter()
                .any(|msg| message_contains(msg, query_lower))
    })
}

/// Check if any text content block in a message contains the query.
fn message_contains(msg: &Message, query_lower: &str) -> bool {
    msg.content.iter().any(|block| {
        matches!(block, ContentBlock::Text { text } if text.to_lowercase().contains(query_lower))
    })
}
