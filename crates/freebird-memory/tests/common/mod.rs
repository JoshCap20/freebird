use chrono::{DateTime, Utc};
use freebird_traits::id::SessionId;
use freebird_traits::memory::{Conversation, Turn};
use freebird_traits::provider::{ContentBlock, Message, Role};

/// Build a minimal Conversation for testing.
pub fn make_conversation(
    session_id: &str,
    user_text: &str,
    updated_at: DateTime<Utc>,
) -> Conversation {
    Conversation {
        session_id: SessionId::from_string(session_id),
        system_prompt: None,
        turns: vec![Turn {
            user_message: Message {
                role: Role::User,
                content: vec![ContentBlock::Text {
                    text: user_text.to_string(),
                }],
                timestamp: updated_at,
            },
            assistant_messages: vec![Message {
                role: Role::Assistant,
                content: vec![ContentBlock::Text {
                    text: "Response".to_string(),
                }],
                timestamp: updated_at,
            }],
            tool_invocations: vec![],
            started_at: updated_at,
            completed_at: Some(updated_at),
        }],
        created_at: updated_at,
        updated_at,
        model_id: "test-model".to_string(),
        provider_id: "test-provider".to_string(),
    }
}
