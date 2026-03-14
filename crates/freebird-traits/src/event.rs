//! Event-sourced conversation persistence types and trait.
//!
//! Each action in a conversation (user message, tool call, assistant response)
//! becomes an immutable [`ConversationEvent`]. The [`EventSink`] trait allows
//! the runtime to persist events immediately as they occur, enabling sub-turn
//! crash recovery.

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::id::SessionId;
use crate::memory::{MemoryError, ToolInvocation};
use crate::provider::Message;

/// An immutable event that represents a single action in a conversation.
///
/// Events are appended to an event log and replayed to reconstruct
/// [`Conversation`](crate::memory::Conversation) state. The `turn_index`
/// fields provide deterministic ordering for replay.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ConversationEvent {
    /// Session created with initial metadata.
    SessionCreated {
        system_prompt: Option<String>,
        model_id: String,
        provider_id: String,
    },
    /// Session metadata updated (model/provider/prompt change).
    SessionMetadataUpdated {
        system_prompt: Option<String>,
        model_id: String,
        provider_id: String,
    },
    /// User sent a message, starting a new turn.
    TurnStarted {
        turn_index: usize,
        user_message: Message,
    },
    /// Assistant produced a response (text or `tool_use`).
    AssistantMessage {
        turn_index: usize,
        message_index: usize,
        message: Message,
    },
    /// A tool was invoked and produced output.
    ToolInvoked {
        turn_index: usize,
        invocation_index: usize,
        invocation: ToolInvocation,
    },
    /// Turn completed (assistant finished responding).
    TurnCompleted {
        turn_index: usize,
        completed_at: DateTime<Utc>,
    },
}

impl ConversationEvent {
    /// Returns the event type discriminator as a static string.
    #[must_use]
    pub const fn event_type(&self) -> &'static str {
        match self {
            Self::SessionCreated { .. } => "session_created",
            Self::SessionMetadataUpdated { .. } => "session_metadata_updated",
            Self::TurnStarted { .. } => "turn_started",
            Self::AssistantMessage { .. } => "assistant_message",
            Self::ToolInvoked { .. } => "tool_invoked",
            Self::TurnCompleted { .. } => "turn_completed",
        }
    }
}

/// Sink for persisting conversation events immediately as they occur.
///
/// The runtime calls [`append`](EventSink::append) at each action point in
/// the agentic loop, providing sub-turn crash recovery. The [`Memory`](crate::memory::Memory)
/// trait's [`save`](crate::memory::Memory::save) method then only needs to
/// update session metadata.
#[async_trait]
pub trait EventSink: Send + Sync + 'static {
    /// Append a conversation event to the persistent event log.
    ///
    /// Events are immutable once appended. The implementation is responsible
    /// for assigning sequence numbers and maintaining HMAC chain integrity.
    async fn append(
        &self,
        session_id: &SessionId,
        event: ConversationEvent,
    ) -> Result<(), MemoryError>;
}
