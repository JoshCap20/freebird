//! Channel trait — abstracts over transport layers (CLI, Signal, WebSocket, etc.).

use std::fmt;
use std::pin::Pin;

use async_trait::async_trait;
use futures::Stream;
use serde::{Deserialize, Serialize};

/// Metadata about a channel implementation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChannelInfo {
    pub id: String,
    pub display_name: String,
    pub supports_media: bool,
    pub supports_streaming: bool,
    pub requires_auth: bool,
}

/// An inbound event from a channel.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum InboundEvent {
    Message {
        raw_text: String,
        sender_id: String,
        attachments: Vec<Attachment>,
    },
    Connected {
        sender_id: String,
    },
    Disconnected {
        sender_id: String,
    },
    Command {
        name: String,
        args: Vec<String>,
        sender_id: String,
    },
}

/// An outbound event to send to the user via the channel.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum OutboundEvent {
    Message { text: String, recipient_id: String },
    StreamChunk { text: String, recipient_id: String },
    StreamEnd { recipient_id: String },
    Error { text: String, recipient_id: String },
}

/// A media attachment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Attachment {
    pub filename: String,
    pub media_type: String,
    pub data: Vec<u8>,
}

/// The handle returned by [`Channel::start`].
///
/// Ownership: `inbound` is a single-consumer stream (not Clone).
/// `outbound` is a tokio `mpsc::Sender` (Clone). The runtime owns the
/// stream and may clone the sender for concurrent use.
pub struct ChannelHandle {
    pub inbound: Pin<Box<dyn Stream<Item = InboundEvent> + Send>>,
    pub outbound: tokio::sync::mpsc::Sender<OutboundEvent>,
}

impl fmt::Debug for ChannelHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ChannelHandle")
            .field("inbound", &"<stream>")
            .field("outbound", &self.outbound)
            .finish()
    }
}

/// The core channel trait.
#[async_trait]
pub trait Channel: Send + Sync + 'static {
    fn info(&self) -> &ChannelInfo;
    async fn start(&self) -> Result<ChannelHandle, ChannelError>;
    async fn stop(&self) -> Result<(), ChannelError>;
}

/// Channel-specific errors.
#[derive(Debug, thiserror::Error)]
pub enum ChannelError {
    #[error("channel `{channel}` failed to start: {reason}")]
    StartupFailed { channel: String, reason: String },

    #[error("channel `{channel}` connection lost: {reason}")]
    ConnectionLost { channel: String, reason: String },

    #[error("failed to send message on channel `{channel}`: {reason}")]
    SendFailed { channel: String, reason: String },

    #[error("channel `{channel}` authentication failed")]
    AuthenticationFailed { channel: String },

    #[error("channel IO error: {0}")]
    Io(#[from] std::io::Error),
}
