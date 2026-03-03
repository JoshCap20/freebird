//! Channel trait — abstracts over transport layers (CLI, Signal, WebSocket, etc.).

use std::pin::Pin;

use async_trait::async_trait;
use tokio_stream::Stream;

/// Metadata about a channel implementation.
#[derive(Debug, Clone)]
pub struct ChannelInfo {
    /// Unique identifier (e.g., "cli", "signal", "websocket").
    pub id: String,
    /// Human-readable name (e.g., "Command Line Interface").
    pub display_name: String,
    /// Whether this channel supports rich content (images, files).
    pub supports_media: bool,
    /// Whether this channel supports real-time streaming of responses.
    pub supports_streaming: bool,
    /// Whether this channel requires authentication/pairing before use.
    pub requires_auth: bool,
}

/// An inbound event from a channel.
#[derive(Debug, Clone)]
pub enum InboundEvent {
    /// A new message from the user.
    Message {
        /// Raw text from the user (will be wrapped in Tainted<Untrusted> by the router).
        raw_text: String,
        /// Channel-specific sender identifier (e.g., phone number, username).
        sender_id: String,
        /// Optional media attachments.
        attachments: Vec<Attachment>,
    },
    /// The user has connected/started a session.
    Connected { sender_id: String },
    /// The user has disconnected.
    Disconnected { sender_id: String },
    /// A control command (e.g., /new, /status, /model).
    Command {
        name: String,
        args: Vec<String>,
        sender_id: String,
    },
}

/// An outbound event to send to the user via the channel.
#[derive(Debug, Clone)]
pub enum OutboundEvent {
    /// A complete text response.
    Message { text: String, recipient_id: String },
    /// A streaming text chunk (for channels that support it).
    StreamChunk { text: String, recipient_id: String },
    /// Signal that streaming is complete.
    StreamEnd { recipient_id: String },
    /// An error message to display to the user.
    Error { text: String, recipient_id: String },
}

/// A media attachment (image, file, audio).
#[derive(Debug, Clone)]
pub struct Attachment {
    pub filename: String,
    pub media_type: String,
    pub data: Vec<u8>,
}

/// The handle returned by [`Channel::start`].
pub struct ChannelHandle {
    /// Stream of inbound events from the user.
    pub inbound: Pin<Box<dyn Stream<Item = InboundEvent> + Send>>,
    /// Sender for outbound events to the user.
    pub outbound: tokio::sync::mpsc::Sender<OutboundEvent>,
}

/// The core channel trait.
///
/// Lifecycle: `start()` is called once. It returns a stream of inbound events
/// and a sender for outbound events. The runtime consumes the stream and
/// sends responses via the sender. `stop()` is called during shutdown.
#[async_trait]
pub trait Channel: Send + Sync + 'static {
    /// Return metadata about this channel.
    fn info(&self) -> &ChannelInfo;

    /// Start the channel, returning a handle with inbound stream and outbound sender.
    async fn start(&self) -> Result<ChannelHandle, ChannelError>;

    /// Gracefully stop the channel, closing connections and flushing buffers.
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
