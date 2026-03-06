//! Channel trait — abstracts over transport layers (CLI, Signal, `WebSocket`, etc.).

use std::collections::BTreeSet;
use std::fmt;
use std::pin::Pin;

use async_trait::async_trait;
use futures::Stream;
use serde::{Deserialize, Serialize};

use crate::id::ChannelId;

/// Optional features a channel may support.
///
/// Enum set instead of boolean flags (CLAUDE.md §30) for extensibility.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ChannelFeature {
    Media,
    Streaming,
}

/// Whether a channel requires authentication before processing messages.
///
/// Enum instead of `bool` (CLAUDE.md §30: "`bool` parameters → Enums").
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AuthRequirement {
    /// Channel does not require authentication (e.g., local CLI).
    None,
    /// Channel requires explicit pairing/auth before message processing.
    Required,
}

/// Metadata about a channel implementation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChannelInfo {
    pub id: ChannelId,
    pub display_name: String,
    pub features: BTreeSet<ChannelFeature>,
    pub auth: AuthRequirement,
}

impl ChannelInfo {
    /// Check whether this channel supports a specific feature.
    #[must_use]
    pub fn supports(&self, feature: &ChannelFeature) -> bool {
        self.features.contains(feature)
    }

    /// Whether this channel requires authentication.
    #[must_use]
    pub fn requires_auth(&self) -> bool {
        self.auth == AuthRequirement::Required
    }
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
    Message {
        text: String,
        recipient_id: String,
    },
    StreamChunk {
        text: String,
        recipient_id: String,
    },
    StreamEnd {
        recipient_id: String,
    },
    Error {
        text: String,
        recipient_id: String,
    },
    ToolStart {
        tool_name: String,
        recipient_id: String,
    },
    ToolEnd {
        tool_name: String,
        outcome: String,
        duration_ms: u64,
        recipient_id: String,
    },
    /// The full agentic turn is complete — no more events for this user message.
    TurnComplete {
        recipient_id: String,
    },
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

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn test_channel_feature_serde_roundtrip() {
        for (feature, expected_json) in [
            (ChannelFeature::Media, "\"media\""),
            (ChannelFeature::Streaming, "\"streaming\""),
        ] {
            let json = serde_json::to_string(&feature).unwrap();
            assert_eq!(json, expected_json);
            let back: ChannelFeature = serde_json::from_str(&json).unwrap();
            assert_eq!(back, feature);
        }
    }

    #[test]
    fn test_auth_requirement_serde_roundtrip() {
        for (req, expected_json) in [
            (AuthRequirement::None, "\"none\""),
            (AuthRequirement::Required, "\"required\""),
        ] {
            let json = serde_json::to_string(&req).unwrap();
            assert_eq!(json, expected_json);
            let back: AuthRequirement = serde_json::from_str(&json).unwrap();
            assert_eq!(back, req);
        }
    }

    #[test]
    fn test_channel_info_supports_feature() {
        let info = ChannelInfo {
            id: ChannelId::from("cli"),
            display_name: "Command Line Interface".into(),
            features: BTreeSet::from([ChannelFeature::Streaming]),
            auth: AuthRequirement::None,
        };

        assert!(info.supports(&ChannelFeature::Streaming));
        assert!(!info.supports(&ChannelFeature::Media));
        assert!(!info.requires_auth());
    }

    #[test]
    fn test_channel_info_requires_auth() {
        let info = ChannelInfo {
            id: ChannelId::from("signal"),
            display_name: "Signal".into(),
            features: BTreeSet::from([ChannelFeature::Media, ChannelFeature::Streaming]),
            auth: AuthRequirement::Required,
        };

        assert!(info.requires_auth());
        assert!(info.supports(&ChannelFeature::Media));
    }

    #[test]
    fn test_channel_info_uses_channel_id_newtype() {
        let info = ChannelInfo {
            id: ChannelId::from("websocket"),
            display_name: "WebSocket".into(),
            features: BTreeSet::new(),
            auth: AuthRequirement::Required,
        };
        assert_eq!(info.id.as_str(), "websocket");
    }

    #[test]
    fn test_channel_info_serde_roundtrip() {
        let info = ChannelInfo {
            id: ChannelId::from("cli"),
            display_name: "CLI".into(),
            features: BTreeSet::from([ChannelFeature::Streaming]),
            auth: AuthRequirement::None,
        };

        let json = serde_json::to_string(&info).unwrap();
        let back: ChannelInfo = serde_json::from_str(&json).unwrap();
        assert_eq!(back.id.as_str(), "cli");
        assert!(back.supports(&ChannelFeature::Streaming));
        assert!(!back.requires_auth());
    }
}
