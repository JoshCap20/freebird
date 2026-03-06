//! Wire protocol types for daemon ↔ client communication over TCP + JSON-lines.
//!
//! Both `ClientMessage` and `ServerMessage` are single source of truth — used
//! by `TcpChannel` (daemon side) and `freebird chat` (client side).

use serde::{Deserialize, Serialize};

/// Messages sent by the client (`freebird chat`) to the daemon.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ClientMessage {
    /// User typed a chat message.
    Message { text: String },
    /// User typed a /command.
    Command { name: String, args: Vec<String> },
    /// Client is disconnecting gracefully.
    Disconnect,
}

/// Messages sent by the daemon to a connected client.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ServerMessage {
    /// Complete response text.
    Message { text: String },
    /// Streaming chunk (partial response).
    StreamChunk { text: String },
    /// Stream finished.
    StreamEnd,
    /// Error message.
    Error { text: String },
    /// Command response (e.g., /help output).
    CommandResponse { text: String },
    /// Tool execution started.
    ToolStart { tool_name: String },
    /// Tool execution completed.
    ToolEnd {
        tool_name: String,
        outcome: String,
        duration_ms: u64,
    },
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    // ── ClientMessage serde ──────────────────────────────────────────

    #[test]
    fn client_message_serializes_as_tagged_json() {
        let msg = ClientMessage::Message {
            text: "hello".into(),
        };
        let json = serde_json::to_string(&msg).unwrap();
        assert_eq!(json, r#"{"type":"message","text":"hello"}"#);
    }

    #[test]
    fn client_command_serializes_with_args() {
        let msg = ClientMessage::Command {
            name: "model".into(),
            args: vec!["opus".into()],
        };
        let json = serde_json::to_string(&msg).unwrap();
        assert_eq!(json, r#"{"type":"command","name":"model","args":["opus"]}"#);
    }

    #[test]
    fn client_disconnect_serializes() {
        let msg = ClientMessage::Disconnect;
        let json = serde_json::to_string(&msg).unwrap();
        assert_eq!(json, r#"{"type":"disconnect"}"#);
    }

    #[test]
    fn client_message_roundtrips() {
        for msg in [
            ClientMessage::Message {
                text: "what is 2+2?".into(),
            },
            ClientMessage::Command {
                name: "new".into(),
                args: vec![],
            },
            ClientMessage::Disconnect,
        ] {
            let json = serde_json::to_string(&msg).unwrap();
            let back: ClientMessage = serde_json::from_str(&json).unwrap();
            assert_eq!(msg, back);
        }
    }

    // ── ServerMessage serde ──────────────────────────────────────────

    #[test]
    fn server_message_serializes_as_tagged_json() {
        let msg = ServerMessage::Message { text: "4".into() };
        let json = serde_json::to_string(&msg).unwrap();
        assert_eq!(json, r#"{"type":"message","text":"4"}"#);
    }

    #[test]
    fn server_stream_chunk_serializes() {
        let msg = ServerMessage::StreamChunk {
            text: "partial".into(),
        };
        let json = serde_json::to_string(&msg).unwrap();
        assert_eq!(json, r#"{"type":"stream_chunk","text":"partial"}"#);
    }

    #[test]
    fn server_stream_end_serializes() {
        let msg = ServerMessage::StreamEnd;
        let json = serde_json::to_string(&msg).unwrap();
        assert_eq!(json, r#"{"type":"stream_end"}"#);
    }

    #[test]
    fn server_error_serializes() {
        let msg = ServerMessage::Error {
            text: "boom".into(),
        };
        let json = serde_json::to_string(&msg).unwrap();
        assert_eq!(json, r#"{"type":"error","text":"boom"}"#);
    }

    #[test]
    fn server_command_response_serializes() {
        let msg = ServerMessage::CommandResponse {
            text: "done".into(),
        };
        let json = serde_json::to_string(&msg).unwrap();
        assert_eq!(json, r#"{"type":"command_response","text":"done"}"#);
    }

    #[test]
    fn server_tool_start_serializes() {
        let msg = ServerMessage::ToolStart {
            tool_name: "read_file".into(),
        };
        let json = serde_json::to_string(&msg).unwrap();
        assert_eq!(json, r#"{"type":"tool_start","tool_name":"read_file"}"#);
    }

    #[test]
    fn server_tool_end_serializes() {
        let msg = ServerMessage::ToolEnd {
            tool_name: "read_file".into(),
            outcome: "success".into(),
            duration_ms: 42,
        };
        let json = serde_json::to_string(&msg).unwrap();
        assert_eq!(
            json,
            r#"{"type":"tool_end","tool_name":"read_file","outcome":"success","duration_ms":42}"#
        );
    }

    #[test]
    fn server_message_roundtrips() {
        for msg in [
            ServerMessage::Message {
                text: "hello".into(),
            },
            ServerMessage::StreamChunk {
                text: "chunk".into(),
            },
            ServerMessage::StreamEnd,
            ServerMessage::Error { text: "err".into() },
            ServerMessage::CommandResponse { text: "ok".into() },
            ServerMessage::ToolStart {
                tool_name: "shell".into(),
            },
            ServerMessage::ToolEnd {
                tool_name: "shell".into(),
                outcome: "error".into(),
                duration_ms: 100,
            },
        ] {
            let json = serde_json::to_string(&msg).unwrap();
            let back: ServerMessage = serde_json::from_str(&json).unwrap();
            assert_eq!(msg, back);
        }
    }

    // ── JSON-line framing validation ─────────────────────────────────

    #[test]
    fn json_line_contains_no_embedded_newlines() {
        let msg = ClientMessage::Message {
            text: "line1\nline2".into(),
        };
        let json = serde_json::to_string(&msg).unwrap();
        // serde_json escapes \n as \\n in the JSON string, so no raw newlines
        assert!(
            !json.contains('\n'),
            "JSON-line must not contain raw newlines"
        );
    }

    #[test]
    fn invalid_type_tag_fails_deserialization() {
        let bad = r#"{"type":"unknown","text":"x"}"#;
        let result = serde_json::from_str::<ClientMessage>(bad);
        assert!(result.is_err());
    }
}
