//! TCP channel — accepts JSON-line connections from `freebird chat` clients.
//!
//! Implements the `Channel` trait. Each TCP connection is an independent
//! client identified by `"tcp-{n}"`. Supports multiple concurrent clients.

use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use async_trait::async_trait;
use tokio::io::AsyncWriteExt;
use tokio::net::TcpListener;
use tokio::sync::mpsc;
use tokio_stream::StreamExt as _;
use tokio_stream::wrappers::ReceiverStream;
use tokio_util::codec::{FramedRead, LinesCodec, LinesCodecError};
use tokio_util::sync::CancellationToken;

use freebird_traits::channel::{
    AuthRequirement, Channel, ChannelError, ChannelFeature, ChannelHandle, ChannelInfo,
    InboundEvent, OutboundEvent,
};
use freebird_traits::id::ChannelId;
use freebird_types::protocol::{ClientMessage, ServerMessage};

/// Maximum bytes per JSON line from a client connection.
///
/// Limits per-read memory allocation to prevent OOM from malicious clients
/// sending arbitrarily long lines without a newline terminator.
/// Connections that exceed this limit are terminated — a client sending
/// >64 KiB in a single line is either malicious or broken.
const MAX_LINE_BYTES: usize = 65_536;

/// Per-connection writer senders, keyed by `sender_id` (e.g., `"tcp-0"`).
type WriterMap = HashMap<String, mpsc::Sender<ServerMessage>>;

/// TCP channel — listens on a TCP port, bridges JSON-line protocol to `Channel` trait.
pub struct TcpChannel {
    info: ChannelInfo,
    host: String,
    port: u16,
    cancel: CancellationToken,
}

impl TcpChannel {
    #[must_use]
    pub fn new(host: impl Into<String>, port: u16) -> Self {
        Self {
            info: ChannelInfo {
                id: ChannelId::from("tcp"),
                display_name: "TCP JSON-Lines".into(),
                features: BTreeSet::from([ChannelFeature::Streaming]),
                auth: AuthRequirement::None,
            },
            host: host.into(),
            port,
            cancel: CancellationToken::new(),
        }
    }

    /// Start with a pre-bound listener (for testing).
    pub(crate) fn start_with_listener(&self, listener: TcpListener) -> ChannelHandle {
        let (inbound_tx, inbound_rx) = mpsc::channel::<InboundEvent>(256);
        let (outbound_tx, outbound_rx) = mpsc::channel::<OutboundEvent>(256);

        let cancel = self.cancel.clone();
        let conn_counter = Arc::new(AtomicU64::new(0));

        // Spawn the accept loop
        tokio::spawn(accept_loop(
            listener,
            inbound_tx,
            outbound_rx,
            cancel,
            conn_counter,
        ));

        ChannelHandle {
            inbound: Box::pin(ReceiverStream::new(inbound_rx)),
            outbound: outbound_tx,
        }
    }
}

#[async_trait]
impl Channel for TcpChannel {
    fn info(&self) -> &ChannelInfo {
        &self.info
    }

    async fn start(&self) -> Result<ChannelHandle, ChannelError> {
        let addr = format!("{}:{}", self.host, self.port);
        let listener = TcpListener::bind(&addr)
            .await
            .map_err(|e| ChannelError::StartupFailed {
                channel: "tcp".into(),
                reason: format!("failed to bind {addr}: {e}"),
            })?;

        tracing::info!(addr = %addr, "TCP channel listening");
        Ok(self.start_with_listener(listener))
    }

    async fn stop(&self) -> Result<(), ChannelError> {
        self.cancel.cancel();
        Ok(())
    }
}

/// Accept loop: accepts TCP connections and spawns per-connection tasks.
async fn accept_loop(
    listener: TcpListener,
    inbound_tx: mpsc::Sender<InboundEvent>,
    mut outbound_rx: mpsc::Receiver<OutboundEvent>,
    cancel: CancellationToken,
    conn_counter: Arc<AtomicU64>,
) {
    // Route outbound events to per-connection writers.
    let writers: Arc<tokio::sync::Mutex<WriterMap>> =
        Arc::new(tokio::sync::Mutex::new(HashMap::new()));

    let writers_for_outbound = Arc::clone(&writers);

    // Spawn outbound router: takes OutboundEvents from the runtime and routes
    // them to the correct per-connection writer by recipient_id.
    tokio::spawn(async move {
        while let Some(event) = outbound_rx.recv().await {
            let (recipient_id, server_msg) = outbound_to_server_message(event);
            // Clone the sender inside the lock scope, then drop the guard
            // before the `.send().await` — holding a lock across .await is
            // a concurrency anti-pattern (CLAUDE.md §25, §30).
            let writer_tx = {
                let guard = writers_for_outbound.lock().await;
                guard.get(&recipient_id).cloned()
            };
            if let Some(tx) = writer_tx {
                // Best-effort delivery; if the connection closed, drop silently.
                let _ = tx.send(server_msg).await;
            }
        }
    });

    loop {
        tokio::select! {
            result = listener.accept() => {
                match result {
                    Ok((stream, peer)) => {
                        let id = conn_counter.fetch_add(1, Ordering::Relaxed);
                        let sender_id = format!("tcp-{id}");

                        tracing::info!(peer = %peer, sender_id = %sender_id, "client connected");

                        let (writer_tx, writer_rx) = mpsc::channel::<ServerMessage>(64);
                        {
                            let mut guard = writers.lock().await;
                            guard.insert(sender_id.clone(), writer_tx);
                        }

                        let inbound_tx = inbound_tx.clone();
                        let writers_for_cleanup = Arc::clone(&writers);

                        tokio::spawn(handle_connection(
                            stream,
                            sender_id,
                            inbound_tx,
                            writer_rx,
                            writers_for_cleanup,
                        ));
                    }
                    Err(e) => {
                        tracing::warn!(error = %e, "failed to accept TCP connection");
                    }
                }
            }
            () = cancel.cancelled() => {
                tracing::info!("TCP accept loop shutting down");
                break;
            }
        }
    }
}

/// Handle a single TCP connection: read JSON lines → inbound events, and
/// write server messages from `writer_rx` → JSON lines.
/// Read JSON lines from a single client connection and forward as `InboundEvent`s.
///
/// Uses `LinesCodec` with `MAX_LINE_BYTES` to bound per-line memory allocation,
/// preventing OOM from malicious clients sending arbitrarily long lines.
/// Connections that exceed the limit are terminated (`FramedRead` yields
/// `None` after any codec error, ending the read loop).
async fn read_client_lines(
    read_half: tokio::net::tcp::OwnedReadHalf,
    sender_id: String,
    inbound_tx: mpsc::Sender<InboundEvent>,
) {
    let codec = LinesCodec::new_with_max_length(MAX_LINE_BYTES);
    let mut lines = FramedRead::new(read_half, codec);

    while let Some(result) = lines.next().await {
        match result {
            Ok(line) => {
                let trimmed = line.trim();
                if trimmed.is_empty() {
                    continue;
                }

                match serde_json::from_str::<ClientMessage>(trimmed) {
                    Ok(ClientMessage::Message { text }) => {
                        let _ = inbound_tx
                            .send(InboundEvent::Message {
                                raw_text: text,
                                sender_id: sender_id.clone(),
                                attachments: vec![],
                            })
                            .await;
                    }
                    Ok(ClientMessage::Command { name, args }) => {
                        let _ = inbound_tx
                            .send(InboundEvent::Command {
                                name,
                                args,
                                sender_id: sender_id.clone(),
                            })
                            .await;
                    }
                    Ok(ClientMessage::ApprovalResponse {
                        request_id,
                        approved,
                        reason,
                    }) => {
                        let _ = inbound_tx
                            .send(InboundEvent::ApprovalResponse {
                                request_id,
                                approved,
                                reason,
                                sender_id: sender_id.clone(),
                            })
                            .await;
                    }
                    Ok(ClientMessage::Disconnect) => {
                        break;
                    }
                    Err(e) => {
                        // Truncate logged input to prevent disk exhaustion
                        // from oversized malformed payloads.
                        let preview: &str = if trimmed.len() > 200 {
                            &trimmed[..200]
                        } else {
                            trimmed
                        };
                        tracing::warn!(
                            sender = %sender_id,
                            error = %e,
                            line = %preview,
                            "invalid JSON from client"
                        );
                    }
                }
            }
            Err(LinesCodecError::MaxLineLengthExceeded) => {
                tracing::warn!(
                    sender = %sender_id,
                    max_bytes = MAX_LINE_BYTES,
                    "oversized line from client, disconnecting"
                );
                break;
            }
            Err(e) => {
                tracing::warn!(
                    sender = %sender_id,
                    error = %e,
                    "read error from client"
                );
                break;
            }
        }
    }
}

/// Handle a single TCP connection: read JSON lines → inbound events, and
/// write server messages from `writer_rx` → JSON lines.
async fn handle_connection(
    stream: tokio::net::TcpStream,
    sender_id: String,
    inbound_tx: mpsc::Sender<InboundEvent>,
    mut writer_rx: mpsc::Receiver<ServerMessage>,
    writers: Arc<tokio::sync::Mutex<WriterMap>>,
) {
    let (read_half, mut write_half) = stream.into_split();

    // Send Connected event
    let _ = inbound_tx
        .send(InboundEvent::Connected {
            sender_id: sender_id.clone(),
        })
        .await;

    // Spawn reader task: JSON lines from client → InboundEvent
    let reader_handle = tokio::spawn(read_client_lines(
        read_half,
        sender_id.clone(),
        inbound_tx.clone(),
    ));

    // Spawn writer task: ServerMessage from runtime → JSON lines to client
    let sender_for_writer = sender_id.clone();
    let writer_handle = tokio::spawn(async move {
        while let Some(msg) = writer_rx.recv().await {
            match serde_json::to_string(&msg) {
                Ok(json) => {
                    let line = format!("{json}\n");
                    if write_half.write_all(line.as_bytes()).await.is_err() {
                        break;
                    }
                    if write_half.flush().await.is_err() {
                        break;
                    }
                }
                Err(e) => {
                    tracing::warn!(
                        sender = %sender_for_writer,
                        error = %e,
                        "failed to serialize ServerMessage"
                    );
                }
            }
        }
    });

    // Wait for the reader to finish (client disconnected or errored)
    let _ = reader_handle.await;
    // Abort the writer task — dropping only detaches (orphans) the task.
    writer_handle.abort();

    // Clean up: remove from writers map and send Disconnected
    {
        let mut guard = writers.lock().await;
        guard.remove(&sender_id);
    }

    tracing::info!(sender_id = %sender_id, "client disconnected");

    let _ = inbound_tx
        .send(InboundEvent::Disconnected { sender_id })
        .await;
}

/// Convert an `OutboundEvent` into a (`recipient_id`, `ServerMessage`) pair.
///
/// Note: `ServerMessage::CommandResponse` has no corresponding `OutboundEvent`
/// variant. Command responses currently route through `OutboundEvent::Message`.
/// If distinct client-side rendering is needed, add a `CommandResponse` variant
/// to `OutboundEvent` in `freebird-traits`.
fn outbound_to_server_message(event: OutboundEvent) -> (String, ServerMessage) {
    match event {
        OutboundEvent::Message { text, recipient_id } => {
            (recipient_id, ServerMessage::Message { text })
        }
        OutboundEvent::StreamChunk { text, recipient_id } => {
            (recipient_id, ServerMessage::StreamChunk { text })
        }
        OutboundEvent::StreamEnd { recipient_id } => (recipient_id, ServerMessage::StreamEnd),
        OutboundEvent::Error { text, recipient_id } => {
            (recipient_id, ServerMessage::Error { text })
        }
        OutboundEvent::ToolStart {
            tool_name,
            recipient_id,
        } => (recipient_id, ServerMessage::ToolStart { tool_name }),
        OutboundEvent::ToolEnd {
            tool_name,
            outcome,
            duration_ms,
            recipient_id,
        } => (
            recipient_id,
            ServerMessage::ToolEnd {
                tool_name,
                outcome,
                duration_ms,
            },
        ),
        OutboundEvent::TurnComplete { recipient_id } => (recipient_id, ServerMessage::TurnComplete),
        OutboundEvent::TokenUsage {
            input_tokens,
            output_tokens,
            cache_read_tokens,
            cache_creation_tokens,
            recipient_id,
        } => (
            recipient_id,
            ServerMessage::TokenUsage {
                input_tokens,
                output_tokens,
                cache_read_tokens,
                cache_creation_tokens,
            },
        ),
        OutboundEvent::SessionInfo {
            session_id,
            model_id,
            provider_id,
            recipient_id,
        } => (
            recipient_id,
            ServerMessage::SessionInfo {
                session_id,
                model_id,
                provider_id,
            },
        ),
        OutboundEvent::ApprovalRequest {
            request_id,
            category_json,
            expires_at,
            recipient_id,
        } => (
            recipient_id,
            ServerMessage::ApprovalRequest {
                request_id,
                category_json,
                expires_at,
            },
        ),
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpStream;
    use tokio_stream::StreamExt;

    /// Connect a test client to the channel, returns (reader, writer) halves.
    async fn connect_test_client(port: u16) -> TcpStream {
        TcpStream::connect(format!("127.0.0.1:{port}"))
            .await
            .expect("test client should connect")
    }

    /// Send a JSON line to the server.
    async fn send_json(stream: &mut TcpStream, msg: &ClientMessage) {
        let json = serde_json::to_string(msg).unwrap();
        let line = format!("{json}\n");
        stream.write_all(line.as_bytes()).await.unwrap();
        stream.flush().await.unwrap();
    }

    /// Read a JSON line from the server.
    async fn recv_json(stream: &mut TcpStream) -> ServerMessage {
        let mut buf = Vec::new();
        loop {
            let mut byte = [0u8; 1];
            stream.read_exact(&mut byte).await.unwrap();
            if byte[0] == b'\n' {
                break;
            }
            buf.push(byte[0]);
        }
        serde_json::from_slice(&buf).expect("should be valid ServerMessage JSON")
    }

    /// Bind to a random port for testing.
    async fn bind_random() -> (TcpListener, u16) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        (listener, port)
    }

    #[tokio::test]
    async fn test_client_message_becomes_inbound_event() {
        let (listener, port) = bind_random().await;
        let channel = TcpChannel::new("127.0.0.1", port);
        let mut handle = channel.start_with_listener(listener);

        let mut client = connect_test_client(port).await;

        // Should get Connected event
        let event = handle.inbound.next().await.unwrap();
        assert!(matches!(event, InboundEvent::Connected { .. }));

        // Send a message
        send_json(
            &mut client,
            &ClientMessage::Message {
                text: "hello".into(),
            },
        )
        .await;

        let event = handle.inbound.next().await.unwrap();
        match event {
            InboundEvent::Message {
                raw_text,
                sender_id,
                ..
            } => {
                assert_eq!(raw_text, "hello");
                assert_eq!(sender_id, "tcp-0");
            }
            other => panic!("expected Message, got {other:?}"),
        }

        channel.stop().await.unwrap();
    }

    #[tokio::test]
    async fn test_client_command_becomes_inbound_command() {
        let (listener, port) = bind_random().await;
        let channel = TcpChannel::new("127.0.0.1", port);
        let mut handle = channel.start_with_listener(listener);

        let mut client = connect_test_client(port).await;

        // Consume Connected
        let _ = handle.inbound.next().await.unwrap();

        send_json(
            &mut client,
            &ClientMessage::Command {
                name: "new".into(),
                args: vec![],
            },
        )
        .await;

        let event = handle.inbound.next().await.unwrap();
        match event {
            InboundEvent::Command {
                name,
                args,
                sender_id,
            } => {
                assert_eq!(name, "new");
                assert!(args.is_empty());
                assert_eq!(sender_id, "tcp-0");
            }
            other => panic!("expected Command, got {other:?}"),
        }

        channel.stop().await.unwrap();
    }

    #[tokio::test]
    async fn test_outbound_message_sent_to_client() {
        let (listener, port) = bind_random().await;
        let channel = TcpChannel::new("127.0.0.1", port);
        let mut handle = channel.start_with_listener(listener);

        let mut client = connect_test_client(port).await;

        // Consume Connected
        let event = handle.inbound.next().await.unwrap();
        let sender_id = match event {
            InboundEvent::Connected { sender_id } => sender_id,
            other => panic!("expected Connected, got {other:?}"),
        };

        // Send outbound message from runtime → client
        handle
            .outbound
            .send(OutboundEvent::Message {
                text: "response".into(),
                recipient_id: sender_id,
            })
            .await
            .unwrap();

        let msg = recv_json(&mut client).await;
        assert_eq!(
            msg,
            ServerMessage::Message {
                text: "response".into()
            }
        );

        channel.stop().await.unwrap();
    }

    #[tokio::test]
    async fn test_streaming_chunks_sent_to_client() {
        let (listener, port) = bind_random().await;
        let channel = TcpChannel::new("127.0.0.1", port);
        let mut handle = channel.start_with_listener(listener);

        let mut client = connect_test_client(port).await;

        // Consume Connected
        let event = handle.inbound.next().await.unwrap();
        let sender_id = match event {
            InboundEvent::Connected { sender_id } => sender_id,
            other => panic!("expected Connected, got {other:?}"),
        };

        // Send stream chunks
        handle
            .outbound
            .send(OutboundEvent::StreamChunk {
                text: "hello ".into(),
                recipient_id: sender_id.clone(),
            })
            .await
            .unwrap();

        handle
            .outbound
            .send(OutboundEvent::StreamChunk {
                text: "world".into(),
                recipient_id: sender_id.clone(),
            })
            .await
            .unwrap();

        handle
            .outbound
            .send(OutboundEvent::StreamEnd {
                recipient_id: sender_id,
            })
            .await
            .unwrap();

        assert_eq!(
            recv_json(&mut client).await,
            ServerMessage::StreamChunk {
                text: "hello ".into()
            }
        );
        assert_eq!(
            recv_json(&mut client).await,
            ServerMessage::StreamChunk {
                text: "world".into()
            }
        );
        assert_eq!(recv_json(&mut client).await, ServerMessage::StreamEnd);

        channel.stop().await.unwrap();
    }

    #[tokio::test]
    async fn test_client_disconnect_sends_disconnected_event() {
        let (listener, port) = bind_random().await;
        let channel = TcpChannel::new("127.0.0.1", port);
        let mut handle = channel.start_with_listener(listener);

        let mut client = connect_test_client(port).await;

        // Consume Connected
        let _ = handle.inbound.next().await.unwrap();

        // Send Disconnect message
        send_json(&mut client, &ClientMessage::Disconnect).await;

        // Should get Disconnected event
        let event = handle.inbound.next().await.unwrap();
        assert!(matches!(event, InboundEvent::Disconnected { .. }));

        channel.stop().await.unwrap();
    }

    #[tokio::test]
    async fn test_tcp_close_sends_disconnected_event() {
        let (listener, port) = bind_random().await;
        let channel = TcpChannel::new("127.0.0.1", port);
        let mut handle = channel.start_with_listener(listener);

        let client = connect_test_client(port).await;

        // Consume Connected
        let _ = handle.inbound.next().await.unwrap();

        // Drop the client (TCP close)
        drop(client);

        // Should get Disconnected event
        let event = handle.inbound.next().await.unwrap();
        assert!(matches!(event, InboundEvent::Disconnected { .. }));

        channel.stop().await.unwrap();
    }

    #[tokio::test]
    async fn test_multiple_clients_get_unique_sender_ids() {
        let (listener, port) = bind_random().await;
        let channel = TcpChannel::new("127.0.0.1", port);
        let mut handle = channel.start_with_listener(listener);

        let _client1 = connect_test_client(port).await;
        let event1 = handle.inbound.next().await.unwrap();
        let id1 = match event1 {
            InboundEvent::Connected { sender_id } => sender_id,
            other => panic!("expected Connected, got {other:?}"),
        };

        let _client2 = connect_test_client(port).await;
        let event2 = handle.inbound.next().await.unwrap();
        let id2 = match event2 {
            InboundEvent::Connected { sender_id } => sender_id,
            other => panic!("expected Connected, got {other:?}"),
        };

        assert_ne!(id1, id2, "each client must get a unique sender_id");
        assert_eq!(id1, "tcp-0");
        assert_eq!(id2, "tcp-1");

        channel.stop().await.unwrap();
    }

    #[tokio::test]
    async fn test_outbound_routed_to_correct_client() {
        let (listener, port) = bind_random().await;
        let channel = TcpChannel::new("127.0.0.1", port);
        let mut handle = channel.start_with_listener(listener);

        let mut client1 = connect_test_client(port).await;
        let event1 = handle.inbound.next().await.unwrap();
        let id1 = match event1 {
            InboundEvent::Connected { sender_id } => sender_id,
            other => panic!("expected Connected, got {other:?}"),
        };

        let mut client2 = connect_test_client(port).await;
        let event2 = handle.inbound.next().await.unwrap();
        let id2 = match event2 {
            InboundEvent::Connected { sender_id } => sender_id,
            other => panic!("expected Connected, got {other:?}"),
        };

        // Send a message targeted at client2 only
        handle
            .outbound
            .send(OutboundEvent::Message {
                text: "for client2".into(),
                recipient_id: id2.clone(),
            })
            .await
            .unwrap();

        let msg = recv_json(&mut client2).await;
        assert_eq!(
            msg,
            ServerMessage::Message {
                text: "for client2".into()
            }
        );

        // Send a message targeted at client1
        handle
            .outbound
            .send(OutboundEvent::Message {
                text: "for client1".into(),
                recipient_id: id1,
            })
            .await
            .unwrap();

        let msg = recv_json(&mut client1).await;
        assert_eq!(
            msg,
            ServerMessage::Message {
                text: "for client1".into()
            }
        );

        channel.stop().await.unwrap();
    }

    #[test]
    fn test_outbound_to_server_message_conversion() {
        let cases = [
            (
                OutboundEvent::Message {
                    text: "hi".into(),
                    recipient_id: "tcp-0".into(),
                },
                (
                    "tcp-0".to_string(),
                    ServerMessage::Message { text: "hi".into() },
                ),
            ),
            (
                OutboundEvent::StreamChunk {
                    text: "ch".into(),
                    recipient_id: "tcp-1".into(),
                },
                (
                    "tcp-1".to_string(),
                    ServerMessage::StreamChunk { text: "ch".into() },
                ),
            ),
            (
                OutboundEvent::StreamEnd {
                    recipient_id: "tcp-2".into(),
                },
                ("tcp-2".to_string(), ServerMessage::StreamEnd),
            ),
            (
                OutboundEvent::Error {
                    text: "err".into(),
                    recipient_id: "tcp-3".into(),
                },
                (
                    "tcp-3".to_string(),
                    ServerMessage::Error { text: "err".into() },
                ),
            ),
            (
                OutboundEvent::ToolStart {
                    tool_name: "read_file".into(),
                    recipient_id: "tcp-4".into(),
                },
                (
                    "tcp-4".to_string(),
                    ServerMessage::ToolStart {
                        tool_name: "read_file".into(),
                    },
                ),
            ),
            (
                OutboundEvent::ToolEnd {
                    tool_name: "read_file".into(),
                    outcome: "success".into(),
                    duration_ms: 42,
                    recipient_id: "tcp-5".into(),
                },
                (
                    "tcp-5".to_string(),
                    ServerMessage::ToolEnd {
                        tool_name: "read_file".into(),
                        outcome: "success".into(),
                        duration_ms: 42,
                    },
                ),
            ),
            (
                OutboundEvent::TurnComplete {
                    recipient_id: "tcp-6".into(),
                },
                ("tcp-6".to_string(), ServerMessage::TurnComplete),
            ),
            (
                OutboundEvent::ApprovalRequest {
                    request_id: "req-42".into(),
                    category_json: r#"{"kind":"consent","tool_name":"shell"}"#.into(),
                    expires_at: "2026-03-09T12:00:00Z".into(),
                    recipient_id: "tcp-7".into(),
                },
                (
                    "tcp-7".to_string(),
                    ServerMessage::ApprovalRequest {
                        request_id: "req-42".into(),
                        category_json: r#"{"kind":"consent","tool_name":"shell"}"#.into(),
                        expires_at: "2026-03-09T12:00:00Z".into(),
                    },
                ),
            ),
        ];

        for (input, (expected_id, expected_msg)) in cases {
            let (id, msg) = outbound_to_server_message(input);
            assert_eq!(id, expected_id);
            assert_eq!(msg, expected_msg);
        }
    }

    #[tokio::test]
    async fn test_invalid_json_from_client_is_ignored() {
        let (listener, port) = bind_random().await;
        let channel = TcpChannel::new("127.0.0.1", port);
        let mut handle = channel.start_with_listener(listener);

        let mut client = connect_test_client(port).await;

        // Consume Connected
        let _ = handle.inbound.next().await.unwrap();

        // Send invalid JSON
        client.write_all(b"not valid json\n").await.unwrap();
        client.flush().await.unwrap();

        // Then send a valid message
        send_json(
            &mut client,
            &ClientMessage::Message {
                text: "valid".into(),
            },
        )
        .await;

        // The valid message should come through (invalid was ignored)
        let event = handle.inbound.next().await.unwrap();
        match event {
            InboundEvent::Message { raw_text, .. } => {
                assert_eq!(raw_text, "valid");
            }
            other => panic!("expected Message, got {other:?}"),
        }

        channel.stop().await.unwrap();
    }

    #[tokio::test]
    async fn test_approval_response_from_client_becomes_inbound_event() {
        let (listener, port) = bind_random().await;
        let channel = TcpChannel::new("127.0.0.1", port);
        let mut handle = channel.start_with_listener(listener);

        let mut client = connect_test_client(port).await;

        // Consume Connected
        let _ = handle.inbound.next().await.unwrap();

        // Send an ApprovalResponse from the client
        send_json(
            &mut client,
            &ClientMessage::ApprovalResponse {
                request_id: "req-42".into(),
                approved: true,
                reason: None,
            },
        )
        .await;

        let event = handle.inbound.next().await.unwrap();
        match event {
            InboundEvent::ApprovalResponse {
                request_id,
                approved,
                reason,
                sender_id,
            } => {
                assert_eq!(request_id, "req-42");
                assert!(approved);
                assert!(reason.is_none());
                assert_eq!(sender_id, "tcp-0");
            }
            other => panic!("expected ApprovalResponse, got {other:?}"),
        }

        channel.stop().await.unwrap();
    }

    #[tokio::test]
    async fn test_approval_response_denied_from_client() {
        let (listener, port) = bind_random().await;
        let channel = TcpChannel::new("127.0.0.1", port);
        let mut handle = channel.start_with_listener(listener);

        let mut client = connect_test_client(port).await;

        // Consume Connected
        let _ = handle.inbound.next().await.unwrap();

        send_json(
            &mut client,
            &ClientMessage::ApprovalResponse {
                request_id: "req-99".into(),
                approved: false,
                reason: Some("too dangerous".into()),
            },
        )
        .await;

        let event = handle.inbound.next().await.unwrap();
        match event {
            InboundEvent::ApprovalResponse {
                request_id,
                approved,
                reason,
                ..
            } => {
                assert_eq!(request_id, "req-99");
                assert!(!approved);
                assert_eq!(reason.unwrap(), "too dangerous");
            }
            other => panic!("expected ApprovalResponse, got {other:?}"),
        }

        channel.stop().await.unwrap();
    }

    #[tokio::test]
    async fn test_oversized_line_disconnects_client() {
        let (listener, port) = bind_random().await;
        let channel = TcpChannel::new("127.0.0.1", port);
        let mut handle = channel.start_with_listener(listener);

        let mut client = connect_test_client(port).await;

        // Consume Connected
        let _ = handle.inbound.next().await.unwrap();

        // Send a line exceeding MAX_LINE_BYTES — the server should
        // disconnect the client (a client sending >64 KiB in a single
        // JSON line is either malicious or broken).
        let oversized = "x".repeat(MAX_LINE_BYTES + 100);
        client
            .write_all(format!("{oversized}\n").as_bytes())
            .await
            .unwrap();
        client.flush().await.unwrap();

        let event = handle.inbound.next().await.unwrap();
        assert!(
            matches!(event, InboundEvent::Disconnected { .. }),
            "expected Disconnected after oversized line, got {event:?}"
        );

        channel.stop().await.unwrap();
    }
}
