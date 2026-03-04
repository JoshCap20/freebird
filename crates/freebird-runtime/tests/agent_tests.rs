//! Integration tests for `AgentRuntime`.
//!
//! Uses `MockChannel` and `NoopMemory` to test event routing, command
//! handling, session management, and shutdown behavior.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::BTreeSet;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use std::path::PathBuf;

use freebird_runtime::agent::AgentRuntime;
use freebird_runtime::registry::ProviderRegistry;
use freebird_traits::channel::{
    AuthRequirement, Channel, ChannelError, ChannelHandle, ChannelInfo, InboundEvent, OutboundEvent,
};
use freebird_traits::id::{ChannelId, SessionId};
use freebird_traits::memory::{Conversation, Memory, MemoryError, SessionSummary};
use freebird_types::config::{RuntimeConfig, ToolsConfig};
use tokio::sync::Mutex as TokioMutex;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tokio_util::sync::CancellationToken;

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

/// A test-only channel that:
/// - Receives inbound events from test code via `inbound_tx`
/// - Exposes outbound events to test code via `outbound_rx`
/// - Tracks whether `stop()` was called
struct MockChannel {
    info: ChannelInfo,
    handle: TokioMutex<Option<ChannelHandle>>,
    stopped: Arc<AtomicBool>,
}

impl MockChannel {
    /// Create a mock channel and the sender/receiver for test interaction.
    ///
    /// Returns:
    /// - `MockChannel` to pass to `AgentRuntime::new()`
    /// - `mpsc::Sender<InboundEvent>` for test code to inject events
    /// - `mpsc::Receiver<OutboundEvent>` for test code to read responses
    /// - `Arc<AtomicBool>` to check if `stop()` was called
    fn new() -> (
        Self,
        mpsc::Sender<InboundEvent>,
        mpsc::Receiver<OutboundEvent>,
        Arc<AtomicBool>,
    ) {
        let (inbound_tx, inbound_rx) = mpsc::channel::<InboundEvent>(32);
        let (outbound_tx, outbound_rx) = mpsc::channel::<OutboundEvent>(32);
        let stopped = Arc::new(AtomicBool::new(false));

        let channel = Self {
            info: ChannelInfo {
                id: ChannelId::from("mock"),
                display_name: "Mock Channel".into(),
                features: BTreeSet::new(),
                auth: AuthRequirement::None,
            },
            handle: TokioMutex::new(Some(ChannelHandle {
                inbound: Box::pin(ReceiverStream::new(inbound_rx)),
                outbound: outbound_tx,
            })),
            stopped: Arc::clone(&stopped),
        };

        (channel, inbound_tx, outbound_rx, stopped)
    }
}

#[async_trait::async_trait]
impl Channel for MockChannel {
    fn info(&self) -> &ChannelInfo {
        &self.info
    }

    async fn start(&self) -> Result<ChannelHandle, ChannelError> {
        self.handle
            .lock()
            .await
            .take()
            .ok_or_else(|| ChannelError::StartupFailed {
                channel: "mock".into(),
                reason: "already started".into(),
            })
    }

    async fn stop(&self) -> Result<(), ChannelError> {
        self.stopped.store(true, Ordering::SeqCst);
        Ok(())
    }
}

struct NoopMemory;

#[async_trait::async_trait]
impl Memory for NoopMemory {
    async fn load(&self, _: &SessionId) -> Result<Option<Conversation>, MemoryError> {
        Ok(None)
    }
    async fn save(&self, _: &Conversation) -> Result<(), MemoryError> {
        Ok(())
    }
    async fn list_sessions(&self, _: usize) -> Result<Vec<SessionSummary>, MemoryError> {
        Ok(vec![])
    }
    async fn delete(&self, _: &SessionId) -> Result<(), MemoryError> {
        Ok(())
    }
    async fn search(&self, _: &str, _: usize) -> Result<Vec<SessionSummary>, MemoryError> {
        Ok(vec![])
    }
}

fn make_runtime(channel: MockChannel) -> AgentRuntime {
    AgentRuntime::new(
        ProviderRegistry::new(),
        Box::new(channel),
        vec![],
        Box::new(NoopMemory),
        RuntimeConfig {
            default_model: "test-model".into(),
            default_provider: "test-provider".into(),
            system_prompt: None,
            max_output_tokens: 1024,
            max_tool_rounds: 10,
            temperature: None,
            max_turns_per_session: 10,
            drain_timeout_secs: 1,
        },
        ToolsConfig {
            sandbox_root: PathBuf::from("/tmp/test-sandbox"),
            default_timeout_secs: 30,
        },
        None,
    )
}

/// Helper: extract text from an `OutboundEvent::Message`.
fn message_text(event: &OutboundEvent) -> Option<&str> {
    match event {
        OutboundEvent::Message { text, .. } => Some(text.as_str()),
        _ => None,
    }
}

/// Helper: extract text from an `OutboundEvent::Error`.
fn error_text(event: &OutboundEvent) -> Option<&str> {
    match event {
        OutboundEvent::Error { text, .. } => Some(text.as_str()),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_command_quit_sends_goodbye_and_exits() {
    let (channel, inbound_tx, mut outbound_rx, _stopped) = MockChannel::new();
    let runtime = make_runtime(channel);
    let cancel = CancellationToken::new();

    inbound_tx
        .send(InboundEvent::Command {
            name: "quit".into(),
            args: vec![],
            sender_id: "alice".into(),
        })
        .await
        .unwrap();

    let result = tokio::time::timeout(Duration::from_secs(2), runtime.run(cancel)).await;
    let result = result.expect("runtime should exit within timeout");
    assert!(result.is_ok(), "run() should return Ok(())");

    let event = outbound_rx.recv().await.expect("should receive goodbye");
    assert_eq!(message_text(&event), Some("Goodbye!"));
}

#[tokio::test]
async fn test_command_new_creates_session() {
    let (channel, inbound_tx, mut outbound_rx, _stopped) = MockChannel::new();
    let runtime = make_runtime(channel);
    let cancel = CancellationToken::new();

    inbound_tx
        .send(InboundEvent::Command {
            name: "new".into(),
            args: vec![],
            sender_id: "alice".into(),
        })
        .await
        .unwrap();
    inbound_tx
        .send(InboundEvent::Command {
            name: "quit".into(),
            args: vec![],
            sender_id: "alice".into(),
        })
        .await
        .unwrap();

    let result = tokio::time::timeout(Duration::from_secs(2), runtime.run(cancel)).await;
    result.expect("runtime should exit").unwrap();

    let event = outbound_rx
        .recv()
        .await
        .expect("should receive new session msg");
    let text = message_text(&event).expect("should be Message");
    assert!(
        text.contains("New session started:"),
        "expected 'New session started:', got: {text}"
    );
}

#[tokio::test]
async fn test_command_help_sends_help_text() {
    let (channel, inbound_tx, mut outbound_rx, _stopped) = MockChannel::new();
    let runtime = make_runtime(channel);
    let cancel = CancellationToken::new();

    inbound_tx
        .send(InboundEvent::Command {
            name: "help".into(),
            args: vec![],
            sender_id: "alice".into(),
        })
        .await
        .unwrap();
    inbound_tx
        .send(InboundEvent::Command {
            name: "quit".into(),
            args: vec![],
            sender_id: "alice".into(),
        })
        .await
        .unwrap();

    let result = tokio::time::timeout(Duration::from_secs(2), runtime.run(cancel)).await;
    result.expect("runtime should exit").unwrap();

    let event = outbound_rx.recv().await.expect("should receive help text");
    let text = message_text(&event).expect("should be Message");
    assert!(text.contains("quit"), "help should mention /quit");
    assert!(text.contains("new"), "help should mention /new");
    assert!(text.contains("help"), "help should mention /help");
}

#[tokio::test]
async fn test_command_unknown_sends_error() {
    let (channel, inbound_tx, mut outbound_rx, _stopped) = MockChannel::new();
    let runtime = make_runtime(channel);
    let cancel = CancellationToken::new();

    inbound_tx
        .send(InboundEvent::Command {
            name: "foo".into(),
            args: vec![],
            sender_id: "alice".into(),
        })
        .await
        .unwrap();
    inbound_tx
        .send(InboundEvent::Command {
            name: "quit".into(),
            args: vec![],
            sender_id: "alice".into(),
        })
        .await
        .unwrap();

    let result = tokio::time::timeout(Duration::from_secs(2), runtime.run(cancel)).await;
    result.expect("runtime should exit").unwrap();

    let event = outbound_rx.recv().await.expect("should receive error");
    let text = error_text(&event).expect("should be Error variant");
    assert!(
        text.contains("Unknown command: /foo"),
        "expected 'Unknown command: /foo', got: {text}"
    );
}

#[tokio::test]
async fn test_shutdown_on_cancel() {
    let (channel, _inbound_tx, _outbound_rx, _stopped) = MockChannel::new();
    let runtime = make_runtime(channel);
    let cancel = CancellationToken::new();

    let cancel_clone = cancel.clone();
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(50)).await;
        cancel_clone.cancel();
    });

    let result = tokio::time::timeout(Duration::from_secs(2), runtime.run(cancel)).await;
    let result = result.expect("runtime should exit within timeout");
    assert!(result.is_ok(), "run() should return Ok(()) on cancel");
}

#[tokio::test]
async fn test_eof_exits_cleanly() {
    let (channel, inbound_tx, _outbound_rx, _stopped) = MockChannel::new();
    let runtime = make_runtime(channel);
    let cancel = CancellationToken::new();

    // Drop the sender to close the inbound stream (simulates EOF)
    drop(inbound_tx);

    let result = tokio::time::timeout(Duration::from_secs(2), runtime.run(cancel)).await;
    let result = result.expect("runtime should exit within timeout");
    assert!(result.is_ok(), "run() should return Ok(()) on EOF");
}

#[tokio::test]
async fn test_channel_stop_called() {
    let (channel, inbound_tx, _outbound_rx, stopped) = MockChannel::new();
    let runtime = make_runtime(channel);
    let cancel = CancellationToken::new();

    // Drop sender to trigger EOF → clean exit
    drop(inbound_tx);

    let result = tokio::time::timeout(Duration::from_secs(2), runtime.run(cancel)).await;
    result.expect("runtime should exit").unwrap();

    assert!(
        stopped.load(Ordering::SeqCst),
        "channel.stop() should have been called"
    );
}

#[tokio::test]
async fn test_connected_event_no_crash() {
    let (channel, inbound_tx, mut outbound_rx, _stopped) = MockChannel::new();
    let runtime = make_runtime(channel);
    let cancel = CancellationToken::new();

    inbound_tx
        .send(InboundEvent::Connected {
            sender_id: "alice".into(),
        })
        .await
        .unwrap();
    inbound_tx
        .send(InboundEvent::Command {
            name: "quit".into(),
            args: vec![],
            sender_id: "alice".into(),
        })
        .await
        .unwrap();

    let result = tokio::time::timeout(Duration::from_secs(2), runtime.run(cancel)).await;
    result.expect("runtime should exit").unwrap();

    // The Connected event produces no outbound — next event is Goodbye
    let event = outbound_rx.recv().await.expect("should receive goodbye");
    assert_eq!(message_text(&event), Some("Goodbye!"));
}

#[tokio::test]
async fn test_disconnected_event_no_crash() {
    let (channel, inbound_tx, mut outbound_rx, _stopped) = MockChannel::new();
    let runtime = make_runtime(channel);
    let cancel = CancellationToken::new();

    inbound_tx
        .send(InboundEvent::Disconnected {
            sender_id: "alice".into(),
        })
        .await
        .unwrap();
    inbound_tx
        .send(InboundEvent::Command {
            name: "quit".into(),
            args: vec![],
            sender_id: "alice".into(),
        })
        .await
        .unwrap();

    let result = tokio::time::timeout(Duration::from_secs(2), runtime.run(cancel)).await;
    result.expect("runtime should exit").unwrap();

    let event = outbound_rx.recv().await.expect("should receive goodbye");
    assert_eq!(message_text(&event), Some("Goodbye!"));
}
