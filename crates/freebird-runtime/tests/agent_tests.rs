//! Integration tests for `AgentRuntime`.
//!
//! Uses `MockChannel` and `NoopMemory` to test event routing, command
//! handling, session management, and shutdown behavior.

#![allow(clippy::unwrap_used, clippy::expect_used)]

mod helpers;

use std::sync::atomic::Ordering;
use std::time::Duration;

use freebird_runtime::agent::AgentRuntime;
use freebird_runtime::registry::ProviderRegistry;
use freebird_traits::channel::InboundEvent;
use freebird_traits::id::{ModelId, ProviderId, SessionId};
use freebird_traits::memory::{Conversation, Memory, MemoryError, SessionSummary};
use freebird_types::config::{EditConfig, KnowledgeConfig, RuntimeConfig, ToolsConfig};
use tokio_util::sync::CancellationToken;

use helpers::{MockChannel, error_text, make_tool_executor, message_text};

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

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

// make_tool_executor imported from helpers

fn make_runtime(channel: MockChannel) -> AgentRuntime {
    AgentRuntime::new(
        ProviderRegistry::new(),
        Box::new(channel),
        make_tool_executor(vec![]),
        None,
        Box::new(NoopMemory),
        None,
        KnowledgeConfig::default(),
        RuntimeConfig {
            default_model: ModelId::from("test-model"),
            default_provider: ProviderId::from("test-provider"),
            system_prompt: None,
            max_output_tokens: 1024,
            max_tool_rounds: 10,
            temperature: None,
            max_turns_per_session: 10,
            drain_timeout_secs: 1,
        },
        ToolsConfig {
            sandbox_root: std::env::temp_dir(),
            default_timeout_secs: 30,
            allowed_directories: vec![],
            allowed_shell_commands: vec![],
            max_shell_output_bytes: 1_048_576,
            edit: EditConfig::default(),
        },
        None,
    )
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_command_quit_sends_goodbye_and_exits() {
    let (channel, inbound_tx, mut outbound_rx, _stopped) = MockChannel::new();
    let mut runtime = make_runtime(channel);
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
    let mut runtime = make_runtime(channel);
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
    let mut runtime = make_runtime(channel);
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
    let mut runtime = make_runtime(channel);
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
    let mut runtime = make_runtime(channel);
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
    let mut runtime = make_runtime(channel);
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
    let mut runtime = make_runtime(channel);
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
    let mut runtime = make_runtime(channel);
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
    let mut runtime = make_runtime(channel);
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
