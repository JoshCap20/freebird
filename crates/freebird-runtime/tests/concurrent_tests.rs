//! Integration tests for concurrent message handling in `AgentRuntime`.
//!
//! Tests cover:
//! - Per-session serialization via per-session locks
//! - Cross-session parallelism
//! - Concurrency limit enforcement (semaphore)
//! - Drain phase with timeout
//! - Quit command during in-flight tasks

#![allow(clippy::unwrap_used, clippy::expect_used)]

mod helpers;

use std::collections::BTreeSet;
use std::pin::Pin;
use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use chrono::Utc;
use futures::Stream;
use tokio_util::sync::CancellationToken;

use freebird_memory::in_memory::InMemoryMemory;
use freebird_runtime::agent::AgentRuntime;
use freebird_runtime::registry::ProviderRegistry;
use freebird_traits::channel::InboundEvent;
use freebird_traits::id::{ModelId, ProviderId};
use freebird_traits::provider::{
    CompletionRequest, CompletionResponse, ContentBlock, Message, Provider, ProviderError,
    ProviderFeature, ProviderInfo, Role, StopReason, StreamEvent, TokenUsage,
};
use freebird_types::config::{BudgetConfig, ContextConfig, KnowledgeConfig, RuntimeConfig};

use helpers::{
    MockChannel, QueuedProvider, ResponseFactory, default_tools_config, error_text, make_registry,
    make_tool_executor, message_text, without_status_events,
};

// ---------------------------------------------------------------------------
// Slow provider — introduces a configurable delay per response
// ---------------------------------------------------------------------------

fn slow_text_response(text: &str, delay: Duration) -> ResponseFactory {
    let text = text.to_owned();
    Box::new(move || {
        std::thread::sleep(delay);
        Ok(CompletionResponse {
            message: Message {
                role: Role::Assistant,
                content: vec![ContentBlock::Text { text: text.clone() }],
                timestamp: Utc::now(),
            },
            stop_reason: StopReason::EndTurn,
            usage: TokenUsage::default(),
            model: ModelId::from("test-model"),
        })
    })
}

fn config_with(max_concurrent: usize, drain_secs: u64) -> RuntimeConfig {
    RuntimeConfig {
        default_model: ModelId::from("test-model"),
        default_provider: ProviderId::from("test-provider"),
        system_prompt: None,
        max_output_tokens: 1024,
        max_tool_rounds: 10,
        temperature: None,
        max_turns_per_session: 10,
        drain_timeout_secs: drain_secs,
        max_concurrent_tasks: max_concurrent,
        session: freebird_types::config::SessionConfig::default(),
        context: ContextConfig::default(),
    }
}

// ---------------------------------------------------------------------------
// SlowAsyncProvider — uses tokio::time::sleep (cancellable by JoinSet::shutdown)
// ---------------------------------------------------------------------------

struct SlowAsyncProvider {
    info: ProviderInfo,
    delay: Duration,
    text: String,
}

impl SlowAsyncProvider {
    fn new(text: &str, delay: Duration) -> Self {
        Self {
            info: ProviderInfo {
                id: ProviderId::from("slow-async-provider"),
                display_name: "Slow Async".into(),
                supported_models: vec![],
                features: BTreeSet::from([ProviderFeature::ToolUse]),
            },
            delay,
            text: text.to_owned(),
        }
    }
}

#[async_trait]
impl Provider for SlowAsyncProvider {
    fn info(&self) -> &ProviderInfo {
        &self.info
    }

    async fn validate_credentials(&self) -> Result<(), ProviderError> {
        Ok(())
    }

    async fn complete(
        &self,
        _request: CompletionRequest,
    ) -> Result<CompletionResponse, ProviderError> {
        tokio::time::sleep(self.delay).await;
        Ok(CompletionResponse {
            message: Message {
                role: Role::Assistant,
                content: vec![ContentBlock::Text {
                    text: self.text.clone(),
                }],
                timestamp: Utc::now(),
            },
            stop_reason: StopReason::EndTurn,
            usage: TokenUsage::default(),
            model: ModelId::from("test-model"),
        })
    }

    async fn stream(
        &self,
        _request: CompletionRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamEvent, ProviderError>> + Send>>, ProviderError>
    {
        Err(ProviderError::NotConfigured)
    }
}

// ===========================================================================
// Tests
// ===========================================================================

/// Two messages from different senders should be processed concurrently.
/// If they were serialized, total time would be >= 2 * delay. If concurrent,
/// total time should be close to 1 * delay.
#[tokio::test]
async fn test_concurrent_different_sessions() {
    let delay = Duration::from_millis(200);
    let provider = Arc::new(QueuedProvider::new(vec![
        slow_text_response("Reply A", delay),
        slow_text_response("Reply B", delay),
    ]));

    let (channel, inbound_tx, mut outbound_rx, _) = MockChannel::new();
    let runtime = AgentRuntime::new(
        make_registry(provider),
        Box::new(channel),
        make_tool_executor(vec![]),
        None,
        Arc::new(InMemoryMemory::new()),
        None,
        KnowledgeConfig::default(),
        config_with(8, 5),
        default_tools_config(),
        BudgetConfig::default(),
        24,
        Some(Arc::new(helpers::MockEventSink::new())),
        Some(Arc::new(helpers::MockAuditSink::new())),
    );

    // Send messages from two different senders (different sessions)
    inbound_tx
        .send(InboundEvent::Message {
            raw_text: "Hello from Alice".into(),
            sender_id: "alice".into(),
            attachments: vec![],
        })
        .await
        .unwrap();
    inbound_tx
        .send(InboundEvent::Message {
            raw_text: "Hello from Bob".into(),
            sender_id: "bob".into(),
            attachments: vec![],
        })
        .await
        .unwrap();
    drop(inbound_tx);

    let start = Instant::now();
    let cancel = CancellationToken::new();
    tokio::time::timeout(Duration::from_secs(5), Arc::new(runtime).run(cancel))
        .await
        .expect("runtime should exit within timeout")
        .unwrap();
    let elapsed = start.elapsed();

    // Collect responses
    let mut events = Vec::new();
    while let Ok(event) = outbound_rx.try_recv() {
        events.push(event);
    }
    let events = without_status_events(events);
    let messages: Vec<_> = events.iter().filter_map(|e| message_text(e)).collect();

    assert!(
        messages.contains(&"Reply A"),
        "should contain Reply A, got: {messages:?}"
    );
    assert!(
        messages.contains(&"Reply B"),
        "should contain Reply B, got: {messages:?}"
    );

    // Concurrent execution: both should complete in roughly 1 delay period,
    // not 2x. Allow generous margin for CI variability.
    assert!(
        elapsed < delay * 3,
        "concurrent messages took {elapsed:?}, expected < {:?} (3x one delay). \
         Sequential processing would take >= {:?}",
        delay * 3,
        delay * 2,
    );
}

/// Two messages from the same sender should be serialized by the per-session
/// lock. Total time should be >= 2 * delay.
#[tokio::test]
async fn test_serialized_same_session() {
    let delay = Duration::from_millis(100);
    let provider = Arc::new(QueuedProvider::new(vec![
        slow_text_response("Reply 1", delay),
        slow_text_response("Reply 2", delay),
    ]));

    let (channel, inbound_tx, mut outbound_rx, _) = MockChannel::new();
    let runtime = AgentRuntime::new(
        make_registry(provider),
        Box::new(channel),
        make_tool_executor(vec![]),
        None,
        Arc::new(InMemoryMemory::new()),
        None,
        KnowledgeConfig::default(),
        config_with(8, 5),
        default_tools_config(),
        BudgetConfig::default(),
        24,
        Some(Arc::new(helpers::MockEventSink::new())),
        Some(Arc::new(helpers::MockAuditSink::new())),
    );

    // Send two messages from the same sender (same session → serialized)
    inbound_tx
        .send(InboundEvent::Message {
            raw_text: "First".into(),
            sender_id: "alice".into(),
            attachments: vec![],
        })
        .await
        .unwrap();
    inbound_tx
        .send(InboundEvent::Message {
            raw_text: "Second".into(),
            sender_id: "alice".into(),
            attachments: vec![],
        })
        .await
        .unwrap();
    drop(inbound_tx);

    let start = Instant::now();
    let cancel = CancellationToken::new();
    tokio::time::timeout(Duration::from_secs(5), Arc::new(runtime).run(cancel))
        .await
        .expect("runtime should exit within timeout")
        .unwrap();
    let elapsed = start.elapsed();

    let mut events = Vec::new();
    while let Ok(event) = outbound_rx.try_recv() {
        events.push(event);
    }
    let events = without_status_events(events);
    let messages: Vec<_> = events.iter().filter_map(|e| message_text(e)).collect();

    assert_eq!(
        messages.len(),
        2,
        "should have 2 responses, got: {messages:?}"
    );

    // Serialized: should take >= 2 * delay
    assert!(
        elapsed >= delay * 2,
        "same-session messages took {elapsed:?}, expected >= {:?} (serialized). \
         Indicates per-session lock is not working",
        delay * 2,
    );
}

/// When the semaphore is exhausted, excess messages get "Server is busy" error.
#[tokio::test]
async fn test_concurrency_limit_rejects_excess() {
    let delay = Duration::from_millis(300);
    // max_concurrent_tasks = 1, so only one message at a time
    let provider = Arc::new(QueuedProvider::new(vec![slow_text_response(
        "Slow reply",
        delay,
    )]));

    let (channel, inbound_tx, mut outbound_rx, _) = MockChannel::new();
    let runtime = AgentRuntime::new(
        make_registry(provider),
        Box::new(channel),
        make_tool_executor(vec![]),
        None,
        Arc::new(InMemoryMemory::new()),
        None,
        KnowledgeConfig::default(),
        config_with(1, 5),
        default_tools_config(),
        BudgetConfig::default(),
        24,
        Some(Arc::new(helpers::MockEventSink::new())),
        Some(Arc::new(helpers::MockAuditSink::new())),
    );

    // Send two messages from different senders at once
    // First grabs the single permit, second should be rejected
    inbound_tx
        .send(InboundEvent::Message {
            raw_text: "Hello".into(),
            sender_id: "alice".into(),
            attachments: vec![],
        })
        .await
        .unwrap();

    // Small delay so the first message's task starts and holds the permit
    tokio::time::sleep(Duration::from_millis(50)).await;

    inbound_tx
        .send(InboundEvent::Message {
            raw_text: "Hello too".into(),
            sender_id: "bob".into(),
            attachments: vec![],
        })
        .await
        .unwrap();
    drop(inbound_tx);

    let cancel = CancellationToken::new();
    tokio::time::timeout(Duration::from_secs(5), Arc::new(runtime).run(cancel))
        .await
        .expect("runtime should exit within timeout")
        .unwrap();

    let mut events = Vec::new();
    while let Ok(event) = outbound_rx.try_recv() {
        events.push(event);
    }
    let events = without_status_events(events);

    // One message should succeed, the other should get a "busy" error
    let has_response = events
        .iter()
        .any(|e| message_text(e).is_some_and(|t| t == "Slow reply"));
    let has_busy_error = events
        .iter()
        .any(|e| error_text(e).is_some_and(|t| t.contains("Server is busy")));

    assert!(
        has_response,
        "first message should get a response, got: {events:?}"
    );
    assert!(
        has_busy_error,
        "second message should get busy error, got: {events:?}"
    );
}

/// Quit command during in-flight message tasks should drain tasks before exiting.
#[tokio::test]
async fn test_command_quit_during_inflight() {
    let delay = Duration::from_millis(200);
    let provider = Arc::new(QueuedProvider::new(vec![slow_text_response(
        "Finished", delay,
    )]));

    let (channel, inbound_tx, mut outbound_rx, _) = MockChannel::new();
    let runtime = AgentRuntime::new(
        make_registry(provider),
        Box::new(channel),
        make_tool_executor(vec![]),
        None,
        Arc::new(InMemoryMemory::new()),
        None,
        KnowledgeConfig::default(),
        config_with(8, 5), // 5 second drain — enough time
        default_tools_config(),
        BudgetConfig::default(),
        24,
        Some(Arc::new(helpers::MockEventSink::new())),
        Some(Arc::new(helpers::MockAuditSink::new())),
    );

    // Send a message, then immediately send quit
    inbound_tx
        .send(InboundEvent::Message {
            raw_text: "Hello".into(),
            sender_id: "alice".into(),
            attachments: vec![],
        })
        .await
        .unwrap();

    // Small delay to ensure the message is spawned into JoinSet
    tokio::time::sleep(Duration::from_millis(10)).await;

    inbound_tx
        .send(InboundEvent::Command {
            name: "quit".into(),
            args: vec![],
            sender_id: "alice".into(),
        })
        .await
        .unwrap();
    drop(inbound_tx);

    let cancel = CancellationToken::new();
    tokio::time::timeout(Duration::from_secs(5), Arc::new(runtime).run(cancel))
        .await
        .expect("runtime should exit within timeout")
        .unwrap();

    let mut events = Vec::new();
    while let Ok(event) = outbound_rx.try_recv() {
        events.push(event);
    }
    let events = without_status_events(events);

    // The drain phase should have waited for the in-flight task
    let has_response = events
        .iter()
        .any(|e| message_text(e).is_some_and(|t| t == "Finished"));
    let has_goodbye = events
        .iter()
        .any(|e| message_text(e).is_some_and(|t| t == "Goodbye!"));

    assert!(
        has_response,
        "in-flight task should complete during drain, got: {events:?}"
    );
    assert!(has_goodbye, "quit should send Goodbye!, got: {events:?}");
}

/// Shutdown with drain timeout should abort tasks that exceed the deadline.
#[tokio::test]
async fn test_shutdown_drains_inflight_with_timeout() {
    // Use an async provider so tokio can cancel the sleep during shutdown
    let provider = SlowAsyncProvider::new("Never arrives", Duration::from_secs(30));
    let mut registry = ProviderRegistry::new();
    let id = ProviderId::from("slow-async-provider");
    registry.register(id.clone(), Box::new(provider));
    registry.set_failover_chain(vec![id]);

    let (channel, inbound_tx, mut outbound_rx, _) = MockChannel::new();
    let runtime = AgentRuntime::new(
        registry,
        Box::new(channel),
        make_tool_executor(vec![]),
        None,
        Arc::new(InMemoryMemory::new()),
        None,
        KnowledgeConfig::default(),
        RuntimeConfig {
            default_provider: "slow-async-provider".into(),
            ..config_with(8, 1) // 1 second drain timeout
        },
        default_tools_config(),
        BudgetConfig::default(),
        24,
        Some(Arc::new(helpers::MockEventSink::new())),
        Some(Arc::new(helpers::MockAuditSink::new())),
    );

    inbound_tx
        .send(InboundEvent::Message {
            raw_text: "Hello".into(),
            sender_id: "alice".into(),
            attachments: vec![],
        })
        .await
        .unwrap();
    drop(inbound_tx);

    let start = Instant::now();
    let cancel = CancellationToken::new();
    tokio::time::timeout(Duration::from_secs(10), Arc::new(runtime).run(cancel))
        .await
        .expect("runtime should exit within timeout")
        .unwrap();
    let elapsed = start.elapsed();

    // Should exit after ~1 second (drain timeout), not 30 seconds (task delay)
    assert!(
        elapsed < Duration::from_secs(5),
        "drain timeout should abort long-running task, but took {elapsed:?}"
    );

    // The task was aborted, so no response
    let mut events = Vec::new();
    while let Ok(event) = outbound_rx.try_recv() {
        events.push(event);
    }
    let events = without_status_events(events);
    let has_response = events
        .iter()
        .any(|e| message_text(e).is_some_and(|t| t == "Never arrives"));
    assert!(
        !has_response,
        "aborted task should not produce a response, got: {events:?}"
    );
}
