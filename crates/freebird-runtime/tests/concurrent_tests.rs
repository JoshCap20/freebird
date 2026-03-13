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

use std::collections::{BTreeSet, VecDeque};
use std::pin::Pin;
use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use chrono::Utc;
use futures::Stream;
use tokio::sync::Mutex as TokioMutex;
use tokio::sync::Notify;
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
use freebird_types::config::{BudgetConfig, KnowledgeConfig, RuntimeConfig, SummarizationConfig};

use helpers::{
    MockChannel, default_config, default_tools_config, error_text, make_tool_executor,
    message_text, without_status_events,
};

fn config_with(max_concurrent: usize, drain_secs: u64) -> RuntimeConfig {
    RuntimeConfig {
        drain_timeout_secs: drain_secs,
        max_concurrent_tasks: max_concurrent,
        ..default_config()
    }
}

// ---------------------------------------------------------------------------
// SlowAsyncProvider — queued responses with tokio::time::sleep (async, cancellable)
// ---------------------------------------------------------------------------

struct SlowAsyncResponse {
    text: String,
    delay: Duration,
}

struct SlowAsyncProvider {
    info: ProviderInfo,
    responses: TokioMutex<VecDeque<SlowAsyncResponse>>,
    /// Optional: signalled when `complete()` starts (before the delay).
    /// Tests can wait on this to know the provider is actively processing.
    on_start: Option<Arc<Notify>>,
}

impl SlowAsyncProvider {
    fn new(provider_id: &str, responses: Vec<(&str, Duration)>) -> Self {
        Self {
            info: ProviderInfo {
                id: ProviderId::from(provider_id),
                display_name: "Slow Async".into(),
                supported_models: vec![],
                features: BTreeSet::from([ProviderFeature::ToolUse]),
            },
            responses: TokioMutex::new(
                responses
                    .into_iter()
                    .map(|(text, delay)| SlowAsyncResponse {
                        text: text.to_owned(),
                        delay,
                    })
                    .collect(),
            ),
            on_start: None,
        }
    }

    fn with_start_notify(mut self, notify: Arc<Notify>) -> Self {
        self.on_start = Some(notify);
        self
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
        let resp = self
            .responses
            .lock()
            .await
            .pop_front()
            .expect("SlowAsyncProvider: no more queued responses");

        // Signal that processing has started (before sleeping).
        if let Some(notify) = &self.on_start {
            notify.notify_one();
        }

        tokio::time::sleep(resp.delay).await;
        Ok(CompletionResponse {
            message: Message {
                role: Role::Assistant,
                content: vec![ContentBlock::Text { text: resp.text }],
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

/// Register a `SlowAsyncProvider` into a `ProviderRegistry` and configure failover.
fn make_slow_registry(provider_id: &str, provider: SlowAsyncProvider) -> ProviderRegistry {
    let mut registry = ProviderRegistry::new();
    let id = ProviderId::from(provider_id);
    registry.register(id.clone(), Box::new(provider));
    registry.set_failover_chain(vec![id]);
    registry
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
    let provider = SlowAsyncProvider::new(
        "test-provider",
        vec![("Reply A", delay), ("Reply B", delay)],
    );

    let (channel, inbound_tx, mut outbound_rx, _) = MockChannel::new();
    let runtime = AgentRuntime::new(
        make_slow_registry("test-provider", provider),
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
        None,
        SummarizationConfig::default(),
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
    // not 2x. A serialized execution would take >= 400ms.
    assert!(
        elapsed < delay * 2,
        "concurrent messages took {elapsed:?}, expected < {:?} (2x one delay). \
         Sequential processing would take >= {:?}",
        delay * 2,
        delay * 2,
    );
}

/// Two messages from the same sender should be serialized by the per-session
/// lock. Total time should be >= 2 * delay.
#[tokio::test]
async fn test_serialized_same_session() {
    let delay = Duration::from_millis(100);
    let provider = SlowAsyncProvider::new(
        "test-provider",
        vec![("Reply 1", delay), ("Reply 2", delay)],
    );

    let (channel, inbound_tx, mut outbound_rx, _) = MockChannel::new();
    let runtime = AgentRuntime::new(
        make_slow_registry("test-provider", provider),
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
        None,
        SummarizationConfig::default(),
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
///
/// Uses `Notify` to deterministically wait until the provider is actively
/// processing the first request before sending the second, avoiding races.
#[tokio::test]
async fn test_concurrency_limit_rejects_excess() {
    let delay = Duration::from_millis(300);
    let started = Arc::new(Notify::new());
    // max_concurrent_tasks = 1, so only one message at a time
    let provider = SlowAsyncProvider::new("test-provider", vec![("Slow reply", delay)])
        .with_start_notify(Arc::clone(&started));

    let (channel, inbound_tx, mut outbound_rx, _) = MockChannel::new();
    let runtime = AgentRuntime::new(
        make_slow_registry("test-provider", provider),
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
        None,
        SummarizationConfig::default(),
    );

    let runtime = Arc::new(runtime);
    let cancel = CancellationToken::new();

    // Run the runtime in a background task so we can coordinate message sends.
    let runtime_handle = {
        let runtime = Arc::clone(&runtime);
        let cancel = cancel.clone();
        tokio::spawn(async move { runtime.run(cancel).await })
    };

    // Send first message — it will grab the single semaphore permit.
    inbound_tx
        .send(InboundEvent::Message {
            raw_text: "Hello".into(),
            sender_id: "alice".into(),
            attachments: vec![],
        })
        .await
        .unwrap();

    // Wait until the provider is actively processing (permit held, sleeping).
    // This is deterministic — no timing races.
    tokio::time::timeout(Duration::from_secs(2), started.notified())
        .await
        .expect("provider should start processing within 2s");

    // Now send the second message — the permit is held, so this gets rejected.
    inbound_tx
        .send(InboundEvent::Message {
            raw_text: "Hello too".into(),
            sender_id: "bob".into(),
            attachments: vec![],
        })
        .await
        .unwrap();
    drop(inbound_tx);

    tokio::time::timeout(Duration::from_secs(5), runtime_handle)
        .await
        .expect("runtime should exit within timeout")
        .unwrap()
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
    let started = Arc::new(Notify::new());
    let provider = SlowAsyncProvider::new("test-provider", vec![("Finished", delay)])
        .with_start_notify(Arc::clone(&started));

    let (channel, inbound_tx, mut outbound_rx, _) = MockChannel::new();
    let runtime = AgentRuntime::new(
        make_slow_registry("test-provider", provider),
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
        None,
        SummarizationConfig::default(),
    );

    let runtime = Arc::new(runtime);
    let cancel = CancellationToken::new();

    let runtime_handle = {
        let runtime = Arc::clone(&runtime);
        let cancel = cancel.clone();
        tokio::spawn(async move { runtime.run(cancel).await })
    };

    // Send a message
    inbound_tx
        .send(InboundEvent::Message {
            raw_text: "Hello".into(),
            sender_id: "alice".into(),
            attachments: vec![],
        })
        .await
        .unwrap();

    // Wait until the provider is actively processing (deterministic).
    tokio::time::timeout(Duration::from_secs(2), started.notified())
        .await
        .expect("provider should start processing within 2s");

    // Now send quit — the in-flight task should complete during drain.
    inbound_tx
        .send(InboundEvent::Command {
            name: "quit".into(),
            args: vec![],
            sender_id: "alice".into(),
        })
        .await
        .unwrap();
    drop(inbound_tx);

    tokio::time::timeout(Duration::from_secs(5), runtime_handle)
        .await
        .expect("runtime should exit within timeout")
        .unwrap()
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
    let provider = SlowAsyncProvider::new(
        "test-provider",
        vec![("Never arrives", Duration::from_secs(30))],
    );

    let (channel, inbound_tx, mut outbound_rx, _) = MockChannel::new();
    let runtime = AgentRuntime::new(
        make_slow_registry("test-provider", provider),
        Box::new(channel),
        make_tool_executor(vec![]),
        None,
        Arc::new(InMemoryMemory::new()),
        None,
        KnowledgeConfig::default(),
        config_with(8, 1), // 1 second drain timeout
        default_tools_config(),
        BudgetConfig::default(),
        24,
        Some(Arc::new(helpers::MockEventSink::new())),
        Some(Arc::new(helpers::MockAuditSink::new())),
        None,
        SummarizationConfig::default(),
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

/// After a task completes and releases its semaphore permit, the next
/// message should be processed normally — verifies permits are returned.
#[tokio::test]
async fn test_concurrency_limit_recovers_after_completion() {
    let delay = Duration::from_millis(50);
    let started = Arc::new(Notify::new());
    // max_concurrent_tasks = 1, but the first task will finish before the second arrives.
    let provider = SlowAsyncProvider::new(
        "test-provider",
        vec![("Reply 1", delay), ("Reply 2", delay)],
    )
    .with_start_notify(Arc::clone(&started));

    let (channel, inbound_tx, mut outbound_rx, _) = MockChannel::new();
    let runtime = AgentRuntime::new(
        make_slow_registry("test-provider", provider),
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
        None,
        SummarizationConfig::default(),
    );

    let runtime = Arc::new(runtime);
    let cancel = CancellationToken::new();

    let runtime_handle = {
        let runtime = Arc::clone(&runtime);
        let cancel = cancel.clone();
        tokio::spawn(async move { runtime.run(cancel).await })
    };

    // Send first message — occupies the single permit.
    inbound_tx
        .send(InboundEvent::Message {
            raw_text: "Hello".into(),
            sender_id: "alice".into(),
            attachments: vec![],
        })
        .await
        .unwrap();

    // Wait for the provider to start, then wait for the task to complete.
    tokio::time::timeout(Duration::from_secs(2), started.notified())
        .await
        .expect("provider should start processing");
    // Wait for the first task to finish (delay + margin).
    tokio::time::sleep(delay + Duration::from_millis(200)).await;

    // Now send a second message from a different sender — the permit should be free.
    inbound_tx
        .send(InboundEvent::Message {
            raw_text: "Hello again".into(),
            sender_id: "bob".into(),
            attachments: vec![],
        })
        .await
        .unwrap();
    drop(inbound_tx);

    tokio::time::timeout(Duration::from_secs(5), runtime_handle)
        .await
        .expect("runtime should exit within timeout")
        .unwrap()
        .unwrap();

    let mut events = Vec::new();
    while let Ok(event) = outbound_rx.try_recv() {
        events.push(event);
    }
    let events = without_status_events(events);
    let messages: Vec<_> = events.iter().filter_map(|e| message_text(e)).collect();

    // Both messages should have been processed (no "busy" error).
    assert!(
        messages.contains(&"Reply 1"),
        "first message should succeed, got: {messages:?}"
    );
    assert!(
        messages.contains(&"Reply 2"),
        "second message should succeed after permit recovery, got: {messages:?}"
    );

    // No "busy" errors — permits were properly returned.
    let has_busy_error = events
        .iter()
        .any(|e| error_text(e).is_some_and(|t| t.contains("Server is busy")));
    assert!(
        !has_busy_error,
        "no messages should be rejected, got: {events:?}"
    );
}
