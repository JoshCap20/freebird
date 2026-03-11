//! Integration tests for the consent bridge — end-to-end consent flow.
//!
//! Tests cover:
//! - Consent request forwarded to channel for high-risk tools
//! - Approved consent executes tool and delivers result
//! - Denied consent returns error to provider
//! - Low-risk tools bypass consent even with gate configured
//! - No consent gate means high-risk tools execute freely
//! - Unknown consent response IDs are handled gracefully

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::too_many_lines,
    clippy::needless_continue,
    clippy::match_same_arms
)]

mod helpers;

use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use async_trait::async_trait;
use chrono::Utc;
use tokio::sync::Mutex as TokioMutex;
use tokio_util::sync::CancellationToken;

use freebird_memory::in_memory::InMemoryMemory;
use freebird_runtime::agent::AgentRuntime;
use freebird_runtime::tool_executor::ToolExecutor;
use freebird_security::consent::ConsentGate;
use freebird_traits::channel::{InboundEvent, OutboundEvent};
use freebird_traits::id::ModelId;
use freebird_traits::provider::{
    CompletionResponse, ContentBlock, Message, Role, StopReason, TokenUsage,
};
use freebird_traits::tool::{
    Capability, RiskLevel, SideEffects, Tool, ToolContext, ToolError, ToolInfo, ToolOutcome,
    ToolOutput,
};
use freebird_types::config::KnowledgeConfig;
use helpers::{
    MockChannel, QueuedProvider, ResponseFactory, default_config, default_tools_config,
    make_registry, message_text, without_status_events,
};

// QueuedProvider, ArcProvider, ResponseFactory imported from helpers

// ---------------------------------------------------------------------------
// MockTool — configurable risk level + queued outputs + shared invocation counter
// ---------------------------------------------------------------------------

struct MockTool {
    info: ToolInfo,
    outputs: TokioMutex<VecDeque<Result<ToolOutput, ToolError>>>,
    invocation_count: Arc<AtomicUsize>,
}

impl MockTool {
    fn new(
        name: &str,
        risk_level: RiskLevel,
        capability: Capability,
        outputs: Vec<Result<ToolOutput, ToolError>>,
        counter: Arc<AtomicUsize>,
    ) -> Self {
        Self {
            info: ToolInfo {
                name: name.into(),
                description: format!("Mock tool: {name}"),
                input_schema: serde_json::json!({"type": "object"}),
                required_capability: capability,
                risk_level,
                side_effects: SideEffects::None,
            },
            outputs: TokioMutex::new(VecDeque::from(outputs)),
            invocation_count: counter,
        }
    }
}

#[async_trait]
impl Tool for MockTool {
    fn info(&self) -> &ToolInfo {
        &self.info
    }

    async fn execute(
        &self,
        _input: serde_json::Value,
        _context: &ToolContext<'_>,
    ) -> Result<ToolOutput, ToolError> {
        self.invocation_count.fetch_add(1, Ordering::SeqCst);
        self.outputs
            .lock()
            .await
            .pop_front()
            .expect("MockTool: no more queued outputs")
    }
}

// ---------------------------------------------------------------------------
// Response builders (consent-specific response factories)
// ---------------------------------------------------------------------------

fn text_response(text: &str) -> ResponseFactory {
    let text = text.to_owned();
    Box::new(move || {
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

fn tool_use_response(tool_name: &str, input: serde_json::Value) -> ResponseFactory {
    let tool_name = tool_name.to_owned();
    Box::new(move || {
        Ok(CompletionResponse {
            message: Message {
                role: Role::Assistant,
                content: vec![ContentBlock::ToolUse {
                    id: "tool-call-1".into(),
                    name: tool_name.clone(),
                    input: input.clone(),
                }],
                timestamp: Utc::now(),
            },
            stop_reason: StopReason::ToolUse,
            usage: TokenUsage::default(),
            model: ModelId::from("test-model"),
        })
    })
}

// default_config, default_tools_config, make_registry imported from helpers

// ===========================================================================
// Tests
// ===========================================================================

/// High-risk tool triggers a `ConsentRequest` on the outbound channel.
#[tokio::test]
async fn test_consent_request_forwarded_to_channel() {
    let (channel, inbound_tx, mut outbound_rx, _) = MockChannel::new();

    let provider = Arc::new(QueuedProvider::new(vec![
        tool_use_response("dangerous_tool", serde_json::json!({"action": "delete"})),
        text_response("Done."),
    ]));

    let counter = Arc::new(AtomicUsize::new(0));
    let tool = MockTool::new(
        "dangerous_tool",
        RiskLevel::High,
        Capability::FileDelete,
        vec![Ok(ToolOutput {
            content: "deleted".into(),
            outcome: ToolOutcome::Success,
            metadata: None,
        })],
        Arc::clone(&counter),
    );

    // Create consent gate with High threshold
    let (gate, consent_rx) = ConsentGate::new(RiskLevel::High, Duration::from_secs(5), 10);

    let executor = ToolExecutor::new(
        vec![Box::new(tool)],
        Duration::from_secs(30),
        None,
        vec![],
        Some(gate),
        None,
        None,
    )
    .expect("executor construction should succeed");

    let mut runtime = AgentRuntime::new(
        make_registry(provider),
        Box::new(channel),
        executor,
        Some(consent_rx),
        Box::new(InMemoryMemory::new()),
        None,
        KnowledgeConfig::default(),
        default_config(),
        default_tools_config(),
        None,
    );

    // Send message that triggers the high-risk tool
    inbound_tx
        .send(InboundEvent::Message {
            raw_text: "Delete the file".into(),
            sender_id: "alice".into(),
            attachments: vec![],
        })
        .await
        .unwrap();

    let cancel = CancellationToken::new();
    let cancel_clone = cancel.clone();

    // Run runtime in background, interact with it via channels
    let runtime_handle = tokio::spawn(async move { runtime.run(cancel_clone).await });

    // Wait for the ConsentRequest to appear on the outbound channel
    let mut consent_request_seen = false;
    let mut request_id = String::new();
    let deadline = tokio::time::Instant::now() + Duration::from_secs(5);

    while tokio::time::Instant::now() < deadline {
        match tokio::time::timeout(Duration::from_millis(100), outbound_rx.recv()).await {
            Ok(Some(OutboundEvent::ConsentRequest {
                request_id: rid,
                tool_name,
                risk_level,
                recipient_id,
                ..
            })) => {
                assert_eq!(tool_name, "dangerous_tool");
                assert_eq!(risk_level, "high");
                assert_eq!(recipient_id, "alice");
                request_id = rid;
                consent_request_seen = true;
                break;
            }
            Ok(Some(_)) => continue, // skip other events (status, etc.)
            Ok(None) => break,       // channel closed
            Err(_) => continue,      // timeout, keep polling
        }
    }

    assert!(
        consent_request_seen,
        "expected ConsentRequest on outbound channel"
    );
    assert!(!request_id.is_empty(), "request_id should be non-empty");

    // Approve the consent and then quit
    inbound_tx
        .send(InboundEvent::ConsentResponse {
            request_id,
            approved: true,
            reason: None,
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

    tokio::time::timeout(Duration::from_secs(5), runtime_handle)
        .await
        .expect("runtime should exit within timeout")
        .expect("runtime task should not panic")
        .unwrap();
}

/// Approved consent allows the tool to execute and deliver a result.
#[tokio::test]
async fn test_consent_approved_executes_tool() {
    let (channel, inbound_tx, mut outbound_rx, _) = MockChannel::new();

    let provider = Arc::new(QueuedProvider::new(vec![
        tool_use_response("risky_tool", serde_json::json!({"cmd": "rm -rf"})),
        text_response("Operation complete."),
    ]));

    let counter = Arc::new(AtomicUsize::new(0));
    let tool = MockTool::new(
        "risky_tool",
        RiskLevel::High,
        Capability::ShellExecute,
        vec![Ok(ToolOutput {
            content: "executed".into(),
            outcome: ToolOutcome::Success,
            metadata: None,
        })],
        Arc::clone(&counter),
    );

    let (gate, consent_rx) = ConsentGate::new(RiskLevel::High, Duration::from_secs(5), 10);

    let executor = ToolExecutor::new(
        vec![Box::new(tool)],
        Duration::from_secs(30),
        None,
        vec![],
        Some(gate),
        None,
        None,
    )
    .expect("executor construction should succeed");

    let mut runtime = AgentRuntime::new(
        make_registry(provider),
        Box::new(channel),
        executor,
        Some(consent_rx),
        Box::new(InMemoryMemory::new()),
        None,
        KnowledgeConfig::default(),
        default_config(),
        default_tools_config(),
        None,
    );

    inbound_tx
        .send(InboundEvent::Message {
            raw_text: "Run the command".into(),
            sender_id: "alice".into(),
            attachments: vec![],
        })
        .await
        .unwrap();

    let cancel = CancellationToken::new();
    let cancel_clone = cancel.clone();
    let runtime_handle = tokio::spawn(async move { runtime.run(cancel_clone).await });

    // Wait for consent request, then approve
    let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
    while tokio::time::Instant::now() < deadline {
        match tokio::time::timeout(Duration::from_millis(100), outbound_rx.recv()).await {
            Ok(Some(OutboundEvent::ConsentRequest { request_id, .. })) => {
                inbound_tx
                    .send(InboundEvent::ConsentResponse {
                        request_id,
                        approved: true,
                        reason: None,
                        sender_id: "alice".into(),
                    })
                    .await
                    .unwrap();
                break;
            }
            Ok(Some(_)) => continue,
            Ok(None) => panic!("channel closed before consent request"),
            Err(_) => continue,
        }
    }

    // Wait for the final response message
    let mut final_response = None;
    let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
    while tokio::time::Instant::now() < deadline {
        match tokio::time::timeout(Duration::from_millis(100), outbound_rx.recv()).await {
            Ok(Some(ref evt)) if message_text(evt).is_some() => {
                final_response = message_text(evt).map(String::from);
                break;
            }
            Ok(Some(_)) => continue,
            Ok(None) => break,
            Err(_) => continue,
        }
    }

    // Quit
    inbound_tx
        .send(InboundEvent::Command {
            name: "quit".into(),
            args: vec![],
            sender_id: "alice".into(),
        })
        .await
        .unwrap();

    tokio::time::timeout(Duration::from_secs(5), runtime_handle)
        .await
        .expect("runtime should exit")
        .expect("runtime should not panic")
        .unwrap();

    assert_eq!(
        counter.load(Ordering::SeqCst),
        1,
        "tool should have been called once"
    );
    assert_eq!(
        final_response.as_deref(),
        Some("Operation complete."),
        "should get the final text response after tool execution"
    );
}

/// Denied consent prevents tool execution and sends error to provider.
#[tokio::test]
async fn test_consent_denied_returns_error_to_provider() {
    let (channel, inbound_tx, mut outbound_rx, _) = MockChannel::new();

    let provider = Arc::new(QueuedProvider::new(vec![
        tool_use_response("risky_tool", serde_json::json!({"cmd": "delete"})),
        text_response("I could not complete that — consent was denied."),
    ]));

    let counter = Arc::new(AtomicUsize::new(0));
    let tool = MockTool::new(
        "risky_tool",
        RiskLevel::High,
        Capability::FileDelete,
        vec![Ok(ToolOutput {
            content: "should not run".into(),
            outcome: ToolOutcome::Success,
            metadata: None,
        })],
        Arc::clone(&counter),
    );

    let (gate, consent_rx) = ConsentGate::new(RiskLevel::High, Duration::from_secs(5), 10);

    let executor = ToolExecutor::new(
        vec![Box::new(tool)],
        Duration::from_secs(30),
        None,
        vec![],
        Some(gate),
        None,
        None,
    )
    .expect("executor construction should succeed");

    let mut runtime = AgentRuntime::new(
        make_registry(provider),
        Box::new(channel),
        executor,
        Some(consent_rx),
        Box::new(InMemoryMemory::new()),
        None,
        KnowledgeConfig::default(),
        default_config(),
        default_tools_config(),
        None,
    );

    inbound_tx
        .send(InboundEvent::Message {
            raw_text: "Delete the file".into(),
            sender_id: "alice".into(),
            attachments: vec![],
        })
        .await
        .unwrap();

    let cancel = CancellationToken::new();
    let cancel_clone = cancel.clone();
    let runtime_handle = tokio::spawn(async move { runtime.run(cancel_clone).await });

    // Wait for consent request, then deny
    let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
    while tokio::time::Instant::now() < deadline {
        match tokio::time::timeout(Duration::from_millis(100), outbound_rx.recv()).await {
            Ok(Some(OutboundEvent::ConsentRequest { request_id, .. })) => {
                inbound_tx
                    .send(InboundEvent::ConsentResponse {
                        request_id,
                        approved: false,
                        reason: Some("Not authorized".into()),
                        sender_id: "alice".into(),
                    })
                    .await
                    .unwrap();
                break;
            }
            Ok(Some(_)) => continue,
            Ok(None) => panic!("channel closed before consent request"),
            Err(_) => continue,
        }
    }

    // Wait for the final response — provider sees the denial and responds
    let mut final_response = None;
    let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
    while tokio::time::Instant::now() < deadline {
        match tokio::time::timeout(Duration::from_millis(100), outbound_rx.recv()).await {
            Ok(Some(ref evt)) if message_text(evt).is_some() => {
                final_response = message_text(evt).map(String::from);
                break;
            }
            Ok(Some(_)) => continue,
            Ok(None) => break,
            Err(_) => continue,
        }
    }

    inbound_tx
        .send(InboundEvent::Command {
            name: "quit".into(),
            args: vec![],
            sender_id: "alice".into(),
        })
        .await
        .unwrap();

    tokio::time::timeout(Duration::from_secs(5), runtime_handle)
        .await
        .expect("runtime should exit")
        .expect("runtime should not panic")
        .unwrap();

    assert_eq!(
        counter.load(Ordering::SeqCst),
        0,
        "tool should NOT have been called"
    );
    assert!(
        final_response.is_some(),
        "should get a response after denial (provider sees error and responds)"
    );
}

/// Low-risk tool executes without consent prompt even with gate configured.
#[tokio::test]
async fn test_consent_low_risk_no_prompt() {
    let (channel, inbound_tx, mut outbound_rx, _) = MockChannel::new();

    let provider = Arc::new(QueuedProvider::new(vec![
        tool_use_response("safe_tool", serde_json::json!({"path": "readme.txt"})),
        text_response("File contents: hello"),
    ]));

    let counter = Arc::new(AtomicUsize::new(0));
    let tool = MockTool::new(
        "safe_tool",
        RiskLevel::Low,
        Capability::FileRead,
        vec![Ok(ToolOutput {
            content: "hello".into(),
            outcome: ToolOutcome::Success,
            metadata: None,
        })],
        counter,
    );

    // Gate requires consent for High+, so Low tools should auto-approve
    let (gate, consent_rx) = ConsentGate::new(RiskLevel::High, Duration::from_secs(5), 10);

    let executor = ToolExecutor::new(
        vec![Box::new(tool)],
        Duration::from_secs(30),
        None,
        vec![],
        Some(gate),
        None,
        None,
    )
    .expect("executor construction should succeed");

    let mut runtime = AgentRuntime::new(
        make_registry(provider),
        Box::new(channel),
        executor,
        Some(consent_rx),
        Box::new(InMemoryMemory::new()),
        None,
        KnowledgeConfig::default(),
        default_config(),
        default_tools_config(),
        None,
    );

    inbound_tx
        .send(InboundEvent::Message {
            raw_text: "Read the file".into(),
            sender_id: "alice".into(),
            attachments: vec![],
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

    let cancel = CancellationToken::new();
    tokio::time::timeout(Duration::from_secs(5), runtime.run(cancel))
        .await
        .expect("runtime should exit within timeout")
        .unwrap();

    // Collect all events — none should be ConsentRequest
    let mut events = Vec::new();
    while let Ok(event) = outbound_rx.try_recv() {
        events.push(event);
    }

    let consent_requests: Vec<_> = events
        .iter()
        .filter(|e| matches!(e, OutboundEvent::ConsentRequest { .. }))
        .collect();

    assert!(
        consent_requests.is_empty(),
        "low-risk tool should not trigger consent: got {} requests",
        consent_requests.len()
    );

    // Should have gotten a message response
    let content_events = without_status_events(events);
    let messages: Vec<_> = content_events.iter().filter_map(message_text).collect();
    assert!(
        messages.iter().any(|m| m.contains("hello")),
        "should get the tool result in final response, got: {messages:?}"
    );
}

/// No consent gate — high-risk tool executes without any prompt.
#[tokio::test]
async fn test_consent_no_gate_executes_freely() {
    let (channel, inbound_tx, mut outbound_rx, _) = MockChannel::new();

    let provider = Arc::new(QueuedProvider::new(vec![
        tool_use_response("dangerous_tool", serde_json::json!({"action": "nuke"})),
        text_response("Done nuking."),
    ]));

    let counter = Arc::new(AtomicUsize::new(0));
    let tool = MockTool::new(
        "dangerous_tool",
        RiskLevel::Critical,
        Capability::ShellExecute,
        vec![Ok(ToolOutput {
            content: "nuked".into(),
            outcome: ToolOutcome::Success,
            metadata: None,
        })],
        counter,
    );

    // No consent gate
    let executor = ToolExecutor::new(
        vec![Box::new(tool)],
        Duration::from_secs(30),
        None,
        vec![],
        None,
        None,
        None,
    )
    .expect("executor construction should succeed");

    let mut runtime = AgentRuntime::new(
        make_registry(provider),
        Box::new(channel),
        executor,
        None, // no consent_rx
        Box::new(InMemoryMemory::new()),
        None,
        KnowledgeConfig::default(),
        default_config(),
        default_tools_config(),
        None,
    );

    inbound_tx
        .send(InboundEvent::Message {
            raw_text: "Nuke it".into(),
            sender_id: "alice".into(),
            attachments: vec![],
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

    let cancel = CancellationToken::new();
    tokio::time::timeout(Duration::from_secs(5), runtime.run(cancel))
        .await
        .expect("runtime should exit within timeout")
        .unwrap();

    let mut events = Vec::new();
    while let Ok(event) = outbound_rx.try_recv() {
        events.push(event);
    }

    // No consent requests should appear
    let consent_count = events
        .iter()
        .filter(|e| matches!(e, OutboundEvent::ConsentRequest { .. }))
        .count();
    assert_eq!(consent_count, 0, "no gate means no consent requests");

    // Should have gotten the final response
    let content_events = without_status_events(events);
    let messages: Vec<_> = content_events
        .iter()
        .filter_map(message_text)
        .map(String::from)
        .collect();
    assert!(
        messages.iter().any(|m| m.contains("nuking")),
        "high-risk tool should execute freely without gate, got: {messages:?}"
    );
}

/// Sending a `ConsentResponse` with an unknown `request_id` doesn't crash.
#[tokio::test]
async fn test_consent_response_unknown_id_logged() {
    let (channel, inbound_tx, outbound_rx, _) = MockChannel::new();

    let provider = Arc::new(QueuedProvider::new(vec![]));

    let (gate, consent_rx) = ConsentGate::new(RiskLevel::High, Duration::from_secs(5), 10);

    let executor = ToolExecutor::new(
        vec![],
        Duration::from_secs(30),
        None,
        vec![],
        Some(gate),
        None,
        None,
    )
    .expect("executor construction should succeed");

    let mut runtime = AgentRuntime::new(
        make_registry(provider),
        Box::new(channel),
        executor,
        Some(consent_rx),
        Box::new(InMemoryMemory::new()),
        None,
        KnowledgeConfig::default(),
        default_config(),
        default_tools_config(),
        None,
    );

    // Send a bogus consent response
    inbound_tx
        .send(InboundEvent::ConsentResponse {
            request_id: "nonexistent-id-12345".into(),
            approved: true,
            reason: None,
            sender_id: "alice".into(),
        })
        .await
        .unwrap();

    // Then quit — runtime should handle the unknown ID gracefully
    inbound_tx
        .send(InboundEvent::Command {
            name: "quit".into(),
            args: vec![],
            sender_id: "alice".into(),
        })
        .await
        .unwrap();

    let cancel = CancellationToken::new();
    let result = tokio::time::timeout(Duration::from_secs(5), runtime.run(cancel))
        .await
        .expect("runtime should exit within timeout");

    // Should exit cleanly — no panic, no error
    assert!(
        result.is_ok(),
        "runtime should handle unknown consent ID gracefully"
    );

    // The splitter should have sent an error back to the user about the
    // unknown consent request ID.
    let mut found_error = false;
    let mut outbound_rx = outbound_rx;
    while let Ok(event) = outbound_rx.try_recv() {
        if let OutboundEvent::Error { ref text, .. } = event {
            if text.contains("nonexistent-id-12345") {
                found_error = true;
            }
        }
    }
    assert!(
        found_error,
        "expected error about unknown consent ID in outbound events"
    );
}
