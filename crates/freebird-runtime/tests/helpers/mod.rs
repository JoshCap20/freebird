//! Shared test infrastructure for `freebird-runtime` integration tests.

// Each test binary compiles this module independently; not every binary uses
// every helper, so unused-function warnings are expected and harmless.
#![allow(clippy::unwrap_used, clippy::expect_used, dead_code)]

use std::collections::{BTreeSet, VecDeque};
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Duration;

use async_trait::async_trait;
use futures::Stream;
use tokio::sync::Mutex as TokioMutex;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

use chrono::Utc;
use freebird_runtime::agent::AgentRuntime;
use freebird_runtime::registry::ProviderRegistry;
use freebird_runtime::tool_executor::{ToolExecutor, ToolExecutorBuilder};
use freebird_security::capability::RevocationList;
use freebird_traits::audit::AuditSink;
use freebird_traits::channel::{
    AuthRequirement, Channel, ChannelError, ChannelHandle, ChannelInfo, InboundEvent, OutboundEvent,
};
use freebird_traits::event::{ConversationEvent, EventSink};
use freebird_traits::id::{ChannelId, KnowledgeId, ModelId, ProviderId, SessionId};
use freebird_traits::knowledge::{
    KnowledgeEntry, KnowledgeError, KnowledgeKind, KnowledgeMatch, KnowledgeStore,
};
use freebird_traits::memory::{Conversation, Memory, MemoryError, SessionSummary};
use freebird_traits::provider::{
    CompletionRequest, CompletionResponse, ContentBlock, Message, Provider, ProviderError,
    ProviderFeature, ProviderInfo, Role, StopReason, StreamEvent, TokenUsage,
};
use freebird_traits::tool::Tool;
use freebird_types::config::{ContextConfig, EditConfig, RuntimeConfig, ToolsConfig};
use tokio_util::sync::CancellationToken;

// ---------------------------------------------------------------------------
// MockChannel
// ---------------------------------------------------------------------------

/// A test-only channel that:
/// - Receives inbound events from test code via `inbound_tx`
/// - Exposes outbound events to test code via `outbound_rx`
/// - Tracks whether `stop()` was called
pub struct MockChannel {
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
    pub fn new() -> (
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

#[async_trait]
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

// ---------------------------------------------------------------------------
// QueuedProvider — returns queued responses in order
// ---------------------------------------------------------------------------

pub type ResponseFactory = Box<dyn Fn() -> Result<CompletionResponse, ProviderError> + Send + Sync>;

pub struct QueuedProvider {
    pub info: ProviderInfo,
    pub responses: TokioMutex<VecDeque<ResponseFactory>>,
    pub call_count: AtomicUsize,
}

impl QueuedProvider {
    pub fn new(responses: Vec<ResponseFactory>) -> Self {
        Self {
            info: ProviderInfo {
                id: ProviderId::from("test-provider"),
                display_name: "Test Provider".into(),
                supported_models: vec![],
                features: BTreeSet::from([ProviderFeature::ToolUse]),
            },
            responses: TokioMutex::new(VecDeque::from(responses)),
            call_count: AtomicUsize::new(0),
        }
    }

    pub fn call_count(&self) -> usize {
        self.call_count.load(Ordering::SeqCst)
    }
}

#[async_trait]
impl Provider for QueuedProvider {
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
        self.call_count.fetch_add(1, Ordering::SeqCst);
        let factory = self
            .responses
            .lock()
            .await
            .pop_front()
            .expect("QueuedProvider: no more queued responses");
        factory()
    }

    async fn stream(
        &self,
        _request: CompletionRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamEvent, ProviderError>> + Send>>, ProviderError>
    {
        Err(ProviderError::NotConfigured)
    }
}

/// Wrapper so `Arc<QueuedProvider>` implements `Provider`.
pub struct ArcProvider(pub Arc<QueuedProvider>);

#[async_trait]
impl Provider for ArcProvider {
    fn info(&self) -> &ProviderInfo {
        self.0.info()
    }

    async fn validate_credentials(&self) -> Result<(), ProviderError> {
        self.0.validate_credentials().await
    }

    async fn complete(
        &self,
        request: CompletionRequest,
    ) -> Result<CompletionResponse, ProviderError> {
        self.0.complete(request).await
    }

    async fn stream(
        &self,
        request: CompletionRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamEvent, ProviderError>> + Send>>, ProviderError>
    {
        self.0.stream(request).await
    }
}

// ---------------------------------------------------------------------------
// MockEventSink — records events for test inspection
// ---------------------------------------------------------------------------

/// Records every appended event so tests can verify event emission.
pub struct MockEventSink {
    events: TokioMutex<Vec<(SessionId, ConversationEvent)>>,
}

impl MockEventSink {
    pub fn new() -> Self {
        Self {
            events: TokioMutex::new(Vec::new()),
        }
    }

    pub async fn events(&self) -> Vec<(SessionId, ConversationEvent)> {
        self.events.lock().await.clone()
    }
}

#[async_trait]
impl EventSink for MockEventSink {
    async fn append(
        &self,
        session_id: &SessionId,
        event: ConversationEvent,
    ) -> Result<(), MemoryError> {
        self.events.lock().await.push((session_id.clone(), event));
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// MockAuditSink — records audit events for test inspection
// ---------------------------------------------------------------------------

/// Records every audit event so tests can verify audit logging.
pub struct MockAuditSink {
    events: TokioMutex<Vec<(Option<String>, String, String)>>,
}

impl MockAuditSink {
    pub fn new() -> Self {
        Self {
            events: TokioMutex::new(Vec::new()),
        }
    }

    pub async fn events(&self) -> Vec<(Option<String>, String, String)> {
        self.events.lock().await.clone()
    }
}

#[async_trait]
impl AuditSink for MockAuditSink {
    async fn record(
        &self,
        session_id: Option<&str>,
        event_type: &str,
        event_json: &str,
    ) -> Result<(), MemoryError> {
        self.events.lock().await.push((
            session_id.map(String::from),
            event_type.to_owned(),
            event_json.to_owned(),
        ));
        Ok(())
    }

    async fn verify_chain(&self) -> Result<(), MemoryError> {
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// NoopMemory — returns empty results for all operations
// ---------------------------------------------------------------------------

/// No-op `Memory` implementation for tests that don't exercise session recall.
pub struct NoopMemory;

#[async_trait]
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

// ---------------------------------------------------------------------------
// NoopKnowledgeStore — returns empty results for all operations
// ---------------------------------------------------------------------------

/// No-op `KnowledgeStore` implementation for tests that don't exercise knowledge tools.
pub struct NoopKnowledgeStore;

#[async_trait]
impl KnowledgeStore for NoopKnowledgeStore {
    async fn store(&self, _: KnowledgeEntry) -> Result<KnowledgeId, KnowledgeError> {
        Ok(KnowledgeId::from_string("noop"))
    }
    async fn update(&self, _: &KnowledgeEntry) -> Result<(), KnowledgeError> {
        Ok(())
    }
    async fn get(&self, _: &KnowledgeId) -> Result<Option<KnowledgeEntry>, KnowledgeError> {
        Ok(None)
    }
    async fn delete(&self, _: &KnowledgeId) -> Result<(), KnowledgeError> {
        Ok(())
    }
    async fn search(&self, _: &str, _: usize) -> Result<Vec<KnowledgeMatch>, KnowledgeError> {
        Ok(vec![])
    }
    async fn list_by_kind(
        &self,
        _: &KnowledgeKind,
        _: usize,
    ) -> Result<Vec<KnowledgeEntry>, KnowledgeError> {
        Ok(vec![])
    }
    async fn list_by_tag(&self, _: &str, _: usize) -> Result<Vec<KnowledgeEntry>, KnowledgeError> {
        Ok(vec![])
    }
    async fn replace_kind(
        &self,
        _: &KnowledgeKind,
        _: Vec<KnowledgeEntry>,
    ) -> Result<(), KnowledgeError> {
        Ok(())
    }
    async fn record_access(&self, _: &[KnowledgeId]) -> Result<(), KnowledgeError> {
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Config + runtime helpers
// ---------------------------------------------------------------------------

pub fn default_config() -> RuntimeConfig {
    RuntimeConfig {
        default_model: ModelId::from("test-model"),
        default_provider: ProviderId::from("test-provider"),
        system_prompt: None,
        max_output_tokens: 1024,
        max_tool_rounds: 10,
        temperature: None,
        max_turns_per_session: 10,
        drain_timeout_secs: 1,
        max_concurrent_tasks: 8,
        session: freebird_types::config::SessionConfig::default(),
        context: ContextConfig::default(),
    }
}

pub fn default_tools_config() -> ToolsConfig {
    ToolsConfig {
        sandbox_root: std::env::temp_dir(),
        default_timeout_secs: 30,
        allowed_directories: vec![],
        allowed_shell_commands: vec![],
        max_shell_output_bytes: 1_048_576,
        edit: EditConfig::default(),
        git_timeout_secs: 5,
    }
}

pub fn make_tool_executor(tools: Vec<Box<dyn Tool>>) -> ToolExecutor {
    make_tool_executor_with_audit(tools, Arc::new(MockAuditSink::new()))
}

pub fn make_tool_executor_with_audit(
    tools: Vec<Box<dyn Tool>>,
    audit_sink: Arc<MockAuditSink>,
) -> ToolExecutor {
    ToolExecutorBuilder::new(tools, Duration::from_secs(30))
        .audit_sink(audit_sink as Arc<dyn AuditSink>)
        .knowledge_store(Arc::new(NoopKnowledgeStore) as Arc<dyn KnowledgeStore>)
        .memory(Arc::new(NoopMemory) as Arc<dyn Memory>)
        .revocation_list(Arc::new(RevocationList::new()))
        .build()
        .expect("test tool executor construction should not fail")
}

pub fn make_registry(provider: Arc<QueuedProvider>) -> ProviderRegistry {
    let mut registry = ProviderRegistry::new();
    let id = ProviderId::from("test-provider");
    registry.register(id.clone(), Box::new(ArcProvider(provider)));
    registry.set_failover_chain(vec![id]);
    registry
}

// ---------------------------------------------------------------------------
// Event extraction helpers
// ---------------------------------------------------------------------------

/// Helper: extract text from an `OutboundEvent::Message`.
pub fn message_text(event: &OutboundEvent) -> Option<&str> {
    match event {
        OutboundEvent::Message { text, .. } => Some(text.as_str()),
        _ => None,
    }
}

/// Helper: extract text from an `OutboundEvent::Error`.
pub fn error_text(event: &OutboundEvent) -> Option<&str> {
    match event {
        OutboundEvent::Error { text, .. } => Some(text.as_str()),
        _ => None,
    }
}

/// Filter out status/control events (`ToolStart`, `ToolEnd`, `TurnComplete`),
/// keeping only content events (messages, errors, stream chunks).
pub fn without_status_events(events: Vec<OutboundEvent>) -> Vec<OutboundEvent> {
    events
        .into_iter()
        .filter(|e| {
            !matches!(
                e,
                OutboundEvent::ToolStart { .. }
                    | OutboundEvent::ToolEnd { .. }
                    | OutboundEvent::TurnComplete { .. }
            )
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Response factories — shared across agentic_loop_tests, consent_bridge_tests
// ---------------------------------------------------------------------------

pub fn text_response(text: &str) -> ResponseFactory {
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

pub fn tool_use_response(tool_name: &str, input: serde_json::Value) -> ResponseFactory {
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

pub fn max_tokens_response(text: &str) -> ResponseFactory {
    let text = text.to_owned();
    Box::new(move || {
        Ok(CompletionResponse {
            message: Message {
                role: Role::Assistant,
                content: vec![ContentBlock::Text { text: text.clone() }],
                timestamp: Utc::now(),
            },
            stop_reason: StopReason::MaxTokens,
            usage: TokenUsage::default(),
            model: ModelId::from("test-model"),
        })
    })
}

pub fn stop_sequence_response(text: &str) -> ResponseFactory {
    let text = text.to_owned();
    Box::new(move || {
        Ok(CompletionResponse {
            message: Message {
                role: Role::Assistant,
                content: vec![ContentBlock::Text { text: text.clone() }],
                timestamp: Utc::now(),
            },
            stop_reason: StopReason::StopSequence,
            usage: TokenUsage::default(),
            model: ModelId::from("test-model"),
        })
    })
}

// ---------------------------------------------------------------------------
// send_message_and_collect — shared across agentic_loop_tests, stream_tests
// ---------------------------------------------------------------------------

/// Send a message then quit, run the runtime, and collect all outbound events.
pub async fn send_message_and_collect(
    inbound_tx: mpsc::Sender<InboundEvent>,
    mut outbound_rx: mpsc::Receiver<OutboundEvent>,
    runtime: AgentRuntime,
    text: &str,
) -> Vec<OutboundEvent> {
    inbound_tx
        .send(InboundEvent::Message {
            raw_text: text.into(),
            sender_id: "alice".into(),
            attachments: vec![],
        })
        .await
        .unwrap();
    // Drop sender to trigger EOF — runtime will drain spawned tasks before exiting.
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
    events
}
