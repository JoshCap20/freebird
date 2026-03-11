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

use freebird_runtime::registry::ProviderRegistry;
use freebird_runtime::tool_executor::ToolExecutor;
use freebird_traits::channel::{
    AuthRequirement, Channel, ChannelError, ChannelHandle, ChannelInfo, InboundEvent, OutboundEvent,
};
use freebird_traits::id::{ChannelId, ModelId, ProviderId};
use freebird_traits::provider::{
    CompletionRequest, CompletionResponse, Provider, ProviderError, ProviderFeature, ProviderInfo,
    StreamEvent,
};
use freebird_traits::tool::Tool;
use freebird_types::config::{EditConfig, RuntimeConfig, ToolsConfig};

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
    }
}

pub fn make_tool_executor(tools: Vec<Box<dyn Tool>>) -> ToolExecutor {
    ToolExecutor::new(tools, Duration::from_secs(30), None, vec![], None, None)
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
