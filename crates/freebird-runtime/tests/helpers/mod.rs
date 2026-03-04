//! Shared test infrastructure for `freebird-runtime` integration tests.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::BTreeSet;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use async_trait::async_trait;
use tokio::sync::Mutex as TokioMutex;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

use freebird_traits::channel::{
    AuthRequirement, Channel, ChannelError, ChannelHandle, ChannelInfo, InboundEvent, OutboundEvent,
};
use freebird_traits::id::ChannelId;

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
