//! `AgentRuntime` — the core agentic loop.
//!
//! Wires together channels, providers, tools, memory, and sessions into
//! a single event loop. Starts the channel, consumes inbound events,
//! routes them to handlers, and shuts down gracefully when the
//! cancellation token fires or the inbound stream closes.

use std::pin::Pin;
use std::sync::Arc;

use chrono::Utc;
use freebird_security::approval::ApprovalRequest;
use freebird_security::audit::{AuditEventType, InjectionSource};
use freebird_security::budget::TokenBudget;
use freebird_security::capability::CapabilityGrant;
use freebird_security::error::Severity;
use freebird_security::injection;
use freebird_security::safe_types::{SafeMessage, ScannedModelResponse, ValidationResult};
use freebird_security::taint::Tainted;
use freebird_traits::audit::AuditSink;
use freebird_traits::channel::{
    Channel, ChannelError, ChannelFeature, InboundEvent, OutboundEvent,
};
use freebird_traits::event::{ConversationEvent, EventSink};
use freebird_traits::id::SessionId;
use freebird_traits::knowledge::{KnowledgeMatch, KnowledgeStore};
use freebird_traits::memory::{Conversation, Memory, MemoryError, ToolInvocation, Turn};
use freebird_traits::provider::{
    CompletionRequest, CompletionResponse, ContentBlock, Message, ProviderError, Role, StopReason,
    StreamEvent, TokenUsage, ToolDefinition,
};
use freebird_traits::tool::{Capability, ToolOutcome};
use freebird_types::config::{
    BudgetConfig, ConversationSummary, InjectionResponse, KnowledgeConfig, RuntimeConfig,
    SummarizationConfig, ToolsConfig,
};
// Re-exported for test module via `use super::*`.
#[cfg(test)]
use freebird_types::config::{ContextConfig, InjectionConfig};
use futures::StreamExt;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use freebird_memory::sqlite_summary::SummaryStore;

use crate::history::conversation_to_messages;
use crate::registry::ProviderRegistry;
use crate::stream::StreamAccumulator;
use crate::summarize;

use crate::session::SessionManager;

/// Controls the event loop after handling an event.
///
/// Private — callers of `run()` see `Result<(), RuntimeError>`.
enum LoopAction {
    /// Continue processing events.
    Continue,
    /// Exit the event loop gracefully (e.g., `/quit` command).
    Exit,
}

/// Errors that can occur in the agent runtime.
#[derive(Debug, thiserror::Error)]
pub enum RuntimeError {
    /// A channel operation failed (start or stop).
    #[error("channel error: {0}")]
    Channel(#[from] ChannelError),

    /// A provider operation failed.
    #[error("provider error: {0}")]
    Provider(#[from] ProviderError),

    /// A memory operation failed.
    #[error("memory error: {0}")]
    Memory(#[from] MemoryError),
}

/// The central orchestrator that wires together all subsystems.
///
/// Accepts inbound events from a channel, routes messages through
/// session management, dispatches to providers and tools via the
/// agentic loop, and persists conversations to memory.
pub struct AgentRuntime {
    provider_registry: ProviderRegistry,
    channel: Box<dyn Channel>,
    tool_executor: crate::tool_executor::ToolExecutor,
    /// Receives approval requests from the `ToolExecutor`'s `ApprovalGate`.
    /// Consumed by the approval-forwarder background task.
    approval_rx: Option<tokio::sync::mpsc::Receiver<ApprovalRequest>>,
    memory: Arc<dyn Memory>,
    knowledge_store: Option<Arc<dyn KnowledgeStore>>,
    knowledge_config: KnowledgeConfig,
    config: RuntimeConfig,
    tools_config: ToolsConfig,
    /// Token and tool-round budget limits (ASI08). Used to create a
    /// per-session `TokenBudget` when a new session starts.
    budget_config: BudgetConfig,
    /// Default session TTL in hours. Used to set expiration on capability
    /// grants when per-session auth is not yet wired up.
    default_session_ttl_hours: u64,
    event_sink: Option<Arc<dyn EventSink>>,
    audit_sink: Option<Arc<dyn AuditSink>>,
    summary_store: Option<Arc<SummaryStore>>,
    summarization_config: SummarizationConfig,
    sessions: SessionManager,
}

impl AgentRuntime {
    /// Build a new runtime with all dependencies.
    ///
    /// The `SessionManager` is created internally — it's an implementation
    /// detail, not an external dependency.
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        provider_registry: ProviderRegistry,
        channel: Box<dyn Channel>,
        tool_executor: crate::tool_executor::ToolExecutor,
        approval_rx: Option<tokio::sync::mpsc::Receiver<ApprovalRequest>>,
        memory: Arc<dyn Memory>,
        knowledge_store: Option<Arc<dyn KnowledgeStore>>,
        knowledge_config: KnowledgeConfig,
        config: RuntimeConfig,
        tools_config: ToolsConfig,
        budget_config: BudgetConfig,
        default_session_ttl_hours: u64,
        event_sink: Option<Arc<dyn EventSink>>,
        audit_sink: Option<Arc<dyn AuditSink>>,
        summary_store: Option<Arc<SummaryStore>>,
        summarization_config: SummarizationConfig,
    ) -> Self {
        let sessions = SessionManager::with_config(config.session.clone());
        Self {
            provider_registry,
            channel,
            tool_executor,
            approval_rx,
            memory,
            knowledge_store,
            knowledge_config,
            config,
            tools_config,
            budget_config,
            default_session_ttl_hours,
            event_sink,
            audit_sink,
            summary_store,
            summarization_config,
            sessions,
        }
    }

    /// Returns a reference to the internal [`SessionManager`].
    ///
    /// Primarily useful for tests that need to inject restricted capability
    /// grants after session creation (e.g., capability denial E2E tests).
    #[must_use]
    pub const fn sessions(&self) -> &SessionManager {
        &self.sessions
    }

    /// Look up `max_context_tokens` for the conversation's model from the
    /// provider registry. Returns `None` if no matching model is found.
    fn get_max_context_tokens(&self, conversation: &Conversation) -> Option<u32> {
        for pid in self.provider_registry.provider_ids() {
            if let Some(provider) = self.provider_registry.get(pid) {
                for model in &provider.info().supported_models {
                    if model.id == conversation.model_id {
                        return Some(model.max_context_tokens);
                    }
                }
            }
        }
        None
    }

    /// Attempt to summarize older conversation turns.
    ///
    /// Non-fatal: all errors are logged and skipped. Summarization is retried
    /// on the next turn. Returns the updated summary if one was generated.
    #[allow(clippy::too_many_lines)]
    async fn maybe_summarize(
        &self,
        session_id: &SessionId,
        conversation: &Conversation,
        existing_summary: Option<&ConversationSummary>,
    ) -> Option<ConversationSummary> {
        let store = self.summary_store.as_ref()?;

        if !self.summarization_config.enabled {
            return None;
        }

        // Need max_context_tokens to evaluate threshold
        let Some(max_context_tokens) = self.get_max_context_tokens(conversation) else {
            tracing::debug!(%session_id, "cannot determine max_context_tokens for model — skipping summarization");
            return None;
        };

        // Check if summarization is needed
        let messages = conversation_to_messages(conversation);
        if !summarize::should_summarize(
            &self.summarization_config,
            &messages,
            max_context_tokens,
            conversation.turns.len(),
            existing_summary,
        ) {
            return None;
        }

        // Budget guard: skip if remaining tokens < 2 * max_summary_tokens
        let budget = self.sessions.get_budget(session_id).await;
        if let Some(ref b) = budget {
            let remaining = b.remaining_tokens();
            let required = u64::from(self.summarization_config.max_summary_tokens) * 2;
            if remaining < required {
                tracing::info!(
                    %session_id, remaining, required,
                    "insufficient token budget for summarization — skipping"
                );
                return None;
            }
        }

        // Build the summarization request
        let Some((request, new_summarized_through)) = summarize::build_summary_request(
            &self.summarization_config,
            conversation,
            existing_summary,
            &conversation.model_id,
        ) else {
            tracing::debug!(%session_id, "not enough turns to summarize");
            return None;
        };

        tracing::info!(
            %session_id,
            summarized_through = new_summarized_through,
            "triggering conversation summarization"
        );

        self.audit(
            session_id,
            AuditEventType::PolicyViolation {
                rule: "summarization_triggered".into(),
                context: format!(
                    "summarizing turns 0..={new_summarized_through} ({} total turns)",
                    conversation.turns.len()
                ),
                severity: Severity::Low,
            },
        )
        .await;

        // Call the provider
        let (_provider_id, response) = match self
            .provider_registry
            .complete_with_failover(request)
            .await
        {
            Ok(r) => r,
            Err(e) => {
                tracing::warn!(%session_id, error = %e, "summarization provider call failed — skipping");
                return None;
            }
        };

        // Record token usage (non-fatal if budget exceeded)
        if let Some(ref b) = budget {
            if let Err(e) = b.record_usage(&response.usage) {
                tracing::warn!(%session_id, error = %e, "summarization token usage exceeded budget");
            }
        }

        // Extract summary text from response
        let summary_text = response
            .message
            .content
            .iter()
            .filter_map(|block| {
                if let ContentBlock::Text { text } = block {
                    Some(text.as_str())
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
            .join("\n");

        if summary_text.is_empty() {
            tracing::warn!(%session_id, "summarization response was empty — skipping");
            return None;
        }

        // Scan summary for injection via ScannedModelResponse
        let scanned = ScannedModelResponse::from_raw(&summary_text);
        if scanned.injection_detected() {
            tracing::warn!(
                %session_id,
                "injection detected in summarization response — discarding"
            );
            self.audit(
                session_id,
                AuditEventType::InjectionDetected {
                    pattern: "injection in summarization response".into(),
                    source: InjectionSource::ModelResponse,
                    severity: Severity::Critical,
                },
            )
            .await;
            return None;
        }

        // Build and save the summary
        let original_token_estimate = summarize::estimate_token_count(&messages);
        let summary = ConversationSummary {
            session_id: session_id.clone(),
            text: scanned.content().to_owned(),
            summarized_through_turn: new_summarized_through,
            original_token_estimate,
            generated_at: Utc::now(),
        };

        if let Err(e) = store.save(&summary).await {
            tracing::warn!(%session_id, error = %e, "failed to persist summary — skipping");
            return None;
        }

        tracing::info!(
            %session_id,
            summarized_through = new_summarized_through,
            "conversation summary saved"
        );

        Some(summary)
    }

    /// Create a default [`CapabilityGrant`] for a new session.
    ///
    /// Grants all capabilities scoped to the configured sandbox root with a
    /// time-limited expiration based on `default_session_ttl_hours`. Logs a
    /// warning because this is a permissive fallback — per-session auth
    /// should scope grants from the authenticated credential.
    fn create_default_grant(
        &self,
    ) -> Result<CapabilityGrant, freebird_security::error::SecurityError> {
        tracing::warn!(
            ttl_hours = self.default_session_ttl_hours,
            "using permissive default capability grant — per-session auth not yet wired"
        );
        let ttl_hours = i64::try_from(self.default_session_ttl_hours).map_err(|_| {
            freebird_security::error::SecurityError::InvalidCredential {
                reason: format!(
                    "default_session_ttl_hours value {} exceeds maximum representable duration",
                    self.default_session_ttl_hours
                ),
            }
        })?;
        let expires_at = Utc::now() + chrono::Duration::hours(ttl_hours);
        CapabilityGrant::new(
            [
                Capability::FileRead,
                Capability::FileWrite,
                Capability::FileDelete,
                Capability::ShellExecute,
                Capability::ProcessSpawn,
                Capability::NetworkOutbound,
                Capability::NetworkListen,
                Capability::EnvRead,
            ]
            .into_iter()
            .collect(),
            self.tools_config.sandbox_root.clone(),
            Some(expires_at),
        )
    }

    /// Run the event loop until shutdown.
    ///
    /// Starts the channel, consumes inbound events, routes to handlers.
    /// Exits when:
    /// - `cancel` token is cancelled (external shutdown signal)
    /// - Inbound stream closes (e.g., stdin EOF)
    /// - `/quit` command received
    ///
    /// Calls `channel.stop()` before returning.
    ///
    /// # Errors
    ///
    /// Returns `RuntimeError::Channel` if the channel fails to start or stop.
    pub async fn run(&mut self, cancel: CancellationToken) -> Result<(), RuntimeError> {
        let handle = self.channel.start().await?;
        let outbound = handle.outbound;

        tracing::info!("agent runtime started, waiting for messages");

        // Fan-out inbound events into a splitter task and optional approval
        // bridge task. See `spawn_approval_bridge` for details.
        let (mut main_rx, splitter_task, approval_task) =
            self.spawn_approval_bridge(handle.inbound, &outbound, &cancel);

        loop {
            tokio::select! {
                () = cancel.cancelled() => {
                    tracing::info!("shutdown signal received, stopping runtime");
                    break;
                }
                event = main_rx.recv() => {
                    if let Some(event) = event {
                        if matches!(
                            self.handle_event(event, &outbound).await,
                            LoopAction::Exit,
                        ) {
                            break;
                        }
                    } else {
                        tracing::info!("inbound stream closed");
                        break;
                    }
                }
            }
        }

        // Clean up spawned tasks
        splitter_task.abort();
        if let Some(task) = approval_task {
            task.abort();
        }

        self.channel.stop().await?;
        Ok(())
    }

    /// Spawn background tasks for approval bridge plumbing.
    ///
    /// Returns a receiver for non-approval inbound events, the splitter task
    /// handle, and an optional approval-forwarder task handle.
    ///
    /// **Splitter task**: reads from the raw inbound stream and routes
    /// `ApprovalResponse` events directly to the approval gate's
    /// [`ApprovalResponder`], bypassing the main event loop. All other
    /// events are forwarded to the returned `main_rx`. This is necessary
    /// because `handle_event()` may block on `check_consent()` (awaiting
    /// user approval via oneshot), and the `ApprovalResponse` that unblocks
    /// it also arrives as an `InboundEvent`.
    ///
    /// **Approval-forwarder task**: reads `ApprovalRequest`s from the gate's
    /// mpsc channel and forwards them as `OutboundEvent::ApprovalRequest` to
    /// the user's channel.
    fn spawn_approval_bridge(
        &mut self,
        inbound: Pin<Box<dyn futures::Stream<Item = InboundEvent> + Send>>,
        outbound: &mpsc::Sender<OutboundEvent>,
        cancel: &CancellationToken,
    ) -> (
        mpsc::Receiver<InboundEvent>,
        tokio::task::JoinHandle<()>,
        Option<tokio::task::JoinHandle<()>>,
    ) {
        let (main_tx, main_rx) = mpsc::channel::<InboundEvent>(32);
        let approval_responder = self.tool_executor.approval_responder();
        let has_approval_gate = approval_responder.is_some();

        let splitter_cancel = cancel.clone();
        let splitter_outbound = outbound.clone();
        let splitter_task = tokio::spawn({
            let mut inbound = inbound;
            async move {
                loop {
                    tokio::select! {
                        () = splitter_cancel.cancelled() => break,
                        event = inbound.next() => {
                            match event {
                                Some(InboundEvent::ApprovalResponse {
                                    request_id, approved, reason, sender_id, budget_action,
                                }) if has_approval_gate => {
                                    if let Some(ref resp) = approval_responder {
                                        let response = if let Some(ref action_str) = budget_action {
                                            match freebird_security::approval::BudgetOverrideAction::from_wire(action_str) {
                                                Some(action) => freebird_security::approval::ApprovalResponse::BudgetOverride { action },
                                                None if approved => {
                                                    tracing::warn!(budget_action = %action_str, "unrecognized budget_action, falling back to Approved");
                                                    freebird_security::approval::ApprovalResponse::Approved
                                                }
                                                None => freebird_security::approval::ApprovalResponse::Denied { reason },
                                            }
                                        } else if approved {
                                            freebird_security::approval::ApprovalResponse::Approved
                                        } else {
                                            freebird_security::approval::ApprovalResponse::Denied { reason }
                                        };
                                        if resp.respond(&request_id, response).await {
                                            tracing::info!(
                                                %request_id, %sender_id, approved,
                                                "approval response delivered"
                                            );
                                        } else {
                                            tracing::warn!(
                                                %request_id, %sender_id,
                                                "approval response for unknown or expired request"
                                            );
                                            let _ = splitter_outbound.send(
                                                OutboundEvent::Error {
                                                    text: format!(
                                                        "No pending approval request with id `{request_id}` \
                                                         (expired or already responded)"
                                                    ),
                                                    recipient_id: sender_id,
                                                }
                                            ).await;
                                        }
                                    }
                                }
                                Some(event) => {
                                    if main_tx.send(event).await.is_err() { break; }
                                }
                                None => break,
                            }
                        }
                    }
                }
            }
        });

        let approval_task = self.approval_rx.take().map(|mut approval_rx| {
            let approval_cancel = cancel.clone();
            let approval_outbound = outbound.clone();
            tokio::spawn(async move {
                loop {
                    tokio::select! {
                        () = approval_cancel.cancelled() => break,
                        req = approval_rx.recv() => {
                            match req {
                                Some(req) => {
                                    let category_json = serde_json::to_string(&req.category)
                                        .unwrap_or_else(|_| String::from("{}"));
                                    let event = OutboundEvent::ApprovalRequest {
                                        request_id: req.id,
                                        category_json,
                                        expires_at: req.expires_at.to_rfc3339(),
                                        recipient_id: req.sender_id,
                                    };
                                    if approval_outbound.send(event).await.is_err() {
                                        tracing::warn!(
                                            "approval outbound channel closed; \
                                             approval request dropped"
                                        );
                                        break;
                                    }
                                }
                                None => break,
                            }
                        }
                    }
                }
            })
        });

        (main_rx, splitter_task, approval_task)
    }

    /// Route an inbound event to the appropriate handler.
    ///
    /// Returns `LoopAction::Exit` only for `/quit`. All other events
    /// (including handler errors) return `LoopAction::Continue` — the
    /// runtime is resilient to individual message failures.
    async fn handle_event(
        &self,
        event: InboundEvent,
        outbound: &mpsc::Sender<OutboundEvent>,
    ) -> LoopAction {
        match event {
            InboundEvent::Message {
                raw_text,
                sender_id,
                ..
            } => {
                let session_id = self
                    .sessions
                    .resolve(self.channel.info().id.as_str(), &sender_id)
                    .await;

                // Ensure per-session TokenBudget and CapabilityGrant exist (idempotent).
                if self.sessions.get_budget(&session_id).await.is_none() {
                    self.sessions
                        .set_budget(&session_id, TokenBudget::new(&self.budget_config))
                        .await;
                }
                if self.sessions.get_grant(&session_id).await.is_none() {
                    match self.create_default_grant() {
                        Ok(grant) => {
                            self.sessions.set_grant(&session_id, grant).await;
                        }
                        Err(e) => {
                            tracing::error!(error = %e, "cannot create capability grant for session");
                            send_outbound(
                                outbound,
                                OutboundEvent::Error {
                                    text: "Internal error: sandbox root is not accessible".into(),
                                    recipient_id: sender_id.clone(),
                                },
                            )
                            .await;
                            return LoopAction::Continue;
                        }
                    }
                }

                tracing::info!(%session_id, %sender_id, "handling message");
                self.handle_message(raw_text, sender_id, session_id, outbound)
                    .await;
                LoopAction::Continue
            }
            InboundEvent::Command {
                name,
                args,
                sender_id,
            } => {
                let action = self
                    .handle_command(&name, &args, &sender_id, outbound)
                    .await;
                send_outbound(
                    outbound,
                    OutboundEvent::TurnComplete {
                        recipient_id: sender_id,
                    },
                )
                .await;
                action
            }
            InboundEvent::Connected { sender_id } => {
                tracing::info!(%sender_id, "user connected");
                let channel_id = self.channel.info().id.as_str().to_owned();
                self.audit_no_session(AuditEventType::ChannelConnected {
                    channel_id,
                    remote_addr: Some(sender_id),
                })
                .await;
                LoopAction::Continue
            }
            InboundEvent::Disconnected { sender_id } => {
                tracing::info!(%sender_id, "user disconnected");
                let channel_id = self.channel.info().id.as_str().to_owned();
                self.audit_no_session(AuditEventType::ChannelDisconnected {
                    channel_id,
                    reason: None,
                })
                .await;
                LoopAction::Continue
            }
            InboundEvent::ApprovalResponse {
                request_id,
                sender_id,
                ..
            } => {
                // When an approval gate is configured, ApprovalResponse events
                // are handled by the splitter task (which runs concurrently
                // and can deliver responses while handle_event is blocked).
                // This arm only fires when no approval gate is configured
                // (the splitter passes them through to the main loop).
                tracing::warn!(
                    %request_id,
                    %sender_id,
                    "approval response received but no approval gate is configured"
                );
                LoopAction::Continue
            }
        }
    }

    /// Handle a `/command` event.
    ///
    /// Returns `LoopAction::Exit` only for `/quit`.
    async fn handle_command(
        &self,
        name: &str,
        _args: &[String],
        sender_id: &str,
        outbound: &mpsc::Sender<OutboundEvent>,
    ) -> LoopAction {
        match name {
            "quit" => {
                let _ = outbound
                    .send(OutboundEvent::Message {
                        text: "Goodbye!".into(),
                        recipient_id: sender_id.into(),
                    })
                    .await;
                LoopAction::Exit
            }
            "new" => {
                let session_id = self
                    .sessions
                    .new_session(self.channel.info().id.as_str(), sender_id)
                    .await;
                // Create a fresh budget for the new session.
                self.sessions
                    .set_budget(&session_id, TokenBudget::new(&self.budget_config))
                    .await;
                // Create a fresh capability grant for the new session.
                match self.create_default_grant() {
                    Ok(grant) => {
                        self.sessions.set_grant(&session_id, grant).await;
                        let _ = outbound
                            .send(OutboundEvent::Message {
                                text: format!("New session started: {session_id}"),
                                recipient_id: sender_id.into(),
                            })
                            .await;
                    }
                    Err(e) => {
                        tracing::error!(error = %e, "cannot create capability grant for new session");
                        let _ = outbound
                            .send(OutboundEvent::Error {
                                text: "Failed to start new session: sandbox root is not accessible"
                                    .into(),
                                recipient_id: sender_id.into(),
                            })
                            .await;
                    }
                }
                LoopAction::Continue
            }
            "help" => {
                let help_text = [
                    "Available commands:",
                    "  /quit — exit",
                    "  /new  — start a new session",
                    "  /help — show this message",
                ]
                .join("\n");
                let _ = outbound
                    .send(OutboundEvent::Message {
                        text: help_text,
                        recipient_id: sender_id.into(),
                    })
                    .await;
                LoopAction::Continue
            }
            unknown => {
                let _ = outbound
                    .send(OutboundEvent::Error {
                        text: format!("Unknown command: /{unknown}"),
                        recipient_id: sender_id.into(),
                    })
                    .await;
                LoopAction::Continue
            }
        }
    }

    /// Handle an inbound user message — the agentic loop.
    ///
    /// 1. Wraps input in `Tainted`, validates via `SafeMessage::from_tainted()`
    /// 2. Loads/creates conversation from memory
    /// 3. Builds `CompletionRequest` with tool definitions and history
    /// 4. Calls provider via `ProviderRegistry::complete_with_failover()`
    /// 5. Handles tool-use responses (execute tools, collect results, loop)
    /// 6. Sends final response to user
    /// 7. Persists conversation to memory
    ///
    /// Infallible — all errors are caught and sent as `OutboundEvent::Error`.
    async fn handle_message(
        &self,
        raw_text: String,
        sender_id: String,
        session_id: SessionId,
        outbound: &mpsc::Sender<OutboundEvent>,
    ) {
        self.handle_message_inner(&raw_text, &sender_id, &session_id, outbound)
            .await;

        // Always signal turn complete — even on validation/load errors —
        // so the client knows to re-prompt for input.
        send_outbound(
            outbound,
            OutboundEvent::TurnComplete {
                recipient_id: sender_id,
            },
        )
        .await;
    }

    /// Inner implementation of message handling. Separated so `handle_message`
    /// can unconditionally send `TurnComplete` regardless of early returns.
    async fn handle_message_inner(
        &self,
        raw_text: &str,
        sender_id: &str,
        session_id: &SessionId,
        outbound: &mpsc::Sender<OutboundEvent>,
    ) {
        // 1. Taint input and validate
        let Some(safe_message) = self
            .validate_input(raw_text, sender_id, session_id, outbound)
            .await
        else {
            return;
        };

        // 2. Load or create conversation
        let Some(mut conversation) = self
            .load_or_create_conversation(session_id, sender_id, outbound)
            .await
        else {
            return;
        };

        // 3. Retrieve per-session capability grant
        let Some(grant) = self.sessions.get_grant(session_id).await else {
            tracing::error!(%session_id, "no capability grant found for session");
            send_outbound(
                outbound,
                OutboundEvent::Error {
                    text: "Internal error: session has no capability grant".into(),
                    recipient_id: sender_id.into(),
                },
            )
            .await;
            return;
        };

        // 3b. Load existing summary and attempt summarization
        let existing_summary = if let Some(ref store) = self.summary_store {
            match store.load(session_id).await {
                Ok(s) => s,
                Err(e) => {
                    tracing::warn!(%session_id, error = %e, "failed to load summary — proceeding without");
                    None
                }
            }
        } else {
            None
        };

        // Attempt summarization before the agentic loop. Non-fatal.
        let summary = self
            .maybe_summarize(session_id, &conversation, existing_summary.as_ref())
            .await
            .or(existing_summary);

        // 4. Check streaming support and dispatch to appropriate loop
        let use_streaming = self
            .channel
            .info()
            .features
            .contains(&ChannelFeature::Streaming)
            && self.any_provider_supports_streaming();

        let current_turn = if use_streaming {
            self.run_agentic_loop_streaming(
                &safe_message,
                sender_id,
                session_id,
                &conversation,
                &grant,
                outbound,
                summary.as_ref(),
            )
            .await
        } else {
            self.run_agentic_loop(
                &safe_message,
                sender_id,
                session_id,
                &conversation,
                &grant,
                outbound,
                None,
                summary.as_ref(),
            )
            .await
        };

        // 5. Persist conversation
        conversation.turns.push(current_turn);
        conversation.updated_at = Utc::now();

        if let Err(e) = self.memory.save(&conversation).await {
            tracing::error!(error = %e, "failed to persist conversation");
        }
    }

    /// Validate raw input through taint tracking. Returns `None` if rejected.
    ///
    /// Uses three-state `ValidationResult`:
    /// - `Clean` → proceed immediately
    /// - `Warning` → behavior depends on `injection_config.input_response`:
    ///   - `Block` → reject outright
    ///   - `Prompt` → ask user via `ApprovalGate`; fallback to warn-and-proceed
    ///   - `Allow` → warn and proceed
    /// - `Rejected` → hard failure (e.g., input too long)
    #[allow(clippy::too_many_lines)] // three-way config branch + three-way gate result
    async fn validate_input(
        &self,
        raw_text: &str,
        sender_id: &str,
        session_id: &SessionId,
        outbound: &mpsc::Sender<OutboundEvent>,
    ) -> Option<SafeMessage> {
        let tainted = Tainted::new(raw_text);
        match SafeMessage::from_tainted(&tainted) {
            ValidationResult::Clean(msg) => Some(msg),
            ValidationResult::Warning { message, warning } => {
                let warning_str = warning.to_string();

                // Audit the injection detection.
                self.audit(
                    session_id,
                    AuditEventType::InjectionDetected {
                        pattern: warning_str.clone(),
                        source: InjectionSource::UserInput,
                        severity: Severity::High,
                    },
                )
                .await;

                match self.tool_executor.input_injection_response() {
                    InjectionResponse::Block => {
                        tracing::warn!(
                            session_id = %session_id,
                            "injection detected in user input — blocking per config"
                        );
                        let _ = outbound
                            .send(OutboundEvent::Error {
                                text: format!(
                                    "Message rejected: injection pattern detected — {warning_str}"
                                ),
                                recipient_id: sender_id.into(),
                            })
                            .await;
                        return None;
                    }
                    InjectionResponse::Allow => {
                        tracing::warn!(
                            session_id = %session_id,
                            "injection detected in user input — allowing per config"
                        );
                        let _ = outbound
                            .send(OutboundEvent::Error {
                                text: format!(
                                    "Warning: {warning_str}. Proceeding with your message."
                                ),
                                recipient_id: sender_id.into(),
                            })
                            .await;
                        return Some(message);
                    }
                    InjectionResponse::Prompt => {} // fall through to prompt logic
                }

                let preview = raw_text.chars().take(200).collect::<String>();

                match self
                    .tool_executor
                    .check_security_warning(
                        "injection_input".into(),
                        warning_str.clone(),
                        preview,
                        "user_input".into(),
                        sender_id,
                    )
                    .await
                {
                    Ok(()) => {
                        tracing::info!(
                            session_id = %session_id,
                            "user approved security warning for input injection — proceeding"
                        );
                        Some(message)
                    }
                    Err(freebird_security::approval::ApprovalError::Denied { .. }) => {
                        tracing::warn!(%session_id, "user denied input injection security warning");
                        let _ = outbound
                            .send(OutboundEvent::Error {
                                text: format!(
                                    "Message rejected: security warning denied — {warning_str}"
                                ),
                                recipient_id: sender_id.into(),
                            })
                            .await;
                        None
                    }
                    Err(e) => {
                        // Expired, too many pending, channel closed, or no gate.
                        // Fall back to warn-and-proceed since the sanitized
                        // message is safe to use.
                        tracing::warn!(
                            session_id = %session_id,
                            error = %e,
                            "approval gate unavailable for input security warning — proceeding with sanitized content"
                        );
                        let _ = outbound
                            .send(OutboundEvent::Error {
                                text: format!(
                                    "Warning: {warning_str}. Proceeding with your message."
                                ),
                                recipient_id: sender_id.into(),
                            })
                            .await;
                        Some(message)
                    }
                }
            }
            ValidationResult::Rejected(e) => {
                tracing::warn!(%session_id, "input validation rejected message");
                let _ = outbound
                    .send(OutboundEvent::Error {
                        text: format!("Message rejected: {e}"),
                        recipient_id: sender_id.into(),
                    })
                    .await;
                None
            }
        }
    }

    /// Load conversation from memory, or create a new one. Returns `None` on error.
    async fn load_or_create_conversation(
        &self,
        session_id: &SessionId,
        sender_id: &str,
        outbound: &mpsc::Sender<OutboundEvent>,
    ) -> Option<Conversation> {
        match self.memory.load(session_id).await {
            Ok(Some(conv)) => Some(conv),
            Ok(None) => {
                // Emit SessionCreated event for the new conversation
                self.emit_event(
                    session_id,
                    ConversationEvent::SessionCreated {
                        system_prompt: self.config.system_prompt.clone(),
                        model_id: self.config.default_model.as_str().to_owned(),
                        provider_id: self.config.default_provider.as_str().to_owned(),
                    },
                )
                .await;

                // Record session creation in the security audit log
                let grant = self.create_default_grant();
                let capabilities: Vec<String> = grant
                    .map(|g| g.capabilities().iter().map(|c| format!("{c:?}")).collect())
                    .unwrap_or_default();
                self.audit(session_id, AuditEventType::SessionStarted { capabilities })
                    .await;

                Some(Conversation {
                    session_id: session_id.clone(),
                    system_prompt: self.config.system_prompt.clone(),
                    turns: Vec::new(),
                    created_at: Utc::now(),
                    updated_at: Utc::now(),
                    model_id: self.config.default_model.clone(),
                    provider_id: self.config.default_provider.clone(),
                })
            }
            Err(e) => {
                tracing::error!(error = %e, "failed to load conversation from memory");
                let _ = outbound
                    .send(OutboundEvent::Error {
                        text: format!("Failed to load conversation: {e}"),
                        recipient_id: sender_id.into(),
                    })
                    .await;
                None
            }
        }
    }

    /// Retrieve relevant knowledge entries for auto-injection into the prompt.
    ///
    /// Returns an empty vec when auto-retrieval is disabled, no knowledge store
    /// is configured, or no entries pass the relevance threshold.
    async fn retrieve_knowledge_context(&self, query: &str) -> Vec<KnowledgeMatch> {
        if !self.knowledge_config.auto_retrieve {
            return Vec::new();
        }

        let Some(ref store) = self.knowledge_store else {
            return Vec::new();
        };

        let matches = match store
            .search(query, self.knowledge_config.max_context_entries)
            .await
        {
            Ok(m) => m,
            Err(e) => {
                tracing::warn!(error = %e, "knowledge auto-retrieval failed");
                return Vec::new();
            }
        };

        // BM25: lower (more negative) = more relevant. Keep entries at or below threshold.
        let filtered: Vec<KnowledgeMatch> = matches
            .into_iter()
            .filter(|m| m.rank <= self.knowledge_config.relevance_threshold)
            .collect();

        if filtered.is_empty() {
            return Vec::new();
        }

        // Record access for analytics (fire-and-forget).
        let ids: Vec<_> = filtered.iter().map(|m| m.entry.id.clone()).collect();
        if let Err(e) = store.record_access(&ids).await {
            tracing::warn!(error = %e, "failed to record knowledge access");
        }

        tracing::debug!(
            count = filtered.len(),
            "injecting knowledge context into prompt"
        );

        filtered
    }

    /// Format knowledge matches into a context block for injection into the prompt.
    ///
    /// Returns `None` if there are no matches. Respects the configured
    /// `max_context_tokens` budget (estimated at ~4 chars per token).
    fn format_knowledge_context(&self, matches: &[KnowledgeMatch]) -> Option<String> {
        if matches.is_empty() {
            return None;
        }

        let token_budget_chars = self.knowledge_config.max_context_tokens.saturating_mul(4);
        let mut buf = String::with_capacity(token_budget_chars.min(8192));
        buf.push_str("[RELEVANT CONTEXT]\n");
        let mut remaining = token_budget_chars.saturating_sub(buf.len());

        for m in matches {
            let label = format!("[{:?}] ", m.entry.kind);
            let entry_len = label.len() + m.entry.content.len() + 1; // +1 for newline

            if entry_len > remaining {
                // Fit as much of this entry as possible, then stop.
                let avail = remaining.saturating_sub(label.len() + 1);
                if avail > 0 {
                    buf.push_str(&label);
                    // Truncate at a char boundary.
                    let truncated: String = m.entry.content.chars().take(avail).collect();
                    buf.push_str(&truncated);
                    buf.push('\n');
                }
                break;
            }

            buf.push_str(&label);
            buf.push_str(&m.entry.content);
            buf.push('\n');
            remaining -= entry_len;
        }

        Some(buf)
    }

    /// Build the initial state for an agentic loop: user message, turn, messages, tool defs.
    async fn prepare_agentic_loop(
        &self,
        safe_message: &SafeMessage,
        conversation: &Conversation,
        grant: &CapabilityGrant,
        summary: Option<&ConversationSummary>,
    ) -> (Vec<Message>, Turn, Vec<ToolDefinition>) {
        let user_message = Message {
            role: Role::User,
            content: vec![ContentBlock::Text {
                text: safe_message.as_str().to_owned(),
            }],
            timestamp: Utc::now(),
        };

        let tool_definitions = self.tool_executor.tool_definitions_for_grant(grant);

        let messages = conversation_to_messages(conversation);

        // Apply summarization: skip summarized turns and prepend summary.
        // This must happen before observation collapsing so that collapsed
        // tool outputs are only computed for the unsummarized portion.
        let mut messages = summarize::apply_summary_to_messages(messages, conversation, summary);

        // Collapse stale tool outputs from older turns to free context space.
        // Only affects the wire messages — persisted data retains full output.
        if self.config.context.collapse_tool_outputs {
            crate::observation::collapse_observations(
                &mut messages,
                conversation,
                self.config.context.collapse_after_turns,
            );
        }

        // CLAUDE.md §14: scan loaded conversation history for context injection
        // before sending to provider. Filter out any messages containing injection
        // patterns to prevent memory poisoning attacks.
        // This also catches any injection in the prepended summary text.
        messages.retain(|msg| {
            for block in &msg.content {
                if let ContentBlock::Text { text } = block {
                    if injection::scan_context(text).is_err() {
                        tracing::warn!(
                            role = ?msg.role,
                            "context injection detected in loaded history, removing message"
                        );
                        return false;
                    }
                }
            }
            true
        });

        // CLAUDE.md §5: retrieve relevant knowledge via FTS5 search and inject
        // before the user message so the model has context for its response.
        let knowledge_matches = self.retrieve_knowledge_context(safe_message.as_str()).await;
        if let Some(context_text) = self.format_knowledge_context(&knowledge_matches) {
            messages.push(Message {
                role: Role::User,
                content: vec![ContentBlock::Text { text: context_text }],
                timestamp: Utc::now(),
            });
        }

        messages.push(user_message.clone());

        let current_turn = Turn {
            user_message,
            assistant_messages: Vec::new(),
            tool_invocations: Vec::new(),
            started_at: Utc::now(),
            completed_at: None,
        };

        (messages, current_turn, tool_definitions)
    }

    /// Log and report max tool rounds exceeded.
    async fn log_max_rounds_exceeded(
        &self,
        session_id: &SessionId,
        current_turn: &mut Turn,
        sender_id: &str,
        outbound: &mpsc::Sender<OutboundEvent>,
    ) {
        self.audit(
            session_id,
            AuditEventType::PolicyViolation {
                rule: "max_tool_rounds".into(),
                context: format!(
                    "agentic loop exhausted {} rounds without completing",
                    self.config.max_tool_rounds
                ),
                severity: Severity::Medium,
            },
        )
        .await;
        current_turn.completed_at = Some(Utc::now());
        send_outbound(
            outbound,
            OutboundEvent::Error {
                text: "Maximum tool rounds exceeded. Stopping.".into(),
                recipient_id: sender_id.into(),
            },
        )
        .await;
    }

    /// Resolve the override action string and new limit value for audit/logging.
    fn resolve_budget_override(
        action: &freebird_security::approval::BudgetOverrideAction,
        resource: &freebird_security::budget::BudgetResource,
    ) -> (&'static str, Option<u64>) {
        use freebird_security::approval::BudgetOverrideAction;
        use freebird_security::budget::BudgetResource;

        match action {
            BudgetOverrideAction::ApproveOnce => ("approve_once", None),
            BudgetOverrideAction::RaiseLimit { new_limit } => ("raise_limit", Some(*new_limit)),
            BudgetOverrideAction::DisableLimit => {
                let max = match resource {
                    BudgetResource::ToolRoundsPerTurn => u64::from(u32::MAX),
                    _ => u64::MAX,
                };
                ("disable_limit", Some(max))
            }
        }
    }

    /// Apply a budget limit change to the given budget.
    fn apply_budget_limit(
        budget: &freebird_security::budget::TokenBudget,
        resource: &freebird_security::budget::BudgetResource,
        new_limit: u64,
    ) {
        use freebird_security::budget::BudgetResource;

        match resource {
            BudgetResource::TokensPerRequest => budget.set_max_tokens_per_request(new_limit),
            BudgetResource::TokensPerSession => budget.set_max_tokens_per_session(new_limit),
            BudgetResource::ToolRoundsPerTurn => {
                budget.set_max_tool_rounds_per_turn(u32::try_from(new_limit).unwrap_or(u32::MAX));
            }
            BudgetResource::CostPerSession => budget.set_max_cost_microdollars(new_limit),
        }
    }

    /// Handle a budget exceeded error: request user approval to continue.
    ///
    /// Returns `true` if the user approves (caller should continue the loop),
    /// or `false` if denied/expired (caller should abort).
    async fn handle_budget_exceeded(
        &self,
        session_id: &SessionId,
        error: &freebird_security::error::SecurityError,
        current_turn: &mut Turn,
        sender_id: &str,
        outbound: &mpsc::Sender<OutboundEvent>,
        budget: Option<&freebird_security::budget::TokenBudget>,
    ) -> bool {
        tracing::warn!(%session_id, %error, "token budget exceeded — requesting approval");

        let (resource, resource_str, used_val, limit_val) =
            if let freebird_security::error::SecurityError::BudgetExceeded {
                resource,
                used,
                limit,
            } = error
            {
                (resource.clone(), resource.to_string(), *used, *limit)
            } else {
                return false;
            };

        // Attempt approval via the gate.
        match self
            .tool_executor
            .check_budget_approval(resource_str.clone(), used_val, limit_val, sender_id)
            .await
        {
            Ok(action) => {
                let (override_action_str, new_limit_val) =
                    Self::resolve_budget_override(&action, &resource);

                if let (Some(new_lim), Some(b)) = (new_limit_val, budget) {
                    Self::apply_budget_limit(b, &resource, new_lim);
                    tracing::info!(
                        %session_id, resource = %resource_str,
                        action = %override_action_str, new_limit = new_lim,
                        "budget limit updated by user override"
                    );
                } else {
                    tracing::info!(
                        %session_id, resource = %resource_str,
                        "budget override approved by user (once)"
                    );
                }

                self.audit(
                    session_id,
                    AuditEventType::BudgetExceeded {
                        resource: resource_str,
                        used: used_val,
                        limit: limit_val,
                        approved: true,
                        override_action: Some(override_action_str.into()),
                        new_limit: new_limit_val,
                    },
                )
                .await;
                true
            }
            Err(approval_err) => {
                tracing::warn!(
                    %session_id, resource = %resource_str, %approval_err,
                    "budget override denied"
                );
                self.audit(
                    session_id,
                    AuditEventType::BudgetExceeded {
                        resource: resource_str,
                        used: used_val,
                        limit: limit_val,
                        approved: false,
                        override_action: None,
                        new_limit: None,
                    },
                )
                .await;
                current_turn.completed_at = Some(Utc::now());
                send_outbound(
                    outbound,
                    OutboundEvent::Error {
                        text: format!("Budget exceeded: {error} (approval {approval_err})"),
                        recipient_id: sender_id.into(),
                    },
                )
                .await;
                false
            }
        }
    }

    /// Run the agentic loop: call provider, handle tool use, deliver response.
    ///
    /// When `initial_request` is `Some`, the first loop iteration uses it directly
    /// instead of building a fresh `CompletionRequest`. This supports the streaming
    /// fallback path where the request has already been constructed.
    #[allow(clippy::too_many_arguments, clippy::too_many_lines)]
    async fn run_agentic_loop(
        &self,
        safe_message: &SafeMessage,
        sender_id: &str,
        session_id: &SessionId,
        conversation: &Conversation,
        grant: &CapabilityGrant,
        outbound: &mpsc::Sender<OutboundEvent>,
        initial_request: Option<CompletionRequest>,
        summary: Option<&ConversationSummary>,
    ) -> Turn {
        let (mut messages, mut current_turn, tool_definitions) = self
            .prepare_agentic_loop(safe_message, conversation, grant, summary)
            .await;

        // Emit TurnStarted event
        self.emit_event(
            session_id,
            ConversationEvent::TurnStarted {
                turn_index: conversation.turns.len(),
                user_message: current_turn.user_message.clone(),
            },
        )
        .await;

        let turn_index = conversation.turns.len();
        let mut pending_request = initial_request;

        // Retrieve the per-session budget (if set).
        let budget = self.sessions.get_budget(session_id).await;

        // The budget check is the sole enforcer of tool-round limits.
        // When no budget exists, fall back to config.max_tool_rounds.
        let max_rounds = self.config.max_tool_rounds;

        for round in 0.. {
            // Check tool-round budget before each iteration.
            if let Some(ref b) = budget {
                let round_u32 = u32::try_from(round).unwrap_or(u32::MAX);
                if let Err(e) = b.check_tool_rounds(round_u32) {
                    if !self
                        .handle_budget_exceeded(
                            session_id,
                            &e,
                            &mut current_turn,
                            sender_id,
                            outbound,
                            budget.as_deref(),
                        )
                        .await
                    {
                        return current_turn;
                    }
                    // User approved — continue the loop beyond the limit.
                }
            } else if round >= max_rounds {
                // No budget configured — fall back to config limit.
                break;
            }

            let request = pending_request.take().unwrap_or_else(|| {
                self.build_completion_request(conversation, &messages, &tool_definitions)
            });

            let (_provider_id, response) =
                match self.provider_registry.complete_with_failover(request).await {
                    Ok(r) => r,
                    Err(e) => {
                        tracing::error!(error = %e, "all providers failed");
                        let _ = outbound
                            .send(OutboundEvent::Error {
                                text: format!("Provider error: {e}"),
                                recipient_id: sender_id.into(),
                            })
                            .await;
                        return current_turn;
                    }
                };

            // Record token usage and check per-request/per-session limits.
            if let Some(ref b) = budget {
                if let Err(e) = b.record_usage(&response.usage) {
                    if self
                        .handle_budget_exceeded(
                            session_id,
                            &e,
                            &mut current_turn,
                            sender_id,
                            outbound,
                            budget.as_deref(),
                        )
                        .await
                    {
                        // User approved — force-commit the usage.
                        b.force_record_usage(&response.usage);
                    } else {
                        return current_turn;
                    }
                }
            }

            match response.stop_reason {
                StopReason::ToolUse => {
                    let invocations_before = current_turn.tool_invocations.len();
                    let messages_before = current_turn.assistant_messages.len();

                    self.handle_tool_use_round(
                        &response,
                        &mut messages,
                        &mut current_turn,
                        grant,
                        session_id,
                        sender_id,
                        outbound,
                    )
                    .await;

                    self.emit_new_assistant_message(
                        session_id,
                        &current_turn,
                        turn_index,
                        messages_before,
                    )
                    .await;
                    self.emit_new_tool_invocations(
                        session_id,
                        &current_turn,
                        turn_index,
                        invocations_before,
                    )
                    .await;
                }
                StopReason::EndTurn | StopReason::StopSequence => {
                    let messages_before = current_turn.assistant_messages.len();

                    self.deliver_final_response(
                        &response,
                        sender_id,
                        session_id,
                        &mut current_turn,
                        outbound,
                    )
                    .await;

                    self.emit_new_assistant_message(
                        session_id,
                        &current_turn,
                        turn_index,
                        messages_before,
                    )
                    .await;
                    self.emit_turn_completed(session_id, &current_turn, turn_index)
                        .await;

                    return current_turn;
                }
                StopReason::MaxTokens => {
                    let messages_before = current_turn.assistant_messages.len();

                    self.deliver_truncated_response(
                        &response,
                        sender_id,
                        session_id,
                        &mut current_turn,
                        outbound,
                    )
                    .await;

                    self.emit_new_assistant_message(
                        session_id,
                        &current_turn,
                        turn_index,
                        messages_before,
                    )
                    .await;
                    self.emit_turn_completed(session_id, &current_turn, turn_index)
                        .await;

                    return current_turn;
                }
            }
        }

        self.log_max_rounds_exceeded(session_id, &mut current_turn, sender_id, outbound)
            .await;

        self.emit_turn_completed(session_id, &current_turn, turn_index)
            .await;

        current_turn
    }

    /// Build a `CompletionRequest` from the current conversation state.
    fn build_completion_request(
        &self,
        conversation: &Conversation,
        messages: &[Message],
        tool_definitions: &[ToolDefinition],
    ) -> CompletionRequest {
        let base = conversation.system_prompt.as_deref().unwrap_or("");
        let system_prompt = self.build_effective_system_prompt(
            base,
            conversation.model_id.as_str(),
            tool_definitions,
        );

        CompletionRequest {
            model: conversation.model_id.clone(),
            system_prompt: Some(system_prompt),
            messages: messages.to_vec(),
            tools: tool_definitions.to_vec(),
            max_tokens: self.config.max_output_tokens,
            temperature: self.config.temperature,
            stop_sequences: Vec::new(),
        }
    }

    /// Augment the base system prompt with tool and filesystem access
    /// information so the model knows what it can do.
    fn build_effective_system_prompt(
        &self,
        base: &str,
        model_id: &str,
        tool_definitions: &[ToolDefinition],
    ) -> String {
        use std::fmt::Write;

        let mut prompt = base.to_owned();
        let _ = write!(prompt, "\n\nYou are running on model: {model_id}");

        if tool_definitions.is_empty() {
            return prompt;
        }

        // List available tools by name and description.
        prompt.push_str("\n\nYou have the following tools available:\n");
        for def in tool_definitions {
            let _ = writeln!(prompt, "- **{}**: {}", def.name, def.description);
        }

        // Filesystem access context.
        let sandbox = &self.tools_config.sandbox_root;
        let allowed = &self.tools_config.allowed_directories;

        let _ = write!(
            prompt,
            "\nYour sandbox directory is: {}\n",
            sandbox.display()
        );

        if allowed.is_empty() {
            prompt.push_str("You can only access files within the sandbox directory.");
        } else {
            prompt.push_str("You can also access files in these additional directories:\n");
            for dir in allowed {
                let _ = writeln!(prompt, "- {}", dir.display());
            }
        }

        prompt
    }

    /// Execute tool calls from a `ToolUse` response and append results to the message list.
    #[allow(clippy::too_many_arguments)]
    async fn handle_tool_use_round(
        &self,
        response: &CompletionResponse,
        messages: &mut Vec<Message>,
        current_turn: &mut Turn,
        grant: &CapabilityGrant,
        session_id: &SessionId,
        sender_id: &str,
        outbound: &mpsc::Sender<OutboundEvent>,
    ) {
        self.execute_tool_calls(
            &response.message,
            messages,
            current_turn,
            grant,
            session_id,
            sender_id,
            outbound,
        )
        .await;
    }

    /// Extract tool-use blocks from an assistant message, execute them, scan
    /// outputs for injection, and append tool results to the message list.
    ///
    /// Shared between the non-streaming (`handle_tool_use_round`) and streaming
    /// (`run_agentic_loop_streaming`) paths to avoid duplicating security-critical
    /// tool execution code.
    #[allow(clippy::too_many_arguments)]
    async fn execute_tool_calls(
        &self,
        assistant_message: &Message,
        messages: &mut Vec<Message>,
        current_turn: &mut Turn,
        grant: &CapabilityGrant,
        session_id: &SessionId,
        sender_id: &str,
        outbound: &mpsc::Sender<OutboundEvent>,
    ) {
        // Extract tool_use blocks from the message
        let tool_uses: Vec<(String, String, serde_json::Value)> = assistant_message
            .content
            .iter()
            .filter_map(|block| match block {
                ContentBlock::ToolUse { id, name, input } => {
                    Some((id.clone(), name.clone(), input.clone()))
                }
                _ => None,
            })
            .collect();

        // Record intermediate assistant message in the turn for persistence
        current_turn
            .assistant_messages
            .push(assistant_message.clone());

        // Add assistant message with tool_use blocks to conversation
        messages.push(assistant_message.clone());

        // Execute each tool and collect results
        let mut tool_results = Vec::with_capacity(tool_uses.len());
        for (tool_use_id, tool_name, input) in tool_uses {
            send_outbound(
                outbound,
                OutboundEvent::ToolStart {
                    tool_name: tool_name.clone(),
                    recipient_id: sender_id.into(),
                },
            )
            .await;

            let start = std::time::Instant::now();

            // ToolExecutor handles capability checks, consent gates, timeout,
            // injection scanning, and audit logging. Output is already scanned.
            let output = self
                .tool_executor
                .execute(&tool_name, input.clone(), grant, session_id, sender_id)
                .await;

            let duration_ms = u64::try_from(start.elapsed().as_millis()).unwrap_or(u64::MAX);

            let final_content = output.content;
            let outcome = output.outcome;

            let is_error = outcome == ToolOutcome::Error;

            let outcome_str = if is_error { "error" } else { "success" };
            send_outbound(
                outbound,
                OutboundEvent::ToolEnd {
                    tool_name: tool_name.clone(),
                    outcome: outcome_str.into(),
                    duration_ms,
                    recipient_id: sender_id.into(),
                },
            )
            .await;

            current_turn.tool_invocations.push(ToolInvocation {
                tool_use_id: tool_use_id.clone(),
                tool_name,
                input,
                output: Some(final_content.clone()),
                outcome,
                duration_ms: Some(duration_ms),
            });

            tool_results.push(ContentBlock::ToolResult {
                tool_use_id,
                content: final_content,
                is_error,
            });
        }

        // Add tool results as a user-role message (Anthropic API format)
        messages.push(Message {
            role: Role::User,
            content: tool_results,
            timestamp: Utc::now(),
        });
    }

    /// Deliver a final (non-truncated) response to the user.
    ///
    /// Scans model output for injection via `ScannedModelResponse`. On detection:
    /// - Sends `OutboundEvent::Error` instead of the response
    /// - Does NOT persist the tainted response to memory (prevents poisoning)
    async fn deliver_final_response(
        &self,
        response: &CompletionResponse,
        sender_id: &str,
        session_id: &SessionId,
        current_turn: &mut Turn,
        outbound: &mpsc::Sender<OutboundEvent>,
    ) {
        let response_text = extract_text(&response.message);

        // Skip sending empty responses — the model may have returned
        // only tool-use blocks with no text, or genuinely empty content.
        if response_text.is_empty() {
            tracing::warn!(%session_id, "model returned empty text response, skipping delivery");
            current_turn
                .assistant_messages
                .push(response.message.clone());
            current_turn.completed_at = Some(Utc::now());
            return;
        }

        // Scan model output for injection — BLOCK if detected
        let scanned = ScannedModelResponse::from_raw(&response_text);
        if scanned.injection_detected() {
            self.audit_model_injection(session_id).await;

            let _ = outbound
                .send(OutboundEvent::Error {
                    text: ScannedModelResponse::BLOCKED_MESSAGE.into(),
                    recipient_id: sender_id.into(),
                })
                .await;

            // Do NOT persist the tainted response — prevents memory poisoning
        } else {
            let _ = outbound
                .send(OutboundEvent::Message {
                    text: scanned.into_content(),
                    recipient_id: sender_id.into(),
                })
                .await;

            current_turn
                .assistant_messages
                .push(response.message.clone());
        }
        current_turn.completed_at = Some(Utc::now());
    }

    /// Deliver a truncated response (max tokens reached).
    ///
    /// Scans the partial output for injection before delivery.
    async fn deliver_truncated_response(
        &self,
        response: &CompletionResponse,
        sender_id: &str,
        session_id: &SessionId,
        current_turn: &mut Turn,
        outbound: &mpsc::Sender<OutboundEvent>,
    ) {
        let partial_text = extract_text(&response.message);

        // Scan truncated output for injection — BLOCK if detected
        let scanned = ScannedModelResponse::from_raw(&partial_text);
        if scanned.injection_detected() {
            self.audit_model_injection(session_id).await;

            let _ = outbound
                .send(OutboundEvent::Error {
                    text: ScannedModelResponse::BLOCKED_MESSAGE.into(),
                    recipient_id: sender_id.into(),
                })
                .await;
        } else {
            let _ = outbound
                .send(OutboundEvent::Message {
                    text: format!(
                        "{}\n\n[response truncated — max tokens reached]",
                        scanned.content()
                    ),
                    recipient_id: sender_id.into(),
                })
                .await;

            current_turn
                .assistant_messages
                .push(response.message.clone());
        }
        current_turn.completed_at = Some(Utc::now());
    }

    /// Record an audit event without a session context.
    ///
    /// Used for daemon-level and channel-level events that occur
    /// outside of a specific session. No-op when no `AuditSink` is configured.
    /// Errors are logged but never block the agent loop.
    async fn audit_no_session(&self, event: AuditEventType) {
        if let Some(sink) = &self.audit_sink {
            emit_audit(sink.as_ref(), None, event).await;
        }
    }

    async fn audit(&self, session_id: &SessionId, event: AuditEventType) {
        if let Some(sink) = &self.audit_sink {
            emit_audit(sink.as_ref(), Some(session_id.as_str()), event).await;
        }
    }

    /// Emit a conversation event via the `EventSink` if configured.
    ///
    /// Errors are logged but never block the agent loop.
    async fn emit_event(&self, session_id: &SessionId, event: ConversationEvent) {
        if let Some(sink) = &self.event_sink {
            if let Err(e) = sink.append(session_id, event).await {
                tracing::error!(
                    error = %e,
                    %session_id,
                    "event sink append failed — event-sourced persistence may be incomplete"
                );
            }
        }
    }

    /// Emit an `AssistantMessage` event if a new message was appended to the turn.
    async fn emit_new_assistant_message(
        &self,
        session_id: &SessionId,
        turn: &Turn,
        turn_index: usize,
        messages_before: usize,
    ) {
        if turn.assistant_messages.len() > messages_before {
            if let Some(msg) = turn.assistant_messages.get(messages_before).cloned() {
                self.emit_event(
                    session_id,
                    ConversationEvent::AssistantMessage {
                        turn_index,
                        message_index: messages_before,
                        message: msg,
                    },
                )
                .await;
            }
        }
    }

    /// Emit `ToolInvoked` events for all new tool invocations since `start_index`.
    async fn emit_new_tool_invocations(
        &self,
        session_id: &SessionId,
        turn: &Turn,
        turn_index: usize,
        start_index: usize,
    ) {
        for idx in start_index..turn.tool_invocations.len() {
            if let Some(inv) = turn.tool_invocations.get(idx).cloned() {
                self.emit_event(
                    session_id,
                    ConversationEvent::ToolInvoked {
                        turn_index,
                        invocation_index: idx,
                        invocation: inv,
                    },
                )
                .await;
            }
        }
    }

    /// Emit a `TurnCompleted` event.
    async fn emit_turn_completed(&self, session_id: &SessionId, turn: &Turn, turn_index: usize) {
        self.emit_event(
            session_id,
            ConversationEvent::TurnCompleted {
                turn_index,
                completed_at: turn.completed_at.unwrap_or_else(Utc::now),
            },
        )
        .await;
    }

    /// Log and audit a model output injection detection.
    async fn audit_model_injection(&self, session_id: &SessionId) {
        tracing::warn!("injection detected in model output, blocking delivery");
        self.audit(
            session_id,
            AuditEventType::InjectionDetected {
                pattern: "prompt injection in model output".to_owned(),
                source: InjectionSource::ModelResponse,
                severity: Severity::High,
            },
        )
        .await;
    }

    /// Check whether any provider in the failover chain supports streaming.
    fn any_provider_supports_streaming(&self) -> bool {
        self.provider_registry.any_in_chain_supports_streaming()
    }

    /// Streaming variant of the agentic loop.
    ///
    /// Sends `StreamChunk` events for each text delta, `StreamEnd` between
    /// tool-use rounds and at final response. Falls back to non-streaming
    /// `complete_with_failover()` if stream setup fails.
    ///
    /// Injection scan on accumulated text is audit-only — the text has already
    /// been delivered to the user via `StreamChunk` events.
    #[allow(clippy::too_many_lines)] // budget enforcement adds necessary branches
    #[allow(clippy::too_many_arguments)]
    async fn run_agentic_loop_streaming(
        &self,
        safe_message: &SafeMessage,
        sender_id: &str,
        session_id: &SessionId,
        conversation: &Conversation,
        grant: &CapabilityGrant,
        outbound: &mpsc::Sender<OutboundEvent>,
        summary: Option<&ConversationSummary>,
    ) -> Turn {
        let (mut messages, mut current_turn, tool_definitions) = self
            .prepare_agentic_loop(safe_message, conversation, grant, summary)
            .await;

        let turn_index = conversation.turns.len();

        // Emit TurnStarted event
        self.emit_event(
            session_id,
            ConversationEvent::TurnStarted {
                turn_index,
                user_message: current_turn.user_message.clone(),
            },
        )
        .await;

        // Retrieve the per-session budget (if set).
        let budget = self.sessions.get_budget(session_id).await;

        // The budget check is the sole enforcer of tool-round limits.
        // When no budget exists, fall back to config.max_tool_rounds.
        let max_rounds = self.config.max_tool_rounds;

        for round in 0.. {
            // Check tool-round budget before each iteration.
            if let Some(ref b) = budget {
                let round_u32 = u32::try_from(round).unwrap_or(u32::MAX);
                if let Err(e) = b.check_tool_rounds(round_u32) {
                    if !self
                        .handle_budget_exceeded(
                            session_id,
                            &e,
                            &mut current_turn,
                            sender_id,
                            outbound,
                            budget.as_deref(),
                        )
                        .await
                    {
                        return current_turn;
                    }
                    // User approved — continue the loop beyond the limit.
                }
            } else if round >= max_rounds {
                // No budget configured — fall back to config limit.
                break;
            }

            let request = self.build_completion_request(conversation, &messages, &tool_definitions);

            let event_stream = match self
                .provider_registry
                .stream_with_failover(request.clone())
                .await
            {
                Ok((_provider_id, s)) => s,
                Err(e) => {
                    tracing::warn!(error = %e, "stream setup failed on all providers, falling back to non-streaming");
                    return self
                        .run_agentic_loop(
                            safe_message,
                            sender_id,
                            session_id,
                            conversation,
                            grant,
                            outbound,
                            Some(request),
                            summary,
                        )
                        .await;
                }
            };

            let Some((accumulator, stop_reason, usage)) = Self::consume_stream(
                event_stream,
                sender_id,
                session_id,
                self.audit_sink.as_deref(),
                outbound,
            )
            .await
            else {
                return current_turn;
            };

            // Record token usage and check per-request/per-session limits.
            if let Some(ref b) = budget {
                if let Err(e) = b.record_usage(&usage) {
                    if self
                        .handle_budget_exceeded(
                            session_id,
                            &e,
                            &mut current_turn,
                            sender_id,
                            outbound,
                            budget.as_deref(),
                        )
                        .await
                    {
                        // User approved — force-commit the usage.
                        b.force_record_usage(&usage);
                    } else {
                        return current_turn;
                    }
                }
            }

            // Always send StreamEnd after a complete stream round
            send_outbound(
                outbound,
                OutboundEvent::StreamEnd {
                    recipient_id: sender_id.into(),
                },
            )
            .await;

            match stop_reason {
                StopReason::ToolUse => {
                    let invocations_before = current_turn.tool_invocations.len();
                    let messages_before = current_turn.assistant_messages.len();

                    let assistant_message = accumulator.into_message();
                    self.execute_tool_calls(
                        &assistant_message,
                        &mut messages,
                        &mut current_turn,
                        grant,
                        session_id,
                        sender_id,
                        outbound,
                    )
                    .await;

                    self.emit_new_assistant_message(
                        session_id,
                        &current_turn,
                        turn_index,
                        messages_before,
                    )
                    .await;
                    self.emit_new_tool_invocations(
                        session_id,
                        &current_turn,
                        turn_index,
                        invocations_before,
                    )
                    .await;
                }
                StopReason::EndTurn | StopReason::StopSequence => {
                    let msg = accumulator.into_message();
                    self.audit_streaming_injection(session_id, &msg).await;
                    let msg_idx = current_turn.assistant_messages.len();
                    current_turn.assistant_messages.push(msg.clone());
                    current_turn.completed_at = Some(Utc::now());

                    self.emit_event(
                        session_id,
                        ConversationEvent::AssistantMessage {
                            turn_index,
                            message_index: msg_idx,
                            message: msg,
                        },
                    )
                    .await;
                    self.emit_turn_completed(session_id, &current_turn, turn_index)
                        .await;

                    return current_turn;
                }
                StopReason::MaxTokens => {
                    let msg = accumulator.into_message();
                    self.audit_streaming_injection(session_id, &msg).await;
                    send_outbound(
                        outbound,
                        OutboundEvent::Message {
                            text: "[response truncated — max tokens reached]".into(),
                            recipient_id: sender_id.into(),
                        },
                    )
                    .await;
                    let msg_idx = current_turn.assistant_messages.len();
                    current_turn.assistant_messages.push(msg.clone());
                    current_turn.completed_at = Some(Utc::now());

                    self.emit_event(
                        session_id,
                        ConversationEvent::AssistantMessage {
                            turn_index,
                            message_index: msg_idx,
                            message: msg,
                        },
                    )
                    .await;
                    self.emit_turn_completed(session_id, &current_turn, turn_index)
                        .await;

                    return current_turn;
                }
            }
        }

        self.log_max_rounds_exceeded(session_id, &mut current_turn, sender_id, outbound)
            .await;

        self.emit_turn_completed(session_id, &current_turn, turn_index)
            .await;

        current_turn
    }

    /// Consume a provider stream, forwarding text deltas as `StreamChunk` events.
    ///
    /// Returns `Some((accumulator, stop_reason, usage))` on success, `None` if a
    /// mid-stream error occurred (error events are sent to the user and
    /// audit-logged).
    async fn consume_stream(
        mut event_stream: std::pin::Pin<
            Box<dyn futures::Stream<Item = Result<StreamEvent, ProviderError>> + Send>,
        >,
        sender_id: &str,
        session_id: &SessionId,
        audit_sink: Option<&dyn AuditSink>,
        outbound: &mpsc::Sender<OutboundEvent>,
    ) -> Option<(StreamAccumulator, StopReason, TokenUsage)> {
        let mut accumulator = StreamAccumulator::new();

        while let Some(event_result) = event_stream.next().await {
            match event_result {
                Ok(StreamEvent::TextDelta(text)) => {
                    accumulator.push_text_delta(&text);
                    send_outbound(
                        outbound,
                        OutboundEvent::StreamChunk {
                            text,
                            recipient_id: sender_id.into(),
                        },
                    )
                    .await;
                }
                Ok(StreamEvent::ToolUse { id, name, input }) => {
                    accumulator.push_tool_use(id, name, input);
                }
                Ok(StreamEvent::Done { stop_reason, usage }) => {
                    return Some((accumulator, stop_reason, usage));
                }
                Ok(StreamEvent::Error(e)) => {
                    tracing::error!(error = %e, "stream error mid-response");
                    Self::audit_stream_error(
                        audit_sink,
                        session_id,
                        "stream_error",
                        &format!("mid-stream error: {e}"),
                    )
                    .await;
                    send_stream_error(outbound, sender_id, &format!("Stream error: {e}")).await;
                    return None;
                }
                Err(e) => {
                    tracing::error!(error = %e, "provider error during streaming");
                    Self::audit_stream_error(
                        audit_sink,
                        session_id,
                        "stream_provider_error",
                        &format!("provider error during streaming: {e}"),
                    )
                    .await;
                    send_stream_error(outbound, sender_id, &format!("Provider error: {e}")).await;
                    return None;
                }
            }
        }

        // Stream ended without a Done event
        send_stream_error(
            outbound,
            sender_id,
            "Stream ended unexpectedly without completion",
        )
        .await;
        None
    }

    /// Record an audit event for a stream error.
    async fn audit_stream_error(
        audit_sink: Option<&dyn AuditSink>,
        session_id: &SessionId,
        rule: &str,
        context: &str,
    ) {
        if let Some(sink) = audit_sink {
            let event = AuditEventType::PolicyViolation {
                rule: rule.into(),
                context: context.into(),
                severity: Severity::Medium,
            };
            emit_audit(sink, Some(session_id.as_str()), event).await;
        }
    }

    /// Audit-only injection scan for streaming responses.
    ///
    /// Text has already been delivered via `StreamChunk`, so we cannot block it.
    /// We log the detection for forensics.
    async fn audit_streaming_injection(&self, session_id: &SessionId, message: &Message) {
        let response_text = extract_text(message);
        let scanned = ScannedModelResponse::from_raw(&response_text);
        if scanned.injection_detected() {
            self.audit_model_injection(session_id).await;
        }
    }
}

/// Send an outbound event, ignoring channel-closed errors.
async fn send_outbound(outbound: &mpsc::Sender<OutboundEvent>, event: OutboundEvent) {
    let _ = outbound.send(event).await;
}

/// Send a `StreamEnd` followed by an `Error` event — used by `consume_stream`
/// error paths.
async fn send_stream_error(outbound: &mpsc::Sender<OutboundEvent>, sender_id: &str, text: &str) {
    send_outbound(
        outbound,
        OutboundEvent::StreamEnd {
            recipient_id: sender_id.into(),
        },
    )
    .await;
    send_outbound(
        outbound,
        OutboundEvent::Error {
            text: text.into(),
            recipient_id: sender_id.into(),
        },
    )
    .await;
}

/// Serialize and record an audit event via an [`AuditSink`].
///
/// Extracts the serde `"type"` tag from the serialized [`AuditEventType`]
/// for the `event_type` column, then records the full JSON. Errors are
/// logged but never propagated — audit must not block the agent loop.
pub async fn emit_audit(sink: &dyn AuditSink, session_id: Option<&str>, event: AuditEventType) {
    let event_value = match serde_json::to_value(&event) {
        Ok(v) => v,
        Err(e) => {
            tracing::error!(error = %e, "failed to serialize audit event — event lost");
            return;
        }
    };
    let event_type = event_value
        .get("type")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown");
    let event_json = event_value.to_string();
    if let Err(e) = sink.record(session_id, event_type, &event_json).await {
        tracing::error!(error = %e, "audit sink recording failed");
    }
}

/// Join all `ContentBlock::Text` blocks in a message.
fn extract_text(message: &Message) -> String {
    let total_len: usize = message
        .content
        .iter()
        .filter_map(|block| match block {
            ContentBlock::Text { text } => Some(text.len()),
            _ => None,
        })
        .sum();
    let mut result = String::with_capacity(total_len);
    for block in &message.content {
        if let ContentBlock::Text { text } = block {
            result.push_str(text);
        }
    }
    result
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::indexing_slicing)]
mod tests {
    use super::*;
    use freebird_types::config::EditConfig;

    // -- extract_text --

    #[test]
    fn test_extract_text_single_text_block() {
        let msg = Message {
            role: Role::Assistant,
            content: vec![ContentBlock::Text {
                text: "hello".into(),
            }],
            timestamp: Utc::now(),
        };
        assert_eq!(extract_text(&msg), "hello");
    }

    #[test]
    fn test_extract_text_multiple_text_blocks() {
        let msg = Message {
            role: Role::Assistant,
            content: vec![
                ContentBlock::Text {
                    text: "part1".into(),
                },
                ContentBlock::Text {
                    text: "part2".into(),
                },
            ],
            timestamp: Utc::now(),
        };
        assert_eq!(extract_text(&msg), "part1part2");
    }

    #[test]
    fn test_extract_text_skips_non_text_blocks() {
        let msg = Message {
            role: Role::Assistant,
            content: vec![
                ContentBlock::Text {
                    text: "before".into(),
                },
                ContentBlock::ToolUse {
                    id: "1".into(),
                    name: "tool".into(),
                    input: serde_json::json!({}),
                },
                ContentBlock::Text {
                    text: "after".into(),
                },
            ],
            timestamp: Utc::now(),
        };
        assert_eq!(extract_text(&msg), "beforeafter");
    }

    #[test]
    fn test_extract_text_empty_content() {
        let msg = Message {
            role: Role::Assistant,
            content: vec![],
            timestamp: Utc::now(),
        };
        assert_eq!(extract_text(&msg), "");
    }

    #[test]
    fn test_extract_text_no_text_blocks() {
        let msg = Message {
            role: Role::Assistant,
            content: vec![ContentBlock::ToolUse {
                id: "1".into(),
                name: "tool".into(),
                input: serde_json::json!({}),
            }],
            timestamp: Utc::now(),
        };
        assert_eq!(extract_text(&msg), "");
    }

    // -- format_knowledge_context --

    use freebird_traits::id::KnowledgeId;
    use freebird_traits::knowledge::{KnowledgeEntry, KnowledgeKind, KnowledgeSource};
    use std::collections::BTreeSet;

    fn make_match(kind: KnowledgeKind, content: &str, rank: f64) -> KnowledgeMatch {
        KnowledgeMatch {
            entry: KnowledgeEntry {
                id: KnowledgeId::from_string("test-id"),
                kind,
                content: content.to_owned(),
                tags: BTreeSet::new(),
                source: KnowledgeSource::System,
                confidence: 1.0,
                session_id: None,
                created_at: Utc::now(),
                updated_at: Utc::now(),
                access_count: 0,
                last_accessed: None,
            },
            rank,
        }
    }

    fn test_runtime_with_knowledge_config(config: KnowledgeConfig) -> AgentRuntime {
        AgentRuntime::new(
            ProviderRegistry::new(),
            Box::new(NullChannel),
            crate::tool_executor::ToolExecutor::new(
                vec![],
                std::time::Duration::from_secs(30),
                None,
                vec![],
                None,
                None,
                None,
                None,
                InjectionConfig::default(),
                None,
            )
            .unwrap(),
            None,
            Arc::new(NullMemory),
            None,
            config,
            RuntimeConfig {
                default_model: freebird_traits::id::ModelId::from("m"),
                default_provider: freebird_traits::id::ProviderId::from("p"),
                system_prompt: None,
                max_output_tokens: 1024,
                max_tool_rounds: 1,
                temperature: None,
                max_turns_per_session: 10,
                drain_timeout_secs: 1,
                session: freebird_types::config::SessionConfig::default(),
                context: ContextConfig::default(),
            },
            ToolsConfig {
                sandbox_root: std::env::temp_dir(),
                default_timeout_secs: 30,
                allowed_directories: vec![],
                allowed_shell_commands: vec![],
                max_shell_output_bytes: 1_048_576,
                edit: EditConfig::default(),
                git_timeout_secs: 5,
            },
            BudgetConfig::default(),
            24, // default_session_ttl_hours
            None,
            None,
            None,
            SummarizationConfig::default(),
        )
    }

    /// Minimal channel for unit tests — never used, just satisfies the type.
    struct NullChannel;
    #[async_trait::async_trait]
    impl freebird_traits::channel::Channel for NullChannel {
        fn info(&self) -> &freebird_traits::channel::ChannelInfo {
            static INFO: std::sync::OnceLock<freebird_traits::channel::ChannelInfo> =
                std::sync::OnceLock::new();
            INFO.get_or_init(|| freebird_traits::channel::ChannelInfo {
                id: freebird_traits::id::ChannelId::from("null"),
                display_name: "null".into(),
                features: BTreeSet::new(),
                auth: freebird_traits::channel::AuthRequirement::None,
            })
        }
        async fn start(
            &self,
        ) -> Result<freebird_traits::channel::ChannelHandle, freebird_traits::channel::ChannelError>
        {
            Err(freebird_traits::channel::ChannelError::StartupFailed {
                channel: "null".into(),
                reason: "null channel".into(),
            })
        }
        async fn stop(&self) -> Result<(), freebird_traits::channel::ChannelError> {
            Ok(())
        }
    }

    /// Minimal memory for unit tests.
    struct NullMemory;
    #[async_trait::async_trait]
    impl freebird_traits::memory::Memory for NullMemory {
        async fn load(
            &self,
            _: &SessionId,
        ) -> Result<Option<Conversation>, freebird_traits::memory::MemoryError> {
            Ok(None)
        }
        async fn save(&self, _: &Conversation) -> Result<(), freebird_traits::memory::MemoryError> {
            Ok(())
        }
        async fn list_sessions(
            &self,
            _: usize,
        ) -> Result<
            Vec<freebird_traits::memory::SessionSummary>,
            freebird_traits::memory::MemoryError,
        > {
            Ok(vec![])
        }
        async fn delete(&self, _: &SessionId) -> Result<(), freebird_traits::memory::MemoryError> {
            Ok(())
        }
        async fn search(
            &self,
            _: &str,
            _: usize,
        ) -> Result<
            Vec<freebird_traits::memory::SessionSummary>,
            freebird_traits::memory::MemoryError,
        > {
            Ok(vec![])
        }
    }

    #[test]
    fn test_format_knowledge_context_empty_returns_none() {
        let rt = test_runtime_with_knowledge_config(KnowledgeConfig::default());
        assert!(rt.format_knowledge_context(&[]).is_none());
    }

    #[test]
    fn test_format_knowledge_context_formats_entries() {
        let rt = test_runtime_with_knowledge_config(KnowledgeConfig::default());
        let matches = vec![
            make_match(KnowledgeKind::LearnedPattern, "Use pattern X", -2.0),
            make_match(KnowledgeKind::ErrorResolution, "Fix Y with Z", -1.5),
        ];

        let result = rt.format_knowledge_context(&matches).unwrap();
        assert!(result.starts_with("[RELEVANT CONTEXT]\n"));
        assert!(result.contains("[LearnedPattern] Use pattern X"));
        assert!(result.contains("[ErrorResolution] Fix Y with Z"));
    }

    #[test]
    fn test_format_knowledge_context_respects_token_budget() {
        let config = KnowledgeConfig {
            max_context_tokens: 10, // ~40 chars budget
            ..KnowledgeConfig::default()
        };
        let rt = test_runtime_with_knowledge_config(config);

        let matches = vec![
            make_match(KnowledgeKind::LearnedPattern, "short", -2.0),
            make_match(
                KnowledgeKind::ErrorResolution,
                "this is a very long entry that should be truncated or excluded",
                -1.5,
            ),
        ];

        let result = rt.format_knowledge_context(&matches).unwrap();
        // Should have the header and at most the first entry fully.
        // The second entry should be truncated or excluded due to budget.
        assert!(result.starts_with("[RELEVANT CONTEXT]\n"));
        assert!(result.len() <= 60); // 10 tokens * 4 chars + some slack for the first entry
    }

    #[tokio::test]
    async fn test_retrieve_knowledge_context_disabled() {
        let config = KnowledgeConfig {
            auto_retrieve: false,
            ..KnowledgeConfig::default()
        };
        let rt = test_runtime_with_knowledge_config(config);
        let result = rt.retrieve_knowledge_context("hello").await;
        assert!(result.is_empty());
    }

    #[tokio::test]
    async fn test_retrieve_knowledge_context_no_store() {
        // Default test runtime has no knowledge store (None)
        let rt = test_runtime_with_knowledge_config(KnowledgeConfig::default());
        let result = rt.retrieve_knowledge_context("hello").await;
        assert!(result.is_empty());
    }
}
