//! `AgentRuntime` — the core agentic loop.
//!
//! Wires together channels, providers, tools, memory, and sessions into
//! a single event loop. Starts the channel, consumes inbound events,
//! routes them to handlers, and shuts down gracefully when the
//! cancellation token fires or the inbound stream closes.

use std::pin::Pin;

use chrono::Utc;
use freebird_security::audit::{AuditEventType, AuditLogger, InjectionSource};
use freebird_security::capability::CapabilityGrant;
use freebird_security::consent::ConsentRequest;
use freebird_security::error::Severity;
use freebird_security::injection;
use freebird_security::safe_types::{SafeMessage, ScannedModelResponse};
use freebird_security::taint::Tainted;
use freebird_traits::channel::{
    Channel, ChannelError, ChannelFeature, InboundEvent, OutboundEvent,
};
use freebird_traits::id::SessionId;
use freebird_traits::memory::{Conversation, Memory, MemoryError, ToolInvocation, Turn};
use freebird_traits::provider::{
    CompletionRequest, CompletionResponse, ContentBlock, Message, ProviderError, Role, StopReason,
    StreamEvent, TokenUsage, ToolDefinition,
};
use freebird_traits::tool::{Capability, ToolOutcome};
use freebird_types::config::{RuntimeConfig, ToolsConfig};
use futures::StreamExt;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::history::conversation_to_messages;
use crate::registry::ProviderRegistry;
use crate::stream::StreamAccumulator;

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
    /// Receives consent requests from the `ToolExecutor`'s `ConsentGate`.
    /// Consumed by the third arm of the `run()` select loop.
    consent_rx: Option<tokio::sync::mpsc::Receiver<ConsentRequest>>,
    memory: Box<dyn Memory>,
    config: RuntimeConfig,
    tools_config: ToolsConfig,
    audit: Option<AuditLogger>,
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
        consent_rx: Option<tokio::sync::mpsc::Receiver<ConsentRequest>>,
        memory: Box<dyn Memory>,
        config: RuntimeConfig,
        tools_config: ToolsConfig,
        audit: Option<AuditLogger>,
    ) -> Self {
        Self {
            provider_registry,
            channel,
            tool_executor,
            consent_rx,
            memory,
            config,
            tools_config,
            audit,
            sessions: SessionManager::new(),
        }
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

        // Fan-out inbound events into a splitter task and optional consent
        // bridge task. See `spawn_inbound_splitter` and
        // `spawn_consent_forwarder` for details.
        let (mut main_rx, splitter_task, consent_task) =
            self.spawn_consent_bridge(handle.inbound, &outbound, &cancel);

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
        if let Some(task) = consent_task {
            task.abort();
        }

        self.channel.stop().await?;
        Ok(())
    }

    /// Spawn background tasks for consent bridge plumbing.
    ///
    /// Returns a receiver for non-consent inbound events, the splitter task
    /// handle, and an optional consent-forwarder task handle.
    ///
    /// **Splitter task**: reads from the raw inbound stream and routes
    /// `ConsentResponse` events directly to the consent gate's
    /// [`ConsentResponder`], bypassing the main event loop. All other
    /// events are forwarded to the returned `main_rx`. This is necessary
    /// because `handle_event()` may block on `check_consent()` (awaiting
    /// user approval via oneshot), and the `ConsentResponse` that unblocks
    /// it also arrives as an `InboundEvent`.
    ///
    /// **Consent-forwarder task**: reads `ConsentRequest`s from the gate's
    /// mpsc channel and forwards them as `OutboundEvent::ConsentRequest` to
    /// the user's channel.
    fn spawn_consent_bridge(
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
        let consent_responder = self.tool_executor.consent_responder();
        let has_consent_gate = consent_responder.is_some();

        let splitter_cancel = cancel.clone();
        let splitter_task = tokio::spawn({
            let mut inbound = inbound;
            async move {
                loop {
                    tokio::select! {
                        () = splitter_cancel.cancelled() => break,
                        event = inbound.next() => {
                            match event {
                                Some(InboundEvent::ConsentResponse {
                                    request_id, approved, reason, sender_id,
                                }) if has_consent_gate => {
                                    if let Some(ref resp) = consent_responder {
                                        let response = if approved {
                                            freebird_security::consent::ConsentResponse::Approved
                                        } else {
                                            freebird_security::consent::ConsentResponse::Denied { reason }
                                        };
                                        if !resp.respond(&request_id, response).await {
                                            tracing::warn!(
                                                %request_id, %sender_id,
                                                "consent response for unknown or expired request"
                                            );
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

        let consent_task = self.consent_rx.take().map(|mut consent_rx| {
            let consent_cancel = cancel.clone();
            let consent_outbound = outbound.clone();
            tokio::spawn(async move {
                loop {
                    tokio::select! {
                        () = consent_cancel.cancelled() => break,
                        req = consent_rx.recv() => {
                            match req {
                                Some(req) => {
                                    let event = OutboundEvent::ConsentRequest {
                                        request_id: req.id,
                                        tool_name: req.tool_name,
                                        description: req.description,
                                        risk_level: format!("{:?}", req.risk_level),
                                        action_summary: req.action_summary,
                                        expires_at: req.expires_at.to_rfc3339(),
                                        recipient_id: req.sender_id,
                                    };
                                    let _ = consent_outbound.send(event).await;
                                }
                                None => break,
                            }
                        }
                    }
                }
            })
        });

        (main_rx, splitter_task, consent_task)
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
                LoopAction::Continue
            }
            InboundEvent::Disconnected { sender_id } => {
                tracing::info!(%sender_id, "user disconnected");
                LoopAction::Continue
            }
            InboundEvent::ConsentResponse {
                request_id,
                sender_id,
                ..
            } => {
                // When a consent gate is configured, ConsentResponse events
                // are handled by the splitter task (which runs concurrently
                // and can deliver responses while handle_event is blocked).
                // This arm only fires when no consent gate is configured
                // (the splitter passes them through to the main loop).
                tracing::warn!(
                    %request_id,
                    %sender_id,
                    "consent response received but no consent gate is configured"
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
                let _ = outbound
                    .send(OutboundEvent::Message {
                        text: format!("New session started: {session_id}"),
                        recipient_id: sender_id.into(),
                    })
                    .await;
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

        // 3. Check streaming support and dispatch to appropriate loop
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
                outbound,
            )
            .await
        } else {
            self.run_agentic_loop(
                &safe_message,
                sender_id,
                session_id,
                &conversation,
                outbound,
                None,
            )
            .await
        };

        // 4. Persist conversation
        conversation.turns.push(current_turn);
        conversation.updated_at = Utc::now();

        if let Err(e) = self.memory.save(&conversation).await {
            tracing::error!(error = %e, "failed to persist conversation");
        }
    }

    /// Validate raw input through taint tracking. Returns `None` if rejected.
    async fn validate_input(
        &self,
        raw_text: &str,
        sender_id: &str,
        session_id: &SessionId,
        outbound: &mpsc::Sender<OutboundEvent>,
    ) -> Option<SafeMessage> {
        let tainted = Tainted::new(raw_text);
        match SafeMessage::from_tainted(&tainted) {
            Ok(msg) => Some(msg),
            Err(e) => {
                self.audit(
                    session_id,
                    AuditEventType::InjectionDetected {
                        pattern: format!("{e}"),
                        source: InjectionSource::UserInput,
                        severity: Severity::High,
                    },
                )
                .await;
                let _ = outbound
                    .send(OutboundEvent::Error {
                        text: "Message rejected by input validation.".into(),
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
            Ok(None) => Some(Conversation {
                session_id: session_id.clone(),
                system_prompt: self.config.system_prompt.clone(),
                turns: Vec::new(),
                created_at: Utc::now(),
                updated_at: Utc::now(),
                model_id: self.config.default_model.clone(),
                provider_id: self.config.default_provider.clone(),
            }),
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

    /// Build the initial state for an agentic loop: user message, turn, messages, tool defs.
    fn prepare_agentic_loop(
        &self,
        safe_message: &SafeMessage,
        conversation: &Conversation,
    ) -> (Vec<Message>, Turn, Vec<ToolDefinition>) {
        let user_message = Message {
            role: Role::User,
            content: vec![ContentBlock::Text {
                text: safe_message.as_str().to_owned(),
            }],
            timestamp: Utc::now(),
        };

        let tool_definitions = self.tool_executor.tool_definitions();

        let mut messages = conversation_to_messages(conversation);

        // CLAUDE.md §14: scan loaded conversation history for context injection
        // before sending to provider. Filter out any messages containing injection
        // patterns to prevent memory poisoning attacks.
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

    /// Run the agentic loop: call provider, handle tool use, deliver response.
    ///
    /// When `initial_request` is `Some`, the first loop iteration uses it directly
    /// instead of building a fresh `CompletionRequest`. This supports the streaming
    /// fallback path where the request has already been constructed.
    async fn run_agentic_loop(
        &self,
        safe_message: &SafeMessage,
        sender_id: &str,
        session_id: &SessionId,
        conversation: &Conversation,
        outbound: &mpsc::Sender<OutboundEvent>,
        initial_request: Option<CompletionRequest>,
    ) -> Turn {
        let (mut messages, mut current_turn, tool_definitions) =
            self.prepare_agentic_loop(safe_message, conversation);

        let mut pending_request = initial_request;

        for _round in 0..self.config.max_tool_rounds {
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

            match response.stop_reason {
                StopReason::ToolUse => {
                    self.handle_tool_use_round(
                        &response,
                        &mut messages,
                        &mut current_turn,
                        session_id,
                        sender_id,
                        outbound,
                    )
                    .await;
                }
                StopReason::EndTurn | StopReason::StopSequence => {
                    self.deliver_final_response(
                        &response,
                        sender_id,
                        session_id,
                        &mut current_turn,
                        outbound,
                    )
                    .await;
                    return current_turn;
                }
                StopReason::MaxTokens => {
                    self.deliver_truncated_response(
                        &response,
                        sender_id,
                        session_id,
                        &mut current_turn,
                        outbound,
                    )
                    .await;
                    return current_turn;
                }
            }
        }

        self.log_max_rounds_exceeded(session_id, &mut current_turn, sender_id, outbound)
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
        let system_prompt =
            self.build_effective_system_prompt(base, conversation.model_id.as_str());

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
    fn build_effective_system_prompt(&self, base: &str, model_id: &str) -> String {
        use std::fmt::Write;

        let mut prompt = base.to_owned();
        let _ = write!(prompt, "\n\nYou are running on model: {model_id}");

        if self.tool_executor.tool_count() == 0 {
            return prompt;
        }

        // List available tools by name and description.
        prompt.push_str("\n\nYou have the following tools available:\n");
        for def in &self.tool_executor.tool_definitions() {
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
    async fn handle_tool_use_round(
        &self,
        response: &CompletionResponse,
        messages: &mut Vec<Message>,
        current_turn: &mut Turn,
        session_id: &SessionId,
        sender_id: &str,
        outbound: &mpsc::Sender<OutboundEvent>,
    ) {
        self.execute_tool_calls(
            &response.message,
            messages,
            current_turn,
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
    async fn execute_tool_calls(
        &self,
        assistant_message: &Message,
        messages: &mut Vec<Message>,
        current_turn: &mut Turn,
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

        // TODO(#27): Replace with per-session capability grants derived from session auth.
        let grant = match CapabilityGrant::new(
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
            None,
        ) {
            Ok(g) => g,
            Err(e) => {
                tracing::error!(error = %e, "cannot create capability grant — skipping tool execution");
                send_outbound(
                    outbound,
                    OutboundEvent::Error {
                        text: "Internal error: sandbox root is not accessible".into(),
                        recipient_id: sender_id.into(),
                    },
                )
                .await;
                return;
            }
        };

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
                .execute(&tool_name, input.clone(), &grant, session_id, sender_id)
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

    /// Record an audit event if an audit logger is configured.
    ///
    /// No-op when `self.audit` is `None`. Errors from the audit logger
    /// are intentionally discarded — audit failures must never block the
    /// agent loop.
    async fn audit(&self, session_id: &SessionId, event: AuditEventType) {
        if let Some(audit) = &self.audit {
            let _ = audit.record(session_id.as_str(), event).await;
        }
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
    async fn run_agentic_loop_streaming(
        &self,
        safe_message: &SafeMessage,
        sender_id: &str,
        session_id: &SessionId,
        conversation: &Conversation,
        outbound: &mpsc::Sender<OutboundEvent>,
    ) -> Turn {
        let (mut messages, mut current_turn, tool_definitions) =
            self.prepare_agentic_loop(safe_message, conversation);

        for _round in 0..self.config.max_tool_rounds {
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
                            outbound,
                            Some(request),
                        )
                        .await;
                }
            };

            // TODO(#31): wire `usage` to TokenBudget for per-request enforcement
            let Some((accumulator, stop_reason, _usage)) = Self::consume_stream(
                event_stream,
                sender_id,
                session_id,
                self.audit.as_ref(),
                outbound,
            )
            .await
            else {
                return current_turn;
            };

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
                    let assistant_message = accumulator.into_message();
                    self.execute_tool_calls(
                        &assistant_message,
                        &mut messages,
                        &mut current_turn,
                        session_id,
                        sender_id,
                        outbound,
                    )
                    .await;
                }
                StopReason::EndTurn | StopReason::StopSequence => {
                    let msg = accumulator.into_message();
                    self.audit_streaming_injection(session_id, &msg).await;
                    current_turn.assistant_messages.push(msg);
                    current_turn.completed_at = Some(Utc::now());
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
                    current_turn.assistant_messages.push(msg);
                    current_turn.completed_at = Some(Utc::now());
                    return current_turn;
                }
            }
        }

        self.log_max_rounds_exceeded(session_id, &mut current_turn, sender_id, outbound)
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
        audit: Option<&AuditLogger>,
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
                        audit,
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
                        audit,
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
        audit: Option<&AuditLogger>,
        session_id: &SessionId,
        rule: &str,
        context: &str,
    ) {
        if let Some(a) = audit {
            let _ = a
                .record(
                    session_id.as_str(),
                    AuditEventType::PolicyViolation {
                        rule: rule.into(),
                        context: context.into(),
                        severity: Severity::Medium,
                    },
                )
                .await;
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
}
