//! `AgentRuntime` — the core agentic loop.
//!
//! Wires together channels, providers, tools, memory, and sessions into
//! a single event loop. Starts the channel, consumes inbound events,
//! routes them to handlers, and shuts down gracefully when the
//! cancellation token fires or the inbound stream closes.

use std::time::Duration;

use chrono::Utc;
use freebird_security::audit::{
    AuditEventType, AuditLogger, CapabilityCheckResult, InjectionSource,
};
use freebird_security::error::Severity;
use freebird_security::safe_types::{SafeMessage, ScannedModelResponse, ScannedToolOutput};
use freebird_security::taint::Tainted;
use freebird_traits::channel::{Channel, ChannelError, InboundEvent, OutboundEvent};
use freebird_traits::id::{ModelId, SessionId};
use freebird_traits::memory::{Conversation, Memory, MemoryError, ToolInvocation, Turn};
use freebird_traits::provider::{
    CompletionRequest, CompletionResponse, ContentBlock, Message, ProviderError, Role, StopReason,
    ToolDefinition,
};
use freebird_traits::tool::{Tool, ToolContext, ToolOutput};
use freebird_types::config::{RuntimeConfig, ToolsConfig};
use futures::StreamExt;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::registry::ProviderRegistry;
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
    tools: Vec<Box<dyn Tool>>,
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
    pub fn new(
        provider_registry: ProviderRegistry,
        channel: Box<dyn Channel>,
        tools: Vec<Box<dyn Tool>>,
        memory: Box<dyn Memory>,
        config: RuntimeConfig,
        tools_config: ToolsConfig,
        audit: Option<AuditLogger>,
    ) -> Self {
        Self {
            provider_registry,
            channel,
            tools,
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
    pub async fn run(&self, cancel: CancellationToken) -> Result<(), RuntimeError> {
        let handle = self.channel.start().await?;
        let mut inbound = handle.inbound;
        let outbound = handle.outbound;

        tracing::info!("agent runtime started, waiting for messages");

        loop {
            tokio::select! {
                () = cancel.cancelled() => {
                    tracing::info!("shutdown signal received, stopping runtime");
                    break;
                }
                event = inbound.next() => {
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

        self.channel.stop().await?;
        Ok(())
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
                self.handle_command(&name, &args, &sender_id, outbound)
                    .await
            }
            InboundEvent::Connected { sender_id } => {
                tracing::info!(%sender_id, "user connected");
                LoopAction::Continue
            }
            InboundEvent::Disconnected { sender_id } => {
                tracing::info!(%sender_id, "user disconnected");
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
        // 1. Taint input and validate
        let Some(safe_message) = self
            .validate_input(&raw_text, &sender_id, &session_id, outbound)
            .await
        else {
            return;
        };

        // 2. Load or create conversation
        let Some(mut conversation) = self
            .load_or_create_conversation(&session_id, &sender_id, outbound)
            .await
        else {
            return;
        };

        // 3. Build user message and start the agentic loop
        let current_turn = self
            .run_agentic_loop(
                &safe_message,
                &sender_id,
                &session_id,
                &conversation,
                outbound,
            )
            .await;

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
                if let Some(audit) = &self.audit {
                    let _ = audit
                        .record(
                            session_id.as_str(),
                            AuditEventType::InjectionDetected {
                                pattern: format!("{e}"),
                                source: InjectionSource::UserInput,
                                severity: Severity::High,
                            },
                        )
                        .await;
                }
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

    /// Run the agentic loop: call provider, handle tool use, deliver response.
    async fn run_agentic_loop(
        &self,
        safe_message: &SafeMessage,
        sender_id: &str,
        session_id: &SessionId,
        conversation: &Conversation,
        outbound: &mpsc::Sender<OutboundEvent>,
    ) -> Turn {
        let user_message = Message {
            role: Role::User,
            content: vec![ContentBlock::Text {
                text: safe_message.as_str().to_owned(),
            }],
            timestamp: Utc::now(),
        };

        let tool_definitions: Vec<ToolDefinition> =
            self.tools.iter().map(|t| t.to_definition()).collect();

        let mut messages = conversation_to_messages(conversation);
        messages.push(user_message.clone());

        let mut current_turn = Turn {
            user_message,
            assistant_response: None,
            tool_invocations: Vec::new(),
            started_at: Utc::now(),
            completed_at: None,
        };

        for _round in 0..self.config.max_tool_rounds {
            let request = self.build_completion_request(conversation, &messages, &tool_definitions);

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

        // Max rounds exceeded — this is a policy-relevant event
        if let Some(audit) = &self.audit {
            let _ = audit
                .record(
                    session_id.as_str(),
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
        }
        current_turn.completed_at = Some(Utc::now());
        let _ = outbound
            .send(OutboundEvent::Error {
                text: "Maximum tool rounds exceeded. Stopping.".into(),
                recipient_id: sender_id.into(),
            })
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
        CompletionRequest {
            model: ModelId::from(conversation.model_id.as_str()),
            system_prompt: conversation.system_prompt.clone(),
            messages: messages.to_vec(),
            tools: tool_definitions.to_vec(),
            max_tokens: self.config.max_output_tokens,
            temperature: self.config.temperature,
            stop_sequences: Vec::new(),
        }
    }

    /// Execute tool calls from a `ToolUse` response and append results to the message list.
    async fn handle_tool_use_round(
        &self,
        response: &CompletionResponse,
        messages: &mut Vec<Message>,
        current_turn: &mut Turn,
        session_id: &SessionId,
    ) {
        // Extract tool_use blocks from the response
        let tool_uses: Vec<(String, String, serde_json::Value)> = response
            .message
            .content
            .iter()
            .filter_map(|block| match block {
                ContentBlock::ToolUse { id, name, input } => {
                    Some((id.clone(), name.clone(), input.clone()))
                }
                _ => None,
            })
            .collect();

        // Add assistant message with tool_use blocks to conversation
        messages.push(response.message.clone());

        // Execute each tool and collect results
        let mut tool_results = Vec::new();
        for (tool_use_id, tool_name, input) in tool_uses {
            let start = std::time::Instant::now();

            let output = self.execute_tool(&tool_name, &input, session_id).await;

            let duration_ms = u64::try_from(start.elapsed().as_millis()).unwrap_or(u64::MAX);

            // Scan tool output for injection — BLOCK if detected
            let (final_content, is_error) = match ScannedToolOutput::from_raw(&output.content) {
                Ok(scanned) => (scanned.as_str().to_owned(), output.is_error),
                Err(e) => {
                    tracing::warn!(tool = %tool_name, "injection detected in tool output, blocking");
                    if let Some(audit) = &self.audit {
                        let _ = audit
                            .record(
                                session_id.as_str(),
                                AuditEventType::InjectionDetected {
                                    pattern: format!("{e}"),
                                    source: InjectionSource::ToolOutput,
                                    severity: Severity::High,
                                },
                            )
                            .await;
                    }
                    (
                        "Tool output blocked: potential prompt injection detected".to_owned(),
                        true,
                    )
                }
            };

            current_turn.tool_invocations.push(ToolInvocation {
                tool_use_id: tool_use_id.clone(),
                tool_name,
                input,
                output: Some(final_content.clone()),
                is_error,
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
            current_turn.assistant_response = Some(response.message.clone());
            current_turn.completed_at = Some(Utc::now());
            return;
        }

        // Scan model output for injection — BLOCK if detected
        match ScannedModelResponse::from_raw(&response_text) {
            Ok(scanned) => {
                let _ = outbound
                    .send(OutboundEvent::Message {
                        text: scanned.as_str().to_owned(),
                        recipient_id: sender_id.into(),
                    })
                    .await;

                current_turn.assistant_response = Some(response.message.clone());
                current_turn.completed_at = Some(Utc::now());
            }
            Err(e) => {
                tracing::warn!("injection detected in model output, blocking delivery");
                if let Some(audit) = &self.audit {
                    let _ = audit
                        .record(
                            session_id.as_str(),
                            AuditEventType::InjectionDetected {
                                pattern: format!("{e}"),
                                source: InjectionSource::ModelResponse,
                                severity: Severity::High,
                            },
                        )
                        .await;
                }

                let _ = outbound
                    .send(OutboundEvent::Error {
                        text:
                            "Response blocked: potential prompt injection detected in model output."
                                .into(),
                        recipient_id: sender_id.into(),
                    })
                    .await;

                // Do NOT persist the tainted response — prevents memory poisoning
                current_turn.completed_at = Some(Utc::now());
            }
        }
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
        match ScannedModelResponse::from_raw(&partial_text) {
            Ok(scanned) => {
                let _ = outbound
                    .send(OutboundEvent::Message {
                        text: format!(
                            "{}\n\n[response truncated — max tokens reached]",
                            scanned.as_str()
                        ),
                        recipient_id: sender_id.into(),
                    })
                    .await;

                current_turn.assistant_response = Some(response.message.clone());
                current_turn.completed_at = Some(Utc::now());
            }
            Err(e) => {
                tracing::warn!("injection detected in truncated model output, blocking delivery");
                if let Some(audit) = &self.audit {
                    let _ = audit
                        .record(
                            session_id.as_str(),
                            AuditEventType::InjectionDetected {
                                pattern: format!("{e}"),
                                source: InjectionSource::ModelResponse,
                                severity: Severity::High,
                            },
                        )
                        .await;
                }

                let _ = outbound
                    .send(OutboundEvent::Error {
                        text:
                            "Response blocked: potential prompt injection detected in model output."
                                .into(),
                        recipient_id: sender_id.into(),
                    })
                    .await;

                current_turn.completed_at = Some(Utc::now());
            }
        }
    }

    /// Execute a tool by name, with timeout. Returns `ToolOutput` unconditionally.
    async fn execute_tool(
        &self,
        tool_name: &str,
        input: &serde_json::Value,
        session_id: &SessionId,
    ) -> ToolOutput {
        let Some(tool) = self.find_tool(tool_name) else {
            if let Some(audit) = &self.audit {
                let _ = audit
                    .record(
                        session_id.as_str(),
                        AuditEventType::ToolInvocation {
                            tool_name: tool_name.into(),
                            capability_check: CapabilityCheckResult::Denied {
                                reason: "tool not found".into(),
                            },
                        },
                    )
                    .await;
            }
            return ToolOutput {
                content: format!("Error: tool `{tool_name}` not found"),
                is_error: true,
                metadata: None,
            };
        };

        if let Some(audit) = &self.audit {
            let _ = audit
                .record(
                    session_id.as_str(),
                    AuditEventType::ToolInvocation {
                        tool_name: tool_name.into(),
                        capability_check: CapabilityCheckResult::Granted,
                    },
                )
                .await;
        }

        let context = ToolContext {
            session_id,
            sandbox_root: &self.tools_config.sandbox_root,
            // TODO(#27): Use per-session capability grants instead of empty slice
            granted_capabilities: &[],
        };

        let timeout = Duration::from_secs(self.tools_config.default_timeout_secs);

        match tokio::time::timeout(timeout, tool.execute(input.clone(), &context)).await {
            Ok(Ok(output)) => output,
            Ok(Err(e)) => ToolOutput {
                content: format!("Error: {e}"),
                is_error: true,
                metadata: None,
            },
            Err(_) => ToolOutput {
                content: format!("Error: tool `{tool_name}` timed out"),
                is_error: true,
                metadata: None,
            },
        }
    }

    /// Look up a tool by name.
    fn find_tool(&self, name: &str) -> Option<&dyn Tool> {
        self.tools
            .iter()
            .find(|t| t.info().name == name)
            .map(AsRef::as_ref)
    }
}

/// Join all `ContentBlock::Text` blocks in a message.
fn extract_text(message: &Message) -> String {
    let mut result = String::new();
    for block in &message.content {
        if let ContentBlock::Text { text } = block {
            result.push_str(text);
        }
    }
    result
}

/// Reconstruct a flat message list from conversation turns.
///
/// Inline helper — will be replaced by a proper abstraction in #18.
fn conversation_to_messages(conversation: &Conversation) -> Vec<Message> {
    // Each turn produces at least 2 messages (user + assistant), possibly 3 (+ tool results)
    let mut messages = Vec::with_capacity(conversation.turns.len() * 2);

    for turn in &conversation.turns {
        messages.push(turn.user_message.clone());

        if let Some(ref response) = turn.assistant_response {
            messages.push(response.clone());

            // If the response contained tool_use blocks, also add tool results
            let has_tool_use = response
                .content
                .iter()
                .any(|b| matches!(b, ContentBlock::ToolUse { .. }));

            if has_tool_use {
                let tool_results: Vec<ContentBlock> = turn
                    .tool_invocations
                    .iter()
                    .map(|inv| ContentBlock::ToolResult {
                        tool_use_id: inv.tool_use_id.clone(),
                        content: inv.output.clone().unwrap_or_default(),
                        is_error: inv.is_error,
                    })
                    .collect();

                if !tool_results.is_empty() {
                    messages.push(Message {
                        role: Role::User,
                        content: tool_results,
                        timestamp: turn.started_at,
                    });
                }
            }
        }
    }

    messages
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::indexing_slicing)]
mod tests {
    use super::*;
    use freebird_traits::memory::ToolInvocation;

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

    // -- conversation_to_messages --

    #[test]
    fn test_conversation_to_messages_empty() {
        let conv = Conversation {
            session_id: SessionId::from("s1"),
            system_prompt: None,
            turns: vec![],
            created_at: Utc::now(),
            updated_at: Utc::now(),
            model_id: "m".into(),
            provider_id: "p".into(),
        };
        let msgs = conversation_to_messages(&conv);
        assert!(msgs.is_empty());
    }

    #[test]
    fn test_conversation_to_messages_simple_turn() {
        let conv = Conversation {
            session_id: SessionId::from("s1"),
            system_prompt: None,
            turns: vec![Turn {
                user_message: Message {
                    role: Role::User,
                    content: vec![ContentBlock::Text { text: "hi".into() }],
                    timestamp: Utc::now(),
                },
                assistant_response: Some(Message {
                    role: Role::Assistant,
                    content: vec![ContentBlock::Text {
                        text: "hello".into(),
                    }],
                    timestamp: Utc::now(),
                }),
                tool_invocations: vec![],
                started_at: Utc::now(),
                completed_at: Some(Utc::now()),
            }],
            created_at: Utc::now(),
            updated_at: Utc::now(),
            model_id: "m".into(),
            provider_id: "p".into(),
        };
        let msgs = conversation_to_messages(&conv);
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0].role, Role::User);
        assert_eq!(msgs[1].role, Role::Assistant);
    }

    #[test]
    fn test_conversation_to_messages_tool_turn() {
        let conv = Conversation {
            session_id: SessionId::from("s1"),
            system_prompt: None,
            turns: vec![Turn {
                user_message: Message {
                    role: Role::User,
                    content: vec![ContentBlock::Text {
                        text: "read file".into(),
                    }],
                    timestamp: Utc::now(),
                },
                assistant_response: Some(Message {
                    role: Role::Assistant,
                    content: vec![ContentBlock::ToolUse {
                        id: "call-1".into(),
                        name: "read_file".into(),
                        input: serde_json::json!({}),
                    }],
                    timestamp: Utc::now(),
                }),
                tool_invocations: vec![ToolInvocation {
                    tool_use_id: "call-1".into(),
                    tool_name: "read_file".into(),
                    input: serde_json::json!({}),
                    output: Some("file content".into()),
                    is_error: false,
                    duration_ms: Some(10),
                }],
                started_at: Utc::now(),
                completed_at: Some(Utc::now()),
            }],
            created_at: Utc::now(),
            updated_at: Utc::now(),
            model_id: "m".into(),
            provider_id: "p".into(),
        };
        let msgs = conversation_to_messages(&conv);
        // User message + assistant tool_use + user tool_result
        assert_eq!(msgs.len(), 3);
        assert_eq!(msgs[0].role, Role::User);
        assert_eq!(msgs[1].role, Role::Assistant);
        assert_eq!(msgs[2].role, Role::User);
        // Verify the tool result content
        assert!(matches!(
            &msgs[2].content[0],
            ContentBlock::ToolResult {
                tool_use_id,
                content,
                is_error,
            } if tool_use_id == "call-1" && content == "file content" && !is_error
        ));
    }

    #[test]
    fn test_conversation_to_messages_incomplete_turn() {
        let conv = Conversation {
            session_id: SessionId::from("s1"),
            system_prompt: None,
            turns: vec![Turn {
                user_message: Message {
                    role: Role::User,
                    content: vec![ContentBlock::Text { text: "hi".into() }],
                    timestamp: Utc::now(),
                },
                assistant_response: None, // no response yet
                tool_invocations: vec![],
                started_at: Utc::now(),
                completed_at: None,
            }],
            created_at: Utc::now(),
            updated_at: Utc::now(),
            model_id: "m".into(),
            provider_id: "p".into(),
        };
        let msgs = conversation_to_messages(&conv);
        // Only the user message
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0].role, Role::User);
    }

    #[test]
    fn test_conversation_to_messages_multi_turn() {
        let conv = Conversation {
            session_id: SessionId::from("s1"),
            system_prompt: None,
            turns: vec![
                Turn {
                    user_message: Message {
                        role: Role::User,
                        content: vec![ContentBlock::Text {
                            text: "first".into(),
                        }],
                        timestamp: Utc::now(),
                    },
                    assistant_response: Some(Message {
                        role: Role::Assistant,
                        content: vec![ContentBlock::Text {
                            text: "response1".into(),
                        }],
                        timestamp: Utc::now(),
                    }),
                    tool_invocations: vec![],
                    started_at: Utc::now(),
                    completed_at: Some(Utc::now()),
                },
                Turn {
                    user_message: Message {
                        role: Role::User,
                        content: vec![ContentBlock::Text {
                            text: "second".into(),
                        }],
                        timestamp: Utc::now(),
                    },
                    assistant_response: Some(Message {
                        role: Role::Assistant,
                        content: vec![ContentBlock::Text {
                            text: "response2".into(),
                        }],
                        timestamp: Utc::now(),
                    }),
                    tool_invocations: vec![],
                    started_at: Utc::now(),
                    completed_at: Some(Utc::now()),
                },
            ],
            created_at: Utc::now(),
            updated_at: Utc::now(),
            model_id: "m".into(),
            provider_id: "p".into(),
        };
        let msgs = conversation_to_messages(&conv);
        assert_eq!(msgs.len(), 4);
        assert_eq!(msgs[0].role, Role::User);
        assert_eq!(msgs[1].role, Role::Assistant);
        assert_eq!(msgs[2].role, Role::User);
        assert_eq!(msgs[3].role, Role::Assistant);
    }

    #[test]
    fn test_conversation_to_messages_tool_with_no_output() {
        let conv = Conversation {
            session_id: SessionId::from("s1"),
            system_prompt: None,
            turns: vec![Turn {
                user_message: Message {
                    role: Role::User,
                    content: vec![ContentBlock::Text {
                        text: "read file".into(),
                    }],
                    timestamp: Utc::now(),
                },
                assistant_response: Some(Message {
                    role: Role::Assistant,
                    content: vec![ContentBlock::ToolUse {
                        id: "call-1".into(),
                        name: "read_file".into(),
                        input: serde_json::json!({}),
                    }],
                    timestamp: Utc::now(),
                }),
                tool_invocations: vec![ToolInvocation {
                    tool_use_id: "call-1".into(),
                    tool_name: "read_file".into(),
                    input: serde_json::json!({}),
                    output: None, // no output recorded
                    is_error: false,
                    duration_ms: None,
                }],
                started_at: Utc::now(),
                completed_at: Some(Utc::now()),
            }],
            created_at: Utc::now(),
            updated_at: Utc::now(),
            model_id: "m".into(),
            provider_id: "p".into(),
        };
        let msgs = conversation_to_messages(&conv);
        assert_eq!(msgs.len(), 3);
        // Tool result should have empty string content (from unwrap_or_default)
        assert!(matches!(
            &msgs[2].content[0],
            ContentBlock::ToolResult {
                content,
                ..
            } if content.is_empty()
        ));
    }
}
