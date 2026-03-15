//! Agentic loop implementations and budget helpers for `AgentRuntime`.

use chrono::Utc;
use freebird_security::audit::AuditEventType;
use freebird_security::capability::CapabilityGrant;
use freebird_security::error::Severity;
use freebird_security::injection;
use freebird_security::safe_types::SafeMessage;
use freebird_traits::id::SessionId;
use freebird_traits::memory::{Conversation, Turn};
use freebird_traits::provider::{
    CompletionRequest, ContentBlock, Message, ProviderError, Role, StopReason, StreamEvent,
    TokenUsage, ToolDefinition,
};
use freebird_types::config::ConversationSummary;
use futures::StreamExt;
use tokio::sync::mpsc;

use freebird_traits::channel::OutboundEvent;

use crate::history::conversation_to_messages;
use crate::stream::StreamAccumulator;
use crate::summarize;

use super::{AgentRuntime, send_outbound, send_stream_error};

/// Control flow outcome from budget check helpers.
///
/// Used by `check_round_budget` and `record_and_check_usage` to tell
/// the caller whether the agentic loop should continue, break (max
/// rounds without budget), or return early (user denied approval).
pub(super) enum BudgetOutcome {
    /// Budget check passed (or no budget configured) — continue the loop.
    Continue,
    /// No budget configured and the static max-rounds limit was hit.
    BreakLoop,
    /// Budget exceeded and the user denied (or timed out) — abort the turn.
    ReturnTurn,
}

impl AgentRuntime {
    /// Build the initial state for an agentic loop: user message, turn, messages, tool defs.
    pub(super) async fn prepare_agentic_loop(
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
    pub(super) async fn log_max_rounds_exceeded(
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
    pub(super) async fn handle_budget_exceeded(
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

    /// Check the tool-round budget before each agentic loop iteration.
    ///
    /// When a budget is configured, delegates to `check_tool_rounds` and routes
    /// any exceeded error through the approval gate.  When no budget exists,
    /// falls back to the static `max_rounds` limit from config.
    #[expect(clippy::too_many_arguments, reason = "complex orchestration logic")]
    pub(super) async fn check_round_budget(
        &self,
        round: usize,
        max_rounds: usize,
        budget: Option<&freebird_security::budget::TokenBudget>,
        session_id: &SessionId,
        current_turn: &mut Turn,
        sender_id: &str,
        outbound: &mpsc::Sender<OutboundEvent>,
    ) -> BudgetOutcome {
        if let Some(b) = budget {
            let round_u32 = u32::try_from(round).unwrap_or(u32::MAX);
            if let Err(e) = b.check_tool_rounds(round_u32) {
                if !self
                    .handle_budget_exceeded(
                        session_id,
                        &e,
                        current_turn,
                        sender_id,
                        outbound,
                        budget,
                    )
                    .await
                {
                    return BudgetOutcome::ReturnTurn;
                }
                // User approved — continue the loop beyond the limit.
            }
        } else if round >= max_rounds {
            // No budget configured — fall back to config limit.
            return BudgetOutcome::BreakLoop;
        }
        BudgetOutcome::Continue
    }

    /// Record token usage after a provider response and check per-request /
    /// per-session limits.
    ///
    /// When the limit is exceeded, routes the error through the approval gate.
    /// On approval the usage is force-committed; on denial the caller should
    /// abort the loop.
    pub(super) async fn record_and_check_usage(
        &self,
        usage: &TokenUsage,
        budget: Option<&freebird_security::budget::TokenBudget>,
        session_id: &SessionId,
        current_turn: &mut Turn,
        sender_id: &str,
        outbound: &mpsc::Sender<OutboundEvent>,
    ) -> BudgetOutcome {
        if let Some(b) = budget {
            if let Err(e) = b.record_usage(usage) {
                if self
                    .handle_budget_exceeded(
                        session_id,
                        &e,
                        current_turn,
                        sender_id,
                        outbound,
                        budget,
                    )
                    .await
                {
                    // User approved — force-commit the usage.
                    b.force_record_usage(usage);
                } else {
                    return BudgetOutcome::ReturnTurn;
                }
            }
        }
        BudgetOutcome::Continue
    }

    /// Run the agentic loop: call provider, handle tool use, deliver response.
    ///
    /// When `initial_request` is `Some`, the first loop iteration uses it directly
    /// instead of building a fresh `CompletionRequest`. This supports the streaming
    /// fallback path where the request has already been constructed.
    #[expect(
        clippy::too_many_arguments,
        clippy::too_many_lines,
        reason = "complex orchestration logic"
    )]
    pub(super) async fn run_agentic_loop(
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
            freebird_traits::event::ConversationEvent::TurnStarted {
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
            match self
                .check_round_budget(
                    round,
                    max_rounds,
                    budget.as_deref(),
                    session_id,
                    &mut current_turn,
                    sender_id,
                    outbound,
                )
                .await
            {
                BudgetOutcome::Continue => {}
                BudgetOutcome::BreakLoop => break,
                BudgetOutcome::ReturnTurn => return current_turn,
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

            if matches!(
                self.record_and_check_usage(
                    &response.usage,
                    budget.as_deref(),
                    session_id,
                    &mut current_turn,
                    sender_id,
                    outbound,
                )
                .await,
                BudgetOutcome::ReturnTurn
            ) {
                return current_turn;
            }

            match response.stop_reason {
                StopReason::ToolUse => {
                    let invocations_before = current_turn.tool_invocations.len();
                    let messages_before = current_turn.assistant_messages.len();

                    self.execute_tool_calls(
                        &response.message,
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

    /// Streaming variant of the agentic loop.
    ///
    /// Sends `StreamChunk` events for each text delta, `StreamEnd` between
    /// tool-use rounds and at final response. Falls back to non-streaming
    /// `complete_with_failover()` if stream setup fails.
    ///
    /// Injection scan on accumulated text is audit-only — the text has already
    /// been delivered to the user via `StreamChunk` events.
    #[expect(
        clippy::too_many_lines,
        clippy::too_many_arguments,
        reason = "complex orchestration logic"
    )]
    pub(super) async fn run_agentic_loop_streaming(
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
            freebird_traits::event::ConversationEvent::TurnStarted {
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
            match self
                .check_round_budget(
                    round,
                    max_rounds,
                    budget.as_deref(),
                    session_id,
                    &mut current_turn,
                    sender_id,
                    outbound,
                )
                .await
            {
                BudgetOutcome::Continue => {}
                BudgetOutcome::BreakLoop => break,
                BudgetOutcome::ReturnTurn => return current_turn,
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

            if matches!(
                self.record_and_check_usage(
                    &usage,
                    budget.as_deref(),
                    session_id,
                    &mut current_turn,
                    sender_id,
                    outbound,
                )
                .await,
                BudgetOutcome::ReturnTurn
            ) {
                return current_turn;
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
                        freebird_traits::event::ConversationEvent::AssistantMessage {
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
                        freebird_traits::event::ConversationEvent::AssistantMessage {
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
    pub(super) async fn consume_stream(
        mut event_stream: std::pin::Pin<
            Box<dyn futures::Stream<Item = Result<StreamEvent, ProviderError>> + Send>,
        >,
        sender_id: &str,
        session_id: &SessionId,
        audit_sink: Option<&dyn freebird_traits::audit::AuditSink>,
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
}
