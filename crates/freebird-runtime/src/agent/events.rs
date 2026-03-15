//! Event emission and audit logging helpers for `AgentRuntime`.

use freebird_security::audit::{AuditEventType, InjectionSource};
use freebird_security::error::Severity;
use freebird_security::safe_types::ScannedModelResponse;
use freebird_traits::audit::AuditSink;
use freebird_traits::event::ConversationEvent;
use freebird_traits::id::SessionId;
use freebird_traits::memory::Turn;
use freebird_traits::provider::Message;

use super::{AgentRuntime, extract_text};

impl AgentRuntime {
    /// Record an audit event without a session context.
    ///
    /// Used for daemon-level and channel-level events that occur
    /// outside of a specific session. No-op when no `AuditSink` is configured.
    /// Errors are logged but never block the agent loop.
    pub(crate) async fn audit_no_session(&self, event: AuditEventType) {
        if let Some(sink) = &self.audit_sink {
            emit_audit(sink.as_ref(), None, event).await;
        }
    }

    pub(crate) async fn audit(&self, session_id: &SessionId, event: AuditEventType) {
        if let Some(sink) = &self.audit_sink {
            emit_audit(sink.as_ref(), Some(session_id.as_str()), event).await;
        }
    }

    /// Emit a conversation event via the `EventSink` if configured.
    ///
    /// Errors are logged but never block the agent loop.
    pub(crate) async fn emit_event(&self, session_id: &SessionId, event: ConversationEvent) {
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
    pub(crate) async fn emit_new_assistant_message(
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
    pub(crate) async fn emit_new_tool_invocations(
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
    pub(crate) async fn emit_turn_completed(
        &self,
        session_id: &SessionId,
        turn: &Turn,
        turn_index: usize,
    ) {
        self.emit_event(
            session_id,
            ConversationEvent::TurnCompleted {
                turn_index,
                completed_at: turn.completed_at.unwrap_or_else(chrono::Utc::now),
            },
        )
        .await;
    }

    /// Log and audit a model output injection detection.
    pub(crate) async fn audit_model_injection(&self, session_id: &SessionId) {
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

    /// Record an audit event for a stream error.
    pub(crate) async fn audit_stream_error(
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
    pub(crate) async fn audit_streaming_injection(
        &self,
        session_id: &SessionId,
        message: &Message,
    ) {
        let response_text = extract_text(message);
        let scanned = ScannedModelResponse::from_raw(&response_text);
        if scanned.injection_detected() {
            self.audit_model_injection(session_id).await;
        }
    }
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
