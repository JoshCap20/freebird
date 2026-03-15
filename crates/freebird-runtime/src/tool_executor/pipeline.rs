//! Security pipeline steps for tool execution.
//!
//! Contains the capability check, secret guard, approval gate, knowledge
//! consent escalation, and injection scan steps that run before/after each
//! tool invocation. These are `impl ToolExecutor` methods called from
//! `execute()` in `mod.rs`.

use freebird_security::approval::ApprovalError;
use freebird_security::audit::{AuditEventType, CapabilityCheckResult, InjectionSource};
use freebird_security::error::Severity;
use freebird_security::safe_types::ScannedToolOutput;
use freebird_security::secret_guard::SecretCheckResult;
use freebird_traits::id::SessionId;
use freebird_traits::knowledge::KnowledgeKind;
use freebird_traits::tool::{RiskLevel, ToolInfo, ToolOutcome, ToolOutput};
use freebird_types::config::InjectionResponse;

use super::ToolExecutor;

/// Result of the secret guard input check, used to control flow in `execute()`.
pub(super) enum SecretGuardInputResult {
    /// Input is safe, no action needed.
    Safe,
    /// Input triggers consent escalation with modified `ToolInfo`.
    EscalatedInfo(ToolInfo),
    /// Input is blocked; return the enclosed `ToolOutput` immediately.
    Blocked(ToolOutput),
}

impl ToolExecutor {
    // ── Audit helpers ───────────────────────────────────────────────

    /// Log an audit event via the `AuditSink`, swallowing errors with a warning.
    ///
    /// Centralizes the `if let Some(sink) = &self.audit_sink` boilerplate so
    /// every call site is a single line.
    pub(super) async fn audit_log(&self, session_id: &SessionId, event: AuditEventType) {
        if let Some(sink) = &self.audit_sink {
            crate::agent::emit_audit(sink.as_ref(), Some(session_id.as_str()), event).await;
        }
    }

    pub(super) async fn audit_tool_invocation(
        &self,
        session_id: &SessionId,
        tool_name: &str,
        result: CapabilityCheckResult,
    ) {
        self.audit_log(
            session_id,
            AuditEventType::ToolInvocation {
                tool_name: tool_name.into(),
                capability_check: result,
            },
        )
        .await;
    }

    pub(super) async fn audit_policy_violation(
        &self,
        session_id: &SessionId,
        rule: &str,
        context: &str,
        severity: Severity,
    ) {
        self.audit_log(
            session_id,
            AuditEventType::PolicyViolation {
                rule: rule.into(),
                context: context.into(),
                severity,
            },
        )
        .await;
    }

    async fn audit_approval_granted(&self, session_id: &SessionId, context: &str) {
        self.audit_log(
            session_id,
            AuditEventType::ApprovalGranted {
                context: context.into(),
            },
        )
        .await;
    }

    async fn audit_approval_denied(
        &self,
        session_id: &SessionId,
        context: &str,
        reason: Option<&str>,
    ) {
        self.audit_log(
            session_id,
            AuditEventType::ApprovalDenied {
                context: context.into(),
                reason: reason.map(Into::into),
            },
        )
        .await;
    }

    async fn audit_approval_expired(&self, session_id: &SessionId, context: &str) {
        self.audit_log(
            session_id,
            AuditEventType::ApprovalExpired {
                context: context.into(),
            },
        )
        .await;
    }

    pub(super) async fn audit_injection_detected(&self, session_id: &SessionId, pattern: &str) {
        self.audit_log(
            session_id,
            AuditEventType::InjectionDetected {
                pattern: pattern.into(),
                source: InjectionSource::ToolOutput,
                severity: Severity::High,
            },
        )
        .await;
    }

    async fn audit_secret_access(
        &self,
        session_id: &SessionId,
        tool_name: &str,
        reason: &str,
        blocked: bool,
    ) {
        let event = if blocked {
            AuditEventType::SecretAccessBlocked {
                tool_name: tool_name.into(),
                reason: reason.into(),
            }
        } else {
            AuditEventType::SecretAccessConsent {
                tool_name: tool_name.into(),
                reason: reason.into(),
            }
        };
        self.audit_log(session_id, event).await;
    }

    pub(super) async fn audit_secret_redacted(&self, session_id: &SessionId, tool_name: &str) {
        self.audit_log(
            session_id,
            AuditEventType::SecretRedacted {
                tool_name: tool_name.into(),
            },
        )
        .await;
    }

    // ── Consent gate ────────────────────────────────────────────────

    /// Check consent for a tool invocation. Returns `Some(ToolOutput)` to abort
    /// execution (denied/expired/error), or `None` to proceed.
    pub(super) async fn check_consent(
        &self,
        tool_name: &str,
        tool_info: &freebird_traits::tool::ToolInfo,
        input: &serde_json::Value,
        session_id: &SessionId,
        sender_id: &str,
    ) -> Option<ToolOutput> {
        // Truncate action summary to avoid leaking large tool inputs (e.g. file contents)
        // through the consent channel. 512 chars is enough for a human to make a decision.
        const MAX_SUMMARY_LEN: usize = 512;

        let consent = self.approval_gate.as_ref()?;
        let raw_summary =
            serde_json::to_string(input).unwrap_or_else(|_| "<unserializable input>".into());
        let action_summary = if raw_summary.len() > MAX_SUMMARY_LEN {
            // Find a char-boundary-safe truncation point to avoid panicking
            // on multi-byte UTF-8 sequences (e.g. CJK, emoji in tool input).
            let truncate_at = raw_summary
                .char_indices()
                .map(|(i, _)| i)
                .take_while(|&i| i <= MAX_SUMMARY_LEN)
                .last()
                .unwrap_or(0);
            format!(
                "{}… ({} bytes total)",
                &raw_summary[..truncate_at],
                raw_summary.len()
            )
        } else {
            raw_summary
        };
        match consent
            .check_consent(tool_info, action_summary, sender_id)
            .await
        {
            Ok(outcome) => {
                if outcome == freebird_security::approval::ConsentOutcome::Approved {
                    self.audit_approval_granted(session_id, tool_name).await;
                    tracing::info!(tool = %tool_name, %session_id, "consent approved by user");
                }
                None
            }
            Err(ApprovalError::Denied { context, reason }) => {
                tracing::warn!(%context, %reason, %session_id, "approval denied");
                self.audit_approval_denied(session_id, &context, Some(&reason))
                    .await;
                Some(ToolOutput {
                    content: format!("Approval denied for {context}: {reason}"),
                    outcome: ToolOutcome::Error,
                    metadata: None,
                })
            }
            Err(ApprovalError::Expired {
                context,
                timeout_secs,
            }) => {
                tracing::warn!(%context, timeout_secs, %session_id, "approval expired");
                self.audit_approval_expired(session_id, &context).await;
                Some(ToolOutput {
                    content: format!("Approval expired for {context} after {timeout_secs}s"),
                    outcome: ToolOutcome::Error,
                    metadata: None,
                })
            }
            Err(ApprovalError::TooManyPending { context, max }) => {
                tracing::warn!(%context, max, %session_id, "too many pending approval requests");
                self.audit_approval_denied(
                    session_id,
                    &context,
                    Some(&format!("too many pending requests (max {max})")),
                )
                .await;
                Some(ToolOutput {
                    content: format!(
                        "Too many pending approval requests ({max}); denying {context}"
                    ),
                    outcome: ToolOutcome::Error,
                    metadata: None,
                })
            }
            Err(ApprovalError::ChannelClosed) => {
                tracing::warn!(tool = %tool_name, %session_id, "approval channel closed");
                self.audit_approval_denied(session_id, tool_name, Some("approval channel closed"))
                    .await;
                Some(ToolOutput {
                    content: "Approval channel closed — cannot request approval".into(),
                    outcome: ToolOutcome::Error,
                    metadata: None,
                })
            }
        }
    }

    // ── Knowledge consent escalation ────────────────────────────────

    /// Escalate consent-gated knowledge tool calls to `High` risk.
    ///
    /// The three mutable knowledge tools (`store_knowledge`, `update_knowledge`,
    /// `delete_knowledge`) are declared `Medium` risk so that routine agent
    /// writes (learned patterns, error resolutions, session insights) do not
    /// require user approval. However, kinds that carry persistent configuration
    /// or user-declared preferences (`system_config`, `tool_capability`,
    /// `user_preference`) MUST go through the consent gate regardless of the
    /// global threshold — they modify agent behaviour in ways the user should
    /// explicitly authorise.
    ///
    /// This method inspects the raw tool input for a `"kind"` field and, if the
    /// parsed kind returns `true` from [`KnowledgeKind::requires_consent`],
    /// returns an escalated clone of `tool_info` with `risk_level = High`.
    /// The escalated info is then passed to `check_consent` exactly like the
    /// secret-guard escalation path, so no call site needs to know about it.
    ///
    /// For `delete_knowledge` the kind is not in the input (only an `id` is).
    /// Deletion of consent-gated entries is handled by the tool itself at
    /// execution time — this method returns `None` for that tool.
    ///
    /// Returns `Some(escalated_info)` if escalation is needed, `None` otherwise.
    pub(super) fn check_knowledge_consent_escalation(
        tool_name: &str,
        input: &serde_json::Value,
        tool_info: &ToolInfo,
    ) -> Option<ToolInfo> {
        // Only the two tools that accept a `kind` field in their input need
        // pre-execution escalation. `delete_knowledge` operates on an opaque
        // ID so we cannot inspect the kind before fetching from the store.
        // TODO: Should also gate delete_knowledge
        if !matches!(tool_name, "store_knowledge" | "update_knowledge") {
            return None;
        }

        // If the input has no `kind` field, or it cannot be parsed, we do not
        // escalate — the tool's own validation will reject the call.
        let kind_str = input.get("kind")?.as_str()?;
        let json_quoted = format!("\"{kind_str}\"");
        let kind: KnowledgeKind = serde_json::from_str(&json_quoted).ok()?;

        if !kind.requires_consent() {
            return None;
        }

        tracing::info!(
            tool = %tool_name,
            %kind_str,
            "knowledge consent escalation: consent-gated kind requires approval"
        );

        let mut escalated = tool_info.clone();
        escalated.risk_level = RiskLevel::High;
        escalated.description = format!(
            "{} [CONSENT REQUIRED: kind `{kind_str}` requires human approval]",
            escalated.description
        );
        Some(escalated)
    }

    // ── Secret guard ────────────────────────────────────────────────

    /// Check the secret guard for sensitive patterns in tool input.
    pub(super) async fn check_secret_guard_input(
        &self,
        tool_name: &str,
        input: &serde_json::Value,
        tool_info: &ToolInfo,
        session_id: &SessionId,
    ) -> SecretGuardInputResult {
        let Some(guard) = &self.secret_guard else {
            return SecretGuardInputResult::Safe;
        };
        match guard.check_tool_input(tool_name, input) {
            SecretCheckResult::Safe => SecretGuardInputResult::Safe,
            SecretCheckResult::RequiresConsent { reason } => {
                tracing::info!(
                    tool = %tool_name,
                    %reason,
                    "secret guard escalating to consent"
                );
                self.audit_secret_access(session_id, tool_name, &reason, false)
                    .await;
                let mut info = tool_info.clone();
                info.risk_level = RiskLevel::Critical;
                info.description = format!("{} [SECRET GUARD: {}]", info.description, reason);
                SecretGuardInputResult::EscalatedInfo(info)
            }
            SecretCheckResult::Blocked { reason } => {
                tracing::warn!(
                    tool = %tool_name,
                    %reason,
                    "secret guard blocked tool invocation"
                );
                self.audit_secret_access(session_id, tool_name, &reason, true)
                    .await;
                SecretGuardInputResult::Blocked(ToolOutput {
                    content: format!("Secret access blocked for tool `{tool_name}`: {reason}"),
                    outcome: ToolOutcome::Error,
                    metadata: None,
                })
            }
        }
    }

    /// Redact secrets from tool output if the secret guard is configured to do so.
    pub(super) async fn maybe_redact_output(
        &self,
        output: ToolOutput,
        tool_name: &str,
        session_id: &SessionId,
    ) -> ToolOutput {
        let Some(guard) = &self.secret_guard else {
            return output;
        };
        if !guard.should_redact_output() || output.outcome == ToolOutcome::Error {
            return output;
        }
        let (redacted, was_redacted) =
            freebird_security::secret_guard::SecretGuard::redact_output(&output.content);
        if was_redacted {
            tracing::info!(tool = %tool_name, "redacted secrets in tool output");
            self.audit_secret_redacted(session_id, tool_name).await;
            ToolOutput {
                content: redacted,
                outcome: output.outcome,
                metadata: output.metadata,
            }
        } else {
            output
        }
    }

    // ── Output injection scan ───────────────────────────────────────

    /// Scan non-error tool output for prompt injection patterns.
    ///
    /// Behavior depends on `injection_config.tool_output_response`:
    /// - `Block` → synthetic error, no user prompt
    /// - `Prompt` → ask user via `ApprovalGate`; approve passes original, deny blocks
    /// - `Allow` → warn in logs, pass through original content
    pub(super) async fn scan_output_for_injection(
        &self,
        output: ToolOutput,
        tool_name: &str,
        session_id: &SessionId,
        sender_id: &str,
    ) -> ToolOutput {
        if output.outcome == ToolOutcome::Error {
            return output;
        }
        let scanned = ScannedToolOutput::from_raw(&output.content);
        if scanned.injection_detected() {
            self.audit_injection_detected(session_id, "prompt injection pattern in tool output")
                .await;

            match self.injection_config.tool_output_response {
                InjectionResponse::Block => {
                    tracing::warn!(
                        tool = %tool_name,
                        session_id = %session_id,
                        "injection detected in tool output — blocking per config"
                    );
                    return ToolOutput {
                        content: format!(
                            "Tool output from `{tool_name}` was blocked: injection pattern detected"
                        ),
                        outcome: ToolOutcome::Error,
                        metadata: None,
                    };
                }
                InjectionResponse::Allow => {
                    tracing::warn!(
                        tool = %tool_name,
                        session_id = %session_id,
                        "injection detected in tool output — allowing per config"
                    );
                    return ToolOutput {
                        content: scanned.into_original_or_content(),
                        outcome: output.outcome,
                        metadata: output.metadata,
                    };
                }
                InjectionResponse::Prompt => {} // fall through to prompt logic
            }

            let preview = output.content.chars().take(200).collect::<String>();

            match self
                .check_security_warning(
                    "injection_tool_output".into(),
                    "prompt injection pattern in tool output".into(),
                    preview,
                    format!("tool:{tool_name}"),
                    sender_id,
                )
                .await
            {
                Ok(()) => {
                    tracing::info!(
                        tool = %tool_name,
                        session_id = %session_id,
                        "user approved tool output injection warning — proceeding with original content"
                    );
                    ToolOutput {
                        content: scanned.into_original_or_content(),
                        outcome: output.outcome,
                        metadata: output.metadata,
                    }
                }
                Err(ApprovalError::Denied { reason, .. }) => {
                    tracing::warn!(
                        tool = %tool_name,
                        %session_id,
                        ?reason,
                        "user denied tool output injection warning — returning synthetic error"
                    );
                    ToolOutput {
                        content: format!(
                            "Tool output from `{tool_name}` was rejected: injection pattern detected and denied by user"
                        ),
                        outcome: ToolOutcome::Error,
                        metadata: None,
                    }
                }
                Err(e) => {
                    // Expired, too many pending, channel closed, or no gate.
                    // Fall back to warn-and-proceed since the content may be
                    // legitimate (security docs, test fixtures, etc.).
                    tracing::warn!(
                        tool = %tool_name,
                        session_id = %session_id,
                        error = %e,
                        "approval gate unavailable for tool output injection — proceeding with original content"
                    );
                    ToolOutput {
                        content: scanned.into_original_or_content(),
                        outcome: output.outcome,
                        metadata: output.metadata,
                    }
                }
            }
        } else {
            output
        }
    }
}
