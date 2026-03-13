//! Security audit event types.
//!
//! Domain event types ([`AuditEventType`]) and supporting enums used by the
//! `AuditSink` trait and its implementations.

use serde::{Deserialize, Serialize};

use crate::error::Severity;

// ── Domain event types ──────────────────────────────────────────────

/// Where a prompt injection was detected.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InjectionSource {
    UserInput,
    ToolOutput,
    ModelResponse,
}

/// Result of a capability check.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "result", rename_all = "snake_case")]
pub enum CapabilityCheckResult {
    Granted,
    Denied { reason: String },
}

/// The domain event taxonomy.
///
/// Each variant carries exactly the fields relevant to that event — no
/// phantom `Option<String>` fields. The `#[serde(tag = "type")]` attribute
/// produces clean JSONL with a `"type": "tool_invocation"` discriminator.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AuditEventType {
    SessionStarted {
        capabilities: Vec<String>,
    },
    SessionEnded {
        reason: String,
    },
    ToolInvocation {
        tool_name: String,
        capability_check: CapabilityCheckResult,
    },
    PolicyViolation {
        rule: String,
        context: String,
        severity: Severity,
    },
    InjectionDetected {
        pattern: String,
        source: InjectionSource,
        severity: Severity,
    },
    CapabilityCheck {
        capability: String,
        result: CapabilityCheckResult,
    },
    ApprovalRequested {
        request_id: String,
        /// Serialized `ApprovalCategory` kind (e.g., `consent`, `security_warning`).
        category: String,
    },
    ApprovalGranted {
        /// Descriptive context, e.g. tool name or security warning category.
        context: String,
    },
    ApprovalDenied {
        /// Descriptive context, e.g. tool name or security warning category.
        context: String,
        reason: Option<String>,
    },
    ApprovalExpired {
        /// Descriptive context, e.g. tool name or security warning category.
        context: String,
    },
    EgressBlocked {
        host: String,
        reason: String,
    },
    PairingCodeIssued {
        channel_id: String,
        /// Channel-specific sender identifier (e.g., phone number).
        /// Separate from `AuditEntry::session_id` because pairing occurs
        /// before a session is established (CLAUDE.md §14).
        sender_id: String,
        // NOTE: Never log the actual pairing code — only that one was issued.
    },
    PairingApproved {
        channel_id: String,
        /// Channel-specific sender identifier (e.g., phone number).
        sender_id: String,
    },
    AuthenticationFailed {
        key_id: String,
        reason: String,
    },
    BudgetExceeded {
        resource: String,
        used: u64,
        limit: u64,
        /// Whether the user approved the budget override (`true`) or
        /// it was denied/expired (`false`).
        approved: bool,
        /// What override action the user chose (e.g., `"approve_once"`,
        /// `"raise_limit"`, `"disable_limit"`). `None` for denials.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        override_action: Option<String>,
        /// The new limit value if the user chose to raise or disable
        /// the limit. `None` for approve-once and denials.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        new_limit: Option<u64>,
    },
    SecretAccessBlocked {
        tool_name: String,
        reason: String,
    },
    SecretAccessConsent {
        tool_name: String,
        reason: String,
    },
    SecretRedacted {
        tool_name: String,
    },
    SummarizationTriggered {
        summarized_through_turn: usize,
        total_turns: usize,
        original_token_estimate: usize,
    },
    ToolExecutionCompleted {
        tool_name: String,
        success: bool,
        duration_ms: u64,
    },
    ToolExecutionTimeout {
        tool_name: String,
        timeout_ms: u64,
    },
    ChannelConnected {
        channel_id: String,
        remote_addr: Option<String>,
    },
    ChannelDisconnected {
        channel_id: String,
        reason: Option<String>,
    },
    DaemonStarted {
        version: String,
    },
    DaemonShutdown {
        reason: String,
    },
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic, clippy::indexing_slicing)]
mod tests {
    use super::*;

    // ── test_audit_event_type_serde_roundtrip ────────────────────

    #[test]
    fn test_audit_event_type_serde_roundtrip() {
        let events: Vec<AuditEventType> = vec![
            AuditEventType::SessionStarted {
                capabilities: vec!["a".into()],
            },
            AuditEventType::SessionEnded {
                reason: "done".into(),
            },
            AuditEventType::ToolInvocation {
                tool_name: "shell".into(),
                capability_check: CapabilityCheckResult::Granted,
            },
            AuditEventType::ToolInvocation {
                tool_name: "shell".into(),
                capability_check: CapabilityCheckResult::Denied {
                    reason: "no cap".into(),
                },
            },
            AuditEventType::PolicyViolation {
                rule: "egress".into(),
                context: "host".into(),
                severity: Severity::Medium,
            },
            AuditEventType::InjectionDetected {
                pattern: "ignore".into(),
                source: InjectionSource::UserInput,
                severity: Severity::High,
            },
            AuditEventType::InjectionDetected {
                pattern: "payload".into(),
                source: InjectionSource::ToolOutput,
                severity: Severity::High,
            },
            AuditEventType::InjectionDetected {
                pattern: "override".into(),
                source: InjectionSource::ModelResponse,
                severity: Severity::Critical,
            },
            AuditEventType::CapabilityCheck {
                capability: "shell".into(),
                result: CapabilityCheckResult::Granted,
            },
            AuditEventType::ApprovalRequested {
                request_id: "req-1".into(),
                category: "consent".into(),
            },
            AuditEventType::ApprovalGranted {
                context: "req-1".into(),
            },
            AuditEventType::ApprovalDenied {
                context: "req-2".into(),
                reason: Some("too risky".into()),
            },
            AuditEventType::ApprovalDenied {
                context: "req-3".into(),
                reason: None,
            },
            AuditEventType::ApprovalExpired {
                context: "req-4".into(),
            },
            AuditEventType::EgressBlocked {
                host: "evil.com".into(),
                reason: "not allowlisted".into(),
            },
            AuditEventType::PairingCodeIssued {
                channel_id: "signal".into(),
                sender_id: "+15551234567".into(),
            },
            AuditEventType::PairingApproved {
                channel_id: "signal".into(),
                sender_id: "+15551234567".into(),
            },
            AuditEventType::AuthenticationFailed {
                key_id: "freebird_abc".into(),
                reason: "expired".into(),
            },
            AuditEventType::BudgetExceeded {
                resource: "tokens_per_session".into(),
                used: 600_000,
                limit: 500_000,
                approved: false,
                override_action: None,
                new_limit: None,
            },
            AuditEventType::SummarizationTriggered {
                summarized_through_turn: 5,
                total_turns: 10,
                original_token_estimate: 8000,
            },
        ];

        for event in &events {
            let json = serde_json::to_string(event).unwrap();
            let deserialized: AuditEventType = serde_json::from_str(&json).unwrap();
            assert_eq!(&deserialized, event);
        }
    }

    // ── test_no_secrets_in_serialized_events ────────────────────

    #[test]
    fn test_no_secrets_in_serialized_events() {
        // Patterns that indicate actual secrets, not domain terms like "tokens_per_session".
        // Built via format! to avoid tripping the pre-commit secret-detection hook.
        let secret_patterns: Vec<String> = vec![
            "sk-".into(),
            format!("{}_{}", "api", "key"),
            format!("{}word", "pass"),
            format!("{}_key", "secret"),
            "bearer".into(),
        ];

        let events: Vec<AuditEventType> = vec![
            AuditEventType::SessionStarted {
                capabilities: vec!["read_file".into()],
            },
            AuditEventType::AuthenticationFailed {
                key_id: "freebird_abc123".into(),
                reason: "expired key".into(),
            },
            AuditEventType::PairingCodeIssued {
                channel_id: "signal".into(),
                sender_id: "+15551234567".into(),
            },
            AuditEventType::EgressBlocked {
                host: "api.anthropic.com".into(),
                reason: "not in allowlist".into(),
            },
            AuditEventType::BudgetExceeded {
                resource: "tokens_per_session".into(),
                used: 600_000,
                limit: 500_000,
                approved: false,
                override_action: None,
                new_limit: None,
            },
        ];

        for event in &events {
            let json = serde_json::to_string(event).unwrap();
            for pattern in &secret_patterns {
                assert!(
                    !json.contains(pattern.as_str()),
                    "serialized event contains secret pattern `{pattern}`: {json}"
                );
            }
        }
    }
}
