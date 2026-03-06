//! Consent gates — human-in-the-loop approval for high-risk tool operations.
//!
//! Implements OWASP ASI09 defense against Human-Agent Trust Exploitation.
//! Tools at or above the configured risk threshold require explicit human
//! approval before execution.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use chrono::{DateTime, TimeDelta, Utc};
use freebird_traits::tool::{RiskLevel, ToolInfo};
use serde::{Deserialize, Serialize};
use tokio::sync::{Mutex, mpsc, oneshot};

/// A consent request presented to the user via their active channel.
///
/// The runtime bridges this to the channel's `OutboundEvent`. On CLI, it
/// renders as an inline prompt. On messaging channels, it could be
/// rendered as approve/deny buttons.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsentRequest {
    /// Unique ID for correlating request → response.
    pub id: String,
    /// Which tool is requesting consent.
    pub tool_name: String,
    /// Tool description for human context (from `ToolInfo.description`).
    pub description: String,
    /// Risk classification.
    pub risk_level: RiskLevel,
    /// What the tool will do — MUST be machine-generated from the actual
    /// tool input, never from LLM-provided text.
    pub action_summary: String,
    /// When the request was created.
    pub requested_at: DateTime<Utc>,
    /// When the request expires if not answered.
    pub expires_at: DateTime<Utc>,
}

/// The user's response to a consent request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConsentResponse {
    Approved,
    Denied { reason: Option<String> },
}

/// Errors specific to consent operations.
///
/// These are workflow errors, not security violations. The `ToolExecutor`
/// catches them and converts to `ToolOutput { outcome: ToolOutcome::Error }`.
#[derive(Debug, thiserror::Error)]
pub enum ConsentError {
    #[error("consent denied for tool `{tool}`: {reason}")]
    Denied { tool: String, reason: String },

    #[error("consent request for tool `{tool}` expired after {timeout_secs}s")]
    Expired { tool: String, timeout_secs: u64 },

    #[error("too many pending consent requests ({max}); denying tool `{tool}`")]
    TooManyPending { tool: String, max: usize },

    #[error("consent channel closed")]
    ChannelClosed,
}

/// The consent gate sits between capability checks and tool execution
/// in the `ToolExecutor` pipeline. It blocks tool invocation for High/Critical
/// risk tools until the user explicitly approves.
///
/// Uses `mpsc + oneshot` channels:
/// - `request_tx`: sends `ConsentRequest`s to the runtime (which forwards to
///   the user's channel)
/// - `pending`: maps request IDs to oneshot senders. When the user responds,
///   the runtime calls `respond()` which sends via the oneshot.
/// - `check()` awaits the oneshot with a timeout.
pub struct ConsentGate {
    /// Tools at or above this risk level require consent.
    threshold: RiskLevel,
    /// How long to wait for a consent response.
    consent_ttl: Duration,
    /// Channel to send consent requests to the runtime.
    request_tx: mpsc::Sender<ConsentRequest>,
    /// Pending response senders, keyed by request ID.
    pending: Arc<Mutex<HashMap<String, oneshot::Sender<ConsentResponse>>>>,
    /// Maximum concurrent pending consent requests (prevents LLM flooding).
    max_pending: usize,
}

impl ConsentGate {
    /// Create a new consent gate and its request receiver.
    ///
    /// The returned `mpsc::Receiver<ConsentRequest>` is consumed by the runtime,
    /// which forwards requests to the user's active channel.
    #[must_use]
    pub fn new(
        threshold: RiskLevel,
        consent_ttl: Duration,
        max_pending: usize,
    ) -> (Self, mpsc::Receiver<ConsentRequest>) {
        let channel_capacity = max_pending.max(16);
        let (request_tx, request_rx) = mpsc::channel(channel_capacity);
        let gate = Self {
            threshold,
            consent_ttl,
            request_tx,
            pending: Arc::new(Mutex::new(HashMap::new())),
            max_pending,
        };
        (gate, request_rx)
    }

    /// Check if a tool invocation requires consent.
    ///
    /// If `tool_info.risk_level < self.threshold`, returns `Ok(())`
    /// immediately (auto-approved). Tools **at or above** the threshold
    /// require consent.
    ///
    /// `action_summary` MUST be machine-generated from the actual tool
    /// input (e.g., by the `ToolExecutor` serializing the input JSON).
    /// Never pass LLM-generated text as the summary.
    ///
    /// # Errors
    ///
    /// Returns [`ConsentError::Denied`] if the user explicitly denies,
    /// [`ConsentError::Expired`] on timeout, [`ConsentError::TooManyPending`]
    /// if the rate limit is hit, or [`ConsentError::ChannelClosed`] if the
    /// runtime receiver was dropped.
    pub async fn check(
        &self,
        tool_info: &ToolInfo,
        action_summary: String,
    ) -> Result<(), ConsentError> {
        if tool_info.risk_level < self.threshold {
            return Ok(());
        }

        let request_id = uuid::Uuid::new_v4().to_string();
        let now = Utc::now();
        let ttl_delta =
            TimeDelta::from_std(self.consent_ttl).unwrap_or_else(|_| TimeDelta::seconds(60));

        let request = ConsentRequest {
            id: request_id.clone(),
            tool_name: tool_info.name.clone(),
            description: tool_info.description.clone(),
            risk_level: tool_info.risk_level.clone(),
            action_summary,
            requested_at: now,
            expires_at: now + ttl_delta,
        };

        // Single lock acquisition: rate-limit check + insert (no TOCTOU).
        let (response_tx, response_rx) = oneshot::channel();
        {
            let mut pending = self.pending.lock().await;
            if pending.len() >= self.max_pending {
                return Err(ConsentError::TooManyPending {
                    tool: tool_info.name.clone(),
                    max: self.max_pending,
                });
            }
            pending.insert(request_id.clone(), response_tx);
        }

        // Send request to runtime. If the receiver is dropped, clean up.
        if self.request_tx.send(request).await.is_err() {
            self.pending.lock().await.remove(&request_id);
            return Err(ConsentError::ChannelClosed);
        }

        // Wait for response with timeout.
        let tool_name = tool_info.name.clone();
        let timeout_secs = self.consent_ttl.as_secs();

        match tokio::time::timeout(self.consent_ttl, response_rx).await {
            Ok(Ok(ConsentResponse::Approved)) => Ok(()),
            Ok(Ok(ConsentResponse::Denied { reason })) => Err(ConsentError::Denied {
                tool: tool_name,
                reason: reason.unwrap_or_else(|| "user denied".into()),
            }),
            // Oneshot sender dropped (Ok(Err)) or timeout elapsed (Err).
            Ok(Err(_)) | Err(_) => {
                self.pending.lock().await.remove(&request_id);
                Err(ConsentError::Expired {
                    tool: tool_name,
                    timeout_secs,
                })
            }
        }
    }

    /// Called by the runtime when the user responds to a consent request.
    ///
    /// Returns `true` if the response was delivered, `false` if the
    /// request ID was not found (already expired or already responded).
    pub async fn respond(&self, request_id: &str, response: ConsentResponse) -> bool {
        self.pending
            .lock()
            .await
            .remove(request_id)
            .is_some_and(|sender| sender.send(response).is_ok())
    }

    /// Returns the number of currently pending consent requests.
    pub async fn pending_count(&self) -> usize {
        self.pending.lock().await.len()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;
    use freebird_traits::tool::{Capability, SideEffects};

    fn make_tool_info(name: &str, risk_level: RiskLevel) -> ToolInfo {
        ToolInfo {
            name: name.to_string(),
            description: format!("{name} tool"),
            input_schema: serde_json::json!({"type": "object"}),
            required_capability: Capability::FileRead,
            risk_level,
            side_effects: SideEffects::None,
        }
    }

    // ── Auto-approval threshold ────────────────────────────────────

    #[tokio::test]
    async fn test_below_threshold_auto_approved() {
        let (gate, mut rx) = ConsentGate::new(RiskLevel::Medium, Duration::from_secs(60), 5);
        let tool = make_tool_info("read_file", RiskLevel::Low);

        let result = gate.check(&tool, "reading /tmp/foo".into()).await;
        assert!(result.is_ok());

        // No request should have been sent.
        assert!(rx.try_recv().is_err());
    }

    #[tokio::test]
    async fn test_at_threshold_requires_consent() {
        let (gate, mut rx) = ConsentGate::new(RiskLevel::Medium, Duration::from_secs(60), 5);
        let gate = Arc::new(gate);
        let tool = make_tool_info("write_file", RiskLevel::Medium);

        let gate_clone = Arc::clone(&gate);
        let tool_clone = tool.clone();
        let handle = tokio::spawn(async move {
            gate_clone
                .check(&tool_clone, "writing /tmp/bar".into())
                .await
        });

        // Receive the consent request and approve it.
        let request = rx.recv().await.unwrap();
        assert_eq!(request.tool_name, "write_file");
        assert_eq!(request.risk_level, RiskLevel::Medium);

        gate.respond(&request.id, ConsentResponse::Approved).await;

        let result = handle.await.unwrap();
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_above_threshold_requires_consent() {
        let (gate, mut rx) = ConsentGate::new(RiskLevel::Medium, Duration::from_secs(60), 5);
        let gate = Arc::new(gate);
        let tool = make_tool_info("shell", RiskLevel::High);

        let gate_clone = Arc::clone(&gate);
        let tool_clone = tool.clone();
        let handle =
            tokio::spawn(
                async move { gate_clone.check(&tool_clone, "running rm -rf".into()).await },
            );

        let request = rx.recv().await.unwrap();
        gate.respond(&request.id, ConsentResponse::Approved).await;

        assert!(handle.await.unwrap().is_ok());
    }

    #[tokio::test]
    async fn test_critical_at_critical_threshold_requires_consent() {
        let (gate, mut rx) = ConsentGate::new(RiskLevel::Critical, Duration::from_secs(60), 5);
        let gate = Arc::new(gate);
        let tool = make_tool_info("modify_config", RiskLevel::Critical);

        let gate_clone = Arc::clone(&gate);
        let tool_clone = tool.clone();
        let handle = tokio::spawn(async move {
            gate_clone
                .check(&tool_clone, "changing config".into())
                .await
        });

        let request = rx.recv().await.unwrap();
        assert_eq!(request.risk_level, RiskLevel::Critical);
        gate.respond(&request.id, ConsentResponse::Approved).await;

        assert!(handle.await.unwrap().is_ok());
    }

    #[tokio::test]
    async fn test_all_below_critical_auto_approved() {
        let (gate, mut rx) = ConsentGate::new(RiskLevel::Critical, Duration::from_secs(60), 5);

        for (name, risk) in [
            ("read_file", RiskLevel::Low),
            ("write_file", RiskLevel::Medium),
            ("shell", RiskLevel::High),
        ] {
            let tool = make_tool_info(name, risk);
            assert!(gate.check(&tool, "action".into()).await.is_ok());
        }

        // No requests should have been sent.
        assert!(rx.try_recv().is_err());
    }

    // ── Consent flow ───────────────────────────────────────────────

    #[tokio::test]
    async fn test_consent_approved() {
        let (gate, mut rx) = ConsentGate::new(RiskLevel::High, Duration::from_secs(60), 5);
        let gate = Arc::new(gate);
        let tool = make_tool_info("shell", RiskLevel::High);

        let gate_clone = Arc::clone(&gate);
        let tool_clone = tool.clone();
        let handle =
            tokio::spawn(async move { gate_clone.check(&tool_clone, "ls -la".into()).await });

        let req = rx.recv().await.unwrap();
        gate.respond(&req.id, ConsentResponse::Approved).await;

        assert!(handle.await.unwrap().is_ok());
    }

    #[tokio::test]
    async fn test_consent_denied_with_reason() {
        let (gate, mut rx) = ConsentGate::new(RiskLevel::High, Duration::from_secs(60), 5);
        let gate = Arc::new(gate);
        let tool = make_tool_info("shell", RiskLevel::High);

        let gate_clone = Arc::clone(&gate);
        let tool_clone = tool.clone();
        let handle =
            tokio::spawn(async move { gate_clone.check(&tool_clone, "rm -rf /".into()).await });

        let req = rx.recv().await.unwrap();
        gate.respond(
            &req.id,
            ConsentResponse::Denied {
                reason: Some("too risky".into()),
            },
        )
        .await;

        let err = handle.await.unwrap().unwrap_err();
        match err {
            ConsentError::Denied { tool, reason } => {
                assert_eq!(tool, "shell");
                assert_eq!(reason, "too risky");
            }
            other => panic!("expected Denied, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_consent_denied_no_reason_defaults() {
        let (gate, mut rx) = ConsentGate::new(RiskLevel::High, Duration::from_secs(60), 5);
        let gate = Arc::new(gate);
        let tool = make_tool_info("shell", RiskLevel::High);

        let gate_clone = Arc::clone(&gate);
        let tool_clone = tool.clone();
        let handle =
            tokio::spawn(async move { gate_clone.check(&tool_clone, "action".into()).await });

        let req = rx.recv().await.unwrap();
        gate.respond(&req.id, ConsentResponse::Denied { reason: None })
            .await;

        let err = handle.await.unwrap().unwrap_err();
        match err {
            ConsentError::Denied { reason, .. } => {
                assert_eq!(reason, "user denied");
            }
            other => panic!("expected Denied, got {other:?}"),
        }
    }

    #[tokio::test(start_paused = true)]
    async fn test_consent_timeout() {
        let ttl = Duration::from_secs(5);
        let (gate, _rx) = ConsentGate::new(RiskLevel::High, ttl, 5);
        let tool = make_tool_info("shell", RiskLevel::High);

        let result = gate.check(&tool, "action".into()).await;

        match result.unwrap_err() {
            ConsentError::Expired { tool, timeout_secs } => {
                assert_eq!(tool, "shell");
                assert_eq!(timeout_secs, 5);
            }
            other => panic!("expected Expired, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_consent_request_fields() {
        let (gate, mut rx) = ConsentGate::new(RiskLevel::High, Duration::from_secs(30), 5);
        let gate = Arc::new(gate);
        let tool = make_tool_info("network_request", RiskLevel::High);

        let gate_clone = Arc::clone(&gate);
        let tool_clone = tool.clone();
        let handle = tokio::spawn(async move {
            gate_clone
                .check(&tool_clone, "GET https://example.com".into())
                .await
        });

        let req = rx.recv().await.unwrap();
        assert_eq!(req.tool_name, "network_request");
        assert_eq!(req.description, "network_request tool");
        assert_eq!(req.risk_level, RiskLevel::High);
        assert_eq!(req.action_summary, "GET https://example.com");
        assert!(req.requested_at < req.expires_at);
        assert!(!req.id.is_empty());

        // Clean up by approving.
        gate.respond(&req.id, ConsentResponse::Approved).await;
        handle.await.unwrap().unwrap();
    }

    // ── Cleanup and edge cases ─────────────────────────────────────

    #[tokio::test(start_paused = true)]
    async fn test_timeout_cleans_up_pending() {
        let ttl = Duration::from_secs(1);
        let (gate, _rx) = ConsentGate::new(RiskLevel::High, ttl, 5);
        let tool = make_tool_info("shell", RiskLevel::High);

        let _ = gate.check(&tool, "action".into()).await;

        assert_eq!(gate.pending_count().await, 0);
    }

    #[tokio::test]
    async fn test_sender_drop_cleans_up_pending() {
        let (gate, rx) = ConsentGate::new(RiskLevel::High, Duration::from_secs(60), 5);
        let tool = make_tool_info("shell", RiskLevel::High);

        // Drop the receiver — the mpsc send will still succeed because
        // there's buffer space, but the oneshot sender will be dropped
        // when nothing responds.
        drop(rx);

        let result = gate.check(&tool, "action".into()).await;

        // Should get ChannelClosed since receiver was dropped.
        assert!(matches!(result, Err(ConsentError::ChannelClosed)));
        assert_eq!(gate.pending_count().await, 0);
    }

    #[tokio::test]
    async fn test_respond_unknown_id_returns_false() {
        let (gate, _rx) = ConsentGate::new(RiskLevel::High, Duration::from_secs(60), 5);

        let delivered = gate.respond("nonexistent", ConsentResponse::Approved).await;
        assert!(!delivered);
    }

    #[tokio::test(start_paused = true)]
    async fn test_respond_after_timeout_returns_false() {
        let ttl = Duration::from_secs(1);
        let (gate, mut rx) = ConsentGate::new(RiskLevel::High, ttl, 5);
        let gate = Arc::new(gate);
        let tool = make_tool_info("shell", RiskLevel::High);

        let gate_clone = Arc::clone(&gate);
        let tool_clone = tool.clone();
        let handle =
            tokio::spawn(async move { gate_clone.check(&tool_clone, "action".into()).await });

        // Grab the request but don't respond — let it timeout.
        let req = rx.recv().await.unwrap();

        // Wait for check to complete via timeout.
        let result = handle.await.unwrap();
        assert!(matches!(result, Err(ConsentError::Expired { .. })));

        // Late response should return false.
        let delivered = gate.respond(&req.id, ConsentResponse::Approved).await;
        assert!(!delivered);
    }

    #[tokio::test]
    async fn test_closed_channel_returns_error_and_cleans_up() {
        let (gate, rx) = ConsentGate::new(RiskLevel::High, Duration::from_secs(60), 5);
        let tool = make_tool_info("shell", RiskLevel::High);

        drop(rx);

        let result = gate.check(&tool, "action".into()).await;
        assert!(matches!(result, Err(ConsentError::ChannelClosed)));
        assert_eq!(gate.pending_count().await, 0);
    }

    // ── Rate limiting ──────────────────────────────────────────────

    #[tokio::test]
    async fn test_max_pending_enforced() {
        let (gate, mut rx) = ConsentGate::new(RiskLevel::High, Duration::from_secs(600), 2);
        let gate = Arc::new(gate);
        let tool = make_tool_info("shell", RiskLevel::High);

        // Start 2 pending checks (they will block waiting for responses).
        let gate1 = Arc::clone(&gate);
        let t1 = tool.clone();
        let _h1 = tokio::spawn(async move { gate1.check(&t1, "a1".into()).await });

        let gate2 = Arc::clone(&gate);
        let t2 = tool.clone();
        let _h2 = tokio::spawn(async move { gate2.check(&t2, "a2".into()).await });

        // Wait for both requests to arrive (proves they registered in pending).
        let _r1 = rx.recv().await.unwrap();
        let _r2 = rx.recv().await.unwrap();

        // 3rd should fail with TooManyPending.
        let result = gate.check(&tool, "a3".into()).await;
        match result.unwrap_err() {
            ConsentError::TooManyPending { tool, max } => {
                assert_eq!(tool, "shell");
                assert_eq!(max, 2);
            }
            other => panic!("expected TooManyPending, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_pending_cleared_after_response_allows_new_request() {
        let (gate, mut rx) = ConsentGate::new(RiskLevel::High, Duration::from_secs(60), 1);
        let gate = Arc::new(gate);
        let tool = make_tool_info("shell", RiskLevel::High);

        // First request.
        let gate_clone = Arc::clone(&gate);
        let t = tool.clone();
        let h = tokio::spawn(async move { gate_clone.check(&t, "first".into()).await });

        let req = rx.recv().await.unwrap();
        gate.respond(&req.id, ConsentResponse::Approved).await;
        h.await.unwrap().unwrap();

        // Second request should succeed (slot is free).
        let gate_clone = Arc::clone(&gate);
        let t = tool.clone();
        let h = tokio::spawn(async move { gate_clone.check(&t, "second".into()).await });

        let req = rx.recv().await.unwrap();
        gate.respond(&req.id, ConsentResponse::Approved).await;
        h.await.unwrap().unwrap();
    }

    // ── Concurrent requests ────────────────────────────────────────

    #[tokio::test]
    async fn test_concurrent_requests_independent() {
        let (gate, mut rx) = ConsentGate::new(RiskLevel::High, Duration::from_secs(60), 5);
        let gate = Arc::new(gate);

        let tool_a = make_tool_info("shell_a", RiskLevel::High);
        let tool_b = make_tool_info("shell_b", RiskLevel::High);

        let ga = Arc::clone(&gate);
        let ta = tool_a.clone();
        let ha = tokio::spawn(async move { ga.check(&ta, "action a".into()).await });

        let gb = Arc::clone(&gate);
        let tb = tool_b.clone();
        let hb = tokio::spawn(async move { gb.check(&tb, "action b".into()).await });

        // Receive both requests.
        let r1 = rx.recv().await.unwrap();
        let r2 = rx.recv().await.unwrap();

        // Approve one, deny the other (order may vary).
        let (approve_id, deny_id) = if r1.tool_name == "shell_a" {
            (r1.id, r2.id)
        } else {
            (r2.id, r1.id)
        };

        gate.respond(&approve_id, ConsentResponse::Approved).await;
        gate.respond(
            &deny_id,
            ConsentResponse::Denied {
                reason: Some("nope".into()),
            },
        )
        .await;

        let ra = ha.await.unwrap();
        let rb = hb.await.unwrap();

        // shell_a was approved, shell_b was denied.
        assert!(ra.is_ok());
        assert!(matches!(rb, Err(ConsentError::Denied { .. })));
    }

    // ── ConsentError Display ───────────────────────────────────────

    #[test]
    fn test_consent_error_display_messages() {
        let denied = ConsentError::Denied {
            tool: "shell".into(),
            reason: "too risky".into(),
        };
        assert_eq!(
            denied.to_string(),
            "consent denied for tool `shell`: too risky"
        );

        let expired = ConsentError::Expired {
            tool: "shell".into(),
            timeout_secs: 60,
        };
        assert_eq!(
            expired.to_string(),
            "consent request for tool `shell` expired after 60s"
        );

        let too_many = ConsentError::TooManyPending {
            tool: "shell".into(),
            max: 5,
        };
        assert_eq!(
            too_many.to_string(),
            "too many pending consent requests (5); denying tool `shell`"
        );

        let closed = ConsentError::ChannelClosed;
        assert_eq!(closed.to_string(), "consent channel closed");
    }

    // ── Serde roundtrips ───────────────────────────────────────────

    #[test]
    fn test_consent_request_serde_roundtrip() {
        let now = Utc::now();
        let req = ConsentRequest {
            id: "abc-123".into(),
            tool_name: "shell".into(),
            description: "execute shell commands".into(),
            risk_level: RiskLevel::High,
            action_summary: "rm -rf /tmp/test".into(),
            requested_at: now,
            expires_at: now + TimeDelta::seconds(60),
        };

        let json = serde_json::to_string(&req).unwrap();
        let back: ConsentRequest = serde_json::from_str(&json).unwrap();

        assert_eq!(back.id, req.id);
        assert_eq!(back.tool_name, req.tool_name);
        assert_eq!(back.description, req.description);
        assert_eq!(back.risk_level, req.risk_level);
        assert_eq!(back.action_summary, req.action_summary);
        assert_eq!(back.requested_at, req.requested_at);
        assert_eq!(back.expires_at, req.expires_at);
    }

    #[test]
    fn test_consent_response_serde_roundtrip() {
        let approved = ConsentResponse::Approved;
        let json = serde_json::to_string(&approved).unwrap();
        let back: ConsentResponse = serde_json::from_str(&json).unwrap();
        assert!(matches!(back, ConsentResponse::Approved));

        let denied = ConsentResponse::Denied {
            reason: Some("too dangerous".into()),
        };
        let json = serde_json::to_string(&denied).unwrap();
        let back: ConsentResponse = serde_json::from_str(&json).unwrap();
        match back {
            ConsentResponse::Denied { reason } => {
                assert_eq!(reason.unwrap(), "too dangerous");
            }
            ConsentResponse::Approved => panic!("expected Denied"),
        }
    }
}
