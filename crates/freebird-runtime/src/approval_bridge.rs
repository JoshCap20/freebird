//! Approval bridge — routes approval events between the channel and the
//! [`ApprovalGate`](freebird_security::approval::ApprovalGate).
//!
//! Spawns two background tasks:
//!
//! 1. **Splitter**: reads from the raw inbound stream and routes
//!    `ApprovalResponse` events directly to the
//!    [`ApprovalResponder`](freebird_security::approval::ApprovalResponder),
//!    bypassing the main event loop. All other events are forwarded to the
//!    returned `main_rx`. This is necessary because the message handler may
//!    block on `check_consent()` (awaiting user approval via oneshot), and the
//!    `ApprovalResponse` that unblocks it also arrives as an `InboundEvent`.
//!
//! 2. **Forwarder**: reads `ApprovalRequest`s from the gate's mpsc channel and
//!    forwards them as `OutboundEvent::ApprovalRequest` to the user's channel.

use std::pin::Pin;

use freebird_security::approval::{ApprovalRequest, ApprovalResponder};
use freebird_traits::channel::{InboundEvent, OutboundEvent};
use futures::StreamExt;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

/// Handles returned by [`spawn`] — the caller must hold these to keep the
/// background tasks alive and abort them on shutdown.
pub struct ApprovalBridgeHandles {
    /// Receiver for non-approval inbound events.
    pub main_rx: mpsc::Receiver<InboundEvent>,
    /// The splitter task that routes `ApprovalResponse` events.
    pub splitter_task: JoinHandle<()>,
    /// The forwarder task (present only when an approval gate is configured).
    pub approval_task: Option<JoinHandle<()>>,
}

/// Spawn the approval bridge tasks.
///
/// See module-level docs for the responsibilities of each task.
pub fn spawn(
    inbound: Pin<Box<dyn futures::Stream<Item = InboundEvent> + Send>>,
    outbound: &mpsc::Sender<OutboundEvent>,
    cancel: &CancellationToken,
    approval_responder: Option<ApprovalResponder>,
    approval_rx: Option<mpsc::Receiver<ApprovalRequest>>,
) -> ApprovalBridgeHandles {
    let (main_tx, main_rx) = mpsc::channel::<InboundEvent>(32);
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

    let approval_task = approval_rx.map(|mut approval_rx| {
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

    ApprovalBridgeHandles {
        main_rx,
        splitter_task,
        approval_task,
    }
}
