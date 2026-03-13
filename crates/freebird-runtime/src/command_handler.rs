//! Command dispatcher for `/command` events.
//!
//! Handles slash commands (`/quit`, `/new`, `/help`) received from the
//! channel and returns a [`LoopAction`] indicating whether the event loop
//! should continue or exit.

use std::path::Path;

use freebird_traits::channel::OutboundEvent;
use freebird_types::config::BudgetConfig;
use tokio::sync::mpsc;

use crate::session::SessionManager;

/// Controls the event loop after handling an event.
///
/// Private — callers of `run()` see `Result<(), RuntimeError>`.
pub enum LoopAction {
    /// Continue processing events.
    Continue,
    /// Exit the event loop gracefully (e.g., `/quit` command).
    Exit,
}

/// Borrowed context needed by command handlers.
pub struct CommandContext<'a> {
    pub channel_id: &'a str,
    pub sessions: &'a SessionManager,
    pub budget_config: &'a BudgetConfig,
    pub sandbox_root: &'a Path,
    pub ttl_hours: u64,
    pub outbound: &'a mpsc::Sender<OutboundEvent>,
}

/// Dispatch a `/command` event.
///
/// Returns [`LoopAction::Exit`] only for `/quit`.
pub async fn handle_command(
    name: &str,
    _args: &[String],
    sender_id: &str,
    ctx: &CommandContext<'_>,
) -> LoopAction {
    match name {
        "quit" => {
            let _ = ctx
                .outbound
                .send(OutboundEvent::Message {
                    text: "Goodbye!".into(),
                    recipient_id: sender_id.into(),
                })
                .await;
            LoopAction::Exit
        }
        "new" => {
            let session_id = ctx.sessions.new_session(ctx.channel_id, sender_id).await;
            match ctx
                .sessions
                .initialize_session_state(
                    &session_id,
                    ctx.budget_config,
                    ctx.sandbox_root,
                    ctx.ttl_hours,
                    true,
                )
                .await
            {
                Ok(()) => {
                    let _ = ctx
                        .outbound
                        .send(OutboundEvent::Message {
                            text: format!("New session started: {session_id}"),
                            recipient_id: sender_id.into(),
                        })
                        .await;
                }
                Err(e) => {
                    tracing::error!(error = %e, "cannot create capability grant for new session");
                    let _ = ctx
                        .outbound
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
            let _ = ctx
                .outbound
                .send(OutboundEvent::Message {
                    text: help_text,
                    recipient_id: sender_id.into(),
                })
                .await;
            LoopAction::Continue
        }
        unknown => {
            let _ = ctx
                .outbound
                .send(OutboundEvent::Error {
                    text: format!("Unknown command: /{unknown}"),
                    recipient_id: sender_id.into(),
                })
                .await;
            LoopAction::Continue
        }
    }
}
