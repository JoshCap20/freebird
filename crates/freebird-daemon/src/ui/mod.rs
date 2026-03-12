//! Interactive TUI chat — Claude Code-inspired terminal interface.
//!
//! Uses crossterm for raw mode terminal control. The architecture is:
//! - Scrolling output above (native scrollback preserved)
//! - Fixed input area at the bottom (redrawn in place)
//! - Spinner animation for tool execution
//! - Inline token usage and session info display

pub mod completion;
pub mod consent;
pub mod input;
pub mod output;
pub mod raw_writer;
pub mod spinner;
pub mod status;
pub mod theme;

use std::io::{self, Write};
use std::time::Duration;

use anyhow::{Context, Result};
use crossterm::event::{Event, EventStream, KeyEvent};
use crossterm::terminal;
use crossterm::{cursor, execute};
use futures::StreamExt;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::net::TcpStream;

use freebird_types::protocol::{ClientMessage, ServerMessage};

/// Sentinel value returned by [`parse_approval_category`] when the category
/// is a `security_warning` kind. Used to drive display branching in
/// [`render_consent_header`] without comparing against a raw string literal
/// at the call site.
const RISK_LEVEL_SECURITY_WARNING: &str = "warning";

/// Sentinel value returned by [`parse_approval_category`] when the category
/// is a `budget_exceeded` kind.
const RISK_LEVEL_BUDGET_EXCEEDED: &str = "budget";

use self::consent::{ConsentAction, ConsentSelector};
use self::input::{InputAction, InputEditor};
use self::output::OutputRenderer;
use self::raw_writer::RawWriter;
use self::spinner::ToolSpinner;
use self::status::StatusBar;

/// Interval for spinner animation ticks.
const SPINNER_TICK_MS: u64 = 80;

/// The interactive TUI chat session.
pub struct TtyChat {
    /// Raw mode writer that translates `\n` → `\r\n`.
    writer: RawWriter,
    /// Input editor with history and multi-line support.
    input: InputEditor,
    /// Output renderer for server responses.
    output: OutputRenderer,
    /// Tool execution spinner.
    spinner: ToolSpinner,
    /// Session/token info display.
    status: StatusBar,
    /// Active consent selector (None when in normal input mode).
    consent: Option<ConsentSelector>,
}

impl TtyChat {
    /// Create a new TUI chat session.
    fn new() -> Result<Self> {
        let (width, _height) = terminal::size().context("failed to query terminal size")?;

        Ok(Self {
            writer: RawWriter::new(),
            input: InputEditor::new(width),
            output: OutputRenderer::new(),
            spinner: ToolSpinner::new(),
            status: StatusBar::new(),
            consent: None,
        })
    }

    /// Run the interactive chat session over the given TCP stream.
    ///
    /// Takes ownership of the stream. Enables raw mode, runs the event loop,
    /// and restores the terminal on exit (including panic).
    pub async fn run(stream: TcpStream) -> Result<()> {
        // Install panic hook to restore terminal on panic.
        let original_hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            let _ = terminal::disable_raw_mode();
            let _ = execute!(io::stdout(), cursor::Show);
            original_hook(info);
        }));

        // Enable raw mode for full key control.
        terminal::enable_raw_mode().context("failed to enable raw mode")?;

        let result = {
            let mut chat = Self::new()?;
            chat.event_loop(stream).await
        };

        // Always restore terminal state.
        let _ = terminal::disable_raw_mode();
        let _ = execute!(io::stdout(), cursor::Show);

        result
    }

    /// The main event loop — races keyboard events, socket messages, and spinner ticks.
    async fn event_loop(&mut self, stream: TcpStream) -> Result<()> {
        let (socket_read, mut socket_write) = stream.into_split();
        let socket_reader = BufReader::new(socket_read);
        let mut socket_lines = socket_reader.lines();

        let mut key_events = EventStream::new();

        // Draw the initial prompt.
        writeln!(self.writer)?;
        self.input.render(&mut self.writer)?;

        let spinner_interval = tokio::time::interval(Duration::from_millis(SPINNER_TICK_MS));
        tokio::pin!(spinner_interval);

        loop {
            tokio::select! {
                // ── Keyboard events ──────────────────────────────────────
                maybe_event = key_events.next() => {
                    let Some(event_result) = maybe_event else {
                        break; // Event stream closed
                    };

                    let event = event_result.context("reading terminal event")?;

                    match event {
                        Event::Key(key_event) => {
                            match self.handle_key_event(key_event, &mut socket_write).await? {
                                LoopControl::Continue => {}
                                LoopControl::Exit => break,
                            }
                        }
                        Event::Resize(width, _height) => {
                            self.input.set_term_width(width);
                            self.input.render(&mut self.writer)?;
                        }
                        _ => {}
                    }
                }

                // ── Server messages ──────────────────────────────────────
                line = socket_lines.next_line() => {
                    if let Some(line) = line.context("reading from daemon")? {
                        // handle_server_line may return a pending message to send
                        // (e.g., auto-deny of a superseded consent request).
                        if let Some(pending) = self.handle_server_line(&line)? {
                            crate::chat::send_client_message(&mut socket_write, &pending).await?;
                        }
                    } else {
                        // Server disconnected.
                        self.save_input_area()?;
                        writeln!(self.writer)?;
                        theme::write_error_styled(
                            &mut self.writer,
                            "daemon disconnected \u{2014} run 'freebird serve' to restart"
                        )?;
                        break;
                    }
                }

                // ── Spinner / consent expiry tick ─────────────────────────
                _ = spinner_interval.tick(), if self.spinner.is_active() || self.consent.is_some() => {
                    if self.spinner.is_active() {
                        self.spinner.tick(&mut self.writer)?;
                    }
                    // Check consent expiry on each tick (~80ms granularity).
                    if self.consent.as_ref().is_some_and(ConsentSelector::is_expired) {
                        if let Some(sel) = self.consent.take() {
                            if let Some(msg) = Self::consent_action_to_message(sel.auto_deny("timeout")) {
                                crate::chat::send_client_message(&mut socket_write, &msg).await?;
                            }
                            ConsentSelector::clear(&mut self.writer)?;
                            sel.render_outcome(&mut self.writer, false)?;
                            self.input.render(&mut self.writer)?;
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Handle a keyboard event, returning whether to continue or exit.
    async fn handle_key_event<W: tokio::io::AsyncWrite + Unpin>(
        &mut self,
        key_event: KeyEvent,
        socket_write: &mut W,
    ) -> Result<LoopControl> {
        // If consent selector is active, route key events there instead.
        if let Some(sel) = &mut self.consent {
            let consent_action = sel.handle_key(key_event);
            match consent_action {
                ConsentAction::Redraw => {
                    // Temporarily take the selector to avoid borrow conflict
                    // with save_input_area (which borrows &mut self).
                    if let Some(sel) = self.consent.take() {
                        ConsentSelector::clear(&mut self.writer)?;
                        sel.render(&mut self.writer)?;
                        self.consent = Some(sel);
                    }
                }
                ConsentAction::Confirmed {
                    request_id,
                    approved,
                    reason,
                } => {
                    // Clear selector lines and show a collapsed outcome summary.
                    if let Some(sel) = self.consent.take() {
                        ConsentSelector::clear(&mut self.writer)?;
                        sel.render_outcome(&mut self.writer, approved)?;
                    }
                    let msg = ClientMessage::ApprovalResponse {
                        request_id,
                        approved,
                        reason,
                    };
                    crate::chat::send_client_message(socket_write, &msg).await?;
                    self.input.render(&mut self.writer)?;
                }
                ConsentAction::None => {}
            }
            return Ok(LoopControl::Continue);
        }

        let action = self.input.handle_key(key_event);

        match action {
            InputAction::Submit(text) => {
                // Clear the input area and echo the user's message permanently
                // in the scrollback (like Claude Code / ChatGPT).
                self.save_input_area()?;
                let trimmed = text.trim().to_string();
                self.echo_user_input(&trimmed)?;

                let parse_result = crate::chat::parse_user_input(&trimmed);

                match parse_result {
                    crate::chat::ParseResult::Quit => {
                        crate::chat::send_client_message(socket_write, &ClientMessage::Disconnect)
                            .await?;
                        return Ok(LoopControl::Exit);
                    }
                    crate::chat::ParseResult::LocalOutput(output_text) => {
                        write!(self.writer, "{output_text}")?;
                        self.writer.flush()?;
                        writeln!(self.writer)?;
                        self.input.render(&mut self.writer)?;
                    }
                    crate::chat::ParseResult::Send(msg) => {
                        crate::chat::send_client_message(socket_write, &msg).await?;
                        // Don't redraw input — wait for server response
                    }
                }
            }
            InputAction::Redraw => {
                self.input.render(&mut self.writer)?;
            }
            InputAction::Quit => {
                crate::chat::send_client_message(socket_write, &ClientMessage::Disconnect).await?;
                return Ok(LoopControl::Exit);
            }
            InputAction::None => {}
        }

        Ok(LoopControl::Continue)
    }

    /// Handle a JSON-line from the server.
    ///
    /// Returns an optional `ClientMessage` that must be sent to the daemon
    /// (used for auto-deny of superseded consent requests).
    fn handle_server_line(&mut self, line: &str) -> Result<Option<ClientMessage>> {
        let msg: ServerMessage = match serde_json::from_str(line) {
            Ok(msg) => msg,
            Err(e) => {
                self.save_input_area()?;
                theme::write_error_styled(&mut self.writer, &format!("protocol error: {e}"))?;
                self.input.render(&mut self.writer)?;
                return Ok(None);
            }
        };
        let mut pending_send: Option<ClientMessage> = None;

        match msg {
            ServerMessage::Message { text } => {
                self.save_input_area()?;
                self.output.write_message(&mut self.writer, &text)?;
            }
            ServerMessage::StreamChunk { text } => {
                if !self.output.is_streaming() {
                    self.save_input_area()?;
                }
                self.output.write_stream_chunk(&mut self.writer, &text)?;
            }
            ServerMessage::StreamEnd => {
                self.output.write_stream_end(&mut self.writer)?;
            }
            ServerMessage::Error { text } => {
                self.save_input_area()?;
                self.output.write_error(&mut self.writer, &text)?;
            }
            ServerMessage::CommandResponse { text } => {
                self.save_input_area()?;
                self.output
                    .write_command_response(&mut self.writer, &text)?;
            }
            ServerMessage::ToolStart { tool_name } => {
                self.save_input_area()?;
                writeln!(self.writer)?;
                self.spinner.start(&mut self.writer, &tool_name)?;
            }
            ServerMessage::ToolEnd {
                tool_name,
                outcome,
                duration_ms,
            } => {
                self.save_input_area()?;
                self.spinner
                    .stop(&mut self.writer, &tool_name, &outcome, duration_ms)?;
            }
            ServerMessage::TurnComplete => {
                self.output.turn_complete();
                writeln!(self.writer)?;
                self.input.render(&mut self.writer)?;
            }
            ServerMessage::TokenUsage {
                input_tokens,
                output_tokens,
                ..
            } => {
                self.save_input_area()?;
                self.status
                    .show_token_usage(&mut self.writer, input_tokens, output_tokens)?;
                self.input.render(&mut self.writer)?;
            }
            ServerMessage::SessionInfo {
                session_id,
                model_id,
                ..
            } => {
                self.save_input_area()?;
                self.status
                    .set_session_info(&mut self.writer, &model_id, &session_id)?;
                self.input.render(&mut self.writer)?;
            }
            ServerMessage::ApprovalRequest {
                request_id,
                category_json,
                expires_at,
            } => {
                // Stop the spinner — the tool is paused awaiting approval.
                if self.spinner.is_active() {
                    self.spinner.pause(&mut self.writer)?;
                }
                // If there's already a pending approval, auto-deny it.
                if let Some(prev) = self.consent.take() {
                    pending_send = Self::consent_action_to_message(prev.auto_deny("superseded"));
                }
                self.save_input_area()?;
                // Parse category_json to extract display info for
                // polymorphic rendering (Consent vs SecurityWarning).
                let (display_name, action_summary, risk_level) =
                    Self::parse_approval_category(&category_json);
                self.render_consent_header(&display_name, &action_summary, &risk_level)?;
                // Create the interactive selector (falls back to text hint if expired/unparseable).
                if let Some(sel) =
                    ConsentSelector::new(request_id.clone(), display_name.clone(), &expires_at)
                {
                    sel.render(&mut self.writer)?;
                    self.consent = Some(sel);
                } else {
                    // Expired or unparseable — show the old text-based hint as fallback.
                    self.render_consent_fallback(&request_id)?;
                    self.input.render(&mut self.writer)?;
                }
            }
        }

        Ok(pending_send)
    }

    /// Render the approval request header with polymorphic styling.
    ///
    /// - Consent requests: yellow "CONSENT REQUIRED" with tool name and risk level
    /// - Security warnings: red "SECURITY WARNING" with threat details
    /// - Budget exceeded: yellow "BUDGET EXCEEDED" with resource and usage
    ///
    /// The interactive selector or text-based fallback is rendered separately.
    fn render_consent_header(
        &mut self,
        display_name: &str,
        action_summary: &str,
        risk_level: &str,
    ) -> std::io::Result<()> {
        use crossterm::{
            queue,
            style::{Attribute, Color, ResetColor, SetAttribute, SetForegroundColor},
        };

        let is_security_warning = risk_level == RISK_LEVEL_SECURITY_WARNING;
        let is_budget_exceeded = risk_level == RISK_LEVEL_BUDGET_EXCEEDED;

        let (banner_color, banner_text) = if is_security_warning {
            (Color::Red, "SECURITY WARNING")
        } else if is_budget_exceeded {
            (Color::Yellow, "BUDGET EXCEEDED")
        } else {
            (Color::Yellow, "CONSENT REQUIRED")
        };

        writeln!(self.writer)?;
        queue!(
            self.writer,
            SetAttribute(Attribute::Bold),
            SetForegroundColor(banner_color),
        )?;
        write!(self.writer, "  {banner_text}")?;
        queue!(self.writer, ResetColor, SetAttribute(Attribute::Reset))?;

        writeln!(self.writer)?;
        queue!(
            self.writer,
            SetAttribute(Attribute::Dim),
            SetForegroundColor(Color::White),
        )?;
        if is_security_warning {
            write!(self.writer, "  Source: {display_name}")?;
        } else if is_budget_exceeded {
            write!(self.writer, "  Resource: {display_name}")?;
        } else {
            write!(self.writer, "  Tool: {display_name} (risk: {risk_level})")?;
        }
        queue!(self.writer, ResetColor, SetAttribute(Attribute::Reset))?;

        writeln!(self.writer)?;
        queue!(
            self.writer,
            SetAttribute(Attribute::Dim),
            SetForegroundColor(Color::White),
        )?;
        let label = if is_security_warning {
            "Detail"
        } else if is_budget_exceeded {
            "Usage"
        } else {
            "Action"
        };
        write!(self.writer, "  {label}: {action_summary}")?;
        queue!(self.writer, ResetColor, SetAttribute(Attribute::Reset))?;

        writeln!(self.writer)?;
        self.writer.flush()
    }

    /// Fallback text hint when the consent selector can't be created
    /// (expired or unparseable timestamp).
    fn render_consent_fallback(&mut self, request_id: &str) -> std::io::Result<()> {
        use crossterm::{
            queue,
            style::{Color, ResetColor, SetForegroundColor},
        };
        queue!(self.writer, SetForegroundColor(Color::DarkGrey))?;
        write!(
            self.writer,
            "  /approve {request_id}  or  /deny {request_id} [reason]"
        )?;
        queue!(self.writer, ResetColor)?;
        writeln!(self.writer)?;
        self.writer.flush()
    }

    /// Parse `category_json` into display fields for the approval header.
    ///
    /// Returns `(header_label, action_summary, risk_info)`.
    fn parse_approval_category(category_json: &str) -> (String, String, String) {
        // Use serde_json::Value to parse without needing serde derive.
        let Ok(val) = serde_json::from_str::<serde_json::Value>(category_json) else {
            return ("approval".into(), category_json.to_string(), "—".into());
        };

        let kind = val.get("kind").and_then(|v| v.as_str()).unwrap_or("");
        let str_field = |key: &str| {
            val.get(key)
                .and_then(|v| v.as_str())
                .unwrap_or("—")
                .to_string()
        };

        match kind {
            "consent" => {
                let tool_name = str_field("tool_name");
                let risk_level = str_field("risk_level");
                let action_summary = str_field("action_summary");
                (tool_name, action_summary, risk_level)
            }
            "security_warning" => {
                let threat_type = str_field("threat_type");
                let detected_pattern = str_field("detected_pattern");
                let content_preview = str_field("content_preview");
                let source = str_field("source");

                let header = format!("security: {threat_type} ({source})");
                let summary = format!("Pattern: {detected_pattern} \u{2014} {content_preview}");
                // Truncate summary to 200 chars for display.
                let summary = if summary.chars().count() > 200 {
                    let truncated: String = summary.chars().take(197).collect();
                    format!("{truncated}...")
                } else {
                    summary
                };
                (header, summary, RISK_LEVEL_SECURITY_WARNING.into())
            }
            "budget_exceeded" => {
                let resource = str_field("resource");
                let used = val
                    .get("used")
                    .and_then(serde_json::Value::as_u64)
                    .unwrap_or(0);
                let limit = val
                    .get("limit")
                    .and_then(serde_json::Value::as_u64)
                    .unwrap_or(0);

                let header = format!("budget: {resource}");
                let summary = format!("Used {used}, limit {limit} \u{2014} approve to continue");
                (header, summary, RISK_LEVEL_BUDGET_EXCEEDED.into())
            }
            _ => ("approval".into(), category_json.to_string(), "—".into()),
        }
    }

    /// Convert a [`ConsentAction::Confirmed`] into a [`ClientMessage`].
    ///
    /// Returns `None` for non-Confirmed actions.
    fn consent_action_to_message(action: ConsentAction) -> Option<ClientMessage> {
        if let ConsentAction::Confirmed {
            request_id,
            approved,
            reason,
        } = action
        {
            Some(ClientMessage::ApprovalResponse {
                request_id,
                approved,
                reason,
            })
        } else {
            None
        }
    }

    /// Echo the user's submitted message into the scrollback so it stays
    /// visible above the bot's response.
    fn echo_user_input(&mut self, text: &str) -> std::io::Result<()> {
        theme::write_prompt_styled(&mut self.writer)?;
        writeln!(self.writer, "{text}")?;
        self.writer.flush()
    }

    /// Clear the full input area (including wrapped/multi-line rows) before
    /// writing output so that stale content doesn't remain on screen.
    fn save_input_area(&mut self) -> std::io::Result<()> {
        self.input.clear_visual_area(&mut self.writer)
    }
}

/// Loop control signal.
enum LoopControl {
    Continue,
    Exit,
}
