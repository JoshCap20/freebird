//! Interactive TUI chat — Claude Code-inspired terminal interface.
//!
//! Uses crossterm for raw mode terminal control. The architecture is:
//! - Scrolling output above (native scrollback preserved)
//! - Fixed input area at the bottom (redrawn in place)
//! - Spinner animation for tool execution
//! - Inline token usage and session info display

pub mod completion;
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
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::TcpStream;

use freebird_types::protocol::{ClientMessage, ServerMessage};

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
                        self.handle_server_line(&line)?;
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

                // ── Spinner tick ─────────────────────────────────────────
                _ = spinner_interval.tick(), if self.spinner.is_active() => {
                    self.spinner.tick(&mut self.writer)?;
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
        let action = self.input.handle_key(key_event);

        match action {
            InputAction::Submit(text) => {
                // Clear the input line and move to a new line for the response.
                self.save_input_area()?;

                let trimmed = text.trim().to_string();
                let parse_result = crate::chat::parse_user_input(&trimmed);

                match parse_result {
                    crate::chat::ParseResult::Quit => {
                        send_client_message_async(socket_write, &ClientMessage::Disconnect).await?;
                        return Ok(LoopControl::Exit);
                    }
                    crate::chat::ParseResult::LocalOutput(output_text) => {
                        write!(self.writer, "{output_text}")?;
                        self.writer.flush()?;
                        writeln!(self.writer)?;
                        self.input.render(&mut self.writer)?;
                    }
                    crate::chat::ParseResult::Send(msg) => {
                        send_client_message_async(socket_write, &msg).await?;
                        // Don't redraw input — wait for server response
                    }
                }
            }
            InputAction::Redraw => {
                self.input.render(&mut self.writer)?;
            }
            InputAction::Quit => {
                send_client_message_async(socket_write, &ClientMessage::Disconnect).await?;
                return Ok(LoopControl::Exit);
            }
            InputAction::None => {}
        }

        Ok(LoopControl::Continue)
    }

    /// Handle a JSON-line from the server.
    fn handle_server_line(&mut self, line: &str) -> Result<()> {
        let msg: ServerMessage = match serde_json::from_str(line) {
            Ok(msg) => msg,
            Err(e) => {
                self.save_input_area()?;
                theme::write_error_styled(&mut self.writer, &format!("protocol error: {e}"))?;
                self.input.render(&mut self.writer)?;
                return Ok(());
            }
        };

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
            ServerMessage::ConsentRequest {
                request_id,
                tool_name,
                action_summary,
                risk_level,
                ..
            } => {
                // Stop the spinner — the tool is paused awaiting consent.
                if self.spinner.is_active() {
                    self.spinner.pause(&mut self.writer)?;
                }
                self.save_input_area()?;
                self.render_consent_request(&request_id, &tool_name, &action_summary, &risk_level)?;
                self.input.render(&mut self.writer)?;
            }
        }

        Ok(())
    }

    /// Render a consent request with risk coloring and request ID.
    fn render_consent_request(
        &mut self,
        request_id: &str,
        tool_name: &str,
        action_summary: &str,
        risk_level: &str,
    ) -> std::io::Result<()> {
        use crossterm::{
            queue,
            style::{Attribute, Color, ResetColor, SetAttribute, SetForegroundColor},
        };
        writeln!(self.writer)?;
        queue!(
            self.writer,
            SetAttribute(Attribute::Bold),
            SetForegroundColor(Color::Yellow),
        )?;
        write!(self.writer, "  CONSENT REQUIRED")?;
        queue!(self.writer, ResetColor, SetAttribute(Attribute::Reset))?;

        writeln!(self.writer)?;
        queue!(
            self.writer,
            SetAttribute(Attribute::Dim),
            SetForegroundColor(Color::White),
        )?;
        write!(self.writer, "  Tool: {tool_name} (risk: {risk_level})")?;
        queue!(self.writer, ResetColor, SetAttribute(Attribute::Reset))?;

        writeln!(self.writer)?;
        queue!(
            self.writer,
            SetAttribute(Attribute::Dim),
            SetForegroundColor(Color::White),
        )?;
        write!(self.writer, "  Action: {action_summary}")?;
        queue!(self.writer, ResetColor, SetAttribute(Attribute::Reset))?;

        writeln!(self.writer)?;
        queue!(self.writer, SetForegroundColor(Color::DarkGrey))?;
        write!(
            self.writer,
            "  /approve {request_id}  or  /deny {request_id} [reason]"
        )?;
        queue!(self.writer, ResetColor)?;
        writeln!(self.writer)?;

        self.writer.flush()
    }

    /// Save the input area (clear the current prompt line) before writing output.
    fn save_input_area(&mut self) -> std::io::Result<()> {
        self.output.clear_input_line(&mut self.writer)
    }
}

/// Loop control signal.
enum LoopControl {
    Continue,
    Exit,
}

/// Send a `ClientMessage` as JSON over an async writer.
async fn send_client_message_async<W: tokio::io::AsyncWrite + Unpin>(
    writer: &mut W,
    msg: &ClientMessage,
) -> Result<()> {
    let json = serde_json::to_string(msg).context("serializing client message")?;
    writer
        .write_all(json.as_bytes())
        .await
        .context("writing to socket")?;
    writer.write_all(b"\n").await?;
    writer.flush().await.context("flushing socket")?;
    Ok(())
}
