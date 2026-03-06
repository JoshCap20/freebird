//! Chat client — thin TCP client for `freebird chat`.
//!
//! Connects to a running daemon, bridges user input/output to the JSON-line
//! protocol defined in [`freebird_types::protocol`].

use anyhow::{Context, Result};
use tokio::io::{AsyncBufRead, AsyncBufReadExt, AsyncWrite, AsyncWriteExt, BufReader};
use tokio::net::TcpStream;

use freebird_types::protocol::{ClientMessage, ServerMessage};

// ── ANSI styling ─────────────────────────────────────────────────────────────

/// Raw ANSI escape codes for terminal styling.
///
/// Only used when `is_tty == true` — non-tty output stays plain for test
/// determinism and piped usage.
mod style {
    pub const RESET: &str = "\x1b[0m";
    pub const BOLD: &str = "\x1b[1m";
    pub const DIM: &str = "\x1b[2m";
    pub const CYAN: &str = "\x1b[36m";
    pub const GREEN: &str = "\x1b[32m";
    pub const RED: &str = "\x1b[31m";
    pub const YELLOW: &str = "\x1b[33m";
    pub const MAGENTA: &str = "\x1b[35m";

    #[must_use]
    pub fn user_prompt() -> String {
        format!("{BOLD}{CYAN}You:{RESET} ")
    }

    #[must_use]
    pub fn bot_prefix() -> String {
        format!("{BOLD}{GREEN}Freebird:{RESET} ")
    }

    #[must_use]
    pub fn error_prefix(text: &str) -> String {
        format!("{BOLD}{RED}error:{RESET} {text}\n")
    }

    #[must_use]
    pub fn tool_start(name: &str) -> String {
        format!("{DIM}{YELLOW}  ⚙ {name}...{RESET}\n")
    }

    #[must_use]
    pub fn tool_end(name: &str, outcome: &str, ms: u64) -> String {
        format!("{DIM}{YELLOW}  ✓ {name} ({outcome}, {ms}ms){RESET}\n")
    }

    #[must_use]
    pub fn system_msg(text: &str) -> String {
        format!("{DIM}{MAGENTA}[{text}]{RESET}\n")
    }
}

// ── Help text ────────────────────────────────────────────────────────────────

/// Client-side help text printed by `/help` without a daemon round-trip.
const HELP_TEXT: &str = "\
Available commands:
  /help            Show this help message
  /new             Start a new conversation
  /quit, /exit     Disconnect from the daemon
";

// ── Input parsing ─────────────────────────────────────────────────────────────

/// The result of parsing a line of user input.
///
/// Separating local-only actions from network messages lets the loop decide
/// what to do without needing out-of-band return values.
#[derive(Debug, PartialEq, Eq)]
pub enum ParseResult {
    /// Send this message to the daemon.
    Send(ClientMessage),
    /// Print this text locally and do nothing else.
    LocalOutput(String),
    /// Disconnect gracefully.
    Quit,
}

/// Parse a line of user input into a [`ParseResult`].
///
/// - `/quit` and `/exit`  → [`ParseResult::Quit`]
/// - `/help`              → [`ParseResult::LocalOutput`] (no round-trip)
/// - `/`  (bare slash)    → [`ParseResult::LocalOutput`] with a usage hint
/// - `/cmd [args…]`       → [`ParseResult::Send`] with a `Command` message
/// - anything else        → [`ParseResult::Send`] with a `Message`
#[must_use]
pub fn parse_user_input(line: &str) -> ParseResult {
    match line {
        "/quit" | "/exit" => return ParseResult::Quit,
        "/help" => return ParseResult::LocalOutput(HELP_TEXT.to_string()),
        "/" => {
            return ParseResult::LocalOutput(
                "Unknown command '/'. Type /help for available commands.\n".to_string(),
            );
        }
        _ => {}
    }

    line.strip_prefix('/').map_or_else(
        || {
            ParseResult::Send(ClientMessage::Message {
                text: line.to_string(),
            })
        },
        |cmd| {
            let mut parts = cmd.splitn(2, ' ');
            // Infallible: splitn(2, ' ') always yields at least one element.
            let name = parts.next().unwrap_or_default().to_string();
            let args = parts
                .next()
                .map(|a| a.split_whitespace().map(String::from).collect())
                .unwrap_or_default();
            ParseResult::Send(ClientMessage::Command { name, args })
        },
    )
}

// ── Rendering ─────────────────────────────────────────────────────────────────

/// Render a [`ServerMessage`] as user-facing text.
///
/// Streaming chunks are returned as-is (no trailing newline) so they can be
/// printed incrementally. The caller is responsible for printing the
/// `Freebird: ` prefix before the first chunk of each response.
///
/// When `is_tty` is true, ANSI styling is applied to errors and tool events.
#[must_use]
pub fn render_server_message(msg: &ServerMessage, is_tty: bool) -> String {
    match msg {
        ServerMessage::Message { text } | ServerMessage::CommandResponse { text } => {
            format!("{text}\n")
        }
        ServerMessage::StreamChunk { text } => text.clone(),
        ServerMessage::StreamEnd => "\n".to_string(),
        ServerMessage::Error { text } => {
            if is_tty {
                style::error_prefix(text)
            } else {
                format!("error: {text}\n")
            }
        }
        ServerMessage::ToolStart { tool_name } => {
            if is_tty {
                style::tool_start(tool_name)
            } else {
                format!("[tool: {tool_name}...]\n")
            }
        }
        ServerMessage::ToolEnd {
            tool_name,
            outcome,
            duration_ms,
        } => {
            if is_tty {
                style::tool_end(tool_name, outcome, *duration_ms)
            } else {
                format!("[tool: {tool_name} {outcome} {duration_ms}ms]\n")
            }
        }
        // TurnComplete is a control signal — no visible output.
        ServerMessage::TurnComplete => String::new(),
    }
}

// ── Chat loop ─────────────────────────────────────────────────────────────────

/// Run the chat client loop with injectable I/O (for testing).
///
/// Reads lines from `user_input`, sends them as JSON-line [`ClientMessage`]s
/// to the daemon via `stream`, reads JSON-line [`ServerMessage`]s from the
/// daemon, and renders them to `user_output`.
///
/// The `is_tty` flag controls whether interactive prompts (`You: `,
/// `Freebird: `) are written. Pass `true` when connected to a real terminal
/// and `false` in tests / piped usage so assertions on `user_output` remain
/// deterministic.
pub async fn run_chat_with_io<I, O>(
    stream: TcpStream,
    user_input: I,
    mut user_output: O,
    is_tty: bool,
) -> Result<()>
where
    I: AsyncBufRead + Unpin,
    O: AsyncWrite + Unpin,
{
    let (socket_read, mut socket_write) = stream.into_split();
    let socket_reader = BufReader::new(socket_read);

    let mut input_lines = user_input.lines();
    let mut socket_lines = socket_reader.lines();

    // True while we are mid-stream (between first StreamChunk and StreamEnd).
    // Used to suppress the "Freebird: " prefix on every chunk after the first.
    let mut in_stream = false;

    // Print the first prompt before the loop so the user sees it immediately.
    write_prompt(&mut user_output, is_tty).await?;

    loop {
        tokio::select! {
            line = input_lines.next_line() => {
                if let Some(line) = line.context("reading user input")? {
                    let trimmed = line.trim().to_string();
                    if trimmed.is_empty() {
                        // Re-print prompt on blank input
                        write_prompt(&mut user_output, is_tty).await?;
                        continue;
                    }

                    match parse_user_input(&trimmed) {
                        ParseResult::Quit => {
                            send_client_message(&mut socket_write, &ClientMessage::Disconnect).await?;
                            return Ok(());
                        }
                        ParseResult::LocalOutput(text) => {
                            user_output.write_all(text.as_bytes()).await.context("writing local output")?;
                            user_output.flush().await.context("flushing local output")?;
                            // Re-print prompt after local output
                            write_prompt(&mut user_output, is_tty).await?;
                        }
                        ParseResult::Send(client_msg) => {
                            send_client_message(&mut socket_write, &client_msg).await?;
                        }
                    }
                } else {
                    // stdin EOF — disconnect gracefully
                    send_client_message(&mut socket_write, &ClientMessage::Disconnect).await?;
                    return Ok(());
                }
            }
            line = socket_lines.next_line() => {
                if let Some(line) = line.context("reading from daemon")? {
                    in_stream = handle_daemon_line(
                        &line, &mut user_output, is_tty, in_stream,
                    ).await?;
                } else {
                    // Server disconnected — tell the user and suggest next steps.
                    let notice = if is_tty {
                        style::system_msg("daemon disconnected — run 'freebird serve' to restart")
                    } else {
                        "[daemon disconnected]\nRun 'freebird serve' to start the daemon.\n"
                            .to_string()
                    };
                    user_output.write_all(notice.as_bytes()).await?;
                    user_output.flush().await?;
                    return Ok(());
                }
            }
        }
    }
}

// ── Wire helpers ──────────────────────────────────────────────────────────────

/// Process a single JSON-line from the daemon.
///
/// Returns the updated `in_stream` flag so the caller can track streaming state.
async fn handle_daemon_line<O: AsyncWrite + Unpin>(
    line: &str,
    user_output: &mut O,
    is_tty: bool,
    in_stream: bool,
) -> Result<bool> {
    let msg = match serde_json::from_str::<ServerMessage>(line) {
        Ok(msg) => msg,
        Err(e) => {
            let err_msg = format!("[protocol error: {e}]\n");
            user_output.write_all(err_msg.as_bytes()).await?;
            user_output.flush().await?;
            return Ok(in_stream);
        }
    };

    // Print "Freebird: " prefix before the first token of each response.
    // Tool events render their own prefix, so they skip this.
    let needs_prefix = is_tty
        && match &msg {
            ServerMessage::StreamChunk { .. } if !in_stream => true,
            ServerMessage::Message { .. }
            | ServerMessage::CommandResponse { .. }
            | ServerMessage::Error { .. } => true,
            _ => false,
        };

    if needs_prefix {
        // needs_prefix is only true when is_tty is true, so always use styled.
        let prefix = style::bot_prefix();
        user_output
            .write_all(prefix.as_bytes())
            .await
            .context("writing response prefix")?;
    }

    // Track streaming state.
    let new_in_stream = matches!(&msg, ServerMessage::StreamChunk { .. });

    // Write rendered content first, then prompt — order matters for StreamEnd
    // so the newline terminates the response before the next prompt appears.
    let rendered = render_server_message(&msg, is_tty);
    user_output
        .write_all(rendered.as_bytes())
        .await
        .context("writing to output")?;
    user_output.flush().await.context("flushing output")?;

    // Only print the "You: " prompt after TurnComplete — the server signals
    // when the entire agentic turn is done. Intermediate messages, stream ends,
    // and tool events do NOT trigger a prompt.
    if matches!(&msg, ServerMessage::TurnComplete) {
        write_prompt(user_output, is_tty).await?;
    }

    Ok(new_in_stream)
}

/// Write the user input prompt (styled when tty).
async fn write_prompt<O: AsyncWrite + Unpin>(output: &mut O, is_tty: bool) -> Result<()> {
    if !is_tty {
        return Ok(());
    }
    let prompt = style::user_prompt();
    output
        .write_all(prompt.as_bytes())
        .await
        .context("writing prompt")?;
    output.flush().await.context("flushing prompt")?;
    Ok(())
}

/// Send a [`ClientMessage`] as a JSON line over the socket.
pub async fn send_client_message<W: AsyncWrite + Unpin>(
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

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::indexing_slicing)]
mod tests {
    use super::*;
    use std::time::Duration;
    use tokio::io::{AsyncReadExt, BufReader, duplex};
    use tokio::net::TcpListener;
    use tokio::sync::mpsc;
    use tokio::time::timeout;

    const TEST_TIMEOUT: Duration = Duration::from_secs(5);

    async fn bind_random() -> (TcpListener, u16) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        (listener, port)
    }

    // ── parse_user_input unit tests ────────────────────────────────────

    #[test]
    fn parse_plain_message() {
        assert_eq!(
            parse_user_input("hello world"),
            ParseResult::Send(ClientMessage::Message {
                text: "hello world".into()
            })
        );
    }

    #[test]
    fn parse_command_no_args() {
        assert_eq!(
            parse_user_input("/new"),
            ParseResult::Send(ClientMessage::Command {
                name: "new".into(),
                args: vec![]
            })
        );
    }

    #[test]
    fn parse_command_with_args() {
        assert_eq!(
            parse_user_input("/model opus fast"),
            ParseResult::Send(ClientMessage::Command {
                name: "model".into(),
                args: vec!["opus".into(), "fast".into()]
            })
        );
    }

    #[test]
    fn parse_quit_returns_quit() {
        assert_eq!(parse_user_input("/quit"), ParseResult::Quit);
        assert_eq!(parse_user_input("/exit"), ParseResult::Quit);
    }

    /// Bare `/` used to produce `Command { name: "", args: [] }` — now it
    /// returns a local usage hint instead of sending a malformed command.
    #[test]
    fn parse_bare_slash_returns_local_output() {
        let result = parse_user_input("/");
        assert!(
            matches!(result, ParseResult::LocalOutput(_)),
            "bare '/' should produce LocalOutput, got {result:?}"
        );
        if let ParseResult::LocalOutput(text) = result {
            assert!(
                text.contains("/help"),
                "usage hint should mention /help, got: {text}"
            );
        }
    }

    #[test]
    fn parse_help_returns_local_output() {
        let result = parse_user_input("/help");
        assert!(
            matches!(result, ParseResult::LocalOutput(_)),
            "/help should produce LocalOutput, got {result:?}"
        );
        if let ParseResult::LocalOutput(text) = result {
            assert!(text.contains("/quit"), "help text should mention /quit");
            assert!(text.contains("/new"), "help text should mention /new");
        }
    }

    // ── render_server_message unit tests ───────────────────────────────

    #[test]
    fn render_message() {
        let msg = ServerMessage::Message { text: "hi".into() };
        assert_eq!(render_server_message(&msg, false), "hi\n");
    }

    #[test]
    fn render_stream_chunk() {
        let msg = ServerMessage::StreamChunk {
            text: "partial".into(),
        };
        assert_eq!(render_server_message(&msg, false), "partial");
    }

    #[test]
    fn render_stream_end() {
        assert_eq!(
            render_server_message(&ServerMessage::StreamEnd, false),
            "\n"
        );
    }

    #[test]
    fn render_error() {
        let msg = ServerMessage::Error {
            text: "boom".into(),
        };
        assert_eq!(render_server_message(&msg, false), "error: boom\n");
    }

    #[test]
    fn render_command_response() {
        let msg = ServerMessage::CommandResponse { text: "ok".into() };
        assert_eq!(render_server_message(&msg, false), "ok\n");
    }

    #[test]
    fn render_tool_start_plain() {
        let msg = ServerMessage::ToolStart {
            tool_name: "read_file".into(),
        };
        assert_eq!(render_server_message(&msg, false), "[tool: read_file...]\n");
    }

    #[test]
    fn render_tool_end_plain() {
        let msg = ServerMessage::ToolEnd {
            tool_name: "read_file".into(),
            outcome: "success".into(),
            duration_ms: 42,
        };
        assert_eq!(
            render_server_message(&msg, false),
            "[tool: read_file success 42ms]\n"
        );
    }

    #[test]
    fn render_tool_start_tty_has_ansi() {
        let msg = ServerMessage::ToolStart {
            tool_name: "shell".into(),
        };
        let rendered = render_server_message(&msg, true);
        assert!(
            rendered.contains("\x1b["),
            "tty tool_start should contain ANSI codes"
        );
        assert!(rendered.contains("shell"), "should contain tool name");
    }

    #[test]
    fn render_error_tty_has_ansi() {
        let msg = ServerMessage::Error {
            text: "boom".into(),
        };
        let rendered = render_server_message(&msg, true);
        assert!(
            rendered.contains("\x1b["),
            "tty error should contain ANSI codes"
        );
        assert!(rendered.contains("boom"), "should contain error text");
    }

    #[test]
    fn render_tool_end_tty_has_ansi() {
        let msg = ServerMessage::ToolEnd {
            tool_name: "read_file".into(),
            outcome: "success".into(),
            duration_ms: 42,
        };
        let rendered = render_server_message(&msg, true);
        assert!(
            rendered.contains("\x1b["),
            "tty tool_end should contain ANSI codes"
        );
        assert!(rendered.contains("read_file"), "should contain tool name");
        assert!(rendered.contains("42ms"), "should contain duration");
    }

    #[test]
    fn render_turn_complete_returns_empty() {
        assert_eq!(
            render_server_message(&ServerMessage::TurnComplete, false),
            ""
        );
        assert_eq!(
            render_server_message(&ServerMessage::TurnComplete, true),
            ""
        );
    }

    // ── Integration tests (is_tty = false) ────────────────────────────
    //
    // All integration tests pass `is_tty = false` so prompts are never written
    // to `user_output`, keeping assertions on output content deterministic.

    #[tokio::test]
    async fn stdin_eof_sends_disconnect() {
        let (listener, port) = bind_random().await;
        let (received_tx, mut received_rx) = mpsc::channel::<ClientMessage>(16);

        tokio::spawn(async move {
            let (stream, _) = listener.accept().await.unwrap();
            let (read_half, _write_half) = stream.into_split();
            let mut lines = BufReader::new(read_half).lines();
            while let Ok(Some(line)) = lines.next_line().await {
                if let Ok(msg) = serde_json::from_str::<ClientMessage>(&line) {
                    let _ = received_tx.send(msg).await;
                }
            }
        });

        let stream = TcpStream::connect(format!("127.0.0.1:{port}"))
            .await
            .unwrap();

        let input = BufReader::new(&b""[..]);
        let (output_writer, _output_reader) = duplex(4096);

        run_chat_with_io(stream, input, output_writer, false)
            .await
            .unwrap();

        let msg = timeout(TEST_TIMEOUT, received_rx.recv())
            .await
            .expect("timed out")
            .expect("channel closed");
        assert_eq!(msg, ClientMessage::Disconnect);
    }

    #[tokio::test]
    async fn quit_sends_disconnect() {
        let (listener, port) = bind_random().await;
        let (received_tx, mut received_rx) = mpsc::channel::<ClientMessage>(16);

        tokio::spawn(async move {
            let (stream, _) = listener.accept().await.unwrap();
            let (read_half, _write_half) = stream.into_split();
            let mut lines = BufReader::new(read_half).lines();
            while let Ok(Some(line)) = lines.next_line().await {
                if let Ok(msg) = serde_json::from_str::<ClientMessage>(&line) {
                    let _ = received_tx.send(msg).await;
                }
            }
        });

        let stream = TcpStream::connect(format!("127.0.0.1:{port}"))
            .await
            .unwrap();

        let input = BufReader::new(&b"/quit\n"[..]);
        let (output_writer, _output_reader) = duplex(4096);

        run_chat_with_io(stream, input, output_writer, false)
            .await
            .unwrap();

        let msg = timeout(TEST_TIMEOUT, received_rx.recv())
            .await
            .expect("timed out")
            .expect("channel closed");
        assert_eq!(msg, ClientMessage::Disconnect);
    }

    #[tokio::test]
    async fn message_sent_to_daemon() {
        let (listener, port) = bind_random().await;
        let (received_tx, mut received_rx) = mpsc::channel::<ClientMessage>(16);

        tokio::spawn(async move {
            let (stream, _) = listener.accept().await.unwrap();
            let (read_half, _write_half) = stream.into_split();
            let mut lines = BufReader::new(read_half).lines();
            while let Ok(Some(line)) = lines.next_line().await {
                if let Ok(msg) = serde_json::from_str::<ClientMessage>(&line) {
                    let _ = received_tx.send(msg).await;
                }
            }
        });

        let stream = TcpStream::connect(format!("127.0.0.1:{port}"))
            .await
            .unwrap();

        let input = BufReader::new(&b"hello\n"[..]);
        let (output_writer, _output_reader) = duplex(4096);

        run_chat_with_io(stream, input, output_writer, false)
            .await
            .unwrap();

        let msg1 = timeout(TEST_TIMEOUT, received_rx.recv())
            .await
            .expect("timed out")
            .expect("channel closed");
        assert_eq!(
            msg1,
            ClientMessage::Message {
                text: "hello".into()
            }
        );

        let msg2 = timeout(TEST_TIMEOUT, received_rx.recv())
            .await
            .expect("timed out")
            .expect("channel closed");
        assert_eq!(msg2, ClientMessage::Disconnect);
    }

    #[tokio::test]
    async fn server_message_rendered_to_output() {
        let (listener, port) = bind_random().await;

        tokio::spawn(async move {
            let (stream, _) = listener.accept().await.unwrap();
            let (_read_half, mut write_half) = stream.into_split();
            let msg = ServerMessage::Message {
                text: "hello from daemon".into(),
            };
            let json = serde_json::to_string(&msg).unwrap();
            write_half
                .write_all(format!("{json}\n").as_bytes())
                .await
                .unwrap();
            write_half.flush().await.unwrap();
            drop(write_half);
        });

        let stream = TcpStream::connect(format!("127.0.0.1:{port}"))
            .await
            .unwrap();

        let (_stdin_write, stdin_read) = duplex(1024);
        let input = BufReader::new(stdin_read);
        let (output_writer, mut output_reader) = duplex(4096);

        run_chat_with_io(stream, input, output_writer, false)
            .await
            .unwrap();

        let mut output = String::new();
        output_reader.read_to_string(&mut output).await.unwrap();
        assert!(
            output.contains("hello from daemon\n"),
            "expected daemon message in output, got: {output}"
        );
        assert!(
            output.contains("[daemon disconnected]"),
            "expected disconnect notice in output, got: {output}"
        );
    }

    /// Daemon disconnect message must now include the 'freebird serve' hint.
    #[tokio::test]
    async fn daemon_disconnect_includes_serve_hint() {
        let (listener, port) = bind_random().await;

        tokio::spawn(async move {
            let (stream, _) = listener.accept().await.unwrap();
            // Close immediately without sending anything.
            drop(stream);
        });

        let stream = TcpStream::connect(format!("127.0.0.1:{port}"))
            .await
            .unwrap();

        let (_stdin_write, stdin_read) = duplex(1024);
        let input = BufReader::new(stdin_read);
        let (output_writer, mut output_reader) = duplex(4096);

        run_chat_with_io(stream, input, output_writer, false)
            .await
            .unwrap();

        let mut output = String::new();
        output_reader.read_to_string(&mut output).await.unwrap();
        assert!(
            output.contains("freebird serve"),
            "disconnect message should mention 'freebird serve', got: {output}"
        );
    }

    /// `/help` must be handled locally — no message should be sent to the daemon.
    #[tokio::test]
    async fn help_command_is_local_no_daemon_message() {
        let (listener, port) = bind_random().await;
        let (received_tx, mut received_rx) = mpsc::channel::<ClientMessage>(16);

        tokio::spawn(async move {
            let (stream, _) = listener.accept().await.unwrap();
            let (read_half, _write_half) = stream.into_split();
            let mut lines = BufReader::new(read_half).lines();
            while let Ok(Some(line)) = lines.next_line().await {
                if let Ok(msg) = serde_json::from_str::<ClientMessage>(&line) {
                    let _ = received_tx.send(msg).await;
                }
            }
        });

        let stream = TcpStream::connect(format!("127.0.0.1:{port}"))
            .await
            .unwrap();

        // /help then /quit
        let input = BufReader::new(&b"/help\n/quit\n"[..]);
        let (output_writer, mut output_reader) = duplex(4096);

        run_chat_with_io(stream, input, output_writer, false)
            .await
            .unwrap();

        // Only message the daemon should have received is Disconnect
        let msg = timeout(TEST_TIMEOUT, received_rx.recv())
            .await
            .expect("timed out")
            .expect("channel closed");
        assert_eq!(
            msg,
            ClientMessage::Disconnect,
            "/help should not send anything to the daemon"
        );

        // Help text should appear in local output
        let mut output = String::new();
        output_reader.read_to_string(&mut output).await.unwrap();
        assert!(
            output.contains("/quit"),
            "help output should contain /quit, got: {output}"
        );
    }

    /// Bare `/` must produce a local usage hint, not send a command to the daemon.
    #[tokio::test]
    async fn bare_slash_is_local_no_daemon_message() {
        let (listener, port) = bind_random().await;
        let (received_tx, mut received_rx) = mpsc::channel::<ClientMessage>(16);

        tokio::spawn(async move {
            let (stream, _) = listener.accept().await.unwrap();
            let (read_half, _write_half) = stream.into_split();
            let mut lines = BufReader::new(read_half).lines();
            while let Ok(Some(line)) = lines.next_line().await {
                if let Ok(msg) = serde_json::from_str::<ClientMessage>(&line) {
                    let _ = received_tx.send(msg).await;
                }
            }
        });

        let stream = TcpStream::connect(format!("127.0.0.1:{port}"))
            .await
            .unwrap();

        let input = BufReader::new(&b"/\n/quit\n"[..]);
        let (output_writer, mut output_reader) = duplex(4096);

        run_chat_with_io(stream, input, output_writer, false)
            .await
            .unwrap();

        // Only Disconnect should reach the daemon
        let msg = timeout(TEST_TIMEOUT, received_rx.recv())
            .await
            .expect("timed out")
            .expect("channel closed");
        assert_eq!(
            msg,
            ClientMessage::Disconnect,
            "bare '/' should not send a command to the daemon"
        );

        // Local output should contain the usage hint
        let mut output = String::new();
        output_reader.read_to_string(&mut output).await.unwrap();
        assert!(
            output.contains("/help"),
            "bare '/' output should mention /help, got: {output}"
        );
    }

    /// With `is_tty` = true, prompts appear in output; content is still correct.
    #[tokio::test]
    async fn tty_mode_writes_prompts_to_output() {
        let (listener, port) = bind_random().await;

        tokio::spawn(async move {
            let (stream, _) = listener.accept().await.unwrap();
            let (_read_half, mut write_half) = stream.into_split();
            let msg = ServerMessage::Message {
                text: "pong".into(),
            };
            let json = serde_json::to_string(&msg).unwrap();
            write_half
                .write_all(format!("{json}\n").as_bytes())
                .await
                .unwrap();
            write_half.flush().await.unwrap();
            drop(write_half);
        });

        let stream = TcpStream::connect(format!("127.0.0.1:{port}"))
            .await
            .unwrap();

        let (_stdin_write, stdin_read) = duplex(1024);
        let input = BufReader::new(stdin_read);
        let (output_writer, mut output_reader) = duplex(4096);

        run_chat_with_io(stream, input, output_writer, true)
            .await
            .unwrap();

        let mut output = String::new();
        output_reader.read_to_string(&mut output).await.unwrap();

        assert!(
            output.contains("You:"),
            "tty mode should write styled 'You:' prompt, got: {output}"
        );
        assert!(
            output.contains("Freebird:"),
            "tty mode should write styled 'Freebird:' prefix, got: {output}"
        );
        assert!(
            output.contains("\x1b["),
            "tty mode should include ANSI escape codes, got: {output}"
        );
        assert!(
            output.contains("pong"),
            "response text should still appear, got: {output}"
        );
    }

    /// `TurnComplete` triggers a new "You: " prompt in TTY mode.
    #[tokio::test]
    async fn tty_turn_complete_triggers_reprompt() {
        let (listener, port) = bind_random().await;

        tokio::spawn(async move {
            let (stream, _) = listener.accept().await.unwrap();
            let (_read_half, mut write_half) = stream.into_split();

            // Send a complete turn: Message then TurnComplete
            for msg in [
                ServerMessage::Message {
                    text: "hello".into(),
                },
                ServerMessage::TurnComplete,
            ] {
                let json = serde_json::to_string(&msg).unwrap();
                write_half
                    .write_all(format!("{json}\n").as_bytes())
                    .await
                    .unwrap();
            }
            write_half.flush().await.unwrap();
            drop(write_half);
        });

        let stream = TcpStream::connect(format!("127.0.0.1:{port}"))
            .await
            .unwrap();

        let (_stdin_write, stdin_read) = duplex(1024);
        let input = BufReader::new(stdin_read);
        let (output_writer, mut output_reader) = duplex(4096);

        run_chat_with_io(stream, input, output_writer, true)
            .await
            .unwrap();

        let mut output = String::new();
        output_reader.read_to_string(&mut output).await.unwrap();

        // "You:" should appear at least twice: initial prompt + after TurnComplete
        let you_count = output.matches("You:").count();
        assert!(
            you_count >= 2,
            "expected at least 2 'You:' prompts (initial + after TurnComplete), got {you_count} in: {output}"
        );
    }
}
