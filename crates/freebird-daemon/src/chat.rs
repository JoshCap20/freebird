//! Chat client — thin TCP client for `freebird chat`.
//!
//! Connects to a running daemon, bridges user input/output to the JSON-line
//! protocol defined in [`freebird_types::protocol`].

use anyhow::{Context, Result};
use tokio::io::{AsyncBufRead, AsyncBufReadExt, AsyncWrite, AsyncWriteExt, BufReader};
use tokio::net::TcpStream;

use freebird_types::protocol::{ClientMessage, ServerMessage};

/// Parse a line of user input into a [`ClientMessage`].
///
/// Returns `None` for `/quit` and `/exit` (signals the caller to disconnect).
pub fn parse_user_input(line: &str) -> Option<ClientMessage> {
    if line == "/quit" || line == "/exit" {
        return None;
    }

    line.strip_prefix('/').map_or_else(
        || {
            Some(ClientMessage::Message {
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
            Some(ClientMessage::Command { name, args })
        },
    )
}

/// Render a [`ServerMessage`] as user-facing text.
pub fn render_server_message(msg: &ServerMessage) -> String {
    match msg {
        ServerMessage::Message { text } | ServerMessage::CommandResponse { text } => {
            format!("{text}\n")
        }
        ServerMessage::StreamChunk { text } => text.clone(),
        ServerMessage::StreamEnd => "\n".to_string(),
        ServerMessage::Error { text } => format!("error: {text}\n"),
    }
}

/// Run the chat client loop with injectable I/O (for testing).
///
/// Reads lines from `user_input`, sends them as JSON-line [`ClientMessage`]s
/// to the daemon via `stream`, reads JSON-line [`ServerMessage`]s from the
/// daemon, and renders them to `user_output`.
pub async fn run_chat_with_io<I, O>(
    stream: TcpStream,
    user_input: I,
    mut user_output: O,
) -> Result<()>
where
    I: AsyncBufRead + Unpin,
    O: AsyncWrite + Unpin,
{
    let (socket_read, mut socket_write) = stream.into_split();
    let socket_reader = BufReader::new(socket_read);

    let mut input_lines = user_input.lines();
    let mut socket_lines = socket_reader.lines();

    loop {
        tokio::select! {
            line = input_lines.next_line() => {
                if let Some(line) = line.context("reading user input")? {
                    let trimmed = line.trim().to_string();
                    if trimmed.is_empty() {
                        continue;
                    }

                    let Some(client_msg) = parse_user_input(&trimmed) else {
                        send_client_message(&mut socket_write, &ClientMessage::Disconnect).await?;
                        return Ok(());
                    };

                    send_client_message(&mut socket_write, &client_msg).await?;
                } else {
                    // stdin EOF — disconnect gracefully
                    send_client_message(&mut socket_write, &ClientMessage::Disconnect).await?;
                    return Ok(());
                }
            }
            line = socket_lines.next_line() => {
                if let Some(line) = line.context("reading from daemon")? {
                    match serde_json::from_str::<ServerMessage>(&line) {
                        Ok(msg) => {
                            let rendered = render_server_message(&msg);
                            user_output.write_all(rendered.as_bytes()).await
                                .context("writing to output")?;
                            user_output.flush().await
                                .context("flushing output")?;
                        }
                        Err(e) => {
                            let err_msg = format!("[protocol error: {e}]\n");
                            user_output.write_all(err_msg.as_bytes()).await?;
                            user_output.flush().await?;
                        }
                    }
                } else {
                    // Server disconnected
                    user_output.write_all(b"[daemon disconnected]\n").await?;
                    user_output.flush().await?;
                    return Ok(());
                }
            }
        }
    }
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

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::indexing_slicing)]
mod tests {
    use super::*;
    use std::time::Duration;
    use tokio::io::{AsyncReadExt, duplex};
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
            Some(ClientMessage::Message {
                text: "hello world".into()
            })
        );
    }

    #[test]
    fn parse_command_no_args() {
        assert_eq!(
            parse_user_input("/new"),
            Some(ClientMessage::Command {
                name: "new".into(),
                args: vec![]
            })
        );
    }

    #[test]
    fn parse_command_with_args() {
        assert_eq!(
            parse_user_input("/model opus fast"),
            Some(ClientMessage::Command {
                name: "model".into(),
                args: vec!["opus".into(), "fast".into()]
            })
        );
    }

    #[test]
    fn parse_quit_and_exit() {
        assert_eq!(parse_user_input("/quit"), None);
        assert_eq!(parse_user_input("/exit"), None);
    }

    #[test]
    fn parse_slash_alone_is_empty_command() {
        assert_eq!(
            parse_user_input("/"),
            Some(ClientMessage::Command {
                name: String::new(),
                args: vec![]
            })
        );
    }

    // ── render_server_message unit tests ───────────────────────────────

    #[test]
    fn render_message() {
        let msg = ServerMessage::Message { text: "hi".into() };
        assert_eq!(render_server_message(&msg), "hi\n");
    }

    #[test]
    fn render_stream_chunk() {
        let msg = ServerMessage::StreamChunk {
            text: "partial".into(),
        };
        assert_eq!(render_server_message(&msg), "partial");
    }

    #[test]
    fn render_stream_end() {
        assert_eq!(render_server_message(&ServerMessage::StreamEnd), "\n");
    }

    #[test]
    fn render_error() {
        let msg = ServerMessage::Error {
            text: "boom".into(),
        };
        assert_eq!(render_server_message(&msg), "error: boom\n");
    }

    #[test]
    fn render_command_response() {
        let msg = ServerMessage::CommandResponse { text: "ok".into() };
        assert_eq!(render_server_message(&msg), "ok\n");
    }

    // ── Integration tests ──────────────────────────────────────────────

    #[tokio::test]
    async fn stdin_eof_sends_disconnect() {
        let (listener, port) = bind_random().await;
        let (received_tx, mut received_rx) = mpsc::channel::<ClientMessage>(16);

        // Mock daemon: read and record client messages
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

        // Empty stdin (immediate EOF)
        let input = BufReader::new(&b""[..]);
        let (output_writer, _output_reader) = duplex(4096);

        run_chat_with_io(stream, input, output_writer)
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

        run_chat_with_io(stream, input, output_writer)
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

        // User types "hello" then stdin EOF triggers disconnect
        let input = BufReader::new(&b"hello\n"[..]);
        let (output_writer, _output_reader) = duplex(4096);

        run_chat_with_io(stream, input, output_writer)
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

        // Mock daemon: send a message then close connection
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

        // Stdin: open but empty (won't send anything)
        let (_stdin_write, stdin_read) = duplex(1024);
        let input = BufReader::new(stdin_read);
        let (output_writer, mut output_reader) = duplex(4096);

        run_chat_with_io(stream, input, output_writer)
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
}
