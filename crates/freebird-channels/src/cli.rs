//! CLI (stdin/stdout) channel implementation.
//!
//! Single-user, local-only channel that reads from stdin and writes to
//! stdout/stderr. Does not require authentication or pairing
//! (CLAUDE.md §14: "CLI is exempt because it runs locally").

use std::collections::BTreeSet;
use std::sync::atomic::{AtomicBool, Ordering};

use async_trait::async_trait;
use tokio::io::{self, AsyncBufRead, AsyncBufReadExt, AsyncWrite, AsyncWriteExt, BufReader};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tokio_util::sync::CancellationToken;

use freebird_traits::channel::{
    AuthRequirement, Channel, ChannelError, ChannelFeature, ChannelHandle, ChannelInfo,
    InboundEvent, OutboundEvent,
};
use freebird_traits::id::ChannelId;

/// Default prompt printed to stderr before each input line.
const DEFAULT_PROMPT: &str = "> ";

/// Sender ID for all CLI events. CLI is a single-user local channel.
const SENDER_ID: &str = "local";

/// CLI channel — reads from stdin, writes to stdout/stderr.
///
/// This is a single-user, local-only channel that does not require
/// authentication or pairing.
pub struct CliChannel {
    info: ChannelInfo,
    prompt: String,
    cancel: CancellationToken,
    started: AtomicBool,
}

impl CliChannel {
    /// Create a CLI channel with the default prompt (`"> "`).
    #[must_use]
    pub fn new() -> Self {
        Self::with_prompt(DEFAULT_PROMPT)
    }

    /// Create a CLI channel with a custom prompt string.
    ///
    /// The prompt is printed to stderr before each line of user input.
    #[must_use]
    pub fn with_prompt(prompt: impl Into<String>) -> Self {
        Self {
            info: ChannelInfo {
                id: ChannelId::from("cli"),
                display_name: "Command Line Interface".into(),
                features: BTreeSet::from([ChannelFeature::Streaming]),
                auth: AuthRequirement::None,
            },
            prompt: prompt.into(),
            cancel: CancellationToken::new(),
            started: AtomicBool::new(false),
        }
    }

    /// Start the channel with injected I/O sources.
    ///
    /// This is the real implementation. [`Channel::start`] delegates here
    /// with stdin/stdout/stderr. Tests call this directly with in-memory buffers.
    ///
    /// Takes two separate stderr writers because the inbound task (prompts) and
    /// outbound task (errors) each need exclusive ownership. In production,
    /// `io::stderr()` returns a new handle per call; in tests, use two duplex streams.
    pub(crate) fn start_with_io<R, W, E1, E2>(
        &self,
        reader: R,
        mut stdout: W,
        mut prompt_stderr: E1,
        mut error_stderr: E2,
    ) -> ChannelHandle
    where
        R: AsyncBufRead + Send + Unpin + 'static,
        W: AsyncWrite + Send + Unpin + 'static,
        E1: AsyncWrite + Send + Unpin + 'static,
        E2: AsyncWrite + Send + Unpin + 'static,
    {
        let (inbound_tx, inbound_rx) = mpsc::channel::<InboundEvent>(32);
        let (outbound_tx, mut outbound_rx) = mpsc::channel::<OutboundEvent>(32);

        let prompt = self.prompt.clone();
        let cancel = self.cancel.clone();

        // Spawn inbound (stdin reader) task
        tokio::spawn(async move {
            let _ = inbound_tx
                .send(InboundEvent::Connected {
                    sender_id: SENDER_ID.into(),
                })
                .await;

            let mut lines = reader.lines();

            loop {
                let _ = prompt_stderr.write_all(prompt.as_bytes()).await;
                let _ = prompt_stderr.flush().await;

                tokio::select! {
                    () = cancel.cancelled() => {
                        tracing::debug!("inbound task cancelled");
                        break;
                    }
                    result = lines.next_line() => {
                        match result {
                            Ok(Some(line)) => {
                                if let Some(event) = parse_input_line(&line) {
                                    if inbound_tx.send(event).await.is_err() {
                                        break;
                                    }
                                }
                            }
                            Ok(None) => {
                                let _ = inbound_tx
                                    .send(InboundEvent::Disconnected {
                                        sender_id: SENDER_ID.into(),
                                    })
                                    .await;
                                break;
                            }
                            Err(e) => {
                                tracing::error!(error = %e, "stdin read error");
                                break;
                            }
                        }
                    }
                }
            }
        });

        // Spawn outbound (stdout/error_stderr writer) task
        let cancel_out = self.cancel.clone();
        tokio::spawn(async move {
            loop {
                tokio::select! {
                    () = cancel_out.cancelled() => break,
                    event = outbound_rx.recv() => {
                        match event {
                            Some(OutboundEvent::Message { text, .. }) => {
                                let _ = stdout.write_all(text.as_bytes()).await;
                                let _ = stdout.write_all(b"\n").await;
                                let _ = stdout.flush().await;
                            }
                            Some(OutboundEvent::StreamChunk { text, .. }) => {
                                let _ = stdout.write_all(text.as_bytes()).await;
                                let _ = stdout.flush().await;
                            }
                            Some(OutboundEvent::StreamEnd { .. }) => {
                                let _ = stdout.write_all(b"\n").await;
                                let _ = stdout.flush().await;
                            }
                            Some(OutboundEvent::Error { text, .. }) => {
                                let _ = error_stderr.write_all(b"error: ").await;
                                let _ = error_stderr.write_all(text.as_bytes()).await;
                                let _ = error_stderr.write_all(b"\n").await;
                                let _ = error_stderr.flush().await;
                            }
                            None => break,
                        }
                    }
                }
            }
        });

        ChannelHandle {
            inbound: Box::pin(ReceiverStream::new(inbound_rx)),
            outbound: outbound_tx,
        }
    }
}

impl Default for CliChannel {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Channel for CliChannel {
    fn info(&self) -> &ChannelInfo {
        &self.info
    }

    async fn start(&self) -> Result<ChannelHandle, ChannelError> {
        if self.started.swap(true, Ordering::SeqCst) {
            return Err(ChannelError::StartupFailed {
                channel: "cli".into(),
                reason: "channel already started".into(),
            });
        }

        Ok(self.start_with_io(
            BufReader::new(io::stdin()),
            io::stdout(),
            io::stderr(),
            io::stderr(),
        ))
    }

    async fn stop(&self) -> Result<(), ChannelError> {
        self.cancel.cancel();
        Ok(())
    }
}

/// Parse a raw input line into an [`InboundEvent`].
///
/// Returns `None` for empty or whitespace-only lines (silently skipped).
/// Lines starting with `/` are parsed as commands; everything else is a message.
fn parse_input_line(line: &str) -> Option<InboundEvent> {
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return None;
    }

    trimmed.strip_prefix('/').map_or_else(
        || {
            Some(InboundEvent::Message {
                raw_text: trimmed.to_string(),
                sender_id: SENDER_ID.into(),
                attachments: vec![],
            })
        },
        |rest| {
            let mut parts = rest.splitn(2, ' ');
            let name = parts.next().unwrap_or_default().to_string();
            let args = parts
                .next()
                .map(|a| a.split_whitespace().map(String::from).collect())
                .unwrap_or_default();
            Some(InboundEvent::Command {
                name,
                args,
                sender_id: SENDER_ID.into(),
            })
        },
    )
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]
mod tests {
    use super::*;
    use tokio::io::{AsyncReadExt, BufReader};
    use tokio_stream::StreamExt;

    // ── Helpers ──────────────────────────────────────────────────────

    fn extract_message_text(event: &InboundEvent) -> &str {
        match event {
            InboundEvent::Message { raw_text, .. } => raw_text,
            other => panic!("expected Message, got {other:?}"),
        }
    }

    fn extract_command(event: &InboundEvent) -> (&str, &[String]) {
        match event {
            InboundEvent::Command { name, args, .. } => (name, args),
            other => panic!("expected Command, got {other:?}"),
        }
    }

    fn assert_connected(event: &InboundEvent) {
        match event {
            InboundEvent::Connected { sender_id } => assert_eq!(sender_id, "local"),
            other => panic!("expected Connected, got {other:?}"),
        }
    }

    fn assert_disconnected(event: &InboundEvent) {
        match event {
            InboundEvent::Disconnected { sender_id } => assert_eq!(sender_id, "local"),
            other => panic!("expected Disconnected, got {other:?}"),
        }
    }

    /// Returns (handle, `stdout_read`, `prompt_stderr_read`, `error_stderr_read`, channel).
    async fn start_channel_with_stdin(
        stdin_bytes: &[u8],
    ) -> (
        ChannelHandle,
        tokio::io::DuplexStream,
        tokio::io::DuplexStream,
        tokio::io::DuplexStream,
        CliChannel,
    ) {
        let (mut stdin_write, stdin_read) = tokio::io::duplex(4096);
        let (stdout_write, stdout_read) = tokio::io::duplex(4096);
        let (prompt_stderr_write, prompt_stderr_read) = tokio::io::duplex(4096);
        let (error_stderr_write, error_stderr_read) = tokio::io::duplex(4096);

        stdin_write.write_all(stdin_bytes).await.unwrap();
        drop(stdin_write); // EOF

        let channel = CliChannel::new();
        let handle = channel.start_with_io(
            BufReader::new(stdin_read),
            stdout_write,
            prompt_stderr_write,
            error_stderr_write,
        );

        (
            handle,
            stdout_read,
            prompt_stderr_read,
            error_stderr_read,
            channel,
        )
    }

    // ── Unit tests: parse_input_line ────────────────────────────────

    #[test]
    fn test_parse_regular_message() {
        let event = parse_input_line("hello world").unwrap();
        assert_eq!(extract_message_text(&event), "hello world");
    }

    #[test]
    fn test_parse_command_simple() {
        let event = parse_input_line("/quit").unwrap();
        let (name, args) = extract_command(&event);
        assert_eq!(name, "quit");
        assert!(args.is_empty());
    }

    #[test]
    fn test_parse_command_with_args() {
        let event = parse_input_line("/new my session").unwrap();
        let (name, args) = extract_command(&event);
        assert_eq!(name, "new");
        assert_eq!(args, &["my", "session"]);
    }

    #[test]
    fn test_parse_empty_line() {
        assert!(parse_input_line("").is_none());
    }

    #[test]
    fn test_parse_whitespace_only() {
        assert!(parse_input_line("   \t  ").is_none());
    }

    #[test]
    fn test_parse_slash_only() {
        let event = parse_input_line("/").unwrap();
        let (name, args) = extract_command(&event);
        assert_eq!(name, "");
        assert!(args.is_empty());
    }

    #[test]
    fn test_parse_leading_trailing_whitespace() {
        let event = parse_input_line("  hello  ").unwrap();
        assert_eq!(extract_message_text(&event), "hello");
    }

    #[test]
    fn test_parse_command_extra_whitespace() {
        let event = parse_input_line("/help   arg1   arg2").unwrap();
        let (name, args) = extract_command(&event);
        assert_eq!(name, "help");
        assert_eq!(args, &["arg1", "arg2"]);
    }

    // ── Unit tests: struct and metadata ─────────────────────────────

    #[test]
    fn test_channel_info() {
        let channel = CliChannel::new();
        let info = channel.info();
        assert_eq!(info.id.as_str(), "cli");
        assert_eq!(info.display_name, "Command Line Interface");
        assert!(info.supports(&ChannelFeature::Streaming));
        assert!(!info.supports(&ChannelFeature::Media));
        assert!(!info.requires_auth());
    }

    #[test]
    fn test_default_prompt() {
        let channel = CliChannel::new();
        assert_eq!(channel.prompt, "> ");
    }

    #[test]
    fn test_custom_prompt() {
        let channel = CliChannel::with_prompt("you> ");
        assert_eq!(channel.prompt, "you> ");
    }

    #[test]
    fn test_default_trait() {
        let channel = CliChannel::default();
        assert_eq!(channel.info().id.as_str(), "cli");
        assert_eq!(channel.prompt, "> ");
    }

    // ── Async tests: start_with_io integration ──────────────────────

    #[tokio::test]
    async fn test_connected_event_on_start() {
        let (mut handle, _, _, _, _) = start_channel_with_stdin(b"").await;
        let event = handle.inbound.next().await.unwrap();
        assert_connected(&event);
    }

    #[tokio::test]
    async fn test_disconnected_event_on_eof() {
        let (mut handle, _, _, _, _) = start_channel_with_stdin(b"").await;
        let _ = handle.inbound.next().await; // Connected
        let event = handle.inbound.next().await.unwrap();
        assert_disconnected(&event);
    }

    #[tokio::test]
    async fn test_message_flows_through() {
        let (mut handle, _, _, _, _) = start_channel_with_stdin(b"hello\n").await;
        let _ = handle.inbound.next().await; // Connected
        let event = handle.inbound.next().await.unwrap();
        assert_eq!(extract_message_text(&event), "hello");
    }

    #[tokio::test]
    async fn test_command_flows_through() {
        let (mut handle, _, _, _, _) = start_channel_with_stdin(b"/quit\n").await;
        let _ = handle.inbound.next().await; // Connected
        let event = handle.inbound.next().await.unwrap();
        let (name, args) = extract_command(&event);
        assert_eq!(name, "quit");
        assert!(args.is_empty());
    }

    #[tokio::test]
    async fn test_outbound_message_written() {
        let (handle, mut stdout_read, _, _, _) = start_channel_with_stdin(b"").await;

        handle
            .outbound
            .send(OutboundEvent::Message {
                text: "hello".into(),
                recipient_id: "local".into(),
            })
            .await
            .unwrap();

        drop(handle);
        tokio::task::yield_now().await;

        let mut buf = vec![0u8; 256];
        let n = tokio::time::timeout(
            std::time::Duration::from_millis(500),
            stdout_read.read(&mut buf),
        )
        .await
        .unwrap()
        .unwrap();

        let output = std::str::from_utf8(&buf[..n]).unwrap();
        assert_eq!(output, "hello\n");
    }

    #[tokio::test]
    async fn test_outbound_stream_chunk_no_newline() {
        let (handle, mut stdout_read, _, _, _) = start_channel_with_stdin(b"").await;

        handle
            .outbound
            .send(OutboundEvent::StreamChunk {
                text: "hel".into(),
                recipient_id: "local".into(),
            })
            .await
            .unwrap();

        drop(handle);
        tokio::task::yield_now().await;

        let mut buf = vec![0u8; 256];
        let n = tokio::time::timeout(
            std::time::Duration::from_millis(500),
            stdout_read.read(&mut buf),
        )
        .await
        .unwrap()
        .unwrap();

        let output = std::str::from_utf8(&buf[..n]).unwrap();
        assert_eq!(output, "hel");
    }

    #[tokio::test]
    async fn test_outbound_stream_end_newline() {
        let (handle, mut stdout_read, _, _, _) = start_channel_with_stdin(b"").await;

        handle
            .outbound
            .send(OutboundEvent::StreamEnd {
                recipient_id: "local".into(),
            })
            .await
            .unwrap();

        drop(handle);
        tokio::task::yield_now().await;

        let mut buf = vec![0u8; 256];
        let n = tokio::time::timeout(
            std::time::Duration::from_millis(500),
            stdout_read.read(&mut buf),
        )
        .await
        .unwrap()
        .unwrap();

        let output = std::str::from_utf8(&buf[..n]).unwrap();
        assert_eq!(output, "\n");
    }

    #[tokio::test]
    async fn test_outbound_error_to_stderr() {
        let (handle, _, _, mut error_stderr_read, _) = start_channel_with_stdin(b"").await;

        handle
            .outbound
            .send(OutboundEvent::Error {
                text: "something".into(),
                recipient_id: "local".into(),
            })
            .await
            .unwrap();

        drop(handle);
        tokio::task::yield_now().await;

        let mut buf = vec![0u8; 4096];
        let n = tokio::time::timeout(
            std::time::Duration::from_millis(500),
            error_stderr_read.read(&mut buf),
        )
        .await
        .unwrap()
        .unwrap();

        let output = std::str::from_utf8(&buf[..n]).unwrap();
        assert_eq!(output, "error: something\n");
    }

    #[tokio::test]
    async fn test_stop_closes_inbound_stream() {
        let (stdin_write, stdin_read) = tokio::io::duplex(4096);
        let (stdout_write, _stdout_read) = tokio::io::duplex(4096);
        let (prompt_stderr_write, _prompt_stderr_read) = tokio::io::duplex(4096);
        let (error_stderr_write, _error_stderr_read) = tokio::io::duplex(4096);

        // Keep stdin open so the channel blocks on read
        let _keep_open = stdin_write;

        let channel = CliChannel::new();
        let mut handle = channel.start_with_io(
            BufReader::new(stdin_read),
            stdout_write,
            prompt_stderr_write,
            error_stderr_write,
        );

        let _ = handle.inbound.next().await; // Connected

        channel.stop().await.unwrap();

        let next =
            tokio::time::timeout(std::time::Duration::from_millis(500), handle.inbound.next())
                .await
                .unwrap();

        assert!(next.is_none(), "stream should end after stop()");
    }

    #[tokio::test]
    async fn test_double_start_fails() {
        let channel = CliChannel::new();
        channel.started.store(true, Ordering::SeqCst);

        let result = channel.start().await;
        assert!(result.is_err());

        let err = result.unwrap_err();
        match err {
            ChannelError::StartupFailed { channel, reason } => {
                assert_eq!(channel, "cli");
                assert_eq!(reason, "channel already started");
            }
            other => panic!("expected StartupFailed, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_prompt_written_to_stderr() {
        let (mut handle, _, mut prompt_stderr_read, _, _) =
            start_channel_with_stdin(b"hello\n").await;

        // Drain inbound events to let the tasks run
        let _ = handle.inbound.next().await; // Connected
        let _ = handle.inbound.next().await; // Message
        let _ = handle.inbound.next().await; // Disconnected (EOF)

        let mut buf = vec![0u8; 4096];
        let n = tokio::time::timeout(
            std::time::Duration::from_millis(500),
            prompt_stderr_read.read(&mut buf),
        )
        .await
        .unwrap()
        .unwrap();

        let output = std::str::from_utf8(&buf[..n]).unwrap();
        assert!(
            output.contains("> "),
            "stderr should contain prompt '> ', got: {output:?}"
        );
    }

    // ── Property-based test ─────────────────────────────────────────

    mod proptests {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            #[test]
            fn test_parse_input_line_never_panics(input in "\\PC*") {
                let _ = parse_input_line(&input);
            }
        }
    }
}
