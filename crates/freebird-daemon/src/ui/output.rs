//! Output renderer — streaming text display and message formatting.
//!
//! Manages rendering of server responses to the terminal, handling
//! streaming chunks, tool events, errors, and session info.

use std::io::Write;

use crossterm::cursor::MoveToColumn;
use crossterm::queue;
use crossterm::terminal::{Clear, ClearType};

use super::theme;

/// Tracks state for rendering server output.
pub struct OutputRenderer {
    /// True while mid-stream (between first `StreamChunk` and `StreamEnd`).
    in_stream: bool,
    /// Whether we've printed the bot prefix for the current response.
    prefix_printed: bool,
}

impl OutputRenderer {
    /// Create a new output renderer.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            in_stream: false,
            prefix_printed: false,
        }
    }

    /// Whether we are currently in a streaming response.
    #[must_use]
    pub const fn is_streaming(&self) -> bool {
        self.in_stream
    }

    /// Render a complete message from the bot.
    pub fn write_message<W: Write>(&mut self, w: &mut W, text: &str) -> std::io::Result<()> {
        if !self.prefix_printed {
            theme::write_bot_prefix_styled(w)?;
            self.prefix_printed = true;
        }
        write!(w, "{text}")?;
        writeln!(w)?;
        w.flush()?;
        self.prefix_printed = false;
        Ok(())
    }

    /// Render a streaming chunk.
    pub fn write_stream_chunk<W: Write>(&mut self, w: &mut W, text: &str) -> std::io::Result<()> {
        if !self.prefix_printed {
            theme::write_bot_prefix_styled(w)?;
            self.prefix_printed = true;
        }
        write!(w, "{text}")?;
        w.flush()?;
        self.in_stream = true;
        Ok(())
    }

    /// Handle stream end.
    pub fn write_stream_end<W: Write>(&mut self, w: &mut W) -> std::io::Result<()> {
        if self.in_stream {
            writeln!(w)?;
            w.flush()?;
        }
        self.in_stream = false;
        self.prefix_printed = false;
        Ok(())
    }

    /// Render an error message.
    pub fn write_error<W: Write>(&mut self, w: &mut W, text: &str) -> std::io::Result<()> {
        if !self.prefix_printed {
            writeln!(w)?;
        }
        theme::write_error_styled(w, text)?;
        self.prefix_printed = false;
        Ok(())
    }

    /// Render a command response.
    pub fn write_command_response<W: Write>(
        &mut self,
        w: &mut W,
        text: &str,
    ) -> std::io::Result<()> {
        if !self.prefix_printed {
            theme::write_bot_prefix_styled(w)?;
        }
        writeln!(w, "{text}")?;
        w.flush()?;
        self.prefix_printed = false;
        Ok(())
    }

    /// Signal that a turn is complete — resets state for next interaction.
    pub const fn turn_complete(&mut self) {
        self.in_stream = false;
        self.prefix_printed = false;
    }

    /// Prepare for output by clearing the input area line.
    ///
    /// Call this before writing any output to ensure the input prompt
    /// doesn't get mixed with response text. Takes `&self` (rather than
    /// being a free function) so future enhancements can inspect renderer
    /// state (e.g. multi-line input height) to clear the correct number
    /// of lines.
    #[allow(clippy::unused_self)] // Kept as method for future state-aware clearing
    pub fn clear_input_line<W: Write>(&self, w: &mut W) -> std::io::Result<()> {
        queue!(w, MoveToColumn(0), Clear(ClearType::CurrentLine))?;

        w.flush()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn new_renderer_not_streaming() {
        let r = OutputRenderer::new();
        assert!(!r.is_streaming());
    }

    #[test]
    fn stream_chunk_sets_streaming() {
        let mut r = OutputRenderer::new();
        let mut buf = Vec::new();
        r.write_stream_chunk(&mut buf, "hello").unwrap();
        assert!(r.is_streaming());
    }

    #[test]
    fn stream_end_clears_streaming() {
        let mut r = OutputRenderer::new();
        let mut buf = Vec::new();
        r.write_stream_chunk(&mut buf, "hello").unwrap();
        r.write_stream_end(&mut buf).unwrap();
        assert!(!r.is_streaming());
    }

    #[test]
    fn turn_complete_resets() {
        let mut r = OutputRenderer::new();
        let mut buf = Vec::new();
        r.write_stream_chunk(&mut buf, "test").unwrap();
        r.turn_complete();
        assert!(!r.is_streaming());
    }

    #[test]
    fn message_writes_content() {
        let mut r = OutputRenderer::new();
        let mut buf = Vec::new();
        r.write_message(&mut buf, "hello world").unwrap();
        let output = String::from_utf8_lossy(&buf);
        assert!(output.contains("hello world"));
    }

    #[test]
    fn error_writes_content() {
        let mut r = OutputRenderer::new();
        let mut buf = Vec::new();
        r.write_error(&mut buf, "boom").unwrap();
        let output = String::from_utf8_lossy(&buf);
        assert!(output.contains("boom"));
    }
}
