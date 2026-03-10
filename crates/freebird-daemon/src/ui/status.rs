//! Status bar — inline display of token usage, model info, and session data.
//!
//! Unlike a fixed-position status bar (which would require alternate screen mode),
//! this renders inline at appropriate moments to preserve native scrollback.

use std::io::Write;

use super::theme;

/// Manages display of session and usage information.
pub struct StatusBar {
    /// Current model name (set on first `SessionInfo`).
    model_name: Option<String>,
    /// Current session ID (set on first `SessionInfo`).
    session_id: Option<String>,
    /// Whether the session header has been displayed.
    header_shown: bool,
}

impl StatusBar {
    /// Create a new status bar.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            model_name: None,
            session_id: None,
            header_shown: false,
        }
    }

    /// Update session info and display the header if this is the first time.
    pub fn set_session_info<W: Write>(
        &mut self,
        w: &mut W,
        model_id: &str,
        session_id: &str,
    ) -> std::io::Result<()> {
        self.model_name = Some(model_id.to_string());
        self.session_id = Some(session_id.to_string());

        if !self.header_shown {
            theme::write_session_header_styled(w, model_id, session_id)?;
            self.header_shown = true;
        }

        Ok(())
    }

    /// Display token usage after a turn completes.
    #[allow(clippy::unused_self)] // Method on StatusBar for future cumulative tracking
    pub fn show_token_usage<W: Write>(
        &self,
        w: &mut W,
        input_tokens: u32,
        output_tokens: u32,
    ) -> std::io::Result<()> {
        theme::write_token_usage_styled(w, input_tokens, output_tokens)
    }

    /// Whether the session header has been displayed.
    #[must_use]
    #[allow(dead_code)]
    pub const fn header_shown(&self) -> bool {
        self.header_shown
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn new_status_bar_no_header() {
        let s = StatusBar::new();
        assert!(!s.header_shown());
    }

    #[test]
    fn set_session_info_shows_header() {
        let mut s = StatusBar::new();
        let mut buf = Vec::new();
        s.set_session_info(&mut buf, "claude-sonnet-4-6", "abc12345-def")
            .unwrap();
        assert!(s.header_shown());
        let output = String::from_utf8_lossy(&buf);
        assert!(output.contains("claude-sonnet"));
    }

    #[test]
    fn set_session_info_twice_only_shows_header_once() {
        let mut s = StatusBar::new();
        let mut buf = Vec::new();
        s.set_session_info(&mut buf, "model1", "session1").unwrap();
        let len_after_first = buf.len();
        s.set_session_info(&mut buf, "model2", "session2").unwrap();
        // Second call should not write more header output
        assert_eq!(buf.len(), len_after_first);
    }

    #[test]
    fn show_token_usage_writes_output() {
        let s = StatusBar::new();
        let mut buf = Vec::new();
        s.show_token_usage(&mut buf, 1234, 567).unwrap();
        let output = String::from_utf8_lossy(&buf);
        assert!(output.contains("1,234"));
        assert!(output.contains("567"));
    }
}
