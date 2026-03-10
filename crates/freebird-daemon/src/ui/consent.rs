//! Interactive consent selector widget for the TUI.
//!
//! When the daemon requests consent for a high-risk tool invocation, this
//! widget replaces the normal input editor with a two-option selector
//! (Approve / Deny) that the user navigates with arrow keys or shortcut keys.

use std::io::Write;

use chrono::{DateTime, Utc};
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use crossterm::queue;
use crossterm::style::{Attribute, Color, ResetColor, SetAttribute, SetForegroundColor};

/// Which option is currently highlighted in the consent selector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ConsentChoice {
    Approve,
    Deny,
}

impl ConsentChoice {
    /// Toggle between Approve and Deny.
    const fn toggle(self) -> Self {
        match self {
            Self::Approve => Self::Deny,
            Self::Deny => Self::Approve,
        }
    }
}

/// Result of handling a key event in consent mode.
#[derive(Debug)]
#[must_use]
pub enum ConsentAction {
    /// User confirmed their choice — send this response.
    Confirmed {
        request_id: String,
        approved: bool,
        reason: Option<String>,
    },
    /// Selection changed — redraw the selector.
    Redraw,
    /// No action needed.
    None,
}

/// Interactive consent selector widget.
///
/// Renders a two-option selector (Approve / Deny) and handles keyboard
/// navigation. The widget does NOT send messages itself — it returns
/// [`ConsentAction`] and the caller (`TtyChat`) sends the response.
pub struct ConsentSelector {
    /// The pending consent request ID.
    request_id: String,
    /// Currently highlighted choice.
    choice: ConsentChoice,
    /// When this request expires.
    expires_at: DateTime<Utc>,
}

impl ConsentSelector {
    /// Create a new consent selector from a received `ConsentRequest`.
    ///
    /// `expires_at_str` is the RFC 3339 timestamp from the server.
    /// Returns `None` if the timestamp is unparseable or already expired.
    #[must_use]
    pub fn new(request_id: String, expires_at_str: &str) -> Option<Self> {
        let expires_at = expires_at_str.parse::<DateTime<Utc>>().ok()?;
        if expires_at <= Utc::now() {
            return None;
        }
        Some(Self {
            request_id,
            choice: ConsentChoice::Approve,
            expires_at,
        })
    }

    /// Handle a key event. Returns the action for the caller.
    pub fn handle_key(&mut self, key: KeyEvent) -> ConsentAction {
        // Ctrl+C → deny immediately
        if key.code == KeyCode::Char('c') && key.modifiers.contains(KeyModifiers::CONTROL) {
            return self.confirm(false, None);
        }

        match key.code {
            // Navigation: toggle selection
            KeyCode::Up | KeyCode::Down | KeyCode::Tab => {
                self.choice = self.choice.toggle();
                ConsentAction::Redraw
            }
            // Confirm current selection
            KeyCode::Enter => {
                let approved = self.choice == ConsentChoice::Approve;
                self.confirm(approved, None)
            }
            // Immediate approve shortcuts
            KeyCode::Char('y' | 'a') => self.confirm(true, None),
            // Immediate deny shortcuts / Escape → deny
            KeyCode::Char('n' | 'd') | KeyCode::Esc => self.confirm(false, None),
            // Everything else is ignored
            _ => ConsentAction::None,
        }
    }

    /// Render the selector widget.
    ///
    /// Outputs two lines:
    /// ```text
    ///   > Approve [y]    ← selected (bold cyan)
    ///     Deny [n]       ← unselected (grey)
    /// ```
    pub fn render<W: Write>(&self, w: &mut W) -> std::io::Result<()> {
        // Approve line
        self.render_option(w, ConsentChoice::Approve, "Approve", "[y]")?;
        writeln!(w)?;
        // Deny line
        self.render_option(w, ConsentChoice::Deny, "Deny", "[n]")?;
        writeln!(w)?;
        w.flush()
    }

    /// Whether the consent request has expired.
    #[must_use]
    pub fn is_expired(&self) -> bool {
        Utc::now() >= self.expires_at
    }

    /// Build a [`ConsentAction::Confirmed`] for auto-deny (timeout or superseded).
    pub fn auto_deny(&self, reason: &str) -> ConsentAction {
        ConsentAction::Confirmed {
            request_id: self.request_id.clone(),
            approved: false,
            reason: Some(reason.to_string()),
        }
    }

    // ── Private helpers ──────────────────────────────────────────────────

    fn confirm(&self, approved: bool, reason: Option<String>) -> ConsentAction {
        ConsentAction::Confirmed {
            request_id: self.request_id.clone(),
            approved,
            reason,
        }
    }

    fn render_option<W: Write>(
        &self,
        w: &mut W,
        option: ConsentChoice,
        label: &str,
        hint: &str,
    ) -> std::io::Result<()> {
        let selected = self.choice == option;
        if selected {
            queue!(
                w,
                SetAttribute(Attribute::Bold),
                SetForegroundColor(Color::Cyan),
            )?;
            write!(w, "  > {label} {hint}")?;
            queue!(w, ResetColor, SetAttribute(Attribute::Reset))?;
        } else {
            queue!(w, SetForegroundColor(Color::DarkGrey))?;
            write!(w, "    {label} {hint}")?;
            queue!(w, ResetColor)?;
        }
        Ok(())
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;
    use chrono::Duration;
    use crossterm::event::{KeyCode, KeyEvent, KeyEventKind, KeyEventState, KeyModifiers};

    /// Helper to create a `KeyEvent` for a given key code.
    fn key(code: KeyCode) -> KeyEvent {
        KeyEvent {
            code,
            modifiers: KeyModifiers::NONE,
            kind: KeyEventKind::Press,
            state: KeyEventState::NONE,
        }
    }

    /// Helper to create a Ctrl+key event.
    fn ctrl_key(code: KeyCode) -> KeyEvent {
        KeyEvent {
            code,
            modifiers: KeyModifiers::CONTROL,
            kind: KeyEventKind::Press,
            state: KeyEventState::NONE,
        }
    }

    /// Helper to create a selector with a future expiry.
    fn make_selector(request_id: &str) -> ConsentSelector {
        let future = (Utc::now() + Duration::minutes(5)).to_rfc3339();
        ConsentSelector::new(request_id.to_string(), &future).unwrap()
    }

    // ── Construction tests ───────────────────────────────────────────

    #[test]
    fn test_new_returns_none_for_expired_timestamp() {
        let past = (Utc::now() - Duration::minutes(1)).to_rfc3339();
        assert!(ConsentSelector::new("req-1".to_string(), &past).is_none());
    }

    #[test]
    fn test_new_returns_some_for_valid_timestamp() {
        let sel = make_selector("req-1");
        assert_eq!(sel.choice, ConsentChoice::Approve);
    }

    #[test]
    fn test_new_returns_none_for_invalid_timestamp() {
        assert!(ConsentSelector::new("req-1".to_string(), "not-a-date").is_none());
    }

    // ── Navigation tests ─────────────────────────────────────────────

    #[test]
    fn test_toggle_choice() {
        let mut sel = make_selector("req-1");
        assert_eq!(sel.choice, ConsentChoice::Approve);

        let action = sel.handle_key(key(KeyCode::Down));
        assert!(matches!(action, ConsentAction::Redraw));
        assert_eq!(sel.choice, ConsentChoice::Deny);

        let action = sel.handle_key(key(KeyCode::Up));
        assert!(matches!(action, ConsentAction::Redraw));
        assert_eq!(sel.choice, ConsentChoice::Approve);

        // Tab also toggles
        let action = sel.handle_key(key(KeyCode::Tab));
        assert!(matches!(action, ConsentAction::Redraw));
        assert_eq!(sel.choice, ConsentChoice::Deny);
    }

    // ── Confirm tests ────────────────────────────────────────────────

    #[test]
    fn test_enter_confirms_approve() {
        let mut sel = make_selector("req-1");
        let action = sel.handle_key(key(KeyCode::Enter));
        match action {
            ConsentAction::Confirmed {
                approved, reason, ..
            } => {
                assert!(approved);
                assert!(reason.is_none());
            }
            _ => panic!("expected Confirmed"),
        }
    }

    #[test]
    fn test_enter_confirms_deny() {
        let mut sel = make_selector("req-1");
        let _ = sel.handle_key(key(KeyCode::Down)); // Toggle to Deny
        let action = sel.handle_key(key(KeyCode::Enter));
        match action {
            ConsentAction::Confirmed {
                approved, reason, ..
            } => {
                assert!(!approved);
                assert!(reason.is_none());
            }
            _ => panic!("expected Confirmed"),
        }
    }

    // ── Shortcut key tests ───────────────────────────────────────────

    #[test]
    fn test_y_key_immediately_approves() {
        let mut sel = make_selector("req-1");
        let action = sel.handle_key(key(KeyCode::Char('y')));
        match action {
            ConsentAction::Confirmed { approved, .. } => assert!(approved),
            _ => panic!("expected Confirmed"),
        }
    }

    #[test]
    fn test_a_key_immediately_approves() {
        let mut sel = make_selector("req-1");
        let action = sel.handle_key(key(KeyCode::Char('a')));
        match action {
            ConsentAction::Confirmed { approved, .. } => assert!(approved),
            _ => panic!("expected Confirmed"),
        }
    }

    #[test]
    fn test_n_key_immediately_denies() {
        let mut sel = make_selector("req-1");
        let action = sel.handle_key(key(KeyCode::Char('n')));
        match action {
            ConsentAction::Confirmed { approved, .. } => assert!(!approved),
            _ => panic!("expected Confirmed"),
        }
    }

    #[test]
    fn test_d_key_immediately_denies() {
        let mut sel = make_selector("req-1");
        let action = sel.handle_key(key(KeyCode::Char('d')));
        match action {
            ConsentAction::Confirmed { approved, .. } => assert!(!approved),
            _ => panic!("expected Confirmed"),
        }
    }

    // ── Escape / Ctrl+C tests ────────────────────────────────────────

    #[test]
    fn test_escape_denies() {
        let mut sel = make_selector("req-1");
        let action = sel.handle_key(key(KeyCode::Esc));
        match action {
            ConsentAction::Confirmed {
                approved, reason, ..
            } => {
                assert!(!approved);
                assert!(reason.is_none());
            }
            _ => panic!("expected Confirmed"),
        }
    }

    #[test]
    fn test_ctrl_c_denies() {
        let mut sel = make_selector("req-1");
        let action = sel.handle_key(ctrl_key(KeyCode::Char('c')));
        match action {
            ConsentAction::Confirmed {
                approved, reason, ..
            } => {
                assert!(!approved);
                assert!(reason.is_none());
            }
            _ => panic!("expected Confirmed"),
        }
    }

    // ── Expiry tests ─────────────────────────────────────────────────

    #[test]
    fn test_is_expired() {
        // Can't construct via new() (it rejects expired), so test via a future
        // expiry and check the inverse.
        let sel = make_selector("req-1");
        assert!(!sel.is_expired());

        // For actual expiry, we construct one that expires in the past by
        // building the struct directly (testing internal state).
        let expired_sel = ConsentSelector {
            request_id: "req-expired".to_string(),
            choice: ConsentChoice::Approve,
            expires_at: Utc::now() - Duration::seconds(1),
        };
        assert!(expired_sel.is_expired());
    }

    // ── auto_deny tests ──────────────────────────────────────────────

    #[test]
    fn test_auto_deny() {
        let sel = make_selector("req-42");
        let action = sel.auto_deny("timeout");
        match action {
            ConsentAction::Confirmed {
                request_id,
                approved,
                reason,
            } => {
                assert_eq!(request_id, "req-42");
                assert!(!approved);
                assert_eq!(reason.as_deref(), Some("timeout"));
            }
            _ => panic!("expected Confirmed"),
        }
    }

    // ── Render tests ─────────────────────────────────────────────────

    #[test]
    fn test_render_approve_selected() {
        let sel = make_selector("req-1");
        let mut buf = Vec::new();
        sel.render(&mut buf).unwrap();
        let output = String::from_utf8_lossy(&buf);
        assert!(output.contains("> Approve [y]"), "got: {output}");
        // Deny should not have the `>` indicator
        assert!(
            !output.contains("> Deny"),
            "Deny should not be selected, got: {output}"
        );
        assert!(output.contains("Deny [n]"), "Deny option should be present");
    }

    #[test]
    fn test_render_deny_selected() {
        let mut sel = make_selector("req-1");
        let _ = sel.handle_key(key(KeyCode::Down)); // Toggle to Deny
        let mut buf = Vec::new();
        sel.render(&mut buf).unwrap();
        let output = String::from_utf8_lossy(&buf);
        assert!(output.contains("> Deny [n]"), "got: {output}");
        assert!(
            !output.contains("> Approve [y]"),
            "Approve should not be selected, got: {output}"
        );
    }

    // ── Unrelated key tests ──────────────────────────────────────────

    #[test]
    fn test_unrelated_keys_produce_none() {
        let mut sel = make_selector("req-1");
        assert!(matches!(
            sel.handle_key(key(KeyCode::Char('x'))),
            ConsentAction::None
        ));
        assert!(matches!(
            sel.handle_key(key(KeyCode::Char('3'))),
            ConsentAction::None
        ));
        assert!(matches!(
            sel.handle_key(key(KeyCode::F(1))),
            ConsentAction::None
        ));
        // Choice should be unchanged
        assert_eq!(sel.choice, ConsentChoice::Approve);
    }

    // ── Request ID preservation ──────────────────────────────────────

    #[test]
    fn test_request_id_preserved() {
        let mut sel = make_selector("req-42");
        let action = sel.handle_key(key(KeyCode::Enter));
        match action {
            ConsentAction::Confirmed { request_id, .. } => {
                assert_eq!(request_id, "req-42");
            }
            _ => panic!("expected Confirmed"),
        }
    }
}
