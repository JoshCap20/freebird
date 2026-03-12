//! Interactive consent selector widget for the TUI.
//!
//! When the daemon requests consent for a high-risk tool invocation, this
//! widget replaces the normal input editor with a two-option selector
//! (Approve / Deny) that the user navigates with arrow keys or shortcut keys.

use std::io::Write;

use chrono::{DateTime, Utc};
use crossterm::cursor::MoveUp;
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use crossterm::queue;
use crossterm::style::{Attribute, Color, ResetColor, SetAttribute, SetForegroundColor};
use crossterm::terminal::{Clear, ClearType};

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

/// Budget-specific approval choices (4 options instead of 2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BudgetChoice {
    ApproveOnce,
    DoubleLimit,
    DisableLimit,
    Deny,
}

impl BudgetChoice {
    /// Move to the next option (wraps around).
    const fn next(self) -> Self {
        match self {
            Self::ApproveOnce => Self::DoubleLimit,
            Self::DoubleLimit => Self::DisableLimit,
            Self::DisableLimit => Self::Deny,
            Self::Deny => Self::ApproveOnce,
        }
    }

    /// Move to the previous option (wraps around).
    const fn prev(self) -> Self {
        match self {
            Self::ApproveOnce => Self::Deny,
            Self::DoubleLimit => Self::ApproveOnce,
            Self::DisableLimit => Self::DoubleLimit,
            Self::Deny => Self::DisableLimit,
        }
    }
}

/// Budget info passed when creating a budget-mode consent selector.
#[derive(Debug, Clone)]
pub struct BudgetInfo {
    /// The current limit value.
    pub current_limit: u64,
}

/// Which mode the selector is in.
#[derive(Debug, Clone)]
enum SelectorMode {
    /// Standard 2-option (Approve / Deny).
    Standard(ConsentChoice),
    /// Budget 4-option (Approve once / Double limit / Disable limit / Deny).
    Budget {
        choice: BudgetChoice,
        info: BudgetInfo,
    },
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
        /// Budget override action string, if this was a budget approval.
        budget_action: Option<String>,
    },
    /// Selection changed — redraw the selector.
    Redraw,
    /// No action needed.
    None,
}

/// Number of terminal lines for the standard 2-option selector.
const STANDARD_SELECTOR_LINES: u16 = 2;

/// Number of terminal lines for the budget 4-option selector.
const BUDGET_SELECTOR_LINES: u16 = 4;

/// Interactive consent selector widget.
///
/// Renders either a two-option selector (Approve / Deny) or a four-option
/// budget selector (Approve once / Double limit / Disable limit / Deny).
/// The widget does NOT send messages itself — it returns [`ConsentAction`]
/// and the caller (`TtyChat`) sends the response.
pub struct ConsentSelector {
    /// The pending consent request ID.
    request_id: String,
    /// Tool name for outcome display after confirmation.
    tool_name: String,
    /// Selector mode (standard or budget).
    mode: SelectorMode,
    /// When this request expires.
    expires_at: DateTime<Utc>,
}

impl ConsentSelector {
    /// Create a new standard (2-option) consent selector.
    ///
    /// `expires_at_str` is the RFC 3339 timestamp from the server.
    /// Returns `None` if the timestamp is unparseable or already expired.
    #[must_use]
    pub fn new(request_id: String, tool_name: String, expires_at_str: &str) -> Option<Self> {
        let expires_at = expires_at_str.parse::<DateTime<Utc>>().ok()?;
        if expires_at <= Utc::now() {
            return None;
        }
        Some(Self {
            request_id,
            tool_name,
            mode: SelectorMode::Standard(ConsentChoice::Approve),
            expires_at,
        })
    }

    /// Create a budget-mode (4-option) consent selector.
    ///
    /// Returns `None` if the timestamp is unparseable or already expired.
    #[must_use]
    pub fn new_budget(
        request_id: String,
        tool_name: String,
        expires_at_str: &str,
        budget_info: BudgetInfo,
    ) -> Option<Self> {
        let expires_at = expires_at_str.parse::<DateTime<Utc>>().ok()?;
        if expires_at <= Utc::now() {
            return None;
        }
        Some(Self {
            request_id,
            tool_name,
            mode: SelectorMode::Budget {
                choice: BudgetChoice::ApproveOnce,
                info: budget_info,
            },
            expires_at,
        })
    }

    /// Handle a key event. Returns the action for the caller.
    pub fn handle_key(&mut self, key: KeyEvent) -> ConsentAction {
        // Ctrl+C → deny immediately
        if key.code == KeyCode::Char('c') && key.modifiers.contains(KeyModifiers::CONTROL) {
            return self.confirm_deny(None);
        }

        match &mut self.mode {
            SelectorMode::Standard(choice) => {
                Self::handle_key_standard(key, choice, &self.request_id)
            }
            SelectorMode::Budget { choice, info } => {
                Self::handle_key_budget(key, choice, info, &self.request_id)
            }
        }
    }

    /// Render the selector widget.
    pub fn render<W: Write>(&self, w: &mut W) -> std::io::Result<()> {
        match &self.mode {
            SelectorMode::Standard(choice) => {
                Self::render_line(w, *choice == ConsentChoice::Approve, "Approve", "[y]")?;
                writeln!(w)?;
                Self::render_line(w, *choice == ConsentChoice::Deny, "Deny", "[n]")?;
                writeln!(w)?;
            }
            SelectorMode::Budget { choice, info } => {
                let doubled = info.current_limit.saturating_mul(2);
                let doubled_label = format!("Double limit to {doubled}");

                Self::render_line(
                    w,
                    *choice == BudgetChoice::ApproveOnce,
                    "Approve once",
                    "[1]",
                )?;
                writeln!(w)?;
                Self::render_line(
                    w,
                    *choice == BudgetChoice::DoubleLimit,
                    &doubled_label,
                    "[2]",
                )?;
                writeln!(w)?;
                Self::render_line(
                    w,
                    *choice == BudgetChoice::DisableLimit,
                    "Disable limit",
                    "[3]",
                )?;
                writeln!(w)?;
                Self::render_line(w, *choice == BudgetChoice::Deny, "Deny", "[n]")?;
                writeln!(w)?;
            }
        }
        w.flush()
    }

    /// Erase the selector lines from the terminal.
    ///
    /// Moves cursor up past the selector lines and clears from there down.
    /// Call this before re-rendering or when dismissing the selector.
    pub fn clear<W: Write>(&self, w: &mut W) -> std::io::Result<()> {
        let lines = self.selector_lines();
        queue!(w, MoveUp(lines), Clear(ClearType::FromCursorDown))?;
        w.flush()
    }

    /// Render a one-line summary after the user confirms or the request expires.
    pub fn render_outcome<W: Write>(&self, w: &mut W, approved: bool) -> std::io::Result<()> {
        if approved {
            queue!(
                w,
                SetAttribute(Attribute::Bold),
                SetForegroundColor(Color::Green),
            )?;
            write!(w, "  \u{2713} Approved: {}", self.tool_name)?;
        } else {
            queue!(
                w,
                SetAttribute(Attribute::Bold),
                SetForegroundColor(Color::Red),
            )?;
            write!(w, "  \u{2717} Denied: {}", self.tool_name)?;
        }
        queue!(w, ResetColor, SetAttribute(Attribute::Reset))?;
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
            budget_action: None,
        }
    }

    // ── Private helpers ──────────────────────────────────────────────────

    /// Number of terminal lines occupied by this selector.
    const fn selector_lines(&self) -> u16 {
        match &self.mode {
            SelectorMode::Standard(_) => STANDARD_SELECTOR_LINES,
            SelectorMode::Budget { .. } => BUDGET_SELECTOR_LINES,
        }
    }

    fn confirm_deny(&self, reason: Option<String>) -> ConsentAction {
        ConsentAction::Confirmed {
            request_id: self.request_id.clone(),
            approved: false,
            reason,
            budget_action: None,
        }
    }

    fn handle_key_standard(
        key: KeyEvent,
        choice: &mut ConsentChoice,
        request_id: &str,
    ) -> ConsentAction {
        match key.code {
            KeyCode::Up | KeyCode::Down | KeyCode::Tab => {
                *choice = choice.toggle();
                ConsentAction::Redraw
            }
            KeyCode::Enter => {
                let approved = *choice == ConsentChoice::Approve;
                ConsentAction::Confirmed {
                    request_id: request_id.to_string(),
                    approved,
                    reason: None,
                    budget_action: None,
                }
            }
            KeyCode::Char('y' | 'a') => ConsentAction::Confirmed {
                request_id: request_id.to_string(),
                approved: true,
                reason: None,
                budget_action: None,
            },
            KeyCode::Char('n' | 'd') | KeyCode::Esc => ConsentAction::Confirmed {
                request_id: request_id.to_string(),
                approved: false,
                reason: None,
                budget_action: None,
            },
            _ => ConsentAction::None,
        }
    }

    fn handle_key_budget(
        key: KeyEvent,
        choice: &mut BudgetChoice,
        info: &BudgetInfo,
        request_id: &str,
    ) -> ConsentAction {
        match key.code {
            KeyCode::Down | KeyCode::Tab => {
                *choice = choice.next();
                ConsentAction::Redraw
            }
            KeyCode::Up => {
                *choice = choice.prev();
                ConsentAction::Redraw
            }
            KeyCode::Enter => Self::budget_confirm(*choice, info, request_id),
            KeyCode::Char('y' | '1') => {
                Self::budget_confirm(BudgetChoice::ApproveOnce, info, request_id)
            }
            KeyCode::Char('2') => Self::budget_confirm(BudgetChoice::DoubleLimit, info, request_id),
            KeyCode::Char('3') => {
                Self::budget_confirm(BudgetChoice::DisableLimit, info, request_id)
            }
            KeyCode::Char('n') | KeyCode::Esc => {
                Self::budget_confirm(BudgetChoice::Deny, info, request_id)
            }
            _ => ConsentAction::None,
        }
    }

    fn budget_confirm(choice: BudgetChoice, info: &BudgetInfo, request_id: &str) -> ConsentAction {
        // Wire format strings match freebird_security::approval::BudgetOverrideAction::to_wire()
        let (approved, budget_action) = match choice {
            BudgetChoice::ApproveOnce => (true, Some("approve_once".to_string())),
            BudgetChoice::DoubleLimit => {
                let doubled = info.current_limit.saturating_mul(2);
                (true, Some(format!("raise_limit:{doubled}")))
            }
            BudgetChoice::DisableLimit => (true, Some("disable_limit".to_string())),
            BudgetChoice::Deny => (false, None),
        };
        ConsentAction::Confirmed {
            request_id: request_id.to_string(),
            approved,
            reason: None,
            budget_action,
        }
    }

    fn render_line<W: Write>(
        w: &mut W,
        selected: bool,
        label: &str,
        hint: &str,
    ) -> std::io::Result<()> {
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

    /// Helper to create a standard selector with a future expiry.
    fn make_selector(request_id: &str) -> ConsentSelector {
        let future = (Utc::now() + Duration::minutes(5)).to_rfc3339();
        ConsentSelector::new(request_id.to_string(), "test_tool".to_string(), &future).unwrap()
    }

    /// Helper to create a budget selector with a future expiry.
    fn make_budget_selector(request_id: &str, current_limit: u64) -> ConsentSelector {
        let future = (Utc::now() + Duration::minutes(5)).to_rfc3339();
        ConsentSelector::new_budget(
            request_id.to_string(),
            "test_tool".to_string(),
            &future,
            BudgetInfo { current_limit },
        )
        .unwrap()
    }

    // ── Construction tests ───────────────────────────────────────────

    #[test]
    fn test_new_returns_none_for_expired_timestamp() {
        let past = (Utc::now() - Duration::minutes(1)).to_rfc3339();
        assert!(ConsentSelector::new("req-1".to_string(), "tool".to_string(), &past).is_none());
    }

    #[test]
    fn test_new_returns_some_for_valid_timestamp() {
        let sel = make_selector("req-1");
        assert!(matches!(
            sel.mode,
            SelectorMode::Standard(ConsentChoice::Approve)
        ));
    }

    #[test]
    fn test_new_returns_none_for_invalid_timestamp() {
        assert!(
            ConsentSelector::new("req-1".to_string(), "tool".to_string(), "not-a-date").is_none()
        );
    }

    #[test]
    fn test_new_budget_returns_some_for_valid_timestamp() {
        let sel = make_budget_selector("req-1", 32768);
        assert!(matches!(sel.mode, SelectorMode::Budget { .. }));
    }

    #[test]
    fn test_new_budget_returns_none_for_expired() {
        let past = (Utc::now() - Duration::minutes(1)).to_rfc3339();
        let info = BudgetInfo {
            current_limit: 32768,
        };
        assert!(
            ConsentSelector::new_budget("req-1".to_string(), "tool".to_string(), &past, info)
                .is_none()
        );
    }

    // ── Standard navigation tests ────────────────────────────────────

    #[test]
    fn test_toggle_choice() {
        let mut sel = make_selector("req-1");
        assert!(matches!(
            sel.mode,
            SelectorMode::Standard(ConsentChoice::Approve)
        ));

        let action = sel.handle_key(key(KeyCode::Down));
        assert!(matches!(action, ConsentAction::Redraw));
        assert!(matches!(
            sel.mode,
            SelectorMode::Standard(ConsentChoice::Deny)
        ));

        let action = sel.handle_key(key(KeyCode::Up));
        assert!(matches!(action, ConsentAction::Redraw));
        assert!(matches!(
            sel.mode,
            SelectorMode::Standard(ConsentChoice::Approve)
        ));

        // Tab also toggles
        let action = sel.handle_key(key(KeyCode::Tab));
        assert!(matches!(action, ConsentAction::Redraw));
        assert!(matches!(
            sel.mode,
            SelectorMode::Standard(ConsentChoice::Deny)
        ));
    }

    // ── Standard confirm tests ───────────────────────────────────────

    #[test]
    fn test_enter_confirms_approve() {
        let mut sel = make_selector("req-1");
        let action = sel.handle_key(key(KeyCode::Enter));
        match action {
            ConsentAction::Confirmed {
                approved,
                reason,
                budget_action,
                ..
            } => {
                assert!(approved);
                assert!(reason.is_none());
                assert!(budget_action.is_none());
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

    // ── Standard shortcut key tests ──────────────────────────────────

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
        let sel = make_selector("req-1");
        assert!(!sel.is_expired());

        let expired_sel = ConsentSelector {
            request_id: "req-expired".to_string(),
            tool_name: "test_tool".to_string(),
            mode: SelectorMode::Standard(ConsentChoice::Approve),
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
                budget_action,
            } => {
                assert_eq!(request_id, "req-42");
                assert!(!approved);
                assert_eq!(reason.as_deref(), Some("timeout"));
                assert!(budget_action.is_none());
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

    // ── Outcome render tests ───────────────────────────────────────

    #[test]
    fn test_render_outcome_approved() {
        let sel = make_selector("req-1");
        let mut buf = Vec::new();
        sel.render_outcome(&mut buf, true).unwrap();
        let output = String::from_utf8_lossy(&buf);
        assert!(output.contains("Approved: test_tool"), "got: {output}");
    }

    #[test]
    fn test_render_outcome_denied() {
        let sel = make_selector("req-1");
        let mut buf = Vec::new();
        sel.render_outcome(&mut buf, false).unwrap();
        let output = String::from_utf8_lossy(&buf);
        assert!(output.contains("Denied: test_tool"), "got: {output}");
    }

    // ── Standard unrelated key tests ─────────────────────────────────

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
        assert!(matches!(
            sel.mode,
            SelectorMode::Standard(ConsentChoice::Approve)
        ));
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

    // ── Budget mode tests ────────────────────────────────────────────

    #[test]
    fn test_budget_navigation_wraps() {
        let mut sel = make_budget_selector("req-1", 32768);

        // Down cycles: ApproveOnce → DoubleLimit → DisableLimit → Deny → ApproveOnce
        for expected in [
            BudgetChoice::DoubleLimit,
            BudgetChoice::DisableLimit,
            BudgetChoice::Deny,
            BudgetChoice::ApproveOnce,
        ] {
            let action = sel.handle_key(key(KeyCode::Down));
            assert!(matches!(action, ConsentAction::Redraw));
            match &sel.mode {
                SelectorMode::Budget { choice, .. } => assert_eq!(*choice, expected),
                SelectorMode::Standard(_) => panic!("expected Budget mode"),
            }
        }

        // Up goes backwards
        let _ = sel.handle_key(key(KeyCode::Up));
        match &sel.mode {
            SelectorMode::Budget { choice, .. } => assert_eq!(*choice, BudgetChoice::Deny),
            SelectorMode::Standard(_) => panic!("expected Budget mode"),
        }
    }

    #[test]
    fn test_budget_key_1_approves_once() {
        let mut sel = make_budget_selector("req-1", 32768);
        let action = sel.handle_key(key(KeyCode::Char('1')));
        match action {
            ConsentAction::Confirmed {
                approved,
                budget_action,
                ..
            } => {
                assert!(approved);
                assert_eq!(budget_action.as_deref(), Some("approve_once"));
            }
            _ => panic!("expected Confirmed"),
        }
    }

    #[test]
    fn test_budget_key_y_approves_once() {
        let mut sel = make_budget_selector("req-1", 32768);
        let action = sel.handle_key(key(KeyCode::Char('y')));
        match action {
            ConsentAction::Confirmed {
                approved,
                budget_action,
                ..
            } => {
                assert!(approved);
                assert_eq!(budget_action.as_deref(), Some("approve_once"));
            }
            _ => panic!("expected Confirmed"),
        }
    }

    #[test]
    fn test_budget_key_2_doubles_limit() {
        let mut sel = make_budget_selector("req-1", 32768);
        let action = sel.handle_key(key(KeyCode::Char('2')));
        match action {
            ConsentAction::Confirmed {
                approved,
                budget_action,
                ..
            } => {
                assert!(approved);
                assert_eq!(budget_action.as_deref(), Some("raise_limit:65536"));
            }
            _ => panic!("expected Confirmed"),
        }
    }

    #[test]
    fn test_budget_key_3_disables_limit() {
        let mut sel = make_budget_selector("req-1", 32768);
        let action = sel.handle_key(key(KeyCode::Char('3')));
        match action {
            ConsentAction::Confirmed {
                approved,
                budget_action,
                ..
            } => {
                assert!(approved);
                assert_eq!(budget_action.as_deref(), Some("disable_limit"));
            }
            _ => panic!("expected Confirmed"),
        }
    }

    #[test]
    fn test_budget_key_n_denies() {
        let mut sel = make_budget_selector("req-1", 32768);
        let action = sel.handle_key(key(KeyCode::Char('n')));
        match action {
            ConsentAction::Confirmed {
                approved,
                budget_action,
                ..
            } => {
                assert!(!approved);
                assert!(budget_action.is_none());
            }
            _ => panic!("expected Confirmed"),
        }
    }

    #[test]
    fn test_budget_enter_confirms_current_selection() {
        let mut sel = make_budget_selector("req-1", 32768);
        // Move to DoubleLimit
        let _ = sel.handle_key(key(KeyCode::Down));
        let action = sel.handle_key(key(KeyCode::Enter));
        match action {
            ConsentAction::Confirmed {
                approved,
                budget_action,
                ..
            } => {
                assert!(approved);
                assert_eq!(budget_action.as_deref(), Some("raise_limit:65536"));
            }
            _ => panic!("expected Confirmed"),
        }
    }

    #[test]
    fn test_budget_render_shows_four_options() {
        let sel = make_budget_selector("req-1", 32768);
        let mut buf = Vec::new();
        sel.render(&mut buf).unwrap();
        let output = String::from_utf8_lossy(&buf);
        assert!(output.contains("Approve once"), "got: {output}");
        assert!(output.contains("Double limit to 65536"), "got: {output}");
        assert!(output.contains("Disable limit"), "got: {output}");
        assert!(output.contains("Deny"), "got: {output}");
    }

    #[test]
    fn test_budget_render_highlights_selected() {
        let sel = make_budget_selector("req-1", 32768);
        let mut buf = Vec::new();
        sel.render(&mut buf).unwrap();
        let output = String::from_utf8_lossy(&buf);
        assert!(
            output.contains("> Approve once [1]"),
            "first option should be selected, got: {output}"
        );
    }

    #[test]
    fn test_budget_ctrl_c_denies() {
        let mut sel = make_budget_selector("req-1", 32768);
        let action = sel.handle_key(ctrl_key(KeyCode::Char('c')));
        match action {
            ConsentAction::Confirmed {
                approved,
                budget_action,
                ..
            } => {
                assert!(!approved);
                assert!(budget_action.is_none());
            }
            _ => panic!("expected Confirmed"),
        }
    }

    #[test]
    fn test_budget_saturating_double() {
        // When current limit is very large, doubling saturates to u64::MAX
        let sel = make_budget_selector("req-1", u64::MAX);
        let mut buf = Vec::new();
        sel.render(&mut buf).unwrap();
        let output = String::from_utf8_lossy(&buf);
        let max_str = u64::MAX.to_string();
        assert!(
            output.contains(&format!("Double limit to {max_str}")),
            "should saturate, got: {output}"
        );
    }

    #[test]
    fn test_budget_selector_lines() {
        let standard = make_selector("req-1");
        assert_eq!(standard.selector_lines(), 2);

        let budget = make_budget_selector("req-1", 32768);
        assert_eq!(budget.selector_lines(), 4);
    }
}
