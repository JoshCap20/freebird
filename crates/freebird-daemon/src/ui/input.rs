//! Input editor — multi-line editing, history, and key handling.
//!
//! Renders a user input area at the bottom of the terminal. Handles key events
//! from crossterm, supports multi-line editing, command history navigation,
//! and tab completion for `/commands`.

use std::io::Write;

use crossterm::cursor::{MoveToColumn, MoveUp};
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use crossterm::style::{Attribute, Color, ResetColor, SetAttribute, SetForegroundColor};
use crossterm::terminal::{Clear, ClearType};

use super::completion::CommandCompleter;
use super::theme;

/// What the main loop should do after handling a key event.
#[derive(Debug)]
pub enum InputAction {
    /// User submitted their input — process it.
    Submit(String),
    /// Input changed — redraw the input area.
    Redraw,
    /// User wants to quit (Ctrl+D on empty or Ctrl+C twice).
    Quit,
    /// Nothing to do.
    None,
}

/// Manages the user's input buffer, cursor position, and history.
pub struct InputEditor {
    /// Current buffer lines (supports multi-line editing).
    lines: Vec<String>,
    /// Cursor row within the buffer (0-indexed).
    cursor_row: usize,
    /// Cursor column within the current line (0-indexed, in chars).
    cursor_col: usize,
    /// Command history (most recent last).
    history: Vec<String>,
    /// Current history navigation index (None = editing new input).
    history_index: Option<usize>,
    /// Saved input when navigating history.
    saved_input: Option<String>,
    /// Terminal width for wrapping calculations.
    term_width: u16,
    /// Tab completion state.
    completer: CommandCompleter,
    /// Whether the previous key was Ctrl+C (for double-Ctrl+C quit).
    last_was_ctrl_c: bool,
    /// The cursor's visual row from the top of the input area after the
    /// last `render()` call. Used to navigate back to the top on re-render
    /// or clear.
    last_cursor_visual_row: u16,
}

impl InputEditor {
    /// Create a new input editor.
    #[must_use]
    pub fn new(term_width: u16) -> Self {
        Self {
            lines: vec![String::new()],
            cursor_row: 0,
            cursor_col: 0,
            history: Vec::new(),
            history_index: None,
            saved_input: None,
            term_width,
            completer: CommandCompleter::new(),
            last_was_ctrl_c: false,
            last_cursor_visual_row: 0,
        }
    }

    /// Update the terminal width (e.g., on resize).
    pub const fn set_term_width(&mut self, width: u16) {
        self.term_width = width;
    }

    /// Get the current input as a single string (lines joined with newlines).
    #[must_use]
    pub fn content(&self) -> String {
        self.lines.join("\n")
    }

    /// Whether the input buffer is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.lines.len() == 1 && self.lines.first().is_none_or(String::is_empty)
    }

    /// Handle a key event and return the action for the main loop.
    pub fn handle_key(&mut self, key: KeyEvent) -> InputAction {
        let is_ctrl_c =
            key.code == KeyCode::Char('c') && key.modifiers.contains(KeyModifiers::CONTROL);

        // Reset completion on non-Tab keys
        if key.code != KeyCode::Tab {
            self.completer.reset();
        }

        // Track double Ctrl+C
        if !is_ctrl_c {
            self.last_was_ctrl_c = false;
        }

        if key.modifiers.contains(KeyModifiers::CONTROL) {
            if let Some(action) = self.handle_ctrl_key(key) {
                return action;
            }
        }

        self.handle_standard_key(key)
    }

    /// Handle Ctrl+key combinations. Returns `Some(action)` if handled.
    fn handle_ctrl_key(&mut self, key: KeyEvent) -> Option<InputAction> {
        match key.code {
            KeyCode::Char('c') => {
                if self.is_empty() {
                    if self.last_was_ctrl_c {
                        return Some(InputAction::Quit);
                    }
                    self.last_was_ctrl_c = true;
                    return Some(InputAction::None);
                }
                self.clear();
                self.last_was_ctrl_c = false;
                Some(InputAction::Redraw)
            }
            KeyCode::Char('d') => Some(if self.is_empty() {
                InputAction::Quit
            } else {
                InputAction::None
            }),
            KeyCode::Char('a') => {
                self.cursor_col = 0;
                Some(InputAction::Redraw)
            }
            KeyCode::Char('e') => {
                self.cursor_col = self.current_line_len();
                Some(InputAction::Redraw)
            }
            KeyCode::Char('u') => {
                if let Some(line) = self.lines.get_mut(self.cursor_row) {
                    line.clear();
                    self.cursor_col = 0;
                }
                Some(InputAction::Redraw)
            }
            KeyCode::Char('w') => {
                self.delete_word_backward();
                Some(InputAction::Redraw)
            }
            _ => None,
        }
    }

    /// Handle non-Ctrl key events.
    fn handle_standard_key(&mut self, key: KeyEvent) -> InputAction {
        match key.code {
            KeyCode::Enter
                if key
                    .modifiers
                    .intersects(KeyModifiers::SHIFT | KeyModifiers::ALT) =>
            {
                self.insert_newline();
                InputAction::Redraw
            }
            KeyCode::Enter => {
                let content = self.content();
                if content.trim().is_empty() {
                    return InputAction::Redraw;
                }
                self.push_history(&content);
                self.clear();
                InputAction::Submit(content)
            }
            KeyCode::Char(ch) => {
                self.insert_char(ch);
                InputAction::Redraw
            }
            KeyCode::Backspace => redraw_if(self.backspace()),
            KeyCode::Delete => redraw_if(self.delete_char()),
            KeyCode::Left => {
                self.move_left();
                InputAction::Redraw
            }
            KeyCode::Right => {
                self.move_right();
                InputAction::Redraw
            }
            KeyCode::Up => self.handle_up(),
            KeyCode::Down => self.handle_down(),
            KeyCode::Home => {
                self.cursor_col = 0;
                InputAction::Redraw
            }
            KeyCode::End => {
                self.cursor_col = self.current_line_len();
                InputAction::Redraw
            }
            KeyCode::Tab => {
                let input = self.content();
                self.completer
                    .complete(&input)
                    .map_or(InputAction::None, |completed| {
                        self.set_content(&completed);
                        InputAction::Redraw
                    })
            }
            _ => InputAction::None,
        }
    }

    /// Handle Up arrow — move within multi-line buffer or navigate history.
    fn handle_up(&mut self) -> InputAction {
        if self.cursor_row > 0 {
            self.cursor_row -= 1;
            self.clamp_cursor_col();
            InputAction::Redraw
        } else {
            redraw_if(self.history_up())
        }
    }

    /// Handle Down arrow — move within multi-line buffer or navigate history.
    fn handle_down(&mut self) -> InputAction {
        if self.cursor_row + 1 < self.lines.len() {
            self.cursor_row += 1;
            self.clamp_cursor_col();
            InputAction::Redraw
        } else {
            redraw_if(self.history_down())
        }
    }

    /// Render the input area to the writer.
    ///
    /// Clears the entire previous input area (including wrapped rows and
    /// multi-line content), redraws prompt + text, and positions the cursor.
    pub fn render<W: Write>(&mut self, w: &mut W) -> std::io::Result<()> {
        use crossterm::queue;

        let prompt_width = 3u16; // "> " = 3 chars

        // Move from the cursor's last visual position to the top of the
        // input area, then clear everything from there to the bottom of
        // the screen. This erases all stale wrapped/multi-line rows.
        if self.last_cursor_visual_row > 0 {
            queue!(w, MoveUp(self.last_cursor_visual_row))?;
        }
        queue!(w, MoveToColumn(0), Clear(ClearType::FromCursorDown))?;

        // Draw the prompt.
        theme::write_prompt_styled(w)?;

        // Draw the input text.
        for (i, line) in self.lines.iter().enumerate() {
            if i > 0 {
                writeln!(w)?;
                // Indent continuation lines to align with prompt.
                write!(w, "   ")?;
            }
            write!(w, "{line}")?;
        }

        // Draw completion hint (dim text after cursor).
        if self.lines.len() == 1 {
            let input = self.content();
            if let Some(hint) = self.completer.hint(&input) {
                queue!(
                    w,
                    SetAttribute(Attribute::Dim),
                    SetForegroundColor(Color::DarkGrey),
                )?;
                write!(w, "{hint}")?;
                queue!(w, ResetColor, SetAttribute(Attribute::Reset))?;
            }
        }

        // ── Cursor positioning ──────────────────────────────────────────
        // Compute the cursor's visual row from the top of the input area.
        let cursor_indent = if self.cursor_row == 0 {
            prompt_width
        } else {
            3u16
        };
        let cursor_col_u16 = u16::try_from(self.cursor_col).unwrap_or(u16::MAX);

        let mut cursor_visual_row: u16 = 0;
        // Add visual rows from lines above the cursor row.
        for i in 0..self.cursor_row {
            let indent = if i == 0 { prompt_width } else { 3u16 };
            let chars = u16::try_from(self.lines.get(i).map_or(0, |l| l.chars().count()))
                .unwrap_or(u16::MAX);
            cursor_visual_row =
                cursor_visual_row.saturating_add(visual_rows(indent, chars, self.term_width));
        }
        // Add wrapped rows within the cursor's own line.
        if let Some(wrapped) = cursor_indent
            .saturating_add(cursor_col_u16)
            .checked_div(self.term_width)
        {
            cursor_visual_row += wrapped;
        }

        // Compute total visual rows for all content.
        let mut total_visual_rows: u16 = 0;
        for (i, line) in self.lines.iter().enumerate() {
            let indent = if i == 0 { prompt_width } else { 3u16 };
            let chars = u16::try_from(line.chars().count()).unwrap_or(u16::MAX);
            total_visual_rows =
                total_visual_rows.saturating_add(visual_rows(indent, chars, self.term_width));
        }

        // Move up from the bottom of the rendered content to the cursor row.
        let visual_below = total_visual_rows.saturating_sub(cursor_visual_row + 1);
        if visual_below > 0 {
            queue!(w, MoveUp(visual_below))?;
        }

        let raw_col = cursor_indent.saturating_add(cursor_col_u16);
        let col_offset = if self.term_width > 0 {
            raw_col % self.term_width
        } else {
            raw_col
        };
        queue!(w, MoveToColumn(col_offset))?;

        // Remember cursor position for the next render/clear.
        self.last_cursor_visual_row = cursor_visual_row;

        w.flush()
    }

    /// Clear the entire visual area occupied by the input and reset tracking.
    ///
    /// Call this before writing output above the input area (e.g. server
    /// responses) so that stale wrapped/multi-line rows are erased.
    pub fn clear_visual_area<W: Write>(&mut self, w: &mut W) -> std::io::Result<()> {
        use crossterm::queue;
        if self.last_cursor_visual_row > 0 {
            queue!(w, MoveUp(self.last_cursor_visual_row))?;
        }
        queue!(w, MoveToColumn(0), Clear(ClearType::FromCursorDown))?;
        self.last_cursor_visual_row = 0;
        w.flush()
    }

    /// Get the number of terminal lines the input area occupies.
    #[must_use]
    #[allow(dead_code)]
    pub fn height(&self) -> u16 {
        u16::try_from(self.lines.len()).unwrap_or(u16::MAX)
    }

    // ── Private helpers ──────────────────────────────────────────────────────

    fn current_line_len(&self) -> usize {
        self.lines
            .get(self.cursor_row)
            .map_or(0, |l| l.chars().count())
    }

    fn clamp_cursor_col(&mut self) {
        let len = self.current_line_len();
        if self.cursor_col > len {
            self.cursor_col = len;
        }
    }

    fn insert_char(&mut self, ch: char) {
        if let Some(line) = self.lines.get_mut(self.cursor_row) {
            let byte_idx = char_to_byte_index(line, self.cursor_col);
            line.insert(byte_idx, ch);
            self.cursor_col += 1;
        }
    }

    fn insert_newline(&mut self) {
        if let Some(line) = self.lines.get(self.cursor_row).cloned() {
            let byte_idx = char_to_byte_index(&line, self.cursor_col);
            let remainder = line.get(byte_idx..).unwrap_or_default().to_string();
            if let Some(current) = self.lines.get_mut(self.cursor_row) {
                current.truncate(byte_idx);
            }
            self.cursor_row += 1;
            self.lines.insert(self.cursor_row, remainder);
            self.cursor_col = 0;
        }
    }

    fn backspace(&mut self) -> bool {
        if self.cursor_col > 0 {
            if let Some(line) = self.lines.get_mut(self.cursor_row) {
                let byte_idx = char_to_byte_index(line, self.cursor_col - 1);
                let next_byte_idx = char_to_byte_index(line, self.cursor_col);
                line.drain(byte_idx..next_byte_idx);
                self.cursor_col -= 1;
                return true;
            }
        } else if self.cursor_row > 0 {
            // Join with previous line
            let current = self.lines.remove(self.cursor_row);
            self.cursor_row -= 1;
            if let Some(prev) = self.lines.get_mut(self.cursor_row) {
                self.cursor_col = prev.chars().count();
                prev.push_str(&current);
            }
            return true;
        }
        false
    }

    fn delete_char(&mut self) -> bool {
        if let Some(line) = self.lines.get(self.cursor_row) {
            let len = line.chars().count();
            if self.cursor_col < len {
                let byte_idx = char_to_byte_index(line, self.cursor_col);
                let next_byte_idx = char_to_byte_index(line, self.cursor_col + 1);
                if let Some(line) = self.lines.get_mut(self.cursor_row) {
                    line.drain(byte_idx..next_byte_idx);
                }
                return true;
            } else if self.cursor_row + 1 < self.lines.len() {
                // Join with next line
                let next = self.lines.remove(self.cursor_row + 1);
                if let Some(current) = self.lines.get_mut(self.cursor_row) {
                    current.push_str(&next);
                }
                return true;
            }
        }
        false
    }

    fn delete_word_backward(&mut self) {
        if let Some(line) = self.lines.get_mut(self.cursor_row) {
            if self.cursor_col == 0 {
                return;
            }
            let chars: Vec<char> = line.chars().collect();
            let mut pos = self.cursor_col;
            // Skip trailing whitespace
            while pos > 0 && chars.get(pos - 1).is_some_and(|c| c.is_whitespace()) {
                pos -= 1;
            }
            // Skip word characters
            while pos > 0 && chars.get(pos - 1).is_some_and(|c| !c.is_whitespace()) {
                pos -= 1;
            }
            let start_byte = char_to_byte_index(line, pos);
            let end_byte = char_to_byte_index(line, self.cursor_col);
            line.drain(start_byte..end_byte);
            self.cursor_col = pos;
        }
    }

    fn move_left(&mut self) {
        if self.cursor_col > 0 {
            self.cursor_col -= 1;
        } else if self.cursor_row > 0 {
            self.cursor_row -= 1;
            self.cursor_col = self.current_line_len();
        }
    }

    fn move_right(&mut self) {
        let len = self.current_line_len();
        if self.cursor_col < len {
            self.cursor_col += 1;
        } else if self.cursor_row + 1 < self.lines.len() {
            self.cursor_row += 1;
            self.cursor_col = 0;
        }
    }

    fn history_up(&mut self) -> bool {
        if self.history.is_empty() {
            return false;
        }

        match self.history_index {
            None => {
                // Save current input and start browsing history
                self.saved_input = Some(self.content());
                self.history_index = Some(self.history.len() - 1);
            }
            Some(0) => return false, // Already at oldest
            Some(idx) => {
                self.history_index = Some(idx - 1);
            }
        }

        if let Some(idx) = self.history_index {
            if let Some(entry) = self.history.get(idx).cloned() {
                self.set_content(&entry);
            }
        }
        true
    }

    fn history_down(&mut self) -> bool {
        let Some(idx) = self.history_index else {
            return false;
        };

        if idx + 1 < self.history.len() {
            self.history_index = Some(idx + 1);
            if let Some(entry) = self.history.get(idx + 1).cloned() {
                self.set_content(&entry);
            }
        } else {
            // Back to current input
            self.history_index = None;
            let saved = self.saved_input.take().unwrap_or_default();
            self.set_content(&saved);
        }
        true
    }

    fn push_history(&mut self, content: &str) {
        let trimmed = content.trim().to_string();
        if trimmed.is_empty() {
            return;
        }
        // Don't duplicate the last entry
        if self.history.last() != Some(&trimmed) {
            self.history.push(trimmed);
        }
        self.history_index = None;
        self.saved_input = None;
    }

    fn clear(&mut self) {
        self.lines = vec![String::new()];
        self.cursor_row = 0;
        self.cursor_col = 0;
        self.history_index = None;
        self.saved_input = None;
    }

    fn set_content(&mut self, content: &str) {
        self.lines = content.split('\n').map(String::from).collect();
        if self.lines.is_empty() {
            self.lines.push(String::new());
        }
        self.cursor_row = self.lines.len() - 1;
        self.cursor_col = self.current_line_len();
    }
}

/// Return `Redraw` if changed, `None` otherwise.
const fn redraw_if(changed: bool) -> InputAction {
    if changed {
        InputAction::Redraw
    } else {
        InputAction::None
    }
}

/// Convert a character index to a byte index in a string.
fn char_to_byte_index(s: &str, char_idx: usize) -> usize {
    s.char_indices()
        .nth(char_idx)
        .map_or(s.len(), |(byte_idx, _)| byte_idx)
}

/// How many visual terminal rows a line occupies, given a left indent
/// of `indent` chars and a terminal width of `tw` columns.
const fn visual_rows(indent: u16, char_count: u16, tw: u16) -> u16 {
    if tw == 0 {
        return 1;
    }
    let total = indent.saturating_add(char_count);
    if total == 0 {
        return 1;
    }
    total.div_ceil(tw)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::indexing_slicing)]
mod tests {
    use super::*;

    #[test]
    fn new_editor_is_empty() {
        let editor = InputEditor::new(80);
        assert!(editor.is_empty());
        assert_eq!(editor.content(), "");
    }

    #[test]
    fn insert_char_updates_content() {
        let mut editor = InputEditor::new(80);
        editor.insert_char('h');
        editor.insert_char('i');
        assert_eq!(editor.content(), "hi");
        assert_eq!(editor.cursor_col, 2);
    }

    #[test]
    fn backspace_removes_char() {
        let mut editor = InputEditor::new(80);
        editor.insert_char('a');
        editor.insert_char('b');
        assert!(editor.backspace());
        assert_eq!(editor.content(), "a");
        assert_eq!(editor.cursor_col, 1);
    }

    #[test]
    fn backspace_at_start_returns_false() {
        let mut editor = InputEditor::new(80);
        assert!(!editor.backspace());
    }

    #[test]
    fn insert_newline_splits_line() {
        let mut editor = InputEditor::new(80);
        editor.insert_char('a');
        editor.insert_char('b');
        editor.cursor_col = 1; // Position between a and b
        editor.insert_newline();
        assert_eq!(editor.lines.len(), 2);
        assert_eq!(editor.lines[0], "a");
        assert_eq!(editor.lines[1], "b");
        assert_eq!(editor.cursor_row, 1);
        assert_eq!(editor.cursor_col, 0);
    }

    #[test]
    fn history_navigation() {
        let mut editor = InputEditor::new(80);
        editor.push_history("first");
        editor.push_history("second");

        assert!(editor.history_up());
        assert_eq!(editor.content(), "second");

        assert!(editor.history_up());
        assert_eq!(editor.content(), "first");

        assert!(editor.history_down());
        assert_eq!(editor.content(), "second");

        assert!(editor.history_down());
        assert_eq!(editor.content(), ""); // back to empty
    }

    #[test]
    fn history_up_on_empty_returns_false() {
        let mut editor = InputEditor::new(80);
        assert!(!editor.history_up());
    }

    #[test]
    fn clear_resets_state() {
        let mut editor = InputEditor::new(80);
        editor.insert_char('x');
        editor.clear();
        assert!(editor.is_empty());
        assert_eq!(editor.cursor_col, 0);
        assert_eq!(editor.cursor_row, 0);
    }

    #[test]
    fn set_content_multiline() {
        let mut editor = InputEditor::new(80);
        editor.set_content("line1\nline2\nline3");
        assert_eq!(editor.lines.len(), 3);
        assert_eq!(editor.cursor_row, 2);
        assert_eq!(editor.cursor_col, 5);
    }

    #[test]
    fn char_to_byte_index_ascii() {
        assert_eq!(char_to_byte_index("hello", 0), 0);
        assert_eq!(char_to_byte_index("hello", 3), 3);
        assert_eq!(char_to_byte_index("hello", 5), 5);
    }

    #[test]
    fn char_to_byte_index_unicode() {
        let s = "h\u{00e9}llo"; // héllo
        assert_eq!(char_to_byte_index(s, 0), 0);
        assert_eq!(char_to_byte_index(s, 1), 1); // 'é' starts at byte 1
        assert_eq!(char_to_byte_index(s, 2), 3); // 'l' starts at byte 3 (é is 2 bytes)
    }

    #[test]
    fn delete_word_backward() {
        let mut editor = InputEditor::new(80);
        editor.set_content("hello world");
        editor.delete_word_backward();
        assert_eq!(editor.content(), "hello ");
    }

    #[test]
    fn move_left_right() {
        let mut editor = InputEditor::new(80);
        editor.insert_char('a');
        editor.insert_char('b');
        assert_eq!(editor.cursor_col, 2);
        editor.move_left();
        assert_eq!(editor.cursor_col, 1);
        editor.move_right();
        assert_eq!(editor.cursor_col, 2);
    }

    #[test]
    fn duplicate_history_entries_deduplicated() {
        let mut editor = InputEditor::new(80);
        editor.push_history("same");
        editor.push_history("same");
        assert_eq!(editor.history.len(), 1);
    }

    #[test]
    fn visual_rows_basic() {
        // 3 indent + 10 chars = 13, fits in 80 cols = 1 row
        assert_eq!(visual_rows(3, 10, 80), 1);
        // 3 indent + 77 chars = 80, exactly fits = 1 row
        assert_eq!(visual_rows(3, 77, 80), 1);
        // 3 indent + 78 chars = 81, wraps to 2 rows
        assert_eq!(visual_rows(3, 78, 80), 2);
        // 3 indent + 157 chars = 160, exactly 2 rows
        assert_eq!(visual_rows(3, 157, 80), 2);
    }

    #[test]
    fn visual_rows_zero_width_safe() {
        // Should not panic on zero terminal width
        assert_eq!(visual_rows(3, 10, 0), 1);
        assert_eq!(visual_rows(0, 0, 0), 1);
    }

    #[test]
    fn cursor_wraps_at_term_width() {
        let mut editor = InputEditor::new(20);
        // Type 25 characters — prompt(3) + 25 = 28, wraps on a 20-col terminal
        for _ in 0..25 {
            editor.insert_char('x');
        }
        let mut buf = Vec::new();
        editor.render(&mut buf).unwrap();
        let output = String::from_utf8_lossy(&buf);
        // The cursor should be at column (3+25) % 20 = 8
        // MoveToColumn uses 0-indexed crossterm column positioning
        assert!(
            output.contains("\x1b[9G") || output.contains("\x1b[8G"),
            "cursor should wrap to column 8 on 20-col terminal, got: {output:?}"
        );
    }

    #[test]
    fn cursor_no_wrap_short_input() {
        let mut editor = InputEditor::new(80);
        for _ in 0..5 {
            editor.insert_char('a');
        }
        let mut buf = Vec::new();
        editor.render(&mut buf).unwrap();
        let output = String::from_utf8_lossy(&buf);
        // prompt(3) + 5 = 8, MoveToColumn(8) = \x1b[9G (1-indexed in escape)
        assert!(
            output.contains("\x1b[9G"),
            "cursor at col 8 on 80-col terminal, got: {output:?}"
        );
    }
}
