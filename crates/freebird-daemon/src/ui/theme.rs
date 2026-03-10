//! TUI-only styling functions for the interactive crossterm chat mode.
//!
//! All colors, styles, and formatting helpers used by the TUI components
//! live here. The pipe-mode path (`chat.rs`) has its own inline `style` module.

use std::io::Write;

use crossterm::queue;
use crossterm::style::{Attribute, Color, ResetColor, SetAttribute, SetForegroundColor};

// ── Crossterm styled writes (for TUI mode) ──────────────────────────────────

/// Write the user input prompt using crossterm styling.
pub fn write_prompt_styled<W: Write>(w: &mut W) -> std::io::Result<()> {
    queue!(
        w,
        SetAttribute(Attribute::Bold),
        SetForegroundColor(Color::Cyan),
    )?;
    write!(w, ">")?;
    queue!(w, ResetColor, SetAttribute(Attribute::Reset))?;
    write!(w, "  ")?;
    w.flush()
}

/// Write the bot prefix using crossterm styling.
pub fn write_bot_prefix_styled<W: Write>(w: &mut W) -> std::io::Result<()> {
    writeln!(w)?;
    queue!(
        w,
        SetAttribute(Attribute::Bold),
        SetForegroundColor(Color::Green),
    )?;
    write!(w, "freebird >")?;
    queue!(w, ResetColor, SetAttribute(Attribute::Reset))?;
    write!(w, " ")?;
    w.flush()
}

/// Write an error using crossterm styling.
pub fn write_error_styled<W: Write>(w: &mut W, text: &str) -> std::io::Result<()> {
    queue!(
        w,
        SetAttribute(Attribute::Bold),
        SetForegroundColor(Color::Red),
    )?;
    write!(w, "error:")?;
    queue!(w, ResetColor, SetAttribute(Attribute::Reset))?;
    writeln!(w, " {text}")?;
    w.flush()
}

/// Write token usage line using crossterm styling.
pub fn write_token_usage_styled<W: Write>(
    w: &mut W,
    input_tokens: u32,
    output_tokens: u32,
) -> std::io::Result<()> {
    writeln!(w)?;
    queue!(
        w,
        SetAttribute(Attribute::Dim),
        SetForegroundColor(Color::DarkGrey),
    )?;
    write!(
        w,
        "  {} tokens in \u{2022} {} tokens out",
        format_number(input_tokens),
        format_number(output_tokens),
    )?;
    queue!(w, ResetColor, SetAttribute(Attribute::Reset))?;
    writeln!(w)?;
    w.flush()
}

/// Write session header using crossterm styling.
pub fn write_session_header_styled<W: Write>(
    w: &mut W,
    model: &str,
    session_id: &str,
) -> std::io::Result<()> {
    queue!(
        w,
        SetAttribute(Attribute::Dim),
        SetForegroundColor(Color::DarkGrey),
    )?;
    write!(
        w,
        "  model: {model} \u{2022} session: {}",
        truncate_id(session_id)
    )?;
    queue!(w, ResetColor, SetAttribute(Attribute::Reset))?;
    writeln!(w)?;
    w.flush()
}

// ── Helpers ──────────────────────────────────────────────────────────────────

/// Format a number with comma separators (e.g., 1,234).
#[must_use]
fn format_number(n: u32) -> String {
    let s = n.to_string();
    let mut result = String::with_capacity(s.len() + s.len() / 3);
    for (i, ch) in s.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 {
            result.push(',');
        }
        result.push(ch);
    }
    result.chars().rev().collect()
}

/// Truncate a session ID to the first 8 characters for display.
#[must_use]
fn truncate_id(id: &str) -> &str {
    id.char_indices()
        .nth(8)
        .map_or(id, |(byte_pos, _)| id.get(..byte_pos).unwrap_or(id))
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn format_number_basic() {
        assert_eq!(format_number(0), "0");
        assert_eq!(format_number(42), "42");
        assert_eq!(format_number(999), "999");
        assert_eq!(format_number(1000), "1,000");
        assert_eq!(format_number(1_234_567), "1,234,567");
    }

    #[test]
    fn truncate_id_short() {
        assert_eq!(truncate_id("abc"), "abc");
        assert_eq!(truncate_id("12345678"), "12345678");
    }

    #[test]
    fn truncate_id_long() {
        assert_eq!(truncate_id("123456789abcdef"), "12345678");
    }
}
