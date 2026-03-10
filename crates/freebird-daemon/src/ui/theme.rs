//! Theme — all colors, styles, and formatting functions in one place.
//!
//! Used by both the interactive TUI mode and the plain pipe mode.
//! The `is_tty` parameter controls whether ANSI styling is applied.

use std::io::Write;

use crossterm::queue;
use crossterm::style::{Attribute, Color, ResetColor, SetAttribute, SetForegroundColor};

/// ANSI escape codes for direct string formatting (pipe-compatible path).
#[allow(dead_code)]
pub mod ansi {
    pub const RESET: &str = "\x1b[0m";
    pub const BOLD: &str = "\x1b[1m";
    pub const DIM: &str = "\x1b[2m";
    pub const CYAN: &str = "\x1b[36m";
    pub const GREEN: &str = "\x1b[32m";
    pub const RED: &str = "\x1b[31m";
    pub const YELLOW: &str = "\x1b[33m";
    pub const MAGENTA: &str = "\x1b[35m";
    pub const BLUE: &str = "\x1b[34m";
    pub const WHITE: &str = "\x1b[37m";
}

// ── Styled string builders (for pipe mode) ──────────────────────────────────

/// User input prompt: `> ` in bold cyan.
#[must_use]
#[allow(dead_code)]
pub fn user_prompt() -> String {
    format!("{}{}>{}  ", ansi::BOLD, ansi::CYAN, ansi::RESET)
}

/// Bot response prefix: `freebird >` in bold green.
#[must_use]
#[allow(dead_code)]
pub fn bot_prefix() -> String {
    format!("\n{}{}freebird >{} ", ansi::BOLD, ansi::GREEN, ansi::RESET)
}

/// Error prefix: `error:` in bold red.
#[must_use]
#[allow(dead_code)]
pub fn error_prefix(text: &str) -> String {
    format!("{}{}error:{} {text}\n", ansi::BOLD, ansi::RED, ansi::RESET)
}

/// Tool start: spinning indicator placeholder (for pipe mode, just show gear).
#[must_use]
#[allow(dead_code)]
pub fn tool_start(name: &str) -> String {
    format!(
        "{}{}  \u{25CB} {name}...{}",
        ansi::DIM,
        ansi::YELLOW,
        ansi::RESET
    )
}

/// Tool end: checkmark or cross based on outcome.
#[must_use]
#[allow(dead_code)]
pub fn tool_end(name: &str, outcome: &str, ms: u64) -> String {
    let icon = if outcome == "success" {
        "\u{2713}"
    } else {
        "\u{2717}"
    };
    format!(
        "{}{}  {icon} {name} ({outcome}, {ms}ms){}",
        ansi::DIM,
        ansi::YELLOW,
        ansi::RESET
    )
}

/// System message: `[text]` in dim magenta.
#[must_use]
#[allow(dead_code)]
pub fn system_msg(text: &str) -> String {
    format!("{}{}[{text}]{}\n", ansi::DIM, ansi::MAGENTA, ansi::RESET)
}

/// Token usage display: dim right-aligned.
#[must_use]
#[allow(dead_code)]
pub fn token_usage_line(input_tokens: u32, output_tokens: u32) -> String {
    format!(
        "\n{}{}{} tokens in \u{2022} {} tokens out{}\n",
        ansi::DIM,
        ansi::WHITE,
        format_number(input_tokens),
        format_number(output_tokens),
        ansi::RESET,
    )
}

/// Session info header.
#[must_use]
#[allow(dead_code)]
pub fn session_header(model: &str, session_id: &str) -> String {
    format!(
        "{}{}model: {model} \u{2022} session: {}{}\n",
        ansi::DIM,
        ansi::WHITE,
        truncate_id(session_id),
        ansi::RESET,
    )
}

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

/// Write tool start (circle indicator) using crossterm styling.
#[allow(dead_code)]
pub fn write_tool_start_styled<W: Write>(w: &mut W, name: &str) -> std::io::Result<()> {
    queue!(
        w,
        SetAttribute(Attribute::Dim),
        SetForegroundColor(Color::Yellow),
    )?;
    write!(w, "  \u{25CB} {name}...")?;
    queue!(w, ResetColor, SetAttribute(Attribute::Reset))?;
    w.flush()
}

/// Write tool end (check/cross) using crossterm styling.
#[allow(dead_code)]
pub fn write_tool_end_styled<W: Write>(
    w: &mut W,
    name: &str,
    outcome: &str,
    ms: u64,
) -> std::io::Result<()> {
    let icon = if outcome == "success" {
        "\u{2713}"
    } else {
        "\u{2717}"
    };
    queue!(
        w,
        SetAttribute(Attribute::Dim),
        SetForegroundColor(Color::Yellow),
    )?;
    write!(w, "  {icon} {name} ({outcome}, {ms}ms)")?;
    queue!(w, ResetColor, SetAttribute(Attribute::Reset))?;
    writeln!(w)?;
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

    #[test]
    fn user_prompt_contains_arrow() {
        let p = user_prompt();
        assert!(p.contains('>'), "prompt should contain >");
    }

    #[test]
    fn bot_prefix_contains_freebird() {
        let p = bot_prefix();
        assert!(p.contains("freebird"), "prefix should contain freebird");
    }

    #[test]
    fn error_prefix_contains_text() {
        let e = error_prefix("boom");
        assert!(e.contains("boom"));
        assert!(e.contains("error:"));
    }
}
