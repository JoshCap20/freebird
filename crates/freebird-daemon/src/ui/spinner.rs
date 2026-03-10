//! Tool spinner — animated braille spinner for tool execution status.

use std::io::Write;
use std::time::Instant;

use crossterm::cursor::MoveToColumn;
use crossterm::queue;
use crossterm::style::{Attribute, Color, ResetColor, SetAttribute, SetForegroundColor};
use crossterm::terminal::{Clear, ClearType};

/// Braille spinner frames for smooth animation.
const SPINNER_FRAMES: &[char] = &[
    '\u{2800}', // ⠀ (blank for contrast)
    '\u{280B}', // ⠋
    '\u{2819}', // ⠙
    '\u{2839}', // ⠹
    '\u{2838}', // ⠸
    '\u{283C}', // ⠼
    '\u{2834}', // ⠴
    '\u{2826}', // ⠦
    '\u{2827}', // ⠧
    '\u{2807}', // ⠇
    '\u{280F}', // ⠏
];

/// A spinner that shows animated progress for a running tool.
pub struct ToolSpinner {
    /// Name of the currently running tool (None if idle).
    active_tool: Option<ActiveTool>,
    /// Current frame index.
    frame_idx: usize,
    /// Number of lines the spinner occupies (for clearing).
    lines_occupied: u16,
}

struct ActiveTool {
    name: String,
    started_at: Instant,
}

impl ToolSpinner {
    /// Create a new idle spinner.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            active_tool: None,
            frame_idx: 0,
            lines_occupied: 0,
        }
    }

    /// Whether a tool is currently running.
    #[must_use]
    pub const fn is_active(&self) -> bool {
        self.active_tool.is_some()
    }

    /// Start the spinner for a tool.
    pub fn start<W: Write>(&mut self, w: &mut W, tool_name: &str) -> std::io::Result<()> {
        self.active_tool = Some(ActiveTool {
            name: tool_name.to_string(),
            started_at: Instant::now(),
        });
        self.frame_idx = 1; // Start at first visible frame
        self.render(w)
    }

    /// Advance the spinner by one frame and redraw.
    pub fn tick<W: Write>(&mut self, w: &mut W) -> std::io::Result<()> {
        if self.active_tool.is_none() {
            return Ok(());
        }
        self.frame_idx = (self.frame_idx + 1) % SPINNER_FRAMES.len();
        self.redraw(w)
    }

    /// Stop the spinner and show the final result.
    pub fn stop<W: Write>(
        &mut self,
        w: &mut W,
        tool_name: &str,
        outcome: &str,
        duration_ms: u64,
    ) -> std::io::Result<()> {
        // Clear the spinner line
        self.clear(w)?;

        let icon = if outcome == "success" {
            '\u{2713}'
        } else {
            '\u{2717}'
        };
        let color = if outcome == "success" {
            Color::Green
        } else {
            Color::Red
        };

        queue!(w, SetAttribute(Attribute::Dim), SetForegroundColor(color),)?;
        write!(w, "  {icon} {tool_name} ({outcome}, {duration_ms}ms)")?;
        queue!(w, ResetColor, SetAttribute(Attribute::Reset))?;
        writeln!(w)?;
        w.flush()?;

        self.active_tool = None;
        self.lines_occupied = 0;
        Ok(())
    }

    /// Render the spinner (initial draw).
    fn render<W: Write>(&mut self, w: &mut W) -> std::io::Result<()> {
        let Some(tool) = &self.active_tool else {
            return Ok(());
        };

        let frame = SPINNER_FRAMES.get(self.frame_idx).copied().unwrap_or(' ');

        queue!(
            w,
            SetAttribute(Attribute::Dim),
            SetForegroundColor(Color::Yellow),
        )?;
        write!(w, "  {frame} {}...", tool.name)?;
        queue!(w, ResetColor, SetAttribute(Attribute::Reset))?;
        w.flush()?;

        self.lines_occupied = 1;
        Ok(())
    }

    /// Redraw the spinner in place (update the frame character).
    fn redraw<W: Write>(&self, w: &mut W) -> std::io::Result<()> {
        let Some(tool) = &self.active_tool else {
            return Ok(());
        };

        let frame = SPINNER_FRAMES.get(self.frame_idx).copied().unwrap_or(' ');

        // Move to start of current line and clear it
        queue!(w, MoveToColumn(0), Clear(ClearType::CurrentLine))?;
        queue!(
            w,
            SetAttribute(Attribute::Dim),
            SetForegroundColor(Color::Yellow),
        )?;
        write!(w, "  {frame} {}...", tool.name)?;
        queue!(w, ResetColor, SetAttribute(Attribute::Reset))?;
        w.flush()
    }

    /// Clear the spinner line(s).
    fn clear<W: Write>(&self, w: &mut W) -> std::io::Result<()> {
        if self.lines_occupied == 0 {
            return Ok(());
        }
        queue!(w, MoveToColumn(0), Clear(ClearType::CurrentLine))?;
        w.flush()
    }

    /// Get the elapsed time for the current tool (for display if needed).
    #[must_use]
    #[allow(dead_code)]
    pub fn elapsed_ms(&self) -> Option<u64> {
        self.active_tool
            .as_ref()
            .map(|t| u64::try_from(t.started_at.elapsed().as_millis()).unwrap_or(u64::MAX))
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn spinner_starts_inactive() {
        let s = ToolSpinner::new();
        assert!(!s.is_active());
        assert!(s.elapsed_ms().is_none());
    }

    #[test]
    fn spinner_start_makes_active() {
        let mut s = ToolSpinner::new();
        let mut buf = Vec::new();
        s.start(&mut buf, "read_file").unwrap();
        assert!(s.is_active());
        assert!(s.elapsed_ms().is_some());
        let output = String::from_utf8_lossy(&buf);
        assert!(output.contains("read_file"));
    }

    #[test]
    fn spinner_stop_makes_inactive() {
        let mut s = ToolSpinner::new();
        let mut buf = Vec::new();
        s.start(&mut buf, "read_file").unwrap();
        s.stop(&mut buf, "read_file", "success", 42).unwrap();
        assert!(!s.is_active());
    }

    #[test]
    fn spinner_tick_advances_frame() {
        let mut s = ToolSpinner::new();
        let mut buf = Vec::new();
        s.start(&mut buf, "shell").unwrap();
        let initial = s.frame_idx;
        s.tick(&mut buf).unwrap();
        assert_ne!(s.frame_idx, initial);
    }

    #[test]
    fn spinner_tick_when_idle_is_noop() {
        let mut s = ToolSpinner::new();
        let mut buf = Vec::new();
        s.tick(&mut buf).unwrap();
        assert!(buf.is_empty());
    }
}
