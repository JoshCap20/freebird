//! Command completion — tab completion and command registry for `/commands`.
//!
//! This module is the **single source of truth** for all known client commands.
//! Both tab completion and `/help` output are derived from [`COMMANDS`].

use std::fmt::Write;

/// A known command definition.
pub struct CommandDef {
    /// The command name (without the leading `/`).
    pub name: &'static str,
    /// Human-readable description shown in `/help` output.
    pub description: &'static str,
}

/// All known commands available in the chat client.
///
/// This is the canonical command list — tab completion and help text are both
/// derived from it. To add a new command, add it here.
pub const COMMANDS: &[CommandDef] = &[
    CommandDef {
        name: "help",
        description: "Show available commands",
    },
    CommandDef {
        name: "quit",
        description: "Disconnect from the daemon",
    },
    CommandDef {
        name: "exit",
        description: "Disconnect from the daemon",
    },
    CommandDef {
        name: "new",
        description: "Start a new conversation",
    },
    CommandDef {
        name: "model",
        description: "Switch model",
    },
    CommandDef {
        name: "status",
        description: "Show session status",
    },
    CommandDef {
        name: "approve",
        description: "Approve a consent request",
    },
    CommandDef {
        name: "deny",
        description: "Deny a consent request",
    },
];

/// Generate help text from the command registry.
///
/// Skips `/exit` (duplicate of `/quit`) to keep the output clean.
#[must_use]
pub fn generate_help_text() -> String {
    let mut text = String::from("Available commands:\n");
    for cmd in COMMANDS {
        if cmd.name == "exit" {
            continue;
        }
        let _ = writeln!(text, "  /{:<16}{}", cmd.name, cmd.description);
    }
    text
}

/// Manages tab completion state for `/commands`.
pub struct CommandCompleter {
    /// Filtered matches for the current prefix.
    matches: Vec<&'static str>,
    /// Current cycle index within matches.
    cycle_index: usize,
    /// The prefix that generated the current matches.
    active_prefix: String,
}

impl CommandCompleter {
    /// Create a new completer.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            matches: Vec::new(),
            cycle_index: 0,
            active_prefix: String::new(),
        }
    }

    /// Get the next completion for the given input line.
    ///
    /// Returns `Some(completed_line)` if a completion is available, `None` otherwise.
    /// The input should start with `/`.
    #[must_use]
    pub fn complete(&mut self, input: &str) -> Option<String> {
        let prefix = input.strip_prefix('/')?;

        // If the prefix is the same, cycle to next match; otherwise recompute.
        if prefix == self.active_prefix {
            if !self.matches.is_empty() {
                self.cycle_index = (self.cycle_index + 1) % self.matches.len();
            }
        } else {
            self.active_prefix = prefix.to_string();
            self.matches = COMMANDS
                .iter()
                .filter(|cmd| cmd.name.starts_with(prefix))
                .map(|cmd| cmd.name)
                .collect();
            self.cycle_index = 0;
        }

        self.matches
            .get(self.cycle_index)
            .map(|name| format!("/{name}"))
    }

    /// Get the inline hint for the current input (the first completion, dimmed).
    ///
    /// Returns the suffix to display after the cursor (not the full command).
    #[must_use]
    #[allow(clippy::unused_self)] // Method for API consistency with complete()
    pub fn hint(&self, input: &str) -> Option<&'static str> {
        let prefix = input.strip_prefix('/')?;
        if prefix.is_empty() {
            return None;
        }

        COMMANDS
            .iter()
            .find(|cmd| cmd.name.starts_with(prefix) && cmd.name != prefix)
            .and_then(|cmd| cmd.name.get(prefix.len()..))
    }

    /// Reset completion state (call on any non-Tab keystroke).
    pub fn reset(&mut self) {
        self.matches.clear();
        self.cycle_index = 0;
        self.active_prefix.clear();
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn complete_basic() {
        let mut c = CommandCompleter::new();
        assert_eq!(c.complete("/he"), Some("/help".to_string()));
    }

    #[test]
    fn complete_cycles() {
        let mut c = CommandCompleter::new();
        // /e matches exit
        let first = c.complete("/e");
        assert_eq!(first, Some("/exit".to_string()));
        // cycling with same prefix
        let second = c.complete("/e");
        // could be exit again if only one match, which is fine
        assert!(second.is_some());
    }

    #[test]
    fn complete_no_match() {
        let mut c = CommandCompleter::new();
        assert_eq!(c.complete("/zzz"), None);
    }

    #[test]
    fn complete_non_slash_returns_none() {
        let mut c = CommandCompleter::new();
        assert_eq!(c.complete("hello"), None);
    }

    #[test]
    fn hint_basic() {
        let c = CommandCompleter::new();
        assert_eq!(c.hint("/he"), Some("lp"));
    }

    #[test]
    fn hint_no_match() {
        let c = CommandCompleter::new();
        assert_eq!(c.hint("/zzz"), None);
    }

    #[test]
    fn hint_exact_match_returns_none() {
        let c = CommandCompleter::new();
        assert_eq!(c.hint("/help"), None);
    }

    #[test]
    fn reset_clears_state() {
        let mut c = CommandCompleter::new();
        let _ = c.complete("/he");
        c.reset();
        assert!(c.matches.is_empty());
    }

    #[test]
    fn generate_help_text_contains_all_commands() {
        let text = generate_help_text();
        for name in ["help", "quit", "new", "model", "status", "approve", "deny"] {
            assert!(
                text.contains(&format!("/{name}")),
                "help text should contain /{name}, got: {text}"
            );
        }
        // /exit is skipped (duplicate of /quit)
        assert!(
            !text.contains("/exit"),
            "help text should not contain /exit (duplicate of /quit)"
        );
    }
}
