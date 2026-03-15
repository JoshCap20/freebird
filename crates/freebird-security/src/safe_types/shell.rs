//! Safe shell argument and bash command types.

use crate::error::SecurityError;
use crate::taint::Tainted;

// ── SafeShellArg ─────────────────────────────────────────────────

/// A shell argument that has been validated against forbidden characters.
///
/// Produced by: tool input extraction.
/// Consumed by: shell tool.
#[derive(Debug)]
pub struct SafeShellArg(String);

const MAX_ARG_LEN: usize = 4096;

/// Characters that could enable command injection.
///
/// Covers: pipes, command chaining, variable expansion, subshells,
/// redirection, quoting (which can break out of quoted contexts),
/// backslash escaping, glob expansion, and history expansion.
const FORBIDDEN_ARG_CHARS: &[char] = &[
    '|', ';', '&', '`', '$', '(', ')', '{', '}', '[', ']', '<', '>', '\'', '"', '\\', '*', '?',
    '!', '#', '~', '\n', '\r',
];

impl SafeShellArg {
    /// Validate untrusted input as a safe shell argument.
    ///
    /// - Rejects empty or whitespace-only input
    /// - Enforces maximum length (4KB)
    /// - Rejects characters that could enable command injection
    /// - Strips null bytes
    ///
    /// # Errors
    ///
    /// Returns `SecurityError::ForbiddenCharacter` if a disallowed character is found.
    /// Returns `SecurityError::InputTooLong` if input exceeds the length limit.
    pub fn from_tainted(t: &Tainted) -> Result<Self, SecurityError> {
        let raw = t.inner();

        if raw.trim().is_empty() {
            return Err(SecurityError::ForbiddenCharacter {
                character: ' ',
                context: "shell argument must not be empty".into(),
            });
        }

        if raw.len() > MAX_ARG_LEN {
            return Err(SecurityError::InputTooLong {
                max: MAX_ARG_LEN,
                actual: raw.len(),
            });
        }

        if raw.contains('\0') {
            return Err(SecurityError::ForbiddenCharacter {
                character: '\0',
                context: "null byte in shell argument".into(),
            });
        }

        if let Some(&c) = raw
            .chars()
            .collect::<Vec<_>>()
            .iter()
            .find(|c| FORBIDDEN_ARG_CHARS.contains(c))
        {
            return Err(SecurityError::ForbiddenCharacter {
                character: c,
                context: format!("forbidden character '{c}' in shell argument"),
            });
        }

        Ok(Self(raw.to_string()))
    }

    /// Access the validated shell argument.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

// ── SafeBashCommand ──────────────────────────────────────────────

/// A bash command string that has been length-checked and sanitized.
///
/// Unlike [`SafeShellArg`], this type intentionally allows shell
/// metacharacters (`|`, `;`, `&`, `$`, etc.) because it is designed
/// for raw `bash -c` execution where pipes, redirects, and compound
/// commands are the whole point.
///
/// Validation is limited to:
/// - Non-empty / non-whitespace-only
/// - Maximum length (32 KiB)
/// - No null bytes
/// - Control characters stripped (except `\n`, `\t`)
///
/// Produced by: tool input extraction (`extract_bash_command`).
/// Consumed by: `BashExecTool`.
#[derive(Debug)]
pub struct SafeBashCommand(String);

/// Maximum length of a bash command string (same as `SafeMessage`).
const MAX_BASH_COMMAND_LEN: usize = 32_768;

impl SafeBashCommand {
    /// Validate untrusted input as a bash command.
    ///
    /// - Rejects empty or whitespace-only input
    /// - Enforces maximum length (32 KiB)
    /// - Rejects null bytes
    /// - Strips control characters except `\n` and `\t`
    ///
    /// # Errors
    ///
    /// Returns `SecurityError::ForbiddenCharacter` if the input is empty
    /// or contains null bytes.
    /// Returns `SecurityError::InputTooLong` if input exceeds the length limit.
    pub fn from_tainted(t: &Tainted) -> Result<Self, SecurityError> {
        let raw = t.inner();

        if raw.trim().is_empty() {
            return Err(SecurityError::ForbiddenCharacter {
                character: ' ',
                context: "bash command must not be empty".into(),
            });
        }

        if raw.len() > MAX_BASH_COMMAND_LEN {
            return Err(SecurityError::InputTooLong {
                max: MAX_BASH_COMMAND_LEN,
                actual: raw.len(),
            });
        }

        if raw.contains('\0') {
            return Err(SecurityError::ForbiddenCharacter {
                character: '\0',
                context: "null byte in bash command".into(),
            });
        }

        // Strip control characters except newline and tab (same policy as SafeMessage).
        let clean: String = raw
            .chars()
            .filter(|c| !c.is_control() || *c == '\n' || *c == '\t')
            .collect();

        // Re-check after sanitization: input of only control chars becomes empty.
        if clean.trim().is_empty() {
            return Err(SecurityError::ForbiddenCharacter {
                character: ' ',
                context: "bash command must not be empty after sanitization".into(),
            });
        }

        Ok(Self(clean))
    }

    /// Access the validated bash command.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic
)]
mod tests {
    use super::*;
    use crate::taint::Tainted;

    // ── SafeBashCommand tests ────────────────────────────────────

    #[test]
    fn test_bash_command_accepts_pipes_and_metacharacters() {
        let t = Tainted::new("echo hello | grep h && ls; cat foo");
        let cmd = SafeBashCommand::from_tainted(&t).unwrap();
        assert_eq!(cmd.as_str(), "echo hello | grep h && ls; cat foo");
    }

    #[test]
    fn test_bash_command_accepts_dollar_backtick_redirects() {
        let t = Tainted::new("echo $HOME > /tmp/out 2>&1 `whoami`");
        let cmd = SafeBashCommand::from_tainted(&t).unwrap();
        assert!(cmd.as_str().contains('$'));
        assert!(cmd.as_str().contains('`'));
        assert!(cmd.as_str().contains('>'));
    }

    #[test]
    fn test_bash_command_rejects_empty() {
        let t = Tainted::new("");
        let err = SafeBashCommand::from_tainted(&t).unwrap_err();
        assert!(matches!(err, SecurityError::ForbiddenCharacter { .. }));
    }

    #[test]
    fn test_bash_command_rejects_whitespace_only() {
        let t = Tainted::new("   \t\n  ");
        let err = SafeBashCommand::from_tainted(&t).unwrap_err();
        assert!(matches!(err, SecurityError::ForbiddenCharacter { .. }));
    }

    #[test]
    fn test_bash_command_rejects_too_long() {
        let long = "a".repeat(MAX_BASH_COMMAND_LEN + 1);
        let t = Tainted::new(&long);
        let err = SafeBashCommand::from_tainted(&t).unwrap_err();
        assert!(matches!(err, SecurityError::InputTooLong { .. }));
    }

    #[test]
    fn test_bash_command_rejects_null_byte() {
        let t = Tainted::new("ls\0-la");
        let err = SafeBashCommand::from_tainted(&t).unwrap_err();
        assert!(matches!(err, SecurityError::ForbiddenCharacter { .. }));
    }

    #[test]
    fn test_bash_command_strips_control_chars() {
        let t = Tainted::new("echo\x07hello\x1bworld");
        let cmd = SafeBashCommand::from_tainted(&t).unwrap();
        assert_eq!(cmd.as_str(), "echohelloworld");
    }

    #[test]
    fn test_bash_command_rejects_only_control_chars() {
        let t = Tainted::new("\x07\x1b\x03");
        let err = SafeBashCommand::from_tainted(&t).unwrap_err();
        assert!(matches!(err, SecurityError::ForbiddenCharacter { .. }));
    }

    #[test]
    fn test_bash_command_preserves_newlines_and_tabs() {
        let t = Tainted::new("echo hello\n\techo world");
        let cmd = SafeBashCommand::from_tainted(&t).unwrap();
        assert_eq!(cmd.as_str(), "echo hello\n\techo world");
    }

    // ── SafeShellArg tests ───────────────────────────────────────

    #[test]
    fn test_shell_arg_accepts_safe_input() {
        let t = Tainted::new("hello-world_123");
        let arg = SafeShellArg::from_tainted(&t).unwrap();
        assert_eq!(arg.as_str(), "hello-world_123");
    }

    #[test]
    fn test_shell_arg_rejects_pipe() {
        let t = Tainted::new("foo | bar");
        let err = SafeShellArg::from_tainted(&t).unwrap_err();
        assert!(matches!(err, SecurityError::ForbiddenCharacter { .. }));
    }

    #[test]
    fn test_shell_arg_rejects_semicolon() {
        let t = Tainted::new("foo; rm -rf /");
        let err = SafeShellArg::from_tainted(&t).unwrap_err();
        assert!(matches!(err, SecurityError::ForbiddenCharacter { .. }));
    }

    #[test]
    fn test_shell_arg_rejects_backtick() {
        let t = Tainted::new("foo`whoami`");
        let err = SafeShellArg::from_tainted(&t).unwrap_err();
        assert!(matches!(err, SecurityError::ForbiddenCharacter { .. }));
    }

    #[test]
    fn test_shell_arg_rejects_dollar() {
        let t = Tainted::new("$HOME");
        let err = SafeShellArg::from_tainted(&t).unwrap_err();
        assert!(matches!(err, SecurityError::ForbiddenCharacter { .. }));
    }

    #[test]
    fn test_shell_arg_rejects_empty() {
        let t = Tainted::new("");
        let err = SafeShellArg::from_tainted(&t).unwrap_err();
        assert!(matches!(err, SecurityError::ForbiddenCharacter { .. }));
    }

    #[test]
    fn test_shell_arg_rejects_too_long() {
        let long = "a".repeat(MAX_ARG_LEN + 1);
        let t = Tainted::new(&long);
        let err = SafeShellArg::from_tainted(&t).unwrap_err();
        assert!(matches!(err, SecurityError::InputTooLong { .. }));
    }

    #[test]
    fn test_shell_arg_rejects_null_byte() {
        let t = Tainted::new("foo\0bar");
        let err = SafeShellArg::from_tainted(&t).unwrap_err();
        assert!(matches!(err, SecurityError::ForbiddenCharacter { .. }));
    }
}
