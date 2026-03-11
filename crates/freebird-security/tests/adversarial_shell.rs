//! Adversarial tests for `SafeShellArg` — command injection attempts.

#![allow(
    clippy::all,
    clippy::pedantic,
    clippy::nursery,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

use freebird_security::safe_types::SafeShellArg;
use freebird_security::taint::Tainted;

// ---------------------------------------------------------------------------
// Injection attempts
// ---------------------------------------------------------------------------

#[test]
fn newline_injection() {
    let t = Tainted::new("safe-arg\nrm -rf /");
    let result = SafeShellArg::from_tainted(&t);
    assert!(result.is_err(), "newline must be rejected");
}

#[test]
fn carriage_return_injection() {
    let t = Tainted::new("safe-arg\rrm -rf /");
    let result = SafeShellArg::from_tainted(&t);
    assert!(result.is_err(), "carriage return must be rejected");
}

#[test]
fn pipe_injection() {
    let t = Tainted::new("file.txt | cat /etc/passwd");
    let result = SafeShellArg::from_tainted(&t);
    assert!(result.is_err(), "pipe must be rejected");
}

#[test]
fn semicolon_injection() {
    let t = Tainted::new("file.txt; rm -rf /");
    let result = SafeShellArg::from_tainted(&t);
    assert!(result.is_err(), "semicolon must be rejected");
}

#[test]
fn backtick_injection() {
    let t = Tainted::new("file.txt`whoami`");
    let result = SafeShellArg::from_tainted(&t);
    assert!(result.is_err(), "backtick must be rejected");
}

#[test]
fn dollar_subshell_injection() {
    let t = Tainted::new("file.txt$(whoami)");
    let result = SafeShellArg::from_tainted(&t);
    assert!(result.is_err(), "dollar subshell must be rejected");
}

#[test]
fn null_byte_injection() {
    let t = Tainted::new("file.txt\0--exec");
    let result = SafeShellArg::from_tainted(&t);
    assert!(result.is_err(), "null byte must be rejected");
}

// ---------------------------------------------------------------------------
// All forbidden chars individually
// ---------------------------------------------------------------------------

#[test]
fn all_forbidden_chars_individually_rejected() {
    let forbidden = [
        '|', ';', '&', '`', '$', '(', ')', '{', '}', '[', ']', '<', '>', '\'', '"', '\\', '*', '?',
        '!', '#', '~', '\n', '\r',
    ];

    for ch in &forbidden {
        let input = format!("safe{ch}arg");
        let t = Tainted::new(&input);
        let result = SafeShellArg::from_tainted(&t);
        assert!(
            result.is_err(),
            "character {:?} (U+{:04X}) must be rejected",
            ch,
            *ch as u32
        );
    }
}

// ---------------------------------------------------------------------------
// Edge cases
// ---------------------------------------------------------------------------

#[test]
fn empty_arg_rejected() {
    let t = Tainted::new("");
    let result = SafeShellArg::from_tainted(&t);
    assert!(result.is_err(), "empty arg must be rejected");
}

#[test]
fn whitespace_only_arg_rejected() {
    let t = Tainted::new("   ");
    let result = SafeShellArg::from_tainted(&t);
    assert!(result.is_err(), "whitespace-only arg must be rejected");
}

#[test]
fn very_long_arg_rejected() {
    let long = "a".repeat(5000);
    let t = Tainted::new(&long);
    let result = SafeShellArg::from_tainted(&t);
    assert!(result.is_err(), "arg exceeding 4096 chars must be rejected");
}

#[test]
fn valid_arg_passes() {
    let t = Tainted::new("simple-filename.txt");
    let result = SafeShellArg::from_tainted(&t);
    assert!(result.is_ok(), "simple filename should pass");
}

#[test]
fn arg_with_dashes_and_dots_passes() {
    let t = Tainted::new("my-file.2024-01-01.txt");
    let result = SafeShellArg::from_tainted(&t);
    assert!(result.is_ok(), "dashes and dots should be allowed");
}

#[test]
fn arg_with_unicode_letters_passes() {
    // Unicode letters without forbidden chars should pass
    let t = Tainted::new("archivo-español");
    let result = SafeShellArg::from_tainted(&t);
    assert!(result.is_ok(), "Unicode letters should be allowed");
}

// ---------------------------------------------------------------------------
// proptest: any string with forbidden char is rejected
// ---------------------------------------------------------------------------

#[cfg(test)]
mod proptest_shell {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn string_with_forbidden_char_rejected(
            prefix in "[a-z]{1,10}",
            forbidden in prop::sample::select(vec![
                '|', ';', '&', '`', '$', '(', ')', '{', '}', '[', ']',
                '<', '>', '\'', '"', '\\', '*', '?', '!', '#', '~', '\n', '\r'
            ]),
            suffix in "[a-z]{1,10}"
        ) {
            let input = format!("{prefix}{forbidden}{suffix}");
            let t = Tainted::new(&input);
            let result = SafeShellArg::from_tainted(&t);
            prop_assert!(
                result.is_err(),
                "string containing {:?} should be rejected",
                forbidden
            );
        }
    }
}
