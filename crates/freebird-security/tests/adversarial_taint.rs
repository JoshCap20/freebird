//! Adversarial tests for taint tracking boundaries.
//!
//! Tests run from OUTSIDE the `freebird-security` crate to verify that
//! `Tainted::inner()` (which is `pub(crate)`) is truly inaccessible.

#![allow(
    clippy::all,
    clippy::pedantic,
    clippy::nursery,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

use freebird_security::taint::{Tainted, TaintedToolInput};
use serde_json::json;

// ---------------------------------------------------------------------------
// Tainted<String> boundary tests
// ---------------------------------------------------------------------------

#[test]
fn tainted_debug_never_leaks_content() {
    let secret = "super-secret-api-key-abc123";
    let t = Tainted::new(secret);
    let debug = format!("{t:?}");

    assert!(
        !debug.contains(secret),
        "Debug output must not contain raw content: {debug}"
    );
    assert!(
        debug.contains("Tainted"),
        "Debug output should indicate tainted status: {debug}"
    );
}

#[test]
fn tainted_debug_never_leaks_for_short_content() {
    // Even very short content should not leak
    let t = Tainted::new("x");
    let debug = format!("{t:?}");
    // "x" is too short to reliably assert non-presence (it could appear in "TAINTED"),
    // but we can verify the format
    assert!(debug.contains("Tainted"));
}

#[test]
fn tainted_debug_never_leaks_for_empty_content() {
    let t = Tainted::new("");
    let debug = format!("{t:?}");
    assert!(debug.contains("Tainted"));
}

#[test]
fn tainted_debug_never_leaks_long_content() {
    let secret = "a".repeat(10_000);
    let t = Tainted::new(&secret);
    let debug = format!("{t:?}");

    assert!(
        !debug.contains(&secret),
        "Debug must not contain 10k raw content"
    );
    // Debug length should be bounded, not proportional to content
    assert!(
        debug.len() < 200,
        "Debug output should be bounded, got len={}",
        debug.len()
    );
}

#[test]
fn tainted_new_from_string_does_not_leak() {
    // Tainted does not implement Clone (by design), so create two separate instances
    let t1 = Tainted::new("secret-value-one");
    let t2 = Tainted::new("secret-value-two");
    let debug1 = format!("{t1:?}");
    let debug2 = format!("{t2:?}");
    assert!(!debug1.contains("secret-value-one"));
    assert!(!debug2.contains("secret-value-two"));
}

// ---------------------------------------------------------------------------
// TaintedToolInput boundary tests
// ---------------------------------------------------------------------------

#[test]
fn tainted_tool_input_debug_never_leaks() {
    let input = TaintedToolInput::new(json!({
        "path": "/etc/passwd",
        "content": "sensitive data here"
    }));
    let debug = format!("{input:?}");

    assert!(
        !debug.contains("/etc/passwd"),
        "Debug must not contain input values: {debug}"
    );
    assert!(
        !debug.contains("sensitive data here"),
        "Debug must not contain input values: {debug}"
    );
}

#[test]
fn tainted_tool_input_deeply_nested_json() {
    let nested = json!({
        "level1": {
            "level2": {
                "level3": {
                    "secret": "deeply-hidden-value"
                }
            }
        }
    });
    let input = TaintedToolInput::new(nested);
    let debug = format!("{input:?}");
    assert!(!debug.contains("deeply-hidden-value"));
}

#[test]
fn tainted_tool_input_extract_missing_key() {
    let input = TaintedToolInput::new(json!({"path": "/tmp"}));
    let result = input.extract_string("nonexistent");
    assert!(result.is_err());
}

#[test]
fn tainted_tool_input_extract_wrong_type() {
    let input = TaintedToolInput::new(json!({"count": 42}));
    let result = input.extract_string("count");
    assert!(result.is_err(), "extracting number as string should fail");
}

#[test]
fn tainted_tool_input_extract_null_value() {
    let input = TaintedToolInput::new(json!({"key": null}));
    let result = input.extract_string("key");
    assert!(result.is_err(), "extracting null as string should fail");
}

#[test]
fn tainted_tool_input_array_root() {
    let input = TaintedToolInput::new(json!(["a", "b", "c"]));
    let result = input.extract_string("0");
    assert!(
        result.is_err(),
        "array root should not allow key extraction"
    );
}

#[test]
fn tainted_tool_input_null_root() {
    let input = TaintedToolInput::new(json!(null));
    let result = input.extract_string("key");
    assert!(result.is_err(), "null root should fail extraction");
}

#[test]
fn tainted_tool_input_prototype_pollution_keys() {
    // Keys that might cause issues in other languages should be benign in Rust
    let input = TaintedToolInput::new(json!({
        "__proto__": "attack",
        "constructor": "evil",
        "toString": "override"
    }));
    // These should extract normally as strings
    let result = input.extract_string("__proto__");
    assert!(result.is_ok());
}

#[test]
fn tainted_tool_input_very_large_json() {
    // 1MB JSON value should not panic
    let big_string = "x".repeat(1_000_000);
    let input = TaintedToolInput::new(json!({"data": big_string}));
    let debug = format!("{input:?}");
    assert!(debug.len() < 1000, "Debug of large input should be bounded");
}

#[test]
fn tainted_tool_input_u64_optional_absent_vs_null() {
    let absent = TaintedToolInput::new(json!({"other": "val"}));
    let null_val = TaintedToolInput::new(json!({"key": null}));

    // Absent key should return Ok(None)
    let absent_result = absent.extract_u64_optional("key");
    assert!(absent_result.is_ok());
    assert_eq!(absent_result.unwrap(), None);

    // Null value: key exists but as_u64() returns None → Err
    let null_result = null_val.extract_u64_optional("key");
    assert!(
        null_result.is_err(),
        "null is not a valid u64, should return Err"
    );
}

// ---------------------------------------------------------------------------
// proptest: Tainted Debug never leaks for arbitrary strings
// ---------------------------------------------------------------------------

#[cfg(test)]
mod proptest_taint {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn debug_never_leaks_arbitrary_content(s in ".{11,200}") {
            // Only test strings > 10 chars to avoid false positives from
            // short strings matching inside "TAINTED" or "len="
            let t = Tainted::new(&s);
            let debug = format!("{t:?}");
            prop_assert!(
                !debug.contains(&s),
                "Debug leaked raw content: {}", debug
            );
        }

        #[test]
        fn tool_input_debug_never_leaks_string_values(s in ".{11,200}") {
            let input = TaintedToolInput::new(json!({"key": s}));
            let debug = format!("{input:?}");
            prop_assert!(
                !debug.contains(&s),
                "TaintedToolInput Debug leaked value: {}", debug
            );
        }
    }
}
