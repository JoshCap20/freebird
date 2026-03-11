//! Adversarial tests for hash-chained audit log — tamper detection,
//! entry manipulation, edge cases.

#![allow(
    clippy::all,
    clippy::pedantic,
    clippy::nursery,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

use std::io::Write;

use freebird_security::audit::{AuditEventType, AuditLogger};
use ring::hmac;

fn test_signing_key() -> hmac::Key {
    hmac::Key::new(hmac::HMAC_SHA256, b"test-audit-key-for-adversarial-tests")
}

fn test_event() -> AuditEventType {
    AuditEventType::SessionStarted {
        capabilities: vec!["read_file".into()],
    }
}

// ---------------------------------------------------------------------------
// Tamper detection
// ---------------------------------------------------------------------------

#[tokio::test]
async fn modified_entry_detected() {
    let dir = tempfile::tempdir().unwrap();
    let log_path = dir.path().join("audit.jsonl");

    {
        let logger = AuditLogger::new(&log_path, test_signing_key()).unwrap();
        logger.record("sess-001", test_event()).await.unwrap();
        logger.record("sess-002", test_event()).await.unwrap();
        logger.record("sess-003", test_event()).await.unwrap();
    }

    // Tamper with the second line
    let content = std::fs::read_to_string(&log_path).unwrap();
    let tampered = content.replace("sess-002", "sess-hacked");
    std::fs::write(&log_path, tampered).unwrap();

    let result = AuditLogger::verify_chain(&log_path, &test_signing_key());
    assert!(
        result.is_err(),
        "modified entry should break the hash chain"
    );
}

#[tokio::test]
async fn reordered_entries_detected() {
    let dir = tempfile::tempdir().unwrap();
    let log_path = dir.path().join("audit.jsonl");

    {
        let logger = AuditLogger::new(&log_path, test_signing_key()).unwrap();
        logger.record("first", test_event()).await.unwrap();
        logger.record("second", test_event()).await.unwrap();
        logger.record("third", test_event()).await.unwrap();
    }

    let content = std::fs::read_to_string(&log_path).unwrap();
    let mut lines: Vec<&str> = content.lines().collect();
    assert!(lines.len() >= 3);
    lines.swap(1, 2);
    let reordered = lines.join("\n") + "\n";
    std::fs::write(&log_path, reordered).unwrap();

    let result = AuditLogger::verify_chain(&log_path, &test_signing_key());
    assert!(
        result.is_err(),
        "reordered entries should break the hash chain"
    );
}

#[tokio::test]
async fn deleted_entry_detected() {
    let dir = tempfile::tempdir().unwrap();
    let log_path = dir.path().join("audit.jsonl");

    {
        let logger = AuditLogger::new(&log_path, test_signing_key()).unwrap();
        logger.record("keep-1", test_event()).await.unwrap();
        logger.record("delete-me", test_event()).await.unwrap();
        logger.record("keep-2", test_event()).await.unwrap();
    }

    let content = std::fs::read_to_string(&log_path).unwrap();
    let lines: Vec<&str> = content.lines().collect();
    let without_middle = format!("{}\n{}\n", lines[0], lines[2]);
    std::fs::write(&log_path, without_middle).unwrap();

    let result = AuditLogger::verify_chain(&log_path, &test_signing_key());
    assert!(result.is_err(), "deleted entry should break the hash chain");
}

// ---------------------------------------------------------------------------
// Edge cases
// ---------------------------------------------------------------------------

#[test]
fn empty_log_verifies_ok() {
    let dir = tempfile::tempdir().unwrap();
    let log_path = dir.path().join("audit.jsonl");

    std::fs::write(&log_path, "").unwrap();

    let result = AuditLogger::verify_chain(&log_path, &test_signing_key());
    assert!(result.is_ok(), "empty log should verify successfully");
}

#[tokio::test]
async fn single_entry_log_verifies_ok() {
    let dir = tempfile::tempdir().unwrap();
    let log_path = dir.path().join("audit.jsonl");

    {
        let logger = AuditLogger::new(&log_path, test_signing_key()).unwrap();
        logger.record("only-one", test_event()).await.unwrap();
    }

    let result = AuditLogger::verify_chain(&log_path, &test_signing_key());
    assert!(result.is_ok(), "single entry should verify successfully");
}

#[tokio::test]
async fn truncated_last_entry_handled() {
    let dir = tempfile::tempdir().unwrap();
    let log_path = dir.path().join("audit.jsonl");

    {
        let logger = AuditLogger::new(&log_path, test_signing_key()).unwrap();
        logger.record("entry-1", test_event()).await.unwrap();
        logger.record("entry-2", test_event()).await.unwrap();
    }

    // Truncate the last entry (simulate crash mid-write)
    let content = std::fs::read_to_string(&log_path).unwrap();
    let truncated = &content[..content.len().saturating_sub(10)];
    std::fs::write(&log_path, truncated).unwrap();

    // Truncated entry should fail verification (malformed JSON or broken hash chain)
    let result = AuditLogger::verify_chain(&log_path, &test_signing_key());
    assert!(
        result.is_err(),
        "truncated entry should fail verification, not silently pass"
    );
}

#[tokio::test]
async fn wrong_signing_key_fails_verification() {
    let dir = tempfile::tempdir().unwrap();
    let log_path = dir.path().join("audit.jsonl");

    {
        let logger = AuditLogger::new(&log_path, test_signing_key()).unwrap();
        logger.record("signed-entry", test_event()).await.unwrap();
    }

    let wrong_key = hmac::Key::new(hmac::HMAC_SHA256, b"wrong-key");
    let result = AuditLogger::verify_chain(&log_path, &wrong_key);
    assert!(result.is_err(), "verification with wrong key should fail");
}

#[tokio::test]
async fn appended_garbage_detected() {
    let dir = tempfile::tempdir().unwrap();
    let log_path = dir.path().join("audit.jsonl");

    {
        let logger = AuditLogger::new(&log_path, test_signing_key()).unwrap();
        logger.record("legit", test_event()).await.unwrap();
    }

    let mut file = std::fs::OpenOptions::new()
        .append(true)
        .open(&log_path)
        .unwrap();
    writeln!(file, "{{\"garbage\": true}}").unwrap();

    let result = AuditLogger::verify_chain(&log_path, &test_signing_key());
    assert!(
        result.is_err(),
        "appended non-audit-entry should fail verification"
    );
}

#[test]
fn nonexistent_log_verifies_ok() {
    let dir = tempfile::tempdir().unwrap();
    let log_path = dir.path().join("does_not_exist.jsonl");

    let result = AuditLogger::verify_chain(&log_path, &test_signing_key());
    assert!(
        result.is_ok(),
        "nonexistent log should verify ok (nothing to verify)"
    );
}
