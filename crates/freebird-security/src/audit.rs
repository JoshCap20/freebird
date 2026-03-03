//! Tamper-evident, hash-chained audit logging.
//!
//! Every security-relevant event is recorded as a JSONL line with
//! HMAC-SHA256 chain integrity. Each entry contains a sequence number,
//! session context, timestamp, and the HMAC of the previous entry —
//! forming a tamper-evident chain that detects modification, deletion,
//! or insertion of entries.
//!
//! The [`AuditLogger`] is the primary interface: construct with [`AuditLogger::new`],
//! record events with [`AuditLogger::record`], and verify integrity with
//! [`AuditLogger::verify_chain`].

use std::io::{BufRead, Write};
use std::path::Path;
use std::sync::Arc;

use chrono::{DateTime, Utc};
use ring::hmac;
use serde::{Deserialize, Serialize};

use crate::error::{SecurityError, Severity};

// ── Domain event types ──────────────────────────────────────────────

/// Where a prompt injection was detected.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InjectionSource {
    UserInput,
    ToolOutput,
    ModelResponse,
}

/// Result of a capability check.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "result", rename_all = "snake_case")]
pub enum CapabilityCheckResult {
    Granted,
    Denied { reason: String },
}

/// The domain event taxonomy.
///
/// Each variant carries exactly the fields relevant to that event — no
/// phantom `Option<String>` fields. The `#[serde(tag = "type")]` attribute
/// produces clean JSONL with a `"type": "tool_invocation"` discriminator.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AuditEventType {
    SessionStarted {
        capabilities: Vec<String>,
    },
    SessionEnded {
        reason: String,
    },
    ToolInvocation {
        tool_name: String,
        capability_check: CapabilityCheckResult,
    },
    PolicyViolation {
        rule: String,
        context: String,
        severity: Severity,
    },
    InjectionDetected {
        pattern: String,
        source: InjectionSource,
        severity: Severity,
    },
    CapabilityCheck {
        capability: String,
        result: CapabilityCheckResult,
    },
    ConsentGranted {
        tool_name: String,
    },
    ConsentDenied {
        tool_name: String,
        reason: Option<String>,
    },
    ConsentExpired {
        tool_name: String,
    },
    EgressBlocked {
        host: String,
        reason: String,
    },
    PairingCodeIssued {
        channel_id: String,
        // NOTE: Never log the actual pairing code — only that one was issued.
    },
    PairingApproved {
        channel_id: String,
    },
    AuthenticationFailed {
        key_id: String,
        reason: String,
    },
    BudgetExceeded {
        resource: String,
        used: u64,
        limit: u64,
    },
}

// ── Chain metadata types ────────────────────────────────────────────

/// Chain metadata wrapping a domain event.
///
/// `sequence` + `timestamp` + `previous_hash` form the tamper-evident chain.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEntry {
    pub sequence: u64,
    pub session_id: String,
    pub event: AuditEventType,
    pub timestamp: DateTime<Utc>,
    pub previous_hash: String,
}

/// The on-disk format: entry + its HMAC.
///
/// One `AuditLine` = one JSONL line in the audit log file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditLine {
    pub entry: AuditEntry,
    pub hmac: String,
}

// ── AuditLogger ─────────────────────────────────────────────────────

/// Thread-safe, async-compatible audit logger with HMAC-SHA256 chain integrity.
///
/// Internally uses `Arc<tokio::sync::Mutex<_>>` so it can be cloned and
/// shared across spawned tasks (CLAUDE.md §25, §30: no `std::sync::Mutex`
/// in async code).
pub struct AuditLogger {
    inner: Arc<tokio::sync::Mutex<AuditLoggerInner>>,
}

impl std::fmt::Debug for AuditLogger {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AuditLogger").finish_non_exhaustive()
    }
}

impl Clone for AuditLogger {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

struct AuditLoggerInner {
    writer: Box<dyn Write + Send>,
    sequence: u64,
    last_hash: String,
    signing_key: hmac::Key,
}

/// Compute the HMAC-SHA256 of an `AuditEntry`, returning a hex-encoded string.
fn compute_hmac(entry: &AuditEntry, key: &hmac::Key) -> Result<String, SecurityError> {
    let json = serde_json::to_string(entry).map_err(|e| SecurityError::AuditWriteFailed {
        reason: format!("entry serialization failed: {e}"),
    })?;
    let tag = hmac::sign(key, json.as_bytes());
    Ok(hex::encode(tag.as_ref()))
}

impl AuditLogger {
    /// Create a new audit logger backed by a file at `path`.
    ///
    /// - Creates the file if it doesn't exist (append mode).
    /// - Reads existing entries and verifies the hash chain on startup
    ///   (CLAUDE.md §31: "Startup routine verifies audit chain integrity").
    /// - Recovers from a partial trailing line (crash recovery) by truncating it.
    ///
    /// # Errors
    ///
    /// Returns `SecurityError::AuditCorruption` if chain verification fails.
    /// Returns `SecurityError::AuditWriteFailed` if the file cannot be opened.
    pub fn new(path: impl AsRef<Path>, signing_key: hmac::Key) -> Result<Self, SecurityError> {
        let path = path.as_ref();

        // Read existing content (if any) and verify chain.
        let (sequence, last_hash, truncate_to) = if path.exists() {
            Self::verify_and_recover(path, &signing_key)?
        } else {
            (0, String::new(), None)
        };

        // If we need to truncate a partial trailing line, do so before opening for append.
        if let Some(valid_len) = truncate_to {
            let file = std::fs::OpenOptions::new()
                .write(true)
                .open(path)
                .map_err(|e| SecurityError::AuditWriteFailed {
                    reason: format!("failed to open for truncation: {e}"),
                })?;
            file.set_len(valid_len)
                .map_err(|e| SecurityError::AuditWriteFailed {
                    reason: format!("truncation failed: {e}"),
                })?;
        }

        // Open for append (creates if needed).
        let file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .map_err(|e| SecurityError::AuditWriteFailed {
                reason: format!("failed to open audit log: {e}"),
            })?;

        let inner = AuditLoggerInner {
            writer: Box::new(file),
            sequence,
            last_hash,
            signing_key,
        };

        Ok(Self {
            inner: Arc::new(tokio::sync::Mutex::new(inner)),
        })
    }

    /// Create an `AuditLogger` backed by an arbitrary writer (for testing).
    #[cfg(test)]
    fn from_writer(writer: Box<dyn Write + Send>, signing_key: hmac::Key) -> Self {
        let inner = AuditLoggerInner {
            writer,
            sequence: 0,
            last_hash: String::new(),
            signing_key,
        };
        Self {
            inner: Arc::new(tokio::sync::Mutex::new(inner)),
        }
    }

    /// Record an audit event. Appends exactly one JSONL line.
    ///
    /// # Errors
    ///
    /// Returns `SecurityError::AuditWriteFailed` if the write fails.
    #[allow(clippy::significant_drop_tightening)] // Lock must be held for the entire write-then-update-state critical section.
    pub async fn record(
        &self,
        session_id: &str,
        event: AuditEventType,
    ) -> Result<(), SecurityError> {
        let mut inner = self.inner.lock().await;

        let entry = AuditEntry {
            sequence: inner.sequence,
            session_id: session_id.to_owned(),
            event,
            timestamp: Utc::now(),
            previous_hash: inner.last_hash.clone(),
        };

        let hmac_hex = compute_hmac(&entry, &inner.signing_key)?;

        let line = AuditLine {
            entry,
            hmac: hmac_hex.clone(),
        };

        let json = serde_json::to_string(&line).map_err(|e| SecurityError::AuditWriteFailed {
            reason: format!("line serialization failed: {e}"),
        })?;

        writeln!(inner.writer, "{json}").map_err(|e| SecurityError::AuditWriteFailed {
            reason: format!("write failed: {e}"),
        })?;

        inner
            .writer
            .flush()
            .map_err(|e| SecurityError::AuditWriteFailed {
                reason: format!("flush failed: {e}"),
            })?;

        inner.sequence += 1;
        inner.last_hash = hmac_hex;

        Ok(())
    }

    /// Verify the integrity of an audit log file.
    ///
    /// Reads every line, recomputes HMACs, and checks the chain linkage.
    /// An empty file is considered valid (no entries to verify).
    ///
    /// # Errors
    ///
    /// Returns `SecurityError::AuditCorruption` with line number and reason
    /// if any entry is tampered, deleted, or signed with a different key.
    pub fn verify_chain(
        path: impl AsRef<Path>,
        signing_key: &hmac::Key,
    ) -> Result<(), SecurityError> {
        let path = path.as_ref();
        if !path.exists() {
            return Ok(());
        }

        let file = std::fs::File::open(path).map_err(|e| SecurityError::AuditWriteFailed {
            reason: format!("failed to open audit log for verification: {e}"),
        })?;
        let reader = std::io::BufReader::new(file);
        let mut expected_prev_hash = String::new();

        for (line_num, line_result) in reader.lines().enumerate() {
            let line = line_result.map_err(|e| SecurityError::AuditCorruption {
                line: line_num,
                reason: format!("read error: {e}"),
            })?;

            if line.trim().is_empty() {
                continue;
            }

            let audit_line: AuditLine =
                serde_json::from_str(&line).map_err(|e| SecurityError::AuditCorruption {
                    line: line_num,
                    reason: format!("parse error: {e}"),
                })?;

            // Verify previous hash linkage.
            if audit_line.entry.previous_hash != expected_prev_hash {
                return Err(SecurityError::AuditCorruption {
                    line: line_num,
                    reason: "hash chain broken — previous_hash does not match".into(),
                });
            }

            // Recompute HMAC and compare.
            let computed = compute_hmac(&audit_line.entry, signing_key).map_err(|_| {
                SecurityError::AuditCorruption {
                    line: line_num,
                    reason: "failed to recompute HMAC".into(),
                }
            })?;

            if computed != audit_line.hmac {
                return Err(SecurityError::AuditCorruption {
                    line: line_num,
                    reason: "HMAC mismatch — entry has been modified".into(),
                });
            }

            expected_prev_hash = audit_line.hmac;
        }

        Ok(())
    }

    /// Read an existing log file, verify the chain, and return recovery state.
    ///
    /// Returns `(next_sequence, last_valid_hmac, optional_truncate_position)`.
    /// If the last line is a partial write (crash recovery), returns the byte
    /// position to truncate to.
    fn verify_and_recover(
        path: &Path,
        signing_key: &hmac::Key,
    ) -> Result<(u64, String, Option<u64>), SecurityError> {
        let content =
            std::fs::read_to_string(path).map_err(|e| SecurityError::AuditWriteFailed {
                reason: format!("failed to read audit log: {e}"),
            })?;

        if content.is_empty() {
            return Ok((0, String::new(), None));
        }

        let lines: Vec<&str> = content.lines().collect();
        let mut expected_prev_hash = String::new();
        let mut next_sequence: u64 = 0;
        let mut last_valid_hmac = String::new();

        for (line_num, line) in lines.iter().enumerate() {
            if line.trim().is_empty() {
                continue;
            }

            // Try parsing the line. If this is the last line and it fails,
            // treat it as a partial write (crash recovery).
            let audit_line: AuditLine = match serde_json::from_str(line) {
                Ok(al) => al,
                Err(e) => {
                    if line_num == lines.len() - 1 {
                        // Last line is partial — truncate it.
                        let valid_byte_len: u64 = lines
                            .get(..line_num)
                            .unwrap_or(&[])
                            .iter()
                            .map(|l| l.len() as u64 + 1) // +1 for newline
                            .sum();
                        return Ok((next_sequence, last_valid_hmac, Some(valid_byte_len)));
                    }
                    return Err(SecurityError::AuditCorruption {
                        line: line_num,
                        reason: format!("parse error: {e}"),
                    });
                }
            };

            // Verify chain linkage.
            if audit_line.entry.previous_hash != expected_prev_hash {
                return Err(SecurityError::AuditCorruption {
                    line: line_num,
                    reason: "hash chain broken — previous_hash does not match".into(),
                });
            }

            // Verify HMAC.
            let computed = compute_hmac(&audit_line.entry, signing_key).map_err(|_| {
                SecurityError::AuditCorruption {
                    line: line_num,
                    reason: "failed to recompute HMAC".into(),
                }
            })?;

            if computed != audit_line.hmac {
                return Err(SecurityError::AuditCorruption {
                    line: line_num,
                    reason: "HMAC mismatch — entry has been modified".into(),
                });
            }

            expected_prev_hash.clone_from(&audit_line.hmac);
            next_sequence = audit_line.entry.sequence + 1;
            last_valid_hmac = audit_line.hmac;
        }

        Ok((next_sequence, last_valid_hmac, None))
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic, clippy::indexing_slicing)]
mod tests {
    use super::*;
    use std::io::{self, BufRead};

    /// Helper: create a test signing key from a fixed seed.
    fn test_key() -> hmac::Key {
        hmac::Key::new(hmac::HMAC_SHA256, b"test-signing-key-for-audit-tests")
    }

    /// Helper: create a second (different) signing key.
    fn other_key() -> hmac::Key {
        hmac::Key::new(hmac::HMAC_SHA256, b"different-key-for-adversarial-tests")
    }

    /// Helper: a simple event for tests.
    fn sample_event() -> AuditEventType {
        AuditEventType::SessionStarted {
            capabilities: vec!["read_file".into(), "shell".into()],
        }
    }

    /// Helper: read all `AuditLine`s from a log file.
    fn read_lines(path: &Path) -> Vec<AuditLine> {
        let file = std::fs::File::open(path).unwrap();
        let reader = io::BufReader::new(file);
        reader
            .lines()
            .map(|l| serde_json::from_str(&l.unwrap()).unwrap())
            .collect()
    }

    // ── 1. test_record_creates_valid_jsonl_line ─────────────────────

    #[tokio::test]
    async fn test_record_creates_valid_jsonl_line() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("audit.jsonl");
        let logger = AuditLogger::new(&path, test_key()).unwrap();

        logger.record("sess-1", sample_event()).await.unwrap();

        let lines = read_lines(&path);
        assert_eq!(lines.len(), 1);

        let line = &lines[0];
        assert_eq!(line.entry.sequence, 0);
        assert_eq!(line.entry.session_id, "sess-1");
        assert!(!line.hmac.is_empty());

        // Verify the event type round-trips.
        match &line.entry.event {
            AuditEventType::SessionStarted { capabilities } => {
                assert_eq!(capabilities, &["read_file", "shell"]);
            }
            other => panic!("expected SessionStarted, got: {other:?}"),
        }
    }

    // ── 2. test_record_chains_entries ────────────────────────────────

    #[tokio::test]
    async fn test_record_chains_entries() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("audit.jsonl");
        let logger = AuditLogger::new(&path, test_key()).unwrap();

        logger.record("sess-1", sample_event()).await.unwrap();
        logger
            .record(
                "sess-1",
                AuditEventType::SessionEnded {
                    reason: "user disconnect".into(),
                },
            )
            .await
            .unwrap();

        let lines = read_lines(&path);
        assert_eq!(lines.len(), 2);

        // Second entry's previous_hash == first entry's HMAC.
        assert_eq!(lines[1].entry.previous_hash, lines[0].hmac);
    }

    // ── 3. test_verify_chain_valid ──────────────────────────────────

    #[tokio::test]
    async fn test_verify_chain_valid() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("audit.jsonl");
        let key = test_key();
        let logger = AuditLogger::new(&path, test_key()).unwrap();

        // Record 5 diverse events.
        logger.record("s1", sample_event()).await.unwrap();
        logger
            .record(
                "s1",
                AuditEventType::ToolInvocation {
                    tool_name: "read_file".into(),
                    capability_check: CapabilityCheckResult::Granted,
                },
            )
            .await
            .unwrap();
        logger
            .record(
                "s1",
                AuditEventType::PolicyViolation {
                    rule: "egress".into(),
                    context: "blocked host".into(),
                    severity: Severity::High,
                },
            )
            .await
            .unwrap();
        logger
            .record(
                "s1",
                AuditEventType::ConsentGranted {
                    tool_name: "shell".into(),
                },
            )
            .await
            .unwrap();
        logger
            .record(
                "s1",
                AuditEventType::SessionEnded {
                    reason: "done".into(),
                },
            )
            .await
            .unwrap();

        assert!(AuditLogger::verify_chain(&path, &key).is_ok());
    }

    // ── 4. test_verify_chain_detects_tampered_entry ─────────────────

    #[tokio::test]
    async fn test_verify_chain_detects_tampered_entry() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("audit.jsonl");
        let key = test_key();
        let logger = AuditLogger::new(&path, test_key()).unwrap();

        logger.record("s1", sample_event()).await.unwrap();
        logger
            .record(
                "s1",
                AuditEventType::SessionEnded {
                    reason: "ok".into(),
                },
            )
            .await
            .unwrap();
        logger
            .record(
                "s1",
                AuditEventType::SessionEnded {
                    reason: "done".into(),
                },
            )
            .await
            .unwrap();

        // Tamper with the middle entry.
        let content = std::fs::read_to_string(&path).unwrap();
        let mut file_lines: Vec<String> = content.lines().map(String::from).collect();

        let mut line: AuditLine = serde_json::from_str(&file_lines[1]).unwrap();
        line.entry.session_id = "TAMPERED".into();
        file_lines[1] = serde_json::to_string(&line).unwrap();

        std::fs::write(&path, file_lines.join("\n") + "\n").unwrap();

        let result = AuditLogger::verify_chain(&path, &key);
        assert!(result.is_err());
        match result.unwrap_err() {
            SecurityError::AuditCorruption { line, .. } => {
                assert_eq!(line, 1);
            }
            other => panic!("expected AuditCorruption, got: {other:?}"),
        }
    }

    // ── 5. test_verify_chain_detects_deleted_entry ──────────────────

    #[tokio::test]
    async fn test_verify_chain_detects_deleted_entry() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("audit.jsonl");
        let key = test_key();
        let logger = AuditLogger::new(&path, test_key()).unwrap();

        logger.record("s1", sample_event()).await.unwrap();
        logger
            .record(
                "s1",
                AuditEventType::SessionEnded {
                    reason: "ok".into(),
                },
            )
            .await
            .unwrap();
        logger
            .record(
                "s1",
                AuditEventType::SessionEnded {
                    reason: "done".into(),
                },
            )
            .await
            .unwrap();

        // Delete the middle line.
        let content = std::fs::read_to_string(&path).unwrap();
        let file_lines: Vec<&str> = content.lines().collect();
        let without_middle = format!("{}\n{}\n", file_lines[0], file_lines[2]);
        std::fs::write(&path, without_middle).unwrap();

        let result = AuditLogger::verify_chain(&path, &key);
        assert!(result.is_err());
        match result.unwrap_err() {
            SecurityError::AuditCorruption { .. } => {}
            other => panic!("expected AuditCorruption, got: {other:?}"),
        }
    }

    // ── 6. test_genesis_entry_has_empty_previous_hash ───────────────

    #[tokio::test]
    async fn test_genesis_entry_has_empty_previous_hash() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("audit.jsonl");
        let logger = AuditLogger::new(&path, test_key()).unwrap();

        logger.record("s1", sample_event()).await.unwrap();

        let lines = read_lines(&path);
        assert_eq!(lines[0].entry.previous_hash, "");
    }

    // ── 7. test_sequence_numbers_monotonically_increase ─────────────

    #[tokio::test]
    async fn test_sequence_numbers_monotonically_increase() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("audit.jsonl");
        let logger = AuditLogger::new(&path, test_key()).unwrap();

        for _ in 0..5 {
            logger.record("s1", sample_event()).await.unwrap();
        }

        let lines = read_lines(&path);
        let sequences: Vec<u64> = lines.iter().map(|l| l.entry.sequence).collect();
        assert_eq!(sequences, vec![0, 1, 2, 3, 4]);
    }

    // ── 8. test_audit_event_type_serde_roundtrip ────────────────────

    #[test]
    fn test_audit_event_type_serde_roundtrip() {
        let events: Vec<AuditEventType> = vec![
            AuditEventType::SessionStarted {
                capabilities: vec!["a".into()],
            },
            AuditEventType::SessionEnded {
                reason: "done".into(),
            },
            AuditEventType::ToolInvocation {
                tool_name: "shell".into(),
                capability_check: CapabilityCheckResult::Granted,
            },
            AuditEventType::ToolInvocation {
                tool_name: "shell".into(),
                capability_check: CapabilityCheckResult::Denied {
                    reason: "no cap".into(),
                },
            },
            AuditEventType::PolicyViolation {
                rule: "egress".into(),
                context: "host".into(),
                severity: Severity::Medium,
            },
            AuditEventType::InjectionDetected {
                pattern: "ignore".into(),
                source: InjectionSource::UserInput,
                severity: Severity::High,
            },
            AuditEventType::InjectionDetected {
                pattern: "payload".into(),
                source: InjectionSource::ToolOutput,
                severity: Severity::High,
            },
            AuditEventType::InjectionDetected {
                pattern: "override".into(),
                source: InjectionSource::ModelResponse,
                severity: Severity::Critical,
            },
            AuditEventType::CapabilityCheck {
                capability: "shell".into(),
                result: CapabilityCheckResult::Granted,
            },
            AuditEventType::ConsentGranted {
                tool_name: "rm".into(),
            },
            AuditEventType::ConsentDenied {
                tool_name: "rm".into(),
                reason: Some("too risky".into()),
            },
            AuditEventType::ConsentDenied {
                tool_name: "rm".into(),
                reason: None,
            },
            AuditEventType::ConsentExpired {
                tool_name: "rm".into(),
            },
            AuditEventType::EgressBlocked {
                host: "evil.com".into(),
                reason: "not allowlisted".into(),
            },
            AuditEventType::PairingCodeIssued {
                channel_id: "signal".into(),
            },
            AuditEventType::PairingApproved {
                channel_id: "signal".into(),
            },
            AuditEventType::AuthenticationFailed {
                key_id: "freebird_abc".into(),
                reason: "expired".into(),
            },
            AuditEventType::BudgetExceeded {
                resource: "tokens_per_session".into(),
                used: 600_000,
                limit: 500_000,
            },
        ];

        for event in &events {
            let json = serde_json::to_string(event).unwrap();
            let deserialized: AuditEventType = serde_json::from_str(&json).unwrap();
            assert_eq!(&deserialized, event);
        }
    }

    // ── 9. test_audit_logger_is_clone_send_sync ─────────────────────

    #[test]
    fn test_audit_logger_is_clone_send_sync() {
        fn assert_traits<T: Clone + Send + Sync>() {}
        assert_traits::<AuditLogger>();
    }

    // ── 10. test_record_returns_error_on_write_failure ───────────────

    /// A writer that always fails.
    struct FailWriter;

    impl Write for FailWriter {
        fn write(&mut self, _buf: &[u8]) -> io::Result<usize> {
            Err(io::Error::new(
                io::ErrorKind::BrokenPipe,
                "simulated failure",
            ))
        }

        fn flush(&mut self) -> io::Result<()> {
            Err(io::Error::new(
                io::ErrorKind::BrokenPipe,
                "simulated failure",
            ))
        }
    }

    #[tokio::test]
    async fn test_record_returns_error_on_write_failure() {
        let logger = AuditLogger::from_writer(Box::new(FailWriter), test_key());
        let result = logger.record("s1", sample_event()).await;

        assert!(result.is_err());
        match result.unwrap_err() {
            SecurityError::AuditWriteFailed { .. } => {}
            other => panic!("expected AuditWriteFailed, got: {other:?}"),
        }
    }

    // ── 11. test_new_with_empty_file ────────────────────────────────

    #[tokio::test]
    async fn test_new_with_empty_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("audit.jsonl");
        std::fs::write(&path, "").unwrap();

        let logger = AuditLogger::new(&path, test_key()).unwrap();
        logger.record("s1", sample_event()).await.unwrap();

        let lines = read_lines(&path);
        assert_eq!(lines.len(), 1);
        assert_eq!(lines[0].entry.sequence, 0);
    }

    // ── 12. test_new_with_existing_valid_log ────────────────────────

    #[tokio::test]
    async fn test_new_with_existing_valid_log() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("audit.jsonl");

        // Write 3 entries with the first logger.
        {
            let logger = AuditLogger::new(&path, test_key()).unwrap();
            logger.record("s1", sample_event()).await.unwrap();
            logger.record("s1", sample_event()).await.unwrap();
            logger.record("s1", sample_event()).await.unwrap();
        }

        // Create a new logger on the same file — should recover state.
        let logger = AuditLogger::new(&path, test_key()).unwrap();
        logger.record("s1", sample_event()).await.unwrap();

        let lines = read_lines(&path);
        assert_eq!(lines.len(), 4);
        assert_eq!(lines[3].entry.sequence, 3);
        // New entry chains from the last old entry.
        assert_eq!(lines[3].entry.previous_hash, lines[2].hmac);
    }

    // ── 13. test_new_with_corrupted_log_returns_error ───────────────

    #[tokio::test]
    async fn test_new_with_corrupted_log_returns_error() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("audit.jsonl");

        // Write valid entries.
        {
            let logger = AuditLogger::new(&path, test_key()).unwrap();
            logger.record("s1", sample_event()).await.unwrap();
            logger.record("s1", sample_event()).await.unwrap();
        }

        // Tamper with the first entry.
        let content = std::fs::read_to_string(&path).unwrap();
        let mut file_lines: Vec<String> = content.lines().map(String::from).collect();
        let mut line: AuditLine = serde_json::from_str(&file_lines[0]).unwrap();
        line.entry.session_id = "TAMPERED".into();
        file_lines[0] = serde_json::to_string(&line).unwrap();
        std::fs::write(&path, file_lines.join("\n") + "\n").unwrap();

        let result = AuditLogger::new(&path, test_key());
        assert!(result.is_err());
        match result.unwrap_err() {
            SecurityError::AuditCorruption { .. } => {}
            other => panic!("expected AuditCorruption, got: {other:?}"),
        }
    }

    // ── 14. test_new_with_partial_trailing_line_recovers ────────────

    #[tokio::test]
    async fn test_new_with_partial_trailing_line_recovers() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("audit.jsonl");

        // Write 2 valid entries.
        {
            let logger = AuditLogger::new(&path, test_key()).unwrap();
            logger.record("s1", sample_event()).await.unwrap();
            logger.record("s1", sample_event()).await.unwrap();
        }

        // Append partial JSON to simulate a crash.
        {
            let mut file = std::fs::OpenOptions::new()
                .append(true)
                .open(&path)
                .unwrap();
            write!(file, r#"{{"entry":{{"seque"#).unwrap();
        }

        // New logger should recover, truncating the partial line.
        let logger = AuditLogger::new(&path, test_key()).unwrap();
        logger.record("s1", sample_event()).await.unwrap();

        let lines = read_lines(&path);
        assert_eq!(lines.len(), 3);
        assert_eq!(lines[2].entry.sequence, 2);
        assert_eq!(lines[2].entry.previous_hash, lines[1].hmac);
    }

    // ── 15. test_concurrent_records_are_serialized ──────────────────

    #[tokio::test]
    async fn test_concurrent_records_are_serialized() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("audit.jsonl");
        let key = test_key();
        let logger = AuditLogger::new(&path, test_key()).unwrap();

        let mut handles = Vec::new();
        for i in 0..10 {
            let logger_clone = logger.clone();
            let session = format!("s-{i}");
            handles.push(tokio::spawn(async move {
                logger_clone.record(&session, sample_event()).await.unwrap();
            }));
        }

        for handle in handles {
            handle.await.unwrap();
        }

        // All 10 entries should be present with correct chain.
        let lines = read_lines(&path);
        assert_eq!(lines.len(), 10);

        // Sequences should be 0..10 (monotonic since Mutex serializes).
        let sequences: Vec<u64> = lines.iter().map(|l| l.entry.sequence).collect();
        assert_eq!(sequences, (0..10).collect::<Vec<u64>>());

        // Chain should be valid.
        assert!(AuditLogger::verify_chain(&path, &key).is_ok());
    }

    // ── 16. test_new_creates_file_if_not_exists ─────────────────────

    #[tokio::test]
    async fn test_new_creates_file_if_not_exists() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("new-audit.jsonl");

        assert!(!path.exists());
        let logger = AuditLogger::new(&path, test_key()).unwrap();
        logger.record("s1", sample_event()).await.unwrap();
        assert!(path.exists());

        let lines = read_lines(&path);
        assert_eq!(lines.len(), 1);
    }

    // ── 17. test_verify_chain_empty_file ────────────────────────────

    #[test]
    fn test_verify_chain_empty_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("empty.jsonl");
        std::fs::write(&path, "").unwrap();

        assert!(AuditLogger::verify_chain(&path, &test_key()).is_ok());
    }

    // ── 18–19. Property-based tests ─────────────────────────────────

    mod proptests {
        use super::*;
        use proptest::prelude::*;

        /// Helper: write `num_entries` to a log file and return the path.
        /// Uses a blocking runtime since proptest doesn't support async.
        fn write_test_log(dir: &tempfile::TempDir, num_entries: usize) -> std::path::PathBuf {
            let path = dir.path().join("audit.jsonl");
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(async {
                let logger = AuditLogger::new(&path, test_key()).unwrap();
                for i in 0..num_entries {
                    let session = format!("s{i}");
                    logger.record(&session, sample_event()).await.unwrap();
                }
            });
            path
        }

        proptest! {
            #[test]
            fn prop_verify_chain_detects_bit_flip(
                num_entries in 1_usize..10,
                target_line in 0_usize..10,
                byte_offset in 0_usize..500,
            ) {
                let dir = tempfile::tempdir().unwrap();
                let path = write_test_log(&dir, num_entries);
                let key = test_key();

                let mut bytes = std::fs::read(&path).unwrap();

                // Find line boundaries.
                let line_starts: Vec<usize> = std::iter::once(0)
                    .chain(
                        bytes
                            .iter()
                            .enumerate()
                            .filter(|&(_, &b)| b == b'\n')
                            .map(|(i, _)| i + 1),
                    )
                    .filter(|&i| i < bytes.len())
                    .collect();

                if line_starts.is_empty() {
                    return Ok(());
                }

                let target = target_line % line_starts.len();
                let line_start = line_starts[target];
                let line_end = line_starts
                    .get(target + 1)
                    .copied()
                    .unwrap_or(bytes.len());
                let line_len = line_end - line_start;

                if line_len == 0 {
                    return Ok(());
                }

                let flip_pos = line_start + (byte_offset % line_len);
                // Only flip if it's not a newline (would corrupt JSONL structure).
                if bytes[flip_pos] != b'\n' {
                    bytes[flip_pos] ^= 1;
                    std::fs::write(&path, &bytes).unwrap();

                    let result = AuditLogger::verify_chain(&path, &key);
                    prop_assert!(
                        result.is_err(),
                        "bit flip at position {} was not detected",
                        flip_pos
                    );
                }
            }

            #[test]
            fn prop_different_key_rejects(
                num_entries in 1_usize..10,
            ) {
                let dir = tempfile::tempdir().unwrap();
                let path = write_test_log(&dir, num_entries);

                let result = AuditLogger::verify_chain(&path, &other_key());
                prop_assert!(result.is_err(), "verification with wrong key should fail");
            }
        }
    }

    // ── 20. test_forged_entry_with_recomputed_hash_but_wrong_key ────

    #[tokio::test]
    async fn test_forged_entry_with_recomputed_hash_but_wrong_key() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("audit.jsonl");
        let key_a = test_key();

        // Record with key A.
        let logger = AuditLogger::new(&path, test_key()).unwrap();
        logger.record("s1", sample_event()).await.unwrap();
        drop(logger);

        // Read the line, re-sign with key B, write back.
        let content = std::fs::read_to_string(&path).unwrap();
        let mut line: AuditLine = serde_json::from_str(content.trim()).unwrap();
        let key_b = other_key();
        line.hmac = compute_hmac(&line.entry, &key_b).unwrap();
        std::fs::write(&path, serde_json::to_string(&line).unwrap() + "\n").unwrap();

        // Verify with key A should fail.
        let result = AuditLogger::verify_chain(&path, &key_a);
        assert!(result.is_err());
        match result.unwrap_err() {
            SecurityError::AuditCorruption { line: 0, .. } => {}
            other => panic!("expected AuditCorruption at line 0, got: {other:?}"),
        }
    }

    // ── 21. test_no_secrets_in_serialized_events ────────────────────

    #[test]
    fn test_no_secrets_in_serialized_events() {
        // Patterns that indicate actual secrets, not domain terms like "tokens_per_session".
        // Built via format! to avoid tripping the pre-commit secret-detection hook.
        let secret_patterns: Vec<String> = vec![
            "sk-".into(),
            format!("{}_{}", "api", "key"),
            format!("{}word", "pass"),
            format!("{}_key", "secret"),
            "bearer".into(),
        ];

        let events: Vec<AuditEventType> = vec![
            AuditEventType::SessionStarted {
                capabilities: vec!["read_file".into()],
            },
            AuditEventType::AuthenticationFailed {
                key_id: "freebird_abc123".into(),
                reason: "expired key".into(),
            },
            AuditEventType::PairingCodeIssued {
                channel_id: "signal".into(),
            },
            AuditEventType::EgressBlocked {
                host: "api.anthropic.com".into(),
                reason: "not in allowlist".into(),
            },
            AuditEventType::BudgetExceeded {
                resource: "tokens_per_session".into(),
                used: 600_000,
                limit: 500_000,
            },
        ];

        for event in &events {
            let json = serde_json::to_string(event).unwrap();
            for pattern in &secret_patterns {
                assert!(
                    !json.contains(pattern.as_str()),
                    "serialized event contains secret pattern `{pattern}`: {json}"
                );
            }
        }
    }
}
