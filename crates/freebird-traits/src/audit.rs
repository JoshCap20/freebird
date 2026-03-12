//! Audit sink trait — abstracts over audit event persistence backends.
//!
//! Backed by `SQLite`, enabling unified encrypted storage with conversation events.
//!
//! The trait takes stringly-typed event data to avoid coupling `freebird-traits`
//! to security-specific types (`AuditEventType` lives in `freebird-security`).

use async_trait::async_trait;

use crate::memory::MemoryError;

/// Sink for persisting security audit events with HMAC chain integrity.
///
/// Each event is stored with a sequence number and HMAC chain linking it
/// to the previous event, forming a tamper-evident log.
///
/// The `event_type` and `event_json` parameters are stringly-typed so that
/// this trait can live in `freebird-traits` without depending on
/// `freebird-security`'s `AuditEventType` enum.
#[async_trait]
pub trait AuditSink: Send + Sync + 'static {
    /// Record an audit event.
    ///
    /// # Parameters
    ///
    /// - `session_id`: The session that generated this event, if any.
    ///   Security events like authentication failures may have no session.
    /// - `event_type`: The event type discriminator (e.g., `"injection_detected"`).
    /// - `event_json`: The full event payload serialized as JSON.
    async fn record(
        &self,
        session_id: Option<&str>,
        event_type: &str,
        event_json: &str,
    ) -> Result<(), MemoryError>;

    /// Verify the integrity of the audit event chain.
    ///
    /// Reads all audit events in sequence order and verifies the HMAC
    /// chain linkage. Returns `MemoryError::IntegrityViolation` if any
    /// entry has been tampered with.
    async fn verify_chain(&self) -> Result<(), MemoryError>;
}
