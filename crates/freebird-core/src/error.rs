//! Core error types for `FreebirdBuilder` and `FreebirdApp`.

use freebird_traits::memory::MemoryError;

/// Errors from building or running a `FreebirdApp`.
#[derive(Debug, thiserror::Error)]
pub enum CoreError {
    /// Configuration validation failed.
    #[error("configuration error: {0}")]
    Config(String),

    /// Database initialization or key derivation failed.
    #[error("database initialization failed: {0}")]
    Database(#[source] anyhow::Error),

    /// Provider construction or credential validation failed.
    #[error("provider initialization failed: {0}")]
    Provider(#[source] anyhow::Error),

    /// Tool registry construction failed (e.g. duplicate tool names).
    #[error("tool registry error: {0}")]
    ToolRegistry(#[source] anyhow::Error),

    /// Knowledge store bootstrap failed.
    #[error("knowledge bootstrap failed: {0}")]
    Knowledge(#[source] anyhow::Error),

    /// Audit chain integrity verification failed on startup.
    #[error("audit chain integrity violation: {0}")]
    AuditIntegrity(#[source] MemoryError),

    /// Agent runtime error during execution.
    #[error("runtime error: {0}")]
    Runtime(#[source] anyhow::Error),

    /// Channel construction failed.
    #[error("channel construction failed: {0}")]
    Channel(String),

    /// Security subsystem construction failed.
    #[error("security initialization failed: {0}")]
    Security(#[source] anyhow::Error),
}
