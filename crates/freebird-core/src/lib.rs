//! Freebird core — composition root and builder for wiring subsystems.
//!
//! This crate provides [`FreebirdBuilder`] to construct a fully-wired
//! [`FreebirdApp`] from an [`AppConfig`](freebird_types::config::AppConfig).
//! The daemon binary becomes a thin shell that loads config, calls the builder,
//! and manages process-level concerns (signals, logging).

#![deny(clippy::all, clippy::pedantic, clippy::nursery)]
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
#![deny(clippy::indexing_slicing)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::missing_errors_doc, clippy::must_use_candidate)]

use std::sync::Arc;

use tokio_util::sync::CancellationToken;

pub mod builder;
pub mod config;
pub mod database;
pub mod error;
pub mod knowledge;
pub mod providers;
pub mod tools;
pub mod util;

pub use builder::{ChannelStrategy, FreebirdBuilder};
pub use database::PassphraseStrategy;
pub use error::CoreError;

/// A fully-wired `FreeBird` instance, ready to run.
pub struct FreebirdApp {
    pub(crate) runtime: freebird_runtime::agent::AgentRuntime,
    pub(crate) audit_sink: Option<Arc<dyn freebird_traits::audit::AuditSink>>,
    pub(crate) memory: Arc<dyn freebird_traits::memory::Memory>,
    pub(crate) revocation_list: Arc<freebird_security::capability::RevocationList>,
}

impl FreebirdApp {
    /// Run the agent until shutdown signal or error.
    ///
    /// Consumes `self` because the runtime takes ownership of the channel
    /// and tool executor.
    pub async fn run(self, token: CancellationToken) -> Result<(), CoreError> {
        Arc::new(self.runtime)
            .run(token)
            .await
            .map_err(|e| CoreError::Runtime(e.into()))
    }

    /// Access the audit sink for pre-run/post-run audit events.
    pub fn audit_sink(&self) -> Option<&Arc<dyn freebird_traits::audit::AuditSink>> {
        self.audit_sink.as_ref()
    }

    /// Access the memory backend (e.g. for replay).
    pub fn memory(&self) -> &Arc<dyn freebird_traits::memory::Memory> {
        &self.memory
    }

    /// Access the revocation list for immediate session capability revocation.
    pub const fn revocation_list(&self) -> &Arc<freebird_security::capability::RevocationList> {
        &self.revocation_list
    }
}
