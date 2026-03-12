//! Shared test utilities for tool test modules.
//!
//! Provides a [`TestHarness`] that owns a temp directory, session ID, and
//! capabilities, giving tool tests a zero-boilerplate `context()` method.

use std::path::{Path, PathBuf};

use freebird_traits::id::SessionId;
use freebird_traits::tool::{Capability, ToolContext};

/// Test harness that owns the temp directory, session ID, and capabilities,
/// providing a zero-boilerplate `context()` method for tool tests.
pub struct TestHarness {
    _tmp: tempfile::TempDir,
    sandbox: PathBuf,
    session_id: SessionId,
    capabilities: Vec<Capability>,
    allowed_directories: Vec<PathBuf>,
}

impl TestHarness {
    /// Create a new test harness with the given capabilities.
    ///
    /// The sandbox path is canonicalized for consistency with `SafeFilePath`
    /// resolution.
    ///
    /// # Panics
    ///
    /// Panics if the temp directory cannot be created or canonicalized.
    #[must_use]
    #[allow(clippy::expect_used)] // Test helper: panic on infra failures
    pub fn with_capabilities(capabilities: Vec<Capability>) -> Self {
        let tmp = tempfile::tempdir().expect("failed to create temp dir");
        let sandbox = tmp
            .path()
            .canonicalize()
            .expect("failed to canonicalize temp dir");
        Self {
            _tmp: tmp,
            sandbox,
            session_id: SessionId::from_string("test-session"),
            capabilities,
            allowed_directories: vec![],
        }
    }

    /// Builder method: set additional allowed directories.
    #[must_use]
    pub fn with_allowed_directories(mut self, dirs: Vec<PathBuf>) -> Self {
        self.allowed_directories = dirs;
        self
    }

    /// Returns the sandbox root path (canonicalized).
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.sandbox
    }

    /// Build a `ToolContext` referencing this harness's sandbox, session,
    /// and capabilities. Borrows from `self` — the context cannot outlive
    /// the harness.
    #[must_use]
    pub fn context(&self) -> ToolContext<'_> {
        ToolContext {
            session_id: &self.session_id,
            sandbox_root: &self.sandbox,
            granted_capabilities: &self.capabilities,
            allowed_directories: &self.allowed_directories,
            knowledge_store: None,
        }
    }
}
