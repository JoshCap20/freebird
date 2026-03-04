//! Graceful shutdown coordinator.
//!
//! Provides [`ShutdownCoordinator`], the single coordination point for daemon
//! shutdown. It listens for OS signals (SIGINT/SIGTERM) or programmatic
//! cancellation, distributes a [`CancellationToken`] to subsystems, and waits
//! for a configurable drain timeout before the process exits.

use std::fmt;
use std::time::Duration;

use tokio_util::sync::CancellationToken;

/// Describes why shutdown was initiated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShutdownSignal {
    /// SIGINT received (Ctrl-C).
    Interrupt,
    /// SIGTERM received (systemd stop, `kill`).
    Terminate,
    /// Shutdown triggered programmatically via [`ShutdownCoordinator::trigger`]
    /// or a pre-cancelled token.
    Programmatic,
}

impl fmt::Display for ShutdownSignal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Interrupt => write!(f, "SIGINT"),
            Self::Terminate => write!(f, "SIGTERM"),
            Self::Programmatic => write!(f, "programmatic"),
        }
    }
}

/// Errors that can occur during shutdown coordination.
#[derive(Debug, thiserror::Error)]
pub enum ShutdownError {
    /// Failed to install an OS signal handler.
    #[error("failed to install signal handler: {0}")]
    SignalHandler(std::io::Error),
}

/// Coordinates graceful shutdown across all daemon subsystems.
///
/// Owns a [`CancellationToken`] and a drain timeout. Subsystems clone the
/// token via [`token()`](Self::token) and monitor it for cancellation.
/// The daemon awaits [`wait_for_signal()`](Self::wait_for_signal) and then
/// calls [`drain()`](Self::drain) to give in-flight work time to complete.
///
/// Not `Clone` — there is exactly one authoritative coordinator per daemon.
pub struct ShutdownCoordinator {
    token: CancellationToken,
    drain_timeout: Duration,
}

impl ShutdownCoordinator {
    /// Create a new coordinator with the given drain timeout.
    ///
    /// The drain timeout is how long [`drain()`](Self::drain) waits after
    /// cancellation to give in-flight work time to complete.
    #[must_use]
    pub fn new(drain_timeout: Duration) -> Self {
        Self {
            token: CancellationToken::new(),
            drain_timeout,
        }
    }

    /// Returns a clone of the cancellation token for subsystems to monitor.
    ///
    /// Subsystems should call `token.cancelled().await` or check
    /// `token.is_cancelled()` to detect shutdown. Child tokens can be
    /// created via `token.child_token()` for hierarchical cancellation.
    #[must_use]
    pub fn token(&self) -> CancellationToken {
        self.token.clone()
    }

    /// Returns the configured drain timeout.
    #[must_use]
    pub const fn drain_timeout(&self) -> Duration {
        self.drain_timeout
    }

    /// Cancel the token programmatically (e.g., on `/quit` command or in tests).
    ///
    /// Idempotent — calling multiple times is safe.
    pub fn trigger(&self) {
        tracing::info!("shutdown triggered programmatically");
        self.token.cancel();
    }

    /// Wait for a shutdown signal (SIGINT, SIGTERM, or programmatic cancellation).
    ///
    /// Installs signal handlers and blocks until one fires or the token is
    /// cancelled externally. Cancels the token before returning (idempotent).
    ///
    /// # Errors
    ///
    /// Returns [`ShutdownError::SignalHandler`] if OS signal handlers cannot
    /// be installed.
    pub async fn wait_for_signal(&self) -> Result<ShutdownSignal, ShutdownError> {
        #[cfg(unix)]
        {
            use tokio::signal::unix::{SignalKind, signal};

            let mut sigint =
                signal(SignalKind::interrupt()).map_err(ShutdownError::SignalHandler)?;
            let mut sigterm =
                signal(SignalKind::terminate()).map_err(ShutdownError::SignalHandler)?;

            let received = tokio::select! {
                _ = sigint.recv() => ShutdownSignal::Interrupt,
                _ = sigterm.recv() => ShutdownSignal::Terminate,
                () = self.token.cancelled() => ShutdownSignal::Programmatic,
            };

            tracing::info!(%received, "received shutdown signal");
            self.token.cancel();
            Ok(received)
        }

        #[cfg(not(unix))]
        {
            tokio::select! {
                result = tokio::signal::ctrl_c() => {
                    result.map_err(ShutdownError::SignalHandler)?;
                    tracing::info!(signal = %ShutdownSignal::Interrupt, "received shutdown signal");
                    self.token.cancel();
                    Ok(ShutdownSignal::Interrupt)
                }
                () = self.token.cancelled() => {
                    tracing::info!(signal = %ShutdownSignal::Programmatic, "received shutdown signal");
                    self.token.cancel(); // idempotent
                    Ok(ShutdownSignal::Programmatic)
                }
            }
        }
    }

    /// Wait for the drain timeout to elapse.
    ///
    /// Call this after cancellation to give in-flight work time to complete.
    /// This is a simple time-based wait — it does NOT track whether tasks
    /// have actually finished. Subsystems are responsible for checking the
    /// cancellation token and winding down within this window.
    pub async fn drain(&self) {
        tracing::info!(
            timeout_ms = self.drain_timeout.as_millis(),
            "draining in-flight work"
        );
        tokio::time::sleep(self.drain_timeout).await;
        tracing::info!("drain complete");
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_token_not_cancelled_initially() {
        let coordinator = ShutdownCoordinator::new(Duration::from_secs(30));
        assert!(!coordinator.token().is_cancelled());
    }

    #[tokio::test]
    async fn test_trigger_cancels_token() {
        let coordinator = ShutdownCoordinator::new(Duration::from_secs(30));
        coordinator.trigger();
        assert!(coordinator.token().is_cancelled());
    }

    #[tokio::test]
    async fn test_token_clone_propagates_cancellation() {
        let coordinator = ShutdownCoordinator::new(Duration::from_secs(30));
        let cloned = coordinator.token();
        assert!(!cloned.is_cancelled());

        coordinator.trigger();
        assert!(cloned.is_cancelled());
    }

    #[tokio::test]
    async fn test_child_token_propagates_cancellation() {
        let coordinator = ShutdownCoordinator::new(Duration::from_secs(30));
        let child = coordinator.token().child_token();
        assert!(!child.is_cancelled());

        coordinator.trigger();
        assert!(child.is_cancelled());
    }

    #[tokio::test]
    async fn test_drain_timeout_returns_configured_value() {
        let coordinator = ShutdownCoordinator::new(Duration::from_secs(42));
        assert_eq!(coordinator.drain_timeout(), Duration::from_secs(42));
    }

    #[tokio::test]
    async fn test_drain_completes_after_timeout() {
        let coordinator = ShutdownCoordinator::new(Duration::from_millis(50));
        let start = tokio::time::Instant::now();

        coordinator.drain().await;

        let elapsed = start.elapsed();
        assert!(
            elapsed >= Duration::from_millis(10),
            "drain completed too fast: {elapsed:?}"
        );
        assert!(
            elapsed < Duration::from_millis(200),
            "drain took too long: {elapsed:?}"
        );
    }

    #[tokio::test]
    async fn test_drain_zero_timeout() {
        let coordinator = ShutdownCoordinator::new(Duration::ZERO);
        // Must not hang — completes immediately.
        coordinator.drain().await;
    }

    #[tokio::test]
    async fn test_trigger_idempotent() {
        let coordinator = ShutdownCoordinator::new(Duration::from_secs(30));
        coordinator.trigger();
        coordinator.trigger(); // must not panic
        assert!(coordinator.token().is_cancelled());
    }

    #[tokio::test]
    async fn test_wait_for_signal_returns_on_precancelled_token() {
        let coordinator = ShutdownCoordinator::new(Duration::from_secs(30));
        coordinator.trigger(); // pre-cancel

        let signal = coordinator
            .wait_for_signal()
            .await
            .expect("wait_for_signal should not fail");

        assert_eq!(signal, ShutdownSignal::Programmatic);
    }

    #[tokio::test]
    async fn test_wait_for_signal_returns_on_concurrent_trigger() {
        let coordinator = ShutdownCoordinator::new(Duration::from_secs(30));
        let token = coordinator.token();

        // Spawn a task that triggers after a short delay.
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(20)).await;
            token.cancel();
        });

        let signal = coordinator
            .wait_for_signal()
            .await
            .expect("wait_for_signal should not fail");

        assert_eq!(signal, ShutdownSignal::Programmatic);
    }

    #[test]
    fn test_shutdown_signal_display() {
        assert_eq!(ShutdownSignal::Interrupt.to_string(), "SIGINT");
        assert_eq!(ShutdownSignal::Terminate.to_string(), "SIGTERM");
        assert_eq!(ShutdownSignal::Programmatic.to_string(), "programmatic");
    }
}
