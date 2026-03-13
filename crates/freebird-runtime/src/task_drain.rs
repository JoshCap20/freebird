//! `JoinSet` drain utility for graceful shutdown.
//!
//! Reaps completed tasks from a `JoinSet`, aborting any that exceed a
//! configurable timeout. Used by the agent runtime's shutdown sequence.

/// Drain in-flight tasks from a `JoinSet`, aborting any that
/// exceed the timeout.
///
/// Waits up to `timeout_secs` for each remaining task to complete. If the
/// deadline expires before all tasks finish, the remaining tasks are aborted
/// via [`tokio::task::JoinSet::shutdown`].
pub async fn drain_tasks(tasks: &mut tokio::task::JoinSet<()>, timeout_secs: u64) {
    let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(timeout_secs);
    tracing::info!(
        remaining = tasks.len(),
        timeout_secs,
        "draining in-flight tasks"
    );
    loop {
        match tokio::time::timeout_at(deadline, tasks.join_next()).await {
            Ok(Some(Ok(()))) => {}
            Ok(Some(Err(e))) => {
                tracing::error!(error = %e, "task panicked during drain");
            }
            Ok(None) => break,
            Err(_) => {
                tracing::warn!(
                    remaining = tasks.len(),
                    "drain timeout expired, aborting remaining tasks"
                );
                tasks.shutdown().await;
                break;
            }
        }
    }
}
