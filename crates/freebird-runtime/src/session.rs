//! Session lifecycle manager.
//!
//! Maps `(channel_id, sender_id)` pairs to [`SessionId`] values, enabling
//! multi-channel, multi-user support. The [`SessionManager`] is consumed by
//! the agent runtime event loop to route inbound events to the correct
//! conversation context.

use std::collections::HashMap;
use std::sync::Arc;

use freebird_security::budget::TokenBudget;
use freebird_security::capability::CapabilityGrant;
use freebird_traits::id::SessionId;
use freebird_types::id::new_session_id;
use tokio::sync::RwLock;

/// Maps `(channel_id, sender_id)` pairs to [`SessionId`] values and
/// tracks per-session [`TokenBudget`]s and [`CapabilityGrant`]s.
///
/// Thread-safe via internal [`RwLock`]. All methods take `&self`, so the
/// manager can be stored as a direct field on the agent runtime without
/// `Arc` wrapping. The `RwLock` provides correctness if the runtime evolves
/// to concurrent event handling.
pub struct SessionManager {
    sessions: RwLock<HashMap<(String, String), SessionId>>,
    /// Per-session token budgets. Wrapped in `Arc` because `TokenBudget`
    /// uses `AtomicU64` internally (interior mutability), and we need to
    /// return owned handles through the `RwLock`.
    budgets: RwLock<HashMap<SessionId, Arc<TokenBudget>>>,
    /// Per-session capability grants. Each session has a scoped set of
    /// permissions that determine which tools it can invoke.
    grants: RwLock<HashMap<SessionId, CapabilityGrant>>,
}

impl SessionManager {
    /// Create an empty session manager.
    #[must_use]
    pub fn new() -> Self {
        Self {
            sessions: RwLock::new(HashMap::new()),
            budgets: RwLock::new(HashMap::new()),
            grants: RwLock::new(HashMap::new()),
        }
    }

    /// Returns the existing [`SessionId`] for this `(channel, sender)` pair,
    /// or creates and inserts a new one.
    ///
    /// Uses read-then-write locking with double-check: the fast path
    /// (existing session) only acquires a read lock. A write lock is
    /// taken only for new sessions, with a re-check to handle races.
    #[must_use = "the returned SessionId identifies the user's conversation"]
    pub async fn resolve(&self, channel_id: &str, sender_id: &str) -> SessionId {
        // Fast path: read lock only
        let key = (channel_id.to_owned(), sender_id.to_owned());
        {
            let sessions = self.sessions.read().await;
            if let Some(id) = sessions.get(&key) {
                return id.clone();
            }
        }

        // Slow path: write lock with double-check
        let mut sessions = self.sessions.write().await;
        sessions.entry(key).or_insert_with(new_session_id).clone()
    }

    /// Creates a new [`SessionId`] for this `(channel, sender)` pair,
    /// replacing any existing session. Used for the `/new` command.
    ///
    /// Cleans up the old session's grant and budget to avoid stale entries
    /// accumulating in memory.
    #[must_use = "the returned SessionId identifies the new conversation"]
    pub async fn new_session(&self, channel_id: &str, sender_id: &str) -> SessionId {
        let key = (channel_id.to_owned(), sender_id.to_owned());
        let id = new_session_id();

        let old_id = {
            let mut sessions = self.sessions.write().await;
            sessions.insert(key, id.clone())
        };

        // Clean up old session's grant and budget if one existed.
        if let Some(old_id) = old_id {
            self.grants.write().await.remove(&old_id);
            self.budgets.write().await.remove(&old_id);
        }

        id
    }

    /// Returns the current [`SessionId`] without creating a new one.
    /// Returns `None` for unknown `(channel, sender)` pairs.
    pub async fn get(&self, channel_id: &str, sender_id: &str) -> Option<SessionId> {
        let key = (channel_id.to_owned(), sender_id.to_owned());
        let sessions = self.sessions.read().await;
        sessions.get(&key).cloned()
    }

    /// Removes the session mapping for this `(channel, sender)` pair.
    /// Returns the removed [`SessionId`], or `None` if no mapping existed.
    pub async fn remove(&self, channel_id: &str, sender_id: &str) -> Option<SessionId> {
        let key = (channel_id.to_owned(), sender_id.to_owned());
        let mut sessions = self.sessions.write().await;
        sessions.remove(&key)
    }

    /// Associate a [`TokenBudget`] with a session.
    ///
    /// Replaces any existing budget for the given session.
    pub async fn set_budget(&self, session_id: &SessionId, budget: TokenBudget) {
        let mut budgets = self.budgets.write().await;
        budgets.insert(session_id.clone(), Arc::new(budget));
    }

    /// Returns the [`TokenBudget`] for a session, if one has been set.
    ///
    /// The returned `Arc` allows callers to record usage without holding
    /// the `RwLock` — `TokenBudget` uses `AtomicU64` for interior mutability.
    pub async fn get_budget(&self, session_id: &SessionId) -> Option<Arc<TokenBudget>> {
        let budgets = self.budgets.read().await;
        budgets.get(session_id).cloned()
    }

    /// Associate a [`CapabilityGrant`] with a session.
    ///
    /// Replaces any existing grant for the given session.
    pub async fn set_grant(&self, session_id: &SessionId, grant: CapabilityGrant) {
        let mut grants = self.grants.write().await;
        grants.insert(session_id.clone(), grant);
    }

    /// Returns the [`CapabilityGrant`] for a session, if one has been set.
    pub async fn get_grant(&self, session_id: &SessionId) -> Option<CapabilityGrant> {
        let grants = self.grants.read().await;
        grants.get(session_id).cloned()
    }

    /// Returns the number of active session mappings.
    pub async fn session_count(&self) -> usize {
        let sessions = self.sessions.read().await;
        sessions.len()
    }
}

impl Default for SessionManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;

    #[tokio::test]
    async fn test_resolve_creates_new_session() {
        let mgr = SessionManager::new();
        let id = mgr.resolve("cli", "alice").await;
        assert!(!id.as_str().is_empty());
    }

    #[tokio::test]
    async fn test_resolve_returns_same_session() {
        let mgr = SessionManager::new();
        let id1 = mgr.resolve("cli", "alice").await;
        let id2 = mgr.resolve("cli", "alice").await;
        assert_eq!(id1, id2);
    }

    #[tokio::test]
    async fn test_different_senders_get_different_sessions() {
        let mgr = SessionManager::new();
        let alice = mgr.resolve("cli", "alice").await;
        let bob = mgr.resolve("cli", "bob").await;
        assert_ne!(alice, bob);
    }

    #[tokio::test]
    async fn test_different_channels_get_different_sessions() {
        let mgr = SessionManager::new();
        let cli = mgr.resolve("cli", "alice").await;
        let signal = mgr.resolve("signal", "alice").await;
        assert_ne!(cli, signal);
    }

    #[tokio::test]
    async fn test_new_session_replaces_existing() {
        let mgr = SessionManager::new();
        let old = mgr.resolve("cli", "alice").await;
        let new = mgr.new_session("cli", "alice").await;
        assert_ne!(old, new);
    }

    #[tokio::test]
    async fn test_new_session_updates_get() {
        let mgr = SessionManager::new();
        let _old = mgr.resolve("cli", "alice").await;
        let new = mgr.new_session("cli", "alice").await;
        let current = mgr.get("cli", "alice").await;
        assert_eq!(current, Some(new));
    }

    #[tokio::test]
    async fn test_get_returns_none_for_unknown() {
        let mgr = SessionManager::new();
        assert!(mgr.get("cli", "alice").await.is_none());
    }

    #[tokio::test]
    async fn test_get_returns_some_for_known() {
        let mgr = SessionManager::new();
        let expected = mgr.resolve("cli", "alice").await;
        let result = mgr.get("cli", "alice").await;
        assert_eq!(result, Some(expected));
    }

    #[tokio::test]
    async fn test_remove_returns_removed_session() {
        let mgr = SessionManager::new();
        let id = mgr.resolve("cli", "alice").await;
        let removed = mgr.remove("cli", "alice").await;
        assert_eq!(removed, Some(id));
        assert!(mgr.get("cli", "alice").await.is_none());
    }

    #[tokio::test]
    async fn test_remove_unknown_returns_none() {
        let mgr = SessionManager::new();
        assert!(mgr.remove("cli", "alice").await.is_none());
    }

    #[tokio::test]
    async fn test_session_count() {
        let mgr = SessionManager::new();
        let _ = mgr.resolve("cli", "alice").await;
        let _ = mgr.resolve("cli", "bob").await;
        let _ = mgr.resolve("signal", "alice").await;
        assert_eq!(mgr.session_count().await, 3);
    }

    #[tokio::test]
    async fn test_session_count_after_remove() {
        let mgr = SessionManager::new();
        let _ = mgr.resolve("cli", "alice").await;
        let _ = mgr.resolve("cli", "bob").await;
        let _ = mgr.remove("cli", "alice").await;
        assert_eq!(mgr.session_count().await, 1);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn test_concurrent_resolve_same_key() {
        let mgr = Arc::new(SessionManager::new());
        let mut handles = Vec::with_capacity(10);

        for _ in 0..10 {
            let mgr = Arc::clone(&mgr);
            handles.push(tokio::spawn(
                async move { mgr.resolve("cli", "alice").await },
            ));
        }

        let mut ids = Vec::with_capacity(10);
        for handle in handles {
            #[allow(clippy::expect_used)] // Test: JoinError means a task panicked — surface it
            ids.push(handle.await.expect("task should not panic"));
        }

        // All 10 tasks must return the same SessionId
        let unique: std::collections::HashSet<_> = ids.iter().map(SessionId::as_str).collect();
        assert_eq!(unique.len(), 1);
        assert_eq!(mgr.session_count().await, 1);
    }

    #[tokio::test]
    async fn test_default_is_empty() {
        let mgr = SessionManager::default();
        assert_eq!(mgr.session_count().await, 0);
    }
}
