//! Session lifecycle manager.
//!
//! Maps `(channel_id, sender_id)` pairs to [`SessionId`] values, enabling
//! multi-channel, multi-user support. The [`SessionManager`] is consumed by
//! the agent runtime event loop to route inbound events to the correct
//! conversation context.
//!
//! Supports TTL-based expiration and LRU eviction to bound memory usage.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use freebird_security::budget::TokenBudget;
use freebird_security::capability::CapabilityGrant;
use freebird_traits::id::SessionId;
use freebird_types::config::SessionConfig;
use freebird_types::id::new_session_id;
use tokio::sync::RwLock;

/// Per-session metadata tracked alongside the session mapping.
struct SessionEntry {
    id: SessionId,
    last_accessed: Instant,
}

/// Maps `(channel_id, sender_id)` pairs to [`SessionId`] values and
/// tracks per-session [`TokenBudget`]s and [`CapabilityGrant`]s.
///
/// Thread-safe via internal [`RwLock`]. All methods take `&self`, so the
/// manager can be stored as a direct field on the agent runtime without
/// `Arc` wrapping. The `RwLock` provides correctness if the runtime evolves
/// to concurrent event handling.
///
/// Session eviction: expired sessions (older than `session_ttl_secs`) and
/// excess sessions beyond `max_sessions` are cleaned up during `resolve()`
/// and `new_session()` operations.
pub struct SessionManager {
    sessions: RwLock<HashMap<(String, String), SessionEntry>>,
    /// Per-session token budgets. Wrapped in `Arc` because `TokenBudget`
    /// uses `AtomicU64` internally (interior mutability), and we need to
    /// return owned handles through the `RwLock`.
    budgets: RwLock<HashMap<SessionId, Arc<TokenBudget>>>,
    /// Per-session capability grants. Each session has a scoped set of
    /// permissions that determine which tools it can invoke.
    grants: RwLock<HashMap<SessionId, CapabilityGrant>>,
    /// Configuration for session limits.
    config: SessionConfig,
}

impl SessionManager {
    /// Create an empty session manager with default configuration.
    #[must_use]
    pub fn new() -> Self {
        Self {
            sessions: RwLock::new(HashMap::new()),
            budgets: RwLock::new(HashMap::new()),
            grants: RwLock::new(HashMap::new()),
            config: SessionConfig::default(),
        }
    }

    /// Create a session manager with the given configuration.
    #[must_use]
    pub fn with_config(config: SessionConfig) -> Self {
        Self {
            sessions: RwLock::new(HashMap::new()),
            budgets: RwLock::new(HashMap::new()),
            grants: RwLock::new(HashMap::new()),
            config,
        }
    }

    /// Returns the existing [`SessionId`] for this `(channel, sender)` pair,
    /// or creates and inserts a new one.
    ///
    /// Uses read-then-write locking with double-check: the fast path
    /// (existing session) only acquires a read lock. A write lock is
    /// taken only for new sessions, with a re-check to handle races.
    ///
    /// Triggers eviction of expired and excess sessions.
    #[must_use = "the returned SessionId identifies the user's conversation"]
    pub async fn resolve(&self, channel_id: &str, sender_id: &str) -> SessionId {
        // Fast path: read lock only
        let key = (channel_id.to_owned(), sender_id.to_owned());
        {
            let mut sessions = self.sessions.write().await;
            if let Some(entry) = sessions.get_mut(&key) {
                entry.last_accessed = Instant::now();
                return entry.id.clone();
            }
        }

        // Slow path: write lock with insert
        let mut sessions = self.sessions.write().await;
        // Double-check after acquiring write lock
        if let Some(entry) = sessions.get_mut(&key) {
            entry.last_accessed = Instant::now();
            return entry.id.clone();
        }
        let entry = SessionEntry {
            id: new_session_id(),
            last_accessed: Instant::now(),
        };
        let id = entry.id.clone();
        sessions.insert(key, entry);
        // Drop the write lock before running eviction (which also takes write locks)
        drop(sessions);

        // Run eviction after inserting
        self.evict_expired().await;
        self.evict_lru().await;

        id
    }

    /// Creates a new [`SessionId`] for this `(channel, sender)` pair,
    /// replacing any existing session. Used for the `/new` command.
    ///
    /// Cleans up the old session's grant and budget to avoid stale entries
    /// accumulating in memory.
    #[must_use = "the returned SessionId identifies the new conversation"]
    pub async fn new_session(&self, channel_id: &str, sender_id: &str) -> SessionId {
        let key = (channel_id.to_owned(), sender_id.to_owned());
        let entry = SessionEntry {
            id: new_session_id(),
            last_accessed: Instant::now(),
        };
        let id = entry.id.clone();

        let old_id = {
            let mut sessions = self.sessions.write().await;
            sessions.insert(key, entry).map(|e| e.id)
        };

        // Clean up old session's grant and budget if one existed.
        if let Some(old_id) = old_id {
            self.grants.write().await.remove(&old_id);
            self.budgets.write().await.remove(&old_id);
        }

        // Run eviction after inserting
        self.evict_expired().await;
        self.evict_lru().await;

        id
    }

    /// Returns the current [`SessionId`] without creating a new one.
    /// Returns `None` for unknown `(channel, sender)` pairs.
    pub async fn get(&self, channel_id: &str, sender_id: &str) -> Option<SessionId> {
        let key = (channel_id.to_owned(), sender_id.to_owned());
        let mut sessions = self.sessions.write().await;
        if let Some(entry) = sessions.get_mut(&key) {
            entry.last_accessed = Instant::now();
            Some(entry.id.clone())
        } else {
            None
        }
    }

    /// Removes the session mapping for this `(channel, sender)` pair.
    /// Returns the removed [`SessionId`], or `None` if no mapping existed.
    ///
    /// Also cleans up any associated grant and budget to prevent stale
    /// entries from accumulating in memory.
    pub async fn remove(&self, channel_id: &str, sender_id: &str) -> Option<SessionId> {
        let key = (channel_id.to_owned(), sender_id.to_owned());
        let removed_id = {
            let mut sessions = self.sessions.write().await;
            sessions.remove(&key).map(|e| e.id)
        };

        if let Some(ref id) = removed_id {
            self.grants.write().await.remove(id);
            self.budgets.write().await.remove(id);
        }

        removed_id
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

    /// Remove sessions that have exceeded the configured TTL.
    async fn evict_expired(&self) {
        let ttl = std::time::Duration::from_secs(self.config.session_ttl_secs);
        let now = Instant::now();

        let expired_ids = {
            let mut sessions = self.sessions.write().await;
            let mut expired_keys = Vec::new();
            for (k, entry) in sessions.iter() {
                if now.duration_since(entry.last_accessed) > ttl {
                    expired_keys.push(k.clone());
                }
            }
            let mut ids = Vec::with_capacity(expired_keys.len());
            for k in expired_keys {
                if let Some(entry) = sessions.remove(&k) {
                    ids.push(entry.id);
                }
            }
            ids
        };

        if expired_ids.is_empty() {
            return;
        }
        tracing::info!(count = expired_ids.len(), "evicted expired sessions");
        self.cleanup_evicted(&expired_ids).await;
    }

    /// Remove least-recently-used sessions if the count exceeds `max_sessions`.
    async fn evict_lru(&self) {
        let max = self.config.max_sessions;

        let evicted_ids = {
            let mut sessions = self.sessions.write().await;
            if sessions.len() <= max {
                return;
            }
            let excess = sessions.len() - max;

            // Sort entries by last_accessed (oldest first)
            let mut entries: Vec<_> = sessions
                .iter()
                .map(|(k, e)| (k.clone(), e.last_accessed))
                .collect();
            entries.sort_by_key(|(_, ts)| *ts);

            let mut ids = Vec::with_capacity(excess);
            for (k, _) in entries.into_iter().take(excess) {
                if let Some(entry) = sessions.remove(&k) {
                    ids.push(entry.id);
                }
            }
            ids
        };

        if evicted_ids.is_empty() {
            return;
        }
        tracing::info!(count = evicted_ids.len(), "evicted LRU sessions");
        self.cleanup_evicted(&evicted_ids).await;
    }

    /// Remove grants and budgets for evicted session IDs.
    async fn cleanup_evicted(&self, ids: &[SessionId]) {
        {
            let mut grants = self.grants.write().await;
            for id in ids {
                grants.remove(id);
            }
        }
        {
            let mut budgets = self.budgets.write().await;
            for id in ids {
                budgets.remove(id);
            }
        }
    }
}

impl Default for SessionManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
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

    // ── Budget and grant management ─────────────────────────────

    #[tokio::test]
    async fn test_set_and_get_budget() {
        let mgr = SessionManager::new();
        let id = mgr.resolve("cli", "alice").await;

        let config = freebird_types::config::BudgetConfig {
            max_tokens_per_session: 1000,
            max_tokens_per_request: 200,
            max_tool_rounds_per_turn: 3,
            max_cost_microdollars: 5_000_000,
        };
        mgr.set_budget(&id, freebird_security::budget::TokenBudget::new(&config))
            .await;

        let budget = mgr.get_budget(&id).await;
        assert!(budget.is_some());
        assert_eq!(budget.unwrap().remaining_tokens(), 1000);
    }

    #[tokio::test]
    async fn test_get_budget_returns_none_for_unknown() {
        let mgr = SessionManager::new();
        let id = freebird_types::id::new_session_id();
        assert!(mgr.get_budget(&id).await.is_none());
    }

    #[tokio::test]
    async fn test_set_and_get_grant() {
        let mgr = SessionManager::new();
        let id = mgr.resolve("cli", "alice").await;

        let grant = freebird_security::capability::CapabilityGrant::new(
            std::iter::once(freebird_traits::tool::Capability::FileRead).collect(),
            std::env::current_dir().unwrap(),
            None,
        )
        .unwrap();
        mgr.set_grant(&id, grant).await;

        let retrieved = mgr.get_grant(&id).await;
        assert!(retrieved.is_some());
    }

    #[tokio::test]
    async fn test_get_grant_returns_none_for_unknown() {
        let mgr = SessionManager::new();
        let id = freebird_types::id::new_session_id();
        assert!(mgr.get_grant(&id).await.is_none());
    }

    #[tokio::test]
    async fn test_new_session_cleans_up_old_budget_and_grant() {
        let mgr = SessionManager::new();
        let old_id = mgr.resolve("cli", "alice").await;

        let config = freebird_types::config::BudgetConfig::default();
        mgr.set_budget(
            &old_id,
            freebird_security::budget::TokenBudget::new(&config),
        )
        .await;
        let grant = freebird_security::capability::CapabilityGrant::new(
            std::iter::once(freebird_traits::tool::Capability::FileRead).collect(),
            std::env::current_dir().unwrap(),
            None,
        )
        .unwrap();
        mgr.set_grant(&old_id, grant).await;

        // Create new session — old budget and grant should be cleaned up.
        let _new_id = mgr.new_session("cli", "alice").await;
        assert!(mgr.get_budget(&old_id).await.is_none());
        assert!(mgr.get_grant(&old_id).await.is_none());
    }

    #[tokio::test]
    async fn test_remove_cleans_up_budget_and_grant() {
        let mgr = SessionManager::new();
        let id = mgr.resolve("cli", "alice").await;

        let config = freebird_types::config::BudgetConfig::default();
        mgr.set_budget(&id, freebird_security::budget::TokenBudget::new(&config))
            .await;
        let grant = freebird_security::capability::CapabilityGrant::new(
            std::iter::once(freebird_traits::tool::Capability::FileRead).collect(),
            std::env::current_dir().unwrap(),
            None,
        )
        .unwrap();
        mgr.set_grant(&id, grant).await;

        // Remove session — budget and grant should be cleaned up.
        let removed = mgr.remove("cli", "alice").await;
        assert_eq!(removed, Some(id.clone()));
        assert!(mgr.get_budget(&id).await.is_none());
        assert!(mgr.get_grant(&id).await.is_none());
    }

    // ── TTL eviction tests ──────────────────────────────────────

    #[tokio::test]
    async fn test_expired_sessions_are_evicted() {
        // Use a 1-second TTL so we can expire alice but keep bob fresh.
        let config = SessionConfig {
            max_sessions: 100,
            session_ttl_secs: 1,
        };
        let mgr = SessionManager::with_config(config);

        let id1 = mgr.resolve("cli", "alice").await;
        assert!(!id1.as_str().is_empty());

        // Wait for alice's TTL to expire
        tokio::time::sleep(std::time::Duration::from_millis(1100)).await;

        // Resolving bob triggers eviction after insert.
        // Alice's session is expired, bob's is fresh.
        let _id2 = mgr.resolve("cli", "bob").await;

        // Alice's session should have been evicted
        assert!(mgr.get("cli", "alice").await.is_none());
        // Bob's session should still exist
        assert!(mgr.get("cli", "bob").await.is_some());
    }

    #[tokio::test]
    async fn test_lru_eviction_removes_oldest() {
        let config = SessionConfig {
            max_sessions: 2,
            session_ttl_secs: 86_400,
        };
        let mgr = SessionManager::with_config(config);

        let _id1 = mgr.resolve("cli", "alice").await;
        tokio::time::sleep(std::time::Duration::from_millis(5)).await;
        let _id2 = mgr.resolve("cli", "bob").await;
        tokio::time::sleep(std::time::Duration::from_millis(5)).await;

        // Adding a third session triggers eviction after insert.
        // Carol is inserted (count=3), then evict_lru removes alice (count=2).
        let _id3 = mgr.resolve("cli", "carol").await;

        assert_eq!(
            mgr.session_count().await,
            2,
            "session count should be 2 after LRU eviction"
        );
        // Alice (oldest) should be evicted
        assert!(
            mgr.get("cli", "alice").await.is_none(),
            "oldest session (alice) should be evicted"
        );
        // Bob and carol should remain
        assert!(mgr.get("cli", "bob").await.is_some());
        assert!(mgr.get("cli", "carol").await.is_some());
    }

    #[tokio::test]
    async fn test_eviction_cleans_up_budgets_and_grants() {
        let config = SessionConfig {
            max_sessions: 100,
            session_ttl_secs: 1,
        };
        let mgr = SessionManager::with_config(config);

        let id = mgr.resolve("cli", "alice").await;
        let budget_config = freebird_types::config::BudgetConfig::default();
        mgr.set_budget(
            &id,
            freebird_security::budget::TokenBudget::new(&budget_config),
        )
        .await;
        let grant = freebird_security::capability::CapabilityGrant::new(
            std::iter::once(freebird_traits::tool::Capability::FileRead).collect(),
            std::env::current_dir().unwrap(),
            None,
        )
        .unwrap();
        mgr.set_grant(&id, grant).await;

        // Wait for alice's TTL to expire, then trigger eviction
        tokio::time::sleep(std::time::Duration::from_millis(1100)).await;
        let _id2 = mgr.resolve("cli", "bob").await;

        // Budget and grant for evicted session should be gone
        assert!(mgr.get_budget(&id).await.is_none());
        assert!(mgr.get_grant(&id).await.is_none());
    }
}
