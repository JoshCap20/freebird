//! Shared edit history for undo/rollback support.
//!
//! Stores per-session, per-file version history (ring buffer) and named
//! checkpoints (in-memory snapshots). Shared via `Arc<EditHistory>` between
//! the edit, undo, and checkpoint tools.

use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;
use std::sync::{PoisonError, RwLock};
use std::time::{Duration, Instant};

use freebird_traits::id::SessionId;

/// Maximum number of previous versions kept per file per session.
const MAX_VERSIONS_PER_FILE: usize = 10;

/// Maximum number of active checkpoints per session.
const MAX_CHECKPOINTS_PER_SESSION: usize = 5;

/// Checkpoints auto-expire after this duration.
const CHECKPOINT_TTL: Duration = Duration::from_secs(3600);

/// A previous version of a file, stored before an edit.
struct FileVersion {
    content: String,
    #[allow(dead_code)]
    timestamp: Instant,
}

/// A named checkpoint: snapshots of modified files at a point in time.
struct Checkpoint {
    files: HashMap<PathBuf, String>,
    created_at: Instant,
}

/// Per-session edit state: undo history + checkpoints.
struct SessionState {
    /// Per-file ring buffer of previous contents (LIFO, max 10).
    file_versions: HashMap<PathBuf, VecDeque<FileVersion>>,
    /// Named checkpoints keyed by name.
    checkpoints: HashMap<String, Checkpoint>,
    /// Insertion order for checkpoint eviction (front = oldest).
    checkpoint_order: VecDeque<String>,
}

impl SessionState {
    fn new() -> Self {
        Self {
            file_versions: HashMap::new(),
            checkpoints: HashMap::new(),
            checkpoint_order: VecDeque::new(),
        }
    }
}

/// Thread-safe, session-scoped edit history.
///
/// Shared via `Arc<EditHistory>` between the edit, undo, and checkpoint tools.
/// Uses `std::sync::RwLock` because all critical sections are sub-microsecond
/// in-memory operations with no `.await`.
pub struct EditHistory {
    sessions: RwLock<HashMap<SessionId, SessionState>>,
}

// `significant_drop_tightening` is suppressed: lock guards must be held for
// the entire critical section, not dropped after a single use.
#[allow(clippy::significant_drop_tightening)]
impl EditHistory {
    pub fn new() -> Self {
        Self {
            sessions: RwLock::new(HashMap::new()),
        }
    }

    /// Record the pre-edit content of a file. Pushes onto the per-file
    /// ring buffer, evicting the oldest entry if at capacity.
    pub fn record_pre_edit(&self, session_id: &SessionId, path: PathBuf, content: String) {
        let mut sessions = self
            .sessions
            .write()
            .unwrap_or_else(PoisonError::into_inner);
        let state = sessions
            .entry(session_id.clone())
            .or_insert_with(SessionState::new);

        let versions = state.file_versions.entry(path).or_default();
        if versions.len() >= MAX_VERSIONS_PER_FILE {
            versions.pop_front();
        }
        versions.push_back(FileVersion {
            content,
            timestamp: Instant::now(),
        });
    }

    /// Pop the most recent version from a file's undo stack.
    /// Returns `None` if no history exists for this session/path.
    pub fn pop_last_version(&self, session_id: &SessionId, path: &PathBuf) -> Option<String> {
        let mut sessions = self
            .sessions
            .write()
            .unwrap_or_else(PoisonError::into_inner);
        let state = sessions.get_mut(session_id)?;
        let versions = state.file_versions.get_mut(path)?;
        let version = versions.pop_back()?;
        // Clean up empty entries
        if versions.is_empty() {
            state.file_versions.remove(path);
        }
        Some(version.content)
    }

    /// Return all file paths that have been modified in this session.
    pub fn modified_files(&self, session_id: &SessionId) -> Vec<PathBuf> {
        let sessions = self.sessions.read().unwrap_or_else(PoisonError::into_inner);
        sessions
            .get(session_id)
            .map(|s| s.file_versions.keys().cloned().collect())
            .unwrap_or_default()
    }

    /// Return how many undo steps remain for a specific file.
    pub fn version_count(&self, session_id: &SessionId, path: &PathBuf) -> usize {
        let sessions = self.sessions.read().unwrap_or_else(PoisonError::into_inner);
        sessions
            .get(session_id)
            .and_then(|s| s.file_versions.get(path))
            .map_or(0, VecDeque::len)
    }

    /// Create a named checkpoint from the given file snapshots.
    ///
    /// Evicts expired checkpoints first, then the oldest if at capacity.
    /// Rejects duplicate checkpoint names.
    pub fn create_checkpoint(
        &self,
        session_id: &SessionId,
        name: String,
        files: HashMap<PathBuf, String>,
    ) -> Result<(), &'static str> {
        let mut sessions = self
            .sessions
            .write()
            .unwrap_or_else(PoisonError::into_inner);
        let state = sessions
            .entry(session_id.clone())
            .or_insert_with(SessionState::new);

        if state.checkpoints.contains_key(&name) {
            return Err("checkpoint with this name already exists");
        }

        // Evict expired checkpoints first
        let now = Instant::now();
        let expired: Vec<String> = state
            .checkpoints
            .iter()
            .filter(|(_, cp)| now.duration_since(cp.created_at) >= CHECKPOINT_TTL)
            .map(|(n, _)| n.clone())
            .collect();
        for expired_name in &expired {
            state.checkpoints.remove(expired_name);
            state.checkpoint_order.retain(|n| n != expired_name);
        }

        // Evict oldest if still at capacity
        if state.checkpoints.len() >= MAX_CHECKPOINTS_PER_SESSION {
            if let Some(oldest) = state.checkpoint_order.pop_front() {
                state.checkpoints.remove(&oldest);
            }
        }

        state.checkpoint_order.push_back(name.clone());
        state.checkpoints.insert(
            name,
            Checkpoint {
                files,
                created_at: Instant::now(),
            },
        );

        Ok(())
    }

    /// Take (consume) a named checkpoint, returning its file snapshots.
    ///
    /// The checkpoint is removed from state. Returns an error if the
    /// checkpoint doesn't exist or has expired.
    pub fn take_checkpoint(
        &self,
        session_id: &SessionId,
        name: &str,
    ) -> Result<HashMap<PathBuf, String>, &'static str> {
        let mut sessions = self
            .sessions
            .write()
            .unwrap_or_else(PoisonError::into_inner);
        let state = sessions
            .get_mut(session_id)
            .ok_or("no checkpoints exist for this session")?;

        let checkpoint = state
            .checkpoints
            .remove(name)
            .ok_or("checkpoint not found")?;

        state.checkpoint_order.retain(|n| n != name);

        if Instant::now().duration_since(checkpoint.created_at) >= CHECKPOINT_TTL {
            return Err("checkpoint has expired");
        }

        Ok(checkpoint.files)
    }

    /// Remove all state for a session.
    #[allow(dead_code)] // Public API for future runtime integration
    pub fn cleanup_session(&self, session_id: &SessionId) {
        let mut sessions = self
            .sessions
            .write()
            .unwrap_or_else(PoisonError::into_inner);
        sessions.remove(session_id);
    }

    /// Insert a checkpoint with a custom creation time. Test-only.
    #[cfg(test)]
    pub fn insert_checkpoint_at(
        &self,
        session_id: &SessionId,
        name: String,
        files: HashMap<PathBuf, String>,
        created_at: Instant,
    ) {
        let mut sessions = self
            .sessions
            .write()
            .unwrap_or_else(PoisonError::into_inner);
        let state = sessions
            .entry(session_id.clone())
            .or_insert_with(SessionState::new);
        state.checkpoint_order.push_back(name.clone());
        state
            .checkpoints
            .insert(name, Checkpoint { files, created_at });
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use freebird_traits::id::SessionId;

    fn session(name: &str) -> SessionId {
        SessionId::from_string(name.to_string())
    }

    fn path(name: &str) -> PathBuf {
        PathBuf::from(name)
    }

    // ── Ring buffer tests ──────────────────────────────────────

    #[test]
    fn test_ring_buffer_lifo_order() {
        let history = EditHistory::new();
        let sid = session("s1");
        let p = path("/tmp/file.rs");

        history.record_pre_edit(&sid, p.clone(), "v1".into());
        history.record_pre_edit(&sid, p.clone(), "v2".into());
        history.record_pre_edit(&sid, p.clone(), "v3".into());

        assert_eq!(history.pop_last_version(&sid, &p), Some("v3".into()));
        assert_eq!(history.pop_last_version(&sid, &p), Some("v2".into()));
        assert_eq!(history.pop_last_version(&sid, &p), Some("v1".into()));
        assert_eq!(history.pop_last_version(&sid, &p), None);
    }

    #[test]
    fn test_ring_buffer_evicts_oldest_at_capacity() {
        let history = EditHistory::new();
        let sid = session("s1");
        let p = path("/tmp/file.rs");

        // Record 11 versions — oldest should be evicted
        for i in 0..=10 {
            history.record_pre_edit(&sid, p.clone(), format!("v{i}"));
        }

        assert_eq!(history.version_count(&sid, &p), MAX_VERSIONS_PER_FILE);

        // v0 was evicted; oldest remaining is v1
        let mut popped = Vec::new();
        while let Some(v) = history.pop_last_version(&sid, &p) {
            popped.push(v);
        }
        assert_eq!(popped.len(), 10);
        assert_eq!(popped.last().unwrap(), "v1"); // oldest remaining
        assert!(!popped.contains(&"v0".to_string())); // evicted
    }

    #[test]
    fn test_pop_empty_returns_none() {
        let history = EditHistory::new();
        let sid = session("s1");
        let p = path("/tmp/unknown.rs");

        assert_eq!(history.pop_last_version(&sid, &p), None);

        // Also for unknown session
        let unknown = session("unknown");
        assert_eq!(history.pop_last_version(&unknown, &p), None);
    }

    #[test]
    fn test_session_isolation() {
        let history = EditHistory::new();
        let s1 = session("s1");
        let s2 = session("s2");
        let p = path("/tmp/file.rs");

        history.record_pre_edit(&s1, p.clone(), "s1-content".into());
        history.record_pre_edit(&s2, p.clone(), "s2-content".into());

        assert_eq!(history.pop_last_version(&s1, &p), Some("s1-content".into()));
        assert_eq!(history.pop_last_version(&s2, &p), Some("s2-content".into()));
    }

    // ── Checkpoint tests ───────────────────────────────────────

    #[test]
    fn test_checkpoint_create_and_take() {
        let history = EditHistory::new();
        let sid = session("s1");

        let files = HashMap::from([
            (path("/tmp/a.rs"), String::from("content-a")),
            (path("/tmp/b.rs"), String::from("content-b")),
        ]);

        history
            .create_checkpoint(&sid, "cp1".into(), files)
            .unwrap();

        let restored = history.take_checkpoint(&sid, "cp1").unwrap();
        assert_eq!(restored.len(), 2);
        assert_eq!(restored.get(&path("/tmp/a.rs")).unwrap(), "content-a");
        assert_eq!(restored.get(&path("/tmp/b.rs")).unwrap(), "content-b");
    }

    #[test]
    fn test_checkpoint_evicts_oldest_at_capacity() {
        let history = EditHistory::new();
        let sid = session("s1");

        // Create 5 checkpoints (at capacity)
        for i in 0..5 {
            history
                .create_checkpoint(
                    &sid,
                    format!("cp{i}"),
                    HashMap::from([(path("/tmp/f.rs"), format!("v{i}"))]),
                )
                .unwrap();
        }

        // 6th should evict cp0 (oldest)
        history
            .create_checkpoint(
                &sid,
                "cp5".into(),
                HashMap::from([(path("/tmp/f.rs"), "v5".into())]),
            )
            .unwrap();

        // cp0 was evicted
        let result = history.take_checkpoint(&sid, "cp0");
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "checkpoint not found");

        // cp1 through cp5 still exist
        for i in 1..=5 {
            let restored = history.take_checkpoint(&sid, &format!("cp{i}")).unwrap();
            assert_eq!(restored.get(&path("/tmp/f.rs")).unwrap(), &format!("v{i}"));
        }
    }

    #[test]
    fn test_checkpoint_expiry_rejected() {
        let history = EditHistory::new();
        let sid = session("s1");

        // Insert a checkpoint with a creation time 2 hours in the past
        history.insert_checkpoint_at(
            &sid,
            "expired-cp".into(),
            HashMap::from([(path("/tmp/f.rs"), "old".into())]),
            Instant::now()
                .checked_sub(Duration::from_secs(7200))
                .unwrap(),
        );

        let result = history.take_checkpoint(&sid, "expired-cp");
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "checkpoint has expired");
    }

    #[test]
    fn test_checkpoint_duplicate_name_rejected() {
        let history = EditHistory::new();
        let sid = session("s1");

        history
            .create_checkpoint(&sid, "dup".into(), HashMap::new())
            .unwrap();

        let result = history.create_checkpoint(&sid, "dup".into(), HashMap::new());
        assert!(result.is_err());
        assert_eq!(
            result.unwrap_err(),
            "checkpoint with this name already exists"
        );
    }

    #[test]
    fn test_modified_files_tracks_edited_paths() {
        let history = EditHistory::new();
        let sid = session("s1");

        history.record_pre_edit(&sid, path("/tmp/a.rs"), "a".into());
        history.record_pre_edit(&sid, path("/tmp/b.rs"), "b".into());
        history.record_pre_edit(&sid, path("/tmp/a.rs"), "a2".into());

        let mut files = history.modified_files(&sid);
        files.sort();
        assert_eq!(files, vec![path("/tmp/a.rs"), path("/tmp/b.rs")]);
    }

    #[test]
    fn test_cleanup_session_removes_all_state() {
        let history = EditHistory::new();
        let sid = session("s1");

        history.record_pre_edit(&sid, path("/tmp/f.rs"), "content".into());
        history
            .create_checkpoint(&sid, "cp".into(), HashMap::new())
            .unwrap();

        history.cleanup_session(&sid);

        assert!(history.modified_files(&sid).is_empty());
        assert_eq!(history.version_count(&sid, &path("/tmp/f.rs")), 0);
        assert!(history.take_checkpoint(&sid, "cp").is_err());
    }

    #[test]
    fn test_checkpoint_not_found_for_unknown_session() {
        let history = EditHistory::new();
        let sid = session("unknown");

        let result = history.take_checkpoint(&sid, "cp");
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "no checkpoints exist for this session");
    }
}
