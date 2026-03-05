//! `CapabilityGrant` system for scoped permissions.
//!
//! Every tool invocation must pass a capability check before execution.
//! Sub-agents receive grants that are provably strict subsets of their
//! parent's grant — in capabilities, sandbox scope, and expiration.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use freebird_traits::tool::Capability;

use crate::error::SecurityError;

/// Raw deserialization target for `CapabilityGrant`.
///
/// Used by `TryFrom` to enforce post-deserialization validation:
/// the `sandbox_root` must exist and be canonicalizable.
#[derive(Deserialize)]
struct RawCapabilityGrant {
    capabilities: BTreeSet<Capability>,
    sandbox_root: PathBuf,
    expires_at: Option<DateTime<Utc>>,
}

/// A scoped, optionally time-limited set of capabilities bound to a sandbox root.
///
/// Invariants (enforced by construction):
/// - Capabilities are an explicit allow-set (deny-by-default).
/// - Sandbox root is canonicalized at construction time (parse, don't validate).
/// - Sub-grants are strict subsets: fewer-or-equal capabilities,
///   equal-or-narrower sandbox, equal-or-earlier expiration.
/// - An expired grant cannot produce sub-grants.
///
/// Uses `BTreeSet` (not `HashSet`) for deterministic serialization order.
/// This ensures stable debug output, predictable test assertions, and
/// future-proof compatibility with HMAC signing of structures containing grants.
///
/// # Deserialization safety
///
/// Deserialization validates that `sandbox_root` exists and can be
/// canonicalized. This prevents deserialized grants from bypassing the
/// path validation performed by `CapabilityGrant::new()`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "RawCapabilityGrant")]
pub struct CapabilityGrant {
    capabilities: BTreeSet<Capability>,
    sandbox_root: PathBuf,
    expires_at: Option<DateTime<Utc>>,
}

impl TryFrom<RawCapabilityGrant> for CapabilityGrant {
    type Error = String;

    fn try_from(raw: RawCapabilityGrant) -> Result<Self, Self::Error> {
        let resolved = raw.sandbox_root.canonicalize().map_err(|e| {
            format!(
                "sandbox_root `{}` cannot be canonicalized: {e}",
                raw.sandbox_root.display()
            )
        })?;

        Ok(Self {
            capabilities: raw.capabilities,
            sandbox_root: resolved,
            expires_at: raw.expires_at,
        })
    }
}

impl CapabilityGrant {
    /// Create a new grant.
    ///
    /// `sandbox_root` is canonicalized at construction to ensure consistent
    /// path comparison in `sub_grant()`. This follows the "parse, don't validate"
    /// principle: if the sandbox root doesn't exist, fail now rather than later.
    ///
    /// # Errors
    ///
    /// Returns `SecurityError::PathResolution` if `sandbox_root` cannot be
    /// canonicalized (e.g., directory doesn't exist).
    pub fn new(
        capabilities: BTreeSet<Capability>,
        sandbox_root: PathBuf,
        expires_at: Option<DateTime<Utc>>,
    ) -> Result<Self, SecurityError> {
        let resolved = sandbox_root
            .canonicalize()
            .map_err(|e| SecurityError::PathResolution {
                path: sandbox_root,
                source: e,
            })?;

        Ok(Self {
            capabilities,
            sandbox_root: resolved,
            expires_at,
        })
    }

    /// Check if this grant includes the required capability and is still valid.
    ///
    /// Checks expiration FIRST, then capability. This ordering matters:
    /// - `GrantExpired` -> caller should re-authenticate.
    /// - `CapabilityDenied` -> caller lacks permissions (different recovery path).
    ///
    /// An expired grant with a valid capability still returns `GrantExpired`.
    ///
    /// # Errors
    ///
    /// - `SecurityError::GrantExpired` if the grant has expired.
    /// - `SecurityError::CapabilityDenied` if the capability is not in the grant.
    #[must_use = "security check result must not be silently discarded"]
    pub fn check(&self, required: &Capability) -> Result<(), SecurityError> {
        self.check_expiration()?;

        if !self.capabilities.contains(required) {
            return Err(SecurityError::CapabilityDenied {
                capability: required.clone(),
            });
        }

        Ok(())
    }

    /// Check multiple capabilities atomically.
    ///
    /// Checks expiration first, then verifies all capabilities are present.
    /// Returns the first missing capability on failure.
    ///
    /// # Errors
    ///
    /// - `SecurityError::GrantExpired` if the grant has expired.
    /// - `SecurityError::CapabilityDenied` with the first missing capability.
    #[must_use = "security check result must not be silently discarded"]
    pub fn check_all(&self, required: &[Capability]) -> Result<(), SecurityError> {
        self.check_expiration()?;

        for cap in required {
            if !self.capabilities.contains(cap) {
                return Err(SecurityError::CapabilityDenied {
                    capability: cap.clone(),
                });
            }
        }

        Ok(())
    }

    /// Create a sub-grant that is a strict subset of this grant.
    ///
    /// Checks parent expiration first — an expired grant CANNOT produce sub-grants.
    /// This prevents time-based privilege escalation where a long-running sub-agent
    /// continues to derive grants after the parent session has expired.
    ///
    /// Then enforces three containment invariants:
    ///
    /// 1. **Capabilities**: child must be a subset of parent.
    /// 2. **Sandbox**: child sandbox must be equal to or a subdirectory of parent sandbox.
    ///    Uses `canonicalize()` + `starts_with()` (same approach as `SafeFilePath`).
    /// 3. **Expiration**: child cannot outlive parent. Clamping rules:
    ///    - Parent `Some(T)`, Child `None` -> child clamped to `Some(T)`.
    ///    - Parent `Some(T1)`, Child `Some(T2)` where `T2 > T1` -> clamped to `Some(T1)`.
    ///    - Parent `None`, Child anything -> allowed as-is.
    ///
    /// # Errors
    ///
    /// - `SecurityError::GrantExpired` if the parent grant has expired.
    /// - `SecurityError::SubGrantExceedsParent` if child requests capabilities not in parent.
    /// - `SecurityError::PathResolution` if child sandbox cannot be canonicalized.
    /// - `SecurityError::SubGrantSandboxEscape` if child sandbox is outside parent sandbox.
    pub fn sub_grant(
        &self,
        capabilities: BTreeSet<Capability>,
        sandbox_root: PathBuf,
        expires_at: Option<DateTime<Utc>>,
    ) -> Result<Self, SecurityError> {
        // 1. Check parent expiration — expired grants cannot produce sub-grants
        self.check_expiration()?;

        // 2. Capability containment — child must be subset of parent
        let denied: Vec<Capability> = capabilities
            .iter()
            .filter(|cap| !self.capabilities.contains(cap))
            .cloned()
            .collect();

        if !denied.is_empty() {
            return Err(SecurityError::SubGrantExceedsParent { denied });
        }

        // 3. Sandbox containment — child must be within parent sandbox
        let child_sandbox =
            sandbox_root
                .canonicalize()
                .map_err(|e| SecurityError::PathResolution {
                    path: sandbox_root,
                    source: e,
                })?;

        if !child_sandbox.starts_with(&self.sandbox_root) {
            return Err(SecurityError::SubGrantSandboxEscape {
                child: child_sandbox,
                parent: self.sandbox_root.clone(),
            });
        }

        // 4. Expiration clamping — child cannot outlive parent
        let clamped_expiration = match (self.expires_at, expires_at) {
            (Some(parent_exp), None) => Some(parent_exp),
            (Some(parent_exp), Some(child_exp)) => Some(parent_exp.min(child_exp)),
            (None, child_exp) => child_exp,
        };

        Ok(Self {
            capabilities,
            sandbox_root: child_sandbox,
            expires_at: clamped_expiration,
        })
    }

    /// Read-only view of granted capabilities.
    #[must_use]
    pub const fn capabilities(&self) -> &BTreeSet<Capability> {
        &self.capabilities
    }

    /// The sandbox root this grant is bound to (canonicalized at construction).
    #[must_use]
    pub fn sandbox_root(&self) -> &Path {
        &self.sandbox_root
    }

    /// Whether this grant has expired (returns false if no expiration set).
    #[must_use]
    pub fn is_expired(&self) -> bool {
        self.expires_at.is_some_and(|exp| Utc::now() > exp)
    }

    /// The expiration time, if set.
    #[must_use]
    pub const fn expires_at(&self) -> Option<DateTime<Utc>> {
        self.expires_at
    }

    /// Check expiration and return `GrantExpired` error if past due.
    /// Extracted to avoid `expect()` on the invariant that `is_expired()` implies
    /// `expires_at` is `Some`.
    fn check_expiration(&self) -> Result<(), SecurityError> {
        if let Some(exp) = self.expires_at {
            if Utc::now() > exp {
                return Err(SecurityError::GrantExpired { expired_at: exp });
            }
        }
        Ok(())
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;

    use chrono::Duration;
    use tempfile::TempDir;

    /// Helper: create a temp directory and return its canonical path.
    fn sandbox() -> (TempDir, PathBuf) {
        let tmp = TempDir::new().expect("failed to create temp dir");
        let path = tmp.path().canonicalize().expect("failed to canonicalize");
        (tmp, path)
    }

    /// Helper: create a `BTreeSet` from a slice of capabilities.
    fn caps(items: &[Capability]) -> BTreeSet<Capability> {
        items.iter().cloned().collect()
    }

    /// Helper: all capabilities.
    ///
    /// Guarded by a static assertion against `CAPABILITY_VARIANT_COUNT` so
    /// adding a new variant without updating this list is a compile error.
    fn all_caps() -> BTreeSet<Capability> {
        const _: () = assert!(
            freebird_traits::tool::CAPABILITY_VARIANT_COUNT == 8,
            "Capability variant added — update all_caps() and arbitrary_capability()"
        );
        caps(&[
            Capability::FileRead,
            Capability::FileWrite,
            Capability::FileDelete,
            Capability::ShellExecute,
            Capability::ProcessSpawn,
            Capability::NetworkOutbound,
            Capability::NetworkListen,
            Capability::EnvRead,
        ])
    }

    // ── Construction tests ─────────────────────────────────────────

    #[test]
    fn test_new_valid() {
        let (_tmp, path) = sandbox();
        let grant = CapabilityGrant::new(
            caps(&[Capability::FileRead, Capability::FileWrite]),
            path,
            Some(Utc::now() + Duration::hours(1)),
        )
        .expect("should be Ok");

        assert_eq!(grant.capabilities().len(), 2);
        assert!(grant.capabilities().contains(&Capability::FileRead));
        assert!(grant.capabilities().contains(&Capability::FileWrite));
    }

    #[test]
    fn test_new_canonicalizes_sandbox() {
        let tmp = TempDir::new().expect("failed to create temp dir");
        let subdir = tmp.path().join("a").join("b");
        std::fs::create_dir_all(&subdir).expect("failed to create subdir");

        // Path with `..` segments
        let non_canonical = subdir.join("..").join("b");
        let grant =
            CapabilityGrant::new(BTreeSet::new(), non_canonical, None).expect("should be Ok");

        let canonical = subdir.canonicalize().expect("failed to canonicalize");
        assert_eq!(grant.sandbox_root(), canonical.as_path());
    }

    #[test]
    fn test_new_nonexistent_sandbox() {
        let result =
            CapabilityGrant::new(BTreeSet::new(), PathBuf::from("/nonexistent/path"), None);

        let err = result.expect_err("should be Err");
        assert!(
            matches!(err, SecurityError::PathResolution { .. }),
            "expected PathResolution, got {err:?}"
        );
    }

    #[test]
    fn test_new_empty_capabilities() {
        let (_tmp, path) = sandbox();
        let grant = CapabilityGrant::new(BTreeSet::new(), path, None).expect("should be Ok");
        assert!(grant.capabilities().is_empty());
    }

    // ── Capability check tests ─────────────────────────────────────

    #[test]
    fn test_check_granted_capability() {
        let (_tmp, path) = sandbox();
        let grant =
            CapabilityGrant::new(caps(&[Capability::FileRead]), path, None).expect("should be Ok");
        assert!(grant.check(&Capability::FileRead).is_ok());
    }

    #[test]
    fn test_check_denied_capability() {
        let (_tmp, path) = sandbox();
        let grant =
            CapabilityGrant::new(caps(&[Capability::FileRead]), path, None).expect("should be Ok");

        let err = grant
            .check(&Capability::ShellExecute)
            .expect_err("should be Err");
        assert!(
            matches!(
                err,
                SecurityError::CapabilityDenied {
                    capability: Capability::ShellExecute
                }
            ),
            "expected CapabilityDenied(ShellExecute), got {err:?}"
        );
    }

    #[test]
    fn test_check_expired_grant_with_valid_capability() {
        let (_tmp, path) = sandbox();
        let grant = CapabilityGrant::new(
            caps(&[Capability::FileRead]),
            path,
            Some(Utc::now() - Duration::hours(1)),
        )
        .expect("should be Ok");

        let err = grant
            .check(&Capability::FileRead)
            .expect_err("should be Err");
        assert!(
            matches!(err, SecurityError::GrantExpired { .. }),
            "expected GrantExpired, got {err:?}"
        );
    }

    #[test]
    fn test_check_expired_grant_with_missing_capability() {
        let (_tmp, path) = sandbox();
        let grant = CapabilityGrant::new(
            caps(&[Capability::FileRead]),
            path,
            Some(Utc::now() - Duration::hours(1)),
        )
        .expect("should be Ok");

        // Missing capability, but expiration takes precedence
        let err = grant
            .check(&Capability::ShellExecute)
            .expect_err("should be Err");
        assert!(
            matches!(err, SecurityError::GrantExpired { .. }),
            "expected GrantExpired (not CapabilityDenied), got {err:?}"
        );
    }

    #[test]
    fn test_check_no_expiration() {
        let (_tmp, path) = sandbox();
        let grant =
            CapabilityGrant::new(caps(&[Capability::FileRead]), path, None).expect("should be Ok");
        assert!(grant.check(&Capability::FileRead).is_ok());
    }

    #[test]
    fn test_check_all_all_present() {
        let (_tmp, path) = sandbox();
        let grant = CapabilityGrant::new(
            caps(&[Capability::FileRead, Capability::FileWrite]),
            path,
            None,
        )
        .expect("should be Ok");

        assert!(
            grant
                .check_all(&[Capability::FileRead, Capability::FileWrite])
                .is_ok()
        );
    }

    #[test]
    fn test_check_all_one_missing() {
        let (_tmp, path) = sandbox();
        let grant =
            CapabilityGrant::new(caps(&[Capability::FileRead]), path, None).expect("should be Ok");

        let err = grant
            .check_all(&[Capability::FileRead, Capability::ShellExecute])
            .expect_err("should be Err");
        assert!(
            matches!(
                err,
                SecurityError::CapabilityDenied {
                    capability: Capability::ShellExecute
                }
            ),
            "expected CapabilityDenied(ShellExecute), got {err:?}"
        );
    }

    #[test]
    fn test_check_all_expired() {
        let (_tmp, path) = sandbox();
        let grant = CapabilityGrant::new(
            caps(&[Capability::FileRead]),
            path,
            Some(Utc::now() - Duration::hours(1)),
        )
        .expect("should be Ok");

        let err = grant
            .check_all(&[Capability::FileRead])
            .expect_err("should be Err");
        assert!(
            matches!(err, SecurityError::GrantExpired { .. }),
            "expected GrantExpired, got {err:?}"
        );
    }

    #[test]
    fn test_empty_capability_set_denies_everything() {
        let (_tmp, path) = sandbox();
        let grant = CapabilityGrant::new(BTreeSet::new(), path, None).expect("should be Ok");

        let err = grant
            .check(&Capability::FileRead)
            .expect_err("should be Err");
        assert!(
            matches!(
                err,
                SecurityError::CapabilityDenied {
                    capability: Capability::FileRead
                }
            ),
            "expected CapabilityDenied(FileRead), got {err:?}"
        );
    }

    // ── Sub-grant tests: parent expiration ─────────────────────────

    #[test]
    fn test_sub_grant_from_expired_parent() {
        let (_tmp, path) = sandbox();
        let parent = CapabilityGrant::new(
            all_caps(),
            path.clone(),
            Some(Utc::now() - Duration::hours(1)),
        )
        .expect("should be Ok");

        let err = parent
            .sub_grant(BTreeSet::new(), path, None)
            .expect_err("should be Err");
        assert!(
            matches!(err, SecurityError::GrantExpired { .. }),
            "expected GrantExpired, got {err:?}"
        );
    }

    // ── Sub-grant tests: capabilities ──────────────────────────────

    #[test]
    fn test_sub_grant_valid_subset() {
        let (_tmp, path) = sandbox();
        let parent = CapabilityGrant::new(
            caps(&[Capability::FileRead, Capability::FileWrite]),
            path.clone(),
            None,
        )
        .expect("should be Ok");

        let child = parent
            .sub_grant(caps(&[Capability::FileRead]), path, None)
            .expect("should be Ok");
        assert_eq!(child.capabilities().len(), 1);
        assert!(child.capabilities().contains(&Capability::FileRead));
        assert!(!child.capabilities().contains(&Capability::FileWrite));
    }

    #[test]
    fn test_sub_grant_equal_set() {
        let (_tmp, path) = sandbox();
        let parent = CapabilityGrant::new(caps(&[Capability::FileRead]), path.clone(), None)
            .expect("should be Ok");

        let child = parent
            .sub_grant(caps(&[Capability::FileRead]), path, None)
            .expect("should be Ok");
        assert_eq!(child.capabilities(), parent.capabilities());
    }

    #[test]
    fn test_sub_grant_empty_set() {
        let (_tmp, path) = sandbox();
        let parent = CapabilityGrant::new(caps(&[Capability::FileRead]), path.clone(), None)
            .expect("should be Ok");

        let child = parent
            .sub_grant(BTreeSet::new(), path, None)
            .expect("should be Ok");
        assert!(child.capabilities().is_empty());
    }

    #[test]
    fn test_sub_grant_exceeds_parent() {
        let (_tmp, path) = sandbox();
        let parent = CapabilityGrant::new(caps(&[Capability::FileRead]), path.clone(), None)
            .expect("should be Ok");

        let err = parent
            .sub_grant(caps(&[Capability::ShellExecute]), path, None)
            .expect_err("should be Err");
        match err {
            SecurityError::SubGrantExceedsParent { denied } => {
                assert_eq!(denied, vec![Capability::ShellExecute]);
            }
            other => panic!("expected SubGrantExceedsParent, got {other:?}"),
        }
    }

    #[test]
    fn test_sub_grant_partially_exceeds() {
        let (_tmp, path) = sandbox();
        let parent = CapabilityGrant::new(caps(&[Capability::FileRead]), path.clone(), None)
            .expect("should be Ok");

        let err = parent
            .sub_grant(
                caps(&[Capability::FileRead, Capability::ShellExecute]),
                path,
                None,
            )
            .expect_err("should be Err");
        match err {
            SecurityError::SubGrantExceedsParent { denied } => {
                assert_eq!(denied, vec![Capability::ShellExecute]);
            }
            other => panic!("expected SubGrantExceedsParent, got {other:?}"),
        }
    }

    // ── Sub-grant tests: sandbox ───────────────────────────────────

    #[test]
    fn test_sub_grant_sandbox_subdirectory() {
        let tmp = TempDir::new().expect("failed to create temp dir");
        let parent_path = tmp.path().canonicalize().expect("canonicalize");
        let child_path = parent_path.join("subdir");
        std::fs::create_dir(&child_path).expect("create subdir");

        let parent =
            CapabilityGrant::new(all_caps(), parent_path, None).expect("parent should be Ok");

        let child = parent
            .sub_grant(BTreeSet::new(), child_path, None)
            .expect("child should be Ok");
        assert!(child.sandbox_root().starts_with(parent.sandbox_root()));
    }

    #[test]
    fn test_sub_grant_sandbox_same() {
        let (_tmp, path) = sandbox();
        let parent =
            CapabilityGrant::new(all_caps(), path.clone(), None).expect("parent should be Ok");

        let child = parent
            .sub_grant(BTreeSet::new(), path, None)
            .expect("child should be Ok");
        assert_eq!(child.sandbox_root(), parent.sandbox_root());
    }

    #[test]
    fn test_sub_grant_sandbox_escapes() {
        let tmp_a = TempDir::new().expect("create tmp_a");
        let tmp_b = TempDir::new().expect("create tmp_b");
        let path_a = tmp_a.path().canonicalize().expect("canonicalize a");
        let path_b = tmp_b.path().canonicalize().expect("canonicalize b");

        let parent = CapabilityGrant::new(all_caps(), path_a, None).expect("parent should be Ok");

        let err = parent
            .sub_grant(BTreeSet::new(), path_b, None)
            .expect_err("should be Err");
        assert!(
            matches!(err, SecurityError::SubGrantSandboxEscape { .. }),
            "expected SubGrantSandboxEscape, got {err:?}"
        );
    }

    #[test]
    fn test_sub_grant_sandbox_traversal() {
        let tmp = TempDir::new().expect("create temp dir");
        let parent_path = tmp.path().join("a");
        let sibling_path = tmp.path().join("b");
        std::fs::create_dir_all(&parent_path).expect("create a");
        std::fs::create_dir_all(&sibling_path).expect("create b");

        let parent =
            CapabilityGrant::new(all_caps(), parent_path, None).expect("parent should be Ok");

        // Try to escape via `..`
        let traversal = tmp.path().join("a").join("..").join("b");
        let err = parent
            .sub_grant(BTreeSet::new(), traversal, None)
            .expect_err("should be Err");
        assert!(
            matches!(err, SecurityError::SubGrantSandboxEscape { .. }),
            "expected SubGrantSandboxEscape, got {err:?}"
        );
    }

    #[test]
    fn test_sub_grant_sandbox_nonexistent() {
        let (_tmp, path) = sandbox();
        let parent = CapabilityGrant::new(all_caps(), path, None).expect("parent should be Ok");

        let err = parent
            .sub_grant(BTreeSet::new(), PathBuf::from("/nonexistent/path"), None)
            .expect_err("should be Err");
        assert!(
            matches!(err, SecurityError::PathResolution { .. }),
            "expected PathResolution, got {err:?}"
        );
    }

    // ── Sub-grant tests: expiration ────────────────────────────────

    #[test]
    fn test_sub_grant_expiration_within_parent() {
        let (_tmp, path) = sandbox();
        let now = Utc::now();
        let parent_exp = now + Duration::minutes(10);
        let child_exp = now + Duration::minutes(5);

        let parent = CapabilityGrant::new(all_caps(), path.clone(), Some(parent_exp))
            .expect("parent should be Ok");

        let child = parent
            .sub_grant(BTreeSet::new(), path, Some(child_exp))
            .expect("child should be Ok");
        assert_eq!(child.expires_at(), Some(child_exp));
    }

    #[test]
    fn test_sub_grant_expiration_exceeds_parent_clamped() {
        let (_tmp, path) = sandbox();
        let now = Utc::now();
        let parent_exp = now + Duration::minutes(10);
        let child_exp = now + Duration::minutes(20);

        let parent = CapabilityGrant::new(all_caps(), path.clone(), Some(parent_exp))
            .expect("parent should be Ok");

        let child = parent
            .sub_grant(BTreeSet::new(), path, Some(child_exp))
            .expect("child should be Ok");
        // Clamped to parent's expiration
        assert_eq!(child.expires_at(), Some(parent_exp));
    }

    #[test]
    fn test_sub_grant_no_expiration_parent_has_expiration() {
        let (_tmp, path) = sandbox();
        let parent_exp = Utc::now() + Duration::minutes(10);

        let parent = CapabilityGrant::new(all_caps(), path.clone(), Some(parent_exp))
            .expect("parent should be Ok");

        let child = parent
            .sub_grant(BTreeSet::new(), path, None)
            .expect("child should be Ok");
        // Clamped to parent's expiration
        assert_eq!(child.expires_at(), Some(parent_exp));
    }

    #[test]
    fn test_sub_grant_parent_no_expiration_child_has_expiration() {
        let (_tmp, path) = sandbox();
        let child_exp = Utc::now() + Duration::minutes(5);

        let parent =
            CapabilityGrant::new(all_caps(), path.clone(), None).expect("parent should be Ok");

        let child = parent
            .sub_grant(BTreeSet::new(), path, Some(child_exp))
            .expect("child should be Ok");
        assert_eq!(child.expires_at(), Some(child_exp));
    }

    #[test]
    fn test_sub_grant_both_no_expiration() {
        let (_tmp, path) = sandbox();
        let parent =
            CapabilityGrant::new(all_caps(), path.clone(), None).expect("parent should be Ok");

        let child = parent
            .sub_grant(BTreeSet::new(), path, None)
            .expect("child should be Ok");
        assert_eq!(child.expires_at(), None);
    }

    // ── Accessor tests ─────────────────────────────────────────────

    #[test]
    fn test_capabilities_accessor() {
        let (_tmp, path) = sandbox();
        let input_caps = caps(&[Capability::FileRead, Capability::NetworkOutbound]);
        let grant = CapabilityGrant::new(input_caps.clone(), path, None).expect("should be Ok");
        assert_eq!(grant.capabilities(), &input_caps);
    }

    #[test]
    fn test_sandbox_root_accessor_canonical() {
        let tmp = TempDir::new().expect("create temp dir");
        let subdir = tmp.path().join("a");
        std::fs::create_dir(&subdir).expect("create subdir");

        let non_canonical = tmp.path().join("a").join(".").join(".");
        let grant =
            CapabilityGrant::new(BTreeSet::new(), non_canonical, None).expect("should be Ok");

        let canonical = subdir.canonicalize().expect("canonicalize");
        assert_eq!(grant.sandbox_root(), canonical.as_path());
    }

    #[test]
    fn test_is_expired_future() {
        let (_tmp, path) = sandbox();
        let grant =
            CapabilityGrant::new(BTreeSet::new(), path, Some(Utc::now() + Duration::hours(1)))
                .expect("should be Ok");
        assert!(!grant.is_expired());
    }

    #[test]
    fn test_is_expired_past() {
        let (_tmp, path) = sandbox();
        let grant =
            CapabilityGrant::new(BTreeSet::new(), path, Some(Utc::now() - Duration::hours(1)))
                .expect("should be Ok");
        assert!(grant.is_expired());
    }

    #[test]
    fn test_is_expired_none() {
        let (_tmp, path) = sandbox();
        let grant = CapabilityGrant::new(BTreeSet::new(), path, None).expect("should be Ok");
        assert!(!grant.is_expired());
    }

    #[test]
    fn test_expires_at_accessor() {
        let (_tmp, path) = sandbox();
        let exp = Utc::now() + Duration::hours(1);
        let grant = CapabilityGrant::new(BTreeSet::new(), path, Some(exp)).expect("should be Ok");
        assert_eq!(grant.expires_at(), Some(exp));
    }

    // ── Serialization tests ────────────────────────────────────────

    #[test]
    fn test_serde_roundtrip() {
        let (_tmp, path) = sandbox();
        let grant = CapabilityGrant::new(
            caps(&[Capability::FileRead, Capability::FileWrite]),
            path,
            Some(Utc::now() + Duration::hours(1)),
        )
        .expect("should be Ok");

        let json = serde_json::to_string(&grant).expect("serialize");
        let deserialized: CapabilityGrant = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(grant, deserialized);
    }

    #[test]
    fn test_serde_deterministic() {
        let (_tmp, path) = sandbox();
        let grant = CapabilityGrant::new(
            caps(&[Capability::FileWrite, Capability::FileRead]),
            path,
            None,
        )
        .expect("should be Ok");

        let json1 = serde_json::to_string(&grant).expect("serialize 1");
        let json2 = serde_json::to_string(&grant).expect("serialize 2");
        assert_eq!(json1, json2, "BTreeSet should produce deterministic JSON");
    }

    // ── Deserialization safety tests ──────────────────────────────

    #[test]
    fn test_deserialized_grant_enforces_sandbox_on_sub_grant() {
        // Simulate a grant loaded from disk whose sandbox_root is stale
        // (e.g., the directory was deleted and recreated elsewhere).
        // sub_grant() must still enforce containment via canonicalize().
        let tmp = TempDir::new().expect("create temp dir");
        let parent_path = tmp.path().canonicalize().expect("canonicalize");
        let child_dir = parent_path.join("child");
        std::fs::create_dir(&child_dir).expect("create child dir");

        // Construct via new(), serialize, then deserialize — round-tripped
        // grants have canonical paths only if the filesystem hasn't changed.
        let grant =
            CapabilityGrant::new(all_caps(), parent_path.clone(), None).expect("should be Ok");
        let json = serde_json::to_string(&grant).expect("serialize");
        let deserialized: CapabilityGrant = serde_json::from_str(&json).expect("deserialize");

        // sub_grant to a subdirectory should still work
        let child = deserialized
            .sub_grant(BTreeSet::new(), child_dir, None)
            .expect("sub_grant to subdirectory should succeed");
        assert!(child.sandbox_root().starts_with(&parent_path));

        // sub_grant to an outside directory should still be rejected
        let other = TempDir::new().expect("create other dir");
        let other_path = other.path().canonicalize().expect("canonicalize");
        let err = deserialized
            .sub_grant(BTreeSet::new(), other_path, None)
            .expect_err("sub_grant outside sandbox should fail");
        assert!(
            matches!(err, SecurityError::SubGrantSandboxEscape { .. }),
            "expected SubGrantSandboxEscape, got {err:?}"
        );
    }

    #[test]
    fn test_deserialize_rejects_nonexistent_sandbox_root() {
        // Craft JSON with a sandbox_root that does not exist on disk.
        // The custom TryFrom<RawCapabilityGrant> should reject it.
        let json = r#"{"capabilities":["file_read"],"sandbox_root":"/nonexistent/fake/path","expires_at":null}"#;
        let result: Result<CapabilityGrant, _> = serde_json::from_str(json);
        let err = result.expect_err("should reject nonexistent sandbox_root");
        assert!(
            err.to_string().contains("cannot be canonicalized"),
            "expected canonicalization error, got: {err}"
        );
    }

    // ── Property-based tests ───────────────────────────────────────

    mod prop_tests {
        use super::*;
        use proptest::prelude::*;

        fn arbitrary_capability() -> impl Strategy<Value = Capability> {
            prop_oneof![
                Just(Capability::FileRead),
                Just(Capability::FileWrite),
                Just(Capability::FileDelete),
                Just(Capability::ShellExecute),
                Just(Capability::ProcessSpawn),
                Just(Capability::NetworkOutbound),
                Just(Capability::NetworkListen),
                Just(Capability::EnvRead),
            ]
        }

        fn btree_set_of_caps(max: usize) -> impl Strategy<Value = BTreeSet<Capability>> {
            proptest::collection::btree_set(arbitrary_capability(), 0..max)
        }

        proptest! {
            #[test]
            fn sub_grant_capabilities_always_subset(
                parent_caps in btree_set_of_caps(8),
                child_caps in btree_set_of_caps(8),
            ) {
                let tmp = TempDir::new().expect("create temp dir");
                let path = tmp.path().canonicalize().expect("canonicalize");
                let parent = CapabilityGrant::new(parent_caps, path.clone(), None)
                    .expect("parent should be Ok");

                if let Ok(child) = parent.sub_grant(child_caps, path, None) {
                    prop_assert!(child.capabilities().is_subset(parent.capabilities()));
                }
            }

            #[test]
            fn sub_grant_never_outlives_parent(
                parent_ttl in 1i64..3600,
                child_ttl in 0i64..7200,
            ) {
                let tmp = TempDir::new().expect("create temp dir");
                let path = tmp.path().canonicalize().expect("canonicalize");
                let now = Utc::now();
                let parent_exp = now + Duration::seconds(parent_ttl);
                let child_exp = now + Duration::seconds(child_ttl);

                let parent = CapabilityGrant::new(all_caps(), path.clone(), Some(parent_exp))
                    .expect("parent should be Ok");

                if let Ok(child) = parent.sub_grant(BTreeSet::new(), path, Some(child_exp)) {
                    if let Some(exp) = child.expires_at() {
                        prop_assert!(exp <= parent_exp);
                    }
                }
            }

            #[test]
            fn sub_grant_none_expiration_clamped_to_parent(
                parent_ttl in 1i64..3600,
            ) {
                let tmp = TempDir::new().expect("create temp dir");
                let path = tmp.path().canonicalize().expect("canonicalize");
                let now = Utc::now();
                let parent_exp = now + Duration::seconds(parent_ttl);

                let parent = CapabilityGrant::new(all_caps(), path.clone(), Some(parent_exp))
                    .expect("parent should be Ok");

                let child = parent
                    .sub_grant(BTreeSet::new(), path, None)
                    .expect("child should be Ok");
                prop_assert_eq!(child.expires_at(), Some(parent_exp));
            }

            #[test]
            fn sub_grant_sandbox_always_within_parent(
                subdirs in proptest::collection::vec("[a-z]{1,5}", 0..3),
            ) {
                let tmp = TempDir::new().expect("create temp dir");
                let parent_path = tmp.path().canonicalize().expect("canonicalize");
                let child_path = subdirs.iter().fold(parent_path.clone(), |p, s| p.join(s));
                std::fs::create_dir_all(&child_path).expect("create subdirs");

                let parent = CapabilityGrant::new(all_caps(), parent_path, None)
                    .expect("parent should be Ok");

                if let Ok(child) = parent.sub_grant(BTreeSet::new(), child_path, None) {
                    prop_assert!(child.sandbox_root().starts_with(parent.sandbox_root()));
                }
            }
        }
    }
}
