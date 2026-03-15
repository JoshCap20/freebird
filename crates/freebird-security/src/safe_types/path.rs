//! Safe filesystem path types.

use std::path::{Path, PathBuf};

use crate::error::SecurityError;
use crate::taint::Tainted;

// ── SafeFilePath ─────────────────────────────────────────────────

/// A filesystem path that has been canonicalized and verified to be
/// within the sandbox root.
///
/// Produced by: tool input extraction.
/// Consumed by: filesystem tools (`read_file`, `write_file`).
#[derive(Debug, Clone)]
pub struct SafeFilePath {
    resolved: PathBuf,
    root: PathBuf,
}

impl SafeFilePath {
    /// Shared early-rejection checks for all path constructors.
    ///
    /// Rejects inputs that are categorically invalid regardless of which
    /// constructor is being used:
    /// - Empty or whitespace-only (masks tool input bugs)
    /// - Null bytes (OS-level path separator confusion)
    /// - Absolute paths (`PathBuf::join` silently discards the root)
    fn reject_invalid_raw(raw: &str, sandbox: &Path) -> Result<(), SecurityError> {
        let traversal = || SecurityError::PathTraversal {
            attempted: PathBuf::from(raw),
            sandbox: sandbox.to_owned(),
        };

        if raw.trim().is_empty() {
            return Err(traversal());
        }

        if raw.contains('\0') {
            return Err(traversal());
        }

        if raw.starts_with('/') || raw.starts_with('\\') {
            return Err(traversal());
        }

        Ok(())
    }

    /// Validate untrusted input as a filesystem path within a sandbox.
    ///
    /// - Rejects empty, whitespace-only, and absolute paths
    /// - Rejects null bytes
    /// - Canonicalizes (resolves symlinks, `..`, `.`)
    /// - Verifies result is within sandbox root
    ///
    /// # Errors
    ///
    /// Returns `SecurityError::PathTraversal` if the path is empty, absolute,
    /// contains null bytes, or escapes the sandbox after canonicalization.
    /// Returns `SecurityError::PathResolution` if canonicalization fails
    /// (e.g., the path does not exist).
    pub fn from_tainted(t: &Tainted, sandbox: &Path) -> Result<Self, SecurityError> {
        let raw = t.inner();
        Self::reject_invalid_raw(raw, sandbox)?;

        let root = sandbox
            .canonicalize()
            .map_err(|e| SecurityError::PathResolution {
                path: sandbox.to_owned(),
                source: e,
            })?;

        let candidate = root.join(raw);
        let resolved = candidate
            .canonicalize()
            .map_err(|e| SecurityError::PathResolution {
                path: candidate,
                source: e,
            })?;

        if !resolved.starts_with(&root) {
            return Err(SecurityError::PathTraversal {
                attempted: resolved,
                sandbox: root,
            });
        }

        Ok(Self { resolved, root })
    }

    /// Validate untrusted input for file creation (target may not exist yet).
    ///
    /// `std::fs::canonicalize()` requires the path to exist. For `write_file`
    /// creating new files, the full path doesn't exist yet. This variant:
    /// - Canonicalizes the parent directory (which must exist)
    /// - Validates the filename component
    /// - If the full path happens to exist (overwrite case), canonicalizes it
    ///   to catch symlink escapes
    /// - Verifies resolved path is within sandbox
    ///
    /// # Errors
    ///
    /// - `PathTraversal` — empty input, absolute path, null bytes, sandbox escape,
    ///   no filename component, trailing slash, or existing symlink pointing outside sandbox
    /// - `PathResolution` — parent directory doesn't exist or sandbox can't be canonicalized
    ///
    /// # Known Limitation: TOCTOU
    ///
    /// `canonicalize()` resolves at call time. Between validation and use, an
    /// attacker with write access inside the sandbox could create a symlink.
    /// The overwrite-safe check mitigates the existing-symlink case but cannot
    /// prevent race-condition symlink creation. See `openat2(2)` / `cap-std`
    /// for kernel-level fix.
    pub fn from_tainted_for_creation(t: &Tainted, sandbox: &Path) -> Result<Self, SecurityError> {
        let raw = t.inner();
        Self::reject_invalid_raw(raw, sandbox)?;

        let traversal = || SecurityError::PathTraversal {
            attempted: PathBuf::from(raw),
            sandbox: sandbox.to_owned(),
        };

        // Reject trailing slash (indicates directory, not file)
        if raw.ends_with('/') || raw.ends_with('\\') {
            return Err(traversal());
        }

        let path = Path::new(raw);

        // Extract filename — reject if none (catches "." and "..")
        let filename = path.file_name().ok_or_else(traversal)?;

        // Canonicalize sandbox root
        let root = sandbox
            .canonicalize()
            .map_err(|e| SecurityError::PathResolution {
                path: sandbox.to_owned(),
                source: e,
            })?;

        // Canonicalize parent directory (must exist).
        // SAFETY-ARGUMENT: parent() returns None only for root ("/") or empty (""),
        // both rejected by reject_invalid_raw above.
        let parent = path.parent().unwrap_or_else(|| Path::new(""));
        let parent_candidate = root.join(parent);
        let canonical_parent =
            parent_candidate
                .canonicalize()
                .map_err(|e| SecurityError::PathResolution {
                    path: parent_candidate,
                    source: e,
                })?;

        // Verify parent is within sandbox
        if !canonical_parent.starts_with(&root) {
            return Err(SecurityError::PathTraversal {
                attempted: canonical_parent,
                sandbox: root,
            });
        }

        let resolved = canonical_parent.join(filename);

        // Overwrite-safe: attempt to canonicalize the resolved path to detect
        // symlink escapes. A symlink to /etc/passwd at this path would pass
        // parent validation but canonicalize reveals the escape.
        //
        // We canonicalize unconditionally instead of checking exists() first,
        // eliminating a TOCTOU race between the existence check and canonicalize.
        match resolved.canonicalize() {
            Ok(actual) => {
                if !actual.starts_with(&root) {
                    return Err(SecurityError::PathTraversal {
                        attempted: actual,
                        sandbox: root,
                    });
                }
                Ok(Self {
                    resolved: actual,
                    root,
                })
            }
            // NotFound means the path doesn't exist yet — this is the creation
            // case, so no symlink escape is possible at this location.
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Self { resolved, root }),
            Err(e) => Err(SecurityError::PathResolution {
                path: resolved,
                source: e,
            }),
        }
    }

    // ── Multi-root absolute path support ────────────────────────────

    /// Validate an untrusted path that may be absolute (within an allowed
    /// directory) or relative (within the sandbox).
    ///
    /// Allowed roots extend the sandbox: an absolute path that resolves
    /// within any root is accepted. Relative paths are always resolved
    /// against `sandbox` (the primary root).
    ///
    /// # Invariant
    ///
    /// All entries in `allowed_dirs` **must** be pre-canonicalized by the
    /// caller (the daemon does this at startup). Non-canonical entries will
    /// silently fail to match because `starts_with` compares canonical
    /// resolved paths against the entries as-is.
    ///
    /// # Errors
    ///
    /// Returns `SecurityError::PathTraversal` if the path escapes all roots.
    /// Returns `SecurityError::PathResolution` if canonicalization fails.
    pub fn from_tainted_multi_root(
        t: &Tainted,
        sandbox: &Path,
        allowed_dirs: &[PathBuf],
    ) -> Result<Self, SecurityError> {
        let raw = t.inner();

        // Relative paths always resolve against sandbox only.
        if !raw.starts_with('/') && !raw.starts_with('\\') {
            return Self::from_tainted(t, sandbox);
        }

        Self::resolve_absolute(raw, sandbox, allowed_dirs)
    }

    /// Like [`from_tainted_multi_root`](Self::from_tainted_multi_root) but
    /// for file creation (target may not exist yet).
    ///
    /// # Invariant
    ///
    /// Same pre-canonicalization requirement on `allowed_dirs`.
    ///
    /// # Errors
    ///
    /// Returns path-related errors if the absolute path escapes all roots,
    /// or delegates to `from_tainted_for_creation` for relative paths.
    pub fn from_tainted_for_creation_multi_root(
        t: &Tainted,
        sandbox: &Path,
        allowed_dirs: &[PathBuf],
    ) -> Result<Self, SecurityError> {
        let raw = t.inner();

        if !raw.starts_with('/') && !raw.starts_with('\\') {
            return Self::from_tainted_for_creation(t, sandbox);
        }

        Self::resolve_absolute_for_creation(raw, sandbox, allowed_dirs)
    }

    /// Shared early-rejection for absolute paths: empty, whitespace, null bytes.
    fn reject_invalid_absolute(raw: &str, sandbox: &Path) -> Result<(), SecurityError> {
        if raw.trim().is_empty() || raw.contains('\0') {
            return Err(SecurityError::PathTraversal {
                attempted: PathBuf::from(raw),
                sandbox: sandbox.to_owned(),
            });
        }
        Ok(())
    }

    /// Canonicalize sandbox and debug-assert that `allowed_dirs` are canonical.
    ///
    /// Allowed directories are canonicalized once at daemon startup. This
    /// method verifies the invariant in debug builds and returns the
    /// canonical sandbox root.
    fn prepare_roots(sandbox: &Path, allowed_dirs: &[PathBuf]) -> Result<PathBuf, SecurityError> {
        let sandbox_root = sandbox
            .canonicalize()
            .map_err(|e| SecurityError::PathResolution {
                path: sandbox.to_owned(),
                source: e,
            })?;

        // In debug builds, verify the caller's invariant. In release builds
        // non-canonical entries simply won't match (safe but silent).
        for dir in allowed_dirs {
            debug_assert!(
                dir.is_absolute(),
                "allowed_dirs entry must be absolute: {}",
                dir.display()
            );
        }

        Ok(sandbox_root)
    }

    /// Check whether `resolved` is within `sandbox_root` or any allowed dir.
    fn check_containment(
        resolved: PathBuf,
        sandbox_root: &Path,
        allowed_dirs: &[PathBuf],
    ) -> Result<Self, SecurityError> {
        if resolved.starts_with(sandbox_root) {
            return Ok(Self {
                resolved,
                root: sandbox_root.to_owned(),
            });
        }

        for dir in allowed_dirs {
            if resolved.starts_with(dir) {
                return Ok(Self {
                    resolved,
                    root: dir.clone(),
                });
            }
        }

        Err(SecurityError::PathTraversal {
            attempted: resolved,
            sandbox: sandbox_root.to_owned(),
        })
    }

    /// Resolve an absolute path against sandbox + allowed directories.
    fn resolve_absolute(
        raw: &str,
        sandbox: &Path,
        allowed_dirs: &[PathBuf],
    ) -> Result<Self, SecurityError> {
        Self::reject_invalid_absolute(raw, sandbox)?;

        let candidate = PathBuf::from(raw);
        let resolved = candidate
            .canonicalize()
            .map_err(|e| SecurityError::PathResolution {
                path: candidate,
                source: e,
            })?;

        let sandbox_root = Self::prepare_roots(sandbox, allowed_dirs)?;
        Self::check_containment(resolved, &sandbox_root, allowed_dirs)
    }

    /// Resolve an absolute path for creation against sandbox + allowed dirs.
    fn resolve_absolute_for_creation(
        raw: &str,
        sandbox: &Path,
        allowed_dirs: &[PathBuf],
    ) -> Result<Self, SecurityError> {
        Self::reject_invalid_absolute(raw, sandbox)?;

        let traversal = || SecurityError::PathTraversal {
            attempted: PathBuf::from(raw),
            sandbox: sandbox.to_owned(),
        };

        if raw.ends_with('/') || raw.ends_with('\\') {
            return Err(traversal());
        }

        let path = Path::new(raw);
        let filename = path.file_name().ok_or_else(traversal)?;
        let parent = path.parent().ok_or_else(traversal)?;

        let canonical_parent =
            parent
                .canonicalize()
                .map_err(|e| SecurityError::PathResolution {
                    path: parent.to_owned(),
                    source: e,
                })?;

        let resolved = canonical_parent.join(filename);
        let sandbox_root = Self::prepare_roots(sandbox, allowed_dirs)?;

        // Find which root contains the parent directory.
        let root = if canonical_parent.starts_with(&sandbox_root) {
            sandbox_root
        } else if let Some(dir) = allowed_dirs
            .iter()
            .find(|d| canonical_parent.starts_with(*d))
        {
            dir.clone()
        } else {
            return Err(SecurityError::PathTraversal {
                attempted: resolved,
                sandbox: sandbox_root,
            });
        };

        Self::overwrite_safe_check(resolved, root, raw)
    }

    /// Symlink escape check for creation paths that already exist.
    fn overwrite_safe_check(
        resolved: PathBuf,
        root: PathBuf,
        raw: &str,
    ) -> Result<Self, SecurityError> {
        match resolved.canonicalize() {
            Ok(actual) => {
                if !actual.starts_with(&root) {
                    return Err(SecurityError::PathTraversal {
                        attempted: actual,
                        sandbox: root,
                    });
                }
                Ok(Self {
                    resolved: actual,
                    root,
                })
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Self { resolved, root }),
            Err(e) => Err(SecurityError::PathResolution {
                path: PathBuf::from(raw),
                source: e,
            }),
        }
    }

    /// Access the validated, canonicalized path.
    #[must_use]
    pub fn as_path(&self) -> &Path {
        &self.resolved
    }

    /// Access the sandbox root this path was validated against.
    #[must_use]
    pub fn root(&self) -> &Path {
        &self.root
    }
}

// ── SafeFileContent ──────────────────────────────────────────────

/// Opaque wrapper for file content extracted from tainted tool input.
///
/// No validation beyond the taint boundary — file content is arbitrary
/// text. This type exists solely to bridge the `pub(crate)` taint
/// boundary: `TaintedToolInput::extract_file_content()` returns this
/// type, and `WriteFileTool` consumes it via `.as_str()`.
#[derive(Debug)]
pub struct SafeFileContent(String);

impl SafeFileContent {
    /// Wrap raw file content from tainted input.
    ///
    /// No content validation — file content is arbitrary text.
    pub(crate) const fn new(content: String) -> Self {
        Self(content)
    }

    /// Access the file content.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// The byte length of the content.
    #[must_use]
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Whether the content is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic
)]
mod tests {
    use super::*;
    use crate::taint::Tainted;
    use std::os::unix::fs::symlink;
    use tempfile::TempDir;

    // ── SafeFilePath tests ───────────────────────────────────────

    #[test]
    fn test_rejects_empty_path() {
        let tmp = TempDir::new().unwrap();
        let t = Tainted::new("");
        let err = SafeFilePath::from_tainted(&t, tmp.path()).unwrap_err();
        assert!(matches!(err, SecurityError::PathTraversal { .. }));
    }

    #[test]
    fn test_rejects_whitespace_path() {
        let tmp = TempDir::new().unwrap();
        let t = Tainted::new("   ");
        let err = SafeFilePath::from_tainted(&t, tmp.path()).unwrap_err();
        assert!(matches!(err, SecurityError::PathTraversal { .. }));
    }

    #[test]
    fn test_rejects_null_byte() {
        let tmp = TempDir::new().unwrap();
        let t = Tainted::new("file\0.txt");
        let err = SafeFilePath::from_tainted(&t, tmp.path()).unwrap_err();
        assert!(matches!(err, SecurityError::PathTraversal { .. }));
    }

    #[test]
    fn test_rejects_absolute_path() {
        let tmp = TempDir::new().unwrap();
        let t = Tainted::new("/etc/passwd");
        let err = SafeFilePath::from_tainted(&t, tmp.path()).unwrap_err();
        assert!(matches!(err, SecurityError::PathTraversal { .. }));
    }

    #[test]
    fn test_rejects_dotdot_escape() {
        let tmp = TempDir::new().unwrap();
        std::fs::write(tmp.path().join("file.txt"), "ok").unwrap();
        let t = Tainted::new("../../../etc/passwd");
        let err = SafeFilePath::from_tainted(&t, tmp.path()).unwrap_err();
        assert!(matches!(
            err,
            SecurityError::PathTraversal { .. } | SecurityError::PathResolution { .. }
        ));
    }

    #[test]
    fn test_accepts_valid_path() {
        let tmp = TempDir::new().unwrap();
        std::fs::write(tmp.path().join("hello.txt"), "content").unwrap();
        let t = Tainted::new("hello.txt");
        let safe = SafeFilePath::from_tainted(&t, tmp.path()).unwrap();
        assert!(safe.as_path().ends_with("hello.txt"));
    }

    #[test]
    fn test_from_tainted_dotdot_within_sandbox_ok() {
        let tmp = TempDir::new().unwrap();
        let sub = tmp.path().join("a").join("b");
        std::fs::create_dir_all(&sub).unwrap();
        std::fs::write(tmp.path().join("a").join("target.txt"), "ok").unwrap();

        let t = Tainted::new("a/b/../target.txt");
        let safe = SafeFilePath::from_tainted(&t, tmp.path()).unwrap();
        assert!(safe.as_path().ends_with("target.txt"));
        assert!(
            safe.as_path()
                .starts_with(tmp.path().canonicalize().unwrap())
        );
    }

    #[test]
    fn test_from_tainted_symlink_within_sandbox_ok() {
        let tmp = TempDir::new().unwrap();
        let real = tmp.path().join("real.txt");
        let link = tmp.path().join("link.txt");
        std::fs::write(&real, "content").unwrap();
        symlink(&real, &link).unwrap();

        let t = Tainted::new("link.txt");
        let safe = SafeFilePath::from_tainted(&t, tmp.path()).unwrap();
        // Resolved through symlink, still within sandbox
        assert!(
            safe.as_path()
                .starts_with(tmp.path().canonicalize().unwrap())
        );
    }

    // ── SafeFilePath for creation tests ──────────────────────────

    #[test]
    fn test_creation_new_file() {
        let tmp = TempDir::new().unwrap();
        let t = Tainted::new("newfile.txt");
        let safe = SafeFilePath::from_tainted_for_creation(&t, tmp.path()).unwrap();
        assert!(safe.as_path().ends_with("newfile.txt"));
    }

    #[test]
    fn test_creation_rejects_trailing_slash() {
        let tmp = TempDir::new().unwrap();
        let t = Tainted::new("dir/");
        let err = SafeFilePath::from_tainted_for_creation(&t, tmp.path()).unwrap_err();
        assert!(matches!(err, SecurityError::PathTraversal { .. }));
    }

    #[test]
    fn test_creation_rejects_dotdot_escape() {
        let tmp = TempDir::new().unwrap();
        let t = Tainted::new("../../escape.txt");
        let err = SafeFilePath::from_tainted_for_creation(&t, tmp.path()).unwrap_err();
        assert!(matches!(
            err,
            SecurityError::PathTraversal { .. } | SecurityError::PathResolution { .. }
        ));
    }

    // ── Multi-root absolute path tests ───────────────────────────

    #[test]
    fn test_multi_root_relative_delegates_to_sandbox() {
        let tmp = TempDir::new().unwrap();
        std::fs::write(tmp.path().join("file.txt"), "content").unwrap();

        let t = Tainted::new("file.txt");
        let safe = SafeFilePath::from_tainted_multi_root(&t, tmp.path(), &[]).unwrap();
        assert!(safe.as_path().ends_with("file.txt"));
    }

    #[test]
    fn test_multi_root_absolute_in_allowed_dir_accepted() {
        let sandbox = TempDir::new().unwrap();
        let allowed = TempDir::new().unwrap();
        std::fs::write(allowed.path().join("code.rs"), "fn main() {}").unwrap();

        let allowed_canonical = allowed.path().canonicalize().unwrap();
        let abs_path = allowed_canonical.join("code.rs");
        let t = Tainted::new(abs_path.to_str().unwrap());

        let safe = SafeFilePath::from_tainted_multi_root(
            &t,
            sandbox.path(),
            std::slice::from_ref(&allowed_canonical),
        )
        .unwrap();
        assert_eq!(safe.as_path(), abs_path);
        assert_eq!(safe.root(), allowed_canonical);
    }

    #[test]
    fn test_multi_root_absolute_in_sandbox_accepted() {
        let sandbox = TempDir::new().unwrap();
        std::fs::write(sandbox.path().join("data.txt"), "ok").unwrap();

        let sandbox_canonical = sandbox.path().canonicalize().unwrap();
        let abs_path = sandbox_canonical.join("data.txt");
        let t = Tainted::new(abs_path.to_str().unwrap());

        let safe = SafeFilePath::from_tainted_multi_root(&t, sandbox.path(), &[]).unwrap();
        assert_eq!(safe.as_path(), abs_path);
    }

    #[test]
    fn test_multi_root_absolute_outside_all_roots_rejected() {
        let sandbox = TempDir::new().unwrap();
        let allowed = TempDir::new().unwrap();
        let outside = TempDir::new().unwrap();
        std::fs::write(outside.path().join("secret.txt"), "secret").unwrap();

        let allowed_canonical = allowed.path().canonicalize().unwrap();
        let abs_path = outside.path().canonicalize().unwrap().join("secret.txt");
        let t = Tainted::new(abs_path.to_str().unwrap());

        let err = SafeFilePath::from_tainted_multi_root(&t, sandbox.path(), &[allowed_canonical])
            .unwrap_err();
        assert!(matches!(err, SecurityError::PathTraversal { .. }));
    }

    #[test]
    fn test_multi_root_absolute_traversal_escape_rejected() {
        let sandbox = TempDir::new().unwrap();
        let allowed = TempDir::new().unwrap();
        std::fs::create_dir_all(allowed.path().join("sub")).unwrap();

        let allowed_canonical = allowed.path().canonicalize().unwrap();
        // Try to traverse out of allowed dir
        let escape_path = format!("{}/../../../etc/passwd", allowed_canonical.display());
        let t = Tainted::new(&escape_path);

        let err = SafeFilePath::from_tainted_multi_root(&t, sandbox.path(), &[allowed_canonical]);
        // Either PathResolution (doesn't exist) or PathTraversal (escaped)
        assert!(err.is_err());
    }

    #[test]
    fn test_multi_root_absolute_symlink_escape_rejected() {
        let sandbox = TempDir::new().unwrap();
        let allowed = TempDir::new().unwrap();
        let outside = TempDir::new().unwrap();

        // Create a file outside allowed dir
        let outside_file = outside.path().join("secret.txt");
        std::fs::write(&outside_file, "secret").unwrap();

        // Create symlink inside allowed dir pointing outside
        let link = allowed.path().join("escape_link.txt");
        symlink(&outside_file, &link).unwrap();

        let allowed_canonical = allowed.path().canonicalize().unwrap();
        let abs_path = format!("{}/escape_link.txt", allowed_canonical.display());
        let t = Tainted::new(&abs_path);

        let err = SafeFilePath::from_tainted_multi_root(&t, sandbox.path(), &[allowed_canonical])
            .unwrap_err();
        assert!(matches!(err, SecurityError::PathTraversal { .. }));
    }

    #[test]
    fn test_multi_root_absolute_null_byte_rejected() {
        let sandbox = TempDir::new().unwrap();
        let t = Tainted::new("/some/path\0/file.txt");

        let err = SafeFilePath::from_tainted_multi_root(&t, sandbox.path(), &[]).unwrap_err();
        assert!(matches!(err, SecurityError::PathTraversal { .. }));
    }

    #[test]
    fn test_multi_root_root_path_rejected() {
        let sandbox = TempDir::new().unwrap();
        let t = Tainted::new("/");

        let err = SafeFilePath::from_tainted_multi_root(&t, sandbox.path(), &[]).unwrap_err();
        assert!(err.to_string().contains("traversal") || err.to_string().contains("resolution"));
    }

    #[test]
    fn test_multi_root_empty_allowed_dirs() {
        let sandbox = TempDir::new().unwrap();
        let outside = TempDir::new().unwrap();
        std::fs::write(outside.path().join("file.txt"), "content").unwrap();

        let abs_path = outside.path().canonicalize().unwrap().join("file.txt");
        let t = Tainted::new(abs_path.to_str().unwrap());

        let err = SafeFilePath::from_tainted_multi_root(&t, sandbox.path(), &[]).unwrap_err();
        assert!(matches!(err, SecurityError::PathTraversal { .. }));
    }

    // ── Multi-root creation tests ────────────────────────────────

    #[test]
    fn test_multi_root_creation_in_allowed_dir_accepted() {
        let sandbox = TempDir::new().unwrap();
        let allowed = TempDir::new().unwrap();

        let allowed_canonical = allowed.path().canonicalize().unwrap();
        let abs_path = format!("{}/new_file.txt", allowed_canonical.display());
        let t = Tainted::new(&abs_path);

        let safe = SafeFilePath::from_tainted_for_creation_multi_root(
            &t,
            sandbox.path(),
            std::slice::from_ref(&allowed_canonical),
        )
        .unwrap();
        assert!(safe.as_path().ends_with("new_file.txt"));
        assert_eq!(safe.root(), allowed_canonical);
    }

    #[test]
    fn test_multi_root_creation_outside_all_roots_rejected() {
        let sandbox = TempDir::new().unwrap();
        let outside = TempDir::new().unwrap();

        let abs_path = format!(
            "{}/evil.txt",
            outside.path().canonicalize().unwrap().display()
        );
        let t = Tainted::new(&abs_path);

        let err = SafeFilePath::from_tainted_for_creation_multi_root(&t, sandbox.path(), &[])
            .unwrap_err();
        assert!(matches!(err, SecurityError::PathTraversal { .. }));
    }

    #[test]
    fn test_multi_root_creation_symlink_escape_rejected() {
        let sandbox = TempDir::new().unwrap();
        let allowed = TempDir::new().unwrap();
        let outside = TempDir::new().unwrap();

        // Create existing file outside allowed dir
        let outside_file = outside.path().join("target.txt");
        std::fs::write(&outside_file, "target").unwrap();

        // Create symlink inside allowed dir pointing to outside file
        let link = allowed.path().join("escape.txt");
        symlink(&outside_file, &link).unwrap();

        let allowed_canonical = allowed.path().canonicalize().unwrap();
        let abs_path = format!("{}/escape.txt", allowed_canonical.display());
        let t = Tainted::new(&abs_path);

        let err = SafeFilePath::from_tainted_for_creation_multi_root(
            &t,
            sandbox.path(),
            &[allowed_canonical],
        )
        .unwrap_err();
        assert!(matches!(err, SecurityError::PathTraversal { .. }));
    }

    #[test]
    fn test_multi_root_creation_trailing_slash_rejected() {
        let sandbox = TempDir::new().unwrap();
        let allowed = TempDir::new().unwrap();

        let allowed_canonical = allowed.path().canonicalize().unwrap();
        let abs_path = format!("{}/dir/", allowed_canonical.display());
        let t = Tainted::new(&abs_path);

        let err = SafeFilePath::from_tainted_for_creation_multi_root(
            &t,
            sandbox.path(),
            &[allowed_canonical],
        )
        .unwrap_err();
        assert!(matches!(err, SecurityError::PathTraversal { .. }));
    }

    #[test]
    fn test_multi_root_creation_relative_delegates_to_sandbox() {
        let sandbox = TempDir::new().unwrap();
        let t = Tainted::new("new_file.txt");

        let safe =
            SafeFilePath::from_tainted_for_creation_multi_root(&t, sandbox.path(), &[]).unwrap();
        assert!(safe.as_path().ends_with("new_file.txt"));
    }

    // ── Property-based tests ─────────────────────────────────────

    #[cfg(test)]
    mod proptest_tests {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            #[test]
            fn prop_safe_file_path_never_escapes_sandbox(input in "\\PC*") {
                let tmp = TempDir::new().unwrap();
                // Any input — safe or adversarial — must never produce a path outside sandbox
                if let Ok(safe) = SafeFilePath::from_tainted(&Tainted::new(&input), tmp.path()) {
                    let canonical_sandbox = tmp.path().canonicalize().unwrap();
                    assert!(
                        safe.as_path().starts_with(&canonical_sandbox),
                        "path escaped sandbox: {:?} not in {:?}",
                        safe.as_path(),
                        canonical_sandbox
                    );
                }
            }

            #[test]
            fn prop_from_tainted_for_creation_never_escapes_sandbox(input in "\\PC*") {
                let tmp = TempDir::new().unwrap();
                if let Ok(safe) = SafeFilePath::from_tainted_for_creation(&Tainted::new(&input), tmp.path()) {
                    let canonical_sandbox = tmp.path().canonicalize().unwrap();
                    assert!(
                        safe.as_path().starts_with(&canonical_sandbox),
                        "from_tainted_for_creation escaped sandbox: {:?} not in {:?}",
                        safe.as_path(),
                        canonical_sandbox
                    );
                }
            }

            #[test]
            fn prop_multi_root_never_escapes_any_root(input in "\\PC*") {
                let sandbox = TempDir::new().unwrap();
                let allowed = TempDir::new().unwrap();
                let allowed_canonical = allowed.path().canonicalize().unwrap();
                let sandbox_canonical = sandbox.path().canonicalize().unwrap();

                if let Ok(safe) = SafeFilePath::from_tainted_multi_root(
                    &Tainted::new(&input),
                    sandbox.path(),
                    std::slice::from_ref(&allowed_canonical),
                ) {
                    assert!(
                        safe.as_path().starts_with(&sandbox_canonical)
                            || safe.as_path().starts_with(&allowed_canonical),
                        "multi_root escaped all roots: {:?}",
                        safe.as_path()
                    );
                }
            }
        }
    }
}
