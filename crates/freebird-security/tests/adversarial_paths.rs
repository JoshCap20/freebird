//! Adversarial tests for `SafeFilePath` — path traversal, symlink escapes,
//! null bytes, encoding tricks.

#![allow(
    clippy::all,
    clippy::pedantic,
    clippy::nursery,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

mod helpers;

use freebird_security::safe_types::SafeFilePath;
use freebird_security::taint::Tainted;

// ---------------------------------------------------------------------------
// Basic traversal attacks
// ---------------------------------------------------------------------------

#[test]
fn double_dot_traversal_single() {
    let dir = tempfile::tempdir().unwrap();
    helpers::sandbox_with_file(&dir, "safe.txt", "ok");

    let t = Tainted::new("../escape.txt");
    let result = SafeFilePath::from_tainted(&t, dir.path());
    assert!(result.is_err(), "single ../ traversal must be rejected");
}

#[test]
fn double_dot_traversal_triple() {
    let dir = tempfile::tempdir().unwrap();
    helpers::sandbox_with_file(&dir, "safe.txt", "ok");

    let t = Tainted::new("../../../etc/passwd");
    let result = SafeFilePath::from_tainted(&t, dir.path());
    assert!(result.is_err(), "triple ../ traversal must be rejected");
}

#[test]
fn dot_dot_in_middle() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(dir.path().join("sub")).unwrap();
    helpers::sandbox_with_file(&dir, "sub/file.txt", "ok");

    let t = Tainted::new("sub/../../etc/passwd");
    let result = SafeFilePath::from_tainted(&t, dir.path());
    assert!(result.is_err(), "../ in middle of path must be rejected");
}

// ---------------------------------------------------------------------------
// Null byte attacks
// ---------------------------------------------------------------------------

#[test]
fn null_byte_in_path() {
    let dir = tempfile::tempdir().unwrap();
    let t = Tainted::new("file.txt\0.sh");
    let result = SafeFilePath::from_tainted(&t, dir.path());
    assert!(result.is_err(), "null byte in path must be rejected");
}

#[test]
fn null_byte_before_traversal() {
    let dir = tempfile::tempdir().unwrap();
    let t = Tainted::new("safe\0/../etc/passwd");
    let result = SafeFilePath::from_tainted(&t, dir.path());
    assert!(result.is_err(), "null byte must be rejected");
}

// ---------------------------------------------------------------------------
// Absolute path rejection
// ---------------------------------------------------------------------------

#[test]
fn absolute_path_unix() {
    let dir = tempfile::tempdir().unwrap();
    let t = Tainted::new("/etc/passwd");
    let result = SafeFilePath::from_tainted(&t, dir.path());
    assert!(result.is_err(), "absolute paths must be rejected");
}

#[test]
fn leading_backslash_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let t = Tainted::new("\\etc\\passwd");
    let result = SafeFilePath::from_tainted(&t, dir.path());
    assert!(result.is_err(), "leading backslash must be rejected");
}

// ---------------------------------------------------------------------------
// Symlink escapes
// ---------------------------------------------------------------------------

#[cfg(unix)]
#[test]
fn symlink_escape_direct() {
    use std::os::unix::fs::symlink;
    let dir = tempfile::tempdir().unwrap();
    let outside = tempfile::tempdir().unwrap();
    std::fs::write(outside.path().join("secret.txt"), "secret").unwrap();

    // Create symlink inside sandbox pointing outside
    symlink(outside.path(), dir.path().join("escape_link")).unwrap();

    let t = Tainted::new("escape_link/secret.txt");
    let result = SafeFilePath::from_tainted(&t, dir.path());
    assert!(
        result.is_err(),
        "symlink pointing outside sandbox must be rejected"
    );
}

#[cfg(unix)]
#[test]
fn symlink_chain_escape() {
    let dir = tempfile::tempdir().unwrap();
    let outside = tempfile::tempdir().unwrap();
    std::fs::write(outside.path().join("target.txt"), "secret").unwrap();

    // Create chain: link_0 -> link_1 -> link_2 -> outside/target.txt
    let entry = helpers::create_symlink_chain(dir.path(), 3, &outside.path().join("target.txt"));

    let link_name = entry.file_name().unwrap().to_str().unwrap();
    let t = Tainted::new(link_name);
    let result = SafeFilePath::from_tainted(&t, dir.path());
    assert!(
        result.is_err(),
        "symlink chain escaping sandbox must be rejected"
    );
}

#[cfg(unix)]
#[test]
fn symlink_to_parent_directory() {
    use std::os::unix::fs::symlink;
    let dir = tempfile::tempdir().unwrap();
    let parent = dir.path().parent().unwrap();

    symlink(parent, dir.path().join("parent_link")).unwrap();

    let t = Tainted::new("parent_link/anything");
    let result = SafeFilePath::from_tainted(&t, dir.path());
    assert!(
        result.is_err(),
        "symlink to parent directory must be rejected"
    );
}

// ---------------------------------------------------------------------------
// Creation mode tests
// ---------------------------------------------------------------------------

#[test]
fn creation_mode_traversal_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let t = Tainted::new("../../newfile.txt");
    let result = SafeFilePath::from_tainted_for_creation(&t, dir.path());
    assert!(result.is_err(), "creation mode must reject traversal");
}

#[test]
fn creation_mode_valid_new_file() {
    let dir = tempfile::tempdir().unwrap();
    let t = Tainted::new("newfile.txt");
    let result = SafeFilePath::from_tainted_for_creation(&t, dir.path());
    assert!(result.is_ok(), "creating a new file in sandbox should work");
}

#[test]
fn creation_mode_trailing_slash_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let t = Tainted::new("subdir/");
    let result = SafeFilePath::from_tainted_for_creation(&t, dir.path());
    assert!(
        result.is_err(),
        "trailing slash indicates directory, not file"
    );
}

#[cfg(unix)]
#[test]
fn creation_mode_symlink_overwrite_escape() {
    use std::os::unix::fs::symlink;
    let dir = tempfile::tempdir().unwrap();
    let outside = tempfile::tempdir().unwrap();
    let target = outside.path().join("overwrite_target.txt");
    std::fs::write(&target, "original").unwrap();

    // Symlink inside sandbox pointing to file outside sandbox
    symlink(&target, dir.path().join("escape.txt")).unwrap();

    let t = Tainted::new("escape.txt");
    let result = SafeFilePath::from_tainted_for_creation(&t, dir.path());
    assert!(
        result.is_err(),
        "creation mode must detect symlink overwrite escape"
    );
}

#[cfg(unix)]
#[test]
fn creation_mode_parent_is_symlink_to_outside() {
    use std::os::unix::fs::symlink;
    let dir = tempfile::tempdir().unwrap();
    let outside = tempfile::tempdir().unwrap();

    // Symlink dir inside sandbox pointing to outside dir
    symlink(outside.path(), dir.path().join("subdir")).unwrap();

    let t = Tainted::new("subdir/newfile.txt");
    let result = SafeFilePath::from_tainted_for_creation(&t, dir.path());
    assert!(
        result.is_err(),
        "parent being a symlink to outside must be rejected in creation mode"
    );
}

// ---------------------------------------------------------------------------
// Path edge cases
// ---------------------------------------------------------------------------

#[test]
fn double_slash_path() {
    let dir = tempfile::tempdir().unwrap();
    helpers::sandbox_with_file(&dir, "file.txt", "ok");

    let t = Tainted::new("file.txt");
    let result = SafeFilePath::from_tainted(&t, dir.path());
    assert!(result.is_ok(), "simple path should work");

    // Double slash should not enable traversal
    let t2 = Tainted::new(".//file.txt");
    let _ = SafeFilePath::from_tainted(&t2, dir.path());
    // Whether this passes depends on canonicalization — no panic is the requirement
}

#[test]
fn dot_only_filenames() {
    let dir = tempfile::tempdir().unwrap();

    let dot = Tainted::new(".");
    let dotdot = Tainted::new("..");
    let dotdotdot = Tainted::new("...");

    // "." and ".." should be rejected (traversal/not-a-file)
    // "..." is unusual but depends on implementation
    assert!(SafeFilePath::from_tainted(&dotdot, dir.path()).is_err());
    // "." resolves to sandbox root itself — may or may not be an error
    let _ = SafeFilePath::from_tainted(&dot, dir.path());
    let _ = SafeFilePath::from_tainted(&dotdotdot, dir.path());
}

#[test]
fn empty_path_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let t = Tainted::new("");
    let result = SafeFilePath::from_tainted(&t, dir.path());
    assert!(result.is_err(), "empty path must be rejected");
}

#[test]
fn whitespace_only_path_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let t = Tainted::new("   ");
    let result = SafeFilePath::from_tainted(&t, dir.path());
    assert!(result.is_err(), "whitespace-only path must be rejected");
}

#[test]
fn extremely_long_path_component() {
    let dir = tempfile::tempdir().unwrap();
    let long_name = "a".repeat(10_000);
    let t = Tainted::new(&long_name);
    let result = SafeFilePath::from_tainted(&t, dir.path());
    // Should return an error (OS limit), not panic
    assert!(result.is_err(), "extremely long path should fail");
}

#[test]
fn deeply_nested_path() {
    let dir = tempfile::tempdir().unwrap();
    let deep = helpers::create_deep_dirs(dir.path(), 50);
    std::fs::write(deep.join("file.txt"), "ok").unwrap();

    // Build the relative path
    let mut rel = String::new();
    for i in 0..50 {
        rel.push_str(&format!("d{i}/"));
    }
    rel.push_str("file.txt");

    let t = Tainted::new(&rel);
    let result = SafeFilePath::from_tainted(&t, dir.path());
    assert!(
        result.is_ok(),
        "deeply nested path within sandbox should work"
    );
}

#[test]
fn hidden_dotfile_in_sandbox() {
    let dir = tempfile::tempdir().unwrap();
    helpers::sandbox_with_file(&dir, ".hidden", "secret");

    let t = Tainted::new(".hidden");
    let result = SafeFilePath::from_tainted(&t, dir.path());
    assert!(
        result.is_ok(),
        "dotfiles within sandbox should be accessible"
    );
}

#[test]
fn encoded_path_separators_not_decoded() {
    let dir = tempfile::tempdir().unwrap();
    // %2F is URL-encoded slash — path API should NOT URL-decode
    let t = Tainted::new("subdir%2F..%2F..%2Fetc%2Fpasswd");
    let result = SafeFilePath::from_tainted(&t, dir.path());
    // Should either fail (path doesn't exist) or treat as literal filename
    // Either way, must not traverse
    assert!(result.is_err());
}

// ---------------------------------------------------------------------------
// proptest: arbitrary paths never escape sandbox
// ---------------------------------------------------------------------------

#[cfg(test)]
mod proptest_paths {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn arbitrary_path_never_escapes_sandbox(input in "\\PC{1,100}") {
            let dir = tempfile::tempdir().unwrap();
            let t = Tainted::new(&input);
            if let Ok(safe) = SafeFilePath::from_tainted(&t, dir.path()) {
                // If it succeeded, verify it's actually within the sandbox
                let sandbox_canonical = dir.path().canonicalize().unwrap();
                prop_assert!(
                    safe.as_path().starts_with(&sandbox_canonical),
                    "path {:?} escaped sandbox {:?}",
                    safe.as_path(),
                    sandbox_canonical
                );
            }
            // Errors are fine — they mean the path was rejected
        }

        #[test]
        fn arbitrary_path_creation_never_escapes_sandbox(input in "\\PC{1,100}") {
            let dir = tempfile::tempdir().unwrap();
            let t = Tainted::new(&input);
            if let Ok(safe) = SafeFilePath::from_tainted_for_creation(&t, dir.path()) {
                let sandbox_canonical = dir.path().canonicalize().unwrap();
                prop_assert!(
                    safe.as_path().starts_with(&sandbox_canonical),
                    "creation path {:?} escaped sandbox {:?}",
                    safe.as_path(),
                    sandbox_canonical
                );
            }
        }
    }
}
