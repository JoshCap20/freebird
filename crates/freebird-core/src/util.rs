//! Shared utility functions for path expansion and directory merging.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

/// Expand `~` prefix to the user's home directory.
///
/// Only expands a leading `~` or `~/` — does not expand `~user`.
pub fn expand_tilde(path: &Path) -> Result<PathBuf> {
    let s = path.to_string_lossy();
    if s == "~" {
        home::home_dir().context("cannot resolve home directory for `~` expansion")
    } else if let Some(rest) = s.strip_prefix("~/") {
        let home = home::home_dir().context("cannot resolve home directory for `~/` expansion")?;
        Ok(home.join(rest))
    } else {
        Ok(path.to_owned())
    }
}

/// Merge additional allowed directories into the tools configuration.
///
/// Expands tilde paths and canonicalizes each directory. Duplicates are
/// silently skipped.
pub fn merge_allow_dirs(
    tools_config: &mut freebird_types::config::ToolsConfig,
    allow_dirs: Vec<PathBuf>,
) -> Result<()> {
    for dir in allow_dirs {
        let expanded = expand_tilde(&dir)?;
        let canonical = expanded.canonicalize().with_context(|| {
            format!(
                "--allow-dir path `{}` does not exist or cannot be resolved",
                dir.display()
            )
        })?;
        if !tools_config.allowed_directories.contains(&canonical) {
            tracing::info!(dir = %canonical.display(), "allowing additional directory");
            tools_config.allowed_directories.push(canonical);
        }
    }
    Ok(())
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::indexing_slicing, clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn test_expand_tilde_home() {
        let result = expand_tilde(Path::new("~")).unwrap();
        if let Some(home) = home::home_dir() {
            assert_eq!(result, home);
        }
    }

    #[test]
    fn test_expand_tilde_home_subpath() {
        let result = expand_tilde(Path::new("~/foo/bar")).unwrap();
        if let Some(home) = home::home_dir() {
            assert_eq!(result, home.join("foo/bar"));
        }
    }

    #[test]
    fn test_expand_tilde_absolute_passthrough() {
        let result = expand_tilde(Path::new("/tmp/test")).unwrap();
        assert_eq!(result, PathBuf::from("/tmp/test"));
    }

    #[test]
    fn test_expand_tilde_relative_passthrough() {
        let result = expand_tilde(Path::new("data/sessions")).unwrap();
        assert_eq!(result, PathBuf::from("data/sessions"));
    }

    #[test]
    fn test_expand_tilde_tilde_user_not_expanded() {
        let result = expand_tilde(Path::new("~otheruser/path")).unwrap();
        assert_eq!(result, PathBuf::from("~otheruser/path"));
    }

    #[test]
    fn test_expand_tilde_empty_path() {
        let result = expand_tilde(Path::new("")).unwrap();
        assert_eq!(result, PathBuf::from(""));
    }
}
