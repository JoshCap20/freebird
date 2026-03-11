//! Mtime-based tag cache for avoiding redundant tree-sitter parsing.
//!
//! Caches parse results per file, keyed by `(path, mtime)`. Modified files
//! are automatically invalidated. The cache is ephemeral — rebuilt each
//! session from disk if available, gracefully degrading on corruption.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use serde::{Deserialize, Serialize};

use super::graph::Tag;

/// Directory name for cache files inside the sandbox root.
const CACHE_DIR: &str = ".freebird_cache";

/// Cache file name.
const CACHE_FILE: &str = "repo_map_tags.json";

/// Cached parse results for a single file.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct CacheEntry {
    /// Modification time at the point of parsing (seconds + nanos).
    mtime_secs: u64,
    mtime_nanos: u32,
    /// Extracted tags for this file.
    tags: Vec<SerializableTag>,
}

/// Serializable version of `Tag` (avoids exposing internal types to serde).
#[derive(Debug, Clone, Serialize, Deserialize)]
struct SerializableTag {
    name: String,
    line: usize,
    is_definition: bool,
    kind: String,
}

impl SerializableTag {
    fn from_tag(tag: &Tag) -> Self {
        Self {
            name: tag.name.clone(),
            line: tag.line,
            is_definition: tag.is_definition,
            kind: format!("{:?}", tag.kind),
        }
    }

    fn to_tag(&self, file: &Path) -> Tag {
        Tag {
            name: self.name.clone(),
            file: file.to_path_buf(),
            line: self.line,
            is_definition: self.is_definition,
            kind: parse_tag_kind(&self.kind),
        }
    }
}

fn parse_tag_kind(s: &str) -> super::graph::TagKind {
    use super::graph::TagKind;
    match s {
        "Function" => TagKind::Function,
        "Struct" => TagKind::Struct,
        "Enum" => TagKind::Enum,
        "Trait" => TagKind::Trait,
        "TypeAlias" => TagKind::TypeAlias,
        "Const" => TagKind::Const,
        "Static" => TagKind::Static,
        "Macro" => TagKind::Macro,
        "Module" => TagKind::Module,
        _ => TagKind::Reference,
    }
}

/// Per-file tag cache with mtime-based invalidation.
pub(super) struct TagCache {
    entries: HashMap<PathBuf, CacheEntry>,
}

impl TagCache {
    /// Create an empty cache.
    #[must_use]
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
        }
    }

    /// Load cache from disk. Returns an empty cache on any error.
    #[must_use]
    pub fn load(sandbox_root: &Path) -> Self {
        let cache_path = sandbox_root.join(CACHE_DIR).join(CACHE_FILE);
        let Ok(data) = std::fs::read_to_string(&cache_path) else {
            return Self::new();
        };
        serde_json::from_str::<HashMap<PathBuf, CacheEntry>>(&data)
            .map_or_else(|_| Self::new(), |entries| Self { entries })
    }

    /// Save cache to disk. Logs a warning on error, never fails.
    pub fn save(&self, sandbox_root: &Path) {
        let cache_dir = sandbox_root.join(CACHE_DIR);
        if std::fs::create_dir_all(&cache_dir).is_err() {
            tracing::warn!("failed to create cache directory");
            return;
        }
        let cache_path = cache_dir.join(CACHE_FILE);
        match serde_json::to_string(&self.entries) {
            Ok(json) => {
                if let Err(e) = std::fs::write(&cache_path, json) {
                    tracing::warn!(%e, "failed to write tag cache");
                }
            }
            Err(e) => {
                tracing::warn!(%e, "failed to serialize tag cache");
            }
        }
    }

    /// Get cached tags for a file if the cache entry is still fresh.
    ///
    /// Returns `None` if the file is not cached or has been modified since caching.
    pub fn get(&self, file: &Path) -> Option<Vec<Tag>> {
        let entry = self.entries.get(file)?;
        let mtime = std::fs::metadata(file).ok()?.modified().ok()?;
        let (secs, nanos) = system_time_to_parts(mtime);
        if secs == entry.mtime_secs && nanos == entry.mtime_nanos {
            Some(entry.tags.iter().map(|t| t.to_tag(file)).collect())
        } else {
            None
        }
    }

    /// Insert or update the cache for a file.
    pub fn insert(&mut self, file: PathBuf, mtime: SystemTime, tags: &[Tag]) {
        let (secs, nanos) = system_time_to_parts(mtime);
        let serializable: Vec<SerializableTag> =
            tags.iter().map(SerializableTag::from_tag).collect();
        self.entries.insert(
            file,
            CacheEntry {
                mtime_secs: secs,
                mtime_nanos: nanos,
                tags: serializable,
            },
        );
    }

    /// Number of cached files (test utility).
    #[cfg(test)]
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }
}

/// Convert `SystemTime` to `(seconds_since_epoch, nanoseconds)`.
fn system_time_to_parts(t: SystemTime) -> (u64, u32) {
    t.duration_since(SystemTime::UNIX_EPOCH)
        .map_or((0, 0), |d| (d.as_secs(), d.subsec_nanos()))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::indexing_slicing)]
mod tests {
    use super::super::graph::TagKind;
    use super::*;
    use tempfile::TempDir;

    fn make_tag(name: &str, file: &Path, line: usize, is_def: bool) -> Tag {
        Tag {
            name: name.to_owned(),
            file: file.to_path_buf(),
            line,
            is_definition: is_def,
            kind: if is_def {
                TagKind::Function
            } else {
                TagKind::Reference
            },
        }
    }

    #[test]
    fn test_cache_miss_returns_none() {
        let cache = TagCache::new();
        assert!(cache.get(Path::new("/nonexistent.rs")).is_none());
    }

    #[test]
    fn test_cache_hit_returns_data() {
        let tmp = TempDir::new().unwrap();
        let file = tmp.path().join("test.rs");
        std::fs::write(&file, "fn hello() {}").unwrap();
        let mtime = std::fs::metadata(&file).unwrap().modified().unwrap();

        let mut cache = TagCache::new();
        let tags = vec![make_tag("hello", &file, 1, true)];
        cache.insert(file.clone(), mtime, &tags);

        let result = cache.get(&file);
        assert!(result.is_some());
        let cached = result.unwrap();
        assert_eq!(cached.len(), 1);
        assert_eq!(cached[0].name, "hello");
        assert!(cached[0].is_definition);
    }

    #[test]
    fn test_modified_file_invalidates_cache() {
        let tmp = TempDir::new().unwrap();
        let file = tmp.path().join("test.rs");
        std::fs::write(&file, "fn old() {}").unwrap();
        let old_mtime = std::fs::metadata(&file).unwrap().modified().unwrap();

        let mut cache = TagCache::new();
        cache.insert(file.clone(), old_mtime, &[make_tag("old", &file, 1, true)]);

        // Modify the file (change mtime).
        std::thread::sleep(std::time::Duration::from_millis(50));
        std::fs::write(&file, "fn new() {}").unwrap();

        // Cache should miss now.
        assert!(cache.get(&file).is_none());
    }

    #[test]
    fn test_load_save_roundtrip() {
        let tmp = TempDir::new().unwrap();
        let sandbox = tmp.path();
        let file = sandbox.join("lib.rs");
        std::fs::write(&file, "struct Config;").unwrap();
        let mtime = std::fs::metadata(&file).unwrap().modified().unwrap();

        let mut cache = TagCache::new();
        cache.insert(file.clone(), mtime, &[make_tag("Config", &file, 1, true)]);
        cache.save(sandbox);

        let loaded = TagCache::load(sandbox);
        let result = loaded.get(&file);
        assert!(result.is_some());
        assert_eq!(result.unwrap()[0].name, "Config");
    }

    #[test]
    fn test_corrupt_cache_fallback() {
        let tmp = TempDir::new().unwrap();
        let sandbox = tmp.path();
        let cache_dir = sandbox.join(CACHE_DIR);
        std::fs::create_dir_all(&cache_dir).unwrap();
        std::fs::write(cache_dir.join(CACHE_FILE), "not valid json!!!").unwrap();

        let cache = TagCache::load(sandbox);
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn test_load_missing_directory() {
        let cache = TagCache::load(Path::new("/nonexistent/path"));
        assert_eq!(cache.len(), 0);
    }
}
