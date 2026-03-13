//! Auto-detecting change planner tool.
//!
//! Uses `repo_map`'s tree-sitter AST analysis and reference graph to infer
//! dependencies, change kinds, and crate kinds, then delegates to
//! `freebird_types::planner::plan_changes` for topological sorting.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use async_trait::async_trait;

use freebird_security::safe_types::SafeFilePath;
use freebird_security::taint::Tainted;
use freebird_traits::tool::{
    Capability, RiskLevel, SideEffects, Tool, ToolContext, ToolError, ToolInfo, ToolOutcome,
    ToolOutput,
};
use freebird_types::planner::{ChangeKind, CrateKind, PlannedChange};

use crate::common::{extract_optional_bool, extract_optional_usize};
use crate::repo_map::cache::TagCache;
use crate::repo_map::graph::{ReferenceGraph, Tag, TagKind, extract_rust_tags};

// ── Auto-detection helpers ──────────────────────────────────────────

/// Infer `ChangeKind` from the dominant tag type in a file.
///
/// Priority: Trait > Struct/Enum > Impl > Test > Consumer.
fn infer_change_kind(tags: &[Tag], file_path: &Path) -> ChangeKind {
    let path_str = file_path.to_string_lossy();
    if path_str.contains("/tests/") || path_str.contains("/test_") {
        return ChangeKind::Test;
    }

    let mut has_trait = false;
    let mut has_type = false;
    let mut has_impl = false;

    for tag in tags {
        if !tag.is_definition {
            continue;
        }
        match tag.kind {
            TagKind::Trait => has_trait = true,
            TagKind::Struct | TagKind::Enum | TagKind::TypeAlias => has_type = true,
            _ => {}
        }
    }

    // Files with function definitions but no trait/type are classified as
    // Implementation rather than Consumer. This is intentional: standalone free
    // functions are edited *after* types but *before* tests in the dependency
    // order, which matches the typical Rust authoring flow.
    if !has_trait && !has_type {
        for tag in tags {
            if tag.is_definition && matches!(tag.kind, TagKind::Function) {
                has_impl = true;
                break;
            }
        }
    }

    if has_trait {
        ChangeKind::TraitDefinition
    } else if has_type {
        ChangeKind::TypeDefinition
    } else if has_impl {
        ChangeKind::Implementation
    } else {
        ChangeKind::Consumer
    }
}

/// Infer `CrateKind` from file path.
fn infer_crate_kind(path: &Path) -> CrateKind {
    let path_str = path.to_string_lossy();
    if path_str.ends_with("/main.rs") || path_str.contains("/bin/") {
        CrateKind::Binary
    } else {
        CrateKind::Library
    }
}

/// Merge explicit and inferred dependencies, deduplicate, and remove self-references.
fn resolve_dependencies(
    explicit: Option<Vec<usize>>,
    inferred: Vec<usize>,
    self_id: usize,
) -> Vec<usize> {
    let mut deps = if let Some(mut explicit) = explicit {
        for dep in inferred {
            if !explicit.contains(&dep) {
                explicit.push(dep);
            }
        }
        explicit
    } else {
        inferred
    };
    deps.retain(|&d| d != self_id);
    deps
}

/// Infer `depends_on` for a file from the reference graph.
///
/// If file[i] references symbols defined in file[j], and file[j] is
/// also in the change set, then file[i] depends on file[j].
fn infer_dependencies(
    file_path: &Path,
    graph: &ReferenceGraph,
    file_id_map: &HashMap<PathBuf, usize>,
) -> Vec<usize> {
    // Find this file's index in the reference graph
    let graph_files = graph.index_to_file();
    let graph_idx = graph_files.iter().position(|f| f == file_path);

    let Some(idx) = graph_idx else {
        return Vec::new();
    };

    let adjacency = graph.adjacency();
    let Some(refs) = adjacency.get(idx) else {
        return Vec::new();
    };

    // Map referenced graph indices back to change IDs
    let mut deps = Vec::new();
    for &ref_idx in refs {
        if let Some(ref_path) = graph_files.get(ref_idx) {
            if let Some(&change_id) = file_id_map.get(ref_path) {
                deps.push(change_id);
            }
        }
    }
    deps
}

/// Parse and resolve file paths, run tree-sitter, build tags and graph.
fn analyze_files(
    sandbox_root: &Path,
    file_paths: &[PathBuf],
) -> (HashMap<PathBuf, Vec<Tag>>, ReferenceGraph) {
    let mut parser = tree_sitter::Parser::new();
    if let Err(e) = parser.set_language(&tree_sitter_rust::LANGUAGE.into()) {
        tracing::warn!(%e, "failed to set tree-sitter language; AST analysis will be skipped");
    }

    let mut cache = TagCache::load(sandbox_root);
    let mut tags_by_file: HashMap<PathBuf, Vec<Tag>> = HashMap::new();

    for path in file_paths {
        // Paths are already resolved to absolute by SafeFilePath validation
        if !path.to_string_lossy().ends_with(".rs") {
            tracing::debug!(
                ?path,
                "non-Rust file; AST-based dependency inference will not apply"
            );
        }

        // Try cache first
        if let Some(cached) = cache.get(path) {
            tags_by_file.insert(path.clone(), cached);
            continue;
        }

        // Parse the file
        let Ok(source) = std::fs::read(path) else {
            continue;
        };

        let Some(tree) = parser.parse(&source, None) else {
            continue;
        };

        let tags = extract_rust_tags(&tree, &source, path);

        // Update cache
        if let Ok(meta) = std::fs::metadata(path) {
            if let Ok(mtime) = meta.modified() {
                cache.insert(path.clone(), mtime, &tags);
            }
        }

        tags_by_file.insert(path.clone(), tags);
    }

    cache.save(sandbox_root);
    let graph = ReferenceGraph::build(&tags_by_file);
    (tags_by_file, graph)
}

// ── Input parsing ───────────────────────────────────────────────────

/// A single change entry from the tool input (before auto-detection).
#[derive(Debug)]
struct InputChange {
    id: usize,
    file_path: PathBuf,
    description: String,
    depends_on: Option<Vec<usize>>,
    change_kind: Option<ChangeKind>,
    crate_kind: Option<CrateKind>,
}

fn parse_input(
    input: &serde_json::Value,
    sandbox_root: &Path,
    allowed_dirs: &[PathBuf],
) -> Result<(Vec<InputChange>, bool), ToolError> {
    let auto_detect = extract_optional_bool(input, "auto_detect").unwrap_or(true);

    let changes_arr = input
        .get("changes")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| ToolError::InvalidInput {
            tool: PlanEditsTool::NAME.into(),
            reason: "missing or invalid 'changes' array".into(),
        })?;

    let mut changes = Vec::with_capacity(changes_arr.len());

    for (i, entry) in changes_arr.iter().enumerate() {
        let id = extract_optional_usize(entry, "id").ok_or_else(|| ToolError::InvalidInput {
            tool: PlanEditsTool::NAME.into(),
            reason: format!("change[{i}]: missing or invalid 'id'"),
        })?;

        let file_path_str = entry
            .get("file_path")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| ToolError::InvalidInput {
                tool: PlanEditsTool::NAME.into(),
                reason: format!("change[{i}]: missing or invalid 'file_path'"),
            })?;

        // Validate path through taint boundary — rejects traversal, symlinks, etc.
        // Uses `for_creation` variant because the planner orders *future* edits;
        // the target files may not exist yet.
        let tainted_path = Tainted::new(file_path_str);
        let safe_path = SafeFilePath::from_tainted_for_creation_multi_root(
            &tainted_path,
            sandbox_root,
            allowed_dirs,
        )
        .map_err(|e| ToolError::InvalidInput {
            tool: PlanEditsTool::NAME.into(),
            reason: format!("change[{i}]: {e}"),
        })?;

        let description = entry
            .get("description")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("")
            .to_owned();

        let depends_on = entry.get("depends_on").and_then(|v| {
            v.as_array().map(|arr| {
                arr.iter()
                    .filter_map(|x| x.as_u64().and_then(|n| usize::try_from(n).ok()))
                    .collect()
            })
        });

        let change_kind = entry
            .get("change_kind")
            .and_then(serde_json::Value::as_str)
            .and_then(|s| serde_json::from_value(serde_json::Value::String(s.to_owned())).ok());

        let crate_kind = entry
            .get("crate_kind")
            .and_then(serde_json::Value::as_str)
            .and_then(|s| serde_json::from_value(serde_json::Value::String(s.to_owned())).ok());

        // Validate: if auto_detect is false, all fields are required
        if !auto_detect {
            if depends_on.is_none() {
                return Err(ToolError::InvalidInput {
                    tool: PlanEditsTool::NAME.into(),
                    reason: format!("change[{i}]: 'depends_on' required when auto_detect=false"),
                });
            }
            if change_kind.is_none() {
                return Err(ToolError::InvalidInput {
                    tool: PlanEditsTool::NAME.into(),
                    reason: format!("change[{i}]: 'change_kind' required when auto_detect=false"),
                });
            }
            if crate_kind.is_none() {
                return Err(ToolError::InvalidInput {
                    tool: PlanEditsTool::NAME.into(),
                    reason: format!("change[{i}]: 'crate_kind' required when auto_detect=false"),
                });
            }
        }

        changes.push(InputChange {
            id,
            file_path: safe_path.as_path().to_path_buf(),
            description,
            depends_on,
            change_kind,
            crate_kind,
        });
    }

    Ok((changes, auto_detect))
}

// ── Tool implementation ─────────────────────────────────────────────

struct PlanEditsTool {
    info: ToolInfo,
}

impl PlanEditsTool {
    const NAME: &'static str = "plan_edits";

    fn new() -> Self {
        Self {
            info: ToolInfo {
                name: Self::NAME.into(),
                description: "Plan a dependency-aware execution order for multi-file edits. \
                    Accepts file paths with descriptions, auto-detects dependencies via AST \
                    analysis, and returns a topologically sorted plan. \
                    Supports up to 256 changes with dependency chains up to 64 levels deep."
                    .into(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "changes": {
                            "type": "array",
                            "description": "Files to change with optional dependency/kind overrides",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "id": { "type": "integer", "description": "Unique ID for this change" },
                                    "file_path": { "type": "string", "description": "Path to the file to change" },
                                    "description": { "type": "string", "description": "What will change in this file" },
                                    "depends_on": {
                                        "type": "array",
                                        "items": { "type": "integer" },
                                        "description": "IDs this change depends on (auto-detected if omitted)"
                                    },
                                    "change_kind": {
                                        "type": "string",
                                        "enum": ["trait_definition", "type_definition", "implementation", "consumer", "test"],
                                        "description": "Classification of the change (auto-detected if omitted)"
                                    },
                                    "crate_kind": {
                                        "type": "string",
                                        "enum": ["library", "binary"],
                                        "description": "Whether the crate is a library or binary (auto-detected if omitted)"
                                    }
                                },
                                "required": ["id", "file_path", "description"]
                            }
                        },
                        "auto_detect": {
                            "type": "boolean",
                            "description": "Auto-detect dependencies and kinds from AST (default: true)"
                        }
                    },
                    "required": ["changes"]
                }),
                required_capability: Capability::FileRead,
                risk_level: RiskLevel::Low,
                side_effects: SideEffects::None,
            },
        }
    }
}

#[async_trait]
impl Tool for PlanEditsTool {
    fn info(&self) -> &ToolInfo {
        &self.info
    }

    async fn execute(
        &self,
        input: serde_json::Value,
        context: &ToolContext<'_>,
    ) -> Result<ToolOutput, ToolError> {
        let (input_changes, auto_detect) =
            parse_input(&input, context.sandbox_root, context.allowed_directories)?;

        // Paths are already resolved to absolute by SafeFilePath validation
        let file_paths: Vec<PathBuf> = input_changes.iter().map(|c| c.file_path.clone()).collect();

        // Auto-detection: parse files, build reference graph
        let (tags_by_file, graph) = if auto_detect {
            analyze_files(context.sandbox_root, &file_paths)
        } else {
            (HashMap::new(), ReferenceGraph::build(&HashMap::new()))
        };

        // Build file_path → change_id mapping for dependency inference
        let file_id_map: HashMap<PathBuf, usize> = input_changes
            .iter()
            .map(|c| (c.file_path.clone(), c.id))
            .collect();

        // Resolve each change: merge explicit + inferred values
        let mut planned: Vec<PlannedChange> = Vec::with_capacity(input_changes.len());
        for ic in input_changes {
            // Infer change_kind from tags if not explicit
            let change_kind = ic.change_kind.unwrap_or_else(|| {
                tags_by_file
                    .get(&ic.file_path)
                    .map_or(ChangeKind::Consumer, |tags| {
                        infer_change_kind(tags, &ic.file_path)
                    })
            });

            // Infer crate_kind from path if not explicit
            let crate_kind = ic
                .crate_kind
                .unwrap_or_else(|| infer_crate_kind(&ic.file_path));

            // Infer + merge dependencies
            let depends_on = if auto_detect {
                let inferred = infer_dependencies(&ic.file_path, &graph, &file_id_map);
                resolve_dependencies(ic.depends_on, inferred, ic.id)
            } else {
                ic.depends_on.unwrap_or_default()
            };

            planned.push(PlannedChange {
                id: ic.id,
                file_path: ic.file_path,
                description: ic.description,
                depends_on,
                change_kind,
                crate_kind,
            });
        }

        // Run the topological sort
        match freebird_types::planner::plan_changes(planned) {
            Ok(plan) => {
                let json = serde_json::to_string_pretty(&plan).map_err(|e| {
                    ToolError::ExecutionFailed {
                        tool: Self::NAME.into(),
                        reason: format!("failed to serialize plan: {e}"),
                    }
                })?;
                Ok(ToolOutput {
                    content: json,
                    outcome: ToolOutcome::Success,
                    metadata: None,
                })
            }
            Err(e) => Ok(ToolOutput {
                content: e.to_string(),
                outcome: ToolOutcome::Error,
                metadata: None,
            }),
        }
    }
}

/// Return the `plan_edits` tool for registration.
#[must_use]
pub fn planner_tools() -> Vec<Box<dyn Tool>> {
    vec![Box::new(PlanEditsTool::new())]
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::redundant_clone,
    clippy::unnecessary_literal_unwrap
)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    // ── infer_change_kind ───────────────────────────────────────────

    fn make_tag(name: &str, kind: TagKind, is_def: bool) -> Tag {
        Tag {
            name: name.to_owned(),
            file: PathBuf::from("test.rs"),
            line: 1,
            is_definition: is_def,
            kind,
        }
    }

    #[test]
    fn test_infer_change_kind_trait_file() {
        let tags = vec![make_tag("MyTrait", TagKind::Trait, true)];
        assert_eq!(
            infer_change_kind(&tags, Path::new("src/lib.rs")),
            ChangeKind::TraitDefinition
        );
    }

    #[test]
    fn test_infer_change_kind_struct_file() {
        let tags = vec![make_tag("MyStruct", TagKind::Struct, true)];
        assert_eq!(
            infer_change_kind(&tags, Path::new("src/lib.rs")),
            ChangeKind::TypeDefinition
        );
    }

    #[test]
    fn test_infer_change_kind_impl_file() {
        let tags = vec![make_tag("do_stuff", TagKind::Function, true)];
        assert_eq!(
            infer_change_kind(&tags, Path::new("src/lib.rs")),
            ChangeKind::Implementation
        );
    }

    #[test]
    fn test_infer_change_kind_free_functions_is_implementation() {
        // Files with only free functions (no struct/trait/enum) are classified as
        // Implementation, not Consumer. This places them after types but before
        // tests in the secondary sort — matching typical Rust authoring order.
        let tags = vec![
            make_tag("helper_one", TagKind::Function, true),
            make_tag("helper_two", TagKind::Function, true),
        ];
        assert_eq!(
            infer_change_kind(&tags, Path::new("src/utils.rs")),
            ChangeKind::Implementation
        );
    }

    #[test]
    fn test_infer_change_kind_no_definitions_is_consumer() {
        // Files with no definitions at all → Consumer (lowest priority)
        let tags: Vec<Tag> = vec![];
        assert_eq!(
            infer_change_kind(&tags, Path::new("src/lib.rs")),
            ChangeKind::Consumer
        );
    }

    #[test]
    fn test_infer_change_kind_test_file() {
        let tags = vec![make_tag("test_foo", TagKind::Function, true)];
        assert_eq!(
            infer_change_kind(&tags, Path::new("src/tests/foo.rs")),
            ChangeKind::Test
        );
    }

    #[test]
    fn test_infer_change_kind_mixed_trait_impl() {
        let tags = vec![
            make_tag("MyTrait", TagKind::Trait, true),
            make_tag("do_stuff", TagKind::Function, true),
        ];
        assert_eq!(
            infer_change_kind(&tags, Path::new("src/lib.rs")),
            ChangeKind::TraitDefinition
        );
    }

    // ── infer_crate_kind ────────────────────────────────────────────

    #[test]
    fn test_infer_crate_kind_main() {
        assert_eq!(
            infer_crate_kind(Path::new("crates/my-crate/src/main.rs")),
            CrateKind::Binary
        );
    }

    #[test]
    fn test_infer_crate_kind_lib() {
        assert_eq!(
            infer_crate_kind(Path::new("crates/my-crate/src/lib.rs")),
            CrateKind::Library
        );
    }

    // ── infer_dependencies ──────────────────────────────────────────

    #[test]
    fn test_infer_depends_on_cross_file_reference() {
        // File A references a symbol defined in File B
        let file_a = PathBuf::from("/tmp/a.rs");
        let file_b = PathBuf::from("/tmp/b.rs");

        let mut tags_by_file = HashMap::new();
        tags_by_file.insert(
            file_a.clone(),
            vec![Tag {
                name: "Foo".into(),
                file: file_a.clone(),
                line: 1,
                is_definition: false,
                kind: TagKind::Reference,
            }],
        );
        tags_by_file.insert(
            file_b.clone(),
            vec![Tag {
                name: "Foo".into(),
                file: file_b.clone(),
                line: 1,
                is_definition: true,
                kind: TagKind::Struct,
            }],
        );

        let graph = ReferenceGraph::build(&tags_by_file);
        let mut file_id_map = HashMap::new();
        file_id_map.insert(file_a.clone(), 0);
        file_id_map.insert(file_b.clone(), 1);

        let deps = infer_dependencies(&file_a, &graph, &file_id_map);
        assert_eq!(deps, vec![1]); // A depends on B
    }

    #[test]
    fn test_infer_depends_on_no_self_reference() {
        let file_a = PathBuf::from("/tmp/a.rs");

        let mut tags_by_file = HashMap::new();
        tags_by_file.insert(
            file_a.clone(),
            vec![
                Tag {
                    name: "Foo".into(),
                    file: file_a.clone(),
                    line: 1,
                    is_definition: true,
                    kind: TagKind::Struct,
                },
                Tag {
                    name: "Foo".into(),
                    file: file_a.clone(),
                    line: 5,
                    is_definition: false,
                    kind: TagKind::Reference,
                },
            ],
        );

        let graph = ReferenceGraph::build(&tags_by_file);
        let mut file_id_map = HashMap::new();
        file_id_map.insert(file_a.clone(), 0);

        let deps = infer_dependencies(&file_a, &graph, &file_id_map);
        assert!(deps.is_empty()); // No self-dep
    }

    // ── Explicit overrides ──────────────────────────────────────────

    #[test]
    fn test_explicit_override_merges_with_inferred() {
        // Explicit deps should be unioned with inferred deps
        let file_a = PathBuf::from("/tmp/a.rs");
        let file_b = PathBuf::from("/tmp/b.rs");
        let file_c = PathBuf::from("/tmp/c.rs");

        let mut tags_by_file = HashMap::new();
        tags_by_file.insert(
            file_a.clone(),
            vec![Tag {
                name: "Foo".into(),
                file: file_a.clone(),
                line: 1,
                is_definition: false,
                kind: TagKind::Reference,
            }],
        );
        tags_by_file.insert(
            file_b.clone(),
            vec![Tag {
                name: "Foo".into(),
                file: file_b.clone(),
                line: 1,
                is_definition: true,
                kind: TagKind::Struct,
            }],
        );
        tags_by_file.insert(file_c.clone(), vec![]);

        let graph = ReferenceGraph::build(&tags_by_file);
        let mut file_id_map = HashMap::new();
        file_id_map.insert(file_a.clone(), 0);
        file_id_map.insert(file_b.clone(), 1);
        file_id_map.insert(file_c.clone(), 2);

        // Inferred: A depends on B. Explicit: A depends on C.
        let inferred = infer_dependencies(&file_a, &graph, &file_id_map);
        let explicit = vec![2usize];

        let mut merged = resolve_dependencies(Some(explicit), inferred, 0);
        merged.sort_unstable();
        assert_eq!(merged, vec![1, 2]); // Both B and C
    }

    #[test]
    fn test_explicit_change_kind_overrides_inferred() {
        // When explicit change_kind is provided, it wins over AST inference
        let tags = vec![make_tag("MyTrait", TagKind::Trait, true)];
        let inferred = infer_change_kind(&tags, Path::new("src/lib.rs"));
        assert_eq!(inferred, ChangeKind::TraitDefinition);

        // But if explicit is Consumer, Consumer wins
        let explicit: Option<ChangeKind> = Some(ChangeKind::Consumer);
        let result = explicit.unwrap_or(inferred);
        assert_eq!(result, ChangeKind::Consumer);
    }

    #[test]
    fn test_auto_detect_false_requires_all_fields() {
        let tmp = tempfile::TempDir::new().unwrap();
        let sandbox = tmp.path();

        let input = serde_json::json!({
            "changes": [{
                "id": 0,
                "file_path": "foo.rs",
                "description": "test"
                // Missing depends_on, change_kind, crate_kind
            }],
            "auto_detect": false
        });

        let err = parse_input(&input, sandbox, &[]).unwrap_err();
        assert!(matches!(err, ToolError::InvalidInput { .. }));
    }
}
