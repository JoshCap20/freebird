//! Reference graph construction for PageRank-ranked repo maps.
//!
//! Extracts **definitions** (functions, structs, traits, etc.) and **references**
//! (calls, type usage, imports) from tree-sitter parse trees, then builds a
//! directed graph where edges point from referencing files to defining files.
//! `PageRank` over this graph ranks files by cross-file importance.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

// ── Tag types ────────────────────────────────────────────────────

/// A tag extracted from a source file — either a definition or a reference.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(super) struct Tag {
    /// The symbol name (e.g., `"AgentRuntime"`, `"handle_message"`).
    pub name: String,
    /// Absolute path to the file containing this tag.
    pub file: PathBuf,
    /// 1-based line number where the tag occurs.
    pub line: usize,
    /// Whether this tag is a definition (`true`) or a reference (`false`).
    pub is_definition: bool,
    /// The kind of symbol.
    pub kind: TagKind,
}

/// Classification of a tagged symbol.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(super) enum TagKind {
    Function,
    Struct,
    Enum,
    Trait,
    TypeAlias,
    Const,
    Static,
    Macro,
    Module,
    /// A reference where we cannot determine the specific kind.
    Reference,
}

impl TagKind {
    /// Stable string representation for serialization.
    ///
    /// Must stay in sync with `cache::parse_tag_kind`.
    pub(super) const fn as_str(self) -> &'static str {
        match self {
            Self::Function => "Function",
            Self::Struct => "Struct",
            Self::Enum => "Enum",
            Self::Trait => "Trait",
            Self::TypeAlias => "TypeAlias",
            Self::Const => "Const",
            Self::Static => "Static",
            Self::Macro => "Macro",
            Self::Module => "Module",
            Self::Reference => "Reference",
        }
    }
}

// ── Reference graph ──────────────────────────────────────────────

/// Directed reference graph where nodes are files and edges point from
/// a file that **references** a symbol to the file that **defines** it.
///
/// Self-edges (same file) are excluded — they carry no cross-file signal.
pub(super) struct ReferenceGraph {
    index_to_file: Vec<PathBuf>,
    /// `adjacency[i]` = set of file indices that file `i` references.
    adjacency: Vec<HashSet<usize>>,
}

impl ReferenceGraph {
    /// Build a reference graph from tags grouped by file.
    ///
    /// 1. Index all files.
    /// 2. Build a definitions map: name to `[(file_idx, line)]`.
    /// 3. For each reference tag, look up the name in definitions.
    ///    If found in a different file, add edge `ref_file` to `def_file`.
    pub fn build(tags_by_file: &HashMap<PathBuf, Vec<Tag>>) -> Self {
        // Step 1: index files deterministically (sorted for stable output).
        let mut sorted_files: Vec<PathBuf> = tags_by_file.keys().cloned().collect();
        sorted_files.sort();

        let mut file_to_index = HashMap::with_capacity(sorted_files.len());
        for (idx, file) in sorted_files.iter().enumerate() {
            file_to_index.insert(file.clone(), idx);
        }

        let num_files = sorted_files.len();
        let mut adjacency = vec![HashSet::new(); num_files];

        // Step 2: build definitions map.
        let mut definitions: HashMap<String, Vec<(usize, usize)>> = HashMap::new();
        for (file, tags) in tags_by_file {
            let Some(&file_idx) = file_to_index.get(file) else {
                continue;
            };
            for tag in tags {
                if tag.is_definition {
                    definitions
                        .entry(tag.name.clone())
                        .or_default()
                        .push((file_idx, tag.line));
                }
            }
        }

        // Step 3: resolve references → edges (drops moved `tags_by_file` at block end).
        for (file, tags) in tags_by_file {
            let Some(&ref_idx) = file_to_index.get(file) else {
                continue;
            };
            for tag in tags {
                if tag.is_definition {
                    continue;
                }
                if let Some(def_locs) = definitions.get(&tag.name) {
                    for &(def_idx, _line) in def_locs {
                        // Skip self-references.
                        if def_idx != ref_idx {
                            if let Some(set) = adjacency.get_mut(ref_idx) {
                                set.insert(def_idx);
                            }
                        }
                    }
                }
            }
        }

        Self {
            index_to_file: sorted_files,
            adjacency,
        }
    }

    /// Number of files (nodes) in the graph.
    #[must_use]
    pub fn num_files(&self) -> usize {
        self.index_to_file.len()
    }

    /// Borrow the adjacency list (for `PageRank`).
    #[must_use]
    pub fn adjacency(&self) -> &[HashSet<usize>] {
        &self.adjacency
    }

    /// Borrow the index-to-file mapping.
    #[must_use]
    pub fn index_to_file(&self) -> &[PathBuf] {
        &self.index_to_file
    }
}

// ── Rust tag extraction ──────────────────────────────────────────

/// Extract both definition and reference tags from a tree-sitter parse tree.
///
/// Walks the entire AST, emitting:
/// - **Definition tags** for top-level items (functions, structs, enums, traits, etc.)
/// - **Reference tags** for identifiers in call expressions, type usage, use
///   declarations, and macro invocations.
pub(super) fn extract_rust_tags(tree: &tree_sitter::Tree, source: &[u8], file: &Path) -> Vec<Tag> {
    let mut tags = Vec::new();
    let mut cursor = tree.walk();
    walk_node_recursive(&mut cursor, source, file, &mut tags);
    tags
}

/// Map a tree-sitter node kind to a definition `TagKind`, if it is a definition.
fn definition_kind(kind_str: &str) -> Option<TagKind> {
    match kind_str {
        "function_item" | "function_signature_item" => Some(TagKind::Function),
        "struct_item" => Some(TagKind::Struct),
        "enum_item" => Some(TagKind::Enum),
        "trait_item" => Some(TagKind::Trait),
        "type_item" => Some(TagKind::TypeAlias),
        "const_item" => Some(TagKind::Const),
        "static_item" => Some(TagKind::Static),
        "mod_item" => Some(TagKind::Module),
        _ => None,
    }
}

/// Emit a definition tag if the node has a `name` child.
fn try_push_definition(
    node: &tree_sitter::Node<'_>,
    source: &[u8],
    file: &Path,
    kind: TagKind,
    tags: &mut Vec<Tag>,
) {
    if let Some(name) = extract_child_name(node, source) {
        tags.push(Tag {
            name,
            file: file.to_path_buf(),
            line: node.start_position().row + 1,
            is_definition: true,
            kind,
        });
    }
}

/// Emit a reference tag with the given name.
fn push_reference(name: String, node: &tree_sitter::Node<'_>, file: &Path, tags: &mut Vec<Tag>) {
    tags.push(Tag {
        name,
        file: file.to_path_buf(),
        line: node.start_position().row + 1,
        is_definition: false,
        kind: TagKind::Reference,
    });
}

/// Recursive AST walker that collects definition and reference tags.
fn walk_node_recursive(
    cursor: &mut tree_sitter::TreeCursor<'_>,
    source: &[u8],
    file: &Path,
    tags: &mut Vec<Tag>,
) {
    let node = cursor.node();
    let kind_str = node.kind();

    // Definitions — most item types follow the same pattern.
    if let Some(tag_kind) = definition_kind(kind_str) {
        try_push_definition(&node, source, file, tag_kind, tags);
    } else if kind_str == "macro_definition" {
        if let Some(name) = extract_macro_name(&node, source) {
            tags.push(Tag {
                name,
                file: file.to_path_buf(),
                line: node.start_position().row + 1,
                is_definition: true,
                kind: TagKind::Macro,
            });
        }
    } else if kind_str == "use_declaration" {
        // Handle use declarations entirely here — don't recurse further.
        extract_use_names(&node, source, file, tags);
        return;
    } else {
        collect_reference_tags(&node, kind_str, source, file, tags);
    }

    // Recurse into children.
    if cursor.goto_first_child() {
        loop {
            walk_node_recursive(cursor, source, file, tags);
            if !cursor.goto_next_sibling() {
                break;
            }
        }
        cursor.goto_parent();
    }
}

/// Extract reference tags from non-definition nodes.
fn collect_reference_tags(
    node: &tree_sitter::Node<'_>,
    kind_str: &str,
    source: &[u8],
    file: &Path,
    tags: &mut Vec<Tag>,
) {
    match kind_str {
        "call_expression" => {
            if let Some(func_node) = node.child_by_field_name("function") {
                if let Some(name) = extract_reference_name(&func_node, source) {
                    push_reference(name, node, file, tags);
                }
            }
        }
        "type_identifier" if !is_definition_name_node(node) => {
            let name = slice_source(source, node.start_byte(), node.end_byte());
            if !name.is_empty() {
                push_reference(name.to_owned(), node, file, tags);
            }
        }
        "macro_invocation" => {
            let mut child_cursor = node.walk();
            for child in node.children(&mut child_cursor) {
                if child.kind() == "identifier" || child.kind() == "scoped_identifier" {
                    if let Some(name) = extract_reference_name(&child, source) {
                        push_reference(name, node, file, tags);
                    }
                    break;
                }
            }
        }
        _ => {}
    }
}

/// Extract the `name` field child from an AST node (works for most items).
fn extract_child_name(node: &tree_sitter::Node<'_>, source: &[u8]) -> Option<String> {
    let name_node = node.child_by_field_name("name")?;
    let name = slice_source(source, name_node.start_byte(), name_node.end_byte());
    if name.is_empty() {
        None
    } else {
        Some(name.to_owned())
    }
}

/// Extract name from a macro definition (uses first `identifier` child).
fn extract_macro_name(node: &tree_sitter::Node<'_>, source: &[u8]) -> Option<String> {
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        if child.kind() == "identifier" {
            let name = slice_source(source, child.start_byte(), child.end_byte());
            if !name.is_empty() {
                return Some(name.to_owned());
            }
        }
    }
    None
}

/// Extract a reference name from a node that may be an identifier or scoped path.
///
/// For scoped identifiers like `foo::bar::Baz`, returns just the final segment
/// (`Baz`) since that's what matches definition names.
fn extract_reference_name(node: &tree_sitter::Node<'_>, source: &[u8]) -> Option<String> {
    /// Return `Some(name)` if the slice is non-empty, else `None`.
    fn non_empty_name(source: &[u8], n: &tree_sitter::Node<'_>) -> Option<String> {
        let name = slice_source(source, n.start_byte(), n.end_byte());
        (!name.is_empty()).then(|| name.to_owned())
    }

    match node.kind() {
        "identifier" | "field_identifier" => non_empty_name(source, node),
        "scoped_identifier" | "scoped_type_identifier" => node
            .child_by_field_name("name")
            .and_then(|n| non_empty_name(source, &n)),
        "field_expression" => node
            .child_by_field_name("field")
            .and_then(|n| non_empty_name(source, &n)),
        _ => None,
    }
}

/// Check whether a `type_identifier` node is the name child of a definition.
fn is_definition_name_node(node: &tree_sitter::Node<'_>) -> bool {
    if let Some(parent) = node.parent() {
        let parent_kind = parent.kind();
        let is_def_kind = matches!(
            parent_kind,
            "struct_item" | "enum_item" | "trait_item" | "type_item"
        );
        if is_def_kind {
            // Check this node is the `name` field of the parent.
            if let Some(name_node) = parent.child_by_field_name("name") {
                return name_node.id() == node.id();
            }
        }
    }
    false
}

/// Extract imported names from a `use_declaration` node.
///
/// Handles simple uses (`use foo::Bar;`), grouped uses (`use foo::{Bar, Baz};`),
/// and aliased uses (`use foo::Bar as Renamed;`).
fn extract_use_names(
    node: &tree_sitter::Node<'_>,
    source: &[u8],
    file: &Path,
    tags: &mut Vec<Tag>,
) {
    let mut cursor = node.walk();
    collect_use_identifiers(&mut cursor, source, file, tags);
}

/// Recursively collect identifiers from use tree nodes.
fn collect_use_identifiers(
    cursor: &mut tree_sitter::TreeCursor<'_>,
    source: &[u8],
    file: &Path,
    tags: &mut Vec<Tag>,
) {
    let node = cursor.node();

    match node.kind() {
        "identifier" => {
            let name = slice_source(source, node.start_byte(), node.end_byte());
            // Skip single-char lowercase identifiers (likely module segments like `std`).
            if name.len() > 1 || name.chars().next().is_some_and(char::is_uppercase) {
                tags.push(Tag {
                    name: name.to_owned(),
                    file: file.to_path_buf(),
                    line: node.start_position().row + 1,
                    is_definition: false,
                    kind: TagKind::Reference,
                });
            }
        }
        "use_as_clause" => {
            // `Bar as Renamed` — the original name is what we reference.
            let mut child_cursor = node.walk();
            if let Some(first_child) = node.children(&mut child_cursor).next() {
                if first_child.kind() == "identifier" {
                    let name =
                        slice_source(source, first_child.start_byte(), first_child.end_byte());
                    if !name.is_empty() {
                        tags.push(Tag {
                            name: name.to_owned(),
                            file: file.to_path_buf(),
                            line: node.start_position().row + 1,
                            is_definition: false,
                            kind: TagKind::Reference,
                        });
                    }
                }
            }
            return; // Don't recurse further.
        }
        "scoped_identifier" | "scoped_use_list" | "use_list" | "use_declaration"
        | "use_wildcard" => {
            // Recurse into children.
        }
        _ => {
            // Don't recurse into unknown nodes.
            return;
        }
    }

    if cursor.goto_first_child() {
        loop {
            collect_use_identifiers(cursor, source, file, tags);
            if !cursor.goto_next_sibling() {
                break;
            }
        }
        cursor.goto_parent();
    }
}

/// Safely extract a substring from source bytes as UTF-8.
fn slice_source(source: &[u8], start: usize, end: usize) -> &str {
    source
        .get(start..end)
        .and_then(|bytes| std::str::from_utf8(bytes).ok())
        .unwrap_or("")
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::indexing_slicing)]
mod tests {
    use super::*;

    /// Build a reverse index from `index_to_file()` for test assertions.
    fn file_index(graph: &ReferenceGraph) -> HashMap<PathBuf, usize> {
        graph
            .index_to_file()
            .iter()
            .enumerate()
            .map(|(i, p)| (p.clone(), i))
            .collect()
    }

    fn parse_rust(source: &str) -> tree_sitter::Tree {
        let mut parser = tree_sitter::Parser::new();
        parser
            .set_language(&tree_sitter_rust::LANGUAGE.into())
            .unwrap();
        parser.parse(source.as_bytes(), None).unwrap()
    }

    // -- extract_rust_tags --

    #[test]
    fn test_extract_tags_function_definition() {
        let src = "pub fn hello(x: i32) -> bool { true }";
        let tree = parse_rust(src);
        let tags = extract_rust_tags(&tree, src.as_bytes(), Path::new("test.rs"));

        let defs: Vec<_> = tags.iter().filter(|t| t.is_definition).collect();
        assert_eq!(defs.len(), 1);
        assert_eq!(defs[0].name, "hello");
        assert_eq!(defs[0].kind, TagKind::Function);
        assert_eq!(defs[0].line, 1);
    }

    #[test]
    fn test_extract_tags_struct_definition() {
        let src = "pub struct Config { name: String }";
        let tree = parse_rust(src);
        let tags = extract_rust_tags(&tree, src.as_bytes(), Path::new("test.rs"));

        let defs: Vec<_> = tags.iter().filter(|t| t.is_definition).collect();
        assert_eq!(defs.len(), 1);
        assert_eq!(defs[0].name, "Config");
        assert_eq!(defs[0].kind, TagKind::Struct);
    }

    #[test]
    fn test_extract_tags_enum_definition() {
        let src = "enum Color { Red, Blue }";
        let tree = parse_rust(src);
        let tags = extract_rust_tags(&tree, src.as_bytes(), Path::new("test.rs"));

        let defs: Vec<_> = tags.iter().filter(|t| t.is_definition).collect();
        assert_eq!(defs.len(), 1);
        assert_eq!(defs[0].name, "Color");
        assert_eq!(defs[0].kind, TagKind::Enum);
    }

    #[test]
    fn test_extract_tags_trait_definition() {
        let src = "pub trait Handler { fn handle(&self); }";
        let tree = parse_rust(src);
        let tags = extract_rust_tags(&tree, src.as_bytes(), Path::new("test.rs"));

        let defs: Vec<_> = tags.iter().filter(|t| t.is_definition).collect();
        // Trait itself + `handle` method signature inside it.
        assert!(!defs.is_empty());
        let trait_def = defs.iter().find(|t| t.kind == TagKind::Trait).unwrap();
        assert_eq!(trait_def.name, "Handler");
    }

    #[test]
    fn test_extract_tags_const_and_static() {
        let src = "const MAX: usize = 100;\nstatic COUNTER: i32 = 0;";
        let tree = parse_rust(src);
        let tags = extract_rust_tags(&tree, src.as_bytes(), Path::new("test.rs"));

        let defs: Vec<_> = tags.iter().filter(|t| t.is_definition).collect();
        assert_eq!(defs.len(), 2);
        assert_eq!(defs[0].name, "MAX");
        assert_eq!(defs[0].kind, TagKind::Const);
        assert_eq!(defs[1].name, "COUNTER");
        assert_eq!(defs[1].kind, TagKind::Static);
    }

    #[test]
    fn test_extract_tags_type_alias() {
        let src = "type Result<T> = std::result::Result<T, Error>;";
        let tree = parse_rust(src);
        let tags = extract_rust_tags(&tree, src.as_bytes(), Path::new("test.rs"));

        let defs: Vec<_> = tags.iter().filter(|t| t.is_definition).collect();
        assert_eq!(defs.len(), 1);
        assert_eq!(defs[0].name, "Result");
        assert_eq!(defs[0].kind, TagKind::TypeAlias);
    }

    #[test]
    fn test_extract_tags_macro_definition() {
        let src = "macro_rules! my_macro { () => {} }";
        let tree = parse_rust(src);
        let tags = extract_rust_tags(&tree, src.as_bytes(), Path::new("test.rs"));

        let defs: Vec<_> = tags.iter().filter(|t| t.is_definition).collect();
        assert_eq!(defs.len(), 1);
        assert_eq!(defs[0].name, "my_macro");
        assert_eq!(defs[0].kind, TagKind::Macro);
    }

    #[test]
    fn test_extract_tags_module_definition() {
        let src = "mod helpers;";
        let tree = parse_rust(src);
        let tags = extract_rust_tags(&tree, src.as_bytes(), Path::new("test.rs"));

        let defs: Vec<_> = tags.iter().filter(|t| t.is_definition).collect();
        assert_eq!(defs.len(), 1);
        assert_eq!(defs[0].name, "helpers");
        assert_eq!(defs[0].kind, TagKind::Module);
    }

    #[test]
    fn test_extract_tags_function_call_reference() {
        let src = "fn main() { hello(); }";
        let tree = parse_rust(src);
        let tags = extract_rust_tags(&tree, src.as_bytes(), Path::new("test.rs"));

        let refs: Vec<_> = tags.iter().filter(|t| !t.is_definition).collect();
        assert!(
            refs.iter().any(|r| r.name == "hello"),
            "should reference 'hello'"
        );
    }

    #[test]
    fn test_extract_tags_type_reference() {
        let src = "fn process(cfg: Config) -> Result<()> { todo!() }";
        let tree = parse_rust(src);
        let tags = extract_rust_tags(&tree, src.as_bytes(), Path::new("test.rs"));

        let refs: Vec<_> = tags.iter().filter(|t| !t.is_definition).collect();
        let ref_names: Vec<_> = refs.iter().map(|r| r.name.as_str()).collect();
        assert!(
            ref_names.contains(&"Config"),
            "should reference Config: {ref_names:?}"
        );
        assert!(
            ref_names.contains(&"Result"),
            "should reference Result: {ref_names:?}"
        );
    }

    #[test]
    fn test_extract_tags_use_declaration() {
        let src = "use std::collections::HashMap;";
        let tree = parse_rust(src);
        let tags = extract_rust_tags(&tree, src.as_bytes(), Path::new("test.rs"));

        let refs: Vec<_> = tags.iter().filter(|t| !t.is_definition).collect();
        let ref_names: Vec<_> = refs.iter().map(|r| r.name.as_str()).collect();
        assert!(
            ref_names.contains(&"HashMap"),
            "should reference HashMap: {ref_names:?}"
        );
    }

    #[test]
    fn test_extract_tags_grouped_use() {
        let src = "use std::collections::{HashMap, BTreeSet};";
        let tree = parse_rust(src);
        let tags = extract_rust_tags(&tree, src.as_bytes(), Path::new("test.rs"));

        let refs: Vec<_> = tags.iter().filter(|t| !t.is_definition).collect();
        let ref_names: Vec<_> = refs.iter().map(|r| r.name.as_str()).collect();
        assert!(
            ref_names.contains(&"HashMap"),
            "should reference HashMap: {ref_names:?}"
        );
        assert!(
            ref_names.contains(&"BTreeSet"),
            "should reference BTreeSet: {ref_names:?}"
        );
    }

    #[test]
    fn test_extract_tags_scoped_call() {
        let src = "fn main() { foo::bar::process(); }";
        let tree = parse_rust(src);
        let tags = extract_rust_tags(&tree, src.as_bytes(), Path::new("test.rs"));

        let refs: Vec<_> = tags.iter().filter(|t| !t.is_definition).collect();
        // Should extract the final segment "process"
        assert!(
            refs.iter().any(|r| r.name == "process"),
            "should reference 'process': {:?}",
            refs.iter().map(|r| &r.name).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_struct_name_not_double_counted_as_reference() {
        // The struct definition name "Config" should not also appear as a reference.
        let src = "pub struct Config { }";
        let tree = parse_rust(src);
        let tags = extract_rust_tags(&tree, src.as_bytes(), Path::new("test.rs"));

        let refs: Vec<_> = tags
            .iter()
            .filter(|t| !t.is_definition && t.name == "Config")
            .collect();
        assert!(
            refs.is_empty(),
            "struct name should not be a reference: {refs:?}"
        );
    }

    #[test]
    fn test_extract_tags_definitions_and_references_mixed() {
        let src = r"
use std::collections::HashMap;

pub struct Config {
    data: HashMap<String, String>,
}

pub fn process(cfg: Config) -> bool {
    validate(&cfg)
}

fn validate(cfg: &Config) -> bool {
    true
}
";
        let tree = parse_rust(src);
        let tags = extract_rust_tags(&tree, src.as_bytes(), Path::new("test.rs"));

        let def_names: Vec<_> = tags
            .iter()
            .filter(|t| t.is_definition)
            .map(|t| t.name.as_str())
            .collect();
        assert!(def_names.contains(&"Config"));
        assert!(def_names.contains(&"process"));
        assert!(def_names.contains(&"validate"));

        let ref_names: Vec<_> = tags
            .iter()
            .filter(|t| !t.is_definition)
            .map(|t| t.name.as_str())
            .collect();
        assert!(ref_names.contains(&"HashMap"));
        assert!(ref_names.contains(&"Config"));
        assert!(ref_names.contains(&"validate"));
    }

    // -- ReferenceGraph::build --

    #[test]
    fn test_build_graph_empty() {
        let graph = ReferenceGraph::build(&HashMap::new());
        assert_eq!(graph.num_files(), 0);
        assert!(graph.adjacency().is_empty());
    }

    #[test]
    fn test_build_graph_cross_file_edge() {
        // File A references `Config` which is defined in File B.
        let mut tags = HashMap::new();
        tags.insert(
            PathBuf::from("/a.rs"),
            vec![Tag {
                name: "Config".into(),
                file: PathBuf::from("/a.rs"),
                line: 5,
                is_definition: false,
                kind: TagKind::Reference,
            }],
        );
        tags.insert(
            PathBuf::from("/b.rs"),
            vec![Tag {
                name: "Config".into(),
                file: PathBuf::from("/b.rs"),
                line: 1,
                is_definition: true,
                kind: TagKind::Struct,
            }],
        );

        let graph = ReferenceGraph::build(&tags);
        assert_eq!(graph.num_files(), 2);

        // Find indices.
        let a_idx = file_index(&graph)[&PathBuf::from("/a.rs")];
        let b_idx = file_index(&graph)[&PathBuf::from("/b.rs")];

        // A → B edge should exist (A references Config defined in B).
        assert!(graph.adjacency()[a_idx].contains(&b_idx));
        // B → A should not exist.
        assert!(!graph.adjacency()[b_idx].contains(&a_idx));
    }

    #[test]
    fn test_build_graph_self_references_excluded() {
        // File A defines and references `Config` — no self-edge.
        let mut tags = HashMap::new();
        tags.insert(
            PathBuf::from("/a.rs"),
            vec![
                Tag {
                    name: "Config".into(),
                    file: PathBuf::from("/a.rs"),
                    line: 1,
                    is_definition: true,
                    kind: TagKind::Struct,
                },
                Tag {
                    name: "Config".into(),
                    file: PathBuf::from("/a.rs"),
                    line: 10,
                    is_definition: false,
                    kind: TagKind::Reference,
                },
            ],
        );

        let graph = ReferenceGraph::build(&tags);
        let a_idx = file_index(&graph)[&PathBuf::from("/a.rs")];
        assert!(graph.adjacency()[a_idx].is_empty());
    }

    #[test]
    fn test_build_graph_unknown_reference_no_edge() {
        // File A references `Foo` which has no definition anywhere.
        let mut tags = HashMap::new();
        tags.insert(
            PathBuf::from("/a.rs"),
            vec![Tag {
                name: "Foo".into(),
                file: PathBuf::from("/a.rs"),
                line: 1,
                is_definition: false,
                kind: TagKind::Reference,
            }],
        );

        let graph = ReferenceGraph::build(&tags);
        let a_idx = file_index(&graph)[&PathBuf::from("/a.rs")];
        assert!(graph.adjacency()[a_idx].is_empty());
    }

    #[test]
    fn test_build_graph_star_topology() {
        // Files B, C, D all reference `Core` defined in A.
        let mut tags = HashMap::new();
        tags.insert(
            PathBuf::from("/a.rs"),
            vec![Tag {
                name: "Core".into(),
                file: PathBuf::from("/a.rs"),
                line: 1,
                is_definition: true,
                kind: TagKind::Struct,
            }],
        );
        for name in &["/b.rs", "/c.rs", "/d.rs"] {
            tags.insert(
                PathBuf::from(name),
                vec![Tag {
                    name: "Core".into(),
                    file: PathBuf::from(name),
                    line: 1,
                    is_definition: false,
                    kind: TagKind::Reference,
                }],
            );
        }

        let graph = ReferenceGraph::build(&tags);
        assert_eq!(graph.num_files(), 4);

        let a_idx = file_index(&graph)[&PathBuf::from("/a.rs")];
        // All of B, C, D should point to A.
        for name in &["/b.rs", "/c.rs", "/d.rs"] {
            let idx = file_index(&graph)[&PathBuf::from(name)];
            assert!(
                graph.adjacency()[idx].contains(&a_idx),
                "{name} should reference /a.rs"
            );
        }
        // A should have no outgoing edges.
        assert!(graph.adjacency()[a_idx].is_empty());
    }

    #[test]
    fn test_build_graph_deterministic_ordering() {
        // Build twice with same input — should produce identical index mapping.
        let make_tags = || {
            let mut tags = HashMap::new();
            for name in &["/z.rs", "/a.rs", "/m.rs"] {
                tags.insert(
                    PathBuf::from(name),
                    vec![Tag {
                        name: "Sym".into(),
                        file: PathBuf::from(name),
                        line: 1,
                        is_definition: true,
                        kind: TagKind::Function,
                    }],
                );
            }
            tags
        };

        let g1 = ReferenceGraph::build(&make_tags());
        let g2 = ReferenceGraph::build(&make_tags());

        assert_eq!(g1.index_to_file(), g2.index_to_file());
    }
}
