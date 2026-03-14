//! AST-based repository map tool via tree-sitter.
//!
//! Generates a structural overview of Rust codebases — function signatures,
//! struct/enum definitions, trait implementations, and module structure —
//! giving the agent an architectural map without reading every file.
//!
//! In **ranked** mode, builds a cross-file reference graph and runs `PageRank`
//! to surface the most important symbols first within a token budget.

pub(crate) mod cache;
pub(crate) mod graph;
mod pagerank;

use std::collections::{HashMap, VecDeque};
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use async_trait::async_trait;

use freebird_security::taint::TaintedToolInput;
use freebird_traits::tool::{
    Capability, RiskLevel, SideEffects, Tool, ToolContext, ToolError, ToolInfo, ToolOutcome,
    ToolOutput,
};

use crate::common::extract_optional_str;

/// Default maximum files to scan.
const DEFAULT_MAX_FILES: usize = 100;

/// Hard cap on `max_files` parameter.
const MAX_FILES_CAP: usize = 500;

/// Maximum file size to parse (1 MiB). Larger files are skipped.
const MAX_PARSE_FILE_BYTES: u64 = 1024 * 1024;

/// Maximum total output size (512 KiB). Truncate with message if exceeded.
const MAX_OUTPUT_BYTES: usize = 512 * 1024;

/// Directories to always skip during discovery.
const SKIP_DIRS: &[&str] = &[".git", "target", "node_modules", ".hg", ".svn", "vendor"];

/// Returns the repo map tool as a trait object.
#[must_use]
pub fn repo_map_tools() -> Vec<Box<dyn Tool>> {
    vec![Box::new(RepoMapTool::new())]
}

// ── Depth ─────────────────────────────────────────────────────────

/// Level of detail in the output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Depth {
    /// Just kind + name (e.g., `fn process`, `struct Config`).
    Outline,
    /// Full signature without body (default).
    Signatures,
    /// Signatures plus doc comments.
    Full,
}

fn parse_depth(s: &str) -> Result<Depth, String> {
    match s {
        "outline" => Ok(Depth::Outline),
        "signatures" => Ok(Depth::Signatures),
        "full" => Ok(Depth::Full),
        other => Err(format!(
            "invalid depth '{other}': expected 'outline', 'signatures', or 'full'"
        )),
    }
}

// ── Mode ─────────────────────────────────────────────────────────

/// Output mode for the repo map tool.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mode {
    /// File-by-file alphabetical symbol listing (default, backward compatible).
    Structure,
    /// Symbols ranked by cross-file reference importance via `PageRank`.
    Ranked,
}

fn parse_mode(s: &str) -> Result<Mode, String> {
    match s {
        "structure" => Ok(Mode::Structure),
        "ranked" => Ok(Mode::Ranked),
        other => Err(format!(
            "invalid mode '{other}': expected 'structure' or 'ranked'"
        )),
    }
}

/// Default token budget for ranked output (~32k tokens at ~4 chars/token).
const DEFAULT_TOKEN_BUDGET_CHARS: usize = 128 * 1024;

/// Validated input parameters extracted before taint wrapping.
struct ParsedParams {
    depth: Depth,
    mode: Mode,
    max_files: usize,
    token_budget_chars: usize,
}

// ── Symbol types ──────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SymbolKind {
    Function,
    Struct,
    Enum,
    Trait,
    Impl,
    TypeAlias,
    Const,
    Static,
    Module,
    Macro,
}

impl SymbolKind {
    const fn label(self) -> &'static str {
        match self {
            Self::Function => "fn",
            Self::Struct => "struct",
            Self::Enum => "enum",
            Self::Trait => "trait",
            Self::Impl => "impl",
            Self::TypeAlias => "type",
            Self::Const => "const",
            Self::Static => "static",
            Self::Module => "mod",
            Self::Macro => "macro",
        }
    }
}

#[derive(Debug, Clone)]
struct Symbol {
    signature: String,
    doc_comment: Option<String>,
    children: Vec<Self>,
    #[cfg(test)]
    kind: SymbolKind,
    #[cfg(test)]
    name: String,
    #[cfg(test)]
    visibility: Option<String>,
}

// ── LanguageMapper trait ──────────────────────────────────────────

/// Language-specific AST parser. Internal trait for extensibility.
trait LanguageMapper: Send + Sync {
    fn language(&self) -> tree_sitter::Language;
    fn file_extension(&self) -> &'static str;
    fn extract_symbols(&self, tree: &tree_sitter::Tree, source: &[u8], depth: Depth)
    -> Vec<Symbol>;
}

// ── RustMapper ────────────────────────────────────────────────────

struct RustMapper;

impl RustMapper {
    fn node_kind_to_symbol(kind: &str) -> Option<SymbolKind> {
        match kind {
            "function_item" | "function_signature_item" => Some(SymbolKind::Function),
            "struct_item" => Some(SymbolKind::Struct),
            "enum_item" => Some(SymbolKind::Enum),
            "trait_item" => Some(SymbolKind::Trait),
            "impl_item" => Some(SymbolKind::Impl),
            "type_item" => Some(SymbolKind::TypeAlias),
            "const_item" => Some(SymbolKind::Const),
            "static_item" => Some(SymbolKind::Static),
            "mod_item" => Some(SymbolKind::Module),
            "macro_definition" => Some(SymbolKind::Macro),
            _ => None,
        }
    }

    /// Safely extract a substring from source bytes as UTF-8.
    fn slice_source(source: &[u8], start: usize, end: usize) -> &str {
        source
            .get(start..end)
            .and_then(|bytes| std::str::from_utf8(bytes).ok())
            .unwrap_or("")
    }

    /// Extract the name from a node. Most items use the `name` field,
    /// but impl blocks need special handling.
    fn extract_name(node: tree_sitter::Node<'_>, source: &[u8]) -> String {
        if node.kind() == "impl_item" {
            return Self::extract_impl_name(node, source);
        }
        if node.kind() == "macro_definition" {
            let mut cursor = node.walk();
            for child in node.children(&mut cursor) {
                if child.kind() == "identifier" {
                    return Self::slice_source(source, child.start_byte(), child.end_byte())
                        .to_owned();
                }
            }
            return String::new();
        }
        node.child_by_field_name("name")
            .map_or_else(String::new, |name_node| {
                Self::slice_source(source, name_node.start_byte(), name_node.end_byte()).to_owned()
            })
    }

    /// Extract the impl block name: `Trait for Type` or just `Type`.
    fn extract_impl_name(node: tree_sitter::Node<'_>, source: &[u8]) -> String {
        let mut cursor = node.walk();
        let children: Vec<_> = node.children(&mut cursor).collect();

        let mut type_ids: Vec<String> = Vec::new();
        let mut has_for = false;
        for child in &children {
            if child.kind() == "for" {
                has_for = true;
            }
            if child.is_named()
                && (child.kind() == "type_identifier"
                    || child.kind() == "scoped_type_identifier"
                    || child.kind() == "generic_type")
            {
                type_ids.push(
                    Self::slice_source(source, child.start_byte(), child.end_byte()).to_owned(),
                );
            }
        }

        if has_for && type_ids.len() >= 2 {
            let mut result = String::new();
            if let Some(first) = type_ids.first() {
                let _ = write!(result, "{first} for ");
            }
            if let Some(second) = type_ids.get(1) {
                result.push_str(second);
            }
            result
        } else {
            type_ids.into_iter().last().unwrap_or_default()
        }
    }

    /// Extract the visibility modifier from a node.
    fn extract_visibility(node: tree_sitter::Node<'_>, source: &[u8]) -> Option<String> {
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            if child.kind() == "visibility_modifier" {
                return Some(
                    Self::slice_source(source, child.start_byte(), child.end_byte()).to_owned(),
                );
            }
        }
        None
    }

    /// Extract the signature text. For items with bodies, extract everything
    /// before the body. For items without bodies, extract the full node text.
    fn extract_signature(
        node: tree_sitter::Node<'_>,
        source: &[u8],
        kind: SymbolKind,
        depth: Depth,
    ) -> String {
        if depth == Depth::Outline {
            let vis = Self::extract_visibility(node, source);
            let name = Self::extract_name(node, source);
            let label = kind.label();
            return vis.map_or_else(
                || format!("{label} {name}"),
                |v| format!("{v} {label} {name}"),
            );
        }

        // Find body node
        let body_kinds = [
            "block",
            "field_declaration_list",
            "enum_variant_list",
            "declaration_list",
        ];
        let mut cursor = node.walk();
        let body_node = node
            .children(&mut cursor)
            .find(|child| body_kinds.contains(&child.kind()));

        match kind {
            SymbolKind::Macro => {
                let name = Self::extract_name(node, source);
                format!("macro_rules! {name}")
            }
            SymbolKind::Const | SymbolKind::Static => {
                // Strip initializer value: "pub const MAX: u64 = 42;" → "pub const MAX: u64"
                // Prevents leaking hardcoded secrets/values to the LLM.
                let full = Self::slice_source(source, node.start_byte(), node.end_byte());
                full.split('=').next().unwrap_or(full).trim_end().to_owned()
            }
            SymbolKind::TypeAlias => Self::slice_source(source, node.start_byte(), node.end_byte())
                .trim_end()
                .to_owned(),
            _ => body_node.map_or_else(
                || {
                    Self::slice_source(source, node.start_byte(), node.end_byte())
                        .trim_end()
                        .to_owned()
                },
                |body| {
                    Self::slice_source(source, node.start_byte(), body.start_byte())
                        .trim_end()
                        .to_owned()
                },
            ),
        }
    }

    /// Extract children from trait/impl declaration lists.
    fn extract_children(node: tree_sitter::Node<'_>, source: &[u8], depth: Depth) -> Vec<Symbol> {
        let mut cursor = node.walk();
        let body_node = node
            .children(&mut cursor)
            .find(|child| child.kind() == "declaration_list");

        let Some(body) = body_node else {
            return Vec::new();
        };

        let mut children = Vec::new();
        let mut body_cursor = body.walk();
        // Track end of previous *symbol* node, not all children.
        // This keeps doc comments (which are also children) inside the
        // search region for collect_doc_comment_before.
        let mut prev_sym_end: Option<usize> = None;
        for child in body.children(&mut body_cursor) {
            if let Some(sym_kind) = Self::node_kind_to_symbol(child.kind()) {
                let doc = if depth == Depth::Full {
                    Self::collect_doc_comment_before(body, child, source, prev_sym_end)
                } else {
                    None
                };
                children.push(Symbol {
                    signature: Self::extract_signature(child, source, sym_kind, depth),
                    doc_comment: doc,
                    children: Vec::new(),
                    #[cfg(test)]
                    kind: sym_kind,
                    #[cfg(test)]
                    name: Self::extract_name(child, source),
                    #[cfg(test)]
                    visibility: Self::extract_visibility(child, source),
                });
                prev_sym_end = Some(child.end_byte());
            }
        }
        children
    }

    /// Collect `///` doc comment lines immediately preceding a node.
    fn collect_doc_comment_before(
        parent: tree_sitter::Node<'_>,
        target: tree_sitter::Node<'_>,
        source: &[u8],
        prev_item_end: Option<usize>,
    ) -> Option<String> {
        let target_start = target.start_byte();
        let search_start = prev_item_end.unwrap_or_else(|| parent.start_byte());

        let region = Self::slice_source(source, search_start, target_start);
        let doc_lines: Vec<&str> = region
            .lines()
            .map(str::trim)
            .filter(|line| line.starts_with("///"))
            .collect();

        if doc_lines.is_empty() {
            None
        } else {
            Some(doc_lines.join("\n"))
        }
    }

    /// Collect doc comments for top-level items by looking at preceding siblings.
    fn collect_doc_comment_for_top_level(
        node: tree_sitter::Node<'_>,
        source: &[u8],
    ) -> Option<String> {
        let mut doc_lines = Vec::new();
        let mut sibling = node.prev_sibling();
        while let Some(sib) = sibling {
            if sib.kind() == "line_comment" {
                let text = Self::slice_source(source, sib.start_byte(), sib.end_byte()).trim();
                if text.starts_with("///") {
                    doc_lines.push(text.to_owned());
                    sibling = sib.prev_sibling();
                    continue;
                }
            }
            if sib.kind() == "attribute_item" {
                sibling = sib.prev_sibling();
                continue;
            }
            break;
        }
        if doc_lines.is_empty() {
            None
        } else {
            doc_lines.reverse();
            Some(doc_lines.join("\n"))
        }
    }

    fn build_symbol(
        node: tree_sitter::Node<'_>,
        source: &[u8],
        kind: SymbolKind,
        depth: Depth,
    ) -> Symbol {
        let children = match kind {
            SymbolKind::Trait | SymbolKind::Impl => Self::extract_children(node, source, depth),
            _ => Vec::new(),
        };
        Symbol {
            signature: Self::extract_signature(node, source, kind, depth),
            doc_comment: None,
            children,
            #[cfg(test)]
            kind,
            #[cfg(test)]
            name: Self::extract_name(node, source),
            #[cfg(test)]
            visibility: Self::extract_visibility(node, source),
        }
    }
}

impl LanguageMapper for RustMapper {
    fn language(&self) -> tree_sitter::Language {
        tree_sitter_rust::LANGUAGE.into()
    }

    fn file_extension(&self) -> &'static str {
        "rs"
    }

    fn extract_symbols(
        &self,
        tree: &tree_sitter::Tree,
        source: &[u8],
        depth: Depth,
    ) -> Vec<Symbol> {
        let root = tree.root_node();
        let mut symbols = Vec::new();
        let mut cursor = root.walk();

        for node in root.children(&mut cursor) {
            let Some(kind) = Self::node_kind_to_symbol(node.kind()) else {
                continue;
            };

            let mut sym = Self::build_symbol(node, source, kind, depth);

            if depth == Depth::Full {
                sym.doc_comment = Self::collect_doc_comment_for_top_level(node, source);
            }

            symbols.push(sym);
        }

        symbols
    }
}

// ── File discovery ────────────────────────────────────────────────

/// Walk a directory tree and collect files matching the given extension.
///
/// Uses an iterative depth-first stack. Skips hidden directories,
/// `SKIP_DIRS`, and files exceeding `MAX_PARSE_FILE_BYTES`.
fn discover_files(root: &Path, extension: &str, max_files: usize) -> Vec<PathBuf> {
    let mut result = Vec::new();
    let mut stack = VecDeque::new();
    stack.push_back(root.to_path_buf());

    while let Some(dir) = stack.pop_back() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };

        let mut subdirs = Vec::new();
        for entry in entries {
            let Ok(entry) = entry else { continue };
            let Ok(file_type) = entry.file_type() else {
                continue;
            };
            let name = entry.file_name();
            let name_str = name.to_string_lossy();

            if file_type.is_dir() {
                if name_str.starts_with('.') || SKIP_DIRS.iter().any(|&skip| name_str == skip) {
                    continue;
                }
                subdirs.push(entry.path());
            } else if file_type.is_file() {
                let path = entry.path();
                let matches_ext = path
                    .extension()
                    .and_then(|e| e.to_str())
                    .is_some_and(|e| e == extension);
                if !matches_ext {
                    continue;
                }

                let Ok(metadata) = entry.metadata() else {
                    continue;
                };
                if metadata.len() > MAX_PARSE_FILE_BYTES {
                    continue;
                }

                result.push(path);
                if result.len() >= max_files {
                    result.sort();
                    return result;
                }
            }
        }

        subdirs.sort();
        for subdir in subdirs.into_iter().rev() {
            stack.push_back(subdir);
        }
    }

    result.sort();
    result
}

// ── Output formatting ─────────────────────────────────────────────

fn format_repo_map(file_symbols: &[(PathBuf, Vec<Symbol>)], root: &Path) -> String {
    let mut output = String::new();
    let mut total_symbols: usize = 0;
    let mut mapped_files: usize = 0;

    for (path, symbols) in file_symbols {
        if symbols.is_empty() {
            continue;
        }

        let relative = path.strip_prefix(root).unwrap_or(path).to_string_lossy();

        let _ = writeln!(output, "## {relative}");
        let _ = writeln!(output);
        mapped_files += 1;

        for (i, sym) in symbols.iter().enumerate() {
            total_symbols += 1;

            if let Some(ref doc) = sym.doc_comment {
                for doc_line in doc.lines() {
                    let _ = writeln!(output, "  {doc_line}");
                }
            }

            let _ = writeln!(output, "  {}", sym.signature);

            for child in &sym.children {
                total_symbols += 1;
                if let Some(ref doc) = child.doc_comment {
                    for doc_line in doc.lines() {
                        let _ = writeln!(output, "    {doc_line}");
                    }
                }
                let _ = writeln!(output, "    {}", child.signature);
            }

            if i + 1 < symbols.len() {
                let _ = writeln!(output);
            }
        }

        let _ = writeln!(output);
    }

    let _ = write!(
        output,
        "Mapped {mapped_files} file{}, {total_symbols} symbol{}",
        if mapped_files == 1 { "" } else { "s" },
        if total_symbols == 1 { "" } else { "s" },
    );

    output
}

// ── RepoMapTool ───────────────────────────────────────────────────

struct RepoMapTool {
    info: ToolInfo,
}

impl RepoMapTool {
    const NAME: &str = "repo_map";

    fn new() -> Self {
        Self {
            info: ToolInfo {
                name: Self::NAME.into(),
                description: "Generate a structural overview of the codebase using AST parsing. \
                    Shows function signatures, struct/enum definitions, trait implementations, \
                    and module structure. In 'ranked' mode, uses PageRank over cross-file \
                    references to surface the most important symbols first."
                    .into(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Relative directory to map. Defaults to sandbox root."
                        },
                        "language": {
                            "type": "string",
                            "description": "Filter to a specific language: 'rust'. Maps all supported languages if omitted."
                        },
                        "depth": {
                            "type": "string",
                            "description": "Level of detail. 'signatures' (default) shows function/type signatures. 'outline' shows just names. 'full' includes doc comments.",
                            "enum": ["outline", "signatures", "full"]
                        },
                        "max_files": {
                            "type": "integer",
                            "description": "Maximum number of files to process. Default: 100. Max: 500."
                        },
                        "mode": {
                            "type": "string",
                            "description": "Output mode. 'structure' (default) lists symbols by file. 'ranked' orders files by cross-file reference importance (PageRank).",
                            "enum": ["structure", "ranked"]
                        },
                        "token_budget": {
                            "type": "integer",
                            "description": "Approximate character budget for ranked output (default: 131072 ~= 32k tokens). Only used in ranked mode."
                        }
                    },
                    "required": []
                }),
                required_capability: Capability::FileRead,
                risk_level: RiskLevel::Low,
                side_effects: SideEffects::None,
            },
        }
    }

    /// Parse the `max_files` parameter, defaulting and clamping.
    fn parse_max_files(input: &serde_json::Value) -> usize {
        input
            .get("max_files")
            .and_then(serde_json::Value::as_u64)
            .map_or(DEFAULT_MAX_FILES, |v| {
                usize::try_from(v)
                    .unwrap_or(MAX_FILES_CAP)
                    .clamp(1, MAX_FILES_CAP)
            })
    }

    /// Resolve the scan root directory from tainted input.
    ///
    /// The `path` field is optional. If missing, defaults to sandbox root.
    /// If present but invalid (traversal, outside sandbox), returns an error.
    fn resolve_scan_root(
        tainted: &TaintedToolInput,
        context: &ToolContext<'_>,
    ) -> Result<PathBuf, ToolError> {
        match tainted.extract_path_multi_root(
            "path",
            context.sandbox_root,
            context.allowed_directories,
        ) {
            Ok(path) => Ok(path.as_path().to_path_buf()),
            Err(freebird_security::error::SecurityError::MissingField { .. }) => {
                // "path" is optional — default to sandbox root
                Ok(context.sandbox_root.to_path_buf())
            }
            Err(e) => Err(ToolError::InvalidInput {
                tool: Self::NAME.into(),
                reason: e.to_string(),
            }),
        }
    }

    /// Parse and validate tool input parameters before taint wrapping.
    fn parse_input_params(input: &serde_json::Value) -> Result<ParsedParams, ToolError> {
        let depth = match extract_optional_str(input, "depth") {
            Some(s) => parse_depth(s).map_err(|reason| ToolError::InvalidInput {
                tool: Self::NAME.into(),
                reason,
            })?,
            None => Depth::Signatures,
        };

        let mode = match extract_optional_str(input, "mode") {
            Some(s) => parse_mode(s).map_err(|reason| ToolError::InvalidInput {
                tool: Self::NAME.into(),
                reason,
            })?,
            None => Mode::Structure,
        };

        let token_budget_chars = input
            .get("token_budget")
            .and_then(serde_json::Value::as_u64)
            .map_or(DEFAULT_TOKEN_BUDGET_CHARS, |v| {
                usize::try_from(v).unwrap_or(DEFAULT_TOKEN_BUDGET_CHARS)
            });

        if let Some(lang) = extract_optional_str(input, "language") {
            if lang != "rust" {
                return Err(ToolError::InvalidInput {
                    tool: Self::NAME.into(),
                    reason: format!(
                        "unsupported language '{lang}': only 'rust' is supported in v1"
                    ),
                });
            }
        }

        let max_files = Self::parse_max_files(input);

        Ok(ParsedParams {
            depth,
            mode,
            max_files,
            token_budget_chars,
        })
    }

    /// Discover files, parse them, and produce the map output string.
    async fn generate_map(
        scan_root: &Path,
        context: &ToolContext<'_>,
        params: &ParsedParams,
    ) -> Result<String, ToolError> {
        let mapper = RustMapper;
        let ext = mapper.file_extension().to_owned();
        let root_for_discovery = scan_root.to_path_buf();
        let max_files = params.max_files;

        let files = tokio::task::spawn_blocking(move || {
            discover_files(&root_for_discovery, &ext, max_files)
        })
        .await
        .map_err(|e| ToolError::ExecutionFailed {
            tool: Self::NAME.into(),
            reason: format!("file discovery task failed: {e}"),
        })?;

        if files.is_empty() {
            return Ok(format!(
                "No .{} files found in '{}'",
                mapper.file_extension(),
                scan_root
                    .strip_prefix(context.sandbox_root)
                    .unwrap_or(scan_root)
                    .display()
            ));
        }

        let sandbox_root = context.sandbox_root.to_path_buf();
        let scan_root_owned = scan_root.to_path_buf();
        let depth = params.depth;
        let token_budget_chars = params.token_budget_chars;

        match params.mode {
            Mode::Structure => {
                let language = mapper.language();
                let file_symbols = tokio::task::spawn_blocking(move || {
                    Self::parse_files(&files, &language, &mapper, depth)
                })
                .await
                .map_err(|e| ToolError::ExecutionFailed {
                    tool: Self::NAME.into(),
                    reason: format!("parsing task failed: {e}"),
                })?
                .map_err(|reason| ToolError::ExecutionFailed {
                    tool: Self::NAME.into(),
                    reason,
                })?;

                Ok(format_repo_map(&file_symbols, &scan_root_owned))
            }
            Mode::Ranked => {
                let language = mapper.language();
                tokio::task::spawn_blocking(move || {
                    Self::execute_ranked(
                        &files,
                        &language,
                        &mapper,
                        depth,
                        &scan_root_owned,
                        &sandbox_root,
                        token_budget_chars,
                    )
                })
                .await
                .map_err(|e| ToolError::ExecutionFailed {
                    tool: Self::NAME.into(),
                    reason: format!("ranked analysis task failed: {e}"),
                })?
                .map_err(|reason| ToolError::ExecutionFailed {
                    tool: Self::NAME.into(),
                    reason,
                })
            }
        }
    }

    /// Parse files and extract symbols from each.
    fn parse_files(
        files: &[PathBuf],
        language: &tree_sitter::Language,
        mapper: &RustMapper,
        depth: Depth,
    ) -> Result<Vec<(PathBuf, Vec<Symbol>)>, String> {
        let mut parser = tree_sitter::Parser::new();
        if parser.set_language(language).is_err() {
            return Err("failed to set tree-sitter language".to_owned());
        }

        let mut results: Vec<(PathBuf, Vec<Symbol>)> = Vec::with_capacity(files.len());
        for file_path in files {
            // Re-check size to narrow the TOCTOU window from discovery.
            let Ok(metadata) = std::fs::metadata(file_path) else {
                continue;
            };
            if metadata.len() > MAX_PARSE_FILE_BYTES {
                continue;
            }
            let Ok(source) = std::fs::read(file_path) else {
                continue;
            };
            let Some(tree) = parser.parse(&source, None) else {
                continue;
            };
            let symbols = mapper.extract_symbols(&tree, &source, depth);
            if !symbols.is_empty() {
                results.push((file_path.clone(), symbols));
            }
        }
        Ok(results)
    }

    /// Execute the ranked mode: extract tags, build reference graph, run
    /// `PageRank`, then format output with files ordered by importance.
    #[allow(clippy::too_many_arguments)]
    fn execute_ranked(
        files: &[PathBuf],
        language: &tree_sitter::Language,
        mapper: &RustMapper,
        depth: Depth,
        scan_root: &Path,
        sandbox_root: &Path,
        token_budget_chars: usize,
    ) -> Result<String, String> {
        let mut parser = tree_sitter::Parser::new();
        if parser.set_language(language).is_err() {
            return Err("failed to set tree-sitter language".to_owned());
        }

        // Load tag cache.
        let mut tag_cache = cache::TagCache::load(sandbox_root);

        // Extract tags and symbols for each file.
        let mut tags_by_file: HashMap<PathBuf, Vec<graph::Tag>> = HashMap::new();
        let mut symbols_by_file: HashMap<PathBuf, Vec<Symbol>> = HashMap::new();

        for file_path in files {
            let Ok(metadata) = std::fs::metadata(file_path) else {
                continue;
            };
            if metadata.len() > MAX_PARSE_FILE_BYTES {
                continue;
            }

            // Check cache first.
            if let Some(cached_tags) = tag_cache.get(file_path) {
                tags_by_file.insert(file_path.clone(), cached_tags);
                // Still need symbols for output formatting — parse the file.
                let Ok(source) = std::fs::read(file_path) else {
                    continue;
                };
                let Some(tree) = parser.parse(&source, None) else {
                    continue;
                };
                let symbols = mapper.extract_symbols(&tree, &source, depth);
                if !symbols.is_empty() {
                    symbols_by_file.insert(file_path.clone(), symbols);
                }
                continue;
            }

            let Ok(source) = std::fs::read(file_path) else {
                continue;
            };
            let Some(tree) = parser.parse(&source, None) else {
                continue;
            };

            let tags = graph::extract_rust_tags(&tree, &source, file_path);
            let symbols = mapper.extract_symbols(&tree, &source, depth);

            // Update cache.
            if let Ok(mtime) = metadata.modified() {
                tag_cache.insert(file_path.clone(), mtime, &tags);
            }

            if !tags.is_empty() {
                tags_by_file.insert(file_path.clone(), tags);
            }
            if !symbols.is_empty() {
                symbols_by_file.insert(file_path.clone(), symbols);
            }
        }

        // Save cache for next invocation.
        tag_cache.save(sandbox_root);

        // Build reference graph and compute PageRank.
        let ref_graph = graph::ReferenceGraph::build(&tags_by_file);
        let ranks = pagerank::pagerank(ref_graph.adjacency(), ref_graph.num_files());

        Ok(format_ranked_repo_map(
            &ref_graph,
            &ranks,
            &symbols_by_file,
            scan_root,
            token_budget_chars,
        ))
    }
}

// ── Output helpers ───────────────────────────────────────────────

/// Truncate output to `MAX_OUTPUT_BYTES` on a clean line boundary.
fn truncate_output(output: &mut String) {
    if output.len() <= MAX_OUTPUT_BYTES {
        return;
    }
    let mut trunc_at = MAX_OUTPUT_BYTES;
    while !output.is_char_boundary(trunc_at) && trunc_at > 0 {
        trunc_at -= 1;
    }
    output.truncate(trunc_at);
    if let Some(last_nl) = output.rfind('\n') {
        output.truncate(last_nl + 1);
    }
    output.push_str("\n... output truncated (exceeded 512 KiB limit)\n");
}

// ── Ranked output formatting ─────────────────────────────────────

/// Format a ranked repo map: files sorted by `PageRank` score (descending),
/// with symbols within each file, truncated to fit the token budget.
fn format_ranked_repo_map(
    graph: &graph::ReferenceGraph,
    ranks: &[f64],
    symbols_by_file: &HashMap<PathBuf, Vec<Symbol>>,
    root: &Path,
    token_budget_chars: usize,
) -> String {
    // Pair files with their ranks and sort descending.
    let index_to_file = graph.index_to_file();
    let mut file_ranks: Vec<(usize, f64)> = ranks.iter().copied().enumerate().collect();
    file_ranks.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut output = String::new();
    let mut total_symbols: usize = 0;
    let mut mapped_files: usize = 0;

    let _ = writeln!(
        output,
        "[RANKED REPO MAP — files ordered by cross-file reference importance]\n"
    );

    for (idx, _rank) in &file_ranks {
        if output.len() >= token_budget_chars {
            break;
        }

        let Some(file_path) = index_to_file.get(*idx) else {
            continue;
        };
        let Some(symbols) = symbols_by_file.get(file_path) else {
            continue;
        };
        if symbols.is_empty() {
            continue;
        }

        let relative = file_path
            .strip_prefix(root)
            .unwrap_or(file_path)
            .to_string_lossy();
        let _ = writeln!(output, "## {relative}");
        let _ = writeln!(output);
        mapped_files += 1;

        for (i, sym) in symbols.iter().enumerate() {
            if output.len() >= token_budget_chars {
                break;
            }
            total_symbols += 1;

            if let Some(ref doc) = sym.doc_comment {
                for doc_line in doc.lines() {
                    let _ = writeln!(output, "  {doc_line}");
                }
            }
            let _ = writeln!(output, "  {}", sym.signature);

            for child in &sym.children {
                total_symbols += 1;
                if let Some(ref doc) = child.doc_comment {
                    for doc_line in doc.lines() {
                        let _ = writeln!(output, "    {doc_line}");
                    }
                }
                let _ = writeln!(output, "    {}", child.signature);
            }

            if i + 1 < symbols.len() {
                let _ = writeln!(output);
            }
        }
        let _ = writeln!(output);
    }

    let _ = write!(
        output,
        "Ranked {mapped_files} file{}, {total_symbols} symbol{}",
        if mapped_files == 1 { "" } else { "s" },
        if total_symbols == 1 { "" } else { "s" },
    );

    output
}

#[async_trait]
impl Tool for RepoMapTool {
    fn info(&self) -> &ToolInfo {
        &self.info
    }

    async fn execute(
        &self,
        input: serde_json::Value,
        context: &ToolContext<'_>,
    ) -> Result<ToolOutput, ToolError> {
        let params = Self::parse_input_params(&input)?;
        let tainted = TaintedToolInput::new(input);
        let scan_root = Self::resolve_scan_root(&tainted, context)?;

        if !scan_root.is_dir() {
            return Err(ToolError::InvalidInput {
                tool: Self::NAME.into(),
                reason: format!(
                    "'{}' is not a directory",
                    scan_root
                        .strip_prefix(context.sandbox_root)
                        .unwrap_or(&scan_root)
                        .display()
                ),
            });
        }

        let mut output_text = Self::generate_map(&scan_root, context, &params).await?;
        truncate_output(&mut output_text);

        Ok(ToolOutput {
            content: output_text,
            outcome: ToolOutcome::Success,
            metadata: None,
        })
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic, clippy::indexing_slicing)]
mod tests {
    use std::path::PathBuf;

    use freebird_traits::id::SessionId;
    use freebird_traits::tool::{Capability, Tool, ToolContext, ToolError, ToolOutcome};

    use super::*;

    // ── TestHarness ───────────────────────────────────────────────

    struct TestHarness {
        _tmp: tempfile::TempDir,
        sandbox: PathBuf,
        session_id: SessionId,
        capabilities: Vec<Capability>,
        allowed_directories: Vec<PathBuf>,
    }

    impl TestHarness {
        fn new() -> Self {
            let tmp = tempfile::tempdir().unwrap();
            let sandbox = tmp.path().canonicalize().unwrap();
            Self {
                _tmp: tmp,
                sandbox,
                session_id: SessionId::from_string("test-session"),
                capabilities: vec![Capability::FileRead],
                allowed_directories: vec![],
            }
        }

        fn path(&self) -> &Path {
            &self.sandbox
        }

        fn context(&self) -> ToolContext<'_> {
            ToolContext {
                session_id: &self.session_id,
                sandbox_root: &self.sandbox,
                granted_capabilities: &self.capabilities,
                allowed_directories: &self.allowed_directories,
                knowledge_store: None,
                memory: None,
            }
        }
    }

    // ── Phase 1: Pure types ───────────────────────────────────────

    #[test]
    fn test_parse_depth_valid_variants() {
        assert_eq!(parse_depth("outline").unwrap(), Depth::Outline);
        assert_eq!(parse_depth("signatures").unwrap(), Depth::Signatures);
        assert_eq!(parse_depth("full").unwrap(), Depth::Full);
    }

    #[test]
    fn test_parse_depth_invalid() {
        let err = parse_depth("extreme").unwrap_err();
        assert!(err.contains("invalid depth"));
        assert!(err.contains("extreme"));
    }

    #[test]
    fn test_symbol_kind_label() {
        assert_eq!(SymbolKind::Function.label(), "fn");
        assert_eq!(SymbolKind::Struct.label(), "struct");
        assert_eq!(SymbolKind::Enum.label(), "enum");
        assert_eq!(SymbolKind::Trait.label(), "trait");
        assert_eq!(SymbolKind::Impl.label(), "impl");
        assert_eq!(SymbolKind::TypeAlias.label(), "type");
        assert_eq!(SymbolKind::Const.label(), "const");
        assert_eq!(SymbolKind::Static.label(), "static");
        assert_eq!(SymbolKind::Module.label(), "mod");
        assert_eq!(SymbolKind::Macro.label(), "macro");
    }

    // ── Phase 2: File discovery ───────────────────────────────────

    #[test]
    fn test_discover_finds_rs_files() {
        let h = TestHarness::new();
        std::fs::write(h.path().join("main.rs"), "fn main() {}").unwrap();
        std::fs::write(h.path().join("lib.rs"), "pub mod foo;").unwrap();

        let files = discover_files(h.path(), "rs", 100);
        assert_eq!(files.len(), 2);
    }

    #[test]
    fn test_discover_skips_hidden_dirs() {
        let h = TestHarness::new();
        let hidden = h.path().join(".hidden");
        std::fs::create_dir(&hidden).unwrap();
        std::fs::write(hidden.join("secret.rs"), "fn secret() {}").unwrap();
        std::fs::write(h.path().join("visible.rs"), "fn visible() {}").unwrap();

        let files = discover_files(h.path(), "rs", 100);
        assert_eq!(files.len(), 1);
        assert!(files[0].to_string_lossy().contains("visible.rs"));
    }

    #[test]
    fn test_discover_skips_target() {
        let h = TestHarness::new();
        let target = h.path().join("target");
        std::fs::create_dir(&target).unwrap();
        std::fs::write(target.join("build.rs"), "fn build() {}").unwrap();
        std::fs::write(h.path().join("src.rs"), "fn src() {}").unwrap();

        let files = discover_files(h.path(), "rs", 100);
        assert_eq!(files.len(), 1);
    }

    #[test]
    fn test_discover_respects_max() {
        let h = TestHarness::new();
        for i in 0..20 {
            std::fs::write(h.path().join(format!("file{i:02}.rs")), "fn f() {}").unwrap();
        }

        let files = discover_files(h.path(), "rs", 5);
        assert_eq!(files.len(), 5);
    }

    #[test]
    fn test_discover_sorted() {
        let h = TestHarness::new();
        std::fs::write(h.path().join("zebra.rs"), "").unwrap();
        std::fs::write(h.path().join("alpha.rs"), "").unwrap();
        std::fs::write(h.path().join("middle.rs"), "").unwrap();

        let files = discover_files(h.path(), "rs", 100);
        let names: Vec<String> = files
            .iter()
            .map(|p| p.file_name().unwrap().to_string_lossy().into_owned())
            .collect();
        assert_eq!(names, vec!["alpha.rs", "middle.rs", "zebra.rs"]);
    }

    #[test]
    fn test_discover_empty_dir() {
        let h = TestHarness::new();
        let files = discover_files(h.path(), "rs", 100);
        assert!(files.is_empty());
    }

    #[test]
    fn test_discover_recursive() {
        let h = TestHarness::new();
        let sub = h.path().join("src").join("tools");
        std::fs::create_dir_all(&sub).unwrap();
        std::fs::write(sub.join("deep.rs"), "fn deep() {}").unwrap();

        let files = discover_files(h.path(), "rs", 100);
        assert_eq!(files.len(), 1);
        assert!(files[0].to_string_lossy().contains("deep.rs"));
    }

    // ── Phase 3: Symbol extraction ────────────────────────────────

    fn parse_rust(source: &str) -> (tree_sitter::Tree, Vec<u8>) {
        let mut parser = tree_sitter::Parser::new();
        parser
            .set_language(&tree_sitter_rust::LANGUAGE.into())
            .unwrap();
        let bytes = source.as_bytes().to_vec();
        let tree = parser.parse(&bytes, None).unwrap();
        (tree, bytes)
    }

    fn extract(source: &str, depth: Depth) -> Vec<Symbol> {
        let (tree, bytes) = parse_rust(source);
        let mapper = RustMapper;
        mapper.extract_symbols(&tree, &bytes, depth)
    }

    #[test]
    fn test_extract_pub_function_signatures() {
        let symbols = extract("pub fn hello(x: i32) -> bool { true }", Depth::Signatures);
        assert_eq!(symbols.len(), 1);
        assert_eq!(symbols[0].kind, SymbolKind::Function);
        assert_eq!(symbols[0].name, "hello");
        assert!(
            symbols[0]
                .signature
                .contains("pub fn hello(x: i32) -> bool")
        );
        // Should NOT contain the body
        assert!(!symbols[0].signature.contains("true"));
    }

    #[test]
    fn test_extract_function_outline() {
        let symbols = extract("pub fn hello(x: i32) -> bool { true }", Depth::Outline);
        assert_eq!(symbols.len(), 1);
        assert_eq!(symbols[0].signature, "pub fn hello");
    }

    #[test]
    fn test_extract_struct_with_fields() {
        let symbols = extract(
            "pub struct Config { pub name: String, timeout: u64 }",
            Depth::Signatures,
        );
        assert_eq!(symbols.len(), 1);
        assert_eq!(symbols[0].kind, SymbolKind::Struct);
        assert_eq!(symbols[0].name, "Config");
        assert!(symbols[0].signature.contains("pub struct Config"));
    }

    #[test]
    fn test_extract_enum_with_variants() {
        let symbols = extract("pub enum Color { Red, Green, Blue(u8) }", Depth::Signatures);
        assert_eq!(symbols.len(), 1);
        assert_eq!(symbols[0].kind, SymbolKind::Enum);
        assert_eq!(symbols[0].name, "Color");
        assert!(symbols[0].signature.contains("pub enum Color"));
    }

    #[test]
    fn test_extract_trait_with_methods() {
        let symbols = extract(
            "pub trait Greeter { fn greet(&self) -> String; fn name(&self) -> &str; }",
            Depth::Signatures,
        );
        assert_eq!(symbols.len(), 1);
        assert_eq!(symbols[0].kind, SymbolKind::Trait);
        assert_eq!(symbols[0].name, "Greeter");
        assert_eq!(symbols[0].children.len(), 2);
        assert_eq!(symbols[0].children[0].name, "greet");
        assert_eq!(symbols[0].children[1].name, "name");
    }

    #[test]
    fn test_extract_impl_block() {
        let source = "struct Foo;\nimpl Foo { pub fn new() -> Self { Self } fn private(&self) {} }";
        let symbols = extract(source, Depth::Signatures);
        let impl_sym = symbols.iter().find(|s| s.kind == SymbolKind::Impl).unwrap();
        assert_eq!(impl_sym.name, "Foo");
        assert!(impl_sym.children.len() >= 2);
    }

    #[test]
    fn test_extract_trait_impl() {
        let source = "trait Foo {} struct Bar;\nimpl Foo for Bar {}";
        let symbols = extract(source, Depth::Signatures);
        let impl_sym = symbols.iter().find(|s| s.kind == SymbolKind::Impl).unwrap();
        assert!(
            impl_sym.name.contains("Foo") && impl_sym.name.contains("Bar"),
            "impl name should contain both trait and type: {}",
            impl_sym.name
        );
    }

    #[test]
    fn test_extract_type_alias() {
        let symbols = extract(
            "pub type Result<T> = std::result::Result<T, Error>;",
            Depth::Signatures,
        );
        assert_eq!(symbols.len(), 1);
        assert_eq!(symbols[0].kind, SymbolKind::TypeAlias);
        assert_eq!(symbols[0].name, "Result");
        assert!(symbols[0].signature.contains("pub type Result"));
    }

    #[test]
    fn test_extract_const_strips_value() {
        let symbols = extract("pub const MAX: u64 = 42;", Depth::Signatures);
        assert_eq!(symbols.len(), 1);
        assert_eq!(symbols[0].kind, SymbolKind::Const);
        assert_eq!(symbols[0].name, "MAX");
        assert!(
            symbols[0].signature.contains("pub const MAX: u64"),
            "signature should contain type declaration"
        );
        assert!(
            !symbols[0].signature.contains("42"),
            "signature must NOT contain initializer value: {}",
            symbols[0].signature
        );
    }

    #[test]
    fn test_extract_static_strips_value() {
        let symbols = extract(
            "pub static GLOBAL: &str = \"secret_api_key\";",
            Depth::Signatures,
        );
        assert_eq!(symbols.len(), 1);
        assert_eq!(symbols[0].kind, SymbolKind::Static);
        assert_eq!(symbols[0].name, "GLOBAL");
        assert!(
            symbols[0].signature.contains("pub static GLOBAL: &str"),
            "signature should contain type declaration"
        );
        assert!(
            !symbols[0].signature.contains("secret_api_key"),
            "signature must NOT contain initializer value: {}",
            symbols[0].signature
        );
    }

    #[test]
    fn test_extract_type_alias_keeps_definition() {
        let symbols = extract(
            "pub type Result<T> = std::result::Result<T, Error>;",
            Depth::Signatures,
        );
        assert_eq!(symbols.len(), 1);
        assert_eq!(symbols[0].kind, SymbolKind::TypeAlias);
        assert!(
            symbols[0].signature.contains("std::result::Result"),
            "type alias should keep full definition: {}",
            symbols[0].signature
        );
    }

    #[test]
    fn test_extract_module() {
        let symbols = extract("pub mod tools;", Depth::Signatures);
        assert_eq!(symbols.len(), 1);
        assert_eq!(symbols[0].kind, SymbolKind::Module);
        assert_eq!(symbols[0].name, "tools");
    }

    #[test]
    fn test_extract_macro() {
        let symbols = extract("macro_rules! my_macro { () => {}; }", Depth::Signatures);
        assert_eq!(symbols.len(), 1);
        assert_eq!(symbols[0].kind, SymbolKind::Macro);
        assert_eq!(symbols[0].name, "my_macro");
    }

    #[test]
    fn test_extract_doc_comments_full_depth() {
        let source = "/// This is a doc comment\n/// Second line\npub fn documented() {}";
        let symbols = extract(source, Depth::Full);
        assert_eq!(symbols.len(), 1);
        let doc = symbols[0].doc_comment.as_ref().unwrap();
        assert!(doc.contains("This is a doc comment"));
        assert!(doc.contains("Second line"));
    }

    #[test]
    fn test_extract_no_doc_comments_at_signatures() {
        let source = "/// This is a doc comment\npub fn documented() {}";
        let symbols = extract(source, Depth::Signatures);
        assert_eq!(symbols.len(), 1);
        assert!(symbols[0].doc_comment.is_none());
    }

    #[test]
    fn test_extract_visibility_variants() {
        let source = "pub fn public() {}\nfn private() {}\npub(crate) fn crate_vis() {}";
        let symbols = extract(source, Depth::Signatures);
        assert_eq!(symbols.len(), 3);
        assert_eq!(symbols[0].visibility.as_deref(), Some("pub"));
        assert!(symbols[1].visibility.is_none());
        assert_eq!(symbols[2].visibility.as_deref(), Some("pub(crate)"));
    }

    #[test]
    fn test_extract_handles_parse_errors_gracefully() {
        let source = "pub fn broken( { struct { enum }";
        let symbols = extract(source, Depth::Signatures);
        // Parser should return partial results or empty — never panic.
        // The exact count depends on tree-sitter error recovery, but it
        // must be finite and not crash.
        assert!(
            symbols.len() <= 10,
            "unexpectedly many symbols from broken source"
        );
    }

    #[test]
    fn test_skips_use_declarations() {
        let source = "use std::io;\nuse std::path::Path;\npub fn foo() {}";
        let symbols = extract(source, Depth::Signatures);
        assert_eq!(symbols.len(), 1);
        assert_eq!(symbols[0].kind, SymbolKind::Function);
    }

    #[test]
    fn test_skips_function_bodies() {
        let source = "pub fn complex() -> String {\n    let x = 42;\n    format!(\"hello {x}\")\n}";
        let symbols = extract(source, Depth::Signatures);
        assert_eq!(symbols.len(), 1);
        assert!(!symbols[0].signature.contains("let x = 42"));
        assert!(!symbols[0].signature.contains("format!"));
    }

    // ── Phase 4: Output formatting ────────────────────────────────

    #[test]
    fn test_format_outline_depth() {
        let source = "pub fn foo() {}\npub struct Bar {}";
        let symbols = extract(source, Depth::Outline);
        let file_symbols = vec![(PathBuf::from("/root/src/lib.rs"), symbols)];
        let output = format_repo_map(&file_symbols, Path::new("/root"));
        assert!(output.contains("pub fn foo"));
        assert!(output.contains("pub struct Bar"));
        assert!(!output.contains("()"));
    }

    #[test]
    fn test_format_signatures_depth() {
        let source = "pub fn foo(x: i32) -> bool { true }";
        let symbols = extract(source, Depth::Signatures);
        let file_symbols = vec![(PathBuf::from("/root/src/lib.rs"), symbols)];
        let output = format_repo_map(&file_symbols, Path::new("/root"));
        assert!(output.contains("pub fn foo(x: i32) -> bool"));
    }

    #[test]
    fn test_format_full_with_docs() {
        let source = "/// My function\npub fn foo() {}";
        let symbols = extract(source, Depth::Full);
        let file_symbols = vec![(PathBuf::from("/root/src/lib.rs"), symbols)];
        let output = format_repo_map(&file_symbols, Path::new("/root"));
        assert!(output.contains("/// My function"));
        assert!(output.contains("pub fn foo()"));
    }

    #[test]
    fn test_format_relative_paths() {
        let symbols = extract("pub fn foo() {}", Depth::Signatures);
        let file_symbols = vec![(PathBuf::from("/root/src/lib.rs"), symbols)];
        let output = format_repo_map(&file_symbols, Path::new("/root"));
        assert!(output.contains("## src/lib.rs"));
        assert!(!output.contains("/root/"));
    }

    #[test]
    fn test_format_empty_input() {
        let output = format_repo_map(&[], Path::new("/root"));
        assert!(output.contains("Mapped 0 files, 0 symbols"));
    }

    #[test]
    fn test_format_children_indented() {
        let source = "impl Foo { pub fn bar() {} pub fn baz() {} }";
        let symbols = extract(source, Depth::Signatures);
        let file_symbols = vec![(PathBuf::from("/root/src/lib.rs"), symbols)];
        let output = format_repo_map(&file_symbols, Path::new("/root"));
        let lines: Vec<&str> = output.lines().collect();
        let impl_line = lines.iter().find(|l| l.contains("impl Foo")).unwrap();
        let child_line = lines.iter().find(|l| l.contains("fn bar")).unwrap();
        assert!(
            impl_line.starts_with("  "),
            "impl should be 2-space indented"
        );
        assert!(
            child_line.starts_with("    "),
            "child should be 4-space indented"
        );
    }

    // ── Phase 5: Integration — Tool::execute ──────────────────────

    #[tokio::test]
    async fn test_repo_map_default_params() {
        let h = TestHarness::new();
        std::fs::write(h.path().join("main.rs"), "pub fn main() {}").unwrap();

        let tool = RepoMapTool::new();
        let output = tool
            .execute(serde_json::json!({}), &h.context())
            .await
            .unwrap();
        assert!(matches!(output.outcome, ToolOutcome::Success));
        assert!(output.content.contains("pub fn main()"));
        assert!(output.content.contains("Mapped"));
    }

    #[tokio::test]
    async fn test_repo_map_with_explicit_path() {
        let h = TestHarness::new();
        let sub = h.path().join("subdir");
        std::fs::create_dir(&sub).unwrap();
        std::fs::write(sub.join("inner.rs"), "pub fn inner() {}").unwrap();
        std::fs::write(h.path().join("outer.rs"), "pub fn outer() {}").unwrap();

        let tool = RepoMapTool::new();
        let output = tool
            .execute(serde_json::json!({"path": "subdir"}), &h.context())
            .await
            .unwrap();
        assert!(output.content.contains("inner"));
        assert!(!output.content.contains("outer"));
    }

    #[tokio::test]
    async fn test_repo_map_outline_depth() {
        let h = TestHarness::new();
        std::fs::write(
            h.path().join("lib.rs"),
            "pub fn foo(x: i32) -> bool { true }",
        )
        .unwrap();

        let tool = RepoMapTool::new();
        let output = tool
            .execute(serde_json::json!({"depth": "outline"}), &h.context())
            .await
            .unwrap();
        assert!(output.content.contains("pub fn foo"));
        assert!(!output.content.contains("i32"));
    }

    #[tokio::test]
    async fn test_repo_map_full_depth() {
        let h = TestHarness::new();
        std::fs::write(h.path().join("lib.rs"), "/// My function\npub fn foo() {}").unwrap();

        let tool = RepoMapTool::new();
        let output = tool
            .execute(serde_json::json!({"depth": "full"}), &h.context())
            .await
            .unwrap();
        assert!(output.content.contains("/// My function"));
    }

    #[tokio::test]
    async fn test_repo_map_invalid_language() {
        let h = TestHarness::new();
        let tool = RepoMapTool::new();

        let err = tool
            .execute(serde_json::json!({"language": "cobol"}), &h.context())
            .await
            .unwrap_err();
        match err {
            ToolError::InvalidInput { tool, reason } => {
                assert_eq!(tool, "repo_map");
                assert!(reason.contains("cobol"));
            }
            other => panic!("expected InvalidInput, got: {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_repo_map_invalid_depth() {
        let h = TestHarness::new();
        let tool = RepoMapTool::new();

        let err = tool
            .execute(serde_json::json!({"depth": "extreme"}), &h.context())
            .await
            .unwrap_err();
        match err {
            ToolError::InvalidInput { tool, .. } => assert_eq!(tool, "repo_map"),
            other => panic!("expected InvalidInput, got: {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_repo_map_no_files_found() {
        let h = TestHarness::new();
        let tool = RepoMapTool::new();
        let output = tool
            .execute(serde_json::json!({}), &h.context())
            .await
            .unwrap();
        assert!(matches!(output.outcome, ToolOutcome::Success));
        assert!(output.content.contains("No .rs files found"));
    }

    #[tokio::test]
    async fn test_repo_map_max_files_respected() {
        let h = TestHarness::new();
        for i in 0..20 {
            std::fs::write(
                h.path().join(format!("f{i:02}.rs")),
                format!("pub fn f{i}() {{}}"),
            )
            .unwrap();
        }

        let tool = RepoMapTool::new();
        let output = tool
            .execute(serde_json::json!({"max_files": 3}), &h.context())
            .await
            .unwrap();
        let file_headers: Vec<&str> = output
            .content
            .lines()
            .filter(|l| l.starts_with("## "))
            .collect();
        assert!(
            file_headers.len() <= 3,
            "expected at most 3 file headers, got {}",
            file_headers.len()
        );
    }

    #[tokio::test]
    async fn test_repo_map_path_traversal_rejected() {
        let h = TestHarness::new();
        let tool = RepoMapTool::new();

        let err = tool
            .execute(serde_json::json!({"path": "../../etc"}), &h.context())
            .await
            .unwrap_err();
        match err {
            ToolError::InvalidInput { tool, .. } => assert_eq!(tool, "repo_map"),
            other => panic!("expected InvalidInput, got: {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_repo_map_output_uses_relative_paths() {
        let h = TestHarness::new();
        std::fs::write(h.path().join("main.rs"), "pub fn main() {}").unwrap();

        let tool = RepoMapTool::new();
        let output = tool
            .execute(serde_json::json!({}), &h.context())
            .await
            .unwrap();
        let sandbox_str = h.path().to_string_lossy();
        assert!(
            !output.content.contains(sandbox_str.as_ref()),
            "output should NOT contain sandbox root: {}",
            output.content
        );
    }

    #[test]
    fn test_repo_map_factory() {
        let tools = repo_map_tools();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].info().name, "repo_map");
    }

    #[test]
    fn test_repo_map_tool_metadata() {
        let tool = RepoMapTool::new();
        let info = tool.info();
        assert_eq!(info.name, "repo_map");
        assert_eq!(info.required_capability, Capability::FileRead);
        assert_eq!(info.risk_level, RiskLevel::Low);
        assert_eq!(info.side_effects, SideEffects::None);
    }

    // ── parse_max_files tests ──────────────────────────────────────

    #[test]
    fn test_parse_max_files_defaults_when_absent() {
        assert_eq!(
            RepoMapTool::parse_max_files(&serde_json::json!({})),
            DEFAULT_MAX_FILES
        );
    }

    #[test]
    fn test_parse_max_files_clamps_zero_to_one() {
        assert_eq!(
            RepoMapTool::parse_max_files(&serde_json::json!({"max_files": 0})),
            1
        );
    }

    #[test]
    fn test_parse_max_files_clamps_above_cap() {
        assert_eq!(
            RepoMapTool::parse_max_files(&serde_json::json!({"max_files": 9999})),
            MAX_FILES_CAP
        );
    }

    #[test]
    fn test_parse_max_files_handles_non_integer() {
        assert_eq!(
            RepoMapTool::parse_max_files(&serde_json::json!({"max_files": "abc"})),
            DEFAULT_MAX_FILES
        );
    }

    #[test]
    fn test_parse_max_files_handles_negative() {
        // serde_json::Value::as_u64() returns None for negative numbers
        assert_eq!(
            RepoMapTool::parse_max_files(&serde_json::json!({"max_files": -5})),
            DEFAULT_MAX_FILES
        );
    }

    // ── Additional extraction tests ──────────────────────────────

    #[test]
    fn test_extract_impl_without_trait() {
        let source = "struct Foo;\nimpl Foo { fn bar(&self) {} }";
        let symbols = extract(source, Depth::Signatures);
        let impl_sym = symbols.iter().find(|s| s.kind == SymbolKind::Impl).unwrap();
        assert_eq!(impl_sym.name, "Foo");
        assert!(
            !impl_sym.name.contains("for"),
            "plain impl should not contain 'for': {}",
            impl_sym.name
        );
    }

    #[test]
    fn test_extract_trait_method_doc_comments_full_depth() {
        let source =
            "pub trait Foo {\n    /// Method doc\n    fn bar(&self);\n    fn baz(&self);\n}";
        let symbols = extract(source, Depth::Full);
        assert_eq!(symbols.len(), 1);
        assert_eq!(symbols[0].children.len(), 2);
        let bar = &symbols[0].children[0];
        assert!(
            bar.doc_comment.is_some(),
            "method bar should have doc comment at Full depth"
        );
        assert!(bar.doc_comment.as_ref().unwrap().contains("Method doc"));
        let baz = &symbols[0].children[1];
        assert!(
            baz.doc_comment.is_none(),
            "method baz should have no doc comment"
        );
    }

    // ── Additional integration tests ─────────────────────────────

    #[tokio::test]
    async fn test_repo_map_file_path_rejected() {
        let h = TestHarness::new();
        std::fs::write(h.path().join("lib.rs"), "pub fn foo() {}").unwrap();

        let tool = RepoMapTool::new();
        let err = tool
            .execute(serde_json::json!({"path": "lib.rs"}), &h.context())
            .await
            .unwrap_err();
        match err {
            ToolError::InvalidInput { tool, reason } => {
                assert_eq!(tool, "repo_map");
                assert!(
                    reason.contains("not a directory"),
                    "expected 'not a directory' error, got: {reason}"
                );
            }
            other => panic!("expected InvalidInput, got: {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_repo_map_rust_language_accepted() {
        let h = TestHarness::new();
        std::fs::write(h.path().join("lib.rs"), "pub fn foo() {}").unwrap();

        let tool = RepoMapTool::new();
        let output = tool
            .execute(serde_json::json!({"language": "rust"}), &h.context())
            .await
            .unwrap();
        assert!(matches!(output.outcome, ToolOutcome::Success));
        assert!(output.content.contains("pub fn foo()"));
    }

    #[test]
    fn test_discover_skips_symlinks() {
        let h = TestHarness::new();
        std::fs::write(h.path().join("real.rs"), "pub fn real() {}").unwrap();

        // Create a symlinked directory pointing outside the sandbox
        let outside = tempfile::tempdir().unwrap();
        std::fs::write(outside.path().join("escape.rs"), "pub fn escape() {}").unwrap();

        #[cfg(unix)]
        {
            std::os::unix::fs::symlink(outside.path(), h.path().join("linked_dir")).unwrap();
            std::os::unix::fs::symlink(
                outside.path().join("escape.rs"),
                h.path().join("linked_file.rs"),
            )
            .unwrap();
        }

        let files = discover_files(h.path(), "rs", 100);
        // Only real.rs should appear — symlinks are not followed
        assert_eq!(files.len(), 1);
        assert!(files[0].to_string_lossy().contains("real.rs"));
    }

    #[test]
    fn test_truncation_safe_on_multibyte_utf8() {
        // Build a repo map output containing multi-byte characters
        let sym = Symbol {
            signature: "pub fn café() -> Ünïcödé".to_owned(),
            doc_comment: None,
            children: vec![],
            #[cfg(test)]
            kind: SymbolKind::Function,
            #[cfg(test)]
            name: "café".to_owned(),
            #[cfg(test)]
            visibility: Some("pub".to_owned()),
        };
        let file_symbols = vec![(PathBuf::from("/root/src/lib.rs"), vec![sym])];
        let output = format_repo_map(&file_symbols, Path::new("/root"));
        // Verify the output is valid UTF-8 (it always is since String)
        assert!(output.is_char_boundary(0));
        // Verify truncation at arbitrary byte offsets doesn't panic
        for i in 0..output.len() {
            let mut s = output.clone();
            let mut trunc_at = i;
            while !s.is_char_boundary(trunc_at) && trunc_at > 0 {
                trunc_at -= 1;
            }
            s.truncate(trunc_at);
            // Should not panic and should be valid UTF-8
            assert!(s.len() <= i);
        }
    }

    // ── Phase 6: Property tests ───────────────────────────────────

    mod proptests {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            #[test]
            fn discover_never_exceeds_max(max in 1usize..50) {
                let h = TestHarness::new();
                for i in 0..60 {
                    std::fs::write(
                        h.path().join(format!("f{i:03}.rs")),
                        "fn f() {}",
                    ).unwrap();
                }
                let files = discover_files(h.path(), "rs", max);
                prop_assert!(files.len() <= max, "got {} files, max was {}", files.len(), max);
            }

            #[test]
            fn parser_never_panics(source in "\\PC{0,2000}") {
                let mut parser = tree_sitter::Parser::new();
                parser.set_language(&tree_sitter_rust::LANGUAGE.into()).unwrap();
                let bytes = source.as_bytes();
                if let Some(tree) = parser.parse(bytes, None) {
                    let mapper = RustMapper;
                    // Test all depth variants, not just Signatures
                    let _ = mapper.extract_symbols(&tree, bytes, Depth::Outline);
                    let _ = mapper.extract_symbols(&tree, bytes, Depth::Signatures);
                    let _ = mapper.extract_symbols(&tree, bytes, Depth::Full);
                }
            }
        }
    }

    // -- Helper for ranked mode tests --

    fn make_context(root: &Path) -> ToolContext<'_> {
        // Leak a session ID so the borrow lives long enough.
        // This is fine in test code.
        let session_id: &'static SessionId =
            Box::leak(Box::new(SessionId::from_string("test-session")));
        let caps: &'static [Capability] = &[Capability::FileRead];
        let dirs: &'static [PathBuf] = &[];
        ToolContext {
            session_id,
            sandbox_root: root,
            granted_capabilities: caps,
            allowed_directories: dirs,
            knowledge_store: None,
            memory: None,
        }
    }

    // -- Mode parsing --

    #[test]
    fn test_parse_mode_structure() {
        assert_eq!(parse_mode("structure").unwrap(), Mode::Structure);
    }

    #[test]
    fn test_parse_mode_ranked() {
        assert_eq!(parse_mode("ranked").unwrap(), Mode::Ranked);
    }

    #[test]
    fn test_parse_mode_invalid() {
        assert!(parse_mode("invalid").is_err());
    }

    // -- Ranked mode integration --

    #[tokio::test]
    async fn test_ranked_mode_produces_ranked_output() {
        let tmp = tempfile::TempDir::new().unwrap();
        let root = tmp.path();

        // Create files where types.rs defines `Config` which is used by both
        // agent.rs and tools.rs — types.rs should rank highest.
        let types_dir = root.join("src");
        std::fs::create_dir_all(&types_dir).unwrap();

        std::fs::write(
            types_dir.join("types.rs"),
            "pub struct Config { pub name: String }\npub struct SessionId(String);\n",
        )
        .unwrap();
        std::fs::write(
            types_dir.join("agent.rs"),
            "use crate::types::Config;\nfn run(cfg: Config) { }\nfn process(sid: SessionId) { }\n",
        )
        .unwrap();
        std::fs::write(
            types_dir.join("tools.rs"),
            "use crate::types::Config;\nfn execute(cfg: Config) -> bool { true }\n",
        )
        .unwrap();

        let tool = RepoMapTool::new();
        let input = serde_json::json!({
            "path": "src",
            "mode": "ranked"
        });
        let ctx = make_context(root);
        let result = tool.execute(input, &ctx).await.unwrap();

        assert_eq!(result.outcome, ToolOutcome::Success);
        assert!(result.content.contains("[RANKED REPO MAP"));
        assert!(result.content.contains("Ranked"));
    }

    #[tokio::test]
    async fn test_structure_mode_backward_compatible() {
        let tmp = tempfile::TempDir::new().unwrap();
        let root = tmp.path();
        std::fs::write(root.join("lib.rs"), "pub fn hello() { }").unwrap();

        let tool = RepoMapTool::new();
        let input = serde_json::json!({ "mode": "structure" });
        let ctx = make_context(root);
        let result = tool.execute(input, &ctx).await.unwrap();

        assert_eq!(result.outcome, ToolOutcome::Success);
        // Structure mode should NOT have the ranked header.
        assert!(!result.content.contains("[RANKED REPO MAP"));
        assert!(result.content.contains("Mapped"));
    }

    #[tokio::test]
    async fn test_default_mode_is_structure() {
        let tmp = tempfile::TempDir::new().unwrap();
        let root = tmp.path();
        std::fs::write(root.join("lib.rs"), "pub fn hello() { }").unwrap();

        let tool = RepoMapTool::new();
        let input = serde_json::json!({});
        let ctx = make_context(root);
        let result = tool.execute(input, &ctx).await.unwrap();

        // Default should be structure mode.
        assert!(result.content.contains("Mapped"));
    }

    #[tokio::test]
    async fn test_ranked_mode_token_budget_truncates() {
        let tmp = tempfile::TempDir::new().unwrap();
        let root = tmp.path();

        // Create enough files to exceed a tiny budget.
        for i in 0..10 {
            std::fs::write(
                root.join(format!("file{i}.rs")),
                format!("pub fn func_{i}() {{ }}\npub struct Type{i};\n"),
            )
            .unwrap();
        }

        let tool = RepoMapTool::new();
        let input = serde_json::json!({
            "mode": "ranked",
            "token_budget": 200
        });
        let ctx = make_context(root);
        let result = tool.execute(input, &ctx).await.unwrap();

        assert_eq!(result.outcome, ToolOutcome::Success);
        // With a 200-char budget, not all 10 files should appear.
        let file_count: usize = result.content.matches("## file").count();
        assert!(file_count < 10, "should truncate: found {file_count} files");
    }

    #[tokio::test]
    async fn test_ranked_mode_invalid_returns_error() {
        let tmp = tempfile::TempDir::new().unwrap();
        let root = tmp.path();
        std::fs::write(root.join("lib.rs"), "fn x() {}").unwrap();

        let tool = RepoMapTool::new();
        let input = serde_json::json!({ "mode": "invalid_mode" });
        let ctx = make_context(root);
        let result = tool.execute(input, &ctx).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_ranked_mode_heavily_referenced_file_first() {
        let tmp = tempfile::TempDir::new().unwrap();
        let root = tmp.path();

        // core.rs defines `CoreType` used by a.rs, b.rs, c.rs.
        // leaf.rs defines `Leaf` used by nobody.
        std::fs::write(root.join("core.rs"), "pub struct CoreType;\n").unwrap();
        std::fs::write(root.join("leaf.rs"), "pub struct Leaf;\n").unwrap();
        std::fs::write(
            root.join("a.rs"),
            "use crate::core::CoreType;\nfn use_core(c: CoreType) {}\n",
        )
        .unwrap();
        std::fs::write(
            root.join("b.rs"),
            "use crate::core::CoreType;\nfn other(c: CoreType) {}\n",
        )
        .unwrap();
        std::fs::write(
            root.join("c.rs"),
            "use crate::core::CoreType;\nfn another(c: CoreType) {}\n",
        )
        .unwrap();

        let tool = RepoMapTool::new();
        let input = serde_json::json!({ "mode": "ranked" });
        let ctx = make_context(root);
        let result = tool.execute(input, &ctx).await.unwrap();

        // core.rs should appear before leaf.rs in ranked output.
        let core_pos = result.content.find("core.rs");
        let leaf_pos = result.content.find("leaf.rs");
        assert!(
            core_pos.is_some() && leaf_pos.is_some(),
            "both files should appear in output"
        );
        assert!(
            core_pos.unwrap() < leaf_pos.unwrap(),
            "core.rs should rank before leaf.rs"
        );
    }
}
