//! Input validation and syntax validation (tree-sitter) for the edit tool.

use std::fmt::Write;
use std::path::Path;

use freebird_traits::tool::ToolError;

// ── Syntax Validation ────────────────────────────────────────────

/// A syntax error found by tree-sitter.
struct SyntaxError {
    /// 1-indexed line number.
    line: usize,
    /// 1-indexed column number.
    column: usize,
    /// Node kind: `"ERROR"` or `"MISSING"`.
    kind: &'static str,
}

/// Maximum syntax errors to collect before stopping the walk.
const MAX_SYNTAX_ERRORS: usize = 5;

/// Resolve a file extension to a tree-sitter language.
///
/// Returns `None` for unsupported extensions — validation is skipped.
fn language_for_path(path: &Path) -> Option<tree_sitter::Language> {
    let ext = path.extension().and_then(|e| e.to_str())?;
    match ext {
        "rs" => Some(tree_sitter_rust::LANGUAGE.into()),
        _ => None,
    }
}

/// Walk the tree collecting ERROR and MISSING nodes via depth-first traversal.
fn collect_syntax_errors(tree: &tree_sitter::Tree) -> Vec<SyntaxError> {
    let mut errors = Vec::with_capacity(MAX_SYNTAX_ERRORS);
    let mut cursor = tree.walk();

    loop {
        let node = cursor.node();
        if node.is_error() || node.is_missing() {
            let pos = node.start_position();
            errors.push(SyntaxError {
                line: pos.row + 1,
                column: pos.column + 1,
                kind: if node.is_error() { "ERROR" } else { "MISSING" },
            });
            if errors.len() >= MAX_SYNTAX_ERRORS {
                break;
            }
        }

        // Depth-first: try child → sibling → parent's sibling
        if cursor.goto_first_child() {
            continue;
        }
        while !cursor.goto_next_sibling() {
            if !cursor.goto_parent() {
                return errors;
            }
        }
    }

    errors
}

/// Validate that edited content has no syntax errors for supported languages.
///
/// Returns `Ok(())` for unsupported file types (validation skipped).
pub(super) fn validate_syntax(path: &Path, content: &str) -> Result<(), ToolError> {
    let Some(language) = language_for_path(path) else {
        return Ok(());
    };

    let mut parser = tree_sitter::Parser::new();
    parser
        .set_language(&language)
        .map_err(|_| ToolError::ExecutionFailed {
            tool: super::SearchReplaceEditTool::NAME.into(),
            reason: "failed to initialize syntax parser".into(),
        })?;

    let Some(tree) = parser.parse(content.as_bytes(), None) else {
        // parse() returns None on cancellation — treat as skip
        return Ok(());
    };

    let errors = collect_syntax_errors(&tree);
    if errors.is_empty() {
        return Ok(());
    }

    let error_count = errors.len();
    let mut msg = String::from("Edit rejected — syntax errors detected (file not modified):\n");
    for err in &errors {
        let _ = writeln!(msg, "  line {}:{} — {}", err.line, err.column, err.kind);
    }

    tracing::debug!(
        path = %path.display(),
        error_count,
        "syntax validation rejected edit"
    );

    Err(ToolError::ExecutionFailed {
        tool: super::SearchReplaceEditTool::NAME.into(),
        reason: msg,
    })
}

// ── Checkpoint Name Validation ──────────────────────────────────

/// Validate a checkpoint name: alphanumeric, hyphens, underscores, 1–64 chars.
pub(super) fn validate_checkpoint_name(name: &str, tool_name: &str) -> Result<(), ToolError> {
    if name.is_empty() || name.len() > 64 {
        return Err(ToolError::InvalidInput {
            tool: tool_name.into(),
            reason: "checkpoint name must be 1–64 characters".into(),
        });
    }
    if !name
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
    {
        return Err(ToolError::InvalidInput {
            tool: tool_name.into(),
            reason: "checkpoint name must contain only alphanumeric characters, hyphens, and underscores".into(),
        });
    }
    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic, clippy::indexing_slicing)]
mod tests {
    use super::*;

    // ── Syntax validation unit tests ─────────────────────────────

    #[test]
    fn test_language_for_path_rs() {
        let path = Path::new("src/main.rs");
        assert!(language_for_path(path).is_some());
    }

    #[test]
    fn test_language_for_path_unknown() {
        assert!(language_for_path(Path::new("file.txt")).is_none());
        assert!(language_for_path(Path::new("file.py")).is_none());
        assert!(language_for_path(Path::new("file.js")).is_none());
        assert!(language_for_path(Path::new("no_extension")).is_none());
    }

    #[test]
    fn test_validate_syntax_valid_rust() {
        let path = Path::new("test.rs");
        let content = "fn main() {\n    let x = 42;\n}\n";
        assert!(validate_syntax(path, content).is_ok());
    }

    #[test]
    fn test_validate_syntax_invalid_rust() {
        let path = Path::new("test.rs");
        let content = "fn main( {\n    let x = 42;\n}\n";
        let err = validate_syntax(path, content).unwrap_err();
        match err {
            ToolError::ExecutionFailed { reason, .. } => {
                assert!(reason.contains("syntax errors detected"), "got: {reason}");
                assert!(reason.contains("file not modified"), "got: {reason}");
            }
            other => panic!("expected ExecutionFailed, got: {other:?}"),
        }
    }

    #[test]
    fn test_validate_syntax_unknown_extension() {
        let path = Path::new("readme.txt");
        let content = "this is not valid rust at all fn {{{";
        assert!(validate_syntax(path, content).is_ok());
    }

    #[test]
    fn test_validate_syntax_empty_content() {
        let path = Path::new("empty.rs");
        assert!(validate_syntax(path, "").is_ok());
    }

    #[test]
    fn test_collect_syntax_errors_multiple() {
        let mut parser = tree_sitter::Parser::new();
        parser
            .set_language(&tree_sitter_rust::LANGUAGE.into())
            .unwrap();
        // Multiple syntax errors: broken function signatures
        let content = "fn a( { } fn b( { } fn c( { } fn d( { } fn e( { } fn f( { }";
        let tree = parser.parse(content.as_bytes(), None).unwrap();
        let errors = collect_syntax_errors(&tree);
        assert!(
            !errors.is_empty(),
            "should detect syntax errors in broken code"
        );
        assert!(
            errors.len() <= MAX_SYNTAX_ERRORS,
            "should cap at {MAX_SYNTAX_ERRORS}, got {}",
            errors.len()
        );
    }

    #[test]
    fn test_error_message_includes_line_column() {
        let path = Path::new("test.rs");
        // Missing closing paren on line 2
        let content = "fn main() {\n    let x = foo(;\n}\n";
        let err = validate_syntax(path, content).unwrap_err();
        match err {
            ToolError::ExecutionFailed { reason, .. } => {
                // Should contain "line N:M" format
                assert!(
                    reason.contains("line "),
                    "error should include line numbers, got: {reason}"
                );
            }
            other => panic!("expected ExecutionFailed, got: {other:?}"),
        }
    }

    #[test]
    fn test_syntax_validation_latency() {
        // Generate a large valid Rust file
        let mut content = String::with_capacity(300_000);
        for i in 0..10_000 {
            use std::fmt::Write;
            let _ = writeln!(content, "fn func_{i}() {{ let _x = {i}; }}");
        }

        let path = Path::new("big.rs");
        let start = std::time::Instant::now();
        let result = validate_syntax(path, &content);
        let elapsed = start.elapsed();

        assert!(result.is_ok(), "valid large file should pass");
        // In release mode tree-sitter parses this in <10ms.
        // Debug builds are ~20x slower, and CI/parallel workloads add more.
        assert!(
            elapsed.as_millis() < 2000,
            "validation took {}ms, expected <2000ms (debug)",
            elapsed.as_millis()
        );
    }

    // ── Property-based tests ─────────────────────────────────────

    mod proptests {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            /// validate_syntax never panics on arbitrary input.
            #[test]
            fn validate_syntax_never_panics(content in "\\PC{0,2000}") {
                let path = Path::new("test.rs");
                // Must not panic — may return Ok or Err, both are fine
                let _ = validate_syntax(path, &content);
            }
        }
    }
}
