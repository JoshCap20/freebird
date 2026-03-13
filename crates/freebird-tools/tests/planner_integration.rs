//! Integration tests for the `plan_edits` tool.
//!
//! Uses real Rust files in a tempdir to verify AST-based auto-detection,
//! manual mode, cycle detection, mixed explicit+inferred dependencies,
//! and path traversal rejection.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::indexing_slicing)]

use std::path::Path;

use freebird_traits::id::SessionId;
use freebird_traits::tool::{Capability, Tool, ToolContext};

/// Get the `plan_edits` tool from the planner module.
fn plan_edits_tool() -> Box<dyn Tool> {
    freebird_tools::planner::planner_tools()
        .into_iter()
        .next()
        .expect("planner_tools should return at least one tool")
}

/// Integration test harness: owns tempdir, provides zero-boilerplate context.
struct Harness {
    _tmp: tempfile::TempDir,
    sandbox: std::path::PathBuf,
    session_id: SessionId,
    caps: Vec<Capability>,
}

impl Harness {
    fn new() -> Self {
        let tmp = tempfile::tempdir().expect("failed to create temp dir");
        let sandbox = tmp.path().canonicalize().expect("failed to canonicalize");
        Self {
            _tmp: tmp,
            sandbox,
            session_id: SessionId::from_string("test-session"),
            caps: vec![Capability::FileRead],
        }
    }

    fn path(&self) -> &Path {
        &self.sandbox
    }

    fn context(&self) -> ToolContext<'_> {
        ToolContext {
            session_id: &self.session_id,
            sandbox_root: &self.sandbox,
            granted_capabilities: &self.caps,
            allowed_directories: &[],
            knowledge_store: None,
            memory: None,
        }
    }
}

#[tokio::test]
async fn test_auto_detect_with_real_files() {
    let h = Harness::new();

    // Create a trait file
    std::fs::write(
        h.path().join("my_trait.rs"),
        "pub trait MyTrait {\n    fn do_thing(&self);\n}\n",
    )
    .unwrap();

    // Create a struct file that references the trait
    std::fs::write(
        h.path().join("my_struct.rs"),
        "use crate::MyTrait;\npub struct MyStruct;\nimpl MyTrait for MyStruct {\n    fn do_thing(&self) {}\n}\n",
    )
    .unwrap();

    let tool = plan_edits_tool();
    let ctx = h.context();

    let input = serde_json::json!({
        "changes": [
            { "id": 0, "file_path": "my_trait.rs", "description": "Add trait" },
            { "id": 1, "file_path": "my_struct.rs", "description": "Add struct impl" }
        ],
        "auto_detect": true
    });

    let result = tool.execute(input, &ctx).await.unwrap();
    assert_eq!(
        result.outcome,
        freebird_traits::tool::ToolOutcome::Success,
        "tool should succeed: {}",
        result.content
    );

    // Parse the output and verify ordering
    let plan: serde_json::Value = serde_json::from_str(&result.content).unwrap();
    let ordered = plan["ordered_changes"].as_array().unwrap();
    assert_eq!(ordered.len(), 2);

    // Trait file should come before struct file (trait_definition < type_definition)
    let first_kind = ordered[0]["change_kind"].as_str().unwrap();
    assert_eq!(first_kind, "trait_definition");
}

#[tokio::test]
async fn test_manual_mode_with_explicit_deps() {
    let h = Harness::new();
    let tool = plan_edits_tool();
    let ctx = h.context();

    let input = serde_json::json!({
        "changes": [
            {
                "id": 0,
                "file_path": "types.rs",
                "description": "Define types",
                "depends_on": [],
                "change_kind": "type_definition",
                "crate_kind": "library"
            },
            {
                "id": 1,
                "file_path": "impl.rs",
                "description": "Implement types",
                "depends_on": [0],
                "change_kind": "implementation",
                "crate_kind": "library"
            },
            {
                "id": 2,
                "file_path": "test.rs",
                "description": "Test implementation",
                "depends_on": [1],
                "change_kind": "test",
                "crate_kind": "library"
            }
        ],
        "auto_detect": false
    });

    let result = tool.execute(input, &ctx).await.unwrap();
    assert_eq!(
        result.outcome,
        freebird_traits::tool::ToolOutcome::Success,
        "tool should succeed: {}",
        result.content
    );

    let plan: serde_json::Value = serde_json::from_str(&result.content).unwrap();
    let ordered = plan["ordered_changes"].as_array().unwrap();
    assert_eq!(ordered.len(), 3);

    // types -> impl -> test
    assert_eq!(ordered[0]["id"].as_u64().unwrap(), 0);
    assert_eq!(ordered[1]["id"].as_u64().unwrap(), 1);
    assert_eq!(ordered[2]["id"].as_u64().unwrap(), 2);
}

#[tokio::test]
async fn test_cycle_returns_error_outcome() {
    let h = Harness::new();
    let tool = plan_edits_tool();
    let ctx = h.context();

    let input = serde_json::json!({
        "changes": [
            {
                "id": 0,
                "file_path": "a.rs",
                "description": "File A",
                "depends_on": [1],
                "change_kind": "consumer",
                "crate_kind": "library"
            },
            {
                "id": 1,
                "file_path": "b.rs",
                "description": "File B",
                "depends_on": [0],
                "change_kind": "consumer",
                "crate_kind": "library"
            }
        ],
        "auto_detect": false
    });

    let result = tool.execute(input, &ctx).await.unwrap();
    assert_eq!(
        result.outcome,
        freebird_traits::tool::ToolOutcome::Error,
        "cycle should produce error outcome"
    );
    assert!(
        result.content.contains("cycle"),
        "error should mention cycle: {}",
        result.content
    );
}

#[tokio::test]
async fn test_mixed_explicit_and_inferred_deps() {
    let h = Harness::new();

    // Create files: trait_file defines `Foo`, consumer references `Foo`
    std::fs::write(
        h.path().join("traits.rs"),
        "pub trait Foo {\n    fn bar(&self);\n}\n",
    )
    .unwrap();

    std::fs::write(
        h.path().join("consumer.rs"),
        "fn uses_foo(f: &dyn Foo) {\n    f.bar();\n}\n",
    )
    .unwrap();

    // Third file with only explicit dep on consumer
    std::fs::write(h.path().join("my_test.rs"), "// test file\n").unwrap();

    let tool = plan_edits_tool();
    let ctx = h.context();

    let input = serde_json::json!({
        "changes": [
            { "id": 0, "file_path": "traits.rs", "description": "Define trait" },
            { "id": 1, "file_path": "consumer.rs", "description": "Use trait" },
            {
                "id": 2,
                "file_path": "my_test.rs",
                "description": "Test",
                "depends_on": [1],
                "change_kind": "test"
            }
        ],
        "auto_detect": true
    });

    let result = tool.execute(input, &ctx).await.unwrap();
    assert_eq!(
        result.outcome,
        freebird_traits::tool::ToolOutcome::Success,
        "tool should succeed: {}",
        result.content
    );

    let plan: serde_json::Value = serde_json::from_str(&result.content).unwrap();
    let ordered = plan["ordered_changes"].as_array().unwrap();
    assert_eq!(ordered.len(), 3);

    // The test file (id=2) should come after consumer (id=1) due to explicit dep
    let test_pos = ordered
        .iter()
        .position(|c| c["id"].as_u64().unwrap() == 2)
        .unwrap();
    let consumer_pos = ordered
        .iter()
        .position(|c| c["id"].as_u64().unwrap() == 1)
        .unwrap();
    assert!(
        consumer_pos < test_pos,
        "consumer should come before test (consumer={consumer_pos}, test={test_pos})"
    );
}

#[tokio::test]
async fn test_path_traversal_rejected() {
    let h = Harness::new();
    let tool = plan_edits_tool();
    let ctx = h.context();

    let input = serde_json::json!({
        "changes": [
            {
                "id": 0,
                "file_path": "../../etc/passwd",
                "description": "Malicious path",
                "depends_on": [],
                "change_kind": "consumer",
                "crate_kind": "library"
            }
        ],
        "auto_detect": false
    });

    let result = tool.execute(input, &ctx).await;
    assert!(result.is_err(), "path traversal should be rejected");
}
