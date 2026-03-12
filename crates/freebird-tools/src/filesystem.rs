//! Sandboxed filesystem tools: `read_file`, `write_file`, `list_directory`.
//!
//! All path validation flows through `TaintedToolInput::extract_path()` /
//! `extract_path_for_creation()` — tools never touch raw paths. File content
//! for writes uses `TaintedToolInput::extract_file_content()` to bridge the
//! `pub(crate)` taint boundary.

use async_trait::async_trait;
use tokio::io::AsyncReadExt;

use freebird_security::taint::TaintedToolInput;
use freebird_traits::tool::{
    Capability, RiskLevel, SideEffects, Tool, ToolContext, ToolError, ToolInfo, ToolOutcome,
    ToolOutput,
};

/// Maximum file size `read_file` will return (10 MiB).
///
/// Prevents the LLM context window from being flooded with a single
/// file read. Matches the egress `max_body_bytes` default.
const MAX_READ_FILE_BYTES: usize = 10 * 1024 * 1024;

/// Maximum number of entries `list_directory` will return.
///
/// Prevents context window flooding from directories with thousands of files.
const MAX_DIR_ENTRIES: usize = 1000;

/// Returns all filesystem tools as trait objects.
///
/// The sandbox is provided at execution time via `ToolContext`, not at
/// construction — tools are stateless and shared across sessions.
#[must_use]
pub fn filesystem_tools() -> Vec<Box<dyn Tool>> {
    vec![
        Box::new(ReadFileTool::new()),
        Box::new(WriteFileTool::new()),
        Box::new(ListDirectoryTool::new()),
    ]
}

// ── ReadFileTool ────────────────────────────────────────────────────

struct ReadFileTool {
    info: ToolInfo,
}

impl ReadFileTool {
    const NAME: &str = "read_file";

    fn new() -> Self {
        Self {
            info: ToolInfo {
                name: Self::NAME.into(),
                description: "Read the contents of a file as UTF-8 text. Maximum 10 MiB.".into(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Relative path within the sandbox, or absolute path within an allowed directory"
                        }
                    },
                    "required": ["path"]
                }),
                required_capability: Capability::FileRead,
                risk_level: RiskLevel::Low,
                side_effects: SideEffects::None,
            },
        }
    }
}

#[async_trait]
impl Tool for ReadFileTool {
    fn info(&self) -> &ToolInfo {
        &self.info
    }

    async fn execute(
        &self,
        input: serde_json::Value,
        context: &ToolContext<'_>,
    ) -> Result<ToolOutput, ToolError> {
        let tainted = TaintedToolInput::new(input);
        let safe_path = tainted
            .extract_path_multi_root("path", context.sandbox_root, context.allowed_directories)
            .map_err(|e| ToolError::InvalidInput {
                tool: Self::NAME.into(),
                reason: e.to_string(),
            })?;

        // Capped read: open → read up to limit+1 → reject if over limit.
        // Single atomic operation avoids TOCTOU between metadata and read.
        let file = tokio::fs::File::open(safe_path.as_path())
            .await
            .map_err(|e| ToolError::ExecutionFailed {
                tool: Self::NAME.into(),
                reason: e.to_string(),
            })?;

        let cap = MAX_READ_FILE_BYTES + 1;
        let mut buf = Vec::with_capacity(cap.min(8 * 1024));
        file.take(cap as u64)
            .read_to_end(&mut buf)
            .await
            .map_err(|e| ToolError::ExecutionFailed {
                tool: Self::NAME.into(),
                reason: e.to_string(),
            })?;

        if buf.len() > MAX_READ_FILE_BYTES {
            return Err(ToolError::ExecutionFailed {
                tool: Self::NAME.into(),
                reason: format!("file exceeds {MAX_READ_FILE_BYTES} byte limit"),
            });
        }

        let file_content = String::from_utf8(buf).map_err(|_| ToolError::ExecutionFailed {
            tool: Self::NAME.into(),
            reason: "file is not valid UTF-8".into(),
        })?;

        Ok(ToolOutput {
            content: file_content,
            outcome: ToolOutcome::Success,
            metadata: None,
        })
    }
}

// ── WriteFileTool ───────────────────────────────────────────────────

struct WriteFileTool {
    info: ToolInfo,
}

impl WriteFileTool {
    const NAME: &str = "write_file";

    fn new() -> Self {
        Self {
            info: ToolInfo {
                name: Self::NAME.into(),
                description:
                    "Write text content to a file. Creates the file if it doesn't exist. Uses atomic write (temp file + rename)."
                        .into(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Relative path within the sandbox, or absolute path within an allowed directory"
                        },
                        "content": {
                            "type": "string",
                            "description": "Text content to write"
                        }
                    },
                    "required": ["path", "content"]
                }),
                required_capability: Capability::FileWrite,
                risk_level: RiskLevel::Medium,
                side_effects: SideEffects::HasSideEffects,
            },
        }
    }
}

#[async_trait]
impl Tool for WriteFileTool {
    fn info(&self) -> &ToolInfo {
        &self.info
    }

    async fn execute(
        &self,
        input: serde_json::Value,
        context: &ToolContext<'_>,
    ) -> Result<ToolOutput, ToolError> {
        let tainted = TaintedToolInput::new(input);
        let safe_path = tainted
            .extract_path_for_creation_multi_root(
                "path",
                context.sandbox_root,
                context.allowed_directories,
            )
            .map_err(|e| ToolError::InvalidInput {
                tool: Self::NAME.into(),
                reason: e.to_string(),
            })?;
        let file_content =
            tainted
                .extract_file_content("content")
                .map_err(|e| ToolError::InvalidInput {
                    tool: Self::NAME.into(),
                    reason: e.to_string(),
                })?;

        // Atomic write: temp file + rename.
        // Unique suffix avoids collisions from concurrent writes.
        //
        // `file_name()` is always `Some` for a SafeFilePath (validated non-empty,
        // no trailing separator). `to_str()` could fail on non-UTF-8 OS strings,
        // but SafeFilePath is constructed from a UTF-8 `Tainted` input, so the
        // path components are always valid UTF-8. The fallback is a harmless
        // default that only affects the temp file name.
        let file_name = safe_path
            .as_path()
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("file");
        let tmp_path = safe_path.as_path().with_file_name(format!(
            ".{}.{}.tmp",
            file_name,
            std::process::id(),
        ));

        tokio::fs::write(&tmp_path, file_content.as_str())
            .await
            .map_err(|e| ToolError::ExecutionFailed {
                tool: Self::NAME.into(),
                reason: e.to_string(),
            })?;

        if let Err(e) = tokio::fs::rename(&tmp_path, safe_path.as_path()).await {
            // Clean up temp file on rename failure
            let _ = tokio::fs::remove_file(&tmp_path).await;
            return Err(ToolError::ExecutionFailed {
                tool: Self::NAME.into(),
                reason: e.to_string(),
            });
        }

        // Report relative path — don't leak sandbox internals to LLM
        let relative = safe_path
            .as_path()
            .strip_prefix(safe_path.root())
            .unwrap_or(safe_path.as_path());

        Ok(ToolOutput {
            content: format!(
                "Wrote {} bytes to {}",
                file_content.len(),
                relative.display()
            ),
            outcome: ToolOutcome::Success,
            metadata: None,
        })
    }
}

// ── ListDirectoryTool ───────────────────────────────────────────────

struct ListDirectoryTool {
    info: ToolInfo,
}

impl ListDirectoryTool {
    const NAME: &str = "list_directory";

    fn new() -> Self {
        Self {
            info: ToolInfo {
                name: Self::NAME.into(),
                description:
                    "List files and directories. Returns up to 1000 entries sorted alphabetically, each prefixed with type (file/dir/symlink)."
                        .into(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Relative path to directory within the sandbox"
                        }
                    },
                    "required": ["path"]
                }),
                required_capability: Capability::FileRead,
                risk_level: RiskLevel::Low,
                side_effects: SideEffects::None,
            },
        }
    }
}

#[async_trait]
impl Tool for ListDirectoryTool {
    fn info(&self) -> &ToolInfo {
        &self.info
    }

    async fn execute(
        &self,
        input: serde_json::Value,
        context: &ToolContext<'_>,
    ) -> Result<ToolOutput, ToolError> {
        let tainted = TaintedToolInput::new(input);
        let safe_path = tainted
            .extract_path_multi_root("path", context.sandbox_root, context.allowed_directories)
            .map_err(|e| ToolError::InvalidInput {
                tool: Self::NAME.into(),
                reason: e.to_string(),
            })?;

        let mut entries = Vec::new();
        let mut dir = tokio::fs::read_dir(safe_path.as_path())
            .await
            .map_err(|e| ToolError::ExecutionFailed {
                tool: Self::NAME.into(),
                reason: e.to_string(),
            })?;

        let mut truncated = false;
        while let Some(entry) = dir
            .next_entry()
            .await
            .map_err(|e| ToolError::ExecutionFailed {
                tool: Self::NAME.into(),
                reason: e.to_string(),
            })?
        {
            if entries.len() >= MAX_DIR_ENTRIES {
                truncated = true;
                break;
            }
            let file_type = entry.file_type().await.ok();
            let kind = match file_type {
                Some(ft) if ft.is_dir() => "dir",
                Some(ft) if ft.is_symlink() => "symlink",
                _ => "file",
            };
            entries.push(format!("{kind}\t{}", entry.file_name().to_string_lossy()));
        }

        entries.sort();
        if truncated {
            entries.push(format!("... truncated ({MAX_DIR_ENTRIES} entry limit)"));
        }

        Ok(ToolOutput {
            content: if entries.is_empty() {
                "(empty directory)".into()
            } else {
                entries.join("\n")
            },
            outcome: ToolOutcome::Success,
            metadata: None,
        })
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic, clippy::indexing_slicing)]
mod tests {
    use std::io::Write as _;
    use std::path::PathBuf;

    use freebird_traits::id::SessionId;
    use freebird_traits::tool::{Capability, Tool, ToolContext, ToolError};

    use super::*;

    /// Test harness that owns the temp directory, session ID, and capabilities,
    /// providing a zero-boilerplate `context()` method for tool tests.
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
            let sandbox = tmp.path().to_path_buf();
            Self {
                _tmp: tmp,
                sandbox,
                session_id: SessionId::from_string("test-session"),
                capabilities: vec![Capability::FileRead, Capability::FileWrite],
                allowed_directories: vec![],
            }
        }

        fn with_allowed_directories(mut self, dirs: Vec<PathBuf>) -> Self {
            self.allowed_directories = dirs;
            self
        }

        fn path(&self) -> &std::path::Path {
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

    // ── read_file tests ─────────────────────────────────────────

    #[tokio::test]
    async fn test_read_existing_file() {
        let h = TestHarness::new();
        std::fs::write(h.path().join("hello.txt"), "Hello, world!").unwrap();

        let tool = ReadFileTool::new();
        let output = tool
            .execute(serde_json::json!({"path": "hello.txt"}), &h.context())
            .await
            .unwrap();
        assert_eq!(output.content, "Hello, world!");
        assert!(matches!(output.outcome, ToolOutcome::Success));
    }

    #[tokio::test]
    async fn test_read_nonexistent_file() {
        let h = TestHarness::new();
        let tool = ReadFileTool::new();

        let err = tool
            .execute(serde_json::json!({"path": "nope.txt"}), &h.context())
            .await
            .unwrap_err();
        // extract_path calls SafeFilePath::from_tainted which requires the file to exist
        // for canonicalization — so this is InvalidInput (PathResolution)
        match err {
            ToolError::InvalidInput { tool, .. } | ToolError::ExecutionFailed { tool, .. } => {
                assert_eq!(tool, "read_file");
            }
            other => panic!("expected InvalidInput or ExecutionFailed, got: {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_read_path_traversal_rejected() {
        let h = TestHarness::new();
        let tool = ReadFileTool::new();

        let err = tool
            .execute(
                serde_json::json!({"path": "../../../etc/passwd"}),
                &h.context(),
            )
            .await
            .unwrap_err();
        match err {
            ToolError::InvalidInput { tool, .. } => assert_eq!(tool, "read_file"),
            other => panic!("expected InvalidInput, got: {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_read_missing_path_field() {
        let h = TestHarness::new();
        let tool = ReadFileTool::new();

        let err = tool
            .execute(serde_json::json!({}), &h.context())
            .await
            .unwrap_err();
        match err {
            ToolError::InvalidInput { tool, .. } => assert_eq!(tool, "read_file"),
            other => panic!("expected InvalidInput, got: {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_read_non_utf8_file() {
        let h = TestHarness::new();
        let file_path = h.path().join("binary.bin");
        {
            let mut f = std::fs::File::create(&file_path).unwrap();
            f.write_all(&[0xFF, 0xFE, 0x00, 0x01]).unwrap();
        }

        let tool = ReadFileTool::new();
        let err = tool
            .execute(serde_json::json!({"path": "binary.bin"}), &h.context())
            .await
            .unwrap_err();
        match err {
            ToolError::ExecutionFailed { tool, .. } => assert_eq!(tool, "read_file"),
            other => panic!("expected ExecutionFailed, got: {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_read_file_exceeds_size_limit() {
        let h = TestHarness::new();
        let file_path = h.path().join("huge.txt");
        {
            let f = std::fs::File::create(&file_path).unwrap();
            // Set file size to just over the limit without writing all bytes
            f.set_len((MAX_READ_FILE_BYTES + 1) as u64).unwrap();
        }

        let tool = ReadFileTool::new();
        let err = tool
            .execute(serde_json::json!({"path": "huge.txt"}), &h.context())
            .await
            .unwrap_err();
        match &err {
            ToolError::ExecutionFailed { reason, .. } => {
                assert!(
                    reason.contains("exceeds"),
                    "error should mention limit: {reason}"
                );
            }
            other => panic!("expected ExecutionFailed, got: {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_read_empty_file() {
        let h = TestHarness::new();
        std::fs::write(h.path().join("empty.txt"), "").unwrap();

        let tool = ReadFileTool::new();
        let output = tool
            .execute(serde_json::json!({"path": "empty.txt"}), &h.context())
            .await
            .unwrap();
        assert_eq!(output.content, "");
        assert!(matches!(output.outcome, ToolOutcome::Success));
    }

    #[tokio::test]
    async fn test_read_absolute_path_rejected() {
        let h = TestHarness::new();
        let tool = ReadFileTool::new();

        let err = tool
            .execute(serde_json::json!({"path": "/etc/passwd"}), &h.context())
            .await
            .unwrap_err();
        match err {
            ToolError::InvalidInput { tool, .. } => assert_eq!(tool, "read_file"),
            other => panic!("expected InvalidInput, got: {other:?}"),
        }
    }

    // ── write_file tests ────────────────────────────────────────

    #[tokio::test]
    async fn test_write_new_file() {
        let h = TestHarness::new();
        let tool = WriteFileTool::new();

        let output = tool
            .execute(
                serde_json::json!({"path": "new.txt", "content": "hello"}),
                &h.context(),
            )
            .await
            .unwrap();
        assert!(matches!(output.outcome, ToolOutcome::Success));

        let written = std::fs::read_to_string(h.path().join("new.txt")).unwrap();
        assert_eq!(written, "hello");
    }

    #[tokio::test]
    async fn test_write_overwrites_existing() {
        let h = TestHarness::new();
        std::fs::write(h.path().join("existing.txt"), "old content").unwrap();

        let tool = WriteFileTool::new();
        let output = tool
            .execute(
                serde_json::json!({"path": "existing.txt", "content": "new content"}),
                &h.context(),
            )
            .await
            .unwrap();
        assert!(matches!(output.outcome, ToolOutcome::Success));

        let written = std::fs::read_to_string(h.path().join("existing.txt")).unwrap();
        assert_eq!(written, "new content");
    }

    #[tokio::test]
    async fn test_write_path_traversal_rejected() {
        let h = TestHarness::new();
        let tool = WriteFileTool::new();

        let err = tool
            .execute(
                serde_json::json!({"path": "../../etc/evil", "content": "x"}),
                &h.context(),
            )
            .await
            .unwrap_err();
        match err {
            ToolError::InvalidInput { tool, .. } => assert_eq!(tool, "write_file"),
            other => panic!("expected InvalidInput, got: {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_write_missing_path_field() {
        let h = TestHarness::new();
        let tool = WriteFileTool::new();

        let err = tool
            .execute(serde_json::json!({"content": "x"}), &h.context())
            .await
            .unwrap_err();
        match err {
            ToolError::InvalidInput { tool, .. } => assert_eq!(tool, "write_file"),
            other => panic!("expected InvalidInput, got: {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_write_missing_content_field() {
        let h = TestHarness::new();
        let tool = WriteFileTool::new();

        let err = tool
            .execute(serde_json::json!({"path": "file.txt"}), &h.context())
            .await
            .unwrap_err();
        match err {
            ToolError::InvalidInput { tool, .. } => assert_eq!(tool, "write_file"),
            other => panic!("expected InvalidInput, got: {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_write_nonexistent_parent_dir() {
        let h = TestHarness::new();
        let tool = WriteFileTool::new();

        let err = tool
            .execute(
                serde_json::json!({"path": "nodir/file.txt", "content": "x"}),
                &h.context(),
            )
            .await
            .unwrap_err();
        match err {
            ToolError::InvalidInput { tool, .. } => assert_eq!(tool, "write_file"),
            other => panic!("expected InvalidInput, got: {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_write_output_reports_relative_path() {
        let h = TestHarness::new();
        let tool = WriteFileTool::new();

        let output = tool
            .execute(
                serde_json::json!({"path": "report.txt", "content": "data"}),
                &h.context(),
            )
            .await
            .unwrap();
        assert!(
            output.content.contains("report.txt"),
            "output should contain relative path: {}",
            output.content
        );
        let sandbox_str = h.path().to_string_lossy();
        assert!(
            !output.content.contains(sandbox_str.as_ref()),
            "output should NOT contain sandbox root: {}",
            output.content
        );
    }

    #[tokio::test]
    async fn test_write_no_orphaned_temp_on_success() {
        let h = TestHarness::new();
        let tool = WriteFileTool::new();

        tool.execute(
            serde_json::json!({"path": "clean.txt", "content": "data"}),
            &h.context(),
        )
        .await
        .unwrap();

        // No .tmp files should remain
        let entries: Vec<_> = std::fs::read_dir(h.path())
            .unwrap()
            .filter_map(Result::ok)
            .filter(|e| e.file_name().to_string_lossy().ends_with(".tmp"))
            .collect();
        assert!(
            entries.is_empty(),
            "no .tmp files should remain after successful write"
        );
    }

    // ── list_directory tests ────────────────────────────────────

    #[tokio::test]
    async fn test_list_directory_returns_sorted_entries() {
        let h = TestHarness::new();
        std::fs::write(h.path().join("banana.txt"), "").unwrap();
        std::fs::write(h.path().join("apple.txt"), "").unwrap();
        std::fs::create_dir(h.path().join("cherry_dir")).unwrap();

        let tool = ListDirectoryTool::new();
        let output = tool
            .execute(serde_json::json!({"path": "."}), &h.context())
            .await
            .unwrap();
        assert!(matches!(output.outcome, ToolOutcome::Success));

        let lines: Vec<&str> = output.content.lines().collect();
        assert_eq!(lines.len(), 3);
        // Sorted alphabetically by full string: "dir\t..." < "file\t..."
        assert_eq!(lines[0], "dir\tcherry_dir");
        assert_eq!(lines[1], "file\tapple.txt");
        assert_eq!(lines[2], "file\tbanana.txt");
    }

    #[tokio::test]
    async fn test_list_nonexistent_directory() {
        let h = TestHarness::new();
        let tool = ListDirectoryTool::new();

        let err = tool
            .execute(serde_json::json!({"path": "nonexistent"}), &h.context())
            .await
            .unwrap_err();
        match err {
            ToolError::InvalidInput { tool, .. } | ToolError::ExecutionFailed { tool, .. } => {
                assert_eq!(tool, "list_directory");
            }
            other => panic!("expected InvalidInput or ExecutionFailed, got: {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_list_file_not_dir() {
        let h = TestHarness::new();
        std::fs::write(h.path().join("afile.txt"), "content").unwrap();

        let tool = ListDirectoryTool::new();
        let err = tool
            .execute(serde_json::json!({"path": "afile.txt"}), &h.context())
            .await
            .unwrap_err();
        match err {
            ToolError::ExecutionFailed { tool, .. } => assert_eq!(tool, "list_directory"),
            other => panic!("expected ExecutionFailed, got: {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_list_path_traversal_rejected() {
        let h = TestHarness::new();
        let tool = ListDirectoryTool::new();

        let err = tool
            .execute(serde_json::json!({"path": "../../"}), &h.context())
            .await
            .unwrap_err();
        match err {
            ToolError::InvalidInput { tool, .. } => assert_eq!(tool, "list_directory"),
            other => panic!("expected InvalidInput, got: {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_list_empty_directory() {
        let h = TestHarness::new();
        std::fs::create_dir(h.path().join("empty")).unwrap();

        let tool = ListDirectoryTool::new();
        let output = tool
            .execute(serde_json::json!({"path": "empty"}), &h.context())
            .await
            .unwrap();
        assert_eq!(output.content, "(empty directory)");
        assert!(matches!(output.outcome, ToolOutcome::Success));
    }

    #[tokio::test]
    async fn test_write_empty_content() {
        let h = TestHarness::new();
        let tool = WriteFileTool::new();

        let output = tool
            .execute(
                serde_json::json!({"path": "empty.txt", "content": ""}),
                &h.context(),
            )
            .await
            .unwrap();
        assert!(matches!(output.outcome, ToolOutcome::Success));
        assert!(output.content.contains("0 bytes"));

        let written = std::fs::read_to_string(h.path().join("empty.txt")).unwrap();
        assert_eq!(written, "");
    }

    #[tokio::test]
    async fn test_write_absolute_path_rejected() {
        let h = TestHarness::new();
        let tool = WriteFileTool::new();

        let err = tool
            .execute(
                serde_json::json!({"path": "/etc/evil", "content": "x"}),
                &h.context(),
            )
            .await
            .unwrap_err();
        match err {
            ToolError::InvalidInput { tool, .. } => assert_eq!(tool, "write_file"),
            other => panic!("expected InvalidInput, got: {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_list_directory_truncates_at_limit() {
        let h = TestHarness::new();
        let dir = h.path().join("big");
        std::fs::create_dir(&dir).unwrap();
        // Create MAX_DIR_ENTRIES + 5 files to exceed the cap
        for i in 0..MAX_DIR_ENTRIES + 5 {
            std::fs::write(dir.join(format!("f{i:05}.txt")), "").unwrap();
        }

        let tool = ListDirectoryTool::new();
        let output = tool
            .execute(serde_json::json!({"path": "big"}), &h.context())
            .await
            .unwrap();
        assert!(matches!(output.outcome, ToolOutcome::Success));

        let lines: Vec<&str> = output.content.lines().collect();
        // MAX_DIR_ENTRIES entries + 1 truncation message
        assert_eq!(lines.len(), MAX_DIR_ENTRIES + 1);
        let last = lines[lines.len() - 1];
        assert!(
            last.contains("truncated"),
            "last line should be truncation notice: {last}"
        );
    }

    // ── allowed_directories tests ──────────────────────────────

    #[tokio::test]
    async fn test_read_file_via_allowed_directory() {
        let h = TestHarness::new();
        let extra_dir = tempfile::tempdir().unwrap();
        // Canonicalize to resolve macOS /var → /private/var symlink
        let extra_canonical = extra_dir.path().canonicalize().unwrap();
        std::fs::write(extra_canonical.join("external.txt"), "external data").unwrap();

        let h = h.with_allowed_directories(vec![extra_canonical.clone()]);
        let tool = ReadFileTool::new();

        let abs_path = extra_canonical.join("external.txt");
        let output = tool
            .execute(
                serde_json::json!({"path": abs_path.to_str().unwrap()}),
                &h.context(),
            )
            .await
            .unwrap();
        assert_eq!(output.content, "external data");
        assert!(matches!(output.outcome, ToolOutcome::Success));
    }

    #[tokio::test]
    async fn test_write_file_via_allowed_directory() {
        let h = TestHarness::new();
        let extra_dir = tempfile::tempdir().unwrap();
        let extra_canonical = extra_dir.path().canonicalize().unwrap();

        let h = h.with_allowed_directories(vec![extra_canonical.clone()]);
        let tool = WriteFileTool::new();

        let abs_path = extra_canonical.join("written.txt");
        let output = tool
            .execute(
                serde_json::json!({"path": abs_path.to_str().unwrap(), "content": "allowed write"}),
                &h.context(),
            )
            .await
            .unwrap();
        assert!(matches!(output.outcome, ToolOutcome::Success));

        let written = std::fs::read_to_string(&abs_path).unwrap();
        assert_eq!(written, "allowed write");
    }

    #[tokio::test]
    async fn test_absolute_path_rejected_when_not_in_allowed_directories() {
        let h = TestHarness::new();
        let extra_dir = tempfile::tempdir().unwrap();
        std::fs::write(extra_dir.path().join("secret.txt"), "secret").unwrap();

        // Don't add extra_dir to allowed_directories
        let tool = ReadFileTool::new();

        let abs_path = extra_dir.path().canonicalize().unwrap().join("secret.txt");
        let err = tool
            .execute(
                serde_json::json!({"path": abs_path.to_str().unwrap()}),
                &h.context(),
            )
            .await
            .unwrap_err();
        match err {
            ToolError::InvalidInput { tool, .. } => assert_eq!(tool, "read_file"),
            other => panic!("expected InvalidInput, got: {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_list_directory_via_allowed_directory() {
        let h = TestHarness::new();
        let extra_dir = tempfile::tempdir().unwrap();
        let extra_canonical = extra_dir.path().canonicalize().unwrap();
        std::fs::write(extra_canonical.join("visible.txt"), "").unwrap();

        let h = h.with_allowed_directories(vec![extra_canonical.clone()]);
        let tool = ListDirectoryTool::new();

        let output = tool
            .execute(
                serde_json::json!({"path": extra_canonical.to_str().unwrap()}),
                &h.context(),
            )
            .await
            .unwrap();
        assert!(matches!(output.outcome, ToolOutcome::Success));
        assert!(
            output.content.contains("visible.txt"),
            "should list files in allowed dir: {}",
            output.content
        );
    }

    // ── Factory test ────────────────────────────────────────────

    #[test]
    fn test_filesystem_tools_returns_three() {
        let tools = filesystem_tools();
        assert_eq!(tools.len(), 3);

        let mut names: Vec<String> = tools.iter().map(|t| t.info().name.clone()).collect();
        names.sort();
        assert_eq!(names, vec!["list_directory", "read_file", "write_file"]);
    }

    // ── Property-based tests ────────────────────────────────────

    mod proptests {
        use super::*;
        use proptest::prelude::*;

        /// Strategy that generates safe filenames (alphanumeric + limited punctuation).
        fn safe_filename() -> impl Strategy<Value = String> {
            "[a-zA-Z0-9_-]{1,64}\\.(txt|dat|log)"
        }

        /// Strategy that generates arbitrary UTF-8 content for file writes.
        fn file_content() -> impl Strategy<Value = String> {
            proptest::string::string_regex("[\\x20-\\x7E\\n\\t]{0,4096}").unwrap()
        }

        proptest! {
            /// Write then read roundtrip: arbitrary content survives intact.
            #[test]
            fn write_read_roundtrip(
                name in safe_filename(),
                content in file_content(),
            ) {
                let rt = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .unwrap();
                rt.block_on(async {
                    let h = TestHarness::new();
                    let write_tool = WriteFileTool::new();
                    let read_tool = ReadFileTool::new();

                    let write_result = write_tool
                        .execute(
                            serde_json::json!({"path": &name, "content": &content}),
                            &h.context(),
                        )
                        .await;
                    // Write should succeed for valid filenames
                    let output = write_result.unwrap();
                    prop_assert!(matches!(output.outcome, ToolOutcome::Success));

                    // Read it back
                    let read_output = read_tool
                        .execute(serde_json::json!({"path": &name}), &h.context())
                        .await
                        .unwrap();
                    prop_assert_eq!(&read_output.content, &content);
                    Ok(())
                })?;
            }

            /// Path traversal with arbitrary depth never succeeds.
            #[test]
            fn path_traversal_always_rejected(
                depth in 1usize..20,
                suffix in "[a-z]{1,10}",
            ) {
                let rt = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .unwrap();
                rt.block_on(async {
                    let h = TestHarness::new();
                    let tool = ReadFileTool::new();

                    let traversal = format!("{}{}", "../".repeat(depth), suffix);
                    let result = tool
                        .execute(serde_json::json!({"path": &traversal}), &h.context())
                        .await;
                    // Must always fail — never read outside sandbox
                    prop_assert!(result.is_err());
                    Ok(())
                })?;
            }

            /// Write output never leaks the sandbox root path.
            #[test]
            fn write_never_leaks_sandbox_root(
                name in safe_filename(),
            ) {
                let rt = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .unwrap();
                rt.block_on(async {
                    let h = TestHarness::new();
                    let tool = WriteFileTool::new();

                    let output = tool
                        .execute(
                            serde_json::json!({"path": &name, "content": "x"}),
                            &h.context(),
                        )
                        .await
                        .unwrap();
                    let sandbox_str = h.path().to_string_lossy();
                    prop_assert!(
                        !output.content.contains(sandbox_str.as_ref()),
                        "output leaked sandbox root: {}",
                        output.content
                    );
                    Ok(())
                })?;
            }
        }
    }
}
