//! Sandboxed filesystem tools: `read_file`, `write_file`, `list_directory`.
//!
//! All path validation flows through `TaintedToolInput::extract_path()` /
//! `extract_path_for_creation()` — tools never touch raw paths. File content
//! for writes uses `TaintedToolInput::extract_file_content()` to bridge the
//! `pub(crate)` taint boundary.

use std::path::PathBuf;

use async_trait::async_trait;
use tokio::io::AsyncReadExt;

use freebird_security::taint::TaintedToolInput;
use freebird_traits::tool::{
    Capability, RiskLevel, Tool, ToolContext, ToolError, ToolInfo, ToolOutput,
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
/// `sandbox_root` is reserved for future per-tool sandbox configuration.
/// Currently, the sandbox is provided at execution time via `ToolContext`.
#[must_use]
pub fn filesystem_tools(_sandbox_root: PathBuf) -> Vec<Box<dyn Tool>> {
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
                            "description": "Relative path within the sandbox"
                        }
                    },
                    "required": ["path"]
                }),
                required_capability: Capability::FileRead,
                risk_level: RiskLevel::Low,
                has_side_effects: false,
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
            .extract_path("path", context.sandbox_root)
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
            is_error: false,
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
                            "description": "Relative path within the sandbox"
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
                has_side_effects: true,
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
            .extract_path_for_creation("path", context.sandbox_root)
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
            is_error: false,
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
                has_side_effects: false,
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
            .extract_path("path", context.sandbox_root)
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
            is_error: false,
            metadata: None,
        })
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic, clippy::indexing_slicing)]
mod tests {
    use std::io::Write as _;

    use freebird_traits::id::SessionId;
    use freebird_traits::tool::{Capability, Tool, ToolContext, ToolError};

    use super::*;

    fn make_context() -> (SessionId, Vec<Capability>) {
        let session_id = SessionId::from_string("test-session");
        let caps = vec![Capability::FileRead, Capability::FileWrite];
        (session_id, caps)
    }

    // ── read_file tests ─────────────────────────────────────────

    #[tokio::test]
    async fn test_read_existing_file() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join("hello.txt"), "Hello, world!").unwrap();

        let tool = ReadFileTool::new();
        let (sid, caps) = make_context();
        let ctx = ToolContext {
            session_id: &sid,
            sandbox_root: tmp.path(),
            granted_capabilities: &caps,
        };

        let output = tool
            .execute(serde_json::json!({"path": "hello.txt"}), &ctx)
            .await
            .unwrap();
        assert_eq!(output.content, "Hello, world!");
        assert!(!output.is_error);
    }

    #[tokio::test]
    async fn test_read_nonexistent_file() {
        let tmp = tempfile::tempdir().unwrap();
        let tool = ReadFileTool::new();
        let (sid, caps) = make_context();
        let ctx = ToolContext {
            session_id: &sid,
            sandbox_root: tmp.path(),
            granted_capabilities: &caps,
        };

        let err = tool
            .execute(serde_json::json!({"path": "nope.txt"}), &ctx)
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
        let tmp = tempfile::tempdir().unwrap();
        let tool = ReadFileTool::new();
        let (sid, caps) = make_context();
        let ctx = ToolContext {
            session_id: &sid,
            sandbox_root: tmp.path(),
            granted_capabilities: &caps,
        };

        let err = tool
            .execute(serde_json::json!({"path": "../../../etc/passwd"}), &ctx)
            .await
            .unwrap_err();
        match err {
            ToolError::InvalidInput { tool, .. } => assert_eq!(tool, "read_file"),
            other => panic!("expected InvalidInput, got: {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_read_missing_path_field() {
        let tmp = tempfile::tempdir().unwrap();
        let tool = ReadFileTool::new();
        let (sid, caps) = make_context();
        let ctx = ToolContext {
            session_id: &sid,
            sandbox_root: tmp.path(),
            granted_capabilities: &caps,
        };

        let err = tool.execute(serde_json::json!({}), &ctx).await.unwrap_err();
        match err {
            ToolError::InvalidInput { tool, .. } => assert_eq!(tool, "read_file"),
            other => panic!("expected InvalidInput, got: {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_read_non_utf8_file() {
        let tmp = tempfile::tempdir().unwrap();
        let file_path = tmp.path().join("binary.bin");
        {
            let mut f = std::fs::File::create(&file_path).unwrap();
            f.write_all(&[0xFF, 0xFE, 0x00, 0x01]).unwrap();
        }

        let tool = ReadFileTool::new();
        let (sid, caps) = make_context();
        let ctx = ToolContext {
            session_id: &sid,
            sandbox_root: tmp.path(),
            granted_capabilities: &caps,
        };

        let err = tool
            .execute(serde_json::json!({"path": "binary.bin"}), &ctx)
            .await
            .unwrap_err();
        match err {
            ToolError::ExecutionFailed { tool, .. } => assert_eq!(tool, "read_file"),
            other => panic!("expected ExecutionFailed, got: {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_read_file_exceeds_size_limit() {
        let tmp = tempfile::tempdir().unwrap();
        let file_path = tmp.path().join("huge.txt");
        {
            let f = std::fs::File::create(&file_path).unwrap();
            // Set file size to just over the limit without writing all bytes
            f.set_len((MAX_READ_FILE_BYTES + 1) as u64).unwrap();
        }

        let tool = ReadFileTool::new();
        let (sid, caps) = make_context();
        let ctx = ToolContext {
            session_id: &sid,
            sandbox_root: tmp.path(),
            granted_capabilities: &caps,
        };

        let err = tool
            .execute(serde_json::json!({"path": "huge.txt"}), &ctx)
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
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join("empty.txt"), "").unwrap();

        let tool = ReadFileTool::new();
        let (sid, caps) = make_context();
        let ctx = ToolContext {
            session_id: &sid,
            sandbox_root: tmp.path(),
            granted_capabilities: &caps,
        };

        let output = tool
            .execute(serde_json::json!({"path": "empty.txt"}), &ctx)
            .await
            .unwrap();
        assert_eq!(output.content, "");
        assert!(!output.is_error);
    }

    #[tokio::test]
    async fn test_read_absolute_path_rejected() {
        let tmp = tempfile::tempdir().unwrap();
        let tool = ReadFileTool::new();
        let (sid, caps) = make_context();
        let ctx = ToolContext {
            session_id: &sid,
            sandbox_root: tmp.path(),
            granted_capabilities: &caps,
        };

        let err = tool
            .execute(serde_json::json!({"path": "/etc/passwd"}), &ctx)
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
        let tmp = tempfile::tempdir().unwrap();
        let tool = WriteFileTool::new();
        let (sid, caps) = make_context();
        let ctx = ToolContext {
            session_id: &sid,
            sandbox_root: tmp.path(),
            granted_capabilities: &caps,
        };

        let output = tool
            .execute(
                serde_json::json!({"path": "new.txt", "content": "hello"}),
                &ctx,
            )
            .await
            .unwrap();
        assert!(!output.is_error);

        let written = std::fs::read_to_string(tmp.path().join("new.txt")).unwrap();
        assert_eq!(written, "hello");
    }

    #[tokio::test]
    async fn test_write_overwrites_existing() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join("existing.txt"), "old content").unwrap();

        let tool = WriteFileTool::new();
        let (sid, caps) = make_context();
        let ctx = ToolContext {
            session_id: &sid,
            sandbox_root: tmp.path(),
            granted_capabilities: &caps,
        };

        let output = tool
            .execute(
                serde_json::json!({"path": "existing.txt", "content": "new content"}),
                &ctx,
            )
            .await
            .unwrap();
        assert!(!output.is_error);

        let written = std::fs::read_to_string(tmp.path().join("existing.txt")).unwrap();
        assert_eq!(written, "new content");
    }

    #[tokio::test]
    async fn test_write_path_traversal_rejected() {
        let tmp = tempfile::tempdir().unwrap();
        let tool = WriteFileTool::new();
        let (sid, caps) = make_context();
        let ctx = ToolContext {
            session_id: &sid,
            sandbox_root: tmp.path(),
            granted_capabilities: &caps,
        };

        let err = tool
            .execute(
                serde_json::json!({"path": "../../etc/evil", "content": "x"}),
                &ctx,
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
        let tmp = tempfile::tempdir().unwrap();
        let tool = WriteFileTool::new();
        let (sid, caps) = make_context();
        let ctx = ToolContext {
            session_id: &sid,
            sandbox_root: tmp.path(),
            granted_capabilities: &caps,
        };

        let err = tool
            .execute(serde_json::json!({"content": "x"}), &ctx)
            .await
            .unwrap_err();
        match err {
            ToolError::InvalidInput { tool, .. } => assert_eq!(tool, "write_file"),
            other => panic!("expected InvalidInput, got: {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_write_missing_content_field() {
        let tmp = tempfile::tempdir().unwrap();
        let tool = WriteFileTool::new();
        let (sid, caps) = make_context();
        let ctx = ToolContext {
            session_id: &sid,
            sandbox_root: tmp.path(),
            granted_capabilities: &caps,
        };

        let err = tool
            .execute(serde_json::json!({"path": "file.txt"}), &ctx)
            .await
            .unwrap_err();
        match err {
            ToolError::InvalidInput { tool, .. } => assert_eq!(tool, "write_file"),
            other => panic!("expected InvalidInput, got: {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_write_nonexistent_parent_dir() {
        let tmp = tempfile::tempdir().unwrap();
        let tool = WriteFileTool::new();
        let (sid, caps) = make_context();
        let ctx = ToolContext {
            session_id: &sid,
            sandbox_root: tmp.path(),
            granted_capabilities: &caps,
        };

        let err = tool
            .execute(
                serde_json::json!({"path": "nodir/file.txt", "content": "x"}),
                &ctx,
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
        let tmp = tempfile::tempdir().unwrap();
        let tool = WriteFileTool::new();
        let (sid, caps) = make_context();
        let ctx = ToolContext {
            session_id: &sid,
            sandbox_root: tmp.path(),
            granted_capabilities: &caps,
        };

        let output = tool
            .execute(
                serde_json::json!({"path": "report.txt", "content": "data"}),
                &ctx,
            )
            .await
            .unwrap();
        assert!(
            output.content.contains("report.txt"),
            "output should contain relative path: {}",
            output.content
        );
        let sandbox_str = tmp.path().to_string_lossy();
        assert!(
            !output.content.contains(sandbox_str.as_ref()),
            "output should NOT contain sandbox root: {}",
            output.content
        );
    }

    #[tokio::test]
    async fn test_write_no_orphaned_temp_on_success() {
        let tmp = tempfile::tempdir().unwrap();
        let tool = WriteFileTool::new();
        let (sid, caps) = make_context();
        let ctx = ToolContext {
            session_id: &sid,
            sandbox_root: tmp.path(),
            granted_capabilities: &caps,
        };

        tool.execute(
            serde_json::json!({"path": "clean.txt", "content": "data"}),
            &ctx,
        )
        .await
        .unwrap();

        // No .tmp files should remain
        let entries: Vec<_> = std::fs::read_dir(tmp.path())
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
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join("banana.txt"), "").unwrap();
        std::fs::write(tmp.path().join("apple.txt"), "").unwrap();
        std::fs::create_dir(tmp.path().join("cherry_dir")).unwrap();

        let tool = ListDirectoryTool::new();
        let (sid, caps) = make_context();
        let ctx = ToolContext {
            session_id: &sid,
            sandbox_root: tmp.path(),
            granted_capabilities: &caps,
        };

        let output = tool
            .execute(serde_json::json!({"path": "."}), &ctx)
            .await
            .unwrap();
        assert!(!output.is_error);

        let lines: Vec<&str> = output.content.lines().collect();
        assert_eq!(lines.len(), 3);
        // Sorted alphabetically by full string: "dir\t..." < "file\t..."
        assert_eq!(lines[0], "dir\tcherry_dir");
        assert_eq!(lines[1], "file\tapple.txt");
        assert_eq!(lines[2], "file\tbanana.txt");
    }

    #[tokio::test]
    async fn test_list_nonexistent_directory() {
        let tmp = tempfile::tempdir().unwrap();
        let tool = ListDirectoryTool::new();
        let (sid, caps) = make_context();
        let ctx = ToolContext {
            session_id: &sid,
            sandbox_root: tmp.path(),
            granted_capabilities: &caps,
        };

        let err = tool
            .execute(serde_json::json!({"path": "nonexistent"}), &ctx)
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
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join("afile.txt"), "content").unwrap();

        let tool = ListDirectoryTool::new();
        let (sid, caps) = make_context();
        let ctx = ToolContext {
            session_id: &sid,
            sandbox_root: tmp.path(),
            granted_capabilities: &caps,
        };

        let err = tool
            .execute(serde_json::json!({"path": "afile.txt"}), &ctx)
            .await
            .unwrap_err();
        match err {
            ToolError::ExecutionFailed { tool, .. } => assert_eq!(tool, "list_directory"),
            other => panic!("expected ExecutionFailed, got: {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_list_path_traversal_rejected() {
        let tmp = tempfile::tempdir().unwrap();
        let tool = ListDirectoryTool::new();
        let (sid, caps) = make_context();
        let ctx = ToolContext {
            session_id: &sid,
            sandbox_root: tmp.path(),
            granted_capabilities: &caps,
        };

        let err = tool
            .execute(serde_json::json!({"path": "../../"}), &ctx)
            .await
            .unwrap_err();
        match err {
            ToolError::InvalidInput { tool, .. } => assert_eq!(tool, "list_directory"),
            other => panic!("expected InvalidInput, got: {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_list_empty_directory() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::create_dir(tmp.path().join("empty")).unwrap();

        let tool = ListDirectoryTool::new();
        let (sid, caps) = make_context();
        let ctx = ToolContext {
            session_id: &sid,
            sandbox_root: tmp.path(),
            granted_capabilities: &caps,
        };

        let output = tool
            .execute(serde_json::json!({"path": "empty"}), &ctx)
            .await
            .unwrap();
        assert_eq!(output.content, "(empty directory)");
        assert!(!output.is_error);
    }

    #[tokio::test]
    async fn test_write_empty_content() {
        let tmp = tempfile::tempdir().unwrap();
        let tool = WriteFileTool::new();
        let (sid, caps) = make_context();
        let ctx = ToolContext {
            session_id: &sid,
            sandbox_root: tmp.path(),
            granted_capabilities: &caps,
        };

        let output = tool
            .execute(
                serde_json::json!({"path": "empty.txt", "content": ""}),
                &ctx,
            )
            .await
            .unwrap();
        assert!(!output.is_error);
        assert!(output.content.contains("0 bytes"));

        let written = std::fs::read_to_string(tmp.path().join("empty.txt")).unwrap();
        assert_eq!(written, "");
    }

    #[tokio::test]
    async fn test_write_absolute_path_rejected() {
        let tmp = tempfile::tempdir().unwrap();
        let tool = WriteFileTool::new();
        let (sid, caps) = make_context();
        let ctx = ToolContext {
            session_id: &sid,
            sandbox_root: tmp.path(),
            granted_capabilities: &caps,
        };

        let err = tool
            .execute(
                serde_json::json!({"path": "/etc/evil", "content": "x"}),
                &ctx,
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
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("big");
        std::fs::create_dir(&dir).unwrap();
        // Create MAX_DIR_ENTRIES + 5 files to exceed the cap
        for i in 0..MAX_DIR_ENTRIES + 5 {
            std::fs::write(dir.join(format!("f{i:05}.txt")), "").unwrap();
        }

        let tool = ListDirectoryTool::new();
        let (sid, caps) = make_context();
        let ctx = ToolContext {
            session_id: &sid,
            sandbox_root: tmp.path(),
            granted_capabilities: &caps,
        };

        let output = tool
            .execute(serde_json::json!({"path": "big"}), &ctx)
            .await
            .unwrap();
        assert!(!output.is_error);

        let lines: Vec<&str> = output.content.lines().collect();
        // MAX_DIR_ENTRIES entries + 1 truncation message
        assert_eq!(lines.len(), MAX_DIR_ENTRIES + 1);
        let last = lines[lines.len() - 1];
        assert!(
            last.contains("truncated"),
            "last line should be truncation notice: {last}"
        );
    }

    // ── Factory test ────────────────────────────────────────────

    #[test]
    fn test_filesystem_tools_returns_three() {
        let tmp = tempfile::tempdir().unwrap();
        let tools = filesystem_tools(tmp.path().to_path_buf());
        assert_eq!(tools.len(), 3);

        let mut names: Vec<String> = tools.iter().map(|t| t.info().name.clone()).collect();
        names.sort();
        assert_eq!(names, vec!["list_directory", "read_file", "write_file"]);
    }
}
