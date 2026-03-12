//! Session recall tools: `list_sessions`, `search_sessions`, `recall_session`.
//!
//! All tools access the memory backend via `ToolContext::memory`.
//! These are read-only operations — no side effects, no writes.

use std::fmt::Write as _;

use async_trait::async_trait;

use freebird_security::taint::TaintedToolInput;
use freebird_traits::id::SessionId;
use freebird_traits::memory::Memory;
use freebird_traits::tool::{
    Capability, RiskLevel, SideEffects, Tool, ToolContext, ToolError, ToolInfo, ToolOutcome,
    ToolOutput,
};

/// Returns all session tools as trait objects.
#[must_use]
pub fn session_tools() -> Vec<Box<dyn Tool>> {
    vec![
        Box::new(ListSessionsTool::new()),
        Box::new(SearchSessionsTool::new()),
        Box::new(RecallSessionTool::new()),
    ]
}

/// Get the memory backend from context, or return an error.
fn require_memory<'a>(
    context: &'a ToolContext<'_>,
    tool_name: &str,
) -> Result<&'a dyn Memory, ToolError> {
    context.memory.ok_or_else(|| ToolError::ExecutionFailed {
        tool: tool_name.into(),
        reason: "memory backend not configured".into(),
    })
}

/// Format a list of session summaries into human-readable output.
fn format_session_list(sessions: &[freebird_traits::memory::SessionSummary]) -> String {
    let mut output = String::with_capacity(sessions.len() * 128);
    for (i, s) in sessions.iter().enumerate() {
        if i > 0 {
            output.push_str("\n---\n");
        }
        let _ = write!(
            output,
            "Session: {}\n  Model: {} | Turns: {}\n  Created: {} | Updated: {}\n  Preview: {}",
            s.session_id,
            s.model_id,
            s.turn_count,
            s.created_at.format("%Y-%m-%d %H:%M"),
            s.updated_at.format("%Y-%m-%d %H:%M"),
            s.preview,
        );
    }
    output
}

// ── ListSessionsTool ──────────────────────────────────────────

struct ListSessionsTool {
    info: ToolInfo,
}

/// Maximum sessions to return from list.
const MAX_LIST_SESSIONS: usize = 100;

/// Default sessions to return from list.
const DEFAULT_LIST_LIMIT: usize = 20;

impl ListSessionsTool {
    const NAME: &str = "list_sessions";

    fn new() -> Self {
        Self {
            info: ToolInfo {
                name: Self::NAME.into(),
                description: "List recent conversation sessions, ordered by most recently \
                    updated. Returns session ID, timestamps, turn count, and a preview."
                    .into(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "limit": {
                            "type": "integer",
                            "description": "Maximum sessions to return (default: 20, max: 100)"
                        }
                    }
                }),
                required_capability: Capability::FileRead,
                risk_level: RiskLevel::Low,
                side_effects: SideEffects::None,
            },
        }
    }
}

#[async_trait]
impl Tool for ListSessionsTool {
    fn info(&self) -> &ToolInfo {
        &self.info
    }

    async fn execute(
        &self,
        input: serde_json::Value,
        context: &ToolContext<'_>,
    ) -> Result<ToolOutput, ToolError> {
        let memory = require_memory(context, Self::NAME)?;
        let tainted = TaintedToolInput::new(input);

        let limit = tainted
            .extract_file_content("limit")
            .ok()
            .and_then(|s| s.as_str().parse::<usize>().ok())
            .unwrap_or(DEFAULT_LIST_LIMIT)
            .min(MAX_LIST_SESSIONS);

        let sessions =
            memory
                .list_sessions(limit)
                .await
                .map_err(|e| ToolError::ExecutionFailed {
                    tool: Self::NAME.into(),
                    reason: e.to_string(),
                })?;

        if sessions.is_empty() {
            return Ok(ToolOutput {
                content: "No sessions found.".into(),
                outcome: ToolOutcome::Success,
                metadata: None,
            });
        }

        Ok(ToolOutput {
            content: format_session_list(&sessions),
            outcome: ToolOutcome::Success,
            metadata: Some(serde_json::json!({ "count": sessions.len() })),
        })
    }
}

// ── SearchSessionsTool ────────────────────────────────────────

struct SearchSessionsTool {
    info: ToolInfo,
}

/// Maximum search results.
const MAX_SEARCH_SESSIONS: usize = 50;

/// Default search results limit.
const DEFAULT_SEARCH_LIMIT: usize = 10;

impl SearchSessionsTool {
    const NAME: &str = "search_sessions";

    fn new() -> Self {
        Self {
            info: ToolInfo {
                name: Self::NAME.into(),
                description: "Search past conversation sessions by content. Returns matching \
                    sessions ranked by relevance."
                    .into(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum results to return (default: 10, max: 50)"
                        }
                    },
                    "required": ["query"]
                }),
                required_capability: Capability::FileRead,
                risk_level: RiskLevel::Low,
                side_effects: SideEffects::None,
            },
        }
    }
}

#[async_trait]
impl Tool for SearchSessionsTool {
    fn info(&self) -> &ToolInfo {
        &self.info
    }

    async fn execute(
        &self,
        input: serde_json::Value,
        context: &ToolContext<'_>,
    ) -> Result<ToolOutput, ToolError> {
        let memory = require_memory(context, Self::NAME)?;
        let tainted = TaintedToolInput::new(input);

        let query = tainted
            .extract_file_content("query")
            .map_err(|e| ToolError::InvalidInput {
                tool: Self::NAME.into(),
                reason: e.to_string(),
            })?;

        if query.as_str().is_empty() {
            return Err(ToolError::InvalidInput {
                tool: Self::NAME.into(),
                reason: "query must not be empty".into(),
            });
        }

        let limit = tainted
            .extract_file_content("limit")
            .ok()
            .and_then(|s| s.as_str().parse::<usize>().ok())
            .unwrap_or(DEFAULT_SEARCH_LIMIT)
            .min(MAX_SEARCH_SESSIONS);

        let sessions =
            memory
                .search(query.as_str(), limit)
                .await
                .map_err(|e| ToolError::ExecutionFailed {
                    tool: Self::NAME.into(),
                    reason: e.to_string(),
                })?;

        if sessions.is_empty() {
            return Ok(ToolOutput {
                content: "No matching sessions found.".into(),
                outcome: ToolOutcome::Success,
                metadata: None,
            });
        }

        Ok(ToolOutput {
            content: format_session_list(&sessions),
            outcome: ToolOutcome::Success,
            metadata: Some(serde_json::json!({ "count": sessions.len() })),
        })
    }
}

// ── RecallSessionTool ─────────────────────────────────────────

struct RecallSessionTool {
    info: ToolInfo,
}

/// Default maximum turns to include in recall.
const DEFAULT_MAX_TURNS: usize = 50;

/// Maximum content length per field before truncation.
const TRUNCATE_LEN: usize = 2000;

impl RecallSessionTool {
    const NAME: &str = "recall_session";

    fn new() -> Self {
        Self {
            info: ToolInfo {
                name: Self::NAME.into(),
                description: "Recall a past conversation session by ID. Returns a formatted \
                    transcript of the session including user messages, assistant responses, \
                    and tool invocations."
                    .into(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "session_id": {
                            "type": "string",
                            "description": "Session ID (UUID) to recall"
                        },
                        "max_turns": {
                            "type": "integer",
                            "description": "Maximum turns to include (default: 50, most recent)"
                        }
                    },
                    "required": ["session_id"]
                }),
                required_capability: Capability::FileRead,
                risk_level: RiskLevel::Low,
                side_effects: SideEffects::None,
            },
        }
    }
}

/// Truncate a string to `max_len` at a valid char boundary.
fn truncate(s: &str, max_len: usize) -> &str {
    if s.len() <= max_len {
        return s;
    }
    // Find a valid char boundary near max_len
    let mut end = max_len;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    s.get(..end).unwrap_or("")
}

/// Extract text content from a Message's content blocks.
fn message_text(msg: &freebird_traits::provider::Message) -> String {
    msg.content
        .iter()
        .filter_map(|block| {
            if let freebird_traits::provider::ContentBlock::Text { text } = block {
                Some(text.as_str())
            } else {
                None
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}

#[async_trait]
impl Tool for RecallSessionTool {
    fn info(&self) -> &ToolInfo {
        &self.info
    }

    async fn execute(
        &self,
        input: serde_json::Value,
        context: &ToolContext<'_>,
    ) -> Result<ToolOutput, ToolError> {
        let memory = require_memory(context, Self::NAME)?;
        let tainted = TaintedToolInput::new(input);

        let session_id_str =
            tainted
                .extract_file_content("session_id")
                .map_err(|e| ToolError::InvalidInput {
                    tool: Self::NAME.into(),
                    reason: e.to_string(),
                })?;

        // Validate UUID format
        let _uuid = uuid::Uuid::parse_str(session_id_str.as_str()).map_err(|_| {
            ToolError::InvalidInput {
                tool: Self::NAME.into(),
                reason: format!(
                    "invalid session ID format: `{}`",
                    truncate(session_id_str.as_str(), 50)
                ),
            }
        })?;

        let session_id = SessionId::from_string(session_id_str.as_str());

        // Prevent circular self-reference
        if &session_id == context.session_id {
            return Ok(ToolOutput {
                content: "Cannot recall the current session.".into(),
                outcome: ToolOutcome::Success,
                metadata: None,
            });
        }

        let max_turns = tainted
            .extract_file_content("max_turns")
            .ok()
            .and_then(|s| s.as_str().parse::<usize>().ok())
            .unwrap_or(DEFAULT_MAX_TURNS);

        let conversation = memory
            .load(&session_id)
            .await
            .map_err(|e| ToolError::ExecutionFailed {
                tool: Self::NAME.into(),
                reason: e.to_string(),
            })?
            .ok_or_else(|| ToolError::ExecutionFailed {
                tool: Self::NAME.into(),
                reason: format!("session `{session_id}` not found"),
            })?;

        // Take the most recent N turns
        let total_turns = conversation.turns.len();
        let start = total_turns.saturating_sub(max_turns);
        let turns = conversation.turns.get(start..).unwrap_or(&[]);

        let mut output = String::with_capacity(4096);
        let _ = writeln!(
            output,
            "Session: {} | Model: {} | Turns: {}",
            conversation.session_id, conversation.model_id, total_turns
        );
        let _ = writeln!(
            output,
            "Created: {} | Updated: {}",
            conversation.created_at.format("%Y-%m-%d %H:%M"),
            conversation.updated_at.format("%Y-%m-%d %H:%M"),
        );
        if start > 0 {
            let _ = writeln!(
                output,
                "(showing turns {}-{} of {total_turns})",
                start + 1,
                total_turns
            );
        }

        for (i, turn) in turns.iter().enumerate() {
            let turn_num = start + i + 1;
            let _ = writeln!(output, "\n--- Turn {turn_num} ---");

            // User message
            let user_text = message_text(&turn.user_message);
            let _ = writeln!(output, "User: {}", truncate(&user_text, TRUNCATE_LEN));

            // Tool invocations
            for inv in &turn.tool_invocations {
                let _ = write!(output, "[Tool: {} → {:?}", inv.tool_name, inv.outcome);
                if let Some(ms) = inv.duration_ms {
                    let _ = write!(output, " ({ms}ms)");
                }
                let _ = writeln!(output, "]");
            }

            // Assistant messages (final response)
            for msg in &turn.assistant_messages {
                let text = message_text(msg);
                if !text.is_empty() {
                    let _ = writeln!(output, "Assistant: {}", truncate(&text, TRUNCATE_LEN));
                }
            }
        }

        Ok(ToolOutput {
            content: output,
            outcome: ToolOutcome::Success,
            metadata: Some(serde_json::json!({
                "session_id": conversation.session_id.as_str(),
                "total_turns": total_turns,
                "turns_shown": turns.len(),
            })),
        })
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic, clippy::indexing_slicing)]
mod tests {
    use super::*;
    use chrono::Utc;
    use freebird_traits::id::{ModelId, ProviderId};
    use freebird_traits::memory::{Conversation, Turn};
    use freebird_traits::provider::{ContentBlock, Message, Role};
    use freebird_traits::tool::ToolOutcome as TO;

    // ── Helpers ──────────────────────────────────────────────────

    fn make_context<'a>(
        session_id: &'a SessionId,
        sandbox: &'a std::path::Path,
        memory: Option<&'a dyn Memory>,
    ) -> ToolContext<'a> {
        ToolContext {
            session_id,
            sandbox_root: sandbox,
            granted_capabilities: &[Capability::FileRead],
            allowed_directories: &[],
            knowledge_store: None,
            memory,
        }
    }

    fn make_conversation(session_id: &SessionId, turns: Vec<Turn>) -> Conversation {
        let now = Utc::now();
        Conversation {
            session_id: session_id.clone(),
            system_prompt: None,
            turns,
            created_at: now,
            updated_at: now,
            model_id: ModelId::from("claude-3"),
            provider_id: ProviderId::from("anthropic"),
        }
    }

    fn user_msg(text: &str) -> Message {
        Message {
            role: Role::User,
            content: vec![ContentBlock::Text {
                text: text.to_owned(),
            }],
            timestamp: Utc::now(),
        }
    }

    fn assistant_msg(text: &str) -> Message {
        Message {
            role: Role::Assistant,
            content: vec![ContentBlock::Text {
                text: text.to_owned(),
            }],
            timestamp: Utc::now(),
        }
    }

    fn simple_turn(user_text: &str, assistant_text: &str) -> Turn {
        let now = Utc::now();
        Turn {
            user_message: user_msg(user_text),
            assistant_messages: vec![assistant_msg(assistant_text)],
            tool_invocations: vec![],
            started_at: now,
            completed_at: Some(now),
        }
    }

    // ── Unit tests ───────────────────────────────────────────────

    #[test]
    fn test_session_tools_returns_three() {
        let tools = session_tools();
        assert_eq!(tools.len(), 3);

        let mut names: Vec<String> = tools.iter().map(|t| t.info().name.clone()).collect();
        names.sort();
        assert_eq!(
            names,
            vec!["list_sessions", "recall_session", "search_sessions"]
        );
    }

    #[test]
    fn test_tool_risk_levels() {
        for tool in session_tools() {
            let info = tool.info();
            assert_eq!(
                info.risk_level,
                RiskLevel::Low,
                "{} should be Low risk",
                info.name
            );
            assert_eq!(
                info.side_effects,
                SideEffects::None,
                "{} should have no side effects",
                info.name
            );
            assert_eq!(
                info.required_capability,
                Capability::FileRead,
                "{} should require FileRead",
                info.name
            );
        }
    }

    #[test]
    fn test_require_memory_none_errors() {
        let sid = SessionId::from_string(uuid::Uuid::new_v4().to_string());
        let ctx = ToolContext {
            session_id: &sid,
            sandbox_root: std::path::Path::new("/tmp"),
            granted_capabilities: &[],
            allowed_directories: &[],
            knowledge_store: None,
            memory: None,
        };
        let result = require_memory(&ctx, "test_tool");
        assert!(result.is_err());
    }

    #[test]
    fn test_truncate_short_string() {
        assert_eq!(truncate("hello", 10), "hello");
    }

    #[test]
    fn test_truncate_long_string() {
        let long = "a".repeat(100);
        assert_eq!(truncate(&long, 10).len(), 10);
    }

    #[test]
    fn test_truncate_unicode_boundary() {
        // "é" is 2 bytes — truncating at byte 1 should back up to byte 0
        let s = "é";
        let result = truncate(s, 1);
        assert!(result.is_empty() || result.len() <= 1);
    }

    // ── Integration tests (with InMemoryMemory) ─────────────────

    #[tokio::test]
    async fn test_list_sessions_empty() {
        let memory = freebird_memory::in_memory::InMemoryMemory::new();
        let sid = SessionId::from_string(uuid::Uuid::new_v4().to_string());
        let ctx = make_context(&sid, std::path::Path::new("/tmp"), Some(&memory));

        let tool = ListSessionsTool::new();
        let output = tool.execute(serde_json::json!({}), &ctx).await.unwrap();
        assert_eq!(output.outcome, TO::Success);
        assert!(output.content.contains("No sessions found"));
    }

    #[tokio::test]
    async fn test_list_sessions_with_data() {
        let memory = freebird_memory::in_memory::InMemoryMemory::new();
        let sid = SessionId::from_string(uuid::Uuid::new_v4().to_string());
        let conv = make_conversation(&sid, vec![simple_turn("Hello", "Hi there")]);
        memory.save(&conv).await.unwrap();

        let ctx_sid = SessionId::from_string(uuid::Uuid::new_v4().to_string());
        let ctx = make_context(&ctx_sid, std::path::Path::new("/tmp"), Some(&memory));

        let tool = ListSessionsTool::new();
        let output = tool.execute(serde_json::json!({}), &ctx).await.unwrap();
        assert_eq!(output.outcome, TO::Success);
        assert!(output.content.contains(&sid.to_string()));
        assert!(output.content.contains("Turns: 1"));
    }

    #[tokio::test]
    async fn test_list_sessions_respects_limit() {
        let memory = freebird_memory::in_memory::InMemoryMemory::new();
        for _ in 0..5 {
            let sid = SessionId::from_string(uuid::Uuid::new_v4().to_string());
            let conv = make_conversation(&sid, vec![simple_turn("Hi", "Hey")]);
            memory.save(&conv).await.unwrap();
        }

        let ctx_sid = SessionId::from_string(uuid::Uuid::new_v4().to_string());
        let ctx = make_context(&ctx_sid, std::path::Path::new("/tmp"), Some(&memory));

        let tool = ListSessionsTool::new();
        let output = tool
            .execute(serde_json::json!({"limit": "2"}), &ctx)
            .await
            .unwrap();
        assert_eq!(output.outcome, TO::Success);
        // Should only show 2 sessions (2 "---" separators = 1)
        let count: usize = output.content.matches("Session: ").count();
        assert_eq!(count, 2);
    }

    #[tokio::test]
    async fn test_list_sessions_caps_at_max() {
        let memory = freebird_memory::in_memory::InMemoryMemory::new();
        let ctx_sid = SessionId::from_string(uuid::Uuid::new_v4().to_string());
        let ctx = make_context(&ctx_sid, std::path::Path::new("/tmp"), Some(&memory));

        let tool = ListSessionsTool::new();
        // Request more than MAX_LIST_SESSIONS — should be capped
        let output = tool
            .execute(serde_json::json!({"limit": 999}), &ctx)
            .await
            .unwrap();
        // Empty store so no sessions, but the limit was accepted
        assert_eq!(output.outcome, TO::Success);
    }

    #[tokio::test]
    async fn test_search_sessions_finds_match() {
        let memory = freebird_memory::in_memory::InMemoryMemory::new();
        let sid = SessionId::from_string(uuid::Uuid::new_v4().to_string());
        let conv = make_conversation(
            &sid,
            vec![simple_turn(
                "Tell me about Rust ownership",
                "Ownership is...",
            )],
        );
        memory.save(&conv).await.unwrap();

        let ctx_sid = SessionId::from_string(uuid::Uuid::new_v4().to_string());
        let ctx = make_context(&ctx_sid, std::path::Path::new("/tmp"), Some(&memory));

        let tool = SearchSessionsTool::new();
        let output = tool
            .execute(serde_json::json!({"query": "ownership"}), &ctx)
            .await
            .unwrap();
        assert_eq!(output.outcome, TO::Success);
        assert!(output.content.contains(&sid.to_string()));
    }

    #[tokio::test]
    async fn test_search_sessions_empty_query_errors() {
        let memory = freebird_memory::in_memory::InMemoryMemory::new();
        let ctx_sid = SessionId::from_string(uuid::Uuid::new_v4().to_string());
        let ctx = make_context(&ctx_sid, std::path::Path::new("/tmp"), Some(&memory));

        let tool = SearchSessionsTool::new();
        let result = tool.execute(serde_json::json!({"query": ""}), &ctx).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_search_sessions_no_match() {
        let memory = freebird_memory::in_memory::InMemoryMemory::new();
        let sid = SessionId::from_string(uuid::Uuid::new_v4().to_string());
        let conv = make_conversation(&sid, vec![simple_turn("Hello", "Hi")]);
        memory.save(&conv).await.unwrap();

        let ctx_sid = SessionId::from_string(uuid::Uuid::new_v4().to_string());
        let ctx = make_context(&ctx_sid, std::path::Path::new("/tmp"), Some(&memory));

        let tool = SearchSessionsTool::new();
        let output = tool
            .execute(serde_json::json!({"query": "zzzznonexistent"}), &ctx)
            .await
            .unwrap();
        assert!(output.content.contains("No matching sessions"));
    }

    #[tokio::test]
    async fn test_recall_session_formats_transcript() {
        let memory = freebird_memory::in_memory::InMemoryMemory::new();
        let target_sid = SessionId::from_string(uuid::Uuid::new_v4().to_string());
        let conv = make_conversation(
            &target_sid,
            vec![
                simple_turn("What is Rust?", "Rust is a systems language."),
                simple_turn("Tell me more", "It has ownership and borrowing."),
            ],
        );
        memory.save(&conv).await.unwrap();

        let ctx_sid = SessionId::from_string(uuid::Uuid::new_v4().to_string());
        let ctx = make_context(&ctx_sid, std::path::Path::new("/tmp"), Some(&memory));

        let tool = RecallSessionTool::new();
        let output = tool
            .execute(serde_json::json!({"session_id": target_sid.as_str()}), &ctx)
            .await
            .unwrap();
        assert_eq!(output.outcome, TO::Success);
        assert!(output.content.contains("Turn 1"));
        assert!(output.content.contains("Turn 2"));
        assert!(output.content.contains("What is Rust?"));
        assert!(output.content.contains("Rust is a systems language."));
        assert!(output.content.contains("Turns: 2"));
    }

    #[tokio::test]
    async fn test_recall_session_not_found() {
        let memory = freebird_memory::in_memory::InMemoryMemory::new();
        let fake_sid = SessionId::from_string(uuid::Uuid::new_v4().to_string());
        let ctx_sid = SessionId::from_string(uuid::Uuid::new_v4().to_string());
        let ctx = make_context(&ctx_sid, std::path::Path::new("/tmp"), Some(&memory));

        let tool = RecallSessionTool::new();
        let result = tool
            .execute(serde_json::json!({"session_id": fake_sid.as_str()}), &ctx)
            .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_recall_session_skips_current() {
        let memory = freebird_memory::in_memory::InMemoryMemory::new();
        let sid = SessionId::from_string(uuid::Uuid::new_v4().to_string());
        let conv = make_conversation(&sid, vec![simple_turn("Hi", "Hey")]);
        memory.save(&conv).await.unwrap();

        // Use the same session_id as context — should prevent self-reference
        let ctx = make_context(&sid, std::path::Path::new("/tmp"), Some(&memory));

        let tool = RecallSessionTool::new();
        let output = tool
            .execute(serde_json::json!({"session_id": sid.as_str()}), &ctx)
            .await
            .unwrap();
        assert!(output.content.contains("Cannot recall the current session"));
    }

    #[tokio::test]
    async fn test_recall_session_max_turns_truncates() {
        let memory = freebird_memory::in_memory::InMemoryMemory::new();
        let target_sid = SessionId::from_string(uuid::Uuid::new_v4().to_string());
        let turns: Vec<Turn> = (0..10)
            .map(|i| simple_turn(&format!("Q{i}"), &format!("A{i}")))
            .collect();
        let conv = make_conversation(&target_sid, turns);
        memory.save(&conv).await.unwrap();

        let ctx_sid = SessionId::from_string(uuid::Uuid::new_v4().to_string());
        let ctx = make_context(&ctx_sid, std::path::Path::new("/tmp"), Some(&memory));

        let tool = RecallSessionTool::new();
        let output = tool
            .execute(
                serde_json::json!({"session_id": target_sid.as_str(), "max_turns": "3"}),
                &ctx,
            )
            .await
            .unwrap();
        assert_eq!(output.outcome, TO::Success);
        // Should show turns 8, 9, 10 (most recent 3 of 10)
        assert!(output.content.contains("Turn 8"));
        assert!(output.content.contains("Turn 10"));
        assert!(!output.content.contains("Turn 7 ---")); // Turn 7 should not appear (only 8-10)
        assert!(output.content.contains("showing turns 8-10 of 10"));
    }

    #[tokio::test]
    async fn test_recall_session_invalid_uuid() {
        let memory = freebird_memory::in_memory::InMemoryMemory::new();
        let ctx_sid = SessionId::from_string(uuid::Uuid::new_v4().to_string());
        let ctx = make_context(&ctx_sid, std::path::Path::new("/tmp"), Some(&memory));

        let tool = RecallSessionTool::new();
        let result = tool
            .execute(serde_json::json!({"session_id": "not-a-uuid"}), &ctx)
            .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_recall_session_with_tool_invocations() {
        use freebird_traits::memory::ToolInvocation;

        let memory = freebird_memory::in_memory::InMemoryMemory::new();
        let target_sid = SessionId::from_string(uuid::Uuid::new_v4().to_string());
        let now = Utc::now();
        let turn = Turn {
            user_message: user_msg("Read the file"),
            assistant_messages: vec![assistant_msg("Here is the content")],
            tool_invocations: vec![ToolInvocation {
                tool_use_id: "tu_123".into(),
                tool_name: "read_file".into(),
                input: serde_json::json!({"path": "test.rs"}),
                output: Some("fn main() {}".into()),
                outcome: TO::Success,
                duration_ms: Some(42),
            }],
            started_at: now,
            completed_at: Some(now),
        };
        let conv = make_conversation(&target_sid, vec![turn]);
        memory.save(&conv).await.unwrap();

        let ctx_sid = SessionId::from_string(uuid::Uuid::new_v4().to_string());
        let ctx = make_context(&ctx_sid, std::path::Path::new("/tmp"), Some(&memory));

        let tool = RecallSessionTool::new();
        let output = tool
            .execute(serde_json::json!({"session_id": target_sid.as_str()}), &ctx)
            .await
            .unwrap();
        assert!(output.content.contains("read_file"));
        assert!(output.content.contains("Success"));
        assert!(output.content.contains("42ms"));
    }
}
