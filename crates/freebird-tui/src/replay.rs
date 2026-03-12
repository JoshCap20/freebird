//! Session replay formatting — human-readable and JSON output for past sessions.

use std::fmt::Write as _;

use freebird_traits::memory::{Conversation, Turn};
use freebird_traits::provider::{ContentBlock, Role};
use freebird_traits::tool::ToolOutcome;

/// Maximum character length for truncated tool input/output in human-readable format.
const TRUNCATE_LIMIT: usize = 200;

/// Format a conversation as a human-readable session trace.
pub fn format_replay(conv: &Conversation) -> String {
    let mut out = String::with_capacity(4096);

    // Header
    let _ = writeln!(out, "═══ Session Replay ═══");
    let _ = writeln!(out, "Session:  {}", conv.session_id);
    let _ = writeln!(out, "Model:    {} ({})", conv.model_id, conv.provider_id);
    let _ = writeln!(
        out,
        "Created:  {}",
        conv.created_at.format("%Y-%m-%d %H:%M:%S UTC")
    );
    let _ = writeln!(
        out,
        "Updated:  {}",
        conv.updated_at.format("%Y-%m-%d %H:%M:%S UTC")
    );
    let _ = writeln!(out, "Turns:    {}", conv.turns.len());

    for (i, turn) in conv.turns.iter().enumerate() {
        let _ = writeln!(out);
        let _ = writeln!(
            out,
            "─── Turn {} [{}] ───",
            i + 1,
            turn.started_at.format("%H:%M:%S")
        );
        let _ = writeln!(out);

        // User message
        let user_text = extract_text(&turn.user_message.content);
        if !user_text.is_empty() {
            let _ = writeln!(out, "  USER: {user_text}");
            let _ = writeln!(out);
        }

        // Tool invocations
        format_tool_invocations(&mut out, turn);

        // Assistant messages (text only — tool_use blocks shown via invocations)
        for msg in &turn.assistant_messages {
            if msg.role == Role::Assistant {
                let text = extract_text(&msg.content);
                if !text.is_empty() {
                    let _ = writeln!(out, "  ASSISTANT: {text}");
                    let _ = writeln!(out);
                }
            }
        }

        // Completed timestamp
        if let Some(completed) = turn.completed_at {
            let _ = writeln!(out, "  [Completed: {}]", completed.format("%H:%M:%S"));
        }
    }

    out
}

/// Format a conversation as structured JSON for machine consumption.
///
/// # Errors
///
/// Returns an error if serialization fails (should not happen for valid conversations).
pub fn format_replay_json(conv: &Conversation) -> Result<String, serde_json::Error> {
    serde_json::to_string_pretty(conv)
}

/// Format tool invocations within a turn.
fn format_tool_invocations(out: &mut String, turn: &Turn) {
    for inv in &turn.tool_invocations {
        let _ = writeln!(out, "  TOOL: {} (id: {})", inv.tool_name, inv.tool_use_id);

        let input_str = inv.input.to_string();
        let _ = writeln!(
            out,
            "    Input:    {}",
            truncate(&input_str, TRUNCATE_LIMIT)
        );

        if let Some(ref output) = inv.output {
            let _ = writeln!(out, "    Output:   {}", truncate(output, TRUNCATE_LIMIT));
        }

        let outcome_str = match inv.outcome {
            ToolOutcome::Success => "Success",
            ToolOutcome::Error => "Error",
        };
        let _ = writeln!(out, "    Outcome:  {outcome_str}");

        if let Some(ms) = inv.duration_ms {
            let _ = writeln!(out, "    Duration: {ms}ms");
        }

        let _ = writeln!(out);
    }
}

/// Extract concatenated text from content blocks.
fn extract_text(blocks: &[ContentBlock]) -> String {
    let mut text = String::new();
    for block in blocks {
        if let ContentBlock::Text { text: t } = block {
            if !text.is_empty() {
                text.push(' ');
            }
            text.push_str(t);
        }
    }
    text
}

/// Truncate a string to `max_chars` on a char boundary, appending "..." if truncated.
fn truncate(s: &str, max_chars: usize) -> String {
    if s.len() <= max_chars {
        return s.to_owned();
    }
    // Find char boundary at or before max_chars
    let end = s
        .char_indices()
        .nth(max_chars)
        .map_or(s.len(), |(idx, _)| idx);
    let truncated = s.get(..end).unwrap_or(s);
    format!("{truncated}...")
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use chrono::{TimeZone, Utc};
    use freebird_traits::id::{ModelId, ProviderId, SessionId};
    use freebird_traits::memory::ToolInvocation;
    use freebird_traits::provider::Message;

    fn make_conversation(turns: Vec<Turn>) -> Conversation {
        Conversation {
            session_id: SessionId::from("test-session-001"),
            system_prompt: None,
            turns,
            created_at: Utc.with_ymd_and_hms(2025, 6, 15, 10, 0, 0).unwrap(),
            updated_at: Utc.with_ymd_and_hms(2025, 6, 15, 10, 5, 0).unwrap(),
            model_id: ModelId::from("claude-3-opus"),
            provider_id: ProviderId::from("anthropic"),
        }
    }

    fn text_msg(role: Role, text: &str) -> Message {
        Message {
            role,
            content: vec![ContentBlock::Text {
                text: text.to_owned(),
            }],
            timestamp: Utc::now(),
        }
    }

    #[test]
    fn test_format_replay_empty_conversation() {
        let conv = make_conversation(vec![]);
        let output = format_replay(&conv);

        assert!(output.contains("═══ Session Replay ═══"));
        assert!(output.contains("Session:  test-session-001"));
        assert!(output.contains("Model:    claude-3-opus (anthropic)"));
        assert!(output.contains("Turns:    0"));
        // No turn dividers
        assert!(!output.contains("─── Turn"));
    }

    #[test]
    fn test_format_replay_single_turn() {
        let turn = Turn {
            user_message: text_msg(Role::User, "Hello, world!"),
            assistant_messages: vec![text_msg(Role::Assistant, "Hi there!")],
            tool_invocations: vec![],
            started_at: Utc.with_ymd_and_hms(2025, 6, 15, 10, 0, 0).unwrap(),
            completed_at: Some(Utc.with_ymd_and_hms(2025, 6, 15, 10, 0, 5).unwrap()),
        };
        let conv = make_conversation(vec![turn]);
        let output = format_replay(&conv);

        assert!(output.contains("Turns:    1"));
        assert!(output.contains("─── Turn 1 [10:00:00] ───"));
        assert!(output.contains("USER: Hello, world!"));
        assert!(output.contains("ASSISTANT: Hi there!"));
        assert!(output.contains("[Completed: 10:00:05]"));
    }

    #[test]
    fn test_format_replay_with_tool_invocations() {
        let turn = Turn {
            user_message: text_msg(Role::User, "Read file.txt"),
            assistant_messages: vec![text_msg(Role::Assistant, "Done.")],
            tool_invocations: vec![ToolInvocation {
                tool_use_id: "tu_123".into(),
                tool_name: "read_file".into(),
                input: serde_json::json!({"path": "/tmp/file.txt"}),
                output: Some("file contents here".into()),
                outcome: ToolOutcome::Success,
                duration_ms: Some(42),
            }],
            started_at: Utc.with_ymd_and_hms(2025, 6, 15, 10, 1, 0).unwrap(),
            completed_at: Some(Utc.with_ymd_and_hms(2025, 6, 15, 10, 1, 2).unwrap()),
        };
        let conv = make_conversation(vec![turn]);
        let output = format_replay(&conv);

        assert!(output.contains("TOOL: read_file (id: tu_123)"));
        assert!(output.contains("Input:    {\"path\":\"/tmp/file.txt\"}"));
        assert!(output.contains("Output:   file contents here"));
        assert!(output.contains("Outcome:  Success"));
        assert!(output.contains("Duration: 42ms"));
    }

    #[test]
    fn test_format_replay_multi_turn() {
        let turns = vec![
            Turn {
                user_message: text_msg(Role::User, "First question"),
                assistant_messages: vec![text_msg(Role::Assistant, "First answer")],
                tool_invocations: vec![],
                started_at: Utc.with_ymd_and_hms(2025, 6, 15, 10, 0, 0).unwrap(),
                completed_at: Some(Utc.with_ymd_and_hms(2025, 6, 15, 10, 0, 3).unwrap()),
            },
            Turn {
                user_message: text_msg(Role::User, "Second question"),
                assistant_messages: vec![text_msg(Role::Assistant, "Second answer")],
                tool_invocations: vec![],
                started_at: Utc.with_ymd_and_hms(2025, 6, 15, 10, 1, 0).unwrap(),
                completed_at: Some(Utc.with_ymd_and_hms(2025, 6, 15, 10, 1, 5).unwrap()),
            },
        ];
        let conv = make_conversation(turns);
        let output = format_replay(&conv);

        assert!(output.contains("Turns:    2"));
        assert!(output.contains("─── Turn 1"));
        assert!(output.contains("─── Turn 2"));
        assert!(output.contains("First question"));
        assert!(output.contains("Second question"));
    }

    #[test]
    fn test_format_replay_truncates_long_content() {
        let long_input = "x".repeat(300);
        let turn = Turn {
            user_message: text_msg(Role::User, "Test"),
            assistant_messages: vec![],
            tool_invocations: vec![ToolInvocation {
                tool_use_id: "tu_456".into(),
                tool_name: "write_file".into(),
                input: serde_json::Value::String(long_input),
                output: Some("y".repeat(300)),
                outcome: ToolOutcome::Error,
                duration_ms: None,
            }],
            started_at: Utc::now(),
            completed_at: None,
        };
        let conv = make_conversation(vec![turn]);
        let output = format_replay(&conv);

        // Input is JSON-stringified, so it has quotes: "\"xxx...xxx\""
        // The truncation should fire on the stringified version
        assert!(output.contains("..."));
        assert!(output.contains("Outcome:  Error"));
    }

    #[test]
    fn test_format_replay_json_roundtrip() {
        let conv = make_conversation(vec![]);
        let json = format_replay_json(&conv).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(
            parsed.get("session_id").and_then(|v| v.as_str()),
            Some("test-session-001")
        );
        assert!(parsed.get("turns").and_then(|v| v.as_array()).is_some());
    }

    #[test]
    fn test_format_replay_incomplete_turn() {
        let turn = Turn {
            user_message: text_msg(Role::User, "Still thinking..."),
            assistant_messages: vec![],
            tool_invocations: vec![],
            started_at: Utc.with_ymd_and_hms(2025, 6, 15, 10, 2, 0).unwrap(),
            completed_at: None,
        };
        let conv = make_conversation(vec![turn]);
        let output = format_replay(&conv);

        assert!(output.contains("USER: Still thinking..."));
        assert!(!output.contains("[Completed:"));
    }

    #[test]
    fn test_truncate_short_string_unchanged() {
        assert_eq!(truncate("hello", 200), "hello");
    }

    #[test]
    fn test_truncate_long_string_with_ellipsis() {
        let long = "a".repeat(250);
        let result = truncate(&long, 200);
        assert!(result.ends_with("..."));
        // 200 chars + "..."
        assert_eq!(result.len(), 203);
    }

    #[test]
    fn test_truncate_unicode_boundary_safe() {
        // "é" is 2 bytes. Ensure we don't split mid-char.
        let s = "é".repeat(150);
        let result = truncate(&s, 100);
        assert!(result.ends_with("..."));
        // Should not panic or produce invalid UTF-8
        assert!(result.is_char_boundary(result.len() - 3));
    }
}
