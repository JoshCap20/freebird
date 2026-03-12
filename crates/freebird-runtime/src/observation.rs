//! Observation collapsing — compresses stale tool outputs in conversation
//! history to prevent context window exhaustion in long sessions.
//!
//! Tool outputs from turns older than a configurable threshold are replaced
//! with compact one-line summaries. This preserves the action history (what
//! tools were called and whether they succeeded) while removing raw content
//! that is no longer relevant.
//!
//! **Key invariant**: Collapsing only affects the wire messages sent to the
//! provider. Persisted data (`Conversation`/`Turn`/`ToolInvocation`) retains
//! full raw output. HMAC chains are unaffected.

use freebird_traits::memory::Conversation;
use freebird_traits::provider::{ContentBlock, Message, Role};

/// Collapse tool outputs from old turns into compact summaries.
///
/// Walks the flat `messages` list (produced by `conversation_to_messages()`)
/// and replaces `ToolResult` content in turns older than `keep_recent` with
/// one-line summaries. Uses turn boundaries from `conversation` to determine
/// which messages belong to old turns.
///
/// # Arguments
///
/// * `messages` — mutable message list (modified in-place)
/// * `conversation` — the source conversation (for turn structure + tool metadata)
/// * `keep_recent` — number of recent turns to keep fully intact
pub fn collapse_observations(
    messages: &mut [Message],
    conversation: &Conversation,
    keep_recent: usize,
) {
    let total_turns = conversation.turns.len();
    if total_turns <= keep_recent {
        return;
    }

    let turns_to_collapse = total_turns - keep_recent;

    // Walk messages, tracking which turn we're in. Each turn produces:
    //   1 user message + N assistant messages + M tool-result messages
    // We map each turn's ToolInvocation data for rich summaries.
    let mut message_idx = 0;
    for (turn_idx, turn) in conversation.turns.iter().enumerate() {
        if turn_idx >= turns_to_collapse {
            break;
        }

        // Skip user message
        if message_idx >= messages.len() {
            break;
        }
        message_idx += 1;

        // Walk assistant messages + their tool results
        let mut invocation_cursor = 0;
        for assistant_msg in &turn.assistant_messages {
            if message_idx >= messages.len() {
                break;
            }

            // Count ToolUse blocks in the original assistant message
            let tool_use_count = assistant_msg
                .content
                .iter()
                .filter(|b| matches!(b, ContentBlock::ToolUse { .. }))
                .count();

            // Skip the assistant message itself (never collapsed)
            message_idx += 1;

            if tool_use_count > 0 {
                // The next message should be a User-role message with ToolResult blocks
                if messages
                    .get(message_idx)
                    .is_some_and(|m| m.role == Role::User)
                {
                    let end = (invocation_cursor + tool_use_count).min(turn.tool_invocations.len());
                    let invocation_slice = turn.tool_invocations.get(invocation_cursor..end);

                    // Collapse each ToolResult in this message
                    if let Some(msg) = messages.get_mut(message_idx) {
                        for (block_idx, block) in msg.content.iter_mut().enumerate() {
                            if let ContentBlock::ToolResult {
                                content, is_error, ..
                            } = block
                            {
                                let inv = invocation_slice.and_then(|s| s.get(block_idx));
                                let tool_name = inv.map_or("unknown", |i| i.tool_name.as_str());
                                let input = inv.map(|i| &i.input);
                                *content =
                                    summarize_tool_output(tool_name, input, content, *is_error);
                            }
                        }
                    }

                    invocation_cursor = end;
                    message_idx += 1;
                }
            }
        }
    }
}

/// Generate a compact one-line summary for a tool's output.
///
/// Parses the tool input JSON for key fields (path, pattern, command) and
/// optionally parses the output for metrics. Falls back to a generic
/// byte-count summary on parse failure.
fn summarize_tool_output(
    tool_name: &str,
    input: Option<&serde_json::Value>,
    output: &str,
    is_error: bool,
) -> String {
    let error_prefix = if is_error { "ERROR " } else { "" };

    let summary = match tool_name {
        "read_file" => summarize_read_file(input, output),
        "file_viewer" => summarize_file_viewer(input, output),
        "grep_search" => summarize_grep(input, output),
        "glob_find" => summarize_glob(input, output),
        "search_replace_edit" => summarize_edit(input),
        "write_file" => summarize_write_file(input),
        "list_directory" => summarize_list_dir(input),
        "shell" => summarize_shell(input, output),
        "bash_exec" => summarize_bash(input, output),
        "repo_map" => "[repo_map: generated overview]".into(),
        "cargo_verify" => summarize_cargo_verify(input, output),
        "http_request" => summarize_http(input, output),
        _ => format!("[{tool_name}: {} bytes output]", output.len()),
    };

    if is_error {
        // Insert error prefix after the opening bracket
        if let Some(rest) = summary.strip_prefix('[') {
            return format!("[{error_prefix}{rest}");
        }
    }

    summary
}

fn extract_str<'a>(input: Option<&'a serde_json::Value>, field: &str) -> Option<&'a str> {
    input?.get(field)?.as_str()
}

fn summarize_read_file(input: Option<&serde_json::Value>, output: &str) -> String {
    let path = extract_str(input, "path").unwrap_or("?");
    let line_count = output.lines().count();
    format!("[read {path}: {line_count} lines]")
}

fn summarize_file_viewer(input: Option<&serde_json::Value>, output: &str) -> String {
    let path = extract_str(input, "path").unwrap_or("?");
    let start = input
        .and_then(|v| v.get("start_line"))
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(1);
    let line_count = output.lines().count();
    let end = start
        .saturating_add(line_count.try_into().unwrap_or(u64::MAX))
        .saturating_sub(1);
    format!("[viewed {path}: lines {start}-{end}]")
}

fn summarize_grep(input: Option<&serde_json::Value>, output: &str) -> String {
    let pattern = extract_str(input, "pattern").unwrap_or("?");
    let match_count = output.lines().count();
    format!("[grep '{pattern}': {match_count} matches]")
}

fn summarize_glob(input: Option<&serde_json::Value>, output: &str) -> String {
    let pattern = extract_str(input, "pattern").unwrap_or("?");
    let file_count = output.lines().filter(|l| !l.is_empty()).count();
    format!("[glob '{pattern}': {file_count} files found]")
}

fn summarize_edit(input: Option<&serde_json::Value>) -> String {
    let path = extract_str(input, "path").unwrap_or("?");
    format!("[edited {path}: replaced content]")
}

fn summarize_write_file(input: Option<&serde_json::Value>) -> String {
    let path = extract_str(input, "path").unwrap_or("?");
    format!("[wrote {path}]")
}

fn summarize_list_dir(input: Option<&serde_json::Value>) -> String {
    let path = extract_str(input, "path").unwrap_or("?");
    format!("[listed {path}]")
}

fn summarize_shell(input: Option<&serde_json::Value>, output: &str) -> String {
    let command = extract_str(input, "command").unwrap_or("?");
    // Truncate command display to 40 chars
    let display_cmd = if command.len() > 40 {
        format!("{}...", &command[..37])
    } else {
        command.to_owned()
    };
    let exit_code = extract_exit_code(output);
    format!("[shell '{display_cmd}': exit {exit_code}]")
}

fn summarize_bash(input: Option<&serde_json::Value>, output: &str) -> String {
    let command = extract_str(input, "command").unwrap_or("?");
    let display_cmd = if command.len() > 40 {
        format!("{}...", &command[..37])
    } else {
        command.to_owned()
    };
    let exit_code = extract_exit_code(output);
    format!("[bash '{display_cmd}': exit {exit_code}]")
}

fn summarize_cargo_verify(input: Option<&serde_json::Value>, output: &str) -> String {
    let subcommand = extract_str(input, "subcommand").unwrap_or("?");
    let status = if output.contains("error") || output.contains("FAILED") {
        "failed"
    } else {
        "ok"
    };
    format!("[cargo_verify '{subcommand}': {status}]")
}

fn summarize_http(input: Option<&serde_json::Value>, output: &str) -> String {
    let method = extract_str(input, "method").unwrap_or("GET");
    let url = extract_str(input, "url").unwrap_or("?");
    // Truncate URL to 50 chars
    let display_url = if url.len() > 50 {
        format!("{}...", &url[..47])
    } else {
        url.to_owned()
    };
    let status = extract_http_status(output);
    format!("[http {method} {display_url}: {status}]")
}

/// Try to extract an exit code from shell/bash output.
/// Looks for common patterns like "exit code: N" or "Exit code: N".
fn extract_exit_code(output: &str) -> &str {
    // Look for the last line that might contain exit code info
    for line in output.lines().rev() {
        let lower = line.to_lowercase();
        if lower.contains("exit code") || lower.contains("exit status") {
            // Try to find the number at the end
            if let Some(num) = line.split_whitespace().last() {
                if num.parse::<i32>().is_ok() {
                    return num;
                }
            }
        }
    }
    "?"
}

/// Try to extract HTTP status from response output.
fn extract_http_status(output: &str) -> String {
    // Look for "Status: NNN" or "HTTP/... NNN" patterns
    for line in output.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("Status:") {
            return trimmed.to_owned();
        }
        if trimmed.starts_with("HTTP/") {
            if let Some(code) = trimmed.split_whitespace().nth(1) {
                return format!("status {code}");
            }
        }
    }
    format!("{} bytes", output.len())
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::indexing_slicing)]
mod tests {
    use super::*;
    use chrono::Utc;
    use freebird_traits::id::{ModelId, ProviderId, SessionId};
    use freebird_traits::memory::{ToolInvocation, Turn};
    use freebird_traits::tool::ToolOutcome;

    use crate::history::conversation_to_messages;

    // -- Test helpers --

    fn user_msg(text: &str) -> Message {
        Message {
            role: Role::User,
            content: vec![ContentBlock::Text {
                text: text.to_owned(),
            }],
            timestamp: Utc::now(),
        }
    }

    fn assistant_text_msg(text: &str) -> Message {
        Message {
            role: Role::Assistant,
            content: vec![ContentBlock::Text {
                text: text.to_owned(),
            }],
            timestamp: Utc::now(),
        }
    }

    fn tool_use_msg(tool_use_id: &str, tool_name: &str) -> Message {
        Message {
            role: Role::Assistant,
            content: vec![ContentBlock::ToolUse {
                id: tool_use_id.to_owned(),
                name: tool_name.to_owned(),
                input: serde_json::json!({"arg": "value"}),
            }],
            timestamp: Utc::now(),
        }
    }

    fn tool_invocation(
        id: &str,
        name: &str,
        input: serde_json::Value,
        output: Option<&str>,
    ) -> ToolInvocation {
        ToolInvocation {
            tool_use_id: id.to_owned(),
            tool_name: name.to_owned(),
            input,
            output: output.map(str::to_owned),
            outcome: ToolOutcome::Success,
            duration_ms: Some(100),
        }
    }

    fn error_invocation(
        id: &str,
        name: &str,
        input: serde_json::Value,
        output: &str,
    ) -> ToolInvocation {
        ToolInvocation {
            tool_use_id: id.to_owned(),
            tool_name: name.to_owned(),
            input,
            output: Some(output.to_owned()),
            outcome: ToolOutcome::Error,
            duration_ms: Some(50),
        }
    }

    fn simple_turn(user_text: &str, response_text: &str) -> Turn {
        Turn {
            user_message: user_msg(user_text),
            assistant_messages: vec![assistant_text_msg(response_text)],
            tool_invocations: Vec::new(),
            started_at: Utc::now(),
            completed_at: Some(Utc::now()),
        }
    }

    fn tool_turn(
        user_text: &str,
        tool_use_id: &str,
        tool_name: &str,
        input: serde_json::Value,
        output: &str,
        response_text: &str,
    ) -> Turn {
        Turn {
            user_message: user_msg(user_text),
            assistant_messages: vec![
                Message {
                    role: Role::Assistant,
                    content: vec![ContentBlock::ToolUse {
                        id: tool_use_id.to_owned(),
                        name: tool_name.to_owned(),
                        input: input.clone(),
                    }],
                    timestamp: Utc::now(),
                },
                assistant_text_msg(response_text),
            ],
            tool_invocations: vec![tool_invocation(tool_use_id, tool_name, input, Some(output))],
            started_at: Utc::now(),
            completed_at: Some(Utc::now()),
        }
    }

    fn make_conversation(turns: Vec<Turn>) -> Conversation {
        Conversation {
            session_id: SessionId::from("test-session"),
            system_prompt: Some("You are helpful.".into()),
            turns,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            model_id: ModelId::from("test-model"),
            provider_id: ProviderId::from("test-provider"),
        }
    }

    /// Extract all `ToolResult` content strings from messages.
    fn extract_tool_result_contents(messages: &[Message]) -> Vec<String> {
        messages
            .iter()
            .flat_map(|msg| &msg.content)
            .filter_map(|block| match block {
                ContentBlock::ToolResult { content, .. } => Some(content.clone()),
                _ => None,
            })
            .collect()
    }

    // -- Tests --

    #[test]
    fn test_fewer_turns_than_threshold() {
        // 3 turns with threshold=5 — nothing should be collapsed
        let conv = make_conversation(vec![
            tool_turn(
                "t1",
                "c1",
                "read_file",
                serde_json::json!({"path": "a.rs"}),
                "line1\nline2",
                "done",
            ),
            tool_turn(
                "t2",
                "c2",
                "read_file",
                serde_json::json!({"path": "b.rs"}),
                "line1\nline2\nline3",
                "done",
            ),
            simple_turn("t3", "ok"),
        ]);
        let mut messages = conversation_to_messages(&conv);
        let original = messages.clone();
        collapse_observations(&mut messages, &conv, 5);
        assert_eq!(
            messages, original,
            "nothing should change when fewer turns than threshold"
        );
    }

    #[test]
    fn test_recent_turns_preserved() {
        // 7 turns (first 2 are tool turns, rest are simple). collapse_after=5.
        // Turns 3-7 should be fully intact.
        let conv = make_conversation(vec![
            tool_turn(
                "old1",
                "c1",
                "read_file",
                serde_json::json!({"path": "a.rs"}),
                "full content line1\nline2\nline3",
                "done",
            ),
            tool_turn(
                "old2",
                "c2",
                "read_file",
                serde_json::json!({"path": "b.rs"}),
                "other content\nhere",
                "done",
            ),
            simple_turn("recent1", "r1"),
            simple_turn("recent2", "r2"),
            simple_turn("recent3", "r3"),
            simple_turn("recent4", "r4"),
            simple_turn("recent5", "r5"),
        ]);
        let mut messages = conversation_to_messages(&conv);

        // Get the messages for turns 3-7 before collapsing
        // Turn 1: user + assistant(ToolUse) + user(ToolResult) + assistant(text) = 4
        // Turn 2: same = 4
        // Turn 3+: user + assistant = 2 each
        let recent_start = 8; // 4 + 4 = 8 messages for first 2 turns
        let recent_before = messages[recent_start..].to_vec();

        collapse_observations(&mut messages, &conv, 5);

        assert_eq!(
            messages[recent_start..],
            recent_before[..],
            "recent turns (3-7) should be fully preserved"
        );
    }

    #[test]
    fn test_old_tool_output_collapsed() {
        // 7 turns, turn 1 has a read_file. collapse_after=5.
        // Turn 1 tool output should be collapsed to a summary.
        let conv = make_conversation(vec![
            tool_turn(
                "old",
                "c1",
                "read_file",
                serde_json::json!({"path": "src/agent.rs"}),
                "fn main() {\n    println!(\"hello\");\n}\n",
                "done",
            ),
            simple_turn("r1", "ok"),
            simple_turn("r2", "ok"),
            simple_turn("r3", "ok"),
            simple_turn("r4", "ok"),
            simple_turn("r5", "ok"),
            simple_turn("r6", "ok"),
        ]);
        let mut messages = conversation_to_messages(&conv);
        collapse_observations(&mut messages, &conv, 5);

        let tool_results = extract_tool_result_contents(&messages);
        assert_eq!(tool_results.len(), 1);
        assert!(
            tool_results[0].starts_with("[read src/agent.rs:"),
            "expected collapsed summary, got: {}",
            tool_results[0]
        );
        assert!(
            tool_results[0].contains("lines]"),
            "should contain line count"
        );
    }

    #[test]
    fn test_user_messages_never_collapsed() {
        let conv = make_conversation(vec![
            tool_turn(
                "my important question",
                "c1",
                "read_file",
                serde_json::json!({"path": "a.rs"}),
                "content",
                "done",
            ),
            simple_turn("r1", "ok"),
            simple_turn("r2", "ok"),
            simple_turn("r3", "ok"),
            simple_turn("r4", "ok"),
            simple_turn("r5", "ok"),
            simple_turn("r6", "ok"),
        ]);
        let mut messages = conversation_to_messages(&conv);
        collapse_observations(&mut messages, &conv, 5);

        // First message should still be the user's text
        assert_eq!(messages[0].role, Role::User);
        assert!(matches!(
            &messages[0].content[0],
            ContentBlock::Text { text } if text == "my important question"
        ));
    }

    #[test]
    fn test_agent_reasoning_preserved() {
        // Turn 1 has tool use + text response. The text response should be preserved.
        let conv = make_conversation(vec![
            tool_turn(
                "old",
                "c1",
                "read_file",
                serde_json::json!({"path": "a.rs"}),
                "content",
                "I analyzed the file and found X",
            ),
            simple_turn("r1", "ok"),
            simple_turn("r2", "ok"),
            simple_turn("r3", "ok"),
            simple_turn("r4", "ok"),
            simple_turn("r5", "ok"),
            simple_turn("r6", "ok"),
        ]);
        let mut messages = conversation_to_messages(&conv);
        collapse_observations(&mut messages, &conv, 5);

        // Find the assistant text response from turn 1
        let has_reasoning = messages
            .iter()
            .filter(|m| m.role == Role::Assistant)
            .flat_map(|m| &m.content)
            .any(|b| matches!(b, ContentBlock::Text { text } if text == "I analyzed the file and found X"));

        assert!(has_reasoning, "assistant reasoning should be preserved");
    }

    #[test]
    fn test_collapsed_summary_format_read_file() {
        let summary = summarize_tool_output(
            "read_file",
            Some(&serde_json::json!({"path": "src/main.rs"})),
            "line1\nline2\nline3\nline4\nline5\n",
            false,
        );
        assert_eq!(summary, "[read src/main.rs: 5 lines]");
    }

    #[test]
    fn test_collapsed_summary_format_grep() {
        let summary = summarize_tool_output(
            "grep_search",
            Some(&serde_json::json!({"pattern": "SessionId"})),
            "src/agent.rs:10: use SessionId;\nsrc/session.rs:5: pub struct SessionId;\nsrc/types.rs:3: SessionId,\n",
            false,
        );
        assert_eq!(summary, "[grep 'SessionId': 3 matches]");
    }

    #[test]
    fn test_collapsed_summary_format_bash() {
        let summary = summarize_tool_output(
            "bash_exec",
            Some(&serde_json::json!({"command": "cargo test"})),
            "running 42 tests\n...\ntest result: ok\nExit code: 0\n",
            false,
        );
        assert_eq!(summary, "[bash 'cargo test': exit 0]");
    }

    #[test]
    fn test_collapsed_summary_format_edit() {
        let summary = summarize_tool_output(
            "search_replace_edit",
            Some(&serde_json::json!({"path": "src/lib.rs"})),
            "Successfully replaced content",
            false,
        );
        assert_eq!(summary, "[edited src/lib.rs: replaced content]");
    }

    #[test]
    fn test_configurable_threshold() {
        let conv = make_conversation(vec![
            tool_turn(
                "t1",
                "c1",
                "read_file",
                serde_json::json!({"path": "a.rs"}),
                "aaa\nbbb",
                "done",
            ),
            tool_turn(
                "t2",
                "c2",
                "read_file",
                serde_json::json!({"path": "b.rs"}),
                "ccc\nddd",
                "done",
            ),
            tool_turn(
                "t3",
                "c3",
                "read_file",
                serde_json::json!({"path": "c.rs"}),
                "eee\nfff",
                "done",
            ),
            simple_turn("t4", "ok"),
        ]);

        // With threshold=2: turns 1 and 2 should be collapsed
        let mut msgs_t2 = conversation_to_messages(&conv);
        collapse_observations(&mut msgs_t2, &conv, 2);
        let results_t2 = extract_tool_result_contents(&msgs_t2);
        assert!(
            results_t2[0].starts_with("[read a.rs:"),
            "turn 1 should be collapsed"
        );
        assert!(
            results_t2[1].starts_with("[read b.rs:"),
            "turn 2 should be collapsed"
        );
        assert_eq!(results_t2[2], "eee\nfff", "turn 3 should be intact");

        // With threshold=3: only turn 1 should be collapsed
        let mut msgs_t3 = conversation_to_messages(&conv);
        collapse_observations(&mut msgs_t3, &conv, 3);
        let results_t3 = extract_tool_result_contents(&msgs_t3);
        assert!(
            results_t3[0].starts_with("[read a.rs:"),
            "turn 1 should be collapsed"
        );
        assert_eq!(results_t3[1], "ccc\nddd", "turn 2 should be intact");
        assert_eq!(results_t3[2], "eee\nfff", "turn 3 should be intact");
    }

    #[test]
    fn test_disabled_preserves_all() {
        // When keep_recent >= total_turns, nothing is collapsed
        let conv = make_conversation(vec![
            tool_turn(
                "t1",
                "c1",
                "read_file",
                serde_json::json!({"path": "a.rs"}),
                "content1",
                "done",
            ),
            tool_turn(
                "t2",
                "c2",
                "read_file",
                serde_json::json!({"path": "b.rs"}),
                "content2",
                "done",
            ),
        ]);
        let mut messages = conversation_to_messages(&conv);
        let original = messages.clone();
        collapse_observations(&mut messages, &conv, 100);
        assert_eq!(
            messages, original,
            "nothing should change when keep_recent >= total turns"
        );
    }

    #[test]
    fn test_error_tool_output_collapsed() {
        let conv = make_conversation(vec![
            Turn {
                user_message: user_msg("old"),
                assistant_messages: vec![
                    tool_use_msg("c1", "read_file"),
                    assistant_text_msg("file not found"),
                ],
                tool_invocations: vec![error_invocation(
                    "c1",
                    "read_file",
                    serde_json::json!({"path": "missing.rs"}),
                    "Error: No such file or directory",
                )],
                started_at: Utc::now(),
                completed_at: Some(Utc::now()),
            },
            simple_turn("r1", "ok"),
            simple_turn("r2", "ok"),
            simple_turn("r3", "ok"),
            simple_turn("r4", "ok"),
            simple_turn("r5", "ok"),
            simple_turn("r6", "ok"),
        ]);
        let mut messages = conversation_to_messages(&conv);
        collapse_observations(&mut messages, &conv, 5);

        let tool_results = extract_tool_result_contents(&messages);
        assert_eq!(tool_results.len(), 1);
        assert!(
            tool_results[0].contains("ERROR"),
            "error tool output should have ERROR prefix: {}",
            tool_results[0]
        );
    }

    #[test]
    fn test_system_messages_never_collapsed() {
        // conversation_to_messages never emits System-role messages, but verify
        // that even if one were present, it wouldn't be modified (since we only
        // touch ToolResult blocks in User-role messages).
        let conv = make_conversation(vec![
            tool_turn(
                "t1",
                "c1",
                "read_file",
                serde_json::json!({"path": "a.rs"}),
                "content",
                "done",
            ),
            simple_turn("r1", "ok"),
            simple_turn("r2", "ok"),
            simple_turn("r3", "ok"),
            simple_turn("r4", "ok"),
            simple_turn("r5", "ok"),
            simple_turn("r6", "ok"),
        ]);
        let mut messages = conversation_to_messages(&conv);

        // Verify no System messages exist (sanity check)
        assert!(
            !messages.iter().any(|m| m.role == Role::System),
            "conversation_to_messages should never emit System-role messages"
        );

        // The collapsing code only modifies ToolResult blocks, so user text
        // and assistant text in old turns are safe. Already tested above.
        collapse_observations(&mut messages, &conv, 5);
        assert!(
            !messages.iter().any(|m| m.role == Role::System),
            "no System messages should appear after collapsing"
        );
    }

    #[test]
    fn test_multi_tool_turn_collapsed() {
        // Turn with 2 tool uses in separate rounds
        let conv = make_conversation(vec![
            Turn {
                user_message: user_msg("old"),
                assistant_messages: vec![
                    Message {
                        role: Role::Assistant,
                        content: vec![ContentBlock::ToolUse {
                            id: "c1".into(),
                            name: "read_file".into(),
                            input: serde_json::json!({"path": "a.rs"}),
                        }],
                        timestamp: Utc::now(),
                    },
                    Message {
                        role: Role::Assistant,
                        content: vec![ContentBlock::ToolUse {
                            id: "c2".into(),
                            name: "grep_search".into(),
                            input: serde_json::json!({"pattern": "todo"}),
                        }],
                        timestamp: Utc::now(),
                    },
                    assistant_text_msg("analyzed both"),
                ],
                tool_invocations: vec![
                    tool_invocation(
                        "c1",
                        "read_file",
                        serde_json::json!({"path": "a.rs"}),
                        Some("line1\nline2"),
                    ),
                    tool_invocation(
                        "c2",
                        "grep_search",
                        serde_json::json!({"pattern": "todo"}),
                        Some("a.rs:1: todo fix\nb.rs:5: todo later"),
                    ),
                ],
                started_at: Utc::now(),
                completed_at: Some(Utc::now()),
            },
            simple_turn("r1", "ok"),
            simple_turn("r2", "ok"),
            simple_turn("r3", "ok"),
            simple_turn("r4", "ok"),
            simple_turn("r5", "ok"),
            simple_turn("r6", "ok"),
        ]);
        let mut messages = conversation_to_messages(&conv);
        collapse_observations(&mut messages, &conv, 5);

        let tool_results = extract_tool_result_contents(&messages);
        assert_eq!(tool_results.len(), 2);
        assert!(
            tool_results[0].starts_with("[read a.rs:"),
            "first tool should be collapsed: {}",
            tool_results[0]
        );
        assert!(
            tool_results[1].starts_with("[grep 'todo':"),
            "second tool should be collapsed: {}",
            tool_results[1]
        );
    }

    #[test]
    fn test_unknown_tool_generic_summary() {
        let summary = summarize_tool_output(
            "custom_tool",
            Some(&serde_json::json!({})),
            "some output data here",
            false,
        );
        assert_eq!(summary, "[custom_tool: 21 bytes output]");
    }

    #[test]
    fn test_shell_long_command_truncated() {
        let summary = summarize_tool_output(
            "shell",
            Some(
                &serde_json::json!({"command": "find /very/long/path -name '*.rs' -exec grep -l 'pattern' {} +"}),
            ),
            "Exit code: 0",
            false,
        );
        assert!(
            summary.contains("..."),
            "long command should be truncated: {summary}"
        );
        assert!(summary.len() < 100, "summary should be compact");
    }
}
