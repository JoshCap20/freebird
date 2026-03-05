//! Conversation history reconstruction for provider API submission.
//!
//! Converts the domain-model `Conversation`/`Turn` structure into the flat
//! `Vec<Message>` format expected by the Anthropic Messages API. This is
//! the sole translation boundary between the persistence model and the
//! provider wire format.

use freebird_traits::memory::{Conversation, ToolInvocation, Turn};
use freebird_traits::provider::{ContentBlock, Message, Role};

/// Reconstruct a flat `Vec<Message>` from a conversation's turn history.
///
/// Produces the message sequence expected by the Anthropic Messages API:
/// - Each turn's user message
/// - For each assistant message in the turn:
///   - The assistant message itself
///   - If it contains `ToolUse` blocks: a user message with corresponding
///     `ToolResult` blocks (matched by `tool_use_id` via cursor into
///     `tool_invocations`)
///
/// Supports multi-round tool use within a single turn: the model may produce
/// multiple sequential assistant→tool rounds before delivering a final text
/// response, and all intermediate messages are reconstructed correctly.
///
/// Does NOT include the system prompt — that's passed separately in
/// `CompletionRequest.system_prompt`.
#[must_use]
pub fn conversation_to_messages(conversation: &Conversation) -> Vec<Message> {
    let mut messages = Vec::with_capacity(estimate_message_count(conversation));

    for turn in &conversation.turns {
        reconstruct_turn(turn, &mut messages);
    }

    messages
}

/// Estimate total message count for pre-allocation.
///
/// For each turn: 1 user message + N assistant messages + 1 tool-result
/// message per assistant message that contains `ToolUse` blocks.
fn estimate_message_count(conversation: &Conversation) -> usize {
    conversation
        .turns
        .iter()
        .map(|t| {
            let tool_result_messages = t
                .assistant_messages
                .iter()
                .filter(|msg| {
                    msg.content
                        .iter()
                        .any(|b| matches!(b, ContentBlock::ToolUse { .. }))
                })
                .count();
            1 + t.assistant_messages.len() + tool_result_messages
        })
        .sum()
}

/// Reconstruct a single turn into the message sequence.
///
/// Iterates through all `assistant_messages` in order. For each assistant
/// message containing `ToolUse` blocks, takes the corresponding number of
/// `ToolInvocation`s from a cursor position and emits a user-role message
/// with matching `ToolResult` blocks.
fn reconstruct_turn(turn: &Turn, messages: &mut Vec<Message>) {
    // 1. User message (always present)
    messages.push(turn.user_message.clone());

    // 2. Walk assistant messages with a cursor into tool_invocations
    let mut invocation_cursor = 0;

    for assistant_msg in &turn.assistant_messages {
        messages.push(assistant_msg.clone());

        // Count ToolUse blocks in this assistant message
        let tool_use_count = assistant_msg
            .content
            .iter()
            .filter(|b| matches!(b, ContentBlock::ToolUse { .. }))
            .count();

        if tool_use_count > 0 {
            let end = (invocation_cursor + tool_use_count).min(turn.tool_invocations.len());
            if let Some(slice) = turn.tool_invocations.get(invocation_cursor..end) {
                reconstruct_tool_results(slice, turn, messages);
            }
            invocation_cursor = end;
        }
    }
}

/// Emit tool results as a user-role message (Anthropic API format).
///
/// The provided invocations slice is grouped into a single user message
/// with one `ToolResult` block per invocation. This matches the Anthropic API
/// requirement that tool results are sent as a user message following the
/// assistant's `tool_use` message.
fn reconstruct_tool_results(
    invocations: &[ToolInvocation],
    turn: &Turn,
    messages: &mut Vec<Message>,
) {
    if invocations.is_empty() {
        return;
    }

    let tool_results: Vec<ContentBlock> = invocations
        .iter()
        .map(|inv| ContentBlock::ToolResult {
            tool_use_id: inv.tool_use_id.clone(),
            content: inv.output.clone().unwrap_or_default(),
            is_error: inv.is_error,
        })
        .collect();

    messages.push(Message {
        role: Role::User,
        content: tool_results,
        timestamp: turn.started_at,
    });
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::indexing_slicing)]
mod tests {
    use super::*;
    use chrono::Utc;
    use freebird_traits::id::SessionId;
    use freebird_traits::memory::ToolInvocation;

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

    fn assistant_msg(text: &str) -> Message {
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

    fn tool_invocation(id: &str, name: &str, output: Option<&str>) -> ToolInvocation {
        ToolInvocation {
            tool_use_id: id.to_owned(),
            tool_name: name.to_owned(),
            input: serde_json::json!({"arg": "value"}),
            output: output.map(str::to_owned),
            is_error: false,
            duration_ms: Some(100),
        }
    }

    fn error_tool_invocation(id: &str, name: &str, output: &str) -> ToolInvocation {
        ToolInvocation {
            tool_use_id: id.to_owned(),
            tool_name: name.to_owned(),
            input: serde_json::json!({"arg": "value"}),
            output: Some(output.to_owned()),
            is_error: true,
            duration_ms: Some(50),
        }
    }

    fn make_turn(
        user_text: &str,
        assistant_messages: Vec<Message>,
        invocations: Vec<ToolInvocation>,
    ) -> Turn {
        let completed_at = if assistant_messages.is_empty() {
            None
        } else {
            Some(Utc::now())
        };
        Turn {
            user_message: user_msg(user_text),
            assistant_messages,
            tool_invocations: invocations,
            started_at: Utc::now(),
            completed_at,
        }
    }

    fn make_conversation(turns: Vec<Turn>) -> Conversation {
        Conversation {
            session_id: SessionId::from("test-session"),
            system_prompt: Some("You are helpful.".into()),
            turns,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            model_id: "test-model".into(),
            provider_id: "test-provider".into(),
        }
    }

    // -- Tests --

    #[test]
    fn test_empty_conversation() {
        let conv = make_conversation(vec![]);
        let msgs = conversation_to_messages(&conv);
        assert!(msgs.is_empty());
    }

    #[test]
    fn test_simple_turn() {
        let conv = make_conversation(vec![make_turn("hello", vec![assistant_msg("hi")], vec![])]);
        let msgs = conversation_to_messages(&conv);
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0].role, Role::User);
        assert_eq!(msgs[1].role, Role::Assistant);
    }

    #[test]
    fn test_two_simple_turns() {
        let conv = make_conversation(vec![
            make_turn("first", vec![assistant_msg("response1")], vec![]),
            make_turn("second", vec![assistant_msg("response2")], vec![]),
        ]);
        let msgs = conversation_to_messages(&conv);
        assert_eq!(msgs.len(), 4);
        assert_eq!(msgs[0].role, Role::User);
        assert_eq!(msgs[1].role, Role::Assistant);
        assert_eq!(msgs[2].role, Role::User);
        assert_eq!(msgs[3].role, Role::Assistant);
    }

    #[test]
    fn test_in_progress_turn() {
        let conv = make_conversation(vec![make_turn("hello", vec![], vec![])]);
        let msgs = conversation_to_messages(&conv);
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0].role, Role::User);
    }

    #[test]
    fn test_tool_use_single_invocation() {
        let conv = make_conversation(vec![Turn {
            user_message: user_msg("read file"),
            assistant_messages: vec![tool_use_msg("call-1", "read_file")],
            tool_invocations: vec![tool_invocation("call-1", "read_file", Some("file content"))],
            started_at: Utc::now(),
            completed_at: Some(Utc::now()),
        }]);
        let msgs = conversation_to_messages(&conv);
        // user + assistant(ToolUse) + user(ToolResult)
        assert_eq!(msgs.len(), 3);
        assert_eq!(msgs[0].role, Role::User);
        assert_eq!(msgs[1].role, Role::Assistant);
        assert_eq!(msgs[2].role, Role::User);
        // Verify tool result content
        assert!(matches!(
            &msgs[2].content[0],
            ContentBlock::ToolResult {
                tool_use_id,
                content,
                is_error,
            } if tool_use_id == "call-1" && content == "file content" && !is_error
        ));
    }

    #[test]
    fn test_tool_use_three_invocations() {
        // Three tool_use blocks in one assistant message, three invocations
        let asst = Message {
            role: Role::Assistant,
            content: vec![
                ContentBlock::ToolUse {
                    id: "c1".into(),
                    name: "read_file".into(),
                    input: serde_json::json!({}),
                },
                ContentBlock::ToolUse {
                    id: "c2".into(),
                    name: "list_dir".into(),
                    input: serde_json::json!({}),
                },
                ContentBlock::ToolUse {
                    id: "c3".into(),
                    name: "shell".into(),
                    input: serde_json::json!({}),
                },
            ],
            timestamp: Utc::now(),
        };
        let conv = make_conversation(vec![Turn {
            user_message: user_msg("do three things"),
            assistant_messages: vec![asst],
            tool_invocations: vec![
                tool_invocation("c1", "read_file", Some("content-1")),
                tool_invocation("c2", "list_dir", Some("content-2")),
                tool_invocation("c3", "shell", Some("content-3")),
            ],
            started_at: Utc::now(),
            completed_at: Some(Utc::now()),
        }]);
        let msgs = conversation_to_messages(&conv);
        // user + assistant(3 ToolUse blocks) + user(3 ToolResult blocks)
        assert_eq!(msgs.len(), 3);
        assert_eq!(msgs[0].role, Role::User);
        assert_eq!(msgs[1].role, Role::Assistant);
        assert_eq!(msgs[1].content.len(), 3);
        assert_eq!(msgs[2].role, Role::User);
        assert_eq!(msgs[2].content.len(), 3);
    }

    #[test]
    fn test_tool_result_ids_match() {
        let asst = Message {
            role: Role::Assistant,
            content: vec![
                ContentBlock::ToolUse {
                    id: "alpha".into(),
                    name: "tool_a".into(),
                    input: serde_json::json!({}),
                },
                ContentBlock::ToolUse {
                    id: "beta".into(),
                    name: "tool_b".into(),
                    input: serde_json::json!({}),
                },
            ],
            timestamp: Utc::now(),
        };
        let conv = make_conversation(vec![Turn {
            user_message: user_msg("test"),
            assistant_messages: vec![asst],
            tool_invocations: vec![
                tool_invocation("alpha", "tool_a", Some("out-a")),
                tool_invocation("beta", "tool_b", Some("out-b")),
            ],
            started_at: Utc::now(),
            completed_at: Some(Utc::now()),
        }]);
        let msgs = conversation_to_messages(&conv);
        let result_msg = &msgs[2];
        assert_eq!(result_msg.role, Role::User);

        // Extract tool_use_ids from the ToolResult blocks
        let result_ids: Vec<&str> = result_msg
            .content
            .iter()
            .filter_map(|b| match b {
                ContentBlock::ToolResult { tool_use_id, .. } => Some(tool_use_id.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(result_ids, vec!["alpha", "beta"]);
    }

    #[test]
    fn test_roles_alternate_within_turns() {
        // For simple turns (no tools), roles strictly alternate
        let conv = make_conversation(vec![
            make_turn("first", vec![assistant_msg("r1")], vec![]),
            make_turn("second", vec![assistant_msg("r2")], vec![]),
            make_turn("third", vec![assistant_msg("r3")], vec![]),
        ]);
        let msgs = conversation_to_messages(&conv);
        for window in msgs.windows(2) {
            assert_ne!(
                window[0].role, window[1].role,
                "simple turns must produce strictly alternating roles"
            );
        }
    }

    #[test]
    fn test_roles_correct_sequence_with_tools() {
        // When a tool turn has only a ToolUse assistant message (no final text
        // response), the ToolResult (User) is followed by the next turn's
        // user message (User), producing two adjacent User messages.
        let conv = make_conversation(vec![
            make_turn("first", vec![assistant_msg("r1")], vec![]),
            Turn {
                user_message: user_msg("use tool"),
                assistant_messages: vec![tool_use_msg("c1", "read_file")],
                tool_invocations: vec![tool_invocation("c1", "read_file", Some("data"))],
                started_at: Utc::now(),
                completed_at: Some(Utc::now()),
            },
            make_turn("third", vec![assistant_msg("r3")], vec![]),
        ]);
        let msgs = conversation_to_messages(&conv);
        assert_eq!(msgs.len(), 7);
        let expected_roles = [
            Role::User,
            Role::Assistant,
            Role::User,      // turn 2: user message
            Role::Assistant, // turn 2: assistant(ToolUse)
            Role::User,      // turn 2: user(ToolResult)
            Role::User,      // turn 3: user message
            Role::Assistant, // turn 3: assistant response
        ];
        for (i, (msg, expected)) in msgs.iter().zip(expected_roles.iter()).enumerate() {
            assert_eq!(&msg.role, expected, "role mismatch at index {i}");
        }
    }

    #[test]
    fn test_tool_invocation_output_none() {
        let conv = make_conversation(vec![Turn {
            user_message: user_msg("read file"),
            assistant_messages: vec![tool_use_msg("call-1", "read_file")],
            tool_invocations: vec![tool_invocation("call-1", "read_file", None)],
            started_at: Utc::now(),
            completed_at: Some(Utc::now()),
        }]);
        let msgs = conversation_to_messages(&conv);
        assert_eq!(msgs.len(), 3);
        assert!(matches!(
            &msgs[2].content[0],
            ContentBlock::ToolResult { content, .. } if content.is_empty()
        ));
    }

    #[test]
    fn test_tool_invocation_is_error() {
        let conv = make_conversation(vec![Turn {
            user_message: user_msg("do something"),
            assistant_messages: vec![tool_use_msg("call-1", "shell")],
            tool_invocations: vec![error_tool_invocation(
                "call-1",
                "shell",
                "permission denied",
            )],
            started_at: Utc::now(),
            completed_at: Some(Utc::now()),
        }]);
        let msgs = conversation_to_messages(&conv);
        assert!(matches!(
            &msgs[2].content[0],
            ContentBlock::ToolResult {
                is_error,
                content,
                ..
            } if *is_error && content == "permission denied"
        ));
    }

    #[test]
    fn test_system_prompt_not_in_output() {
        let conv = make_conversation(vec![make_turn("hello", vec![assistant_msg("hi")], vec![])]);
        // Conversation has system_prompt set
        assert!(conv.system_prompt.is_some());

        let msgs = conversation_to_messages(&conv);
        // No message should have Role::System
        for msg in &msgs {
            assert_ne!(
                msg.role,
                Role::System,
                "system prompt must not appear in output messages"
            );
        }
    }

    #[test]
    fn test_mixed_turns() {
        // Simple turn + tool turn + simple turn
        let conv = make_conversation(vec![
            make_turn("greeting", vec![assistant_msg("hi")], vec![]),
            Turn {
                user_message: user_msg("read a file"),
                assistant_messages: vec![tool_use_msg("t1", "read_file")],
                tool_invocations: vec![tool_invocation("t1", "read_file", Some("data"))],
                started_at: Utc::now(),
                completed_at: Some(Utc::now()),
            },
            make_turn("thanks", vec![assistant_msg("welcome")], vec![]),
        ]);
        let msgs = conversation_to_messages(&conv);
        // Turn 1: user + assistant = 2
        // Turn 2: user + assistant(ToolUse) + user(ToolResult) = 3
        // Turn 3: user + assistant = 2
        // Total = 7
        assert_eq!(msgs.len(), 7);
        assert_eq!(msgs[0].role, Role::User);
        assert_eq!(msgs[1].role, Role::Assistant);
        assert_eq!(msgs[2].role, Role::User);
        assert_eq!(msgs[3].role, Role::Assistant);
        assert_eq!(msgs[4].role, Role::User);
        assert_eq!(msgs[5].role, Role::User);
        assert_eq!(msgs[6].role, Role::Assistant);
    }

    #[test]
    fn test_interrupted_turn_drops_orphaned_invocations() {
        // When assistant_messages is empty (interrupted/crashed turn) but
        // tool_invocations is non-empty, the invocations are silently dropped.
        // This is correct: without an assistant message containing ToolUse blocks,
        // there's nothing to pair the ToolResult blocks against, and emitting
        // orphaned ToolResults would violate the Anthropic API contract.
        let conv = make_conversation(vec![Turn {
            user_message: user_msg("do something"),
            assistant_messages: vec![],
            tool_invocations: vec![
                tool_invocation("orphan-1", "shell", Some("partial output")),
                tool_invocation("orphan-2", "read_file", Some("data")),
            ],
            started_at: Utc::now(),
            completed_at: None,
        }]);
        let msgs = conversation_to_messages(&conv);
        // Only the user message — no assistant, no tool results
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0].role, Role::User);
    }

    #[test]
    fn test_pre_allocation_estimate() {
        // Empty
        let empty = make_conversation(vec![]);
        assert_eq!(estimate_message_count(&empty), 0);

        // Simple turn (user + assistant, no tools)
        let simple = make_conversation(vec![make_turn("hi", vec![assistant_msg("hello")], vec![])]);
        assert_eq!(estimate_message_count(&simple), 2);
        assert_eq!(
            conversation_to_messages(&simple).len(),
            estimate_message_count(&simple)
        );

        // Tool turn (user + assistant + tool_result)
        let tool = make_conversation(vec![Turn {
            user_message: user_msg("read"),
            assistant_messages: vec![tool_use_msg("c1", "read_file")],
            tool_invocations: vec![tool_invocation("c1", "read_file", Some("data"))],
            started_at: Utc::now(),
            completed_at: Some(Utc::now()),
        }]);
        assert_eq!(estimate_message_count(&tool), 3);
        assert_eq!(
            conversation_to_messages(&tool).len(),
            estimate_message_count(&tool)
        );

        // Incomplete turn (user only)
        let incomplete = make_conversation(vec![make_turn("hi", vec![], vec![])]);
        assert_eq!(estimate_message_count(&incomplete), 1);
        assert_eq!(
            conversation_to_messages(&incomplete).len(),
            estimate_message_count(&incomplete)
        );
    }

    // ─── Multi-Round Tests ──────────────────────────────────────────────

    #[test]
    fn test_multi_round_tool_use_reconstruction() {
        // Two sequential rounds: ToolUse→Result→ToolUse→Result→Final text
        let conv = make_conversation(vec![Turn {
            user_message: user_msg("read two files"),
            assistant_messages: vec![
                tool_use_msg("r1", "read_file"),
                tool_use_msg("r2", "write_file"),
                assistant_msg("done, both files processed"),
            ],
            tool_invocations: vec![
                tool_invocation("r1", "read_file", Some("content-1")),
                tool_invocation("r2", "write_file", Some("ok")),
            ],
            started_at: Utc::now(),
            completed_at: Some(Utc::now()),
        }]);
        let msgs = conversation_to_messages(&conv);
        // user + assistant(ToolUse r1) + user(ToolResult r1) +
        // assistant(ToolUse r2) + user(ToolResult r2) +
        // assistant(final text)
        assert_eq!(msgs.len(), 6);
        assert_eq!(msgs[0].role, Role::User);
        assert_eq!(msgs[1].role, Role::Assistant);
        assert_eq!(msgs[2].role, Role::User);
        assert_eq!(msgs[3].role, Role::Assistant);
        assert_eq!(msgs[4].role, Role::User);
        assert_eq!(msgs[5].role, Role::Assistant);

        // Verify final message is the text response
        assert!(matches!(
            &msgs[5].content[0],
            ContentBlock::Text { text } if text == "done, both files processed"
        ));
    }

    #[test]
    fn test_multi_round_estimate_count() {
        // Multi-round turn: 2 ToolUse messages + 1 final text = 3 assistant messages
        // Expected: 1 (user) + 3 (assistant msgs) + 2 (tool result msgs) = 6
        let conv = make_conversation(vec![Turn {
            user_message: user_msg("multi-round"),
            assistant_messages: vec![
                tool_use_msg("a", "read_file"),
                tool_use_msg("b", "shell"),
                assistant_msg("all done"),
            ],
            tool_invocations: vec![
                tool_invocation("a", "read_file", Some("data")),
                tool_invocation("b", "shell", Some("output")),
            ],
            started_at: Utc::now(),
            completed_at: Some(Utc::now()),
        }]);
        assert_eq!(estimate_message_count(&conv), 6);
        assert_eq!(
            conversation_to_messages(&conv).len(),
            estimate_message_count(&conv)
        );
    }

    #[test]
    fn test_multi_round_tool_result_ids_match() {
        // Verify that each round's ToolResult IDs match the ToolUse IDs
        let conv = make_conversation(vec![Turn {
            user_message: user_msg("sequential tools"),
            assistant_messages: vec![
                tool_use_msg("first-call", "read_file"),
                tool_use_msg("second-call", "shell"),
            ],
            tool_invocations: vec![
                tool_invocation("first-call", "read_file", Some("file-data")),
                tool_invocation("second-call", "shell", Some("shell-output")),
            ],
            started_at: Utc::now(),
            completed_at: Some(Utc::now()),
        }]);
        let msgs = conversation_to_messages(&conv);
        assert_eq!(msgs.len(), 5);

        // Round 1: msgs[1] = assistant(ToolUse "first-call"), msgs[2] = user(ToolResult "first-call")
        assert!(matches!(
            &msgs[2].content[0],
            ContentBlock::ToolResult { tool_use_id, .. } if tool_use_id == "first-call"
        ));

        // Round 2: msgs[3] = assistant(ToolUse "second-call"), msgs[4] = user(ToolResult "second-call")
        assert!(matches!(
            &msgs[4].content[0],
            ContentBlock::ToolResult { tool_use_id, .. } if tool_use_id == "second-call"
        ));
    }

    #[test]
    fn test_mixed_batch_and_sequential_rounds() {
        // Round 1: 2 parallel tools in one message, Round 2: 1 tool
        let batch_msg = Message {
            role: Role::Assistant,
            content: vec![
                ContentBlock::ToolUse {
                    id: "p1".into(),
                    name: "read_file".into(),
                    input: serde_json::json!({}),
                },
                ContentBlock::ToolUse {
                    id: "p2".into(),
                    name: "list_dir".into(),
                    input: serde_json::json!({}),
                },
            ],
            timestamp: Utc::now(),
        };
        let conv = make_conversation(vec![Turn {
            user_message: user_msg("batch then sequential"),
            assistant_messages: vec![
                batch_msg,
                tool_use_msg("s1", "shell"),
                assistant_msg("finished"),
            ],
            tool_invocations: vec![
                tool_invocation("p1", "read_file", Some("data-1")),
                tool_invocation("p2", "list_dir", Some("data-2")),
                tool_invocation("s1", "shell", Some("output")),
            ],
            started_at: Utc::now(),
            completed_at: Some(Utc::now()),
        }]);
        let msgs = conversation_to_messages(&conv);
        // user + assistant(2 ToolUse) + user(2 ToolResult in one msg) +
        // assistant(1 ToolUse) + user(1 ToolResult) + assistant(text)
        assert_eq!(msgs.len(), 6);

        // Batch round: single tool result message with 2 blocks
        assert_eq!(msgs[2].content.len(), 2);
        let batch_ids: Vec<&str> = msgs[2]
            .content
            .iter()
            .filter_map(|b| match b {
                ContentBlock::ToolResult { tool_use_id, .. } => Some(tool_use_id.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(batch_ids, vec!["p1", "p2"]);

        // Sequential round: tool result message has 1 block
        assert_eq!(msgs[4].content.len(), 1);
        assert!(matches!(
            &msgs[4].content[0],
            ContentBlock::ToolResult { tool_use_id, .. } if tool_use_id == "s1"
        ));

        // Final text
        assert!(matches!(
            &msgs[5].content[0],
            ContentBlock::Text { text } if text == "finished"
        ));
    }
}
