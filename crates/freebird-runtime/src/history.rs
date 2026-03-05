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
/// - For tool-use turns: the assistant response (containing `ToolUse` blocks)
///   followed by a user message with corresponding `ToolResult` blocks
/// - Each turn's final assistant response (if present)
///
/// Does NOT include the system prompt — that's passed separately in
/// `CompletionRequest.system_prompt`.
///
/// # Limitations (v1)
///
/// Each `ToolInvocation` is emitted as a separate `ToolResult` in a single
/// user message following the assistant's `ToolUse` response. When the model
/// requests multiple tools simultaneously, they are batched correctly because
/// they originate from the same assistant message. However, the `Turn` data
/// model does not yet support multiple sequential assistant→tool rounds within
/// a single turn. When that is added, this function must be updated.
#[must_use]
pub fn conversation_to_messages(conversation: &Conversation) -> Vec<Message> {
    let mut messages = Vec::with_capacity(estimate_message_count(conversation));

    for turn in &conversation.turns {
        reconstruct_turn(turn, &mut messages);
    }

    messages
}

/// Estimate total message count for pre-allocation.
fn estimate_message_count(conversation: &Conversation) -> usize {
    conversation
        .turns
        .iter()
        .map(|t| {
            1 // user message
            + usize::from(t.assistant_response.is_some())
            + usize::from(!t.tool_invocations.is_empty()) // tool_result message
        })
        .sum()
}

/// Reconstruct a single turn into the message sequence.
fn reconstruct_turn(turn: &Turn, messages: &mut Vec<Message>) {
    // 1. User message (always present)
    messages.push(turn.user_message.clone());

    // 2. Assistant response (if present)
    if let Some(ref response) = turn.assistant_response {
        messages.push(response.clone());

        // 3. If the assistant response had tool_use blocks, emit tool results
        let has_tool_use = response
            .content
            .iter()
            .any(|b| matches!(b, ContentBlock::ToolUse { .. }));

        if has_tool_use {
            reconstruct_tool_results(&turn.tool_invocations, turn, messages);
        }
    }
}

/// Emit tool results as a user-role message (Anthropic API format).
///
/// All tool invocations for this turn are grouped into a single user message
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
        assistant_response: Option<Message>,
        invocations: Vec<ToolInvocation>,
    ) -> Turn {
        let completed_at = if assistant_response.is_some() {
            Some(Utc::now())
        } else {
            None
        };
        Turn {
            user_message: user_msg(user_text),
            assistant_response,
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
        let conv = make_conversation(vec![make_turn("hello", Some(assistant_msg("hi")), vec![])]);
        let msgs = conversation_to_messages(&conv);
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0].role, Role::User);
        assert_eq!(msgs[1].role, Role::Assistant);
    }

    #[test]
    fn test_two_simple_turns() {
        let conv = make_conversation(vec![
            make_turn("first", Some(assistant_msg("response1")), vec![]),
            make_turn("second", Some(assistant_msg("response2")), vec![]),
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
        let conv = make_conversation(vec![make_turn("hello", None, vec![])]);
        let msgs = conversation_to_messages(&conv);
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0].role, Role::User);
    }

    #[test]
    fn test_tool_use_single_invocation() {
        let conv = make_conversation(vec![Turn {
            user_message: user_msg("read file"),
            assistant_response: Some(tool_use_msg("call-1", "read_file")),
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
            assistant_response: Some(asst),
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
            assistant_response: Some(asst),
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
            make_turn("first", Some(assistant_msg("r1")), vec![]),
            make_turn("second", Some(assistant_msg("r2")), vec![]),
            make_turn("third", Some(assistant_msg("r3")), vec![]),
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
        // When a tool turn's ToolResult (User) is followed by the next turn's
        // user message (User), two adjacent User messages appear. This is a
        // known v1 limitation — the Turn model only stores one assistant_response,
        // which is the ToolUse message. A real multi-round conversation would
        // have a final assistant text response after tool use.
        let conv = make_conversation(vec![
            make_turn("first", Some(assistant_msg("r1")), vec![]),
            Turn {
                user_message: user_msg("use tool"),
                assistant_response: Some(tool_use_msg("c1", "read_file")),
                tool_invocations: vec![tool_invocation("c1", "read_file", Some("data"))],
                started_at: Utc::now(),
                completed_at: Some(Utc::now()),
            },
            make_turn("third", Some(assistant_msg("r3")), vec![]),
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
            assistant_response: Some(tool_use_msg("call-1", "read_file")),
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
            assistant_response: Some(tool_use_msg("call-1", "shell")),
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
        let conv = make_conversation(vec![make_turn("hello", Some(assistant_msg("hi")), vec![])]);
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
            make_turn("greeting", Some(assistant_msg("hi")), vec![]),
            Turn {
                user_message: user_msg("read a file"),
                assistant_response: Some(tool_use_msg("t1", "read_file")),
                tool_invocations: vec![tool_invocation("t1", "read_file", Some("data"))],
                started_at: Utc::now(),
                completed_at: Some(Utc::now()),
            },
            make_turn("thanks", Some(assistant_msg("welcome")), vec![]),
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
    fn test_pre_allocation_estimate() {
        // Empty
        let empty = make_conversation(vec![]);
        assert_eq!(estimate_message_count(&empty), 0);

        // Simple turn (user + assistant, no tools)
        let simple = make_conversation(vec![make_turn("hi", Some(assistant_msg("hello")), vec![])]);
        assert_eq!(estimate_message_count(&simple), 2);
        assert_eq!(
            conversation_to_messages(&simple).len(),
            estimate_message_count(&simple)
        );

        // Tool turn (user + assistant + tool_result)
        let tool = make_conversation(vec![Turn {
            user_message: user_msg("read"),
            assistant_response: Some(tool_use_msg("c1", "read_file")),
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
        let incomplete = make_conversation(vec![make_turn("hi", None, vec![])]);
        assert_eq!(estimate_message_count(&incomplete), 1);
        assert_eq!(
            conversation_to_messages(&incomplete).len(),
            estimate_message_count(&incomplete)
        );
    }
}
