//! Conversation summarization — compresses older turns into a compact
//! LLM-generated summary when the context window approaches its limit.
//!
//! All functions in this module are pure (no I/O, no side effects). The
//! runtime integration (`maybe_summarize`, provider calls, persistence)
//! lives in `agent.rs`.
//!
//! **Key invariant**: Turns are NEVER deleted from `Conversation.turns`.
//! The summary controls which turns are skipped during message building only.

use chrono::Utc;
use freebird_traits::id::ModelId;
use freebird_traits::memory::Conversation;
use freebird_traits::provider::{CompletionRequest, ContentBlock, Message, Role};
use freebird_types::config::{ConversationSummary, SummarizationConfig};

use crate::history::conversation_to_messages;

/// Hardcoded system prompt for summarization requests.
/// Never includes user-controlled content.
pub const SUMMARIZATION_SYSTEM_PROMPT: &str = "\
You are summarizing a coding session between a user and an AI assistant.
Produce a concise summary that preserves:

1. The user's overall goal and current progress
2. Key decisions made and their rationale
3. Important file paths, function names, and technical details discussed
4. Any unresolved issues or next steps
5. Tool outcomes that affect the current state (files created/modified/deleted)

Be factual and specific. Include exact file paths and code identifiers.
Do not include raw tool output content — only what was done and the result.";

/// Errors specific to the summarization subsystem.
#[derive(Debug, thiserror::Error)]
pub enum SummarizationError {
    #[error("provider failed to generate summary: {0}")]
    ProviderFailed(#[from] freebird_traits::provider::ProviderError),

    #[error("generated summary failed injection scan: {pattern}")]
    InjectionDetected { pattern: String },

    #[error("not enough turns to summarize (need > {preserve_recent}, have {total})")]
    InsufficientTurns {
        preserve_recent: usize,
        total: usize,
    },

    #[error("session budget too low for summarization (remaining: {remaining}, needed: {needed})")]
    InsufficientBudget { remaining: u64, needed: u64 },
}

/// Estimate token count for a message list using the ~4 chars/token heuristic.
///
/// Intentionally conservative (overestimates) to trigger summarization
/// before actually hitting context limits.
#[must_use]
pub fn estimate_token_count(messages: &[Message]) -> usize {
    messages
        .iter()
        .map(|msg| {
            msg.content
                .iter()
                .map(|block| match block {
                    ContentBlock::Text { text } => text.len(),
                    ContentBlock::ToolUse { name, input, .. } => {
                        name.len() + input.to_string().len()
                    }
                    ContentBlock::ToolResult { content, .. } => content.len(),
                    ContentBlock::Image { data, .. } => data.len(),
                })
                .sum::<usize>()
        })
        .sum::<usize>()
        / 4
}

/// Check whether summarization should trigger.
///
/// Returns `false` if: disabled, under threshold, fewer than
/// `min_turns_before_summarize` turns, or all turns already summarized.
#[must_use]
pub fn should_summarize(
    config: &SummarizationConfig,
    messages: &[Message],
    max_context_tokens: u32,
    conversation_turn_count: usize,
    existing_summary: Option<&ConversationSummary>,
) -> bool {
    if !config.enabled {
        return false;
    }

    if conversation_turn_count < config.min_turns_before_summarize {
        return false;
    }

    // Check if all non-preserved turns are already summarized
    if let Some(summary) = existing_summary {
        let unsummarized_turns =
            conversation_turn_count.saturating_sub(summary.summarized_through_turn + 1);
        if unsummarized_turns <= config.preserve_recent_turns {
            return false;
        }
    }

    let estimated_tokens = estimate_token_count(messages);
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let threshold = (f64::from(max_context_tokens) * config.trigger_threshold) as usize;

    estimated_tokens > threshold
}

/// Build a `CompletionRequest` that asks the provider to summarize.
///
/// The request includes:
/// - The hardcoded summarization system prompt (never user-controlled)
/// - The previous summary text (if re-summarizing)
/// - The turns to be compressed (all except the last `preserve_recent_turns`)
/// - No tool definitions (tools are irrelevant for summarization)
///
/// Returns `None` if there aren't enough unsummarized turns to compress.
#[must_use]
pub fn build_summary_request(
    config: &SummarizationConfig,
    conversation: &Conversation,
    existing_summary: Option<&ConversationSummary>,
    model_id: &ModelId,
) -> Option<(CompletionRequest, usize)> {
    let total_turns = conversation.turns.len();

    if total_turns <= config.preserve_recent_turns {
        return None;
    }

    // Determine which turns to summarize: everything except the last N
    let new_summarized_through = total_turns - config.preserve_recent_turns - 1;

    // Determine the start of turns to include in this summarization request
    let start_turn = existing_summary.map_or(0, |s| s.summarized_through_turn + 1);

    if start_turn > new_summarized_through {
        return None;
    }

    // Build messages from the turns to summarize
    let mut summary_messages: Vec<Message> = Vec::new();

    // If re-summarizing, include the previous summary as context
    if let Some(prev) = existing_summary {
        summary_messages.push(Message {
            role: Role::User,
            content: vec![ContentBlock::Text {
                text: format!("Previous session summary:\n\n{}", prev.text),
            }],
            timestamp: Utc::now(),
        });
    }

    // Include the turns to be summarized
    let all_messages = conversation_to_messages(conversation);

    // Map turn indices to message ranges
    let turn_message_ranges = compute_turn_message_ranges(conversation);

    for (turn_idx, (msg_start, msg_end)) in turn_message_ranges.iter().enumerate() {
        if turn_idx < start_turn || turn_idx > new_summarized_through {
            continue;
        }
        for msg in all_messages.get(*msg_start..*msg_end).unwrap_or_default() {
            summary_messages.push(msg.clone());
        }
    }

    if summary_messages.is_empty() {
        return None;
    }

    // Ensure messages end with a user message requesting the summary
    summary_messages.push(Message {
        role: Role::User,
        content: vec![ContentBlock::Text {
            text: "Please summarize the conversation above.".into(),
        }],
        timestamp: Utc::now(),
    });

    let request = CompletionRequest {
        model: model_id.clone(),
        system_prompt: Some(SUMMARIZATION_SYSTEM_PROMPT.into()),
        messages: summary_messages,
        tools: Vec::new(), // No tool definitions for summarization
        max_tokens: config.max_summary_tokens,
        temperature: Some(0.3), // Low temperature for factual summarization
        stop_sequences: Vec::new(),
    };

    Some((request, new_summarized_through))
}

/// Build a message list that skips summarized turns and prepends
/// the summary text as a User message.
///
/// If no summary exists, returns messages unchanged.
/// If a summary exists, returns: `[summary_message] + messages_from_unsummarized_turns`.
#[must_use]
pub fn apply_summary_to_messages(
    messages: Vec<Message>,
    conversation: &Conversation,
    summary: Option<&ConversationSummary>,
) -> Vec<Message> {
    let Some(summary) = summary else {
        return messages;
    };

    let turn_ranges = compute_turn_message_ranges(conversation);

    // Count how many messages belong to summarized turns
    let skip_messages: usize = turn_ranges
        .iter()
        .enumerate()
        .filter(|(idx, _)| *idx <= summary.summarized_through_turn)
        .map(|(_, (start, end))| end - start)
        .sum();

    if skip_messages == 0 || skip_messages > messages.len() {
        return messages;
    }

    let mut result = Vec::with_capacity(messages.len() - skip_messages + 1);

    // Prepend summary as a User message
    result.push(Message {
        role: Role::User,
        content: vec![ContentBlock::Text {
            text: format!(
                "[Summary of earlier conversation (turns 0-{})]\n\n{}",
                summary.summarized_through_turn, summary.text
            ),
        }],
        timestamp: Utc::now(),
    });

    // Append messages from unsummarized turns
    result.extend(messages.into_iter().skip(skip_messages));

    result
}

/// Compute the message index ranges for each turn.
///
/// Returns a Vec where each entry `(start, end)` is the half-open range
/// of message indices in the flat `conversation_to_messages()` output
/// that correspond to that turn.
fn compute_turn_message_ranges(conversation: &Conversation) -> Vec<(usize, usize)> {
    let mut ranges = Vec::with_capacity(conversation.turns.len());
    let mut offset = 0;

    for turn in &conversation.turns {
        // Each turn produces: 1 user message + N assistant messages +
        // 1 tool-result message per assistant message with ToolUse blocks
        let tool_result_messages = turn
            .assistant_messages
            .iter()
            .filter(|msg| {
                msg.content
                    .iter()
                    .any(|b| matches!(b, ContentBlock::ToolUse { .. }))
            })
            .count();

        let msg_count = 1 + turn.assistant_messages.len() + tool_result_messages;
        ranges.push((offset, offset + msg_count));
        offset += msg_count;
    }

    ranges
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::expect_used
)]
mod tests {
    use super::*;
    use freebird_traits::id::{ModelId, ProviderId, SessionId};
    use freebird_traits::memory::{ToolInvocation, Turn};
    use freebird_traits::tool::ToolOutcome;

    // -- Test helpers --

    fn default_config() -> SummarizationConfig {
        SummarizationConfig::default()
    }

    fn disabled_config() -> SummarizationConfig {
        SummarizationConfig {
            enabled: false,
            ..SummarizationConfig::default()
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

    fn tool_use_msg(id: &str, name: &str) -> Message {
        Message {
            role: Role::Assistant,
            content: vec![ContentBlock::ToolUse {
                id: id.to_owned(),
                name: name.to_owned(),
                input: serde_json::json!({"arg": "value"}),
            }],
            timestamp: Utc::now(),
        }
    }

    fn tool_invocation(id: &str, name: &str, output: &str) -> ToolInvocation {
        ToolInvocation {
            tool_use_id: id.to_owned(),
            tool_name: name.to_owned(),
            input: serde_json::json!({"arg": "value"}),
            output: Some(output.to_owned()),
            outcome: ToolOutcome::Success,
            duration_ms: Some(100),
        }
    }

    fn make_turn(user_text: &str, assistant_text: &str) -> Turn {
        Turn {
            user_message: user_msg(user_text),
            assistant_messages: vec![assistant_msg(assistant_text)],
            tool_invocations: Vec::new(),
            started_at: Utc::now(),
            completed_at: Some(Utc::now()),
        }
    }

    fn make_tool_turn(user_text: &str, tool_id: &str, tool_name: &str, output: &str) -> Turn {
        Turn {
            user_message: user_msg(user_text),
            assistant_messages: vec![tool_use_msg(tool_id, tool_name)],
            tool_invocations: vec![tool_invocation(tool_id, tool_name, output)],
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

    fn make_summary(through: usize, text: &str) -> ConversationSummary {
        ConversationSummary {
            session_id: SessionId::from("test-session"),
            text: text.into(),
            summarized_through_turn: through,
            original_token_estimate: 1000,
            generated_at: Utc::now(),
        }
    }

    // Generate N simple turns to create a long conversation
    fn make_long_conversation(n: usize) -> Conversation {
        let turns: Vec<Turn> = (0..n)
            .map(|i| make_turn(&format!("Question {i}"), &format!("Answer {i}")))
            .collect();
        make_conversation(turns)
    }

    // -- Token estimation tests --

    #[test]
    fn test_estimate_token_count_empty() {
        assert_eq!(estimate_token_count(&[]), 0);
    }

    #[test]
    fn test_estimate_token_count_text_message() {
        // 400 chars / 4 = 100 tokens
        let text = "a".repeat(400);
        let messages = vec![user_msg(&text)];
        let estimate = estimate_token_count(&messages);
        assert_eq!(estimate, 100);
    }

    #[test]
    fn test_estimate_token_count_tool_result() {
        // 2000 chars in tool result / 4 = 500 tokens
        let content = "x".repeat(2000);
        let messages = vec![Message {
            role: Role::User,
            content: vec![ContentBlock::ToolResult {
                tool_use_id: "t1".into(),
                content,
                is_error: false,
            }],
            timestamp: Utc::now(),
        }];
        let estimate = estimate_token_count(&messages);
        assert_eq!(estimate, 500);
    }

    // -- should_summarize tests --

    #[test]
    fn test_should_summarize_under_threshold() {
        let config = default_config();
        // Small messages, well under 75% of 200k context
        let messages = vec![user_msg("hello"), assistant_msg("hi")];
        assert!(!should_summarize(&config, &messages, 200_000, 10, None));
    }

    #[test]
    fn test_should_summarize_over_threshold() {
        let config = default_config();
        // max_context_tokens = 100, threshold = 0.75 → 75 tokens
        // 400 chars / 4 = 100 tokens > 75
        let text = "a".repeat(400);
        let messages = vec![user_msg(&text)];
        assert!(should_summarize(&config, &messages, 100, 10, None));
    }

    #[test]
    fn test_should_summarize_disabled() {
        let config = disabled_config();
        let text = "a".repeat(400);
        let messages = vec![user_msg(&text)];
        assert!(!should_summarize(&config, &messages, 100, 10, None));
    }

    #[test]
    fn test_should_summarize_too_few_turns() {
        let config = default_config(); // min_turns = 8
        let text = "a".repeat(400);
        let messages = vec![user_msg(&text)];
        // Only 5 turns, need 8
        assert!(!should_summarize(&config, &messages, 100, 5, None));
    }

    #[test]
    fn test_should_summarize_already_summarized() {
        let config = SummarizationConfig {
            preserve_recent_turns: 3,
            ..default_config()
        };
        let text = "a".repeat(400);
        let messages = vec![user_msg(&text)];
        // 10 turns, summarized through turn 6 → unsummarized = 3 → equal to preserve = 3 → false
        let summary = make_summary(6, "previous summary");
        assert!(!should_summarize(
            &config,
            &messages,
            100,
            10,
            Some(&summary)
        ));
    }

    // -- build_summary_request tests --

    #[test]
    fn test_build_summary_request_preserves_recent() {
        let config = SummarizationConfig {
            preserve_recent_turns: 2,
            ..default_config()
        };
        let conv = make_long_conversation(5);
        let model_id = ModelId::from("test-model");

        let (request, through) = build_summary_request(&config, &conv, None, &model_id).unwrap();

        // Should summarize turns 0,1,2 (preserve last 2: turns 3,4)
        assert_eq!(through, 2);
        // No tool definitions
        assert!(request.tools.is_empty());
        // System prompt is the hardcoded one
        assert_eq!(
            request.system_prompt.as_deref(),
            Some(SUMMARIZATION_SYSTEM_PROMPT)
        );
    }

    #[test]
    fn test_build_summary_request_insufficient_turns() {
        let config = SummarizationConfig {
            preserve_recent_turns: 5,
            ..default_config()
        };
        let conv = make_long_conversation(3); // Only 3 turns, preserve 5
        let model_id = ModelId::from("test-model");

        assert!(build_summary_request(&config, &conv, None, &model_id).is_none());
    }

    #[test]
    fn test_build_summary_request_system_prompt() {
        let config = SummarizationConfig {
            preserve_recent_turns: 1,
            ..default_config()
        };
        let conv = make_long_conversation(5);
        let model_id = ModelId::from("test-model");

        let (request, _) = build_summary_request(&config, &conv, None, &model_id).unwrap();

        assert_eq!(
            request.system_prompt.as_deref(),
            Some(SUMMARIZATION_SYSTEM_PROMPT)
        );
        assert!(request.tools.is_empty());
        assert_eq!(request.max_tokens, 1024);
    }

    #[test]
    fn test_build_summary_request_includes_previous_summary() {
        let config = SummarizationConfig {
            preserve_recent_turns: 2,
            ..default_config()
        };
        let conv = make_long_conversation(10);
        let model_id = ModelId::from("test-model");
        let prev_summary = make_summary(3, "User discussed file I/O in Rust.");

        let (request, through) =
            build_summary_request(&config, &conv, Some(&prev_summary), &model_id).unwrap();

        // Should summarize from turn 4 through turn 7 (preserve last 2: turns 8,9)
        assert_eq!(through, 7);

        // First message should contain the previous summary
        let ContentBlock::Text {
            text: ref first_text,
        } = request.messages[0].content[0]
        else {
            panic!("expected text block");
        };
        assert!(first_text.contains("Previous session summary"));
        assert!(first_text.contains("User discussed file I/O in Rust."));
    }

    // -- apply_summary_to_messages tests --

    #[test]
    fn test_apply_summary_no_summary() {
        let conv = make_long_conversation(5);
        let messages = conversation_to_messages(&conv);
        let original_len = messages.len();

        let result = apply_summary_to_messages(messages, &conv, None);
        assert_eq!(result.len(), original_len);
    }

    #[test]
    fn test_apply_summary_skips_old_turns() {
        let conv = make_long_conversation(5);
        let messages = conversation_to_messages(&conv);
        // Each simple turn = 2 messages. Total = 10 messages.
        assert_eq!(messages.len(), 10);

        // Summarize through turn 2 (turns 0,1,2 skipped = 6 messages)
        let summary = make_summary(2, "Summary of first three turns.");

        let result = apply_summary_to_messages(messages, &conv, Some(&summary));
        // 1 (summary) + 4 (turns 3,4 = 2 messages each)
        assert_eq!(result.len(), 5);
    }

    #[test]
    fn test_apply_summary_preserves_recent_turns() {
        let conv = make_long_conversation(5);
        let messages = conversation_to_messages(&conv);
        let summary = make_summary(2, "Summary text.");

        let result = apply_summary_to_messages(messages, &conv, Some(&summary));

        // Turns 3 and 4 should be preserved intact
        // result[0] = summary, result[1..5] = turns 3,4
        assert_eq!(result[1].role, Role::User); // Turn 3 user
        assert_eq!(result[2].role, Role::Assistant); // Turn 3 assistant
        assert_eq!(result[3].role, Role::User); // Turn 4 user
        assert_eq!(result[4].role, Role::Assistant); // Turn 4 assistant

        // Verify the content of preserved turns
        match &result[3].content[0] {
            ContentBlock::Text { text } => assert_eq!(text, "Question 4"),
            _ => panic!("expected text block"),
        }
    }

    #[test]
    fn test_apply_summary_prepends_summary_message() {
        let conv = make_long_conversation(5);
        let messages = conversation_to_messages(&conv);
        let summary = make_summary(2, "We discussed ownership.");

        let result = apply_summary_to_messages(messages, &conv, Some(&summary));

        assert_eq!(result[0].role, Role::User);
        match &result[0].content[0] {
            ContentBlock::Text { text } => {
                assert!(text.contains("Summary of earlier conversation"));
                assert!(text.contains("We discussed ownership."));
            }
            _ => panic!("expected text block"),
        }
    }

    #[test]
    fn test_apply_summary_message_sequence_valid() {
        // With tool turns, ensure the message sequence is still valid
        let conv = make_conversation(vec![
            make_turn("q0", "a0"),
            make_tool_turn("q1", "t1", "read_file", "file content"),
            make_turn("q2", "a2"),
            make_turn("q3", "a3"),
            make_turn("q4", "a4"),
        ]);
        let messages = conversation_to_messages(&conv);
        let summary = make_summary(1, "Discussed files.");

        let result = apply_summary_to_messages(messages, &conv, Some(&summary));

        // First message should be summary (User)
        assert_eq!(result[0].role, Role::User);

        // No two adjacent Assistant messages should appear
        // (User messages can be adjacent due to ToolResult messages being User-role)
        for window in result.windows(2) {
            assert!(
                !(window[0].role == Role::Assistant && window[1].role == Role::Assistant),
                "found adjacent assistant messages in output"
            );
        }
    }

    // -- Property-based tests --

    #[test]
    fn test_estimate_token_count_monotonic() {
        // Adding content should never decrease the estimate
        let msg1 = vec![user_msg("hello")];
        let msg2 = vec![
            user_msg("hello"),
            assistant_msg("world, this is a longer response"),
        ];

        assert!(estimate_token_count(&msg2) >= estimate_token_count(&msg1));
    }

    #[test]
    fn test_preserve_recent_turns_invariant() {
        // Summarization should never affect the last N turns
        let conv = make_long_conversation(10);
        let messages = conversation_to_messages(&conv);
        let summary = make_summary(6, "Summary.");

        // Clone last 6 messages (3 turns x 2 msgs each) before applying summary
        let original_last_6: Vec<_> = messages.iter().rev().take(6).cloned().collect();
        let result = apply_summary_to_messages(messages, &conv, Some(&summary));
        let result_last_6: Vec<_> = result.iter().rev().take(6).cloned().collect();

        // Last 6 messages should be identical
        for (orig, res) in original_last_6.iter().zip(result_last_6.iter()) {
            assert_eq!(orig.role, res.role);
            assert_eq!(orig.content.len(), res.content.len());
        }
    }
}
