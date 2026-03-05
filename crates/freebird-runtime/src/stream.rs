//! `StreamAccumulator` — collects streaming events into a final `Message`.
//!
//! Accumulates `StreamEvent` items (text deltas, tool-use blocks) into a
//! complete assistant `Message` for conversation persistence. Used by the
//! streaming variant of the agentic loop.

use chrono::Utc;
use freebird_traits::provider::{ContentBlock, Message, Role};

/// Accumulates streaming events into a final assistant `Message`.
///
/// # Flush semantics
///
/// Text deltas accumulate in an internal buffer. The buffer is flushed
/// (moved into a `ContentBlock::Text`) when:
/// - `push_tool_use()` is called (flushes *before* adding the tool block)
/// - `into_message()` is called (flushes any remaining text)
///
/// Empty buffers are never flushed — no empty `ContentBlock::Text` blocks.
///
/// # Ownership
///
/// `into_message()` consumes `self`. For tool-use re-streaming, create
/// a new `StreamAccumulator` for each provider round.
#[derive(Debug)]
#[must_use]
pub struct StreamAccumulator {
    text_buffer: String,
    content_blocks: Vec<ContentBlock>,
}

impl StreamAccumulator {
    /// Create a new empty accumulator.
    pub const fn new() -> Self {
        Self {
            text_buffer: String::new(),
            content_blocks: Vec::new(),
        }
    }

    /// Append a text delta to the internal buffer.
    pub fn push_text_delta(&mut self, text: &str) {
        self.text_buffer.push_str(text);
    }

    /// Record a tool-use block. Flushes any buffered text first.
    pub fn push_tool_use(&mut self, id: String, name: String, input: serde_json::Value) {
        self.flush_text();
        self.content_blocks
            .push(ContentBlock::ToolUse { id, name, input });
    }

    /// Consume the accumulator and produce an assistant-role `Message`.
    ///
    /// Flushes any remaining buffered text, then builds a `Message` with
    /// `Role::Assistant` and a timestamp of `Utc::now()`.
    #[must_use]
    pub fn into_message(mut self) -> Message {
        self.flush_text();
        Message {
            role: Role::Assistant,
            content: self.content_blocks,
            timestamp: Utc::now(),
        }
    }

    /// Returns `true` if no text or content blocks have been accumulated.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.text_buffer.is_empty() && self.content_blocks.is_empty()
    }

    /// Flush the text buffer into a `ContentBlock::Text` if non-empty.
    fn flush_text(&mut self) {
        if !self.text_buffer.is_empty() {
            let text = std::mem::take(&mut self.text_buffer);
            self.content_blocks.push(ContentBlock::Text { text });
        }
    }
}

impl Default for StreamAccumulator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::indexing_slicing)]
mod tests {
    use super::*;

    #[test]
    fn test_accumulator_new_is_empty() {
        let acc = StreamAccumulator::new();
        assert!(acc.is_empty());
    }

    #[test]
    fn test_accumulator_text_deltas_concatenate() {
        let mut acc = StreamAccumulator::new();
        acc.push_text_delta("Hello");
        acc.push_text_delta(" ");
        acc.push_text_delta("world");

        let msg = acc.into_message();
        assert_eq!(msg.content.len(), 1);
        assert!(matches!(
            &msg.content[0],
            ContentBlock::Text { text } if text == "Hello world"
        ));
    }

    #[test]
    fn test_accumulator_tool_use_added() {
        let mut acc = StreamAccumulator::new();
        acc.push_tool_use(
            "call-1".into(),
            "read_file".into(),
            serde_json::json!({"path": "test.txt"}),
        );

        let msg = acc.into_message();
        assert_eq!(msg.content.len(), 1);
        assert!(matches!(
            &msg.content[0],
            ContentBlock::ToolUse { id, name, .. } if id == "call-1" && name == "read_file"
        ));
    }

    #[test]
    fn test_accumulator_text_flushed_before_tool_use() {
        let mut acc = StreamAccumulator::new();
        acc.push_text_delta("Let me check");
        acc.push_tool_use("call-1".into(), "read_file".into(), serde_json::json!({}));

        let msg = acc.into_message();
        assert_eq!(msg.content.len(), 2);
        assert!(matches!(
            &msg.content[0],
            ContentBlock::Text { text } if text == "Let me check"
        ));
        assert!(matches!(&msg.content[1], ContentBlock::ToolUse { .. }));
    }

    #[test]
    fn test_accumulator_text_after_tool_use() {
        let mut acc = StreamAccumulator::new();
        acc.push_text_delta("before");
        acc.push_tool_use("call-1".into(), "tool".into(), serde_json::json!({}));
        acc.push_text_delta("after");

        let msg = acc.into_message();
        assert_eq!(msg.content.len(), 3);
        assert!(matches!(&msg.content[0], ContentBlock::Text { text } if text == "before"));
        assert!(matches!(&msg.content[1], ContentBlock::ToolUse { .. }));
        assert!(matches!(&msg.content[2], ContentBlock::Text { text } if text == "after"));
    }

    #[test]
    fn test_accumulator_multiple_tools() {
        let mut acc = StreamAccumulator::new();
        acc.push_text_delta("thinking");
        acc.push_tool_use("call-1".into(), "tool_a".into(), serde_json::json!({}));
        acc.push_tool_use("call-2".into(), "tool_b".into(), serde_json::json!({}));

        let msg = acc.into_message();
        // Text flushed before first tool, second tool has no preceding text
        assert_eq!(msg.content.len(), 3);
        assert!(matches!(&msg.content[0], ContentBlock::Text { .. }));
        assert!(matches!(&msg.content[1], ContentBlock::ToolUse { id, .. } if id == "call-1"));
        assert!(matches!(&msg.content[2], ContentBlock::ToolUse { id, .. } if id == "call-2"));
    }

    #[test]
    fn test_accumulator_empty_buffer_not_flushed() {
        let mut acc = StreamAccumulator::new();
        acc.push_tool_use("call-1".into(), "tool".into(), serde_json::json!({}));

        let msg = acc.into_message();
        assert_eq!(msg.content.len(), 1);
        assert!(matches!(&msg.content[0], ContentBlock::ToolUse { .. }));
    }

    #[test]
    fn test_accumulator_into_message_role_and_timestamp() {
        let before = Utc::now();
        let acc = StreamAccumulator::new();
        let msg = acc.into_message();
        let after = Utc::now();

        assert_eq!(msg.role, Role::Assistant);
        assert!(msg.timestamp >= before && msg.timestamp <= after);
    }

    #[test]
    fn test_accumulator_into_message_empty() {
        let acc = StreamAccumulator::new();
        let msg = acc.into_message();

        assert_eq!(msg.role, Role::Assistant);
        assert!(msg.content.is_empty());
    }

    #[test]
    fn test_accumulator_is_empty_after_text() {
        let mut acc = StreamAccumulator::new();
        acc.push_text_delta("some text");
        assert!(!acc.is_empty());
    }

    #[test]
    fn test_accumulator_default() {
        let acc = StreamAccumulator::default();
        assert!(acc.is_empty());
    }
}
