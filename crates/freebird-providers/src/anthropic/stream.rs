//! SSE (Server-Sent Events) parsing and the `SseStreamState` state machine
//! for processing Anthropic's streaming responses.

use std::pin::Pin;

use freebird_traits::provider::{ProviderError, StreamEvent, TokenUsage};
use futures::{Stream, StreamExt as _};

use super::client::{classify_reqwest_error, parse_stop_reason};
use super::types::{ApiStreamError, ApiStreamMessage, ApiStreamUsage, ApiUsage};

/// Maximum SSE buffer size (10 MiB). Prevents unbounded memory growth
/// if the API sends anomalous data or never sends `\n\n` delimiters.
const MAX_SSE_BUFFER_BYTES: usize = 10 * 1024 * 1024;

// ---------------------------------------------------------------------------
// SSE stream state
// ---------------------------------------------------------------------------

/// Internal state for the SSE stream processor.
/// Owned by the `futures::stream::unfold` closure.
pub struct SseStreamState {
    /// Raw byte stream from reqwest, mapped to `Vec<u8>` to avoid a direct `bytes` crate dependency.
    pub byte_stream: Pin<Box<dyn Stream<Item = Result<Vec<u8>, reqwest::Error>> + Send>>,
    /// Buffer for incomplete SSE events (bytes may arrive mid-event).
    pub buffer: String,
    /// Active `tool_use` accumulator (set on `content_block_start`, consumed on `content_block_stop`).
    pub active_tool: Option<ToolAccumulator>,
    /// Token usage captured from `message_start` (carries `input_tokens`).
    pub initial_usage: Option<ApiUsage>,
    /// Whether the stream has terminated.
    pub done: bool,
}

/// Accumulates partial JSON for a `tool_use` content block.
pub struct ToolAccumulator {
    /// Block index from `content_block_start` (used to correlate with `content_block_stop`).
    index: usize,
    /// Tool invocation ID from the API.
    id: String,
    /// Tool name.
    name: String,
    /// Accumulated partial JSON string (concatenated `input_json_delta` values).
    json_parts: String,
}

// ---------------------------------------------------------------------------
// SSE chunk parser
// ---------------------------------------------------------------------------

/// Extract the `data:` payload from a raw SSE event chunk.
///
/// SSE spec rules handled:
/// - Lines starting with `:` are comments (ignored)
/// - `data:` lines are concatenated with `\n` (multi-line data support)
/// - `event:`, `id:`, `retry:` lines are ignored (event type comes from JSON `type` field)
/// - Leading single space after `:` in field names is stripped (per SSE spec)
///
/// Returns `None` if the chunk has no `data:` lines.
pub fn parse_sse_chunk(chunk: &str) -> Option<String> {
    let mut data_parts: Vec<&str> = Vec::with_capacity(1);

    for line in chunk.lines() {
        if line.starts_with(':') {
            // Comment line — skip
            continue;
        }
        if let Some(rest) = line.strip_prefix("data:") {
            // Strip optional single leading space per SSE spec
            let payload = rest.strip_prefix(' ').unwrap_or(rest);
            data_parts.push(payload);
        }
        // Ignore event:, id:, retry:, and other field lines
    }

    if data_parts.is_empty() {
        None
    } else {
        Some(data_parts.join("\n"))
    }
}

// ---------------------------------------------------------------------------
// Event processing
// ---------------------------------------------------------------------------

impl SseStreamState {
    /// Handle a `content_block_delta` SSE event.
    fn process_content_block_delta(
        &mut self,
        value: &serde_json::Value,
    ) -> Option<Result<StreamEvent, ProviderError>> {
        let delta = value.get("delta")?;

        let delta_type = delta
            .get("type")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("");

        match delta_type {
            "text_delta" => {
                let text = delta
                    .get("text")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("")
                    .to_string();
                Some(Ok(StreamEvent::TextDelta(text)))
            }
            "input_json_delta" => {
                let partial_json = delta
                    .get("partial_json")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("");

                if let Some(ref mut tool) = self.active_tool {
                    tool.json_parts.push_str(partial_json);
                } else {
                    tracing::warn!("input_json_delta received without active tool accumulator");
                }
                None
            }
            other => {
                tracing::debug!(delta_type = other, "unknown delta type, skipping");
                None
            }
        }
    }

    /// Handle a `message_delta` SSE event (emits `Done`).
    fn process_message_delta(&self, value: &serde_json::Value) -> StreamEvent {
        let delta = value.get("delta");
        let stop_reason_str = delta
            .and_then(|d| d.get("stop_reason"))
            .and_then(serde_json::Value::as_str);
        let stop_reason = parse_stop_reason(stop_reason_str);

        // Merge usage: prefer message_delta fields, fall back to initial_usage
        let usage = value.get("usage").map_or_else(
            || {
                // No usage in message_delta — use initial_usage if available
                self.initial_usage
                    .as_ref()
                    .map(|u| TokenUsage {
                        input_tokens: u.input_tokens,
                        output_tokens: u.output_tokens,
                        cache_read_tokens: u.cache_read_input_tokens,
                        cache_creation_tokens: u.cache_creation_input_tokens,
                    })
                    .unwrap_or_default()
            },
            |usage_value| {
                let stream_usage: ApiStreamUsage = serde_json::from_value(usage_value.clone())
                    .unwrap_or(ApiStreamUsage {
                        output_tokens: 0,
                        input_tokens: None,
                        cache_read_input_tokens: None,
                        cache_creation_input_tokens: None,
                    });

                let input_tokens = stream_usage
                    .input_tokens
                    .or_else(|| self.initial_usage.as_ref().map(|u| u.input_tokens))
                    .unwrap_or(0);

                TokenUsage {
                    input_tokens,
                    output_tokens: stream_usage.output_tokens,
                    cache_read_tokens: stream_usage.cache_read_input_tokens.or_else(|| {
                        self.initial_usage
                            .as_ref()
                            .and_then(|u| u.cache_read_input_tokens)
                    }),
                    cache_creation_tokens: stream_usage.cache_creation_input_tokens.or_else(|| {
                        self.initial_usage
                            .as_ref()
                            .and_then(|u| u.cache_creation_input_tokens)
                    }),
                }
            },
        );

        StreamEvent::Done { stop_reason, usage }
    }

    /// Handle a `content_block_start` SSE event (sets up tool accumulator).
    fn process_content_block_start(&mut self, value: &serde_json::Value) {
        // Content block indices are small non-negative integers from the API; truncation is safe.
        #[expect(clippy::cast_possible_truncation, reason = "value range checked")]
        let index = value
            .get("index")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0) as usize;

        if let Some(content_block) = value.get("content_block") {
            let block_type = content_block
                .get("type")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("");

            match block_type {
                "tool_use" => {
                    let id = content_block
                        .get("id")
                        .and_then(serde_json::Value::as_str)
                        .unwrap_or("")
                        .to_string();
                    let name = content_block
                        .get("name")
                        .and_then(serde_json::Value::as_str)
                        .unwrap_or("")
                        .to_string();

                    self.active_tool = Some(ToolAccumulator {
                        index,
                        id,
                        name,
                        json_parts: String::new(),
                    });
                }
                "text" => {
                    // No-op: text blocks don't need accumulation
                }
                other => {
                    tracing::debug!(block_type = other, "unknown content_block type, skipping");
                }
            }
        }
    }

    /// Handle a `content_block_stop` SSE event (finalizes tool accumulation).
    fn process_content_block_stop(
        &mut self,
        value: &serde_json::Value,
    ) -> Option<Result<StreamEvent, ProviderError>> {
        // Content block indices are small non-negative integers from the API; truncation is safe.
        #[expect(clippy::cast_possible_truncation, reason = "value range checked")]
        let index = value
            .get("index")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0) as usize;

        // Finalize the active tool accumulator if this stop matches its index
        if let Some(tool) = self.active_tool.take().filter(|t| t.index == index) {
            match serde_json::from_str::<serde_json::Value>(&tool.json_parts) {
                Ok(input) => {
                    return Some(Ok(StreamEvent::ToolUse {
                        id: tool.id,
                        name: tool.name,
                        input,
                    }));
                }
                Err(e) => {
                    return Some(Ok(StreamEvent::Error(format!(
                        "failed to parse tool input JSON: {e}"
                    ))));
                }
            }
        }
        // No active tool or index mismatch — text block ended, no-op
        None
    }

    /// Parse a single SSE JSON event and map it to a [`StreamEvent`].
    ///
    /// Returns `None` for internal events that don't produce user-visible output
    /// (e.g., `ping`, `message_start`, `message_stop`).
    pub fn process_event_data(&mut self, data: &str) -> Option<Result<StreamEvent, ProviderError>> {
        let value: serde_json::Value = match serde_json::from_str(data) {
            Ok(v) => v,
            Err(e) => {
                return Some(Err(ProviderError::Deserialization(format!(
                    "invalid JSON in SSE data: {e}"
                ))));
            }
        };

        let event_type = value
            .get("type")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("");

        match event_type {
            "message_start" => {
                // Capture usage from message.usage (contains `input_tokens`)
                if let Some(message_value) = value.get("message") {
                    if let Ok(msg) =
                        serde_json::from_value::<ApiStreamMessage>(message_value.clone())
                    {
                        self.initial_usage = Some(msg.usage);
                    }
                }
                None
            }

            "content_block_start" => {
                self.process_content_block_start(&value);
                None
            }

            "content_block_delta" => self.process_content_block_delta(&value),

            "content_block_stop" => self.process_content_block_stop(&value),

            "message_delta" => Some(Ok(self.process_message_delta(&value))),

            "message_stop" => {
                self.done = true;
                None
            }

            "ping" => None,

            "error" => {
                if let Some(error_value) = value.get("error") {
                    if let Ok(err) = serde_json::from_value::<ApiStreamError>(error_value.clone()) {
                        return Some(Ok(StreamEvent::Error(format!(
                            "{}: {}",
                            err.error_type, err.message
                        ))));
                    }
                }
                Some(Ok(StreamEvent::Error("unknown error in SSE stream".into())))
            }

            other => {
                tracing::debug!(event_type = other, "unknown SSE event type, skipping");
                None
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Stream unfold driver
// ---------------------------------------------------------------------------

/// Create a `Stream` from an SSE byte stream by buffering and parsing events.
///
/// This is the core streaming logic, factored out of the `Provider::stream()`
/// implementation so it can be tested independently.
pub fn unfold_sse_stream(
    state: SseStreamState,
) -> Pin<Box<dyn Stream<Item = Result<StreamEvent, ProviderError>> + Send>> {
    let stream = futures::stream::unfold(state, |mut state| async move {
        if state.done {
            return None;
        }

        loop {
            // Check if buffer contains a complete SSE event (\n\n boundary)
            if let Some(boundary) = state.buffer.find("\n\n") {
                // Split buffer in-place: drain the chunk + \n\n delimiter,
                // leaving the remainder in `state.buffer` without re-allocating.
                let chunk: String = state.buffer.drain(..boundary).collect();
                state.buffer.drain(..2); // skip \n\n delimiter

                if let Some(data) = parse_sse_chunk(&chunk) {
                    if let Some(result) = state.process_event_data(&data) {
                        return Some((result, state));
                    }
                    // None returned — internal event (ping, message_start, etc.)
                    // Continue loop to process next buffered event or read more bytes
                    continue;
                }
                // No data: lines in chunk — skip and continue
                continue;
            }

            // Check buffer size limit
            if state.buffer.len() > MAX_SSE_BUFFER_BYTES {
                state.done = true;
                return Some((
                    Err(ProviderError::Deserialization(format!(
                        "SSE buffer exceeded {MAX_SSE_BUFFER_BYTES} bytes"
                    ))),
                    state,
                ));
            }

            // Read next chunk from byte stream
            match state.byte_stream.next().await {
                Some(Ok(bytes)) => {
                    match std::str::from_utf8(&bytes) {
                        Ok(text) => {
                            state.buffer.push_str(text);
                            // Continue loop to check for complete events
                        }
                        Err(e) => {
                            state.done = true;
                            return Some((
                                Err(ProviderError::Deserialization(format!(
                                    "invalid UTF-8 in SSE stream: {e}"
                                ))),
                                state,
                            ));
                        }
                    }
                }
                Some(Err(e)) => {
                    state.done = true;
                    return Some((Err(classify_reqwest_error(&e)), state));
                }
                None => {
                    // Stream ended
                    state.done = true;
                    if state.buffer.trim().is_empty() {
                        return None; // Clean termination
                    }
                    // Non-empty buffer at end — try to process remaining data
                    let remaining = std::mem::take(&mut state.buffer);
                    if let Some(data) = parse_sse_chunk(&remaining) {
                        if let Some(result) = state.process_event_data(&data) {
                            return Some((result, state));
                        }
                    }
                    return None;
                }
            }
        }
    });

    Box::pin(stream)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic, clippy::indexing_slicing)]
mod tests {
    use super::*;
    use freebird_traits::provider::StopReason;

    // -----------------------------------------------------------------------
    // SSE parser unit tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_parse_sse_chunk_basic() {
        let chunk = "event: content_block_delta\ndata: {\"type\":\"content_block_delta\"}";
        let result = parse_sse_chunk(chunk);
        assert_eq!(
            result,
            Some("{\"type\":\"content_block_delta\"}".to_string())
        );
    }

    #[test]
    fn test_parse_sse_chunk_ignores_comments() {
        let chunk = ": this is a comment\ndata: {\"type\":\"ping\"}";
        let result = parse_sse_chunk(chunk);
        assert_eq!(result, Some("{\"type\":\"ping\"}".to_string()));
    }

    #[test]
    fn test_parse_sse_chunk_multi_line_data() {
        let chunk = "data: line1\ndata: line2\ndata: line3";
        let result = parse_sse_chunk(chunk);
        assert_eq!(result, Some("line1\nline2\nline3".to_string()));
    }

    #[test]
    fn test_parse_sse_chunk_no_data() {
        let chunk = "event: message_start\nid: 123";
        let result = parse_sse_chunk(chunk);
        assert_eq!(result, None);
    }

    #[test]
    fn test_parse_sse_chunk_data_no_space_after_colon() {
        let chunk = "data:{\"type\":\"ping\"}";
        let result = parse_sse_chunk(chunk);
        assert_eq!(result, Some("{\"type\":\"ping\"}".to_string()));
    }

    // -----------------------------------------------------------------------
    // Event processing unit tests
    // -----------------------------------------------------------------------

    fn make_stream_state() -> SseStreamState {
        SseStreamState {
            byte_stream: Box::pin(futures::stream::empty()),
            buffer: String::new(),
            active_tool: None,
            initial_usage: None,
            done: false,
        }
    }

    #[test]
    fn test_process_text_delta() {
        let mut state = make_stream_state();
        let data = r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}"#;
        let result = state.process_event_data(data);
        match result {
            Some(Ok(StreamEvent::TextDelta(text))) => assert_eq!(text, "Hello"),
            other => panic!("expected TextDelta, got {other:?}"),
        }
    }

    #[test]
    fn test_process_tool_use_accumulation() {
        let mut state = make_stream_state();

        // content_block_start: tool_use
        let start_data = r#"{"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"toolu_123","name":"read_file"}}"#;
        let result = state.process_event_data(start_data);
        assert!(result.is_none(), "content_block_start should return None");
        assert!(state.active_tool.is_some());

        // content_block_delta: input_json_delta (part 1)
        let delta1 = r#"{"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"path\":"}}"#;
        let result = state.process_event_data(delta1);
        assert!(result.is_none());

        // content_block_delta: input_json_delta (part 2)
        let delta2 = r#"{"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"\"test.txt\"}"}}"#;
        let result = state.process_event_data(delta2);
        assert!(result.is_none());

        // content_block_stop
        let stop_data = r#"{"type":"content_block_stop","index":1}"#;
        let result = state.process_event_data(stop_data);
        match result {
            Some(Ok(StreamEvent::ToolUse { id, name, input })) => {
                assert_eq!(id, "toolu_123");
                assert_eq!(name, "read_file");
                assert_eq!(input, serde_json::json!({"path": "test.txt"}));
            }
            other => panic!("expected ToolUse, got {other:?}"),
        }
        assert!(state.active_tool.is_none());
    }

    #[test]
    fn test_process_message_delta_end_turn() {
        let mut state = make_stream_state();
        state.initial_usage = Some(ApiUsage {
            input_tokens: 100,
            output_tokens: 0,
            cache_creation_input_tokens: None,
            cache_read_input_tokens: None,
        });

        let data = r#"{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":42}}"#;
        let result = state.process_event_data(data);
        match result {
            Some(Ok(StreamEvent::Done { stop_reason, usage })) => {
                assert_eq!(stop_reason, StopReason::EndTurn);
                assert_eq!(usage.input_tokens, 100);
                assert_eq!(usage.output_tokens, 42);
            }
            other => panic!("expected Done, got {other:?}"),
        }
    }

    #[test]
    fn test_process_message_delta_tool_use_stop() {
        let mut state = make_stream_state();
        let data = r#"{"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":30}}"#;
        let result = state.process_event_data(data);
        match result {
            Some(Ok(StreamEvent::Done { stop_reason, .. })) => {
                assert_eq!(stop_reason, StopReason::ToolUse);
            }
            other => panic!("expected Done with ToolUse, got {other:?}"),
        }
    }

    #[test]
    fn test_process_ping_ignored() {
        let mut state = make_stream_state();
        let data = r#"{"type":"ping"}"#;
        let result = state.process_event_data(data);
        assert!(result.is_none());
    }

    #[test]
    fn test_process_error_event() {
        let mut state = make_stream_state();
        let data =
            r#"{"type":"error","error":{"type":"overloaded_error","message":"API is overloaded"}}"#;
        let result = state.process_event_data(data);
        match result {
            Some(Ok(StreamEvent::Error(msg))) => {
                assert!(msg.contains("overloaded_error"));
                assert!(msg.contains("API is overloaded"));
            }
            other => panic!("expected Error, got {other:?}"),
        }
    }

    #[test]
    fn test_process_unknown_event_type_ignored() {
        let mut state = make_stream_state();
        let data = r#"{"type":"future_unknown_event","data":"something"}"#;
        let result = state.process_event_data(data);
        assert!(result.is_none());
    }

    #[test]
    fn test_process_unknown_delta_type_ignored() {
        let mut state = make_stream_state();
        let data = r#"{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"hmm"}}"#;
        let result = state.process_event_data(data);
        assert!(result.is_none());
    }

    #[test]
    fn test_process_message_start_captures_usage() {
        let mut state = make_stream_state();
        let data = r#"{"type":"message_start","message":{"id":"msg_1","model":"claude-opus-4-6","usage":{"input_tokens":150,"output_tokens":0}}}"#;
        let result = state.process_event_data(data);
        assert!(result.is_none());
        assert!(state.initial_usage.is_some());
        assert_eq!(state.initial_usage.as_ref().unwrap().input_tokens, 150);
    }

    #[test]
    fn test_process_content_block_stop_no_active_tool() {
        let mut state = make_stream_state();
        let data = r#"{"type":"content_block_stop","index":0}"#;
        let result = state.process_event_data(data);
        assert!(result.is_none());
    }

    #[test]
    fn test_process_invalid_tool_json() {
        let mut state = make_stream_state();

        // Start tool
        let start = r#"{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"t1","name":"test"}}"#;
        state.process_event_data(start);

        // Send malformed JSON partial
        let delta = r#"{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{invalid json"}}"#;
        state.process_event_data(delta);

        // Stop — should produce error because accumulated JSON is invalid
        let stop = r#"{"type":"content_block_stop","index":0}"#;
        let result = state.process_event_data(stop);
        match result {
            Some(Ok(StreamEvent::Error(msg))) => {
                assert!(
                    msg.contains("failed to parse tool input JSON"),
                    "unexpected error: {msg}"
                );
            }
            other => panic!("expected Error for invalid JSON, got {other:?}"),
        }
    }

    #[test]
    fn test_process_empty_partial_json() {
        let mut state = make_stream_state();

        // Start tool
        let start = r#"{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"t1","name":"test"}}"#;
        state.process_event_data(start);

        // Empty delta — should accumulate nothing but not error
        let delta = r#"{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":""}}"#;
        let result = state.process_event_data(delta);
        assert!(result.is_none());
        assert_eq!(state.active_tool.as_ref().unwrap().json_parts, "");
    }

    #[test]
    fn test_process_input_json_delta_no_active_tool() {
        let mut state = make_stream_state();
        let data = r#"{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"key\":\"val\"}"}}"#;
        let result = state.process_event_data(data);
        // Should return None and log a warning, not crash
        assert!(result.is_none());
    }
}
