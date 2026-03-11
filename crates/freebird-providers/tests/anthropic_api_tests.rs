//! Wiremock-based API contract tests for the Anthropic provider.
//!
//! These tests verify wire protocol correctness by mocking the Anthropic
//! Messages API and asserting that:
//! - HTTP requests are correctly formatted
//! - HTTP responses (success + error) are correctly parsed
//! - SSE streaming events are correctly processed
//! - Edge cases (timeouts, partial chunks, mid-stream errors) are handled
//!
//! This file establishes the test pattern for future providers (#42 OpenAI,
//! #43 Ollama, #44 Gemini).

#![allow(
    clippy::all,
    clippy::pedantic,
    clippy::nursery,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic
)]

mod helpers;

use std::time::Duration;

use freebird_traits::provider::{
    ContentBlock, NetworkErrorKind, Provider, ProviderError, StopReason, StreamEvent,
};
use futures::StreamExt;
use helpers::{
    ApiResponseBuilder, SseBuilder, error_response_json, make_provider, make_provider_with_timeout,
    request_with_tools, simple_request,
};
use serde_json::json;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

// ---------------------------------------------------------------------------
// Test 1: Request format verification
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_complete_request_format() {
    let server = MockServer::start().await;

    let response = ApiResponseBuilder::text("Hello!").build();
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(ResponseTemplate::new(200).set_body_json(&response))
        .mount(&server)
        .await;

    let provider = make_provider(&server.uri());
    let request = request_with_tools();
    let _ = provider.complete(request).await.unwrap();

    // Inspect the actual HTTP request body
    let received = server.received_requests().await.unwrap();
    assert_eq!(received.len(), 1);
    let body: serde_json::Value = serde_json::from_slice(&received[0].body).unwrap();

    // model is a top-level string
    assert_eq!(body["model"], "claude-opus-4-6");

    // system prompt is top-level, not in messages
    assert_eq!(body["system"], "You are helpful.");

    // messages array contains only user/assistant roles (no system)
    let messages = body["messages"].as_array().unwrap();
    assert!(!messages.is_empty());
    for msg in messages {
        let role = msg["role"].as_str().unwrap();
        assert!(
            role == "user" || role == "assistant",
            "unexpected role in messages: {role}"
        );
    }

    // tools array present with correct structure
    let tools = body["tools"].as_array().unwrap();
    assert_eq!(tools.len(), 2);
    assert_eq!(tools[0]["name"], "list_directory");
    assert_eq!(tools[1]["name"], "read_file");
    assert!(tools[0]["input_schema"]["properties"]["path"].is_object());

    // max_tokens present
    assert_eq!(body["max_tokens"], 4096);

    // temperature present
    assert_eq!(body["temperature"], 0.7);

    // stream should NOT be present for non-streaming calls
    assert!(body.get("stream").is_none());

    // Required headers
    let headers = &received[0].headers;
    assert_eq!(
        headers.get("anthropic-version").unwrap().to_str().unwrap(),
        "2023-06-01"
    );
    assert_eq!(
        headers.get("content-type").unwrap().to_str().unwrap(),
        "application/json"
    );
    assert_eq!(
        headers.get("x-api-key").unwrap().to_str().unwrap(),
        "test-api-key-12345"
    );
}

// ---------------------------------------------------------------------------
// Test 2: Response parsing (text + tool_use)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_complete_response_parsing() {
    let server = MockServer::start().await;

    let response = ApiResponseBuilder::text("Here are the files:")
        .and_tool_use("tool_001", "list_directory", json!({"path": "/tmp"}))
        .with_stop_reason("tool_use")
        .with_usage(100, 200)
        .build();

    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(ResponseTemplate::new(200).set_body_json(&response))
        .mount(&server)
        .await;

    let provider = make_provider(&server.uri());
    let result = provider.complete(simple_request()).await.unwrap();

    // Verify content blocks
    assert_eq!(result.message.content.len(), 2);
    match &result.message.content[0] {
        ContentBlock::Text { text } => assert_eq!(text, "Here are the files:"),
        other => panic!("expected Text, got: {other:?}"),
    }
    match &result.message.content[1] {
        ContentBlock::ToolUse { id, name, input } => {
            assert_eq!(id, "tool_001");
            assert_eq!(name, "list_directory");
            assert_eq!(input, &json!({"path": "/tmp"}));
        }
        other => panic!("expected ToolUse, got: {other:?}"),
    }

    assert_eq!(result.stop_reason, StopReason::ToolUse);
    assert_eq!(result.usage.input_tokens, 100);
    assert_eq!(result.usage.output_tokens, 200);
}

// ---------------------------------------------------------------------------
// Test 3: SSE streaming events
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_stream_sse_events() {
    let server = MockServer::start().await;

    let sse_body = SseBuilder::new()
        .message_start(50)
        .ping()
        .text_block_start(0)
        .text_delta("Hello, ")
        .text_delta("world!")
        .content_block_stop(0)
        .message_delta("end_turn", 10)
        .message_stop()
        .build();

    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(ResponseTemplate::new(200).set_body_raw(sse_body, "text/event-stream"))
        .mount(&server)
        .await;

    let provider = make_provider(&server.uri());
    let stream = provider.stream(simple_request()).await.unwrap();
    let events: Vec<StreamEvent> = stream.filter_map(|r| async { r.ok() }).collect().await;

    // Should get: TextDelta("Hello, "), TextDelta("world!"), Done
    assert!(
        events.len() >= 3,
        "expected at least 3 events, got {}",
        events.len()
    );

    match &events[0] {
        StreamEvent::TextDelta(t) => assert_eq!(t, "Hello, "),
        other => panic!("expected TextDelta, got: {other:?}"),
    }
    match &events[1] {
        StreamEvent::TextDelta(t) => assert_eq!(t, "world!"),
        other => panic!("expected TextDelta, got: {other:?}"),
    }
    match &events[events.len() - 1] {
        StreamEvent::Done { stop_reason, usage } => {
            assert_eq!(*stop_reason, StopReason::EndTurn);
            assert_eq!(usage.input_tokens, 50);
            assert_eq!(usage.output_tokens, 10);
        }
        other => panic!("expected Done, got: {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// Test 4: SSE streaming with tool use
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_stream_partial_chunks() {
    let server = MockServer::start().await;

    // Stream a tool_use response: the tool JSON arrives in partial deltas
    let sse_body = SseBuilder::new()
        .message_start(30)
        .tool_start(0, "tool_123", "read_file")
        .tool_delta(0, "{\"pa")
        .tool_delta(0, "th\": \"/tmp")
        .tool_delta(0, "/file.txt\"}")
        .content_block_stop(0)
        .message_delta("tool_use", 20)
        .message_stop()
        .build();

    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(ResponseTemplate::new(200).set_body_raw(sse_body, "text/event-stream"))
        .mount(&server)
        .await;

    let provider = make_provider(&server.uri());
    let stream = provider.stream(simple_request()).await.unwrap();
    let events: Vec<StreamEvent> = stream.filter_map(|r| async { r.ok() }).collect().await;

    // Should get: ToolUse (accumulated from deltas), Done
    assert_eq!(events.len(), 2);
    match &events[0] {
        StreamEvent::ToolUse { id, name, input } => {
            assert_eq!(id, "tool_123");
            assert_eq!(name, "read_file");
            assert_eq!(input, &json!({"path": "/tmp/file.txt"}));
        }
        other => panic!("expected ToolUse, got: {other:?}"),
    }
    match &events[1] {
        StreamEvent::Done { stop_reason, .. } => {
            assert_eq!(*stop_reason, StopReason::ToolUse);
        }
        other => panic!("expected Done, got: {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// Test 5: Stream error mid-stream
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_stream_error_mid_stream() {
    let server = MockServer::start().await;

    // Send text deltas then abruptly end (no message_delta or message_stop)
    let sse_body = SseBuilder::new()
        .message_start(30)
        .text_block_start(0)
        .text_delta("Partial ")
        .text_delta("response")
        .build(); // No message_delta/message_stop — stream ends abruptly

    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(ResponseTemplate::new(200).set_body_raw(sse_body, "text/event-stream"))
        .mount(&server)
        .await;

    let provider = make_provider(&server.uri());
    let stream = provider.stream(simple_request()).await.unwrap();
    let events: Vec<Result<StreamEvent, ProviderError>> = stream.collect().await;

    // Should get text deltas but no Done event — stream terminates cleanly
    let ok_events: Vec<_> = events.iter().filter_map(|r| r.as_ref().ok()).collect();
    assert!(ok_events.len() >= 2, "expected at least 2 text deltas");
    match &ok_events[0] {
        StreamEvent::TextDelta(t) => assert_eq!(t, "Partial "),
        other => panic!("expected TextDelta, got: {other:?}"),
    }

    // No Done event should have been produced (stream ended without message_delta)
    let has_done = ok_events
        .iter()
        .any(|e| matches!(e, StreamEvent::Done { .. }));
    assert!(
        !has_done,
        "should not have a Done event when stream ends abruptly"
    );
}

// ---------------------------------------------------------------------------
// Test 6: Rate limit (429)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_rate_limit_429() {
    let server = MockServer::start().await;

    // With retry-after header
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(
            ResponseTemplate::new(429)
                .append_header("retry-after", "30")
                .set_body_json(error_response_json("rate_limit_error", "Rate limited")),
        )
        .mount(&server)
        .await;

    let provider = make_provider(&server.uri());
    let err = provider.complete(simple_request()).await.unwrap_err();

    match err {
        ProviderError::RateLimited { retry_after_ms } => {
            assert_eq!(
                retry_after_ms, 30_000,
                "retry-after should be converted to ms"
            );
        }
        other => panic!("expected RateLimited, got: {other:?}"),
    }
}

#[tokio::test]
async fn test_rate_limit_429_no_retry_after() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(
            ResponseTemplate::new(429)
                .set_body_json(error_response_json("rate_limit_error", "Rate limited")),
        )
        .mount(&server)
        .await;

    let provider = make_provider(&server.uri());
    let err = provider.complete(simple_request()).await.unwrap_err();

    match err {
        ProviderError::RateLimited { retry_after_ms } => {
            assert_eq!(retry_after_ms, 1000, "should use DEFAULT_RETRY_AFTER_MS");
        }
        other => panic!("expected RateLimited, got: {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// Test 7: Auth error (401)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_auth_error_401() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(
            ResponseTemplate::new(401).set_body_json(error_response_json(
                "authentication_error",
                "invalid x-api-key",
            )),
        )
        .mount(&server)
        .await;

    let provider = make_provider(&server.uri());
    let err = provider.complete(simple_request()).await.unwrap_err();

    match err {
        ProviderError::AuthenticationFailed { reason } => {
            assert!(
                reason.contains("invalid x-api-key"),
                "expected error message, got: {reason}"
            );
        }
        other => panic!("expected AuthenticationFailed, got: {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// Test 8: Server error (500)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_server_error_500() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(ResponseTemplate::new(500).set_body_string("Internal Server Error"))
        .mount(&server)
        .await;

    let provider = make_provider(&server.uri());
    let err = provider.complete(simple_request()).await.unwrap_err();

    match err {
        ProviderError::ApiError { status, body } => {
            assert_eq!(status, 500);
            assert!(body.contains("Internal Server Error"));
        }
        other => panic!("expected ApiError, got: {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// Test 9: Timeout
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_timeout() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(ApiResponseBuilder::text("too late").build())
                .set_delay(Duration::from_secs(10)),
        )
        .mount(&server)
        .await;

    // 1-second timeout — response delayed by 10 seconds
    let provider = make_provider_with_timeout(&server.uri(), 1);
    let err = provider.complete(simple_request()).await.unwrap_err();

    match err {
        ProviderError::Network { kind, .. } => {
            assert_eq!(kind, NetworkErrorKind::Timeout);
        }
        other => panic!("expected Network(Timeout), got: {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// Test 10: Invalid JSON response
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_invalid_json_response() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(ResponseTemplate::new(200).set_body_string("this is not json"))
        .mount(&server)
        .await;

    let provider = make_provider(&server.uri());
    let err = provider.complete(simple_request()).await.unwrap_err();

    match err {
        ProviderError::Deserialization(msg) => {
            assert!(!msg.is_empty(), "error message should be non-empty");
        }
        other => panic!("expected Deserialization, got: {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// Test 11: Token usage parsing (including cache tokens)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_token_usage_parsed() {
    let server = MockServer::start().await;

    let response = ApiResponseBuilder::text("response")
        .with_usage(150, 75)
        .with_cache_tokens(50, 25)
        .build();

    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(ResponseTemplate::new(200).set_body_json(&response))
        .mount(&server)
        .await;

    let provider = make_provider(&server.uri());
    let result = provider.complete(simple_request()).await.unwrap();

    assert_eq!(result.usage.input_tokens, 150);
    assert_eq!(result.usage.output_tokens, 75);
    assert_eq!(result.usage.cache_read_tokens, Some(50));
    assert_eq!(result.usage.cache_creation_tokens, Some(25));
}

#[tokio::test]
async fn test_token_usage_parsed_streaming() {
    let server = MockServer::start().await;

    let sse_body = SseBuilder::new()
        .message_start_with_cache(100, 30, 15)
        .text_block_start(0)
        .text_delta("Hi")
        .content_block_stop(0)
        .message_delta("end_turn", 42)
        .message_stop()
        .build();

    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(ResponseTemplate::new(200).set_body_raw(sse_body, "text/event-stream"))
        .mount(&server)
        .await;

    let provider = make_provider(&server.uri());
    let stream = provider.stream(simple_request()).await.unwrap();
    let events: Vec<StreamEvent> = stream.filter_map(|r| async { r.ok() }).collect().await;

    // Find the Done event and check merged usage
    let done = events
        .iter()
        .find(|e| matches!(e, StreamEvent::Done { .. }))
        .expect("should have a Done event");

    match done {
        StreamEvent::Done { usage, .. } => {
            assert_eq!(usage.input_tokens, 100);
            assert_eq!(usage.output_tokens, 42);
            assert_eq!(usage.cache_read_tokens, Some(30));
            assert_eq!(usage.cache_creation_tokens, Some(15));
        }
        _ => unreachable!(),
    }
}

// ---------------------------------------------------------------------------
// Test 12: Tool definitions serialized in request
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_tool_definitions_serialized() {
    let server = MockServer::start().await;

    let response = ApiResponseBuilder::text("I'll help.").build();
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(ResponseTemplate::new(200).set_body_json(&response))
        .mount(&server)
        .await;

    let provider = make_provider(&server.uri());
    let _ = provider.complete(request_with_tools()).await.unwrap();

    let received = server.received_requests().await.unwrap();
    let body: serde_json::Value = serde_json::from_slice(&received[0].body).unwrap();

    let tools = body["tools"].as_array().unwrap();
    assert_eq!(tools.len(), 2);

    // First tool: list_directory
    let tool0 = &tools[0];
    assert_eq!(tool0["name"], "list_directory");
    assert_eq!(tool0["description"], "List files in a directory");
    let schema0 = &tool0["input_schema"];
    assert_eq!(schema0["type"], "object");
    assert!(schema0["properties"]["path"].is_object());
    assert_eq!(schema0["required"], json!(["path"]));

    // Second tool: read_file
    let tool1 = &tools[1];
    assert_eq!(tool1["name"], "read_file");
    assert_eq!(tool1["description"], "Read a file's contents");
    let schema1 = &tool1["input_schema"];
    assert_eq!(schema1["type"], "object");
    assert!(schema1["properties"]["path"].is_object());
    assert_eq!(schema1["required"], json!(["path"]));
}

// ---------------------------------------------------------------------------
// Streaming request format: verify `stream: true` is set
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_stream_request_has_stream_true() {
    let server = MockServer::start().await;

    let sse_body = SseBuilder::new()
        .message_start(10)
        .text_block_start(0)
        .text_delta("ok")
        .content_block_stop(0)
        .message_delta("end_turn", 5)
        .message_stop()
        .build();

    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(ResponseTemplate::new(200).set_body_raw(sse_body, "text/event-stream"))
        .mount(&server)
        .await;

    let provider = make_provider(&server.uri());
    let stream = provider.stream(simple_request()).await.unwrap();
    // Consume the stream to trigger the request
    let _: Vec<_> = stream.collect::<Vec<_>>().await;

    let received = server.received_requests().await.unwrap();
    let body: serde_json::Value = serde_json::from_slice(&received[0].body).unwrap();
    assert_eq!(
        body["stream"], true,
        "streaming request must have stream: true"
    );
}
