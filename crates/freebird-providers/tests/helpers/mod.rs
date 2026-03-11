//! Shared test helpers for provider integration tests.
//!
//! Provides factory functions and builders that construct valid Anthropic API
//! responses and SSE streams. Designed as a reusable template for future
//! provider test suites (OpenAI, Ollama, Gemini).

#![allow(
    clippy::all,
    clippy::pedantic,
    clippy::nursery,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    dead_code
)]

use chrono::Utc;
use freebird_providers::anthropic::{AnthropicConfig, AnthropicProvider};
use freebird_traits::id::ModelId;
use freebird_traits::provider::{CompletionRequest, ContentBlock, Message, Role, ToolDefinition};
use secrecy::SecretString;
use serde_json::{Value, json};

// ---------------------------------------------------------------------------
// Provider factories
// ---------------------------------------------------------------------------

/// Create a provider pointing at a test server with default timeout.
pub fn make_provider(base_url: &str) -> AnthropicProvider {
    AnthropicProvider::new(
        test_api_key(),
        AnthropicConfig {
            base_url: Some(base_url.to_string()),
            default_model: None,
            timeout_secs: None,
        },
    )
    .unwrap()
}

/// Create a provider with a custom timeout (for timeout testing).
pub fn make_provider_with_timeout(base_url: &str, timeout_secs: u64) -> AnthropicProvider {
    AnthropicProvider::new(
        test_api_key(),
        AnthropicConfig {
            base_url: Some(base_url.to_string()),
            default_model: None,
            timeout_secs: Some(timeout_secs),
        },
    )
    .unwrap()
}

/// Create a provider with an OAuth token for auth header testing.
pub fn make_oauth_provider(base_url: &str) -> AnthropicProvider {
    AnthropicProvider::new(
        SecretString::from("sk-ant-oat01-test-oauth-token-12345"),
        AnthropicConfig {
            base_url: Some(base_url.to_string()),
            default_model: None,
            timeout_secs: None,
        },
    )
    .unwrap()
}

pub fn test_api_key() -> SecretString {
    SecretString::from("test-api-key-12345")
}

/// A simple completion request with one user message and a system prompt.
pub fn simple_request() -> CompletionRequest {
    CompletionRequest {
        model: ModelId::from("claude-opus-4-6"),
        system_prompt: Some("You are helpful.".into()),
        messages: vec![Message {
            role: Role::User,
            content: vec![ContentBlock::Text {
                text: "Hello".into(),
            }],
            timestamp: Utc::now(),
        }],
        tools: vec![],
        max_tokens: 1024,
        temperature: None,
        stop_sequences: vec![],
    }
}

/// A completion request that includes tool definitions.
pub fn request_with_tools() -> CompletionRequest {
    CompletionRequest {
        model: ModelId::from("claude-opus-4-6"),
        system_prompt: Some("You are helpful.".into()),
        messages: vec![Message {
            role: Role::User,
            content: vec![ContentBlock::Text {
                text: "What files are in the current directory?".into(),
            }],
            timestamp: Utc::now(),
        }],
        tools: vec![
            ToolDefinition {
                name: "list_directory".into(),
                description: "List files in a directory".into(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Directory path"
                        }
                    },
                    "required": ["path"]
                }),
            },
            ToolDefinition {
                name: "read_file".into(),
                description: "Read a file's contents".into(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "File path"
                        }
                    },
                    "required": ["path"]
                }),
            },
        ],
        max_tokens: 4096,
        temperature: Some(0.7),
        stop_sequences: vec![],
    }
}

// ---------------------------------------------------------------------------
// Response builders
// ---------------------------------------------------------------------------

/// Fluent builder for Anthropic API JSON responses.
pub struct ApiResponseBuilder {
    content: Vec<Value>,
    model: String,
    stop_reason: String,
    input_tokens: u32,
    output_tokens: u32,
    cache_read_tokens: Option<u32>,
    cache_creation_tokens: Option<u32>,
}

impl ApiResponseBuilder {
    /// Start building a response with a text content block.
    pub fn text(text: &str) -> Self {
        Self {
            content: vec![json!({
                "type": "text",
                "text": text
            })],
            model: "claude-opus-4-6".into(),
            stop_reason: "end_turn".into(),
            input_tokens: 25,
            output_tokens: 50,
            cache_read_tokens: None,
            cache_creation_tokens: None,
        }
    }

    /// Start building a response with a tool_use content block.
    pub fn tool_use(id: &str, name: &str, input: Value) -> Self {
        Self {
            content: vec![json!({
                "type": "tool_use",
                "id": id,
                "name": name,
                "input": input
            })],
            model: "claude-opus-4-6".into(),
            stop_reason: "tool_use".into(),
            input_tokens: 25,
            output_tokens: 50,
            cache_read_tokens: None,
            cache_creation_tokens: None,
        }
    }

    /// Add an additional text content block.
    pub fn and_text(mut self, text: &str) -> Self {
        self.content.push(json!({
            "type": "text",
            "text": text
        }));
        self
    }

    /// Add an additional tool_use content block.
    pub fn and_tool_use(mut self, id: &str, name: &str, input: Value) -> Self {
        self.content.push(json!({
            "type": "tool_use",
            "id": id,
            "name": name,
            "input": input
        }));
        self
    }

    pub fn with_stop_reason(mut self, reason: &str) -> Self {
        self.stop_reason = reason.into();
        self
    }

    pub fn with_usage(mut self, input: u32, output: u32) -> Self {
        self.input_tokens = input;
        self.output_tokens = output;
        self
    }

    pub fn with_cache_tokens(mut self, read: u32, creation: u32) -> Self {
        self.cache_read_tokens = Some(read);
        self.cache_creation_tokens = Some(creation);
        self
    }

    pub fn with_model(mut self, model: &str) -> Self {
        self.model = model.into();
        self
    }

    /// Build the final JSON value matching the Anthropic API response format.
    pub fn build(self) -> Value {
        let mut usage = json!({
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
        });
        if let Some(read) = self.cache_read_tokens {
            usage["cache_read_input_tokens"] = json!(read);
        }
        if let Some(creation) = self.cache_creation_tokens {
            usage["cache_creation_input_tokens"] = json!(creation);
        }

        json!({
            "id": "msg_test_001",
            "type": "message",
            "role": "assistant",
            "content": self.content,
            "model": self.model,
            "stop_reason": self.stop_reason,
            "usage": usage
        })
    }
}

// ---------------------------------------------------------------------------
// SSE stream builder
// ---------------------------------------------------------------------------

/// Fluent builder for SSE event streams matching Anthropic's streaming format.
pub struct SseBuilder {
    events: Vec<String>,
}

impl SseBuilder {
    pub fn new() -> Self {
        Self { events: vec![] }
    }

    /// Add a `message_start` event with input token count.
    pub fn message_start(mut self, input_tokens: u32) -> Self {
        self.events.push(format!(
            "event: message_start\ndata: {}\n",
            json!({
                "type": "message_start",
                "message": {
                    "id": "msg_stream_001",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-opus-4-6",
                    "usage": {
                        "input_tokens": input_tokens,
                        "output_tokens": 0
                    }
                }
            })
        ));
        self
    }

    /// Add a `message_start` event with cache token fields.
    pub fn message_start_with_cache(
        mut self,
        input_tokens: u32,
        cache_read: u32,
        cache_creation: u32,
    ) -> Self {
        self.events.push(format!(
            "event: message_start\ndata: {}\n",
            json!({
                "type": "message_start",
                "message": {
                    "id": "msg_stream_001",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-opus-4-6",
                    "usage": {
                        "input_tokens": input_tokens,
                        "output_tokens": 0,
                        "cache_read_input_tokens": cache_read,
                        "cache_creation_input_tokens": cache_creation
                    }
                }
            })
        ));
        self
    }

    /// Add a `content_block_start` event for a text block.
    pub fn text_block_start(mut self, index: usize) -> Self {
        self.events.push(format!(
            "event: content_block_start\ndata: {}\n",
            json!({
                "type": "content_block_start",
                "index": index,
                "content_block": { "type": "text", "text": "" }
            })
        ));
        self
    }

    /// Add a `content_block_delta` with text delta.
    pub fn text_delta(mut self, text: &str) -> Self {
        self.events.push(format!(
            "event: content_block_delta\ndata: {}\n",
            json!({
                "type": "content_block_delta",
                "index": 0,
                "delta": { "type": "text_delta", "text": text }
            })
        ));
        self
    }

    /// Add a `content_block_stop` event.
    pub fn content_block_stop(mut self, index: usize) -> Self {
        self.events.push(format!(
            "event: content_block_stop\ndata: {}\n",
            json!({
                "type": "content_block_stop",
                "index": index
            })
        ));
        self
    }

    /// Add a `content_block_start` event for a tool_use block.
    pub fn tool_start(mut self, index: usize, id: &str, name: &str) -> Self {
        self.events.push(format!(
            "event: content_block_start\ndata: {}\n",
            json!({
                "type": "content_block_start",
                "index": index,
                "content_block": {
                    "type": "tool_use",
                    "id": id,
                    "name": name,
                    "input": {}
                }
            })
        ));
        self
    }

    /// Add a `content_block_delta` with partial tool JSON.
    pub fn tool_delta(mut self, index: usize, partial_json: &str) -> Self {
        self.events.push(format!(
            "event: content_block_delta\ndata: {}\n",
            json!({
                "type": "content_block_delta",
                "index": index,
                "delta": {
                    "type": "input_json_delta",
                    "partial_json": partial_json
                }
            })
        ));
        self
    }

    /// Add a `message_delta` event (final event with stop reason + output tokens).
    pub fn message_delta(mut self, stop_reason: &str, output_tokens: u32) -> Self {
        self.events.push(format!(
            "event: message_delta\ndata: {}\n",
            json!({
                "type": "message_delta",
                "delta": { "stop_reason": stop_reason },
                "usage": { "output_tokens": output_tokens }
            })
        ));
        self
    }

    /// Add a `message_stop` event.
    pub fn message_stop(mut self) -> Self {
        self.events
            .push("event: message_stop\ndata: {\"type\": \"message_stop\"}\n".into());
        self
    }

    /// Add a `ping` event.
    pub fn ping(mut self) -> Self {
        self.events
            .push("event: ping\ndata: {\"type\": \"ping\"}\n".into());
        self
    }

    /// Add an `error` event.
    pub fn error(mut self, error_type: &str, message: &str) -> Self {
        self.events.push(format!(
            "event: error\ndata: {}\n",
            json!({
                "type": "error",
                "error": {
                    "type": error_type,
                    "message": message
                }
            })
        ));
        self
    }

    /// Build the complete SSE stream body as a string.
    pub fn build(self) -> String {
        self.events.join("\n")
    }
}

/// Build an Anthropic error response JSON body.
pub fn error_response_json(error_type: &str, message: &str) -> Value {
    json!({
        "type": "error",
        "error": {
            "type": error_type,
            "message": message
        }
    })
}
