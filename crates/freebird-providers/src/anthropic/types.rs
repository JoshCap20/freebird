//! Wire format types for the Anthropic Messages API.
//!
//! These structs map 1:1 to the JSON request/response bodies of
//! `POST /v1/messages` (both synchronous and streaming).

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Request types
// ---------------------------------------------------------------------------

/// Outbound request to POST /v1/messages.
#[derive(Debug, Serialize)]
pub struct ApiRequest {
    pub model: String,
    pub max_tokens: u32,
    pub messages: Vec<ApiMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub stop_sequences: Vec<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<ApiToolDefinition>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
}

/// A message in Anthropic's format.
#[derive(Debug, Serialize, Deserialize)]
pub struct ApiMessage {
    pub role: String,
    pub content: Vec<ApiContentBlock>,
}

/// Content block in Anthropic's wire format (internally tagged).
#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ApiContentBlock {
    #[serde(rename = "text")]
    Text { text: String },

    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },

    #[serde(rename = "tool_result")]
    ToolResult {
        tool_use_id: String,
        content: String,
        #[serde(default, skip_serializing_if = "std::ops::Not::not")]
        is_error: bool,
    },

    #[serde(rename = "image")]
    Image { source: ApiImageSource },
}

/// Image source block (Anthropic nests image data inside a `source` object).
#[derive(Debug, Serialize, Deserialize)]
pub struct ApiImageSource {
    #[serde(rename = "type")]
    pub source_type: String,
    pub media_type: String,
    pub data: String,
}

/// Tool definition in Anthropic's format.
#[derive(Debug, Serialize)]
pub struct ApiToolDefinition {
    pub name: String,
    pub description: String,
    pub input_schema: serde_json::Value,
}

// ---------------------------------------------------------------------------
// Response types
// ---------------------------------------------------------------------------

/// Inbound response from POST /v1/messages.
#[derive(Debug, Deserialize)]
pub struct ApiResponse {
    /// Deserialized for completeness; not currently used.
    #[allow(dead_code)] // fires in lib but not test (deserialized in tests)
    pub id: String,
    /// Deserialized for completeness; always "assistant" for responses.
    #[allow(dead_code)] // fires in lib but not test (deserialized in tests)
    pub role: String,
    pub content: Vec<ApiContentBlock>,
    pub model: String,
    pub stop_reason: Option<String>,
    pub usage: ApiUsage,
}

/// Token usage from the API response.
#[derive(Debug, Deserialize)]
#[expect(
    clippy::struct_field_names,
    reason = "field names match Anthropic API wire format"
)]
pub struct ApiUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
    #[serde(default)]
    pub cache_creation_input_tokens: Option<u32>,
    #[serde(default)]
    pub cache_read_input_tokens: Option<u32>,
}

/// Anthropic API error response body.
#[derive(Debug, Deserialize)]
pub struct ApiErrorResponse {
    /// Deserialized for completeness; not currently used.
    #[allow(dead_code)] // fires in lib but not test (deserialized in tests)
    #[serde(rename = "type")]
    pub error_type: String,
    pub error: ApiErrorDetail,
}

#[derive(Debug, Deserialize)]
pub struct ApiErrorDetail {
    /// Deserialized for completeness; not currently used.
    #[allow(dead_code)] // fires in lib but not test (deserialized in tests)
    #[serde(rename = "type")]
    pub error_type: String,
    pub message: String,
}

// ---------------------------------------------------------------------------
// SSE streaming types
// ---------------------------------------------------------------------------

/// The initial message metadata from the `message_start` SSE event.
/// Carries input token count — the ONLY place it appears in the stream.
#[derive(Debug, Deserialize)]
pub struct ApiStreamMessage {
    #[allow(dead_code)] // fires in lib but not test (deserialized in tests)
    pub id: String,
    #[allow(dead_code)] // fires in lib but not test (deserialized in tests)
    pub model: String,
    pub usage: ApiUsage,
}

/// Cumulative usage from the `message_delta` SSE event.
#[derive(Debug, Deserialize)]
#[expect(
    clippy::struct_field_names,
    reason = "field names match Anthropic API wire format"
)]
pub struct ApiStreamUsage {
    pub output_tokens: u32,
    #[serde(default)]
    pub input_tokens: Option<u32>,
    #[serde(default)]
    pub cache_read_input_tokens: Option<u32>,
    #[serde(default)]
    pub cache_creation_input_tokens: Option<u32>,
}

/// Error payload from SSE `error` event.
#[derive(Debug, Deserialize)]
pub struct ApiStreamError {
    #[serde(rename = "type")]
    pub error_type: String,
    pub message: String,
}

// ---------------------------------------------------------------------------
// Public configuration types
// ---------------------------------------------------------------------------

/// Anthropic-specific configuration.
#[derive(Debug, Clone)]
pub struct AnthropicConfig {
    /// API base URL override. `None` = `DEFAULT_BASE_URL`.
    pub base_url: Option<String>,
    /// Model override. `None` = `DEFAULT_MODEL`.
    pub default_model: Option<String>,
    /// HTTP request timeout override in seconds. `None` = `REQUEST_TIMEOUT_SECS` (300s).
    pub timeout_secs: Option<u64>,
}

/// How the provider authenticates with the Anthropic API.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AuthKind {
    /// Standard API key from console.anthropic.com -> `x-api-key` header.
    ApiKey,
    /// OAuth token from Claude Pro/Max subscription -> `Authorization: Bearer` header.
    OAuthToken,
}
