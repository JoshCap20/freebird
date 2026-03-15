//! HTTP request building, response parsing, and error classification for the
//! Anthropic Messages API.

use chrono::Utc;
use freebird_traits::id::ModelId;
use freebird_traits::provider::{
    CompletionRequest, CompletionResponse, ContentBlock, Message, NetworkErrorKind, ProviderError,
    Role, StopReason, TokenUsage,
};

use super::types::{
    ApiContentBlock, ApiErrorResponse, ApiImageSource, ApiMessage, ApiRequest, ApiResponse,
    ApiToolDefinition,
};

// ---------------------------------------------------------------------------
// Conversion functions
// ---------------------------------------------------------------------------

/// Convert internal `CompletionRequest` to Anthropic API request body.
///
/// Consumes the request to avoid cloning strings and values.
pub fn build_request_body(request: CompletionRequest) -> ApiRequest {
    let messages: Vec<ApiMessage> = request
        .messages
        .into_iter()
        .filter(|m| m.role != Role::System)
        .map(|m| ApiMessage {
            role: match m.role {
                // System is filtered above; included for exhaustiveness safety
                Role::User | Role::Tool | Role::System => "user".into(),
                Role::Assistant => "assistant".into(),
            },
            content: m.content.into_iter().map(convert_content_block).collect(),
        })
        .collect();

    let tools: Vec<ApiToolDefinition> = request
        .tools
        .into_iter()
        .map(|t| ApiToolDefinition {
            name: t.name,
            description: t.description,
            input_schema: t.input_schema,
        })
        .collect();

    ApiRequest {
        model: request.model.to_string(),
        max_tokens: request.max_tokens,
        messages,
        system: request.system_prompt,
        temperature: request.temperature,
        stop_sequences: request.stop_sequences,
        tools,
        stream: None,
    }
}

/// Convert an internal `ContentBlock` to the Anthropic wire format.
///
/// Takes ownership to move strings rather than clone them.
fn convert_content_block(block: ContentBlock) -> ApiContentBlock {
    match block {
        ContentBlock::Text { text } => ApiContentBlock::Text { text },
        ContentBlock::ToolUse { id, name, input } => ApiContentBlock::ToolUse { id, name, input },
        ContentBlock::ToolResult {
            tool_use_id,
            content,
            is_error,
        } => ApiContentBlock::ToolResult {
            tool_use_id,
            content,
            is_error,
        },
        ContentBlock::Image { media_type, data } => ApiContentBlock::Image {
            source: ApiImageSource {
                source_type: "base64".into(),
                media_type,
                data,
            },
        },
    }
}

impl ApiResponse {
    /// Convert `ApiResponse` to internal `CompletionResponse`.
    pub fn into_completion_response(self) -> CompletionResponse {
        let content: Vec<ContentBlock> = self
            .content
            .into_iter()
            .map(|block| match block {
                ApiContentBlock::Text { text } => ContentBlock::Text { text },
                ApiContentBlock::ToolUse { id, name, input } => {
                    ContentBlock::ToolUse { id, name, input }
                }
                ApiContentBlock::ToolResult {
                    tool_use_id,
                    content,
                    is_error,
                } => ContentBlock::ToolResult {
                    tool_use_id,
                    content,
                    is_error,
                },
                ApiContentBlock::Image { source } => ContentBlock::Image {
                    media_type: source.media_type,
                    data: source.data,
                },
            })
            .collect();

        CompletionResponse {
            message: Message {
                role: Role::Assistant,
                content,
                timestamp: Utc::now(),
            },
            stop_reason: parse_stop_reason(self.stop_reason.as_deref()),
            usage: TokenUsage {
                input_tokens: self.usage.input_tokens,
                output_tokens: self.usage.output_tokens,
                cache_read_tokens: self.usage.cache_read_input_tokens,
                cache_creation_tokens: self.usage.cache_creation_input_tokens,
            },
            model: ModelId::from(self.model),
        }
    }
}

/// Parse Anthropic `stop_reason` string to internal `StopReason` enum.
///
/// Unknown values default to `StopReason::EndTurn` with a warning (forward-compatible).
pub fn parse_stop_reason(s: Option<&str>) -> StopReason {
    match s {
        Some("tool_use") => StopReason::ToolUse,
        Some("max_tokens") => StopReason::MaxTokens,
        Some("stop_sequence") => StopReason::StopSequence,
        Some(unknown) if unknown != "end_turn" => {
            tracing::warn!(
                stop_reason = unknown,
                "unknown stop_reason, defaulting to EndTurn"
            );
            StopReason::EndTurn
        }
        // "end_turn", None, or any matched-but-guarded value
        _ => StopReason::EndTurn,
    }
}

// ---------------------------------------------------------------------------
// Error handling
// ---------------------------------------------------------------------------

/// Default retry-after delay when header is missing (ms).
pub const DEFAULT_RETRY_AFTER_MS: u64 = 1000;

/// Extract the error message from an API error response, with a fallback.
async fn extract_error_message(resp: reqwest::Response, fallback: &str) -> String {
    resp.json::<ApiErrorResponse>()
        .await
        .map_or_else(|_| fallback.to_string(), |e| e.error.message)
}

/// Map a non-success HTTP response to the appropriate `ProviderError`.
///
/// Handles 401, 429, and generic error responses with consistent classification.
pub async fn map_error_response(resp: reqwest::Response) -> ProviderError {
    let status = resp.status().as_u16();

    if status == 401 {
        let reason = extract_error_message(resp, "invalid API key").await;
        return ProviderError::AuthenticationFailed { reason };
    }

    if status == 429 {
        let retry_after_ms = resp
            .headers()
            .get("retry-after")
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.parse::<u64>().ok())
            .map_or(DEFAULT_RETRY_AFTER_MS, |secs| secs * 1000);
        return ProviderError::RateLimited { retry_after_ms };
    }

    let body = resp
        .text()
        .await
        .unwrap_or_else(|e| format!("<failed to read response body: {e}>"));
    ProviderError::ApiError { status, body }
}

/// Classify a reqwest error into `ProviderError` with `NetworkErrorKind`.
pub fn classify_reqwest_error(e: &reqwest::Error) -> ProviderError {
    let message = e.to_string();
    let lower = message.to_lowercase();

    let kind = if e.is_timeout() {
        NetworkErrorKind::Timeout
    } else if e.is_connect() {
        if lower.contains("dns") || lower.contains("resolve") {
            NetworkErrorKind::DnsFailure
        } else if lower.contains("refused") {
            NetworkErrorKind::ConnectionRefused
        } else {
            NetworkErrorKind::Other
        }
    } else if lower.contains("tls") || lower.contains("ssl") {
        NetworkErrorKind::TlsError
    } else {
        NetworkErrorKind::Other
    };

    ProviderError::Network {
        reason: message,
        kind,
        status_code: e.status().map(|s| s.as_u16()),
    }
}
