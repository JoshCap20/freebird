//! Anthropic (Claude) provider implementation.

use std::collections::BTreeSet;
use std::pin::Pin;

use async_trait::async_trait;
use chrono::Utc;
use freebird_traits::id::{ModelId, ProviderId};
use freebird_traits::provider::{
    CompletionRequest, CompletionResponse, ContentBlock, Message, ModelInfo, NetworkErrorKind,
    Provider, ProviderError, ProviderFeature, ProviderInfo, Role, StopReason, StreamEvent,
    TokenUsage,
};
use futures::{Stream, StreamExt as _};
use reqwest::Client;
use secrecy::{ExposeSecret, SecretString};
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Anthropic API version header value.
const API_VERSION: &str = "2023-06-01";
/// Default base URL for the Anthropic API.
const DEFAULT_BASE_URL: &str = "https://api.anthropic.com";
/// Default model ID.
const DEFAULT_MODEL: &str = "claude-opus-4-6-20250929";
/// HTTP request timeout in seconds.
const REQUEST_TIMEOUT_SECS: u64 = 300;
/// Default retry-after delay when header is missing (ms).
const DEFAULT_RETRY_AFTER_MS: u64 = 1000;
/// Maximum SSE buffer size (10 MiB). Prevents unbounded memory growth
/// if the API sends anomalous data or never sends `\n\n` delimiters.
const MAX_SSE_BUFFER_BYTES: usize = 10 * 1024 * 1024;

// ---------------------------------------------------------------------------
// Layer 1: Private API serde types
// ---------------------------------------------------------------------------

/// Outbound request to POST /v1/messages.
#[derive(Debug, Serialize)]
struct ApiRequest {
    model: String,
    max_tokens: u32,
    messages: Vec<ApiMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    stop_sequences: Vec<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<ApiToolDefinition>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
}

/// A message in Anthropic's format.
#[derive(Debug, Serialize, Deserialize)]
struct ApiMessage {
    role: String,
    content: Vec<ApiContentBlock>,
}

/// Content block in Anthropic's wire format (internally tagged).
#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type")]
enum ApiContentBlock {
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
struct ApiImageSource {
    #[serde(rename = "type")]
    source_type: String,
    media_type: String,
    data: String,
}

/// Tool definition in Anthropic's format.
#[derive(Debug, Serialize)]
struct ApiToolDefinition {
    name: String,
    description: String,
    input_schema: serde_json::Value,
}

/// Inbound response from POST /v1/messages.
#[derive(Debug, Deserialize)]
struct ApiResponse {
    /// Deserialized for completeness; not currently used.
    #[allow(dead_code)]
    id: String,
    /// Deserialized for completeness; always "assistant" for responses.
    #[allow(dead_code)]
    role: String,
    content: Vec<ApiContentBlock>,
    model: String,
    stop_reason: Option<String>,
    usage: ApiUsage,
}

/// Token usage from the API response.
#[derive(Debug, Deserialize)]
#[allow(clippy::struct_field_names)]
struct ApiUsage {
    input_tokens: u32,
    output_tokens: u32,
    #[serde(default)]
    cache_creation_input_tokens: Option<u32>,
    #[serde(default)]
    cache_read_input_tokens: Option<u32>,
}

/// Anthropic API error response body.
#[derive(Debug, Deserialize)]
struct ApiErrorResponse {
    /// Deserialized for completeness; not currently used.
    #[allow(dead_code)]
    #[serde(rename = "type")]
    error_type: String,
    error: ApiErrorDetail,
}

#[derive(Debug, Deserialize)]
struct ApiErrorDetail {
    /// Deserialized for completeness; not currently used.
    #[allow(dead_code)]
    #[serde(rename = "type")]
    error_type: String,
    message: String,
}

// ---------------------------------------------------------------------------
// Layer 1b: SSE streaming types
// ---------------------------------------------------------------------------

/// The initial message metadata from the `message_start` SSE event.
/// Carries input token count — the ONLY place it appears in the stream.
#[derive(Debug, Deserialize)]
struct ApiStreamMessage {
    #[allow(dead_code)]
    id: String,
    #[allow(dead_code)]
    model: String,
    usage: ApiUsage,
}

/// Cumulative usage from the `message_delta` SSE event.
#[derive(Debug, Deserialize)]
#[allow(clippy::struct_field_names)]
struct ApiStreamUsage {
    output_tokens: u32,
    #[serde(default)]
    input_tokens: Option<u32>,
    #[serde(default)]
    cache_read_input_tokens: Option<u32>,
    #[serde(default)]
    cache_creation_input_tokens: Option<u32>,
}

/// Error payload from SSE `error` event.
#[derive(Debug, Deserialize)]
struct ApiStreamError {
    #[serde(rename = "type")]
    error_type: String,
    message: String,
}

/// Internal state for the SSE stream processor.
/// Owned by the `futures::stream::unfold` closure.
struct SseStreamState {
    /// Raw byte stream from reqwest, mapped to `Vec<u8>` to avoid a direct `bytes` crate dependency.
    byte_stream: Pin<Box<dyn Stream<Item = Result<Vec<u8>, reqwest::Error>> + Send>>,
    /// Buffer for incomplete SSE events (bytes may arrive mid-event).
    buffer: String,
    /// Active `tool_use` accumulator (set on `content_block_start`, consumed on `content_block_stop`).
    active_tool: Option<ToolAccumulator>,
    /// Token usage captured from `message_start` (carries `input_tokens`).
    initial_usage: Option<ApiUsage>,
    /// Whether the stream has terminated.
    done: bool,
}

/// Accumulates partial JSON for a `tool_use` content block.
struct ToolAccumulator {
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
// Layer 3: Public types
// ---------------------------------------------------------------------------

/// Anthropic-specific configuration.
#[derive(Debug, Clone)]
pub struct AnthropicConfig {
    /// API base URL override. `None` = `DEFAULT_BASE_URL`.
    pub base_url: Option<String>,
    /// Model override. `None` = `DEFAULT_MODEL`.
    pub default_model: Option<String>,
}

/// The Anthropic provider implementation.
pub struct AnthropicProvider {
    client: Client,
    api_key: SecretString,
    base_url: String,
    default_model: String,
    info: ProviderInfo,
}

impl AnthropicProvider {
    /// Construct a new Anthropic provider.
    ///
    /// # Errors
    ///
    /// Returns `ProviderError::Network` if the HTTP client fails to build
    /// (e.g., TLS initialization failure).
    pub fn new(api_key: SecretString, config: AnthropicConfig) -> Result<Self, ProviderError> {
        let client = Client::builder()
            .use_rustls_tls()
            .timeout(std::time::Duration::from_secs(REQUEST_TIMEOUT_SECS))
            .build()
            .map_err(|e| ProviderError::Network {
                reason: format!("failed to build HTTP client: {e}"),
                kind: NetworkErrorKind::TlsError,
                status_code: None,
            })?;

        let base_url = config
            .base_url
            .unwrap_or_else(|| DEFAULT_BASE_URL.to_string());
        let default_model = config
            .default_model
            .unwrap_or_else(|| DEFAULT_MODEL.to_string());

        let info = ProviderInfo {
            id: ProviderId::from("anthropic"),
            display_name: "Anthropic Claude".into(),
            supported_models: vec![
                ModelInfo {
                    id: ModelId::from("claude-opus-4-6-20250929"),
                    display_name: "Claude Opus 4.6".into(),
                    max_context_tokens: 200_000,
                    max_output_tokens: 32_768,
                },
                ModelInfo {
                    id: ModelId::from("claude-sonnet-4-5-20250929"),
                    display_name: "Claude Sonnet 4.5".into(),
                    max_context_tokens: 200_000,
                    max_output_tokens: 16_384,
                },
            ],
            features: BTreeSet::from([
                ProviderFeature::Streaming,
                ProviderFeature::ToolUse,
                ProviderFeature::Vision,
            ]),
        };

        Ok(Self {
            client,
            api_key,
            base_url,
            default_model,
            info,
        })
    }
}

// ---------------------------------------------------------------------------
// Layer 2: Conversion functions
// ---------------------------------------------------------------------------

/// Convert internal `CompletionRequest` to Anthropic API request body.
///
/// Consumes the request to avoid cloning strings and values.
fn build_request_body(request: CompletionRequest) -> ApiRequest {
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
    fn into_completion_response(self) -> CompletionResponse {
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
fn parse_stop_reason(s: Option<&str>) -> StopReason {
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

/// Extract the error message from an API error response, with a fallback.
async fn extract_error_message(resp: reqwest::Response, fallback: &str) -> String {
    resp.json::<ApiErrorResponse>()
        .await
        .map_or_else(|_| fallback.to_string(), |e| e.error.message)
}

/// Map a non-success HTTP response to the appropriate `ProviderError`.
///
/// Handles 401, 429, and generic error responses with consistent classification.
async fn map_error_response(resp: reqwest::Response) -> ProviderError {
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
fn classify_reqwest_error(e: &reqwest::Error) -> ProviderError {
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

// ---------------------------------------------------------------------------
// Layer 2b: SSE parsing functions
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
fn parse_sse_chunk(chunk: &str) -> Option<String> {
    let mut data_parts: Vec<&str> = Vec::new();

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

impl SseStreamState {
    /// Process a JSON `data:` payload from an SSE event.
    ///
    /// Uses two-stage parsing for forward compatibility: parse as [`serde_json::Value`],
    /// extract the `type` field, then deserialize only the relevant substructure.
    /// Unknown event types are silently skipped with a debug log.
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
        #[allow(clippy::cast_possible_truncation)]
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
        #[allow(clippy::cast_possible_truncation)]
        let index = value
            .get("index")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0) as usize;

        // Check if this stop matches our active tool accumulator
        if self.active_tool.as_ref().is_some_and(|t| t.index == index) {
            let tool = self.active_tool.take();
            if let Some(tool) = tool {
                // Parse accumulated JSON
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
        }
        // No active tool or index mismatch — text block ended, no-op
        None
    }

    /// Parse a single SSE JSON event and map it to a [`StreamEvent`].
    ///
    /// Returns `None` for internal events that don't produce user-visible output
    /// (e.g., `ping`, `message_start`, `message_stop`).
    fn process_event_data(&mut self, data: &str) -> Option<Result<StreamEvent, ProviderError>> {
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
// Layer 3: Provider trait implementation
// ---------------------------------------------------------------------------

#[async_trait]
impl Provider for AnthropicProvider {
    fn info(&self) -> &ProviderInfo {
        &self.info
    }

    async fn validate_credentials(&self) -> Result<(), ProviderError> {
        let body = serde_json::json!({
            "model": self.default_model,
            "max_tokens": 1,
            "messages": [{"role": "user", "content": "ping"}]
        });

        let resp = self
            .client
            .post(format!("{}/v1/messages", self.base_url))
            .header("x-api-key", self.api_key.expose_secret())
            .header("anthropic-version", API_VERSION)
            .header("content-type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|ref e| classify_reqwest_error(e))?;

        if !resp.status().is_success() {
            return Err(map_error_response(resp).await);
        }

        Ok(())
    }

    async fn complete(
        &self,
        request: CompletionRequest,
    ) -> Result<CompletionResponse, ProviderError> {
        let api_request = build_request_body(request);

        let resp = self
            .client
            .post(format!("{}/v1/messages", self.base_url))
            .header("x-api-key", self.api_key.expose_secret())
            .header("anthropic-version", API_VERSION)
            .header("content-type", "application/json")
            .json(&api_request)
            .send()
            .await
            .map_err(|ref e| classify_reqwest_error(e))?;

        if !resp.status().is_success() {
            return Err(map_error_response(resp).await);
        }

        let api_response: ApiResponse = resp
            .json()
            .await
            .map_err(|e| ProviderError::Deserialization(e.to_string()))?;

        Ok(api_response.into_completion_response())
    }

    async fn stream(
        &self,
        request: CompletionRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamEvent, ProviderError>> + Send>>, ProviderError>
    {
        let mut api_request = build_request_body(request);
        api_request.stream = Some(true);

        let resp = self
            .client
            .post(format!("{}/v1/messages", self.base_url))
            .header("x-api-key", self.api_key.expose_secret())
            .header("anthropic-version", API_VERSION)
            .header("content-type", "application/json")
            .json(&api_request)
            .send()
            .await
            .map_err(|ref e| classify_reqwest_error(e))?;

        if !resp.status().is_success() {
            return Err(map_error_response(resp).await);
        }

        let byte_stream = resp.bytes_stream().map(|result| result.map(|b| b.to_vec()));

        let state = SseStreamState {
            byte_stream: Box::pin(byte_stream),
            buffer: String::new(),
            active_tool: None,
            initial_usage: None,
            done: false,
        };

        let stream = futures::stream::unfold(state, |mut state| async move {
            if state.done {
                return None;
            }

            loop {
                // Check if buffer contains a complete SSE event (\n\n boundary)
                if let Some(boundary) = state.buffer.find("\n\n") {
                    let chunk = state.buffer[..boundary].to_string();
                    state.buffer = state.buffer[boundary + 2..].to_string();

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

        Ok(Box::pin(stream))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic, clippy::indexing_slicing)]
mod tests {
    use super::*;
    use freebird_traits::id::ModelId;
    use freebird_traits::provider::ToolDefinition;
    use secrecy::SecretString;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    // -----------------------------------------------------------------------
    // Test helpers
    // -----------------------------------------------------------------------

    fn test_api_key() -> SecretString {
        SecretString::from("test-api-key-12345")
    }

    fn make_provider(base_url: &str) -> AnthropicProvider {
        AnthropicProvider::new(
            test_api_key(),
            AnthropicConfig {
                base_url: Some(base_url.to_string()),
                default_model: None,
            },
        )
        .unwrap()
    }

    fn simple_completion_request() -> CompletionRequest {
        CompletionRequest {
            model: ModelId::from("claude-opus-4-6-20250929"),
            system_prompt: Some("You are helpful.".into()),
            messages: vec![
                Message {
                    role: Role::User,
                    content: vec![ContentBlock::Text {
                        text: "Hello".into(),
                    }],
                    timestamp: Utc::now(),
                },
                Message {
                    role: Role::Assistant,
                    content: vec![ContentBlock::Text {
                        text: "Hi there!".into(),
                    }],
                    timestamp: Utc::now(),
                },
            ],
            tools: vec![],
            max_tokens: 1024,
            temperature: Some(0.7),
            stop_sequences: vec![],
        }
    }

    fn success_response_json() -> serde_json::Value {
        serde_json::json!({
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Hello, world!"}
            ],
            "model": "claude-opus-4-6-20250929",
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": 25,
                "output_tokens": 10
            }
        })
    }

    fn error_response_json(error_type: &str, message: &str) -> serde_json::Value {
        serde_json::json!({
            "type": "error",
            "error": {
                "type": error_type,
                "message": message
            }
        })
    }

    // -----------------------------------------------------------------------
    // Unit tests: build_request_body
    // -----------------------------------------------------------------------

    #[test]
    fn test_build_request_body_basic() {
        let request = simple_completion_request();

        let api_req = build_request_body(request);

        assert_eq!(api_req.model, "claude-opus-4-6-20250929");
        assert_eq!(api_req.max_tokens, 1024);
        assert_eq!(api_req.messages.len(), 2);
        assert_eq!(api_req.messages[0].role, "user");
        assert_eq!(api_req.messages[1].role, "assistant");
        assert_eq!(api_req.system, Some("You are helpful.".into()));
    }

    #[test]
    fn test_build_request_body_system_prompt_in_top_level() {
        let request = CompletionRequest {
            system_prompt: Some("Be concise.".into()),
            ..simple_completion_request()
        };

        let api_req = build_request_body(request);

        assert_eq!(api_req.system, Some("Be concise.".into()));
        for msg in &api_req.messages {
            assert_ne!(msg.role, "system");
        }
    }

    #[test]
    fn test_build_request_body_tool_role_mapping() {
        let request = CompletionRequest {
            messages: vec![
                Message {
                    role: Role::User,
                    content: vec![ContentBlock::Text {
                        text: "Read file".into(),
                    }],
                    timestamp: Utc::now(),
                },
                Message {
                    role: Role::Tool,
                    content: vec![ContentBlock::ToolResult {
                        tool_use_id: "tu_1".into(),
                        content: "file contents".into(),
                        is_error: false,
                    }],
                    timestamp: Utc::now(),
                },
            ],
            ..simple_completion_request()
        };

        let api_req = build_request_body(request);

        assert_eq!(api_req.messages[1].role, "user");
    }

    #[test]
    fn test_build_request_body_system_messages_filtered() {
        let request = CompletionRequest {
            messages: vec![
                Message {
                    role: Role::System,
                    content: vec![ContentBlock::Text {
                        text: "secret system msg".into(),
                    }],
                    timestamp: Utc::now(),
                },
                Message {
                    role: Role::User,
                    content: vec![ContentBlock::Text {
                        text: "Hello".into(),
                    }],
                    timestamp: Utc::now(),
                },
            ],
            ..simple_completion_request()
        };

        let api_req = build_request_body(request);

        assert_eq!(api_req.messages.len(), 1);
        assert_eq!(api_req.messages[0].role, "user");
    }

    #[test]
    fn test_build_request_body_with_tools() {
        let request = CompletionRequest {
            tools: vec![ToolDefinition {
                name: "read_file".into(),
                description: "Read a file".into(),
                input_schema: serde_json::json!({"type": "object", "properties": {"path": {"type": "string"}}}),
            }],
            ..simple_completion_request()
        };

        let api_req = build_request_body(request);

        assert_eq!(api_req.tools.len(), 1);
        assert_eq!(api_req.tools[0].name, "read_file");
        assert_eq!(api_req.tools[0].description, "Read a file");
    }

    #[test]
    fn test_build_request_body_empty_optional_fields_omitted() {
        let request = CompletionRequest {
            system_prompt: None,
            tools: vec![],
            stop_sequences: vec![],
            temperature: None,
            ..simple_completion_request()
        };

        let api_req = build_request_body(request);
        let json = serde_json::to_value(&api_req).unwrap();
        let obj = json.as_object().unwrap();

        assert!(!obj.contains_key("system"));
        assert!(!obj.contains_key("tools"));
        assert!(!obj.contains_key("stop_sequences"));
        assert!(!obj.contains_key("temperature"));
        assert!(!obj.contains_key("stream"));
    }

    #[test]
    fn test_build_request_body_image_block() {
        let request = CompletionRequest {
            messages: vec![Message {
                role: Role::User,
                content: vec![ContentBlock::Image {
                    media_type: "image/png".into(),
                    data: "iVBORw0KGgo=".into(),
                }],
                timestamp: Utc::now(),
            }],
            ..simple_completion_request()
        };

        let api_req = build_request_body(request);
        let json = serde_json::to_value(&api_req).unwrap();

        let content = &json["messages"][0]["content"][0];
        assert_eq!(content["type"], "image");
        assert_eq!(content["source"]["type"], "base64");
        assert_eq!(content["source"]["media_type"], "image/png");
        assert_eq!(content["source"]["data"], "iVBORw0KGgo=");
    }

    // -----------------------------------------------------------------------
    // Unit tests: API response deserialization & conversion
    // -----------------------------------------------------------------------

    #[test]
    fn test_api_response_deserialization() {
        let json = r#"{
            "id": "msg_abc123",
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Hello!"},
                {"type": "tool_use", "id": "tu_1", "name": "read_file", "input": {"path": "test.txt"}}
            ],
            "model": "claude-opus-4-6-20250929",
            "stop_reason": "tool_use",
            "usage": {
                "input_tokens": 100,
                "output_tokens": 50,
                "cache_creation_input_tokens": 10,
                "cache_read_input_tokens": 5
            }
        }"#;

        let resp: ApiResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.id, "msg_abc123");
        assert_eq!(resp.content.len(), 2);
        assert_eq!(resp.model, "claude-opus-4-6-20250929");
        assert_eq!(resp.stop_reason, Some("tool_use".into()));
        assert_eq!(resp.usage.input_tokens, 100);
        assert_eq!(resp.usage.output_tokens, 50);
        assert_eq!(resp.usage.cache_creation_input_tokens, Some(10));
        assert_eq!(resp.usage.cache_read_input_tokens, Some(5));
    }

    #[test]
    fn test_api_response_into_completion_response() {
        let json = r#"{
            "id": "msg_abc123",
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Hello!"}
            ],
            "model": "claude-opus-4-6-20250929",
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": 25,
                "output_tokens": 10
            }
        }"#;

        let api_resp: ApiResponse = serde_json::from_str(json).unwrap();
        let resp = api_resp.into_completion_response();

        assert_eq!(resp.model, ModelId::from("claude-opus-4-6-20250929"));
        assert_eq!(resp.stop_reason, StopReason::EndTurn);
        assert_eq!(resp.usage.input_tokens, 25);
        assert_eq!(resp.usage.output_tokens, 10);
        assert_eq!(resp.message.role, Role::Assistant);
        assert_eq!(resp.message.content.len(), 1);
        match &resp.message.content[0] {
            ContentBlock::Text { text } => assert_eq!(text, "Hello!"),
            other => panic!("expected Text, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // Unit tests: parse_stop_reason
    // -----------------------------------------------------------------------

    #[test]
    fn test_parse_stop_reason_all_variants() {
        assert_eq!(parse_stop_reason(Some("end_turn")), StopReason::EndTurn);
        assert_eq!(parse_stop_reason(Some("tool_use")), StopReason::ToolUse);
        assert_eq!(parse_stop_reason(Some("max_tokens")), StopReason::MaxTokens);
        assert_eq!(
            parse_stop_reason(Some("stop_sequence")),
            StopReason::StopSequence
        );
    }

    #[test]
    fn test_parse_stop_reason_unknown_defaults_to_end_turn() {
        assert_eq!(
            parse_stop_reason(Some("some_future_reason")),
            StopReason::EndTurn
        );
    }

    #[test]
    fn test_parse_stop_reason_none_defaults_to_end_turn() {
        assert_eq!(parse_stop_reason(None), StopReason::EndTurn);
    }

    // -----------------------------------------------------------------------
    // Integration tests: wiremock HTTP
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_complete_success() {
        let mock_server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .respond_with(ResponseTemplate::new(200).set_body_json(success_response_json()))
            .mount(&mock_server)
            .await;

        let provider = make_provider(&mock_server.uri());
        let result = provider.complete(simple_completion_request()).await;

        let resp = result.unwrap();
        assert_eq!(resp.model, ModelId::from("claude-opus-4-6-20250929"));
        assert_eq!(resp.stop_reason, StopReason::EndTurn);
        assert_eq!(resp.usage.input_tokens, 25);
        assert_eq!(resp.usage.output_tokens, 10);
        match &resp.message.content[0] {
            ContentBlock::Text { text } => assert_eq!(text, "Hello, world!"),
            other => panic!("expected Text, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_complete_auth_error() {
        let mock_server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .respond_with(
                ResponseTemplate::new(401).set_body_json(error_response_json(
                    "authentication_error",
                    "invalid x-api-key",
                )),
            )
            .mount(&mock_server)
            .await;

        let provider = make_provider(&mock_server.uri());
        let result = provider.complete(simple_completion_request()).await;

        match result {
            Err(ProviderError::AuthenticationFailed { reason }) => {
                assert_eq!(reason, "invalid x-api-key");
            }
            other => panic!("expected AuthenticationFailed, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_complete_rate_limited_with_retry_after() {
        let mock_server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .respond_with(
                ResponseTemplate::new(429)
                    .append_header("retry-after", "30")
                    .set_body_json(error_response_json("rate_limit_error", "rate limited")),
            )
            .mount(&mock_server)
            .await;

        let provider = make_provider(&mock_server.uri());
        let result = provider.complete(simple_completion_request()).await;

        match result {
            Err(ProviderError::RateLimited { retry_after_ms }) => {
                assert_eq!(retry_after_ms, 30_000);
            }
            other => panic!("expected RateLimited, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_complete_rate_limited_no_retry_after() {
        let mock_server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .respond_with(
                ResponseTemplate::new(429)
                    .set_body_json(error_response_json("rate_limit_error", "rate limited")),
            )
            .mount(&mock_server)
            .await;

        let provider = make_provider(&mock_server.uri());
        let result = provider.complete(simple_completion_request()).await;

        match result {
            Err(ProviderError::RateLimited { retry_after_ms }) => {
                assert_eq!(retry_after_ms, DEFAULT_RETRY_AFTER_MS);
            }
            other => panic!("expected RateLimited, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_complete_server_error() {
        let mock_server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .respond_with(ResponseTemplate::new(500).set_body_string("internal server error"))
            .mount(&mock_server)
            .await;

        let provider = make_provider(&mock_server.uri());
        let result = provider.complete(simple_completion_request()).await;

        match result {
            Err(ProviderError::ApiError { status, body }) => {
                assert_eq!(status, 500);
                assert_eq!(body, "internal server error");
            }
            other => panic!("expected ApiError, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_complete_malformed_json() {
        let mock_server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .respond_with(ResponseTemplate::new(200).set_body_string("not json"))
            .mount(&mock_server)
            .await;

        let provider = make_provider(&mock_server.uri());
        let result = provider.complete(simple_completion_request()).await;

        match result {
            Err(ProviderError::Deserialization(msg)) => {
                assert!(!msg.is_empty());
            }
            other => panic!("expected Deserialization, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_validate_credentials_success() {
        let mock_server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .respond_with(ResponseTemplate::new(200).set_body_json(success_response_json()))
            .mount(&mock_server)
            .await;

        let provider = make_provider(&mock_server.uri());
        let result = provider.validate_credentials().await;

        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_validate_credentials_invalid_key() {
        let mock_server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .respond_with(
                ResponseTemplate::new(401).set_body_json(error_response_json(
                    "authentication_error",
                    "invalid x-api-key",
                )),
            )
            .mount(&mock_server)
            .await;

        let provider = make_provider(&mock_server.uri());
        let result = provider.validate_credentials().await;

        match result {
            Err(ProviderError::AuthenticationFailed { reason }) => {
                assert_eq!(reason, "invalid x-api-key");
            }
            other => panic!("expected AuthenticationFailed, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_validate_credentials_server_error() {
        let mock_server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .respond_with(ResponseTemplate::new(500).set_body_string("server error"))
            .mount(&mock_server)
            .await;

        let provider = make_provider(&mock_server.uri());
        let result = provider.validate_credentials().await;

        match result {
            Err(ProviderError::ApiError { status, body }) => {
                assert_eq!(status, 500);
                assert_eq!(body, "server error");
            }
            other => panic!("expected ApiError, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_validate_credentials_rate_limited() {
        let mock_server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .respond_with(
                ResponseTemplate::new(429)
                    .append_header("retry-after", "5")
                    .set_body_json(error_response_json("rate_limit_error", "rate limited")),
            )
            .mount(&mock_server)
            .await;

        let provider = make_provider(&mock_server.uri());
        let result = provider.validate_credentials().await;

        match result {
            Err(ProviderError::RateLimited { retry_after_ms }) => {
                assert_eq!(retry_after_ms, 5_000);
            }
            other => panic!("expected RateLimited, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_request_headers() {
        let mock_server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .respond_with(ResponseTemplate::new(200).set_body_json(success_response_json()))
            .expect(1)
            .mount(&mock_server)
            .await;

        let provider = make_provider(&mock_server.uri());
        provider
            .complete(simple_completion_request())
            .await
            .unwrap();

        let received = &mock_server.received_requests().await.unwrap()[0];
        assert_eq!(
            received.headers.get("x-api-key").unwrap().to_str().unwrap(),
            "test-api-key-12345"
        );
        assert_eq!(
            received
                .headers
                .get("anthropic-version")
                .unwrap()
                .to_str()
                .unwrap(),
            API_VERSION
        );
        assert_eq!(
            received
                .headers
                .get("content-type")
                .unwrap()
                .to_str()
                .unwrap(),
            "application/json"
        );
    }

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
        let data = r#"{"type":"message_start","message":{"id":"msg_1","model":"claude-opus-4-6-20250929","usage":{"input_tokens":150,"output_tokens":0}}}"#;
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

    // -----------------------------------------------------------------------
    // Integration tests (wiremock)
    // -----------------------------------------------------------------------

    /// Build an SSE response body from event strings.
    /// Each event should be a complete SSE event (with event: and data: lines).
    fn build_sse_body(events: &[&str]) -> String {
        use std::fmt::Write;
        events.iter().fold(String::new(), |mut acc, e| {
            let _ = write!(acc, "{e}\n\n");
            acc
        })
    }

    #[tokio::test]
    async fn test_stream_text_response() {
        use futures::StreamExt;

        let server = MockServer::start().await;
        let sse_body = build_sse_body(&[
            "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"model\":\"claude-opus-4-6-20250929\",\"usage\":{\"input_tokens\":25,\"output_tokens\":0}}}",
            "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}",
            "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Hello\"}}",
            "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\", world!\"}}",
            "event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}",
            "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":10}}",
            "event: message_stop\ndata: {\"type\":\"message_stop\"}",
        ]);

        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .respond_with(ResponseTemplate::new(200).set_body_raw(sse_body, "text/event-stream"))
            .mount(&server)
            .await;

        let provider = make_provider(&server.uri());
        let request = simple_completion_request();
        let mut stream = provider.stream(request).await.unwrap();

        let mut events = Vec::new();
        while let Some(item) = stream.next().await {
            events.push(item.unwrap());
        }

        assert_eq!(events.len(), 3);
        assert!(matches!(&events[0], StreamEvent::TextDelta(t) if t == "Hello"));
        assert!(matches!(&events[1], StreamEvent::TextDelta(t) if t == ", world!"));
        assert!(
            matches!(&events[2], StreamEvent::Done { stop_reason, usage } if *stop_reason == StopReason::EndTurn && usage.input_tokens == 25 && usage.output_tokens == 10)
        );
    }

    #[tokio::test]
    async fn test_stream_tool_use_response() {
        use futures::StreamExt;

        let server = MockServer::start().await;
        let sse_body = build_sse_body(&[
            "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"model\":\"claude-opus-4-6-20250929\",\"usage\":{\"input_tokens\":50,\"output_tokens\":0}}}",
            "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"tool_use\",\"id\":\"toolu_abc\",\"name\":\"read_file\"}}",
            "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"path\\\":\"}}",
            "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"\\\"test.txt\\\"}\"}}",
            "event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}",
            "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\"},\"usage\":{\"output_tokens\":20}}",
            "event: message_stop\ndata: {\"type\":\"message_stop\"}",
        ]);

        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .respond_with(ResponseTemplate::new(200).set_body_raw(sse_body, "text/event-stream"))
            .mount(&server)
            .await;

        let provider = make_provider(&server.uri());
        let request = simple_completion_request();
        let mut stream = provider.stream(request).await.unwrap();

        let mut events = Vec::new();
        while let Some(item) = stream.next().await {
            events.push(item.unwrap());
        }

        assert_eq!(events.len(), 2);
        match &events[0] {
            StreamEvent::ToolUse { id, name, input } => {
                assert_eq!(id, "toolu_abc");
                assert_eq!(name, "read_file");
                assert_eq!(input, &serde_json::json!({"path": "test.txt"}));
            }
            other => panic!("expected ToolUse, got {other:?}"),
        }
        assert!(
            matches!(&events[1], StreamEvent::Done { stop_reason, .. } if *stop_reason == StopReason::ToolUse)
        );
    }

    #[tokio::test]
    async fn test_stream_mixed_text_and_tool() {
        use futures::StreamExt;

        let server = MockServer::start().await;
        let sse_body = build_sse_body(&[
            "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"model\":\"claude-opus-4-6-20250929\",\"usage\":{\"input_tokens\":30,\"output_tokens\":0}}}",
            "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}",
            "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Let me read that.\"}}",
            "event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}",
            "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":1,\"content_block\":{\"type\":\"tool_use\",\"id\":\"toolu_xyz\",\"name\":\"read_file\"}}",
            "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"path\\\":\\\"foo.rs\\\"}\"}}",
            "event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":1}",
            "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\"},\"usage\":{\"output_tokens\":25}}",
            "event: message_stop\ndata: {\"type\":\"message_stop\"}",
        ]);

        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .respond_with(ResponseTemplate::new(200).set_body_raw(sse_body, "text/event-stream"))
            .mount(&server)
            .await;

        let provider = make_provider(&server.uri());
        let request = simple_completion_request();
        let mut stream = provider.stream(request).await.unwrap();

        let mut events = Vec::new();
        while let Some(item) = stream.next().await {
            events.push(item.unwrap());
        }

        assert_eq!(events.len(), 3);
        assert!(matches!(&events[0], StreamEvent::TextDelta(t) if t == "Let me read that."));
        assert!(matches!(&events[1], StreamEvent::ToolUse { name, .. } if name == "read_file"));
        assert!(
            matches!(&events[2], StreamEvent::Done { stop_reason, .. } if *stop_reason == StopReason::ToolUse)
        );
    }

    #[tokio::test]
    async fn test_stream_error_event() {
        use futures::StreamExt;

        let server = MockServer::start().await;
        let sse_body = build_sse_body(&[
            "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"model\":\"claude-opus-4-6-20250929\",\"usage\":{\"input_tokens\":10,\"output_tokens\":0}}}",
            "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}",
            "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"partial\"}}",
            "event: error\ndata: {\"type\":\"error\",\"error\":{\"type\":\"overloaded_error\",\"message\":\"Overloaded\"}}",
        ]);

        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .respond_with(ResponseTemplate::new(200).set_body_raw(sse_body, "text/event-stream"))
            .mount(&server)
            .await;

        let provider = make_provider(&server.uri());
        let request = simple_completion_request();
        let mut stream = provider.stream(request).await.unwrap();

        let mut events = Vec::new();
        while let Some(item) = stream.next().await {
            events.push(item.unwrap());
        }

        assert_eq!(events.len(), 2);
        assert!(matches!(&events[0], StreamEvent::TextDelta(t) if t == "partial"));
        assert!(matches!(&events[1], StreamEvent::Error(msg) if msg.contains("Overloaded")));
    }

    #[tokio::test]
    async fn test_stream_http_401() {
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
        let request = simple_completion_request();
        let result = provider.stream(request).await;

        match result {
            Err(ProviderError::AuthenticationFailed { reason }) => {
                assert!(reason.contains("invalid"), "reason: {reason}");
            }
            Err(e) => panic!("expected AuthenticationFailed, got error: {e}"),
            Ok(_) => panic!("expected AuthenticationFailed, got Ok"),
        }
    }

    #[tokio::test]
    async fn test_stream_http_429() {
        let server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .respond_with(
                ResponseTemplate::new(429)
                    .insert_header("retry-after", "2")
                    .set_body_json(error_response_json("rate_limit_error", "rate limited")),
            )
            .mount(&server)
            .await;

        let provider = make_provider(&server.uri());
        let request = simple_completion_request();
        let result = provider.stream(request).await;

        match result {
            Err(ProviderError::RateLimited { retry_after_ms }) => {
                assert_eq!(retry_after_ms, 2000);
            }
            Err(e) => panic!("expected RateLimited, got error: {e}"),
            Ok(_) => panic!("expected RateLimited, got Ok"),
        }
    }

    #[tokio::test]
    async fn test_stream_request_has_stream_true() {
        let server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .respond_with(
                ResponseTemplate::new(200).set_body_raw(
                    build_sse_body(&[
                        "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"model\":\"claude-opus-4-6-20250929\",\"usage\":{\"input_tokens\":5,\"output_tokens\":0}}}",
                        "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":1}}",
                        "event: message_stop\ndata: {\"type\":\"message_stop\"}",
                    ]),
                    "text/event-stream",
                ),
            )
            .mount(&server)
            .await;

        let provider = make_provider(&server.uri());
        let request = simple_completion_request();
        let _stream = provider.stream(request).await.unwrap();

        // Verify the request body had stream: true
        let received = &server.received_requests().await.unwrap()[0];
        let body: serde_json::Value = serde_json::from_slice(&received.body).unwrap();
        assert_eq!(body.get("stream"), Some(&serde_json::Value::Bool(true)));
    }

    #[tokio::test]
    async fn test_stream_partial_byte_chunks() {
        use futures::StreamExt;

        let server = MockServer::start().await;

        // This tests that partial chunks are reassembled correctly.
        // The SSE event is split across the response but should still parse.
        let sse_body = build_sse_body(&[
            "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"model\":\"claude-opus-4-6-20250929\",\"usage\":{\"input_tokens\":5,\"output_tokens\":0}}}",
            "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}",
            "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"chunk\"}}",
            "event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}",
            "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":1}}",
            "event: message_stop\ndata: {\"type\":\"message_stop\"}",
        ]);

        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .respond_with(ResponseTemplate::new(200).set_body_raw(sse_body, "text/event-stream"))
            .mount(&server)
            .await;

        let provider = make_provider(&server.uri());
        let request = simple_completion_request();
        let mut stream = provider.stream(request).await.unwrap();

        let mut events = Vec::new();
        while let Some(item) = stream.next().await {
            events.push(item.unwrap());
        }

        assert_eq!(events.len(), 2);
        assert!(matches!(&events[0], StreamEvent::TextDelta(t) if t == "chunk"));
        assert!(matches!(&events[1], StreamEvent::Done { .. }));
    }

    #[tokio::test]
    async fn test_stream_connection_drop() {
        use futures::StreamExt;

        let server = MockServer::start().await;

        // Response with incomplete SSE (no message_stop, abrupt end)
        let incomplete_body = "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"model\":\"claude-opus-4-6-20250929\",\"usage\":{\"input_tokens\":5,\"output_tokens\":0}}}\n\nevent: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\nevent: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"partial\"}}\n\n";

        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .respond_with(
                ResponseTemplate::new(200).set_body_raw(incomplete_body, "text/event-stream"),
            )
            .mount(&server)
            .await;

        let provider = make_provider(&server.uri());
        let request = simple_completion_request();
        let mut stream = provider.stream(request).await.unwrap();

        let mut events = Vec::new();
        while let Some(item) = stream.next().await {
            events.push(item);
        }

        // Should get at least the text delta; stream ends without Done
        assert!(!events.is_empty());
        let ok_events: Vec<_> = events.iter().filter_map(|e| e.as_ref().ok()).collect();
        assert!(
            ok_events
                .iter()
                .any(|e| matches!(e, StreamEvent::TextDelta(t) if t == "partial"))
        );
    }

    // -----------------------------------------------------------------------
    // Security test: streaming
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_stream_api_key_not_in_errors() {
        let server = MockServer::start().await;
        let key_value = "super-secret-streaming-key";

        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .respond_with(
                ResponseTemplate::new(500)
                    .set_body_json(error_response_json("api_error", "internal server error")),
            )
            .mount(&server)
            .await;

        let provider = AnthropicProvider::new(
            SecretString::from(key_value),
            AnthropicConfig {
                base_url: Some(server.uri()),
                default_model: None,
            },
        )
        .unwrap();

        let result = provider.stream(simple_completion_request()).await;
        let Err(err) = result else {
            panic!("expected error, got Ok");
        };
        let display = format!("{err}");
        let debug = format!("{err:?}");
        assert!(
            !display.contains(key_value),
            "stream error Display leaked the key: {display}"
        );
        assert!(
            !debug.contains(key_value),
            "stream error Debug leaked the key: {debug}"
        );
    }

    // -----------------------------------------------------------------------
    // Security test: non-streaming (existing)
    // -----------------------------------------------------------------------

    #[test]
    fn test_api_key_not_in_error_display() {
        let key_value = "super-secret-key-99999";
        let key = SecretString::from(key_value);

        // The SecretString itself should not leak via Debug
        let debug_output = format!("{key:?}");
        assert!(
            !debug_output.contains(key_value),
            "SecretString Debug leaked the key"
        );

        // ProviderError variants should not contain the key
        let errors: Vec<ProviderError> = vec![
            ProviderError::AuthenticationFailed {
                reason: "invalid API key".into(),
            },
            ProviderError::ApiError {
                status: 500,
                body: "internal error".into(),
            },
            ProviderError::Network {
                reason: "connection failed".into(),
                kind: NetworkErrorKind::Other,
                status_code: None,
            },
        ];

        for err in &errors {
            let display = format!("{err}");
            let debug = format!("{err:?}");
            assert!(
                !display.contains(key_value),
                "error Display leaked the key: {display}"
            );
            assert!(
                !debug.contains(key_value),
                "error Debug leaked the key: {debug}"
            );
        }
    }
}
