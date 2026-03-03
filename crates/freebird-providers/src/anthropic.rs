//! Anthropic (Claude) provider implementation.

use std::pin::Pin;

use async_trait::async_trait;
use chrono::Utc;
use freebird_traits::provider::{
    CompletionRequest, CompletionResponse, ContentBlock, Message, ModelInfo, NetworkErrorKind,
    Provider, ProviderError, ProviderInfo, Role, StopReason, StreamEvent, TokenUsage,
};
use futures::Stream;
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
    #[allow(dead_code)]
    id: String,
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
    #[allow(dead_code)]
    #[serde(rename = "type")]
    error_type: String,
    error: ApiErrorDetail,
}

#[derive(Debug, Deserialize)]
struct ApiErrorDetail {
    #[allow(dead_code)]
    #[serde(rename = "type")]
    error_type: String,
    message: String,
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
            id: "anthropic".into(),
            display_name: "Anthropic Claude".into(),
            supported_models: vec![
                ModelInfo {
                    id: "claude-opus-4-6-20250929".into(),
                    display_name: "Claude Opus 4.6".into(),
                    max_context_tokens: 200_000,
                    max_output_tokens: 32_768,
                },
                ModelInfo {
                    id: "claude-sonnet-4-5-20250929".into(),
                    display_name: "Claude Sonnet 4.5".into(),
                    max_context_tokens: 200_000,
                    max_output_tokens: 16_384,
                },
            ],
            supports_streaming: true,
            supports_tool_use: true,
            supports_vision: true,
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
fn build_request_body(request: &CompletionRequest) -> ApiRequest {
    let messages: Vec<ApiMessage> = request
        .messages
        .iter()
        .filter(|m| m.role != Role::System)
        .map(|m| ApiMessage {
            role: match m.role {
                Role::User | Role::Tool | Role::System => "user".into(),
                Role::Assistant => "assistant".into(),
            },
            content: m.content.iter().map(convert_content_block).collect(),
        })
        .collect();

    let tools: Vec<ApiToolDefinition> = request
        .tools
        .iter()
        .map(|t| ApiToolDefinition {
            name: t.name.clone(),
            description: t.description.clone(),
            input_schema: t.input_schema.clone(),
        })
        .collect();

    ApiRequest {
        model: request.model.clone(),
        max_tokens: request.max_tokens,
        messages,
        system: request.system_prompt.clone(),
        temperature: request.temperature,
        stop_sequences: request.stop_sequences.clone(),
        tools,
        stream: None,
    }
}

/// Convert an internal `ContentBlock` to the Anthropic wire format.
fn convert_content_block(block: &ContentBlock) -> ApiContentBlock {
    match block {
        ContentBlock::Text { text } => ApiContentBlock::Text { text: text.clone() },
        ContentBlock::ToolUse { id, name, input } => ApiContentBlock::ToolUse {
            id: id.clone(),
            name: name.clone(),
            input: input.clone(),
        },
        ContentBlock::ToolResult {
            tool_use_id,
            content,
            is_error,
        } => ApiContentBlock::ToolResult {
            tool_use_id: tool_use_id.clone(),
            content: content.clone(),
            is_error: *is_error,
        },
        ContentBlock::Image { media_type, data } => ApiContentBlock::Image {
            source: ApiImageSource {
                source_type: "base64".into(),
                media_type: media_type.clone(),
                data: data.clone(),
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
            model: self.model,
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

        let status = resp.status().as_u16();

        if status == 401 {
            let reason = extract_error_message(resp, "invalid API key").await;
            return Err(ProviderError::AuthenticationFailed { reason });
        }

        if !resp.status().is_success() {
            let body_text = resp.text().await.unwrap_or_default();
            return Err(ProviderError::ApiError {
                status,
                body: body_text,
            });
        }

        Ok(())
    }

    async fn complete(
        &self,
        request: CompletionRequest,
    ) -> Result<CompletionResponse, ProviderError> {
        let api_request = build_request_body(&request);

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

        let status = resp.status().as_u16();

        if status == 401 {
            let reason = extract_error_message(resp, "invalid API key").await;
            return Err(ProviderError::AuthenticationFailed { reason });
        }

        if status == 429 {
            let retry_after_ms = resp
                .headers()
                .get("retry-after")
                .and_then(|v| v.to_str().ok())
                .and_then(|v| v.parse::<u64>().ok())
                .map_or(DEFAULT_RETRY_AFTER_MS, |secs| secs * 1000);
            return Err(ProviderError::RateLimited { retry_after_ms });
        }

        if !resp.status().is_success() {
            let body_text = resp.text().await.unwrap_or_default();
            return Err(ProviderError::ApiError {
                status,
                body: body_text,
            });
        }

        let api_response: ApiResponse = resp
            .json()
            .await
            .map_err(|e| ProviderError::Deserialization(e.to_string()))?;

        Ok(api_response.into_completion_response())
    }

    async fn stream(
        &self,
        _request: CompletionRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamEvent, ProviderError>> + Send>>, ProviderError>
    {
        Err(ProviderError::NotConfigured)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic, clippy::indexing_slicing)]
mod tests {
    use super::*;
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
            model: "claude-opus-4-6-20250929".into(),
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

    // -----------------------------------------------------------------------
    // Unit tests: build_request_body
    // -----------------------------------------------------------------------

    #[test]
    fn test_build_request_body_basic() {
        let request = simple_completion_request();

        let api_req = build_request_body(&request);

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

        let api_req = build_request_body(&request);

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

        let api_req = build_request_body(&request);

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

        let api_req = build_request_body(&request);

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

        let api_req = build_request_body(&request);

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

        let api_req = build_request_body(&request);
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

        let api_req = build_request_body(&request);
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

        assert_eq!(resp.model, "claude-opus-4-6-20250929");
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
        assert_eq!(resp.model, "claude-opus-4-6-20250929");
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
            .respond_with(ResponseTemplate::new(401).set_body_json(serde_json::json!({
                "type": "error",
                "error": {
                    "type": "authentication_error",
                    "message": "invalid x-api-key"
                }
            })))
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
                    .set_body_json(serde_json::json!({
                        "type": "error",
                        "error": {"type": "rate_limit_error", "message": "rate limited"}
                    })),
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
            .respond_with(ResponseTemplate::new(429).set_body_json(serde_json::json!({
                "type": "error",
                "error": {"type": "rate_limit_error", "message": "rate limited"}
            })))
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
            .respond_with(ResponseTemplate::new(401).set_body_json(serde_json::json!({
                "type": "error",
                "error": {
                    "type": "authentication_error",
                    "message": "invalid x-api-key"
                }
            })))
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
    async fn test_stream_returns_not_configured() {
        let provider = make_provider("http://localhost:0");
        let result = provider.stream(simple_completion_request()).await;

        assert!(
            matches!(result, Err(ProviderError::NotConfigured)),
            "expected NotConfigured"
        );
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
    // Security test
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
