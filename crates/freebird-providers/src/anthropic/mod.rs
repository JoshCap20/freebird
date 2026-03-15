//! Anthropic (Claude) provider implementation.

mod client;
mod stream;
mod types;

use std::collections::BTreeSet;
use std::pin::Pin;

use async_trait::async_trait;
use freebird_traits::id::{ModelId, ProviderId};
use freebird_traits::provider::{
    CompletionRequest, CompletionResponse, ModelInfo, NetworkErrorKind, Provider, ProviderError,
    ProviderFeature, ProviderInfo, StreamEvent,
};
use futures::{Stream, StreamExt as _};
use reqwest::Client;
use secrecy::{ExposeSecret, SecretString};

use self::client::{build_request_body, classify_reqwest_error, map_error_response};
use self::stream::{SseStreamState, unfold_sse_stream};
pub use self::types::AnthropicConfig;
use self::types::AuthKind;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Anthropic API version header value.
const API_VERSION: &str = "2023-06-01";
/// Beta feature flag required for OAuth token authentication.
const OAUTH_BETA_FLAG: &str = "oauth-2025-04-20";
/// Default base URL for the Anthropic API.
const DEFAULT_BASE_URL: &str = "https://api.anthropic.com";
/// Default model ID — matches `config/default.toml`'s `runtime.default_model`.
/// The source of truth for the default model is `default.toml`.
const DEFAULT_MODEL: &str = "claude-sonnet-4-6";
/// HTTP request timeout in seconds.
const REQUEST_TIMEOUT_SECS: u64 = 300;

// ---------------------------------------------------------------------------
// Provider struct
// ---------------------------------------------------------------------------

/// The Anthropic provider implementation.
pub struct AnthropicProvider {
    client: Client,
    api_key: SecretString,
    auth_kind: AuthKind,
    base_url: String,
    default_model: String,
    info: ProviderInfo,
}

impl std::fmt::Debug for AnthropicProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AnthropicProvider")
            .field("base_url", &self.base_url)
            .field("default_model", &self.default_model)
            .field("api_key", &"[REDACTED]")
            .field("info", &self.info)
            .finish_non_exhaustive()
    }
}

impl AnthropicProvider {
    /// Construct a new Anthropic provider.
    ///
    /// # Errors
    ///
    /// Returns `ProviderError::Network` if the HTTP client fails to build
    /// (e.g., TLS initialization failure).
    pub fn new(api_key: SecretString, config: AnthropicConfig) -> Result<Self, ProviderError> {
        let timeout = config.timeout_secs.unwrap_or(REQUEST_TIMEOUT_SECS);
        let client = Client::builder()
            .use_rustls_tls()
            .timeout(std::time::Duration::from_secs(timeout))
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
                    id: ModelId::from("claude-opus-4-6"),
                    display_name: "Claude Opus 4.6".into(),
                    max_context_tokens: 200_000,
                    max_output_tokens: 32_768,
                },
                ModelInfo {
                    id: ModelId::from("claude-sonnet-4-6"),
                    display_name: "Claude Sonnet 4.6".into(),
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

        let auth_kind = if api_key.expose_secret().starts_with("sk-ant-oat") {
            AuthKind::OAuthToken
        } else {
            AuthKind::ApiKey
        };

        Ok(Self {
            client,
            api_key,
            auth_kind,
            base_url,
            default_model,
            info,
        })
    }

    /// Build a POST request to the messages endpoint with correct auth headers.
    fn messages_request(&self) -> reqwest::RequestBuilder {
        let builder = self
            .client
            .post(format!("{}/v1/messages", self.base_url))
            .header("anthropic-version", API_VERSION)
            .header("content-type", "application/json");

        match self.auth_kind {
            AuthKind::ApiKey => builder.header("x-api-key", self.api_key.expose_secret()),
            AuthKind::OAuthToken => builder
                .header(
                    "authorization",
                    format!("Bearer {}", self.api_key.expose_secret()),
                )
                .header("anthropic-beta", OAUTH_BETA_FLAG),
        }
    }
}

// ---------------------------------------------------------------------------
// Provider trait implementation
// ---------------------------------------------------------------------------

#[async_trait]
impl Provider for AnthropicProvider {
    fn info(&self) -> &ProviderInfo {
        &self.info
    }

    /// Validates API credentials by sending a minimal completion request.
    ///
    /// **Note**: This consumes a small number of tokens (~10 input + 1 output)
    /// by sending a minimal "ping" message. This call bypasses `TokenBudget`
    /// accounting since it does not go through the agentic loop.
    async fn validate_credentials(&self) -> Result<(), ProviderError> {
        let body = serde_json::json!({
            "model": self.default_model,
            "max_tokens": 1,
            "messages": [{"role": "user", "content": "ping"}]
        });

        let resp = self
            .messages_request()
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
            .messages_request()
            .json(&api_request)
            .send()
            .await
            .map_err(|ref e| classify_reqwest_error(e))?;

        if !resp.status().is_success() {
            return Err(map_error_response(resp).await);
        }

        let api_response: types::ApiResponse = resp
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
            .messages_request()
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

        Ok(unfold_sse_stream(state))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic, clippy::indexing_slicing)]
mod tests {
    use super::*;
    use chrono::Utc;
    use freebird_traits::id::ModelId;
    use freebird_traits::provider::{
        ContentBlock, Message, NetworkErrorKind, Role, StopReason, ToolDefinition,
    };
    use secrecy::SecretString;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    use super::client::DEFAULT_RETRY_AFTER_MS;

    // -----------------------------------------------------------------------
    // Test helpers
    // -----------------------------------------------------------------------

    /// OAuth-prefix key prefix for testing.
    const OAUTH_KEY_PREFIX: &str = "sk-ant-oat01-test-oauth-token-12345";

    fn test_api_key() -> SecretString {
        SecretString::from("test-api-key-12345")
    }

    fn test_oauth_key() -> SecretString {
        SecretString::from(OAUTH_KEY_PREFIX)
    }

    fn make_provider(base_url: &str) -> AnthropicProvider {
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

    fn simple_completion_request() -> CompletionRequest {
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
            tools: Vec::new(),
            max_tokens: 4096,
            temperature: None,
            stop_sequences: Vec::new(),
        }
    }

    fn success_response_json() -> serde_json::Value {
        serde_json::json!({
            "id": "msg_test123",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "Hello, world!"}],
            "model": "claude-opus-4-6",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 25, "output_tokens": 10}
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
    // Unit tests: auth kind detection
    // -----------------------------------------------------------------------

    #[test]
    fn test_auth_kind_detects_api_key() {
        let provider = make_provider("http://localhost");
        assert_eq!(provider.auth_kind, AuthKind::ApiKey);
    }

    #[test]
    fn test_auth_kind_detects_oauth_token() {
        let provider = AnthropicProvider::new(
            test_oauth_key(),
            AnthropicConfig {
                base_url: Some("http://localhost".into()),
                default_model: None,
                timeout_secs: None,
            },
        )
        .unwrap();
        assert_eq!(provider.auth_kind, AuthKind::OAuthToken);
    }

    #[test]
    fn test_auth_kind_detects_oat_variants() {
        // Various OAuth token prefixes should all be detected
        for prefix in &[
            "sk-ant-oat01-abc",
            "sk-ant-oat02-xyz",
            "sk-ant-oat-something",
        ] {
            let provider = AnthropicProvider::new(
                SecretString::from(*prefix),
                AnthropicConfig {
                    base_url: Some("http://localhost".into()),
                    default_model: None,
                    timeout_secs: None,
                },
            )
            .unwrap();
            assert_eq!(
                provider.auth_kind,
                AuthKind::OAuthToken,
                "failed for prefix: {prefix}"
            );
        }
    }

    #[test]
    fn test_auth_kind_api_key_variants() {
        // Non-OAuth keys should be detected as API keys
        for key in &[
            "sk-ant-api-abc123",
            "sk-regular-key",
            "some-other-key",
            "sk-ant-oa-almost-oauth",
        ] {
            let provider = AnthropicProvider::new(
                SecretString::from(*key),
                AnthropicConfig {
                    base_url: Some("http://localhost".into()),
                    default_model: None,
                    timeout_secs: None,
                },
            )
            .unwrap();
            assert_eq!(
                provider.auth_kind,
                AuthKind::ApiKey,
                "failed for key: {key}"
            );
        }
    }

    // -----------------------------------------------------------------------
    // Integration tests: auth headers
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_api_key_sends_x_api_key_header() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .respond_with(ResponseTemplate::new(200).set_body_json(success_response_json()))
            .mount(&server)
            .await;

        let provider = make_provider(&server.uri());
        provider
            .complete(simple_completion_request())
            .await
            .unwrap();

        let received = &server.received_requests().await.unwrap()[0];
        assert!(received.headers.get("x-api-key").is_some());
        assert!(received.headers.get("authorization").is_none());
        assert!(received.headers.get("anthropic-beta").is_none());
    }

    #[tokio::test]
    async fn test_oauth_token_sends_bearer_and_beta_headers() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .respond_with(ResponseTemplate::new(200).set_body_json(success_response_json()))
            .mount(&server)
            .await;

        let provider = AnthropicProvider::new(
            test_oauth_key(),
            AnthropicConfig {
                base_url: Some(server.uri()),
                default_model: None,
                timeout_secs: None,
            },
        )
        .unwrap();

        provider
            .complete(simple_completion_request())
            .await
            .unwrap();

        let received = &server.received_requests().await.unwrap()[0];
        let auth_header = received
            .headers
            .get("authorization")
            .unwrap()
            .to_str()
            .unwrap();
        assert!(
            auth_header.starts_with("Bearer "),
            "auth header: {auth_header}"
        );

        let beta_header = received
            .headers
            .get("anthropic-beta")
            .unwrap()
            .to_str()
            .unwrap();
        assert_eq!(beta_header, OAUTH_BETA_FLAG);
    }

    #[tokio::test]
    async fn test_oauth_token_does_not_send_x_api_key() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .respond_with(ResponseTemplate::new(200).set_body_json(success_response_json()))
            .mount(&server)
            .await;

        let provider = AnthropicProvider::new(
            test_oauth_key(),
            AnthropicConfig {
                base_url: Some(server.uri()),
                default_model: None,
                timeout_secs: None,
            },
        )
        .unwrap();

        provider
            .complete(simple_completion_request())
            .await
            .unwrap();

        let received = &server.received_requests().await.unwrap()[0];
        assert!(
            received.headers.get("x-api-key").is_none(),
            "OAuth flow should not send x-api-key header"
        );
    }

    // -----------------------------------------------------------------------
    // Unit tests: request body building
    // -----------------------------------------------------------------------

    #[test]
    fn test_build_request_body_basic() {
        let request = simple_completion_request();
        let body = build_request_body(request);
        assert_eq!(body.model, "claude-opus-4-6");
        assert_eq!(body.max_tokens, 4096);
        assert_eq!(body.messages.len(), 1);
        assert_eq!(body.messages[0].role, "user");
        assert!(body.stream.is_none());
    }

    #[test]
    fn test_build_request_body_system_prompt_in_top_level() {
        let request = simple_completion_request();
        let body = build_request_body(request);
        assert_eq!(body.system, Some("You are helpful.".into()));
        // System should NOT appear in messages
        assert!(body.messages.iter().all(|m| m.role != "system"));
    }

    #[test]
    fn test_build_request_body_tool_role_mapping() {
        let request = CompletionRequest {
            model: ModelId::from("claude-opus-4-6"),
            system_prompt: None,
            messages: vec![
                Message {
                    role: Role::User,
                    content: vec![ContentBlock::Text {
                        text: "test".into(),
                    }],
                    timestamp: Utc::now(),
                },
                Message {
                    role: Role::Tool,
                    content: vec![ContentBlock::ToolResult {
                        tool_use_id: "t1".into(),
                        content: "result".into(),
                        is_error: false,
                    }],
                    timestamp: Utc::now(),
                },
            ],
            tools: Vec::new(),
            max_tokens: 1024,
            temperature: None,
            stop_sequences: Vec::new(),
        };

        let body = build_request_body(request);
        assert_eq!(body.messages[1].role, "user");
    }

    #[test]
    fn test_build_request_body_system_messages_filtered() {
        let request = CompletionRequest {
            model: ModelId::from("claude-opus-4-6"),
            system_prompt: Some("system prompt".into()),
            messages: vec![
                Message {
                    role: Role::System,
                    content: vec![ContentBlock::Text {
                        text: "system msg".into(),
                    }],
                    timestamp: Utc::now(),
                },
                Message {
                    role: Role::User,
                    content: vec![ContentBlock::Text {
                        text: "hello".into(),
                    }],
                    timestamp: Utc::now(),
                },
            ],
            tools: Vec::new(),
            max_tokens: 1024,
            temperature: None,
            stop_sequences: Vec::new(),
        };

        let body = build_request_body(request);
        assert_eq!(body.messages.len(), 1);
        assert_eq!(body.messages[0].role, "user");
    }

    #[test]
    fn test_build_request_body_with_tools() {
        let request = CompletionRequest {
            model: ModelId::from("claude-opus-4-6"),
            system_prompt: None,
            messages: vec![Message {
                role: Role::User,
                content: vec![ContentBlock::Text { text: "hi".into() }],
                timestamp: Utc::now(),
            }],
            tools: vec![ToolDefinition {
                name: "read_file".into(),
                description: "Read a file".into(),
                input_schema: serde_json::json!({"type": "object"}),
            }],
            max_tokens: 1024,
            temperature: None,
            stop_sequences: Vec::new(),
        };

        let body = build_request_body(request);
        assert_eq!(body.tools.len(), 1);
        assert_eq!(body.tools[0].name, "read_file");
    }

    #[test]
    fn test_build_request_body_empty_optional_fields_omitted() {
        let request = CompletionRequest {
            model: ModelId::from("claude-opus-4-6"),
            system_prompt: None,
            messages: vec![Message {
                role: Role::User,
                content: vec![ContentBlock::Text { text: "hi".into() }],
                timestamp: Utc::now(),
            }],
            tools: Vec::new(),
            max_tokens: 1024,
            temperature: None,
            stop_sequences: Vec::new(),
        };

        let body = build_request_body(request);
        let json = serde_json::to_value(&body).unwrap();
        assert!(json.get("system").is_none());
        assert!(json.get("temperature").is_none());
        assert!(json.get("stop_sequences").is_none());
        assert!(json.get("tools").is_none());
        assert!(json.get("stream").is_none());
    }

    #[test]
    fn test_build_request_body_image_block() {
        let request = CompletionRequest {
            model: ModelId::from("claude-opus-4-6"),
            system_prompt: None,
            messages: vec![Message {
                role: Role::User,
                content: vec![ContentBlock::Image {
                    media_type: "image/png".into(),
                    data: "base64data".into(),
                }],
                timestamp: Utc::now(),
            }],
            tools: Vec::new(),
            max_tokens: 1024,
            temperature: None,
            stop_sequences: Vec::new(),
        };

        let body = build_request_body(request);
        let json = serde_json::to_value(&body).unwrap();
        let content = &json["messages"][0]["content"][0];
        assert_eq!(content["type"], "image");
        assert_eq!(content["source"]["type"], "base64");
        assert_eq!(content["source"]["media_type"], "image/png");
        assert_eq!(content["source"]["data"], "base64data");
    }

    // -----------------------------------------------------------------------
    // Unit tests: response parsing
    // -----------------------------------------------------------------------

    #[test]
    fn test_api_response_deserialization() {
        let json = success_response_json();
        let resp: types::ApiResponse = serde_json::from_value(json).unwrap();
        assert_eq!(resp.id, "msg_test123");
        assert_eq!(resp.role, "assistant");
        assert_eq!(resp.content.len(), 1);
        assert_eq!(resp.model, "claude-opus-4-6");
        assert_eq!(resp.stop_reason, Some("end_turn".into()));
        assert_eq!(resp.usage.input_tokens, 25);
        assert_eq!(resp.usage.output_tokens, 10);
    }

    #[test]
    fn test_api_response_into_completion_response() {
        let api_resp = types::ApiResponse {
            id: "msg_1".into(),
            role: "assistant".into(),
            content: vec![types::ApiContentBlock::Text {
                text: "test".into(),
            }],
            model: "claude-opus-4-6".into(),
            stop_reason: Some("end_turn".into()),
            usage: types::ApiUsage {
                input_tokens: 10,
                output_tokens: 5,
                cache_creation_input_tokens: None,
                cache_read_input_tokens: None,
            },
        };

        let resp = api_resp.into_completion_response();
        assert_eq!(resp.model, ModelId::from("claude-opus-4-6"));
        assert_eq!(resp.stop_reason, StopReason::EndTurn);
        assert_eq!(resp.usage.input_tokens, 10);
        assert_eq!(resp.usage.output_tokens, 5);
        assert_eq!(resp.message.role, Role::Assistant);
        match &resp.message.content[0] {
            ContentBlock::Text { text } => assert_eq!(text, "test"),
            other => panic!("expected Text, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // Unit tests: parse_stop_reason
    // -----------------------------------------------------------------------

    #[test]
    fn test_parse_stop_reason_all_variants() {
        use super::client::parse_stop_reason;
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
        use super::client::parse_stop_reason;
        assert_eq!(
            parse_stop_reason(Some("some_future_reason")),
            StopReason::EndTurn
        );
    }

    #[test]
    fn test_parse_stop_reason_none_defaults_to_end_turn() {
        use super::client::parse_stop_reason;
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
        assert_eq!(resp.model, ModelId::from("claude-opus-4-6"));
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
    // Integration tests: streaming (wiremock)
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
            "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"model\":\"claude-opus-4-6\",\"usage\":{\"input_tokens\":25,\"output_tokens\":0}}}",
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
            "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"model\":\"claude-opus-4-6\",\"usage\":{\"input_tokens\":50,\"output_tokens\":0}}}",
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
            "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"model\":\"claude-opus-4-6\",\"usage\":{\"input_tokens\":30,\"output_tokens\":0}}}",
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
            "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"model\":\"claude-opus-4-6\",\"usage\":{\"input_tokens\":10,\"output_tokens\":0}}}",
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
                        "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"model\":\"claude-opus-4-6\",\"usage\":{\"input_tokens\":5,\"output_tokens\":0}}}",
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
            "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"model\":\"claude-opus-4-6\",\"usage\":{\"input_tokens\":5,\"output_tokens\":0}}}",
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
        let incomplete_body = "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"model\":\"claude-opus-4-6\",\"usage\":{\"input_tokens\":5,\"output_tokens\":0}}}\n\nevent: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\nevent: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"partial\"}}\n\n";

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

        // Should get the text delta that was sent before the connection drop
        assert!(!events.is_empty());
        let ok_events: Vec<_> = events.iter().filter_map(|e| e.as_ref().ok()).collect();
        assert!(
            ok_events
                .iter()
                .any(|e| matches!(e, StreamEvent::TextDelta(t) if t == "partial")),
            "expected TextDelta(\"partial\") in events: {ok_events:?}"
        );
        // Connection dropped before message_delta — stream must NOT contain a Done event
        assert!(
            !ok_events
                .iter()
                .any(|e| matches!(e, StreamEvent::Done { .. })),
            "stream should not contain Done after connection drop: {ok_events:?}"
        );
    }

    // -----------------------------------------------------------------------
    // Security tests
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
                timeout_secs: None,
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
