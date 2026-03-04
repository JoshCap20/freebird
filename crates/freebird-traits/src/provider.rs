//! Provider trait — abstracts over LLM backends (Anthropic, `OpenAI`, Ollama, etc.).

use std::collections::BTreeSet;
use std::pin::Pin;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use futures::Stream;
use serde::{Deserialize, Serialize};

use crate::id::{ModelId, ProviderId};

/// The role of a participant in a conversation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    User,
    Assistant,
    System,
    Tool,
}

/// Optional features a provider may support.
///
/// Using an enum set instead of boolean flags (CLAUDE.md §30) makes the
/// feature surface extensible: adding a new feature is an enum variant,
/// not a struct field change across every constructor.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProviderFeature {
    Streaming,
    ToolUse,
    Vision,
}

/// Metadata about a provider implementation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderInfo {
    pub id: ProviderId,
    pub display_name: String,
    pub supported_models: Vec<ModelInfo>,
    pub features: BTreeSet<ProviderFeature>,
}

impl ProviderInfo {
    /// Check whether this provider supports a specific feature.
    #[must_use]
    pub fn supports(&self, feature: &ProviderFeature) -> bool {
        self.features.contains(feature)
    }
}

/// Metadata about a specific model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub id: ModelId,
    pub display_name: String,
    pub max_context_tokens: u32,
    pub max_output_tokens: u32,
}

/// The input to a provider completion request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionRequest {
    pub model: ModelId,
    pub system_prompt: Option<String>,
    pub messages: Vec<Message>,
    pub tools: Vec<ToolDefinition>,
    pub max_tokens: u32,
    pub temperature: Option<f32>,
    pub stop_sequences: Vec<String>,
}

/// A message in a conversation.
#[allow(clippy::derive_partial_eq_without_eq)] // serde_json::Value does not impl Eq
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: Vec<ContentBlock>,
    pub timestamp: DateTime<Utc>,
}

/// A single piece of content within a message.
#[allow(clippy::derive_partial_eq_without_eq)] // serde_json::Value does not impl Eq
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlock {
    Text {
        text: String,
    },
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    ToolResult {
        tool_use_id: String,
        content: String,
        is_error: bool,
    },
    Image {
        media_type: String,
        data: String,
    },
}

/// A complete (non-streaming) response from the provider.
#[allow(clippy::derive_partial_eq_without_eq)] // serde_json::Value does not impl Eq
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompletionResponse {
    pub message: Message,
    pub stop_reason: StopReason,
    pub usage: TokenUsage,
    pub model: ModelId,
}

/// Why the model stopped generating.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StopReason {
    EndTurn,
    ToolUse,
    MaxTokens,
    StopSequence,
}

/// Token usage for cost tracking.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TokenUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub cache_read_tokens: Option<u32>,
    pub cache_creation_tokens: Option<u32>,
}

/// A chunk of a streaming response.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum StreamEvent {
    TextDelta(String),
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    Done {
        stop_reason: StopReason,
        usage: TokenUsage,
    },
    Error(String),
}

/// A tool definition sent to the provider so it knows what tools are available.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub input_schema: serde_json::Value,
}

/// The core provider trait. Every LLM backend implements this.
#[async_trait]
pub trait Provider: Send + Sync + 'static {
    fn info(&self) -> &ProviderInfo;

    async fn validate_credentials(&self) -> Result<(), ProviderError>;

    async fn complete(
        &self,
        request: CompletionRequest,
    ) -> Result<CompletionResponse, ProviderError>;

    async fn stream(
        &self,
        request: CompletionRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamEvent, ProviderError>> + Send>>, ProviderError>;
}

/// Classifies the kind of network failure for targeted retry strategies.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NetworkErrorKind {
    Timeout,
    ConnectionRefused,
    DnsFailure,
    TlsError,
    Other,
}

/// Provider-specific errors.
#[derive(Debug, thiserror::Error)]
pub enum ProviderError {
    #[error("authentication failed: {reason}")]
    AuthenticationFailed { reason: String },

    #[error("rate limited, retry after {retry_after_ms}ms")]
    RateLimited { retry_after_ms: u64 },

    #[error("model `{model}` not found or not supported")]
    ModelNotFound { model: String },

    #[error("context window exceeded: {used} tokens used, {max} max")]
    ContextOverflow { used: u32, max: u32 },

    #[error("provider API error: {status} — {body}")]
    ApiError { status: u16, body: String },

    #[error("network error ({kind:?}): {reason}")]
    Network {
        reason: String,
        kind: NetworkErrorKind,
        status_code: Option<u16>,
    },

    #[error("deserialization error: {0}")]
    Deserialization(String),

    #[error("provider not configured")]
    NotConfigured,
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_feature_serde_roundtrip() {
        for (feature, expected_json) in [
            (ProviderFeature::Streaming, "\"streaming\""),
            (ProviderFeature::ToolUse, "\"tool_use\""),
            (ProviderFeature::Vision, "\"vision\""),
        ] {
            let json = serde_json::to_string(&feature).unwrap();
            assert_eq!(json, expected_json);
            let back: ProviderFeature = serde_json::from_str(&json).unwrap();
            assert_eq!(back, feature);
        }
    }

    #[test]
    fn test_provider_info_supports_feature() {
        let info = ProviderInfo {
            id: ProviderId::from("anthropic"),
            display_name: "Anthropic Claude".into(),
            supported_models: vec![],
            features: BTreeSet::from([ProviderFeature::Streaming, ProviderFeature::ToolUse]),
        };

        assert!(info.supports(&ProviderFeature::Streaming));
        assert!(info.supports(&ProviderFeature::ToolUse));
        assert!(!info.supports(&ProviderFeature::Vision));
    }

    #[test]
    fn test_provider_info_uses_provider_id_newtype() {
        let info = ProviderInfo {
            id: ProviderId::from("anthropic"),
            display_name: "Anthropic Claude".into(),
            supported_models: vec![],
            features: BTreeSet::new(),
        };
        assert_eq!(info.id.as_str(), "anthropic");
    }

    #[test]
    fn test_model_info_uses_model_id_newtype() {
        let model = ModelInfo {
            id: ModelId::from("claude-opus-4-6-20250929"),
            display_name: "Claude Opus 4.6".into(),
            max_context_tokens: 200_000,
            max_output_tokens: 32_768,
        };
        assert_eq!(model.id.as_str(), "claude-opus-4-6-20250929");
    }

    #[test]
    fn test_provider_info_serde_roundtrip() {
        let info = ProviderInfo {
            id: ProviderId::from("anthropic"),
            display_name: "Anthropic Claude".into(),
            supported_models: vec![ModelInfo {
                id: ModelId::from("claude-opus-4-6"),
                display_name: "Claude Opus 4.6".into(),
                max_context_tokens: 200_000,
                max_output_tokens: 32_768,
            }],
            features: BTreeSet::from([
                ProviderFeature::Streaming,
                ProviderFeature::ToolUse,
                ProviderFeature::Vision,
            ]),
        };

        let json = serde_json::to_string(&info).unwrap();
        let back: ProviderInfo = serde_json::from_str(&json).unwrap();
        assert_eq!(back.id.as_str(), "anthropic");
        assert_eq!(back.features.len(), 3);
        assert!(back.supports(&ProviderFeature::Vision));
    }

    #[test]
    fn test_completion_request_model_is_model_id() {
        let request = CompletionRequest {
            model: ModelId::from("claude-opus-4-6-20250929"),
            system_prompt: None,
            messages: vec![],
            tools: vec![],
            max_tokens: 1024,
            temperature: None,
            stop_sequences: vec![],
        };
        assert_eq!(request.model.as_str(), "claude-opus-4-6-20250929");

        // Serde roundtrip preserves the newtype
        let json = serde_json::to_string(&request).unwrap();
        let back: CompletionRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(back.model.as_str(), "claude-opus-4-6-20250929");
    }

    #[test]
    fn test_completion_response_model_is_model_id() {
        let response = CompletionResponse {
            message: Message {
                role: Role::Assistant,
                content: vec![ContentBlock::Text {
                    text: "hello".into(),
                }],
                timestamp: chrono::Utc::now(),
            },
            stop_reason: StopReason::EndTurn,
            usage: TokenUsage::default(),
            model: ModelId::from("claude-opus-4-6-20250929"),
        };
        assert_eq!(response.model.as_str(), "claude-opus-4-6-20250929");
    }

    #[test]
    fn test_provider_feature_btreeset_deterministic_order() {
        // BTreeSet gives deterministic iteration — important for HMAC stability
        let features = BTreeSet::from([
            ProviderFeature::Vision,
            ProviderFeature::Streaming,
            ProviderFeature::ToolUse,
        ]);
        let json1 = serde_json::to_string(&features).unwrap();

        let features2 = BTreeSet::from([
            ProviderFeature::ToolUse,
            ProviderFeature::Streaming,
            ProviderFeature::Vision,
        ]);
        let json2 = serde_json::to_string(&features2).unwrap();

        assert_eq!(
            json1, json2,
            "BTreeSet serialization must be order-independent"
        );
    }
}
