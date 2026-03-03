//! Provider trait — abstracts over LLM backends (Anthropic, `OpenAI`, Ollama, etc.).

use std::pin::Pin;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use futures::Stream;
use serde::{Deserialize, Serialize};

/// The role of a participant in a conversation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    User,
    Assistant,
    System,
    Tool,
}

/// Metadata about a provider implementation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderInfo {
    pub id: String,
    pub display_name: String,
    pub supported_models: Vec<ModelInfo>,
    pub supports_streaming: bool,
    pub supports_tool_use: bool,
    pub supports_vision: bool,
}

/// Metadata about a specific model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub id: String,
    pub display_name: String,
    pub max_context_tokens: u32,
    pub max_output_tokens: u32,
}

/// The input to a provider completion request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionRequest {
    pub model: String,
    pub system_prompt: Option<String>,
    pub messages: Vec<Message>,
    pub tools: Vec<ToolDefinition>,
    pub max_tokens: u32,
    pub temperature: Option<f32>,
    pub stop_sequences: Vec<String>,
}

/// A message in a conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: Vec<ContentBlock>,
    pub timestamp: DateTime<Utc>,
}

/// A single piece of content within a message.
#[derive(Debug, Clone, Serialize, Deserialize)]
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
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionResponse {
    pub message: Message,
    pub stop_reason: StopReason,
    pub usage: TokenUsage,
    pub model: String,
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
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
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
