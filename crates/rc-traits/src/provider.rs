//! Provider trait — abstracts over LLM backends (Anthropic, OpenAI, Ollama, etc.).

use std::pin::Pin;

use async_trait::async_trait;
use futures::Stream;
use serde::{Deserialize, Serialize};

/// Metadata about a provider implementation.
#[derive(Debug, Clone)]
pub struct ProviderInfo {
    /// Unique identifier (e.g., "anthropic", "openai", "ollama").
    pub id: String,
    /// Human-readable name (e.g., "Anthropic Claude").
    pub display_name: String,
    /// Which models this provider supports.
    pub supported_models: Vec<ModelInfo>,
    /// Whether this provider supports streaming responses.
    pub supports_streaming: bool,
    /// Whether this provider supports tool use natively.
    pub supports_tool_use: bool,
    /// Whether this provider supports image input.
    pub supports_vision: bool,
}

/// Metadata about a specific model.
#[derive(Debug, Clone)]
pub struct ModelInfo {
    /// Model identifier sent to the API (e.g., "claude-opus-4-6-20250929").
    pub id: String,
    /// Human-readable name (e.g., "Claude Opus 4.6").
    pub display_name: String,
    /// Maximum context window in tokens.
    pub max_context_tokens: u32,
    /// Maximum output tokens.
    pub max_output_tokens: u32,
}

/// The input to a provider completion request.
#[derive(Debug, Clone)]
pub struct CompletionRequest {
    pub model: String,
    pub system_prompt: Option<String>,
    pub messages: Vec<Message>,
    pub tools: Vec<ToolDefinition>,
    pub max_tokens: u32,
    pub temperature: Option<f32>,
    pub stop_sequences: Vec<String>,
}

/// A message in a conversation (simplified for trait boundary).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: Vec<ContentBlock>,
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
#[derive(Debug, Clone)]
pub struct CompletionResponse {
    pub message: Message,
    pub stop_reason: StopReason,
    pub usage: TokenUsage,
    pub model: String,
}

/// Why the model stopped generating.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StopReason {
    EndTurn,
    ToolUse,
    MaxTokens,
    StopSequence,
}

/// Token usage for cost tracking.
#[derive(Debug, Clone, Default)]
pub struct TokenUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub cache_read_tokens: Option<u32>,
    pub cache_creation_tokens: Option<u32>,
}

/// A chunk of a streaming response.
#[derive(Debug, Clone)]
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
    /// Return metadata about this provider.
    fn info(&self) -> &ProviderInfo;

    /// Validate that the configured credentials are working.
    async fn validate_credentials(&self) -> Result<(), ProviderError>;

    /// Send a completion request and get a full response.
    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse, ProviderError>;

    /// Send a completion request and get a streaming response.
    async fn stream(
        &self,
        request: CompletionRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamEvent, ProviderError>> + Send>>, ProviderError>;
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

    #[error("network error: {0}")]
    Network(String),

    #[error("deserialization error: {0}")]
    Deserialization(String),

    #[error("provider not configured")]
    NotConfigured,
}
