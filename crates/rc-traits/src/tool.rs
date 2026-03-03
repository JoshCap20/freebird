//! Tool trait — abstracts over agent capabilities (filesystem, shell, network, etc.).

use async_trait::async_trait;

use crate::provider::ToolDefinition;

/// Metadata describing a tool for both the runtime and the LLM.
#[derive(Debug, Clone)]
pub struct ToolInfo {
    /// Unique name (matches what the LLM will call, e.g., "read_file").
    pub name: String,
    /// Human-readable description sent to the LLM.
    pub description: String,
    /// JSON Schema for the tool's input parameters.
    pub input_schema: serde_json::Value,
    /// Whether this tool performs I/O (affects sandboxing decisions).
    pub has_side_effects: bool,
}

/// The result of a tool execution.
#[derive(Debug, Clone)]
pub struct ToolOutput {
    pub content: String,
    pub is_error: bool,
    pub metadata: Option<serde_json::Value>,
}

/// The core tool trait.
#[async_trait]
pub trait Tool: Send + Sync + 'static {
    /// Return metadata about this tool.
    fn info(&self) -> &ToolInfo;

    /// Convert this tool's info into the [`ToolDefinition`] format sent to providers.
    fn to_definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: self.info().name.clone(),
            description: self.info().description.clone(),
            input_schema: self.info().input_schema.clone(),
        }
    }

    /// Execute the tool with the given input.
    async fn execute(
        &self,
        input: serde_json::Value,
    ) -> Result<ToolOutput, ToolError>;
}

/// Tool-specific errors.
#[derive(Debug, thiserror::Error)]
pub enum ToolError {
    #[error("tool `{tool}` execution failed: {reason}")]
    ExecutionFailed { tool: String, reason: String },

    #[error("tool `{tool}` timed out after {timeout_ms}ms")]
    Timeout { tool: String, timeout_ms: u64 },

    #[error("tool `{tool}` input validation failed: {reason}")]
    InvalidInput { tool: String, reason: String },

    #[error("security violation in tool `{tool}`: {reason}")]
    SecurityViolation { tool: String, reason: String },
}
