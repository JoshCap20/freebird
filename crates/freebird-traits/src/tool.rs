//! Tool trait — abstracts over agent capabilities (filesystem, shell, network, etc.).

use std::path::Path;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::id::SessionId;
use crate::provider::ToolDefinition;

/// Capabilities that tools may require.
///
/// Finer-grained than a simple read/write/execute model — separating
/// `FileDelete` from `FileWrite`, `ProcessSpawn` from `ShellExecute`,
/// and inbound vs. outbound network access enables least-privilege grants.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Capability {
    FileRead,
    FileWrite,
    FileDelete,
    ShellExecute,
    ProcessSpawn,
    NetworkOutbound,
    NetworkListen,
    EnvRead,
}

/// How dangerous a tool invocation is — drives consent prompts and audit depth.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RiskLevel {
    Low,
    Medium,
    High,
    Critical,
}

/// Metadata describing a tool for both the runtime and the LLM.
#[derive(Debug, Clone)]
pub struct ToolInfo {
    pub name: String,
    pub description: String,
    pub input_schema: serde_json::Value,
    pub required_capability: Capability,
    pub risk_level: RiskLevel,
    pub has_side_effects: bool,
}

/// Context passed to every tool invocation by the runtime.
pub struct ToolContext<'a> {
    pub session_id: &'a SessionId,
    pub sandbox_root: &'a Path,
}

/// The result of a tool execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolOutput {
    pub content: String,
    pub is_error: bool,
    pub metadata: Option<serde_json::Value>,
}

/// The core tool trait.
#[async_trait]
pub trait Tool: Send + Sync + 'static {
    fn info(&self) -> &ToolInfo;

    fn to_definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: self.info().name.clone(),
            description: self.info().description.clone(),
            input_schema: self.info().input_schema.clone(),
        }
    }

    async fn execute(
        &self,
        input: serde_json::Value,
        context: &ToolContext<'_>,
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
