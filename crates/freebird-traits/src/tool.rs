//! Tool trait — abstracts over agent capabilities (filesystem, shell, network, etc.).

use std::path::{Path, PathBuf};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::id::SessionId;
use crate::provider::ToolDefinition;

/// Capabilities that tools may require.
///
/// Finer-grained than a simple read/write/execute model — separating
/// `FileDelete` from `FileWrite`, `ProcessSpawn` from `ShellExecute`,
/// and inbound vs. outbound network access enables least-privilege grants.
///
/// # Variant ordering contract
///
/// `Ord` is derived, so variant declaration order determines `BTreeSet`
/// iteration and serialization order. Reordering variants changes the
/// byte-level representation of serialized `CapabilityGrant`s, which
/// breaks HMAC signatures over persisted grants. **Append new variants
/// at the end only.**
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
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

/// The number of variants in [`Capability`]. Used by test helpers to
/// detect when a new variant is added without updating exhaustive lists.
pub const CAPABILITY_VARIANT_COUNT: usize = 8;

/// How dangerous a tool invocation is — drives consent prompts and audit depth.
///
/// # Variant ordering contract
///
/// `Ord` is derived so that `Low < Medium < High < Critical`. The consent
/// gate (CLAUDE.md §15) compares `tool_risk >= config.require_consent_above`
/// — this ordering makes that comparison correct.  **Append new variants
/// at the end only; inserting in the middle changes comparison semantics.**
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RiskLevel {
    Low,
    Medium,
    High,
    Critical,
}

/// Whether a tool has observable side effects (writes, mutations, network calls).
///
/// Replaces the `bool` anti-pattern (CLAUDE.md §23). Used by the runtime to
/// decide whether a consent gate or extra audit logging is warranted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SideEffects {
    None,
    HasSideEffects,
}

/// Whether a tool execution succeeded or failed.
///
/// Replaces the `bool` anti-pattern (CLAUDE.md §23). Used in `ToolOutput`
/// and `ToolInvocation` to indicate the outcome of a tool call.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolOutcome {
    Success,
    Error,
}

/// Metadata describing a tool for both the runtime and the LLM.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolInfo {
    pub name: String,
    pub description: String,
    pub input_schema: serde_json::Value,
    pub required_capability: Capability,
    pub risk_level: RiskLevel,
    pub side_effects: SideEffects,
}

/// Context passed to every tool invocation by the runtime.
///
/// The runtime verifies capability grants and logs audit events before
/// calling `Tool::execute`. This context provides sandbox boundaries and
/// the granted capabilities for tools that need sub-capability checks.
pub struct ToolContext<'a> {
    pub session_id: &'a SessionId,
    pub sandbox_root: &'a Path,
    pub granted_capabilities: &'a [Capability],
    /// Additional directories beyond `sandbox_root` that tools may access.
    /// Absolute paths provided by the user via `--allow-dir`.
    pub allowed_directories: &'a [PathBuf],
    /// Knowledge store for knowledge tools. `None` if not configured.
    pub knowledge_store: Option<&'a dyn crate::knowledge::KnowledgeStore>,
    /// Memory backend for session recall tools. `None` if not configured.
    pub memory: Option<&'a dyn crate::memory::Memory>,
}

impl std::fmt::Debug for ToolContext<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ToolContext")
            .field("session_id", &self.session_id)
            .field("sandbox_root", &self.sandbox_root)
            .field("granted_capabilities", &self.granted_capabilities)
            .field("allowed_directories", &self.allowed_directories)
            .field("knowledge_store", &self.knowledge_store.is_some())
            .field("memory", &self.memory.is_some())
            .finish()
    }
}

/// The result of a tool execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolOutput {
    pub content: String,
    pub outcome: ToolOutcome,
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

    #[error("approval denied: {context}")]
    ApprovalDenied { context: String },

    #[error("approval expired: {context}")]
    ApprovalExpired { context: String },
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn test_risk_level_ordering_low_through_critical() {
        assert!(RiskLevel::Low < RiskLevel::Medium);
        assert!(RiskLevel::Medium < RiskLevel::High);
        assert!(RiskLevel::High < RiskLevel::Critical);
    }

    #[test]
    fn test_risk_level_consent_gate_comparison() {
        // Simulates the consent gate check: tool_risk >= config.require_consent_above
        let threshold = RiskLevel::High;
        assert!(RiskLevel::Critical >= threshold);
        assert!(RiskLevel::High >= threshold);
        assert!(RiskLevel::Medium < threshold);
        assert!(RiskLevel::Low < threshold);
    }

    #[test]
    fn test_risk_level_serde_roundtrip() {
        for (level, expected_json) in [
            (RiskLevel::Low, "\"low\""),
            (RiskLevel::Medium, "\"medium\""),
            (RiskLevel::High, "\"high\""),
            (RiskLevel::Critical, "\"critical\""),
        ] {
            let json = serde_json::to_string(&level).unwrap();
            assert_eq!(json, expected_json);
            let back: RiskLevel = serde_json::from_str(&json).unwrap();
            assert_eq!(back, level);
        }
    }
}
