//! `ToolExecutor` — the single security chokepoint for all tool invocations.
//!
//! Every tool call flows through [`ToolExecutor::execute`], which enforces the
//! mandatory security sequence from CLAUDE.md §11.2:
//!
//! 1. Tool lookup
//! 2. Capability + expiration check via [`CapabilityGrant::check`]
//! 3. Consent gate (TODO #29)
//! 4. Audit logging
//! 5. Execution with timeout
//! 6. Injection scan on output via [`ScannedToolOutput::from_raw`]

use std::collections::HashMap;
use std::time::Duration;

use freebird_security::audit::AuditLogger;
use freebird_security::capability::CapabilityGrant;
use freebird_traits::provider::ToolDefinition;
use freebird_traits::tool::Tool;

/// The single security boundary through which all tool calls flow.
///
/// Centralizes capability checks, timeout enforcement, injection scanning,
/// and audit logging so that no call site can accidentally skip a security
/// step. Constructed once at startup and shared (immutably) for the
/// lifetime of the runtime.
pub struct ToolExecutor {
    tools: HashMap<String, Box<dyn Tool>>,
    default_timeout: Duration,
    audit: Option<AuditLogger>,
}

impl std::fmt::Debug for ToolExecutor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ToolExecutor")
            .field("tool_count", &self.tools.len())
            .field("default_timeout", &self.default_timeout)
            .field("has_audit", &self.audit.is_some())
            .finish()
    }
}

impl ToolExecutor {
    /// Create a new executor from a list of tools, a default timeout,
    /// and an optional audit logger.
    ///
    /// # Errors
    ///
    /// Returns an error if two or more tools share the same name.
    /// Duplicate tool names are a configuration bug — fail loudly at
    /// startup rather than silently overwriting (CLAUDE.md §3.4).
    pub fn new(
        tools: Vec<Box<dyn Tool>>,
        default_timeout: Duration,
        audit: Option<AuditLogger>,
    ) -> Result<Self, anyhow::Error> {
        let mut map = HashMap::with_capacity(tools.len());
        for tool in tools {
            let name = tool.info().name.clone();
            if map.contains_key(&name) {
                anyhow::bail!("duplicate tool name: `{name}`");
            }
            map.insert(name, tool);
        }
        Ok(Self {
            tools: map,
            default_timeout,
            audit,
        })
    }

    /// Return definitions for all registered tools (sent to provider).
    ///
    /// Sorted by tool name for deterministic provider API calls.
    #[must_use]
    pub fn tool_definitions(&self) -> Vec<ToolDefinition> {
        let mut defs: Vec<_> = self.tools.values().map(|t| t.to_definition()).collect();
        defs.sort_by(|a, b| a.name.cmp(&b.name));
        defs
    }

    /// Return definitions for tools the given grant permits.
    ///
    /// Filters out tools whose `required_capability` is not in the grant
    /// (or if the grant is expired). Used by the runtime to send only
    /// callable tools to the provider.
    #[must_use]
    pub fn tool_definitions_for_grant(&self, grant: &CapabilityGrant) -> Vec<ToolDefinition> {
        let mut defs: Vec<_> = self
            .tools
            .values()
            .filter(|t| grant.check(&t.info().required_capability).is_ok())
            .map(|t| t.to_definition())
            .collect();
        defs.sort_by(|a, b| a.name.cmp(&b.name));
        defs
    }

    /// Look up a tool by name.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&dyn Tool> {
        self.tools.get(name).map(AsRef::as_ref)
    }

    /// Number of registered tools.
    #[must_use]
    pub fn tool_count(&self) -> usize {
        self.tools.len()
    }
}
