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

use freebird_security::audit::{
    AuditEventType, AuditLogger, CapabilityCheckResult, InjectionSource,
};
use freebird_security::capability::CapabilityGrant;
use freebird_security::error::Severity;
use freebird_security::safe_types::ScannedToolOutput;
use freebird_traits::id::SessionId;
use freebird_traits::provider::ToolDefinition;
use freebird_traits::tool::{Capability, Tool, ToolContext, ToolOutput};

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

    /// Execute a tool by name. This is the ONLY entry point for tool execution.
    ///
    /// Enforces the mandatory security sequence from CLAUDE.md §11.2:
    /// 1. Tool lookup → error if not found (audit: Denied)
    /// 2. Capability check via `CapabilityGrant::check()` (audit: Denied)
    /// 3. TODO #29: consent gate for High/Critical risk tools
    /// 4. Audit: Granted
    /// 5. Build `ToolContext` from grant data and execute with timeout
    /// 6. Injection scan on non-error output via `ScannedToolOutput::from_raw()`
    ///
    /// **Infallible**: returns `ToolOutput` with `is_error=true` on failure.
    pub async fn execute(
        &self,
        tool_name: &str,
        input: serde_json::Value,
        grant: &CapabilityGrant,
        session_id: &SessionId,
    ) -> ToolOutput {
        // 1. Tool lookup
        let Some(tool) = self.tools.get(tool_name) else {
            self.audit_tool_invocation(
                session_id,
                tool_name,
                CapabilityCheckResult::Denied {
                    reason: "tool not found".into(),
                },
            )
            .await;
            return ToolOutput {
                content: format!("Unknown tool: {tool_name}"),
                is_error: true,
                metadata: None,
            };
        };

        // 2. Capability + expiration check
        if let Err(e) = grant.check(&tool.info().required_capability) {
            self.audit_tool_invocation(
                session_id,
                tool_name,
                CapabilityCheckResult::Denied {
                    reason: format!("{e}"),
                },
            )
            .await;
            return ToolOutput {
                content: format!("Capability denied for tool `{tool_name}`: {e}"),
                is_error: true,
                metadata: None,
            };
        }

        // 3. TODO #29: consent gate for High/Critical risk tools
        //    When consent gates land, High/Critical risk tools will block
        //    here until the user approves or the request times out.

        // 4. Audit: capability check passed
        self.audit_tool_invocation(session_id, tool_name, CapabilityCheckResult::Granted)
            .await;

        // 5. Build ToolContext from grant data and execute with timeout
        let caps_vec: Vec<Capability> = grant.capabilities().iter().cloned().collect();
        let context = ToolContext {
            session_id,
            sandbox_root: grant.sandbox_root(),
            granted_capabilities: &caps_vec,
        };

        let output =
            match tokio::time::timeout(self.default_timeout, tool.execute(input, &context)).await {
                Ok(Ok(output)) => output,
                Ok(Err(e)) => ToolOutput {
                    content: format!("Tool error: {e}"),
                    is_error: true,
                    metadata: None,
                },
                Err(_elapsed) => {
                    self.audit_policy_violation(
                        session_id,
                        "tool_timeout",
                        &format!(
                            "tool `{tool_name}` exceeded {}ms timeout",
                            self.default_timeout.as_millis()
                        ),
                        Severity::Medium,
                    )
                    .await;
                    return ToolOutput {
                        content: format!(
                            "Tool `{tool_name}` timed out after {}ms",
                            self.default_timeout.as_millis()
                        ),
                        is_error: true,
                        metadata: None,
                    };
                }
            };

        // 6. Injection scan on non-error output — BLOCK if detected.
        //    Error output is generated by our code, not untrusted sources.
        if output.is_error {
            return output;
        }

        match ScannedToolOutput::from_raw(&output.content) {
            Ok(_scanned) => output,
            Err(e) => {
                tracing::warn!(
                    tool = %tool_name,
                    session_id = %session_id,
                    error = %e,
                    "injection detected in tool output, blocking"
                );
                self.audit_injection_detected(session_id, &format!("{e}"))
                    .await;
                ToolOutput {
                    content: "Tool output blocked: potential prompt injection detected".into(),
                    is_error: true,
                    metadata: None,
                }
            }
        }
    }

    // ── Private helpers ─────────────────────────────────────────────

    async fn audit_tool_invocation(
        &self,
        session_id: &SessionId,
        tool_name: &str,
        result: CapabilityCheckResult,
    ) {
        if let Some(audit) = &self.audit {
            let _ = audit
                .record(
                    session_id.as_str(),
                    AuditEventType::ToolInvocation {
                        tool_name: tool_name.into(),
                        capability_check: result,
                    },
                )
                .await;
        }
    }

    async fn audit_policy_violation(
        &self,
        session_id: &SessionId,
        rule: &str,
        context: &str,
        severity: Severity,
    ) {
        if let Some(audit) = &self.audit {
            let _ = audit
                .record(
                    session_id.as_str(),
                    AuditEventType::PolicyViolation {
                        rule: rule.into(),
                        context: context.into(),
                        severity,
                    },
                )
                .await;
        }
    }

    async fn audit_injection_detected(&self, session_id: &SessionId, pattern: &str) {
        if let Some(audit) = &self.audit {
            let _ = audit
                .record(
                    session_id.as_str(),
                    AuditEventType::InjectionDetected {
                        pattern: pattern.into(),
                        source: InjectionSource::ToolOutput,
                        severity: Severity::High,
                    },
                )
                .await;
        }
    }
}
