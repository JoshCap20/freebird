//! Tool registry for managing built-in tools.
//!
//! `ToolRegistry` stores registered [`Tool`] implementations and provides
//! lookup by name, iteration, and tool definition generation for LLM requests.
//!
//! Analogous to [`ProviderRegistry`](super::registry::ProviderRegistry) but
//! for tools rather than LLM providers.

use std::collections::HashMap;

use freebird_traits::provider::ToolDefinition;
use freebird_traits::tool::Tool;

/// Manages registered tools and provides name-based lookup.
///
/// Constructed via `&mut self` methods at startup. During operation, only
/// `&self` methods are callable — Rust ownership enforces immutability
/// once the registry is moved into the `AgentRuntime`.
pub struct ToolRegistry {
    tools: HashMap<String, Box<dyn Tool>>,
    /// Insertion order, so tool definitions are presented to the LLM in a
    /// stable, predictable order (filesystem first, then shell, then network).
    insertion_order: Vec<String>,
}

impl ToolRegistry {
    /// Create an empty registry with no tools.
    #[must_use]
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
            insertion_order: Vec::new(),
        }
    }

    /// Register a tool. The tool's `info().name` is used as the key.
    ///
    /// If a tool with the same name was already registered, it is replaced
    /// and the old tool is returned.
    pub fn register(&mut self, tool: Box<dyn Tool>) -> Option<Box<dyn Tool>> {
        let name = tool.info().name.clone();
        let old = self.tools.insert(name.clone(), tool);
        if old.is_none() {
            self.insertion_order.push(name);
        }
        old
    }

    /// Register multiple tools at once.
    pub fn register_all(&mut self, tools: Vec<Box<dyn Tool>>) {
        for tool in tools {
            self.register(tool);
        }
    }

    /// Look up a tool by name.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&dyn Tool> {
        self.tools.get(name).map(AsRef::as_ref)
    }

    /// Return all registered tool names in insertion order.
    #[must_use]
    pub fn tool_names(&self) -> &[String] {
        &self.insertion_order
    }

    /// Return the number of registered tools.
    #[must_use]
    pub fn tool_count(&self) -> usize {
        self.tools.len()
    }

    /// Return true if no tools are registered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }

    /// Iterate over all registered tools in insertion order.
    pub fn iter(&self) -> impl Iterator<Item = &dyn Tool> {
        self.insertion_order
            .iter()
            .filter_map(|name| self.tools.get(name).map(AsRef::as_ref))
    }

    /// Generate tool definitions for all registered tools, in insertion order.
    ///
    /// Used when building `CompletionRequest`s for the LLM.
    #[must_use]
    pub fn to_definitions(&self) -> Vec<ToolDefinition> {
        self.iter().map(Tool::to_definition).collect()
    }
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::indexing_slicing)]
mod tests {
    use super::*;

    use async_trait::async_trait;
    use freebird_traits::tool::{
        Capability, RiskLevel, SideEffects, ToolContext, ToolError, ToolInfo, ToolOutput,
    };

    struct MockTool {
        info: ToolInfo,
    }

    impl MockTool {
        fn new(name: &str) -> Self {
            Self {
                info: ToolInfo {
                    name: name.into(),
                    description: format!("mock {name}"),
                    input_schema: serde_json::json!({"type": "object"}),
                    required_capability: Capability::FileRead,
                    risk_level: RiskLevel::Low,
                    side_effects: SideEffects::None,
                },
            }
        }
    }

    #[async_trait]
    impl Tool for MockTool {
        fn info(&self) -> &ToolInfo {
            &self.info
        }

        async fn execute(
            &self,
            _input: serde_json::Value,
            _context: &ToolContext<'_>,
        ) -> Result<ToolOutput, ToolError> {
            Ok(ToolOutput {
                content: "ok".into(),
                outcome: freebird_traits::tool::ToolOutcome::Success,
                metadata: None,
            })
        }
    }

    #[test]
    fn test_new_is_empty() {
        let registry = ToolRegistry::new();
        assert!(registry.is_empty());
        assert_eq!(registry.tool_count(), 0);
        assert!(registry.tool_names().is_empty());
    }

    #[test]
    fn test_register_and_get() {
        let mut registry = ToolRegistry::new();
        registry.register(Box::new(MockTool::new("read_file")));

        assert!(!registry.is_empty());
        assert_eq!(registry.tool_count(), 1);
        assert!(registry.get("read_file").is_some());
        assert!(registry.get("nonexistent").is_none());
    }

    #[test]
    fn test_register_returns_old_on_duplicate() {
        let mut registry = ToolRegistry::new();
        let old = registry.register(Box::new(MockTool::new("read_file")));
        assert!(old.is_none());

        let old = registry.register(Box::new(MockTool::new("read_file")));
        assert!(old.is_some());
        assert_eq!(registry.tool_count(), 1);
    }

    #[test]
    fn test_register_all() {
        let mut registry = ToolRegistry::new();
        registry.register_all(vec![
            Box::new(MockTool::new("read_file")),
            Box::new(MockTool::new("write_file")),
            Box::new(MockTool::new("shell")),
        ]);
        assert_eq!(registry.tool_count(), 3);
    }

    #[test]
    fn test_insertion_order_preserved() {
        let mut registry = ToolRegistry::new();
        registry.register(Box::new(MockTool::new("c_tool")));
        registry.register(Box::new(MockTool::new("a_tool")));
        registry.register(Box::new(MockTool::new("b_tool")));

        assert_eq!(registry.tool_names(), &["c_tool", "a_tool", "b_tool"]);
    }

    #[test]
    fn test_iter_returns_insertion_order() {
        let mut registry = ToolRegistry::new();
        registry.register(Box::new(MockTool::new("first")));
        registry.register(Box::new(MockTool::new("second")));

        let names: Vec<&str> = registry.iter().map(|t| t.info().name.as_str()).collect();
        assert_eq!(names, vec!["first", "second"]);
    }

    #[test]
    fn test_to_definitions() {
        let mut registry = ToolRegistry::new();
        registry.register(Box::new(MockTool::new("alpha")));
        registry.register(Box::new(MockTool::new("beta")));

        let defs = registry.to_definitions();
        assert_eq!(defs.len(), 2);
        assert_eq!(defs[0].name, "alpha");
        assert_eq!(defs[1].name, "beta");
    }

    #[test]
    fn test_duplicate_does_not_double_insertion_order() {
        let mut registry = ToolRegistry::new();
        registry.register(Box::new(MockTool::new("tool_a")));
        registry.register(Box::new(MockTool::new("tool_b")));
        registry.register(Box::new(MockTool::new("tool_a"))); // duplicate

        assert_eq!(registry.tool_names(), &["tool_a", "tool_b"]);
        assert_eq!(registry.tool_count(), 2);
    }
}
