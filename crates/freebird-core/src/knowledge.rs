//! System knowledge bootstrapping.
//!
//! Populates the knowledge store with tool capabilities and system configuration
//! on startup. Uses `replace_kind()` for idempotent updates — safe to call on
//! every boot.

use std::collections::BTreeSet;

use chrono::Utc;

use freebird_runtime::tool_registry::ToolRegistry;
use freebird_traits::id::KnowledgeId;
use freebird_traits::knowledge::{
    KnowledgeEntry, KnowledgeError, KnowledgeKind, KnowledgeSource, KnowledgeStore,
};
use freebird_types::config::AppConfig;

/// Populate the knowledge store with system-level entries on startup.
///
/// Two categories are populated:
/// - **`ToolCapability`**: one entry per registered tool (name, description, capability, risk).
/// - **`SystemConfig`**: key runtime configuration facts.
pub async fn populate_system_knowledge(
    store: &dyn KnowledgeStore,
    tool_registry: &ToolRegistry,
    config: &AppConfig,
) -> Result<(), KnowledgeError> {
    let now = Utc::now();

    // --- ToolCapability entries ---
    let mut tool_entries = Vec::with_capacity(tool_registry.tool_count());
    for tool in tool_registry.iter() {
        let info = tool.info();
        let content = format!(
            "Tool: {}\nDescription: {}\nCapability: {:?}\nRisk: {:?}\nSide effects: {:?}",
            info.name,
            info.description,
            info.required_capability,
            info.risk_level,
            info.side_effects,
        );
        tool_entries.push(KnowledgeEntry {
            id: KnowledgeId::from_string(uuid::Uuid::new_v4().to_string()),
            kind: KnowledgeKind::ToolCapability,
            content,
            tags: BTreeSet::from(["tool".to_owned(), info.name.clone()]),
            source: KnowledgeSource::System,
            confidence: 1.0,
            session_id: None,
            created_at: now,
            updated_at: now,
            access_count: 0,
            last_accessed: None,
        });
    }

    store
        .replace_kind(&KnowledgeKind::ToolCapability, tool_entries)
        .await?;

    // --- SystemConfig entries ---
    let config_facts = [
        format!(
            "Default provider: {}, default model: {}",
            config.runtime.default_provider, config.runtime.default_model,
        ),
        format!(
            "Max tool rounds per turn: {}, max output tokens: {}",
            config.runtime.max_tool_rounds, config.runtime.max_output_tokens,
        ),
        format!(
            "Consent required above: {:?}, consent timeout: {}s",
            config.security.require_consent_above, config.security.consent_timeout_secs,
        ),
        format!(
            "Knowledge auto-retrieve: {}, max context entries: {}, relevance threshold: {}",
            config.knowledge.auto_retrieve,
            config.knowledge.max_context_entries,
            config.knowledge.relevance_threshold,
        ),
    ];

    let config_entries: Vec<KnowledgeEntry> = config_facts
        .into_iter()
        .map(|content| KnowledgeEntry {
            id: KnowledgeId::from_string(uuid::Uuid::new_v4().to_string()),
            kind: KnowledgeKind::SystemConfig,
            content,
            tags: BTreeSet::from(["config".to_owned()]),
            source: KnowledgeSource::System,
            confidence: 1.0,
            session_id: None,
            created_at: now,
            updated_at: now,
            access_count: 0,
            last_accessed: None,
        })
        .collect();

    let entry_count = config_entries.len();
    store
        .replace_kind(&KnowledgeKind::SystemConfig, config_entries)
        .await?;

    tracing::info!(
        tool_capabilities = tool_registry.tool_count(),
        system_config = entry_count,
        "system knowledge bootstrapped"
    );

    Ok(())
}
