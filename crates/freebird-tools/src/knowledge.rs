//! Knowledge store tools: `store_knowledge`, `search_knowledge`,
//! `update_knowledge`, `delete_knowledge`.
//!
//! All tools access the knowledge store via `ToolContext::knowledge_store`.
//! Content is scanned for sensitive material before writes. Consent-gated
//! kinds (`SystemConfig`, `ToolCapability`, `UserPreference`) are noted in
//! tool output but enforcement is delegated to the consent gate layer.

use std::collections::BTreeSet;
use std::fmt::Write as _;

use async_trait::async_trait;
use chrono::Utc;

use freebird_security::sensitive::contains_sensitive_content;
use freebird_security::taint::TaintedToolInput;
use freebird_traits::id::KnowledgeId;
use freebird_traits::knowledge::{KnowledgeEntry, KnowledgeKind, KnowledgeSource};
use freebird_traits::tool::{
    Capability, RiskLevel, SideEffects, Tool, ToolContext, ToolError, ToolInfo, ToolOutcome,
    ToolOutput,
};

/// Returns all knowledge tools as trait objects.
#[must_use]
pub fn knowledge_tools() -> Vec<Box<dyn Tool>> {
    vec![
        Box::new(StoreKnowledgeTool::new()),
        Box::new(SearchKnowledgeTool::new()),
        Box::new(UpdateKnowledgeTool::new()),
        Box::new(DeleteKnowledgeTool::new()),
    ]
}

/// Parse a `KnowledgeKind` from a `snake_case` string.
fn parse_kind(s: &str) -> Result<KnowledgeKind, String> {
    // serde_json expects a JSON string, so wrap in quotes
    let json = format!("\"{s}\"");
    serde_json::from_str(&json).map_err(|_| {
        format!(
            "invalid kind `{s}`: expected one of system_config, tool_capability, \
             user_preference, learned_pattern, error_resolution, session_insight"
        )
    })
}

/// Get the knowledge store from context, or return an error.
fn require_knowledge_store<'a>(
    context: &'a ToolContext<'_>,
    tool_name: &str,
) -> Result<&'a dyn freebird_traits::knowledge::KnowledgeStore, ToolError> {
    context
        .knowledge_store
        .ok_or_else(|| ToolError::ExecutionFailed {
            tool: tool_name.into(),
            reason: "knowledge store not configured".into(),
        })
}

// ── StoreKnowledgeTool ──────────────────────────────────────────

struct StoreKnowledgeTool {
    info: ToolInfo,
}

impl StoreKnowledgeTool {
    const NAME: &str = "store_knowledge";

    fn new() -> Self {
        Self {
            info: ToolInfo {
                name: Self::NAME.into(),
                description: "Store a new knowledge entry. Content is scanned for sensitive \
                    material. Consent-gated kinds (system_config, tool_capability, \
                    user_preference) require human approval."
                    .into(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "kind": {
                            "type": "string",
                            "description": "Knowledge category: system_config, tool_capability, user_preference, learned_pattern, error_resolution, session_insight",
                            "enum": ["system_config", "tool_capability", "user_preference", "learned_pattern", "error_resolution", "session_insight"]
                        },
                        "content": {
                            "type": "string",
                            "description": "The knowledge content to store"
                        },
                        "tags": {
                            "type": "string",
                            "description": "Comma-separated tags for categorization"
                        },
                        "confidence": {
                            "type": "number",
                            "description": "Confidence score 0.0-1.0 (default: 0.8)"
                        }
                    },
                    "required": ["kind", "content"]
                }),
                required_capability: Capability::FileWrite,
                risk_level: RiskLevel::Medium,
                side_effects: SideEffects::HasSideEffects,
            },
        }
    }
}

#[async_trait]
impl Tool for StoreKnowledgeTool {
    fn info(&self) -> &ToolInfo {
        &self.info
    }

    async fn execute(
        &self,
        input: serde_json::Value,
        context: &ToolContext<'_>,
    ) -> Result<ToolOutput, ToolError> {
        let store = require_knowledge_store(context, Self::NAME)?;
        let tainted = TaintedToolInput::new(input);

        let kind_str =
            tainted
                .extract_file_content("kind")
                .map_err(|e| ToolError::InvalidInput {
                    tool: Self::NAME.into(),
                    reason: e.to_string(),
                })?;
        let kind = parse_kind(kind_str.as_str()).map_err(|reason| ToolError::InvalidInput {
            tool: Self::NAME.into(),
            reason,
        })?;

        let body =
            tainted
                .extract_file_content("content")
                .map_err(|e| ToolError::InvalidInput {
                    tool: Self::NAME.into(),
                    reason: e.to_string(),
                })?;

        // Sensitive content scan — runs before consent gate
        if let Some(reason) = contains_sensitive_content(body.as_str()) {
            return Err(ToolError::SecurityViolation {
                tool: Self::NAME.into(),
                reason: format!("content blocked: {reason}"),
            });
        }

        // Parse optional tags (comma-separated string)
        let tags = tainted.extract_file_content("tags").map_or_else(
            |_| BTreeSet::new(),
            |tags_str| {
                tags_str
                    .as_str()
                    .split(',')
                    .map(|s| s.trim().to_owned())
                    .filter(|s| !s.is_empty())
                    .collect()
            },
        );

        // Parse optional confidence (must be in 0.0..=1.0)
        let confidence = match tainted.extract_file_content("confidence") {
            Err(_) => 0.8, // not provided — use default
            Ok(s) => {
                let val = s
                    .as_str()
                    .parse::<f32>()
                    .map_err(|_| ToolError::InvalidInput {
                        tool: Self::NAME.into(),
                        reason: format!("confidence must be a number, got: {}", s.as_str()),
                    })?;
                if !(0.0..=1.0).contains(&val) {
                    return Err(ToolError::InvalidInput {
                        tool: Self::NAME.into(),
                        reason: format!("confidence must be between 0.0 and 1.0, got: {val}"),
                    });
                }
                val
            }
        };

        let now = Utc::now();
        let id = KnowledgeId::from_string(uuid::Uuid::new_v4().to_string());
        let entry = KnowledgeEntry {
            id: id.clone(),
            kind: kind.clone(),
            content: body.as_str().to_owned(),
            tags,
            source: if kind.requires_consent() {
                KnowledgeSource::User
            } else {
                KnowledgeSource::Agent
            },
            confidence,
            session_id: Some(context.session_id.clone()),
            created_at: now,
            updated_at: now,
            access_count: 0,
            last_accessed: None,
        };

        let stored_id = store
            .store(entry)
            .await
            .map_err(|e| ToolError::ExecutionFailed {
                tool: Self::NAME.into(),
                reason: e.to_string(),
            })?;

        let consent_note = if kind.requires_consent() {
            " (consent-gated kind)"
        } else {
            ""
        };

        Ok(ToolOutput {
            content: format!("Stored knowledge entry {stored_id}{consent_note}"),
            outcome: ToolOutcome::Success,
            metadata: Some(serde_json::json!({ "id": stored_id.as_str() })),
        })
    }
}

// ── SearchKnowledgeTool ─────────────────────────────────────────

struct SearchKnowledgeTool {
    info: ToolInfo,
}

impl SearchKnowledgeTool {
    const NAME: &str = "search_knowledge";

    fn new() -> Self {
        Self {
            info: ToolInfo {
                name: Self::NAME.into(),
                description: "Search the knowledge store using full-text search. Returns \
                    entries ranked by relevance (BM25)."
                    .into(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query (supports FTS5 syntax)"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum results to return (default: 10, max: 50)"
                        },
                        "kind": {
                            "type": "string",
                            "description": "Optional: filter by knowledge kind",
                            "enum": ["system_config", "tool_capability", "user_preference", "learned_pattern", "error_resolution", "session_insight"]
                        }
                    },
                    "required": ["query"]
                }),
                required_capability: Capability::FileRead,
                risk_level: RiskLevel::Low,
                side_effects: SideEffects::None,
            },
        }
    }
}

/// Maximum search results to return.
const MAX_SEARCH_RESULTS: usize = 50;

/// Default search results limit.
const DEFAULT_SEARCH_LIMIT: usize = 10;

#[async_trait]
impl Tool for SearchKnowledgeTool {
    fn info(&self) -> &ToolInfo {
        &self.info
    }

    async fn execute(
        &self,
        input: serde_json::Value,
        context: &ToolContext<'_>,
    ) -> Result<ToolOutput, ToolError> {
        let store = require_knowledge_store(context, Self::NAME)?;
        let tainted = TaintedToolInput::new(input);

        let query = tainted
            .extract_file_content("query")
            .map_err(|e| ToolError::InvalidInput {
                tool: Self::NAME.into(),
                reason: e.to_string(),
            })?;

        // Parse optional limit
        let limit = tainted
            .extract_file_content("limit")
            .ok()
            .and_then(|s| s.as_str().parse::<usize>().ok())
            .unwrap_or(DEFAULT_SEARCH_LIMIT)
            .min(MAX_SEARCH_RESULTS);

        // Parse optional kind filter
        let kind_filter = tainted
            .extract_file_content("kind")
            .ok()
            .and_then(|s| parse_kind(s.as_str()).ok());

        let matches =
            store
                .search(query.as_str(), limit)
                .await
                .map_err(|e| ToolError::ExecutionFailed {
                    tool: Self::NAME.into(),
                    reason: e.to_string(),
                })?;

        // Filter by kind if specified
        let filtered: Vec<_> = if let Some(ref kind) = kind_filter {
            matches
                .into_iter()
                .filter(|m| &m.entry.kind == kind)
                .collect()
        } else {
            matches
        };

        if filtered.is_empty() {
            return Ok(ToolOutput {
                content: "No matching knowledge entries found.".into(),
                outcome: ToolOutcome::Success,
                metadata: None,
            });
        }

        // Record access for retrieved entries
        let ids: Vec<KnowledgeId> = filtered.iter().map(|m| m.entry.id.clone()).collect();
        // Best-effort access recording — don't fail the search if this errors
        if let Err(e) = store.record_access(&ids).await {
            tracing::warn!(error = %e, "failed to record access for knowledge entries");
        }

        // Format results
        let mut output = String::new();
        for (i, m) in filtered.iter().enumerate() {
            if i > 0 {
                output.push_str("\n---\n");
            }
            let tags_display = if m.entry.tags.is_empty() {
                "(none)".to_owned()
            } else {
                m.entry.tags.iter().cloned().collect::<Vec<_>>().join(", ")
            };
            let _ = write!(
                output,
                "[{}] ({:?}) {}\nTags: {}\nConfidence: {:.1} | Updated: {}",
                m.entry.id,
                m.entry.kind,
                m.entry.content,
                tags_display,
                m.entry.confidence,
                m.entry.updated_at.format("%Y-%m-%d %H:%M"),
            );
        }

        Ok(ToolOutput {
            content: output,
            outcome: ToolOutcome::Success,
            metadata: Some(serde_json::json!({ "count": filtered.len() })),
        })
    }
}

// ── UpdateKnowledgeTool ─────────────────────────────────────────

struct UpdateKnowledgeTool {
    info: ToolInfo,
}

impl UpdateKnowledgeTool {
    const NAME: &str = "update_knowledge";

    fn new() -> Self {
        Self {
            info: ToolInfo {
                name: Self::NAME.into(),
                description: "Update an existing knowledge entry's content, tags, or confidence. \
                    Consent-gated kinds require human approval."
                    .into(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "id": {
                            "type": "string",
                            "description": "The knowledge entry ID to update"
                        },
                        "content": {
                            "type": "string",
                            "description": "New content (optional — omit to keep existing)"
                        },
                        "tags": {
                            "type": "string",
                            "description": "New comma-separated tags (optional — omit to keep existing)"
                        },
                        "confidence": {
                            "type": "number",
                            "description": "New confidence score 0.0-1.0 (optional)"
                        }
                    },
                    "required": ["id"]
                }),
                required_capability: Capability::FileWrite,
                risk_level: RiskLevel::Medium,
                side_effects: SideEffects::HasSideEffects,
            },
        }
    }
}

#[async_trait]
impl Tool for UpdateKnowledgeTool {
    fn info(&self) -> &ToolInfo {
        &self.info
    }

    async fn execute(
        &self,
        input: serde_json::Value,
        context: &ToolContext<'_>,
    ) -> Result<ToolOutput, ToolError> {
        let store = require_knowledge_store(context, Self::NAME)?;
        let tainted = TaintedToolInput::new(input);

        let id_str = tainted
            .extract_file_content("id")
            .map_err(|e| ToolError::InvalidInput {
                tool: Self::NAME.into(),
                reason: e.to_string(),
            })?;
        let id = KnowledgeId::from_string(id_str.as_str());

        // Fetch existing entry
        let mut entry = store
            .get(&id)
            .await
            .map_err(|e| ToolError::ExecutionFailed {
                tool: Self::NAME.into(),
                reason: e.to_string(),
            })?
            .ok_or_else(|| ToolError::ExecutionFailed {
                tool: Self::NAME.into(),
                reason: format!("knowledge entry `{id}` not found"),
            })?;

        // Apply updates — content check has an early return, so suppress the
        // `useless_let_if_seq` lint that can't handle that pattern.
        #[expect(
            clippy::useless_let_if_seq,
            reason = "early return in conditional makes let-if-seq unavoidable"
        )]
        let mut changed = false;

        if let Ok(new_content) = tainted.extract_file_content("content") {
            // Sensitive content scan
            if let Some(reason) = contains_sensitive_content(new_content.as_str()) {
                return Err(ToolError::SecurityViolation {
                    tool: Self::NAME.into(),
                    reason: format!("content blocked: {reason}"),
                });
            }
            new_content.as_str().clone_into(&mut entry.content);
            changed = true;
        }

        if let Ok(tags_str) = tainted.extract_file_content("tags") {
            entry.tags = tags_str
                .as_str()
                .split(',')
                .map(|s| s.trim().to_owned())
                .filter(|s| !s.is_empty())
                .collect();
            changed = true;
        }

        if let Ok(conf_str) = tainted.extract_file_content("confidence") {
            if let Ok(conf) = conf_str.as_str().parse::<f32>() {
                entry.confidence = conf;
                changed = true;
            }
        }

        if !changed {
            return Ok(ToolOutput {
                content: "No fields to update were provided.".into(),
                outcome: ToolOutcome::Success,
                metadata: None,
            });
        }

        entry.updated_at = Utc::now();

        store
            .update(&entry)
            .await
            .map_err(|e| ToolError::ExecutionFailed {
                tool: Self::NAME.into(),
                reason: e.to_string(),
            })?;

        let consent_note = if entry.kind.requires_consent() {
            " (consent-gated kind)"
        } else {
            ""
        };

        Ok(ToolOutput {
            content: format!("Updated knowledge entry {id}{consent_note}"),
            outcome: ToolOutcome::Success,
            metadata: Some(serde_json::json!({ "id": id.as_str() })),
        })
    }
}

// ── DeleteKnowledgeTool ─────────────────────────────────────────

struct DeleteKnowledgeTool {
    info: ToolInfo,
}

impl DeleteKnowledgeTool {
    const NAME: &str = "delete_knowledge";

    fn new() -> Self {
        Self {
            info: ToolInfo {
                name: Self::NAME.into(),
                description: "Delete a knowledge entry by ID. Consent-gated kinds require \
                    human approval."
                    .into(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "id": {
                            "type": "string",
                            "description": "The knowledge entry ID to delete"
                        }
                    },
                    "required": ["id"]
                }),
                required_capability: Capability::FileDelete,
                risk_level: RiskLevel::Medium,
                side_effects: SideEffects::HasSideEffects,
            },
        }
    }
}

#[async_trait]
impl Tool for DeleteKnowledgeTool {
    fn info(&self) -> &ToolInfo {
        &self.info
    }

    async fn execute(
        &self,
        input: serde_json::Value,
        context: &ToolContext<'_>,
    ) -> Result<ToolOutput, ToolError> {
        let store = require_knowledge_store(context, Self::NAME)?;
        let tainted = TaintedToolInput::new(input);

        let id_str = tainted
            .extract_file_content("id")
            .map_err(|e| ToolError::InvalidInput {
                tool: Self::NAME.into(),
                reason: e.to_string(),
            })?;
        let id = KnowledgeId::from_string(id_str.as_str());

        // Verify entry exists and check if consent-gated
        let entry = store
            .get(&id)
            .await
            .map_err(|e| ToolError::ExecutionFailed {
                tool: Self::NAME.into(),
                reason: e.to_string(),
            })?
            .ok_or_else(|| ToolError::ExecutionFailed {
                tool: Self::NAME.into(),
                reason: format!("knowledge entry `{id}` not found"),
            })?;

        let consent_note = if entry.kind.requires_consent() {
            " (consent-gated kind)"
        } else {
            ""
        };

        store
            .delete(&id)
            .await
            .map_err(|e| ToolError::ExecutionFailed {
                tool: Self::NAME.into(),
                reason: e.to_string(),
            })?;

        Ok(ToolOutput {
            content: format!("Deleted knowledge entry {id}{consent_note}"),
            outcome: ToolOutcome::Success,
            metadata: Some(serde_json::json!({ "id": id.as_str() })),
        })
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic, clippy::indexing_slicing)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_kind_valid() {
        assert_eq!(
            parse_kind("system_config").unwrap(),
            KnowledgeKind::SystemConfig
        );
        assert_eq!(
            parse_kind("learned_pattern").unwrap(),
            KnowledgeKind::LearnedPattern
        );
        assert_eq!(
            parse_kind("error_resolution").unwrap(),
            KnowledgeKind::ErrorResolution
        );
    }

    #[test]
    fn test_parse_kind_invalid() {
        assert!(parse_kind("invalid").is_err());
        assert!(parse_kind("").is_err());
        assert!(parse_kind("SystemConfig").is_err()); // not snake_case
    }

    #[test]
    fn test_knowledge_tools_returns_four() {
        let tools = knowledge_tools();
        assert_eq!(tools.len(), 4);

        let mut names: Vec<String> = tools.iter().map(|t| t.info().name.clone()).collect();
        names.sort();
        assert_eq!(
            names,
            vec![
                "delete_knowledge",
                "search_knowledge",
                "store_knowledge",
                "update_knowledge",
            ]
        );
    }

    #[test]
    fn test_tool_risk_levels() {
        let tools = knowledge_tools();
        for tool in &tools {
            let info = tool.info();
            match info.name.as_str() {
                "search_knowledge" => {
                    assert_eq!(info.risk_level, RiskLevel::Low);
                    assert_eq!(info.side_effects, SideEffects::None);
                    assert_eq!(info.required_capability, Capability::FileRead);
                }
                "store_knowledge" | "update_knowledge" => {
                    assert_eq!(info.risk_level, RiskLevel::Medium);
                    assert_eq!(info.side_effects, SideEffects::HasSideEffects);
                    assert_eq!(info.required_capability, Capability::FileWrite);
                }
                "delete_knowledge" => {
                    assert_eq!(info.risk_level, RiskLevel::Medium);
                    assert_eq!(info.side_effects, SideEffects::HasSideEffects);
                    assert_eq!(info.required_capability, Capability::FileDelete);
                }
                other => panic!("unexpected tool: {other}"),
            }
        }
    }
}
