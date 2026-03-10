//! Knowledge store trait — abstracts over persistent agent knowledge backends.

use std::collections::BTreeSet;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::id::{KnowledgeId, SessionId};

/// Categories of knowledge the agent can store and retrieve.
///
/// # Consent rules
///
/// `SystemConfig`, `ToolCapability`, and `UserPreference` require human consent
/// for any write, update, or delete. Agent-owned kinds (`LearnedPattern`,
/// `ErrorResolution`, `SessionInsight`) can be modified autonomously.
///
/// # Variant ordering contract
///
/// `Ord` is derived — do not reorder existing variants. Append new variants at the end.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KnowledgeKind {
    SystemConfig,
    ToolCapability,
    UserPreference,
    LearnedPattern,
    ErrorResolution,
    SessionInsight,
}

impl KnowledgeKind {
    /// Whether modifying entries of this kind requires human consent.
    #[must_use]
    pub const fn requires_consent(&self) -> bool {
        matches!(
            self,
            Self::SystemConfig | Self::ToolCapability | Self::UserPreference
        )
    }

    /// Whether the agent owns entries of this kind (can write without consent).
    #[must_use]
    pub const fn agent_owned(&self) -> bool {
        !self.requires_consent()
    }
}

/// Who created a knowledge entry.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KnowledgeSource {
    /// Auto-populated by the system at startup.
    System,
    /// Explicitly declared by the user (via consent-gated tool call).
    User,
    /// Inferred by the agent during conversation.
    Agent,
}

/// A single knowledge entry in the knowledge store.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeEntry {
    pub id: KnowledgeId,
    pub kind: KnowledgeKind,
    pub content: String,
    pub tags: BTreeSet<String>,
    pub source: KnowledgeSource,
    /// Confidence score (0.0–1.0). System entries default to 1.0.
    pub confidence: f32,
    /// Session that created this entry. `None` for system-populated entries.
    pub session_id: Option<SessionId>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    /// How many times this entry was retrieved for context injection.
    pub access_count: u64,
    /// Last time this entry was retrieved for context injection.
    pub last_accessed: Option<DateTime<Utc>>,
}

/// A ranked search result from the knowledge store.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeMatch {
    pub entry: KnowledgeEntry,
    /// BM25 relevance score. Lower (more negative) = more relevant.
    pub rank: f64,
}

/// The core knowledge store trait.
#[async_trait]
pub trait KnowledgeStore: Send + Sync + 'static {
    /// Store a new knowledge entry. Returns the assigned ID.
    async fn store(&self, entry: KnowledgeEntry) -> Result<KnowledgeId, KnowledgeError>;

    /// Update an existing entry's content, tags, or confidence.
    async fn update(&self, entry: &KnowledgeEntry) -> Result<(), KnowledgeError>;

    /// Retrieve a single entry by ID.
    async fn get(&self, id: &KnowledgeId) -> Result<Option<KnowledgeEntry>, KnowledgeError>;

    /// Delete an entry by ID.
    async fn delete(&self, id: &KnowledgeId) -> Result<(), KnowledgeError>;

    /// FTS5 ranked search. Returns entries ordered by BM25 relevance.
    async fn search(
        &self,
        query: &str,
        limit: usize,
    ) -> Result<Vec<KnowledgeMatch>, KnowledgeError>;

    /// List entries filtered by kind, ordered by `updated_at` descending.
    async fn list_by_kind(
        &self,
        kind: &KnowledgeKind,
        limit: usize,
    ) -> Result<Vec<KnowledgeEntry>, KnowledgeError>;

    /// List entries that contain a specific tag.
    async fn list_by_tag(
        &self,
        tag: &str,
        limit: usize,
    ) -> Result<Vec<KnowledgeEntry>, KnowledgeError>;

    /// Replace all entries of a given kind (for system auto-population).
    ///
    /// Deletes all existing entries with `kind`, then inserts `entries`.
    /// Runs in a single transaction for atomicity.
    async fn replace_kind(
        &self,
        kind: &KnowledgeKind,
        entries: Vec<KnowledgeEntry>,
    ) -> Result<(), KnowledgeError>;

    /// Record that entries were accessed (bumps `access_count`, sets `last_accessed`).
    async fn record_access(&self, ids: &[KnowledgeId]) -> Result<(), KnowledgeError>;
}

/// Knowledge store errors.
#[derive(Debug, thiserror::Error)]
pub enum KnowledgeError {
    #[error("knowledge entry `{id}` not found")]
    NotFound { id: KnowledgeId },

    #[error("storage I/O error: {0}")]
    Io(String),

    #[error("serialization error: {0}")]
    Serialization(String),

    #[error("database error: {0}")]
    Database(String),
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn test_knowledge_kind_requires_consent() {
        assert!(KnowledgeKind::SystemConfig.requires_consent());
        assert!(KnowledgeKind::ToolCapability.requires_consent());
        assert!(KnowledgeKind::UserPreference.requires_consent());
        assert!(!KnowledgeKind::LearnedPattern.requires_consent());
        assert!(!KnowledgeKind::ErrorResolution.requires_consent());
        assert!(!KnowledgeKind::SessionInsight.requires_consent());
    }

    #[test]
    fn test_knowledge_kind_agent_owned_inverse_of_requires_consent() {
        for kind in [
            KnowledgeKind::SystemConfig,
            KnowledgeKind::ToolCapability,
            KnowledgeKind::UserPreference,
            KnowledgeKind::LearnedPattern,
            KnowledgeKind::ErrorResolution,
            KnowledgeKind::SessionInsight,
        ] {
            assert_eq!(kind.agent_owned(), !kind.requires_consent());
        }
    }

    #[test]
    fn test_knowledge_kind_serde_roundtrip() {
        for (kind, expected) in [
            (KnowledgeKind::SystemConfig, "\"system_config\""),
            (KnowledgeKind::ToolCapability, "\"tool_capability\""),
            (KnowledgeKind::UserPreference, "\"user_preference\""),
            (KnowledgeKind::LearnedPattern, "\"learned_pattern\""),
            (KnowledgeKind::ErrorResolution, "\"error_resolution\""),
            (KnowledgeKind::SessionInsight, "\"session_insight\""),
        ] {
            let json = serde_json::to_string(&kind).unwrap();
            assert_eq!(json, expected);
            let back: KnowledgeKind = serde_json::from_str(&json).unwrap();
            assert_eq!(back, kind);
        }
    }

    #[test]
    fn test_knowledge_source_serde_roundtrip() {
        for (source, expected) in [
            (KnowledgeSource::System, "\"system\""),
            (KnowledgeSource::User, "\"user\""),
            (KnowledgeSource::Agent, "\"agent\""),
        ] {
            let json = serde_json::to_string(&source).unwrap();
            assert_eq!(json, expected);
            let back: KnowledgeSource = serde_json::from_str(&json).unwrap();
            assert_eq!(back, source);
        }
    }

    #[test]
    fn test_knowledge_kind_ordering() {
        assert!(KnowledgeKind::SystemConfig < KnowledgeKind::ToolCapability);
        assert!(KnowledgeKind::ToolCapability < KnowledgeKind::UserPreference);
        assert!(KnowledgeKind::UserPreference < KnowledgeKind::LearnedPattern);
    }

    #[test]
    fn test_knowledge_error_display() {
        let err = KnowledgeError::NotFound {
            id: KnowledgeId::from_string("test-id"),
        };
        assert_eq!(err.to_string(), "knowledge entry `test-id` not found");

        let err = KnowledgeError::Database("connection failed".into());
        assert_eq!(err.to_string(), "database error: connection failed");
    }
}
