//! Knowledge retrieval and context building helpers for `AgentRuntime`.

use freebird_traits::knowledge::KnowledgeMatch;

use super::AgentRuntime;

impl AgentRuntime {
    /// Retrieve relevant knowledge entries for auto-injection into the prompt.
    ///
    /// Returns an empty vec when auto-retrieval is disabled, no knowledge store
    /// is configured, or no entries pass the relevance threshold.
    pub(crate) async fn retrieve_knowledge_context(&self, query: &str) -> Vec<KnowledgeMatch> {
        if !self.knowledge_config.auto_retrieve {
            return Vec::new();
        }

        let Some(ref store) = self.knowledge_store else {
            return Vec::new();
        };

        let matches = match store
            .search(query, self.knowledge_config.max_context_entries)
            .await
        {
            Ok(m) => m,
            Err(e) => {
                tracing::warn!(error = %e, "knowledge auto-retrieval failed");
                return Vec::new();
            }
        };

        // BM25: lower (more negative) = more relevant. Keep entries at or below threshold.
        let filtered: Vec<KnowledgeMatch> = matches
            .into_iter()
            .filter(|m| m.rank <= self.knowledge_config.relevance_threshold)
            .collect();

        if filtered.is_empty() {
            return Vec::new();
        }

        // Record access for analytics (fire-and-forget).
        let ids: Vec<_> = filtered.iter().map(|m| m.entry.id.clone()).collect();
        if let Err(e) = store.record_access(&ids).await {
            tracing::warn!(error = %e, "failed to record knowledge access");
        }

        tracing::debug!(
            count = filtered.len(),
            "injecting knowledge context into prompt"
        );

        filtered
    }

    /// Format knowledge matches into a context block for injection into the prompt.
    ///
    /// Returns `None` if there are no matches. Respects the configured
    /// `max_context_tokens` budget (estimated at ~4 chars per token).
    pub(crate) fn format_knowledge_context(&self, matches: &[KnowledgeMatch]) -> Option<String> {
        if matches.is_empty() {
            return None;
        }

        let token_budget_chars = self.knowledge_config.max_context_tokens.saturating_mul(4);
        let mut buf = String::with_capacity(token_budget_chars.min(8192));
        buf.push_str("[RELEVANT CONTEXT]\n");
        let mut remaining = token_budget_chars.saturating_sub(buf.len());

        for m in matches {
            let label = format!("[{:?}] ", m.entry.kind);
            let entry_len = label.len() + m.entry.content.len() + 1; // +1 for newline

            if entry_len > remaining {
                // Fit as much of this entry as possible, then stop.
                let avail = remaining.saturating_sub(label.len() + 1);
                if avail > 0 {
                    buf.push_str(&label);
                    // Truncate at a char boundary.
                    let truncated: String = m.entry.content.chars().take(avail).collect();
                    buf.push_str(&truncated);
                    buf.push('\n');
                }
                break;
            }

            buf.push_str(&label);
            buf.push_str(&m.entry.content);
            buf.push('\n');
            remaining -= entry_len;
        }

        Some(buf)
    }
}
