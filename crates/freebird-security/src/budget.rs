//! Token budget enforcement for per-session and per-request limits (CLAUDE.md §13 — ASI08).
//!
//! Prevents unbounded token/compute consumption by enforcing hard limits on:
//! - Tokens per individual provider request
//! - Tokens per session (cumulative)
//! - Tool rounds per agentic turn

use std::fmt;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

use freebird_traits::provider::TokenUsage;
use freebird_types::config::BudgetConfig;

use crate::error::SecurityError;

/// Identifies which budget resource was exceeded.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BudgetResource {
    /// Per-request token limit.
    TokensPerRequest,
    /// Cumulative per-session token limit.
    TokensPerSession,
    /// Tool rounds per agentic turn.
    ToolRoundsPerTurn,
}

impl fmt::Display for BudgetResource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TokensPerRequest => write!(f, "tokens_per_request"),
            Self::TokensPerSession => write!(f, "tokens_per_session"),
            Self::ToolRoundsPerTurn => write!(f, "tool_rounds_per_turn"),
        }
    }
}

/// Enforces token and tool-round budgets for a single agent session.
///
/// Thread-safe via atomics — multiple tasks can record usage and modify
/// limits concurrently. Only input and output tokens are counted; cache
/// tokens are excluded.
///
/// Limits can be raised or disabled at runtime (e.g., after user approves
/// a budget override) via the `set_*` methods.
pub struct TokenBudget {
    max_tokens_per_session: AtomicU64,
    max_tokens_per_request: AtomicU64,
    max_tool_rounds_per_turn: AtomicU32,
    tokens_used: AtomicU64,
}

impl TokenBudget {
    /// Create a new budget from typed configuration.
    #[must_use]
    pub const fn new(config: &BudgetConfig) -> Self {
        Self {
            max_tokens_per_session: AtomicU64::new(config.max_tokens_per_session),
            max_tokens_per_request: AtomicU64::new(config.max_tokens_per_request),
            max_tool_rounds_per_turn: AtomicU32::new(config.max_tool_rounds_per_turn),
            tokens_used: AtomicU64::new(0),
        }
    }

    /// Record token usage from a provider response.
    ///
    /// Checks per-request limit first, then per-session limit. Only
    /// `input_tokens` and `output_tokens` are counted — cache tokens are
    /// excluded to avoid double-counting cached content.
    ///
    /// On session limit exceeded, the atomic counter is rolled back so
    /// the budget remains in a consistent state.
    ///
    /// # Errors
    ///
    /// Returns `SecurityError::BudgetExceeded` if either limit is exceeded.
    pub fn record_usage(&self, usage: &TokenUsage) -> Result<(), SecurityError> {
        let request_tokens = u64::from(usage.input_tokens) + u64::from(usage.output_tokens);

        // Check per-request limit first.
        let per_request = self.max_tokens_per_request.load(Ordering::Relaxed);
        if request_tokens > per_request {
            return Err(SecurityError::BudgetExceeded {
                resource: BudgetResource::TokensPerRequest,
                used: request_tokens,
                limit: per_request,
            });
        }

        // Atomically add to session total.
        let previous = self
            .tokens_used
            .fetch_add(request_tokens, Ordering::Relaxed);
        let new_total = previous + request_tokens;

        // Check per-session limit — rollback on exceeded.
        let per_session = self.max_tokens_per_session.load(Ordering::Relaxed);
        if new_total > per_session {
            self.tokens_used
                .fetch_sub(request_tokens, Ordering::Relaxed);
            return Err(SecurityError::BudgetExceeded {
                resource: BudgetResource::TokensPerSession,
                used: new_total,
                limit: per_session,
            });
        }

        Ok(())
    }

    /// Check whether the current tool round is within budget.
    ///
    /// `current_round` is zero-indexed: round 0 is the first iteration.
    /// A limit of 3 allows rounds 0, 1, 2 and rejects round 3+.
    ///
    /// # Errors
    ///
    /// Returns `SecurityError::BudgetExceeded` if `current_round` is at or
    /// beyond the configured maximum.
    pub fn check_tool_rounds(&self, current_round: u32) -> Result<(), SecurityError> {
        let max_rounds = self.max_tool_rounds_per_turn.load(Ordering::Relaxed);
        if current_round >= max_rounds {
            return Err(SecurityError::BudgetExceeded {
                resource: BudgetResource::ToolRoundsPerTurn,
                used: u64::from(current_round),
                limit: u64::from(max_rounds),
            });
        }
        Ok(())
    }

    /// Returns the number of tokens remaining before the session limit.
    #[must_use]
    pub fn remaining_tokens(&self) -> u64 {
        self.max_tokens_per_session
            .load(Ordering::Relaxed)
            .saturating_sub(self.tokens_used.load(Ordering::Relaxed))
    }

    /// Returns the total tokens consumed in this session so far.
    #[must_use]
    pub fn tokens_used(&self) -> u64 {
        self.tokens_used.load(Ordering::Relaxed)
    }

    /// Returns the configured maximum tool rounds per turn.
    #[must_use]
    pub fn max_tool_rounds(&self) -> u32 {
        self.max_tool_rounds_per_turn.load(Ordering::Relaxed)
    }

    /// Returns the current per-request token limit.
    #[must_use]
    pub fn max_tokens_per_request(&self) -> u64 {
        self.max_tokens_per_request.load(Ordering::Relaxed)
    }

    /// Returns the current per-session token limit.
    #[must_use]
    pub fn max_tokens_per_session(&self) -> u64 {
        self.max_tokens_per_session.load(Ordering::Relaxed)
    }

    /// Update the per-request token limit at runtime.
    ///
    /// Used after the user approves a budget override (raise or disable).
    pub fn set_max_tokens_per_request(&self, new_limit: u64) {
        self.max_tokens_per_request
            .store(new_limit, Ordering::Relaxed);
    }

    /// Update the per-session token limit at runtime.
    ///
    /// Used after the user approves a budget override (raise or disable).
    pub fn set_max_tokens_per_session(&self, new_limit: u64) {
        self.max_tokens_per_session
            .store(new_limit, Ordering::Relaxed);
    }

    /// Update the maximum tool rounds per turn at runtime.
    ///
    /// Used after the user approves a budget override (raise or disable).
    pub fn set_max_tool_rounds_per_turn(&self, new_limit: u32) {
        self.max_tool_rounds_per_turn
            .store(new_limit, Ordering::Relaxed);
    }

    /// Record token usage unconditionally, bypassing per-request and
    /// per-session limits. Used after the user explicitly approves a
    /// budget-exceeded approval request.
    pub fn force_record_usage(&self, usage: &TokenUsage) {
        let request_tokens = u64::from(usage.input_tokens) + u64::from(usage.output_tokens);
        self.tokens_used
            .fetch_add(request_tokens, Ordering::Relaxed);
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;

    fn default_config() -> BudgetConfig {
        BudgetConfig::default()
    }

    fn small_config() -> BudgetConfig {
        BudgetConfig {
            max_tokens_per_session: 1000,
            max_tokens_per_request: 200,
            max_tool_rounds_per_turn: 3,
        }
    }

    fn usage(input: u32, output: u32) -> TokenUsage {
        TokenUsage {
            input_tokens: input,
            output_tokens: output,
            cache_read_tokens: None,
            cache_creation_tokens: None,
        }
    }

    fn usage_with_cache(
        input: u32,
        output: u32,
        cache_read: u32,
        cache_creation: u32,
    ) -> TokenUsage {
        TokenUsage {
            input_tokens: input,
            output_tokens: output,
            cache_read_tokens: Some(cache_read),
            cache_creation_tokens: Some(cache_creation),
        }
    }

    // ── Per-request limit tests ──────────────────────────────────

    #[test]
    fn record_usage_within_per_request_limit() {
        let budget = TokenBudget::new(&small_config());
        let result = budget.record_usage(&usage(100, 50));
        assert!(result.is_ok());
        assert_eq!(budget.tokens_used(), 150);
    }

    #[test]
    fn record_usage_at_per_request_limit() {
        let budget = TokenBudget::new(&small_config());
        let result = budget.record_usage(&usage(100, 100));
        assert!(result.is_ok());
        assert_eq!(budget.tokens_used(), 200);
    }

    #[test]
    fn record_usage_exceeds_per_request_limit() {
        let budget = TokenBudget::new(&small_config());
        let result = budget.record_usage(&usage(150, 100));
        assert!(result.is_err());
        let err = result.unwrap_err();
        match &err {
            SecurityError::BudgetExceeded {
                resource,
                used,
                limit,
            } => {
                assert_eq!(*resource, BudgetResource::TokensPerRequest);
                assert_eq!(*used, 250);
                assert_eq!(*limit, 200);
            }
            other => panic!("expected BudgetExceeded, got: {other:?}"),
        }
        // Session counter should not be incremented on per-request failure.
        assert_eq!(budget.tokens_used(), 0);
    }

    // ── Per-session limit tests ──────────────────────────────────

    #[test]
    fn record_usage_cumulative_within_session_limit() {
        let budget = TokenBudget::new(&small_config());
        budget.record_usage(&usage(50, 50)).unwrap();
        budget.record_usage(&usage(50, 50)).unwrap();
        assert_eq!(budget.tokens_used(), 200);
    }

    #[test]
    fn record_usage_at_session_limit() {
        let budget = TokenBudget::new(&small_config());
        // 5 requests of 200 each = 1000 = max_tokens_per_session
        for _ in 0..5 {
            budget.record_usage(&usage(100, 100)).unwrap();
        }
        assert_eq!(budget.tokens_used(), 1000);
    }

    #[test]
    fn record_usage_exceeds_session_limit() {
        let budget = TokenBudget::new(&small_config());
        // Use 900 tokens first
        for _ in 0..9 {
            budget.record_usage(&usage(50, 50)).unwrap();
        }
        assert_eq!(budget.tokens_used(), 900);

        // This 200-token request would push to 1100, exceeding 1000
        let result = budget.record_usage(&usage(100, 100));
        assert!(result.is_err());
        match result.unwrap_err() {
            SecurityError::BudgetExceeded {
                resource,
                used,
                limit,
            } => {
                assert_eq!(resource, BudgetResource::TokensPerSession);
                assert_eq!(used, 1100);
                assert_eq!(limit, 1000);
            }
            other => panic!("expected BudgetExceeded, got: {other:?}"),
        }
        // Counter should be rolled back.
        assert_eq!(budget.tokens_used(), 900);
    }

    #[test]
    fn record_usage_rollback_allows_subsequent_smaller_request() {
        let budget = TokenBudget::new(&small_config());
        // Use 900 tokens
        for _ in 0..9 {
            budget.record_usage(&usage(50, 50)).unwrap();
        }
        // Exceed with 200 tokens
        assert!(budget.record_usage(&usage(100, 100)).is_err());
        assert_eq!(budget.tokens_used(), 900);
        // Smaller request of 100 should fit (900 + 100 = 1000)
        assert!(budget.record_usage(&usage(50, 50)).is_ok());
        assert_eq!(budget.tokens_used(), 1000);
    }

    // ── Cache tokens not counted ─────────────────────────────────

    #[test]
    fn cache_read_tokens_not_counted_in_request() {
        let budget = TokenBudget::new(&small_config());
        let result = budget.record_usage(&usage_with_cache(50, 50, 10_000, 0));
        assert!(result.is_ok());
        assert_eq!(budget.tokens_used(), 100);
    }

    #[test]
    fn cache_creation_tokens_not_counted_in_request() {
        let budget = TokenBudget::new(&small_config());
        let result = budget.record_usage(&usage_with_cache(50, 50, 0, 5_000));
        assert!(result.is_ok());
        assert_eq!(budget.tokens_used(), 100);
    }

    #[test]
    fn cache_tokens_not_counted_in_session_cumulative() {
        let budget = TokenBudget::new(&small_config());
        // With cache tokens that would blow the budget if counted
        for _ in 0..5 {
            budget
                .record_usage(&usage_with_cache(100, 100, 50_000, 50_000))
                .unwrap();
        }
        // Only input+output = 200 * 5 = 1000
        assert_eq!(budget.tokens_used(), 1000);
    }

    // ── Tool rounds tests ────────────────────────────────────────

    #[test]
    fn check_tool_rounds_within_limit() {
        // max_tool_rounds_per_turn = 3 means rounds 0, 1, 2 are valid.
        let budget = TokenBudget::new(&small_config());
        assert!(budget.check_tool_rounds(0).is_ok());
        assert!(budget.check_tool_rounds(1).is_ok());
        assert!(budget.check_tool_rounds(2).is_ok());
    }

    #[test]
    fn check_tool_rounds_at_limit_boundary() {
        // Round 3 (0-indexed) is the 4th attempt — exceeds a limit of 3.
        let budget = TokenBudget::new(&small_config());
        assert!(budget.check_tool_rounds(3).is_err());
    }

    #[test]
    fn check_tool_rounds_exceeds_limit() {
        let budget = TokenBudget::new(&small_config());
        let result = budget.check_tool_rounds(4);
        assert!(result.is_err());
        match result.unwrap_err() {
            SecurityError::BudgetExceeded {
                resource,
                used,
                limit,
            } => {
                assert_eq!(resource, BudgetResource::ToolRoundsPerTurn);
                assert_eq!(used, 4);
                assert_eq!(limit, 3);
            }
            other => panic!("expected BudgetExceeded, got: {other:?}"),
        }
    }

    #[test]
    fn check_tool_rounds_zero_is_valid() {
        let budget = TokenBudget::new(&small_config());
        assert!(budget.check_tool_rounds(0).is_ok());
    }

    #[test]
    fn check_tool_rounds_zero_budget_rejects_all() {
        let config = BudgetConfig {
            max_tool_rounds_per_turn: 0,
            ..small_config()
        };
        let budget = TokenBudget::new(&config);
        // A limit of 0 means no rounds are allowed.
        assert!(budget.check_tool_rounds(0).is_err());
    }

    // ── Query methods ────────────────────────────────────────────

    #[test]
    fn remaining_tokens_starts_at_max() {
        let budget = TokenBudget::new(&small_config());
        assert_eq!(budget.remaining_tokens(), 1000);
    }

    #[test]
    fn remaining_tokens_decreases_after_usage() {
        let budget = TokenBudget::new(&small_config());
        budget.record_usage(&usage(50, 50)).unwrap();
        assert_eq!(budget.remaining_tokens(), 900);
    }

    #[test]
    fn tokens_used_starts_at_zero() {
        let budget = TokenBudget::new(&small_config());
        assert_eq!(budget.tokens_used(), 0);
    }

    #[test]
    fn max_tool_rounds_returns_configured_value() {
        let budget = TokenBudget::new(&small_config());
        assert_eq!(budget.max_tool_rounds(), 3);
    }

    // ── BudgetConfig defaults and serde ──────────────────────────

    #[test]
    fn budget_config_defaults() {
        let config = BudgetConfig::default();
        assert_eq!(config.max_tokens_per_session, 500_000);
        assert_eq!(config.max_tokens_per_request, 32_768);
        assert_eq!(config.max_tool_rounds_per_turn, 10);
    }

    #[test]
    fn budget_config_serde_roundtrip() {
        let config = BudgetConfig {
            max_tokens_per_session: 100_000,
            max_tokens_per_request: 16_384,
            max_tool_rounds_per_turn: 5,
        };
        let json = serde_json::to_string(&config).unwrap();
        let back: BudgetConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(back.max_tokens_per_session, 100_000);
        assert_eq!(back.max_tokens_per_request, 16_384);
        assert_eq!(back.max_tool_rounds_per_turn, 5);
    }

    #[test]
    fn default_config_creates_budget_with_correct_limits() {
        let config = default_config();
        let budget = TokenBudget::new(&config);
        assert_eq!(budget.remaining_tokens(), 500_000);
        assert_eq!(budget.max_tool_rounds(), 10);
    }

    // ── BudgetResource display ───────────────────────────────────

    #[test]
    fn budget_resource_display_snake_case() {
        assert_eq!(
            BudgetResource::TokensPerRequest.to_string(),
            "tokens_per_request"
        );
        assert_eq!(
            BudgetResource::TokensPerSession.to_string(),
            "tokens_per_session"
        );
        assert_eq!(
            BudgetResource::ToolRoundsPerTurn.to_string(),
            "tool_rounds_per_turn"
        );
    }

    // ── Error display format ─────────────────────────────────────

    #[test]
    fn budget_exceeded_error_display_format() {
        let err = SecurityError::BudgetExceeded {
            resource: BudgetResource::TokensPerSession,
            used: 600_000,
            limit: 500_000,
        };
        assert_eq!(
            err.to_string(),
            "budget exceeded for `tokens_per_session`: used 600000, limit 500000"
        );
    }

    // ── force_record_usage tests ──────────────────────────────────

    #[test]
    fn force_record_usage_bypasses_per_request_limit() {
        let budget = TokenBudget::new(&small_config());
        // 250 exceeds per-request limit of 200 — normal record would fail.
        let big_usage = usage(150, 100);
        assert!(budget.record_usage(&big_usage).is_err());
        assert_eq!(budget.tokens_used(), 0);

        // Force bypasses the limit.
        budget.force_record_usage(&big_usage);
        assert_eq!(budget.tokens_used(), 250);
    }

    #[test]
    fn force_record_usage_bypasses_session_limit() {
        let budget = TokenBudget::new(&small_config());
        // Fill to 900.
        for _ in 0..9 {
            budget.record_usage(&usage(50, 50)).unwrap();
        }
        // 200 more would exceed 1000 session limit.
        let over_usage = usage(100, 100);
        assert!(budget.record_usage(&over_usage).is_err());
        assert_eq!(budget.tokens_used(), 900);

        // Force bypasses the limit.
        budget.force_record_usage(&over_usage);
        assert_eq!(budget.tokens_used(), 1100);
    }

    // ── Runtime limit mutation tests ──────────────────────────────

    #[test]
    fn set_max_tokens_per_request_raises_limit() {
        let budget = TokenBudget::new(&small_config());
        // 250 exceeds per-request limit of 200.
        assert!(budget.record_usage(&usage(150, 100)).is_err());

        // Raise the limit.
        budget.set_max_tokens_per_request(400);
        assert_eq!(budget.max_tokens_per_request(), 400);

        // Now 250 fits within the new limit.
        assert!(budget.record_usage(&usage(150, 100)).is_ok());
    }

    #[test]
    fn set_max_tokens_per_session_raises_limit() {
        let budget = TokenBudget::new(&small_config());
        // Fill to 900.
        for _ in 0..9 {
            budget.record_usage(&usage(50, 50)).unwrap();
        }
        // 200 more would exceed 1000 session limit.
        assert!(budget.record_usage(&usage(100, 100)).is_err());

        // Double the session limit.
        budget.set_max_tokens_per_session(2000);
        assert_eq!(budget.max_tokens_per_session(), 2000);

        // Now 200 more fits (900 + 200 = 1100 ≤ 2000).
        assert!(budget.record_usage(&usage(100, 100)).is_ok());
    }

    #[test]
    fn set_max_tool_rounds_per_turn_raises_limit() {
        let budget = TokenBudget::new(&small_config());
        // Round 3 exceeds limit of 3 (0-indexed).
        assert!(budget.check_tool_rounds(3).is_err());

        // Raise to 10.
        budget.set_max_tool_rounds_per_turn(10);
        assert_eq!(budget.max_tool_rounds(), 10);

        // Round 3 now allowed.
        assert!(budget.check_tool_rounds(3).is_ok());
    }

    #[test]
    fn set_max_tokens_per_request_to_max_disables_limit() {
        let budget = TokenBudget::new(&small_config());
        budget.set_max_tokens_per_request(u64::MAX);
        budget.set_max_tokens_per_session(u64::MAX);

        // Any request size should pass.
        assert!(budget.record_usage(&usage(u32::MAX, 0)).is_ok());
    }
}
