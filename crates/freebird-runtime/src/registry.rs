//! Provider registry with failover support.
//!
//! `ProviderRegistry` manages multiple [`Provider`] instances and routes
//! completion requests through a configurable failover chain. When the
//! primary provider fails with a transient error, the registry tries the
//! next provider in the chain. Non-retriable errors short-circuit
//! immediately.

use std::collections::HashMap;
use std::pin::Pin;

use freebird_traits::id::ProviderId;
use freebird_traits::provider::{
    CompletionRequest, CompletionResponse, Provider, ProviderError, StreamEvent,
};
use futures::Stream;

/// Manages registered providers and provides failover semantics for
/// completion requests.
///
/// Constructed via `&mut self` methods at startup. During operation, only
/// `&self` methods are callable — Rust ownership enforces immutability
/// once the registry is moved into the `AgentRuntime`.
pub struct ProviderRegistry {
    providers: HashMap<ProviderId, Box<dyn Provider>>,
    failover_chain: Vec<ProviderId>,
}

impl ProviderRegistry {
    /// Create an empty registry with no providers.
    #[must_use]
    pub fn new() -> Self {
        Self {
            providers: HashMap::new(),
            failover_chain: Vec::new(),
        }
    }

    /// Register a provider under the given ID.
    ///
    /// If a provider was already registered under this ID, it is replaced
    /// and the old provider is returned (follows `HashMap::insert` semantics).
    pub fn register(
        &mut self,
        id: ProviderId,
        provider: Box<dyn Provider>,
    ) -> Option<Box<dyn Provider>> {
        self.providers.insert(id, provider)
    }

    /// Set the failover chain — an ordered list of provider IDs to try.
    ///
    /// Logs `tracing::warn!` for any ID in the chain not currently registered.
    /// Does not error — the chain may be set before all providers are registered.
    pub fn set_failover_chain(&mut self, chain: Vec<ProviderId>) {
        for id in &chain {
            if !self.providers.contains_key(id) {
                tracing::warn!(
                    provider_id = %id,
                    "provider in failover chain but not registered"
                );
            }
        }
        self.failover_chain = chain;
    }

    /// Look up a specific provider by ID.
    #[must_use]
    pub fn get(&self, id: &ProviderId) -> Option<&dyn Provider> {
        self.providers.get(id).map(AsRef::as_ref)
    }

    /// Return all registered provider IDs.
    #[must_use]
    pub fn provider_ids(&self) -> Vec<&ProviderId> {
        self.providers.keys().collect()
    }

    /// Return the number of registered providers.
    #[must_use]
    pub fn provider_count(&self) -> usize {
        self.providers.len()
    }

    /// Return true if no providers are registered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.providers.is_empty()
    }

    /// Look up a model by ID across all registered providers.
    ///
    /// Scans every provider's `supported_models` list and returns the first
    /// match. Returns `None` if no provider advertises this model.
    #[must_use]
    pub fn get_model_info(
        &self,
        model_id: &freebird_traits::id::ModelId,
    ) -> Option<&freebird_traits::provider::ModelInfo> {
        self.providers.values().find_map(|provider| {
            provider
                .info()
                .supported_models
                .iter()
                .find(|m| m.id == *model_id)
        })
    }

    /// Return `true` if any provider in the failover chain advertises streaming support.
    ///
    /// Used as a pre-check before attempting the streaming path — avoids the
    /// overhead of `stream_with_failover()` when no provider supports it.
    #[must_use]
    pub fn any_in_chain_supports_streaming(&self) -> bool {
        use freebird_traits::provider::ProviderFeature;
        self.failover_chain.iter().any(|id| {
            self.providers
                .get(id)
                .is_some_and(|p| p.info().supports(&ProviderFeature::Streaming))
        })
    }

    /// Try the failover chain in order. Returns the responding provider's ID
    /// alongside the response.
    ///
    /// The request is cloned before each attempt.
    ///
    /// # Errors
    ///
    /// Returns `ProviderError::NotConfigured` if the failover chain is empty or
    /// all chain IDs are unregistered. Returns the last retriable error if all
    /// providers fail with retriable errors. Non-retriable errors short-circuit
    /// immediately.
    ///
    /// # Failover behavior
    ///
    /// - Retriable errors (`RateLimited`, `Network`, `ApiError` with status >= 500):
    ///   log warn, try next provider
    /// - Non-retriable errors (`AuthenticationFailed`, `ModelNotFound`,
    ///   `ContextOverflow`, `ApiError` with status < 500, `Deserialization`,
    ///   `NotConfigured`): return error immediately (short-circuit)
    /// - Empty failover chain: return `ProviderError::NotConfigured`
    /// - All providers exhausted: return the last error
    /// - Unregistered ID in chain: skip with warn, try next
    pub async fn complete_with_failover(
        &self,
        request: CompletionRequest,
    ) -> Result<(ProviderId, CompletionResponse), ProviderError> {
        if self.failover_chain.is_empty() {
            return Err(ProviderError::NotConfigured);
        }

        let mut last_error = ProviderError::NotConfigured;

        for provider_id in &self.failover_chain {
            let Some(provider) = self.providers.get(provider_id) else {
                tracing::warn!(
                    provider_id = %provider_id,
                    "provider in failover chain but not registered, skipping"
                );
                continue;
            };

            match provider.complete(request.clone()).await {
                Ok(response) => return Ok((provider_id.clone(), response)),
                Err(e) if is_retriable(&e) => {
                    tracing::warn!(
                        provider_id = %provider_id,
                        error = %e,
                        "provider failed with retriable error, trying next"
                    );
                    last_error = e;
                }
                Err(e) => return Err(e),
            }
        }

        Err(last_error)
    }

    /// Try the failover chain for stream setup. Returns the responding
    /// provider's ID alongside the stream.
    ///
    /// Failover applies only to stream *setup* — once a stream is established,
    /// mid-stream errors are handled by the caller (reconnecting mid-stream is
    /// not meaningful). Uses the same retriability classification as
    /// `complete_with_failover`.
    ///
    /// # Errors
    ///
    /// Returns `ProviderError::NotConfigured` if the failover chain is empty or
    /// all chain IDs are unregistered. Returns the last retriable error if all
    /// providers fail with retriable errors. Non-retriable errors short-circuit
    /// immediately.
    pub async fn stream_with_failover(
        &self,
        request: CompletionRequest,
    ) -> Result<
        (
            ProviderId,
            Pin<Box<dyn Stream<Item = Result<StreamEvent, ProviderError>> + Send>>,
        ),
        ProviderError,
    > {
        if self.failover_chain.is_empty() {
            return Err(ProviderError::NotConfigured);
        }

        let mut last_error = ProviderError::NotConfigured;

        for provider_id in &self.failover_chain {
            let Some(provider) = self.providers.get(provider_id) else {
                tracing::warn!(
                    provider_id = %provider_id,
                    "provider in failover chain but not registered, skipping"
                );
                continue;
            };

            match provider.stream(request.clone()).await {
                Ok(stream) => return Ok((provider_id.clone(), stream)),
                Err(e) if is_retriable(&e) => {
                    tracing::warn!(
                        provider_id = %provider_id,
                        error = %e,
                        "stream setup failed with retriable error, trying next"
                    );
                    last_error = e;
                }
                Err(e) => return Err(e),
            }
        }

        Err(last_error)
    }
}

impl Default for ProviderRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Determine whether a `ProviderError` is retriable (another provider
/// might succeed where this one failed).
///
/// This is a private helper because retry policy is a runtime concern —
/// different consumers might classify differently.
const fn is_retriable(error: &ProviderError) -> bool {
    match error {
        // Transient: another provider may have capacity / different infra
        ProviderError::RateLimited { .. } | ProviderError::Network { .. } => true,
        // Server error: provider is broken, not our request
        ProviderError::ApiError { status, .. } => *status >= 500,
        // Non-retriable: our request is wrong or creds are bad
        ProviderError::AuthenticationFailed { .. }
        | ProviderError::ModelNotFound { .. }
        | ProviderError::ContextOverflow { .. }
        | ProviderError::Deserialization(_)
        | ProviderError::NotConfigured => false,
    }
}

#[cfg(test)]
#[allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::panic,
    clippy::similar_names,
    clippy::items_after_statements
)]
mod tests {
    use super::*;

    use std::collections::BTreeSet;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use async_trait::async_trait;
    use chrono::Utc;
    use freebird_traits::id::ModelId;
    use freebird_traits::provider::{
        Message, ModelInfo, ProviderFeature, ProviderInfo, StopReason, TokenUsage,
    };

    // ── Mock infrastructure ─────────────────────────────────────────

    /// A factory function that produces a `Result<CompletionResponse, ProviderError>`
    /// on each call. This avoids the need for `Clone` on `ProviderError`.
    type ResultFactory = Box<dyn Fn() -> Result<CompletionResponse, ProviderError> + Send + Sync>;

    struct MockProvider {
        info: ProviderInfo,
        factory: ResultFactory,
        call_count: AtomicUsize,
    }

    impl MockProvider {
        fn new_ok(text: &str) -> Self {
            let text = text.to_owned();
            Self {
                info: make_provider_info("mock"),
                factory: Box::new(move || Ok(make_response(&text))),
                call_count: AtomicUsize::new(0),
            }
        }

        fn new_err(make_error: impl Fn() -> ProviderError + Send + Sync + 'static) -> Self {
            Self {
                info: make_provider_info("mock"),
                factory: Box::new(move || Err(make_error())),
                call_count: AtomicUsize::new(0),
            }
        }

        fn call_count(&self) -> usize {
            self.call_count.load(Ordering::Relaxed)
        }
    }

    #[async_trait]
    impl Provider for MockProvider {
        fn info(&self) -> &ProviderInfo {
            &self.info
        }

        async fn validate_credentials(&self) -> Result<(), ProviderError> {
            Ok(())
        }

        async fn complete(
            &self,
            _request: CompletionRequest,
        ) -> Result<CompletionResponse, ProviderError> {
            self.call_count.fetch_add(1, Ordering::Relaxed);
            (self.factory)()
        }

        async fn stream(
            &self,
            _request: CompletionRequest,
        ) -> Result<
            Pin<Box<dyn Stream<Item = Result<StreamEvent, ProviderError>> + Send>>,
            ProviderError,
        > {
            Ok(Box::pin(futures::stream::empty()))
        }
    }

    fn make_provider_info(name: &str) -> ProviderInfo {
        ProviderInfo {
            id: ProviderId::from_string(name),
            display_name: name.to_owned(),
            supported_models: vec![ModelInfo {
                id: ModelId::from_string("test-model"),
                display_name: "Test Model".to_owned(),
                max_context_tokens: 4096,
                max_output_tokens: 1024,
            }],
            features: BTreeSet::from([ProviderFeature::Streaming]),
        }
    }

    fn make_request() -> CompletionRequest {
        use freebird_traits::provider::{ContentBlock, Role};
        CompletionRequest {
            model: ModelId::from_string("test-model"),
            system_prompt: None,
            messages: vec![Message {
                role: Role::User,
                content: vec![ContentBlock::Text {
                    text: "hello".to_owned(),
                }],
                timestamp: Utc::now(),
            }],
            tools: vec![],
            max_tokens: 100,
            temperature: None,
            stop_sequences: vec![],
        }
    }

    fn make_response(text: &str) -> CompletionResponse {
        use freebird_traits::provider::{ContentBlock, Role};
        CompletionResponse {
            message: Message {
                role: Role::Assistant,
                content: vec![ContentBlock::Text {
                    text: text.to_owned(),
                }],
                timestamp: Utc::now(),
            },
            stop_reason: StopReason::EndTurn,
            usage: TokenUsage::default(),
            model: ModelId::from_string("test-model"),
        }
    }

    fn id(s: &str) -> ProviderId {
        ProviderId::from_string(s)
    }

    // ── Registry CRUD tests ─────────────────────────────────────────

    #[test]
    fn test_new_is_empty() {
        let registry = ProviderRegistry::new();
        assert!(registry.is_empty());
        assert_eq!(registry.provider_count(), 0);
        assert!(registry.provider_ids().is_empty());
    }

    #[test]
    fn test_register_and_get() {
        let mut registry = ProviderRegistry::new();
        let provider_id = id("anthropic");
        registry.register(provider_id.clone(), Box::new(MockProvider::new_ok("hi")));

        assert!(registry.get(&provider_id).is_some());
        assert_eq!(registry.provider_count(), 1);
        assert!(!registry.is_empty());
    }

    #[test]
    fn test_register_returns_old_on_duplicate() {
        let mut registry = ProviderRegistry::new();
        let provider_id = id("anthropic");

        let first = registry.register(provider_id.clone(), Box::new(MockProvider::new_ok("v1")));
        assert!(first.is_none());

        let second = registry.register(provider_id, Box::new(MockProvider::new_ok("v2")));
        assert!(second.is_some());
        assert_eq!(registry.provider_count(), 1);
    }

    #[test]
    fn test_provider_ids_returns_all() {
        let mut registry = ProviderRegistry::new();
        let ids = ["alpha", "beta", "gamma"];
        for name in &ids {
            registry.register(id(name), Box::new(MockProvider::new_ok("ok")));
        }

        let registered: Vec<String> = registry
            .provider_ids()
            .into_iter()
            .map(|p| p.as_str().to_owned())
            .collect();

        for name in &ids {
            assert!(
                registered.iter().any(|r| r == *name),
                "expected {name} in registered IDs"
            );
        }
    }

    #[test]
    fn test_get_unknown_returns_none() {
        let registry = ProviderRegistry::new();
        assert!(registry.get(&id("nonexistent")).is_none());
    }

    // ── Shared Arc wrapper for call-count tracking after move ──────

    /// Wraps an `Arc<MockProvider>` so it can be used as `Box<dyn Provider>`
    /// while retaining access to `call_count()` via the cloned `Arc`.
    struct ArcProvider(std::sync::Arc<MockProvider>);

    #[async_trait]
    impl Provider for ArcProvider {
        fn info(&self) -> &ProviderInfo {
            self.0.info()
        }
        async fn validate_credentials(&self) -> Result<(), ProviderError> {
            self.0.validate_credentials().await
        }
        async fn complete(
            &self,
            request: CompletionRequest,
        ) -> Result<CompletionResponse, ProviderError> {
            self.0.complete(request).await
        }
        async fn stream(
            &self,
            request: CompletionRequest,
        ) -> Result<
            Pin<Box<dyn Stream<Item = Result<StreamEvent, ProviderError>> + Send>>,
            ProviderError,
        > {
            self.0.stream(request).await
        }
    }

    // ── Failover logic tests ────────────────────────────────────────

    #[tokio::test]
    async fn test_complete_success_returns_provider_id() {
        let mut registry = ProviderRegistry::new();
        let pid = id("primary");
        registry.register(pid.clone(), Box::new(MockProvider::new_ok("response")));
        registry.set_failover_chain(vec![pid.clone()]);

        let (returned_id, response) = registry
            .complete_with_failover(make_request())
            .await
            .expect("should succeed");

        assert_eq!(returned_id, pid);
        assert_eq!(response.stop_reason, StopReason::EndTurn);
    }

    #[tokio::test]
    async fn test_failover_on_rate_limit() {
        let mut registry = ProviderRegistry::new();
        let first = id("first");
        let second = id("second");

        registry.register(
            first.clone(),
            Box::new(MockProvider::new_err(|| ProviderError::RateLimited {
                retry_after_ms: 1000,
            })),
        );
        registry.register(
            second.clone(),
            Box::new(MockProvider::new_ok("from-second")),
        );
        registry.set_failover_chain(vec![first, second.clone()]);

        let (returned_id, _) = registry
            .complete_with_failover(make_request())
            .await
            .expect("should succeed via failover");

        assert_eq!(returned_id, second);
    }

    #[tokio::test]
    async fn test_failover_on_network_error() {
        let mut registry = ProviderRegistry::new();
        let first = id("first");
        let second = id("second");

        registry.register(
            first.clone(),
            Box::new(MockProvider::new_err(|| ProviderError::Network {
                reason: "connection reset".into(),
                kind: freebird_traits::provider::NetworkErrorKind::Other,
                status_code: None,
            })),
        );
        registry.register(
            second.clone(),
            Box::new(MockProvider::new_ok("from-second")),
        );
        registry.set_failover_chain(vec![first, second.clone()]);

        let (returned_id, _) = registry
            .complete_with_failover(make_request())
            .await
            .expect("should succeed via failover");

        assert_eq!(returned_id, second);
    }

    #[tokio::test]
    async fn test_failover_on_server_error_5xx() {
        let mut registry = ProviderRegistry::new();
        let first = id("first");
        let second = id("second");

        registry.register(
            first.clone(),
            Box::new(MockProvider::new_err(|| ProviderError::ApiError {
                status: 502,
                body: "bad gateway".into(),
            })),
        );
        registry.register(
            second.clone(),
            Box::new(MockProvider::new_ok("from-second")),
        );
        registry.set_failover_chain(vec![first, second.clone()]);

        let (returned_id, _) = registry
            .complete_with_failover(make_request())
            .await
            .expect("should succeed via failover");

        assert_eq!(returned_id, second);
    }

    /// Helper: assert that the given error factory short-circuits failover
    /// (the second provider is never called) and the returned error matches
    /// the `$pattern`.
    async fn assert_short_circuits(
        error_factory: impl Fn() -> ProviderError + Send + Sync + 'static,
    ) -> ProviderError {
        let mut registry = ProviderRegistry::new();
        let first = id("first");
        let second = id("second");

        let p2 = std::sync::Arc::new(MockProvider::new_ok("should-not-reach"));
        let p2_ref = p2.clone();

        registry.register(
            first.clone(),
            Box::new(MockProvider::new_err(error_factory)),
        );
        registry.register(second.clone(), Box::new(ArcProvider(p2)));
        registry.set_failover_chain(vec![first, second]);

        let result = registry.complete_with_failover(make_request()).await;
        assert!(result.is_err(), "expected error, got {result:?}");
        assert_eq!(
            p2_ref.call_count(),
            0,
            "second provider should not be called"
        );
        result.unwrap_err()
    }

    #[tokio::test]
    async fn test_no_failover_on_client_error_4xx() {
        let err = assert_short_circuits(|| ProviderError::ApiError {
            status: 400,
            body: "bad request".into(),
        })
        .await;
        assert!(matches!(err, ProviderError::ApiError { status: 400, .. }));
    }

    #[tokio::test]
    async fn test_auth_failure_short_circuits() {
        let err = assert_short_circuits(|| ProviderError::AuthenticationFailed {
            reason: "invalid key".into(),
        })
        .await;
        assert!(matches!(err, ProviderError::AuthenticationFailed { .. }));
    }

    #[tokio::test]
    async fn test_model_not_found_short_circuits() {
        let err = assert_short_circuits(|| ProviderError::ModelNotFound {
            model: "nonexistent".into(),
        })
        .await;
        assert!(matches!(err, ProviderError::ModelNotFound { .. }));
    }

    #[tokio::test]
    async fn test_context_overflow_short_circuits() {
        let err = assert_short_circuits(|| ProviderError::ContextOverflow {
            used: 200_000,
            max: 100_000,
        })
        .await;
        assert!(matches!(err, ProviderError::ContextOverflow { .. }));
    }

    #[tokio::test]
    async fn test_deserialization_short_circuits() {
        let err = assert_short_circuits(|| ProviderError::Deserialization("bad json".into())).await;
        assert!(matches!(err, ProviderError::Deserialization(_)));
    }

    #[tokio::test]
    async fn test_all_providers_fail_returns_last_error() {
        let mut registry = ProviderRegistry::new();
        let first = id("first");
        let second = id("second");

        registry.register(
            first.clone(),
            Box::new(MockProvider::new_err(|| ProviderError::RateLimited {
                retry_after_ms: 500,
            })),
        );
        registry.register(
            second.clone(),
            Box::new(MockProvider::new_err(|| ProviderError::Network {
                reason: "timeout".into(),
                kind: freebird_traits::provider::NetworkErrorKind::Timeout,
                status_code: None,
            })),
        );
        registry.set_failover_chain(vec![first, second]);

        let result = registry.complete_with_failover(make_request()).await;
        assert!(
            matches!(result, Err(ProviderError::Network { .. })),
            "expected last error (Network), got {result:?}"
        );
    }

    #[tokio::test]
    async fn test_empty_chain_returns_not_configured() {
        let registry = ProviderRegistry::new();
        let result = registry.complete_with_failover(make_request()).await;
        assert!(matches!(result, Err(ProviderError::NotConfigured)));
    }

    #[tokio::test]
    async fn test_failover_respects_chain_order() {
        let mut registry = ProviderRegistry::new();
        let a = id("a");
        let b = id("b");
        let c = id("c");

        let pa = std::sync::Arc::new(MockProvider::new_err(|| ProviderError::RateLimited {
            retry_after_ms: 100,
        }));
        let pb = std::sync::Arc::new(MockProvider::new_err(|| ProviderError::Network {
            reason: "down".into(),
            kind: freebird_traits::provider::NetworkErrorKind::ConnectionRefused,
            status_code: None,
        }));
        let pc = std::sync::Arc::new(MockProvider::new_ok("from-c"));

        let pa_ref = pa.clone();
        let pb_ref = pb.clone();
        let pc_ref = pc.clone();

        registry.register(a.clone(), Box::new(ArcProvider(pa)));
        registry.register(b.clone(), Box::new(ArcProvider(pb)));
        registry.register(c.clone(), Box::new(ArcProvider(pc)));
        registry.set_failover_chain(vec![a, b, c.clone()]);

        let (returned_id, _) = registry
            .complete_with_failover(make_request())
            .await
            .expect("should succeed on third provider");

        assert_eq!(returned_id, c);
        assert_eq!(pa_ref.call_count(), 1);
        assert_eq!(pb_ref.call_count(), 1);
        assert_eq!(pc_ref.call_count(), 1);
    }

    #[tokio::test]
    async fn test_unregistered_id_in_chain_skipped() {
        let mut registry = ProviderRegistry::new();
        let valid = id("valid");
        registry.register(valid.clone(), Box::new(MockProvider::new_ok("ok")));
        registry.set_failover_chain(vec![id("ghost"), valid.clone()]);

        let (returned_id, _) = registry
            .complete_with_failover(make_request())
            .await
            .expect("should skip unregistered and succeed on valid");

        assert_eq!(returned_id, valid);
    }

    #[tokio::test]
    async fn test_all_chain_ids_unregistered() {
        let mut registry = ProviderRegistry::new();
        registry.register(id("exists"), Box::new(MockProvider::new_ok("ok")));
        registry.set_failover_chain(vec![id("ghost1"), id("ghost2")]);

        let result = registry.complete_with_failover(make_request()).await;
        assert!(matches!(result, Err(ProviderError::NotConfigured)));
    }

    // ── Stream failover tests ────────────────────────────────────────

    /// Mock provider that fails on stream setup with a configurable error.
    struct StreamFailProvider {
        info: ProviderInfo,
        stream_error: Box<dyn Fn() -> ProviderError + Send + Sync>,
    }

    #[async_trait]
    impl Provider for StreamFailProvider {
        fn info(&self) -> &ProviderInfo {
            &self.info
        }
        async fn validate_credentials(&self) -> Result<(), ProviderError> {
            Ok(())
        }
        async fn complete(
            &self,
            _request: CompletionRequest,
        ) -> Result<CompletionResponse, ProviderError> {
            Ok(make_response("complete-fallback"))
        }
        async fn stream(
            &self,
            _request: CompletionRequest,
        ) -> Result<
            Pin<Box<dyn Stream<Item = Result<StreamEvent, ProviderError>> + Send>>,
            ProviderError,
        > {
            Err((self.stream_error)())
        }
    }

    #[tokio::test]
    async fn test_stream_with_failover_success() {
        let mut registry = ProviderRegistry::new();
        let pid = id("streamer");
        registry.register(pid.clone(), Box::new(MockProvider::new_ok("ok")));
        registry.set_failover_chain(vec![pid.clone()]);

        let result = registry.stream_with_failover(make_request()).await;
        assert!(result.is_ok());
        let (returned_id, _stream) = result.expect("should succeed");
        assert_eq!(returned_id, pid);
    }

    #[tokio::test]
    async fn test_stream_with_failover_empty_chain() {
        let registry = ProviderRegistry::new();
        let result = registry.stream_with_failover(make_request()).await;
        assert!(matches!(result, Err(ProviderError::NotConfigured)));
    }

    #[tokio::test]
    async fn test_stream_with_failover_retries_on_retriable_error() {
        let mut registry = ProviderRegistry::new();
        let first = id("first");
        let second = id("second");

        registry.register(
            first.clone(),
            Box::new(StreamFailProvider {
                info: make_provider_info("first"),
                stream_error: Box::new(|| ProviderError::RateLimited {
                    retry_after_ms: 100,
                }),
            }),
        );
        registry.register(second.clone(), Box::new(MockProvider::new_ok("ok")));
        registry.set_failover_chain(vec![first, second.clone()]);

        let (returned_id, _stream) = registry
            .stream_with_failover(make_request())
            .await
            .expect("should succeed via failover");
        assert_eq!(returned_id, second);
    }

    #[tokio::test]
    async fn test_stream_with_failover_short_circuits_on_auth_failure() {
        let mut registry = ProviderRegistry::new();
        let first = id("first");
        let second = id("second");

        registry.register(
            first.clone(),
            Box::new(StreamFailProvider {
                info: make_provider_info("first"),
                stream_error: Box::new(|| ProviderError::AuthenticationFailed {
                    reason: "bad key".into(),
                }),
            }),
        );
        registry.register(second.clone(), Box::new(MockProvider::new_ok("ok")));
        registry.set_failover_chain(vec![first, second]);

        let result = registry.stream_with_failover(make_request()).await;
        assert!(matches!(
            result,
            Err(ProviderError::AuthenticationFailed { .. })
        ));
    }

    #[tokio::test]
    async fn test_stream_with_failover_all_fail_returns_last_error() {
        let mut registry = ProviderRegistry::new();
        let first = id("first");
        let second = id("second");

        registry.register(
            first.clone(),
            Box::new(StreamFailProvider {
                info: make_provider_info("first"),
                stream_error: Box::new(|| ProviderError::RateLimited {
                    retry_after_ms: 100,
                }),
            }),
        );
        registry.register(
            second.clone(),
            Box::new(StreamFailProvider {
                info: make_provider_info("second"),
                stream_error: Box::new(|| ProviderError::Network {
                    reason: "timeout".into(),
                    kind: freebird_traits::provider::NetworkErrorKind::Timeout,
                    status_code: None,
                }),
            }),
        );
        registry.set_failover_chain(vec![first, second]);

        let Err(err) = registry.stream_with_failover(make_request()).await else {
            panic!("expected error from all-fail stream failover");
        };
        assert!(
            matches!(err, ProviderError::Network { .. }),
            "expected last error (Network), got {err:?}"
        );
    }

    // ── any_in_chain_supports_streaming tests ────────────────────────

    #[test]
    fn test_any_in_chain_supports_streaming_true() {
        let mut registry = ProviderRegistry::new();
        let pid = id("streaming");
        registry.register(pid.clone(), Box::new(MockProvider::new_ok("ok")));
        registry.set_failover_chain(vec![pid]);
        assert!(registry.any_in_chain_supports_streaming());
    }

    #[test]
    fn test_any_in_chain_supports_streaming_empty_chain() {
        let registry = ProviderRegistry::new();
        assert!(!registry.any_in_chain_supports_streaming());
    }

    #[test]
    fn test_any_in_chain_supports_streaming_no_streaming_provider() {
        use freebird_traits::provider::ProviderFeature;
        let mut registry = ProviderRegistry::new();
        let pid = id("non-streaming");

        // Create a provider with no streaming feature
        let mut info = make_provider_info("non-streaming");
        info.features = BTreeSet::from([ProviderFeature::ToolUse]);

        struct NoStreamProvider(ProviderInfo);
        #[async_trait]
        impl Provider for NoStreamProvider {
            fn info(&self) -> &ProviderInfo {
                &self.0
            }
            async fn validate_credentials(&self) -> Result<(), ProviderError> {
                Ok(())
            }
            async fn complete(
                &self,
                _request: CompletionRequest,
            ) -> Result<CompletionResponse, ProviderError> {
                Ok(make_response("ok"))
            }
            async fn stream(
                &self,
                _request: CompletionRequest,
            ) -> Result<
                Pin<Box<dyn Stream<Item = Result<StreamEvent, ProviderError>> + Send>>,
                ProviderError,
            > {
                Err(ProviderError::NotConfigured)
            }
        }

        registry.register(pid.clone(), Box::new(NoStreamProvider(info)));
        registry.set_failover_chain(vec![pid]);
        assert!(!registry.any_in_chain_supports_streaming());
    }

    // ── Edge case tests ─────────────────────────────────────────────

    #[test]
    fn test_set_failover_chain_does_not_panic_on_unknown() {
        let mut registry = ProviderRegistry::new();
        // Should not panic, just warn
        registry.set_failover_chain(vec![id("unknown1"), id("unknown2")]);
    }

    #[tokio::test]
    async fn test_complete_clones_request() {
        // Verify that both providers receive valid requests (clone works correctly)
        let mut registry = ProviderRegistry::new();
        let first = id("first");
        let second = id("second");

        struct CallTracker {
            info: ProviderInfo,
            call_count: AtomicUsize,
            should_fail: bool,
        }

        #[async_trait]
        impl Provider for CallTracker {
            fn info(&self) -> &ProviderInfo {
                &self.info
            }
            async fn validate_credentials(&self) -> Result<(), ProviderError> {
                Ok(())
            }
            async fn complete(
                &self,
                request: CompletionRequest,
            ) -> Result<CompletionResponse, ProviderError> {
                self.call_count.fetch_add(1, Ordering::Relaxed);
                // Verify the request is valid (not consumed/moved)
                assert!(!request.messages.is_empty(), "request should have messages");
                if self.should_fail {
                    Err(ProviderError::RateLimited {
                        retry_after_ms: 100,
                    })
                } else {
                    Ok(make_response("cloned-ok"))
                }
            }
            async fn stream(
                &self,
                _request: CompletionRequest,
            ) -> Result<
                Pin<Box<dyn Stream<Item = Result<StreamEvent, ProviderError>> + Send>>,
                ProviderError,
            > {
                Ok(Box::pin(futures::stream::empty()))
            }
        }

        registry.register(
            first.clone(),
            Box::new(CallTracker {
                info: make_provider_info("first"),
                call_count: AtomicUsize::new(0),
                should_fail: true,
            }),
        );
        registry.register(
            second.clone(),
            Box::new(CallTracker {
                info: make_provider_info("second"),
                call_count: AtomicUsize::new(0),
                should_fail: false,
            }),
        );
        registry.set_failover_chain(vec![first, second]);

        let result = registry.complete_with_failover(make_request()).await;
        assert!(
            result.is_ok(),
            "second provider should succeed with cloned request"
        );
    }

    // ── is_retriable unit tests ─────────────────────────────────────

    #[test]
    fn test_is_retriable_classification() {
        // Retriable
        assert!(is_retriable(&ProviderError::RateLimited {
            retry_after_ms: 100
        }));
        assert!(is_retriable(&ProviderError::Network {
            reason: "timeout".into(),
            kind: freebird_traits::provider::NetworkErrorKind::Timeout,
            status_code: None,
        }));
        assert!(is_retriable(&ProviderError::ApiError {
            status: 500,
            body: "internal".into(),
        }));
        assert!(is_retriable(&ProviderError::ApiError {
            status: 502,
            body: "bad gateway".into(),
        }));
        assert!(is_retriable(&ProviderError::ApiError {
            status: 503,
            body: "unavailable".into(),
        }));

        // Non-retriable
        assert!(!is_retriable(&ProviderError::AuthenticationFailed {
            reason: "bad key".into(),
        }));
        assert!(!is_retriable(&ProviderError::ModelNotFound {
            model: "x".into(),
        }));
        assert!(!is_retriable(&ProviderError::ContextOverflow {
            used: 200_000,
            max: 100_000,
        }));
        assert!(!is_retriable(&ProviderError::ApiError {
            status: 400,
            body: "bad request".into(),
        }));
        assert!(!is_retriable(&ProviderError::ApiError {
            status: 403,
            body: "forbidden".into(),
        }));
        assert!(!is_retriable(&ProviderError::Deserialization(
            "bad json".into()
        )));
        assert!(!is_retriable(&ProviderError::NotConfigured));
    }
}
