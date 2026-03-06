//! Provider registry construction.
//!
//! Builds and populates the [`ProviderRegistry`] from application configuration,
//! including credential loading from environment variables and startup validation.

use std::collections::HashSet;

use anyhow::{Context, Result, bail};
use secrecy::SecretString;

use freebird_providers::anthropic::{AnthropicConfig, AnthropicProvider};
use freebird_runtime::registry::ProviderRegistry;
use freebird_traits::id::ProviderId;
use freebird_traits::provider::Provider;
use freebird_types::config::{AppConfig, ProviderConfig, ProviderKind};

/// Build and populate the provider registry from configuration.
///
/// # Environment variables
///
/// - `ANTHROPIC_API_KEY`: Required when an Anthropic provider is configured.
///   (Deviates from CLAUDE.md §23's `OPENCLAW_PROVIDERS__0__API_KEY` convention —
///   we use the vendor-standard name for developer ergonomics.)
pub async fn build_provider_registry(config: &AppConfig) -> Result<ProviderRegistry> {
    let mut registry = ProviderRegistry::new();

    for provider_config in &config.providers {
        match provider_config.kind {
            ProviderKind::Anthropic => match build_anthropic_provider(provider_config).await {
                Ok(provider) => {
                    registry.register(provider_config.id.clone(), Box::new(provider));
                }
                Err(e) => {
                    tracing::warn!(
                        provider = %provider_config.id,
                        error = %e,
                        "failed to initialize provider, skipping"
                    );
                }
            },
            ProviderKind::OpenAi => {
                tracing::warn!(
                    provider = %provider_config.id,
                    "OpenAI provider not yet implemented, skipping"
                );
            }
            ProviderKind::Ollama => {
                tracing::warn!(
                    provider = %provider_config.id,
                    "Ollama provider not yet implemented, skipping"
                );
            }
        }
    }

    // Fail fast if no providers were successfully initialized. Without this,
    // the daemon would accept TCP connections but fail on every message with
    // a confusing error from `complete_with_failover()`.
    if registry.provider_ids().is_empty() {
        bail!(
            "no providers were successfully initialized \
             (check ANTHROPIC_API_KEY and provider configuration)"
        );
    }

    // Build failover chain from config order, but only include providers
    // that were actually registered (unimplemented providers are skipped above).
    let registered: HashSet<ProviderId> = registry.provider_ids().into_iter().cloned().collect();
    let failover_chain = config
        .providers
        .iter()
        .map(|p| p.id.clone())
        .filter(|id| registered.contains(id))
        .collect();
    registry.set_failover_chain(failover_chain);

    Ok(registry)
}

/// Build a single Anthropic provider from config, loading credentials from env.
async fn build_anthropic_provider(provider_config: &ProviderConfig) -> Result<AnthropicProvider> {
    let api_key = std::env::var("ANTHROPIC_API_KEY")
        .map(|s| SecretString::from(s.trim().to_owned()))
        .map_err(|e| match e {
            std::env::VarError::NotPresent => {
                anyhow::anyhow!("ANTHROPIC_API_KEY environment variable not set")
            }
            std::env::VarError::NotUnicode(_) => {
                anyhow::anyhow!("ANTHROPIC_API_KEY contains invalid UTF-8")
            }
        })?;

    let anthropic_config = AnthropicConfig {
        base_url: provider_config.base_url.clone(),
        default_model: provider_config
            .default_model
            .as_ref()
            .map(|m| m.as_str().to_owned()),
    };

    let provider_id = provider_config.id.clone();
    let provider = AnthropicProvider::new(api_key, anthropic_config)
        .with_context(|| format!("failed to create provider `{provider_id}`"))?;

    match provider.validate_credentials().await {
        Ok(()) => tracing::info!(provider = %provider_config.id, "credentials validated"),
        Err(e) => tracing::warn!(
            provider = %provider_config.id,
            error = %e,
            "credential validation failed — provider registered but may fail at request time"
        ),
    }

    Ok(provider)
}
