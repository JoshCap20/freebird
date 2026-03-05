//! Provider registry construction.
//!
//! Builds and populates the [`ProviderRegistry`] from application configuration,
//! including credential loading from environment variables and startup validation.

use std::collections::HashSet;

use anyhow::{Context, Result};
use secrecy::SecretString;

use freebird_providers::anthropic::{AnthropicConfig, AnthropicProvider};
use freebird_runtime::registry::ProviderRegistry;
use freebird_traits::id::ProviderId;
use freebird_traits::provider::Provider;
use freebird_types::config::{AppConfig, ProviderKind};

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
            ProviderKind::Anthropic => {
                let api_key = std::env::var("ANTHROPIC_API_KEY")
                    .map(SecretString::from)
                    .map_err(|e| match e {
                        std::env::VarError::NotPresent => anyhow::anyhow!(
                            "ANTHROPIC_API_KEY environment variable not set \
                             (required for provider `{}`)",
                            provider_config.id,
                        ),
                        std::env::VarError::NotUnicode(_) => anyhow::anyhow!(
                            "ANTHROPIC_API_KEY environment variable contains invalid UTF-8 \
                             (required for provider `{}`)",
                            provider_config.id,
                        ),
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

                provider.validate_credentials().await.with_context(|| {
                    format!(
                        "credential validation failed for provider `{}`",
                        provider_config.id,
                    )
                })?;

                tracing::info!(provider = %provider_config.id, "credentials validated");

                registry.register(provider_config.id.clone(), Box::new(provider));
            }
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
