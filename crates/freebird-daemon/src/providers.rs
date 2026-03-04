//! Provider registry construction.
//!
//! Builds and populates the [`ProviderRegistry`] from application configuration,
//! including credential loading from environment variables and startup validation.

use anyhow::{Context, Result};
use secrecy::SecretString;

use freebird_providers::anthropic::{AnthropicConfig, AnthropicProvider};
use freebird_runtime::registry::ProviderRegistry;
use freebird_traits::id::ProviderId;
use freebird_traits::provider::Provider;
use freebird_types::config::{AppConfig, ProviderKind};

/// Build and populate the provider registry from configuration.
pub async fn build_provider_registry(config: &AppConfig) -> Result<ProviderRegistry> {
    let mut registry = ProviderRegistry::new();

    for provider_config in &config.providers {
        match provider_config.kind {
            ProviderKind::Anthropic => {
                let api_key = std::env::var("ANTHROPIC_API_KEY")
                    .map(SecretString::from)
                    .map_err(|_| {
                        anyhow::anyhow!(
                            "ANTHROPIC_API_KEY environment variable not set \
                             (required for provider `{}`)",
                            provider_config.id,
                        )
                    })?;

                let anthropic_config = AnthropicConfig {
                    base_url: provider_config.base_url.clone(),
                    default_model: provider_config.default_model.clone(),
                };

                let provider = AnthropicProvider::new(api_key, anthropic_config).context(
                    format!("failed to create provider `{}`", provider_config.id),
                )?;

                provider.validate_credentials().await.context(format!(
                    "credential validation failed for provider `{}`",
                    provider_config.id,
                ))?;

                tracing::info!(provider = %provider_config.id, "credentials validated");

                registry.register(
                    ProviderId::from_string(provider_config.id.clone()),
                    Box::new(provider),
                );
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

    let failover_chain = config
        .providers
        .iter()
        .map(|p| ProviderId::from_string(p.id.clone()))
        .collect();
    registry.set_failover_chain(failover_chain);

    Ok(registry)
}
