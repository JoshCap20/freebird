//! Freebird daemon — the composition root.
//!
//! This is the only crate that knows about concrete types (`AnthropicProvider`,
//! `CliChannel`, `FileMemory`). Everything else works through trait objects.
//! See CLAUDE.md §3.1: trait-driven extensibility.

#![deny(clippy::all, clippy::pedantic, clippy::nursery)]
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
#![deny(clippy::indexing_slicing)]
#![allow(clippy::module_name_repetitions)]

use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Context, Result, bail};
use figment::Figment;
use figment::providers::{Env, Format, Toml};
use secrecy::SecretString;
use tracing_subscriber::EnvFilter;

use freebird_channels::cli::CliChannel;
use freebird_memory::file::FileMemory;
use freebird_providers::anthropic::{AnthropicConfig, AnthropicProvider};
use freebird_runtime::agent::AgentRuntime;
use freebird_runtime::registry::ProviderRegistry;
use freebird_runtime::shutdown::ShutdownCoordinator;
use freebird_traits::id::ProviderId;
use freebird_traits::provider::Provider;
use freebird_types::config::{AppConfig, ChannelKind, ProviderKind};

#[tokio::main]
async fn main() -> Result<()> {
    // 1. LOGGING — before anything else, so config errors are visible.
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()))
        .init();

    tracing::info!("freebird starting");

    // 2. CONFIGURATION — TOML base, env overrides.
    let config = load_config()?;
    tracing::debug!(?config.runtime, "loaded configuration");

    // 3. VALIDATE — fail fast on invalid configuration.
    validate_config(&config)?;

    // 4. PROVIDER REGISTRY
    let registry = build_provider_registry(&config).await?;

    // 5. CHANNEL — construct with configured prompt (or default).
    let cli_config = config
        .channels
        .iter()
        .find(|c| matches!(c.kind, ChannelKind::Cli))
        .context("cli channel config missing")?;
    let channel: Box<dyn freebird_traits::channel::Channel> =
        cli_config.prompt.as_ref().map_or_else(
            || Box::new(CliChannel::new()) as Box<dyn freebird_traits::channel::Channel>,
            |prompt| Box::new(CliChannel::with_prompt(prompt)),
        );

    // 6. MEMORY
    let memory_dir = expand_tilde(
        &config
            .memory
            .base_dir
            .clone()
            .unwrap_or_else(|| PathBuf::from("~/.freebird/conversations")),
    );
    let memory = FileMemory::new(memory_dir).context("failed to initialize file memory backend")?;

    // 7. TOOLS (none initially — added per-issue later)
    let tools: Vec<Box<dyn freebird_traits::tool::Tool>> = vec![];

    // 8. SHUTDOWN COORDINATOR
    let shutdown = ShutdownCoordinator::new(Duration::from_secs(config.runtime.drain_timeout_secs));
    let token = shutdown.token();
    let drain_timeout = shutdown.drain_timeout();

    tokio::spawn(async move {
        match shutdown.wait_for_signal().await {
            Ok(signal) => tracing::info!(%signal, "shutdown initiated"),
            Err(e) => {
                tracing::error!(%e, "failed to install signal handler, triggering shutdown");
                shutdown.trigger();
            }
        }
    });

    // 9. AGENT RUNTIME
    let runtime = AgentRuntime::new(
        registry,
        channel,
        tools,
        Box::new(memory),
        config.runtime,
        config.tools,
        None, // audit logger — wired in a later issue
    );

    // 10. RUN — awaited directly (NOT raced via tokio::select! against
    //     wait_for_signal) so run() calls channel.stop() before returning.
    let run_result = runtime.run(token).await;
    match &run_result {
        Ok(()) => tracing::info!("runtime exited cleanly"),
        Err(e) => tracing::error!(%e, "runtime error"),
    }

    // 11. DRAIN — give in-flight work time to complete.
    if drain_timeout > Duration::ZERO {
        tracing::info!(?drain_timeout, "draining in-flight work");
        tokio::time::sleep(drain_timeout).await;
    }

    tracing::info!("freebird stopped");
    run_result.map_err(Into::into)
}

/// Load configuration from TOML file with environment variable overrides.
fn load_config() -> Result<AppConfig> {
    let config_path =
        std::env::var("FREEBIRD_CONFIG").unwrap_or_else(|_| "config/default.toml".into());

    Figment::new()
        .merge(Toml::file(&config_path))
        .merge(Env::prefixed("FREEBIRD_").split("__"))
        .extract()
        .context(format!("failed to load configuration from `{config_path}`"))
}

/// Validate configuration invariants that cannot be expressed in types.
fn validate_config(config: &AppConfig) -> Result<()> {
    if config.providers.is_empty() {
        bail!("at least one provider must be configured in [[providers]]");
    }

    let cli_count = config
        .channels
        .iter()
        .filter(|c| matches!(c.kind, ChannelKind::Cli))
        .count();

    if cli_count == 0 {
        bail!("at least one CLI channel must be configured in [[channels]]");
    }
    if cli_count > 1 {
        bail!("only one CLI channel is supported (found {cli_count})");
    }

    Ok(())
}

/// Build and populate the provider registry from configuration.
async fn build_provider_registry(config: &AppConfig) -> Result<ProviderRegistry> {
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

    let mut failover_chain = Vec::with_capacity(config.providers.len());
    for p in &config.providers {
        failover_chain.push(ProviderId::from_string(p.id.clone()));
    }
    registry.set_failover_chain(failover_chain);

    Ok(registry)
}

/// Expand `~` prefix to the user's home directory.
///
/// Only expands a leading `~` or `~/` — does not expand `~user`.
/// Returns the path unchanged if `~` cannot be resolved.
fn expand_tilde(path: &Path) -> PathBuf {
    let s = path.to_string_lossy();
    if s == "~" {
        home::home_dir().unwrap_or_else(|| PathBuf::from("."))
    } else if let Some(rest) = s.strip_prefix("~/") {
        home::home_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join(rest)
    } else {
        path.to_owned()
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::indexing_slicing, clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn test_expand_tilde_home() {
        let result = expand_tilde(Path::new("~"));
        // Should resolve to home dir (or "." if no home found)
        if let Some(home) = home::home_dir() {
            assert_eq!(result, home);
        } else {
            assert_eq!(result, PathBuf::from("."));
        }
    }

    #[test]
    fn test_expand_tilde_home_subpath() {
        let result = expand_tilde(Path::new("~/foo/bar"));
        if let Some(home) = home::home_dir() {
            assert_eq!(result, home.join("foo/bar"));
        } else {
            assert_eq!(result, PathBuf::from("./foo/bar"));
        }
    }

    #[test]
    fn test_expand_tilde_absolute_passthrough() {
        let result = expand_tilde(Path::new("/tmp/test"));
        assert_eq!(result, PathBuf::from("/tmp/test"));
    }

    #[test]
    fn test_expand_tilde_relative_passthrough() {
        let result = expand_tilde(Path::new("data/sessions"));
        assert_eq!(result, PathBuf::from("data/sessions"));
    }

    #[test]
    fn test_expand_tilde_tilde_user_not_expanded() {
        let result = expand_tilde(Path::new("~otheruser/path"));
        assert_eq!(result, PathBuf::from("~otheruser/path"));
    }

    #[test]
    fn test_expand_tilde_empty_path() {
        let result = expand_tilde(Path::new(""));
        assert_eq!(result, PathBuf::from(""));
    }

    // --- validate_config tests ---

    /// Build a valid `AppConfig` from TOML for test mutation.
    fn valid_config() -> AppConfig {
        use figment::providers::{Format, Toml};

        let toml = r#"
[runtime]
default_model = "test"
default_provider = "test"
max_output_tokens = 1024
max_tool_rounds = 5
max_turns_per_session = 10
drain_timeout_secs = 5

[[providers]]
id = "test"
kind = "anthropic"

[[channels]]
id = "cli"
kind = "cli"

[tools]
sandbox_root = "/tmp"
default_timeout_secs = 10

[memory]
kind = "file"

[security]
max_tool_calls_per_turn = 10
require_consent_above = "high"

[logging]
level = "info"
format = "pretty"
"#;
        Figment::new()
            .merge(Toml::string(toml))
            .extract()
            .expect("test config should deserialize")
    }

    #[test]
    fn test_validate_config_valid() {
        let config = valid_config();
        assert!(validate_config(&config).is_ok());
    }

    #[test]
    fn test_validate_config_empty_providers_errors() {
        let mut config = valid_config();
        config.providers.clear();
        let err = validate_config(&config).unwrap_err();
        assert!(
            err.to_string().contains("at least one provider"),
            "expected provider error, got: {err}"
        );
    }

    #[test]
    fn test_validate_config_no_cli_channel_errors() {
        let mut config = valid_config();
        config.channels.clear();
        let err = validate_config(&config).unwrap_err();
        assert!(
            err.to_string().contains("at least one CLI channel"),
            "expected CLI channel error, got: {err}"
        );
    }

    #[test]
    fn test_validate_config_multiple_cli_channels_errors() {
        let mut config = valid_config();
        let second_cli = config.channels[0].clone();
        config.channels.push(second_cli);
        let err = validate_config(&config).unwrap_err();
        assert!(
            err.to_string().contains("only one CLI channel"),
            "expected multi-CLI error, got: {err}"
        );
    }
}
