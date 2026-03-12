//! `FreebirdBuilder` — wires all subsystems into a ready-to-run `FreebirdApp`.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Context as _;

use freebird_channels::tcp::TcpChannel;
use freebird_runtime::agent::AgentRuntime;
use freebird_types::config::AppConfig;

use crate::database::{DatabaseComponents, PassphraseStrategy};
use crate::error::CoreError;
use crate::{FreebirdApp, config, database, knowledge, providers, tools, util};

/// Strategy for constructing the channel.
pub enum ChannelStrategy {
    /// Build a `TcpChannel` from `config.daemon` (production default).
    Tcp,
    /// Use a pre-built channel (for tests or alternative transports).
    Custom(Box<dyn freebird_traits::channel::Channel>),
}

/// Builder for constructing a fully-wired `FreebirdApp`.
///
/// # Example
///
/// ```rust,no_run
/// # use freebird_core::{FreebirdBuilder, FreebirdApp};
/// # async fn example() -> Result<(), freebird_core::CoreError> {
/// let config = freebird_core::config::load_config().map_err(|e| freebird_core::CoreError::Config(e.to_string()))?;
/// let app = FreebirdBuilder::from_config(config)
///     .build()
///     .await?;
/// # Ok(())
/// # }
/// ```
pub struct FreebirdBuilder {
    config: AppConfig,
    passphrase: PassphraseStrategy,
    channel: ChannelStrategy,
    extra_allow_dirs: Vec<PathBuf>,
}

impl FreebirdBuilder {
    /// Start from an `AppConfig`. Uses production defaults for passphrase
    /// resolution and TCP channel.
    #[allow(clippy::missing_const_for_fn)] // Vec::new() in const context is nightly-only
    pub fn from_config(config: AppConfig) -> Self {
        Self {
            config,
            passphrase: PassphraseStrategy::AutoResolve { allow_prompt: true },
            channel: ChannelStrategy::Tcp,
            extra_allow_dirs: Vec::new(),
        }
    }

    /// Override how the DB passphrase is resolved.
    ///
    /// Default: `AutoResolve { allow_prompt: true }`.
    #[must_use]
    pub fn passphrase_strategy(mut self, strategy: PassphraseStrategy) -> Self {
        self.passphrase = strategy;
        self
    }

    /// Override the channel. Default: `Tcp` (from `config.daemon`).
    #[must_use]
    pub fn channel(mut self, strategy: ChannelStrategy) -> Self {
        self.channel = strategy;
        self
    }

    /// Add extra allowed directories (replaces CLI `--allow-dir`).
    #[must_use]
    pub fn allow_dirs(mut self, dirs: Vec<PathBuf>) -> Self {
        self.extra_allow_dirs = dirs;
        self
    }

    /// Build the fully-wired system.
    ///
    /// This is async because provider credential validation and knowledge
    /// population hit the network/DB.
    ///
    /// # Errors
    ///
    /// Returns `CoreError` if any subsystem fails to initialize.
    #[allow(clippy::too_many_lines)] // composition root — naturally long
    pub async fn build(self) -> Result<FreebirdApp, CoreError> {
        // 1. VALIDATE + ENFORCE
        config::validate_config(&self.config).map_err(|e| CoreError::Config(e.to_string()))?;
        let config = config::enforce_security_minimums(self.config);

        // 2. DATABASE
        let DatabaseComponents {
            memory,
            knowledge_store,
            event_sink,
            audit_sink,
            db: _db,
        } = database::init_database(&config, &self.passphrase).map_err(CoreError::Database)?;

        // 3. VERIFY AUDIT CHAIN
        if let Some(ref sink) = audit_sink {
            sink.verify_chain()
                .await
                .map_err(CoreError::AuditIntegrity)?;
            tracing::info!("audit chain integrity verified");
        }

        // 4. EMIT DaemonStarted audit event
        if let Some(ref sink) = audit_sink {
            let event = freebird_security::audit::AuditEventType::DaemonStarted {
                version: env!("CARGO_PKG_VERSION").to_string(),
            };
            freebird_runtime::agent::emit_audit(sink.as_ref(), None, event).await;
        }

        // 5. PROVIDER REGISTRY
        let registry = providers::build_provider_registry(&config)
            .await
            .map_err(CoreError::Provider)?;

        // 6. CHANNEL
        let channel: Box<dyn freebird_traits::channel::Channel> = match self.channel {
            ChannelStrategy::Tcp => Box::new(TcpChannel::new(
                config.daemon.host.to_string(),
                config.daemon.port,
            )),
            ChannelStrategy::Custom(ch) => ch,
        };

        // 7. TOOLS
        let tool_registry = tools::build_tool_registry(&config)
            .context("failed to build tool registry")
            .map_err(CoreError::ToolRegistry)?;

        // 7b. BOOTSTRAP KNOWLEDGE
        if let Some(ref store) = knowledge_store {
            knowledge::populate_system_knowledge(store.as_ref(), &tool_registry, &config)
                .await
                .map_err(CoreError::Knowledge)?;
        }

        // 8. SANDBOX + ALLOW DIRS
        let mut tools_config = config.tools;
        tools_config.sandbox_root = util::expand_tilde(&tools_config.sandbox_root)
            .map_err(|e| CoreError::Config(e.to_string()))?;
        tokio::fs::create_dir_all(&tools_config.sandbox_root)
            .await
            .with_context(|| {
                format!(
                    "failed to create sandbox directory `{}`",
                    tools_config.sandbox_root.display()
                )
            })
            .map_err(CoreError::Database)?;
        util::merge_allow_dirs(&mut tools_config, self.extra_allow_dirs)
            .map_err(|e| CoreError::Config(e.to_string()))?;

        // 9. APPROVAL GATE
        let (approval_gate, approval_rx) = freebird_security::approval::ApprovalGate::new(
            config.security.require_consent_above.clone(),
            Duration::from_secs(config.security.consent_timeout_secs),
            config.security.max_pending_consent_requests,
        );
        tracing::info!(
            threshold = ?config.security.require_consent_above,
            timeout_secs = config.security.consent_timeout_secs,
            max_pending = config.security.max_pending_consent_requests,
            "approval gate configured"
        );

        // Clone knowledge store for runtime (ToolExecutor also needs its own Arc).
        let ks_for_runtime = knowledge_store.clone();

        // 10. SECRET GUARD
        let secret_guard = if config.security.secret_guard.enabled {
            Some(
                freebird_security::secret_guard::SecretGuard::from_config(
                    &config.security.secret_guard,
                )
                .context("failed to construct SecretGuard from config")
                .map_err(CoreError::Security)?,
            )
        } else {
            tracing::info!("secret guard disabled via config");
            None
        };

        // 11. TOOL EXECUTOR
        let tool_executor = freebird_runtime::tool_executor::ToolExecutor::new(
            tool_registry.into_tools(),
            Duration::from_secs(tools_config.default_timeout_secs),
            audit_sink.clone(),
            tools_config.allowed_directories.clone(),
            Some(approval_gate),
            knowledge_store,
            Some(Arc::clone(&memory)),
            secret_guard,
            config.security.injection.clone(),
        )
        .context("failed to construct ToolExecutor (duplicate tool names?)")
        .map_err(CoreError::ToolRegistry)?;

        // 12. AGENT RUNTIME
        let runtime = AgentRuntime::new(
            registry,
            channel,
            tool_executor,
            Some(approval_rx),
            memory.clone(),
            ks_for_runtime,
            config.knowledge,
            config.runtime,
            tools_config,
            config.security.budgets,
            config.security.default_session_ttl_hours,
            event_sink,
            audit_sink.clone(),
        );

        Ok(FreebirdApp {
            runtime,
            audit_sink,
            memory,
        })
    }
}
