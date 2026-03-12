//! Freebird daemon — the composition root.
//!
//! Single binary with `clap` subcommands:
//! - `freebird serve`  — start daemon with TCP listener
//! - `freebird chat`   — connect to running daemon for interactive chat
//! - `freebird status` — check if daemon is running
//! - `freebird stop`   — send graceful shutdown to daemon
//! - `freebird replay` — replay a past session as a detailed trace

#![deny(clippy::all, clippy::pedantic, clippy::nursery)]
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
#![deny(clippy::indexing_slicing)]
#![allow(clippy::module_name_repetitions)]

use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand};
use figment::Figment;
use figment::providers::{Env, Format, Toml};
use tracing_subscriber::EnvFilter;

use secrecy::ExposeSecret as _;

use freebird_channels::tcp::TcpChannel;
use freebird_memory::sqlite::SqliteDb;
use freebird_memory::sqlite_knowledge::SqliteKnowledgeStore;
use freebird_memory::sqlite_memory::SqliteMemory;
use freebird_runtime::agent::AgentRuntime;
use freebird_runtime::shutdown::ShutdownCoordinator;
use freebird_security::secret_guard::SecretGuard;
use freebird_types::config::AppConfig;

mod providers;
mod tools;

/// `Freebird` AI agent daemon.
#[derive(Parser)]
#[command(name = "freebird", version, about)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

/// Available subcommands.
#[derive(Subcommand)]
enum Commands {
    /// Start the daemon, listen on TCP for client connections.
    Serve {
        /// Additional directories the agent may access (repeatable).
        /// Accepts absolute paths or ~ paths (e.g., ~/Documents/myproject).
        #[arg(long = "allow-dir", short = 'a')]
        allow_dirs: Vec<PathBuf>,
    },
    /// Connect to a running daemon for interactive chat.
    Chat,
    /// Check if the daemon is running (probes TCP port).
    Status,
    /// Send graceful shutdown to the daemon.
    Stop,
    /// Replay a past session as a detailed trace.
    Replay {
        /// Session ID to replay (UUID).
        session_id: Option<String>,
        /// Replay the most recent session.
        #[arg(long)]
        last: bool,
        /// Output as JSON instead of human-readable trace.
        #[arg(long)]
        json: bool,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Serve { allow_dirs } => cmd_serve(allow_dirs).await,
        Commands::Chat => cmd_chat().await,
        Commands::Status => cmd_status().await,
        Commands::Stop => cmd_stop().await,
        Commands::Replay {
            session_id,
            last,
            json,
        } => cmd_replay(session_id, last, json).await,
    }
}

/// `freebird serve` — start daemon with TCP channel.
#[allow(clippy::too_many_lines)] // composition root — naturally long
async fn cmd_serve(allow_dirs: Vec<PathBuf>) -> Result<()> {
    // 1. LOGGING — before anything else, so config errors are visible.
    // Intentional silent fallback: if RUST_LOG is absent or unparseable, default
    // to "info". We can't log the parse error because tracing isn't initialized yet.
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()))
        .init();

    tracing::info!("freebird serve starting");

    // 2. CONFIGURATION
    let config = load_config()?;
    tracing::debug!(?config.runtime, "loaded configuration");

    // 3. VALIDATE
    validate_config(&config)?;
    let config = enforce_security_minimums(config);

    // 4. PROVIDER REGISTRY
    let registry = providers::build_provider_registry(&config).await?;

    // 5. CHANNEL — TcpChannel from daemon config.
    let channel: Box<dyn freebird_traits::channel::Channel> = Box::new(TcpChannel::new(
        config.daemon.host.to_string(),
        config.daemon.port,
    ));

    // 6. MEMORY — SQLite with SQLCipher encryption
    let (memory, knowledge_store, db, _db_salt) = init_sqlite(&config)?;

    // 6a. EVENT SINK + AUDIT SINK — event-sourced persistence into the same SQLite DB
    let event_sink: Option<std::sync::Arc<dyn freebird_traits::event::EventSink>> =
        Some(std::sync::Arc::new(
            freebird_memory::sqlite_event::SqliteEventSink::new(std::sync::Arc::clone(&db)),
        ));
    let audit_sink: Option<std::sync::Arc<dyn freebird_traits::audit::AuditSink>> =
        Some(std::sync::Arc::new(
            freebird_memory::sqlite_audit::SqliteAuditSink::new(std::sync::Arc::clone(&db)),
        ));

    // 6b. VERIFY AUDIT CHAIN — detect tampering between restarts (ASI06, #111)
    if let Some(ref sink) = audit_sink {
        sink.verify_chain().await.with_context(
            || "audit chain integrity check failed — the audit log may have been tampered with",
        )?;
        tracing::info!("audit chain integrity verified");
    }

    // 6c. EMIT DaemonStarted audit event
    if let Some(ref sink) = audit_sink {
        let event = freebird_security::audit::AuditEventType::DaemonStarted {
            version: env!("CARGO_PKG_VERSION").to_string(),
        };
        emit_audit_event(sink.as_ref(), None, &event).await;
    }

    // 7. TOOLS — build registry before moving config.tools
    let tool_registry =
        tools::build_tool_registry(&config).context("failed to build tool registry")?;

    // 7b. BOOTSTRAP — populate system knowledge (tool capabilities, system config).
    // Must run before config.tools is moved below.
    if let Some(ref store) = knowledge_store {
        populate_system_knowledge(store.as_ref(), &tool_registry, &config)
            .await
            .context("failed to populate system knowledge")?;
    }

    let mut tools_config = config.tools;
    tools_config.sandbox_root = expand_tilde(&tools_config.sandbox_root)?;
    tokio::fs::create_dir_all(&tools_config.sandbox_root)
        .await
        .with_context(|| {
            format!(
                "failed to create sandbox directory `{}`",
                tools_config.sandbox_root.display()
            )
        })?;

    merge_allow_dirs(&mut tools_config, allow_dirs)?;

    // 8. SHUTDOWN COORDINATOR
    let shutdown = ShutdownCoordinator::new(Duration::from_secs(config.runtime.drain_timeout_secs));
    let token = shutdown.token();
    let drain_timeout = shutdown.drain_timeout();

    let signal_handle = tokio::spawn(async move {
        match shutdown.wait_for_signal().await {
            Ok(signal) => tracing::info!(%signal, "shutdown initiated"),
            Err(e) => {
                tracing::error!(%e, "failed to install signal handler, triggering shutdown");
                shutdown.trigger();
            }
        }
    });

    // 9. APPROVAL GATE — unified human-in-the-loop for consent + security warnings
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

    // Clone knowledge store for the runtime (ToolExecutor also needs its own Arc).
    let ks_for_runtime = knowledge_store.clone();

    // 10. SECRET GUARD — intercepts tool inputs and redacts secrets from outputs
    let secret_guard = if config.security.secret_guard.enabled {
        Some(
            SecretGuard::from_config(&config.security.secret_guard)
                .context("failed to construct SecretGuard from config")?,
        )
    } else {
        tracing::info!("secret guard disabled via config");
        None
    };

    // 11. TOOL EXECUTOR — consumes the registry, adds security pipeline (incl. secret guard)
    let tool_executor = freebird_runtime::tool_executor::ToolExecutor::new(
        tool_registry.into_tools(),
        std::time::Duration::from_secs(tools_config.default_timeout_secs),
        audit_sink.clone(),
        tools_config.allowed_directories.clone(),
        Some(approval_gate),
        knowledge_store,
        Some(std::sync::Arc::clone(&memory)),
        secret_guard,
        config.security.injection.clone(),
    )
    .context("failed to construct ToolExecutor (duplicate tool names?)")?;

    // 13. AGENT RUNTIME
    // Clone audit_sink before moving it into the runtime so we can
    // emit DaemonShutdown after the runtime exits.
    let audit_sink_for_shutdown = audit_sink.clone();
    let mut runtime = AgentRuntime::new(
        registry,
        channel,
        tool_executor,
        Some(approval_rx),
        memory,
        ks_for_runtime,
        config.knowledge,
        config.runtime,
        tools_config,
        config.security.budgets,
        config.security.default_session_ttl_hours,
        event_sink,
        audit_sink,
    );

    // 14. RUN
    let run_result = runtime.run(token).await;
    match &run_result {
        Ok(()) => tracing::info!("runtime exited cleanly"),
        Err(e) => tracing::error!(%e, "runtime error"),
    }

    // 14b. EMIT DaemonShutdown audit event
    if let Some(ref sink) = audit_sink_for_shutdown {
        let reason = match &run_result {
            Ok(()) => "clean shutdown".to_string(),
            Err(e) => format!("runtime error: {e}"),
        };
        let event = freebird_security::audit::AuditEventType::DaemonShutdown { reason };
        emit_audit_event(sink.as_ref(), None, &event).await;
    }

    // 15. DRAIN
    if drain_timeout > Duration::ZERO {
        tracing::info!(?drain_timeout, "draining in-flight work");
        let _ = tokio::time::timeout(drain_timeout, signal_handle).await;
    } else {
        signal_handle.abort();
    }

    tracing::info!("freebird stopped");
    run_result.map_err(Into::into)
}

/// Emit an audit event to the sink, handling serialization errors gracefully.
///
/// Delegates to the shared `emit_audit` helper in `freebird-runtime`.
async fn emit_audit_event(
    sink: &dyn freebird_traits::audit::AuditSink,
    session_id: Option<&str>,
    event: &freebird_security::audit::AuditEventType,
) {
    freebird_runtime::agent::emit_audit(sink, session_id, event.clone()).await;
}

/// `freebird chat` — interactive client that connects to the daemon.
///
/// Uses the rich TUI mode when connected to a real terminal, otherwise
/// falls back to plain pipe mode.
async fn cmd_chat() -> Result<()> {
    let config = load_config()?;
    let addr = format!("{}:{}", config.daemon.host, config.daemon.port);

    let stream = tokio::net::TcpStream::connect(&addr)
        .await
        .with_context(|| format!("failed to connect to daemon at {addr}"))?;

    let is_tty = std::io::IsTerminal::is_terminal(&std::io::stdout());

    if !is_tty {
        eprintln!("Connected to {addr}");
    }

    freebird_tui::chat::run_chat(stream, is_tty).await
}

/// `freebird status` — check if daemon is running by probing the TCP port.
async fn cmd_status() -> Result<()> {
    let config = load_config()?;
    let addr = format!("{}:{}", config.daemon.host, config.daemon.port);

    if tokio::net::TcpStream::connect(&addr).await.is_ok() {
        println!("freebird daemon is running at {addr}");
    } else {
        println!("freebird daemon is not running (could not connect to {addr})");
    }

    Ok(())
}

/// `freebird stop` — send graceful shutdown command to daemon.
async fn cmd_stop() -> Result<()> {
    let config = load_config()?;
    let addr = format!("{}:{}", config.daemon.host, config.daemon.port);

    let mut stream = tokio::net::TcpStream::connect(&addr)
        .await
        .with_context(|| format!("daemon is not running (could not connect to {addr})"))?;

    let msg = freebird_types::protocol::ClientMessage::Command {
        name: "shutdown".into(),
        args: vec![],
    };
    freebird_tui::chat::send_client_message(&mut stream, &msg)
        .await
        .context("sending shutdown command")?;

    println!("shutdown command sent to daemon at {addr}");
    Ok(())
}

/// `freebird replay` — display a past session as a detailed trace.
///
/// Opens the encrypted database (prompts for key if needed), resolves the
/// session via `--last` or an explicit session ID, and prints the formatted
/// replay to stdout.
async fn cmd_replay(session_id: Option<String>, last: bool, json: bool) -> Result<()> {
    let config = load_config()?;

    if session_id.is_none() && !last {
        bail!("specify a session ID or use --last to replay the most recent session");
    }

    let (memory, _knowledge, _db, _salt) = init_sqlite(&config)?;

    // Resolve session ID
    let sid = if last {
        let sessions = memory
            .list_sessions(1)
            .await
            .context("failed to list sessions")?;
        sessions
            .into_iter()
            .next()
            .map(|s| s.session_id)
            .ok_or_else(|| anyhow::anyhow!("no sessions found"))?
    } else {
        // session_id is Some(...) due to the bail above
        let raw = session_id.ok_or_else(|| anyhow::anyhow!("missing session ID"))?;
        freebird_traits::id::SessionId::from(raw.as_str())
    };

    let conversation = memory
        .load(&sid)
        .await
        .with_context(|| format!("failed to load session `{sid}`"))?
        .ok_or_else(|| anyhow::anyhow!("session `{sid}` not found"))?;

    if json {
        let output = freebird_tui::replay::format_replay_json(&conversation)
            .context("failed to serialize session")?;
        println!("{output}");
    } else {
        let output = freebird_tui::replay::format_replay(&conversation);
        print!("{output}");
    }

    Ok(())
}

/// Load configuration from TOML file with environment variable overrides.
fn load_config() -> Result<AppConfig> {
    let config_path =
        std::env::var("FREEBIRD_CONFIG").unwrap_or_else(|_| "config/default.toml".into());

    Figment::new()
        .merge(Toml::file(&config_path))
        .merge(Env::prefixed("FREEBIRD_").split("__"))
        .extract()
        .with_context(|| format!("failed to load configuration from `{config_path}`"))
}

/// Validate configuration invariants that cannot be expressed in types.
fn validate_config(config: &AppConfig) -> Result<()> {
    if config.providers.is_empty() {
        bail!("at least one provider must be configured in [[providers]]");
    }

    // Budget limits must be positive
    if config.security.budgets.max_tokens_per_session == 0 {
        bail!("security.budgets.max_tokens_per_session must be > 0");
    }
    if config.security.budgets.max_tokens_per_request == 0 {
        bail!("security.budgets.max_tokens_per_request must be > 0");
    }
    if config.security.budgets.max_tool_rounds_per_turn == 0 {
        bail!("security.budgets.max_tool_rounds_per_turn must be > 0");
    }
    if config.runtime.max_tool_rounds == 0 {
        bail!("runtime.max_tool_rounds must be > 0");
    }

    // Egress allowed_hosts must be valid domain-like strings
    for host in &config.security.egress.allowed_hosts {
        if host.is_empty() || host.contains(' ') {
            bail!(
                "security.egress.allowed_hosts contains invalid entry: `{host}` \
                 (must be non-empty and contain no spaces)"
            );
        }
    }

    // PBKDF2 iterations minimum
    if config.memory.pbkdf2_iterations < 1000 {
        bail!(
            "memory.pbkdf2_iterations must be >= 1000 (got {})",
            config.memory.pbkdf2_iterations
        );
    }

    // Session TTL must be positive
    if config.runtime.session.session_ttl_secs == 0 {
        bail!("runtime.session.session_ttl_secs must be > 0");
    }

    Ok(())
}

/// Enforce minimum security thresholds regardless of env var overrides.
///
/// Prevents misconfiguration (accidental or adversarial) from weakening
/// security below safe minimums. Values that exceed limits are clamped
/// with a warning.
fn enforce_security_minimums(mut config: AppConfig) -> AppConfig {
    use freebird_traits::tool::RiskLevel;

    // require_consent_above must be at most High (Critical would disable all consent)
    if config.security.require_consent_above > RiskLevel::High {
        tracing::warn!(
            original = ?config.security.require_consent_above,
            clamped = ?RiskLevel::High,
            "require_consent_above exceeds maximum — clamping to High"
        );
        config.security.require_consent_above = RiskLevel::High;
    }

    // max_tokens_per_session must be > 0
    if config.security.budgets.max_tokens_per_session == 0 {
        tracing::warn!("max_tokens_per_session is 0 — clamping to 1");
        config.security.budgets.max_tokens_per_session = 1;
    }

    // max_tool_rounds_per_turn must be > 0 and <= 50
    if config.security.budgets.max_tool_rounds_per_turn == 0 {
        tracing::warn!("max_tool_rounds_per_turn is 0 — clamping to 1");
        config.security.budgets.max_tool_rounds_per_turn = 1;
    }
    if config.security.budgets.max_tool_rounds_per_turn > 50 {
        tracing::warn!(
            original = config.security.budgets.max_tool_rounds_per_turn,
            clamped = 50,
            "max_tool_rounds_per_turn exceeds maximum — clamping to 50"
        );
        config.security.budgets.max_tool_rounds_per_turn = 50;
    }

    config
}

/// Result of [`init_sqlite`]: memory backend, optional knowledge store, raw DB handle, and salt.
type SqliteComponents = (
    std::sync::Arc<dyn freebird_traits::memory::Memory>,
    Option<std::sync::Arc<dyn freebird_traits::knowledge::KnowledgeStore>>,
    std::sync::Arc<freebird_memory::sqlite::SqliteDb>,
    Vec<u8>,
);

/// Initialize the `SQLite`-backed memory and knowledge store.
///
/// Opens (or creates) an encrypted `SQLCipher` database, derives the key via
/// PBKDF2-HMAC-SHA256, and returns both `SqliteMemory` and an `Arc`-wrapped
/// `SqliteKnowledgeStore` for injection into the `ToolExecutor`.
fn init_sqlite(config: &AppConfig) -> Result<SqliteComponents> {
    use std::sync::Arc;

    let db_path = config
        .memory
        .db_path
        .as_ref()
        .map(|p| expand_tilde(p))
        .transpose()?
        .unwrap_or_else(|| {
            home::home_dir().map_or_else(
                || PathBuf::from(".freebird/freebird.db"),
                |h| h.join(".freebird/freebird.db"),
            )
        });

    let salt_path = db_path.with_extension("salt");
    let salt = freebird_security::db_key::load_or_create_salt(&salt_path)
        .context("failed to load or create database salt")?;

    let passphrase = freebird_security::db_key::resolve_passphrase(
        config.memory.keyfile_path.as_deref(),
        true, // allow interactive prompt
    )
    .context("failed to resolve database encryption key")?;

    let key =
        freebird_security::db_key::derive_key(&passphrase, &salt, config.memory.pbkdf2_iterations)
            .context("failed to derive database encryption key")?;

    // Derive HMAC signing key from passphrase + salt for event and audit chain integrity.
    // Uses HMAC-SHA256 with the passphrase as key and a domain-separated salt as message,
    // ensuring the signing key cannot be reconstructed without the secret passphrase.
    let signing_tag = ring::hmac::sign(
        &ring::hmac::Key::new(
            ring::hmac::HMAC_SHA256,
            passphrase.expose_secret().as_bytes(),
        ),
        format!("freebird-event-signing|{}", hex::encode(&salt)).as_bytes(),
    );
    let signing_key = ring::hmac::Key::new(ring::hmac::HMAC_SHA256, signing_tag.as_ref());

    let db =
        SqliteDb::open(&db_path, &key, signing_key).context("failed to open encrypted database")?;
    let db = Arc::new(db);

    tracing::info!(path = %db_path.display(), "encrypted database opened");

    let memory: Arc<dyn freebird_traits::memory::Memory> = Arc::new(SqliteMemory::new(
        Arc::clone(&db),
        config.memory.verify_on_load,
    ));
    let knowledge: Arc<dyn freebird_traits::knowledge::KnowledgeStore> =
        Arc::new(SqliteKnowledgeStore::new(Arc::clone(&db)));

    Ok((memory, Some(knowledge), db, salt))
}

/// Populate the knowledge store with system-level entries on startup.
///
/// Uses `replace_kind()` for idempotent updates — safe to call on every boot.
/// Two categories are populated:
///
/// - **`ToolCapability`**: one entry per registered tool (name, description, capability, risk).
/// - **`SystemConfig`**: key runtime configuration facts.
async fn populate_system_knowledge(
    store: &dyn freebird_traits::knowledge::KnowledgeStore,
    tool_registry: &freebird_runtime::tool_registry::ToolRegistry,
    config: &AppConfig,
) -> Result<()> {
    use std::collections::BTreeSet;

    use chrono::Utc;
    use freebird_traits::id::KnowledgeId;
    use freebird_traits::knowledge::{KnowledgeEntry, KnowledgeKind, KnowledgeSource};

    let now = Utc::now();

    // --- ToolCapability entries ---
    let mut tool_entries = Vec::with_capacity(tool_registry.tool_count());
    for tool in tool_registry.iter() {
        let info = tool.info();
        let content = format!(
            "Tool: {}\nDescription: {}\nCapability: {:?}\nRisk: {:?}\nSide effects: {:?}",
            info.name,
            info.description,
            info.required_capability,
            info.risk_level,
            info.side_effects,
        );
        tool_entries.push(KnowledgeEntry {
            id: KnowledgeId::from_string(uuid::Uuid::new_v4().to_string()),
            kind: KnowledgeKind::ToolCapability,
            content,
            tags: BTreeSet::from(["tool".to_owned(), info.name.clone()]),
            source: KnowledgeSource::System,
            confidence: 1.0,
            session_id: None,
            created_at: now,
            updated_at: now,
            access_count: 0,
            last_accessed: None,
        });
    }

    store
        .replace_kind(&KnowledgeKind::ToolCapability, tool_entries)
        .await
        .context("failed to populate ToolCapability knowledge")?;

    // --- SystemConfig entries ---
    let config_facts = [
        format!(
            "Default provider: {}, default model: {}",
            config.runtime.default_provider, config.runtime.default_model,
        ),
        format!(
            "Max tool rounds per turn: {}, max output tokens: {}",
            config.runtime.max_tool_rounds, config.runtime.max_output_tokens,
        ),
        format!(
            "Consent required above: {:?}, consent timeout: {}s",
            config.security.require_consent_above, config.security.consent_timeout_secs,
        ),
        format!(
            "Knowledge auto-retrieve: {}, max context entries: {}, relevance threshold: {}",
            config.knowledge.auto_retrieve,
            config.knowledge.max_context_entries,
            config.knowledge.relevance_threshold,
        ),
    ];

    let config_entries: Vec<KnowledgeEntry> = config_facts
        .into_iter()
        .map(|content| KnowledgeEntry {
            id: KnowledgeId::from_string(uuid::Uuid::new_v4().to_string()),
            kind: KnowledgeKind::SystemConfig,
            content,
            tags: BTreeSet::from(["config".to_owned()]),
            source: KnowledgeSource::System,
            confidence: 1.0,
            session_id: None,
            created_at: now,
            updated_at: now,
            access_count: 0,
            last_accessed: None,
        })
        .collect();

    let entry_count = config_entries.len();
    store
        .replace_kind(&KnowledgeKind::SystemConfig, config_entries)
        .await
        .context("failed to populate SystemConfig knowledge")?;

    tracing::info!(
        tool_capabilities = tool_registry.tool_count(),
        system_config = entry_count,
        "system knowledge bootstrapped"
    );

    Ok(())
}

/// Merge CLI `--allow-dir` flags into the tools configuration.
fn merge_allow_dirs(
    tools_config: &mut freebird_types::config::ToolsConfig,
    allow_dirs: Vec<PathBuf>,
) -> Result<()> {
    for dir in allow_dirs {
        let expanded = expand_tilde(&dir)?;
        let canonical = expanded.canonicalize().with_context(|| {
            format!(
                "--allow-dir path `{}` does not exist or cannot be resolved",
                dir.display()
            )
        })?;
        if !tools_config.allowed_directories.contains(&canonical) {
            tracing::info!(dir = %canonical.display(), "allowing additional directory");
            tools_config.allowed_directories.push(canonical);
        }
    }
    Ok(())
}

/// Expand `~` prefix to the user's home directory.
///
/// Only expands a leading `~` or `~/` — does not expand `~user`.
fn expand_tilde(path: &Path) -> Result<PathBuf> {
    let s = path.to_string_lossy();
    if s == "~" {
        home::home_dir().context("cannot resolve home directory for `~` expansion")
    } else if let Some(rest) = s.strip_prefix("~/") {
        let home = home::home_dir().context("cannot resolve home directory for `~/` expansion")?;
        Ok(home.join(rest))
    } else {
        Ok(path.to_owned())
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::indexing_slicing, clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn test_expand_tilde_home() {
        let result = expand_tilde(Path::new("~")).unwrap();
        if let Some(home) = home::home_dir() {
            assert_eq!(result, home);
        }
    }

    #[test]
    fn test_expand_tilde_home_subpath() {
        let result = expand_tilde(Path::new("~/foo/bar")).unwrap();
        if let Some(home) = home::home_dir() {
            assert_eq!(result, home.join("foo/bar"));
        }
    }

    #[test]
    fn test_expand_tilde_absolute_passthrough() {
        let result = expand_tilde(Path::new("/tmp/test")).unwrap();
        assert_eq!(result, PathBuf::from("/tmp/test"));
    }

    #[test]
    fn test_expand_tilde_relative_passthrough() {
        let result = expand_tilde(Path::new("data/sessions")).unwrap();
        assert_eq!(result, PathBuf::from("data/sessions"));
    }

    #[test]
    fn test_expand_tilde_tilde_user_not_expanded() {
        let result = expand_tilde(Path::new("~otheruser/path")).unwrap();
        assert_eq!(result, PathBuf::from("~otheruser/path"));
    }

    #[test]
    fn test_expand_tilde_empty_path() {
        let result = expand_tilde(Path::new("")).unwrap();
        assert_eq!(result, PathBuf::from(""));
    }

    // --- validate_config tests ---

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
    fn test_validate_config_no_channels_still_valid() {
        let mut config = valid_config();
        config.channels.clear();
        assert!(
            validate_config(&config).is_ok(),
            "channels are optional — TcpChannel uses daemon config, not [[channels]]"
        );
    }

    #[test]
    fn test_daemon_config_defaults_used_when_absent() {
        let config = valid_config();
        assert_eq!(
            config.daemon.host,
            std::net::IpAddr::V4(std::net::Ipv4Addr::LOCALHOST)
        );
        assert_eq!(config.daemon.port, 7531);
    }

    #[test]
    fn test_validate_config_zero_max_tool_rounds_errors() {
        let mut config = valid_config();
        config.runtime.max_tool_rounds = 0;
        let err = validate_config(&config).unwrap_err();
        assert!(err.to_string().contains("max_tool_rounds"), "got: {err}");
    }

    #[test]
    fn test_validate_config_zero_budget_tokens_errors() {
        let mut config = valid_config();
        config.security.budgets.max_tokens_per_session = 0;
        let err = validate_config(&config).unwrap_err();
        assert!(
            err.to_string().contains("max_tokens_per_session"),
            "got: {err}"
        );
    }

    #[test]
    fn test_validate_config_invalid_egress_host_errors() {
        let mut config = valid_config();
        config.security.egress.allowed_hosts = vec!["valid.com".into(), String::new()];
        let err = validate_config(&config).unwrap_err();
        assert!(err.to_string().contains("allowed_hosts"), "got: {err}");
    }

    #[test]
    fn test_validate_config_low_pbkdf2_errors() {
        let mut config = valid_config();
        config.memory.pbkdf2_iterations = 500;
        let err = validate_config(&config).unwrap_err();
        assert!(err.to_string().contains("pbkdf2_iterations"), "got: {err}");
    }

    #[test]
    fn test_enforce_security_minimums_clamps_consent() {
        let mut config = valid_config();
        config.security.require_consent_above = freebird_traits::tool::RiskLevel::Critical;
        let config = enforce_security_minimums(config);
        assert_eq!(
            config.security.require_consent_above,
            freebird_traits::tool::RiskLevel::High
        );
    }

    #[test]
    fn test_enforce_security_minimums_clamps_tool_rounds() {
        let mut config = valid_config();
        config.security.budgets.max_tool_rounds_per_turn = 100;
        let config = enforce_security_minimums(config);
        assert_eq!(config.security.budgets.max_tool_rounds_per_turn, 50);
    }
}
