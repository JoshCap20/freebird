//! Freebird daemon — thin binary shell.
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

use std::path::PathBuf;
use std::time::Duration;

use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand};
use tracing_subscriber::EnvFilter;

use freebird_runtime::shutdown::ShutdownCoordinator;

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
async fn cmd_serve(allow_dirs: Vec<PathBuf>) -> Result<()> {
    // 1. LOGGING — before anything else, so config errors are visible.
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()))
        .init();

    tracing::info!("freebird serve starting");

    // 2. BUILD — all composition logic lives in freebird-core
    let config = freebird_core::config::load_config()?;
    tracing::debug!(?config.runtime, "loaded configuration");

    let drain_timeout_secs = config.runtime.drain_timeout_secs;

    let app = freebird_core::FreebirdBuilder::from_config(config)
        .allow_dirs(allow_dirs)
        .build()
        .await
        .map_err(|e| anyhow::anyhow!("{e}"))?;

    // 3. SHUTDOWN COORDINATOR — signal handling is process-level
    let shutdown = ShutdownCoordinator::new(Duration::from_secs(drain_timeout_secs));
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

    // 4. EMIT DaemonStarted audit event (lifecycle events belong in the daemon)
    if let Some(sink) = app.audit_sink() {
        let event = freebird_security::audit::AuditEventType::DaemonStarted {
            version: env!("CARGO_PKG_VERSION").to_string(),
        };
        freebird_runtime::agent::emit_audit(sink.as_ref(), None, event).await;
    }

    // 5. RUN
    let audit_sink_for_shutdown = app.audit_sink().cloned();
    let run_result = app.run(token).await;
    match &run_result {
        Ok(()) => tracing::info!("runtime exited cleanly"),
        Err(e) => tracing::error!(%e, "runtime error"),
    }

    // 6. EMIT DaemonShutdown audit event
    if let Some(ref sink) = audit_sink_for_shutdown {
        let reason = match &run_result {
            Ok(()) => "clean shutdown".to_string(),
            Err(e) => format!("runtime error: {e}"),
        };
        let event = freebird_security::audit::AuditEventType::DaemonShutdown { reason };
        freebird_runtime::agent::emit_audit(sink.as_ref(), None, event).await;
    }

    // 7. DRAIN
    if drain_timeout > Duration::ZERO {
        tracing::info!(?drain_timeout, "draining in-flight work");
        let _ = tokio::time::timeout(drain_timeout, signal_handle).await;
    } else {
        signal_handle.abort();
    }

    tracing::info!("freebird stopped");
    run_result.map_err(|e| anyhow::anyhow!("{e}"))
}

/// `freebird chat` — interactive client that connects to the daemon.
async fn cmd_chat() -> Result<()> {
    let config = freebird_core::config::load_config()?;
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
    let config = freebird_core::config::load_config()?;
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
    let config = freebird_core::config::load_config()?;
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
async fn cmd_replay(session_id: Option<String>, last: bool, json: bool) -> Result<()> {
    let config = freebird_core::config::load_config()?;

    if session_id.is_none() && !last {
        bail!("specify a session ID or use --last to replay the most recent session");
    }

    let db_components = freebird_core::database::init_database(
        &config,
        &freebird_core::PassphraseStrategy::AutoResolve { allow_prompt: true },
    )
    .context("failed to initialize database for replay")?;

    // Resolve session ID
    let sid = if last {
        let sessions = db_components
            .memory
            .list_sessions(1)
            .await
            .context("failed to list sessions")?;
        sessions
            .into_iter()
            .next()
            .map(|s| s.session_id)
            .ok_or_else(|| anyhow::anyhow!("no sessions found"))?
    } else {
        let raw = session_id.ok_or_else(|| anyhow::anyhow!("missing session ID"))?;
        freebird_traits::id::SessionId::from(raw.as_str())
    };

    let conversation = db_components
        .memory
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
