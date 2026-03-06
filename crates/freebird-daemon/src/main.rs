//! Freebird daemon — the composition root.
//!
//! Single binary with `clap` subcommands:
//! - `freebird serve`  — start daemon with TCP listener
//! - `freebird chat`   — connect to running daemon for interactive chat
//! - `freebird status` — check if daemon is running
//! - `freebird stop`   — send graceful shutdown to daemon

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

use freebird_channels::tcp::TcpChannel;
use freebird_memory::file::FileMemory;
use freebird_runtime::agent::AgentRuntime;
use freebird_runtime::shutdown::ShutdownCoordinator;
use freebird_types::config::AppConfig;

mod chat;
mod providers;

/// `FreeBird` AI agent daemon.
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
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Serve { allow_dirs } => cmd_serve(allow_dirs).await,
        Commands::Chat => cmd_chat().await,
        Commands::Status => cmd_status().await,
        Commands::Stop => cmd_stop().await,
    }
}

/// `freebird serve` — start daemon with TCP channel.
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

    // 4. PROVIDER REGISTRY
    let registry = providers::build_provider_registry(&config).await?;

    // 5. CHANNEL — TcpChannel from daemon config.
    let channel: Box<dyn freebird_traits::channel::Channel> = Box::new(TcpChannel::new(
        config.daemon.host.to_string(),
        config.daemon.port,
    ));

    // 6. MEMORY
    let memory_dir = expand_tilde(
        &config
            .memory
            .base_dir
            .clone()
            .unwrap_or_else(|| PathBuf::from("~/.freebird/conversations")),
    )?;
    let memory = tokio::task::spawn_blocking(move || FileMemory::new(memory_dir))
        .await
        .context("file memory init task panicked")?
        .context("failed to initialize file memory backend")?;

    // 7. TOOLS
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

    // Merge CLI --allow-dir flags with any configured allowed_directories.
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

    let tools: Vec<Box<dyn freebird_traits::tool::Tool>> =
        freebird_tools::filesystem::filesystem_tools(tools_config.sandbox_root.clone());

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

    // 9. AGENT RUNTIME
    let runtime = AgentRuntime::new(
        registry,
        channel,
        tools,
        Box::new(memory),
        config.runtime,
        tools_config,
        None, // audit logger — wired in a later issue
    );

    // 10. RUN
    let run_result = runtime.run(token).await;
    match &run_result {
        Ok(()) => tracing::info!("runtime exited cleanly"),
        Err(e) => tracing::error!(%e, "runtime error"),
    }

    // 11. DRAIN
    if drain_timeout > Duration::ZERO {
        tracing::info!(?drain_timeout, "draining in-flight work");
        let _ = tokio::time::timeout(drain_timeout, signal_handle).await;
    } else {
        signal_handle.abort();
    }

    tracing::info!("freebird stopped");
    run_result.map_err(Into::into)
}

/// `freebird chat` — thin TCP client that connects to the daemon.
async fn cmd_chat() -> Result<()> {
    let config = load_config()?;
    let addr = format!("{}:{}", config.daemon.host, config.daemon.port);

    let stream = tokio::net::TcpStream::connect(&addr)
        .await
        .with_context(|| format!("failed to connect to daemon at {addr}"))?;

    eprintln!("Connected to freebird daemon at {addr}");
    eprintln!("Type /quit to disconnect, /help for commands.\n");

    let stdin = tokio::io::BufReader::new(tokio::io::stdin());
    let stdout = tokio::io::stdout();

    chat::run_chat_with_io(stream, stdin, stdout, true).await
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
    chat::send_client_message(&mut stream, &msg)
        .await
        .context("sending shutdown command")?;

    println!("shutdown command sent to daemon at {addr}");
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
}
