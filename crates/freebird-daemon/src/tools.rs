//! Tool registry construction.
//!
//! Builds and populates the [`ToolRegistry`] from application configuration,
//! including the filesystem, shell, and network tools.

use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};

use freebird_runtime::tool_registry::ToolRegistry;
use freebird_security::egress::EgressPolicy;
use freebird_tools::network::NetworkToolConfig;
use freebird_types::config::AppConfig;

/// Build and populate the tool registry from configuration.
///
/// Registers:
/// - Filesystem tools (`read_file`, `write_file`, `list_directory`)
/// - Shell tool (`shell`)
/// - Network tool (`http_request`) — gated by [`EgressPolicy`] built from
///   `config.security.egress`
pub fn build_tool_registry(config: &AppConfig) -> Result<ToolRegistry> {
    let mut registry = ToolRegistry::new();

    // Filesystem tools — no config needed beyond ToolsConfig (passed at runtime).
    registry.register_all(freebird_tools::filesystem::filesystem_tools());

    // Shell tool — allowed commands and output limit from ToolsConfig.
    registry.register(freebird_tools::shell::shell_tool(
        config.tools.allowed_shell_commands.clone(),
        config.tools.max_shell_output_bytes,
    ));

    // Network tool — gated by egress policy from SecurityConfig.
    let egress_policy = build_egress_policy(config);
    let network_config = NetworkToolConfig {
        max_response_bytes: config.security.egress.max_response_bytes,
        request_timeout: Duration::from_secs(config.security.egress.request_timeout_secs),
    };
    let client = build_http_client().context("failed to build HTTP client for network tool")?;
    registry.register(freebird_tools::network::network_tool(
        client,
        egress_policy,
        network_config,
    ));

    tracing::info!(
        tool_count = registry.tool_count(),
        tools = ?registry.tool_names(),
        "tool registry initialized"
    );

    Ok(registry)
}

/// Build the [`EgressPolicy`] from the egress configuration.
fn build_egress_policy(config: &AppConfig) -> Arc<EgressPolicy> {
    let allowed_hosts = config
        .security
        .egress
        .allowed_hosts
        .iter()
        .cloned()
        .collect();
    let allowed_ports = config
        .security
        .egress
        .allowed_ports
        .iter()
        .copied()
        .collect();

    Arc::new(EgressPolicy::new(allowed_hosts, allowed_ports))
}

/// Build a `reqwest::Client` with security-hardened defaults.
///
/// Per CLAUDE.md dependency policy:
/// - `rustls` TLS only (never openssl)
/// - No automatic redirect following (prevents egress policy bypass)
/// - Connect timeout of 10 seconds (separate from per-request timeout)
fn build_http_client() -> Result<reqwest::Client, reqwest::Error> {
    reqwest::Client::builder()
        .use_rustls_tls()
        .redirect(reqwest::redirect::Policy::none())
        .connect_timeout(Duration::from_secs(10))
        .build()
}
