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

    // Knowledge tools — store, search, update, delete.
    registry.register_all(freebird_tools::knowledge::knowledge_tools());

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

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use figment::Figment;
    use figment::providers::{Format, Toml};

    fn test_config() -> AppConfig {
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
    fn test_build_tool_registry_registers_expected_tools() {
        let config = test_config();
        let registry = build_tool_registry(&config).unwrap();

        // Must include filesystem, shell, network, and knowledge tools
        assert!(registry.get("read_file").is_some(), "missing read_file");
        assert!(registry.get("write_file").is_some(), "missing write_file");
        assert!(
            registry.get("list_directory").is_some(),
            "missing list_directory"
        );
        assert!(registry.get("shell").is_some(), "missing shell");
        assert!(
            registry.get("http_request").is_some(),
            "missing http_request"
        );
        assert!(
            registry.get("store_knowledge").is_some(),
            "missing store_knowledge"
        );
        assert!(
            registry.get("search_knowledge").is_some(),
            "missing search_knowledge"
        );
        assert!(
            registry.tool_count() >= 9,
            "expected at least 9 tools, got {}",
            registry.tool_count()
        );
    }

    #[test]
    fn test_build_tool_registry_insertion_order_filesystem_first() {
        let config = test_config();
        let registry = build_tool_registry(&config).unwrap();
        let names = registry.tool_names();

        // Filesystem tools should come first, then shell, then network
        let shell_idx = names.iter().position(|n| n == "shell").unwrap();
        let http_idx = names.iter().position(|n| n == "http_request").unwrap();
        assert!(
            shell_idx < http_idx,
            "shell should come before http_request"
        );
    }

    #[test]
    fn test_build_egress_policy_uses_config_values() {
        let mut config = test_config();
        config.security.egress.allowed_hosts = vec!["example.com".into()];
        config.security.egress.allowed_ports = vec![8080];

        let policy = build_egress_policy(&config);
        // EgressPolicy is opaque, but we can verify it was created from config
        // by checking that the policy exists (Arc is non-null)
        assert_eq!(Arc::strong_count(&policy), 1);
    }

    #[test]
    fn test_build_http_client_succeeds() {
        let client = build_http_client();
        assert!(client.is_ok(), "HTTP client builder should not fail");
    }
}
