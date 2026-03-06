//! Sandboxed HTTP request tool.
//!
//! All outbound requests are gated by the egress allowlist (`EgressPolicy`)
//! and sensitive headers are blocked. URL validation flows through the taint
//! boundary: `TaintedToolInput` → `SafeUrl::from_tainted()`.
//!
//! Response bodies are read via streaming with a configurable byte cap to
//! prevent OOM from oversized responses. Redirects are disabled at the
//! `reqwest::Client` level to prevent egress policy bypass.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use reqwest::Method;

use freebird_security::egress::EgressPolicy;
use freebird_security::taint::TaintedToolInput;
use freebird_traits::tool::{
    Capability, RiskLevel, SideEffects, Tool, ToolContext, ToolError, ToolInfo, ToolOutcome,
    ToolOutput,
};

/// Headers blocked to prevent credential injection, virtual host routing,
/// request smuggling, and proxy abuse from LLM-controlled input.
const BLOCKED_HEADERS: &[&str] = &[
    "host",
    "authorization",
    "proxy-authorization",
    "cookie",
    "set-cookie",
    "connection",
    "transfer-encoding",
    "te",
    "upgrade",
    "x-forwarded-for",
    "x-forwarded-host",
];

/// Configuration for the network request tool.
#[derive(Debug, Clone)]
pub struct NetworkToolConfig {
    /// Maximum response body bytes to read. Default: 1 MiB.
    pub max_response_bytes: usize,

    /// Per-request timeout covering the entire send + read cycle.
    /// Prevents slow-drip servers from holding connections indefinitely.
    /// Default: 30 seconds.
    pub request_timeout: Duration,
}

impl Default for NetworkToolConfig {
    fn default() -> Self {
        Self {
            max_response_bytes: 1_048_576,
            request_timeout: Duration::from_secs(30),
        }
    }
}

/// Sandboxed HTTP request tool. All outbound requests are gated by the
/// egress allowlist and sensitive headers are blocked.
///
/// The `reqwest::Client` is injected by the daemon's composition root.
/// In production it **MUST** be built with:
/// - `redirect::Policy::none()` — prevents egress policy bypass via redirects
/// - `use_rustls_tls()` — per CLAUDE.md dependency policy (never openssl)
/// - `connect_timeout(Duration::from_secs(10))` — separate from per-request timeout
pub struct HttpRequestTool {
    client: reqwest::Client,
    egress_policy: Arc<EgressPolicy>,
    config: NetworkToolConfig,
    info: ToolInfo,
}

impl HttpRequestTool {
    const NAME: &str = "http_request";

    /// Create a new HTTP request tool.
    ///
    /// The `client` should be built by the caller with appropriate TLS,
    /// redirect, and connection timeout settings. Tests inject a client
    /// configured for plain HTTP against wiremock.
    #[must_use]
    pub fn new(
        client: reqwest::Client,
        egress_policy: Arc<EgressPolicy>,
        config: NetworkToolConfig,
    ) -> Self {
        Self {
            client,
            egress_policy,
            config,
            info: ToolInfo {
                name: Self::NAME.into(),
                description: "Make an HTTP request to an allowlisted host. \
                    Only HTTPS URLs to pre-approved hosts are permitted."
                    .into(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "HTTPS URL to request (must be an allowlisted host)"
                        },
                        "method": {
                            "type": "string",
                            "enum": ["GET", "POST", "PUT", "DELETE"],
                            "description": "HTTP method (default: GET)"
                        },
                        "body": {
                            "type": "string",
                            "description": "Request body (for POST/PUT)"
                        },
                        "headers": {
                            "type": "object",
                            "additionalProperties": { "type": "string" },
                            "description": "Custom headers (some headers are blocked for security)"
                        }
                    },
                    "required": ["url"]
                }),
                required_capability: Capability::NetworkOutbound,
                risk_level: RiskLevel::High,
                side_effects: SideEffects::HasSideEffects,
            },
        }
    }
}

/// Validated request parameters, ready to send.
/// Produced by `validate_input()`, consumed by `send_request()`.
#[derive(Debug)]
struct ValidatedRequest {
    url: String,
    method: Method,
    headers: HashMap<String, String>,
    body: Option<String>,
}

#[async_trait]
impl Tool for HttpRequestTool {
    fn info(&self) -> &ToolInfo {
        &self.info
    }

    async fn execute(
        &self,
        input: serde_json::Value,
        _context: &ToolContext<'_>,
    ) -> Result<ToolOutput, ToolError> {
        let request = self.validate_input(input)?;
        self.send_request(request).await
    }
}

impl HttpRequestTool {
    /// Parse and validate LLM input. The URL goes through the taint boundary
    /// (`TaintedToolInput` → `SafeUrl`). Other fields are parsed from the raw
    /// JSON — they are NOT security-critical for egress:
    ///
    /// - Method: matched against a fixed whitelist
    /// - Headers: blocked list prevents credential/routing injection
    /// - Body: arbitrary outbound content, not interpreted by our system
    fn validate_input(&self, input: serde_json::Value) -> Result<ValidatedRequest, ToolError> {
        // 1. Method whitelist (read before consuming input into TaintedToolInput)
        let method = match input.get("method").and_then(|v| v.as_str()) {
            None | Some("GET") => Method::GET,
            Some("POST") => Method::POST,
            Some("PUT") => Method::PUT,
            Some("DELETE") => Method::DELETE,
            Some(other) => {
                return Err(ToolError::InvalidInput {
                    tool: Self::NAME.into(),
                    reason: format!("unsupported HTTP method: {other}"),
                });
            }
        };

        // 2. Headers — block sensitive headers, reject non-string values
        let mut headers = HashMap::new();
        if let Some(obj) = input.get("headers").and_then(|v| v.as_object()) {
            for (key, value) in obj {
                let key_lower = key.to_lowercase();
                if BLOCKED_HEADERS.contains(&key_lower.as_str()) {
                    return Err(ToolError::InvalidInput {
                        tool: Self::NAME.into(),
                        reason: format!("header `{key}` is blocked for security"),
                    });
                }
                let val = value.as_str().ok_or_else(|| ToolError::InvalidInput {
                    tool: Self::NAME.into(),
                    reason: format!("header `{key}` value must be a string"),
                })?;
                headers.insert(key.clone(), val.to_owned());
            }
        }

        // 3. Body (optional)
        let body = input.get("body").and_then(|v| v.as_str()).map(String::from);

        // 4. URL through taint boundary (HTTPS + egress policy enforced).
        //    Done last so we can consume `input` without cloning.
        let tainted = TaintedToolInput::new(input);
        let safe_url = tainted
            .extract_url("url", &self.egress_policy)
            .map_err(|e| ToolError::InvalidInput {
                tool: Self::NAME.into(),
                reason: e.to_string(),
            })?;

        Ok(ValidatedRequest {
            url: safe_url.as_str().to_owned(),
            method,
            headers,
            body,
        })
    }

    /// Execute a validated HTTP request. Wraps the entire send + read cycle
    /// in a per-request timeout.
    async fn send_request(&self, request: ValidatedRequest) -> Result<ToolOutput, ToolError> {
        tokio::time::timeout(
            self.config.request_timeout,
            self.send_request_inner(request),
        )
        .await
        .map_err(|_| ToolError::Timeout {
            tool: Self::NAME.into(),
            timeout_ms: u64::try_from(self.config.request_timeout.as_millis()).unwrap_or(u64::MAX),
        })?
    }

    /// Inner implementation without timeout wrapper.
    async fn send_request_inner(&self, request: ValidatedRequest) -> Result<ToolOutput, ToolError> {
        let mut req = self.client.request(request.method, &request.url);

        for (key, value) in &request.headers {
            req = req.header(key.as_str(), value.as_str());
        }

        if let Some(body) = request.body {
            req = req.body(body);
        }

        let response = req.send().await.map_err(|e| ToolError::ExecutionFailed {
            tool: Self::NAME.into(),
            reason: e.to_string(),
        })?;

        let status = response.status().as_u16();

        let resp_headers: serde_json::Map<String, serde_json::Value> = response
            .headers()
            .iter()
            .filter_map(|(k, v)| {
                v.to_str()
                    .ok()
                    .map(|val| (k.to_string(), serde_json::Value::String(val.to_owned())))
            })
            .collect();

        let (body_text, truncated) = self.read_body_capped(response).await?;

        let result = serde_json::json!({
            "status": status,
            "headers": resp_headers,
            "body": body_text,
            "truncated": truncated,
        });

        let outcome = if status >= 400 {
            ToolOutcome::Error
        } else {
            ToolOutcome::Success
        };

        Ok(ToolOutput {
            content: result.to_string(),
            outcome,
            metadata: Some(serde_json::json!({ "status": status })),
        })
    }

    /// Read response body in chunks, stopping after `max_response_bytes`.
    /// Never buffers more than the limit + one chunk (~16 KiB) in memory.
    async fn read_body_capped(
        &self,
        response: reqwest::Response,
    ) -> Result<(String, bool), ToolError> {
        use futures::StreamExt;

        let cap = self.config.max_response_bytes;
        let mut body = Vec::with_capacity(cap.min(65_536));
        let mut stream = response.bytes_stream();
        let mut total = 0usize;
        let mut truncated = false;

        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(|e| ToolError::ExecutionFailed {
                tool: Self::NAME.into(),
                reason: e.to_string(),
            })?;

            let remaining = cap.saturating_sub(total);
            if remaining == 0 {
                truncated = true;
                break;
            }

            let take = chunk.len().min(remaining);
            // SAFETY invariant: `take = chunk.len().min(remaining)` ≤ `chunk.len()`
            // so `get(..take)` always returns `Some`. Fallback to full chunk
            // is unreachable but avoids a panic path.
            if let Some(slice) = chunk.get(..take) {
                body.extend_from_slice(slice);
            }
            total += take;

            if take < chunk.len() {
                truncated = true;
                break;
            }
        }

        let text = String::from_utf8_lossy(&body).into_owned();
        Ok((text, truncated))
    }
}

/// Convenience factory for the tool registry.
#[must_use]
pub fn network_tool(
    client: reqwest::Client,
    egress_policy: Arc<EgressPolicy>,
    config: NetworkToolConfig,
) -> Box<dyn Tool> {
    Box::new(HttpRequestTool::new(client, egress_policy, config))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::indexing_slicing)]
mod tests {
    use wiremock::matchers::{body_string, header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    use super::*;

    // ── Test Helpers ──────────────────────────────────────────────────

    /// Egress policy that allows only `api.anthropic.com:443`.
    fn test_policy() -> Arc<EgressPolicy> {
        Arc::new(EgressPolicy::new(
            std::iter::once("api.anthropic.com".into()).collect(),
            std::iter::once(443).collect(),
        ))
    }

    /// Egress policy that allows a wiremock server's host + port.
    fn mock_policy(server: &MockServer) -> Arc<EgressPolicy> {
        let uri: url::Url = server.uri().parse().unwrap();
        let host = uri.host_str().unwrap().to_owned();
        let port = uri.port().unwrap_or(80);
        Arc::new(EgressPolicy::new(
            std::iter::once(host).collect(),
            std::iter::once(port).collect(),
        ))
    }

    /// Build a tool wired to a wiremock server (HTTP, no TLS).
    fn mock_tool(server: &MockServer) -> HttpRequestTool {
        mock_tool_with_config(server, NetworkToolConfig::default())
    }

    fn mock_tool_with_config(server: &MockServer, config: NetworkToolConfig) -> HttpRequestTool {
        let client = reqwest::Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .unwrap();
        HttpRequestTool::new(client, mock_policy(server), config)
    }

    /// Build a tool that only allows HTTPS to api.anthropic.com:443.
    /// Used for validation-only tests (no actual HTTP calls).
    fn validation_tool() -> HttpRequestTool {
        let client = reqwest::Client::builder().build().unwrap();
        HttpRequestTool::new(client, test_policy(), NetworkToolConfig::default())
    }

    /// Valid HTTPS URL for the test policy (for validation-only tests).
    fn valid_url() -> &'static str {
        "https://api.anthropic.com/v1/messages"
    }

    // ── Validation Tests ─────────────────────────────────────────────

    #[test]
    fn test_validate_missing_url_returns_error() {
        let tool = validation_tool();
        let input = serde_json::json!({"method": "GET"});
        let err = tool.validate_input(input).unwrap_err();
        assert!(matches!(err, ToolError::InvalidInput { .. }));
        assert!(err.to_string().contains("url"));
    }

    #[test]
    fn test_validate_non_string_url_returns_error() {
        let tool = validation_tool();
        let input = serde_json::json!({"url": 42});
        let err = tool.validate_input(input).unwrap_err();
        assert!(matches!(err, ToolError::InvalidInput { .. }));
    }

    #[test]
    fn test_validate_invalid_url_returns_error() {
        let tool = validation_tool();
        let input = serde_json::json!({"url": "not a url"});
        let err = tool.validate_input(input).unwrap_err();
        assert!(matches!(err, ToolError::InvalidInput { .. }));
    }

    #[test]
    fn test_validate_http_url_rejected() {
        let tool = validation_tool();
        let input = serde_json::json!({"url": "http://api.anthropic.com/v1/messages"});
        let err = tool.validate_input(input).unwrap_err();
        assert!(matches!(err, ToolError::InvalidInput { .. }));
        assert!(err.to_string().to_lowercase().contains("https"));
    }

    #[test]
    fn test_validate_non_allowlisted_host_rejected() {
        let tool = validation_tool();
        let input = serde_json::json!({"url": "https://evil.com/exfiltrate"});
        let err = tool.validate_input(input).unwrap_err();
        assert!(matches!(err, ToolError::InvalidInput { .. }));
        assert!(err.to_string().contains("allowlist"));
    }

    #[test]
    fn test_validate_unsupported_method_rejected() {
        let tool = validation_tool();
        let input = serde_json::json!({"url": valid_url(), "method": "PATCH"});
        let err = tool.validate_input(input).unwrap_err();
        assert!(matches!(err, ToolError::InvalidInput { .. }));
        assert!(err.to_string().contains("PATCH"));
    }

    #[test]
    fn test_validate_blocked_header_authorization() {
        let tool = validation_tool();
        let input = serde_json::json!({
            "url": valid_url(),
            "headers": {"authorization": "Bearer stolen"}
        });
        let err = tool.validate_input(input).unwrap_err();
        assert!(matches!(err, ToolError::InvalidInput { .. }));
        assert!(err.to_string().contains("blocked"));
    }

    #[test]
    fn test_validate_blocked_header_host() {
        let tool = validation_tool();
        let input = serde_json::json!({
            "url": valid_url(),
            "headers": {"Host": "evil.com"}
        });
        let err = tool.validate_input(input).unwrap_err();
        assert!(matches!(err, ToolError::InvalidInput { .. }));
        assert!(err.to_string().contains("blocked"));
    }

    #[test]
    fn test_validate_blocked_header_cookie() {
        let tool = validation_tool();
        let input = serde_json::json!({
            "url": valid_url(),
            "headers": {"Cookie": "session=abc"}
        });
        let err = tool.validate_input(input).unwrap_err();
        assert!(matches!(err, ToolError::InvalidInput { .. }));
        assert!(err.to_string().contains("blocked"));
    }

    #[test]
    fn test_validate_blocked_header_x_forwarded_for() {
        let tool = validation_tool();
        let input = serde_json::json!({
            "url": valid_url(),
            "headers": {"X-Forwarded-For": "127.0.0.1"}
        });
        let err = tool.validate_input(input).unwrap_err();
        assert!(matches!(err, ToolError::InvalidInput { .. }));
        assert!(err.to_string().contains("blocked"));
    }

    #[test]
    fn test_validate_non_string_header_value_rejected() {
        let tool = validation_tool();
        let input = serde_json::json!({
            "url": valid_url(),
            "headers": {"X-Count": 42}
        });
        let err = tool.validate_input(input).unwrap_err();
        assert!(matches!(err, ToolError::InvalidInput { .. }));
        assert!(err.to_string().contains("must be a string"));
    }

    #[test]
    fn test_validate_defaults_method_to_get() {
        let tool = validation_tool();
        let input = serde_json::json!({"url": valid_url()});
        let req = tool.validate_input(input).unwrap();
        assert_eq!(req.method, Method::GET);
    }

    #[test]
    fn test_validate_accepts_all_supported_methods() {
        let tool = validation_tool();
        for (name, expected) in [
            ("GET", Method::GET),
            ("POST", Method::POST),
            ("PUT", Method::PUT),
            ("DELETE", Method::DELETE),
        ] {
            let input = serde_json::json!({"url": valid_url(), "method": name});
            let req = tool.validate_input(input).unwrap();
            assert_eq!(req.method, expected, "method {name} should parse");
        }
    }

    #[test]
    fn test_validate_safe_headers_accepted() {
        let tool = validation_tool();
        let input = serde_json::json!({
            "url": valid_url(),
            "headers": {
                "Content-Type": "application/json",
                "Accept": "text/html"
            }
        });
        let req = tool.validate_input(input).unwrap();
        assert_eq!(req.headers.get("Content-Type").unwrap(), "application/json");
        assert_eq!(req.headers.get("Accept").unwrap(), "text/html");
    }

    #[test]
    fn test_validate_valid_input_with_all_fields() {
        let tool = validation_tool();
        let input = serde_json::json!({
            "url": valid_url(),
            "method": "POST",
            "body": "hello",
            "headers": {"Content-Type": "text/plain"}
        });
        let req = tool.validate_input(input).unwrap();
        assert_eq!(req.method, Method::POST);
        assert_eq!(req.body.as_deref(), Some("hello"));
        assert_eq!(req.headers.get("Content-Type").unwrap(), "text/plain");
        assert!(req.url.contains("api.anthropic.com"));
    }

    // ── HTTP Execution Tests (wiremock) ──────────────────────────────

    #[tokio::test]
    async fn test_get_request_returns_body_and_status() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/data"))
            .respond_with(ResponseTemplate::new(200).set_body_string("hello world"))
            .mount(&server)
            .await;

        let tool = mock_tool(&server);
        let req = ValidatedRequest {
            url: format!("{}/data", server.uri()),
            method: Method::GET,
            headers: HashMap::new(),
            body: None,
        };
        let output = tool.send_request(req).await.unwrap();
        assert_eq!(output.outcome, ToolOutcome::Success);

        let parsed: serde_json::Value = serde_json::from_str(&output.content).unwrap();
        assert_eq!(parsed["status"], 200);
        assert_eq!(parsed["body"], "hello world");
    }

    #[tokio::test]
    async fn test_post_request_sends_body() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/submit"))
            .and(body_string("request data"))
            .respond_with(ResponseTemplate::new(200))
            .mount(&server)
            .await;

        let tool = mock_tool(&server);
        let req = ValidatedRequest {
            url: format!("{}/submit", server.uri()),
            method: Method::POST,
            headers: HashMap::new(),
            body: Some("request data".into()),
        };
        let output = tool.send_request(req).await.unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&output.content).unwrap();
        assert_eq!(parsed["status"], 200);
    }

    #[tokio::test]
    async fn test_custom_headers_sent() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/api"))
            .and(header("content-type", "application/json"))
            .respond_with(ResponseTemplate::new(200))
            .mount(&server)
            .await;

        let tool = mock_tool(&server);
        let mut headers = HashMap::new();
        headers.insert("Content-Type".into(), "application/json".into());
        let req = ValidatedRequest {
            url: format!("{}/api", server.uri()),
            method: Method::GET,
            headers,
            body: None,
        };
        let output = tool.send_request(req).await.unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&output.content).unwrap();
        assert_eq!(parsed["status"], 200);
    }

    #[tokio::test]
    async fn test_response_body_truncated_at_limit() {
        let server = MockServer::start().await;
        let large_body = "x".repeat(500);
        Mock::given(method("GET"))
            .and(path("/big"))
            .respond_with(ResponseTemplate::new(200).set_body_string(large_body))
            .mount(&server)
            .await;

        let config = NetworkToolConfig {
            max_response_bytes: 100,
            ..NetworkToolConfig::default()
        };
        let tool = mock_tool_with_config(&server, config);
        let req = ValidatedRequest {
            url: format!("{}/big", server.uri()),
            method: Method::GET,
            headers: HashMap::new(),
            body: None,
        };
        let output = tool.send_request(req).await.unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&output.content).unwrap();
        assert_eq!(parsed["truncated"], true);
        let body = parsed["body"].as_str().unwrap();
        assert!(body.len() <= 100);
    }

    #[tokio::test]
    async fn test_response_body_not_truncated_when_within_limit() {
        let server = MockServer::start().await;
        let small_body = "y".repeat(50);
        Mock::given(method("GET"))
            .and(path("/small"))
            .respond_with(ResponseTemplate::new(200).set_body_string(small_body.clone()))
            .mount(&server)
            .await;

        let config = NetworkToolConfig {
            max_response_bytes: 1000,
            ..NetworkToolConfig::default()
        };
        let tool = mock_tool_with_config(&server, config);
        let req = ValidatedRequest {
            url: format!("{}/small", server.uri()),
            method: Method::GET,
            headers: HashMap::new(),
            body: None,
        };
        let output = tool.send_request(req).await.unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&output.content).unwrap();
        assert_eq!(parsed["truncated"], false);
        assert_eq!(parsed["body"].as_str().unwrap(), small_body);
    }

    #[tokio::test]
    async fn test_4xx_response_sets_outcome_error() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/notfound"))
            .respond_with(ResponseTemplate::new(404))
            .mount(&server)
            .await;

        let tool = mock_tool(&server);
        let req = ValidatedRequest {
            url: format!("{}/notfound", server.uri()),
            method: Method::GET,
            headers: HashMap::new(),
            body: None,
        };
        let output = tool.send_request(req).await.unwrap();
        assert_eq!(output.outcome, ToolOutcome::Error);
    }

    #[tokio::test]
    async fn test_5xx_response_sets_outcome_error() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/error"))
            .respond_with(ResponseTemplate::new(500))
            .mount(&server)
            .await;

        let tool = mock_tool(&server);
        let req = ValidatedRequest {
            url: format!("{}/error", server.uri()),
            method: Method::GET,
            headers: HashMap::new(),
            body: None,
        };
        let output = tool.send_request(req).await.unwrap();
        assert_eq!(output.outcome, ToolOutcome::Error);
    }

    #[tokio::test]
    async fn test_2xx_response_sets_outcome_success() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/ok"))
            .respond_with(ResponseTemplate::new(200))
            .mount(&server)
            .await;

        let tool = mock_tool(&server);
        let req = ValidatedRequest {
            url: format!("{}/ok", server.uri()),
            method: Method::GET,
            headers: HashMap::new(),
            body: None,
        };
        let output = tool.send_request(req).await.unwrap();
        assert_eq!(output.outcome, ToolOutcome::Success);
    }

    #[tokio::test]
    async fn test_network_error_returns_execution_failed() {
        let server = MockServer::start().await;
        let tool = mock_tool(&server);
        // Point to a port that's definitely not listening
        let req = ValidatedRequest {
            url: "http://127.0.0.1:1/unreachable".into(),
            method: Method::GET,
            headers: HashMap::new(),
            body: None,
        };
        let err = tool.send_request(req).await.unwrap_err();
        assert!(matches!(err, ToolError::ExecutionFailed { .. }));
    }

    #[tokio::test]
    async fn test_redirect_not_followed() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/redirect"))
            .respond_with(
                ResponseTemplate::new(301).insert_header("Location", "http://evil.com/steal"),
            )
            .mount(&server)
            .await;

        let tool = mock_tool(&server);
        let req = ValidatedRequest {
            url: format!("{}/redirect", server.uri()),
            method: Method::GET,
            headers: HashMap::new(),
            body: None,
        };
        let output = tool.send_request(req).await.unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&output.content).unwrap();
        assert_eq!(parsed["status"], 301);
        assert!(
            parsed["headers"]["location"]
                .as_str()
                .unwrap()
                .contains("evil.com")
        );
    }

    #[tokio::test]
    async fn test_response_headers_included() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/headers"))
            .respond_with(ResponseTemplate::new(200).insert_header("x-custom", "custom-value"))
            .mount(&server)
            .await;

        let tool = mock_tool(&server);
        let req = ValidatedRequest {
            url: format!("{}/headers", server.uri()),
            method: Method::GET,
            headers: HashMap::new(),
            body: None,
        };
        let output = tool.send_request(req).await.unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&output.content).unwrap();
        assert_eq!(parsed["headers"]["x-custom"], "custom-value");
    }

    #[tokio::test]
    async fn test_empty_response_body() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/empty"))
            .respond_with(ResponseTemplate::new(204))
            .mount(&server)
            .await;

        let tool = mock_tool(&server);
        let req = ValidatedRequest {
            url: format!("{}/empty", server.uri()),
            method: Method::GET,
            headers: HashMap::new(),
            body: None,
        };
        let output = tool.send_request(req).await.unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&output.content).unwrap();
        assert_eq!(parsed["body"], "");
        assert_eq!(parsed["truncated"], false);
    }

    // ── Metadata & Factory Tests ─────────────────────────────────────

    #[test]
    fn test_tool_info_has_correct_capability_and_risk() {
        let tool = validation_tool();
        let info = tool.info();
        assert_eq!(info.name, "http_request");
        assert_eq!(info.required_capability, Capability::NetworkOutbound);
        assert_eq!(info.risk_level, RiskLevel::High);
        assert_eq!(info.side_effects, SideEffects::HasSideEffects);
    }

    #[test]
    fn test_network_tool_factory_returns_boxed_tool() {
        let client = reqwest::Client::builder().build().unwrap();
        let tool = network_tool(client, test_policy(), NetworkToolConfig::default());
        assert_eq!(tool.info().name, "http_request");
    }

    // ── Blocked Header Exhaustiveness ────────────────────────────────

    #[test]
    fn test_all_blocked_headers_are_rejected() {
        let tool = validation_tool();
        for &blocked in BLOCKED_HEADERS {
            let input = serde_json::json!({
                "url": valid_url(),
                "headers": { blocked: "value" }
            });
            let err = tool.validate_input(input).unwrap_err();
            assert!(
                matches!(err, ToolError::InvalidInput { .. }),
                "header `{blocked}` should be rejected"
            );
            assert!(
                err.to_string().contains("blocked"),
                "error for `{blocked}` should mention 'blocked': {err}"
            );
        }
    }
}
