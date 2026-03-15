//! Security-related configuration structs.

use freebird_traits::tool::RiskLevel;
use serde::{Deserialize, Serialize};

/// Security policy configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityConfig {
    pub max_tool_calls_per_turn: usize,
    /// Minimum risk level that requires explicit human consent before tool
    /// execution. Tools at this level or above trigger the consent gate.
    pub require_consent_above: RiskLevel,
    /// How long (in seconds) to wait for a user to respond to a consent
    /// request before auto-denying. Default: 60.
    #[serde(default = "default_consent_timeout_secs")]
    pub consent_timeout_secs: u64,
    /// Maximum number of simultaneous pending consent requests.
    /// Prevents LLM flooding attacks. Default: 5.
    #[serde(default = "default_max_pending_consent")]
    pub max_pending_consent_requests: usize,
    /// Default session TTL in hours. Sessions expire after this duration
    /// unless a more specific TTL is provided. Default: 24 hours.
    #[serde(default = "default_session_ttl_hours")]
    pub default_session_ttl_hours: u64,
    /// Network egress policy. Controls which hosts the agent can contact.
    #[serde(default)]
    pub egress: EgressConfig,
    /// Secret guard policy. Controls detection and action for tool
    /// invocations that may access or expose secrets.
    #[serde(default)]
    pub secret_guard: SecretGuardConfig,
    /// Token and tool-round budget limits.
    #[serde(default)]
    pub budgets: BudgetConfig,
    /// Injection detection response configuration.
    #[serde(default)]
    pub injection: InjectionConfig,
}

const fn default_session_ttl_hours() -> u64 {
    24
}

const fn default_consent_timeout_secs() -> u64 {
    60
}

const fn default_max_pending_consent() -> usize {
    5
}

/// Network egress allowlist configuration (CLAUDE.md \u{00a7}12 \u{2014} ASI01).
///
/// Default is deny-all with only provider API hosts permitted.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EgressConfig {
    /// Hosts the agent is allowed to contact (e.g., `["api.anthropic.com"]`).
    #[serde(default = "default_egress_allowed_hosts")]
    pub allowed_hosts: Vec<String>,
    /// Ports the agent is allowed to contact. Default: `[443]`.
    #[serde(default = "default_egress_allowed_ports")]
    pub allowed_ports: Vec<u16>,
    /// Maximum response body bytes the network tool will read. Default: 1 MiB.
    #[serde(default = "default_egress_max_response_bytes")]
    pub max_response_bytes: usize,
    /// Per-request timeout in seconds for outbound HTTP. Default: 30.
    #[serde(default = "default_egress_request_timeout_secs")]
    pub request_timeout_secs: u64,
    /// Maximum requests per 60-second sliding window. Default: 60.
    /// Prevents rapid-fire exfiltration bursts (CLAUDE.md \u{00a7}12 \u{2014} ASI01).
    #[serde(default = "default_egress_rate_limit_per_minute")]
    pub rate_limit_per_minute: u32,
    /// Maximum request body bytes the network tool will send. Default: 1 MiB.
    /// Prevents large data exfiltration to allowlisted hosts (CLAUDE.md \u{00a7}12).
    #[serde(default = "default_egress_max_request_body_bytes")]
    pub max_request_body_bytes: usize,
}

impl Default for EgressConfig {
    fn default() -> Self {
        Self {
            allowed_hosts: default_egress_allowed_hosts(),
            allowed_ports: default_egress_allowed_ports(),
            max_response_bytes: default_egress_max_response_bytes(),
            request_timeout_secs: default_egress_request_timeout_secs(),
            rate_limit_per_minute: default_egress_rate_limit_per_minute(),
            max_request_body_bytes: default_egress_max_request_body_bytes(),
        }
    }
}

fn default_egress_allowed_hosts() -> Vec<String> {
    vec!["api.anthropic.com".into(), "api.openai.com".into()]
}

fn default_egress_allowed_ports() -> Vec<u16> {
    vec![443]
}

const fn default_egress_max_response_bytes() -> usize {
    102_400 // 100 KiB — ~25k tokens, fits comfortably in LLM context
}

const fn default_egress_request_timeout_secs() -> u64 {
    30
}

const fn default_egress_rate_limit_per_minute() -> u32 {
    60
}

const fn default_egress_max_request_body_bytes() -> usize {
    1_048_576 // 1 MiB — prevents data exfiltration
}

/// What action the secret guard takes when a sensitive input is detected.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SecretGuardAction {
    /// Escalate to consent gate with `RiskLevel::Critical` — the user must
    /// explicitly approve the access.
    Consent,
    /// Deny outright without prompting. Use in headless/automated deployments.
    Block,
}

/// Secret guard configuration — detects and gates tool invocations that
/// access sensitive files, run secret-revealing commands, or produce output
/// containing credentials.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecretGuardConfig {
    /// Whether the secret guard is active. Default: `true`.
    #[serde(default = "default_secret_guard_enabled")]
    pub enabled: bool,
    /// Action on detection. Default: `consent`.
    #[serde(default = "default_secret_guard_action")]
    pub action: SecretGuardAction,
    /// Whether to redact detected secrets in tool output before returning
    /// to the LLM context. Default: `true`.
    #[serde(default = "default_secret_guard_redact_output")]
    pub redact_output: bool,
    /// Additional file patterns to treat as sensitive (glob syntax).
    /// Merged with built-in patterns.
    #[serde(default)]
    pub extra_sensitive_file_patterns: Vec<String>,
    /// Additional shell command patterns to treat as sensitive (regex syntax).
    /// Merged with built-in patterns.
    #[serde(default)]
    pub extra_sensitive_command_patterns: Vec<String>,
}

impl Default for SecretGuardConfig {
    fn default() -> Self {
        Self {
            enabled: default_secret_guard_enabled(),
            action: default_secret_guard_action(),
            redact_output: default_secret_guard_redact_output(),
            extra_sensitive_file_patterns: Vec::new(),
            extra_sensitive_command_patterns: Vec::new(),
        }
    }
}

const fn default_secret_guard_enabled() -> bool {
    true
}

const fn default_secret_guard_action() -> SecretGuardAction {
    SecretGuardAction::Consent
}

const fn default_secret_guard_redact_output() -> bool {
    true
}

/// How injection detection responds when a pattern is found.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InjectionResponse {
    /// Block the content outright — no user choice.
    Block,
    /// Warn the user and ask whether to proceed (default for input/tool output).
    Prompt,
    /// Allow the content through without warning.
    Allow,
}

/// Injection detection response configuration.
///
/// Controls how each injection detection layer responds when a pattern
/// is found. Model output and loaded context are always blocked
/// (non-configurable) because they represent compromised trust boundaries.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InjectionConfig {
    /// Response for injection patterns in user input. Default: `prompt`.
    #[serde(default = "default_injection_input_response")]
    pub input_response: InjectionResponse,
    /// Response for injection patterns in tool output. Default: `prompt`.
    #[serde(default = "default_injection_tool_output_response")]
    pub tool_output_response: InjectionResponse,
    /// Timeout in seconds for security prompts. Default: 60.
    #[serde(default = "default_injection_prompt_timeout_secs")]
    pub prompt_timeout_secs: u64,
}

impl Default for InjectionConfig {
    fn default() -> Self {
        Self {
            input_response: default_injection_input_response(),
            tool_output_response: default_injection_tool_output_response(),
            prompt_timeout_secs: default_injection_prompt_timeout_secs(),
        }
    }
}

const fn default_injection_input_response() -> InjectionResponse {
    InjectionResponse::Prompt
}

const fn default_injection_tool_output_response() -> InjectionResponse {
    InjectionResponse::Prompt
}

const fn default_injection_prompt_timeout_secs() -> u64 {
    60
}

/// Token and tool-round budget limits (CLAUDE.md \u{00a7}13 \u{2014} ASI08).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BudgetConfig {
    /// Maximum total tokens (input + output) per session.
    pub max_tokens_per_session: u64,
    /// Maximum tokens (input + output) per single provider request.
    pub max_tokens_per_request: u64,
    /// Maximum tool rounds in a single agentic turn.
    pub max_tool_rounds_per_turn: u32,
    /// Maximum cost per session in microdollars (1 microdollar = $0.000001).
    /// Default: 5,000,000 = $5.00.
    #[serde(default = "default_max_cost_microdollars")]
    pub max_cost_microdollars: u64,
}

const fn default_max_cost_microdollars() -> u64 {
    5_000_000 // $5.00
}

impl Default for BudgetConfig {
    fn default() -> Self {
        Self {
            max_tokens_per_session: 500_000,
            max_tokens_per_request: 32_768,
            max_tool_rounds_per_turn: 10,
            max_cost_microdollars: default_max_cost_microdollars(),
        }
    }
}
