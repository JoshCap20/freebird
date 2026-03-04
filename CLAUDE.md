# FreeBird — Rust Development Guide for Claude Code

> **Purpose**: This document is the authoritative reference for building FreeBird, a secure, always-running AI agent daemon written in Rust. Every section is a hard constraint, not a suggestion. When in doubt, prioritize security over convenience and explicitness over magic.
>
> **Design philosophy**: FreeBird is modeled after the open-source OpenClaw/ZeroClaw agent ecosystem but rebuilt from scratch in Rust with compile-time security guarantees. We take the best architectural ideas (trait-driven extensibility, channel/provider abstraction, local-first operation) and discard the security debt (exposed tokens in URLs, unsandboxed tool execution, no taint tracking, 36% of third-party skills containing prompt injection per Cisco research).

---

## Table of Contents

**Part I — System Architecture**

1. [Project Overview](#1-project-overview)
2. [Architecture & Workspace Layout](#2-architecture--workspace-layout)
3. [Core Principles](#3-core-principles)
4. [Message Flow & Data Architecture](#4-message-flow--data-architecture)

**Part II — Core Agent Subsystems**

5. [Provider Trait & Model Abstraction](#5-provider-trait--model-abstraction)
6. [Channel Trait & Transport Abstraction](#6-channel-trait--transport-abstraction)
7. [Tool Trait & Execution Sandbox](#7-tool-trait--execution-sandbox)
8. [Memory Trait & Conversation State](#8-memory-trait--conversation-state)
9. [Agent Runtime Loop](#9-agent-runtime-loop)

**Part III — Security (OWASP ASI Top 10 Aligned)**

10. [Type-Driven Security Patterns](#10-type-driven-security-patterns)
11. [Agentic Security Model](#11-agentic-security-model)
12. [Prompt Injection Defense (ASI01/ASI02)](#12-prompt-injection-defense-asi01asi02)
13. [Authentication & Session Management](#13-authentication--session-management)
14. [Channel Pairing & Identity Verification (ASI03)](#14-channel-pairing--identity-verification-asi03)
15. [Consent Gates & Human-in-the-Loop (ASI09)](#15-consent-gates--human-in-the-loop-asi09)
16. [Network Egress Control & Data Exfiltration Defense (ASI01)](#16-network-egress-control--data-exfiltration-defense-asi01)
17. [Token Budgets & Resource Exhaustion Defense (ASI08)](#17-token-budgets--resource-exhaustion-defense-asi08)
18. [Memory & Context Integrity (ASI06)](#18-memory--context-integrity-asi06)
19. [Tamper-Evident Audit Logging](#19-tamper-evident-audit-logging)
20. [Supply Chain Security (ASI04)](#20-supply-chain-security-asi04)

**Part IV — Infrastructure**

21. [Error Handling Strategy](#21-error-handling-strategy)
22. [Daemon Lifecycle & Graceful Shutdown](#22-daemon-lifecycle--graceful-shutdown)
23. [Configuration & Secrets Management](#23-configuration--secrets-management)
24. [Logging, Tracing & Audit](#24-logging-tracing--audit)
25. [Concurrency Patterns](#25-concurrency-patterns)

**Part V — Quality**

26. [Testing Strategy](#26-testing-strategy)
27. [Dependency Policy](#27-dependency-policy)
28. [Code Style & Linting](#28-code-style--linting)
29. [Performance Guidelines](#29-performance-guidelines)
30. [Common Anti-Patterns to Avoid](#30-common-anti-patterns-to-avoid)
31. [Security Checklist](#31-security-checklist)

---

# Part I — System Architecture

---

## 1. Project Overview

**FreeBird** is a persistent AI agent service that:

- Runs as a daemon on a host machine (always on, single static binary).
- Accepts user communication via pluggable **channels** (CLI first, then Signal, WebSocket, etc.).
- Routes conversations through pluggable **providers** (Anthropic/Opus first, extensible to OpenAI, Ollama, etc.).
- Executes agentic tasks via pluggable **tools** (filesystem, shell, network) within a sandboxed capability system.
- Persists conversation state via pluggable **memory** backends (file-based first, extensible to SQLite, vector stores).
- Authenticates users via **session keys** and authenticates to providers via scoped **API credentials**.
- Enforces taint-based data flow tracking, path traversal guards, capability-scoped permissions, prompt injection defense, and structured security audit logging — all at **compile time** where possible.

### Prior Art & Lessons Learned

| Project | What we take | What we fix |
|---|---|---|
| **FreeBird** (TS, 430k LOC) | Channel/provider abstraction, skill system concept, heartbeat daemon, gateway architecture | Exposed tokens in URLs (CVE-2026-25253 CVSS 8.8), no taint tracking, 36% of ClawHub skills contain prompt injection (Cisco), unsandboxed tool execution, monolithic codebase |
| **ZeroClaw** (Rust, ~4k LOC) | Trait-driven extensibility, single-binary deployment, <5MB RAM, sub-10ms boot | Minimal security model, no compile-time capability enforcement, no audit logging, limited tool sandboxing |
| **nanobot** (Rust, ~4k LOC) | Minimal footprint proof-of-concept | Too minimal for production security requirements |

---

## 2. Architecture & Workspace Layout

### Crate Topology

```
freebird/
├── Cargo.toml                       # Workspace root
├── deny.toml                        # cargo-deny: license + advisory audit
├── clippy.toml                      # Clippy configuration
├── rustfmt.toml                     # Formatting rules
├── config/
│   └── default.toml                 # Default configuration
├── crates/
│   ├── freebird-traits/                   # ALL public traits live here — zero freebird-* dependencies
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── provider.rs          # Provider trait
│   │       ├── channel.rs           # Channel trait
│   │       ├── tool.rs              # Tool trait
│   │       └── memory.rs            # Memory trait
│   ├── freebird-types/                    # Shared domain types — depends only on freebird-traits
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── message.rs           # Message, Turn, Role, Content enums
│   │       ├── session.rs           # SessionId, SessionState
│   │       ├── config.rs            # Typed config structs
│   │       └── id.rs                # Newtype IDs (SessionId, InvocationId, etc.)
│   ├── freebird-security/                 # Taint, SafePath, capabilities, audit — no I/O
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── taint.rs             # Tainted<T> type system
│   │       ├── paths.rs             # SafePath validation
│   │       ├── capability.rs        # CapabilityGrant system
│   │       ├── audit.rs             # Structured audit events
│   │       ├── injection.rs         # Prompt injection detection heuristics
│   │       └── error.rs             # SecurityError enum
│   ├── freebird-runtime/                  # Agent loop, session management, orchestration
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── agent.rs             # AgentRuntime — the core loop
│   │       ├── router.rs            # Channel → Agent message routing
│   │       ├── session.rs           # Session lifecycle manager
│   │       ├── registry.rs          # Dynamic provider/channel/tool registration
│   │       └── shutdown.rs          # Graceful shutdown coordinator
│   ├── freebird-providers/                # Provider implementations
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── anthropic.rs         # Anthropic (Claude) provider
│   │       └── openai.rs            # OpenAI provider (future)
│   ├── freebird-channels/                 # Channel implementations
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── cli.rs               # CLI (stdin/stdout) channel
│   │       ├── signal.rs            # Signal messenger channel (future)
│   │       └── ws.rs                # WebSocket channel (future)
│   ├── freebird-tools/                    # Built-in tool implementations
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── filesystem.rs        # Sandboxed file read/write
│   │       ├── shell.rs             # Sandboxed command execution
│   │       └── network.rs           # Sandboxed HTTP requests
│   ├── freebird-memory/                   # Memory backend implementations
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── file.rs              # File-based conversation storage
│   │       └── sqlite.rs            # SQLite backend (future)
│   └── freebird-daemon/                   # Binary — thin main.rs
│       └── src/
│           └── main.rs
```

### Dependency DAG (Strict — Enforced by Cargo)

```
freebird-traits          (zero freebird-* deps — the root of the type universe)
    ↑
freebird-types           (depends on: freebird-traits)
    ↑
freebird-security        (depends on: freebird-types, freebird-traits)
    ↑
freebird-runtime         (depends on: freebird-traits, freebird-types, freebird-security)
    ↑
freebird-providers       (depends on: freebird-traits, freebird-types)
freebird-channels        (depends on: freebird-traits, freebird-types, freebird-security)
freebird-tools           (depends on: freebird-traits, freebird-types, freebird-security)
freebird-memory          (depends on: freebird-traits, freebird-types)
    ↑
freebird-daemon          (depends on: ALL — this is the composition root)
```

**Critical rule**: `freebird-traits` has **ZERO** `freebird-*` dependencies. External deps (`async-trait`, `serde`, `thiserror`, etc.) are the minimum needed for trait signatures and associated types. This means any implementation can be swapped without rebuilding the core. `freebird-security` depends on **no other `freebird-*` crate except `freebird-types` and `freebird-traits`** — it cannot be compromised by a vulnerability in a provider or channel.

### Workspace Cargo.toml

```toml
[workspace]
resolver = "2"
members = ["crates/*"]

[workspace.package]
version = "0.1.0"
edition = "2024"
rust-version = "1.85"
license = "MIT OR Apache-2.0"

[workspace.dependencies]
# Async
tokio = { version = "1", features = ["full"] }
tokio-util = { version = "0.7", features = ["rt"] }
tokio-stream = "0.1"
futures = "0.3"
async-trait = "0.1"

# Serialization
serde = { version = "1", features = ["derive"] }
serde_json = "1"

# Errors
thiserror = "2"
anyhow = "1"

# Logging
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["json", "env-filter"] }

# Crypto & secrets
secrecy = { version = "0.10", features = ["serde"] }
ring = "0.17"
rustls = "0.23"

# HTTP client (for providers)
reqwest = { version = "0.12", default-features = false, features = ["rustls-tls", "json", "stream"] }

# Web framework (for WebSocket channel)
axum = { version = "0.8", features = ["ws"] }
tower = "0.5"
tower-http = { version = "0.6", features = ["trace", "cors", "request-id"] }

# Config
figment = { version = "0.10", features = ["toml", "env"] }

# Time
chrono = { version = "0.4", features = ["serde"] }

# IDs
uuid = { version = "1", features = ["v4", "serde"] }

# Testing
tempfile = "3"
wiremock = "0.6"
proptest = "1"
```

---

## 3. Core Principles

### 3.1 Trait-Driven Extensibility

Every subsystem (provider, channel, tool, memory) is defined as a trait in `freebird-traits`. Implementations live in separate crates. The runtime discovers and composes them via configuration. Adding a new provider or channel means writing a new struct that implements the trait — zero changes to existing code.

### 3.2 Make Illegal States Unrepresentable

```rust
// BAD — implicit states, boolean flags
struct AgentSession {
    is_authenticated: bool,
    is_connected: bool,
    provider: Option<Box<dyn Provider>>,
}

// GOOD — each state carries only its valid data
enum AgentSession {
    Unauthenticated,
    Authenticated { credential: SessionCredential },
    Connected {
        credential: SessionCredential,
        provider: Box<dyn Provider>,
        channel: Box<dyn Channel>,
    },
    Processing {
        credential: SessionCredential,
        provider: Box<dyn Provider>,
        channel: Box<dyn Channel>,
        current_turn: Turn,
    },
}
```

### 3.3 Parse, Don't Validate

Transform unstructured data into typed structures at system boundaries. Once parsed, validity is guaranteed by the type.

### 3.4 Fail Loudly at Boundaries, Propagate Gracefully Inside

All input validation happens at the transport edge. Internal code propagates errors with `?` and added `.context()`.

### 3.5 Zero `unsafe` in Application Code

The only acceptable `unsafe` is inside FFI wrappers with `// SAFETY:` comments. Application logic must be 100% safe Rust.

### 3.6 Ownership Communicates Intent

```rust
fn inspect(data: &Config)          // borrowing: I look but don't touch
fn mutate(data: &mut Config)       // exclusive borrow: I will change this
fn consume(data: Config)           // ownership transfer: caller loses access
fn produce() -> Config             // returning owned: caller now owns this
```

---

## 4. Message Flow & Data Architecture

### 4.1 The Complete Message Path

This is the heartbeat of the system. Every interaction follows this exact flow:

```
┌──────────┐     ┌───────────┐     ┌───────────┐     ┌──────────┐     ┌──────────┐
│ Channel   │────▶│ Router    │────▶│ Agent     │────▶│ Provider │────▶│ LLM API  │
│ (CLI,     │     │ (taint +  │     │ Runtime   │     │ (Anthropic│     │ (Claude, │
│  Signal)  │     │  auth)    │     │ (loop)    │     │  OpenAI)  │     │  GPT)    │
└──────────┘     └───────────┘     └───────────┘     └──────────┘     └──────────┘
     ▲                                   │                                    │
     │                                   ▼                                    │
     │                            ┌───────────┐                               │
     │                            │ Tool      │                               │
     │                            │ Executor  │◀──────────────────────────────┘
     │                            │ (sandbox) │   (tool_use blocks in response)
     │                            └───────────┘
     │                                   │
     │                                   ▼
     │                            ┌───────────┐
     │                            │ Memory    │
     │                            │ (persist) │
     │                            └───────────┘
     │                                   │
     └───────────────────────────────────┘
              (final response)
```

### 4.2 Core Message Types

These types flow through the entire system. They live in `freebird-types` and are the lingua franca between all subsystems.

```rust
// freebird-types/src/message.rs

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// The role of a participant in a conversation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    User,
    Assistant,
    System,
    Tool,
}

/// A single piece of content within a message.
/// Models like Claude support multi-part messages (text + images + tool results).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlock {
    Text {
        text: String,
    },
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    ToolResult {
        tool_use_id: String,
        content: String,
        is_error: bool,
    },
    Image {
        media_type: String,
        /// Base64-encoded image data
        data: String,
    },
}

/// A single message in a conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: Vec<ContentBlock>,
    pub timestamp: DateTime<Utc>,
}

/// A complete conversation turn: user message + assistant response + any tool calls.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Turn {
    pub user_message: Message,
    pub assistant_response: Option<Message>,
    pub tool_invocations: Vec<ToolInvocation>,
    pub started_at: DateTime<Utc>,
    pub completed_at: Option<DateTime<Utc>>,
}

/// Record of a single tool invocation within a turn.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolInvocation {
    pub tool_use_id: String,
    pub tool_name: String,
    pub input: serde_json::Value,
    pub output: Option<String>,
    pub is_error: bool,
    pub duration_ms: Option<u64>,
}

/// A complete conversation (ordered list of turns + metadata).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Conversation {
    pub session_id: SessionId,
    pub system_prompt: Option<String>,
    pub turns: Vec<Turn>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub model_id: String,
    pub provider_id: String,
}
```

### 4.3 Newtype IDs

All identifiers are newtyped to prevent mixing:

```rust
// freebird-types/src/id.rs

macro_rules! define_id {
    ($name:ident) => {
        #[derive(Debug, Clone, Hash, Eq, PartialEq, Serialize, Deserialize)]
        pub struct $name(String);

        impl $name {
            pub fn generate() -> Self {
                Self(uuid::Uuid::new_v4().to_string())
            }

            pub fn from_string(s: impl Into<String>) -> Self {
                Self(s.into())
            }

            pub fn as_str(&self) -> &str {
                &self.0
            }
        }

        impl std::fmt::Display for $name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "{}", self.0)
            }
        }
    };
}

define_id!(SessionId);
define_id!(InvocationId);
define_id!(ChannelId);
define_id!(ProviderId);
define_id!(ToolId);
```

---

# Part II — Core Agent Subsystems

---

## 5. Provider Trait & Model Abstraction

The Provider trait abstracts over LLM backends. Each provider (Anthropic, OpenAI, Ollama) implements this trait. The runtime doesn't know or care which model it's talking to — it only speaks through this interface.

### 5.1 Trait Definition

```rust
// freebird-traits/src/provider.rs

use async_trait::async_trait;
use std::pin::Pin;
use futures::Stream;

/// Metadata about a provider implementation.
#[derive(Debug, Clone)]
pub struct ProviderInfo {
    /// Unique identifier (e.g., "anthropic", "openai", "ollama").
    pub id: String,
    /// Human-readable name (e.g., "Anthropic Claude").
    pub display_name: String,
    /// Which models this provider supports.
    pub supported_models: Vec<ModelInfo>,
    /// Whether this provider supports streaming responses.
    pub supports_streaming: bool,
    /// Whether this provider supports tool use natively.
    pub supports_tool_use: bool,
    /// Whether this provider supports image input.
    pub supports_vision: bool,
}

/// Metadata about a specific model.
#[derive(Debug, Clone)]
pub struct ModelInfo {
    /// Model identifier sent to the API (e.g., "claude-opus-4-6-20250929").
    pub id: String,
    /// Human-readable name (e.g., "Claude Opus 4.6").
    pub display_name: String,
    /// Maximum context window in tokens.
    pub max_context_tokens: u32,
    /// Maximum output tokens.
    pub max_output_tokens: u32,
}

/// The input to a provider completion request.
#[derive(Debug, Clone)]
pub struct CompletionRequest {
    pub model: String,
    pub system_prompt: Option<String>,
    pub messages: Vec<Message>,
    pub tools: Vec<ToolDefinition>,
    pub max_tokens: u32,
    pub temperature: Option<f32>,
    pub stop_sequences: Vec<String>,
}

/// A complete (non-streaming) response from the provider.
#[derive(Debug, Clone)]
pub struct CompletionResponse {
    pub message: Message,
    pub stop_reason: StopReason,
    pub usage: TokenUsage,
    pub model: String,
}

/// Why the model stopped generating.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StopReason {
    EndTurn,
    ToolUse,
    MaxTokens,
    StopSequence,
}

/// Token usage for cost tracking.
#[derive(Debug, Clone, Default)]
pub struct TokenUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub cache_read_tokens: Option<u32>,
    pub cache_creation_tokens: Option<u32>,
}

/// A chunk of a streaming response.
#[derive(Debug, Clone)]
pub enum StreamEvent {
    /// A delta of text content.
    TextDelta(String),
    /// A complete tool use block (streamed tool use sends the full block at end).
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    /// The stream has ended.
    Done {
        stop_reason: StopReason,
        usage: TokenUsage,
    },
    /// An error occurred mid-stream.
    Error(String),
}

/// A tool definition sent to the provider so it knows what tools are available.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub input_schema: serde_json::Value,
}

/// The core provider trait. Every LLM backend implements this.
#[async_trait]
pub trait Provider: Send + Sync + 'static {
    /// Return metadata about this provider.
    fn info(&self) -> &ProviderInfo;

    /// Validate that the configured credentials are working.
    /// Called at startup and periodically to detect key expiry.
    async fn validate_credentials(&self) -> Result<(), ProviderError>;

    /// Send a completion request and get a full response.
    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse, ProviderError>;

    /// Send a completion request and get a streaming response.
    /// Returns a pinned stream of events.
    async fn stream(
        &self,
        request: CompletionRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamEvent, ProviderError>> + Send>>, ProviderError>;
}

/// Provider-specific errors.
#[derive(Debug, thiserror::Error)]
pub enum ProviderError {
    #[error("authentication failed: {reason}")]
    AuthenticationFailed { reason: String },

    #[error("rate limited, retry after {retry_after_ms}ms")]
    RateLimited { retry_after_ms: u64 },

    #[error("model `{model}` not found or not supported")]
    ModelNotFound { model: String },

    #[error("context window exceeded: {used} tokens used, {max} max")]
    ContextOverflow { used: u32, max: u32 },

    #[error("provider API error: {status} — {body}")]
    ApiError { status: u16, body: String },

    #[error("network error: {0}")]
    Network(#[from] reqwest::Error),

    #[error("deserialization error: {0}")]
    Deserialization(String),

    #[error("provider not configured")]
    NotConfigured,
}
```

### 5.2 Anthropic Implementation (First Provider)

```rust
// freebird-providers/src/anthropic.rs

use freebird_traits::provider::*;
use secrecy::{ExposeSecret, SecretString};
use reqwest::Client;

pub struct AnthropicProvider {
    client: Client,
    api_key: SecretString,
    base_url: String,
    default_model: String,
    info: ProviderInfo,
}

impl AnthropicProvider {
    pub fn new(api_key: SecretString, config: AnthropicConfig) -> Self {
        let info = ProviderInfo {
            id: "anthropic".into(),
            display_name: "Anthropic Claude".into(),
            supported_models: vec![
                ModelInfo {
                    id: "claude-opus-4-6-20250929".into(),
                    display_name: "Claude Opus 4.6".into(),
                    max_context_tokens: 200_000,
                    max_output_tokens: 32_768,
                },
                ModelInfo {
                    id: "claude-sonnet-4-5-20250929".into(),
                    display_name: "Claude Sonnet 4.5".into(),
                    max_context_tokens: 200_000,
                    max_output_tokens: 16_384,
                },
            ],
            supports_streaming: true,
            supports_tool_use: true,
            supports_vision: true,
        };

        Self {
            client: Client::builder()
                .use_rustls_tls()
                .timeout(std::time::Duration::from_secs(300))
                .build()
                .expect("failed to build HTTP client"),
            api_key,
            base_url: config.base_url.unwrap_or_else(|| "https://api.anthropic.com".into()),
            default_model: config.default_model.unwrap_or_else(|| "claude-opus-4-6-20250929".into()),
            info,
        }
    }
}

#[async_trait]
impl Provider for AnthropicProvider {
    fn info(&self) -> &ProviderInfo {
        &self.info
    }

    async fn validate_credentials(&self) -> Result<(), ProviderError> {
        // Lightweight request to verify the key works
        let resp = self.client
            .post(format!("{}/v1/messages", self.base_url))
            .header("x-api-key", self.api_key.expose_secret())
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json")
            .json(&serde_json::json!({
                "model": self.default_model,
                "max_tokens": 1,
                "messages": [{"role": "user", "content": "ping"}]
            }))
            .send()
            .await?;

        if resp.status() == 401 {
            return Err(ProviderError::AuthenticationFailed {
                reason: "invalid API key".into(),
            });
        }

        Ok(())
    }

    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse, ProviderError> {
        let body = self.build_request_body(&request);

        let resp = self.client
            .post(format!("{}/v1/messages", self.base_url))
            .header("x-api-key", self.api_key.expose_secret())
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json")
            .json(&body)
            .send()
            .await?;

        let status = resp.status().as_u16();

        if status == 429 {
            let retry_after = resp.headers()
                .get("retry-after")
                .and_then(|v| v.to_str().ok())
                .and_then(|v| v.parse::<u64>().ok())
                .unwrap_or(1000);
            return Err(ProviderError::RateLimited { retry_after_ms: retry_after });
        }

        if !resp.status().is_success() {
            let body = resp.text().await.unwrap_or_default();
            return Err(ProviderError::ApiError { status, body });
        }

        let api_response: AnthropicResponse = resp.json().await
            .map_err(|e| ProviderError::Deserialization(e.to_string()))?;

        Ok(api_response.into_completion_response())
    }

    async fn stream(
        &self,
        request: CompletionRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamEvent, ProviderError>> + Send>>, ProviderError> {
        let mut body = self.build_request_body(&request);
        body["stream"] = serde_json::Value::Bool(true);

        let resp = self.client
            .post(format!("{}/v1/messages", self.base_url))
            .header("x-api-key", self.api_key.expose_secret())
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json")
            .json(&body)
            .send()
            .await?;

        if !resp.status().is_success() {
            let status = resp.status().as_u16();
            let body = resp.text().await.unwrap_or_default();
            return Err(ProviderError::ApiError { status, body });
        }

        // Parse SSE stream into StreamEvent items
        let stream = self.parse_sse_stream(resp.bytes_stream());
        Ok(Box::pin(stream))
    }
}
```

### 5.3 Provider Registry & Failover

```rust
// freebird-runtime/src/registry.rs (provider portion)

pub struct ProviderRegistry {
    providers: HashMap<ProviderId, Box<dyn Provider>>,
    failover_chain: Vec<ProviderId>,
}

impl ProviderRegistry {
    /// Try the primary provider, then fall through the failover chain.
    pub async fn complete_with_failover(
        &self,
        request: CompletionRequest,
    ) -> Result<(ProviderId, CompletionResponse), ProviderError> {
        let mut last_error = None;

        for provider_id in &self.failover_chain {
            let provider = self.providers.get(provider_id)
                .ok_or(ProviderError::NotConfigured)?;

            match provider.complete(request.clone()).await {
                Ok(response) => return Ok((provider_id.clone(), response)),
                Err(e @ ProviderError::AuthenticationFailed { .. }) => return Err(e),
                Err(e) => {
                    tracing::warn!(
                        provider = %provider_id,
                        error = %e,
                        "provider failed, trying next in failover chain"
                    );
                    last_error = Some(e);
                }
            }
        }

        Err(last_error.unwrap_or(ProviderError::NotConfigured))
    }
}
```

---

## 6. Channel Trait & Transport Abstraction

A Channel is a bidirectional communication interface between the user and the agent. CLI reads from stdin and writes to stdout. Signal would bridge to signal-cli. WebSocket would accept connections via axum. They all look identical to the runtime.

### 6.1 Trait Definition

```rust
// freebird-traits/src/channel.rs

use async_trait::async_trait;
use tokio_stream::Stream;
use std::pin::Pin;

/// Metadata about a channel implementation.
#[derive(Debug, Clone)]
pub struct ChannelInfo {
    /// Unique identifier (e.g., "cli", "signal", "websocket").
    pub id: String,
    /// Human-readable name (e.g., "Command Line Interface").
    pub display_name: String,
    /// Whether this channel supports rich content (images, files).
    pub supports_media: bool,
    /// Whether this channel supports real-time streaming of responses.
    pub supports_streaming: bool,
    /// Whether this channel requires authentication/pairing before use.
    pub requires_auth: bool,
}

/// An inbound event from a channel.
#[derive(Debug, Clone)]
pub enum InboundEvent {
    /// A new message from the user.
    Message {
        /// Raw text from the user (will be wrapped in Tainted<Untrusted> by the router).
        raw_text: String,
        /// Channel-specific sender identifier (e.g., phone number, username).
        sender_id: String,
        /// Optional media attachments.
        attachments: Vec<Attachment>,
    },
    /// The user has connected/started a session.
    Connected { sender_id: String },
    /// The user has disconnected.
    Disconnected { sender_id: String },
    /// A control command (e.g., /new, /status, /model).
    Command { name: String, args: Vec<String>, sender_id: String },
}

/// An outbound event to send to the user via the channel.
#[derive(Debug, Clone)]
pub enum OutboundEvent {
    /// A complete text response.
    Message { text: String, recipient_id: String },
    /// A streaming text chunk (for channels that support it).
    StreamChunk { text: String, recipient_id: String },
    /// Signal that streaming is complete.
    StreamEnd { recipient_id: String },
    /// An error message to display to the user.
    Error { text: String, recipient_id: String },
}

/// A media attachment (image, file, audio).
#[derive(Debug, Clone)]
pub struct Attachment {
    pub filename: String,
    pub media_type: String,
    pub data: Vec<u8>,
}

/// The core channel trait.
///
/// Lifecycle: `start()` is called once. It returns a stream of inbound events
/// and a sender for outbound events. The runtime consumes the stream and
/// sends responses via the sender. `stop()` is called during shutdown.
#[async_trait]
pub trait Channel: Send + Sync + 'static {
    /// Return metadata about this channel.
    fn info(&self) -> &ChannelInfo;

    /// Start the channel. Returns:
    /// - A stream of inbound events (messages from users).
    /// - A sender to push outbound events (responses to users).
    ///
    /// This is the key design: the channel owns its own event loop internally
    /// (reading stdin, listening on a socket, polling Signal). It exposes a
    /// uniform stream/sender interface to the runtime.
    async fn start(&self) -> Result<ChannelHandle, ChannelError>;

    /// Gracefully stop the channel, closing connections and flushing buffers.
    async fn stop(&self) -> Result<(), ChannelError>;
}

/// The handle returned by Channel::start().
pub struct ChannelHandle {
    /// Stream of inbound events from the user.
    pub inbound: Pin<Box<dyn Stream<Item = InboundEvent> + Send>>,
    /// Sender for outbound events to the user.
    pub outbound: tokio::sync::mpsc::Sender<OutboundEvent>,
}

/// Channel-specific errors.
#[derive(Debug, thiserror::Error)]
pub enum ChannelError {
    #[error("channel `{channel}` failed to start: {reason}")]
    StartupFailed { channel: String, reason: String },

    #[error("channel `{channel}` connection lost: {reason}")]
    ConnectionLost { channel: String, reason: String },

    #[error("failed to send message on channel `{channel}`: {reason}")]
    SendFailed { channel: String, reason: String },

    #[error("channel `{channel}` authentication failed")]
    AuthenticationFailed { channel: String },

    #[error("channel IO error: {0}")]
    Io(#[from] std::io::Error),
}
```

### 6.2 CLI Channel Implementation (First Channel)

```rust
// freebird-channels/src/cli.rs

use freebird_traits::channel::*;
use tokio::io::{self, AsyncBufReadExt, BufReader};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

pub struct CliChannel {
    info: ChannelInfo,
    prompt: String,
}

impl CliChannel {
    pub fn new(config: CliConfig) -> Self {
        Self {
            info: ChannelInfo {
                id: "cli".into(),
                display_name: "Command Line Interface".into(),
                supports_media: false,
                supports_streaming: true,
                requires_auth: false,
            },
            prompt: config.prompt.unwrap_or_else(|| "you> ".into()),
        }
    }
}

#[async_trait]
impl Channel for CliChannel {
    fn info(&self) -> &ChannelInfo {
        &self.info
    }

    async fn start(&self) -> Result<ChannelHandle, ChannelError> {
        let (inbound_tx, inbound_rx) = mpsc::channel::<InboundEvent>(32);
        let (outbound_tx, mut outbound_rx) = mpsc::channel::<OutboundEvent>(32);

        let prompt = self.prompt.clone();

        // Spawn stdin reader task
        tokio::spawn(async move {
            let stdin = BufReader::new(io::stdin());
            let mut lines = stdin.lines();

            // Send initial connected event
            let _ = inbound_tx.send(InboundEvent::Connected {
                sender_id: "local".into(),
            }).await;

            loop {
                // Print prompt (using stderr so it doesn't mix with piped output)
                eprint!("{prompt}");

                match lines.next_line().await {
                    Ok(Some(line)) => {
                        let trimmed = line.trim().to_string();
                        if trimmed.is_empty() {
                            continue;
                        }

                        // Parse /commands
                        let event = if let Some(cmd) = trimmed.strip_prefix('/') {
                            let parts: Vec<&str> = cmd.splitn(2, ' ').collect();
                            InboundEvent::Command {
                                name: parts[0].to_string(),
                                args: parts.get(1)
                                    .map(|a| a.split_whitespace().map(String::from).collect())
                                    .unwrap_or_default(),
                                sender_id: "local".into(),
                            }
                        } else {
                            InboundEvent::Message {
                                raw_text: trimmed,
                                sender_id: "local".into(),
                                attachments: vec![],
                            }
                        };

                        if inbound_tx.send(event).await.is_err() {
                            break; // runtime shut down
                        }
                    }
                    Ok(None) => {
                        // EOF (stdin closed, e.g., piped input ended)
                        let _ = inbound_tx.send(InboundEvent::Disconnected {
                            sender_id: "local".into(),
                        }).await;
                        break;
                    }
                    Err(e) => {
                        tracing::error!(error = %e, "error reading stdin");
                        break;
                    }
                }
            }
        });

        // Spawn stdout writer task
        tokio::spawn(async move {
            while let Some(event) = outbound_rx.recv().await {
                match event {
                    OutboundEvent::Message { text, .. } => {
                        println!("{text}");
                    }
                    OutboundEvent::StreamChunk { text, .. } => {
                        print!("{text}");
                        // Flush to ensure streaming output appears immediately
                        use std::io::Write;
                        let _ = std::io::stdout().flush();
                    }
                    OutboundEvent::StreamEnd { .. } => {
                        println!(); // newline after stream
                    }
                    OutboundEvent::Error { text, .. } => {
                        eprintln!("error: {text}");
                    }
                }
            }
        });

        Ok(ChannelHandle {
            inbound: Box::pin(ReceiverStream::new(inbound_rx)),
            outbound: outbound_tx,
        })
    }

    async fn stop(&self) -> Result<(), ChannelError> {
        // CLI channel: nothing to clean up
        Ok(())
    }
}
```

### 6.3 Channel Extensibility Pattern

Adding Signal later means implementing the same trait:

```rust
// freebird-channels/src/signal.rs (future)

pub struct SignalChannel {
    info: ChannelInfo,
    signal_cli_path: PathBuf,
    phone_number: String,
    // ...
}

#[async_trait]
impl Channel for SignalChannel {
    fn info(&self) -> &ChannelInfo { &self.info }

    async fn start(&self) -> Result<ChannelHandle, ChannelError> {
        // Spawn signal-cli in JSON-RPC mode
        // Bridge its events to InboundEvent stream
        // Bridge OutboundEvent sender to signal-cli send commands
        todo!()
    }

    async fn stop(&self) -> Result<(), ChannelError> {
        // Kill signal-cli subprocess gracefully
        todo!()
    }
}
```

The runtime requires **zero changes** to support this — it just calls `channel.start()` and processes the stream.

---

## 7. Tool Trait & Execution Sandbox

Tools are how the agent acts on the world. Every tool invocation is gated by the capability system, sandboxed, and audit-logged.

### 7.1 Trait Definition

```rust
// freebird-traits/src/tool.rs

use async_trait::async_trait;

/// Metadata describing a tool for both the runtime and the LLM.
#[derive(Debug, Clone)]
pub struct ToolInfo {
    /// Unique name (matches what the LLM will call, e.g., "read_file").
    pub name: String,
    /// Human-readable description sent to the LLM.
    pub description: String,
    /// JSON Schema for the tool's input parameters.
    pub input_schema: serde_json::Value,
    /// Which capability is required to invoke this tool.
    pub required_capability: Capability,
    /// Whether this tool performs I/O (affects sandboxing decisions).
    pub has_side_effects: bool,
}

/// The result of a tool execution.
#[derive(Debug, Clone)]
pub struct ToolOutput {
    pub content: String,
    pub is_error: bool,
    pub metadata: Option<serde_json::Value>,
}

/// The core tool trait.
#[async_trait]
pub trait Tool: Send + Sync + 'static {
    /// Return metadata about this tool.
    fn info(&self) -> &ToolInfo;

    /// Convert this tool's info into the ToolDefinition format
    /// that gets sent to the provider.
    fn to_definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: self.info().name.clone(),
            description: self.info().description.clone(),
            input_schema: self.info().input_schema.clone(),
        }
    }

    /// Execute the tool. The runtime has already verified:
    /// 1. The capability grant allows this tool.
    /// 2. The input has been taint-checked.
    /// This method performs the actual work.
    async fn execute(
        &self,
        input: serde_json::Value,
        context: &ToolContext,
    ) -> Result<ToolOutput, ToolError>;
}

/// Context passed to every tool invocation.
pub struct ToolContext {
    pub session_id: SessionId,
    pub grant: CapabilityGrant,
    pub sandbox_root: SafePath,
    pub audit: AuditLogger,
}

#[derive(Debug, thiserror::Error)]
pub enum ToolError {
    #[error("tool `{tool}` execution failed: {reason}")]
    ExecutionFailed { tool: String, reason: String },

    #[error("tool `{tool}` timed out after {timeout_ms}ms")]
    Timeout { tool: String, timeout_ms: u64 },

    #[error("tool `{tool}` input validation failed: {reason}")]
    InvalidInput { tool: String, reason: String },

    #[error("security violation in tool `{tool}`: {0}")]
    SecurityViolation(String, #[source] SecurityError),
}
```

### 7.2 Tool Executor (The Security Boundary)

The `ToolExecutor` sits between the runtime and individual tools. It enforces the security invariants **before** any tool code runs.

```rust
// freebird-runtime/src/tool_executor.rs

pub struct ToolExecutor {
    tools: HashMap<String, Box<dyn Tool>>,
    default_timeout: Duration,
}

impl ToolExecutor {
    /// Execute a tool call from the LLM, enforcing all security checks.
    pub async fn execute(
        &self,
        tool_name: &str,
        input: serde_json::Value,
        grant: &CapabilityGrant,
        session_id: &SessionId,
        audit: &AuditLogger,
    ) -> ToolOutput {
        // 1. Look up the tool
        let tool = match self.tools.get(tool_name) {
            Some(t) => t,
            None => {
                audit.record(AuditEvent::ToolInvocation {
                    session_id: session_id.to_string(),
                    tool_name: tool_name.into(),
                    capability_check: CapabilityCheckResult::Denied {
                        reason: "tool not found".into(),
                    },
                    timestamp: Utc::now(),
                });
                return ToolOutput {
                    content: format!("Error: tool `{tool_name}` not found"),
                    is_error: true,
                    metadata: None,
                };
            }
        };

        // 2. Check capability
        if let Err(e) = grant.check(&tool.info().required_capability) {
            audit.record(AuditEvent::ToolInvocation {
                session_id: session_id.to_string(),
                tool_name: tool_name.into(),
                capability_check: CapabilityCheckResult::Denied {
                    reason: format!("{e}"),
                },
                timestamp: Utc::now(),
            });
            return ToolOutput {
                content: format!("Error: insufficient capability — {e}"),
                is_error: true,
                metadata: None,
            };
        }

        audit.record(AuditEvent::ToolInvocation {
            session_id: session_id.to_string(),
            tool_name: tool_name.into(),
            capability_check: CapabilityCheckResult::Granted,
            timestamp: Utc::now(),
        });

        // 3. Execute with timeout
        let context = ToolContext {
            session_id: session_id.clone(),
            grant: grant.clone(),
            sandbox_root: grant.sandbox_root().clone(),
            audit: audit.clone(),
        };

        match tokio::time::timeout(
            self.default_timeout,
            tool.execute(input, &context),
        ).await {
            Ok(Ok(output)) => output,
            Ok(Err(e)) => ToolOutput {
                content: format!("Error: {e}"),
                is_error: true,
                metadata: None,
            },
            Err(_) => {
                audit.record(AuditEvent::PolicyViolation {
                    session_id: session_id.to_string(),
                    rule: "tool_timeout".into(),
                    context: format!("tool `{tool_name}` exceeded {:.1}s timeout", self.default_timeout.as_secs_f64()),
                    severity: Severity::Medium,
                    timestamp: Utc::now(),
                });
                ToolOutput {
                    content: format!("Error: tool `{tool_name}` timed out"),
                    is_error: true,
                    metadata: None,
                }
            }
        }
    }
}
```

---

## 8. Memory Trait & Conversation State

Memory persists conversations across turns and sessions. File-based first (plain JSON files like ZeroClaw/FreeBird), extensible to SQLite or vector stores.

### 8.1 Trait Definition

```rust
// freebird-traits/src/memory.rs

use async_trait::async_trait;

/// The core memory trait for persisting conversations.
#[async_trait]
pub trait Memory: Send + Sync + 'static {
    /// Load a conversation by session ID. Returns None if not found.
    async fn load(&self, session_id: &SessionId) -> Result<Option<Conversation>, MemoryError>;

    /// Save/update a conversation.
    async fn save(&self, conversation: &Conversation) -> Result<(), MemoryError>;

    /// List all session IDs, most recent first.
    async fn list_sessions(&self, limit: usize) -> Result<Vec<SessionSummary>, MemoryError>;

    /// Delete a conversation by session ID.
    async fn delete(&self, session_id: &SessionId) -> Result<(), MemoryError>;

    /// Search conversations by content (for future semantic search).
    async fn search(&self, query: &str, limit: usize) -> Result<Vec<SessionSummary>, MemoryError>;
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionSummary {
    pub session_id: SessionId,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub turn_count: usize,
    pub model_id: String,
    /// First ~100 chars of the first user message, for display.
    pub preview: String,
}

#[derive(Debug, thiserror::Error)]
pub enum MemoryError {
    #[error("session `{session_id}` not found")]
    NotFound { session_id: String },

    #[error("storage I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("serialization error: {0}")]
    Serialization(String),

    #[error("storage is read-only")]
    ReadOnly,
}
```

### 8.2 File Memory Implementation

```rust
// freebird-memory/src/file.rs

use std::path::PathBuf;
use tokio::fs;

pub struct FileMemory {
    base_dir: PathBuf,
}

impl FileMemory {
    pub async fn new(base_dir: PathBuf) -> Result<Self, MemoryError> {
        fs::create_dir_all(&base_dir).await?;
        Ok(Self { base_dir })
    }

    fn session_path(&self, session_id: &SessionId) -> PathBuf {
        self.base_dir.join(format!("{}.json", session_id.as_str()))
    }
}

#[async_trait]
impl Memory for FileMemory {
    async fn load(&self, session_id: &SessionId) -> Result<Option<Conversation>, MemoryError> {
        let path = self.session_path(session_id);
        if !path.exists() {
            return Ok(None);
        }

        let data = fs::read_to_string(&path).await?;
        let conversation: Conversation = serde_json::from_str(&data)
            .map_err(|e| MemoryError::Serialization(e.to_string()))?;

        Ok(Some(conversation))
    }

    async fn save(&self, conversation: &Conversation) -> Result<(), MemoryError> {
        let path = self.session_path(&conversation.session_id);
        let data = serde_json::to_string_pretty(conversation)
            .map_err(|e| MemoryError::Serialization(e.to_string()))?;

        // Atomic write: write to temp file, then rename
        let temp_path = path.with_extension("json.tmp");
        fs::write(&temp_path, &data).await?;
        fs::rename(&temp_path, &path).await?;

        Ok(())
    }

    async fn list_sessions(&self, limit: usize) -> Result<Vec<SessionSummary>, MemoryError> {
        let mut entries = Vec::new();
        let mut dir = fs::read_dir(&self.base_dir).await?;

        while let Some(entry) = dir.next_entry().await? {
            let path = entry.path();
            if path.extension().map_or(false, |e| e == "json") {
                if let Ok(data) = fs::read_to_string(&path).await {
                    if let Ok(conv) = serde_json::from_str::<Conversation>(&data) {
                        entries.push(SessionSummary {
                            session_id: conv.session_id.clone(),
                            created_at: conv.created_at,
                            updated_at: conv.updated_at,
                            turn_count: conv.turns.len(),
                            model_id: conv.model_id.clone(),
                            preview: conv.turns.first()
                                .and_then(|t| t.user_message.content.first())
                                .map(|c| match c {
                                    ContentBlock::Text { text } => text.chars().take(100).collect(),
                                    _ => "[non-text]".into(),
                                })
                                .unwrap_or_default(),
                        });
                    }
                }
            }
        }

        entries.sort_by(|a, b| b.updated_at.cmp(&a.updated_at));
        entries.truncate(limit);

        Ok(entries)
    }

    async fn delete(&self, session_id: &SessionId) -> Result<(), MemoryError> {
        let path = self.session_path(session_id);
        if path.exists() {
            fs::remove_file(&path).await?;
        }
        Ok(())
    }

    async fn search(&self, _query: &str, _limit: usize) -> Result<Vec<SessionSummary>, MemoryError> {
        // Basic implementation: linear scan with substring matching.
        // Future: replace with SQLite FTS or vector similarity.
        Ok(vec![])
    }
}
```

---

## 9. Agent Runtime Loop

This is the central orchestrator — the beating heart of FreeBird. It wires channels, providers, tools, and memory together into the agentic loop.

### 9.1 The Runtime

```rust
// freebird-runtime/src/agent.rs

use tokio::sync::mpsc;
use tokio_stream::StreamExt;

pub struct AgentRuntime {
    provider_registry: ProviderRegistry,
    tool_executor: ToolExecutor,
    memory: Box<dyn Memory>,
    config: RuntimeConfig,
    audit: AuditLogger,
    shutdown: ShutdownCoordinator,
}

impl AgentRuntime {
    /// Build the runtime from configuration.
    pub async fn build(config: AppConfig) -> Result<Self> {
        let audit = AuditLogger::new(&config.logging)?;

        // Build provider registry
        let mut provider_registry = ProviderRegistry::new();
        for provider_config in &config.providers {
            let provider = build_provider(provider_config)?;
            provider.validate_credentials().await
                .context(format!("credentials invalid for provider `{}`", provider.info().id))?;
            provider_registry.register(provider);
        }
        provider_registry.set_failover_chain(config.provider_failover_chain.clone());

        // Build tool executor
        let tool_executor = ToolExecutor::new(
            build_tools(&config.tools, &config.security)?,
            config.tools.default_timeout,
        );

        // Build memory backend
        let memory = build_memory(&config.memory).await?;

        Ok(Self {
            provider_registry,
            tool_executor,
            memory,
            config: config.runtime,
            audit,
            shutdown: ShutdownCoordinator::new(config.runtime.drain_timeout),
        })
    }

    /// Run the agent, listening on all configured channels until shutdown.
    pub async fn run(self, channels: Vec<Box<dyn Channel>>) -> Result<()> {
        let (event_tx, mut event_rx) = mpsc::channel::<RouterEvent>(256);

        // Start all channels and fan their inbound events into a single stream
        let mut channel_senders: HashMap<String, mpsc::Sender<OutboundEvent>> = HashMap::new();

        for channel in &channels {
            let handle = channel.start().await
                .context(format!("failed to start channel `{}`", channel.info().id))?;

            channel_senders.insert(channel.info().id.clone(), handle.outbound);

            let channel_id = channel.info().id.clone();
            let tx = event_tx.clone();

            // Fan-in: forward this channel's inbound events to the central router
            tokio::spawn(async move {
                let mut inbound = handle.inbound;
                while let Some(event) = inbound.next().await {
                    if tx.send(RouterEvent {
                        channel_id: channel_id.clone(),
                        event,
                    }).await.is_err() {
                        break;
                    }
                }
            });

            self.audit.record(AuditEvent::SessionStarted {
                session_id: "system".into(),
                capabilities: vec![format!("channel:{}", channel.info().id)],
                timestamp: Utc::now(),
            });

            tracing::info!(channel = %channel.info().id, "channel started");
        }

        drop(event_tx); // drop our copy so the loop ends when all channels close

        // The main agent loop
        tracing::info!("agent runtime started, waiting for messages");

        loop {
            tokio::select! {
                Some(router_event) = event_rx.recv() => {
                    let outbound_tx = channel_senders
                        .get(&router_event.channel_id)
                        .cloned();

                    if let Some(tx) = outbound_tx {
                        self.handle_event(router_event, tx).await;
                    }
                }
                _ = self.shutdown.wait_for_signal() => {
                    tracing::info!("shutdown signal received, draining");
                    break;
                }
            }
        }

        // Graceful shutdown: stop all channels
        for channel in &channels {
            if let Err(e) = channel.stop().await {
                tracing::warn!(channel = %channel.info().id, error = %e, "error stopping channel");
            }
        }

        Ok(())
    }

    /// Handle a single inbound event — the core agent turn.
    async fn handle_event(&self, event: RouterEvent, outbound: mpsc::Sender<OutboundEvent>) {
        match event.event {
            InboundEvent::Message { raw_text, sender_id, attachments } => {
                self.handle_message(
                    &event.channel_id,
                    &sender_id,
                    raw_text,
                    attachments,
                    &outbound,
                ).await;
            }
            InboundEvent::Command { name, args, sender_id } => {
                self.handle_command(&name, &args, &sender_id, &outbound).await;
            }
            InboundEvent::Connected { sender_id } => {
                tracing::info!(channel = %event.channel_id, sender = %sender_id, "user connected");
            }
            InboundEvent::Disconnected { sender_id } => {
                tracing::info!(channel = %event.channel_id, sender = %sender_id, "user disconnected");
            }
        }
    }

    /// The agentic loop for a single user message.
    /// This is where tool use happens: the LLM may request multiple
    /// tool calls before producing a final response.
    async fn handle_message(
        &self,
        channel_id: &str,
        sender_id: &str,
        raw_text: String,
        _attachments: Vec<Attachment>,
        outbound: &mpsc::Sender<OutboundEvent>,
    ) {
        // 1. Taint the input
        let tainted = Tainted::<Untrusted>::new(&raw_text);

        // 2. Sanitize
        let sanitized = match tainted.sanitize(validators::user_message) {
            Ok(clean) => clean,
            Err(e) => {
                self.audit.record(AuditEvent::PolicyViolation {
                    session_id: sender_id.into(),
                    rule: "input_validation".into(),
                    context: format!("{e}"),
                    severity: Severity::Medium,
                    timestamp: Utc::now(),
                });
                let _ = outbound.send(OutboundEvent::Error {
                    text: "Message rejected by input validation.".into(),
                    recipient_id: sender_id.into(),
                }).await;
                return;
            }
        };

        // 3. Load or create conversation
        let session_id = self.resolve_session(channel_id, sender_id).await;
        let mut conversation = self.memory.load(&session_id).await
            .ok()
            .flatten()
            .unwrap_or_else(|| Conversation::new(
                session_id.clone(),
                self.config.system_prompt.clone(),
                self.config.default_model.clone(),
                self.config.default_provider.clone(),
            ));

        // 4. Append user message
        let user_message = Message {
            role: Role::User,
            content: vec![ContentBlock::Text { text: sanitized.value().to_string() }],
            timestamp: Utc::now(),
        };

        // 5. Build completion request
        let tool_definitions: Vec<ToolDefinition> = self.tool_executor
            .available_tools()
            .map(|t| t.to_definition())
            .collect();

        let mut messages = conversation.to_api_messages();
        messages.push(user_message.clone());

        let mut current_turn = Turn {
            user_message,
            assistant_response: None,
            tool_invocations: vec![],
            started_at: Utc::now(),
            completed_at: None,
        };

        // 6. THE AGENTIC LOOP — keep going until the model produces a final response
        let grant = self.resolve_capability_grant(sender_id).await;
        let max_tool_rounds = self.config.max_tool_rounds.unwrap_or(10);

        for round in 0..max_tool_rounds {
            let request = CompletionRequest {
                model: conversation.model_id.clone(),
                system_prompt: conversation.system_prompt.clone(),
                messages: messages.clone(),
                tools: tool_definitions.clone(),
                max_tokens: self.config.max_output_tokens,
                temperature: self.config.temperature,
                stop_sequences: vec![],
            };

            // Call the provider
            let (provider_id, response) = match self.provider_registry
                .complete_with_failover(request).await
            {
                Ok(r) => r,
                Err(e) => {
                    tracing::error!(error = %e, "all providers failed");
                    let _ = outbound.send(OutboundEvent::Error {
                        text: format!("Provider error: {e}"),
                        recipient_id: sender_id.into(),
                    }).await;
                    return;
                }
            };

            // Check if the model wants to use tools
            match response.stop_reason {
                StopReason::ToolUse => {
                    // Extract tool_use blocks from the response
                    let tool_uses: Vec<_> = response.message.content.iter()
                        .filter_map(|block| match block {
                            ContentBlock::ToolUse { id, name, input } => {
                                Some((id.clone(), name.clone(), input.clone()))
                            }
                            _ => None,
                        })
                        .collect();

                    // Add assistant message with tool_use to the conversation
                    messages.push(response.message.clone());

                    // Execute each tool and collect results
                    let mut tool_results = Vec::new();
                    for (tool_use_id, tool_name, input) in tool_uses {
                        let start = std::time::Instant::now();

                        let output = self.tool_executor.execute(
                            &tool_name, input.clone(), &grant, &session_id, &self.audit,
                        ).await;

                        let duration_ms = start.elapsed().as_millis() as u64;

                        current_turn.tool_invocations.push(ToolInvocation {
                            tool_use_id: tool_use_id.clone(),
                            tool_name: tool_name.clone(),
                            input,
                            output: Some(output.content.clone()),
                            is_error: output.is_error,
                            duration_ms: Some(duration_ms),
                        });

                        tool_results.push(ContentBlock::ToolResult {
                            tool_use_id,
                            content: output.content,
                            is_error: output.is_error,
                        });
                    }

                    // Add tool results as a user-role message (Anthropic API format)
                    messages.push(Message {
                        role: Role::User,
                        content: tool_results,
                        timestamp: Utc::now(),
                    });

                    // Continue the loop — let the model decide what to do next
                    tracing::debug!(round, "tool round completed, continuing agentic loop");
                }
                StopReason::EndTurn | StopReason::StopSequence => {
                    // Model produced a final response — send it to the user
                    let response_text = response.message.content.iter()
                        .filter_map(|block| match block {
                            ContentBlock::Text { text } => Some(text.as_str()),
                            _ => None,
                        })
                        .collect::<Vec<_>>()
                        .join("");

                    // 7. Prompt injection check on model output
                    if let Err(e) = injection::scan_output(&response_text) {
                        self.audit.record(AuditEvent::PolicyViolation {
                            session_id: session_id.to_string(),
                            rule: "output_injection_scan".into(),
                            context: format!("{e}"),
                            severity: Severity::High,
                            timestamp: Utc::now(),
                        });
                        // Still deliver but flag it
                        tracing::warn!("potential prompt injection detected in model output");
                    }

                    let _ = outbound.send(OutboundEvent::Message {
                        text: response_text,
                        recipient_id: sender_id.into(),
                    }).await;

                    current_turn.assistant_response = Some(response.message);
                    current_turn.completed_at = Some(Utc::now());
                    break;
                }
                StopReason::MaxTokens => {
                    // Truncated response — deliver what we have
                    let partial_text = response.message.content.iter()
                        .filter_map(|block| match block {
                            ContentBlock::Text { text } => Some(text.as_str()),
                            _ => None,
                        })
                        .collect::<Vec<_>>()
                        .join("");

                    let _ = outbound.send(OutboundEvent::Message {
                        text: format!("{partial_text}\n\n[response truncated — max tokens reached]"),
                        recipient_id: sender_id.into(),
                    }).await;

                    current_turn.assistant_response = Some(response.message);
                    current_turn.completed_at = Some(Utc::now());
                    break;
                }
            }
        }

        // 8. Persist the conversation
        conversation.turns.push(current_turn);
        conversation.updated_at = Utc::now();

        if let Err(e) = self.memory.save(&conversation).await {
            tracing::error!(error = %e, "failed to persist conversation");
        }
    }
}

/// An event from a channel, tagged with its source.
struct RouterEvent {
    channel_id: String,
    event: InboundEvent,
}
```

---

# Part III — Security

---

## 10. Type-Driven Security Patterns

### 10.1 Taint Tracking System

All data from external sources MUST be wrapped in `Tainted<Untrusted>`. The type system prevents unsanitized data from reaching business logic.

```rust
// freebird-security/src/taint.rs

use std::marker::PhantomData;

pub trait TrustLevel: Send + Sync + 'static {}

pub struct Untrusted;
impl TrustLevel for Untrusted {}

pub struct Sanitized;
impl TrustLevel for Sanitized {}

#[derive(Debug, Clone)]
pub struct Tainted<T: TrustLevel> {
    inner: String,
    _trust: PhantomData<T>,
}

impl Tainted<Untrusted> {
    pub fn new(raw: impl Into<String>) -> Self {
        Self { inner: raw.into(), _trust: PhantomData }
    }

    pub fn sanitize<F>(self, validator: F) -> Result<Tainted<Sanitized>, SecurityError>
    where
        F: FnOnce(&str) -> Result<String, SecurityError>,
    {
        let clean = validator(&self.inner)?;
        Ok(Tainted { inner: clean, _trust: PhantomData })
    }

    /// Read-only access for logging/diagnostics ONLY.
    pub fn raw_for_logging(&self) -> &str {
        &self.inner
    }
}

impl Tainted<Sanitized> {
    pub fn value(&self) -> &str {
        &self.inner
    }
}
```

**Rules**: Never implement `Deref` or `AsRef<str>` on `Tainted<Untrusted>`. Never bypass `sanitize()`.

### 10.2 Safe Path System

```rust
// freebird-security/src/paths.rs

#[derive(Debug, Clone)]
pub struct SafePath {
    resolved: PathBuf,
    root: PathBuf,
}

impl SafePath {
    pub fn new(root: impl AsRef<Path>, user_path: impl AsRef<Path>) -> Result<Self, SecurityError> {
        let root = root.as_ref().canonicalize()
            .map_err(|e| SecurityError::PathResolution { path: root.as_ref().to_owned(), source: e })?;
        let candidate = root.join(user_path.as_ref());
        let resolved = candidate.canonicalize()
            .map_err(|e| SecurityError::PathResolution { path: candidate, source: e })?;

        if !resolved.starts_with(&root) {
            return Err(SecurityError::PathTraversal { attempted: resolved, sandbox: root });
        }

        Ok(Self { resolved, root })
    }

    pub fn as_path(&self) -> &Path { &self.resolved }
    pub fn root(&self) -> &Path { &self.root }
}
```

### 10.3 Output Taint Types

Tool output and model responses flow through safe types that enforce injection scanning at construction time. Unlike `SafeMessage` (which validates user input), these types enforce output-side taint boundaries — preventing indirect prompt injection from poisoning the LLM context or user-facing responses.

```rust
// freebird-security/src/safe_types.rs

/// Scanned tool output — blocks injection before content enters the LLM context.
/// Prevents indirect prompt injection via tool results (e.g., a file containing
/// "ignore previous instructions").
#[derive(Debug)]
pub struct ScannedToolOutput(String);

impl ScannedToolOutput {
    pub fn from_raw(content: &str) -> Result<Self, SecurityError> {
        injection::scan_output(content)?;
        Ok(Self(content.to_owned()))
    }

    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Scanned model response — blocks injection before delivery to the user.
/// Prevents compromised model output from reaching the channel. When blocked,
/// the tainted response is NOT persisted to memory (prevents memory poisoning).
#[derive(Debug)]
pub struct ScannedModelResponse(String);

impl ScannedModelResponse {
    pub fn from_raw(content: &str) -> Result<Self, SecurityError> {
        injection::scan_output(content)?;
        Ok(Self(content.to_owned()))
    }

    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}
```

**Enforcement boundary**: Because `freebird-traits` cannot depend on `freebird-security`, these types are enforced in `freebird-runtime` (the agentic loop), not at the trait level. The `Tool` trait returns raw `ToolOutput`; the runtime wraps it in `ScannedToolOutput::from_raw()` before passing content to the LLM.

**Behavior on detection**:
- **Tool output injection**: Blocked. A synthetic error `ToolResult { is_error: true }` replaces the raw content. The agentic loop continues — the LLM sees the error and can retry or respond without the tainted content.
- **Model output injection**: Blocked. `OutboundEvent::Error` sent to the user. The tainted response is NOT saved as `assistant_response` in the turn, preventing memory poisoning.

---

## 11. Agentic Security Model

### 11.1 Capability System

(Same as previous guide — see CapabilityGrant, Capability enum, derive_for_sub_agent.)

### 11.2 Tool Execution Invariant

Every tool invocation follows this **mandatory** sequence, enforced by `ToolExecutor`:

1. Capability check → deny if missing
2. Input validation → reject if malformed
3. Resource boundary check (SafePath, allowed hosts) → reject if out of bounds
4. Audit log the attempt (pass or fail)
5. Execute with timeout
6. Audit log the result
7. Scan output for injection before returning to LLM

---

## 12. Prompt Injection Defense

This is the most critical security concern for autonomous agents. FreeBird's ecosystem has a 36% skill injection rate (Cisco). Our defense is layered.

### 12.1 Input Scanning

```rust
// freebird-security/src/injection.rs

/// Heuristic patterns that indicate prompt injection attempts.
/// These are checked on ALL input before it reaches the provider.
const INJECTION_PATTERNS: &[&str] = &[
    "ignore previous instructions",
    "ignore all previous",
    "disregard your instructions",
    "you are now",
    "new instructions:",
    "system prompt:",
    "ADMIN OVERRIDE",
    "jailbreak",
    "DAN mode",
    "<|im_start|>",      // ChatML injection
    "```system",          // markdown system block injection
];

pub fn scan_input(text: &str) -> Result<(), SecurityError> {
    let lower = text.to_lowercase();
    for pattern in INJECTION_PATTERNS {
        if lower.contains(pattern) {
            return Err(SecurityError::PotentialInjection {
                pattern: pattern.to_string(),
                severity: Severity::High,
            });
        }
    }
    Ok(())
}

/// Scan tool output before it's sent back to the LLM.
/// This catches cases where a tool reads a file containing injection payloads
/// (the "indirect injection" vector that hit FreeBird).
pub fn scan_output(text: &str) -> Result<(), SecurityError> {
    let lower = text.to_lowercase();
    for pattern in INJECTION_PATTERNS {
        if lower.contains(pattern) {
            return Err(SecurityError::PotentialInjection {
                pattern: pattern.to_string(),
                severity: Severity::High,
            });
        }
    }
    Ok(())
}
```

### 12.2 Defense Layers

| Layer | What it catches | Where it runs |
|---|---|---|
| Input taint + sanitization | Malformed/malicious user input | Router (before agent) |
| Input injection scan | Known prompt injection patterns | Router (before agent) |
| Capability system | Unauthorized tool access | ToolExecutor (before tool) |
| SafePath | Path traversal in tool args | Tool implementation |
| Tool output scan (`ScannedToolOutput`) | Indirect injection via tool results — **blocks** with synthetic error | Agent loop (after tool, before LLM) |
| Model output scan (`ScannedModelResponse`) | Compromised model responses — **blocks** delivery + prevents memory poisoning | Agent loop (before channel) |
| Audit logging | Post-incident forensics | Every layer |

### 12.3 Reader Agent Pattern (Future)

For high-risk operations (reading untrusted files, scraping web content), use a **separate, tool-disabled reader agent** to summarize content before passing it to the main agent. This architectural isolation prevents indirect injection from reaching the agent that has tool access — the pattern recommended by both Cisco and Microsoft in their FreeBird security analyses.

---

## 13. Authentication & Session Management

### 13.1 Session Key Authentication

Users authenticate via session keys — cryptographically random tokens issued and verified by the daemon.

```rust
// freebird-security/src/auth.rs

use ring::rand::{SecureRandom, SystemRandom};
use ring::digest;
use secrecy::{SecretString, ExposeSecret};

/// A session credential used to authenticate a user.
#[derive(Debug, Clone)]
pub struct SessionCredential {
    /// The key ID (public, used for lookup).
    pub key_id: String,
    /// SHA-256 hash of the actual key (stored, never the raw key).
    pub key_hash: String,
    /// When this key was issued.
    pub issued_at: DateTime<Utc>,
    /// When this key expires (None = never, but discouraged).
    pub expires_at: Option<DateTime<Utc>>,
    /// Which capabilities this session key grants.
    pub capability_grant: CapabilityGrant,
}

/// Generate a new session key. Returns the raw key (to give to the user ONCE)
/// and the credential (to store).
pub fn generate_session_key(
    capabilities: CapabilityGrant,
    ttl: Option<Duration>,
) -> (SecretString, SessionCredential) {
    let rng = SystemRandom::new();
    let mut key_bytes = [0u8; 32];
    rng.fill(&mut key_bytes).expect("system RNG failed");

    let raw_key = hex::encode(key_bytes);
    let key_hash = hex::encode(digest::digest(&digest::SHA256, raw_key.as_bytes()));
    let key_id = format!("freebird_{}", &key_hash[..12]);

    let credential = SessionCredential {
        key_id,
        key_hash,
        issued_at: Utc::now(),
        expires_at: ttl.map(|d| Utc::now() + d),
        capability_grant: capabilities,
    };

    (SecretString::from(raw_key), credential)
}

/// Verify a session key against stored credentials.
pub fn verify_session_key(
    raw_key: &str,
    stored: &SessionCredential,
) -> Result<&CapabilityGrant, SecurityError> {
    // Check expiry first
    if let Some(expires) = stored.expires_at {
        if Utc::now() > expires {
            return Err(SecurityError::SessionExpired {
                key_id: stored.key_id.clone(),
                expired_at: expires,
            });
        }
    }

    // Constant-time comparison via hash
    let provided_hash = hex::encode(digest::digest(&digest::SHA256, raw_key.as_bytes()));
    if provided_hash != stored.key_hash {
        return Err(SecurityError::InvalidSessionKey {
            key_id: stored.key_id.clone(),
        });
    }

    Ok(&stored.capability_grant)
}
```

### 13.2 Provider Credential Management

Provider API keys (Anthropic, OpenAI) are stored encrypted and loaded via `secrecy::SecretString`:

```rust
// freebird-types/src/config.rs

#[derive(Debug, Deserialize)]
pub struct ProviderCredential {
    /// Provider ID (e.g., "anthropic").
    pub provider: String,
    /// API key — wrapped in SecretString, redacted in Debug/logs.
    pub api_key: SecretString,
    /// Optional base URL override (for proxies, local models).
    pub base_url: Option<String>,
    /// Default model for this provider.
    pub default_model: Option<String>,
}
```

### 13.3 Auth Flow

```
User starts session
    │
    ├─ CLI channel: session key passed via --key flag or env var
    ├─ Signal channel: pairing code flow (like FreeBird's DM pairing)
    └─ WebSocket channel: key in initial handshake header
    │
    ▼
Router verifies session key
    │
    ├─ Invalid → reject, audit log, increment failed-auth counter
    └─ Valid → extract CapabilityGrant, attach to session
    │
    ▼
All subsequent requests in this session use the attached CapabilityGrant
```

---

## 14. Channel Pairing & Identity Verification (ASI03)

**OWASP ASI03 — Identity & Privilege Abuse**: Leaked credentials let agents operate far beyond their intended scope. Channel pairing prevents unauthorized senders from interacting with the agent entirely.

### 14.1 Pairing State Machine

Every channel that accepts inbound connections from unknown senders (Signal, WebSocket, future channels) MUST implement the pairing flow. CLI is exempt because it runs locally.

```rust
// freebird-security/src/pairing.rs

use ring::rand::{SecureRandom, SystemRandom};

/// The state of a channel sender's pairing status.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PairingState {
    /// Sender has never been seen. No messages are processed.
    Unknown,
    /// A pairing code has been issued. Awaiting approval.
    PendingApproval {
        code: String,
        issued_at: DateTime<Utc>,
        expires_at: DateTime<Utc>,
        attempts: u8,
    },
    /// Sender has been approved. Messages are processed.
    Paired {
        approved_at: DateTime<Utc>,
        approved_by: String, // who approved (e.g., "cli", "admin-session")
        capability_grant: CapabilityGrant,
    },
    /// Sender has been explicitly blocked.
    Blocked {
        blocked_at: DateTime<Utc>,
        reason: String,
    },
}

pub struct PairingManager {
    store: Box<dyn PairingStore>,
    config: PairingConfig,
}

#[derive(Debug, Clone, Deserialize)]
pub struct PairingConfig {
    /// How long a pairing code is valid.
    pub code_ttl_minutes: u64,
    /// Maximum pending pairing requests per channel.
    pub max_pending_per_channel: u8,
    /// Maximum failed pairing attempts before auto-block.
    pub max_failed_attempts: u8,
    /// Length of the pairing code.
    pub code_length: usize,
}

impl Default for PairingConfig {
    fn default() -> Self {
        Self {
            code_ttl_minutes: 60,
            max_pending_per_channel: 3,
            max_failed_attempts: 5,
            code_length: 6,
        }
    }
}

impl PairingManager {
    /// Handle an inbound message from a sender. Returns whether to process it.
    pub async fn gate_inbound(
        &self,
        channel_id: &str,
        sender_id: &str,
        audit: &AuditLogger,
    ) -> PairingDecision {
        let state = self.store.get_state(channel_id, sender_id).await
            .unwrap_or(PairingState::Unknown);

        match state {
            PairingState::Paired { ref capability_grant, .. } => {
                PairingDecision::Allow { grant: capability_grant.clone() }
            }
            PairingState::Blocked { ref reason, .. } => {
                audit.record(AuditEvent::PolicyViolation {
                    session_id: sender_id.into(),
                    rule: "blocked_sender".into(),
                    context: reason.clone(),
                    severity: Severity::Medium,
                    timestamp: Utc::now(),
                });
                PairingDecision::Reject { reason: "sender is blocked".into() }
            }
            PairingState::PendingApproval { ref expires_at, attempts, .. } => {
                if Utc::now() > *expires_at {
                    // Expired — issue new code if under attempt limit
                    if attempts < self.config.max_failed_attempts {
                        self.issue_pairing_code(channel_id, sender_id, audit).await
                    } else {
                        self.block_sender(channel_id, sender_id, "max pairing attempts exceeded", audit).await;
                        PairingDecision::Reject { reason: "too many failed attempts".into() }
                    }
                } else {
                    PairingDecision::PendingPairing {
                        message: "A pairing request is pending. Approve via CLI: `freebird pair approve`".into(),
                    }
                }
            }
            PairingState::Unknown => {
                self.issue_pairing_code(channel_id, sender_id, audit).await
            }
        }
    }

    /// Generate a cryptographically random pairing code.
    fn generate_code(&self) -> String {
        let rng = SystemRandom::new();
        let mut bytes = vec![0u8; self.config.code_length];
        rng.fill(&mut bytes).expect("system RNG failed");

        // Convert to numeric code (e.g., "847291")
        bytes.iter()
            .map(|b| (b % 10).to_string())
            .collect::<String>()
            [..self.config.code_length]
            .to_string()
    }

    async fn issue_pairing_code(
        &self,
        channel_id: &str,
        sender_id: &str,
        audit: &AuditLogger,
    ) -> PairingDecision {
        let code = self.generate_code();
        let now = Utc::now();

        let state = PairingState::PendingApproval {
            code: code.clone(),
            issued_at: now,
            expires_at: now + chrono::Duration::minutes(self.config.code_ttl_minutes as i64),
            attempts: 0,
        };

        let _ = self.store.set_state(channel_id, sender_id, &state).await;

        audit.record(AuditEvent::PairingCodeIssued {
            channel_id: channel_id.into(),
            sender_id: sender_id.into(),
            // NEVER log the actual code — only that one was issued
            timestamp: now,
        });

        PairingDecision::PendingPairing {
            message: format!(
                "Pairing required. Your code: {code}. \
                 Ask the owner to approve: `freebird pair approve --code {code}`"
            ),
        }
    }

    /// Approve a pairing via CLI. Validates the code, transitions to Paired.
    pub async fn approve(
        &self,
        channel_id: &str,
        sender_id: &str,
        provided_code: &str,
        capabilities: CapabilityGrant,
        audit: &AuditLogger,
    ) -> Result<(), SecurityError> {
        let state = self.store.get_state(channel_id, sender_id).await
            .ok_or(SecurityError::PairingNotFound)?;

        match state {
            PairingState::PendingApproval { code, expires_at, .. } => {
                if Utc::now() > expires_at {
                    return Err(SecurityError::PairingExpired);
                }
                // Constant-time comparison
                if !ring::constant_time::verify_slices_are_equal(
                    code.as_bytes(),
                    provided_code.as_bytes(),
                ).is_ok() {
                    return Err(SecurityError::InvalidPairingCode);
                }

                let paired = PairingState::Paired {
                    approved_at: Utc::now(),
                    approved_by: "cli".into(),
                    capability_grant: capabilities,
                };
                self.store.set_state(channel_id, sender_id, &paired).await?;

                audit.record(AuditEvent::PairingApproved {
                    channel_id: channel_id.into(),
                    sender_id: sender_id.into(),
                    timestamp: Utc::now(),
                });

                Ok(())
            }
            _ => Err(SecurityError::PairingNotPending),
        }
    }
}

pub enum PairingDecision {
    /// Message should be processed with this capability grant.
    Allow { grant: CapabilityGrant },
    /// Message should be rejected.
    Reject { reason: String },
    /// Sender needs to complete pairing. Send this message back.
    PendingPairing { message: String },
}
```

### 14.2 Integration with Router

The pairing gate sits in the router, **before** taint processing — unpaired senders never reach the agent:

```rust
// In the router's handle_event:
async fn handle_event(&self, event: RouterEvent, outbound: mpsc::Sender<OutboundEvent>) {
    // CLI channel skips pairing (local-only)
    if event.channel_id == "cli" {
        return self.process_event(event, outbound, self.default_grant.clone()).await;
    }

    // All other channels: gate through pairing
    let sender_id = event.event.sender_id();
    let decision = self.pairing.gate_inbound(
        &event.channel_id, sender_id, &self.audit,
    ).await;

    match decision {
        PairingDecision::Allow { grant } => {
            self.process_event(event, outbound, grant).await;
        }
        PairingDecision::Reject { reason } => {
            let _ = outbound.send(OutboundEvent::Error {
                text: reason,
                recipient_id: sender_id.into(),
            }).await;
        }
        PairingDecision::PendingPairing { message } => {
            let _ = outbound.send(OutboundEvent::Message {
                text: message,
                recipient_id: sender_id.into(),
            }).await;
        }
    }
}
```

---

## 15. Consent Gates & Human-in-the-Loop (ASI09)

**OWASP ASI09 — Human-Agent Trust Exploitation**: Confident, polished agent explanations can mislead human operators into approving harmful actions. Consent gates ensure that high-risk, irreversible, or sensitive operations require explicit human approval **before** execution.

### 15.1 Risk Classification

Every tool action is classified by risk level. The classification determines whether the action can proceed automatically or requires human approval.

```rust
// freebird-security/src/consent.rs

/// Risk classification for tool actions.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum RiskLevel {
    /// Read-only, no side effects. Auto-approved.
    Low,
    /// Side effects but reversible (write file, create dir). Auto-approved if capability granted.
    Medium,
    /// Irreversible or high-impact (delete, send email, execute shell, network request).
    /// Requires explicit human consent.
    High,
    /// System-level or security-impacting (modify config, change permissions, install software).
    /// Always requires human consent, even if capability is granted.
    Critical,
}

/// A consent request presented to the user.
#[derive(Debug, Clone, Serialize)]
pub struct ConsentRequest {
    pub id: String,
    pub tool_name: String,
    pub description: String,
    pub risk_level: RiskLevel,
    /// What exactly the tool will do, in plain language.
    pub action_summary: String,
    /// What resources will be affected.
    pub affected_resources: Vec<String>,
    /// Is this action reversible?
    pub reversible: bool,
    pub requested_at: DateTime<Utc>,
    pub expires_at: DateTime<Utc>,
}

/// The user's response to a consent request.
#[derive(Debug, Clone)]
pub enum ConsentResponse {
    Approved,
    Denied { reason: Option<String> },
    Expired,
}

/// The consent gate that sits inside the ToolExecutor.
pub struct ConsentGate {
    /// Policy: which risk levels require consent.
    require_consent_above: RiskLevel,
    /// How long a consent request is valid.
    consent_ttl: Duration,
    /// Channel to send consent requests to the user.
    consent_tx: mpsc::Sender<ConsentRequest>,
    /// Channel to receive consent responses.
    consent_rx: Arc<Mutex<HashMap<String, oneshot::Sender<ConsentResponse>>>>,
}

impl ConsentGate {
    /// Check if this tool invocation requires consent.
    /// If yes, send a request and wait for approval.
    pub async fn check(
        &self,
        tool: &dyn Tool,
        input: &serde_json::Value,
        audit: &AuditLogger,
    ) -> Result<(), ToolError> {
        let risk = tool.info().risk_level.clone();

        if risk <= self.require_consent_above {
            return Ok(()); // auto-approved
        }

        let request = ConsentRequest {
            id: uuid::Uuid::new_v4().to_string(),
            tool_name: tool.info().name.clone(),
            description: tool.info().description.clone(),
            risk_level: risk.clone(),
            action_summary: tool.describe_action(input),
            affected_resources: tool.affected_resources(input),
            reversible: !tool.info().has_side_effects,
            requested_at: Utc::now(),
            expires_at: Utc::now() + self.consent_ttl,
        };

        // Send to user via channel
        let (response_tx, response_rx) = oneshot::channel();
        {
            let mut pending = self.consent_rx.lock().await;
            pending.insert(request.id.clone(), response_tx);
        }

        self.consent_tx.send(request.clone()).await
            .map_err(|_| ToolError::ConsentChannelClosed)?;

        audit.record(AuditEvent::ConsentRequested {
            tool_name: request.tool_name.clone(),
            risk_level: format!("{risk:?}"),
            timestamp: Utc::now(),
        });

        // Wait for response with timeout
        match tokio::time::timeout(self.consent_ttl, response_rx).await {
            Ok(Ok(ConsentResponse::Approved)) => {
                audit.record(AuditEvent::ConsentGranted {
                    tool_name: request.tool_name,
                    timestamp: Utc::now(),
                });
                Ok(())
            }
            Ok(Ok(ConsentResponse::Denied { reason })) => {
                audit.record(AuditEvent::ConsentDenied {
                    tool_name: request.tool_name,
                    reason: reason.clone(),
                    timestamp: Utc::now(),
                });
                Err(ToolError::ConsentDenied {
                    tool: tool.info().name.clone(),
                    reason: reason.unwrap_or_else(|| "user denied".into()),
                })
            }
            _ => {
                audit.record(AuditEvent::ConsentExpired {
                    tool_name: request.tool_name,
                    timestamp: Utc::now(),
                });
                Err(ToolError::ConsentExpired {
                    tool: tool.info().name.clone(),
                })
            }
        }
    }
}
```

### 15.2 Tool Risk Level Annotation

Every tool declares its risk level in `ToolInfo`:

```rust
// Update to ToolInfo (in freebird-traits/src/tool.rs)

pub struct ToolInfo {
    pub name: String,
    pub description: String,
    pub input_schema: serde_json::Value,
    pub required_capability: Capability,
    pub has_side_effects: bool,
    /// NEW: Risk classification for consent gating.
    pub risk_level: RiskLevel,
}
```

Default risk levels for built-in tools:

| Tool | Risk Level | Consent Required? |
|---|---|---|
| `read_file` | Low | No |
| `list_directory` | Low | No |
| `write_file` | Medium | No (if capability granted) |
| `shell` (read-only: ls, cat) | Medium | No |
| `shell` (write: rm, mv, cp) | High | **Yes** |
| `shell` (arbitrary) | Critical | **Always** |
| `network_request` | High | **Yes** |
| `send_message` (via channel) | High | **Yes** |
| `modify_config` | Critical | **Always** |

### 15.3 Presentation to User

On CLI, consent requests appear inline:

```
🔒 CONSENT REQUIRED
Tool: shell
Risk: High (irreversible)
Action: Execute `rm -rf /tmp/build-cache`
Affected: /tmp/build-cache (directory, ~2.3GB)

Approve? [y/N]:
```

On messaging channels (Signal, WebSocket), the consent request is sent as a structured message with approve/deny buttons.

---

## 16. Network Egress Control & Data Exfiltration Defense (ASI01)

**OWASP ASI01 — Agent Goal Hijacking**: The most dangerous attack turns the agent into an exfiltration engine. A prompt injection causes the agent to silently `curl` sensitive data to an attacker-controlled server. This was the exact vector in FreeBird's CVE-2026-25253.

### 16.1 Egress Allowlist

The agent can ONLY make outbound network requests to explicitly allowed hosts. This is enforced at the tool level AND at the HTTP client level (belt and suspenders).

```rust
// freebird-security/src/egress.rs

pub struct EgressPolicy {
    /// Hosts the agent is allowed to contact.
    allowed_hosts: HashSet<String>,
    /// Ports the agent is allowed to use (default: 443 only).
    allowed_ports: HashSet<u16>,
    /// Maximum request body size (prevents exfiltration of large data).
    max_request_body_bytes: usize,
    /// Maximum number of outbound requests per minute.
    rate_limit_per_minute: u32,
}

impl EgressPolicy {
    pub fn check_request(
        &self,
        url: &url::Url,
        body_size: usize,
        audit: &AuditLogger,
    ) -> Result<(), SecurityError> {
        // Check host
        let host = url.host_str().ok_or(SecurityError::EgressBlocked {
            reason: "no host in URL".into(),
        })?;

        if !self.allowed_hosts.contains(host) {
            audit.record(AuditEvent::EgressBlocked {
                host: host.into(),
                reason: "host not in allowlist".into(),
                timestamp: Utc::now(),
            });
            return Err(SecurityError::EgressBlocked {
                reason: format!("host `{host}` not in egress allowlist"),
            });
        }

        // Check port
        let port = url.port().unwrap_or(443);
        if !self.allowed_ports.contains(&port) {
            return Err(SecurityError::EgressBlocked {
                reason: format!("port {port} not allowed"),
            });
        }

        // Check body size (prevents data exfiltration)
        if body_size > self.max_request_body_bytes {
            audit.record(AuditEvent::PolicyViolation {
                session_id: "system".into(),
                rule: "egress_body_size".into(),
                context: format!("body size {body_size} exceeds limit {}", self.max_request_body_bytes),
                severity: Severity::High,
                timestamp: Utc::now(),
            });
            return Err(SecurityError::EgressBlocked {
                reason: "request body exceeds maximum size".into(),
            });
        }

        Ok(())
    }
}
```

### 16.2 Default Egress Configuration

```toml
# config/default.toml

[security.egress]
# ONLY provider APIs by default. Everything else is blocked.
allowed_hosts = [
    "api.anthropic.com",
    "api.openai.com",
]
allowed_ports = [443]
max_request_body_bytes = 1048576  # 1MB
rate_limit_per_minute = 60
```

Adding any host requires a config change — the agent cannot modify its own egress policy.

### 16.3 DNS Rebinding Prevention

Validate that resolved IPs are not private/loopback addresses (prevents SSRF to internal services):

```rust
pub fn validate_resolved_ip(ip: std::net::IpAddr) -> Result<(), SecurityError> {
    if ip.is_loopback() || ip.is_private() || ip.is_link_local() {
        return Err(SecurityError::EgressBlocked {
            reason: format!("resolved to private/loopback IP: {ip}"),
        });
    }
    Ok(())
}
```

---

## 17. Token Budgets & Resource Exhaustion Defense (ASI08)

**OWASP ASI08 — Agent Resource & Service Exhaustion**: An attacker (or runaway agent) can consume unbounded tokens/compute, causing financial damage or denial of service.

### 17.1 Token Budget System

```rust
// freebird-security/src/budget.rs

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// A budget that tracks and limits token consumption per session.
#[derive(Debug)]
pub struct TokenBudget {
    /// Maximum input + output tokens per session.
    max_tokens_per_session: u64,
    /// Maximum tokens per single request.
    max_tokens_per_request: u64,
    /// Maximum number of tool rounds per turn.
    max_tool_rounds_per_turn: u32,
    /// Maximum total cost in microdollars per session.
    max_cost_microdollars: u64,
    /// Running counters
    tokens_used: AtomicU64,
    cost_microdollars: AtomicU64,
}

impl TokenBudget {
    pub fn record_usage(&self, usage: &TokenUsage) -> Result<(), SecurityError> {
        let total = usage.input_tokens as u64 + usage.output_tokens as u64;

        // Check per-request limit
        if total > self.max_tokens_per_request {
            return Err(SecurityError::BudgetExceeded {
                resource: "tokens_per_request".into(),
                used: total,
                limit: self.max_tokens_per_request,
            });
        }

        // Check cumulative session limit
        let new_total = self.tokens_used.fetch_add(total, Ordering::Relaxed) + total;
        if new_total > self.max_tokens_per_session {
            return Err(SecurityError::BudgetExceeded {
                resource: "tokens_per_session".into(),
                used: new_total,
                limit: self.max_tokens_per_session,
            });
        }

        Ok(())
    }

    pub fn check_tool_rounds(&self, current_round: u32) -> Result<(), SecurityError> {
        if current_round >= self.max_tool_rounds_per_turn {
            return Err(SecurityError::BudgetExceeded {
                resource: "tool_rounds_per_turn".into(),
                used: current_round as u64,
                limit: self.max_tool_rounds_per_turn as u64,
            });
        }
        Ok(())
    }

    pub fn remaining_tokens(&self) -> u64 {
        let used = self.tokens_used.load(Ordering::Relaxed);
        self.max_tokens_per_session.saturating_sub(used)
    }
}
```

### 17.2 Default Budget Configuration

```toml
[security.budgets]
max_tokens_per_session = 500000      # 500k tokens per session
max_tokens_per_request = 32768       # 32k per single request
max_tool_rounds_per_turn = 10        # max 10 tool calls in a single turn
max_cost_microdollars = 5000000      # $5.00 per session
```

---

## 18. Memory & Context Integrity (ASI06)

**OWASP ASI06 — Memory & Context Poisoning**: An attacker poisons the agent's conversation history or long-term memory to influence future behavior. If the agent's memory files are tampered with, all subsequent decisions are compromised.

### 18.1 Conversation Integrity Verification

```rust
// freebird-security/src/integrity.rs

use ring::hmac;

/// Compute an HMAC over a conversation to detect tampering.
pub fn sign_conversation(
    conversation: &Conversation,
    signing_key: &hmac::Key,
) -> String {
    let serialized = serde_json::to_string(conversation)
        .expect("conversation serialization should never fail");
    let tag = hmac::sign(signing_key, serialized.as_bytes());
    hex::encode(tag.as_ref())
}

/// Verify a conversation's HMAC before loading.
pub fn verify_conversation(
    conversation: &Conversation,
    expected_signature: &str,
    signing_key: &hmac::Key,
) -> Result<(), SecurityError> {
    let serialized = serde_json::to_string(conversation)
        .expect("conversation serialization should never fail");
    let expected_bytes = hex::decode(expected_signature)
        .map_err(|_| SecurityError::IntegrityViolation {
            resource: "conversation".into(),
            reason: "invalid signature encoding".into(),
        })?;

    hmac::verify(signing_key, serialized.as_bytes(), &expected_bytes)
        .map_err(|_| SecurityError::IntegrityViolation {
            resource: format!("conversation:{}", conversation.session_id),
            reason: "HMAC verification failed — file may have been tampered with".into(),
        })
}
```

### 18.2 Memory File Format

Conversation files include a signature footer:

```json
{
    "session_id": "abc-123",
    "turns": [ ... ],
    "_signature": "a1b2c3d4e5f6..."
}
```

The `FileMemory` implementation signs on save and verifies on load. If verification fails, the conversation is quarantined (moved to a `quarantine/` directory) and the user is notified.

### 18.3 Context Window Poisoning Defense

Before appending tool outputs to the context window, scan for content that resembles system prompts or instruction overrides:

```rust
/// Scan content that will be injected into the context window.
/// Catches indirect prompt injection via tool outputs.
pub fn scan_context_injection(content: &str) -> Result<(), SecurityError> {
    let suspicious_patterns = &[
        "you are now",
        "new system prompt",
        "ignore all previous",
        "your instructions are",
        "<|system|>",
        "[INST]",
        "<<SYS>>",
    ];

    let lower = content.to_lowercase();
    for pattern in suspicious_patterns {
        if lower.contains(pattern) {
            return Err(SecurityError::ContextPoisoningAttempt {
                pattern: pattern.to_string(),
            });
        }
    }
    Ok(())
}
```

---

## 19. Tamper-Evident Audit Logging

### 19.1 Hash-Chained Audit Log

Each audit log entry includes the hash of the previous entry, creating a tamper-evident chain. If an attacker modifies or deletes a log entry, the chain breaks.

```rust
// freebird-security/src/audit.rs

use ring::digest;

pub struct AuditLogger {
    writer: Box<dyn std::io::Write + Send>,
    last_hash: String,
    signing_key: hmac::Key,
}

impl AuditLogger {
    pub fn record(&mut self, event: AuditEvent) {
        let entry = AuditEntry {
            sequence: self.next_sequence(),
            event,
            timestamp: Utc::now(),
            previous_hash: self.last_hash.clone(),
        };

        let serialized = serde_json::to_string(&entry)
            .expect("audit entry serialization failed");

        // Hash this entry (including the previous hash) to form the chain
        let hash = hex::encode(
            digest::digest(&digest::SHA256, serialized.as_bytes())
        );

        let line = AuditLine {
            entry,
            hash: hash.clone(),
        };

        // Write as single JSON line (atomic append)
        writeln!(self.writer, "{}", serde_json::to_string(&line).unwrap())
            .expect("audit log write failed — this is a critical failure");

        self.last_hash = hash;
    }
}

#[derive(Debug, Serialize)]
struct AuditEntry {
    sequence: u64,
    event: AuditEvent,
    timestamp: DateTime<Utc>,
    previous_hash: String,
}

#[derive(Debug, Serialize)]
struct AuditLine {
    entry: AuditEntry,
    hash: String,
}
```

### 19.2 Audit Log Verification

```rust
/// Verify the integrity of the entire audit log.
/// Returns the first broken link if tampered.
pub fn verify_audit_chain(log_path: &Path) -> Result<(), SecurityError> {
    let file = std::fs::File::open(log_path)?;
    let reader = std::io::BufReader::new(file);
    let mut expected_prev_hash = String::new(); // genesis

    for (line_num, line) in reader.lines().enumerate() {
        let line = line?;
        let audit_line: AuditLine = serde_json::from_str(&line)
            .map_err(|e| SecurityError::AuditCorruption {
                line: line_num,
                reason: format!("parse error: {e}"),
            })?;

        // Verify previous hash matches
        if audit_line.entry.previous_hash != expected_prev_hash {
            return Err(SecurityError::AuditCorruption {
                line: line_num,
                reason: "hash chain broken — log has been tampered with".into(),
            });
        }

        // Verify this entry's hash
        let entry_json = serde_json::to_string(&audit_line.entry)?;
        let computed_hash = hex::encode(
            digest::digest(&digest::SHA256, entry_json.as_bytes())
        );

        if computed_hash != audit_line.hash {
            return Err(SecurityError::AuditCorruption {
                line: line_num,
                reason: "entry hash mismatch — entry has been modified".into(),
            });
        }

        expected_prev_hash = audit_line.hash;
    }

    Ok(())
}
```

---

## 20. Supply Chain Security (ASI04)

**OWASP ASI04 — Agentic Supply Chain Vulnerabilities**: Hidden components and unchecked dependencies. FreeBird's ClawHub had 36% of skills containing prompt injection (Cisco finding).

### 20.1 No Runtime Skill Loading (v1)

Unlike FreeBird, FreeBird v1 does **NOT** support dynamic skill installation. All tools are compiled into the binary. This eliminates the supply chain attack surface entirely for the initial release.

Future skill system (v2+) requirements:
- Skills are sandboxed in a WASM runtime (e.g., `wasmtime`) with no host access.
- Skills declare required capabilities upfront.
- Skills are content-addressed (SHA-256 hash) and signed.
- Skill installation requires explicit human approval via consent gate.
- Skills cannot access the agent's memory, config, or other skills.

### 20.2 Dependency Auditing

```bash
# Run on every CI build
cargo deny check advisories   # known CVEs
cargo deny check licenses     # license compatibility
cargo deny check bans         # banned crates (openssl, etc.)
cargo deny check sources      # only crates.io, no git deps

# Weekly
cargo audit                   # RustSec advisory database
```

### 20.3 deny.toml Configuration

```toml
[advisories]
vulnerability = "deny"
unmaintained = "warn"

[licenses]
allow = ["MIT", "Apache-2.0", "BSD-2-Clause", "BSD-3-Clause", "ISC"]
default = "deny"

[bans]
multiple-versions = "warn"
deny = [
    { name = "openssl" },
    { name = "openssl-sys" },
]

[sources]
unknown-registry = "deny"
unknown-git = "deny"
allow-git = []
```

---

# Part IV — Infrastructure

---

## 21. Error Handling Strategy

**Library crates** (`freebird-traits`, `freebird-security`, `freebird-providers`, etc.): Use `thiserror` with precise, matchable error enums. Each crate defines its own error type.

**Binary crate** (`freebird-daemon`): Use `anyhow` with `.context()` for ergonomic error chains.

**Hard rules**:

- NEVER `.unwrap()` in production code.
- NEVER `.expect()` without a comment explaining why the invariant holds.
- Use `?` for propagation. Add `.context()` at meaningful boundaries.
- Panics are reserved for true programmer errors (violated invariants), never recoverable conditions.

---

## 22. Daemon Lifecycle & Graceful Shutdown

### 15.1 Thin main.rs (Under 30 Lines)

```rust
// freebird-daemon/src/main.rs

use anyhow::{Context, Result};

#[tokio::main]
async fn main() -> Result<()> {
    let config = freebird_runtime::config::load()
        .context("failed to load configuration")?;

    freebird_runtime::logging::init(&config.logging)
        .context("failed to initialize logging")?;

    let channels = freebird_runtime::build_channels(&config.channels)
        .context("failed to build channels")?;

    let runtime = freebird_runtime::AgentRuntime::build(config)
        .await
        .context("failed to build agent runtime")?;

    runtime.run(channels).await
}
```

### 15.2 Graceful Shutdown

```rust
use tokio::signal;
use tokio_util::sync::CancellationToken;

pub struct ShutdownCoordinator {
    token: CancellationToken,
    drain_timeout: Duration,
}

impl ShutdownCoordinator {
    pub async fn wait_for_signal(&self) {
        let ctrl_c = signal::ctrl_c();
        let mut sigterm = signal::unix::signal(signal::unix::SignalKind::terminate())
            .expect("failed to register SIGTERM handler");

        tokio::select! {
            _ = ctrl_c => tracing::info!("received SIGINT"),
            _ = sigterm.recv() => tracing::info!("received SIGTERM"),
        }

        self.token.cancel();
    }
}
```

---

## 23. Configuration & Secrets Management

### 16.1 Layered Configuration

```toml
# config/default.toml

[runtime]
system_prompt = "You are FreeBird, a helpful AI assistant."
default_model = "claude-opus-4-6-20250929"
default_provider = "anthropic"
max_output_tokens = 8192
max_tool_rounds = 10
temperature = 0.7
drain_timeout_seconds = 30

[security]
sandbox_root = "~/.freebird/sandbox"
max_agent_depth = 3
session_key_ttl_hours = 720  # 30 days
rate_limit_requests_per_minute = 60

[[providers]]
provider = "anthropic"
# api_key loaded from OPENCLAW_PROVIDERS__0__API_KEY env var
default_model = "claude-opus-4-6-20250929"

[[channels]]
type = "cli"
prompt = "you> "

[tools]
default_timeout_seconds = 30
enabled = ["read_file", "write_file", "shell"]

[tools.shell]
allowed_commands = ["ls", "cat", "grep", "find", "git"]

[memory]
backend = "file"
base_dir = "~/.freebird/conversations"

[logging]
level = "info"
audit_log_path = "~/.freebird/audit.jsonl"
```

### 16.2 Secrets

- Provider API keys: `OPENCLAW_PROVIDERS__0__API_KEY` environment variable.
- Session keys: generated via `freebird keygen` CLI command, stored in `~/.freebird/keys.json`.
- All secrets wrapped in `secrecy::SecretString` — zeroized on drop, redacted in Debug.
- NEVER hardcode secrets. NEVER log secrets. NEVER include secrets in error messages.

---

## 24. Logging, Tracing & Audit

Use `tracing` for everything. Console output for developers. JSON lines to audit file for SIEM.

```rust
pub fn init(config: &LoggingConfig) -> Result<()> {
    let env_filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new(&config.level));

    let json_layer = fmt::layer()
        .json()
        .with_writer(move || {
            std::fs::OpenOptions::new()
                .create(true).append(true)
                .open(&config.audit_log_path)
                .expect("failed to open audit log")
        });

    let console_layer = fmt::layer().pretty().with_target(true);

    tracing_subscriber::registry()
        .with(env_filter)
        .with(json_layer)
        .with(console_layer)
        .init();

    Ok(())
}
```

---

## 25. Concurrency Patterns

- **Prefer channels over shared state.** The runtime uses `mpsc` channels to fan-in events from multiple channels.
- **Use `tokio::sync::Mutex`** for async code, never `std::sync::Mutex`.
- **Never hold a lock across `.await`**.
- **Use `tokio::select!`** for racing futures (message receive vs. shutdown signal).
- **Use `CancellationToken`** for cooperative shutdown across spawned tasks.

---

# Part V — Quality

---

## 26. Testing Strategy

### Unit Tests

Every module has `#[cfg(test)]` tests. Security modules have adversarial tests:

```rust
#[test]
fn taint_system_prevents_bypass() {
    let raw = Tainted::<Untrusted>::new("../../../etc/passwd");
    // raw.value() — MUST NOT COMPILE (this IS the test)

    let result = raw.sanitize(|s| {
        if s.contains("..") {
            Err(SecurityError::TaintViolation { reason: "traversal".into() })
        } else {
            Ok(s.to_owned())
        }
    });
    assert!(result.is_err());
}
```

### Property-Based Tests

```rust
use proptest::prelude::*;

proptest! {
    #[test]
    fn safe_path_never_escapes_sandbox(input in ".*") {
        let tmp = tempfile::tempdir().unwrap();
        if let Ok(path) = SafePath::new(tmp.path(), &input) {
            assert!(path.as_path().starts_with(tmp.path()));
        }
    }
}
```

### Integration Tests

Cross-crate tests that wire up the full stack with mocked providers:

```rust
#[tokio::test]
async fn agent_loop_handles_tool_use_round_trip() {
    let mock_provider = MockProvider::new()
        .with_response(tool_use_response("read_file", json!({"path": "test.txt"})))
        .then_response(end_turn_response("File contents: hello"));

    let runtime = TestRuntime::new()
        .with_provider(mock_provider)
        .with_tool(MockReadFileTool::new("hello"))
        .build().await;

    let response = runtime.send_message("Read test.txt").await;
    assert!(response.text.contains("hello"));
    assert_eq!(runtime.tool_invocation_count(), 1);
}
```

---

## 27. Dependency Policy

| Purpose | Crate | Notes |
|---|---|---|
| Async | `tokio` | Pin major |
| Traits | `async-trait` | Until Rust has native async traits in all positions |
| Serde | `serde`, `serde_json` | Universal |
| Errors (lib) | `thiserror` | Precise enums |
| Errors (bin) | `anyhow` | Ergonomic context |
| Logging | `tracing` | Structured, async-aware |
| HTTP client | `reqwest` + `rustls-tls` | NEVER openssl |
| Crypto | `ring` | For hashing, random |
| Secrets | `secrecy` | Zeroize + redact |
| Config | `figment` | Layered sources |

**No `openssl-sys`** anywhere in the dep tree. Run `cargo deny check` on every build.

---

## 28. Code Style & Linting

```rust
#![deny(clippy::all)]
#![deny(clippy::pedantic)]
#![deny(clippy::nursery)]
#![deny(clippy::unwrap_used)]
#![deny(clippy::expect_used)]
#![deny(clippy::panic)]
#![deny(clippy::indexing_slicing)]
#![allow(clippy::module_name_repetitions)]
```

Run `cargo fmt` on save. Treat clippy warnings as errors.

---

## 29. Performance Guidelines

- Accept `&str` in parameters, return `String` when ownership is needed.
- Use iterator chains over manual loops (zero-cost abstractions).
- Use `Vec::with_capacity()` when size is known.
- Use `Cow<'_, str>` to avoid unnecessary allocations.
- Pre-allocate buffers in hot paths.
- Profile before optimizing — correctness and security first.

---

## 30. Common Anti-Patterns to Avoid

| Anti-Pattern | Do This Instead |
|---|---|
| `.unwrap()` in production | `?` with `.context()` |
| `.clone()` to satisfy borrow checker | Restructure ownership or use `Arc` |
| `String` in function parameters | `&str` or `impl AsRef<str>` |
| `bool` parameters | Enums: `Mode::Verbose` not `true` |
| `pub` struct fields | Methods + builder pattern |
| `Box<dyn Error>` in libraries | `thiserror` enums |
| `std::sync::Mutex` in async | `tokio::sync::Mutex` |
| `println!` for logging | `tracing::info!` |
| Magic strings for events | Enums |
| Raw `PathBuf` in tool I/O | `SafePath` |
| Raw `String` from network | `Tainted<Untrusted>` |

---

## 31. Security Checklist

### Input Handling

- [ ] All external input enters as `Tainted<Untrusted>`
- [ ] Sanitization happens before business logic
- [ ] Input scanned for prompt injection patterns
- [ ] Validators are unit-tested with adversarial inputs

### Path Safety

- [ ] All filesystem ops use `SafePath`
- [ ] Symlinks resolved before boundary checks
- [ ] Path traversal tests include `..`, symlinks, null bytes, encoded slashes

### Capabilities

- [ ] Every tool checks capabilities before execution
- [ ] Grants have expiration times
- [ ] Sub-agent grants are strict subsets of parent
- [ ] No ambient authority / global admin bypass

### Prompt Injection Defense

- [ ] Input scanning on all user messages
- [ ] Output scanning on all tool results via `ScannedToolOutput` — blocks with synthetic error on detection
- [ ] Output scanning on model responses via `ScannedModelResponse` — blocks delivery and prevents memory poisoning on detection
- [ ] Reader agent pattern for untrusted external content (future)

### Auth & Sessions

- [ ] Session keys are cryptographically random (32 bytes from SystemRandom)
- [ ] Keys are stored as SHA-256 hashes, never raw
- [ ] Keys have configurable TTL
- [ ] Failed auth attempts are rate-limited and audit-logged
- [ ] Provider API keys use `SecretString` — never logged, never serialized

### Channel Pairing (ASI03)

- [ ] All non-CLI channels require explicit pairing before message processing
- [ ] Pairing codes are cryptographically random, time-limited (≤5 min), single-use
- [ ] Code comparison uses constant-time equality (`ring::constant_time::verify_slices_are_equal`)
- [ ] Failed pairing attempts are capped (≤3) before auto-block
- [ ] `PairingState` transitions are audited (Unknown → PendingApproval → Paired/Blocked)
- [ ] Router rejects all `InboundEvent`s from unpaired channels

### Consent Gates (ASI09)

- [ ] All tools are classified by `RiskLevel` (Low / Medium / High / Critical)
- [ ] `High` and `Critical` tools require explicit human approval before execution
- [ ] Consent requests include tool name, sanitized arguments, and risk justification
- [ ] Consent timeouts default to 60s — no indefinite waits
- [ ] Consent decisions are audit-logged with channel, tool, and approval/denial

### Network Egress (ASI01)

- [ ] All outbound HTTP is routed through `EgressPolicy` — no direct `reqwest` calls
- [ ] Default policy is deny-all; only explicitly allowlisted hosts are reachable
- [ ] Response body size is capped (`max_body_bytes`, default 10 MiB)
- [ ] DNS rebinding prevention: resolved IPs checked against private ranges (10.x, 172.16-31.x, 192.168.x, 127.x, ::1)

### Token Budgets (ASI08)

- [ ] Per-session token limits enforced via `TokenBudget` with `AtomicU64`
- [ ] Per-request limits prevent single-turn abuse
- [ ] Per-turn limits cap individual LLM calls within a tool-use loop
- [ ] Budget exhaustion returns a descriptive error — never silently truncates

### Memory Integrity (ASI06)

- [ ] All persisted conversations are HMAC-signed (ring HMAC-SHA256)
- [ ] Signature verified on every load — tampered files quarantined, not loaded
- [ ] HMAC key derived from a server-side secret, never stored alongside conversation files
- [ ] Context injection scan runs on loaded conversation history before sending to provider

### Audit Log Integrity

- [ ] Every audit entry includes a SHA-256 hash of the previous entry (hash chain)
- [ ] Chain can be verified from genesis entry forward — any tampering detected
- [ ] Audit entries are append-only JSON lines — no in-place mutation
- [ ] Startup routine verifies audit chain integrity

### Supply Chain (ASI04)

- [ ] No runtime plugin/extension loading — all code compiled into the binary
- [ ] `cargo deny check advisories` passes in CI (blocks known CVEs)
- [ ] `cargo deny check bans` blocks `openssl-sys` and other disallowed crates
- [ ] `cargo deny check licenses` enforces allowlist (MIT, Apache-2.0, BSD-2/3-Clause, ISC, Zlib)
- [ ] `cargo audit` runs in CI with `--deny-warnings`

### Concurrency

- [ ] No `std::sync::Mutex` in async code
- [ ] No locks held across `.await`
- [ ] Graceful shutdown with drain timeout

### Dependencies

- [ ] `cargo deny check advisories` passes
- [ ] No `openssl-sys` in dep tree
- [ ] No new `unsafe` without justification

---

## Appendix A: Build Order for v0.1

**Phase 1 — Skeleton** (get `cargo build` working)

1. Create workspace with all crate stubs
2. Define traits in `freebird-traits` (Provider, Channel, Tool, Memory)
3. Define message types in `freebird-types`
4. Implement `SecurityError` in `freebird-security`

**Phase 2 — Minimum Viable Agent** (talk to Claude from the terminal)

5. Implement `CliChannel` in `freebird-channels`
6. Implement `AnthropicProvider` in `freebird-providers`
7. Implement `FileMemory` in `freebird-memory`
8. Implement `AgentRuntime` with basic loop (no tools) in `freebird-runtime`
9. Wire everything in `freebird-daemon/main.rs`
10. Test: type a message, get a Claude response in the terminal

**Phase 3 — Security Layer** (harden before adding tools)

11. Implement `Tainted<T>` taint system
12. Implement `SafePath`
13. Implement `CapabilityGrant`
14. Implement session key auth
15. Implement prompt injection scanning
16. Implement tamper-evident audit logging (hash-chained)

**Phase 4 — Channel Pairing & Consent** (identity + human-in-the-loop)

17. Implement `PairingManager` with state machine
18. Integrate pairing check into router (reject unpaired channels)
19. Implement `ConsentGate` with risk classification
20. Wire consent gates into `ToolExecutor` for High/Critical tools
21. Test: pair a channel, verify unpaired channel is rejected

**Phase 5 — Tool System** (agent can act)

22. Implement `ToolExecutor` with capability + consent checks
23. Implement `read_file` tool (with SafePath)
24. Implement `shell` tool (with command allowlist)
25. Wire tool loop into agent runtime
26. Test: ask Claude to read a file, watch it use the tool

**Phase 6 — Defense in Depth** (network + budgets + memory integrity)

27. Implement `EgressPolicy` with host allowlist + DNS rebinding prevention
28. Implement `TokenBudget` with per-session/request/turn limits
29. Implement HMAC-signed memory (sign on save, verify on load)
30. Implement context injection scan on loaded conversations
31. Test: exceed token budget, tamper with a conversation file — verify rejection

**Phase 7 — Polish**

32. Streaming responses
33. `/commands` (new, status, model, help)
34. Configuration validation at startup
35. Daemon mode (systemd/launchd)
36. Signal channel implementation (with pairing flow)
37. Supply chain CI gates (`cargo deny`, `cargo audit`)

---

## Appendix B: Quick Reference — Key Trait Signatures

```rust
// Provider — talk to LLMs
trait Provider: Send + Sync + 'static {
    fn info(&self) -> &ProviderInfo;
    async fn validate_credentials(&self) -> Result<(), ProviderError>;
    async fn complete(&self, req: CompletionRequest) -> Result<CompletionResponse, ProviderError>;
    async fn stream(&self, req: CompletionRequest) -> Result<Pin<Box<dyn Stream<Item = Result<StreamEvent, ProviderError>> + Send>>, ProviderError>;
}

// Channel — talk to users
trait Channel: Send + Sync + 'static {
    fn info(&self) -> &ChannelInfo;
    async fn start(&self) -> Result<ChannelHandle, ChannelError>;
    async fn stop(&self) -> Result<(), ChannelError>;
}

// Tool — act on the world
trait Tool: Send + Sync + 'static {
    fn info(&self) -> &ToolInfo;
    fn to_definition(&self) -> ToolDefinition;
    async fn execute(&self, input: serde_json::Value, ctx: &ToolContext) -> Result<ToolOutput, ToolError>;
}

// Memory — remember conversations
trait Memory: Send + Sync + 'static {
    async fn load(&self, id: &SessionId) -> Result<Option<Conversation>, MemoryError>;
    async fn save(&self, conv: &Conversation) -> Result<(), MemoryError>;
    async fn list_sessions(&self, limit: usize) -> Result<Vec<SessionSummary>, MemoryError>;
    async fn delete(&self, id: &SessionId) -> Result<(), MemoryError>;
    async fn search(&self, query: &str, limit: usize) -> Result<Vec<SessionSummary>, MemoryError>;
}

// Security primitives
Tainted::<Untrusted>::new(raw) -> Tainted<Untrusted>
Tainted::<Untrusted>::sanitize(self, F) -> Result<Tainted<Sanitized>, SecurityError>
Tainted::<Sanitized>::value(&self) -> &str
SafePath::new(root, user_path) -> Result<SafePath, SecurityError>
CapabilityGrant::check(&self, required: &Capability) -> Result<(), SecurityError>
```