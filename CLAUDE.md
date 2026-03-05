# FreeBird — Rust Development Guide

> **Purpose**: Authoritative reference for building FreeBird, a secure, always-running AI agent daemon in Rust. Every section is a hard constraint, not a suggestion. Prioritize security over convenience and explicitness over magic.
>
> **Design philosophy**: Modeled after the open-source OpenClaw/ZeroClaw agent ecosystem but rebuilt from scratch in Rust with compile-time security guarantees. We take the best ideas (trait-driven extensibility, channel/provider abstraction, local-first operation) and discard the security debt (exposed tokens in URLs, unsandboxed tool execution, no taint tracking, 36% of third-party skills containing prompt injection per Cisco research).

---

## Table of Contents

1. [Workspace Layout & Dependencies](#1-workspace-layout--dependencies)
2. [Core Principles](#2-core-principles)
3. [Message Flow](#3-message-flow)
4. [Trait Signatures](#4-trait-signatures)
5. [Agent Runtime Loop](#5-agent-runtime-loop)
6. [Type-Driven Security](#6-type-driven-security)
7. [Agentic Security Model](#7-agentic-security-model)
8. [Prompt Injection Defense](#8-prompt-injection-defense)
9. [Auth & Session Management](#9-auth--session-management)
10. [Channel Pairing — ASI03](#10-channel-pairing--asi03)
11. [Consent Gates — ASI09](#11-consent-gates--asi09)
12. [Network Egress — ASI01](#12-network-egress--asi01)
13. [Token Budgets — ASI08](#13-token-budgets--asi08)
14. [Memory Integrity — ASI06](#14-memory-integrity--asi06)
15. [Audit Logging](#15-audit-logging)
16. [Supply Chain — ASI04](#16-supply-chain--asi04)
17. [Error Handling](#17-error-handling)
18. [Daemon Lifecycle](#18-daemon-lifecycle)
19. [Configuration & Secrets](#19-configuration--secrets)
20. [Logging & Concurrency](#20-logging--concurrency)
21. [Testing Strategy](#21-testing-strategy)
22. [Code Style & Linting](#22-code-style--linting)
23. [Anti-Patterns](#23-anti-patterns)
24. [Security Checklist](#24-security-checklist)

---

## 1. Workspace Layout & Dependencies

### Crate Topology

```
freebird/
├── Cargo.toml                       # Workspace root
├── deny.toml                        # cargo-deny: license + advisory audit
├── config/
│   └── default.toml                 # Default configuration
├── crates/
│   ├── freebird-traits/             # ALL public traits — zero freebird-* deps
│   ├── freebird-types/              # Shared domain types — depends only on freebird-traits
│   ├── freebird-security/           # Taint, SafePath, capabilities, audit — no I/O
│   ├── freebird-runtime/            # Agent loop, session mgmt, orchestration
│   ├── freebird-providers/          # Provider implementations (Anthropic, etc.)
│   ├── freebird-channels/           # Channel implementations (CLI, etc.)
│   ├── freebird-tools/              # Built-in tool implementations
│   ├── freebird-memory/             # Memory backend implementations
│   └── freebird-daemon/             # Binary — thin composition root
```

### Dependency DAG (Strict — Enforced by Cargo)

```
freebird-traits          (zero freebird-* deps — root of the type universe)
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
freebird-daemon          (depends on: ALL — composition root)
```

**Critical rules**:
- `freebird-traits` has **ZERO** `freebird-*` dependencies. Any implementation can be swapped without rebuilding the core.
- `freebird-security` depends on **no other `freebird-*` crate except `freebird-types` and `freebird-traits`** — it cannot be compromised by a vulnerability in a provider or channel.

### Dependency Policy

| Purpose | Crate | Notes |
|---|---|---|
| Async | `tokio` (full features) | Pin major |
| Traits | `async-trait` | Until Rust has native async traits in all positions |
| Serde | `serde`, `serde_json`, `toml` | Universal serialization |
| Errors (lib) | `thiserror` | Precise matchable enums |
| Errors (bin) | `anyhow` | Ergonomic context chains |
| Logging | `tracing`, `tracing-subscriber` | Structured, async-aware |
| HTTP client | `reqwest` + `rustls-tls` | **NEVER openssl** |
| Crypto | `ring` | Hashing, HMAC, random |
| Secrets | `secrecy` | Zeroize + redact in Debug |
| Config | `figment` (toml, env) | Layered config sources |
| Time | `chrono` | Timestamps |
| IDs | `uuid` (v4, serde) | Newtype IDs |
| URLs | `url` | URL parsing + validation |
| Testing | `tempfile`, `wiremock`, `proptest` | Dev-only |

**No `openssl-sys`** anywhere in the dep tree. Run `cargo deny check` on every build. See `Cargo.toml` (workspace root) for exact versions.

---

## 2. Core Principles

### 2.1 Trait-Driven Extensibility

Every subsystem (provider, channel, tool, memory) is defined as a trait in `freebird-traits`. Implementations live in separate crates. Adding a new provider or channel means writing a new struct that implements the trait — zero changes to existing code.

### 2.2 Make Illegal States Unrepresentable

```rust
// GOOD — each state carries only its valid data
enum AgentSession {
    Unauthenticated,
    Authenticated { credential: SessionCredential },
    Connected { credential: SessionCredential, provider: Box<dyn Provider>, channel: Box<dyn Channel> },
    Processing { credential: SessionCredential, provider: Box<dyn Provider>, channel: Box<dyn Channel>, current_turn: Turn },
}
```

### 2.3 Parse, Don't Validate

Transform unstructured data into typed structures at system boundaries. Once parsed, validity is guaranteed by the type.

### 2.4 Fail Loudly at Boundaries, Propagate Gracefully Inside

All input validation happens at the transport edge. Internal code propagates errors with `?` and added `.context()`.

### 2.5 Zero `unsafe` in Application Code

The only acceptable `unsafe` is inside FFI wrappers with `// SAFETY:` comments. Application logic must be 100% safe Rust.

### 2.6 Ownership Communicates Intent

```rust
fn inspect(data: &Config)          // borrowing: I look but don't touch
fn mutate(data: &mut Config)       // exclusive borrow: I will change this
fn consume(data: Config)           // ownership transfer: caller loses access
fn produce() -> Config             // returning owned: caller now owns this
```

---

## 3. Message Flow

### The Complete Message Path

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

**Core types**: `Role`, `ContentBlock`, `Message`, `CompletionRequest`, `CompletionResponse`, `StreamEvent`, `StopReason`, `TokenUsage` — see `crates/freebird-traits/src/provider.rs`.

**Conversation types**: `Turn`, `ToolInvocation`, `Conversation`, `SessionSummary` — see `crates/freebird-traits/src/memory.rs`.

**Newtype IDs**: `SessionId`, `ChannelId`, `ProviderId`, `ModelId` — see `crates/freebird-traits/src/id.rs`. All IDs use `define_id!` macro with `generate()` (UUID v4) and `from_string()`.

---

## 4. Trait Signatures

All traits live in `freebird-traits` (zero `freebird-*` deps). Implementations in separate crates.

### Provider — talk to LLMs

See `crates/freebird-traits/src/provider.rs`.

```rust
#[async_trait]
pub trait Provider: Send + Sync + 'static {
    fn info(&self) -> &ProviderInfo;
    async fn validate_credentials(&self) -> Result<(), ProviderError>;
    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse, ProviderError>;
    async fn stream(&self, request: CompletionRequest) -> Result<Pin<Box<dyn Stream<Item = Result<StreamEvent, ProviderError>> + Send>>, ProviderError>;
}
```

Key design notes:
- `ProviderInfo.features` uses `BTreeSet<ProviderFeature>` (not bool flags). Check with `info().supports(&ProviderFeature::Streaming)`.
- `ProviderError::Network` uses structured `NetworkErrorKind { Timeout, ConnectionRefused, DnsFailure, TlsError, Other }` — not `#[from] reqwest::Error`.
- IDs use newtypes: `ProviderInfo.id` is `ProviderId`, `ModelInfo.id` is `ModelId`, `CompletionRequest.model` is `ModelId`.

### Channel — talk to users

See `crates/freebird-traits/src/channel.rs`.

```rust
#[async_trait]
pub trait Channel: Send + Sync + 'static {
    fn info(&self) -> &ChannelInfo;
    async fn start(&self) -> Result<ChannelHandle, ChannelError>;
    async fn stop(&self) -> Result<(), ChannelError>;
}
```

Key design notes:
- `ChannelInfo.features` uses `BTreeSet<ChannelFeature>` (Media, Streaming).
- Auth uses `AuthRequirement` enum (`None`, `Required`) — not a bool.
- `ChannelHandle` returns `{ inbound: Pin<Box<dyn Stream<Item = InboundEvent> + Send>>, outbound: mpsc::Sender<OutboundEvent> }`.
- Channel owns its event loop internally; runtime consumes the uniform stream/sender interface.

### Tool — act on the world

See `crates/freebird-traits/src/tool.rs`.

```rust
#[async_trait]
pub trait Tool: Send + Sync + 'static {
    fn info(&self) -> &ToolInfo;
    fn to_definition(&self) -> ToolDefinition; // default impl provided
    async fn execute(&self, input: serde_json::Value, context: &ToolContext<'_>) -> Result<ToolOutput, ToolError>;
}
```

Key design notes:
- `ToolContext<'a>` uses borrowed references: `session_id: &'a SessionId`, `sandbox_root: &'a Path`, `granted_capabilities: &'a [Capability]`.
- `Capability` enum: `FileRead`, `FileWrite`, `FileDelete`, `ShellExecute`, `ProcessSpawn`, `NetworkOutbound`, `NetworkListen`, `EnvRead`. **Append new variants at the end only** (Ord-derived, affects HMAC signatures).
- `RiskLevel` enum: `Low < Medium < High < Critical` (Ord-derived for consent gate comparison).

### Memory — persist conversations

See `crates/freebird-traits/src/memory.rs`.

```rust
#[async_trait]
pub trait Memory: Send + Sync + 'static {
    async fn load(&self, session_id: &SessionId) -> Result<Option<Conversation>, MemoryError>;
    async fn save(&self, conversation: &Conversation) -> Result<(), MemoryError>;
    async fn list_sessions(&self, limit: usize) -> Result<Vec<SessionSummary>, MemoryError>;
    async fn delete(&self, session_id: &SessionId) -> Result<(), MemoryError>;
    async fn search(&self, query: &str, limit: usize) -> Result<Vec<SessionSummary>, MemoryError>;
}
```

---

## 5. Agent Runtime Loop

See `crates/freebird-runtime/src/agent.rs` for the full implementation.

### Conceptual Flow

1. **Receive** `InboundEvent` from channel via fan-in `mpsc` channel
2. **Taint** raw input: `Tainted::new(&raw_text)`
3. **Sanitize** via `SafeMessage::from_tainted()` — rejects injection patterns, enforces length
4. **Load or create** conversation from memory backend
5. **Build** `CompletionRequest` with conversation history + tool definitions
6. **Agentic loop** (up to `max_tool_rounds`):
   - Call provider (`complete_with_failover`)
   - If `StopReason::ToolUse` → execute tools via `ToolExecutor`, wrap output in `ScannedToolOutput::from_raw()`, append results, continue loop
   - If `StopReason::EndTurn` → wrap response in `ScannedModelResponse::from_raw()`, deliver to channel, break
   - If `StopReason::MaxTokens` → deliver truncated response, break
7. **Persist** conversation to memory

### Key Invariants

- `handle_message()` is **infallible** — all errors are caught and sent as `OutboundEvent::Error`, never propagated
- **Tool output taint**: `ScannedToolOutput::from_raw()` wraps tool results before they enter the LLM context. On injection detection, a synthetic `ToolResult { is_error: true }` replaces the raw content
- **Model output taint**: `ScannedModelResponse::from_raw()` wraps model responses before delivery. On injection detection, the response is **NOT** saved to memory (prevents memory poisoning)
- Tool execution is always gated by capability check + timeout
- `CancellationToken` enables cooperative shutdown mid-loop

---

## 6. Type-Driven Security

### Taint Tracking

See `crates/freebird-security/src/taint.rs`.

All data from external sources MUST be wrapped in `Tainted`. The security boundary is enforced by `pub(crate)` visibility on `inner()`:

- `Tainted::new(raw)` — wraps untrusted input. Debug impl shows `[TAINTED len=N]`, never the content.
- `tainted.inner()` — `pub(crate)` only. Only safe type factories inside `freebird-security` can access raw content.
- `TaintedToolInput::new(value)` — wraps untrusted `serde_json::Value` from LLM tool calls. Provides `extract_string()`, `extract_path()`, `extract_shell_arg()`, `extract_url()` methods that return safe types.

**Rules**: Never implement `Deref`, `Display`, `AsRef<str>`, or `Into<String>` on `Tainted`. Never bypass `pub(crate) inner()`.

### Safe Type Factories

See `crates/freebird-security/src/safe_types.rs`.

Each safe type is constructed from `Tainted` via a factory that validates + sanitizes:

| Safe Type | Factory | What it validates |
|---|---|---|
| `SafeMessage` | `from_tainted(&Tainted)` | Injection scan, length ≤ 32k, strips control chars |
| `SafeFilePath` | `from_tainted(&Tainted, &Path)` | Path traversal, symlink resolution, null bytes, sandbox containment |
| `SafeFilePath` | `from_tainted_for_creation(...)` | Same + parent dir must exist, no overwriting symlinks |
| `SafeShellArg` | `from_tainted(&Tainted)` | Rejects `|;&\`'"*?` and other shell metacharacters |
| `SafeUrl` | `from_tainted(&Tainted, &EgressPolicy)` | HTTPS only, host allowlist, DNS rebinding check |
| `Redacted` | `from_tainted(&Tainted)` | Truncates + strips control chars for safe logging |

### Output Taint Types

| Type | Purpose | On detection |
|---|---|---|
| `ScannedToolOutput` | Blocks injection before tool output enters LLM context | Synthetic error `ToolResult { is_error: true }` replaces raw content |
| `ScannedModelResponse` | Blocks injection before model output reaches user | `OutboundEvent::Error` sent; response NOT saved to memory |

Both are constructed via `::from_raw(content)` which calls `injection::scan_output()`. Enforcement happens in `freebird-runtime` (the agentic loop), not at the trait level.

---

## 7. Agentic Security Model

### Capability System

See `crates/freebird-security/src/capability.rs`.

`CapabilityGrant` wraps a `BTreeSet<Capability>` + `sandbox_root` + `expires_at`. Key properties:
- `check(&Capability)` — returns `Err(SecurityError)` if capability not granted or expired
- `derive_sub_grant(subset, narrower_sandbox)` — for sub-agents; subset must be ⊆ parent, expiry must be ≤ parent
- Time-based privilege escalation prevention: sub-grant expiry is clamped to parent's expiry

### Tool Execution Invariant

Every tool invocation follows this **mandatory** sequence, enforced by `ToolExecutor`:

1. Capability check → deny if missing or expired
2. Input validation → reject if malformed (via `TaintedToolInput`)
3. Resource boundary check (`SafeFilePath`, allowed hosts) → reject if out of bounds
4. Audit log the attempt (pass or fail)
5. Execute with timeout (`tokio::time::timeout`)
6. Audit log the result
7. Scan output for injection via `ScannedToolOutput::from_raw()` before returning to LLM

---

## 8. Prompt Injection Defense

### Defense Layers

| Layer | What it catches | Where it runs |
|---|---|---|
| Input taint + `SafeMessage` | Malformed/malicious user input | Router (before agent) |
| Input injection scan | Known prompt injection patterns | `SafeMessage::from_tainted()` |
| Capability system | Unauthorized tool access | ToolExecutor (before tool) |
| `SafeFilePath` | Path traversal in tool args | Tool implementation |
| `ScannedToolOutput` | Indirect injection via tool results | Agent loop (after tool, before LLM) |
| `ScannedModelResponse` | Compromised model responses | Agent loop (before channel) |
| Audit logging | Post-incident forensics | Every layer |

### Injection Scanning

See `crates/freebird-security/src/injection.rs`.

Three scan functions: `scan_input()`, `scan_output()`, `scan_context()`. Each scans for known injection patterns (e.g., "ignore previous instructions", ChatML markers, system block injection). Uses dual normalization (stripped + spaced) to defeat Unicode zero-width character evasion.

### Reader Agent Pattern (Future)

For high-risk operations (reading untrusted files, scraping web content), use a **separate, tool-disabled reader agent** to summarize content before passing it to the main agent. This architectural isolation prevents indirect injection from reaching the agent that has tool access.

---

## 9. Auth & Session Management

### Session Keys

Users authenticate via session keys — 32-byte cryptographically random tokens (`ring::rand::SystemRandom`).

- Keys are stored as **SHA-256 hashes**, never raw
- Keys have configurable TTL (default: 30 days)
- Each key carries a `CapabilityGrant` that scopes what the session can do
- Failed auth attempts are rate-limited and audit-logged
- Verification uses constant-time comparison via hash

### Provider Credentials

- API keys wrapped in `secrecy::SecretString` — zeroized on drop, redacted in Debug
- Loaded from environment variables (e.g., `FREEBIRD_PROVIDERS__0__API_KEY`)
- **NEVER** hardcode secrets, log secrets, or include secrets in error messages

### Auth Flow

```
User starts session
    ├─ CLI: session key via --key flag or env var (local-only, pairing exempt)
    ├─ Signal: pairing code flow (see §10)
    └─ WebSocket: key in initial handshake header
    │
    ▼
Router verifies session key → extract CapabilityGrant → attach to session
```

---

## 10. Channel Pairing — ASI03

**OWASP ASI03 — Identity & Privilege Abuse**: Prevents unauthorized senders from interacting with the agent.

### Design

All non-CLI channels require explicit pairing before message processing. The pairing gate sits in the router, **before** taint processing — unpaired senders never reach the agent.

### Pairing State Machine

```
Unknown → PendingApproval { code, issued_at, expires_at, attempts } → Paired { capability_grant }
                                                                    → Blocked { reason }
```

### Key Rules

- Pairing codes: cryptographically random (`ring::rand::SystemRandom`), time-limited (default 60 min), numeric (e.g., "847291")
- Code comparison: constant-time via `ring::constant_time::verify_slices_are_equal`
- Max failed attempts before auto-block (default: 5)
- Approval via CLI: `freebird pair approve --code <code>` — transitions sender to `Paired` with scoped capabilities
- Pairing codes are **NEVER** logged — only that one was issued
- All state transitions are audit-logged

---

## 11. Consent Gates — ASI09

**OWASP ASI09 — Human-Agent Trust Exploitation**: Ensures high-risk operations require explicit human approval before execution.

### Risk Classification

Every tool action is classified by `RiskLevel`. The consent gate compares `tool_risk >= config.require_consent_above`.

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

### Consent Flow

1. `ConsentGate::check()` evaluates tool's `RiskLevel` against policy threshold
2. If consent required: send `ConsentRequest` to user via channel (tool name, action summary, affected resources, reversibility)
3. Wait for response with timeout (default: 60s)
4. On approval → execute. On denial/timeout → return `ToolError::ConsentDenied`/`ConsentExpired`
5. All consent decisions are audit-logged

---

## 12. Network Egress — ASI01

**OWASP ASI01 — Agent Goal Hijacking**: Prevents the agent from being turned into an exfiltration engine.

### Design

See `crates/freebird-security/src/egress.rs`.

The agent can **ONLY** make outbound network requests to explicitly allowed hosts. Enforced at both the tool level AND HTTP client level.

### Default Policy

```toml
[security.egress]
allowed_hosts = ["api.anthropic.com", "api.openai.com"]
allowed_ports = [443]
max_request_body_bytes = 1048576  # 1MB — prevents data exfiltration
rate_limit_per_minute = 60
```

Adding any host requires a config change — the agent cannot modify its own egress policy.

### DNS Rebinding Prevention

Validate that resolved IPs are not private/loopback addresses (10.x, 172.16-31.x, 192.168.x, 127.x, ::1). Prevents SSRF to internal services.

---

## 13. Token Budgets — ASI08

**OWASP ASI08 — Agent Resource & Service Exhaustion**: Prevents unbounded token/compute consumption.

### Budget System

`TokenBudget` tracks consumption per session using `AtomicU64` counters:
- **Per-session limit**: max total input + output tokens (default: 500k)
- **Per-request limit**: max tokens per single LLM call (default: 32k)
- **Per-turn limit**: max tool rounds in a single agentic loop (default: 10)
- **Cost limit**: max microdollars per session (default: $5.00)

On budget exhaustion, returns a descriptive `SecurityError::BudgetExceeded` — never silently truncates. The agentic loop checks `budget.check_tool_rounds(round)` before each iteration and `budget.record_usage(&usage)` after each provider response.

---

## 14. Memory Integrity — ASI06

**OWASP ASI06 — Memory & Context Poisoning**: Protects conversation history from tampering.

### Conversation Signing

- Every persisted conversation is HMAC-signed (`ring::hmac::Key` with SHA-256)
- Signature verified on every load — tampered files quarantined (moved to `quarantine/` directory), not loaded
- HMAC key derived from server-side secret, never stored alongside conversation files

### Context Injection Defense

Before appending loaded conversation history to a provider request, `scan_context()` checks for content resembling system prompts or instruction overrides (e.g., "you are now", "new system prompt", ChatML markers `<|system|>`, `[INST]`, `<<SYS>>`).

---

## 15. Audit Logging

### Hash-Chained Audit Log

Each audit entry includes the SHA-256 hash of the previous entry, creating a tamper-evident chain:

```jsonl
{"entry": {"sequence": 0, "event": {...}, "timestamp": "...", "previous_hash": ""}, "hash": "a1b2..."}
{"entry": {"sequence": 1, "event": {...}, "timestamp": "...", "previous_hash": "a1b2..."}, "hash": "c3d4..."}
```

- Append-only JSON lines — no in-place mutation
- If an entry is modified or deleted, the hash chain breaks and `verify_audit_chain()` detects it
- Startup routine verifies chain integrity
- See `crates/freebird-security/src/audit.rs`

---

## 16. Supply Chain — ASI04

**OWASP ASI04 — Agentic Supply Chain Vulnerabilities**

### v1: No Runtime Plugin Loading

All tools are compiled into the binary. This eliminates the supply chain attack surface entirely. Future skill system (v2+) will require WASM sandboxing, content-addressed signing, and explicit human approval.

### CI Gates

```bash
cargo deny check advisories   # known CVEs — blocks build
cargo deny check licenses     # allowlist: MIT, Apache-2.0, BSD-2/3-Clause, ISC, Zlib
cargo deny check bans         # banned: openssl, openssl-sys
cargo deny check sources      # only crates.io, no git deps
cargo audit --deny-warnings   # weekly RustSec advisory check
```

See `deny.toml` for configuration.

---

## 17. Error Handling

**Library crates** (`freebird-traits`, `freebird-security`, `freebird-providers`, etc.): `thiserror` with precise, matchable error enums. Each crate defines its own error type.

**Binary crate** (`freebird-daemon`): `anyhow` with `.context()` for ergonomic error chains.

**Hard rules**:
- NEVER `.unwrap()` in production code
- NEVER `.expect()` without a comment explaining why the invariant holds
- Use `?` for propagation. Add `.context()` at meaningful boundaries
- Panics are reserved for true programmer errors (violated invariants), never recoverable conditions

---

## 18. Daemon Lifecycle

`freebird-daemon/src/main.rs` is a **thin composition root** — under 30 lines of logic. It loads config, initializes logging, builds channels/providers/memory, constructs `AgentRuntime`, and calls `runtime.run(channels)`.

**Graceful shutdown**: `ShutdownCoordinator` listens for SIGINT/SIGTERM, cancels via `CancellationToken`, drains in-flight requests within a configurable timeout.

See `crates/freebird-daemon/src/main.rs` and `crates/freebird-runtime/src/shutdown.rs`.

---

## 19. Configuration & Secrets

- **Layered**: `config/default.toml` → environment variables via `figment`
- **Env var convention**: `FREEBIRD_SECTION__KEY` (double underscore for nesting)
- **Provider API keys**: loaded from env vars (e.g., `FREEBIRD_PROVIDERS__0__API_KEY`), wrapped in `SecretString`
- **Session keys**: generated via `freebird keygen`, stored in `~/.freebird/keys.json`
- **NEVER** hardcode secrets. **NEVER** log secrets. **NEVER** include secrets in error messages.

See `config/default.toml` and `crates/freebird-types/src/config.rs` for typed config structs.

---

## 20. Logging & Concurrency

### Logging

Use `tracing` for everything. Console output (pretty) for developers, JSON lines to audit file for SIEM. Initialize via `tracing-subscriber` with `EnvFilter`.

### Concurrency Rules

- **Prefer channels over shared state.** The runtime uses `mpsc` channels to fan-in events from multiple channels.
- **Use `tokio::sync::Mutex`** for async code, **never** `std::sync::Mutex` (exception: nanosecond-held locks with no `.await`).
- **Never hold a lock across `.await`.**
- **Use `tokio::select!`** for racing futures (message receive vs. shutdown signal).
- **Use `CancellationToken`** for cooperative shutdown across spawned tasks.

---

## 21. Testing Strategy

### Unit Tests

Every module has `#[cfg(test)]` tests. Security modules have adversarial tests with injection payloads, path traversal attempts, and Unicode evasion.

Gate with `#[allow(clippy::unwrap_used)]` on test modules (production code must use `?`).

### Property-Based Tests

`proptest` for security-critical invariants:
- `SafeFilePath` never escapes sandbox for arbitrary input
- `Tainted` never leaks raw content through any public API
- Injection scanner handles arbitrary Unicode without panic

### Integration Tests

Cross-crate tests in `crates/*/tests/` that wire up the full stack with mocked providers. Test the complete agent loop: send message → provider responds with tool_use → tool executes → provider responds with end_turn → response delivered.

---

## 22. Code Style & Linting

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

## 23. Anti-Patterns

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
| Raw `PathBuf` in tool I/O | `SafeFilePath` |
| Raw `String` from network | `Tainted` |
| `bool` feature flags | `BTreeSet<FeatureEnum>` with `.supports()` |

---

## 24. Security Checklist

### Input Handling

- [ ] All external input enters as `Tainted`
- [ ] Sanitization via safe type factories before business logic
- [ ] Input scanned for prompt injection patterns
- [ ] Validators are unit-tested with adversarial inputs

### Path Safety

- [ ] All filesystem ops use `SafeFilePath`
- [ ] Symlinks resolved before boundary checks
- [ ] Path traversal tests include `..`, symlinks, null bytes, encoded slashes

### Capabilities

- [ ] Every tool checks capabilities before execution
- [ ] Grants have expiration times
- [ ] Sub-agent grants are strict subsets of parent
- [ ] No ambient authority / global admin bypass

### Prompt Injection Defense

- [ ] Input scanning on all user messages via `SafeMessage`
- [ ] Output scanning on all tool results via `ScannedToolOutput` — blocks with synthetic error
- [ ] Output scanning on model responses via `ScannedModelResponse` — blocks delivery + prevents memory poisoning
- [ ] Reader agent pattern for untrusted external content (future)

### Auth & Sessions

- [ ] Session keys are cryptographically random (32 bytes from `SystemRandom`)
- [ ] Keys stored as SHA-256 hashes, never raw
- [ ] Keys have configurable TTL
- [ ] Failed auth attempts are rate-limited and audit-logged
- [ ] Provider API keys use `SecretString` — never logged, never serialized

### Channel Pairing (ASI03)

- [ ] All non-CLI channels require explicit pairing before message processing
- [ ] Pairing codes are cryptographically random, time-limited, single-use
- [ ] Code comparison uses constant-time equality
- [ ] Failed pairing attempts capped before auto-block
- [ ] Router rejects all `InboundEvent`s from unpaired channels

### Consent Gates (ASI09)

- [ ] All tools classified by `RiskLevel`
- [ ] `High` and `Critical` tools require explicit human approval
- [ ] Consent requests include tool name, action summary, risk justification
- [ ] Consent timeouts default to 60s — no indefinite waits
- [ ] Consent decisions are audit-logged

### Network Egress (ASI01)

- [ ] All outbound HTTP routed through `EgressPolicy` — no direct `reqwest` calls
- [ ] Default policy is deny-all; only allowlisted hosts reachable
- [ ] DNS rebinding prevention: resolved IPs checked against private ranges

### Token Budgets (ASI08)

- [ ] Per-session token limits enforced via `TokenBudget`
- [ ] Per-request and per-turn limits prevent single-turn abuse
- [ ] Budget exhaustion returns descriptive error — never silently truncates

### Memory Integrity (ASI06)

- [ ] Persisted conversations HMAC-signed
- [ ] Signature verified on every load — tampered files quarantined
- [ ] Context injection scan on loaded history before sending to provider

### Audit Log Integrity

- [ ] Every audit entry includes SHA-256 hash of previous entry (hash chain)
- [ ] Audit entries are append-only JSON lines
- [ ] Startup verifies audit chain integrity

### Supply Chain (ASI04)

- [ ] No runtime plugin loading — all code compiled into binary
- [ ] `cargo deny check` passes in CI
- [ ] `cargo audit` runs with `--deny-warnings`

### Concurrency

- [ ] No `std::sync::Mutex` in async code
- [ ] No locks held across `.await`
- [ ] Graceful shutdown with drain timeout

### Dependencies

- [ ] `cargo deny check advisories` passes
- [ ] No `openssl-sys` in dep tree
- [ ] No new `unsafe` without justification
