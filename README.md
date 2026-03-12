# FreeBird

[![CI](https://github.com/JoshCap20/freebird/actions/workflows/ci.yml/badge.svg)](https://github.com/JoshCap20/freebird/actions/workflows/ci.yml)

A security-first AI agent runtime written in Rust.

Freebird is a Rust reimplementation of [OpenClaw](https://github.com/openclaw) — an always-on AI agent you communicate with through pluggable channels (CLI, Signal, and more). I rebuilt it from scratch because **agentic AI demands a fundamentally different security posture** than OpenClaw provides today. Additionally, I really wanted to name a project `freebird`.

## Why Rust? Why Rebuild?

OpenClaw is a capable agent, but its architecture inherits trust assumptions that don't hold up under adversarial conditions. Recent findings make the case clearly:

- **36% of community-shared ClawHub skills contain prompt injection payloads** (Cisco Talos, 2026). Any public skill marketplace is an attack surface.
- **CVE-2026-25253** (CVSS 8.8) demonstrated that OpenClaw's tool output pipeline trusts data it shouldn't, allowing a crafted tool response to escalate privileges.
- **The OWASP Agentic Security Initiative (ASI) Top 10** catalogues the systemic risks — goal hijacking, identity abuse, memory poisoning, resource exhaustion — that agentic systems must defend against by design, not by patch.

Freebird addresses these risks at the architecture level, using Rust's type system and ownership model to enforce invariants that are impossible to express in TypeScript.

## Security Model

### No Public Skill Marketplace

OpenClaw's ClawHub is a convenience that doubles as an injection vector. Freebird takes a different approach: ask the agent to build what you need, or provide a tool you trust. There is no runtime plugin loading — every tool is compiled into the binary, audited at build time, and subject to capability checks at execution time.

### Compile-Time Taint Tracking

All external input — user messages, tool outputs, API responses — enters the system wrapped in `Tainted`. It cannot be used in tool invocations or passed to the LLM without first passing through an explicit sanitization boundary. Taint tracking is enforced at compile time — the type system prevents unsanitized data from reaching dangerous operations. Injection *detection* (pattern scanning) is a runtime check, but the guarantee that every external input passes through a sanitization boundary is structural. Injection detection response is configurable per boundary — `block`, `prompt` (escalate to human approval), or `allow` — with model output and context injections always blocked.

### Explicit Approval for Sensitive Actions

High-risk tool executions (file writes outside the sandbox, shell commands, network requests) require human approval through an approval gate. The agent cannot silently escalate. Every tool is classified by risk level, and `High` and `Critical` operations pause for confirmation before proceeding. The approval system supports two categories: `Consent` (action-driven, risk-based) and `SecurityWarning` (threat-driven, e.g., injection detection).

### Secret Guard

Tool invocations that access sensitive files (`.env`, private keys, credentials) or run secret-revealing commands (`printenv`, `cat ~/.ssh/*`) are detected by the `SecretGuard`. On detection, the action is escalated to `RiskLevel::Critical` approval or blocked outright, depending on configuration. Tool output is also scanned and detected secrets are redacted before returning to the LLM context.

### Channel Pairing & Identity Verification

The security layer includes cryptographic pairing primitives for remote channels — time-limited pairing codes, constant-time verification, and auto-block after failed attempts. The current TCP channel is local-only and pairing-exempt; enforcement will be wired when remote transports (Signal, WebSocket) are added.

### Sandboxed File Access

All filesystem operations go through `SafeFilePath`, which canonicalizes paths, resolves symlinks, and enforces directory boundaries before any I/O occurs. Path traversal via `..`, symlinks, null bytes, or encoded slashes is validated at construction time via `SafeFilePath::from_tainted()`, ensuring invalid paths never reach I/O operations.

### Tamper-Evident Audit Logging

Every security-relevant action — tool invocations, approval decisions, injection detections, session creation, pairing events — is recorded in an HMAC-chained audit log stored in the encrypted database. Each entry's HMAC covers the previous entry's HMAC, forming a global tamper-evident chain. If a single entry is altered or removed, the chain breaks and `verify_chain()` detects it.

### Memory Integrity

Conversations are persisted as immutable event logs with per-session HMAC chains. Each event's HMAC covers the session ID, sequence number, event data, timestamp, and the previous event's HMAC — forming a tamper-evident chain. On every load, the full chain is verified and tampered sessions are rejected before they reach the LLM context. This defends against the memory poisoning attacks described in OWASP ASI-06.

### Encrypted Database Storage

All persistent data — conversations, knowledge entries, and full-text search indexes — is stored in a single SQLCipher-encrypted SQLite database. Encryption is AES-256-CBC at the page level, transparent to application code.

| Property | Detail |
|----------|--------|
| Cipher | AES-256-CBC (SQLCipher) |
| Key derivation | PBKDF2-HMAC-SHA256, 100k iterations (configurable) |
| Key sources | `FREEBIRD_DB_KEY` env var → keyfile → interactive prompt |
| Key handling | `secrecy::SecretString` — zeroized on drop, redacted in Debug |
| Journal mode | WAL (crash-resilient, concurrent reads) |

The encryption key is never logged, never included in error messages, and never stored alongside the database.

### Knowledge Store

Freebird maintains a long-term knowledge store that persists facts, preferences, and learnings across sessions. Knowledge is stored in the encrypted database with FTS5 full-text search (BM25-ranked, porter stemmer tokenizer).

- **6 knowledge kinds**: `SystemConfig`, `ToolCapability`, `UserPreference`, `LearnedPattern`, `ErrorResolution`, `SessionInsight` — each classified as agent-owned or protected (protected kinds require human consent to modify)
- **Sensitive content filter**: Blocks storage of API keys, passwords, PEM blocks, and other credential material
- **Auto-retrieval**: Relevant knowledge is injected into every conversation turn (configurable threshold and token budget)
- **4 tools**: `store_knowledge`, `search_knowledge`, `update_knowledge`, `delete_knowledge`

### Token Budget Enforcement

Per-session, per-request, and per-turn token limits prevent runaway cost and resource exhaustion (OWASP ASI-08). Budgets are enforced with atomic counters — no race conditions, no silent overruns.

### Network Egress Control

Outbound HTTP is deny-by-default. Only explicitly allowlisted hosts are reachable. Responses are size-capped, and resolved IPs are checked against private ranges to prevent DNS rebinding attacks.

### Threat Model Summary

| Trust Boundary | Trust Level | Key Mitigations |
|----------------|-------------|-----------------|
| User input | Untrusted | `Tainted` → `SafeMessage`, injection scanning |
| LLM provider | Semi-trusted | Output scanning via `ScannedModelResponse`, capability system |
| Tool output | Untrusted | `ScannedToolOutput`, injection scanning, sandbox |
| Filesystem | Trusted-but-sandboxed | `SafeFilePath`, directory boundary enforcement |
| Network | Hostile | Egress allowlist, DNS rebinding prevention, HTTPS only |
| Stored data | Integrity-critical | SQLCipher encryption, per-session HMAC event chains, global HMAC audit chain |
| Channel peers | Local-only (TCP); pairing planned for remote channels | Local binding, pairing primitives available |

## Architecture

Freebird is structured as a Rust workspace with strict crate boundaries:

```
freebird-traits      Zero-dependency trait definitions (Provider, Channel, Tool, Memory, KnowledgeStore, EventSink, AuditSink)
freebird-types       Shared message types and domain objects
freebird-security    Taint, safe types, capabilities, approval gates, secret guard, injection, audit, budgets, egress, auth
freebird-runtime     Agent loop, session management, tool execution, event emission, audit integration
freebird-providers   LLM integrations (Anthropic with streaming + tool use)
freebird-channels    Transport integrations (TCP channel with JSON-line protocol)
freebird-tools       Built-in tool implementations
freebird-memory      SQLCipher-encrypted event-sourced conversations, knowledge store, audit sink (all FTS5-indexed)
freebird-daemon      Binary entry point, config, lifecycle, TUI chat client
```

### Built-in Tools

| Tool | Module | Risk Level |
|------|--------|------------|
| `read_file`, `list_directory` | filesystem | Low |
| `write_file` | filesystem | Medium |
| `search_replace_edit` | edit | Medium |
| `grep_search` | grep | Low |
| `glob_find` | glob_find | Low |
| `file_viewer` | viewer | Low |
| `shell` | shell | High |
| `bash_exec` | bash | Critical |
| `http_request` | network | High |
| `store_knowledge`, `search_knowledge`, `update_knowledge`, `delete_knowledge` | knowledge | Low |
| `repo_map` | repo_map | Low |
| `cargo_verify` | cargo_verify | Medium |
| `list_sessions`, `search_sessions`, `recall_session` | session | Low |

Dependencies flow in one direction. `freebird-traits` and `freebird-types` depend on nothing internal. Security primitives live in `freebird-security` and are used everywhere — they aren't an afterthought bolted onto the runtime.

### Event-Sourced Conversation Persistence

Conversations are persisted as immutable event logs rather than mutable JSON blobs. Every action in the agentic loop — user message, assistant response, tool invocation, turn completion — is appended as a `ConversationEvent` to the encrypted database immediately as it occurs. This provides:

- **Sub-turn crash recovery**: If the daemon crashes mid-turn, the partial state is recoverable by replaying the event log up to the last committed event
- **Per-session HMAC chains**: Each event's HMAC covers the previous event's HMAC, forming a tamper-evident chain per session. On load, the entire chain is verified — a single tampered row causes the session to be rejected with `IntegrityViolation`
- **Denormalized metadata**: Session listing and search operate against a `session_metadata` table maintained by triggers, avoiding full event replay for common operations
- **FTS5 conversation search**: Event data is indexed via FTS5 triggers with porter stemmer tokenization, and search queries are escaped to prevent FTS5 syntax injection

### Security Audit Sink

Security audit events (tool executions, approval decisions, injection detections, session creation) are persisted to a separate `audit_events` table in the same encrypted database. The audit log uses a **global HMAC chain** (not per-session) — every event links to the previous via HMAC, making the entire audit trail tamper-evident and verifiable via `AuditSink::verify_chain()`.

## Status

The core system is functional: daemon with TCP channel, Anthropic provider (streaming + tool use), 16 built-in tools, and all security layers described above are wired and enforced. Persistent storage uses SQLCipher-encrypted SQLite with event-sourced conversations, FTS5 search for both knowledge and conversations, and HMAC-chained audit logging.

**Remaining gaps**: Session auth uses a default permissive capability grant (all capabilities scoped to sandbox). Channel pairing primitives exist but aren't enforced on the TCP channel (local-only). Multi-channel routing is stubbed pending additional transports.

## Getting Started

### Prerequisites

- Rust toolchain (stable, 1.85+)
- An Anthropic API key

### Configuration

Copy and edit the default config:

```bash
cp config/default.toml config/local.toml
export FREEBIRD_CONFIG=config/local.toml
```

Set your Anthropic API key:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

### Installation

Install the `freebird` binary to `~/.cargo/bin/`:

```bash
cargo install --path crates/freebird-daemon
```

### Running the Daemon

Start the daemon:

```bash
freebird serve
```

Connect with the chat client (in another terminal):

```bash
freebird chat
```

> **During development**, you can skip installation and run directly:
> ```bash
> cargo run -p freebird-daemon -- serve
> cargo run -p freebird-daemon -- chat
> ```

### CLI Reference

```
freebird serve   Start the daemon with TCP listener
freebird chat    Connect to a running daemon for interactive chat
freebird status  Check if the daemon is running
freebird stop    Send graceful shutdown to the daemon
freebird replay  Replay a past session as a detailed trace
```

#### `freebird serve` Options

| Flag | Short | Description |
|------|-------|-------------|
| `--allow-dir <PATH>` | `-a` | Grant the agent access to an additional directory (repeatable) |

By default, the agent can only access files within its sandbox directory (`~/.freebird/sandbox`). Use `--allow-dir` to grant access to additional directories — for example, to let the agent work on a project:

```bash
freebird serve --allow-dir ~/Documents/myproject
```

Multiple directories can be allowed:

```bash
freebird serve \
  --allow-dir ~/Documents/project-a \
  --allow-dir ~/src/project-b
```

Allowed directories can also be set in `config/default.toml`:

```toml
[tools]
sandbox_root = "~/.freebird/sandbox"
allowed_directories = ["~/Documents/myproject", "~/src/other"]
```

Paths support `~` expansion. CLI flags are merged with any directories set in config. Relative paths from the agent always resolve against the sandbox; absolute paths are validated against the sandbox and all allowed directories.

#### `freebird replay` Options

| Flag | Description |
|------|-------------|
| `<SESSION_ID>` | Session ID (UUID) to replay |
| `--last` | Replay the most recent session |
| `--json` | Output as JSON instead of human-readable trace |

```bash
# Replay the most recent session
freebird replay --last

# Replay a specific session as JSON
freebird replay abc123-def4-5678-... --json
```

## License

TBD