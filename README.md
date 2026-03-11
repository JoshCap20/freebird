# Freebird - Work in Progress

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

All external input — user messages, tool outputs, API responses — enters the system as `Tainted<Untrusted>`. It cannot be used in tool invocations or passed to the LLM without first passing through an explicit sanitization boundary. This isn't a convention; the Rust compiler enforces it. You cannot accidentally pass unsanitized data to a dangerous operation.

### Explicit Approval for Sensitive Actions

High-risk tool executions (file writes outside the sandbox, shell commands, network requests) require human approval through a consent gate. The agent cannot silently escalate. Every tool is classified by risk level, and `High` and `Critical` operations pause for confirmation before proceeding.

### Channel Pairing & Identity Verification

Non-CLI channels (Signal, future integrations) must complete a cryptographic pairing handshake before the agent will process any messages. Unpaired channels are rejected at the router level — no parsing, no processing, no attack surface.

### Sandboxed File Access

All filesystem operations go through `SafePath`, which canonicalizes paths, resolves symlinks, and enforces directory boundaries before any I/O occurs. Path traversal via `..`, symlinks, null bytes, or encoded slashes is caught at construction time, not at use time.

### Tamper-Evident Audit Logging

Every action — user messages, tool invocations, approval decisions, pairing events — is recorded in a hash-chained audit log. Each entry includes the SHA-256 hash of the previous entry. If a single log line is altered or removed, the chain breaks and the tampering is detected on the next verification pass.

### Memory Integrity

Persisted conversations are HMAC-signed. On every load, the signature is verified. Tampered conversation files are quarantined automatically — they never reach the LLM context. This defends against the memory poisoning attacks described in OWASP ASI-06.

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

- **6 knowledge kinds**: `fact`, `preference`, `learned_pattern`, `correction`, `context`, `reference` — each classified as agent-owned or protected
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
| Stored data | Integrity-critical | SQLCipher encryption, HMAC signing |
| Channel peers | Untrusted until paired | Cryptographic pairing, constant-time verification |

## Architecture

Freebird is structured as a Rust workspace with strict crate boundaries:

```
freebird-traits      Zero-dependency trait definitions (Provider, Channel, Tool, Memory)
freebird-types       Shared message types and domain objects
freebird-security    Taint system, SafePath, capabilities, pairing, consent gates
freebird-runtime     Agent loop, routing, token budgets
freebird-providers   LLM integrations (Anthropic first)
freebird-channels    Transport integrations (CLI first, Signal planned)
freebird-tools       Built-in tool implementations
freebird-memory      SQLCipher-encrypted conversation + knowledge persistence with FTS5
freebird-daemon      Binary entry point, config, lifecycle
```

Dependencies flow in one direction. `freebird-traits` and `freebird-types` depend on nothing internal. Security primitives live in `freebird-security` and are used everywhere — they aren't an afterthought bolted onto the runtime.

Will likely eventually move crates to separate repos if project is ever big enough

## Status

Freebird is in early development. The current focus is on the core agent loop, CLI channel, Anthropic provider, and the security layer described above.

## Getting Started

### Prerequisites

- Rust toolchain (stable, 1.75+)
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

## License

TBD