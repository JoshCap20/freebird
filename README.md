# Freebird - Work in Progress

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

### Token Budget Enforcement

Per-session, per-request, and per-turn token limits prevent runaway cost and resource exhaustion (OWASP ASI-08). Budgets are enforced with atomic counters — no race conditions, no silent overruns.

### Network Egress Control

Outbound HTTP is deny-by-default. Only explicitly allowlisted hosts are reachable. Responses are size-capped, and resolved IPs are checked against private ranges to prevent DNS rebinding attacks.

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
freebird-memory      Conversation persistence with HMAC signing
freebird-daemon      Binary entry point, config, lifecycle
```

Dependencies flow in one direction. `freebird-traits` and `freebird-types` depend on nothing internal. Security primitives live in `freebird-security` and are used everywhere — they aren't an afterthought bolted onto the runtime.

## Status

Freebird is in early development. The current focus is on the core agent loop, CLI channel, Anthropic provider, and the security layer described above.

## License

TBD