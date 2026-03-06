# Consent Bridge Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Wire ConsentGate end-to-end so high-risk tools prompt the user for approval before executing.

**Architecture:** Phase 1 replaces direct `tool.execute()` in AgentRuntime with `ToolExecutor`, bringing the full security pipeline into the runtime. Phase 2 adds a consent bridge — a `select!` arm that forwards `ConsentRequest`s to the user and routes responses back to the gate.

**Tech Stack:** Rust, tokio (select!, mpsc, oneshot), freebird-security (ConsentGate), freebird-runtime (AgentRuntime, ToolExecutor)

**Design doc:** `docs/plans/2026-03-06-consent-bridge-design.md`

---

## Phase 1: ToolExecutor Integration into AgentRuntime

### Task 1: Add `sender_id` to `ConsentRequest` and `ConsentGate::check()`

**Files:**
- Modify: `crates/freebird-security/src/consent.rs`

**Step 1: Update `ConsentRequest` struct — add `sender_id` field**

In `consent.rs`, add `sender_id: String` to the `ConsentRequest` struct after `action_summary`:

```rust
pub struct ConsentRequest {
    pub id: String,
    pub tool_name: String,
    pub description: String,
    pub risk_level: RiskLevel,
    pub action_summary: String,
    pub sender_id: String, // NEW — who triggered this, for routing consent prompts
    pub requested_at: DateTime<Utc>,
    pub expires_at: DateTime<Utc>,
}
```

**Step 2: Update `check()` signature — add `sender_id: &str` param**

Change the signature and include `sender_id` in the `ConsentRequest` construction:

```rust
pub async fn check(
    &self,
    tool_info: &ToolInfo,
    action_summary: String,
    sender_id: &str,           // NEW
) -> Result<(), ConsentError> {
```

Inside `check()`, when building the `ConsentRequest` (around line 141-151), add:

```rust
sender_id: sender_id.to_owned(),
```

**Step 3: Update all 21 consent tests**

Every call to `gate.check(tool_info, summary)` becomes `gate.check(tool_info, summary, "test-sender")`. This is a mechanical search-replace.

Pattern: `gate.check(&info, ` → `gate.check(&info, ` stays the same, but add `, "test-sender"` before the closing `)`.

Also update any tests that inspect `ConsentRequest` fields to account for the new `sender_id` field.

**Step 4: Run tests**

```bash
cargo test -p freebird-security -- consent
cargo clippy -p freebird-security --all-targets -- -D warnings
```

Expected: all consent tests pass, no clippy warnings.

**Step 5: Commit**

```bash
git add crates/freebird-security/src/consent.rs
git commit -m "feat(security): add sender_id to ConsentRequest for consent routing"
```

---

### Task 2: Propagate `sender_id` through `ToolExecutor::execute()`

**Files:**
- Modify: `crates/freebird-runtime/src/tool_executor.rs`

**Step 1: Add `sender_id: &str` to `execute()` and `check_consent()`**

Update `execute()` signature:

```rust
pub async fn execute(
    &self,
    tool_name: &str,
    input: serde_json::Value,
    grant: &CapabilityGrant,
    session_id: &SessionId,
    sender_id: &str,           // NEW
) -> ToolOutput {
```

Pass it through to `check_consent()`:

```rust
if let Some(output) = self
    .check_consent(tool_name, tool.info(), &input, session_id, sender_id)
    .await
{
    return output;
}
```

Update `check_consent()` signature:

```rust
async fn check_consent(
    &self,
    tool_name: &str,
    tool_info: &freebird_traits::tool::ToolInfo,
    input: &serde_json::Value,
    session_id: &SessionId,
    sender_id: &str,           // NEW
) -> Option<ToolOutput> {
```

Inside `check_consent()`, pass `sender_id` to `consent.check()`:

```rust
match consent.check(tool_info, action_summary, sender_id).await {
```

**Step 2: Update all ~32 tool_executor tests**

Every call to `executor.execute(name, input, &grant, &session_id)` becomes `executor.execute(name, input, &grant, &session_id, "test-sender")`. Mechanical search-replace.

**Step 3: Run tests**

```bash
cargo test -p freebird-runtime -- tool_executor
cargo clippy -p freebird-runtime --all-targets -- -D warnings
```

Expected: all 32 tool_executor tests pass, no clippy warnings.

**Step 4: Commit**

```bash
git add crates/freebird-runtime/src/tool_executor.rs
git commit -m "feat(runtime): propagate sender_id through ToolExecutor::execute()"
```

---

### Task 3: Replace `Vec<Box<dyn Tool>>` with `ToolExecutor` in `AgentRuntime`

**Files:**
- Modify: `crates/freebird-runtime/src/agent.rs`

This is the largest task. It changes the struct, constructor, and all `self.tools` / `self.tools_config` references.

**Step 1: Update struct and constructor**

Replace:

```rust
pub struct AgentRuntime {
    provider_registry: ProviderRegistry,
    channel: Box<dyn Channel>,
    tools: Vec<Box<dyn Tool>>,
    memory: Box<dyn Memory>,
    config: RuntimeConfig,
    tools_config: ToolsConfig,
    audit: Option<AuditLogger>,
    sessions: SessionManager,
}
```

With:

```rust
pub struct AgentRuntime {
    provider_registry: ProviderRegistry,
    channel: Box<dyn Channel>,
    tool_executor: crate::tool_executor::ToolExecutor,
    consent_rx: Option<tokio::sync::mpsc::Receiver<freebird_security::consent::ConsentRequest>>,
    memory: Box<dyn Memory>,
    config: RuntimeConfig,
    audit: Option<AuditLogger>,
    sessions: SessionManager,
}
```

Update constructor:

```rust
pub fn new(
    provider_registry: ProviderRegistry,
    channel: Box<dyn Channel>,
    tool_executor: crate::tool_executor::ToolExecutor,
    consent_rx: Option<tokio::sync::mpsc::Receiver<freebird_security::consent::ConsentRequest>>,
    memory: Box<dyn Memory>,
    config: RuntimeConfig,
    audit: Option<AuditLogger>,
) -> Self {
    Self {
        provider_registry,
        channel,
        tool_executor,
        consent_rx,
        memory,
        config,
        audit,
        sessions: SessionManager::new(),
    }
}
```

**Step 2: Update `prepare_agentic_loop()` — tool definitions**

Replace:

```rust
let tool_definitions: Vec<ToolDefinition> =
    self.tools.iter().map(|t| t.to_definition()).collect();
```

With:

```rust
let tool_definitions = self.tool_executor.tool_definitions();
```

**Step 3: Update `build_effective_system_prompt()` — tool list and sandbox info**

This method currently iterates `self.tools` and reads `self.tools_config`. It needs to get this info from `ToolExecutor` instead.

The simplest approach: add a helper method to `ToolExecutor` that returns the info needed, OR keep `ToolsConfig` as a separate field in `AgentRuntime` just for prompt building. The cleanest: keep a `sandbox_root` and `allowed_directories` in `AgentRuntime` (they're config, not runtime state).

Actually — the simplest approach is: add a `sandbox_root: PathBuf` and `allowed_directories: Vec<PathBuf>` to `RuntimeConfig` or keep `ToolsConfig` as an additional parameter to `new()`. This avoids changing `ToolExecutor`'s API.

**Decision:** Keep `tools_config: ToolsConfig` as a field for system prompt building only. The `ToolExecutor` uses its own copy for execution. This duplicates the config reference but avoids coupling prompt building to `ToolExecutor`.

So the struct actually becomes:

```rust
pub struct AgentRuntime {
    provider_registry: ProviderRegistry,
    channel: Box<dyn Channel>,
    tool_executor: crate::tool_executor::ToolExecutor,
    consent_rx: Option<tokio::sync::mpsc::Receiver<freebird_security::consent::ConsentRequest>>,
    memory: Box<dyn Memory>,
    config: RuntimeConfig,
    tools_config: ToolsConfig,  // kept for system prompt building
    audit: Option<AuditLogger>,
    sessions: SessionManager,
}
```

And constructor takes `tools_config` too:

```rust
pub fn new(
    provider_registry: ProviderRegistry,
    channel: Box<dyn Channel>,
    tool_executor: crate::tool_executor::ToolExecutor,
    consent_rx: Option<tokio::sync::mpsc::Receiver<freebird_security::consent::ConsentRequest>>,
    memory: Box<dyn Memory>,
    config: RuntimeConfig,
    tools_config: ToolsConfig,
    audit: Option<AuditLogger>,
) -> Self
```

**Step 4: Update `execute_tool_calls()` — delegate to ToolExecutor**

Replace the body of the tool execution loop (lines 715-777 approximately). The key change is replacing:

```rust
let output = self.execute_tool(&tool_name, &input, session_id).await;

// Scan tool output for injection — BLOCK if detected
let scanned = ScannedToolOutput::from_raw(&output.content);
let (final_content, outcome) = if scanned.injection_detected() {
    // ... audit, block ...
} else {
    (scanned.into_content(), output.outcome)
};
```

With:

```rust
let output = self.tool_executor.execute(
    &tool_name, input.clone(), &grant, session_id, sender_id,
).await;

// ToolExecutor already handles injection scanning.
// Output with injection detected comes back as ToolOutcome::Error.
let final_content = output.content;
let outcome = output.outcome;
```

This means `execute_tool_calls()` needs a `grant: &CapabilityGrant` parameter. Create a default permissive grant for now:

```rust
use freebird_security::capability::CapabilityGrant;
use freebird_traits::tool::Capability;

/// Create a permissive grant for the given sandbox root.
/// TODO(#27): Replace with per-session capability grants.
fn default_grant(sandbox_root: &Path) -> CapabilityGrant {
    let all_caps = [
        Capability::FileRead,
        Capability::FileWrite,
        Capability::FileDelete,
        Capability::ShellExecute,
        Capability::ProcessSpawn,
        Capability::NetworkOutbound,
        Capability::NetworkListen,
        Capability::EnvRead,
    ]
    .into_iter()
    .collect();
    CapabilityGrant::new(all_caps, sandbox_root.to_path_buf(), None)
        .expect("default grant construction should not fail")
}
```

Place this as a private function at the bottom of `agent.rs`, or as a method on `AgentRuntime`.

**Step 5: Remove `execute_tool()` and `find_tool()` methods**

Delete the `execute_tool()` method (lines ~909-965) and `find_tool()` method (lines ~1186-1191). All tool execution now goes through `self.tool_executor.execute()`.

**Step 6: Remove unused imports**

After the refactor, these imports become unused:
- `ScannedToolOutput` (ToolExecutor handles this internally)
- `ToolContext` (ToolExecutor builds this internally)
- `Tool` (no longer stored directly)
- `CapabilityCheckResult` (ToolExecutor handles audit internally) — keep if still used by other methods
- `InjectionSource` — keep if still used by model injection audit

Check and remove only truly unused imports.

**Step 7: Run tests (expect failures — test fixtures not updated yet)**

```bash
cargo check -p freebird-runtime
```

Expected: compiles but tests won't build yet (constructor signature changed).

**Step 8: Commit (compiles, tests not yet updated)**

```bash
git add crates/freebird-runtime/src/agent.rs
git commit -m "refactor(runtime): replace direct tool execution with ToolExecutor

AgentRuntime now delegates all tool execution to ToolExecutor, bringing
capability checks, consent gates, timeout, injection scanning, and audit
logging into the security pipeline. Removes redundant execute_tool() and
find_tool() methods.

TODO(#27): default_grant() creates permissive capabilities — replace with
per-session grants."
```

---

### Task 4: Update test fixtures for new `AgentRuntime::new()` signature

**Files:**
- Modify: `crates/freebird-runtime/tests/agent_tests.rs`
- Modify: `crates/freebird-runtime/tests/agentic_loop_tests.rs`
- Modify: `crates/freebird-runtime/tests/stream_tests.rs`
- Modify: `crates/freebird-runtime/tests/helpers/mod.rs` (if it has shared construction logic)

**Step 1: Create a shared test helper to build ToolExecutor from tools**

In each test file's helper section (or in `helpers/mod.rs`), add:

```rust
use crate::helpers::default_tools_config; // if shared
use freebird_runtime::tool_executor::ToolExecutor;
use std::time::Duration;

fn make_tool_executor(tools: Vec<Box<dyn Tool>>) -> ToolExecutor {
    ToolExecutor::new(
        tools,
        Duration::from_secs(30),
        None, // no audit in most tests
        vec![],
        None, // no consent gate in most tests
    )
    .expect("test tool executor")
}
```

**Step 2: Update every `AgentRuntime::new()` call**

Pattern — replace:

```rust
AgentRuntime::new(
    registry,
    Box::new(channel),
    tools,                    // Vec<Box<dyn Tool>>
    memory,
    config,
    tools_config,
    audit,
)
```

With:

```rust
AgentRuntime::new(
    registry,
    Box::new(channel),
    make_tool_executor(tools), // ToolExecutor
    None,                      // consent_rx
    memory,
    config,
    tools_config,
    audit,
)
```

This is mechanical. There are approximately:
- 1 call in `agent_tests.rs` (via `make_runtime()`)
- 12 calls in `agentic_loop_tests.rs` (via `make_test_runtime()` + direct)
- 8 calls in `stream_tests.rs`

Update `make_runtime()` and `make_test_runtime()` helpers first, then fix any direct `AgentRuntime::new()` calls.

**Step 3: Run full test suite**

```bash
cargo test -p freebird-runtime
cargo clippy -p freebird-runtime --all-targets -- -D warnings
```

Expected: all ~169 runtime tests pass, no clippy warnings.

**Step 4: Run workspace tests**

```bash
cargo test --workspace
```

Expected: all ~751 tests pass.

**Step 5: Commit**

```bash
git add crates/freebird-runtime/tests/
git commit -m "test(runtime): update test fixtures for ToolExecutor integration"
```

---

## Phase 2: Consent Bridge

### Task 5: Add consent bridge to `run()` select! loop

**Files:**
- Modify: `crates/freebird-runtime/src/agent.rs`

**Step 1: Add the `recv_consent` helper function**

Add at the bottom of `agent.rs` (near `send_outbound`):

```rust
/// Receive from the consent channel, or pend forever if no gate is configured.
async fn recv_consent(
    rx: &mut Option<mpsc::Receiver<freebird_security::consent::ConsentRequest>>,
) -> Option<freebird_security::consent::ConsentRequest> {
    match rx.as_mut() {
        Some(rx) => rx.recv().await,
        None => std::future::pending().await,
    }
}
```

**Step 2: Add `forward_consent_request()` method to `AgentRuntime`**

```rust
/// Forward a consent request from the gate to the user's channel.
///
/// TODO: In future, broadcast to all active channels or route to a
/// preferred approval channel (e.g. Signal on phone). Currently sends
/// to the channel/sender that triggered the tool call.
async fn forward_consent_request(
    &self,
    req: freebird_security::consent::ConsentRequest,
    outbound: &mpsc::Sender<OutboundEvent>,
) {
    let event = OutboundEvent::ConsentRequest {
        request_id: req.id,
        tool_name: req.tool_name,
        description: req.description,
        risk_level: format!("{:?}", req.risk_level),
        action_summary: req.action_summary,
        expires_at: req.expires_at.to_rfc3339(),
        recipient_id: req.sender_id,
    };
    send_outbound(outbound, event).await;
}
```

**Step 3: Update `run()` — add consent arm to select! loop**

The `run()` method needs `&mut self` instead of `&self` because `recv_consent` takes `&mut Option<Receiver>`. Update signature:

```rust
pub async fn run(&mut self, cancel: CancellationToken) -> Result<(), RuntimeError> {
```

Update the loop:

```rust
loop {
    tokio::select! {
        () = cancel.cancelled() => {
            tracing::info!("shutdown signal received, stopping runtime");
            break;
        }
        Some(req) = recv_consent(&mut self.consent_rx) => {
            self.forward_consent_request(req, &outbound).await;
        }
        event = inbound.next() => {
            if let Some(event) = event {
                if matches!(
                    self.handle_event(event, &outbound).await,
                    LoopAction::Exit,
                ) {
                    break;
                }
            } else {
                tracing::info!("inbound stream closed");
                break;
            }
        }
    }
}
```

**Step 4: Update `handle_event()` — wire ConsentResponse**

Replace the stub:

```rust
InboundEvent::ConsentResponse {
    request_id,
    approved,
    reason,
    sender_id,
} => {
    let response = if approved {
        freebird_security::consent::ConsentResponse::Approved
    } else {
        freebird_security::consent::ConsentResponse::Denied { reason }
    };
    let delivered = self
        .tool_executor
        .consent_respond(&request_id, response)
        .await;
    if !delivered {
        tracing::warn!(
            %request_id,
            %sender_id,
            "consent response for unknown or expired request"
        );
    }
    LoopAction::Continue
}
```

**Step 5: Check compilation**

```bash
cargo check -p freebird-runtime
```

If `run()` changed from `&self` to `&mut self`, callers need updating. Check tests and daemon.

**Step 6: Update test callers of `run()` if needed**

Tests that call `runtime.run(cancel)` may need `mut runtime` bindings. This is mechanical.

**Step 7: Run tests**

```bash
cargo test -p freebird-runtime
cargo clippy -p freebird-runtime --all-targets -- -D warnings
```

Expected: all existing tests pass (consent bridge doesn't affect non-consent flows).

**Step 8: Commit**

```bash
git add crates/freebird-runtime/src/agent.rs
git commit -m "feat(runtime): add consent bridge — forward requests and route responses

Adds select! arm to run() that forwards ConsentRequests from the gate to
the user's channel. Wires InboundEvent::ConsentResponse to call
tool_executor.consent_respond(), completing the end-to-end consent flow.

TODO: In future, broadcast to all active channels or route to preferred
approval channel (e.g. Signal)."
```

---

### Task 6: Write consent bridge integration tests

**Files:**
- Create: `crates/freebird-runtime/tests/consent_bridge_tests.rs`

These tests wire the full stack: MockChannel → AgentRuntime → ToolExecutor → ConsentGate → user prompt → approval → tool execution.

**Step 1: Write test scaffold and helpers**

```rust
//! Integration tests for the consent bridge — end-to-end consent flow.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::indexing_slicing)]

use std::sync::Arc;
use std::time::Duration;

use freebird_runtime::tool_executor::ToolExecutor;
use freebird_runtime::AgentRuntime;
use freebird_security::consent::ConsentGate;
use freebird_traits::channel::{InboundEvent, OutboundEvent};
use freebird_traits::tool::RiskLevel;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

mod helpers;
use helpers::{default_config, default_tools_config, MockChannel};

// Re-use or create mock tool/provider helpers as needed from existing test patterns.
```

**Step 2: Write `test_consent_request_forwarded_to_channel`**

Send a message that triggers a high-risk tool. Verify the channel receives `OutboundEvent::ConsentRequest` with the correct `request_id`, `tool_name`, and `recipient_id`.

**Step 3: Write `test_consent_approved_executes_tool`**

Send message → provider returns ToolUse for high-risk tool → user receives ConsentRequest → send `InboundEvent::ConsentResponse { approved: true }` → tool executes → provider gets result → final response delivered.

**Step 4: Write `test_consent_denied_returns_error_to_provider`**

Same flow but deny. Verify provider sees a tool error result containing "Consent denied".

**Step 5: Write `test_consent_low_risk_no_prompt`**

Low-risk tool with consent gate configured → no ConsentRequest sent, tool executes directly.

**Step 6: Write `test_consent_no_gate_executes_freely`**

No consent gate → high-risk tool executes without prompting.

**Step 7: Write `test_consent_response_unknown_id_logged`**

Send an `InboundEvent::ConsentResponse` with a bogus `request_id`. Verify no crash, processing continues.

**Step 8: Run all tests**

```bash
cargo test -p freebird-runtime
cargo clippy -p freebird-runtime --all-targets -- -D warnings
cargo test --workspace
```

Expected: all tests pass.

**Step 9: Commit**

```bash
git add crates/freebird-runtime/tests/consent_bridge_tests.rs
git commit -m "test(runtime): add consent bridge integration tests

Tests the full consent flow: message → tool use → consent prompt →
approval/denial → tool execution/error → response delivery."
```

---

### Task 7: Final verification and cleanup

**Step 1: Run all quality gates**

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
cargo check --workspace --all-targets
```

All must pass.

**Step 2: Review the full diff**

```bash
git log --oneline master..HEAD
git diff master...HEAD --stat
```

Verify commit history tells a clean story.

**Step 3: Push and update PR**

```bash
git push -u origin feat/issue-29-consent-gates --force-with-lease
```

Update PR #64 description to include the consent bridge work, or create a new PR.

---

## Summary of Commits

| # | Message | Phase |
|---|---------|-------|
| 1 | `feat(security): add sender_id to ConsentRequest for consent routing` | 1 |
| 2 | `feat(runtime): propagate sender_id through ToolExecutor::execute()` | 1 |
| 3 | `refactor(runtime): replace direct tool execution with ToolExecutor` | 1 |
| 4 | `test(runtime): update test fixtures for ToolExecutor integration` | 1 |
| 5 | `feat(runtime): add consent bridge — forward requests and route responses` | 2 |
| 6 | `test(runtime): add consent bridge integration tests` | 2 |

## Risk Notes

- **`run()` becomes `&mut self`**: Required because `recv_consent` needs `&mut` on the receiver. All callers must be updated. This is a one-line change per call site (`let runtime` → `let mut runtime`).
- **Default permissive grant**: `default_grant()` uses `.expect()` which is normally prohibited. Add a `// SAFETY:` style comment explaining this is an infallible construction from valid constants. This is a known gap until #27 (per-session grants) is implemented.
- **`tools_config` duplication**: Both `ToolExecutor` and `AgentRuntime` hold `ToolsConfig` data. `ToolExecutor` uses it for execution, `AgentRuntime` uses it for system prompt building. This is acceptable — the config is immutable after construction.
