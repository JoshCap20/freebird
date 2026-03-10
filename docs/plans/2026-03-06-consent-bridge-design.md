# Consent Bridge Design — Wire ConsentGate End-to-End

**Date**: 2026-03-06
**Status**: Approved
**Scope**: Phase 1 (ToolExecutor integration) + Phase 2 (consent bridge)

---

## Problem

`ConsentGate` and `ToolExecutor` are fully implemented and wired together, but `AgentRuntime` doesn't use `ToolExecutor`. It stores `Vec<Box<dyn Tool>>` and calls `tool.execute()` directly, bypassing the entire security pipeline (capability checks, consent gates, timeout enforcement, injection scanning, audit logging).

Additionally, the two ends of the consent flow are disconnected:
- **Outbound**: `ConsentGate::check()` sends `ConsentRequest` to an mpsc receiver that nobody reads
- **Inbound**: `InboundEvent::ConsentResponse` is logged and ignored in `handle_event()`

The user never sees consent prompts. High-risk tools either time out after 60s or execute without approval.

## Architecture

```
User (CLI/Signal)            AgentRuntime                    ToolExecutor            ConsentGate
     │                            │                              │                      │
     │  message                   │                              │                      │
     ├───────────────────────────>│  taint → sanitize → provider │                      │
     │                            │  provider returns ToolUse    │                      │
     │                            │                              │                      │
     │                            │  execute(name, input, grant, │                      │
     │                            │    session_id, sender_id)    │                      │
     │                            ├─────────────────────────────>│                      │
     │                            │                              │ 1. capability check  │
     │                            │                              │ 2. check_consent()   │
     │                            │                              ├─────────────────────>│
     │                            │                              │                      │ ConsentRequest
     │                            │    select! arm fires         │                      │ on mpsc
     │   OutboundEvent::         <├──────────────────────────────┼──────────────────────┤
     │     ConsentRequest         │  forward_consent_request()   │                      │ (blocks on
     │<───────────────────────────┤                              │                      │  oneshot)
     │                            │                              │                      │
     │  /approve                  │                              │                      │
     ├───────────────────────────>│  InboundEvent::              │                      │
     │                            │    ConsentResponse           │                      │
     │                            │  consent_respond(id,Approved)│                      │
     │                            ├─────────────────────────────>│  gate.respond()      │
     │                            │                              ├─────────────────────>│
     │                            │                              │  check() → Ok(())  <─┤ oneshot
     │                            │                              │ 3. audit: granted    │
     │                            │                              │ 4. execute tool      │
     │                            │                              │ 5. injection scan    │
     │                            │          ToolOutput <────────┤                      │
     │                            │                              │                      │
     │  response                  │                              │                      │
     │<───────────────────────────┤                              │                      │
```

## Phase 1: ToolExecutor Integration into AgentRuntime

### Struct changes

```rust
// BEFORE
pub struct AgentRuntime {
    tools: Vec<Box<dyn Tool>>,
    tools_config: ToolsConfig,
    // ...
}

// AFTER
pub struct AgentRuntime {
    tool_executor: ToolExecutor,
    consent_rx: Option<mpsc::Receiver<ConsentRequest>>,
    // ...
}
```

### Constructor changes

The daemon (composition root) constructs all pieces:

```rust
// Daemon builds ConsentGate from SecurityConfig
let (consent_gate, consent_rx) = ConsentGate::new(
    security_config.require_consent_above,
    Duration::from_secs(security_config.consent_timeout_secs),
    security_config.max_pending_consent_requests,
);

// Daemon builds ToolExecutor with ConsentGate
let tool_executor = ToolExecutor::new(
    tools,
    Duration::from_secs(tools_config.default_timeout_secs),
    audit.clone(),
    tools_config.allowed_directories,
    Some(consent_gate),
)?;

// Daemon passes both to AgentRuntime
let runtime = AgentRuntime::new(
    provider_registry, channel, tool_executor, Some(consent_rx),
    memory, config, audit,
);
```

### execute_tool() refactor

All tool execution delegates to ToolExecutor. Removes ~40 lines of redundant security logic from agent.rs (manual timeout, injection scan, tool lookup).

```rust
// BEFORE — manual timeout + injection scan
let output = match tokio::time::timeout(timeout, tool.execute(input, &ctx)).await { ... };
let scanned = ScannedToolOutput::from_raw(&output.content);

// AFTER — single delegation
let output = self.tool_executor.execute(
    tool_name, input, &grant, &session_id, &sender_id
).await;
```

### Capability grants

ToolExecutor requires a `CapabilityGrant`. Per-session grants are issue #27 (not yet implemented). For now, create a permissive default grant at session creation that allows all capabilities within the sandbox root. This matches current behavior — no regression.

### What gets removed from agent.rs

- Manual `tokio::time::timeout` on tool calls
- Manual `ScannedToolOutput::from_raw()` after tool execution
- Tool lookup by name
- Direct `tool.execute()` calls
- `tools_config` field

### What stays in agent.rs

- `ScannedModelResponse::from_raw()` on LLM output (not a tool concern)
- `SafeMessage::from_tainted()` on user input (not a tool concern)
- Agentic loop structure (provider calls, stop reason handling)
- Conversation persistence

## Phase 2: Consent Bridge

### Sender routing

`sender_id: String` is added to `ConsentRequest` and propagated through `ConsentGate::check()` and `ToolExecutor::execute()`. The request carries its own routing info — no runtime-side reverse lookups.

Propagation chain:
```
handle_message(sender_id)
  → run_agentic_loop(sender_id)
    → tool_executor.execute(..., sender_id)
      → check_consent(..., sender_id)
        → gate.check(tool_info, summary, sender_id)
          → ConsentRequest { sender_id, ... }
```

### Bridge in run() select! loop

No spawned tasks. A third arm in the existing select! loop:

```rust
loop {
    tokio::select! {
        () = cancel.cancelled() => break,

        Some(req) = recv_consent(&mut self.consent_rx) => {
            // TODO: In future, broadcast to all active channels or route
            // to a preferred approval channel (e.g. Signal on phone).
            // Currently sends to the channel that triggered the tool call.
            self.forward_consent_request(req, &outbound).await;
        }

        event = inbound.next() => { /* existing dispatch */ }
    }
}

// Helper for Option<Receiver>
async fn recv_consent(
    rx: &mut Option<mpsc::Receiver<ConsentRequest>>,
) -> Option<ConsentRequest> {
    match rx.as_mut() {
        Some(rx) => rx.recv().await,
        None => std::future::pending().await,
    }
}
```

### forward_consent_request()

```rust
async fn forward_consent_request(
    &self,
    req: ConsentRequest,
    outbound: &mpsc::Sender<OutboundEvent>,
) {
    // TODO: Route to preferred approval channel (Signal, all channels, etc.)
    // For now, sends back to the channel/sender that triggered the tool call.
    let event = OutboundEvent::ConsentRequest {
        request_id: req.id,
        tool_name: req.tool_name,
        description: req.description,
        risk_level: req.risk_level.to_string(),
        action_summary: req.action_summary,
        expires_at: req.expires_at.to_rfc3339(),
        recipient_id: req.sender_id,
    };
    let _ = outbound.send(event).await;
}
```

### InboundEvent::ConsentResponse handler

Replace the "ignoring" stub:

```rust
InboundEvent::ConsentResponse { request_id, approved, reason, sender_id } => {
    let response = if approved {
        ConsentResponse::Approved
    } else {
        ConsentResponse::Denied { reason }
    };
    let delivered = self.tool_executor
        .consent_respond(&request_id, response).await;
    if !delivered {
        tracing::warn!(%request_id, %sender_id,
            "consent response for unknown/expired request");
    }
    LoopAction::Continue
}
```

## Files Modified

### Phase 1

| File | Change |
|------|--------|
| `crates/freebird-runtime/src/agent.rs` | Replace `tools` + `tools_config` with `tool_executor` + `consent_rx`. Refactor `execute_tool()`. Remove redundant security code. Add default grant. |
| `crates/freebird-runtime/tests/agentic_loop_tests.rs` | Update `make_runtime()` fixture to construct ToolExecutor. |
| `crates/freebird-runtime/tests/agent_tests.rs` | Same fixture update. |
| `crates/freebird-runtime/tests/stream_tests.rs` | Same fixture update. |

### Phase 2

| File | Change |
|------|--------|
| `crates/freebird-security/src/consent.rs` | Add `sender_id: String` to `ConsentRequest`. Add `sender_id: &str` param to `check()`. Update tests. |
| `crates/freebird-runtime/src/tool_executor.rs` | Add `sender_id: &str` to `execute()` and `check_consent()`. Update tests. |
| `crates/freebird-runtime/src/agent.rs` | Add consent bridge to `select!` loop. Handle `ConsentResponse`. Pass `sender_id` to `execute()`. |
| `crates/freebird-runtime/tests/consent_bridge_tests.rs` | New. Integration tests for full consent flow. |

## Test Strategy

### Phase 1 (ToolExecutor integration)

| Test | Validates |
|------|-----------|
| Existing 23 agentic loop tests | Tool execution works through ToolExecutor |
| Existing 9 agent event tests | Commands, connect/disconnect unchanged |
| Existing 17 streaming tests | Streaming + tool use unaffected |
| `test_tool_executor_used_not_bypassed` | Denied capability blocks execution (proves ToolExecutor is in path) |
| `test_default_grant_allows_tool_execution` | Permissive grant doesn't regress behavior |

### Phase 2 (consent bridge)

| Test | Validates |
|------|-----------|
| `test_consent_request_forwarded_to_channel` | High-risk tool → user sees ConsentRequest with correct fields |
| `test_consent_approved_executes_tool` | Approve → tool runs → response delivered |
| `test_consent_denied_returns_error_to_provider` | Deny → provider sees tool error |
| `test_consent_timeout_returns_error` | No response → expires → error |
| `test_consent_low_risk_no_prompt` | Low-risk tool → no prompt, executes directly |
| `test_consent_no_gate_executes_freely` | No gate → all tools execute without prompts |
| `test_consent_response_unknown_id_logged` | Stale response → warn, no crash |
| `test_consent_multiple_concurrent_tools` | Two high-risk tools → two prompts, independent |
| `test_consent_bridge_shutdown_clean` | Cancel mid-consent → clean exit |

### Not tested (future work)

- Multi-channel broadcast for consent routing
- Signal-specific consent rendering
- Different approvers per risk level

## Scope Estimate

- ~6 files modified
- ~53 existing tests updated mechanically (add sender_id param)
- ~10 new tests
- ~60 lines new production code
- ~40 lines removed from agent.rs
