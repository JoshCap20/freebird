# FreeBird Build Order for v0.1

> Extracted from CLAUDE.md Appendix A. See GitHub issues for tracking.

## Phase 1 — Skeleton (get `cargo build` working)

1. Create workspace with all crate stubs
2. Define traits in `freebird-traits` (Provider, Channel, Tool, Memory)
3. Define message types in `freebird-types`
4. Implement `SecurityError` in `freebird-security`

## Phase 2 — Minimum Viable Agent (talk to Claude from the terminal)

5. Implement `CliChannel` in `freebird-channels`
6. Implement `AnthropicProvider` in `freebird-providers`
7. Implement `FileMemory` in `freebird-memory`
8. Implement `AgentRuntime` with basic loop (no tools) in `freebird-runtime`
9. Wire everything in `freebird-daemon/main.rs`
10. Test: type a message, get a Claude response in the terminal

## Phase 3 — Security Layer (harden before adding tools)

11. Implement `Tainted` taint system
12. Implement `SafeFilePath`
13. Implement `CapabilityGrant`
14. Implement session key auth
15. Implement prompt injection scanning
16. Implement tamper-evident audit logging (hash-chained)

## Phase 4 — Channel Pairing & Consent (identity + human-in-the-loop)

17. Implement `PairingManager` with state machine
18. Integrate pairing check into router (reject unpaired channels)
19. Implement `ConsentGate` with risk classification
20. Wire consent gates into `ToolExecutor` for High/Critical tools
21. Test: pair a channel, verify unpaired channel is rejected

## Phase 5 — Tool System (agent can act)

22. Implement `ToolExecutor` with capability + consent checks
23. Implement `read_file` tool (with SafeFilePath)
24. Implement `shell` tool (with command allowlist)
25. Wire tool loop into agent runtime
26. Test: ask Claude to read a file, watch it use the tool

## Phase 6 — Defense in Depth (network + budgets + memory integrity)

27. Implement `EgressPolicy` with host allowlist + DNS rebinding prevention
28. Implement `TokenBudget` with per-session/request/turn limits
29. Implement HMAC-signed memory (sign on save, verify on load)
30. Implement context injection scan on loaded conversations
31. Test: exceed token budget, tamper with a conversation file — verify rejection

## Phase 7 — Polish

32. Streaming responses
33. `/commands` (new, status, model, help)
34. Configuration validation at startup
35. Daemon mode (systemd/launchd)
36. Signal channel implementation (with pairing flow)
37. Supply chain CI gates (`cargo deny`, `cargo audit`)
