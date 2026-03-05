# Daemon/Client Separation Design

> **Date**: 2026-03-04
> **Status**: Approved
> **Branch**: TBD (off `feat/issue-19-daemon-composition`)

## Problem

Today, `freebird` is a single process: the daemon boots the `AgentRuntime` with a `CliChannel` that reads stdin and writes stdout. Tracing logs interleave with chat output. The agent can't run as a background service. There's no way to connect multiple clients.

## Decision

Split into **daemon** (`freebird serve`) and **client** (`freebird chat`) communicating over TCP (`127.0.0.1`) with JSON-line framing. Single binary with `clap` subcommands.

### IPC Choice: TCP on 127.0.0.1 + JSON-Lines

**Evaluated alternatives:**
- **Unix domain socket**: More secure by default (browsers can't connect), but not cross-platform and requires separate mechanism for remote access. Good defense-in-depth but auth solves the same problem.
- **gRPC over UDS**: Adds `tonic`/`prost`/`protoc` deps for a protocol that is internal and easily changed later. A2A and MCP are separate channel implementations with their own protocol specs; they won't share proto definitions with the internal daemon-CLI IPC.

**Why TCP + JSON-lines:**
- Cross-platform (Windows support if ever needed)
- Easier to debug (telnet, netcat)
- Natural path to remote access (`freebird chat --host remote:port`)
- Container/Docker friendly (shared network, not filesystem)
- Session key auth (CLAUDE.md §13) prevents browser-based attacks that hit OpenClaw
- Bind to `127.0.0.1` only by default — not `0.0.0.0`
- Bidirectional async (both sides read/write independently) — supports future orchestrator/sub-agent push notifications
- Protocol is extensible (add serde variants without breaking existing clients)

## Architecture

```
BEFORE:
  freebird (single process)
    ├── AgentRuntime
    ├── CliChannel (reads stdin, writes stdout)
    ├── ProviderRegistry
    └── Memory

AFTER:
  freebird serve (daemon process, no terminal UI)
    ├── AgentRuntime
    ├── TcpChannel (accepts TCP connections on 127.0.0.1:PORT, implements Channel trait)
    ├── ProviderRegistry
    ├── Memory
    └── Logs to stderr / file (no stdout UI)

  freebird chat (thin client process)
    ├── Connects to 127.0.0.1:PORT
    ├── Reads stdin → sends JSON to socket
    └── Reads JSON from socket → writes to stdout
```

Key insight: **`TcpChannel` replaces `CliChannel` as the daemon's channel.** It implements the same `Channel` trait. `AgentRuntime` doesn't change — it still calls `channel.start()` and processes the `InboundEvent`/`OutboundEvent` stream.

## Command Structure

```
freebird serve          # Start daemon, listen on TCP. Foreground by default.
freebird chat           # Connect to running daemon, interactive chat.
freebird status         # Check if daemon is running (probes TCP port).
freebird stop           # Send graceful shutdown to daemon via TCP.
```

Single binary with `clap` subcommands. No separate binaries.

## Wire Protocol

JSON-lines over TCP (`127.0.0.1:7531`). Each message is a single JSON object followed by `\n`.

### Client → Daemon

```rust
/// Messages sent by `freebird chat` to the daemon.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ClientMessage {
    /// User typed a message.
    Message { text: String },
    /// User typed a /command.
    Command { name: String, args: Vec<String> },
    /// Client is disconnecting gracefully.
    Disconnect,
}
```

### Daemon → Client

```rust
/// Messages sent by the daemon to a connected client.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ServerMessage {
    /// Complete response text.
    Message { text: String },
    /// Streaming chunk (partial response).
    StreamChunk { text: String },
    /// Stream finished.
    StreamEnd,
    /// Error message.
    Error { text: String },
    /// Command response (e.g., /status output).
    CommandResponse { text: String },
}
```

These map 1:1 to the existing `InboundEvent`/`OutboundEvent` enums. The `sender_id`/`recipient_id` fields are removed from the wire protocol because the TCP connection itself identifies the client. The `TcpChannel` assigns an internal `sender_id` per connection (e.g., `"tcp-0"`, `"tcp-1"`) and routes `OutboundEvent`s to the correct connection.

### Example Exchange

```
→  {"type":"message","text":"what is 2+2?"}\n
←  {"type":"stream_chunk","text":"2 + 2"}\n
←  {"type":"stream_chunk","text":" = 4"}\n
←  {"type":"stream_end"}\n

→  {"type":"command","name":"new","args":[]}\n
←  {"type":"command_response","text":"New session started."}\n

→  {"type":"disconnect"}\n
```

## TcpChannel Implementation

New file: `freebird-channels/src/tcp.rs`

Implements `Channel` trait:
- `start()`: Binds `TcpListener` on `127.0.0.1:PORT`, spawns accept loop. Each connection gets a reader task (JSON lines → `InboundEvent`s on the shared inbound stream) and a writer task (matching `OutboundEvent`s → JSON lines back to that connection).
- `stop()`: Cancels the accept loop, drops all connections.
- `info()`: Returns `ChannelInfo` with `id: "tcp"`, `features: {Streaming}`, `auth: None` (auth deferred to session key issue).

Connection lifecycle:
1. Client connects → `InboundEvent::Connected { sender_id: "tcp-{n}" }`
2. Client sends JSON lines → parsed into `InboundEvent::Message` / `InboundEvent::Command`
3. Runtime sends `OutboundEvent` with matching `recipient_id` → routed to correct connection
4. Client disconnects → `InboundEvent::Disconnected { sender_id: "tcp-{n}" }`

Multiple concurrent clients are supported — each connection is independent.

## Chat Client Implementation

The `freebird chat` subcommand handler (in `freebird-daemon/src/chat.rs` or similar):

1. Connect to `127.0.0.1:PORT` (fail with helpful error if daemon not running)
2. Spawn two tasks:
   - **Reader**: reads JSON lines from socket, matches on `ServerMessage`, prints to stdout (stream chunks without newline, messages with newline, errors to stderr)
   - **Writer**: reads stdin line by line, wraps in `ClientMessage`, writes JSON to socket
3. On stdin EOF or `/quit`: send `Disconnect`, close socket, exit.

The prompt (`you> `) and all terminal UX lives here, not in the daemon.

## Config Changes

```toml
# New section in config/default.toml
[daemon]
host = "127.0.0.1"
port = 7531
pid_file = "~/.freebird/freebird.pid"
```

Added to `AppConfig`:

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DaemonConfig {
    pub host: String,
    pub port: u16,
    pub pid_file: PathBuf,
}
```

The `[[channels]]` config section is retained for future channels (Signal, WebSocket). The TCP listener is always active when `freebird serve` runs.

## What Changes Where

| Crate | Changes |
|---|---|
| **freebird-daemon** | Add `clap`. Split `main.rs` into subcommands: `serve`, `chat`, `status`, `stop`. `serve` uses `TcpChannel`. `chat` is thin TCP client. |
| **freebird-channels** | Add `tcp.rs` — `TcpChannel` implementing `Channel` trait. |
| **freebird-types** | Add `ClientMessage`, `ServerMessage` protocol enums. Add `DaemonConfig` to `AppConfig`. |
| **freebird-runtime** | **No changes.** |
| **freebird-traits** | **No changes.** |
| **freebird-providers** | **No changes.** |
| **freebird-memory** | **No changes.** |
| **freebird-security** | **No changes.** |

## New Dependency

- `clap` (with `derive` feature) — for subcommand parsing. Well-established, standard Rust CLI crate.

## Log Separation (Solves Original Problem)

- **`freebird serve`**: Logs go to stderr / log file. No terminal UI. Can run in background, via systemd/launchd, nohup, etc.
- **`freebird chat`**: Clean interactive experience. Only prompt + agent responses. Zero log noise.

## Future Extensions (Not In Scope)

- **Orchestrator/sub-agent pattern**: The protocol is bidirectional async, so adding `TaskDelegated`/`TaskProgress`/`TaskCompleted` server messages later is backward-compatible.
- **A2A / MCP**: Separate `Channel` implementations with their own listeners, not related to the internal TCP protocol.
- **Remote access**: Change `host` to `0.0.0.0` in config to accept remote connections. Requires session key auth first.
- **Authentication**: Session key auth (CLAUDE.md §13) should be wired in before exposing beyond localhost.

## Testing Strategy

- **TcpChannel unit tests**: Connect test `TcpStream` pairs to verify JSON-line framing, multi-client routing, disconnect handling.
- **Integration test**: Start `TcpChannel`, connect a test client, send a message, verify `InboundEvent` appears on the channel's stream. Send an `OutboundEvent`, verify the test client receives the correct `ServerMessage`.
- **End-to-end**: `freebird serve` in background + `freebird chat` with piped input, verify response comes back.
