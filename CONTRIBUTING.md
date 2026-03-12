# Contributing to FreeBird

Thank you for your interest in contributing to FreeBird! Whether you're reporting a bug, suggesting a feature, or submitting code, your help is valued.

## Getting Started

### Prerequisites

- **Rust toolchain**: Install via [rustup](https://rustup.rs/) (stable channel)
- **SQLCipher development headers**: Required for the `rusqlite` `bundled-sqlcipher` feature
  - macOS: `brew install sqlcipher`
  - Ubuntu/Debian: `apt install libsqlcipher-dev`
- **cargo-deny** (optional, for supply chain checks): `cargo install cargo-deny`

### Building

```bash
git clone https://github.com/JoshCap20/freebird.git
cd freebird
cargo build --workspace
```

### Running Tests

```bash
cargo test --workspace
```

### Running the Daemon

```bash
# Start the server
cargo run -- serve

# Connect a chat client
cargo run -- chat
```

## Reporting Bugs

1. Check [existing issues](https://github.com/JoshCap20/freebird/issues) to avoid duplicates
2. Open a new issue using the **Bug Report** template
3. Include: Rust version (`rustc --version`), OS, steps to reproduce, expected vs. actual behavior

## Suggesting Features

Open a new issue using the **Feature Request** template. Include the motivation, proposed solution, and alternatives you considered.

## Security Vulnerabilities

**Do NOT file security vulnerabilities as public issues.** See [SECURITY.md](SECURITY.md) for responsible disclosure instructions.

## Submitting Code

### Workflow

1. Fork the repository
2. Create a branch: `feat/issue-<N>-<short-description>` or `fix/issue-<N>-<short-description>`
3. Write tests first (TDD is strongly encouraged)
4. Implement your changes
5. Run quality gates (see below)
6. Open a pull request against `master`

### Commit Messages

We use conventional commits:

```
type(scope): imperative description

# Examples:
feat(runtime): add graceful shutdown timeout
fix(security): reject null bytes in SafeFilePath
test(providers): add streaming cancellation test
refactor(tools): extract shared validation logic
```

Types: `feat`, `fix`, `refactor`, `test`, `docs`, `chore`, `security`
Scope: crate name without `freebird-` prefix (e.g., `security`, `runtime`, `providers`)

### Quality Gates

All PRs must pass these before merge:

```bash
# Formatting
cargo fmt --all -- --check

# Linting (warnings are errors)
cargo clippy --workspace --all-targets -- -D warnings

# Tests
cargo test --workspace

# Supply chain audit (optional but appreciated)
cargo deny check
```

### Code Style

- `cargo fmt` on all code
- Strict clippy lints (see `lib.rs` files for deny directives)
- No `.unwrap()` or `.expect()` in production code — use `?` with `.context()`
- No `unsafe` in application code
- No `println!` — use `tracing::info!`, `tracing::debug!`, etc.
- No `std::sync::Mutex` in async code — use `tokio::sync::Mutex`
- Accept `&str` parameters, return `String` when ownership transfer is needed
- Use `thiserror` for library error types, `anyhow` only in the binary crate

For the full list, see [CLAUDE.md](CLAUDE.md) sections 22 (Code Style) and 23 (Anti-Patterns).

### Architecture

FreeBird is organized as a Cargo workspace with strict crate dependency rules. Before making changes, familiarize yourself with:

- **Crate topology** and dependency DAG (CLAUDE.md section 1)
- **Trait definitions** in `freebird-traits/` — all extensibility points
- **Security model** — taint tracking, capability system, approval gates (CLAUDE.md sections 6-8)

The `freebird-traits` crate has **zero** `freebird-*` dependencies. The `freebird-security` crate depends only on `freebird-traits` and `freebird-types`. These invariants are enforced by Cargo and must never be violated.

## License

By contributing to FreeBird, you agree that your contributions will be dual-licensed under MIT and Apache 2.0, as described in [LICENSE](LICENSE).
