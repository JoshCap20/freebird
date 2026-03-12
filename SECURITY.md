# Security Policy

FreeBird is a security-first project. We take vulnerability reports seriously and appreciate responsible disclosure.

## Reporting a Vulnerability

**Please do NOT open a public GitHub issue for security vulnerabilities.**

Instead, use [GitHub Private Vulnerability Reporting](https://github.com/JoshCap20/freebird/security/advisories/new) to submit your report confidentially.

### What to include

- Description of the vulnerability
- Steps to reproduce
- Affected component (e.g., `freebird-security`, `freebird-runtime`, tool executor)
- Impact assessment (what could an attacker achieve?)
- Suggested fix, if you have one

### What to expect

- **Acknowledgment** within 48 hours
- **Initial assessment** within 7 days
- **Fix timeline** depends on severity — critical issues are prioritized for immediate patching

## Scope

The following are considered security vulnerabilities:

- Prompt injection that bypasses taint tracking or injection scanning
- Path traversal that escapes the sandbox boundary
- Capability escalation (tool executing without required capability grant)
- Approval gate bypass (high-risk action executing without human consent)
- Secret leakage (API keys, tokens, or credentials exposed in logs, errors, or LLM context)
- Memory poisoning (tampered conversation history accepted without detection)
- Audit log tampering (hash chain integrity bypass)
- Network egress policy bypass (requests to non-allowlisted hosts)
- Authentication bypass (session key verification circumvented)
- Denial of service via token budget bypass

The following are **not** security vulnerabilities (file as regular issues):

- Injection patterns that are detected and handled per configured policy (`block`/`prompt`/`allow`)
- Performance issues or resource usage within configured budget limits
- Feature requests for additional security controls

## Supported Versions

| Version | Supported |
|---------|-----------|
| `master` (HEAD) | Yes |
| Released tags | Yes (latest only) |

## Coordinated Disclosure

We follow coordinated disclosure. Once a fix is available, we will:

1. Merge the fix
2. Publish a GitHub Security Advisory
3. Credit the reporter (unless they prefer anonymity)

We ask that reporters allow up to 90 days for a fix before public disclosure.

## Security Architecture

For details on FreeBird's security model, see the [Security Checklist](CLAUDE.md#24-security-checklist) in CLAUDE.md. Key defenses include:

- Compile-time taint tracking on all external input
- Capability-based tool authorization with expiring grants
- Human approval gates for high-risk operations (OWASP ASI09)
- HMAC-signed conversation history (OWASP ASI06)
- Host-allowlist network egress policy (OWASP ASI01)
- Token budget enforcement (OWASP ASI08)
- Hash-chained audit logging
- SQLCipher-encrypted persistence
