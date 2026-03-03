# RClaw

Rust-based AI agent

## Security Concerns

The main reason for creating my custom Rust OpenClaw is security. 

Studies show an exceeding amount of community tools containing prompt injections. While this is just one way of injection, any output from a tool can still contain malicious input.

- No supported public skill-sharing, simply ask the agent to build what you need or provide it with one you trust.
- Explicit approval required for using tainted tool data in a tool innvocation. We will use taint flow detection.
- Explicit approval required for operating outside of defined directoy.
- Audit logs for tracability, both of user-authorized actions and the tools used by the agent.
