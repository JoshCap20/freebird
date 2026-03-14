//! Freebird terminal UI — interactive chat client and session replay.
//!
//! This crate provides:
//! - [`chat`] — TCP client bridging user I/O to the daemon's JSON-line protocol
//! - [`replay`] — Human-readable and JSON formatters for past session traces
//! - [`ui`] — Interactive TUI with crossterm raw mode, multi-line editing, spinners, etc.

// These lints were implicitly suppressed when this code lived inside the daemon
// bin crate. Allow them here to avoid adding boilerplate to every pub function
// in what is effectively an internal crate.
#![allow(
    clippy::missing_errors_doc,
    clippy::must_use_candidate,
    clippy::new_without_default
)]

pub mod chat;
pub mod replay;
pub mod ui;
