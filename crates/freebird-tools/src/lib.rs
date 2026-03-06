//! Built-in tool implementations.

#![deny(clippy::all, clippy::pedantic, clippy::nursery)]
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
#![deny(clippy::indexing_slicing)]
#![allow(clippy::module_name_repetitions)]

pub mod filesystem;
pub mod shell;

// network module intentionally omitted until implementation (tracked as #26).
// Adding an empty `pub mod network` would expose a public module with zero exports.
