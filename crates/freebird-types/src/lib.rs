//! Shared domain types for Freebird.
//!
//! Config structs, ID generation utilities, and types not referenced
//! in trait signatures. Consumers import `freebird-traits` directly.

#![deny(clippy::all, clippy::pedantic, clippy::nursery)]
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
#![deny(clippy::indexing_slicing)]
#![allow(clippy::module_name_repetitions)]

pub mod config;
pub mod id;
