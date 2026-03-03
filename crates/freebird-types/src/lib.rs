//! Shared domain types for Freebird.
//!
//! Re-exports core types from `freebird-traits` and adds config structs,
//! ID generation utilities, and types not referenced in trait signatures.

pub mod config;
pub mod id;

pub use freebird_traits;
