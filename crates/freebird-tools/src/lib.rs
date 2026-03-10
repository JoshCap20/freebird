//! Built-in tool implementations.

#![deny(clippy::all, clippy::pedantic, clippy::nursery)]
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
#![deny(clippy::indexing_slicing)]
#![allow(clippy::module_name_repetitions)]

pub mod cargo_verify;
mod common;
pub mod edit;
pub mod filesystem;
pub mod glob_find;
pub mod grep;
pub mod knowledge;
pub mod network;
pub mod repo_map;
pub mod shell;
pub mod viewer;
