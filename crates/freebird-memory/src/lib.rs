//! Memory backend implementations.

#![deny(clippy::all, clippy::pedantic, clippy::nursery)]
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
#![deny(clippy::indexing_slicing)]
#![allow(clippy::module_name_repetitions)]

pub mod event;
pub mod file;
mod helpers;
pub mod in_memory;
pub mod sqlite;
pub mod sqlite_audit;
pub mod sqlite_event;
pub mod sqlite_knowledge;
pub mod sqlite_memory;
