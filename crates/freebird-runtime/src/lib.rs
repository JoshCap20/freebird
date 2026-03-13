//! Agent runtime loop, session management, and orchestration.

#![deny(clippy::all, clippy::pedantic, clippy::nursery)]
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
#![deny(clippy::indexing_slicing)]
#![allow(clippy::module_name_repetitions)]

pub mod agent;
mod approval_bridge;
mod command_handler;
pub mod history;
pub mod observation;
pub mod registry;
pub mod router;
pub mod session;
pub mod shutdown;
pub mod stream;
pub mod summarize;
mod task_drain;
pub mod tool_executor;
pub mod tool_registry;
