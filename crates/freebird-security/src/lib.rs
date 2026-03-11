//! Security primitives for Freebird.

#![deny(clippy::all, clippy::pedantic, clippy::nursery)]
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
#![deny(clippy::indexing_slicing)]
#![allow(clippy::module_name_repetitions)]

pub mod audit;
pub mod auth;
pub mod budget;
pub mod capability;
pub mod consent;
pub mod db_key;
pub mod egress;
pub mod error;
pub mod injection;
pub mod safe_types;
pub mod secret_guard;
pub mod sensitive;
pub mod taint;
