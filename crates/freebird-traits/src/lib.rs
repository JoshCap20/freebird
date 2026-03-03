//! Core trait definitions and associated types for Freebird.
//!
//! This crate is the root of the internal dependency DAG. It has **zero**
//! `freebird-*` dependencies. External deps are the minimum needed for
//! async trait signatures and serialization.

pub mod channel;
pub mod id;
pub mod memory;
pub mod provider;
pub mod tool;
