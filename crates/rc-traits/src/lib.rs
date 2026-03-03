//! Core trait definitions.
//!
//! This crate defines the public traits that every subsystem implements:
//! [`Provider`], [`Channel`], [`Tool`], and [`Memory`]. It has zero `rc-*`
//! dependencies — all implementation crates depend on this, never the reverse.

pub mod channel;
pub mod memory;
pub mod provider;
pub mod tool;
