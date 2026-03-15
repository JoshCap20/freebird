//! Context-specific safe output types.
//!
//! Each safe type has a private inner value, a `from_tainted()` factory with
//! context-specific validation, and a context-specific accessor. The type name
//! IS the context — `SafeFilePath` cannot be used where `SafeShellArg` is
//! expected, and vice versa.
//!
//! All inner fields are private. Construction is only possible through
//! `from_tainted()` factories which live in this crate. This ensures every
//! instance has been validated for its specific context.

mod message;
mod output;
mod path;
mod shell;
mod url;

pub use message::{SafeMessage, ScannedModelResponse, ValidationResult};
pub use output::{Redacted, ScannedToolOutput};
pub use path::{SafeFileContent, SafeFilePath};
pub use shell::{SafeBashCommand, SafeShellArg};
pub use url::SafeUrl;
