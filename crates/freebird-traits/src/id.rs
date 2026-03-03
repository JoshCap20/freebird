//! Newtype IDs to prevent mixing different identifier domains.
//!
//! Generation (uuid-based) lives in `freebird-types::id` to keep
//! the `uuid` crate out of this foundation crate.

use serde::{Deserialize, Serialize};

macro_rules! define_id {
    ($(#[$meta:meta])* $name:ident) => {
        $(#[$meta])*
        #[derive(Debug, Clone, Hash, Eq, PartialEq, Serialize, Deserialize)]
        pub struct $name(String);

        impl $name {
            pub fn from_string(s: impl Into<String>) -> Self {
                Self(s.into())
            }

            pub fn as_str(&self) -> &str {
                &self.0
            }
        }

        impl std::fmt::Display for $name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str(&self.0)
            }
        }

        impl From<String> for $name {
            fn from(s: String) -> Self {
                Self(s)
            }
        }

        impl From<&str> for $name {
            fn from(s: &str) -> Self {
                Self(s.to_owned())
            }
        }
    };
}

define_id!(
    /// Identifies a conversation session.
    SessionId
);
define_id!(
    /// Identifies a single tool invocation within a turn.
    InvocationId
);
define_id!(
    /// Identifies a channel instance.
    ChannelId
);
define_id!(
    /// Identifies a provider instance.
    ProviderId
);
define_id!(
    /// Identifies a registered tool.
    ToolId
);
