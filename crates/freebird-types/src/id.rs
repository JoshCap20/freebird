//! ID generation utilities.
//!
//! The newtype IDs are defined in `freebird_traits::id`. This module provides
//! uuid-based generation that is not available in the traits crate.

use freebird_traits::id::*;

pub fn new_session_id() -> SessionId {
    SessionId::from_string(uuid::Uuid::new_v4().to_string())
}

pub fn new_invocation_id() -> InvocationId {
    InvocationId::from_string(uuid::Uuid::new_v4().to_string())
}

pub fn new_channel_id() -> ChannelId {
    ChannelId::from_string(uuid::Uuid::new_v4().to_string())
}

pub fn new_provider_id() -> ProviderId {
    ProviderId::from_string(uuid::Uuid::new_v4().to_string())
}

pub fn new_tool_id() -> ToolId {
    ToolId::from_string(uuid::Uuid::new_v4().to_string())
}
