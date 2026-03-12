//! Event replay and HMAC chain computation for event-sourced persistence.
//!
//! Provides functions to replay a sequence of [`ConversationEvent`]s into a
//! [`Conversation`] struct, and to compute/verify HMAC chains on stored events.

#![allow(clippy::significant_drop_tightening)]

use chrono::{DateTime, Utc};
use freebird_traits::event::ConversationEvent;
use freebird_traits::id::{ModelId, ProviderId, SessionId};
use freebird_traits::memory::{Conversation, MemoryError, Turn};
use ring::hmac;

/// A stored event row with chain integrity metadata.
#[derive(Debug, Clone)]
pub struct StoredEvent {
    pub(crate) session_id: String,
    pub(crate) sequence: i64,
    pub(crate) event: ConversationEvent,
    pub(crate) timestamp: DateTime<Utc>,
    pub(crate) previous_hmac: String,
    pub(crate) hmac: String,
}

/// Replay a sequence of events into a [`Conversation`].
///
/// Events must be ordered by `sequence` (ascending) and belong to the same
/// session. Returns `None` if the event list is empty.
///
/// # Errors
///
/// Returns `MemoryError::Serialization` if events reference invalid indices,
/// or `MemoryError::IntegrityViolation` if a `SessionCreated` event is missing.
pub fn replay_events_to_conversation(
    session_id: &SessionId,
    events: &[StoredEvent],
) -> Result<Option<Conversation>, MemoryError> {
    if events.is_empty() {
        return Ok(None);
    }

    let mut conversation = Conversation {
        session_id: session_id.clone(),
        system_prompt: None,
        turns: Vec::new(),
        created_at: events.first().map_or_else(Utc::now, |e| e.timestamp),
        updated_at: events.last().map_or_else(Utc::now, |e| e.timestamp),
        model_id: ModelId::from_string("unknown"),
        provider_id: ProviderId::from_string("unknown"),
    };

    for stored in events {
        apply_event(&mut conversation, &stored.event, stored.timestamp)?;
    }

    Ok(Some(conversation))
}

/// Apply a single event to a conversation being reconstructed.
fn apply_event(
    conv: &mut Conversation,
    event: &ConversationEvent,
    timestamp: DateTime<Utc>,
) -> Result<(), MemoryError> {
    match event {
        ConversationEvent::SessionCreated {
            system_prompt,
            model_id,
            provider_id,
        } => {
            conv.system_prompt.clone_from(system_prompt);
            conv.model_id = ModelId::from_string(model_id);
            conv.provider_id = ProviderId::from_string(provider_id);
            conv.created_at = timestamp;
        }
        ConversationEvent::SessionMetadataUpdated {
            system_prompt,
            model_id,
            provider_id,
        } => {
            conv.system_prompt.clone_from(system_prompt);
            conv.model_id = ModelId::from_string(model_id);
            conv.provider_id = ProviderId::from_string(provider_id);
        }
        ConversationEvent::TurnStarted { user_message, .. } => {
            conv.turns.push(Turn {
                user_message: user_message.clone(),
                assistant_messages: Vec::new(),
                tool_invocations: Vec::new(),
                started_at: timestamp,
                completed_at: None,
            });
        }
        ConversationEvent::AssistantMessage {
            turn_index,
            message,
            ..
        } => {
            let turn = conv.turns.get_mut(*turn_index).ok_or_else(|| {
                MemoryError::Serialization(format!(
                    "AssistantMessage references invalid turn_index {turn_index}"
                ))
            })?;
            turn.assistant_messages.push(message.clone());
        }
        ConversationEvent::ToolInvoked {
            turn_index,
            invocation,
            ..
        } => {
            let turn = conv.turns.get_mut(*turn_index).ok_or_else(|| {
                MemoryError::Serialization(format!(
                    "ToolInvoked references invalid turn_index {turn_index}"
                ))
            })?;
            turn.tool_invocations.push(invocation.clone());
        }
        ConversationEvent::TurnCompleted {
            turn_index,
            completed_at,
        } => {
            let turn = conv.turns.get_mut(*turn_index).ok_or_else(|| {
                MemoryError::Serialization(format!(
                    "TurnCompleted references invalid turn_index {turn_index}"
                ))
            })?;
            turn.completed_at = Some(*completed_at);
        }
    }

    conv.updated_at = timestamp;
    Ok(())
}

/// Compute the HMAC-SHA256 for a conversation event entry.
///
/// The HMAC covers: `session_id|sequence|event_json|timestamp|previous_hmac`.
///
/// # Errors
///
/// Returns `MemoryError::Serialization` if the event cannot be serialized.
pub fn compute_event_hmac(
    session_id: &str,
    sequence: i64,
    event: &ConversationEvent,
    timestamp: &str,
    previous_hmac: &str,
    key: &hmac::Key,
) -> Result<String, MemoryError> {
    let event_json = serde_json::to_string(event)
        .map_err(|e| MemoryError::Serialization(format!("event serialization: {e}")))?;
    let data = format!("{session_id}|{sequence}|{event_json}|{timestamp}|{previous_hmac}");
    let tag = hmac::sign(key, data.as_bytes());
    Ok(hex::encode(tag.as_ref()))
}

/// Compute the HMAC-SHA256 for an audit event entry.
///
/// The HMAC covers: `sequence|session_id|event_type|event_json|timestamp|previous_hmac`.
#[must_use]
pub fn compute_audit_hmac(
    sequence: i64,
    session_id: Option<&str>,
    event_type: &str,
    event_json: &str,
    timestamp: &str,
    previous_hmac: &str,
    key: &hmac::Key,
) -> String {
    let sid = session_id.unwrap_or("");
    let data = format!("{sequence}|{sid}|{event_type}|{event_json}|{timestamp}|{previous_hmac}");
    let tag = hmac::sign(key, data.as_bytes());
    hex::encode(tag.as_ref())
}

/// Verify the HMAC chain integrity of a sequence of stored events.
///
/// # Errors
///
/// Returns `MemoryError::IntegrityViolation` if the chain is broken.
pub fn verify_event_chain(events: &[StoredEvent], key: &hmac::Key) -> Result<(), MemoryError> {
    let mut expected_previous = String::new();

    for stored in events {
        // Verify previous_hmac linkage
        if stored.previous_hmac != expected_previous {
            return Err(MemoryError::IntegrityViolation {
                reason: format!(
                    "event chain broken at sequence {}: expected previous_hmac `{}`, got `{}`",
                    stored.sequence, expected_previous, stored.previous_hmac
                ),
            });
        }

        // Recompute HMAC and verify
        let computed = compute_event_hmac(
            &stored.session_id,
            stored.sequence,
            &stored.event,
            &stored.timestamp.to_rfc3339(),
            &stored.previous_hmac,
            key,
        )?;

        if computed != stored.hmac {
            return Err(MemoryError::IntegrityViolation {
                reason: format!(
                    "HMAC mismatch at sequence {}: event may have been tampered with",
                    stored.sequence
                ),
            });
        }

        expected_previous.clone_from(&stored.hmac);
    }

    Ok(())
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic
)]
mod tests {
    use chrono::Utc;
    use freebird_traits::event::ConversationEvent;
    use freebird_traits::id::SessionId;
    use freebird_traits::memory::ToolInvocation;
    use freebird_traits::provider::{ContentBlock, Message, Role};
    use freebird_traits::tool::ToolOutcome;

    use super::*;

    fn make_message(role: Role, text: &str) -> Message {
        Message {
            role,
            content: vec![ContentBlock::Text { text: text.into() }],
            timestamp: Utc::now(),
        }
    }

    fn make_stored_event(session_id: &str, sequence: i64, event: ConversationEvent) -> StoredEvent {
        StoredEvent {
            session_id: session_id.into(),
            sequence,
            event,
            timestamp: Utc::now(),
            previous_hmac: String::new(),
            hmac: String::new(),
        }
    }

    fn test_key() -> hmac::Key {
        hmac::Key::new(hmac::HMAC_SHA256, b"test-key")
    }

    #[test]
    fn test_replay_empty_returns_none() {
        let sid = SessionId::from_string("s1");
        let result = replay_events_to_conversation(&sid, &[]).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_replay_session_created() {
        let sid = SessionId::from_string("s1");
        let events = vec![make_stored_event(
            "s1",
            0,
            ConversationEvent::SessionCreated {
                system_prompt: Some("hello".into()),
                model_id: "claude".into(),
                provider_id: "anthropic".into(),
            },
        )];

        let conv = replay_events_to_conversation(&sid, &events)
            .unwrap()
            .unwrap();
        assert_eq!(conv.system_prompt.as_deref(), Some("hello"));
        assert_eq!(conv.model_id.as_str(), "claude");
        assert_eq!(conv.provider_id.as_str(), "anthropic");
        assert!(conv.turns.is_empty());
    }

    #[test]
    fn test_replay_full_turn() {
        let sid = SessionId::from_string("s1");
        let now = Utc::now();
        let events = vec![
            make_stored_event(
                "s1",
                0,
                ConversationEvent::SessionCreated {
                    system_prompt: None,
                    model_id: "m1".into(),
                    provider_id: "p1".into(),
                },
            ),
            make_stored_event(
                "s1",
                1,
                ConversationEvent::TurnStarted {
                    turn_index: 0,
                    user_message: make_message(Role::User, "hello"),
                },
            ),
            make_stored_event(
                "s1",
                2,
                ConversationEvent::AssistantMessage {
                    turn_index: 0,
                    message_index: 0,
                    message: make_message(Role::Assistant, "hi there"),
                },
            ),
            make_stored_event(
                "s1",
                3,
                ConversationEvent::TurnCompleted {
                    turn_index: 0,
                    completed_at: now,
                },
            ),
        ];

        let conv = replay_events_to_conversation(&sid, &events)
            .unwrap()
            .unwrap();
        assert_eq!(conv.turns.len(), 1);
        assert_eq!(conv.turns[0].assistant_messages.len(), 1);
        assert!(conv.turns[0].completed_at.is_some());
    }

    #[test]
    fn test_replay_with_tool_invocation() {
        let sid = SessionId::from_string("s1");
        let invocation = ToolInvocation {
            tool_use_id: "tu1".into(),
            tool_name: "read_file".into(),
            input: serde_json::json!({"path": "/tmp/test"}),
            output: Some("file contents".into()),
            outcome: ToolOutcome::Success,
            duration_ms: Some(42),
        };

        let events = vec![
            make_stored_event(
                "s1",
                0,
                ConversationEvent::SessionCreated {
                    system_prompt: None,
                    model_id: "m1".into(),
                    provider_id: "p1".into(),
                },
            ),
            make_stored_event(
                "s1",
                1,
                ConversationEvent::TurnStarted {
                    turn_index: 0,
                    user_message: make_message(Role::User, "read file"),
                },
            ),
            make_stored_event(
                "s1",
                2,
                ConversationEvent::ToolInvoked {
                    turn_index: 0,
                    invocation_index: 0,
                    invocation,
                },
            ),
        ];

        let conv = replay_events_to_conversation(&sid, &events)
            .unwrap()
            .unwrap();
        assert_eq!(conv.turns[0].tool_invocations.len(), 1);
        assert_eq!(conv.turns[0].tool_invocations[0].tool_name, "read_file");
    }

    #[test]
    fn test_replay_invalid_turn_index_errors() {
        let sid = SessionId::from_string("s1");
        let events = vec![
            make_stored_event(
                "s1",
                0,
                ConversationEvent::SessionCreated {
                    system_prompt: None,
                    model_id: "m1".into(),
                    provider_id: "p1".into(),
                },
            ),
            make_stored_event(
                "s1",
                1,
                ConversationEvent::AssistantMessage {
                    turn_index: 99, // invalid
                    message_index: 0,
                    message: make_message(Role::Assistant, "oops"),
                },
            ),
        ];

        let result = replay_events_to_conversation(&sid, &events);
        assert!(result.is_err());
    }

    #[test]
    fn test_hmac_computation_deterministic() {
        let key = test_key();
        let event = ConversationEvent::SessionCreated {
            system_prompt: None,
            model_id: "m1".into(),
            provider_id: "p1".into(),
        };

        let hmac1 = compute_event_hmac("s1", 0, &event, "2025-01-01T00:00:00Z", "", &key).unwrap();
        let hmac2 = compute_event_hmac("s1", 0, &event, "2025-01-01T00:00:00Z", "", &key).unwrap();
        assert_eq!(hmac1, hmac2);
    }

    #[test]
    fn test_hmac_changes_with_different_input() {
        let key = test_key();
        let event = ConversationEvent::SessionCreated {
            system_prompt: None,
            model_id: "m1".into(),
            provider_id: "p1".into(),
        };

        let hmac1 = compute_event_hmac("s1", 0, &event, "2025-01-01T00:00:00Z", "", &key).unwrap();
        let hmac2 = compute_event_hmac("s2", 0, &event, "2025-01-01T00:00:00Z", "", &key).unwrap();
        assert_ne!(hmac1, hmac2);
    }

    #[test]
    fn test_verify_event_chain_valid() {
        let key = test_key();
        let events_data = vec![
            ConversationEvent::SessionCreated {
                system_prompt: None,
                model_id: "m1".into(),
                provider_id: "p1".into(),
            },
            ConversationEvent::TurnStarted {
                turn_index: 0,
                user_message: make_message(Role::User, "hello"),
            },
        ];

        let mut stored_events = Vec::new();
        let mut prev_hmac = String::new();

        #[allow(clippy::cast_possible_wrap)]
        for (i, event) in events_data.into_iter().enumerate() {
            let ts = "2025-01-01T00:00:00+00:00";
            let seq = i as i64;
            let hmac_hex = compute_event_hmac("s1", seq, &event, ts, &prev_hmac, &key).unwrap();

            stored_events.push(StoredEvent {
                session_id: "s1".into(),
                sequence: seq,
                event,
                timestamp: chrono::DateTime::parse_from_rfc3339(ts).unwrap().to_utc(),
                previous_hmac: prev_hmac,
                hmac: hmac_hex.clone(),
            });

            prev_hmac = hmac_hex;
        }

        assert!(verify_event_chain(&stored_events, &key).is_ok());
    }

    #[test]
    fn test_verify_event_chain_tampered() {
        let key = test_key();
        let event = ConversationEvent::SessionCreated {
            system_prompt: None,
            model_id: "m1".into(),
            provider_id: "p1".into(),
        };

        let ts = "2025-01-01T00:00:00+00:00";

        let stored = vec![StoredEvent {
            session_id: "s1".into(),
            sequence: 0,
            event,
            timestamp: chrono::DateTime::parse_from_rfc3339(ts).unwrap().to_utc(),
            previous_hmac: String::new(),
            hmac: "tampered_hmac".into(), // wrong!
        }];

        let result = verify_event_chain(&stored, &key);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("HMAC mismatch"));
    }

    #[test]
    fn test_verify_event_chain_broken_linkage() {
        let key = test_key();
        let event = ConversationEvent::SessionCreated {
            system_prompt: None,
            model_id: "m1".into(),
            provider_id: "p1".into(),
        };

        let ts = "2025-01-01T00:00:00+00:00";
        let hmac_hex = compute_event_hmac("s1", 0, &event, ts, "", &key).unwrap();

        let stored = vec![
            StoredEvent {
                session_id: "s1".into(),
                sequence: 0,
                event: event.clone(),
                timestamp: chrono::DateTime::parse_from_rfc3339(ts).unwrap().to_utc(),
                previous_hmac: String::new(),
                hmac: hmac_hex,
            },
            StoredEvent {
                session_id: "s1".into(),
                sequence: 1,
                event,
                timestamp: chrono::DateTime::parse_from_rfc3339(ts).unwrap().to_utc(),
                previous_hmac: "wrong_linkage".into(), // broken chain
                hmac: "doesn't matter".into(),
            },
        ];

        let result = verify_event_chain(&stored, &key);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("chain broken"));
    }

    #[test]
    fn test_replay_multi_turn_conversation() {
        let sid = SessionId::from_string("s1");
        let now = Utc::now();

        let events = vec![
            make_stored_event(
                "s1",
                0,
                ConversationEvent::SessionCreated {
                    system_prompt: Some("system".into()),
                    model_id: "claude".into(),
                    provider_id: "anthropic".into(),
                },
            ),
            // Turn 0
            make_stored_event(
                "s1",
                1,
                ConversationEvent::TurnStarted {
                    turn_index: 0,
                    user_message: make_message(Role::User, "first"),
                },
            ),
            make_stored_event(
                "s1",
                2,
                ConversationEvent::AssistantMessage {
                    turn_index: 0,
                    message_index: 0,
                    message: make_message(Role::Assistant, "first reply"),
                },
            ),
            make_stored_event(
                "s1",
                3,
                ConversationEvent::TurnCompleted {
                    turn_index: 0,
                    completed_at: now,
                },
            ),
            // Turn 1
            make_stored_event(
                "s1",
                4,
                ConversationEvent::TurnStarted {
                    turn_index: 1,
                    user_message: make_message(Role::User, "second"),
                },
            ),
            make_stored_event(
                "s1",
                5,
                ConversationEvent::AssistantMessage {
                    turn_index: 1,
                    message_index: 0,
                    message: make_message(Role::Assistant, "second reply"),
                },
            ),
            make_stored_event(
                "s1",
                6,
                ConversationEvent::TurnCompleted {
                    turn_index: 1,
                    completed_at: now,
                },
            ),
        ];

        let conv = replay_events_to_conversation(&sid, &events)
            .unwrap()
            .unwrap();
        assert_eq!(conv.turns.len(), 2);
        assert_eq!(conv.system_prompt.as_deref(), Some("system"));
    }

    #[test]
    fn test_replay_incomplete_turn() {
        let sid = SessionId::from_string("s1");

        let events = vec![
            make_stored_event(
                "s1",
                0,
                ConversationEvent::SessionCreated {
                    system_prompt: None,
                    model_id: "m1".into(),
                    provider_id: "p1".into(),
                },
            ),
            make_stored_event(
                "s1",
                1,
                ConversationEvent::TurnStarted {
                    turn_index: 0,
                    user_message: make_message(Role::User, "hello"),
                },
            ),
            // No TurnCompleted — simulates crash
        ];

        let conv = replay_events_to_conversation(&sid, &events)
            .unwrap()
            .unwrap();
        assert_eq!(conv.turns.len(), 1);
        assert!(conv.turns[0].completed_at.is_none());
    }

    #[test]
    fn test_audit_hmac_deterministic() {
        let key = test_key();
        let h1 = compute_audit_hmac(0, Some("s1"), "test", "{}", "ts", "", &key);
        let h2 = compute_audit_hmac(0, Some("s1"), "test", "{}", "ts", "", &key);
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_audit_hmac_none_session() {
        let key = test_key();
        let h1 = compute_audit_hmac(0, None, "test", "{}", "ts", "", &key);
        let h2 = compute_audit_hmac(0, Some("s1"), "test", "{}", "ts", "", &key);
        assert_ne!(h1, h2);
    }
}
