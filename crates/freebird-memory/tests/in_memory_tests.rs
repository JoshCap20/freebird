#![allow(clippy::unwrap_used, clippy::indexing_slicing)]

mod common;

use chrono::{TimeZone, Utc};
use freebird_memory::in_memory::InMemoryMemory;
use freebird_traits::id::{ModelId, ProviderId, SessionId};
use freebird_traits::memory::{Conversation, Memory, MemoryError, Turn};
use freebird_traits::provider::{ContentBlock, Message, Role};

use crate::common::make_conversation;

// ─── Core Operations ────────────────────────────────────────────────────

#[tokio::test]
async fn test_new_is_empty() {
    let mem = InMemoryMemory::new();
    let sessions = mem.list_sessions(100).await.unwrap();
    assert!(sessions.is_empty());
}

#[tokio::test]
async fn test_save_and_load_roundtrip() {
    let mem = InMemoryMemory::new();
    let conv = make_conversation("s1", "hello", Utc::now());
    mem.save(&conv).await.unwrap();

    let loaded = mem
        .load(&SessionId::from_string("s1"))
        .await
        .unwrap()
        .unwrap();
    assert_eq!(loaded, conv);
}

#[tokio::test]
async fn test_load_nonexistent() {
    let mem = InMemoryMemory::new();
    let result = mem
        .load(&SessionId::from_string("nonexistent"))
        .await
        .unwrap();
    assert!(result.is_none());
}

#[tokio::test]
async fn test_save_upsert_overwrites() {
    let mem = InMemoryMemory::new();
    let now = Utc::now();
    let conv_a = make_conversation("s1", "first version", now);
    let conv_b = make_conversation("s1", "second version", now);

    mem.save(&conv_a).await.unwrap();
    mem.save(&conv_b).await.unwrap();

    let loaded = mem
        .load(&SessionId::from_string("s1"))
        .await
        .unwrap()
        .unwrap();
    assert_eq!(loaded, conv_b);
}

#[tokio::test]
async fn test_delete_existing() {
    let mem = InMemoryMemory::new();
    let conv = make_conversation("s1", "hello", Utc::now());
    mem.save(&conv).await.unwrap();
    mem.delete(&SessionId::from_string("s1")).await.unwrap();

    let result = mem.load(&SessionId::from_string("s1")).await.unwrap();
    assert!(result.is_none());
}

#[tokio::test]
async fn test_delete_nonexistent_returns_not_found() {
    let mem = InMemoryMemory::new();
    let result = mem.delete(&SessionId::from_string("gone")).await;
    assert!(matches!(
        result,
        Err(MemoryError::NotFound { session_id }) if session_id.as_str() == "gone"
    ));
}

#[tokio::test]
async fn test_delete_then_delete_again() {
    let mem = InMemoryMemory::new();
    let conv = make_conversation("s1", "hello", Utc::now());
    mem.save(&conv).await.unwrap();
    mem.delete(&SessionId::from_string("s1")).await.unwrap();

    let result = mem.delete(&SessionId::from_string("s1")).await;
    assert!(matches!(result, Err(MemoryError::NotFound { .. })));
}

// ─── Listing ────────────────────────────────────────────────────────────

#[tokio::test]
async fn test_list_sessions_ordering() {
    let mem = InMemoryMemory::new();
    let t1 = Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap();
    let t2 = Utc.with_ymd_and_hms(2024, 6, 1, 0, 0, 0).unwrap();
    let t3 = Utc.with_ymd_and_hms(2025, 1, 1, 0, 0, 0).unwrap();

    mem.save(&make_conversation("old", "first", t1))
        .await
        .unwrap();
    mem.save(&make_conversation("mid", "second", t2))
        .await
        .unwrap();
    mem.save(&make_conversation("new", "third", t3))
        .await
        .unwrap();

    let sessions = mem.list_sessions(10).await.unwrap();
    assert_eq!(sessions.len(), 3);
    assert_eq!(sessions[0].session_id.as_str(), "new");
    assert_eq!(sessions[1].session_id.as_str(), "mid");
    assert_eq!(sessions[2].session_id.as_str(), "old");
}

#[tokio::test]
async fn test_list_sessions_limit() {
    let mem = InMemoryMemory::new();
    for i in 0..5 {
        let t = Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, i).unwrap();
        mem.save(&make_conversation(&format!("s{i}"), "text", t))
            .await
            .unwrap();
    }

    let sessions = mem.list_sessions(2).await.unwrap();
    assert_eq!(sessions.len(), 2);
}

#[tokio::test]
async fn test_list_sessions_limit_zero() {
    let mem = InMemoryMemory::new();
    mem.save(&make_conversation("s1", "text", Utc::now()))
        .await
        .unwrap();

    let sessions = mem.list_sessions(0).await.unwrap();
    assert!(sessions.is_empty());
}

#[tokio::test]
async fn test_list_sessions_limit_exceeds_count() {
    let mem = InMemoryMemory::new();
    mem.save(&make_conversation("s1", "text", Utc::now()))
        .await
        .unwrap();
    mem.save(&make_conversation("s2", "text", Utc::now()))
        .await
        .unwrap();

    let sessions = mem.list_sessions(100).await.unwrap();
    assert_eq!(sessions.len(), 2);
}

// ─── Search ─────────────────────────────────────────────────────────────

#[tokio::test]
async fn test_search_finds_user_message() {
    let mem = InMemoryMemory::new();
    mem.save(&make_conversation("s1", "Hello World", Utc::now()))
        .await
        .unwrap();

    let results = mem.search("hello", 10).await.unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].session_id.as_str(), "s1");
}

#[tokio::test]
async fn test_search_finds_assistant_response() {
    let mem = InMemoryMemory::new();
    let now = Utc::now();
    let conv = Conversation {
        session_id: SessionId::from_string("s1"),
        system_prompt: None,
        turns: vec![Turn {
            user_message: Message {
                role: Role::User,
                content: vec![ContentBlock::Text {
                    text: "question".to_string(),
                }],
                timestamp: now,
            },
            assistant_messages: vec![Message {
                role: Role::Assistant,
                content: vec![ContentBlock::Text {
                    text: "specific phrase here".to_string(),
                }],
                timestamp: now,
            }],
            tool_invocations: vec![],
            started_at: now,
            completed_at: Some(now),
        }],
        created_at: now,
        updated_at: now,
        model_id: ModelId::from("test-model"),
        provider_id: ProviderId::from("test-provider"),
    };
    mem.save(&conv).await.unwrap();

    let results = mem.search("specific phrase", 10).await.unwrap();
    assert_eq!(results.len(), 1);
}

#[tokio::test]
async fn test_search_case_insensitive() {
    let mem = InMemoryMemory::new();
    mem.save(&make_conversation("s1", "Hello World", Utc::now()))
        .await
        .unwrap();

    let results = mem.search("HELLO", 10).await.unwrap();
    assert_eq!(results.len(), 1);
}

#[tokio::test]
async fn test_search_no_match() {
    let mem = InMemoryMemory::new();
    mem.save(&make_conversation("s1", "Hello World", Utc::now()))
        .await
        .unwrap();

    let results = mem.search("nonexistent", 10).await.unwrap();
    assert!(results.is_empty());
}

#[tokio::test]
async fn test_search_empty_query() {
    let mem = InMemoryMemory::new();
    mem.save(&make_conversation("s1", "Hello World", Utc::now()))
        .await
        .unwrap();

    let results = mem.search("", 10).await.unwrap();
    assert!(results.is_empty());
}

#[tokio::test]
async fn test_search_limit() {
    let mem = InMemoryMemory::new();
    for i in 0..5 {
        let t = Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, i).unwrap();
        mem.save(&make_conversation(
            &format!("s{i}"),
            "matching text here",
            t,
        ))
        .await
        .unwrap();
    }

    let results = mem.search("matching", 2).await.unwrap();
    assert_eq!(results.len(), 2);
}

// ─── Summary ────────────────────────────────────────────────────────────

#[tokio::test]
async fn test_summary_preview_first_100_chars() {
    let mem = InMemoryMemory::new();
    let long_text: String = "a".repeat(200);
    mem.save(&make_conversation("s1", &long_text, Utc::now()))
        .await
        .unwrap();

    let sessions = mem.list_sessions(10).await.unwrap();
    assert_eq!(sessions[0].preview.len(), 100);
}

#[tokio::test]
async fn test_summary_preview_short_message() {
    let mem = InMemoryMemory::new();
    mem.save(&make_conversation("s1", "short", Utc::now()))
        .await
        .unwrap();

    let sessions = mem.list_sessions(10).await.unwrap();
    assert_eq!(sessions[0].preview, "short");
}

#[tokio::test]
async fn test_summary_preview_empty_turns() {
    let mem = InMemoryMemory::new();
    let now = Utc::now();
    let conv = Conversation {
        session_id: SessionId::from_string("s1"),
        system_prompt: None,
        turns: vec![],
        created_at: now,
        updated_at: now,
        model_id: ModelId::from("test-model"),
        provider_id: ProviderId::from("test-provider"),
    };
    mem.save(&conv).await.unwrap();

    let sessions = mem.list_sessions(10).await.unwrap();
    assert!(sessions[0].preview.is_empty());
}

#[tokio::test]
async fn test_summary_turn_count() {
    let mem = InMemoryMemory::new();
    let now = Utc::now();
    let make_turn = |text: &str| Turn {
        user_message: Message {
            role: Role::User,
            content: vec![ContentBlock::Text {
                text: text.to_string(),
            }],
            timestamp: now,
        },
        assistant_messages: vec![],
        tool_invocations: vec![],
        started_at: now,
        completed_at: Some(now),
    };

    let conv = Conversation {
        session_id: SessionId::from_string("s1"),
        system_prompt: None,
        turns: vec![make_turn("a"), make_turn("b"), make_turn("c")],
        created_at: now,
        updated_at: now,
        model_id: ModelId::from("test-model"),
        provider_id: ProviderId::from("test-provider"),
    };
    mem.save(&conv).await.unwrap();

    let sessions = mem.list_sessions(10).await.unwrap();
    assert_eq!(sessions[0].turn_count, 3);
}

#[tokio::test]
async fn test_summary_preview_multibyte_unicode() {
    let mem = InMemoryMemory::new();
    // 150 CJK characters (each 3 bytes in UTF-8), preview truncates to 100
    let cjk = "漢".repeat(150);
    mem.save(&make_conversation("s1", &cjk, Utc::now()))
        .await
        .unwrap();

    let sessions = mem.list_sessions(10).await.unwrap();
    assert_eq!(sessions[0].preview.chars().count(), 100);
    assert!(sessions[0].preview.chars().all(|c| c == '漢'));
}

#[tokio::test]
async fn test_search_limit_zero() {
    let mem = InMemoryMemory::new();
    mem.save(&make_conversation("s1", "findme", Utc::now()))
        .await
        .unwrap();

    let results = mem.search("findme", 0).await.unwrap();
    assert!(results.is_empty());
}

// ─── Concurrency ────────────────────────────────────────────────────────

#[tokio::test]
async fn test_concurrent_save_and_load() {
    let mem = std::sync::Arc::new(InMemoryMemory::new());
    let now = Utc::now();

    let mut handles = Vec::with_capacity(10);
    for i in 0..10 {
        let mem = std::sync::Arc::clone(&mem);
        handles.push(tokio::spawn(async move {
            let conv = make_conversation(&format!("concurrent-{i}"), &format!("text-{i}"), now);
            mem.save(&conv).await.unwrap();
        }));
    }

    for handle in handles {
        handle.await.unwrap();
    }

    for i in 0..10 {
        let loaded = mem
            .load(&SessionId::from_string(format!("concurrent-{i}")))
            .await
            .unwrap();
        assert!(loaded.is_some(), "conversation concurrent-{i} should exist");
    }

    let sessions = mem.list_sessions(100).await.unwrap();
    assert_eq!(sessions.len(), 10);
}
