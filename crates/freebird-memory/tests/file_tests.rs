#![allow(clippy::unwrap_used, clippy::indexing_slicing)]

mod common;

use chrono::{TimeZone, Utc};
use freebird_memory::file::FileMemory;
use freebird_traits::id::SessionId;
use freebird_traits::memory::{Conversation, Memory, MemoryError, Turn};
use freebird_traits::provider::{ContentBlock, Message, Role};
use proptest::prelude::*;

use crate::common::make_conversation;

// ─── Core Operations ────────────────────────────────────────────────────

#[test]
fn test_new_creates_dir() {
    let tmp = tempfile::tempdir().unwrap();
    let sub = tmp.path().join("nested").join("dir");
    assert!(!sub.exists());

    let _mem = FileMemory::new(sub.clone()).unwrap();
    assert!(sub.exists());
}

#[test]
fn test_new_canonicalizes_path() {
    let tmp = tempfile::tempdir().unwrap();
    // Create a subdirectory "a" so the path tmp/a/../b resolves correctly
    std::fs::create_dir(tmp.path().join("a")).unwrap();
    let sub = tmp.path().join("a").join("..").join("b");
    let mem = FileMemory::new(sub).unwrap();

    // FileMemory doesn't expose base_dir, but we can verify by checking
    // that saving a file ends up in the canonicalized path
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        let conv = make_conversation("test1", "hello", Utc::now());
        mem.save(&conv).await.unwrap();

        // The file should be in the canonicalized path (tmp/b/), not tmp/a/../b/
        let canonical = tmp.path().join("b");
        assert!(canonical.join("test1.json").exists());
    });
}

#[test]
fn test_new_cleans_orphaned_tmp() {
    let tmp = tempfile::tempdir().unwrap();
    // Place an orphaned .tmp file
    std::fs::write(tmp.path().join("stale.json.tmp"), "garbage").unwrap();
    assert!(tmp.path().join("stale.json.tmp").exists());

    let _mem = FileMemory::new(tmp.path().to_path_buf()).unwrap();

    // The .tmp file should have been cleaned up
    assert!(!tmp.path().join("stale.json.tmp").exists());
}

#[tokio::test]
async fn test_save_and_load_roundtrip() {
    let tmp = tempfile::tempdir().unwrap();
    let mem = FileMemory::new(tmp.path().to_path_buf()).unwrap();
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
    let tmp = tempfile::tempdir().unwrap();
    let mem = FileMemory::new(tmp.path().to_path_buf()).unwrap();

    let result = mem
        .load(&SessionId::from_string("nonexistent"))
        .await
        .unwrap();
    assert!(result.is_none());
}

#[tokio::test]
async fn test_load_corrupt_file() {
    let tmp = tempfile::tempdir().unwrap();
    let mem = FileMemory::new(tmp.path().to_path_buf()).unwrap();

    // Write invalid JSON directly to the file
    std::fs::write(tmp.path().join("bad.json"), "not valid json {{{").unwrap();

    let result = mem.load(&SessionId::from_string("bad")).await;
    assert!(matches!(result, Err(MemoryError::Serialization(_))));
}

#[tokio::test]
async fn test_save_overwrites() {
    let tmp = tempfile::tempdir().unwrap();
    let mem = FileMemory::new(tmp.path().to_path_buf()).unwrap();
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
async fn test_atomic_write_no_tmp_remains() {
    let tmp = tempfile::tempdir().unwrap();
    let mem = FileMemory::new(tmp.path().to_path_buf()).unwrap();

    mem.save(&make_conversation("s1", "hello", Utc::now()))
        .await
        .unwrap();

    // No .tmp files should remain after save
    let has_tmp = std::fs::read_dir(tmp.path())
        .unwrap()
        .flatten()
        .any(|e| e.path().extension().is_some_and(|ext| ext == "tmp"));
    assert!(!has_tmp, "no .tmp files should remain after save");
}

#[tokio::test]
async fn test_delete_existing() {
    let tmp = tempfile::tempdir().unwrap();
    let mem = FileMemory::new(tmp.path().to_path_buf()).unwrap();
    let conv = make_conversation("s1", "hello", Utc::now());

    mem.save(&conv).await.unwrap();
    mem.delete(&SessionId::from_string("s1")).await.unwrap();

    let result = mem.load(&SessionId::from_string("s1")).await.unwrap();
    assert!(result.is_none());
}

#[tokio::test]
async fn test_delete_nonexistent_returns_not_found() {
    let tmp = tempfile::tempdir().unwrap();
    let mem = FileMemory::new(tmp.path().to_path_buf()).unwrap();

    let result = mem.delete(&SessionId::from_string("gone")).await;
    assert!(matches!(
        result,
        Err(MemoryError::NotFound { session_id }) if session_id.as_str() == "gone"
    ));
}

#[tokio::test]
async fn test_delete_then_delete_again() {
    let tmp = tempfile::tempdir().unwrap();
    let mem = FileMemory::new(tmp.path().to_path_buf()).unwrap();
    let conv = make_conversation("s1", "hello", Utc::now());

    mem.save(&conv).await.unwrap();
    mem.delete(&SessionId::from_string("s1")).await.unwrap();

    let result = mem.delete(&SessionId::from_string("s1")).await;
    assert!(matches!(result, Err(MemoryError::NotFound { .. })));
}

// ─── Path Safety ────────────────────────────────────────────────────────

#[tokio::test]
async fn test_path_traversal_rejected() {
    let tmp = tempfile::tempdir().unwrap();
    let mem = FileMemory::new(tmp.path().to_path_buf()).unwrap();

    let result = mem
        .load(&SessionId::from_string("../../../etc/passwd"))
        .await;
    assert!(matches!(result, Err(MemoryError::Io(_))));
}

#[tokio::test]
async fn test_null_byte_rejected() {
    let tmp = tempfile::tempdir().unwrap();
    let mem = FileMemory::new(tmp.path().to_path_buf()).unwrap();

    let result = mem.load(&SessionId::from_string("evil\0id")).await;
    assert!(matches!(result, Err(MemoryError::Io(_))));
}

#[tokio::test]
async fn test_dot_in_session_id_rejected() {
    let tmp = tempfile::tempdir().unwrap();
    let mem = FileMemory::new(tmp.path().to_path_buf()).unwrap();

    let result = mem.load(&SessionId::from_string("evil.json")).await;
    assert!(matches!(result, Err(MemoryError::Io(_))));
}

#[tokio::test]
async fn test_empty_session_id_rejected() {
    let tmp = tempfile::tempdir().unwrap();
    let mem = FileMemory::new(tmp.path().to_path_buf()).unwrap();

    let result = mem.load(&SessionId::from_string("")).await;
    assert!(matches!(result, Err(MemoryError::Io(_))));
}

#[tokio::test]
async fn test_valid_uuid_accepted() {
    let tmp = tempfile::tempdir().unwrap();
    let mem = FileMemory::new(tmp.path().to_path_buf()).unwrap();

    // UUID v4 format: alphanumeric + hyphens — should pass validation
    let result = mem
        .load(&SessionId::from_string(
            "550e8400-e29b-41d4-a716-446655440000",
        ))
        .await
        .unwrap();
    assert!(result.is_none());
}

#[tokio::test]
async fn test_underscore_accepted() {
    let tmp = tempfile::tempdir().unwrap();
    let mem = FileMemory::new(tmp.path().to_path_buf()).unwrap();

    let result = mem
        .load(&SessionId::from_string("test_session_1"))
        .await
        .unwrap();
    assert!(result.is_none());
}

// ─── Listing ────────────────────────────────────────────────────────────

#[tokio::test]
async fn test_list_sessions_ordering() {
    let tmp = tempfile::tempdir().unwrap();
    let mem = FileMemory::new(tmp.path().to_path_buf()).unwrap();
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
    let tmp = tempfile::tempdir().unwrap();
    let mem = FileMemory::new(tmp.path().to_path_buf()).unwrap();

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
    let tmp = tempfile::tempdir().unwrap();
    let mem = FileMemory::new(tmp.path().to_path_buf()).unwrap();
    mem.save(&make_conversation("s1", "text", Utc::now()))
        .await
        .unwrap();

    let sessions = mem.list_sessions(0).await.unwrap();
    assert!(sessions.is_empty());
}

#[tokio::test]
async fn test_list_sessions_limit_exceeds_count() {
    let tmp = tempfile::tempdir().unwrap();
    let mem = FileMemory::new(tmp.path().to_path_buf()).unwrap();
    mem.save(&make_conversation("s1", "text", Utc::now()))
        .await
        .unwrap();
    mem.save(&make_conversation("s2", "text", Utc::now()))
        .await
        .unwrap();

    let sessions = mem.list_sessions(100).await.unwrap();
    assert_eq!(sessions.len(), 2);
}

#[tokio::test]
async fn test_list_sessions_skips_corrupt() {
    let tmp = tempfile::tempdir().unwrap();
    let mem = FileMemory::new(tmp.path().to_path_buf()).unwrap();

    // Save a valid conversation
    mem.save(&make_conversation("good", "hello", Utc::now()))
        .await
        .unwrap();

    // Place a corrupt file
    std::fs::write(tmp.path().join("bad.json"), "not json").unwrap();

    let sessions = mem.list_sessions(10).await.unwrap();
    assert_eq!(sessions.len(), 1);
    assert_eq!(sessions[0].session_id.as_str(), "good");
}

#[tokio::test]
async fn test_list_sessions_skips_tmp_files() {
    let tmp = tempfile::tempdir().unwrap();
    let mem = FileMemory::new(tmp.path().to_path_buf()).unwrap();

    mem.save(&make_conversation("s1", "hello", Utc::now()))
        .await
        .unwrap();

    // Place a .tmp file (would happen during crash)
    std::fs::write(tmp.path().join("orphan.json.tmp"), "{}").unwrap();

    let sessions = mem.list_sessions(10).await.unwrap();
    assert_eq!(sessions.len(), 1);
    assert_eq!(sessions[0].session_id.as_str(), "s1");
}

// ─── Search ─────────────────────────────────────────────────────────────

#[tokio::test]
async fn test_search_finds_user_message() {
    let tmp = tempfile::tempdir().unwrap();
    let mem = FileMemory::new(tmp.path().to_path_buf()).unwrap();
    mem.save(&make_conversation("s1", "Hello World", Utc::now()))
        .await
        .unwrap();

    let results = mem.search("hello", 10).await.unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].session_id.as_str(), "s1");
}

#[tokio::test]
async fn test_search_finds_assistant_response() {
    let tmp = tempfile::tempdir().unwrap();
    let mem = FileMemory::new(tmp.path().to_path_buf()).unwrap();
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
            assistant_response: Some(Message {
                role: Role::Assistant,
                content: vec![ContentBlock::Text {
                    text: "specific phrase here".to_string(),
                }],
                timestamp: now,
            }),
            tool_invocations: vec![],
            started_at: now,
            completed_at: Some(now),
        }],
        created_at: now,
        updated_at: now,
        model_id: "test-model".to_string(),
        provider_id: "test-provider".to_string(),
    };
    mem.save(&conv).await.unwrap();

    let results = mem.search("specific phrase", 10).await.unwrap();
    assert_eq!(results.len(), 1);
}

#[tokio::test]
async fn test_search_case_insensitive() {
    let tmp = tempfile::tempdir().unwrap();
    let mem = FileMemory::new(tmp.path().to_path_buf()).unwrap();
    mem.save(&make_conversation("s1", "Hello World", Utc::now()))
        .await
        .unwrap();

    let results = mem.search("HELLO", 10).await.unwrap();
    assert_eq!(results.len(), 1);
}

#[tokio::test]
async fn test_search_no_match() {
    let tmp = tempfile::tempdir().unwrap();
    let mem = FileMemory::new(tmp.path().to_path_buf()).unwrap();
    mem.save(&make_conversation("s1", "Hello World", Utc::now()))
        .await
        .unwrap();

    let results = mem.search("nonexistent", 10).await.unwrap();
    assert!(results.is_empty());
}

#[tokio::test]
async fn test_search_empty_query() {
    let tmp = tempfile::tempdir().unwrap();
    let mem = FileMemory::new(tmp.path().to_path_buf()).unwrap();
    mem.save(&make_conversation("s1", "Hello World", Utc::now()))
        .await
        .unwrap();

    let results = mem.search("", 10).await.unwrap();
    assert!(results.is_empty());
}

#[tokio::test]
async fn test_search_limit() {
    let tmp = tempfile::tempdir().unwrap();
    let mem = FileMemory::new(tmp.path().to_path_buf()).unwrap();
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
async fn test_preview_truncates_at_100_chars() {
    let tmp = tempfile::tempdir().unwrap();
    let mem = FileMemory::new(tmp.path().to_path_buf()).unwrap();
    let long_text: String = "a".repeat(200);
    mem.save(&make_conversation("s1", &long_text, Utc::now()))
        .await
        .unwrap();

    let sessions = mem.list_sessions(10).await.unwrap();
    assert_eq!(sessions[0].preview.len(), 100);
}

#[tokio::test]
async fn test_preview_short_message() {
    let tmp = tempfile::tempdir().unwrap();
    let mem = FileMemory::new(tmp.path().to_path_buf()).unwrap();
    mem.save(&make_conversation("s1", "short", Utc::now()))
        .await
        .unwrap();

    let sessions = mem.list_sessions(10).await.unwrap();
    assert_eq!(sessions[0].preview, "short");
}

#[tokio::test]
async fn test_preview_empty_turns() {
    let tmp = tempfile::tempdir().unwrap();
    let mem = FileMemory::new(tmp.path().to_path_buf()).unwrap();
    let now = Utc::now();
    let conv = Conversation {
        session_id: SessionId::from_string("s1"),
        system_prompt: None,
        turns: vec![],
        created_at: now,
        updated_at: now,
        model_id: "test-model".to_string(),
        provider_id: "test-provider".to_string(),
    };
    mem.save(&conv).await.unwrap();

    let sessions = mem.list_sessions(10).await.unwrap();
    assert!(sessions[0].preview.is_empty());
}

#[tokio::test]
async fn test_summary_turn_count() {
    let tmp = tempfile::tempdir().unwrap();
    let mem = FileMemory::new(tmp.path().to_path_buf()).unwrap();
    let now = Utc::now();
    let make_turn = |text: &str| Turn {
        user_message: Message {
            role: Role::User,
            content: vec![ContentBlock::Text {
                text: text.to_string(),
            }],
            timestamp: now,
        },
        assistant_response: None,
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
        model_id: "test-model".to_string(),
        provider_id: "test-provider".to_string(),
    };
    mem.save(&conv).await.unwrap();

    let sessions = mem.list_sessions(10).await.unwrap();
    assert_eq!(sessions[0].turn_count, 3);
}

// ─── Property-Based Tests ───────────────────────────────────────────────

proptest! {
    #[test]
    fn proptest_save_load_roundtrip(
        id in "[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}",
        text in ".*",
    ) {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let tmp = tempfile::tempdir().unwrap();
            let mem = FileMemory::new(tmp.path().to_path_buf()).unwrap();
            let conv = make_conversation(&id, &text, Utc::now());
            mem.save(&conv).await.unwrap();

            let loaded = mem
                .load(&SessionId::from_string(&id))
                .await
                .unwrap()
                .unwrap();
            prop_assert_eq!(loaded, conv);
            Ok(())
        })?;
    }

    #[test]
    fn proptest_session_id_never_escapes_base_dir(input in "\\PC{1,100}") {
        let tmp = tempfile::tempdir().unwrap();
        let mem = FileMemory::new(tmp.path().to_path_buf()).unwrap();
        let session_id = SessionId::from_string(&input);

        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            // Either load returns Ok(None) (valid ID, no file) or Err (rejected ID).
            // It must NEVER succeed with a path outside base_dir.
            match mem.load(&session_id).await {
                Ok(None) => {
                    // Valid ID, no file — this is fine. Verify the session_path
                    // would be under base_dir by checking the ID is safe.
                    prop_assert!(
                        input.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
                            && !input.is_empty()
                    );
                }
                Err(MemoryError::Io(_)) => {
                    // Rejected — path-unsafe characters. This is the expected behavior
                    // for traversal attempts.
                }
                other => {
                    prop_assert!(false, "unexpected result: {other:?}");
                }
            }
            Ok(())
        })?;
    }
}
