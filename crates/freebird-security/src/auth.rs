//! Session key authentication for the Freebird daemon.
//!
//! Session keys are 256-bit cryptographically random tokens used to authenticate
//! users connecting to the daemon. Keys are stored as SHA-256 hashes (never raw),
//! verified with constant-time comparison, and support configurable TTL and
//! scoped capabilities.

use std::time::Duration;

use chrono::{DateTime, Utc};
use freebird_traits::tool::Capability;
use ring::rand::{SecureRandom, SystemRandom};
use ring::{digest, hmac};
use secrecy::SecretString;
use serde::{Deserialize, Serialize};

use crate::error::SecurityError;

/// A stored session credential. The raw key is never stored — only its hash.
///
/// This is a serialization type for `~/.freebird/keys.json`. Fields are
/// immutable after construction via [`generate_session_key()`].
///
/// Stores `Vec<Capability>` rather than `CapabilityGrant` because:
/// - `CapabilityGrant` includes `sandbox_root` which is a runtime-configured
///   path that may change between daemon restarts
/// - The runtime constructs `CapabilityGrant` at session creation by combining
///   the credential's capabilities + the runtime's configured sandbox root
///   + the credential's expiration
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionCredential {
    /// Public identifier for this key (e.g., `freebird_a1b2c3d4e5f6`).
    /// Derived from the first 12 hex chars of the key hash.
    pub key_id: String,
    /// SHA-256 hash of the raw key (hex-encoded, 64 chars).
    pub key_hash: String,
    /// When this key was issued.
    pub issued_at: DateTime<Utc>,
    /// When this key expires. `None` means no expiration (discouraged).
    pub expires_at: Option<DateTime<Utc>>,
    /// Which capabilities this key grants.
    pub capabilities: Vec<Capability>,
}

/// Generate a new session key.
///
/// Returns a tuple of:
/// - The raw key as a [`SecretString`] (give to the user ONCE, then discard)
/// - The [`SessionCredential`] (store in `keys.json`; contains only the hash)
///
/// The raw key is a 64-character hex string encoding 32 random bytes (256 bits
/// of entropy). The credential stores the SHA-256 hash of this hex string.
///
/// # Panics
///
/// Panics if the system RNG fails (indicates a broken OS — no recovery possible).
/// This is the only `expect()` in the codebase, matching `ring`'s own guidance.
#[allow(clippy::expect_used)]
#[must_use]
pub fn generate_session_key(
    capabilities: Vec<Capability>,
    ttl: Option<Duration>,
) -> (SecretString, SessionCredential) {
    let rng = SystemRandom::new();
    let mut key_bytes = [0u8; 32];
    rng.fill(&mut key_bytes)
        .expect("system RNG failed — OS entropy source is broken");

    let raw_key = hex::encode(key_bytes);
    let key_hash = hex::encode(digest::digest(&digest::SHA256, raw_key.as_bytes()));
    let key_id = format!("freebird_{}", &key_hash[..12]);

    let now = Utc::now();
    let expires_at = ttl.and_then(|d| chrono::Duration::from_std(d).ok().map(|cd| now + cd));

    let credential = SessionCredential {
        key_id,
        key_hash,
        issued_at: now,
        expires_at,
        capabilities,
    };

    (SecretString::from(raw_key), credential)
}

/// Derive a `key_id` from a raw key without needing the stored credential.
///
/// Used by the router to look up the correct [`SessionCredential`] from
/// `keys.json` before calling [`verify_session_key()`]. The flow is:
/// 1. User provides raw key
/// 2. Router calls `derive_key_id(raw_key)` to get the `key_id`
/// 3. Router looks up [`SessionCredential`] by `key_id`
/// 4. Router calls `verify_session_key(raw_key, &credential)`
#[must_use]
pub fn derive_key_id(raw_key: &str) -> String {
    let hash = hex::encode(digest::digest(&digest::SHA256, raw_key.as_bytes()));
    format!("freebird_{}", &hash[..12])
}

/// Verify a session key against a stored credential.
///
/// Checks expiration FIRST, then performs constant-time hash comparison.
///
/// **Timing analysis**: Checking expiration before hash comparison leaks
/// whether a key is expired vs invalid. This is acceptable because:
/// - The `key_id` is not secret (it's derived from the hash, used in logs)
/// - Knowing a key is expired doesn't help an attacker forge a valid key
/// - The hash comparison itself is constant-time (the security-critical path)
///
/// # Errors
///
/// - [`SecurityError::SessionExpired`] if the key has expired
/// - [`SecurityError::InvalidSessionKey`] if the key hash doesn't match
#[must_use = "security check result must not be silently discarded"]
pub fn verify_session_key<'a>(
    raw_key: &str,
    stored: &'a SessionCredential,
) -> Result<&'a [Capability], SecurityError> {
    // 1. Check expiry first
    if let Some(expires) = stored.expires_at {
        if Utc::now() > expires {
            return Err(SecurityError::SessionExpired {
                key_id: stored.key_id.clone(),
            });
        }
    }

    // 2. Constant-time comparison via HMAC verification.
    //    Hash the provided key, then use HMAC to compare the two hash strings.
    //    ring::hmac::verify is inherently constant-time, preventing timing
    //    side-channel attacks. We HMAC the stored hash to produce a tag, then
    //    verify that HMAC(provided_hash) produces the same tag.
    let provided_hash = hex::encode(digest::digest(&digest::SHA256, raw_key.as_bytes()));
    let verify_key = hmac::Key::new(hmac::HMAC_SHA256, b"freebird-session-key-verify");
    let tag = hmac::sign(&verify_key, stored.key_hash.as_bytes());

    if hmac::verify(&verify_key, provided_hash.as_bytes(), tag.as_ref()).is_err() {
        return Err(SecurityError::InvalidSessionKey {
            key_id: stored.key_id.clone(),
        });
    }

    Ok(&stored.capabilities)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;
    use secrecy::ExposeSecret;

    // ── Key Generation ──────────────────────────────────────────

    #[test]
    fn test_generate_produces_valid_credential() {
        let caps = vec![Capability::FileRead];
        let (_, cred) = generate_session_key(caps, None);

        assert!(cred.key_id.starts_with("freebird_"));
        assert_eq!(cred.key_id.len(), "freebird_".len() + 12);
        assert_eq!(cred.key_hash.len(), 64);
        assert!(cred.key_hash.chars().all(|c| c.is_ascii_hexdigit()));
        // issued_at should be very recent
        let elapsed = Utc::now() - cred.issued_at;
        assert!(elapsed.num_seconds() < 2);
    }

    #[test]
    fn test_generated_key_is_64_hex_chars() {
        let (raw_key, _) = generate_session_key(vec![], None);
        let exposed = raw_key.expose_secret();
        assert_eq!(exposed.len(), 64);
        assert!(exposed.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn test_two_keys_are_different() {
        let (key1, _) = generate_session_key(vec![], None);
        let (key2, _) = generate_session_key(vec![], None);
        assert_ne!(key1.expose_secret(), key2.expose_secret());
    }

    #[test]
    fn test_ttl_none_produces_no_expiration() {
        let (_, cred) = generate_session_key(vec![], None);
        assert!(cred.expires_at.is_none());
    }

    #[test]
    fn test_ttl_sets_correct_expiration() {
        let ttl = Duration::from_secs(3600); // 1 hour
        let before = Utc::now();
        let (_, cred) = generate_session_key(vec![], Some(ttl));
        let after = Utc::now();

        let expires = cred.expires_at.unwrap();
        let one_hour = chrono::Duration::seconds(3600);
        assert!(expires >= before + one_hour - chrono::Duration::seconds(1));
        assert!(expires <= after + one_hour + chrono::Duration::seconds(1));
    }

    #[test]
    fn test_capabilities_stored_in_credential() {
        let caps = vec![Capability::FileRead, Capability::ShellExecute];
        let (_, cred) = generate_session_key(caps.clone(), None);
        assert_eq!(cred.capabilities, caps);
    }

    // ── Key Verification ────────────────────────────────────────

    #[test]
    fn test_valid_key_verifies_successfully() {
        let caps = vec![Capability::FileRead, Capability::FileWrite];
        let (raw_key, cred) = generate_session_key(caps.clone(), None);

        let result = verify_session_key(raw_key.expose_secret(), &cred);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), &caps);
    }

    #[test]
    fn test_wrong_key_returns_invalid() {
        let (_, cred) = generate_session_key(vec![], None);
        let wrong_key = "a".repeat(64);

        let result = verify_session_key(&wrong_key, &cred);
        assert!(result.is_err());
        match result.unwrap_err() {
            SecurityError::InvalidSessionKey { key_id } => {
                assert_eq!(key_id, cred.key_id);
            }
            other => panic!("expected InvalidSessionKey, got {other:?}"),
        }
    }

    #[test]
    fn test_expired_key_returns_expired() {
        let (raw_key, mut cred) = generate_session_key(vec![], None);
        // Manually set expiration to the past
        cred.expires_at = Some(Utc::now() - chrono::Duration::seconds(60));

        let result = verify_session_key(raw_key.expose_secret(), &cred);
        assert!(result.is_err());
        match result.unwrap_err() {
            SecurityError::SessionExpired { key_id } => {
                assert_eq!(key_id, cred.key_id);
            }
            other => panic!("expected SessionExpired, got {other:?}"),
        }
    }

    #[test]
    fn test_no_expiration_never_expires() {
        let (raw_key, cred) = generate_session_key(vec![], None);
        assert!(cred.expires_at.is_none());
        assert!(verify_session_key(raw_key.expose_secret(), &cred).is_ok());
    }

    #[test]
    fn test_key_just_before_expiry_succeeds() {
        let (raw_key, mut cred) = generate_session_key(vec![], None);
        // Set expiration to 10 seconds from now — well within range
        cred.expires_at = Some(Utc::now() + chrono::Duration::seconds(10));
        assert!(verify_session_key(raw_key.expose_secret(), &cred).is_ok());
    }

    #[test]
    fn test_key_just_after_expiry_fails() {
        let (raw_key, mut cred) = generate_session_key(vec![], None);
        // Set expiration to 1 second in the past
        cred.expires_at = Some(Utc::now() - chrono::Duration::seconds(1));

        let result = verify_session_key(raw_key.expose_secret(), &cred);
        assert!(matches!(result, Err(SecurityError::SessionExpired { .. })));
    }

    // ── derive_key_id ───────────────────────────────────────────

    #[test]
    fn test_derive_key_id_matches_generated() {
        let (raw_key, cred) = generate_session_key(vec![], None);
        let derived = derive_key_id(raw_key.expose_secret());
        assert_eq!(derived, cred.key_id);
    }

    #[test]
    fn test_derive_key_id_different_keys_different_ids() {
        let (key1, _) = generate_session_key(vec![], None);
        let (key2, _) = generate_session_key(vec![], None);
        let id1 = derive_key_id(key1.expose_secret());
        let id2 = derive_key_id(key2.expose_secret());
        assert_ne!(id1, id2);
    }

    // ── Security Properties ─────────────────────────────────────

    #[test]
    fn test_raw_key_not_in_credential_debug() {
        let (raw_key, cred) = generate_session_key(vec![], None);
        let debug_output = format!("{cred:?}");
        assert!(
            !debug_output.contains(raw_key.expose_secret()),
            "raw key must not appear in credential Debug output"
        );
    }

    #[test]
    fn test_key_id_does_not_reveal_full_hash() {
        let (_, cred) = generate_session_key(vec![], None);
        // key_id = "freebird_" + 12 hex chars = 21 chars total
        // key_hash = 64 hex chars
        assert!(cred.key_id.len() < cred.key_hash.len());
        let suffix = &cred.key_id["freebird_".len()..];
        assert_eq!(suffix.len(), 12);
        assert!(suffix.chars().all(|c| c.is_ascii_hexdigit()));
    }

    // ── Serde Roundtrip ─────────────────────────────────────────

    #[test]
    fn test_credential_serde_roundtrip() {
        let caps = vec![Capability::FileRead, Capability::ShellExecute];
        let (_, cred) = generate_session_key(caps, Some(Duration::from_secs(86400)));

        let json = serde_json::to_string_pretty(&cred).unwrap();
        let deserialized: SessionCredential = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized, cred);
    }

    // ── Property-Based Test ─────────────────────────────────────

    mod prop {
        use super::*;
        use proptest::prelude::*;
        use secrecy::ExposeSecret;

        fn arb_capability() -> impl Strategy<Value = Capability> {
            prop_oneof![
                Just(Capability::FileRead),
                Just(Capability::FileWrite),
                Just(Capability::FileDelete),
                Just(Capability::ShellExecute),
                Just(Capability::ProcessSpawn),
                Just(Capability::NetworkOutbound),
                Just(Capability::NetworkListen),
                Just(Capability::EnvRead),
            ]
        }

        proptest! {
            #[test]
            fn test_generate_then_verify_always_succeeds(
                caps in proptest::collection::vec(arb_capability(), 0..8),
            ) {
                let (raw_key, cred) = generate_session_key(caps, None);
                let result = verify_session_key(raw_key.expose_secret(), &cred);
                prop_assert!(result.is_ok());
            }
        }
    }
}
