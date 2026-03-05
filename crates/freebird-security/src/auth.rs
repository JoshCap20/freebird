//! Session key authentication for the Freebird daemon.
//!
//! Session keys are 256-bit cryptographically random tokens used to authenticate
//! users connecting to the daemon. Keys are stored as SHA-256 hashes (never raw),
//! verified with constant-time comparison, and support configurable TTL and
//! scoped capabilities.
//!
//! [`SessionCredential`] enforces structural invariants by construction:
//! - Created via [`generate_session_key()`] (always valid)
//! - Deserialized via `serde` with [`#[serde(try_from)]`](serde::Deserialize)
//!   which validates on parse (not after)

use std::time::Duration;

use chrono::{DateTime, Utc};
use freebird_traits::tool::Capability;
use ring::rand::{SecureRandom, SystemRandom};
use ring::{digest, hmac};
use secrecy::SecretString;
use serde::{Deserialize, Serialize};

use crate::error::SecurityError;

/// Prefix for all key IDs, making them greppable in logs.
const KEY_ID_PREFIX: &str = "freebird_";

/// Number of hex characters from the hash used in the key ID.
/// 12 hex chars = 48 bits, giving ~281 trillion unique IDs before collision.
const KEY_ID_HASH_LEN: usize = 12;

/// SHA-256 hash a raw key string and return the hex-encoded digest.
///
/// This is the single source of truth for key hashing. Used by generation,
/// derivation, and verification to ensure consistent behavior.
fn hash_raw_key(raw_key: &str) -> String {
    hex::encode(digest::digest(&digest::SHA256, raw_key.as_bytes()))
}

/// Derive a key ID from a hex-encoded hash string.
fn key_id_from_hash(hash: &str) -> String {
    format!("{KEY_ID_PREFIX}{}", &hash[..KEY_ID_HASH_LEN])
}

/// A stored session credential. The raw key is never stored — only its hash.
///
/// This is a serialization type for `~/.freebird/keys.json`. Fields are
/// private and immutable after construction — access via getter methods.
///
/// **Construction paths** (both enforce invariants):
/// - [`generate_session_key()`] — valid by construction
/// - [`serde::Deserialize`] — validated on parse via
///   [`TryFrom<SessionCredentialUnchecked>`]
///
/// Stores `Vec<Capability>` rather than `CapabilityGrant` because:
/// - `CapabilityGrant` includes `sandbox_root` which is a runtime-configured
///   path that may change between daemon restarts
/// - The runtime constructs `CapabilityGrant` at session creation by combining
///   the credential's capabilities + the runtime's configured sandbox root
///   + the credential's expiration
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "SessionCredentialUnchecked")]
pub struct SessionCredential {
    /// Public identifier (e.g., `freebird_a1b2c3d4e5f6`).
    key_id: String,
    /// SHA-256 hash of the raw key (hex-encoded, 64 chars).
    key_hash: String,
    /// When this key was issued.
    issued_at: DateTime<Utc>,
    /// When this key expires. `None` means no expiration (discouraged).
    expires_at: Option<DateTime<Utc>>,
    /// Which capabilities this key grants.
    capabilities: Vec<Capability>,
}

impl SessionCredential {
    /// Public identifier for this key (e.g., `freebird_a1b2c3d4e5f6`).
    /// Derived from the first 12 hex chars of the key hash.
    #[must_use]
    pub fn key_id(&self) -> &str {
        &self.key_id
    }

    /// SHA-256 hash of the raw key (hex-encoded, 64 chars).
    #[must_use]
    pub fn key_hash(&self) -> &str {
        &self.key_hash
    }

    /// When this key was issued.
    #[must_use]
    pub const fn issued_at(&self) -> DateTime<Utc> {
        self.issued_at
    }

    /// When this key expires. `None` means no expiration (discouraged).
    #[must_use]
    pub const fn expires_at(&self) -> Option<DateTime<Utc>> {
        self.expires_at
    }

    /// Which capabilities this key grants.
    #[must_use]
    pub fn capabilities(&self) -> &[Capability] {
        &self.capabilities
    }

    /// Validate structural invariants. Called by [`TryFrom`] on deserialization
    /// to enforce parse-don't-validate — if you hold a `SessionCredential`,
    /// it is structurally valid.
    fn validate(&self) -> Result<(), SecurityError> {
        if self.key_hash.len() != 64 || !self.key_hash.chars().all(|c| c.is_ascii_hexdigit()) {
            return Err(SecurityError::InvalidCredential {
                reason: "key_hash must be exactly 64 hex characters".into(),
            });
        }

        let expected_id = key_id_from_hash(&self.key_hash);
        if self.key_id != expected_id {
            return Err(SecurityError::InvalidCredential {
                reason: format!(
                    "key_id `{}` does not match hash (expected `{expected_id}`)",
                    self.key_id
                ),
            });
        }

        Ok(())
    }
}

/// Raw deserialization helper for [`SessionCredential`].
///
/// Serde deserializes into this unchecked type first, then
/// [`TryFrom`] validates invariants before producing a
/// [`SessionCredential`]. This enforces parse-don't-validate
/// (CLAUDE.md §2.3).
#[derive(Deserialize)]
struct SessionCredentialUnchecked {
    key_id: String,
    key_hash: String,
    issued_at: DateTime<Utc>,
    expires_at: Option<DateTime<Utc>>,
    capabilities: Vec<Capability>,
}

impl TryFrom<SessionCredentialUnchecked> for SessionCredential {
    type Error = SecurityError;

    fn try_from(raw: SessionCredentialUnchecked) -> Result<Self, Self::Error> {
        let cred = Self {
            key_id: raw.key_id,
            key_hash: raw.key_hash,
            issued_at: raw.issued_at,
            expires_at: raw.expires_at,
            capabilities: raw.capabilities,
        };
        cred.validate()?;
        Ok(cred)
    }
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
/// # Errors
///
/// Returns [`SecurityError::InvalidTtl`] if the TTL is too large for chrono
/// to represent (roughly > 292 billion years). A security-critical module
/// must never silently degrade to no-expiration.
///
/// # Panics
///
/// Panics if the system RNG fails (indicates a broken OS — no recovery possible).
/// This is the only `expect()` in the codebase, matching `ring`'s own guidance.
#[allow(clippy::expect_used)]
pub fn generate_session_key(
    capabilities: Vec<Capability>,
    ttl: Option<Duration>,
) -> Result<(SecretString, SessionCredential), SecurityError> {
    let rng = SystemRandom::new();
    let mut key_bytes = [0u8; 32];
    rng.fill(&mut key_bytes)
        .expect("system RNG failed — OS entropy source is broken");

    // Zeroize raw bytes after encoding — defense-in-depth against stack exposure
    let raw_key = hex::encode(key_bytes);
    key_bytes.fill(0);

    let key_hash = hash_raw_key(&raw_key);
    let key_id = key_id_from_hash(&key_hash);

    let now = Utc::now();
    let expires_at = match ttl {
        Some(d) => {
            let cd = chrono::Duration::from_std(d).map_err(|_| SecurityError::InvalidTtl {
                seconds: d.as_secs(),
            })?;
            Some(now + cd)
        }
        None => None,
    };

    let credential = SessionCredential {
        key_id,
        key_hash,
        issued_at: now,
        expires_at,
        capabilities,
    };

    tracing::debug!(key_id = %credential.key_id, "session key generated");

    Ok((SecretString::from(raw_key), credential))
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
    let hash = hash_raw_key(raw_key);
    key_id_from_hash(&hash)
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
            tracing::warn!(key_id = %stored.key_id, "session key verification failed: expired");
            return Err(SecurityError::SessionExpired {
                key_id: stored.key_id.clone(),
            });
        }
    }

    // 2. Constant-time comparison via HMAC verification.
    //    We use HMAC purely as a constant-time equality check — the key is
    //    static because secrecy of the HMAC key is irrelevant here. What
    //    matters is that ring::hmac::verify runs in constant time, preventing
    //    timing side-channel attacks on the hash comparison.
    let provided_hash = hash_raw_key(raw_key);
    let verify_key = hmac::Key::new(hmac::HMAC_SHA256, b"freebird-session-key-verify");
    let tag = hmac::sign(&verify_key, stored.key_hash.as_bytes());

    if hmac::verify(&verify_key, provided_hash.as_bytes(), tag.as_ref()).is_err() {
        tracing::warn!(key_id = %stored.key_id, "session key verification failed: invalid key");
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

    /// Helper: generate a key and unwrap — tests that don't exercise the
    /// error path of generation can use this to reduce boilerplate.
    fn make_key(caps: Vec<Capability>, ttl: Option<Duration>) -> (SecretString, SessionCredential) {
        generate_session_key(caps, ttl).unwrap()
    }

    // ── Key Generation ──────────────────────────────────────────

    #[test]
    fn test_generate_produces_valid_credential() {
        let caps = vec![Capability::FileRead];
        let (_, cred) = make_key(caps, None);

        assert!(cred.key_id.starts_with(KEY_ID_PREFIX));
        assert_eq!(cred.key_id.len(), KEY_ID_PREFIX.len() + KEY_ID_HASH_LEN);
        assert_eq!(cred.key_hash.len(), 64);
        assert!(cred.key_hash.chars().all(|c| c.is_ascii_hexdigit()));
        // issued_at should be very recent
        let elapsed = Utc::now() - cred.issued_at;
        assert!(elapsed.num_seconds() < 2);
    }

    #[test]
    fn test_getters_match_internal_fields() {
        let caps = vec![Capability::FileRead, Capability::ShellExecute];
        let (_, cred) = make_key(caps, Some(Duration::from_secs(3600)));

        assert_eq!(cred.key_id(), &cred.key_id);
        assert_eq!(cred.key_hash(), &cred.key_hash);
        assert_eq!(cred.issued_at(), cred.issued_at);
        assert_eq!(cred.expires_at(), cred.expires_at);
        assert_eq!(cred.capabilities(), &cred.capabilities[..]);
    }

    #[test]
    fn test_generated_key_is_64_hex_chars() {
        let (raw_key, _) = make_key(vec![], None);
        let exposed = raw_key.expose_secret();
        assert_eq!(exposed.len(), 64);
        assert!(exposed.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn test_two_keys_are_different() {
        let (key1, _) = make_key(vec![], None);
        let (key2, _) = make_key(vec![], None);
        assert_ne!(key1.expose_secret(), key2.expose_secret());
    }

    #[test]
    fn test_ttl_none_produces_no_expiration() {
        let (_, cred) = make_key(vec![], None);
        assert!(cred.expires_at.is_none());
    }

    #[test]
    fn test_ttl_sets_correct_expiration() {
        let ttl = Duration::from_secs(3600); // 1 hour
        let before = Utc::now();
        let (_, cred) = make_key(vec![], Some(ttl));
        let after = Utc::now();

        let expires = cred.expires_at.unwrap();
        let one_hour = chrono::Duration::seconds(3600);
        assert!(expires >= before + one_hour - chrono::Duration::seconds(1));
        assert!(expires <= after + one_hour + chrono::Duration::seconds(1));
    }

    #[test]
    fn test_ttl_overflow_returns_error() {
        let result = generate_session_key(vec![], Some(Duration::MAX));
        assert!(result.is_err());
        match result.unwrap_err() {
            SecurityError::InvalidTtl { seconds } => {
                assert_eq!(seconds, Duration::MAX.as_secs());
            }
            other => panic!("expected InvalidTtl, got {other:?}"),
        }
    }

    #[test]
    fn test_capabilities_stored_in_credential() {
        let caps = vec![Capability::FileRead, Capability::ShellExecute];
        let (_, cred) = make_key(caps.clone(), None);
        assert_eq!(cred.capabilities, caps);
    }

    // ── Key Verification ────────────────────────────────────────

    #[test]
    fn test_valid_key_verifies_successfully() {
        let caps = vec![Capability::FileRead, Capability::FileWrite];
        let (raw_key, cred) = make_key(caps.clone(), None);

        let result = verify_session_key(raw_key.expose_secret(), &cred);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), &caps);
    }

    #[test]
    fn test_wrong_key_returns_invalid() {
        let (_, cred) = make_key(vec![], None);
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
        let (raw_key, mut cred) = make_key(vec![], None);
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
        let (raw_key, cred) = make_key(vec![], None);
        assert!(cred.expires_at.is_none());
        assert!(verify_session_key(raw_key.expose_secret(), &cred).is_ok());
    }

    #[test]
    fn test_key_just_before_expiry_succeeds() {
        let (raw_key, mut cred) = make_key(vec![], None);
        // Set expiration to 10 seconds from now — well within range
        cred.expires_at = Some(Utc::now() + chrono::Duration::seconds(10));
        assert!(verify_session_key(raw_key.expose_secret(), &cred).is_ok());
    }

    #[test]
    fn test_key_just_after_expiry_fails() {
        let (raw_key, mut cred) = make_key(vec![], None);
        // Set expiration to 1 second in the past
        cred.expires_at = Some(Utc::now() - chrono::Duration::seconds(1));

        let result = verify_session_key(raw_key.expose_secret(), &cred);
        assert!(matches!(result, Err(SecurityError::SessionExpired { .. })));
    }

    // ── derive_key_id ───────────────────────────────────────────

    #[test]
    fn test_derive_key_id_matches_generated() {
        let (raw_key, cred) = make_key(vec![], None);
        let derived = derive_key_id(raw_key.expose_secret());
        assert_eq!(derived, cred.key_id);
    }

    #[test]
    fn test_derive_key_id_different_keys_different_ids() {
        let (key1, _) = make_key(vec![], None);
        let (key2, _) = make_key(vec![], None);
        let id1 = derive_key_id(key1.expose_secret());
        let id2 = derive_key_id(key2.expose_secret());
        assert_ne!(id1, id2);
    }

    // ── Security Properties ─────────────────────────────────────

    #[test]
    fn test_raw_key_not_in_credential_debug() {
        let (raw_key, cred) = make_key(vec![], None);
        let debug_output = format!("{cred:?}");
        assert!(
            !debug_output.contains(raw_key.expose_secret()),
            "raw key must not appear in credential Debug output"
        );
    }

    #[test]
    fn test_key_id_does_not_reveal_full_hash() {
        let (_, cred) = make_key(vec![], None);
        assert!(cred.key_id.len() < cred.key_hash.len());
        let suffix = &cred.key_id[KEY_ID_PREFIX.len()..];
        assert_eq!(suffix.len(), KEY_ID_HASH_LEN);
        assert!(suffix.chars().all(|c| c.is_ascii_hexdigit()));
    }

    // ── Credential Validation (parse-don't-validate) ────────────

    #[test]
    fn test_validate_rejects_short_hash() {
        let (_, mut cred) = make_key(vec![], None);
        cred.key_hash = "abcd".into();
        assert!(matches!(
            cred.validate(),
            Err(SecurityError::InvalidCredential { .. })
        ));
    }

    #[test]
    fn test_validate_rejects_non_hex_hash() {
        let (_, mut cred) = make_key(vec![], None);
        cred.key_hash = "g".repeat(64);
        assert!(matches!(
            cred.validate(),
            Err(SecurityError::InvalidCredential { .. })
        ));
    }

    #[test]
    fn test_validate_rejects_mismatched_key_id() {
        let (_, mut cred) = make_key(vec![], None);
        cred.key_id = "freebird_000000000000".into();
        assert!(matches!(
            cred.validate(),
            Err(SecurityError::InvalidCredential { .. })
        ));
    }

    #[test]
    fn test_validate_accepts_valid_credential() {
        let (_, cred) = make_key(vec![Capability::FileRead], None);
        assert!(cred.validate().is_ok());
    }

    #[test]
    fn test_deserialize_rejects_invalid_credential() {
        let (_, cred) = make_key(vec![], None);
        let mut json_val: serde_json::Value = serde_json::to_value(&cred).unwrap();
        json_val
            .as_object_mut()
            .unwrap()
            .insert("key_hash".into(), serde_json::Value::String("abcd".into()));

        let result: Result<SessionCredential, _> = serde_json::from_value(json_val);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("key_hash must be exactly 64 hex characters"),
            "expected validation error in deserialization, got: {err_msg}"
        );
    }

    // ── Serde Roundtrip ─────────────────────────────────────────

    #[test]
    fn test_credential_serde_roundtrip() {
        let caps = vec![Capability::FileRead, Capability::ShellExecute];
        let (_, cred) = make_key(caps, Some(Duration::from_secs(86400)));

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
                let (raw_key, cred) = make_key(caps, None);
                let result = verify_session_key(raw_key.expose_secret(), &cred);
                prop_assert!(result.is_ok());
            }
        }
    }
}
