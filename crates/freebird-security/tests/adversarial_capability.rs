//! Adversarial tests for capability grants — privilege escalation,
//! expiration edge cases, sub-grant chain depth.

#![allow(
    clippy::all,
    clippy::pedantic,
    clippy::nursery,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

use std::collections::BTreeSet;

use chrono::{TimeDelta, Utc};
use freebird_security::capability::CapabilityGrant;
use freebird_traits::tool::Capability;

fn all_capabilities() -> BTreeSet<Capability> {
    BTreeSet::from([
        Capability::FileRead,
        Capability::FileWrite,
        Capability::FileDelete,
        Capability::ShellExecute,
        Capability::ProcessSpawn,
        Capability::NetworkOutbound,
        Capability::NetworkListen,
        Capability::EnvRead,
    ])
}

// ---------------------------------------------------------------------------
// Sub-grant chain depth
// ---------------------------------------------------------------------------

#[test]
fn sub_grant_chain_depth_3() {
    let dir = tempfile::tempdir().unwrap();
    let sandbox_b = dir.path().join("sub_b");
    let sandbox_c = sandbox_b.join("sub_c");
    std::fs::create_dir_all(&sandbox_c).unwrap();

    let caps_a = all_capabilities();
    let caps_b: BTreeSet<Capability> = BTreeSet::from([
        Capability::FileRead,
        Capability::FileWrite,
        Capability::ShellExecute,
    ]);
    let caps_c: BTreeSet<Capability> = BTreeSet::from([Capability::FileRead]);

    let expiry_a = Utc::now() + TimeDelta::hours(1);
    let expiry_b = Utc::now() + TimeDelta::minutes(30);
    let expiry_c = Utc::now() + TimeDelta::minutes(15);

    let grant_a = CapabilityGrant::new(caps_a, dir.path().to_path_buf(), Some(expiry_a)).unwrap();
    let grant_b = grant_a
        .sub_grant(caps_b, sandbox_b, Some(expiry_b))
        .unwrap();
    let grant_c = grant_b
        .sub_grant(caps_c, sandbox_c, Some(expiry_c))
        .unwrap();

    // C should only have FileRead
    assert!(grant_c.check(&Capability::FileRead).is_ok());
    assert!(grant_c.check(&Capability::FileWrite).is_err());
    assert!(grant_c.check(&Capability::ShellExecute).is_err());
}

#[test]
fn sub_grant_cannot_escalate_capabilities() {
    let dir = tempfile::tempdir().unwrap();
    let parent_caps: BTreeSet<Capability> = BTreeSet::from([Capability::FileRead]);
    let child_caps: BTreeSet<Capability> =
        BTreeSet::from([Capability::FileRead, Capability::FileWrite]);

    let parent = CapabilityGrant::new(parent_caps, dir.path().to_path_buf(), None).unwrap();
    let result = parent.sub_grant(child_caps, dir.path().to_path_buf(), None);
    assert!(
        result.is_err(),
        "sub-grant must not escalate beyond parent capabilities"
    );
}

#[test]
fn sub_grant_clamps_expiry_to_parent() {
    let dir = tempfile::tempdir().unwrap();
    let caps: BTreeSet<Capability> = BTreeSet::from([Capability::FileRead]);
    let parent_expiry = Utc::now() + TimeDelta::minutes(30);
    let child_expiry = Utc::now() + TimeDelta::hours(2);

    let parent =
        CapabilityGrant::new(caps.clone(), dir.path().to_path_buf(), Some(parent_expiry)).unwrap();
    let child = parent
        .sub_grant(caps, dir.path().to_path_buf(), Some(child_expiry))
        .unwrap();

    // Implementation clamps child expiry to parent's — child cannot outlive parent
    assert!(
        child.expires_at().unwrap() <= parent_expiry,
        "sub-grant expiry must be clamped to parent expiry"
    );
}

// ---------------------------------------------------------------------------
// Expiration edge cases
// ---------------------------------------------------------------------------

#[test]
fn expired_grant_rejects_all_checks() {
    let dir = tempfile::tempdir().unwrap();
    let caps = all_capabilities();
    let past = Utc::now() - TimeDelta::seconds(1);

    let grant = CapabilityGrant::new(caps, dir.path().to_path_buf(), Some(past)).unwrap();

    assert!(grant.check(&Capability::FileRead).is_err());
    assert!(grant.check(&Capability::ShellExecute).is_err());
    assert!(grant.is_expired());
}

#[test]
fn grant_without_expiry_never_expires() {
    let dir = tempfile::tempdir().unwrap();
    let caps = all_capabilities();

    let grant = CapabilityGrant::new(caps, dir.path().to_path_buf(), None).unwrap();

    assert!(grant.check(&Capability::FileRead).is_ok());
    assert!(!grant.is_expired());
}

// ---------------------------------------------------------------------------
// check_all edge cases
// ---------------------------------------------------------------------------

#[test]
fn check_all_with_empty_required_list() {
    let dir = tempfile::tempdir().unwrap();
    let caps: BTreeSet<Capability> = BTreeSet::from([Capability::FileRead]);

    let grant = CapabilityGrant::new(caps, dir.path().to_path_buf(), None).unwrap();
    assert!(grant.check_all(&[]).is_ok());
}

#[test]
fn check_all_with_missing_capability() {
    let dir = tempfile::tempdir().unwrap();
    let caps: BTreeSet<Capability> = BTreeSet::from([Capability::FileRead]);

    let grant = CapabilityGrant::new(caps, dir.path().to_path_buf(), None).unwrap();
    let result = grant.check_all(&[Capability::FileRead, Capability::FileWrite]);
    assert!(
        result.is_err(),
        "check_all must fail if any capability is missing"
    );
}

// ---------------------------------------------------------------------------
// Sandbox boundary tests
// ---------------------------------------------------------------------------

#[test]
fn sub_grant_sandbox_must_be_within_parent() {
    let dir = tempfile::tempdir().unwrap();
    let other_dir = tempfile::tempdir().unwrap();
    let caps: BTreeSet<Capability> = BTreeSet::from([Capability::FileRead]);

    let parent = CapabilityGrant::new(caps.clone(), dir.path().to_path_buf(), None).unwrap();

    let result = parent.sub_grant(caps, other_dir.path().to_path_buf(), None);
    assert!(
        result.is_err(),
        "sub-grant sandbox must be within parent sandbox"
    );
}

#[test]
fn sub_grant_with_empty_capabilities() {
    let dir = tempfile::tempdir().unwrap();
    let parent = CapabilityGrant::new(all_capabilities(), dir.path().to_path_buf(), None).unwrap();

    let result = parent.sub_grant(BTreeSet::new(), dir.path().to_path_buf(), None);
    assert!(result.is_ok(), "empty capability sub-grant should be valid");

    let empty_grant = result.unwrap();
    assert!(empty_grant.check(&Capability::FileRead).is_err());
}
