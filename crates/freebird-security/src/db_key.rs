//! Database encryption key derivation and source resolution.
//!
//! Derives a SQLCipher-compatible encryption key from a user-provided
//! passphrase using PBKDF2-HMAC-SHA256 with a persistent random salt.
//!
//! Key sources are tried in order: environment variable -> keyfile -> interactive prompt.
//! The agent has no tool or code path to access any of these sources.

use std::path::Path;

use ring::pbkdf2;
use ring::rand::{SecureRandom, SystemRandom};
use secrecy::{ExposeSecret, SecretString};

use crate::error::SecurityError;

/// PBKDF2 algorithm: HMAC-SHA256.
static PBKDF2_ALG: pbkdf2::Algorithm = pbkdf2::PBKDF2_HMAC_SHA256;

/// Length of the derived key in bytes (256-bit for `SQLCipher`).
const KEY_LEN: usize = 32;

/// Length of the salt in bytes.
const SALT_LEN: usize = 32;

/// Environment variable name for the database encryption key.
const ENV_VAR_NAME: &str = "FREEBIRD_DB_KEY";

/// Derives a SQLCipher-compatible hex key from a passphrase and salt.
///
/// Returns a 64-character hex string suitable for `PRAGMA key = 'x"..."'`.
#[must_use]
pub fn derive_key(passphrase: &SecretString, salt: &[u8], iterations: u32) -> SecretString {
    let mut key_bytes = [0u8; KEY_LEN];
    let non_zero_iters = std::num::NonZeroU32::new(iterations).unwrap_or(std::num::NonZeroU32::MIN);
    pbkdf2::derive(
        PBKDF2_ALG,
        non_zero_iters,
        salt,
        passphrase.expose_secret().as_bytes(),
        &mut key_bytes,
    );
    let hex_key = hex::encode(key_bytes);
    // Zeroize the intermediate key bytes
    key_bytes.fill(0);
    SecretString::from(hex_key)
}

/// Load or create the salt file for key derivation.
///
/// On first run, generates a cryptographically random salt and writes it
/// to `salt_path` with 0600 permissions. On subsequent runs, reads the existing salt.
///
/// # Errors
///
/// Returns `SecurityError::KeyfileError` if the salt file cannot be read or created.
pub fn load_or_create_salt(salt_path: &Path) -> Result<Vec<u8>, SecurityError> {
    use std::io::Write;

    // Create parent directory if needed (idempotent)
    if let Some(parent) = salt_path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| {
            SecurityError::KeyfileError(format!(
                "failed to create salt directory `{}`: {e}",
                parent.display()
            ))
        })?;
    }

    // Attempt atomic create — avoids TOCTOU race between exists() and write().
    // If the file already exists, `create_new` returns AlreadyExists and we
    // fall through to read the existing salt.
    match std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(salt_path)
    {
        Ok(mut file) => {
            let rng = SystemRandom::new();
            let mut salt = vec![0u8; SALT_LEN];
            rng.fill(&mut salt).map_err(|_| {
                SecurityError::KeyfileError("failed to generate random salt".into())
            })?;

            file.write_all(&salt).map_err(|e| {
                SecurityError::KeyfileError(format!(
                    "failed to write salt file `{}`: {e}",
                    salt_path.display()
                ))
            })?;

            // Set permissions to 0600 on Unix
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                let perms = std::fs::Permissions::from_mode(0o600);
                std::fs::set_permissions(salt_path, perms).map_err(|e| {
                    SecurityError::KeyfileError(format!("failed to set salt file permissions: {e}"))
                })?;
            }

            Ok(salt)
        }
        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
            // File already exists — read it
            std::fs::read(salt_path).map_err(|e| {
                SecurityError::KeyfileError(format!(
                    "failed to read salt file `{}`: {e}",
                    salt_path.display()
                ))
            })
        }
        Err(e) => Err(SecurityError::KeyfileError(format!(
            "failed to create salt file `{}`: {e}",
            salt_path.display()
        ))),
    }
}

/// Resolve the database passphrase from layered sources.
///
/// Priority: environment variable -> keyfile -> interactive prompt.
///
/// # Errors
///
/// Returns `SecurityError::NoEncryptionKey` if no source provides a key.
pub fn resolve_passphrase(
    keyfile_path: Option<&Path>,
    allow_prompt: bool,
) -> Result<SecretString, SecurityError> {
    // 1. Environment variable
    if let Ok(val) = std::env::var(ENV_VAR_NAME) {
        // SAFETY: Intentional removal of sensitive data from env.
        // The env var name is a static constant, not user-controlled.
        // `remove_var` is unsafe in edition 2024 due to potential data races,
        // but we only call this once during single-threaded startup.
        unsafe {
            std::env::remove_var(ENV_VAR_NAME);
        }
        return Ok(SecretString::from(val));
    }

    // 2. Keyfile
    if let Some(path) = keyfile_path {
        if path.exists() {
            validate_keyfile_permissions(path)?;
            let contents = std::fs::read_to_string(path).map_err(|e| {
                SecurityError::KeyfileError(format!(
                    "failed to read keyfile `{}`: {e}",
                    path.display()
                ))
            })?;
            let trimmed = contents.trim();
            if trimmed.is_empty() {
                return Err(SecurityError::KeyfileError(format!(
                    "keyfile `{}` is empty",
                    path.display()
                )));
            }
            return Ok(SecretString::from(trimmed.to_owned()));
        }
    }

    // 3. Interactive prompt
    if allow_prompt && std::io::IsTerminal::is_terminal(&std::io::stdin()) {
        return prompt_passphrase();
    }

    Err(SecurityError::NoEncryptionKey {
        message: format!(
            "No database key found. Set {ENV_VAR_NAME} env var, \
             create a keyfile, or run interactively."
        ),
    })
}

/// Validate that a keyfile has restrictive permissions (0600 on Unix).
#[cfg(unix)]
fn validate_keyfile_permissions(path: &Path) -> Result<(), SecurityError> {
    use std::os::unix::fs::PermissionsExt;
    let metadata = std::fs::metadata(path).map_err(|e| {
        SecurityError::KeyfileError(format!("failed to stat keyfile `{}`: {e}", path.display()))
    })?;
    let mode = metadata.permissions().mode() & 0o777;
    if mode != 0o600 {
        return Err(SecurityError::InsecureKeyfile {
            path: path.to_path_buf(),
            actual_mode: mode,
            required_mode: 0o600,
        });
    }
    Ok(())
}

#[cfg(not(unix))]
fn validate_keyfile_permissions(_path: &Path) -> Result<(), SecurityError> {
    tracing::warn!("keyfile permission check is not supported on this platform");
    Ok(())
}

/// Read a line from stdin with terminal echo disabled (Unix).
///
/// Uses a drop guard to ensure echo is always restored, even on error or panic.
#[cfg(unix)]
fn read_password_from_stdin() -> std::io::Result<String> {
    use std::os::unix::io::AsRawFd;

    // Drop guard that restores echo even if read_line panics or errors.
    // Defined before statements to satisfy clippy::items_after_statements.
    struct RestoreEcho {
        fd: libc::c_int,
        original: libc::termios,
    }
    impl Drop for RestoreEcho {
        fn drop(&mut self) {
            // SAFETY: restoring previously-saved terminal state on a valid fd
            let rc = unsafe { libc::tcsetattr(self.fd, libc::TCSANOW, &raw const self.original) };
            if rc != 0 {
                tracing::warn!("failed to restore terminal echo after password input");
            }
        }
    }

    let stdin_fd = std::io::stdin().as_raw_fd();

    let mut termios = std::mem::MaybeUninit::uninit();
    // SAFETY: tcgetattr is a standard POSIX API operating on a valid fd.
    if unsafe { libc::tcgetattr(stdin_fd, termios.as_mut_ptr()) } != 0 {
        return Err(std::io::Error::last_os_error());
    }
    // SAFETY: tcgetattr succeeded, so termios is fully initialized.
    let original = unsafe { termios.assume_init() };

    let _guard = RestoreEcho {
        fd: stdin_fd,
        original,
    };

    let mut noecho = original;
    noecho.c_lflag &= !libc::ECHO;
    // SAFETY: disabling ECHO flag via tcsetattr on a valid fd with valid termios.
    if unsafe { libc::tcsetattr(stdin_fd, libc::TCSANOW, &raw const noecho) } != 0 {
        return Err(std::io::Error::last_os_error());
    }

    let mut input = String::new();
    std::io::stdin().read_line(&mut input)?;
    // _guard dropped here → echo restored

    // Print newline since echo was disabled (user's Enter didn't show)
    eprintln!();

    Ok(input)
}

/// Fallback for non-Unix: reads with echo (warns user).
#[cfg(not(unix))]
fn read_password_from_stdin() -> std::io::Result<String> {
    tracing::warn!("password echo suppression is not supported on this platform");
    let mut input = String::new();
    std::io::stdin().read_line(&mut input)?;
    Ok(input)
}

/// Prompt the user for a passphrase via stdin.
fn prompt_passphrase() -> Result<SecretString, SecurityError> {
    use std::io::Write;
    eprint!("Enter database encryption passphrase: ");
    std::io::stderr()
        .flush()
        .map_err(|e| SecurityError::KeyfileError(format!("failed to flush stderr: {e}")))?;

    let mut input = read_password_from_stdin()
        .map_err(|e| SecurityError::KeyfileError(format!("failed to read passphrase: {e}")))?;

    let trimmed = input.trim();
    if trimmed.is_empty() {
        return Err(SecurityError::NoEncryptionKey {
            message: "empty passphrase provided".into(),
        });
    }

    let secret = SecretString::from(trimmed.to_owned());
    // Zeroize the input buffer
    input.clear();
    input.shrink_to_fit();
    Ok(secret)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, unsafe_code)]
mod tests {
    use super::*;

    #[test]
    fn test_derive_key_deterministic_with_same_salt() {
        let passphrase = SecretString::from("test-passphrase".to_owned());
        let salt = b"fixed-salt-for-testing-1234567890";
        let key1 = derive_key(&passphrase, salt, 1000);
        let key2 = derive_key(&passphrase, salt, 1000);
        assert_eq!(key1.expose_secret(), key2.expose_secret());
    }

    #[test]
    fn test_derive_key_different_salt_different_key() {
        let passphrase = SecretString::from("test-passphrase".to_owned());
        let salt1 = b"salt-aaaaaaaaaaaaaaaaaaaaaaaaaaaa";
        let salt2 = b"salt-bbbbbbbbbbbbbbbbbbbbbbbbbbbb";
        let key1 = derive_key(&passphrase, salt1, 1000);
        let key2 = derive_key(&passphrase, salt2, 1000);
        assert_ne!(key1.expose_secret(), key2.expose_secret());
    }

    #[test]
    fn test_derive_key_produces_64_char_hex() {
        let passphrase = SecretString::from("test".to_owned());
        let salt = b"12345678901234567890123456789012";
        let key = derive_key(&passphrase, salt, 1000);
        assert_eq!(key.expose_secret().len(), 64);
        assert!(key.expose_secret().chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn test_load_or_create_salt_creates_new() {
        let dir = tempfile::tempdir().unwrap();
        let salt_path = dir.path().join("db.salt");
        let salt = load_or_create_salt(&salt_path).unwrap();
        assert_eq!(salt.len(), SALT_LEN);
        assert!(salt_path.exists());
    }

    #[test]
    fn test_load_or_create_salt_reads_existing() {
        let dir = tempfile::tempdir().unwrap();
        let salt_path = dir.path().join("db.salt");
        let salt1 = load_or_create_salt(&salt_path).unwrap();
        let salt2 = load_or_create_salt(&salt_path).unwrap();
        assert_eq!(salt1, salt2);
    }

    #[cfg(unix)]
    #[test]
    fn test_validate_keyfile_permissions_rejects_world_readable() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().unwrap();
        let keyfile = dir.path().join("db.key");
        std::fs::write(&keyfile, "secret").unwrap();
        std::fs::set_permissions(&keyfile, std::fs::Permissions::from_mode(0o644)).unwrap();
        let result = validate_keyfile_permissions(&keyfile);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            SecurityError::InsecureKeyfile { .. }
        ));
    }

    #[cfg(unix)]
    #[test]
    fn test_validate_keyfile_permissions_accepts_0600() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().unwrap();
        let keyfile = dir.path().join("db.key");
        std::fs::write(&keyfile, "secret").unwrap();
        std::fs::set_permissions(&keyfile, std::fs::Permissions::from_mode(0o600)).unwrap();
        assert!(validate_keyfile_permissions(&keyfile).is_ok());
    }

    #[test]
    fn test_resolve_passphrase_returns_no_key_when_nothing_available() {
        // Ensure env var is not set
        // SAFETY: Test-only — no concurrent env var access in this test.
        unsafe { std::env::remove_var(ENV_VAR_NAME) };
        let result = resolve_passphrase(None, false);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            SecurityError::NoEncryptionKey { .. }
        ));
    }

    #[test]
    fn test_resolve_passphrase_empty_keyfile_errors() {
        let dir = tempfile::tempdir().unwrap();
        let keyfile = dir.path().join("db.key");
        std::fs::write(&keyfile, "").unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&keyfile, std::fs::Permissions::from_mode(0o600)).unwrap();
        }
        let result = resolve_passphrase(Some(&keyfile), false);
        assert!(result.is_err());
    }
}
