//! Raw mode writer — translates `\n` to `\r\n` for crossterm raw mode.
//!
//! In raw mode, the terminal does not perform automatic CR+LF translation.
//! A bare `\n` only moves the cursor down without returning to column 0,
//! causing text to appear at scattered positions. This wrapper intercepts
//! all writes and ensures every `\n` is preceded by `\r`.

use std::io::{self, Stdout, Write};

/// A writer that translates lone `\n` bytes to `\r\n` sequences.
///
/// Wraps `Stdout` and intercepts all `write` calls to ensure correct
/// line endings in crossterm raw mode. Already-correct `\r\n` sequences
/// are passed through unchanged, even when split across multiple writes.
pub struct RawWriter {
    inner: Stdout,
    /// Last byte written — used to detect `\r\n` split across write calls.
    last_byte: Option<u8>,
}

impl RawWriter {
    /// Create a new raw mode writer wrapping stdout.
    pub fn new() -> Self {
        Self {
            inner: io::stdout(),
            last_byte: None,
        }
    }
}

impl Write for RawWriter {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        if buf.is_empty() {
            return Ok(0);
        }
        // Scan for lone \n (not preceded by \r) and replace with \r\n.
        // We process the entire buffer to maintain correct byte count reporting.
        let mut last = 0;
        for (i, &byte) in buf.iter().enumerate() {
            if byte == b'\n' {
                // Check the preceding byte: either in this buffer or from the last write.
                let prev = if i > 0 {
                    buf.get(i - 1).copied()
                } else {
                    self.last_byte
                };
                if prev != Some(b'\r') {
                    // Write everything up to and not including the \n
                    if let Some(chunk) = buf.get(last..i) {
                        if !chunk.is_empty() {
                            self.inner.write_all(chunk)?;
                        }
                    }
                    // Write \r\n instead
                    self.inner.write_all(b"\r\n")?;
                    last = i + 1;
                }
            }
        }
        // Write any remaining bytes
        if let Some(remainder) = buf.get(last..) {
            if !remainder.is_empty() {
                self.inner.write_all(remainder)?;
            }
        }
        // Track the last byte for cross-write \r\n detection
        self.last_byte = buf.last().copied();
        // Report the original buffer length as consumed
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        self.inner.flush()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::indexing_slicing)]
mod tests {
    use super::*;

    /// Test helper that uses a Vec instead of Stdout, with cross-write tracking.
    struct TestRawWriter {
        buf: Vec<u8>,
        last_byte: Option<u8>,
    }

    impl TestRawWriter {
        fn new() -> Self {
            Self {
                buf: Vec::new(),
                last_byte: None,
            }
        }
    }

    impl Write for TestRawWriter {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            if buf.is_empty() {
                return Ok(0);
            }
            let mut last = 0;
            for (i, &byte) in buf.iter().enumerate() {
                if byte == b'\n' {
                    let prev = if i > 0 {
                        buf.get(i - 1).copied()
                    } else {
                        self.last_byte
                    };
                    if prev != Some(b'\r') {
                        if i > last {
                            self.buf.extend_from_slice(&buf[last..i]);
                        }
                        self.buf.extend_from_slice(b"\r\n");
                        last = i + 1;
                    }
                }
            }
            if last < buf.len() {
                self.buf.extend_from_slice(&buf[last..]);
            }
            self.last_byte = buf.last().copied();
            Ok(buf.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    #[test]
    fn lone_newline_becomes_crlf() {
        let mut w = TestRawWriter::new();
        w.write_all(b"hello\nworld").unwrap();
        assert_eq!(&w.buf, b"hello\r\nworld");
    }

    #[test]
    fn existing_crlf_preserved() {
        let mut w = TestRawWriter::new();
        w.write_all(b"hello\r\nworld").unwrap();
        assert_eq!(&w.buf, b"hello\r\nworld");
    }

    #[test]
    fn multiple_newlines() {
        let mut w = TestRawWriter::new();
        w.write_all(b"a\nb\nc\n").unwrap();
        assert_eq!(&w.buf, b"a\r\nb\r\nc\r\n");
    }

    #[test]
    fn no_newlines_passthrough() {
        let mut w = TestRawWriter::new();
        w.write_all(b"hello world").unwrap();
        assert_eq!(&w.buf, b"hello world");
    }

    #[test]
    fn empty_write() {
        let mut w = TestRawWriter::new();
        w.write_all(b"").unwrap();
        assert!(w.buf.is_empty());
    }

    #[test]
    fn leading_newline() {
        let mut w = TestRawWriter::new();
        w.write_all(b"\nhello").unwrap();
        assert_eq!(&w.buf, b"\r\nhello");
    }

    #[test]
    fn writeln_macro_works() {
        let mut w = TestRawWriter::new();
        writeln!(w, "hello").unwrap();
        assert_eq!(&w.buf, b"hello\r\n");
    }

    #[test]
    fn mixed_crlf_and_lf() {
        let mut w = TestRawWriter::new();
        w.write_all(b"a\r\nb\nc\r\n").unwrap();
        assert_eq!(&w.buf, b"a\r\nb\r\nc\r\n");
    }

    #[test]
    fn crlf_split_across_writes_not_doubled() {
        let mut w = TestRawWriter::new();
        w.write_all(b"line1\r").unwrap();
        w.write_all(b"\nline2").unwrap();
        assert_eq!(&w.buf, b"line1\r\nline2");
    }

    #[test]
    fn lone_lf_split_across_writes_gets_cr() {
        let mut w = TestRawWriter::new();
        w.write_all(b"line1").unwrap();
        w.write_all(b"\nline2").unwrap();
        assert_eq!(&w.buf, b"line1\r\nline2");
    }
}
