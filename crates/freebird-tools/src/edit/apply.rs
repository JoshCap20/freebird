//! Diff application logic: string matching, replacement, and content modification.
//!
//! Contains the layered matching strategy (exact → whitespace-normalized → fuzzy)
//! and indentation-preserving replacement logic.

use freebird_traits::tool::ToolError;

/// Count the line number (1-indexed) where `needle` first appears in `haystack`.
pub(super) fn line_number_of_match(haystack: &str, byte_offset: usize) -> usize {
    haystack
        .get(..byte_offset)
        .unwrap_or(haystack)
        .chars()
        .filter(|&c| c == '\n')
        .count()
        + 1
}

/// Count how many lines `s` spans (at least 1 for non-empty, 0 for empty).
pub(super) fn line_count(s: &str) -> usize {
    if s.is_empty() {
        return 0;
    }
    s.chars().filter(|&c| c == '\n').count() + 1
}

/// Normalize a string for whitespace-tolerant matching:
/// - Trim leading/trailing whitespace per line
/// - Collapse runs of inline whitespace to a single space
/// - Preserve line boundaries
fn normalize_whitespace(s: &str) -> String {
    let mut result: String = s
        .lines()
        .map(|line| line.split_whitespace().collect::<Vec<_>>().join(" "))
        .collect::<Vec<_>>()
        .join("\n");
    // Preserve trailing newline so normalization is idempotent
    if s.ends_with('\n') {
        result.push('\n');
    }
    result
}

/// A line from the original haystack, remembering its byte offset.
struct SourceLine<'a> {
    text: &'a str,
    byte_offset: usize,
}

/// Split `haystack` into lines that remember their byte offsets.
/// Handles both `\n` and `\r\n`. Does NOT use `str::lines()` because that
/// erases terminator length information needed for correct byte-offset math.
///
/// `split_inclusive('\n')` returns the last segment even without a trailing
/// newline, so no special end-of-input handling is needed.
fn split_source_lines(haystack: &str) -> Vec<SourceLine<'_>> {
    if haystack.is_empty() {
        return vec![];
    }
    let mut lines = Vec::new();
    let mut offset = 0;
    for line_with_term in haystack.split_inclusive('\n') {
        let text = line_with_term.strip_suffix('\n').unwrap_or(line_with_term);
        let text = text.strip_suffix('\r').unwrap_or(text);
        lines.push(SourceLine {
            text,
            byte_offset: offset,
        });
        offset += line_with_term.len();
    }
    lines
}

/// Find all byte-offset positions where `needle` appears in `haystack` using
/// normalized (whitespace-collapsed) comparison. Returns the byte offsets and
/// byte lengths in the *original* haystack.
///
/// Correctly handles `\r\n` line endings by tracking actual terminator byte
/// lengths instead of assuming 1-byte `\n`.
fn find_normalized_matches(haystack: &str, needle: &str) -> Vec<(usize, usize)> {
    let norm_needle = normalize_whitespace(needle);
    if norm_needle.is_empty() {
        return vec![];
    }

    let needle_line_count = norm_needle.lines().count();
    let source_lines = split_source_lines(haystack);

    let mut matches = Vec::new();

    if source_lines.len() < needle_line_count {
        return matches;
    }

    for start_idx in 0..=source_lines.len() - needle_line_count {
        let window_slice = source_lines
            .get(start_idx..start_idx + needle_line_count)
            .unwrap_or(&[]);
        let window_texts: Vec<&str> = window_slice.iter().map(|sl| sl.text).collect();
        let window = window_texts.join("\n");
        let norm_window = normalize_whitespace(&window);

        if norm_window == norm_needle {
            let byte_start = window_slice.first().map_or(0, |sl| sl.byte_offset);
            // Total bytes from start of first matched line to end of last matched line's text
            // (excluding the last line's terminator, since we're replacing the text content).
            let byte_len: usize =
                if let (Some(first), Some(last)) = (window_slice.first(), window_slice.last()) {
                    (last.byte_offset + last.text.len()) - first.byte_offset
                } else {
                    0
                };
            matches.push((byte_start, byte_len));
        }
    }

    matches
}

/// Detect the indentation character used in a whitespace prefix.
/// Returns `'\t'` if tabs dominate, `' '` otherwise.
fn detect_indent_char(prefix: &str) -> char {
    let tabs = prefix.chars().filter(|&c| c == '\t').count();
    let spaces = prefix.chars().filter(|&c| c == ' ').count();
    if tabs > spaces { '\t' } else { ' ' }
}

/// Apply indentation preservation: detect the indentation delta between the
/// matched block and the replacement, then shift all replacement lines.
///
/// Tab-aware: detects whether the matched region uses tabs or spaces and
/// replicates the same character for relative indentation offsets.
fn apply_indentation(matched_first_line: &str, new_string: &str) -> String {
    if new_string.is_empty() {
        return String::new();
    }

    let match_indent = matched_first_line.len() - matched_first_line.trim_start().len();
    let new_lines: Vec<&str> = new_string.lines().collect();

    // Detect indentation of the first line of new_string
    let new_indent = new_lines
        .first()
        .map_or(0, |l| l.len() - l.trim_start().len());

    if match_indent == new_indent {
        return new_string.to_string();
    }

    let match_prefix = matched_first_line.get(..match_indent).unwrap_or("");
    let indent_char = detect_indent_char(match_prefix);

    new_lines
        .iter()
        .enumerate()
        .map(|(i, line)| {
            if i == 0 {
                // First line: replace its indentation with match indentation
                format!("{}{}", match_prefix, line.trim_start())
            } else if line.trim().is_empty() {
                // Preserve blank lines as-is
                String::new()
            } else {
                // Other lines: compute relative indent from first line and apply
                // using the same indent character as the matched region
                let line_indent = line.len() - line.trim_start().len();
                let relative = line_indent.saturating_sub(new_indent);
                let relative_str: String = std::iter::repeat_n(indent_char, relative).collect();
                format!("{}{}{}", match_prefix, relative_str, line.trim_start())
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Result of a successful match+replace operation.
pub(super) struct ReplaceResult {
    pub(super) content: String,
    pub(super) start_line: usize,
    pub(super) replaced_lines: usize,
    pub(super) matched_text: String,
    pub(super) adjusted_new: String,
}

/// Build a replaced string from the file content given a byte range and replacement.
fn build_replacement(
    file_content: &str,
    byte_start: usize,
    byte_len: usize,
    new_str: &str,
) -> ReplaceResult {
    let start_line = line_number_of_match(file_content, byte_start);
    let matched_region = file_content
        .get(byte_start..byte_start + byte_len)
        .unwrap_or("");
    let replaced_lines = line_count(matched_region);
    let first_matched_line = matched_region.lines().next().unwrap_or("");
    let adjusted_new = apply_indentation(first_matched_line, new_str);

    let mut result = String::with_capacity(file_content.len());
    result.push_str(file_content.get(..byte_start).unwrap_or(""));
    result.push_str(&adjusted_new);
    result.push_str(file_content.get(byte_start + byte_len..).unwrap_or(""));
    ReplaceResult {
        content: result,
        start_line,
        replaced_lines,
        matched_text: matched_region.to_string(),
        adjusted_new,
    }
}

/// Minimum fraction of lines that must match (after normalization) for a
/// fuzzy window to be considered a candidate. 0.6 allows up to 40% of lines
/// to differ, which handles common LLM errors (wrong variable names, slight
/// reformatting) while still being selective enough to avoid false positives.
const FUZZY_MATCH_THRESHOLD: f64 = 0.6;

/// Compute the fraction of lines in `needle_lines` that match `window_lines`
/// after whitespace normalization. Both slices must have the same length.
fn line_similarity(window_lines: &[&str], needle_lines: &[&str]) -> f64 {
    if window_lines.is_empty() {
        return 0.0;
    }
    let matching = window_lines
        .iter()
        .zip(needle_lines.iter())
        .filter(|(w, n)| normalize_whitespace(w) == normalize_whitespace(n))
        .count();
    #[expect(
        clippy::cast_precision_loss,
        reason = "line counts are small; f64 mantissa overflow is not a concern"
    )]
    {
        matching as f64 / window_lines.len() as f64
    }
}

/// Find the best fuzzy match: a window of N lines where at least
/// `FUZZY_MATCH_THRESHOLD` of lines match after normalization.
///
/// Returns `Some((byte_start, byte_len, score))` if exactly one window
/// achieves the best score above the threshold. Returns `None` if no window
/// passes the threshold, or if multiple windows tie for the best score
/// (ambiguous).
fn find_fuzzy_match(haystack: &str, needle: &str) -> Option<(usize, usize)> {
    let needle_lines: Vec<&str> = needle.lines().collect();
    if needle_lines.is_empty() {
        return None;
    }

    let source_lines = split_source_lines(haystack);
    if source_lines.len() < needle_lines.len() {
        return None;
    }

    let mut best_score: f64 = 0.0;
    let mut best_match: Option<(usize, usize)> = None;
    let mut best_is_unique = true;

    for start_idx in 0..=source_lines.len() - needle_lines.len() {
        let window_slice = source_lines
            .get(start_idx..start_idx + needle_lines.len())
            .unwrap_or(&[]);
        let window_texts: Vec<&str> = window_slice.iter().map(|sl| sl.text).collect();
        let score = line_similarity(&window_texts, &needle_lines);

        if score >= FUZZY_MATCH_THRESHOLD {
            if score > best_score {
                best_score = score;
                let byte_start = window_slice.first().map_or(0, |sl| sl.byte_offset);
                let byte_len = if let (Some(first), Some(last)) =
                    (window_slice.first(), window_slice.last())
                {
                    (last.byte_offset + last.text.len()) - first.byte_offset
                } else {
                    0
                };
                best_match = Some((byte_start, byte_len));
                best_is_unique = true;
            } else if (score - best_score).abs() < f64::EPSILON {
                // Tie — ambiguous, can't pick one
                best_is_unique = false;
            }
        }
    }

    if best_is_unique { best_match } else { None }
}

/// Find the replacement for the given old/new strings in the file content.
/// Returns the replaced file content and metadata, or a `ToolError`.
///
/// Matching strategy (layered, most precise first):
/// 1. **Exact match** — literal string comparison
/// 2. **Whitespace-normalized match** — collapse whitespace per line, compare
/// 3. **Fuzzy line-level match** — per-line similarity scoring with threshold
///
/// Each layer requires a unique match. Ambiguous matches (>1 candidate)
/// always return an error asking the agent to provide more context.
pub(super) fn find_and_replace(
    file_content: &str,
    old_str: &str,
    new_str: &str,
    relative_path: &str,
) -> Result<ReplaceResult, ToolError> {
    let exact_count = file_content.matches(old_str).count();
    match exact_count {
        1 => {
            let byte_offset =
                file_content
                    .find(old_str)
                    .ok_or_else(|| ToolError::ExecutionFailed {
                        tool: super::SearchReplaceEditTool::NAME.into(),
                        reason: "internal: exact match count was 1 but find returned None".into(),
                    })?;
            Ok(build_replacement(
                file_content,
                byte_offset,
                old_str.len(),
                new_str,
            ))
        }
        0 => {
            // Layer 2: whitespace-normalized match
            let norm_matches = find_normalized_matches(file_content, old_str);
            match norm_matches.len() {
                1 => {
                    if let Some(&(byte_start, byte_len)) = norm_matches.first() {
                        return Ok(build_replacement(
                            file_content,
                            byte_start,
                            byte_len,
                            new_str,
                        ));
                    }
                }
                n if n > 1 => {
                    return Err(ToolError::ExecutionFailed {
                        tool: super::SearchReplaceEditTool::NAME.into(),
                        reason: format!(
                            "old_string matches {n} locations in {relative_path} after whitespace normalization, provide more surrounding context"
                        ),
                    });
                }
                _ => {} // 0 normalized matches — fall through to fuzzy
            }

            // Layer 3: fuzzy line-level match
            if let Some((byte_start, byte_len)) = find_fuzzy_match(file_content, old_str) {
                return Ok(build_replacement(
                    file_content,
                    byte_start,
                    byte_len,
                    new_str,
                ));
            }

            Err(ToolError::ExecutionFailed {
                tool: super::SearchReplaceEditTool::NAME.into(),
                reason: format!("old_string not found in {relative_path}"),
            })
        }
        n => Err(ToolError::ExecutionFailed {
            tool: super::SearchReplaceEditTool::NAME.into(),
            reason: format!(
                "old_string matches {n} locations in {relative_path}, provide more surrounding context"
            ),
        }),
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic, clippy::indexing_slicing)]
mod tests {
    use super::*;

    // ── Unit tests for internal helpers ──────────────────────────

    #[test]
    fn test_split_source_lines_lf() {
        let lines = split_source_lines("abc\ndef\n");
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0].text, "abc");
        assert_eq!(lines[0].byte_offset, 0);
        assert_eq!(lines[1].text, "def");
        assert_eq!(lines[1].byte_offset, 4);
    }

    #[test]
    fn test_split_source_lines_crlf() {
        let lines = split_source_lines("abc\r\ndef\r\n");
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0].text, "abc");
        assert_eq!(lines[0].byte_offset, 0);
        assert_eq!(lines[1].text, "def");
        assert_eq!(lines[1].byte_offset, 5); // "abc\r\n" = 5 bytes
    }

    #[test]
    fn test_split_source_lines_no_trailing_newline() {
        let lines = split_source_lines("abc\ndef");
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0].text, "abc");
        assert_eq!(lines[1].text, "def");
        assert_eq!(lines[1].byte_offset, 4);
    }

    #[test]
    fn test_split_source_lines_empty() {
        let lines = split_source_lines("");
        assert!(lines.is_empty());
    }

    #[test]
    fn test_line_similarity_perfect() {
        let score = line_similarity(&["let x = 1;", "let y = 2;"], &["let x = 1;", "let y = 2;"]);
        assert!((score - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_line_similarity_partial() {
        let score = line_similarity(
            &["let x = 1;", "let y = 2;", "let z = 3;"],
            &["let x = 1;", "let WRONG = 2;", "let z = 3;"],
        );
        // 2 out of 3 match
        assert!((score - 2.0 / 3.0).abs() < 0.01);
    }

    #[test]
    fn test_line_similarity_none() {
        let score = line_similarity(&["aaa", "bbb"], &["xxx", "yyy"]);
        assert!((score - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_detect_indent_char_spaces() {
        assert_eq!(detect_indent_char("    "), ' ');
        assert_eq!(detect_indent_char("  "), ' ');
    }

    #[test]
    fn test_detect_indent_char_tabs() {
        assert_eq!(detect_indent_char("\t\t"), '\t');
        assert_eq!(detect_indent_char("\t"), '\t');
    }

    #[test]
    fn test_detect_indent_char_mixed_prefers_majority() {
        // More tabs than spaces → tab
        assert_eq!(detect_indent_char("\t\t "), '\t');
        // More spaces than tabs → space
        assert_eq!(detect_indent_char("\t  "), ' ');
    }

    // ── Property-based tests ─────────────────────────────────────

    mod proptests {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            /// Normalize whitespace is idempotent — normalizing twice
            /// gives the same result as normalizing once.
            #[test]
            fn normalize_whitespace_idempotent(input in "[ \\t\\n\\x20-\\x7E]{0,200}") {
                let once = normalize_whitespace(&input);
                let twice = normalize_whitespace(&once);
                prop_assert_eq!(once, twice);
            }

            /// split_source_lines byte offsets reconstruct original text.
            #[test]
            fn source_lines_cover_text(input in "[\\x20-\\x7E\\n]{1,200}") {
                let lines = split_source_lines(&input);
                for sl in &lines {
                    // Each line's text must be at its stated byte offset
                    let actual = input.get(sl.byte_offset..sl.byte_offset + sl.text.len());
                    prop_assert_eq!(actual, Some(sl.text),
                        "offset {} len {} doesn't match", sl.byte_offset, sl.text.len());
                }
            }
        }
    }
}
