//! # Output truncation helpers
//!
//! Tool results are fed back into the model's context window, which is small on
//! Jetson-class devices. These helpers cap results at
//! [`MAX_TOOL_OUTPUT`](crate::constants::tools::MAX_TOOL_OUTPUT) characters and
//! append an omission note so the model knows the output was truncated.
//!
//! ## References
//!
//! - Context window limits in LLMs: <https://en.wikipedia.org/wiki/Large_language_model#Context_window>
//! - Rust `str::chars` for Unicode-safe truncation: <https://doc.rust-lang.org/std/primitive.str.html#method.chars>

use crate::constants::tools::MAX_TOOL_OUTPUT;

/// Truncate a string to `MAX_TOOL_OUTPUT` characters.
pub(crate) fn truncate(text: String) -> String {
    truncate_to(text, MAX_TOOL_OUTPUT)
}

/// Truncate a string to at most `max` characters, appending an omission note.
///
/// # Parameters
///
/// - `text`: string to truncate.
/// - `max`: maximum number of Unicode characters to keep.
///
/// # Returns
///
/// The original string when it fits, otherwise a truncated string ending with
/// `... (truncated, N chars omitted)`.
pub(super) fn truncate_to(text: String, max: usize) -> String {
    let char_count = text.chars().count();

    if char_count <= max {
        return text;
    }

    let mut kept = text.chars().take(max).collect::<String>();
    kept.push_str(&format!(
        "\n... (truncated, {} chars omitted)",
        char_count - max
    ));

    kept
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn truncate_appends_omission_note() {
        let text = "x".repeat(MAX_TOOL_OUTPUT + 100);
        let shortened = truncate(text);
        assert!(shortened.len() < MAX_TOOL_OUTPUT + 200);
        assert!(shortened.contains("truncated"));
    }

    #[test]
    fn short_text_is_unchanged() {
        let text = "hello".to_string();
        assert_eq!(truncate(text), "hello");
    }

    #[test]
    fn truncate_to_respects_custom_limit() {
        let text = "abcdefgh".to_string();
        let shortened = truncate_to(text, 4);
        assert!(shortened.starts_with("abcd"));
        assert!(shortened.contains("truncated"));
    }

    #[test]
    fn multi_byte_text_within_the_limit_is_unchanged() {
        let text = "😀😀".to_string();

        assert_eq!(truncate_to(text, 2), "😀😀");
    }

    #[test]
    fn truncate_handles_unicode_char_boundaries() {
        let text = "😀😀😀😀".to_string();
        let shortened = truncate_to(text, 2);
        assert_eq!(shortened.chars().next(), Some('😀'));
        assert!(shortened.contains("truncated"));
    }
}
