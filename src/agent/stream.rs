//! # Incremental JSON field streaming
//!
//! Local models are slow enough that waiting for a complete JSON response
//! before printing anything makes the TUI feel frozen. This module solves that
//! by scanning the partial JSON buffer on every token and extracting the two
//! string fields the agent schema cares about: `thought` and `answer`.
//!
//! The model emits a JSON string such as:
//!
//! ```json
//! {"answer":"Here is the code:\nfn main() {}"}
//! ```
//!
//! Inside JSON, a real newline is encoded as the two characters `\` and `n`.
//! If we printed those bytes verbatim the TUI would show literal `\n` instead
//! of a line break, which also breaks markdown/code-fence detection and syntax
//! highlighting. `unescape_json_string` converts JSON escapes back to real
//! characters as the stream arrives.
//!
//! ## References
//!
//! - JSON data model and string escapes, RFC 8259: <https://datatracker.ietf.org/doc/html/rfc8259#section-8.2>
//! - Ollama `/api/chat` streaming API: <https://github.com/ollama/ollama/blob/main/docs/api.md#generate-a-chat-completion>

use super::reporter::Reporter;
use crate::constants::ansi::{ANSI_GREEN_CHECK, COLOR_CYAN, COLOR_RESET};
use crate::ollama::{ChatRequest, OllamaClient};
use anyhow::Result;
use futures_util::StreamExt;

/// Streams a model response, forwarding thought/answer text to the reporter,
/// and returns the complete raw JSON content plus whether the answer was streamed.
///
/// # Parameters
///
/// - `client`: Ollama client used to open the chat stream.
/// - `request`: chat request; `stream` is forced to `true` by the client.
/// - `reporter`: receives streaming chunks and final lines.
/// - `show_thinking`: when `false`, the `thought` field is extracted but not
///   printed to the user.
///
/// # Returns
///
/// `(raw_json, answer_was_streamed)`.
pub(super) async fn stream_model_response<R: Reporter>(
    client: &OllamaClient,
    request: ChatRequest,
    reporter: &mut R,
    show_thinking: bool,
) -> Result<(String, bool)> {
    let mut stream = Box::pin(client.chat_stream(request).await?);
    let mut buffer = String::new();
    let mut thought = StreamState::new("thought", show_thinking);
    let mut answer = StreamState::new("answer", true);

    while let Some(delta) = stream.next().await {
        let delta = delta?;
        buffer.push_str(&delta);
        thought.feed(reporter, &buffer)?;
        answer.feed(reporter, &buffer)?;
    }

    let answer_was_streamed = answer.did_print();
    Ok((buffer.trim().to_string(), answer_was_streamed))
}

/// Lifecycle of a single JSON string field while its value is being streamed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FieldState {
    /// Still receiving characters; the closing quote has not been seen.
    Active,
    /// The field is hidden (thinking disabled) or shadowed by a tool call.
    Skipped,
    /// The closing quote was seen; no more text will arrive.
    Complete,
}

/// Tracks incremental extraction of a JSON string field from a streaming buffer.
pub(super) struct StreamState {
    /// JSON key being extracted (e.g. "thought" or "answer").
    key: &'static str,
    /// Current lifecycle state of this field.
    state: FieldState,
    /// Whether the visual prefix (emoji + label) has been emitted.
    has_printed_prefix: bool,
    /// Number of characters already forwarded to the reporter.
    printed_length: usize,
}

impl StreamState {
    /// Create a new stream tracker for the given JSON key.
    ///
    /// # Parameters
    ///
    /// - `key`: the JSON field name to extract (e.g. "thought", "answer").
    /// - `enabled`: when `false` and key is "thought", output is suppressed.
    pub(super) fn new(key: &'static str, enabled: bool) -> Self {
        let state = if key == "thought" && !enabled {
            FieldState::Skipped
        } else {
            FieldState::Active
        };
        Self {
            key,
            state,
            has_printed_prefix: false,
            printed_length: 0,
        }
    }

    /// Feed a new buffer snapshot and forward any new text to the reporter.
    ///
    /// # Parameters
    ///
    /// - `reporter`: output sink for streaming chunks.
    /// - `buffer`: the accumulated raw JSON response so far.
    pub(super) fn feed<R: Reporter>(&mut self, reporter: &mut R, buffer: &str) -> Result<()> {
        match self.state {
            FieldState::Skipped | FieldState::Complete => return Ok(()),
            FieldState::Active => {}
        }

        // With depth-aware key finding, extract_json_string only matches
        // top-level keys, so nested keys inside tool arguments are already
        // excluded. The only remaining case to guard against is when the
        // response contains a `"tool"` key at the top level but the field
        // we're looking for hasn't appeared yet — in that case the model
        // is calling a tool and won't produce an `answer` field.
        if find_toplevel_key(buffer, "tool").is_some()
            && find_toplevel_key(buffer, self.key).is_none()
        {
            self.state = FieldState::Skipped;
            return Ok(());
        }

        if let Some((text, is_complete)) = extract_json_string(buffer, self.key) {
            let text = unescape_json_string(&text);
            if !self.has_printed_prefix && !text.is_empty() {
                reporter.chunk(&self.prefix());
                self.has_printed_prefix = true;
            }
            if text.len() > self.printed_length && text.is_char_boundary(self.printed_length) {
                reporter.chunk(&text[self.printed_length..]);
                self.printed_length = text.len();
            }
            if is_complete {
                self.state = FieldState::Complete;
                reporter.line(String::new());
            }
        }
        Ok(())
    }

    /// Visual prefix (emoji + label) for this field's output.
    fn prefix(&self) -> String {
        match self.key {
            "thought" => format!("{COLOR_CYAN}🧠 {COLOR_RESET}"),
            "answer" => format!("{ANSI_GREEN_CHECK} "),
            _ => String::new(),
        }
    }

    /// Whether any text was forwarded to the reporter for this field.
    pub(super) fn did_print(&self) -> bool {
        self.has_printed_prefix
    }
}

/// Find the position of a key at the top level of a (possibly partial) JSON object.
///
/// Tracks brace depth and string boundaries so that a key like `"answer"`
/// nested inside `"tool": {"arguments": {"answer": ...}}` is skipped.
/// Only matches keys at depth 1 (direct children of the root object).
fn find_toplevel_key(buffer: &str, key: &str) -> Option<usize> {
    let key_pattern = format!("\"{key}\"");
    let mut depth: usize = 0;
    let mut in_string = false;
    let mut escaped = false;
    let bytes = buffer.as_bytes();
    let mut index = 0;

    while index < bytes.len() {
        let byte = bytes[index];

        if escaped {
            escaped = false;
            index += 1;
            continue;
        }

        if in_string {
            match byte {
                b'\\' => escaped = true,
                b'"' => in_string = false,
                _ => {}
            }
            index += 1;
            continue;
        }

        match byte {
            b'"' => {
                if buffer[index..].starts_with(&key_pattern) && depth == 1 {
                    return Some(index);
                }
                in_string = true;
            }
            b'{' => depth += 1,
            b'}' => depth = depth.saturating_sub(1),
            _ => {}
        }
        index += 1;
    }
    None
}

/// Extract the current value of a JSON string field from a partial JSON buffer.
///
/// Only matches the key at the top level of the JSON object (depth 1), so
/// nested keys inside tool arguments are never confused with the real
/// `thought` or `answer` fields.
///
/// Returns `(value_so_far, is_complete)`.
fn extract_json_string(buffer: &str, key: &str) -> Option<(String, bool)> {
    let start = find_toplevel_key(buffer, key)?;
    let key_pattern_len = key.len() + 2; // `"key"`
    let after_key = &buffer[start + key_pattern_len..];
    let after_key = after_key.trim_start();
    let after_key = after_key.strip_prefix(':')?.trim_start();
    let after_key = after_key.strip_prefix('"')?;

    let mut out = String::new();
    let mut complete = false;
    let mut chars = after_key.chars();
    while let Some(c) = chars.next() {
        match c {
            '\\' => {
                if let Some(n) = chars.next() {
                    out.push('\\');
                    out.push(n);
                }
            }
            '"' => {
                complete = true;
                break;
            }
            _ => out.push(c),
        }
    }
    Some((out, complete))
}

/// Unescape a JSON string value for terminal display.
///
/// The raw JSON keeps escapes like `\n` as two characters; without this the
/// TUI would render literal `\n` instead of actual newlines, which also breaks
/// markdown/code-fence detection and syntax highlighting.
///
/// A trailing backslash or an incomplete `\uXXXX` escape is left pending so
/// incremental streaming does not print a half-escape.
fn unescape_json_string(raw: &str) -> String {
    let mut out = String::new();
    let mut chars = raw.chars();
    while let Some(character) = chars.next() {
        if character != '\\' {
            out.push(character);
            continue;
        }
        match chars.next() {
            Some('n') => out.push('\n'),
            Some('r') => out.push('\r'),
            Some('t') => out.push('\t'),
            Some('\\') => out.push('\\'),
            Some('"') => out.push('"'),
            Some('/') => out.push('/'),
            Some('b') => out.push('\u{0008}'),
            Some('f') => out.push('\u{000C}'),
            Some('u') => {
                let hex: String = chars.clone().take(4).collect();
                if hex.len() == 4 && hex.chars().all(|digit| digit.is_ascii_hexdigit()) {
                    if let Ok(code) = u32::from_str_radix(&hex, 16) {
                        if let Some(character) = char::from_u32(code) {
                            out.push(character);
                        }
                    }
                    for _ in 0..4 {
                        let _ = chars.next();
                    }
                } else {
                    // Incomplete unicode escape; wait for the next chunk.
                    break;
                }
            }
            Some(other) => {
                // Unknown escape: preserve it literally rather than dropping data.
                out.push('\\');
                out.push(other);
            }
            None => break,
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_json_string_incrementally() {
        let key = "answer";
        assert_eq!(
            extract_json_string(r#"{"answer":"hel"#, key),
            Some(("hel".to_string(), false))
        );
        assert_eq!(
            extract_json_string(r#"{"answer":"hello"}"#, key),
            Some(("hello".to_string(), true))
        );
    }

    #[test]
    fn stream_state_skips_when_disabled() {
        let mut state = StreamState::new("thought", false);
        assert_eq!(state.state, FieldState::Skipped);
        state
            .feed(
                &mut super::super::reporter::StdoutReporter,
                r#"{"thought":"secret"}"#,
            )
            .unwrap();
        assert!(!state.did_print());
    }

    #[test]
    fn stream_state_streams_incrementally() {
        struct Sink(Vec<String>);
        impl Reporter for Sink {
            fn line(&mut self, msg: String) {
                self.0.push(msg);
            }
            fn chunk(&mut self, msg: &str) {
                self.0.push(msg.to_string());
            }
        }
        let mut state = StreamState::new("answer", true);
        let mut sink = Sink(Vec::new());
        state.feed(&mut sink, r#"{"answer":"hel"#).unwrap();
        state.feed(&mut sink, r#"{"answer":"hello"}"#).unwrap();
        assert!(sink.0.iter().any(|part| part.contains("hel")));
        assert!(sink.0.iter().any(|part| part.contains("lo")));
    }

    #[test]
    fn stream_state_renders_escaped_newlines_as_real_newlines() {
        struct Sink(Vec<String>);
        impl Reporter for Sink {
            fn line(&mut self, msg: String) {
                self.0.push(msg);
            }
            fn chunk(&mut self, msg: &str) {
                self.0.push(msg.to_string());
            }
        }
        let mut state = StreamState::new("answer", true);
        let mut sink = Sink(Vec::new());
        state
            .feed(&mut sink, r#"{"answer":"line1\nline2"}"#)
            .unwrap();
        assert!(
            sink.0.iter().any(|part| part.contains('\n')),
            "escaped \\n must be rendered as a real newline"
        );
        assert!(
            !sink.0.iter().any(|part| part.contains("\\n")),
            "literal \\n must not appear in streamed output"
        );
    }

    #[test]
    fn unescape_json_string_converts_common_escapes() {
        assert_eq!(unescape_json_string(r#"hello\nworld"#), "hello\nworld");
        assert_eq!(unescape_json_string(r#"tab\there"#), "tab\there");
        assert_eq!(unescape_json_string(r#"quote\"x"#), "quote\"x");
        assert_eq!(unescape_json_string(r#"back\\slash"#), "back\\slash");
        assert_eq!(unescape_json_string(r#"unicode\u0041"#), "unicodeA");
    }

    #[test]
    fn find_toplevel_key_skips_nested_keys() {
        // "answer" inside tool.arguments must NOT match at top level.
        let buffer = r#"{"thought":"thinking","tool":{"name":"write_file","arguments":{"path":"foo","content":"answer"}}}"#;
        assert!(
            find_toplevel_key(buffer, "answer").is_none(),
            "nested 'answer' inside tool.arguments must not match"
        );
        assert!(find_toplevel_key(buffer, "thought").is_some());
        assert!(find_toplevel_key(buffer, "tool").is_some());
    }

    #[test]
    fn find_toplevel_key_matches_top_level() {
        let buffer = r#"{"thought":"hi","answer":"hello"}"#;
        assert!(find_toplevel_key(buffer, "answer").is_some());
        assert!(find_toplevel_key(buffer, "thought").is_some());
    }

    #[test]
    fn find_toplevel_key_handles_key_inside_string_value() {
        // The string value contains the literal text "answer" — must not match.
        let buffer = r#"{"thought":"the answer is 42","tool":{"name":"finish"}}"#;
        assert!(
            find_toplevel_key(buffer, "answer").is_none(),
            "key inside a string value must not match"
        );
    }

    #[test]
    fn extract_json_string_ignores_nested_answer() {
        let buffer = r#"{"thought":"calling tool","tool":{"name":"write_file","arguments":{"answer":"nested"}}}"#;
        assert!(
            extract_json_string(buffer, "answer").is_none(),
            "nested 'answer' inside tool.arguments must not be extracted"
        );
    }

    #[test]
    fn extract_json_string_handles_partial_with_tool() {
        // Partial buffer: tool key seen but answer hasn't arrived yet.
        let buffer = r#"{"thought":"thinking","tool":{"name":"read_file""#;
        assert!(extract_json_string(buffer, "answer").is_none());
    }

    #[test]
    fn unescape_json_string_keeps_incomplete_escape_pending() {
        assert_eq!(unescape_json_string("abc\\"), "abc");
        assert_eq!(unescape_json_string("abc\\u12"), "abc");
        assert_eq!(unescape_json_string("abc\\u1234"), "abc\u{1234}");
    }
}
