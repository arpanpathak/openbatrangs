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

/// Tracks incremental extraction of a JSON string field from a streaming buffer.
pub(super) struct StreamState {
    key: &'static str,
    has_printed_prefix: bool,
    printed_length: usize,
    is_complete: bool,
    is_skipped: bool,
}

impl StreamState {
    pub(super) fn new(key: &'static str, enabled: bool) -> Self {
        Self {
            key,
            has_printed_prefix: false,
            printed_length: 0,
            is_complete: false,
            is_skipped: key == "thought" && !enabled,
        }
    }

    pub(super) fn feed<R: Reporter>(&mut self, reporter: &mut R, buffer: &str) -> Result<()> {
        if self.is_complete || self.is_skipped {
            return Ok(());
        }

        // If a top-level tool call exists before this field, don't stream it —
        // it might be text inside tool arguments rather than the real field.
        if let (Some(field_pos), Some(tool_pos)) =
            (find_key_pos(buffer, self.key), find_key_pos(buffer, "tool"))
        {
            if tool_pos < field_pos {
                self.is_skipped = true;
                return Ok(());
            }
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
                self.is_complete = true;
                reporter.line(String::new());
            }
        }
        Ok(())
    }

    fn prefix(&self) -> String {
        match self.key {
            "thought" => format!("{COLOR_CYAN}🧠 {COLOR_RESET}"),
            "answer" => format!("{ANSI_GREEN_CHECK} "),
            _ => String::new(),
        }
    }

    pub(super) fn did_print(&self) -> bool {
        self.has_printed_prefix
    }
}

fn find_key_pos(buffer: &str, key: &str) -> Option<usize> {
    buffer.find(&format!("\"{key}\""))
}

/// Extract the current value of a JSON string field from a partial JSON buffer.
/// Returns `(value_so_far, is_complete)`.
fn extract_json_string(buffer: &str, key: &str) -> Option<(String, bool)> {
    let key_pattern = format!("\"{key}\"");
    let start = buffer.find(&key_pattern)?;
    let after_key = &buffer[start + key_pattern.len()..];
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
        assert!(state.is_skipped);
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
    fn unescape_json_string_keeps_incomplete_escape_pending() {
        assert_eq!(unescape_json_string("abc\\"), "abc");
        assert_eq!(unescape_json_string("abc\\u12"), "abc");
        assert_eq!(unescape_json_string("abc\\u1234"), "abc\u{1234}");
    }
}
