//! # NDJSON stream parsing for Ollama chat and model pull endpoints
//!
//! Ollama streams responses as newline-delimited JSON. Because HTTP chunks can
//! split a JSON line arbitrarily, the client keeps a string buffer and drains
//! only complete lines. This module owns that buffering/parsing logic so it can
//! be unit-tested without any network access.
//!
//! ## References
//!
//! - NDJSON specification: <https://ndjson.org/>
//! - Ollama streaming chat API: <https://github.com/ollama/ollama/blob/main/docs/api.md#generate-a-chat-completion>

use super::types::Delta;
use serde_json::Value;
use std::str::Utf8Error;

/// Text received so far from a byte stream, decoded as UTF-8.
///
/// A network chunk can end in the middle of a multi-byte character. Decoding
/// each chunk on its own turned both halves into U+FFFD, so an emoji or any
/// non-ASCII letter at a chunk boundary reached the user as `�`. The bytes of
/// an unfinished character wait in `pending` until the next chunk completes it.
#[derive(Debug, Default)]
pub(super) struct TextBuffer {
    /// Decoded text not yet consumed by a drain function.
    pub(super) text: String,
    /// The first bytes of a character whose remaining bytes have not arrived.
    pending: Vec<u8>,
}

impl TextBuffer {
    /// Append one chunk, decoding every complete character in it.
    pub(super) fn push(&mut self, bytes: &[u8]) {
        self.pending.extend_from_slice(bytes);

        let complete = decodable_len(&self.pending);
        self.text
            .push_str(&String::from_utf8_lossy(&self.pending[..complete]));
        self.pending.drain(..complete);
    }
}

/// How many leading bytes of `bytes` can be decoded now: all of them, except
/// a character cut off at the end. Invalid bytes count as decodable; they
/// become U+FFFD.
fn decodable_len(bytes: &[u8]) -> usize {
    std::str::from_utf8(bytes)
        .map_or_else(|error| decodable_len_before(error, bytes.len()), str::len)
}

/// [`decodable_len`] for bytes that are not valid UTF-8: up to the cut-off
/// character when the error is an unfinished character at the end, else all
/// `length` bytes.
fn decodable_len_before(error: Utf8Error, length: usize) -> usize {
    let is_cut_off_at_end = error.error_len().is_none();

    if is_cut_off_at_end {
        return error.valid_up_to();
    }

    length
}

/// Result of draining complete lines from a stream buffer.
pub(super) enum Drain {
    /// A delta is ready to emit.
    Delta(Delta),
    /// The stream's end marker was seen.
    Done,
    /// No complete payload line is available yet.
    NeedMore,
}

/// Remove the first complete line from `buffer` and return it trimmed;
/// `None` while no newline has arrived.
pub(super) fn take_line(buffer: &mut String) -> Option<String> {
    let newline = buffer.find('\n')?;
    let line = buffer[..newline].trim().to_string();

    buffer.drain(..=newline);

    Some(line)
}

/// Consume complete lines from `buffer`, returning the first meaningful event.
///
/// Lines after that event stay in `buffer` for the next call.
pub(super) fn drain_complete_lines(buffer: &mut String) -> Drain {
    std::iter::from_fn(|| take_line(buffer))
        .find_map(|line| parse_stream_line(&line))
        .unwrap_or(Drain::NeedMore)
}

/// Parse one NDJSON line from an Ollama chat stream.
///
/// Returns `None` for non-payload lines (progress, keep-alive, malformed JSON).
fn parse_stream_line(line: &str) -> Option<Drain> {
    let value = serde_json::from_str::<Value>(line).ok()?;

    message_content(&value).or_else(|| done_marker(&value))
}

/// The reply text of a chat stream line, as a content delta.
fn message_content(value: &Value) -> Option<Drain> {
    value
        .pointer("/message/content")
        .and_then(Value::as_str)
        .map(|content| Drain::Delta(Delta::Content(content.to_string())))
}

/// [`Drain::Done`] when a chat stream line carries `"done": true`.
fn done_marker(value: &Value) -> Option<Drain> {
    let is_done = value.get("done").and_then(Value::as_bool).unwrap_or(false);

    is_done.then_some(Drain::Done)
}

/// A meaningful event from an Ollama `/api/pull` NDJSON stream.
pub(super) enum PullLine {
    /// Human-readable status, possibly including download percentage.
    Status(String),
    /// The pull finished successfully.
    Done,
    /// The pull failed with a server-provided error message.
    Error(String),
}

/// Consume complete lines from a pull stream buffer, returning the first event.
pub(super) fn drain_pull_line(buffer: &mut String) -> Option<PullLine> {
    std::iter::from_fn(|| take_line(buffer)).find_map(|line| parse_pull_line(&line))
}

/// Parse one NDJSON line from an Ollama pull stream.
fn parse_pull_line(line: &str) -> Option<PullLine> {
    let value = serde_json::from_str::<Value>(line).ok()?;

    pull_error(&value).or_else(|| pull_status(&value))
}

/// The server's error message on a pull stream line.
fn pull_error(value: &Value) -> Option<PullLine> {
    value
        .get("error")
        .and_then(Value::as_str)
        .map(|error| PullLine::Error(error.to_string()))
}

/// The status on a pull stream line, if it has one.
fn pull_status(value: &Value) -> Option<PullLine> {
    value
        .get("status")
        .and_then(Value::as_str)
        .map(|status| status_line(status, value))
}

/// [`PullLine::Done`] for the `success` status, else the status with the
/// download percentage when the line reports one.
fn status_line(status: &str, value: &Value) -> PullLine {
    if status == "success" {
        return PullLine::Done;
    }

    PullLine::Status(format!("{status}{}", progress_suffix(value)))
}

/// ` 25%` when a pull line reports `completed` and a non-zero `total`, else
/// an empty string.
fn progress_suffix(value: &Value) -> String {
    let completed = value.get("completed").and_then(Value::as_u64);
    let total = value
        .get("total")
        .and_then(Value::as_u64)
        .filter(|total| *total > 0);

    completed
        .zip(total)
        .map_or_else(String::new, |(completed, total)| {
            format!(" {:.0}%", completed as f64 * 100.0 / total as f64)
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The reply text of a drained content delta, `None` for anything else.
    fn content(drain: Drain) -> Option<String> {
        match drain {
            Drain::Delta(Delta::Content(text)) => Some(text),
            Drain::Delta(_) | Drain::Done | Drain::NeedMore => None,
        }
    }

    /// The text of a pull status line, `None` for anything else.
    fn status(line: Option<PullLine>) -> Option<String> {
        match line {
            Some(PullLine::Status(text)) => Some(text),
            Some(PullLine::Done | PullLine::Error(_)) | None => None,
        }
    }

    #[test]
    fn a_character_split_across_chunks_is_decoded_whole() {
        let bytes = "bat 🦇 ünïcode".as_bytes();
        let mut buffer = TextBuffer::default();

        bytes.chunks(1).for_each(|chunk| buffer.push(chunk));

        assert_eq!(buffer.text, "bat 🦇 ünïcode");
    }

    #[test]
    fn invalid_bytes_still_decode_lossily() {
        let mut buffer = TextBuffer::default();

        buffer.push(&[b'a', 0xFF, b'b']);

        assert_eq!(buffer.text, "a\u{FFFD}b");
    }

    #[test]
    fn parses_content_delta_from_stream_line() {
        let line = r#"{"message":{"role":"assistant","content":"hello"},"done":false}"#;

        let parsed = parse_stream_line(line).and_then(content);

        assert_eq!(parsed.as_deref(), Some("hello"));
    }

    #[test]
    fn parses_done_marker_from_stream_line() {
        let line = r#"{"done":true}"#;

        assert!(matches!(parse_stream_line(line), Some(Drain::Done)));
    }

    #[test]
    fn ignores_malformed_stream_line() {
        assert!(parse_stream_line("not json").is_none());
    }

    #[test]
    fn ignores_keep_alive_and_progress_lines() {
        assert!(parse_stream_line(r#"{"keep_alive":true}"#).is_none());
        assert!(parse_stream_line(r#"{"status":"pulling manifest"}"#).is_none());
    }

    #[test]
    fn drains_multiple_lines_in_order() {
        let mut buffer = String::from(
            r#"{"message":{"content":"a"},"done":false}
{"message":{"content":"b"},"done":false}
{"done":true}
"#,
        );

        assert_eq!(
            content(drain_complete_lines(&mut buffer)).as_deref(),
            Some("a")
        );
        assert_eq!(
            content(drain_complete_lines(&mut buffer)).as_deref(),
            Some("b")
        );
        assert!(matches!(drain_complete_lines(&mut buffer), Drain::Done));
        assert!(matches!(drain_complete_lines(&mut buffer), Drain::NeedMore));
    }

    #[test]
    fn skips_blank_and_non_payload_lines() {
        let mut buffer = String::from("\n\nnot json\n{\"message\":{\"content\":\"ok\"}}\n");

        assert_eq!(
            content(drain_complete_lines(&mut buffer)).as_deref(),
            Some("ok")
        );
    }

    #[test]
    fn returns_need_more_for_partial_line() {
        let mut buffer = String::from(r#"{"message":{"content":"par"#);

        assert!(matches!(drain_complete_lines(&mut buffer), Drain::NeedMore));
        assert_eq!(content(Drain::Done), None);
    }

    #[test]
    fn parses_pull_status_with_progress_percentage() {
        let line = r#"{"status":"downloading","digest":"abc","total":100,"completed":25}"#;

        assert_eq!(
            status(parse_pull_line(line)).as_deref(),
            Some("downloading 25%")
        );
    }

    #[test]
    fn parses_pull_success_as_done() {
        let line = r#"{"status":"success"}"#;

        assert!(matches!(parse_pull_line(line), Some(PullLine::Done)));
    }

    #[test]
    fn parses_pull_error() {
        let line = r#"{"error":"model not found"}"#;

        assert!(matches!(
            parse_pull_line(line),
            Some(PullLine::Error(error)) if error == "model not found"
        ));
    }

    #[test]
    fn parses_pull_status_without_progress_fields() {
        let line = r#"{"status":"pulling manifest"}"#;
        let zero_total = r#"{"status":"verifying","total":0,"completed":0}"#;

        assert_eq!(
            status(parse_pull_line(line)).as_deref(),
            Some("pulling manifest")
        );
        assert_eq!(
            status(parse_pull_line(zero_total)).as_deref(),
            Some("verifying")
        );
    }

    #[test]
    fn drains_pull_lines_in_order_and_skips_garbage() {
        let mut buffer = String::from(
            "not json\n{\"status\":\"pulling manifest\"}\n{\"status\":\"downloading\",\"total\":10,\"completed\":5}\n{\"status\":\"success\"}\n",
        );

        assert_eq!(
            status(drain_pull_line(&mut buffer)).as_deref(),
            Some("pulling manifest")
        );
        assert_eq!(
            status(drain_pull_line(&mut buffer)).as_deref(),
            Some("downloading 50%")
        );
        assert!(matches!(drain_pull_line(&mut buffer), Some(PullLine::Done)));
        assert!(drain_pull_line(&mut buffer).is_none());
        assert_eq!(status(None), None);
    }
}
