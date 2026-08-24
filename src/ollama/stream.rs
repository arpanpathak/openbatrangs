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

use super::types::StreamLine;
use serde_json::Value;

/// Result of draining complete NDJSON lines from the stream buffer.
pub(super) enum LineDrain {
    /// A content delta is ready to emit.
    Content(String),
    /// The stream's terminal `done` marker was seen.
    Done,
    /// No complete payload line is available yet.
    NeedMore,
}

/// Consume complete lines from `buffer`, returning the first meaningful event.
pub(super) fn drain_complete_lines(buffer: &mut String) -> LineDrain {
    while let Some(newline_pos) = buffer.find('\n') {
        let line = buffer[..newline_pos].trim().to_string();
        *buffer = buffer[newline_pos + 1..].to_string();
        if line.is_empty() {
            continue;
        }
        match parse_stream_line(&line) {
            Some(StreamLine::Content(content)) => return LineDrain::Content(content),
            Some(StreamLine::Done) => return LineDrain::Done,
            None => {}
        }
    }
    LineDrain::NeedMore
}

/// Parse one NDJSON line from an Ollama chat stream.
///
/// Returns `None` for non-payload lines (progress, keep-alive, malformed JSON).
fn parse_stream_line(line: &str) -> Option<StreamLine> {
    let value = serde_json::from_str::<Value>(line).ok()?;
    if let Some(content) = value
        .pointer("/message/content")
        .and_then(|content| content.as_str())
    {
        return Some(StreamLine::Content(content.to_string()));
    }
    if value
        .get("done")
        .and_then(|done| done.as_bool())
        .unwrap_or(false)
    {
        return Some(StreamLine::Done);
    }
    None
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
    while let Some(newline_pos) = buffer.find('\n') {
        let line = buffer[..newline_pos].trim().to_string();
        *buffer = buffer[newline_pos + 1..].to_string();
        if line.is_empty() {
            continue;
        }
        if let Some(event) = parse_pull_line(&line) {
            return Some(event);
        }
    }
    None
}

/// Parse one NDJSON line from an Ollama pull stream.
fn parse_pull_line(line: &str) -> Option<PullLine> {
    let value = serde_json::from_str::<Value>(line).ok()?;

    if let Some(error) = value.get("error").and_then(|value| value.as_str()) {
        return Some(PullLine::Error(error.to_string()));
    }

    let status = value.get("status").and_then(|value| value.as_str())?;
    if status == "success" {
        return Some(PullLine::Done);
    }

    let progress = match (
        value.get("completed").and_then(|value| value.as_u64()),
        value.get("total").and_then(|value| value.as_u64()),
    ) {
        (Some(completed), Some(total)) if total > 0 => {
            format!(" {:.0}%", completed as f64 * 100.0 / total as f64)
        }
        _ => String::new(),
    };
    Some(PullLine::Status(format!("{status}{progress}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_content_delta_from_stream_line() {
        let line = r#"{"message":{"role":"assistant","content":"hello"},"done":false}"#;
        match parse_stream_line(line) {
            Some(StreamLine::Content(content)) => assert_eq!(content, "hello"),
            _ => panic!("expected content delta"),
        }
    }

    #[test]
    fn parses_done_marker_from_stream_line() {
        let line = r#"{"done":true}"#;
        assert!(matches!(parse_stream_line(line), Some(StreamLine::Done)));
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
        assert!(matches!(
            drain_complete_lines(&mut buffer),
            LineDrain::Content(content) if content == "a"
        ));
        assert!(matches!(
            drain_complete_lines(&mut buffer),
            LineDrain::Content(content) if content == "b"
        ));
        assert!(matches!(drain_complete_lines(&mut buffer), LineDrain::Done));
        assert!(matches!(
            drain_complete_lines(&mut buffer),
            LineDrain::NeedMore
        ));
    }

    #[test]
    fn skips_blank_and_non_payload_lines() {
        let mut buffer = String::from("\n\nnot json\n{\"message\":{\"content\":\"ok\"}}\n");
        assert!(matches!(
            drain_complete_lines(&mut buffer),
            LineDrain::Content(content) if content == "ok"
        ));
    }

    #[test]
    fn returns_need_more_for_partial_line() {
        let mut buffer = String::from(r#"{"message":{"content":"par"#);
        assert!(matches!(
            drain_complete_lines(&mut buffer),
            LineDrain::NeedMore
        ));
    }

    #[test]
    fn parses_pull_status_with_progress_percentage() {
        let line = r#"{"status":"downloading","digest":"abc","total":100,"completed":25}"#;
        match parse_pull_line(line) {
            Some(PullLine::Status(status)) => {
                assert!(status.starts_with("downloading"));
                assert!(status.contains("25%"));
            }
            _ => panic!("expected pull status with progress"),
        }
    }

    #[test]
    fn parses_pull_success_as_done() {
        let line = r#"{"status":"success"}"#;
        assert!(matches!(parse_pull_line(line), Some(PullLine::Done)));
    }

    #[test]
    fn parses_pull_error() {
        let line = r#"{"error":"model not found"}"#;
        match parse_pull_line(line) {
            Some(PullLine::Error(error)) => assert_eq!(error, "model not found"),
            _ => panic!("expected pull error"),
        }
    }

    #[test]
    fn parses_pull_status_without_progress_fields() {
        let line = r#"{"status":"pulling manifest"}"#;
        match parse_pull_line(line) {
            Some(PullLine::Status(status)) => assert_eq!(status, "pulling manifest"),
            _ => panic!("expected pull status"),
        }
    }

    #[test]
    fn drains_pull_lines_in_order_and_skips_garbage() {
        let mut buffer = String::from(
            "not json\n{\"status\":\"pulling manifest\"}\n{\"status\":\"downloading\",\"total\":10,\"completed\":5}\n{\"status\":\"success\"}\n",
        );
        assert!(matches!(
            drain_pull_line(&mut buffer),
            Some(PullLine::Status(status)) if status == "pulling manifest"
        ));
        assert!(matches!(
            drain_pull_line(&mut buffer),
            Some(PullLine::Status(status)) if status.starts_with("downloading") && status.contains("50%")
        ));
        assert!(matches!(drain_pull_line(&mut buffer), Some(PullLine::Done)));
        assert!(drain_pull_line(&mut buffer).is_none());
    }
}
