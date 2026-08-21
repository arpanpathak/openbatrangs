//! NDJSON stream parsing for `POST /api/chat`.

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
}
