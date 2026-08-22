//! Incremental streaming extraction of JSON string fields from model output.

use super::reporter::Reporter;
use crate::constants::ansi::{ANSI_GREEN_CHECK, COLOR_CYAN, COLOR_RESET};
use crate::engine::InferenceBackend;
use crate::ollama::ChatRequest;
use anyhow::Result;
use futures_util::StreamExt;

/// Streams a model response, forwarding thought/answer text to the reporter,
/// and returns the complete raw JSON content plus whether the answer was streamed.
pub(super) async fn stream_model_response<R: Reporter>(
    backend: &dyn InferenceBackend,
    request: ChatRequest,
    reporter: &mut R,
    show_thinking: bool,
) -> Result<(String, bool)> {
    let mut stream = Box::pin(backend.chat_stream(request).await?);
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
}
