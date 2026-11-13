//! # Incremental JSON field streaming
//!
//! Local models are slow enough that waiting for a complete JSON response
//! before printing anything makes the TUI feel frozen. This module solves that
//! by scanning the partial JSON buffer on every token and extracting the two
//! string fields of the agent schema that are shown to the user: `thought`
//! and `answer`.
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

use super::reporter::{AgentEvent, Reporter, TextKind};
use crate::ollama::{ChatRequest, Delta, OllamaClient};
use anyhow::Result;
use futures_util::{Stream, TryStreamExt};

/// Streams a model response, forwarding thought/answer text to the reporter,
/// and returns the complete raw JSON content plus whether the answer was streamed.
///
/// # Parameters
///
/// - `client`: Ollama client used to open the chat stream.
/// - `request`: chat request; `stream` is forced to `true` by the client.
/// - `reporter`: receives streaming chunks and final lines.
/// - `show_thinking`: when `false`, the `thought` field and the model's own
///   reasoning (thinking models) are not printed to the user.
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
    let stream = client.chat_stream(request).await?;

    print_response(stream, reporter, show_thinking).await
}

/// Print every delta of `stream` as it arrives and return the raw JSON
/// content plus whether the answer was streamed.
async fn print_response<R: Reporter>(
    stream: impl Stream<Item = Result<Delta>>,
    reporter: &mut R,
    show_thinking: bool,
) -> Result<(String, bool)> {
    let printer = ResponsePrinter::new(show_thinking);

    let printer = stream
        .try_fold(printer, |mut printer, delta| {
            printer.take(delta, reporter);
            std::future::ready(Ok(printer))
        })
        .await?;

    Ok(printer.finish(reporter))
}

/// The raw text of one JSON-loop response and the state of its live output.
struct ResponsePrinter {
    /// Raw JSON received so far.
    buffer: String,
    /// Streaming state of the `thought` field.
    thought: StreamState,
    /// Streaming state of the `answer` field.
    answer: StreamState,
    /// Whether reasoning from a thinking model is printed.
    show_thinking: bool,
    /// Whether reasoning text is being printed right now.
    is_reasoning: bool,
}

impl ResponsePrinter {
    /// A printer that has received nothing yet.
    fn new(show_thinking: bool) -> Self {
        Self {
            buffer: String::new(),
            thought: StreamState::new("thought", TextKind::Thought, show_thinking),
            answer: StreamState::new("answer", TextKind::Answer, true),
            show_thinking,
            is_reasoning: false,
        }
    }

    /// Print one delta. Tool-call deltas never reach the JSON loop.
    fn take<R: Reporter>(&mut self, delta: Delta, reporter: &mut R) {
        match delta {
            Delta::Thinking(text) => self.reason(&text, reporter),
            Delta::Content(text) => self.content(&text, reporter),
            Delta::ToolCalls(..) => {}
        }
    }

    /// Print reasoning text, when reasoning is shown.
    fn reason<R: Reporter>(&mut self, text: &str, reporter: &mut R) {
        if !self.show_thinking {
            return;
        }

        self.is_reasoning = true;
        reporter.report(AgentEvent::Text {
            kind: TextKind::Reasoning,
            text: text.to_string(),
        });
    }

    /// End the reasoning block, if one is open.
    fn end_reasoning<R: Reporter>(&mut self, reporter: &mut R) {
        if !self.is_reasoning {
            return;
        }

        reporter.report(AgentEvent::TextEnded(TextKind::Reasoning));
        self.is_reasoning = false;
    }

    /// Add reply text to the buffer and print the new part of each field.
    fn content<R: Reporter>(&mut self, text: &str, reporter: &mut R) {
        self.end_reasoning(reporter);
        self.buffer.push_str(text);

        self.thought.feed(reporter, &self.buffer);
        self.answer.feed(reporter, &self.buffer);
    }

    /// Close any open reasoning block and return the trimmed raw JSON plus
    /// whether the answer was printed.
    fn finish<R: Reporter>(mut self, reporter: &mut R) -> (String, bool) {
        self.end_reasoning(reporter);

        (self.buffer.trim().to_string(), self.answer.did_print())
    }
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
    /// What the field's text is, for the reporter.
    kind: TextKind,
    /// Current lifecycle state of this field.
    state: FieldState,
    /// Number of bytes of the field's text already forwarded to the reporter.
    printed_length: usize,
}

impl StreamState {
    /// Create a new stream tracker for the given JSON key.
    ///
    /// # Parameters
    ///
    /// - `key`: the JSON field name to extract (e.g. "thought", "answer").
    /// - `kind`: what the field's text is.
    /// - `enabled`: when `false`, the field is not forwarded at all.
    pub(super) fn new(key: &'static str, kind: TextKind, enabled: bool) -> Self {
        let state = match enabled {
            true => FieldState::Active,
            false => FieldState::Skipped,
        };

        Self {
            key,
            kind,
            state,
            printed_length: 0,
        }
    }

    /// Feed a new buffer snapshot and forward any new text to the reporter.
    ///
    /// # Parameters
    ///
    /// - `reporter`: output sink for streaming chunks.
    /// - `buffer`: the accumulated raw JSON response so far.
    pub(super) fn feed<R: Reporter>(&mut self, reporter: &mut R, buffer: &str) {
        if self.state != FieldState::Active {
            return;
        }

        self.feed_active(reporter, buffer);
    }

    /// Feed a field that is still receiving text.
    ///
    /// Key finding only matches top-level keys, so keys nested inside tool
    /// arguments never match. A top-level `"tool"` key before this field
    /// means the reply is a tool call, which has no `answer` or `thought`
    /// text to print.
    fn feed_active<R: Reporter>(&mut self, reporter: &mut R, buffer: &str) {
        if tool_call_hides(buffer, self.key) {
            self.state = FieldState::Skipped;
            return;
        }

        self.print_field(reporter, buffer);
    }

    /// Print the part of the field's value that has not been printed yet.
    fn print_field<R: Reporter>(&mut self, reporter: &mut R, buffer: &str) {
        let Some((raw_text, is_complete)) = extract_json_string(buffer, self.key) else {
            return;
        };

        let text = unescape_json_string(&raw_text);
        self.print_new_text(reporter, &text);
        self.complete_when(is_complete, reporter);
    }

    /// Print the text after the part already printed.
    fn print_new_text<R: Reporter>(&mut self, reporter: &mut R, text: &str) {
        let new_text = text
            .get(self.printed_length..)
            .filter(|new_text| !new_text.is_empty());

        let Some(new_text) = new_text else {
            return;
        };

        reporter.report(AgentEvent::Text {
            kind: self.kind,
            text: new_text.to_string(),
        });
        self.printed_length = text.len();
    }

    /// End the field's line once its closing quote has arrived.
    fn complete_when<R: Reporter>(&mut self, is_complete: bool, reporter: &mut R) {
        if !is_complete {
            return;
        }

        self.state = FieldState::Complete;
        reporter.report(AgentEvent::TextEnded(self.kind));
    }

    /// Whether any text was forwarded to the reporter for this field.
    pub(super) fn did_print(&self) -> bool {
        self.printed_length > 0
    }
}

/// True when `buffer` has a top-level `"tool"` key but no top-level `key`.
fn tool_call_hides(buffer: &str, key: &str) -> bool {
    let has_tool = find_toplevel_key(buffer, "tool").is_some();
    let has_key = find_toplevel_key(buffer, key).is_some();

    matches!((has_tool, has_key), (true, false))
}

/// Where a JSON scanner is relative to string literals.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Position {
    /// Between tokens, outside any string.
    Outside,
    /// Inside a string literal.
    InString,
    /// Right after a backslash inside a string literal.
    Escaped,
}

/// The position after `byte`, given the position before it.
fn next_position(position: Position, byte: u8) -> Position {
    match (position, byte) {
        (Position::Outside, b'"') => Position::InString,
        (Position::InString, b'"') => Position::Outside,
        (Position::InString, b'\\') => Position::Escaped,
        (Position::Escaped, _) => Position::InString,
        (position, _) => position,
    }
}

/// The brace depth after `byte`; braces inside strings do not count.
fn next_depth(position: Position, depth: usize, byte: u8) -> usize {
    match (position, byte) {
        (Position::Outside, b'{') => depth + 1,
        (Position::Outside, b'}') => depth.saturating_sub(1),
        _ => depth,
    }
}

/// Scan state for finding a key at the top level of a partial JSON object.
struct KeyScanner {
    /// Position relative to string literals.
    position: Position,
    /// Number of open braces outside strings.
    depth: usize,
}

impl KeyScanner {
    /// True when `key_pattern` (the quoted key) starts at `index` at depth 1,
    /// outside any string. Moves the scanner past the byte at `index`.
    fn is_key_at(&mut self, bytes: &[u8], index: usize, key_pattern: &[u8]) -> bool {
        let is_top_level = matches!((self.position, self.depth), (Position::Outside, 1));
        let is_key = is_top_level && bytes[index..].starts_with(key_pattern);

        self.depth = next_depth(self.position, self.depth, bytes[index]);
        self.position = next_position(self.position, bytes[index]);

        is_key
    }
}

/// Find the position of a key at the top level of a (possibly partial) JSON object.
///
/// Tracks brace depth and string boundaries so that a key like `"answer"`
/// nested inside `"tool": {"arguments": {"answer": ...}}` is skipped.
/// Only matches keys at depth 1 (direct children of the root object).
fn find_toplevel_key(buffer: &str, key: &str) -> Option<usize> {
    let key_pattern = format!("\"{key}\"");
    let bytes = buffer.as_bytes();
    let mut scanner = KeyScanner {
        position: Position::Outside,
        depth: 0,
    };

    (0..bytes.len()).find(|&index| scanner.is_key_at(bytes, index, key_pattern.as_bytes()))
}

/// Extract the current value of a JSON string field from a partial JSON buffer.
///
/// Only matches the key at the top level of the JSON object (depth 1), so
/// nested keys inside tool arguments are never confused with the real
/// `thought` or `answer` fields. Escapes are kept as written.
///
/// Returns `(value_so_far, is_complete)`.
fn extract_json_string(buffer: &str, key: &str) -> Option<(String, bool)> {
    let start = find_toplevel_key(buffer, key)?;
    let after_key = &buffer[start + key.len() + 2..];

    string_value_start(after_key).map(read_string_value)
}

/// The text after the `:` and the opening quote that follow a key; `None`
/// when they have not arrived or the value is not a string.
fn string_value_start(after_key: &str) -> Option<&str> {
    after_key
        .trim_start()
        .strip_prefix(':')?
        .trim_start()
        .strip_prefix('"')
}

/// The string value at the start of `text` and whether its closing quote
/// has arrived.
fn read_string_value(text: &str) -> (String, bool) {
    match closing_quote(text) {
        Some(end) => (text[..end].to_string(), true),
        None => (text.to_string(), false),
    }
}

/// The byte index of the first unescaped quote in `text`, which starts
/// inside a string literal.
fn closing_quote(text: &str) -> Option<usize> {
    let mut position = Position::InString;

    text.bytes().position(|byte| {
        let closes = matches!((position, byte), (Position::InString, b'"'));
        position = next_position(position, byte);
        closes
    })
}

/// JSON escapes that stand for one character, and that character.
const SIMPLE_ESCAPES: [(char, char); 8] = [
    ('n', '\n'),
    ('r', '\r'),
    ('t', '\t'),
    ('\\', '\\'),
    ('"', '"'),
    ('/', '/'),
    ('b', '\u{0008}'),
    ('f', '\u{000C}'),
];

/// Unescape a JSON string value for terminal display.
///
/// The raw JSON keeps escapes like `\n` as two characters; without this the
/// TUI would render literal `\n` instead of actual newlines, which also breaks
/// markdown/code-fence detection and syntax highlighting.
///
/// A trailing backslash or an incomplete `\uXXXX` escape is left pending so
/// incremental streaming does not print a half-escape.
fn unescape_json_string(raw: &str) -> String {
    let mut text = String::new();
    let mut rest = Some(raw);

    while let Some(remaining) = rest {
        rest = unescape_next(remaining, &mut text);
    }

    text
}

/// Append `raw` up to and including its first escape to `text`, decoded.
/// Returns the rest, or `None` when `raw` is used up or ends inside an
/// escape.
fn unescape_next<'a>(raw: &'a str, text: &mut String) -> Option<&'a str> {
    let Some((before, after)) = raw.split_once('\\') else {
        text.push_str(raw);
        return None;
    };

    text.push_str(before);
    push_escape(after, text)
}

/// Append the escape at the start of `after`, the text after a backslash.
/// Returns the text after the escape, or `None` when it is incomplete.
fn push_escape<'a>(after: &'a str, text: &mut String) -> Option<&'a str> {
    let mut characters = after.chars();

    match characters.next() {
        Some('u') => push_unicode(characters.as_str(), text),
        Some(marker) => Some(push_simple(marker, characters.as_str(), text)),
        None => None,
    }
}

/// Append the character a one-letter escape stands for. An unknown escape
/// is kept as written rather than dropped. Returns `rest` unchanged.
fn push_simple<'a>(marker: char, rest: &'a str, text: &mut String) -> &'a str {
    let known = SIMPLE_ESCAPES.iter().find(|(letter, _)| *letter == marker);

    match known {
        Some((_, character)) => text.push(*character),
        None => text.extend(['\\', marker]),
    }

    rest
}

/// Append the character of a `\uXXXX` escape whose hex digits start `rest`.
/// Returns the text after the digits, or `None` when fewer than four hex
/// digits have arrived.
fn push_unicode<'a>(rest: &'a str, text: &mut String) -> Option<&'a str> {
    let hex = rest
        .get(..4)
        .filter(|hex| hex.chars().all(|digit| digit.is_ascii_hexdigit()))?;

    push_code_point(hex, text);
    Some(&rest[4..])
}

/// Append the character with code point `hex`; skip a code point that is not
/// a character, such as a lone surrogate.
fn push_code_point(hex: &str, text: &mut String) {
    let character = u32::from_str_radix(hex, 16).ok().and_then(char::from_u32);

    if let Some(character) = character {
        text.push(character);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::fakes::{
        RecordingReporter, Reply, serve_replies, text_delta, thinking_delta, tool_call_delta,
    };
    use crate::constants::ansi::{ANSI_GREEN_CHECK, COLOR_DIM, COLOR_RESET};
    use crate::ollama::{ChatMessage, Role};
    use serde_json::json;

    /// A chat request with one user message.
    fn request() -> ChatRequest {
        ChatRequest {
            model: "test-model".to_string(),
            messages: vec![ChatMessage {
                role: Role::User,
                content: "task".to_string(),
            }],
            stream: false,
            keep_alive: None,
            format: None,
            options: None,
        }
    }

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
        assert_eq!(
            extract_json_string(r#"{"answer":"say \"hi\""}"#, key),
            Some((r#"say \"hi\""#.to_string(), true))
        );
        assert_eq!(extract_json_string(r#"{"answer": 42}"#, key), None);
    }

    #[test]
    fn stream_state_skips_when_disabled() {
        let mut state = StreamState::new("thought", TextKind::Thought, false);
        let mut reporter = RecordingReporter::default();

        state.feed(&mut reporter, r#"{"thought":"secret"}"#);

        assert_eq!(state.state, FieldState::Skipped);
        assert!(!state.did_print());
        assert!(reporter.output.is_empty());
    }

    #[test]
    fn stream_state_streams_incrementally() {
        let mut state = StreamState::new("answer", TextKind::Answer, true);
        let mut reporter = RecordingReporter::default();

        state.feed(&mut reporter, r#"{"answer":"hel"#);
        state.feed(&mut reporter, r#"{"answer":"hel"#);
        state.feed(&mut reporter, r#"{"answer":"hello"}"#);
        state.feed(&mut reporter, r#"{"answer":"hello"} more"#);

        assert_eq!(reporter.output, format!("{ANSI_GREEN_CHECK} hello\n"));
        assert_eq!(state.state, FieldState::Complete);
    }

    #[test]
    fn stream_state_renders_escaped_newlines_as_real_newlines() {
        let mut state = StreamState::new("answer", TextKind::Answer, true);
        let mut reporter = RecordingReporter::default();

        state.feed(&mut reporter, r#"{"answer":"line1\nline2"}"#);

        assert!(reporter.output.contains("line1\nline2"));
        assert!(!reporter.output.contains("\\n"));
    }

    #[test]
    fn a_tool_call_hides_the_answer_field() {
        let mut state = StreamState::new("answer", TextKind::Answer, true);
        let mut reporter = RecordingReporter::default();

        state.feed(
            &mut reporter,
            r#"{"tool":{"name":"read_file","arguments":{"answer":"x"}}}"#,
        );

        assert_eq!(state.state, FieldState::Skipped);
        assert!(reporter.output.is_empty());
    }

    #[test]
    fn unescape_json_string_converts_common_escapes() {
        assert_eq!(unescape_json_string(r#"hello\nworld"#), "hello\nworld");
        assert_eq!(unescape_json_string(r#"tab\there"#), "tab\there");
        assert_eq!(unescape_json_string(r#"quote\"x"#), "quote\"x");
        assert_eq!(unescape_json_string(r#"back\\slash"#), "back\\slash");
        assert_eq!(unescape_json_string(r#"unicodeA"#), "unicodeA");
        assert_eq!(unescape_json_string(r#"\r\/\b\f"#), "\r/\u{0008}\u{000C}");
    }

    #[test]
    fn unescape_json_string_keeps_unknown_escapes_and_skips_surrogates() {
        assert_eq!(unescape_json_string(r#"a\qb"#), r#"a\qb"#);
        assert_eq!(unescape_json_string(r#"a\ud800b"#), "ab");
    }

    #[test]
    fn unescape_json_string_keeps_incomplete_escape_pending() {
        assert_eq!(unescape_json_string("abc\\"), "abc");
        assert_eq!(unescape_json_string("abc\\u12"), "abc");
        assert_eq!(unescape_json_string("abc\\u12zz"), "abc");
        assert_eq!(unescape_json_string("abc\\u1234"), "abc\u{1234}");
    }

    #[test]
    fn find_toplevel_key_skips_nested_keys() {
        // "answer" inside tool.arguments must NOT match at top level.
        let buffer = r#"{"thought":"thinking","tool":{"name":"write_file","arguments":{"path":"foo","content":"answer"}}}"#;

        assert!(find_toplevel_key(buffer, "answer").is_none());
        assert!(find_toplevel_key(buffer, "thought").is_some());
        assert!(find_toplevel_key(buffer, "tool").is_some());
    }

    #[test]
    fn find_toplevel_key_matches_top_level() {
        let buffer = r#"{"thought":"hi \"quoted\" {","answer":"hello"}"#;

        assert!(find_toplevel_key(buffer, "answer").is_some());
        assert!(find_toplevel_key(buffer, "thought").is_some());
    }

    #[test]
    fn find_toplevel_key_handles_key_inside_string_value() {
        // The string value contains the literal text "answer" — must not match.
        let buffer = r#"{"thought":"the answer is 42","tool":{"name":"finish"}}"#;

        assert!(find_toplevel_key(buffer, "answer").is_none());
    }

    #[test]
    fn extract_json_string_ignores_nested_answer() {
        let buffer = r#"{"thought":"calling tool","tool":{"name":"write_file","arguments":{"answer":"nested"}}}"#;

        assert!(extract_json_string(buffer, "answer").is_none());
    }

    #[test]
    fn extract_json_string_handles_partial_with_tool() {
        // Partial buffer: tool key seen but answer hasn't arrived yet.
        let buffer = r#"{"thought":"thinking","tool":{"name":"read_file""#;

        assert!(extract_json_string(buffer, "answer").is_none());
    }

    #[tokio::test]
    async fn streams_an_ollama_answer_and_reports_it_was_printed() -> anyhow::Result<()> {
        let reply = json!({ "thought": "plan", "answer": "done" }).to_string();
        let url = serve_replies(vec![Reply::ollama_stream(&reply)]).await?;
        let client = OllamaClient::new(&url)?;
        let mut reporter = RecordingReporter::default();

        let (content, was_streamed) =
            stream_model_response(&client, request(), &mut reporter, true).await?;

        assert_eq!(content, reply);
        assert!(was_streamed);
        assert!(reporter.output.contains("plan"));
        assert!(reporter.output.contains("done"));

        Ok(())
    }

    #[tokio::test]
    async fn prints_reasoning_only_when_thinking_is_shown() -> anyhow::Result<()> {
        let deltas = [
            thinking_delta("pondering"),
            thinking_delta(" more"),
            tool_call_delta("c1", "list_files", &json!({})),
            text_delta(r#"{"answer":"ok"}"#),
        ];
        let url = serve_replies(vec![
            Reply::openai_stream(&deltas),
            Reply::openai_stream(&deltas),
        ])
        .await?;
        let client = OllamaClient::openai(&url, None, true)?;

        let mut shown = RecordingReporter::default();
        stream_model_response(&client, request(), &mut shown, true).await?;
        assert!(
            shown
                .output
                .contains(&format!("💭 {COLOR_DIM}pondering more"))
        );

        let mut hidden = RecordingReporter::default();
        stream_model_response(&client, request(), &mut hidden, false).await?;
        assert!(!hidden.output.contains("pondering"));

        Ok(())
    }

    #[tokio::test]
    async fn a_reply_that_ends_while_reasoning_closes_the_block() -> anyhow::Result<()> {
        let url = serve_replies(vec![Reply::openai_stream(&[thinking_delta("hmm")])]).await?;
        let client = OllamaClient::openai(&url, None, true)?;
        let mut reporter = RecordingReporter::default();

        let (content, was_streamed) =
            stream_model_response(&client, request(), &mut reporter, true).await?;

        assert_eq!(content, "");
        assert!(!was_streamed);
        assert!(reporter.output.ends_with(&format!("hmm{COLOR_RESET}\n")));

        Ok(())
    }

    #[tokio::test]
    async fn a_failed_request_is_an_error() -> anyhow::Result<()> {
        let url = serve_replies(vec![Reply::server_error("model not loaded")]).await?;
        let client = OllamaClient::new(&url)?;
        let mut reporter = RecordingReporter::default();

        let result = stream_model_response(&client, request(), &mut reporter, true).await;

        assert!(result.is_err());

        Ok(())
    }

    #[tokio::test]
    async fn a_stream_error_stops_the_response() {
        let deltas = futures_util::stream::iter([
            Ok(Delta::Content("{".to_string())),
            Err(anyhow::anyhow!("connection reset")),
        ]);
        let mut reporter = RecordingReporter::default();

        let result = print_response(deltas, &mut reporter, true).await;

        assert!(result.is_err());
    }
}
