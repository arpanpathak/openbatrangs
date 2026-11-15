//! # Gemma tool calls written as text
//!
//! Gemma 4 writes a tool call in its own syntax:
//!
//! ```text
//! <|tool_call>call:write_file{path:<|"|>a.py<|"|>,content:<|"|>print(1)<|"|>}<tool_call|>
//! ```
//!
//! Keys are bare words and every string sits between two `<|"|>` markers;
//! numbers, booleans, arrays and objects are written as in JSON. The server's
//! tool-call parser normally turns this into a structured call, but when it
//! does not, the syntax reaches the reply text. This module reads a complete
//! call from that text, and tells the native loop when the text holds a call
//! marker at all, so a broken call is never taken as the answer.

use regex::Regex;
use serde_json::{Map, Value};

/// Opens a Gemma tool call.
const CALL_START: &str = "<|tool_call>call:";

/// Closes a Gemma tool call.
const CALL_END: &str = "<tool_call|>";

/// Opens and closes a string inside a Gemma tool call.
const STRING_MARKER: &str = "<|\"|>";

/// Markers of a tool call written as text, in Gemma's and in the common
/// `<tool_call>` syntax.
const RAW_CALL_MARKERS: [&str; 2] = ["<|tool_call>", "<tool_call>"];

/// A bare object key after `{` or `,`, with the spaces around it.
const BARE_KEY: &str = r"([{,]\s*)([A-Za-z_][A-Za-z0-9_]*)(\s*:)";

/// Whether `text` holds the start of a tool call written as text.
pub(super) fn has_raw_call_marker(text: &str) -> bool {
    RAW_CALL_MARKERS.iter().any(|marker| text.contains(marker))
}

/// The tool name and the arguments of the first complete Gemma call in
/// `text`; `None` when there is none, or it is cut off or does not parse.
pub(super) fn parse_gemma_call(text: &str) -> Option<(&str, Map<String, Value>)> {
    let (_before, call) = text.split_once(CALL_START)?;
    let (call, _after) = call.split_once(CALL_END)?;
    let brace = call.find('{')?;
    let (name, arguments) = call.split_at(brace);
    let arguments = serde_json::from_str(&as_json(arguments)?).ok()?;

    Some((name.trim(), arguments))
}

/// The arguments of a Gemma call rewritten as JSON: strings quoted and
/// escaped, bare keys quoted. `None` when a string is never closed.
fn as_json(arguments: &str) -> Option<String> {
    let parts: Vec<&str> = arguments.split(STRING_MARKER).collect();
    let bare_key = Regex::new(BARE_KEY).ok()?;

    // Text and strings alternate, so an even count means an open string.
    if parts.len().is_multiple_of(2) {
        return None;
    }

    let json = parts
        .iter()
        .enumerate()
        .map(|(position, part)| match position % 2 {
            0 => bare_key.replace_all(part, "$1\"$2\"$3").into_owned(),
            _ => Value::from(*part).to_string(),
        })
        .collect();

    Some(json)
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::{Context, Result};
    use serde_json::json;

    #[test]
    fn a_complete_call_is_read_with_its_strings_numbers_and_lists() -> Result<()> {
        let text = concat!(
            "I will write it.\n",
            "<|tool_call>call:write_file{path:<|\"|>a.py<|\"|>,",
            "content:<|\"|>print(\"hi\")\nx = {a: 1}, done<|\"|>,",
            "tries:2,options:[<|\"|>x<|\"|>,<|\"|>y<|\"|>],fast:true}<tool_call|>",
        );

        let (name, arguments) = parse_gemma_call(text).context("the call parses")?;

        assert_eq!(name, "write_file");
        assert_eq!(
            Value::Object(arguments),
            json!({
                "path": "a.py",
                "content": "print(\"hi\")\nx = {a: 1}, done",
                "tries": 2,
                "options": ["x", "y"],
                "fast": true,
            })
        );

        Ok(())
    }

    #[test]
    fn a_cut_off_or_broken_call_is_not_read() {
        let cut = "<|tool_call>call:write_file{path:<|\"|>a.py<|\"|>,content:<|\"|>import json";
        let open_string = "<|tool_call>call:write_file{path:<|\"|>a.py}<tool_call|>";
        let not_json = "<|tool_call>call:write_file{path:<|\"|>a<|\"|> oops}<tool_call|>";
        let no_brace = "<|tool_call>call:finish<tool_call|>";
        let trailing = "<|tool_call>call:finish{}[1]<tool_call|>";

        assert_eq!(parse_gemma_call(cut), None);
        assert_eq!(parse_gemma_call(open_string), None);
        assert_eq!(parse_gemma_call(not_json), None);
        assert_eq!(parse_gemma_call(no_brace), None);
        assert_eq!(parse_gemma_call(trailing), None);
        assert_eq!(parse_gemma_call("plain answer"), None);
    }

    #[test]
    fn call_markers_are_found_in_either_syntax() {
        assert!(has_raw_call_marker("text <|tool_call>call:x{"));
        assert!(has_raw_call_marker("<tool_call>{\"name\": \"x\"}"));
        assert!(!has_raw_call_marker("The tool_call field is documented."));
    }
}
