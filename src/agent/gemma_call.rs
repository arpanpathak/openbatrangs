//! # Gemma 4 tool calls written as text
//!
//! Gemma 4's own tool-call syntax is `call:NAME{key:<|"|>text<|"|>,count:3}`,
//! sometimes wrapped in `<|tool_call>` and `<tool_call|>`. `<|"|>` is Gemma's
//! string quote token. When a server does not turn this text into a
//! structured tool call, it reaches the reply text; [`gemma_tool_call`] reads
//! it back as a [`ToolCall`].
//!
//! Each `<|"|>` toggles between text outside and inside a string, so the
//! arguments split on it alternate: even pieces hold keys and bare values,
//! odd pieces hold string values. Quoting both turns the arguments into JSON.

use super::tool::ToolCall;
use crate::constants::agent::{
    GEMMA_CALL_PREFIX, GEMMA_STRING_QUOTE, GEMMA_TOOL_CALL_CLOSE, GEMMA_TOOL_CALL_OPEN,
};
use serde_json::Value;

/// The first Gemma tool call in `reply`; `None` when there is none or its
/// arguments are not strings, numbers and booleans.
pub(super) fn gemma_tool_call(reply: &str) -> Option<ToolCall> {
    let call_start = reply.find(GEMMA_CALL_PREFIX)? + GEMMA_CALL_PREFIX.len();
    let call = first_call(&reply[call_start..]);
    let (name, arguments) = call.split_once('{')?;
    let arguments_end = arguments.rfind('}')?;
    let arguments = serde_json::from_str(&arguments_json(&arguments[..arguments_end])).ok()?;

    Some(ToolCall {
        name: name.trim().to_string(),
        arguments,
    })
}

/// True while `reply` is, or may still become, a Gemma tool call: after
/// leading white space it starts with `call:` or `<|tool_call>`, or with the
/// first characters of one of them.
pub(super) fn may_be_gemma_call(reply: &str) -> bool {
    let start = reply.trim_start();

    [GEMMA_CALL_PREFIX, GEMMA_TOOL_CALL_OPEN]
        .iter()
        .any(|opener| starts_like(start, opener))
}

/// True when `text` starts with `opener`, or `opener` starts with `text`.
fn starts_like(text: &str, opener: &str) -> bool {
    text.starts_with(opener) || opener.starts_with(text)
}

/// The text of the first call, up to its `<tool_call|>` when it has one.
fn first_call(calls: &str) -> &str {
    calls.split(GEMMA_TOOL_CALL_CLOSE).next().unwrap_or(calls)
}

/// A JSON object holding the Gemma `arguments`, the text between the call's
/// braces.
fn arguments_json(arguments: &str) -> String {
    let pieces: Vec<String> = arguments
        .split(GEMMA_STRING_QUOTE)
        .enumerate()
        .map(|(index, piece)| json_piece(index, piece))
        .collect();

    format!("{{{}}}", pieces.concat())
}

/// One piece of the arguments as JSON: a string value for an odd `index`,
/// keys and bare values for an even one.
fn json_piece(index: usize, piece: &str) -> String {
    let is_string = index % 2 == 1;

    if is_string {
        return Value::String(piece.to_string()).to_string();
    }

    with_quoted_keys(piece)
}

/// `piece` with the key of every `key:value` pair quoted; bare numbers and
/// booleans are already JSON.
fn with_quoted_keys(piece: &str) -> String {
    piece
        .split(',')
        .map(with_quoted_key)
        .collect::<Vec<String>>()
        .join(",")
}

/// `pair` with its key quoted; text without a `:` stays as it is.
fn with_quoted_key(pair: &str) -> String {
    let Some((key, value)) = pair.split_once(':') else {
        return pair.to_string();
    };

    format!("{}:{value}", Value::String(key.trim().to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::{Context, Result};
    use serde_json::json;

    #[test]
    fn reads_an_edit_call_with_a_multi_line_string() -> Result<()> {
        let reply = concat!(
            "call:edit_file{new_string:<|\"|>    #[test]\n    fn median_of_three() {\n",
            "        assert_eq!(median(&[3.0, 1.0, 2.0]), Some(2.0));\n    }\n<|\"|>,",
            "old_string:<|\"|>mod tests {<|\"|>,path:<|\"|>src/lib.rs<|\"|>}"
        );

        let call = gemma_tool_call(reply).context("no call")?;

        assert_eq!(call.name, "edit_file");
        assert_eq!(
            call.arguments,
            json!({
                "new_string": "    #[test]\n    fn median_of_three() {\n        assert_eq!(median(&[3.0, 1.0, 2.0]), Some(2.0));\n    }\n",
                "old_string": "mod tests {",
                "path": "src/lib.rs",
            })
        );
        Ok(())
    }

    #[test]
    fn reads_a_wrapped_call_with_numbers_and_booleans() -> Result<()> {
        let reply = concat!(
            "<|tool_call>call:read_file{path:<|\"|>a.rs<|\"|>, start_line:3,whole:false}",
            "<tool_call|><|tool_call>call:finish{summary:<|\"|>later<|\"|>}<tool_call|>"
        );

        let call = gemma_tool_call(reply).context("no call")?;

        assert_eq!(call.name, "read_file");
        assert_eq!(
            call.arguments,
            json!({"path": "a.rs", "start_line": 3, "whole": false})
        );
        Ok(())
    }

    #[test]
    fn reads_a_call_without_arguments() -> Result<()> {
        let call = gemma_tool_call("call:list_files{}").context("no call")?;

        assert_eq!(call.name, "list_files");
        assert_eq!(call.arguments, json!({}));
        Ok(())
    }

    #[test]
    fn rejects_text_that_is_not_a_call() {
        assert!(gemma_tool_call("I will read the file.").is_none());
        assert!(gemma_tool_call("call:list_files").is_none());
        assert!(gemma_tool_call("call:list_files{path:.").is_none());
        assert!(gemma_tool_call("call:read_file{path:src/lib.rs}").is_none());
    }

    #[test]
    fn a_reply_may_be_a_call_until_it_differs_from_the_opener() {
        assert!(may_be_gemma_call(""));
        assert!(may_be_gemma_call("\n ca"));
        assert!(may_be_gemma_call("call:edit_file{"));
        assert!(may_be_gemma_call("<|tool"));
        assert!(may_be_gemma_call("<|tool_call>call:x"));
        assert!(!may_be_gemma_call("cat"));
        assert!(!may_be_gemma_call("I added it. call:x{}"));
    }
}
