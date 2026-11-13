//! # Typed tool-call model and JSON argument parsing
//!
//! The agent model communicates its next action as a JSON object. To avoid
//! passing raw `serde_json::Value` objects around the rest of the codebase,
//! this module converts the model's free-form JSON into the exhaustive [`Tool`]
//! enum. Every tool the agent may call is represented as a variant, and every
//! variant's arguments are validated here.
//!
//! Parsing happens in two stages:
//!
//! 1. [`parse_agent_response`] deserializes the whole model reply into
//!    [`AgentResponse`] (either `answer` or `tool`).
//! 2. [`Tool::from_call`] validates the tool arguments and builds a typed
//!    [`Tool`].
//!
//! ## References
//!
//! - JSON data model and string escapes, RFC 8259: <https://datatracker.ietf.org/doc/html/rfc8259>
//! - Serde JSON: <https://serde.rs/json.html>
//! - OpenAI function calling (the same JSON tool-call idea): <https://platform.openai.com/docs/guides/function-calling>
//! - ReAct: reasoning + acting: <https://arxiv.org/abs/2210.03629>

use crate::constants::agent::{DEFAULT_GREP_MAX_RESULTS, DEFAULT_READ_CHARS};
use anyhow::{Context, Result, anyhow, bail};
use serde::Deserialize;
use serde_json::{Value, json};

/// A raw tool call as deserialized from the model's JSON response.
#[derive(Debug, Deserialize)]
pub(super) struct ToolCall {
    /// Tool name, e.g. `"list_files"`.
    pub(super) name: String,
    /// Raw tool arguments; validated by [`Tool::from_call`].
    pub(super) arguments: Value,
}

/// Typed, exhaustively-matched representation of every tool the agent can invoke.
///
/// Each variant stores only the arguments that tool actually needs. This makes
/// the rest of the codebase free of stringly-typed tool dispatch.
#[derive(Debug)]
pub(super) enum Tool {
    /// Recursively list files under `path`.
    ListFiles {
        /// Relative directory to list; `.` means the workspace root.
        path: String,
    },
    /// Read a text file with a character cap.
    ReadFile {
        /// Relative file path inside the workspace.
        path: String,
        /// Maximum characters to include in the result.
        max_chars: usize,
    },
    /// Regex-search file contents under `path`.
    GrepFiles {
        /// Regular expression to match.
        pattern: String,
        /// Relative directory to search.
        path: String,
        /// Maximum number of matches to return.
        max_results: usize,
    },
    /// Search the current web using local SearXNG.
    WebSearch {
        /// Focused search query.
        query: String,
    },
    /// Read official NVIDIA or GitHub documentation returned by search.
    WebRead {
        /// HTTPS URL of an approved documentation page.
        url: String,
    },
    /// Write (or overwrite) a file.
    WriteFile {
        /// Relative file path inside the workspace.
        path: String,
        /// Full file contents.
        content: String,
    },
    /// Replace a literal string inside an existing file.
    EditFile {
        /// Relative file path inside the workspace.
        path: String,
        /// Exact text to find.
        old_string: String,
        /// Replacement text.
        new_string: String,
        /// Replace every occurrence instead of requiring a unique match.
        replace_all: bool,
    },
    /// Run a shell command in the workspace.
    RunCommand {
        /// Shell command line, executed with `bash -lc`.
        command: String,
    },
    /// Signal that the task is complete.
    Finish {
        /// Short summary of what was done.
        summary: String,
    },
    /// Stop and ask the user a question; their next message is the answer.
    AskUser {
        /// The question to show the user.
        question: String,
    },
}

impl Tool {
    /// Convert a raw [`ToolCall`] into a typed [`Tool`], validating arguments.
    ///
    /// # Parameters
    ///
    /// - `call`: raw tool call from the model.
    ///
    /// # Returns
    ///
    /// A typed [`Tool`], or an error when the tool name is unknown or an
    /// argument has the wrong type.
    pub(super) fn from_call(call: ToolCall) -> Result<Self> {
        let parser = TOOL_PARSERS
            .iter()
            .find(|(name, _)| *name == call.name)
            .map(|(_, parser)| parser);

        let Some(parse) = parser else {
            bail!("unknown tool: {}", call.name);
        };

        parse(&call.arguments)
    }

    /// Human-readable one-line description shown before the tool executes.
    pub(super) fn describe(&self) -> String {
        match self {
            Self::ListFiles { path } => format!("list_files → {path}"),
            Self::ReadFile { path, .. } => format!("read_file → {path}"),
            Self::GrepFiles { pattern, path, .. } => format!("grep_files → {pattern:?} in {path}"),
            Self::WebSearch { query } => format!("web_search → {query}"),
            Self::WebRead { url } => format!("web_read → {url}"),
            Self::WriteFile { path, .. } => format!("write_file → {path}"),
            Self::EditFile { path, .. } => format!("edit_file → {path}"),
            Self::RunCommand { command } => format!("run_command → {command}"),
            Self::Finish { .. } => "finish".to_string(),
            Self::AskUser { question } => format!("ask_user → {question}"),
        }
    }
}

/// Builds one [`Tool`] variant from the call's JSON arguments.
type ToolParser = fn(&Value) -> Result<Tool>;

/// The parser for each tool name the model may call.
const TOOL_PARSERS: [(&str, ToolParser); 10] = [
    ("list_files", list_files_tool),
    ("read_file", read_file_tool),
    ("grep_files", grep_files_tool),
    ("web_search", web_search_tool),
    ("web_read", web_read_tool),
    ("write_file", write_file_tool),
    ("edit_file", edit_file_tool),
    ("run_command", run_command_tool),
    ("finish", finish_tool),
    ("ask_user", ask_user_tool),
];

/// `list_files`: the directory defaults to the workspace root.
fn list_files_tool(args: &Value) -> Result<Tool> {
    Ok(Tool::ListFiles {
        path: path_arg(args)?.unwrap_or(".").to_string(),
    })
}

/// `read_file`: a required path and an optional character cap.
fn read_file_tool(args: &Value) -> Result<Tool> {
    Ok(Tool::ReadFile {
        path: required_path_arg(args, "read_file")?.to_string(),
        max_chars: optional_u64_arg(args, "max_chars")?.unwrap_or(DEFAULT_READ_CHARS as u64)
            as usize,
    })
}

/// `grep_files`: a required pattern, a directory, and a result cap.
fn grep_files_tool(args: &Value) -> Result<Tool> {
    Ok(Tool::GrepFiles {
        pattern: required_string_arg(args, "pattern", "grep_files")?.to_string(),
        path: path_arg(args)?.unwrap_or(".").to_string(),
        max_results: optional_u64_arg(args, "max_results")?
            .unwrap_or(DEFAULT_GREP_MAX_RESULTS as u64) as usize,
    })
}

/// `web_search`: a required query.
fn web_search_tool(args: &Value) -> Result<Tool> {
    Ok(Tool::WebSearch {
        query: required_string_arg(args, "query", "web_search")?.to_string(),
    })
}

/// `web_read`: a required URL.
fn web_read_tool(args: &Value) -> Result<Tool> {
    Ok(Tool::WebRead {
        url: required_string_arg(args, "url", "web_read")?.to_string(),
    })
}

/// `write_file`: a required path and the complete content.
fn write_file_tool(args: &Value) -> Result<Tool> {
    Ok(Tool::WriteFile {
        path: required_path_arg(args, "write_file")?.to_string(),
        content: required_string_arg(args, "content", "write_file")?.to_string(),
    })
}

/// `edit_file`: a path, the text to find, its replacement, and whether to
/// replace every occurrence.
fn edit_file_tool(args: &Value) -> Result<Tool> {
    Ok(Tool::EditFile {
        path: required_path_arg(args, "edit_file")?.to_string(),
        old_string: required_string_arg(args, "old_string", "edit_file")?.to_string(),
        new_string: required_string_arg(args, "new_string", "edit_file")?.to_string(),
        replace_all: optional_bool_arg(args, "replace_all")?.unwrap_or(false),
    })
}

/// `run_command`: a required command line.
fn run_command_tool(args: &Value) -> Result<Tool> {
    Ok(Tool::RunCommand {
        command: required_string_arg(args, "command", "run_command")?.to_string(),
    })
}

/// `finish`: the summary defaults to `done`.
fn finish_tool(args: &Value) -> Result<Tool> {
    Ok(Tool::Finish {
        summary: string_arg(args, "summary")?.unwrap_or("done").to_string(),
    })
}

/// `ask_user`: a required question.
fn ask_user_tool(args: &Value) -> Result<Tool> {
    Ok(Tool::AskUser {
        question: required_string_arg(args, "question", "ask_user")?.to_string(),
    })
}

/// The one-line description of a call to tool `name` with JSON `arguments`;
/// the bare name when the call does not parse.
pub(crate) fn describe_call(name: &str, arguments: &str) -> String {
    let call = ToolCall {
        name: name.to_string(),
        arguments: serde_json::from_str(arguments.trim()).unwrap_or_else(|_error| json!({})),
    };

    Tool::from_call(call).map_or_else(|_error| name.to_string(), |tool| tool.describe())
}

/// The two possible shapes of a model response: either a final `answer` or a
/// single `tool` call. Serde fills the missing field with `None`.
#[derive(Debug, Deserialize)]
pub(super) struct AgentResponse {
    /// Tool call to execute, when the model is not done yet.
    #[serde(default)]
    pub(super) tool: Option<ToolCall>,
    /// Final answer, when the model is done.
    #[serde(default)]
    pub(super) answer: Option<String>,
}

/// Parse the model's raw text into an [`AgentResponse`].
///
/// The model is asked to emit strict JSON, but smaller local models sometimes
/// wrap it in a markdown code fence, so both forms are accepted.
///
/// # Parameters
///
/// - `content`: raw model output.
///
/// # Returns
///
/// A parsed [`AgentResponse`], or an error describing the JSON problem.
pub(super) fn parse_agent_response(content: &str) -> Result<AgentResponse> {
    let value: Value = serde_json::from_str(strip_code_fence(content))
        .with_context(|| format!("invalid JSON from model: {content}"))?;

    serde_json::from_value(nest_flat_tool_call(value))
        .with_context(|| format!("JSON did not match agent schema: {content}"))
}

/// `content` without surrounding whitespace and without a markdown code
/// fence around it, which smaller local models add to their JSON.
pub(super) fn strip_code_fence(content: &str) -> &str {
    content
        .trim()
        .trim_start_matches("```json")
        .trim_start_matches("```")
        .trim_end_matches("```")
        .trim()
}

/// Rewrites the flat shape some models use,
/// `{"tool": "read_file", "arguments": {...}}`, into the documented
/// `{"tool": {"name": "read_file", "arguments": {...}}}`. `args` and
/// `parameters` are accepted in place of `arguments`. Anything else is
/// returned unchanged.
fn nest_flat_tool_call(mut value: Value) -> Value {
    let Some(name) = value
        .get("tool")
        .and_then(Value::as_str)
        .map(str::to_string)
    else {
        return value;
    };
    let arguments = ["arguments", "args", "parameters"]
        .iter()
        .find_map(|key| value.get_mut(*key).map(Value::take))
        .unwrap_or_else(|| Value::Object(serde_json::Map::new()));
    serde_json::json!({ "tool": { "name": name, "arguments": arguments } })
}

/// Require a string argument; error when it is missing or not a string.
pub(super) fn required_string_arg<'a>(
    args: &'a Value,
    key: &str,
    tool_name: &str,
) -> Result<&'a str> {
    string_arg(args, key)?.ok_or_else(|| anyhow!("{tool_name} requires '{key}'"))
}

/// Names models use for a path argument besides the documented `path`.
const PATH_ALIASES: [&str; 5] = ["path", "file_path", "filepath", "file", "filename"];

/// Read the path argument under `path` or one of the names models commonly
/// use instead, so a `file_path` from the model does not fail the call.
///
/// The first alias with a non-null value is used; it must be a string.
pub(super) fn path_arg(args: &Value) -> Result<Option<&str>> {
    let key = PATH_ALIASES
        .iter()
        .find(|key| present_arg(args, key).is_some());

    key.map_or(Ok(None), |key| string_arg(args, key))
}

/// Require the path argument of `tool_name`.
fn required_path_arg<'a>(args: &'a Value, tool_name: &str) -> Result<&'a str> {
    path_arg(args)?.ok_or_else(|| anyhow!("{tool_name} requires 'path'"))
}

/// The argument under `key`, or `None` when it is missing or `null`.
fn present_arg<'a>(args: &'a Value, key: &str) -> Option<&'a Value> {
    args.get(key).filter(|value| !value.is_null())
}

/// Read an optional string argument.
///
/// `null` and missing keys both produce `None`; any other JSON type is an
/// error, so a number is never used as a path.
pub(super) fn string_arg<'a>(args: &'a Value, key: &str) -> Result<Option<&'a str>> {
    present_arg(args, key)
        .map(|value| {
            value
                .as_str()
                .ok_or_else(|| anyhow!("argument '{key}' must be a string"))
        })
        .transpose()
}

/// Read an optional boolean argument.
///
/// `null` and missing keys both produce `None`; any other JSON type is an error.
pub(super) fn optional_bool_arg(args: &Value, key: &str) -> Result<Option<bool>> {
    present_arg(args, key)
        .map(|value| {
            value
                .as_bool()
                .ok_or_else(|| anyhow!("argument '{key}' must be a boolean"))
        })
        .transpose()
}

/// Read an optional non-negative integer argument.
///
/// `null` and missing keys both produce `None`; negative or non-numeric values
/// are rejected.
pub(super) fn optional_u64_arg(args: &Value, key: &str) -> Result<Option<u64>> {
    present_arg(args, key)
        .map(|value| u64_value(value, key))
        .transpose()
}

/// A JSON number as a `u64`; `key` names the argument in the error.
fn u64_value(value: &Value, key: &str) -> Result<u64> {
    let Value::Number(number) = value else {
        bail!("argument '{key}' must be a number");
    };

    number
        .as_u64()
        .ok_or_else(|| anyhow!("argument '{key}' must be a non-negative integer"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// Parse a call to `name` with `arguments`.
    fn tool(name: &str, arguments: Value) -> Result<Tool> {
        Tool::from_call(ToolCall {
            name: name.to_string(),
            arguments,
        })
    }

    #[test]
    fn parses_agent_response_without_code_fence() -> Result<()> {
        let response = parse_agent_response(r#"{"answer":"done"}"#)?;

        assert_eq!(response.answer.as_deref(), Some("done"));
        assert!(response.tool.is_none());

        Ok(())
    }

    #[test]
    fn parses_agent_response_with_code_fence() -> Result<()> {
        let response = parse_agent_response(
            "```json\n{\"tool\":{\"name\":\"list_files\",\"arguments\":{\"path\":\".\"}}}\n```",
        )?;

        assert!(response.tool.is_some());

        Ok(())
    }

    #[test]
    fn rejects_invalid_agent_response() {
        assert!(parse_agent_response("not json").is_err());
        assert!(parse_agent_response(r#"{"answer": 7}"#).is_err());
    }

    #[test]
    fn parses_the_flat_tool_call_shape() -> Result<()> {
        let response =
            parse_agent_response(r#"{"tool": "read_file", "arguments": {"path": "Cargo.toml"}}"#)?;
        let call = response.tool.context("tool call")?;

        assert_eq!(Tool::from_call(call)?.describe(), "read_file → Cargo.toml");

        Ok(())
    }

    #[test]
    fn the_flat_shape_accepts_args_and_no_arguments() -> Result<()> {
        let with_args = parse_agent_response(r#"{"tool": "list_files", "args": {"path": "src"}}"#)?;
        let call = with_args.tool.context("tool call")?;
        assert_eq!(call.arguments, json!({"path": "src"}));

        let without = parse_agent_response(r#"{"tool": "finish"}"#)?;
        let call = without.tool.context("tool call")?;
        assert_eq!(call.arguments, json!({}));

        Ok(())
    }

    #[test]
    fn accepts_file_path_as_the_path_argument() -> Result<()> {
        let parsed = tool("read_file", json!({"file_path": "Cargo.toml"}))?;

        assert_eq!(parsed.describe(), "read_file → Cargo.toml");

        Ok(())
    }

    #[test]
    fn a_null_path_alias_falls_through_to_the_next_one() -> Result<()> {
        let args = json!({"path": null, "filename": "a.rs"});

        assert_eq!(path_arg(&args)?, Some("a.rs"));
        assert_eq!(path_arg(&json!({}))?, None);
        assert!(path_arg(&json!({"file": 3})).is_err());

        Ok(())
    }

    #[test]
    fn rejects_non_string_tool_argument() {
        let args = json!({"path": 42});

        assert!(string_arg(&args, "path").is_err());
    }

    #[test]
    fn tool_from_call_parses_known_tools() -> Result<()> {
        let parsed = tool("write_file", json!({"path": "a.txt", "content": "x"}))?;

        assert_eq!(
            format!("{parsed:?}"),
            r#"WriteFile { path: "a.txt", content: "x" }"#
        );

        Ok(())
    }

    #[test]
    fn every_tool_parses_and_describes_itself() {
        let calls = [
            ("list_files", json!({}), "list_files → ."),
            ("read_file", json!({"path": "a.rs"}), "read_file → a.rs"),
            (
                "grep_files",
                json!({"pattern": "fn"}),
                "grep_files → \"fn\" in .",
            ),
            (
                "web_search",
                json!({"query": "tokio"}),
                "web_search → tokio",
            ),
            (
                "web_read",
                json!({"url": "https://a.b"}),
                "web_read → https://a.b",
            ),
            (
                "write_file",
                json!({"path": "a", "content": ""}),
                "write_file → a",
            ),
            (
                "edit_file",
                json!({"path": "a", "old_string": "x", "new_string": "y", "replace_all": true}),
                "edit_file → a",
            ),
            ("run_command", json!({"command": "ls"}), "run_command → ls"),
            ("finish", json!({}), "finish"),
            (
                "ask_user",
                json!({"question": "Which file?"}),
                "ask_user → Which file?",
            ),
        ];

        for (name, arguments, description) in calls {
            let described = tool(name, arguments).map(|parsed| parsed.describe());
            assert_eq!(described.ok().as_deref(), Some(description));
        }
    }

    #[test]
    fn describe_call_names_a_call_that_does_not_parse() {
        assert_eq!(
            describe_call("read_file", r#"{"path":"a.rs"}"#),
            "read_file → a.rs"
        );
        assert_eq!(describe_call("rm_rf", "not json"), "rm_rf");
    }

    #[test]
    fn defaults_fill_optional_arguments() -> Result<()> {
        let read = tool("read_file", json!({"path": "a"}))?;
        let grep = tool("grep_files", json!({"pattern": "x", "max_results": 5}))?;
        let finish = tool("finish", json!({"summary": null}))?;

        assert!(matches!(
            read,
            Tool::ReadFile {
                max_chars: DEFAULT_READ_CHARS,
                ..
            }
        ));
        assert!(matches!(grep, Tool::GrepFiles { max_results: 5, .. }));
        assert_eq!(format!("{finish:?}"), r#"Finish { summary: "done" }"#);

        Ok(())
    }

    #[test]
    fn wrong_argument_types_are_errors() {
        let bad_calls = [
            ("read_file", json!({})),
            ("read_file", json!({"path": "a", "max_chars": -1})),
            ("read_file", json!({"path": "a", "max_chars": "many"})),
            (
                "edit_file",
                json!({"path": "a", "old_string": "x", "new_string": "y", "replace_all": "yes"}),
            ),
            ("web_search", json!({})),
        ];

        for (name, arguments) in bad_calls {
            assert!(tool(name, arguments).is_err(), "{name} should be rejected");
        }
    }

    #[test]
    fn tool_from_call_rejects_unknown_tool() {
        assert!(tool("rm_rf", json!({})).is_err());
    }

    #[test]
    fn code_fences_are_stripped() {
        assert_eq!(strip_code_fence(" ```json\n{}\n``` "), "{}");
        assert_eq!(strip_code_fence("```\n{}\n```"), "{}");
    }
}
