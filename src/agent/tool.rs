//! Typed tool-call model, argument parsing, and JSON response parsing.

use crate::constants::agent::{DEFAULT_GREP_MAX_RESULTS, DEFAULT_READ_CHARS};
use anyhow::{anyhow, bail, Context, Result};
use serde::Deserialize;
use serde_json::Value;

#[derive(Debug, Deserialize)]
pub(super) struct ToolCall {
    pub(super) name: String,
    pub(super) arguments: Value,
}

/// Typed, exhaustively-matched representation of every tool the agent can invoke.
#[derive(Debug)]
pub(super) enum Tool {
    ListFiles {
        path: String,
    },
    ReadFile {
        path: String,
        max_chars: usize,
    },
    GrepFiles {
        pattern: String,
        path: String,
        max_results: usize,
    },
    WriteFile {
        path: String,
        content: String,
    },
    RunCommand {
        command: String,
    },
    Finish {
        summary: String,
    },
}

impl Tool {
    pub(super) fn from_call(call: ToolCall) -> Result<Self> {
        let args = &call.arguments;
        match call.name.as_str() {
            "list_files" => Ok(Self::ListFiles {
                path: string_arg(args, "path")?.unwrap_or(".").to_string(),
            }),
            "read_file" => Ok(Self::ReadFile {
                path: required_string_arg(args, "path", "read_file")?.to_string(),
                max_chars: optional_u64_arg(args, "max_chars")?.unwrap_or(DEFAULT_READ_CHARS as u64)
                    as usize,
            }),
            "grep_files" => Ok(Self::GrepFiles {
                pattern: required_string_arg(args, "pattern", "grep_files")?.to_string(),
                path: string_arg(args, "path")?.unwrap_or(".").to_string(),
                max_results: optional_u64_arg(args, "max_results")?
                    .unwrap_or(DEFAULT_GREP_MAX_RESULTS as u64)
                    as usize,
            }),
            "write_file" => Ok(Self::WriteFile {
                path: required_string_arg(args, "path", "write_file")?.to_string(),
                content: required_string_arg(args, "content", "write_file")?.to_string(),
            }),
            "run_command" => Ok(Self::RunCommand {
                command: required_string_arg(args, "command", "run_command")?.to_string(),
            }),
            "finish" => Ok(Self::Finish {
                summary: string_arg(args, "summary")?.unwrap_or("done").to_string(),
            }),
            other => bail!("unknown tool: {other}"),
        }
    }

    /// Human-readable one-line description shown before the tool executes.
    pub(super) fn describe(&self) -> String {
        match self {
            Self::ListFiles { path } => format!("list_files → {path}"),
            Self::ReadFile { path, .. } => format!("read_file → {path}"),
            Self::GrepFiles { pattern, path, .. } => format!("grep_files → {pattern:?} in {path}"),
            Self::WriteFile { path, .. } => format!("write_file → {path}"),
            Self::RunCommand { command } => format!("run_command → {command}"),
            Self::Finish { .. } => "finish".to_string(),
        }
    }
}

#[derive(Debug, Deserialize)]
pub(super) struct AgentResponse {
    #[serde(default)]
    pub(super) tool: Option<ToolCall>,
    #[serde(default)]
    pub(super) answer: Option<String>,
}

pub(super) fn parse_agent_response(content: &str) -> Result<AgentResponse> {
    let cleaned = content
        .trim()
        .trim_start_matches("```json")
        .trim_start_matches("```")
        .trim_end_matches("```")
        .trim();
    let value: Value = serde_json::from_str(cleaned)
        .with_context(|| format!("invalid JSON from model: {content}"))?;
    serde_json::from_value(value)
        .with_context(|| format!("JSON did not match agent schema: {content}"))
}

pub(super) fn required_string_arg<'a>(
    args: &'a Value,
    key: &str,
    tool_name: &str,
) -> Result<&'a str> {
    string_arg(args, key)?.ok_or_else(|| anyhow!("{tool_name} requires '{key}'"))
}

pub(super) fn string_arg<'a>(args: &'a Value, key: &str) -> Result<Option<&'a str>> {
    match args.get(key) {
        Some(Value::String(value)) => Ok(Some(value)),
        Some(Value::Null) | None => Ok(None),
        Some(_) => bail!("argument '{key}' must be a string"),
    }
}

pub(super) fn optional_u64_arg(args: &Value, key: &str) -> Result<Option<u64>> {
    match args.get(key) {
        Some(Value::Number(number)) => number
            .as_u64()
            .map(Some)
            .ok_or_else(|| anyhow!("argument '{key}' must be a non-negative integer")),
        Some(Value::Null) | None => Ok(None),
        Some(_) => bail!("argument '{key}' must be a number"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_agent_response_without_code_fence() {
        let response =
            parse_agent_response(r#"{"answer":"done"}"#).expect("valid response should parse");
        assert_eq!(response.answer.as_deref(), Some("done"));
        assert!(response.tool.is_none());
    }

    #[test]
    fn parses_agent_response_with_code_fence() {
        let response = parse_agent_response(
            "```json\n{\"tool\":{\"name\":\"list_files\",\"arguments\":{\"path\":\".\"}}}\n```",
        )
        .expect("valid fenced response should parse");
        assert!(response.tool.is_some());
    }

    #[test]
    fn rejects_invalid_agent_response() {
        assert!(parse_agent_response("not json").is_err());
    }

    #[test]
    fn rejects_non_string_tool_argument() {
        let args = serde_json::json!({"path": 42});
        assert!(string_arg(&args, "path").is_err());
    }

    #[test]
    fn tool_from_call_parses_known_tools() {
        let call = ToolCall {
            name: "write_file".to_string(),
            arguments: serde_json::json!({"path": "a.txt", "content": "x"}),
        };
        let tool = Tool::from_call(call).unwrap();
        assert!(
            matches!(tool, Tool::WriteFile { path, content } if path == "a.txt" && content == "x")
        );
    }

    #[test]
    fn tool_from_call_rejects_unknown_tool() {
        let call = ToolCall {
            name: "rm_rf".to_string(),
            arguments: serde_json::json!({}),
        };
        assert!(Tool::from_call(call).is_err());
    }

    #[test]
    fn tool_describe_is_human_readable() {
        let tool = Tool::ListFiles {
            path: "src".to_string(),
        };
        assert_eq!(tool.describe(), "list_files → src");
    }
}
