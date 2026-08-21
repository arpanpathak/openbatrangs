use crate::ollama::{ChatMessage, ChatRequest, OllamaClient};
use crate::tools;
use anyhow::{anyhow, bail, Context, Result};
use futures_util::StreamExt;
use serde::Deserialize;
use serde_json::Value;
use std::io::Write;
use std::path::{Path, PathBuf};

/// Colors used by the stdout reporter. The TUI reporter strips these before rendering.
const COLOR_CYAN: &str = "\x1b[36m";
const COLOR_MAGENTA: &str = "\x1b[35m";
const COLOR_BOLD: &str = "\x1b[1m";
const COLOR_DIM: &str = "\x1b[2m";
const COLOR_RESET: &str = "\x1b[0m";

/// Model context window caps. Lower max keeps attention faster on edge GPUs.
const MAX_CONTEXT_TOKENS: u64 = 16_384;
const MIN_CONTEXT_TOKENS: u64 = 4_096;

/// Tool argument defaults.
const DEFAULT_READ_CHARS: usize = 8_000;
const DEFAULT_GREP_MAX_RESULTS: usize = 200;
const COMMAND_TIMEOUT_SECONDS: u64 = 120;

/// Conversation trimming keeps the system message, the initial task, and this many recent exchanges.
const MAX_HISTORY_MESSAGES: usize = 40;

/// Sampling temperature for agentic tool-calling determinism.
const AGENT_TEMPERATURE: f64 = 0.2;

/// Receives agent output. `line` is a complete line; `chunk` is streaming text
/// that should be appended to the current live line.
pub trait Reporter: Send {
    fn line(&mut self, msg: String);
    fn chunk(&mut self, msg: &str);
}

/// Prints agent output directly to stdout (used by one-shot mode).
pub struct StdoutReporter;

impl Reporter for StdoutReporter {
    fn line(&mut self, msg: String) {
        if msg.is_empty() {
            println!();
        } else {
            println!("{msg}");
        }
    }

    fn chunk(&mut self, msg: &str) {
        print!("{msg}");
        let _ = std::io::stdout().flush();
    }
}

#[derive(Debug, Clone)]
pub struct AgentConfig {
    pub cwd: PathBuf,
    pub max_steps: usize,
    pub is_read_only: bool,
    pub should_confirm: bool,
    pub show_thinking: bool,
}

#[derive(Debug, Deserialize)]
struct ToolCall {
    name: String,
    arguments: Value,
}

/// Typed, exhaustively-matched representation of every tool the agent can invoke.
#[derive(Debug)]
enum Tool {
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
    fn from_call(call: ToolCall) -> Result<Self> {
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
    fn describe(&self) -> String {
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
struct AgentResponse {
    #[serde(default)]
    tool: Option<ToolCall>,
    #[serde(default)]
    answer: Option<String>,
}

const SYSTEM_PROMPT: &str = r#"You are openBatarangs, an autonomous coding agent running on a local edge device (Jetson-class hardware).

You must respond with ONLY one JSON object. Never use markdown fences. Never add prose outside the JSON.

Two valid response shapes:

1. To call a tool:
{"thought": "short reasoning", "tool": {"name": "tool_name", "arguments": {"arg": "value"}}}

2. To finish:
{"thought": "short reasoning", "answer": "final answer for the user"}

Available tools:
- list_files: arguments {"path": "relative_dir"} — list files in a directory (max depth 5)
- read_file: arguments {"path": "relative_file", "max_chars": 8000} — read a text file
- grep_files: arguments {"pattern": "regex", "path": "relative_dir", "max_results": 200} — search file contents
- write_file: arguments {"path": "relative_file", "content": "full file content"} — write/overwrite a file
- run_command: arguments {"command": "shell command"} — run a read/build/test shell command in the workspace
- finish: arguments {"summary": "done"} — same as answer, use to end

Rules:
- Always use paths relative to the workspace root. Absolute paths and '..' are rejected. "." is the workspace root.
- When asked to analyze the current directory or codebase, start with list_files path "." and then read the key files before concluding.
- Explore before editing. Read files before rewriting them.
- Prefer small, focused edits. Run build/test commands to verify when possible.
- Never invent file contents as done unless you actually wrote them.
- When the task is complete, provide a concise answer with what changed and any commands the user should run.
- Keep tool outputs in mind, but do not repeat them verbatim in the final answer."#;

/// The outcome of one agent iteration.
enum AgentStepOutcome {
    /// The agent answered or hit a terminal parse error.
    Finished,
    /// The agent called a tool and should continue to the next step.
    Continue,
}

/// Immutable inputs shared by every agent step.
struct AgentRunContext<'a> {
    config: &'a AgentConfig,
    client: &'a OllamaClient,
    model: &'a str,
    num_ctx: u64,
}

pub async fn run_agent<R: Reporter>(
    config: &AgentConfig,
    client: &OllamaClient,
    model: &str,
    model_context: u64,
    task: &str,
    reporter: &mut R,
) -> Result<()> {
    ensure_workspace_exists(&config.cwd)?;

    let num_ctx = model_context.clamp(MIN_CONTEXT_TOKENS, MAX_CONTEXT_TOKENS);
    let mut messages = initial_messages(&config.cwd, task);
    let mut changed_files: Vec<String> = Vec::new();

    let step_context = AgentRunContext {
        config,
        client,
        model,
        num_ctx,
    };

    for step in 1..=config.max_steps {
        let outcome = run_agent_step(
            &step_context,
            &mut messages,
            &mut changed_files,
            reporter,
            step,
        )
        .await?;
        match outcome {
            AgentStepOutcome::Finished => return Ok(()),
            AgentStepOutcome::Continue => {}
        }
    }

    reporter.line(format!(
        "\n⚠️  Reached max steps ({}) without a final answer.",
        config.max_steps
    ));
    Ok(())
}

/// Run one agent iteration: stream a response, parse it, and act on it.
async fn run_agent_step<R: Reporter>(
    context: &AgentRunContext<'_>,
    messages: &mut Vec<ChatMessage>,
    changed_files: &mut Vec<String>,
    reporter: &mut R,
    step: usize,
) -> Result<AgentStepOutcome> {
    report_step_start(reporter, step, context.config.max_steps, context.model);

    let request = chat_request(context.model, messages, context.num_ctx);
    let (content, answer_was_streamed) = stream_model_response(
        context.client,
        request,
        reporter,
        context.config.show_thinking,
    )
    .await
    .with_context(|| format!("model call failed at step {step}"))?;

    let response = match parse_agent_response(&content) {
        Ok(parsed) => parsed,
        Err(error) => {
            report_parse_error(reporter, &error, &content);
            return Ok(AgentStepOutcome::Finished);
        }
    };

    if let Some(answer) = response.answer {
        if !answer_was_streamed {
            reporter.line(format!("\n\x1b[32m✅\x1b[0m {answer}"));
        }
        print_changed_files(&context.config.cwd, changed_files, reporter);
        return Ok(AgentStepOutcome::Finished);
    }

    if let Some(tool_call) = response.tool {
        return handle_tool_call(
            context.config,
            messages,
            changed_files,
            reporter,
            &content,
            tool_call,
        )
        .await;
    }

    reporter.line("\n⚠️  Model returned neither an answer nor a tool call.".to_string());
    Ok(AgentStepOutcome::Finished)
}

/// Execute a parsed tool call and append the assistant/tool messages.
async fn handle_tool_call<R: Reporter>(
    config: &AgentConfig,
    messages: &mut Vec<ChatMessage>,
    changed_files: &mut Vec<String>,
    reporter: &mut R,
    content: &str,
    tool_call: ToolCall,
) -> Result<AgentStepOutcome> {
    let tool = Tool::from_call(tool_call)?;
    if let Tool::Finish { summary } = &tool {
        reporter.line(format!("\n\x1b[32m✅\x1b[0m {summary}"));
        print_changed_files(&config.cwd, changed_files, reporter);
        return Ok(AgentStepOutcome::Finished);
    }

    messages.push(ChatMessage {
        role: "assistant".to_string(),
        content: content.to_string(),
    });

    reporter.line(format!(
        "\n{COLOR_MAGENTA}🔧 {}{COLOR_RESET}",
        tool.describe()
    ));
    let result_text = execute_tool_or_report_error(config, &config.cwd, &tool, changed_files).await;
    reporter.line(format!("{COLOR_DIM}{result_text}{COLOR_RESET}"));

    messages.push(ChatMessage {
        role: "user".to_string(),
        content: format!("Tool result:\n{result_text}"),
    });
    trim_messages(messages, MAX_HISTORY_MESSAGES);
    Ok(AgentStepOutcome::Continue)
}

fn report_step_start<R: Reporter>(reporter: &mut R, step: usize, max_steps: usize, model: &str) {
    reporter.line(format!(
        "\n{COLOR_BOLD}{COLOR_CYAN}🦇 Step {step}/{max_steps}{COLOR_RESET} — model: {COLOR_BOLD}{model}{COLOR_RESET}"
    ));
}

fn report_parse_error<R: Reporter>(reporter: &mut R, error: &anyhow::Error, content: &str) {
    reporter.line(format!("\n⚠️  Could not parse model JSON: {error}"));
    reporter.line("Showing raw model output as the final response.".to_string());
    reporter.line(format!("\n{content}"));
}

async fn execute_tool_or_report_error(
    config: &AgentConfig,
    cwd: &Path,
    tool: &Tool,
    changed_files: &mut Vec<String>,
) -> String {
    match execute_tool(config, cwd, tool, changed_files).await {
        Ok(text) => text,
        Err(error) => format!("Tool error: {error:#}"),
    }
}

fn ensure_workspace_exists(cwd: &Path) -> Result<()> {
    if !cwd.is_dir() {
        bail!("working directory does not exist: {}", cwd.display());
    }
    Ok(())
}

fn initial_messages(cwd: &Path, task: &str) -> Vec<ChatMessage> {
    let initial_listing = tools::list_files(cwd, ".", tools::DEFAULT_LIST_DEPTH)
        .unwrap_or_else(|error| format!("(could not list files: {error})"));
    let user_content = format!(
        "Workspace root: {}\n\nInitial file listing:\n{}\n\nTask:\n{}",
        cwd.display(),
        initial_listing,
        task
    );
    vec![
        ChatMessage {
            role: "system".to_string(),
            content: SYSTEM_PROMPT.to_string(),
        },
        ChatMessage {
            role: "user".to_string(),
            content: user_content,
        },
    ]
}

fn chat_request(model: &str, messages: &[ChatMessage], num_ctx: u64) -> ChatRequest {
    ChatRequest {
        model: model.to_string(),
        messages: messages.to_vec(),
        stream: false,
        format: Some(Value::String("json".to_string())),
        options: Some(serde_json::json!({
            "temperature": AGENT_TEMPERATURE,
            "num_ctx": num_ctx,
        })),
    }
}

/// Streams a model response, forwarding thought/answer text to the reporter,
/// and returns the complete raw JSON content plus whether the answer was streamed.
async fn stream_model_response<R: Reporter>(
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
struct StreamState {
    key: &'static str,
    has_printed_prefix: bool,
    printed_length: usize,
    is_complete: bool,
    is_skipped: bool,
}

impl StreamState {
    fn new(key: &'static str, enabled: bool) -> Self {
        Self {
            key,
            has_printed_prefix: false,
            printed_length: 0,
            is_complete: false,
            is_skipped: key == "thought" && !enabled,
        }
    }

    fn feed<R: Reporter>(&mut self, reporter: &mut R, buffer: &str) -> Result<()> {
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
            "answer" => "\n\x1b[32m✅\x1b[0m ".to_string(),
            _ => String::new(),
        }
    }

    fn did_print(&self) -> bool {
        self.has_printed_prefix
    }
}

async fn execute_tool(
    config: &AgentConfig,
    cwd: &Path,
    tool: &Tool,
    changed_files: &mut Vec<String>,
) -> Result<String> {
    match tool {
        Tool::ListFiles { path } => tools::list_files(cwd, path, tools::DEFAULT_LIST_DEPTH),
        Tool::ReadFile { path, max_chars } => {
            let max_chars = (*max_chars).min(tools::MAX_TOOL_OUTPUT);
            tools::read_file(cwd, path, max_chars)
        }
        Tool::GrepFiles {
            pattern,
            path,
            max_results,
        } => tools::grep_files(cwd, pattern, path, *max_results),
        Tool::WriteFile { path, content } => {
            ensure_not_read_only(config)?;
            confirm_or_abort(config, &format!("write file '{path}'?"))?;
            let result = tools::write_file(cwd, path, content)?;
            changed_files.push(path.clone());
            Ok(format!("{result}\n📎 {}", clickable_path(cwd, path)))
        }
        Tool::RunCommand { command } => {
            ensure_not_read_only(config)?;
            confirm_or_abort(config, &format!("run command: {command}"))?;
            tools::run_command(cwd, command, COMMAND_TIMEOUT_SECONDS).await
        }
        Tool::Finish { summary } => Ok(format!("✅ {summary}")),
    }
}

fn ensure_not_read_only(config: &AgentConfig) -> Result<()> {
    if config.is_read_only {
        bail!("this tool is disabled in read-only mode");
    }
    Ok(())
}

fn required_string_arg<'a>(args: &'a Value, key: &str, tool_name: &str) -> Result<&'a str> {
    string_arg(args, key)?.ok_or_else(|| anyhow!("{tool_name} requires '{key}'"))
}

fn string_arg<'a>(args: &'a Value, key: &str) -> Result<Option<&'a str>> {
    match args.get(key) {
        Some(Value::String(value)) => Ok(Some(value)),
        Some(Value::Null) | None => Ok(None),
        Some(_) => bail!("argument '{key}' must be a string"),
    }
}

fn optional_u64_arg(args: &Value, key: &str) -> Result<Option<u64>> {
    match args.get(key) {
        Some(Value::Number(number)) => number
            .as_u64()
            .map(Some)
            .ok_or_else(|| anyhow!("argument '{key}' must be a non-negative integer")),
        Some(Value::Null) | None => Ok(None),
        Some(_) => bail!("argument '{key}' must be a number"),
    }
}

fn confirm_or_abort(config: &AgentConfig, prompt: &str) -> Result<()> {
    if !config.should_confirm {
        return Ok(());
    }
    print!("❓ {prompt} [y/N] ");
    std::io::stdout().flush().ok();
    let mut input = String::new();
    std::io::stdin().read_line(&mut input).ok();
    if !input.trim().eq_ignore_ascii_case("y") {
        bail!("aborted by user");
    }
    Ok(())
}

fn parse_agent_response(content: &str) -> Result<AgentResponse> {
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

fn trim_messages(messages: &mut Vec<ChatMessage>, max: usize) {
    if messages.len() <= max {
        return;
    }
    // Always keep the system message and the initial task message,
    // then the most recent exchanges.
    let system = messages.remove(0);
    let first_user = messages.remove(0);
    let tail = messages.split_off(messages.len().saturating_sub(max - 2));
    messages.clear();
    messages.push(system);
    messages.push(first_user);
    messages.extend(tail);
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

/// Terminal-clickable file path (OSC 8 hyperlink). Always emits an absolute path.
fn clickable_path(cwd: &Path, path: &str) -> String {
    let joined = cwd.join(path);
    let full = std::path::absolute(&joined).unwrap_or(joined);
    let display = full.to_string_lossy();
    let encoded = display
        .replace('%', "%25")
        .replace(' ', "%20")
        .replace('#', "%23")
        .replace('?', "%3F");
    format!("\x1b]8;;file://{encoded}\x1b\\{display}\x1b]8;;\x1b\\")
}

fn print_changed_files<R: Reporter>(cwd: &Path, files: &[String], reporter: &mut R) {
    if files.is_empty() {
        return;
    }
    reporter.line("\n\x1b[1;33m📁 Files changed:\x1b[0m".to_string());
    for file in files {
        reporter.line(format!("   {}", clickable_path(cwd, file)));
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
    fn trims_messages_but_keeps_system_and_first_user() {
        let mut messages = vec![
            ChatMessage {
                role: "system".to_string(),
                content: "system".to_string(),
            },
            ChatMessage {
                role: "user".to_string(),
                content: "task".to_string(),
            },
            ChatMessage {
                role: "assistant".to_string(),
                content: "a1".to_string(),
            },
            ChatMessage {
                role: "user".to_string(),
                content: "u1".to_string(),
            },
            ChatMessage {
                role: "assistant".to_string(),
                content: "a2".to_string(),
            },
        ];
        trim_messages(&mut messages, 3);
        assert_eq!(messages.len(), 3);
        assert_eq!(messages[0].role, "system");
        assert_eq!(messages[1].content, "task");
        assert_eq!(messages[2].content, "a2");
    }

    #[test]
    fn rejects_non_string_tool_argument() {
        let args = serde_json::json!({"path": 42});
        assert!(string_arg(&args, "path").is_err());
    }
}
