use crate::ollama::{ChatMessage, ChatRequest, OllamaClient};
use crate::tools;
use anyhow::{anyhow, bail, Context, Result};
use futures_util::StreamExt;
use serde::Deserialize;
use serde_json::Value;
use std::io::Write;
use std::path::{Path, PathBuf};

const CYAN: &str = "\x1b[36m";
const MAGENTA: &str = "\x1b[35m";
const BOLD: &str = "\x1b[1m";
const DIM: &str = "\x1b[2m";
const RESET: &str = "\x1b[0m";

/// Receives agent output. `line` is a complete line; `chunk` is streaming text
/// that should be appended to the current live line.
pub trait Reporter: Send {
    fn line(&mut self, msg: String);
    fn chunk(&mut self, msg: String);
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

    fn chunk(&mut self, msg: String) {
        print!("{msg}");
        let _ = std::io::stdout().flush();
    }
}

#[derive(Debug, Clone)]
pub struct AgentConfig {
    pub cwd: PathBuf,
    pub max_steps: usize,
    pub read_only: bool,
    pub confirm: bool,
}

#[derive(Debug, Deserialize)]
struct ToolCall {
    name: String,
    arguments: Value,
}

#[derive(Debug, Deserialize)]
struct AgentResponse {
    #[serde(default)]
    #[allow(dead_code)]
    thought: Option<String>,
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
- Always use paths relative to the workspace root. Absolute paths and '..' are rejected.
- Explore before editing. Read files before rewriting them.
- Prefer small, focused edits. Run build/test commands to verify when possible.
- Never invent file contents as done unless you actually wrote them.
- When the task is complete, provide a concise answer with what changed and any commands the user should run.
- Keep tool outputs in mind, but do not repeat them verbatim in the final answer."#;

pub async fn run_agent<R: Reporter>(
    config: &AgentConfig,
    client: &OllamaClient,
    model: &str,
    model_context: u64,
    task: &str,
    reporter: &mut R,
) -> Result<()> {
    let cwd = config.cwd.clone();
    if !cwd.is_dir() {
        bail!("working directory does not exist: {}", cwd.display());
    }

    let num_ctx = model_context.min(32768).max(4096) as u64;
    let initial_listing =
        tools::list_files(&cwd, ".", 5).unwrap_or_else(|e| format!("(could not list files: {e})"));

    let user_content = format!(
        "Workspace root: {}\n\nInitial file listing:\n{}\n\nTask:\n{}",
        cwd.display(),
        initial_listing,
        task
    );

    let mut messages = vec![
        ChatMessage {
            role: "system".to_string(),
            content: SYSTEM_PROMPT.to_string(),
        },
        ChatMessage {
            role: "user".to_string(),
            content: user_content,
        },
    ];

    let mut changed_files: Vec<String> = Vec::new();

    for step in 1..=config.max_steps {
        reporter.line(format!(
            "\n{BOLD}{CYAN}🦇 Step {step}/{max}{RESET} — model: {BOLD}{model}{RESET}",
            BOLD = BOLD,
            CYAN = CYAN,
            RESET = RESET,
            step = step,
            max = config.max_steps,
            model = model
        ));

        let req = ChatRequest {
            model: model.to_string(),
            messages: messages.clone(),
            stream: false,
            format: Some(Value::String("json".to_string())),
            options: Some(serde_json::json!({
                "temperature": 0.2,
                "num_ctx": num_ctx,
            })),
        };

        let mut stream = Box::pin(
            client
                .chat_stream(req)
                .await
                .with_context(|| format!("model call failed at step {step}"))?,
        );

        let mut buffer = String::new();
        let mut thought_prefix_printed = false;
        let mut thought_printed_len = 0usize;
        let mut thought_complete = false;
        let mut answer_prefix_printed = false;
        let mut answer_printed_len = 0usize;
        let mut answer_complete = false;
        let mut answer_skipped = false;

        while let Some(delta) = stream.next().await {
            let delta = delta?;
            buffer.push_str(&delta);

            // If a top-level tool call exists before an answer key, don't try to
            // stream an "answer" that might be inside tool arguments.
            if !answer_skipped {
                if let (Some(a_pos), Some(t_pos)) = (
                    find_key_pos(&buffer, "answer"),
                    find_key_pos(&buffer, "tool"),
                ) {
                    if t_pos < a_pos {
                        answer_skipped = true;
                    }
                }
            }

            if !thought_complete {
                if let Some((text, complete)) = extract_json_string(&buffer, "thought") {
                    if !thought_prefix_printed && !text.is_empty() {
                        reporter.chunk(format!("{}🧠 {}", CYAN, RESET));
                        thought_prefix_printed = true;
                    }
                    if text.len() > thought_printed_len {
                        reporter.chunk(text[thought_printed_len..].to_string());
                        thought_printed_len = text.len();
                    }
                    if complete {
                        thought_complete = true;
                        reporter.line(String::new());
                    }
                }
            }

            if !answer_skipped && !answer_complete {
                if let Some((text, complete)) = extract_json_string(&buffer, "answer") {
                    if !answer_prefix_printed && !text.is_empty() {
                        reporter.chunk(format!("\n\x1b[32m✅\x1b[0m "));
                        answer_prefix_printed = true;
                    }
                    if text.len() > answer_printed_len {
                        reporter.chunk(text[answer_printed_len..].to_string());
                        answer_printed_len = text.len();
                    }
                    if complete {
                        answer_complete = true;
                        reporter.line(String::new());
                    }
                }
            }
        }

        let content = buffer.trim().to_string();
        let parsed = match parse_agent_response(&content) {
            Ok(p) => p,
            Err(e) => {
                reporter.line(format!("\n⚠️  Could not parse model JSON: {e}"));
                reporter.line("Showing raw model output as the final response.".to_string());
                reporter.line(format!("\n{content}"));
                return Ok(());
            }
        };

        if let Some(answer) = parsed.answer {
            if !answer_prefix_printed {
                reporter.line(format!("\n\x1b[32m✅\x1b[0m {answer}"));
            }
            print_changed_files(&cwd, &changed_files, reporter);
            return Ok(());
        }

        if let Some(tool_call) = parsed.tool {
            // finish is a direct final response.
            if tool_call.name == "finish" {
                let summary = get_str(&tool_call.arguments, "summary")?.unwrap_or("done");
                reporter.line(format!("\n\x1b[32m✅\x1b[0m {summary}"));
                print_changed_files(&cwd, &changed_files, reporter);
                return Ok(());
            }

            // Keep the raw JSON in history so the model sees exactly what it sent.
            messages.push(ChatMessage {
                role: "assistant".to_string(),
                content: content.clone(),
            });

            reporter.line(format!(
                "\n{}🔧 {}{}",
                MAGENTA,
                describe_tool(&tool_call),
                RESET
            ));
            let result = execute_tool(config, &cwd, &tool_call, &mut changed_files).await;
            let result_text = match result {
                Ok(text) => text,
                Err(e) => format!("Tool error: {e:#}"),
            };
            reporter.line(format!("{}{}{}", DIM, result_text, RESET));

            messages.push(ChatMessage {
                role: "user".to_string(),
                content: format!("Tool '{}' result:\n{}", tool_call.name, result_text),
            });

            // Keep context bounded: drop the oldest tool exchange if history grows too large.
            trim_messages(&mut messages, 40);
            continue;
        }

        reporter.line("\n⚠️  Model returned neither an answer nor a tool call.".to_string());
        return Ok(());
    }

    reporter.line(format!(
        "\n⚠️  Reached max steps ({}) without a final answer.",
        config.max_steps
    ));
    Ok(())
}

#[allow(clippy::too_many_arguments)]
async fn execute_tool(
    config: &AgentConfig,
    cwd: &Path,
    tool: &ToolCall,
    changed_files: &mut Vec<String>,
) -> Result<String> {
    let name = tool.name.as_str();
    let args = &tool.arguments;

    match name {
        "list_files" => {
            let path = get_str(args, "path")?.unwrap_or(".");
            tools::list_files(cwd, path, 5)
        }
        "read_file" => {
            let path =
                get_str(args, "path")?.ok_or_else(|| anyhow!("read_file requires 'path'"))?;
            let max_chars = get_u64(args, "max_chars")?.unwrap_or(8000) as usize;
            let max_chars = max_chars.min(tools::MAX_TOOL_OUTPUT);
            tools::read_file(cwd, path, max_chars)
        }
        "grep_files" => {
            let pattern = get_str(args, "pattern")?
                .ok_or_else(|| anyhow!("grep_files requires 'pattern'"))?;
            let path = get_str(args, "path")?.unwrap_or(".");
            let max_results = get_u64(args, "max_results")?.unwrap_or(200) as usize;
            tools::grep_files(cwd, pattern, path, max_results)
        }
        "write_file" => {
            let path =
                get_str(args, "path")?.ok_or_else(|| anyhow!("write_file requires 'path'"))?;
            let content = get_str(args, "content")?
                .ok_or_else(|| anyhow!("write_file requires 'content'"))?;
            if config.read_only {
                bail!("write_file is disabled in read-only mode");
            }
            confirm_or_abort(config, &format!("write file '{path}'?"))?;
            let result = tools::write_file(cwd, path, content)?;
            changed_files.push(path.to_string());
            Ok(format!("{result}\n📎 {}", clickable_path(cwd, path)))
        }
        "run_command" => {
            let command = get_str(args, "command")?
                .ok_or_else(|| anyhow!("run_command requires 'command'"))?;
            if config.read_only {
                bail!("run_command is disabled in read-only mode");
            }
            confirm_or_abort(config, &format!("run command: {command}"))?;
            tools::run_command(cwd, command, 120).await
        }
        "finish" => {
            let summary = get_str(args, "summary")?.unwrap_or("done");
            Ok(format!("✅ {summary}"))
        }
        _ => bail!("unknown tool: {name}"),
    }
}

/// Human-readable one-line description of what a tool call is about to do.
fn describe_tool(tool: &ToolCall) -> String {
    let name = tool.name.as_str();
    let args = &tool.arguments;
    match name {
        "write_file" => {
            let path = get_str(args, "path").ok().flatten().unwrap_or("?");
            format!("write_file → {path}")
        }
        "read_file" => {
            let path = get_str(args, "path").ok().flatten().unwrap_or("?");
            format!("read_file → {path}")
        }
        "grep_files" => {
            let pattern = get_str(args, "pattern").ok().flatten().unwrap_or("?");
            let path = get_str(args, "path").ok().flatten().unwrap_or(".");
            format!("grep_files → {pattern:?} in {path}")
        }
        "list_files" => {
            let path = get_str(args, "path").ok().flatten().unwrap_or(".");
            format!("list_files → {path}")
        }
        "run_command" => {
            let cmd = get_str(args, "command").ok().flatten().unwrap_or("?");
            format!("run_command → {cmd}")
        }
        "finish" => "finish".to_string(),
        other => other.to_string(),
    }
}

fn get_str<'a>(args: &'a Value, key: &str) -> Result<Option<&'a str>> {
    match args.get(key) {
        Some(Value::String(s)) => Ok(Some(s)),
        Some(Value::Null) | None => Ok(None),
        Some(_) => bail!("argument '{key}' must be a string"),
    }
}

fn get_u64(args: &Value, key: &str) -> Result<Option<u64>> {
    match args.get(key) {
        Some(Value::Number(n)) => n
            .as_u64()
            .map(Some)
            .ok_or_else(|| anyhow!("argument '{key}' must be a non-negative integer")),
        Some(Value::Null) | None => Ok(None),
        Some(_) => bail!("argument '{key}' must be a number"),
    }
}

fn confirm_or_abort(config: &AgentConfig, prompt: &str) -> Result<()> {
    if !config.confirm {
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
    let resp: AgentResponse = serde_json::from_value(value)
        .with_context(|| format!("JSON did not match agent schema: {content}"))?;
    Ok(resp)
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

fn find_key_pos(buf: &str, key: &str) -> Option<usize> {
    buf.find(&format!("\"{key}\""))
}

/// Extract the current value of a JSON string field from a partial JSON buffer.
/// Returns `(value_so_far, is_complete)`.
fn extract_json_string(buf: &str, key: &str) -> Option<(String, bool)> {
    let key_pattern = format!("\"{key}\"");
    let start = buf.find(&key_pattern)?;
    let after_key = &buf[start + key_pattern.len()..];
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
    for f in files {
        reporter.line(format!("   {}", clickable_path(cwd, f)));
    }
}
