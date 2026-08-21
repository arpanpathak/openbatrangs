use crate::ollama::{ChatMessage, ChatRequest, OllamaClient};
use crate::tools;
use anyhow::{anyhow, bail, Context, Result};
use serde::Deserialize;
use serde_json::Value;
use std::io::Write;
use std::path::PathBuf;

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

pub async fn run_agent(
    config: &AgentConfig,
    client: &OllamaClient,
    model: &str,
    model_context: u64,
    task: &str,
) -> Result<()> {
    let cwd = config.cwd.clone();
    if !cwd.is_dir() {
        bail!("working directory does not exist: {}", cwd.display());
    }

    let num_ctx = model_context.min(32768).max(4096) as u64;
    let initial_listing = tools::list_files(&cwd, ".", 5).unwrap_or_else(|e| format!("(could not list files: {e})"));

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

    for step in 1..=config.max_steps {
        println!("\n🧠 Step {step}/{} — model: {model}", config.max_steps);

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

        let resp = client
            .chat(req)
            .await
            .with_context(|| format!("model call failed at step {step}"))?;
        let content = resp.message.content.trim().to_string();

        println!("🤖 {content}");

        let parsed = match parse_agent_response(&content) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("\n⚠️  Could not parse model JSON: {e}");
                eprintln!("Showing raw model output as the final response.");
                println!("\n{content}");
                return Ok(());
            }
        };

        if let Some(answer) = parsed.answer {
            println!("\n✅ {answer}");
            return Ok(());
        }

        if let Some(tool_call) = parsed.tool {
            // finish is a direct final response.
            if tool_call.name == "finish" {
                let summary = get_str(&tool_call.arguments, "summary")?.unwrap_or("done");
                println!("\n✅ {summary}");
                return Ok(());
            }

            // Keep the raw JSON in history so the model sees exactly what it sent.
            messages.push(ChatMessage {
                role: "assistant".to_string(),
                content: content.clone(),
            });

            let result = execute_tool(config, &cwd, &tool_call).await;
            let result_text = match result {
                Ok(text) => text,
                Err(e) => format!("Tool error: {e:#}"),
            };

            println!("\n🔧 Tool '{}' ->", tool_call.name);
            println!("{}", result_text);

            messages.push(ChatMessage {
                role: "user".to_string(),
                content: format!(
                    "Tool '{}' result:\n{}",
                    tool_call.name, result_text
                ),
            });

            // Keep context bounded: drop the oldest tool exchange if history grows too large.
            trim_messages(&mut messages, 40);
            continue;
        }

        eprintln!("\n⚠️  Model returned neither an answer nor a tool call.");
        return Ok(());
    }

    eprintln!("\n⚠️  Reached max steps ({}) without a final answer.", config.max_steps);
    Ok(())
}

async fn execute_tool(config: &AgentConfig, cwd: &std::path::Path, tool: &ToolCall) -> Result<String> {
    let name = tool.name.as_str();
    let args = &tool.arguments;

    match name {
        "list_files" => {
            let path = get_str(args, "path")?.unwrap_or(".");
            tools::list_files(cwd, path, 5)
        }
        "read_file" => {
            let path = get_str(args, "path")?.ok_or_else(|| anyhow!("read_file requires 'path'"))?;
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
            let path = get_str(args, "path")?
                .ok_or_else(|| anyhow!("write_file requires 'path'"))?;
            let content = get_str(args, "content")?
                .ok_or_else(|| anyhow!("write_file requires 'content'"))?;
            if config.read_only {
                bail!("write_file is disabled in read-only mode");
            }
            confirm_or_abort(config, &format!("write file '{path}'?"))?;
            tools::write_file(cwd, path, content)
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

fn get_str<'a>(args: &'a Value, key: &str) -> Result<Option<&'a str>> {
    match args.get(key) {
        Some(Value::String(s)) => Ok(Some(s)),
        Some(Value::Null) | None => Ok(None),
        Some(_) => bail!("argument '{key}' must be a string"),
    }
}

fn get_u64(args: &Value, key: &str) -> Result<Option<u64>> {
    match args.get(key) {
        Some(Value::Number(n)) => n.as_u64().map(Some).ok_or_else(|| anyhow!("argument '{key}' must be a non-negative integer")),
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
