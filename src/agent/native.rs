//! # Native tool-calling agent loop
//!
//! OpenAI-compatible servers — llama-server in router mode and TensorRT
//! Edge-LLM — accept a `tools` array and reply with structured `tool_calls`.
//! This loop sends the schemas, streams the reply, executes every call, feeds
//! the `tool` results back, and treats a reply without calls as the answer.
//!
//! Tool execution, confirmation, and the file-work gate are shared with the
//! JSON loop in [`super`], so writes and commands behave the same either way.
//!
//! ## References
//!
//! - OpenAI function calling: <https://platform.openai.com/docs/guides/function-calling>
//! - llama.cpp server, router mode: <https://github.com/ggml-org/llama.cpp/tree/master/tools/server>

use super::execute::{execute_tool_or_report_error, print_changed_files};
use super::quality::{requires_file_work, FileWorkGate};
use super::tool::{Tool, ToolCall};
use super::{effective_context, AgentConfig, Confirmer, Reporter};
use crate::constants::agent::{
    AGENT_TEMPERATURE, CHARS_PER_TOKEN, CONTEXT_RESERVE_TOKENS, MAX_HISTORY_MESSAGES,
    MAX_OUTPUT_TOKENS, MIN_OUTPUT_TOKENS, NATIVE_SYSTEM_PROMPT,
};
use crate::ollama::{Delta, OllamaClient, ToolCallDelta};
use anyhow::{Context, Result};
use futures_util::StreamExt;
use serde_json::{json, Value};

/// Whether the loop should keep going after one model turn.
enum Progress {
    /// The model answered; stop.
    Done,
    /// A tool ran, or a shallow finish was rejected; take another step.
    More,
}

/// One message in the native conversation, mapped to the OpenAI chat shape.
#[derive(Debug, Clone)]
enum Message {
    /// System instructions.
    System(String),
    /// A user turn: the task, or feedback we inject.
    User(String),
    /// An assistant turn, optionally requesting tool calls.
    Assistant {
        /// Assistant text; often empty when the turn only calls tools.
        content: String,
        /// Tool calls the assistant requested.
        calls: Vec<Invocation>,
    },
    /// The result of one tool call.
    Tool {
        /// Id of the call this result answers.
        id: String,
        /// Result text handed back to the model.
        content: String,
    },
}

impl Message {
    /// Render this message in the OpenAI `/chat/completions` shape.
    fn to_json(&self) -> Value {
        match self {
            Self::System(text) => json!({ "role": "system", "content": text }),
            Self::User(text) => json!({ "role": "user", "content": text }),
            Self::Assistant { content, calls } if calls.is_empty() => {
                json!({ "role": "assistant", "content": content })
            }
            Self::Assistant { content, calls } => json!({
                "role": "assistant",
                "content": content,
                "tool_calls": calls.iter().map(Invocation::to_json).collect::<Vec<_>>(),
            }),
            Self::Tool { id, content } => {
                json!({ "role": "tool", "tool_call_id": id, "content": content })
            }
        }
    }

    /// Approximate character size, used to budget output tokens.
    fn char_len(&self) -> usize {
        match self {
            Self::System(text) | Self::User(text) => text.len(),
            Self::Assistant { content, calls } => {
                content.len() + calls.iter().map(Invocation::char_len).sum::<usize>()
            }
            Self::Tool { content, .. } => content.len(),
        }
    }
}

/// A fully assembled tool call, ready to execute.
#[derive(Debug, Clone, Default)]
struct Invocation {
    /// Call id from the server; synthesized when the server omits it.
    id: String,
    /// Tool (function) name.
    name: String,
    /// JSON-encoded arguments.
    arguments: String,
}

impl Invocation {
    /// Render this invocation in the OpenAI `tool_calls` item shape.
    fn to_json(&self) -> Value {
        json!({
            "id": self.id,
            "type": "function",
            "function": { "name": self.name, "arguments": self.arguments },
        })
    }

    /// Character size of the name and arguments.
    fn char_len(&self) -> usize {
        self.name.len() + self.arguments.len()
    }
}

/// Accumulates the streamed text and tool-call fragments of one model turn.
#[derive(Debug, Default)]
struct Turn {
    /// Assistant text streamed so far.
    content: String,
    /// Tool calls, assembled by their streamed `index`.
    calls: Vec<Invocation>,
}

impl Turn {
    /// Merge one incremental tool-call chunk into the turn.
    fn absorb(&mut self, delta: ToolCallDelta) {
        let index = delta.index as usize;

        if self.calls.len() <= index {
            self.calls.resize_with(index + 1, Invocation::default);
        }

        let call = &mut self.calls[index];
        if let Some(id) = delta.id.filter(|id| !id.is_empty()) {
            call.id = id;
        }
        if let Some(name) = delta.name.filter(|name| !name.is_empty()) {
            call.name = name;
        }
        call.arguments.push_str(&delta.arguments);
    }
}

/// One run of the native loop, holding the conversation and its limits.
struct Run<'a, R, C> {
    /// Agent runtime configuration.
    config: &'a AgentConfig,
    /// OpenAI-compatible client.
    client: &'a OllamaClient,
    /// Model name to request.
    model: &'a str,
    /// Context window, in tokens.
    num_ctx: u64,
    /// Whether the model is asked to think, from `--no-think` on the client.
    think: bool,
    /// Whether the task asks for file changes.
    needs_files: bool,
    /// Conversation sent to the model.
    messages: Vec<Message>,
    /// Files changed by tools so far.
    changed: Vec<String>,
    /// Gate that rejects shallow finishes.
    gate: FileWorkGate,
    /// Output sink.
    reporter: &'a mut R,
    /// Human-in-the-loop gate for writes and commands.
    confirmer: &'a mut C,
}

impl<R, C> Run<'_, R, C>
where
    R: Reporter,
    C: Confirmer,
{
    /// Run until the model answers or the step budget is spent.
    async fn run(mut self) -> Result<()> {
        for step in 1..=self.config.max_steps {
            if let Progress::Done = self.step(step).await? {
                return Ok(());
            }
        }
        self.reporter.line(format!(
            "\n⚠️  Reached max steps ({}) without a final answer.",
            self.config.max_steps
        ));
        Ok(())
    }

    /// Stream one model turn and act on it.
    async fn step(&mut self, step: usize) -> Result<Progress> {
        self.reporter
            .line(step_header(step, self.config.max_steps, self.model));
        trim_messages(&mut self.messages, MAX_HISTORY_MESSAGES);
        let body = request_body(self.model, &self.messages, self.num_ctx, self.think);
        let turn = stream_turn(self.client, body, self.reporter, self.config.show_thinking)
            .await
            .with_context(|| format!("model call failed at step {step}"))?;
        match turn.calls.as_slice() {
            [] => Ok(self.answer(turn.content)),
            _ => self.tools(turn).await,
        }
    }

    /// Treat a turn without tool calls as the final answer.
    fn answer(&mut self, text: String) -> Progress {
        let answer = text.trim();
        if answer.is_empty() {
            self.reporter
                .line("\n⚠️  The model returned an empty reply.".to_string());
            return Progress::Done;
        }
        if self.gate_blocks() {
            self.messages.push(Message::Assistant {
                content: answer.to_string(),
                calls: Vec::new(),
            });
            return self.blocked();
        }
        self.reporter.line(format!("\n✅ {answer}"));
        print_changed_files(&self.config.cwd, &self.changed, self.reporter);
        Progress::Done
    }

    /// Execute the tool calls of one turn and decide what happens next.
    async fn tools(&mut self, turn: Turn) -> Result<Progress> {
        self.messages.push(Message::Assistant {
            content: turn.content.clone(),
            calls: turn.calls.clone(),
        });
        let Some(summary) = self.execute(&turn.calls).await else {
            return Ok(Progress::More);
        };
        if self.gate_blocks() {
            return Ok(self.blocked());
        }
        self.reporter.line(format!("\n✅ {summary}"));
        print_changed_files(&self.config.cwd, &self.changed, self.reporter);
        Ok(Progress::Done)
    }

    /// Run every call, returning the summary when one of them was `finish`.
    async fn execute(&mut self, calls: &[Invocation]) -> Option<String> {
        let mut summary = None;
        for call in calls {
            summary = self.execute_one(call).await.or(summary);
        }
        summary
    }

    /// Run one call and append its result to the conversation.
    async fn execute_one(&mut self, call: &Invocation) -> Option<String> {
        let id = call_id(call);
        let tool = match parse_call(call) {
            Ok(tool) => tool,
            Err(error) => return self.reject(id, &error),
        };
        match tool {
            Tool::Finish { summary } => {
                self.messages.push(Message::Tool {
                    id,
                    content: "acknowledged".to_string(),
                });
                Some(summary)
            }
            tool => {
                self.reporter.line(format!("\n🔧 {}", tool.describe()));
                let result = execute_tool_or_report_error(
                    self.config,
                    &self.config.cwd,
                    &tool,
                    &mut self.changed,
                    self.confirmer,
                )
                .await;
                self.reporter.line(result.clone());
                self.messages.push(Message::Tool {
                    id,
                    content: result,
                });
                None
            }
        }
    }

    /// Record a rejected call so the model can retry it.
    fn reject(&mut self, id: String, error: &anyhow::Error) -> Option<String> {
        self.reporter
            .line(format!("\n⚠️  Tool call rejected: {error}"));
        self.messages.push(Message::Tool {
            id,
            content: format!("Error: {error}. Retry with the exact argument names."),
        });
        None
    }

    /// True and recorded when a finish must be rejected for lack of file work.
    fn gate_blocks(&mut self) -> bool {
        let blocked = self.gate.should_block_finish(
            self.needs_files,
            self.config.is_read_only,
            self.changed.len(),
        );
        if blocked {
            self.gate.record_retry();
        }
        blocked
    }

    /// Tell the model to keep working and take another step.
    fn blocked(&mut self) -> Progress {
        self.messages
            .push(Message::User(FileWorkGate::feedback().to_string()));
        self.reporter
            .line("\n⚠️  Finish blocked: no files were written yet.".to_string());
        Progress::More
    }
}

/// Run the native tool-calling loop until the model answers or steps run out.
///
/// # Arguments
/// - `config`: agent runtime configuration.
/// - `client`: an OpenAI-compatible client.
/// - `model`: model name to request.
/// - `model_context`: the model's native context window, in tokens.
/// - `task`: the user's task.
/// - `reporter`: output sink.
/// - `confirmer`: human-in-the-loop gate for writes and commands.
///
/// # Returns
/// `Ok(())` when the agent finishes, or an error if a model call fails.
pub(super) async fn run_native_agent<R, C>(
    config: &AgentConfig,
    client: &OllamaClient,
    model: &str,
    model_context: u64,
    task: &str,
    reporter: &mut R,
    confirmer: &mut C,
) -> Result<()>
where
    R: Reporter,
    C: Confirmer,
{
    let think = client.requests_thinking();
    let run = Run {
        config,
        client,
        model,
        num_ctx: effective_context(model_context, config.max_ctx),
        think,
        needs_files: requires_file_work(task),
        messages: vec![
            Message::System(NATIVE_SYSTEM_PROMPT.to_string()),
            Message::User(initial_user_message(config, task, think)),
        ],
        changed: Vec::new(),
        gate: FileWorkGate::default(),
        reporter,
        confirmer,
    };
    run.run().await
}

/// The user turn that opens the conversation.
///
/// A model told not to think stops honouring rules that only live in the
/// system turn: with `enable_thinking: false`, Nemotron-3-Nano writes to the
/// workspace root although the system prompt says to use `out/`, and it obeys
/// again the moment the same words arrive in the user turn. So the task
/// carries the instructions whenever thinking is off.
fn initial_user_message(config: &AgentConfig, task: &str, think: bool) -> String {
    let task = format!("Workspace root: {}\n\nTask:\n{task}", config.cwd.display());
    match think {
        true => task,
        false => format!("Instructions (highest priority):\n{NATIVE_SYSTEM_PROMPT}\n\n{task}"),
    }
}

/// Header printed before each model turn.
fn step_header(step: usize, max_steps: usize, model: &str) -> String {
    format!("\n── step {step}/{max_steps} ── {model}")
}

/// The id for a call: the server's, or a synthesized one when it omitted it.
fn call_id(call: &Invocation) -> String {
    match call.id.as_str() {
        "" => format!("call_{}", call.name),
        id => id.to_string(),
    }
}

/// Turn a streamed invocation into a typed [`Tool`].
fn parse_call(call: &Invocation) -> Result<Tool> {
    Tool::from_call(ToolCall {
        name: call.name.clone(),
        arguments: parse_arguments(&call.arguments),
    })
}

/// Parse a tool call's JSON arguments; empty or invalid input becomes `{}`.
fn parse_arguments(raw: &str) -> Value {
    match raw.trim() {
        "" => json!({}),
        raw => serde_json::from_str(raw).unwrap_or_else(|_| json!({})),
    }
}

/// Build the `/chat/completions` body for one turn.
fn request_body(model: &str, messages: &[Message], num_ctx: u64, think: bool) -> Value {
    let prompt = messages.iter().map(Message::char_len).sum::<usize>();
    let prompt_tokens = (prompt as f64 / CHARS_PER_TOKEN).ceil() as u64;
    let max_tokens = num_ctx
        .saturating_sub(prompt_tokens)
        .saturating_sub(CONTEXT_RESERVE_TOKENS)
        .clamp(MIN_OUTPUT_TOKENS, MAX_OUTPUT_TOKENS);
    json!({
        "model": model,
        "messages": messages.iter().map(Message::to_json).collect::<Vec<_>>(),
        "tools": tools_schema(),
        "tool_choice": "auto",
        "temperature": AGENT_TEMPERATURE,
        "max_tokens": max_tokens,
        "stream": true,
        "chat_template_kwargs": { "enable_thinking": think },
    })
}

/// Stream one model turn, forwarding text to the reporter.
async fn stream_turn<R>(
    client: &OllamaClient,
    body: Value,
    reporter: &mut R,
    show_thinking: bool,
) -> Result<Turn>
where
    R: Reporter,
{
    let mut stream = client.chat_stream_body(body).await?;
    let mut turn = Turn::default();
    let mut in_thinking = false;

    while let Some(delta) = stream.next().await {
        match delta? {
            Delta::Thinking(text) if show_thinking => {
                if !in_thinking {
                    reporter.chunk("💭 ");
                    in_thinking = true;
                }
                reporter.chunk(&text);
            }
            Delta::Thinking(_) => {}
            Delta::Content(text) => {
                if in_thinking {
                    reporter.chunk("\n");
                    in_thinking = false;
                }
                turn.content.push_str(&text);
                reporter.chunk(&text);
            }
            Delta::ToolCalls(calls) => {
                if in_thinking {
                    reporter.chunk("\n");
                    in_thinking = false;
                }
                for call in calls {
                    turn.absorb(call);
                }
            }
        }
    }
    if in_thinking {
        reporter.chunk("\n");
    }
    Ok(turn)
}

/// Drop the oldest exchanges, keeping the system prompt and initial task.
fn trim_messages(messages: &mut Vec<Message>, max: usize) {
    if messages.len() <= max || max < 2 {
        return;
    }
    let system = messages.remove(0);
    let task = messages.remove(0);
    let mut tail = messages.split_off(messages.len().saturating_sub(max - 2));
    while matches!(tail.first(), Some(Message::Tool { .. })) {
        tail.remove(0);
    }
    messages.clear();
    messages.push(system);
    messages.push(task);
    messages.extend(tail);
}

/// JSON schemas for the tools offered to the model.
fn tools_schema() -> Value {
    json!([
        {
            "type": "function",
            "function": {
                "name": "list_files",
                "description": "List files under a directory, recursively, skipping build and cache directories.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": { "type": "string", "description": "Directory relative to the workspace root; defaults to \".\"." }
                    },
                    "required": []
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a text file, truncated to max_chars characters.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": { "type": "string", "description": "File path relative to the workspace root." },
                        "max_chars": { "type": "integer", "description": "Maximum characters to return; defaults to 8000." }
                    },
                    "required": ["path"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "grep_files",
                "description": "Search file contents with a regular expression.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pattern": { "type": "string", "description": "Regular expression to match on each line." },
                        "path": { "type": "string", "description": "Directory to search; defaults to \".\"." },
                        "max_results": { "type": "integer", "description": "Maximum matches to return; defaults to 200." }
                    },
                    "required": ["pattern"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "write_file",
                "description": "Create a file or overwrite it completely. Parent directories are created as needed.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": { "type": "string", "description": "File path relative to the workspace root." },
                        "content": { "type": "string", "description": "Complete file contents." }
                    },
                    "required": ["path", "content"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "edit_file",
                "description": "Replace a literal string inside an existing file. Prefer this over write_file for small changes.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": { "type": "string", "description": "File path relative to the workspace root." },
                        "old_string": { "type": "string", "description": "Exact text to find; must be unique unless replace_all is true." },
                        "new_string": { "type": "string", "description": "Replacement text." },
                        "replace_all": { "type": "boolean", "description": "Replace every occurrence instead of requiring a unique match." }
                    },
                    "required": ["path", "old_string", "new_string"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "run_command",
                "description": "Run a shell command in the workspace and return its combined output.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": { "type": "string", "description": "Shell command line, run with bash -lc." }
                    },
                    "required": ["command"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "finish",
                "description": "Signal that the task is complete. Prefer replying in plain text instead.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "summary": { "type": "string", "description": "Short summary of what was done." }
                    },
                    "required": ["summary"]
                }
            }
        }
    ])
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn assembles_tool_call_fragments_by_index() {
        let mut turn = Turn::default();
        turn.absorb(ToolCallDelta {
            index: 0,
            id: Some("call_a".to_string()),
            name: Some("read_file".to_string()),
            arguments: String::new(),
        });
        turn.absorb(ToolCallDelta {
            index: 0,
            arguments: "{\"path\":".to_string(),
            ..ToolCallDelta::default()
        });
        turn.absorb(ToolCallDelta {
            index: 0,
            arguments: "\"a.rs\"}".to_string(),
            ..ToolCallDelta::default()
        });
        assert_eq!(turn.calls.len(), 1);
        assert_eq!(turn.calls[0].id, "call_a");
        assert_eq!(turn.calls[0].name, "read_file");
        assert_eq!(turn.calls[0].arguments, "{\"path\":\"a.rs\"}");
    }

    #[test]
    fn parallel_tool_calls_keep_their_index() {
        let mut turn = Turn::default();
        turn.absorb(ToolCallDelta {
            index: 1,
            name: Some("run_command".to_string()),
            arguments: "{}".to_string(),
            ..ToolCallDelta::default()
        });
        assert_eq!(turn.calls.len(), 2);
        assert_eq!(turn.calls[1].name, "run_command");
        assert_eq!(turn.calls[0].name, "");
    }

    #[test]
    fn a_piece_of_an_earlier_call_keeps_the_later_ones() {
        let mut turn = Turn::default();
        let piece = |index, arguments: &str| ToolCallDelta {
            index,
            arguments: arguments.to_string(),
            ..ToolCallDelta::default()
        };

        turn.absorb(piece(0, "{\"path\":"));
        turn.absorb(piece(1, "{}"));
        turn.absorb(piece(0, "\"a.rs\"}"));

        assert_eq!(turn.calls.len(), 2);
        assert_eq!(turn.calls[0].arguments, "{\"path\":\"a.rs\"}");
        assert_eq!(turn.calls[1].arguments, "{}");
    }

    #[test]
    fn trim_keeps_system_and_task_and_drops_orphan_results() {
        let mut messages = vec![
            Message::System("system".to_string()),
            Message::User("task".to_string()),
        ];
        for index in 0..30 {
            messages.push(Message::Assistant {
                content: String::new(),
                calls: vec![Invocation {
                    id: format!("c{index}"),
                    name: "read_file".to_string(),
                    arguments: "{}".to_string(),
                }],
            });
            messages.push(Message::Tool {
                id: format!("c{index}"),
                content: "x".to_string(),
            });
        }
        trim_messages(&mut messages, 6);
        assert_eq!(messages[0].to_json()["role"].as_str(), Some("system"));
        assert_eq!(messages[1].to_json()["role"].as_str(), Some("user"));
        assert!(matches!(messages[2], Message::Assistant { .. }));
        assert!(messages.len() <= 6);
    }

    #[test]
    fn tool_schema_offers_edit_and_command_tools() {
        let names: Vec<String> = tools_schema()
            .as_array()
            .expect("schema is an array")
            .iter()
            .filter_map(|tool| tool.pointer("/function/name").and_then(Value::as_str))
            .map(str::to_string)
            .collect();
        assert!(names.contains(&"edit_file".to_string()));
        assert!(names.contains(&"run_command".to_string()));
        assert!(names.contains(&"read_file".to_string()));
    }

    #[test]
    fn invalid_arguments_become_an_empty_object() {
        assert_eq!(parse_arguments("not json"), json!({}));
        assert_eq!(parse_arguments(""), json!({}));
        assert_eq!(parse_arguments(" {\"path\":\"a\"} "), json!({"path": "a"}));
    }

    #[test]
    fn tool_result_messages_carry_the_call_id() {
        let message = Message::Tool {
            id: "call_1".to_string(),
            content: "ok".to_string(),
        };
        assert_eq!(message.to_json()["role"].as_str(), Some("tool"));
        assert_eq!(message.to_json()["tool_call_id"].as_str(), Some("call_1"));
    }

    fn config() -> AgentConfig {
        AgentConfig {
            cwd: PathBuf::from("/tmp/workspace"),
            max_steps: 4,
            is_read_only: false,
            should_confirm: false,
            show_thinking: false,
            max_ctx: None,
        }
    }

    #[test]
    fn the_task_carries_the_rules_only_when_thinking_is_off() {
        let config = config();
        let thinking = initial_user_message(&config, "write it", true);
        assert!(thinking.starts_with("Workspace root: /tmp/workspace"));
        assert!(!thinking.contains(NATIVE_SYSTEM_PROMPT));

        let not_thinking = initial_user_message(&config, "write it", false);
        assert!(not_thinking.contains(NATIVE_SYSTEM_PROMPT));
        assert!(not_thinking.ends_with("Task:\nwrite it"));
    }

    #[test]
    fn the_request_carries_the_thinking_flag_it_was_given() {
        let messages = vec![Message::System("rules".to_string())];
        let flag = |think| {
            request_body("m", &messages, 8192, think)["chat_template_kwargs"]["enable_thinking"]
                .clone()
        };
        assert_eq!(flag(false), json!(false));
        assert_eq!(flag(true), json!(true));
    }
}
