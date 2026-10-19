//! # Native tool-calling agent loop
//!
//! OpenAI-compatible servers — llama-server in router mode and TensorRT
//! Edge-LLM — accept a `tools` array and reply with structured `tool_calls`.
//! This loop sends the schemas with `tool_choice: "required"`, streams the
//! reply, executes every call, and feeds the `tool` results back. Served
//! models that answer with their JSON tool call as plain text instead of the
//! structured channel are detected and executed anyway; anything else is the
//! answer.
//!
//! Tool execution, confirmation, and the file-work gate are shared with the
//! JSON loop in [`super`], so writes and commands behave the same either way.
//!
//! ## References
//!
//! - OpenAI function calling: <https://platform.openai.com/docs/guides/function-calling>
//! - llama.cpp server, router mode: <https://github.com/ggml-org/llama.cpp/tree/master/tools/server>

use super::environment::{environment_facts, project_has_build_manifest};
use super::execute::{execute_tool_or_report_error, print_changed_files};
use super::quality::{command_succeeded, requires_file_work, FileWorkGate};
use super::tool::{Tool, ToolCall};
use super::{effective_context, AgentConfig, Confirmer, HistorySink, Reporter, RunContinuation};
use crate::constants::agent::{
    AGENT_TEMPERATURE, CHARS_PER_TOKEN, CONTEXT_RESERVE_TOKENS, EMPTY_TURN_NUDGE,
    MAX_HISTORY_MESSAGES, MAX_OUTPUT_TOKENS, MIN_OUTPUT_TOKENS, NATIVE_SYSTEM_PROMPT,
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

    /// Rebuild a message from its OpenAI `/chat/completions` shape.
    ///
    /// Unknown roles and malformed entries are dropped, so a resumed
    /// conversation degrades gracefully instead of failing the request.
    fn from_json(value: &Value) -> Option<Self> {
        let role = value.get("role")?.as_str()?;
        let content = value
            .get("content")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();
        match role {
            "system" => Some(Self::System(content)),
            "user" => Some(Self::User(content)),
            "assistant" => {
                let calls = value
                    .get("tool_calls")
                    .and_then(Value::as_array)
                    .map(|calls| calls.iter().filter_map(Invocation::from_json).collect())
                    .unwrap_or_default();
                Some(Self::Assistant { content, calls })
            }
            "tool" => Some(Self::Tool {
                id: value
                    .get("tool_call_id")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string(),
                content,
            }),
            _ => None,
        }
    }

    /// Rebuild a stored conversation, dropping entries that do not parse;
    /// `None` when nothing survives, so the caller can start fresh instead.
    fn conversation(values: &[Value]) -> Option<Vec<Self>> {
        let messages: Vec<Self> = values.iter().filter_map(Self::from_json).collect();
        (!messages.is_empty()).then_some(messages)
    }

    /// The tool calls of this message, when it is an assistant turn.
    fn calls(&self) -> Option<&[Invocation]> {
        match self {
            Self::Assistant { calls, .. } => Some(calls),
            _ => None,
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

    /// Rebuild an invocation from an OpenAI `tool_calls` item.
    fn from_json(value: &Value) -> Option<Self> {
        let function = value.get("function")?;
        Some(Self {
            id: value
                .get("id")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string(),
            name: function
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string(),
            arguments: function
                .get("arguments")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string(),
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
    /// Whether a successful build/test is expected before the task counts as done.
    needs_verification: bool,
    /// Number of consecutive empty model turns already nudged to act.
    empty_retries: u8,
    /// Conversation sent to the model.
    messages: Vec<Message>,
    /// Files changed by tools so far.
    changed_files: Vec<String>,
    /// Gate that rejects shallow finishes.
    finish_gate: FileWorkGate,
    /// Sink receiving conversation snapshots, so an interrupted run can resume.
    sink: Option<&'a mut dyn HistorySink>,
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

    /// Stream one model turn, act on it, and snapshot the conversation.
    async fn step(&mut self, step: usize) -> Result<Progress> {
        self.reporter
            .line(step_header(step, self.config.max_steps, self.model));
        trim_messages(&mut self.messages, MAX_HISTORY_MESSAGES);
        let body = request_body(self.model, &self.messages, self.num_ctx, self.think);
        let turn = stream_turn(self.client, body, self.reporter, self.config.show_thinking)
            .await
            .with_context(|| format!("model call failed at step {step}"))?;

        let progress = self.act(turn).await?;
        self.save_conversation();
        Ok(progress)
    }

    /// Decide what one streamed turn means and act on it.
    async fn act(&mut self, turn: Turn) -> Result<Progress> {
        if !turn.calls.is_empty() {
            self.empty_retries = 0;
            return self.tools(turn).await;
        }

        if let Some(call) = parse_text_tool_call(&turn.content) {
            self.reporter.line(
                "\n⚠️ The model wrote its tool call as text; executing it anyway.".to_string(),
            );
            return self
                .tools(Turn {
                    calls: vec![call],
                    ..turn
                })
                .await;
        }

        if turn.content.trim().is_empty() {
            return Ok(self.handle_empty_turn());
        }

        Ok(self.answer(turn.content))
    }

    /// Nudge a silent model twice, then accept the silence as an answer.
    fn handle_empty_turn(&mut self) -> Progress {
        if self.empty_retries >= 2 {
            return self.answer(String::new());
        }

        self.empty_retries += 1;
        self.messages
            .push(Message::User(EMPTY_TURN_NUDGE.to_string()));
        self.reporter
            .line("\n⚠️ Empty model turn; prompting it to continue with tools.".to_string());
        Progress::More
    }

    /// Treat a turn without tool calls as the final answer.
    fn answer(&mut self, text: String) -> Progress {
        let answer = text.trim();
        if answer.is_empty() {
            self.reporter
                .line("\n⚠️  The model returned an empty reply.".to_string());
            return Progress::Done;
        }
        if let Some(feedback) = self.finish_feedback() {
            self.messages.push(Message::Assistant {
                content: answer.to_string(),
                calls: Vec::new(),
            });
            self.messages.push(Message::User(feedback.to_string()));
            self.reporter
                .line("\n⚠️  Finish rejected: the task is not done.".to_string());
            return Progress::More;
        }
        self.reporter.line(format!("\n✅ {answer}"));
        print_changed_files(&self.config.cwd, &self.changed_files, self.reporter);
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
        if let Some(feedback) = self.finish_feedback() {
            self.messages.push(Message::User(feedback.to_string()));
            self.reporter
                .line("\n⚠️  Finish rejected: the task is not done.".to_string());
            return Ok(Progress::More);
        }
        self.reporter.line(format!("\n✅ {summary}"));
        print_changed_files(&self.config.cwd, &self.changed_files, self.reporter);
        Ok(Progress::Done)
    }

    /// Run every call, returning the summary when one of them was `finish`.
    async fn execute(&mut self, calls: &[Invocation]) -> Option<String> {
        let mut summary = None;
        for call in calls {
            if let Some(found) = self.execute_one(call).await {
                summary = Some(found);
            }
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
                    &mut self.changed_files,
                    self.confirmer,
                )
                .await;
                if matches!(tool, Tool::RunCommand { .. }) && command_succeeded(&result) {
                    self.finish_gate.record_verification();
                }
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

    /// Reject a premature finish and return the feedback the model must hear.
    fn finish_feedback(&mut self) -> Option<&'static str> {
        self.finish_gate.reject_finish(
            self.needs_files,
            self.config.is_read_only,
            self.changed_files.len(),
            self.needs_verification,
        )
    }

    /// Send the current conversation to the snapshot sink, if one is attached.
    fn save_conversation(&mut self) {
        let Some(sink) = &mut self.sink else {
            return;
        };
        let snapshot: Vec<Value> = self.messages.iter().map(Message::to_json).collect();
        sink.save(snapshot);
    }
}

/// Run the native tool-calling loop until the model answers or steps run out.
///
/// # Arguments
/// - `config`: agent runtime configuration.
/// - `client`: an OpenAI-compatible client.
/// - `model`: model name to request.
/// - `task`: the user's task, or the correction when resuming.
/// - `continuation`: the interrupted run to continue and the snapshot sink;
///   `None` in one-shot mode.
/// - `reporter`: output sink.
/// - `confirmer`: human-in-the-loop gate for writes and commands.
///
/// # Returns
/// `Ok(())` when the agent finishes, or an error if a model call fails.
pub(super) async fn run_native_agent<'a, R, C>(
    config: &'a AgentConfig,
    client: &'a OllamaClient,
    model: &'a str,
    task: &str,
    continuation: Option<RunContinuation<'a>>,
    reporter: &'a mut R,
    confirmer: &'a mut C,
) -> Result<()>
where
    R: Reporter,
    C: Confirmer,
{
    let (resume, sink) = continuation
        .map(|run| (run.messages, run.sink))
        .unwrap_or((None, None));
    let think = client.requests_thinking();
    let resuming = resume.is_some();

    // A resumed run already carries its documentation in the conversation;
    // repeating the preflight would refetch the same pages for nothing.
    let docs = if resuming {
        None
    } else {
        super::research::documentation_context(task, &config.searxng_url, reporter).await
    };
    let task_with_docs = match docs {
        Some(docs) => format!("{task}\n\n{docs}\n\nThe bounded documentation preflight is complete. Use the supplied sources and proceed; do not repeat searches."),
        None => task.to_string(),
    };

    let messages = seed_messages(config, &task_with_docs, think, resume);
    let needs_files = messages
        .iter()
        .find_map(|message| match message {
            Message::User(text) => Some(requires_file_work(text)),
            _ => None,
        })
        .unwrap_or(false);

    // Carry the prior run's state forward so the gate does not demand work
    // that was already done before the interrupt.
    let changed_files = previously_changed_files(&messages);
    let finish_gate = gate_for_run(&messages);

    let mut run = Run {
        config,
        client,
        model,
        num_ctx: effective_context(config.model_context, config.max_ctx),
        think,
        needs_files,
        needs_verification: needs_files && project_has_build_manifest(&config.cwd),
        empty_retries: 0,
        messages,
        changed_files,
        finish_gate,
        sink,
        reporter,
        confirmer,
    };
    run.save_conversation();
    run.run().await
}

/// The opening messages: the stored conversation plus the new task when
/// resuming, or the fresh system + task pair otherwise.
fn seed_messages(
    config: &AgentConfig,
    task: &str,
    think: bool,
    resume: Option<Vec<Value>>,
) -> Vec<Message> {
    let Some(mut messages) = resume.as_deref().and_then(Message::conversation) else {
        return fresh_messages(config, task, think);
    };

    messages.push(Message::User(task.to_string()));
    messages
}

/// The system prompt and initial user turn for a fresh run.
fn fresh_messages(config: &AgentConfig, task: &str, think: bool) -> Vec<Message> {
    vec![
        Message::System(NATIVE_SYSTEM_PROMPT.to_string()),
        Message::User(initial_user_message(config, task, think)),
    ]
}

/// The finish gate for a run, carrying a resumed run's verification state
/// forward when its history already shows a successful command.
fn gate_for_run(messages: &[Message]) -> FileWorkGate {
    let mut gate = FileWorkGate::default();
    if verified_in_history(messages) {
        gate.record_verification();
    }
    gate
}

/// True when a tool result in the conversation reports a zero exit code.
fn verified_in_history(messages: &[Message]) -> bool {
    messages.iter().any(|message| match message {
        Message::Tool { content, .. } => command_succeeded(content),
        _ => false,
    })
}

/// Files the prior conversation already wrote, so a resumed run's finish gate
/// does not complain that nothing changed.
fn previously_changed_files(messages: &[Message]) -> Vec<String> {
    let mut files: Vec<String> = Vec::new();

    for call in messages.iter().filter_map(Message::calls).flatten() {
        let Some(path) = written_path(call) else {
            continue;
        };
        if !files.iter().any(|known| known == &path) {
            files.push(path);
        }
    }

    files
}

/// The file an invocation writes, when it is a `write_file` or `edit_file` call.
fn written_path(call: &Invocation) -> Option<String> {
    let is_file_writer = matches!(call.name.as_str(), "write_file" | "edit_file");
    if !is_file_writer {
        return None;
    }

    let arguments: Value = serde_json::from_str(&call.arguments).ok()?;
    let path = arguments.get("path")?.as_str()?;
    Some(path.to_string())
}

/// The user turn that opens a fresh conversation.
///
/// A model told not to think stops honouring rules that only live in the
/// system turn: with `enable_thinking: false`, Nemotron-3-Nano writes to the
/// workspace root although the system prompt says to use `out/`, and it obeys
/// again the moment the same words arrive in the user turn. So the task
/// carries the instructions whenever thinking is off.
fn initial_user_message(config: &AgentConfig, task: &str, think: bool) -> String {
    let task = format!(
        "Workspace root: {}\n\nEnvironment:\n{}\n\nTask:\n{task}",
        config.cwd.display(),
        environment_facts(&config.cwd)
    );
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
        // "required" keeps served models from answering in plain text when a
        // tool call is what the task needs; the `finish` tool is the exit.
        "tool_choice": "required",
        "temperature": AGENT_TEMPERATURE,
        "max_tokens": max_tokens,
        "stream": true,
        "chat_template_kwargs": { "enable_thinking": think },
    })
}

/// Parse a JSON tool call the model wrote as plain text instead of using the
/// server's structured `tool_calls` channel.
///
/// Some served models are trained to answer with `{"name": ..., "arguments": ...}`
/// in the reply text; when `tool_choice` does not force the structured channel
/// the loop would otherwise treat that JSON as the final answer. Known tools
/// are extracted and executed instead; anything else is left to the answer path.
fn parse_text_tool_call(content: &str) -> Option<Invocation> {
    let cleaned = content
        .trim()
        .trim_start_matches("```json")
        .trim_start_matches("```")
        .trim_end_matches("```")
        .trim();
    let value: Value = serde_json::from_str(cleaned).ok()?;
    let name = value.get("name")?.as_str()?.to_string();
    let arguments = value.get("arguments").cloned().unwrap_or_else(|| json!({}));
    // The id is left empty on purpose: `call_id` synthesizes one for calls
    // the server did not identify.
    let invocation = Invocation {
        id: String::new(),
        name,
        arguments: arguments.to_string(),
    };
    parse_call(&invocation).ok()?;
    Some(invocation)
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

/// Drop the oldest turns whole, keeping the system prompt, the initial task,
/// and as many complete recent turns as fit in `max` messages.
///
/// A turn is an assistant message plus its tool results (and any injected
/// user feedback). Cutting inside a turn would either strand tool results
/// without their call or leave an assistant tool call without its results —
/// both shapes the server rejects — so turns are always dropped as a unit.
fn trim_messages(messages: &mut Vec<Message>, max: usize) {
    if messages.len() <= max || max < 2 {
        return;
    }

    let system = messages.remove(0);
    let task = messages.remove(0);

    while messages.len() > max - 2 {
        let mut end = 1;
        while end < messages.len() && matches!(messages[end], Message::Tool { .. }) {
            end += 1;
        }
        messages.drain(0..end);
    }

    // A tail that starts with a tool result has lost its assistant message;
    // drop the orphans so the conversation stays valid.
    while matches!(messages.first(), Some(Message::Tool { .. })) {
        messages.remove(0);
    }

    messages.insert(0, task);
    messages.insert(0, system);
}

/// JSON schemas for the tools offered to the model.
///
/// The web tools are always present: the system prompt advertises them, and a
/// call to a tool that is not in the schema is rejected as unknown.
fn tools_schema() -> Value {
    let tools = json!([
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
                "name": "web_search",
                "description": "Search current web documentation. Prefer official vendor sources for APIs, libraries, GPU programming, and version-sensitive behavior. Return URLs and snippets. Search once or twice, then use web_read on an official result; do not repeat near-identical queries.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": { "type": "string", "description": "Focused search query; include the vendor or project and API/version." }
                    },
                    "required": ["query"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "web_read",
                "description": "Read an official NVIDIA or GitHub documentation page found by web_search. Pages are capped at 128 KiB and only approved HTTPS documentation hosts are allowed.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": { "type": "string", "description": "HTTPS URL from a web_search result on nvidia.github.io, developer.nvidia.com, docs.nvidia.com, or github.com." }
                    },
                    "required": ["url"]
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
                "description": "Signal that the task is complete. Call it only after the work is done and verified.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "summary": { "type": "string", "description": "Short summary of what was done." }
                    },
                    "required": ["summary"]
                }
            }
        }
    ]);
    tools
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
    fn web_tools_are_always_offered() {
        let names: Vec<String> = tools_schema()
            .as_array()
            .expect("schema is an array")
            .iter()
            .filter_map(|tool| tool.pointer("/function/name").and_then(Value::as_str))
            .map(str::to_string)
            .collect();
        assert!(names.contains(&"web_search".to_string()));
        assert!(names.contains(&"web_read".to_string()));
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
            model_context: 8_192,
            max_ctx: None,
            searxng_url: "http://127.0.0.1:8888".to_string(),
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

    #[test]
    fn the_request_requires_tool_calls() {
        let messages = vec![Message::User("task".to_string())];
        let body = request_body("m", &messages, 8192, true);

        assert_eq!(body["tool_choice"], json!("required"));
        assert!(body["tools"]
            .as_array()
            .is_some_and(|tools| !tools.is_empty()));
    }

    #[test]
    fn trims_whole_turns_without_splitting_a_call_from_its_result() {
        let call = |index: usize| Invocation {
            id: format!("c{index}"),
            name: "read_file".to_string(),
            arguments: "{}".to_string(),
        };
        let mut messages = vec![
            Message::System("system".to_string()),
            Message::User("task".to_string()),
            Message::Assistant {
                content: String::new(),
                calls: vec![call(1)],
            },
            Message::Tool {
                id: "c1".to_string(),
                content: "one".to_string(),
            },
            Message::Tool {
                id: "c1b".to_string(),
                content: "one more".to_string(),
            },
            Message::Assistant {
                content: String::new(),
                calls: vec![call(2)],
            },
            Message::Tool {
                id: "c2".to_string(),
                content: "two".to_string(),
            },
        ];
        trim_messages(&mut messages, 4);

        assert_eq!(messages.len(), 4);
        assert!(matches!(messages[0], Message::System(_)));
        assert!(matches!(messages[1], Message::User(_)));
        let Message::Assistant { calls, .. } = &messages[2] else {
            panic!("expected the recent assistant turn to survive");
        };
        assert_eq!(calls[0].id, "c2");
        assert!(matches!(messages[3], Message::Tool { .. }));
    }

    #[test]
    fn trims_never_leave_a_call_without_its_results() {
        // Six turns of two messages each, trimmed to 4: the kept tail must be
        // an assistant message followed by its tool result, never a bare
        // assistant with dangling calls.
        let mut messages = vec![
            Message::System("system".to_string()),
            Message::User("task".to_string()),
        ];
        for index in 0..6 {
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
        trim_messages(&mut messages, 4);

        assert_eq!(messages.len(), 4);
        for window in messages.windows(2) {
            let [first, second] = window else { continue };
            let Message::Assistant { calls, .. } = first else {
                continue;
            };
            assert!(
                !calls.is_empty() && matches!(second, Message::Tool { .. }),
                "a call must be followed by its tool result"
            );
        }
    }

    #[test]
    fn trims_strip_orphan_tool_results() {
        let mut messages = vec![
            Message::System("system".to_string()),
            Message::User("task".to_string()),
            Message::Tool {
                id: "lost".to_string(),
                content: "orphan".to_string(),
            },
            Message::Assistant {
                content: String::new(),
                calls: vec![Invocation {
                    id: "c1".to_string(),
                    name: "read_file".to_string(),
                    arguments: "{}".to_string(),
                }],
            },
            Message::Tool {
                id: "c1".to_string(),
                content: "x".to_string(),
            },
        ];
        trim_messages(&mut messages, 4);

        assert_eq!(messages.len(), 4);
        assert!(matches!(messages[2], Message::Assistant { .. }));
        assert!(matches!(messages[3], Message::Tool { .. }));
    }

    #[test]
    fn messages_round_trip_through_json() {
        let original = Message::Assistant {
            content: "reading".to_string(),
            calls: vec![Invocation {
                id: "call_1".to_string(),
                name: "read_file".to_string(),
                arguments: r#"{"path":"a.rs"}"#.to_string(),
            }],
        };
        let restored = Message::from_json(&original.to_json()).expect("known shape");

        let Message::Assistant { content, calls } = restored else {
            panic!("expected an assistant message");
        };
        assert_eq!(content, "reading");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].id, "call_1");
        assert_eq!(calls[0].name, "read_file");
        assert_eq!(calls[0].arguments, r#"{"path":"a.rs"}"#);
    }

    #[test]
    fn from_json_drops_unknown_roles() {
        assert!(Message::from_json(&json!({"role": "oracle", "content": "x"})).is_none());
    }

    #[test]
    fn text_tool_calls_are_parsed_when_the_tool_is_known() {
        let call = parse_text_tool_call(r#"{"name": "read_file", "arguments": {"path": "a.rs"}}"#)
            .expect("known tool in text form");
        assert_eq!(call.name, "read_file");
        assert_eq!(call.arguments, r#"{"path":"a.rs"}"#);

        let fenced = parse_text_tool_call(
            "```json\n{\"name\": \"list_files\", \"arguments\": {\"path\": \".\"}}\n```",
        )
        .expect("fenced tool call");
        assert_eq!(fenced.name, "list_files");
    }

    #[test]
    fn text_tool_calls_ignore_unknown_tools_and_plain_answers() {
        assert!(parse_text_tool_call(r#"{"name": "rm_rf", "arguments": {}}"#).is_none());
        assert!(parse_text_tool_call("the build failed because of x").is_none());
        assert!(parse_text_tool_call(r#"{"answer": "done"}"#).is_none());
    }

    #[test]
    fn resumed_messages_append_the_correction_as_a_user_turn() {
        let prior = vec![
            json!({"role": "system", "content": "rules"}),
            json!({"role": "user", "content": "original task"}),
            json!({"role": "assistant", "content": "", "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}
            ]}),
            json!({"role": "tool", "tool_call_id": "c1", "content": "file contents"}),
        ];
        let messages = seed_messages(&config(), "use the other API instead", true, Some(prior));

        assert_eq!(messages.len(), 5);
        assert!(matches!(messages[0], Message::System(_)));
        assert!(matches!(messages[1], Message::User(_)));
        let Message::User(correction) = &messages[4] else {
            panic!("expected the correction as the last user turn");
        };
        assert_eq!(correction, "use the other API instead");
    }

    #[test]
    fn resumed_messages_with_bad_history_fall_back_to_a_fresh_start() {
        let messages = seed_messages(
            &config(),
            "task",
            true,
            Some(vec![json!({"role": "oracle", "content": "x"})]),
        );

        assert_eq!(messages.len(), 2);
        assert!(matches!(messages[0], Message::System(_)));
        assert!(matches!(messages[1], Message::User(_)));
    }

    #[test]
    fn previously_changed_files_reads_write_calls_from_history() {
        let messages = vec![Message::Assistant {
            content: String::new(),
            calls: vec![
                Invocation {
                    id: "c1".to_string(),
                    name: "write_file".to_string(),
                    arguments: r#"{"path":"src/lib.rs","content":"x"}"#.to_string(),
                },
                Invocation {
                    id: "c2".to_string(),
                    name: "edit_file".to_string(),
                    arguments: r#"{"path":"README.md"}"#.to_string(),
                },
                Invocation {
                    id: "c3".to_string(),
                    name: "read_file".to_string(),
                    arguments: r#"{"path":"Cargo.toml"}"#.to_string(),
                },
            ],
        }];

        assert_eq!(
            previously_changed_files(&messages),
            vec!["src/lib.rs".to_string(), "README.md".to_string()]
        );
    }

    #[test]
    fn a_successful_command_in_history_seeds_the_gate() {
        let messages = vec![Message::Tool {
            id: "c1".to_string(),
            content: "compiled\n[exit code 0]\n".to_string(),
        }];

        let mut gate = gate_for_run(&messages);
        assert_eq!(gate.reject_finish(true, false, 2, true), None);

        let mut fresh = gate_for_run(&[Message::User("task".to_string())]);
        assert!(fresh.reject_finish(true, false, 2, true).is_some());
    }
}
