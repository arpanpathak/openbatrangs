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

use super::context::{max_prompt_chars, truncate_chars};
use super::environment::{environment_facts, project_has_build_manifest};
use super::execute::{absolute_paths, run_tool_call};
use super::quality::{FinishGate, command_succeeded};
use super::repeats::RepeatGuard;
use super::research::task_with_documentation;
use super::tool::{Tool, ToolCall, strip_code_fence};
use super::{
    AgentConfig, AgentEvent, Confirmer, HistorySink, LoopKind, Reporter, RunContinuation, TextKind,
    earlier_turns_section, effective_context, ensure_workspace_exists,
};
use crate::constants::agent::{
    AGENT_TEMPERATURE, CHARS_PER_TOKEN, CONTEXT_RESERVE_TOKENS, EMPTY_TURN_NUDGE,
    MAX_HISTORY_MESSAGES, MAX_OUTPUT_TOKENS, MIN_OUTPUT_TOKENS, MIN_TOOL_RESULT_CHARS,
    NATIVE_SYSTEM_PROMPT,
};
use crate::ollama::{Delta, OllamaClient, ToolCallDelta};
use anyhow::{Context, Result};
use futures_util::{Stream, TryStreamExt};
use serde_json::{Value, json};

/// Number of empty model turns answered with a nudge before an empty turn
/// counts as the answer.
const MAX_EMPTY_RETRIES: u8 = 2;

/// The text of the tool result that answers an `ask_user` call.
const QUESTION_SHOWN: &str =
    "The question is shown to the user; their answer follows as the next message.";

/// The states of one native run. [`Run::run`] moves from state to state
/// until the run is over.
enum State {
    /// Start the next step, or give up when the step budget is spent.
    Ask,
    /// Stream the model's reply for the current step.
    Stream,
    /// The reply has no structured tool calls: run a call written as text, or
    /// take the text as the answer.
    Reply(Turn),
    /// Run the reply's tool calls.
    RunTools(Turn),
    /// The reply was empty.
    EmptyTurn,
    /// The reply text is the answer.
    Answer(String),
    /// The model called `finish` with this summary.
    Finish(String),
    /// The model asked the user this question.
    AskUser(String),
    /// The step budget ran out before an answer.
    GiveUp,
    /// The run is over.
    Done,
}

/// What the tool calls of one turn asked for besides their results.
#[derive(Default)]
struct CallOutcome {
    /// Summary of the last `finish` call.
    summary: Option<String>,
    /// Question of the last `ask_user` call.
    question: Option<String>,
}

impl CallOutcome {
    /// Ask the user when a call asked a question, finish when a call
    /// finished, and take another step otherwise.
    fn next_state(self) -> State {
        match (self.question, self.summary) {
            (Some(question), _) => State::AskUser(question),
            (None, Some(summary)) => State::Finish(summary),
            (None, None) => State::Ask,
        }
    }
}

/// One message in the native conversation, mapped to the OpenAI chat shape.
#[derive(Debug, Clone)]
enum Message {
    /// System instructions.
    System(String),
    /// A user turn: the task, or feedback the loop adds.
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
            Self::Assistant { content, calls } => assistant_json(content, calls),
            Self::Tool { id, content } => {
                json!({ "role": "tool", "tool_call_id": id, "content": content })
            }
        }
    }

    /// Rebuild a message from its OpenAI `/chat/completions` shape.
    ///
    /// Unknown roles and malformed entries yield `None`, so a resumed
    /// conversation drops them instead of failing the request.
    fn from_json(value: &Value) -> Option<Self> {
        let role = value
            .get("role")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let content = string_field(value, "content");

        match role {
            "system" => Some(Self::System(content)),
            "user" => Some(Self::User(content)),
            "assistant" => Some(Self::Assistant {
                content,
                calls: invocations(value),
            }),
            "tool" => Some(Self::Tool {
                id: string_field(value, "tool_call_id"),
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

    /// True for a tool result.
    fn is_tool_result(&self) -> bool {
        matches!(self, Self::Tool { .. })
    }

    /// Shorten a tool result to at most `max_chars` characters; other
    /// messages stay as they are.
    fn shorten_tool_result(&mut self, max_chars: usize) {
        let Self::Tool { content, .. } = self else {
            return;
        };

        *content = truncate_chars(content, max_chars);
    }

    /// The text of a tool result.
    fn tool_content(&self) -> Option<&str> {
        match self {
            Self::Tool { content, .. } => Some(content),
            _ => None,
        }
    }
}

/// An assistant message in the OpenAI shape; `tool_calls` is present only
/// when the turn calls tools.
fn assistant_json(content: &str, calls: &[Invocation]) -> Value {
    let mut message = json!({ "role": "assistant", "content": content });

    if !calls.is_empty() {
        message["tool_calls"] = calls.iter().map(Invocation::to_json).collect();
    }

    message
}

/// The string under `key`, or an empty string when it is missing or not a
/// string.
fn string_field(value: &Value, key: &str) -> String {
    value
        .get(key)
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string()
}

/// The parsable `tool_calls` items of an assistant message.
fn invocations(value: &Value) -> Vec<Invocation> {
    value
        .get("tool_calls")
        .and_then(Value::as_array)
        .map(|calls| calls.iter().filter_map(Invocation::from_json).collect())
        .unwrap_or_default()
}

/// A fully assembled tool call, ready to execute.
#[derive(Debug, Clone, Default)]
struct Invocation {
    /// Call id from the server; [`identified_calls`] fills it in when the
    /// server omits it.
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
            id: string_field(value, "id"),
            name: string_field(function, "name"),
            arguments: string_field(function, "arguments"),
        })
    }

    /// Character size of the name and arguments.
    fn char_len(&self) -> usize {
        self.name.len() + self.arguments.len()
    }

    /// This call with `fallback` as its id when it has none.
    fn with_id(mut self, fallback: String) -> Self {
        if self.id.is_empty() {
            self.id = fallback;
        }

        self
    }

    /// Merge one streamed chunk: a non-empty id or name replaces the stored
    /// one, and argument text is appended.
    fn merge(&mut self, delta: ToolCallDelta) {
        replace_when_present(&mut self.id, delta.id);
        replace_when_present(&mut self.name, delta.name);
        self.arguments.push_str(&delta.arguments);
    }
}

/// Replace `field` with `value` when `value` is present and not empty.
fn replace_when_present(field: &mut String, value: Option<String>) {
    if let Some(value) = value.filter(|value| !value.is_empty()) {
        *field = value;
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
        let call = self.call_at(delta.index as usize);

        call.merge(delta);
    }

    /// The call at `index`, adding empty calls up to it when needed.
    fn call_at(&mut self, index: usize) -> &mut Invocation {
        if self.calls.len() <= index {
            self.calls.resize_with(index + 1, Invocation::default);
        }

        &mut self.calls[index]
    }
}

/// Collects one streamed turn and prints its text and reasoning live.
struct TurnPrinter {
    /// The turn received so far.
    turn: Turn,
    /// Whether reasoning text is printed.
    show_thinking: bool,
    /// Whether reasoning text is being printed right now.
    is_thinking: bool,
}

impl TurnPrinter {
    /// A printer that has received nothing yet.
    fn new(show_thinking: bool) -> Self {
        Self {
            turn: Turn::default(),
            show_thinking,
            is_thinking: false,
        }
    }

    /// Add one delta to the turn and print its text.
    fn take<R: Reporter>(&mut self, delta: Delta, reporter: &mut R) {
        match delta {
            Delta::Thinking(text) => self.think(&text, reporter),
            Delta::Content(text) => self.content(&text, reporter),
            Delta::ToolCalls(calls) => self.tool_calls(calls, reporter),
        }
    }

    /// Print reasoning text, when reasoning is shown.
    fn think<R: Reporter>(&mut self, text: &str, reporter: &mut R) {
        if !self.show_thinking {
            return;
        }

        self.is_thinking = true;
        reporter.report(AgentEvent::Text {
            kind: TextKind::Reasoning,
            text: text.to_string(),
        });
    }

    /// End the reasoning line, if one is open.
    fn end_thinking<R: Reporter>(&mut self, reporter: &mut R) {
        if !self.is_thinking {
            return;
        }

        reporter.report(AgentEvent::TextEnded(TextKind::Reasoning));
        self.is_thinking = false;
    }

    /// Add reply text to the turn and print it.
    fn content<R: Reporter>(&mut self, text: &str, reporter: &mut R) {
        self.end_thinking(reporter);
        self.turn.content.push_str(text);
        reporter.report(AgentEvent::Text {
            kind: TextKind::Answer,
            text: text.to_string(),
        });
    }

    /// Add tool-call chunks to the turn.
    fn tool_calls<R: Reporter>(&mut self, calls: Vec<ToolCallDelta>, reporter: &mut R) {
        self.end_thinking(reporter);

        for call in calls {
            self.turn.absorb(call);
        }
    }

    /// Close any open reasoning line and return the turn.
    fn finish<R: Reporter>(mut self, reporter: &mut R) -> Turn {
        self.end_thinking(reporter);

        self.turn
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
    /// Characters the messages may use, so the prompt, the tool schema and
    /// the reply fit in `num_ctx`.
    prompt_budget: usize,
    /// Whether the model is asked to think, from `--no-think` on the client.
    think: bool,
    /// Whether the project has a build or test manifest, so changed files
    /// need a passing check before the run may finish.
    needs_check: bool,
    /// Number of the current step; 0 before the first.
    step: usize,
    /// Number of consecutive empty model turns answered with a nudge message.
    empty_retries: u8,
    /// Conversation sent to the model, trimmed to fit the context window.
    messages: Vec<Message>,
    /// The same conversation without trimming, for the transcript.
    full_conversation: Vec<Message>,
    /// Files changed by tools so far.
    changed_files: Vec<String>,
    /// Gate that rejects a finish before changed files are checked.
    finish_gate: FinishGate,
    /// Stops a model that repeats the same tool call.
    repeat_guard: RepeatGuard,
    /// Sink receiving conversation snapshots, so an interrupted run can resume.
    sink: Option<&'a mut dyn HistorySink>,
    /// Sink receiving the untrimmed conversation when the run ends.
    transcript: Option<&'a mut dyn HistorySink>,
    /// Receives the run's events.
    reporter: &'a mut R,
    /// Human-in-the-loop gate for writes and commands.
    confirmer: &'a mut C,
}

impl<R: Reporter, C: Confirmer> Run<'_, R, C> {
    /// Run the states to the end, then hand the whole conversation to the
    /// transcript sink, whether the run answered, gave up or failed.
    async fn run(mut self) -> Result<()> {
        let result = self.run_states().await;
        self.save_transcript();

        result
    }

    /// Move from state to state until the run is over.
    async fn run_states(&mut self) -> Result<()> {
        let mut state = State::Ask;

        loop {
            state = match state {
                State::Ask => self.ask(),
                State::Stream => self.stream().await?,
                State::Reply(turn) => self.reply(turn),
                State::RunTools(turn) => self.run_tools(turn).await?,
                State::EmptyTurn => self.empty_turn(),
                State::Answer(text) => self.answer(&text),
                State::Finish(summary) => self.finish(&summary),
                State::AskUser(question) => self.ask_user(question),
                State::GiveUp => self.give_up(),
                State::Done => return Ok(()),
            };
        }
    }

    /// Save the conversation and start the next step, or give up when the
    /// budget is spent.
    fn ask(&mut self) -> State {
        self.save_conversation();
        self.step += 1;

        if self.step > self.config.max_steps {
            return State::GiveUp;
        }

        self.reporter.report(AgentEvent::StepStarted {
            step: self.step,
            max_steps: self.config.max_steps,
            model: self.model.to_string(),
            loop_kind: LoopKind::Native,
        });
        trim_messages(&mut self.messages, MAX_HISTORY_MESSAGES);
        fit_to_budget(&mut self.messages, self.prompt_budget);

        State::Stream
    }

    /// Stream the model's reply; a reply with tool calls runs them.
    async fn stream(&mut self) -> Result<State> {
        let body = request_body(self.model, &self.messages, self.num_ctx, self.think);

        let turn = stream_turn(self.client, body, self.reporter, self.config.show_thinking)
            .await
            .with_context(|| format!("model call failed at step {}", self.step))?;

        Ok(state_after_turn(turn))
    }

    /// Run a tool call the model wrote as text; otherwise the text is the
    /// answer, or an empty turn.
    fn reply(&mut self, turn: Turn) -> State {
        let Some(call) = parse_text_tool_call(&turn.content) else {
            return state_for_text(turn.content);
        };

        self.reporter
            .notice("\n⚠️ The model wrote its tool call as text; executing it anyway.".to_string());

        State::RunTools(Turn {
            calls: vec![call],
            ..turn
        })
    }

    /// Answer the first two empty turns in a row with a nudge message; the
    /// next one ends the run.
    fn empty_turn(&mut self) -> State {
        if self.empty_retries >= MAX_EMPTY_RETRIES {
            self.reporter
                .notice("\n⚠️  The model returned an empty reply.".to_string());
            return self.done();
        }

        self.empty_retries += 1;
        self.record(Message::User(EMPTY_TURN_NUDGE.to_string()));
        self.reporter
            .notice("\n⚠️ Empty model turn; prompting it to continue with tools.".to_string());

        State::Ask
    }

    /// End the run with the reply text, unless the finish gate rejects it; a
    /// rejected answer stays in the conversation with the feedback after it.
    fn answer(&mut self, text: &str) -> State {
        let answer = text.trim();
        self.empty_retries = 0;

        // The answer stays in the conversation either way: a resumed run
        // needs it, and a rejected one needs it before the feedback.
        self.record(Message::Assistant {
            content: answer.to_string(),
            calls: Vec::new(),
        });

        let Some(feedback) = self.finish_feedback() else {
            return self.report_done(answer, true);
        };

        self.reject_finish(feedback)
    }

    /// Run every tool call of the turn, then finish, ask the user, or take
    /// another step; stop when the model keeps repeating one call.
    async fn run_tools(&mut self, turn: Turn) -> Result<State> {
        let mut outcome = CallOutcome::default();
        let calls = identified_calls(turn.calls, self.step);
        self.empty_retries = 0;
        self.record(Message::Assistant {
            content: turn.content,
            calls: calls.clone(),
        });

        for call in &calls {
            self.execute_one(call, &mut outcome).await;
        }

        self.repeat_guard.allow(outcome.next_state())
    }

    /// End the run with the `finish` summary, unless the finish gate rejects it.
    fn finish(&mut self, summary: &str) -> State {
        let Some(feedback) = self.finish_feedback() else {
            return self.report_done(summary, false);
        };

        self.reject_finish(feedback)
    }

    /// Show the question and end the run; the user's answer resumes it.
    fn ask_user(&mut self, question: String) -> State {
        self.reporter.report(AgentEvent::AskUser { question });

        self.done()
    }

    /// Warn that the step budget ran out without an answer.
    fn give_up(&mut self) -> State {
        self.reporter.report(AgentEvent::StepLimitReached {
            max_steps: self.config.max_steps,
        });

        State::Done
    }

    /// Save the conversation and end the run.
    fn done(&mut self) -> State {
        self.save_conversation();

        State::Done
    }

    /// Send the gate's feedback to the model and take another step.
    fn reject_finish(&mut self, feedback: &str) -> State {
        self.record(Message::User(feedback.to_string()));
        self.reporter
            .notice("\n⚠️  Finish rejected: the task is not done.".to_string());

        State::Ask
    }

    /// Report the answer and the changed files, and end the run.
    fn report_done(&mut self, answer: &str, answer_was_streamed: bool) -> State {
        let changed_files = absolute_paths(&self.config.cwd, &self.changed_files);

        self.reporter.report(AgentEvent::Finished {
            answer: answer.to_string(),
            answer_was_streamed,
            changed_files,
        });

        self.done()
    }

    /// Run one call and append its result to the conversation; a `finish`
    /// or `ask_user` call is recorded in `outcome`.
    async fn execute_one(&mut self, call: &Invocation, outcome: &mut CallOutcome) {
        let id = call.id.clone();

        match parse_call(call) {
            Ok(Tool::Finish { summary }) => {
                outcome.summary = Some(self.acknowledge_finish(id, summary))
            }
            Ok(Tool::AskUser { question }) => {
                outcome.question = Some(self.acknowledge_question(id, question))
            }
            Ok(tool) => self.run_tool(id, call, tool).await,
            Err(error) => self.reject(id, &error),
        }
    }

    /// Answer a `finish` call so every call has its result, and return its
    /// summary.
    fn acknowledge_finish(&mut self, id: String, summary: String) -> String {
        self.record(Message::Tool {
            id,
            content: "acknowledged".to_string(),
        });

        summary
    }

    /// Answer an `ask_user` call so every call has its result, and return the
    /// question.
    fn acknowledge_question(&mut self, id: String, question: String) -> String {
        self.record(Message::Tool {
            id,
            content: QUESTION_SHOWN.to_string(),
        });

        question
    }

    /// Execute one tool, report it, and append its result to the conversation.
    async fn run_tool(&mut self, id: String, call: &Invocation, tool: Tool) {
        self.reporter.report(AgentEvent::ToolCall {
            id: id.clone(),
            name: call.name.clone(),
            arguments: call.arguments.clone(),
        });

        let changes_before = self.changed_files.len();
        let cwd = &self.config.cwd;
        let report = run_tool_call(
            self.config,
            cwd,
            &tool,
            &mut self.changed_files,
            self.confirmer,
        )
        .await;
        let files_changed = self.changed_files.len() > changes_before;
        self.finish_gate
            .record_tool_result(&tool, &report.output, files_changed);

        self.reporter.report(AgentEvent::ToolResult {
            id: id.clone(),
            success: report.success,
            summary: report.summary,
            output: report.output.clone(),
        });
        let content = self
            .repeat_guard
            .model_result(&call.name, &call.arguments, &report.output);
        self.record(Message::Tool { id, content });
    }

    /// Record a rejected call so the model can retry it.
    fn reject(&mut self, id: String, error: &anyhow::Error) {
        self.reporter
            .notice(format!("\n⚠️  Tool call rejected: {error}"));
        self.record(Message::Tool {
            id,
            content: format!("Error: {error}. Retry with the exact argument names."),
        });
    }

    /// The finish gate's feedback when a finish now would be premature;
    /// `None` when the finish may pass.
    fn finish_feedback(&mut self) -> Option<&'static str> {
        self.finish_gate.reject_finish(self.needs_check)
    }

    /// Append `message` to the conversation sent to the model and to the
    /// untrimmed copy kept for the transcript.
    fn record(&mut self, message: Message) {
        self.full_conversation.push(message.clone());
        self.messages.push(message);
    }

    /// Send the untrimmed conversation to the transcript sink, if one is
    /// attached.
    fn save_transcript(&mut self) {
        let Some(transcript) = &mut self.transcript else {
            return;
        };

        let conversation = self
            .full_conversation
            .iter()
            .map(Message::to_json)
            .collect();
        transcript.save(conversation);
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

/// The state after a streamed turn: run its tool calls, or read its text.
fn state_after_turn(turn: Turn) -> State {
    match turn.calls.is_empty() {
        true => State::Reply(turn),
        false => State::RunTools(turn),
    }
}

/// The state for a reply without tool calls: blank text is an empty turn.
fn state_for_text(text: String) -> State {
    match text.trim().is_empty() {
        true => State::EmptyTurn,
        false => State::Answer(text),
    }
}

/// Run the native tool-calling loop until the model answers or steps run out.
///
/// # Arguments
/// - `config`: agent runtime configuration.
/// - `client`: an OpenAI-compatible client.
/// - `model`: model name to request.
/// - `task`: the user's task, or the correction when resuming.
/// - `continuation`: the interrupted run to continue, the snapshot sink and
///   the transcript sink; one-shot mode passes only the transcript sink.
/// - `reporter`: output sink.
/// - `confirmer`: human-in-the-loop gate for writes and commands.
///
/// # Returns
/// `Ok(())` when the agent finishes, or an error if the working directory is
/// missing or a model call fails.
pub(super) async fn run_native_agent<'a, R: Reporter, C: Confirmer>(
    config: &'a AgentConfig,
    client: &'a OllamaClient,
    model: &'a str,
    task: &str,
    continuation: Option<RunContinuation<'a>>,
    reporter: &'a mut R,
    confirmer: &'a mut C,
) -> Result<()> {
    ensure_workspace_exists(&config.cwd)?;

    let RunContinuation {
        messages: resume,
        earlier,
        sink,
        transcript,
    } = continuation.unwrap_or_default();
    let think = client.requests_thinking();
    let resuming = resume.is_some();
    let context_cap = config.max_ctx.or(client.default_context_cap());
    let num_ctx = effective_context(config.model_context, context_cap);

    let opening_task = opening_task(task, config, resuming, reporter).await;
    let opening = Opening {
        task: &opening_task,
        earlier: &earlier_turns_section(&earlier, num_ctx),
        think,
    };
    let messages = seed_messages(config, &opening, resume);

    // Carry the prior run's state forward so the gate does not demand a
    // check that already passed before the interrupt.
    let changed_files = previously_changed_files(&messages);
    let finish_gate = gate_for_run(&messages);

    let run = Run {
        config,
        client,
        model,
        num_ctx,
        prompt_budget: prompt_budget(num_ctx),
        think,
        needs_check: project_has_build_manifest(&config.cwd),
        step: 0,
        empty_retries: 0,
        full_conversation: messages.clone(),
        messages,
        changed_files,
        finish_gate,
        repeat_guard: RepeatGuard::default(),
        sink,
        transcript,
        reporter,
        confirmer,
    };

    run.run().await
}

/// The task for the first user turn. A resumed run already carries its
/// documentation in the conversation; repeating the lookup would fetch the
/// same pages for nothing.
async fn opening_task<R: Reporter>(
    task: &str,
    config: &AgentConfig,
    resuming: bool,
    reporter: &mut R,
) -> String {
    if resuming {
        return task.to_string();
    }

    task_with_documentation(task, config, reporter).await
}

/// What the first user turn of a run is made of.
struct Opening<'a> {
    /// The task, or the correction when resuming.
    task: &'a str,
    /// The earlier turns of the session as a section of the first message;
    /// empty when there are none.
    earlier: &'a str,
    /// Whether the model is asked to think.
    think: bool,
}

/// The opening messages: the stored conversation plus the new task when
/// resuming, or the fresh system + task pair otherwise.
fn seed_messages(
    config: &AgentConfig,
    opening: &Opening,
    resume: Option<Vec<Value>>,
) -> Vec<Message> {
    let Some(mut messages) = resume.as_deref().and_then(Message::conversation) else {
        return fresh_messages(config, opening);
    };

    messages.push(Message::User(opening.task.to_string()));
    messages
}

/// The system prompt and initial user turn for a fresh run.
fn fresh_messages(config: &AgentConfig, opening: &Opening) -> Vec<Message> {
    vec![
        Message::System(NATIVE_SYSTEM_PROMPT.to_string()),
        Message::User(initial_user_message(config, opening)),
    ]
}

/// The finish gate for a run, replaying a resumed run's history: a file
/// written there needs a passing check after it, as in a fresh run.
fn gate_for_run(messages: &[Message]) -> FinishGate {
    let mut gate = FinishGate::default();

    for message in messages {
        gate.record(writes_a_file(message), reports_a_passing_command(message));
    }

    gate
}

/// True for an assistant turn that calls `write_file` or `edit_file`.
fn writes_a_file(message: &Message) -> bool {
    message
        .calls()
        .is_some_and(|calls| calls.iter().any(|call| written_path(call).is_some()))
}

/// True for a tool result that reports a zero exit code.
fn reports_a_passing_command(message: &Message) -> bool {
    message.tool_content().is_some_and(command_succeeded)
}

/// Files the prior conversation already wrote, so a resumed run's finish gate
/// does not complain that nothing changed.
fn previously_changed_files(messages: &[Message]) -> Vec<String> {
    let mut files: Vec<String> = Vec::new();
    let written_paths = messages
        .iter()
        .filter_map(Message::calls)
        .flatten()
        .filter_map(written_path);

    for path in written_paths {
        add_once(&mut files, path);
    }

    files
}

/// Add `path` to `files` unless it is already there.
fn add_once(files: &mut Vec<String>, path: String) {
    if !files.contains(&path) {
        files.push(path);
    }
}

/// The file an invocation writes, when it is a `write_file` or `edit_file` call.
fn written_path(call: &Invocation) -> Option<String> {
    let is_file_writer = matches!(call.name.as_str(), "write_file" | "edit_file");

    if !is_file_writer {
        return None;
    }

    parse_arguments(&call.arguments)
        .get("path")
        .and_then(Value::as_str)
        .map(str::to_string)
}

/// The heading [`initial_user_message`] puts before the task.
const TASK_HEADING: &str = "\n\nTask:\n";

/// The user turn that opens a fresh conversation.
///
/// With `enable_thinking: false`, Nemotron-3-Nano ignores rules that appear
/// only in the system turn: it writes to the workspace root although the
/// system prompt says to use `out/`, and it writes to `out/` again once the
/// same text is in the user turn. So the task carries the instructions
/// whenever thinking is off.
fn initial_user_message(config: &AgentConfig, opening: &Opening) -> String {
    let task = format!(
        "Workspace root: {}\n\nEnvironment:\n{}{}{TASK_HEADING}{}",
        config.cwd.display(),
        environment_facts(&config.cwd),
        opening.earlier,
        opening.task
    );

    match opening.think {
        true => task,
        false => format!("Instructions (highest priority):\n{NATIVE_SYSTEM_PROMPT}\n\n{task}"),
    }
}

/// The calls of the turn at `step`, each with an id: the server's, or
/// `call_<step>_<position>` when the server sent none.
///
/// The ids are set once, before the assistant message is stored, so the
/// message and each tool result carry the same id, and two calls to the same
/// tool get different ids.
fn identified_calls(calls: Vec<Invocation>, step: usize) -> Vec<Invocation> {
    calls
        .into_iter()
        .enumerate()
        .map(|(position, call)| call.with_id(format!("call_{step}_{position}")))
        .collect()
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
    serde_json::from_str(raw.trim())
        .ok()
        .unwrap_or_else(|| json!({}))
}

/// Build the `/chat/completions` body for one turn.
///
/// The tool schema goes with every request, so it counts toward the prompt
/// that `max_tokens` leaves room for.
fn request_body(model: &str, messages: &[Message], num_ctx: u64, think: bool) -> Value {
    let tools = tools_schema();
    let prompt = total_chars(messages) + tools.to_string().len();
    let prompt_tokens = (prompt as f64 / CHARS_PER_TOKEN).ceil() as u64;
    let max_tokens = num_ctx
        .saturating_sub(prompt_tokens)
        .saturating_sub(CONTEXT_RESERVE_TOKENS)
        .clamp(MIN_OUTPUT_TOKENS, MAX_OUTPUT_TOKENS);

    json!({
        "model": model,
        "messages": messages.iter().map(Message::to_json).collect::<Vec<_>>(),
        "tools": tools,
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
/// Some served models put `{"name": ..., "arguments": ...}` in the reply
/// text; when `tool_choice` does not force the structured channel
/// the loop would otherwise treat that JSON as the final answer. Known tools
/// are extracted and executed instead; anything else is left to the answer path.
fn parse_text_tool_call(content: &str) -> Option<Invocation> {
    let invocation = serde_json::from_str::<Value>(strip_code_fence(content))
        .ok()
        .and_then(|value| text_invocation(&value))?;

    let is_known_tool = parse_call(&invocation).is_ok();
    is_known_tool.then_some(invocation)
}

/// The invocation in a parsed `{"name": ..., "arguments": ...}` object.
///
/// The id is left empty on purpose: `identified_calls` gives one to calls the
/// server did not identify.
fn text_invocation(value: &Value) -> Option<Invocation> {
    let name = value.get("name").and_then(Value::as_str)?;
    let arguments = value.get("arguments").cloned().unwrap_or_else(|| json!({}));

    Some(Invocation {
        id: String::new(),
        name: name.to_string(),
        arguments: arguments.to_string(),
    })
}

/// Stream one model turn, forwarding text to the reporter.
async fn stream_turn<R: Reporter>(
    client: &OllamaClient,
    body: Value,
    reporter: &mut R,
    show_thinking: bool,
) -> Result<Turn> {
    let stream = client.chat_stream_body(body).await?;

    collect_turn(stream, reporter, show_thinking).await
}

/// Collect every delta of `stream` into a turn, printing text as it arrives.
async fn collect_turn<R: Reporter>(
    stream: impl Stream<Item = Result<Delta>>,
    reporter: &mut R,
    show_thinking: bool,
) -> Result<Turn> {
    let printer = TurnPrinter::new(show_thinking);

    let printer = stream
        .try_fold(printer, |mut printer, delta| {
            printer.take(delta, reporter);
            std::future::ready(Ok(printer))
        })
        .await?;

    Ok(printer.finish(reporter))
}

/// Drop the oldest turns whole, keeping the system prompt, the initial task,
/// and as many complete recent turns as fit in `max` messages.
///
/// A turn is an assistant message plus its tool results (and any injected
/// user feedback). Cutting inside a turn would either strand tool results
/// without their call or leave an assistant tool call without its results —
/// both shapes the server rejects — so turns are always dropped as a unit.
/// With `max` below 2 the messages are left as they are.
fn trim_messages(messages: &mut Vec<Message>, max: usize) {
    let Some(room) = max.checked_sub(2) else {
        return;
    };

    trim_to_room(messages, room);
}

/// Keep the system prompt, the task, and at most `room` recent messages.
fn trim_to_room(messages: &mut Vec<Message>, room: usize) {
    if messages.len() <= room + 2 {
        return;
    }

    let mut recent = messages.split_off(2);
    drop_oldest_turns(&mut recent, room);
    drop_orphan_results(&mut recent);

    messages.append(&mut recent);
}

/// Drop whole turns from the front until at most `room` messages are left.
fn drop_oldest_turns(messages: &mut Vec<Message>, room: usize) {
    while messages.len() > room {
        let turn_length = first_turn_length(messages);
        messages.drain(..turn_length);
    }
}

/// Length of the first turn: its first message and the tool results after it.
fn first_turn_length(messages: &[Message]) -> usize {
    let results = messages
        .iter()
        .skip(1)
        .take_while(|message| message.is_tool_result())
        .count();

    1 + results
}

/// Characters the messages may use in a `num_ctx` window: the prompt budget
/// minus the tool schema, which goes with every request.
fn prompt_budget(num_ctx: u64) -> usize {
    let schema_chars = tools_schema().to_string().len();

    max_prompt_chars(num_ctx).saturating_sub(schema_chars)
}

/// Make the messages fit in `budget` characters: drop the oldest turns after
/// the system prompt and the task, keeping at least the newest turn, then
/// shorten tool results when that is not enough.
///
/// Turns are dropped whole, so no tool call loses its result.
fn fit_to_budget(messages: &mut Vec<Message>, budget: usize) {
    if total_chars(messages) <= budget {
        return;
    }

    let mut recent = messages.split_off(messages.len().min(2));
    let room = budget.saturating_sub(total_chars(messages));

    drop_turns_over_room(&mut recent, room);
    drop_orphan_results(&mut recent);
    messages.append(&mut recent);
    shorten_tool_results(messages, budget);
}

/// Drop whole turns from the front of `recent` until it fits in `room`
/// characters or only one turn is left.
fn drop_turns_over_room(recent: &mut Vec<Message>, room: usize) {
    while has_turn_to_drop(recent, room) {
        let turn_length = first_turn_length(recent);
        recent.drain(..turn_length);
    }
}

/// True while `recent` is over `room` characters and holds more than one turn.
fn has_turn_to_drop(recent: &[Message], room: usize) -> bool {
    total_chars(recent) > room && first_turn_length(recent) < recent.len()
}

/// Shorten every tool result to an equal share of what the other messages
/// leave of `budget`, when the messages are still over it.
fn shorten_tool_results(messages: &mut [Message], budget: usize) {
    if total_chars(messages) <= budget {
        return;
    }

    let share = tool_result_share(messages, budget);

    messages
        .iter_mut()
        .for_each(|message| message.shorten_tool_result(share));
}

/// The characters each tool result may keep: what the other messages leave
/// of `budget`, split evenly, and never below [`MIN_TOOL_RESULT_CHARS`].
fn tool_result_share(messages: &[Message], budget: usize) -> usize {
    let results: Vec<&str> = messages.iter().filter_map(Message::tool_content).collect();
    let result_chars: usize = results.iter().map(|content| content.len()).sum();
    let other_chars = total_chars(messages) - result_chars;
    let room = budget.saturating_sub(other_chars);

    (room / results.len().max(1)).max(MIN_TOOL_RESULT_CHARS)
}

/// Combined character size of `messages`.
fn total_chars(messages: &[Message]) -> usize {
    messages.iter().map(Message::char_len).sum()
}

/// A tail that starts with a tool result has lost its assistant message;
/// drop the orphans so the conversation stays valid.
fn drop_orphan_results(messages: &mut Vec<Message>) {
    let orphans = messages
        .iter()
        .take_while(|message| message.is_tool_result())
        .count();

    messages.drain(..orphans);
}

/// JSON schemas for the tools offered to the model.
///
/// The web tools are always present: the system prompt advertises them, and a
/// call to a tool that is not in the schema is rejected as unknown.
pub(super) fn tools_schema() -> Value {
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
                "name": "ask_user",
                "description": "Ask the user one short question when the request is unclear and the workspace cannot settle it. The run stops; the user's reply arrives as the next message.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "question": { "type": "string", "description": "The question, in plain words." }
                    },
                    "required": ["question"]
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
                        "summary": { "type": "string", "description": "The only text the user reads. For a question, the complete answer, with its quotes and URLs; for a change, what was done and what the checks reported." }
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
    use crate::agent::fakes::{
        FixedConfirmer, RecordingReporter, RecordingSink, Reply, serve_replies, test_config,
        text_delta, tool_call_delta,
    };
    use crate::ollama::ChatMessage;
    use std::path::{Path, PathBuf};

    /// A `read_file` invocation with id `id`.
    fn read_call(id: &str) -> Invocation {
        Invocation {
            id: id.to_string(),
            name: "read_file".to_string(),
            arguments: "{}".to_string(),
        }
    }

    /// An assistant turn that calls `read_file` with id `id`.
    fn assistant_call(id: &str) -> Message {
        Message::Assistant {
            content: String::new(),
            calls: vec![read_call(id)],
        }
    }

    /// A tool result for call `id`.
    fn tool_result(id: &str) -> Message {
        Message::Tool {
            id: id.to_string(),
            content: "x".to_string(),
        }
    }

    /// The system prompt and task that open every conversation.
    fn opening() -> Vec<Message> {
        vec![
            Message::System("system".to_string()),
            Message::User("task".to_string()),
        ]
    }

    /// The output of a native run of `task` in `root` with `max_steps` steps
    /// against a server that sends `replies` in order, and the run's result.
    async fn run_in(
        root: &Path,
        task: &str,
        max_steps: usize,
        replies: Vec<Reply>,
    ) -> Result<(String, Result<()>)> {
        let url = serve_replies(replies).await?;
        let client = OllamaClient::openai(&url, None, true)?;
        let config = test_config(root, max_steps);
        let mut reporter = RecordingReporter::default();
        let mut confirmer = FixedConfirmer::new(true);

        let result = run_native_agent(
            &config,
            &client,
            "m",
            task,
            None,
            &mut reporter,
            &mut confirmer,
        )
        .await;

        Ok((reporter.output, result))
    }

    /// Like [`run_in`], in a new empty workspace that is removed afterwards.
    async fn run_with_replies(
        task: &str,
        max_steps: usize,
        replies: Vec<Reply>,
    ) -> Result<(String, Result<()>)> {
        let root = crate::test_support::unique_temp_dir("openbatrangs-native-loop");

        let outcome = run_in(&root, task, max_steps, replies).await?;

        std::fs::remove_dir_all(root)?;
        Ok(outcome)
    }

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
            id: Some(String::new()),
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
        let mut messages = opening();

        for index in 0..30 {
            messages.push(assistant_call(&format!("c{index}")));
            messages.push(tool_result(&format!("c{index}")));
        }

        trim_messages(&mut messages, 6);

        assert!(matches!(messages[0], Message::System(_)));
        assert!(matches!(messages[1], Message::User(_)));
        assert!(matches!(messages[2], Message::Assistant { .. }));
        assert!(messages.len() <= 6);
    }

    #[test]
    fn trim_leaves_messages_alone_below_a_limit_of_two() {
        let mut messages = opening();
        messages.push(assistant_call("c1"));

        trim_messages(&mut messages, 1);

        assert_eq!(messages.len(), 3);
    }

    #[test]
    fn trims_whole_turns_without_splitting_a_call_from_its_result() {
        let mut messages = opening();
        messages.extend([
            assistant_call("c1"),
            tool_result("c1"),
            tool_result("c1b"),
            assistant_call("c2"),
            tool_result("c2"),
        ]);

        trim_messages(&mut messages, 4);

        assert_eq!(messages.len(), 4);
        assert!(matches!(messages[0], Message::System(_)));
        assert!(matches!(messages[1], Message::User(_)));
        assert_eq!(
            messages[2].calls().map(|calls| calls[0].id.as_str()),
            Some("c2")
        );
        assert!(messages[3].is_tool_result());
    }

    #[test]
    fn trims_never_leave_a_call_without_its_results() {
        let mut messages = opening();

        for index in 0..6 {
            messages.push(assistant_call(&format!("c{index}")));
            messages.push(tool_result(&format!("c{index}")));
        }

        trim_messages(&mut messages, 4);

        assert_eq!(messages.len(), 4);
        assert_eq!(
            messages[2].calls().map(|calls| calls[0].id.as_str()),
            Some("c5")
        );
        assert!(messages[3].is_tool_result());
    }

    #[test]
    fn trims_strip_orphan_tool_results() {
        let mut messages = opening();
        messages.extend([tool_result("lost"), assistant_call("c1"), tool_result("c1")]);

        trim_messages(&mut messages, 4);

        assert_eq!(messages.len(), 4);
        assert!(matches!(messages[2], Message::Assistant { .. }));
        assert!(messages[3].is_tool_result());
    }

    #[test]
    fn the_schema_offers_every_tool() {
        let schema = tools_schema();
        let names: Vec<&str> = schema
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|tool| tool.pointer("/function/name").and_then(Value::as_str))
            .collect();

        for name in [
            "web_search",
            "web_read",
            "edit_file",
            "run_command",
            "read_file",
            "ask_user",
            "finish",
        ] {
            assert!(names.contains(&name), "{name} is missing");
        }
    }

    #[test]
    fn invalid_arguments_become_an_empty_object() {
        assert_eq!(parse_arguments("not json"), json!({}));
        assert_eq!(parse_arguments(""), json!({}));
        assert_eq!(parse_arguments(" {\"path\":\"a\"} "), json!({"path": "a"}));
    }

    #[test]
    fn tool_result_messages_carry_the_call_id() {
        let message = tool_result("call_1");

        assert_eq!(message.to_json()["role"].as_str(), Some("tool"));
        assert_eq!(message.to_json()["tool_call_id"].as_str(), Some("call_1"));
    }

    #[test]
    fn the_task_carries_the_rules_only_when_thinking_is_off() {
        let config = test_config(Path::new("/tmp/workspace"), 4);

        let opening = |think| Opening {
            task: "write it",
            earlier: "",
            think,
        };

        let thinking = initial_user_message(&config, &opening(true));
        assert!(thinking.starts_with("Workspace root: /tmp/workspace"));
        assert!(!thinking.contains(NATIVE_SYSTEM_PROMPT));

        let not_thinking = initial_user_message(&config, &opening(false));
        assert!(not_thinking.contains(NATIVE_SYSTEM_PROMPT));
        assert!(not_thinking.ends_with("Task:\nwrite it"));
    }

    #[test]
    fn the_request_carries_the_thinking_flag_and_requires_tools() {
        let messages = vec![Message::System("rules".to_string())];
        let flag = |think| {
            request_body("m", &messages, 8192, think)["chat_template_kwargs"]["enable_thinking"]
                .clone()
        };

        assert_eq!(flag(false), json!(false));
        assert_eq!(flag(true), json!(true));

        let body = request_body("m", &messages, 8192, true);
        assert_eq!(body["tool_choice"], json!("required"));
        assert!(
            body["tools"]
                .as_array()
                .is_some_and(|tools| !tools.is_empty())
        );
    }

    #[test]
    fn every_message_round_trips_through_json() {
        let originals = [
            Message::System("rules".to_string()),
            Message::User("task".to_string()),
            Message::Assistant {
                content: "plain answer".to_string(),
                calls: Vec::new(),
            },
            Message::Assistant {
                content: "reading".to_string(),
                calls: vec![read_call("call_1")],
            },
            tool_result("call_1"),
        ];

        for original in originals {
            let restored = Message::from_json(&original.to_json());
            assert_eq!(format!("{restored:?}"), format!("{:?}", Some(original)));
        }
    }

    #[test]
    fn from_json_drops_unknown_roles_and_bad_calls() {
        assert!(Message::from_json(&json!({"role": "oracle", "content": "x"})).is_none());
        assert!(Message::from_json(&json!({"content": "x"})).is_none());

        let assistant = json!({"role": "assistant", "tool_calls": [{"id": "c1"}]});
        let restored = Message::from_json(&assistant);
        assert_eq!(
            restored.as_ref().and_then(Message::calls).map(<[_]>::len),
            Some(0)
        );
    }

    #[test]
    fn text_tool_calls_are_parsed_when_the_tool_is_known() -> Result<()> {
        let call = parse_text_tool_call(r#"{"name": "read_file", "arguments": {"path": "a.rs"}}"#)
            .context("known tool in text form")?;
        assert_eq!(call.name, "read_file");
        assert_eq!(call.arguments, r#"{"path":"a.rs"}"#);

        let fenced = parse_text_tool_call("```json\n{\"name\": \"list_files\"}\n```")
            .context("fenced tool call")?;
        assert_eq!(fenced.name, "list_files");
        assert_eq!(fenced.arguments, "{}");

        Ok(())
    }

    #[test]
    fn text_tool_calls_ignore_unknown_tools_and_plain_answers() {
        assert!(parse_text_tool_call(r#"{"name": "rm_rf", "arguments": {}}"#).is_none());
        assert!(parse_text_tool_call("the build failed because of x").is_none());
        assert!(parse_text_tool_call(r#"{"answer": "done"}"#).is_none());
    }

    #[test]
    fn resumed_messages_append_the_correction_as_a_user_turn() -> Result<()> {
        let prior = vec![
            json!({"role": "system", "content": "rules"}),
            json!({"role": "user", "content": "original task"}),
            json!({"role": "assistant", "content": "", "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}
            ]}),
            json!({"role": "tool", "tool_call_id": "c1", "content": "file contents"}),
        ];
        let config = test_config(Path::new("/tmp"), 4);

        let opening = Opening {
            task: "use the other API instead",
            earlier: "",
            think: true,
        };

        let messages = seed_messages(&config, &opening, Some(prior));

        assert_eq!(messages.len(), 5);
        assert!(matches!(messages[0], Message::System(_)));
        assert_eq!(
            format!("{:?}", messages[4]),
            r#"User("use the other API instead")"#
        );

        Ok(())
    }

    #[test]
    fn resumed_messages_with_bad_history_fall_back_to_a_fresh_start() {
        let config = test_config(Path::new("/tmp"), 4);
        let bad_history = vec![json!({"role": "oracle", "content": "x"})];

        let opening = Opening {
            task: "task",
            earlier: "",
            think: true,
        };

        let messages = seed_messages(&config, &opening, Some(bad_history));

        assert_eq!(messages.len(), 2);
        assert!(matches!(messages[0], Message::System(_)));
        assert!(matches!(messages[1], Message::User(_)));
    }

    #[test]
    fn previously_changed_files_reads_write_calls_from_history() {
        let call = |id: &str, name: &str, arguments: &str| Invocation {
            id: id.to_string(),
            name: name.to_string(),
            arguments: arguments.to_string(),
        };
        let messages = vec![Message::Assistant {
            content: String::new(),
            calls: vec![
                call("c1", "write_file", r#"{"path":"src/lib.rs","content":"x"}"#),
                call("c2", "edit_file", r#"{"path":"README.md"}"#),
                call("c3", "read_file", r#"{"path":"Cargo.toml"}"#),
                call("c4", "edit_file", r#"{"path":"src/lib.rs"}"#),
            ],
        }];

        assert_eq!(
            previously_changed_files(&messages),
            ["src/lib.rs", "README.md"]
        );
    }

    #[test]
    fn the_gate_replays_writes_and_checks_from_history() {
        let write = Invocation {
            id: "c1".to_string(),
            name: "write_file".to_string(),
            arguments: r#"{"path":"a.rs","content":"x"}"#.to_string(),
        };
        let written = Message::Assistant {
            content: String::new(),
            calls: vec![write],
        };
        let passed = Message::Tool {
            id: "c2".to_string(),
            content: "compiled\n[exit code 0]\n".to_string(),
        };

        let mut checked = gate_for_run(&[written.clone(), passed]);
        assert_eq!(checked.reject_finish(true), None);

        let mut unchecked = gate_for_run(&[written, Message::User("task".to_string())]);
        assert!(unchecked.reject_finish(true).is_some());
    }

    #[tokio::test]
    async fn reasoning_is_printed_on_its_own_line_when_shown() -> Result<()> {
        let deltas = || {
            futures_util::stream::iter([
                Ok(Delta::Thinking("plan".to_string())),
                Ok(Delta::Thinking(" more".to_string())),
                Ok(Delta::Content("Hi".to_string())),
                Ok(Delta::Thinking("again".to_string())),
                Ok(Delta::ToolCalls(vec![ToolCallDelta::default()])),
                Ok(Delta::Thinking("last".to_string())),
            ])
        };

        let native_step = || AgentEvent::StepStarted {
            step: 1,
            max_steps: 1,
            model: "m".to_string(),
            loop_kind: LoopKind::Native,
        };

        let mut shown = RecordingReporter::default();
        shown.report(native_step());
        let turn = collect_turn(deltas(), &mut shown, true).await?;
        assert!(
            shown
                .output
                .ends_with(" m\n💭 plan more\nHi💭 again\n💭 last\n")
        );
        assert_eq!(turn.content, "Hi");
        assert_eq!(turn.calls.len(), 1);

        let mut hidden = RecordingReporter::default();
        hidden.report(native_step());
        collect_turn(deltas(), &mut hidden, false).await?;
        assert!(hidden.output.ends_with(" m\nHi"));

        Ok(())
    }

    #[tokio::test]
    async fn a_tool_call_and_a_finish_complete_the_task() -> Result<()> {
        let root = crate::test_support::unique_temp_dir("openbatrangs-native-write");
        let write = json!({"path": "a.txt", "content": "hello"});
        let replies = vec![
            Reply::openai_stream(&[tool_call_delta("c1", "write_file", &write)]),
            Reply::openai_stream(&[tool_call_delta(
                "c2",
                "finish",
                &json!({"summary": "wrote it"}),
            )]),
        ];

        let (output, result) = run_in(&root, "write a.txt", 3, replies).await?;

        result?;
        assert!(output.contains("🔧 write_file → a.txt"));
        assert!(output.contains("✅ wrote it"));
        assert_eq!(std::fs::read_to_string(root.join("a.txt"))?, "hello");

        std::fs::remove_dir_all(root)?;
        Ok(())
    }

    #[test]
    fn calls_without_an_id_get_unique_ids() {
        let calls = vec![read_call(""), read_call("server_id"), read_call("")];

        let ids: Vec<String> = identified_calls(calls, 3)
            .into_iter()
            .map(|call| call.id)
            .collect();

        assert_eq!(ids, vec!["call_3_0", "server_id", "call_3_2"]);
    }

    #[tokio::test]
    async fn a_call_without_an_id_and_its_result_share_one_id() -> Result<()> {
        let root = crate::test_support::unique_temp_dir("openbatrangs-native-ids");
        let replies = vec![
            Reply::openai_stream(&[text_delta(r#"{"name":"list_files","arguments":{}}"#)]),
            Reply::openai_stream(&[text_delta("nothing here")]),
        ];
        let url = serve_replies(replies).await?;
        let client = OllamaClient::openai(&url, None, true)?;
        let config = test_config(&root, 3);
        let mut sink = RecordingSink::default();
        let continuation = RunContinuation {
            sink: Some(&mut sink),
            ..RunContinuation::default()
        };
        let mut reporter = RecordingReporter::default();
        let mut confirmer = FixedConfirmer::new(true);

        run_native_agent(
            &config,
            &client,
            "m",
            "look around",
            Some(continuation),
            &mut reporter,
            &mut confirmer,
        )
        .await?;

        let last = sink.snapshots.last().context("a saved conversation")?;
        assert_eq!(last[2]["tool_calls"][0]["id"], "call_1_0");
        assert_eq!(last[3]["tool_call_id"], "call_1_0");

        std::fs::remove_dir_all(root)?;
        Ok(())
    }

    #[tokio::test]
    async fn a_tool_call_written_as_text_is_executed() -> Result<()> {
        let replies = vec![
            Reply::openai_stream(&[text_delta(r#"{"name":"list_files","arguments":{}}"#)]),
            Reply::openai_stream(&[text_delta("nothing here")]),
        ];

        let (output, result) = run_with_replies("look around", 3, replies).await?;

        result?;
        assert!(output.contains("wrote its tool call as text"));
        assert!(output.contains("🔧 list_files → ."));
        assert!(output.contains("✅ nothing here"));

        Ok(())
    }

    #[tokio::test]
    async fn empty_turns_are_nudged_twice_then_accepted() -> Result<()> {
        let replies = vec![
            Reply::openai_stream(&[]),
            Reply::openai_stream(&[text_delta("  ")]),
            Reply::openai_stream(&[]),
        ];

        let (output, result) = run_with_replies("say hi", 5, replies).await?;

        result?;
        assert_eq!(output.matches("Empty model turn").count(), 2);
        assert!(output.contains("The model returned an empty reply."));

        Ok(())
    }

    #[tokio::test]
    async fn a_tool_call_resets_the_empty_turn_count() -> Result<()> {
        let list = tool_call_delta("c1", "list_files", &json!({}));
        let replies = vec![
            Reply::openai_stream(&[]),
            Reply::openai_stream(&[list]),
            Reply::openai_stream(&[]),
            Reply::openai_stream(&[]),
            Reply::openai_stream(&[text_delta("done")]),
        ];

        let (output, result) = run_with_replies("look around", 5, replies).await?;

        result?;
        assert_eq!(output.matches("Empty model turn").count(), 3);
        assert!(output.contains("✅ done"));

        Ok(())
    }

    #[tokio::test]
    async fn a_tool_call_written_as_text_resets_the_empty_turn_count() -> Result<()> {
        let list = text_delta(r#"{"name":"list_files","arguments":{}}"#);
        let replies = vec![
            Reply::openai_stream(&[]),
            Reply::openai_stream(&[]),
            Reply::openai_stream(&[list]),
            Reply::openai_stream(&[]),
            Reply::openai_stream(&[text_delta("done")]),
        ];

        let (output, result) = run_with_replies("look around", 5, replies).await?;

        result?;
        assert_eq!(output.matches("Empty model turn").count(), 3);
        assert!(output.contains("✅ done"));

        Ok(())
    }

    #[tokio::test]
    async fn a_change_in_a_cargo_project_is_checked_before_finishing() -> Result<()> {
        let root = crate::test_support::unique_temp_dir("openbatrangs-native-check");
        std::fs::write(root.join("Cargo.toml"), "[package]")?;
        let write = json!({"path": "a.txt", "content": "hello"});
        let finish = tool_call_delta("c2", "finish", &json!({"summary": "done"}));
        let replies = vec![
            Reply::openai_stream(&[tool_call_delta("c1", "write_file", &write)]),
            Reply::openai_stream(&[text_delta("all done")]),
            Reply::openai_stream(&[finish]),
        ];

        let (output, result) = run_in(&root, "change a.txt", 3, replies).await?;

        result?;
        assert_eq!(
            output
                .matches("Finish rejected: the task is not done.")
                .count(),
            2
        );
        assert!(output.contains("Reached max steps (3) without a final answer."));

        std::fs::remove_dir_all(root)?;
        Ok(())
    }

    #[tokio::test]
    async fn a_question_with_work_words_is_answered_at_once() -> Result<()> {
        let root = crate::test_support::unique_temp_dir("openbatrangs-native-question");
        std::fs::write(root.join("Cargo.toml"), "[package]")?;
        let code = "```rust\nfn main() {}\n```";
        let finish = tool_call_delta("c1", "finish", &json!({"summary": code}));
        let task = "show me how to write and add tests for a directory walker, just the code";

        let (output, result) =
            run_in(&root, task, 3, vec![Reply::openai_stream(&[finish])]).await?;

        result?;
        assert!(output.contains("✅ ```rust"));
        assert!(!output.contains("🔧"));
        assert!(!output.contains("Finish rejected"));

        let answered = vec![Reply::openai_stream(&[text_delta("Use walkdir.")])];
        let (output, result) = run_in(&root, task, 3, answered).await?;

        result?;
        assert!(output.contains("✅ Use walkdir."));

        std::fs::remove_dir_all(root)?;
        Ok(())
    }

    #[tokio::test]
    async fn ask_user_ends_the_run_after_one_model_call() -> Result<()> {
        let root = crate::test_support::unique_temp_dir("openbatrangs-native-ask");
        let ask = tool_call_delta("c1", "ask_user", &json!({"question": "Which file?"}));
        let url = serve_replies(vec![Reply::openai_stream(&[ask])]).await?;
        let client = OllamaClient::openai(&url, None, true)?;
        let config = test_config(&root, 5);
        let mut sink = RecordingSink::default();
        let continuation = RunContinuation {
            sink: Some(&mut sink),
            ..RunContinuation::default()
        };
        let mut reporter = RecordingReporter::default();
        let mut confirmer = FixedConfirmer::new(true);

        run_native_agent(
            &config,
            &client,
            "m",
            "fix it",
            Some(continuation),
            &mut reporter,
            &mut confirmer,
        )
        .await?;

        let question = AgentEvent::AskUser {
            question: "Which file?".to_string(),
        };
        assert_eq!(reporter.events.last(), Some(&question));
        assert!(reporter.output.contains("❓ Which file?"));
        assert!(!reporter.output.contains("step 2/5"));
        let last = sink.snapshots.last().context("a saved conversation")?;
        assert_eq!(last.len(), 4);
        assert_eq!(last[3]["content"], QUESTION_SHOWN);

        std::fs::remove_dir_all(root)?;
        Ok(())
    }

    #[tokio::test]
    async fn an_unknown_tool_is_rejected_for_a_retry() -> Result<()> {
        let replies = vec![
            Reply::openai_stream(&[tool_call_delta("c1", "rm_rf", &json!({}))]),
            Reply::openai_stream(&[text_delta("ok")]),
        ];

        let (output, result) = run_with_replies("clean up", 3, replies).await?;

        result?;
        assert!(output.contains("Tool call rejected: unknown tool: rm_rf"));
        assert!(output.contains("✅ ok"));

        Ok(())
    }

    /// `count` replies that each call `list_files` on the workspace root.
    fn identical_list_calls(count: usize) -> Vec<Reply> {
        (0..count)
            .map(|index| {
                let id = format!("c{index}");
                Reply::openai_stream(&[tool_call_delta(&id, "list_files", &json!({}))])
            })
            .collect()
    }

    #[tokio::test]
    async fn the_fifth_identical_call_stops_the_run() -> Result<()> {
        let replies = identical_list_calls(5);

        let (output, result) = run_with_replies("look around", 10, replies).await?;

        let error = result.err().map(|error| error.to_string());
        assert_eq!(
            error.as_deref(),
            Some("stopped: the model repeated the same tool call 5 times")
        );
        assert!(output.contains("step 5/10"));
        assert!(!output.contains("step 6/10"));

        Ok(())
    }

    #[tokio::test]
    async fn the_third_identical_call_tells_the_model_to_change_approach() -> Result<()> {
        let root = crate::test_support::unique_temp_dir("openbatrangs-native-repeat");
        let mut replies = identical_list_calls(3);
        let finish = tool_call_delta("done", "finish", &json!({"summary": "looked"}));
        replies.push(Reply::openai_stream(&[finish]));
        let url = serve_replies(replies).await?;
        let client = OllamaClient::openai(&url, None, true)?;
        let config = test_config(&root, 10);
        let mut sink = RecordingSink::default();
        let continuation = RunContinuation {
            sink: Some(&mut sink),
            ..RunContinuation::default()
        };
        let mut reporter = RecordingReporter::default();
        let mut confirmer = FixedConfirmer::new(true);

        run_native_agent(
            &config,
            &client,
            "m",
            "look around",
            Some(continuation),
            &mut reporter,
            &mut confirmer,
        )
        .await?;

        let last = sink.snapshots.last().context("a saved conversation")?;
        let tool_results: Vec<&str> = last
            .iter()
            .filter(|message| message["role"] == "tool")
            .filter_map(|message| message["content"].as_str())
            .collect();
        assert!(!tool_results[1].contains("same result"));
        assert!(tool_results[2].contains("same result 3 times in a row"));

        std::fs::remove_dir_all(root)?;
        Ok(())
    }

    #[tokio::test]
    async fn a_failed_model_call_names_the_step() -> Result<()> {
        let replies = vec![Reply::server_error("no model")];

        let (_output, result) = run_with_replies("say hi", 3, replies).await?;

        assert!(format!("{result:?}").contains("model call failed at step 1"));

        Ok(())
    }

    #[tokio::test]
    async fn a_missing_workspace_is_an_error() -> Result<()> {
        let missing = PathBuf::from("/definitely/not/a/workspace");

        let (_output, result) = run_in(&missing, "say hi", 1, Vec::new()).await?;

        assert!(result.is_err());

        Ok(())
    }

    #[tokio::test]
    async fn a_resumed_run_continues_the_stored_conversation() -> Result<()> {
        let root = crate::test_support::unique_temp_dir("openbatrangs-native-resume");
        let url = serve_replies(vec![Reply::openai_stream(&[text_delta("switched")])]).await?;
        let client = OllamaClient::openai(&url, None, false)?;
        let config = test_config(&root, 2);
        let prior = vec![
            json!({"role": "system", "content": "rules"}),
            json!({"role": "user", "content": "explain the parser"}),
            json!({"role": "assistant", "content": "it parses"}),
        ];
        let mut sink = RecordingSink::default();
        let continuation = RunContinuation {
            messages: Some(prior),
            sink: Some(&mut sink),
            ..RunContinuation::default()
        };
        let mut reporter = RecordingReporter::default();
        let mut confirmer = FixedConfirmer::new(true);

        run_native_agent(
            &config,
            &client,
            "m",
            "use the other docs API",
            Some(continuation),
            &mut reporter,
            &mut confirmer,
        )
        .await?;

        // The prior three messages, the correction, and the answer.
        assert_eq!(sink.snapshots.len(), 2);
        let last = sink.snapshots.last().cloned().unwrap_or_default();
        assert_eq!(last.len(), 5);
        assert_eq!(last[4], json!({"role": "assistant", "content": "switched"}));
        assert!(!reporter.output.contains("web_search"));
        assert!(reporter.output.contains("✅ switched"));

        std::fs::remove_dir_all(root)?;
        Ok(())
    }

    #[tokio::test]
    async fn a_new_run_gets_the_earlier_turns_in_its_first_message() -> Result<()> {
        let root = crate::test_support::unique_temp_dir("openbatrangs-native-earlier");
        let url = serve_replies(vec![Reply::openai_stream(&[text_delta("done")])]).await?;
        let client = OllamaClient::openai(&url, None, false)?;
        let config = test_config(&root, 2);
        let earlier = vec![
            ChatMessage {
                role: crate::ollama::Role::User,
                content: "rename the parser".to_string(),
            },
            ChatMessage {
                role: crate::ollama::Role::Assistant,
                content: "Renamed it to reader.".to_string(),
            },
        ];
        let mut sink = RecordingSink::default();
        let continuation = RunContinuation {
            earlier,
            sink: Some(&mut sink),
            ..RunContinuation::default()
        };
        let mut reporter = RecordingReporter::default();
        let mut confirmer = FixedConfirmer::new(true);

        run_native_agent(
            &config,
            &client,
            "m",
            "print our conversation so far",
            Some(continuation),
            &mut reporter,
            &mut confirmer,
        )
        .await?;

        let first = sink.snapshots.first().cloned().unwrap_or_default();
        let task = first[1]["content"].as_str().unwrap_or_default();
        assert!(task.contains("Earlier in this conversation:"));
        assert!(task.contains("User: rename the parser\nAssistant: Renamed it to reader."));
        assert!(task.ends_with("Task:\nprint our conversation so far"));

        std::fs::remove_dir_all(root)?;
        Ok(())
    }

    /// An assistant turn that calls `read_file` with call id `id`.
    fn read_turn(id: &str) -> Message {
        Message::Assistant {
            content: String::new(),
            calls: vec![Invocation {
                id: id.to_string(),
                name: "read_file".to_string(),
                arguments: "{}".to_string(),
            }],
        }
    }

    #[test]
    fn large_tool_results_are_cut_to_fit_the_context_window() {
        let mut messages = vec![
            Message::System("rules".to_string()),
            Message::User("task".to_string()),
        ];

        for index in 0..19 {
            let id = format!("c{index}");
            messages.push(read_turn(&id));
            messages.push(Message::Tool {
                id,
                content: "x".repeat(20_000),
            });
        }

        let budget = prompt_budget(8_192);
        trim_messages(&mut messages, MAX_HISTORY_MESSAGES);
        fit_to_budget(&mut messages, budget);

        let body = request_body("m", &messages, 8_192, true);
        let prompt_chars = total_chars(&messages) + tools_schema().to_string().len();
        let prompt_tokens = (prompt_chars as f64 / CHARS_PER_TOKEN).ceil() as u64;
        let max_tokens = body["max_tokens"].as_u64().unwrap_or(u64::MAX);
        assert!(prompt_tokens + max_tokens <= 8_192);

        // The newest turn is kept, with its call and its result.
        assert!(matches!(messages[2], Message::Assistant { .. }));
        assert!(messages.last().is_some_and(Message::is_tool_result));
    }

    #[test]
    fn a_turn_too_large_alone_keeps_its_call_and_a_shortened_result() {
        let mut messages = vec![
            Message::System("rules".to_string()),
            Message::User("task".to_string()),
            read_turn("c1"),
            Message::Tool {
                id: "c1".to_string(),
                content: "y".repeat(100_000),
            },
        ];

        fit_to_budget(&mut messages, 4_000);

        assert_eq!(messages.len(), 4);
        assert!(total_chars(&messages) < 4_000 + 100);
    }

    #[test]
    fn messages_within_the_budget_stay_as_they_are() {
        let mut messages = vec![
            Message::System("rules".to_string()),
            Message::User("task".to_string()),
        ];

        fit_to_budget(&mut messages, 1_000);

        assert_eq!(messages.len(), 2);
    }

    /// The resume sink and the transcript sink of a native run of `task` in
    /// `root` with `max_steps` steps against a server that sends `replies`,
    /// and the run's result.
    async fn run_with_transcript(
        root: &Path,
        task: &str,
        max_steps: usize,
        replies: Vec<Reply>,
    ) -> Result<(RecordingSink, RecordingSink, Result<()>)> {
        let url = serve_replies(replies).await?;
        let client = OllamaClient::openai(&url, None, true)?;
        let config = test_config(root, max_steps);
        let mut sink = RecordingSink::default();
        let mut transcript = RecordingSink::default();
        let continuation = RunContinuation {
            sink: Some(&mut sink),
            transcript: Some(&mut transcript),
            ..RunContinuation::default()
        };
        let mut reporter = RecordingReporter::default();
        let mut confirmer = FixedConfirmer::new(true);

        let result = run_native_agent(
            &config,
            &client,
            "m",
            task,
            Some(continuation),
            &mut reporter,
            &mut confirmer,
        )
        .await;

        Ok((sink, transcript, result))
    }

    /// The roles of `messages`, in order.
    fn roles(messages: &[Value]) -> Vec<&str> {
        messages
            .iter()
            .map(|message| message["role"].as_str().unwrap_or("none"))
            .collect()
    }

    #[tokio::test]
    async fn the_transcript_keeps_what_the_context_budget_trims() -> Result<()> {
        let root = crate::test_support::unique_temp_dir("openbatrangs-native-transcript");
        std::fs::write(root.join("big.txt"), "x".repeat(7_000))?;
        let read = json!({"path": "big.txt"});
        let mut replies: Vec<Reply> = (0..4)
            .map(|index| {
                let id = format!("c{index}");
                Reply::openai_stream(&[tool_call_delta(&id, "read_file", &read)])
            })
            .collect();
        replies.push(Reply::openai_stream(&[text_delta("big.txt is all x")]));

        let (sink, transcript, result) =
            run_with_transcript(&root, "read big.txt", 6, replies).await?;

        result?;
        let saved = transcript.snapshots.concat();
        let resumable = sink.snapshots.last().context("a saved conversation")?;
        let expected = vec![
            "system",
            "user",
            "assistant",
            "tool",
            "assistant",
            "tool",
            "assistant",
            "tool",
            "assistant",
            "tool",
            "assistant",
        ];
        assert_eq!(transcript.snapshots.len(), 1);
        assert_eq!(roles(&saved), expected);
        assert!(
            saved[3]["content"]
                .as_str()
                .is_some_and(|text| text.len() >= 7_000)
        );
        assert!(roles(resumable).len() < saved.len());

        std::fs::remove_dir_all(root)?;
        Ok(())
    }

    #[tokio::test]
    async fn the_transcript_is_saved_when_a_model_call_fails() -> Result<()> {
        let root = crate::test_support::unique_temp_dir("openbatrangs-native-transcript-error");
        let replies = vec![
            Reply::openai_stream(&[tool_call_delta("c1", "list_files", &json!({}))]),
            Reply::server_error("model crashed"),
        ];

        let (_sink, transcript, result) =
            run_with_transcript(&root, "look around", 5, replies).await?;

        assert!(result.is_err());
        let saved = transcript.snapshots.concat();
        assert_eq!(roles(&saved), vec!["system", "user", "assistant", "tool"]);
        assert_eq!(saved[2]["tool_calls"][0]["function"]["name"], "list_files");
        assert_eq!(saved[3]["tool_call_id"], "c1");

        std::fs::remove_dir_all(root)?;
        Ok(())
    }

    #[test]
    fn the_shared_tool_definitions_are_the_ones_sent() {
        let body = request_body("m", &opening(), 8_192, true);

        assert_eq!(body["tools"], crate::agent::native_tool_definitions());
    }
}
