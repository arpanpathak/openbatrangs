//! # JSON tool loop for Ollama servers
//!
//! Ollama's `/api/chat` has no tool-call channel the loop relies on, so the
//! model answers every step with one JSON object: a tool call, or the final
//! answer. Each step streams that object, prints its `thought` and `answer`
//! fields as they arrive, and then acts on it:
//!
//! - an answer, or a call to `finish`, ends the run unless the finish gate
//!   rejects it as premature;
//! - a call to `ask_user` ends the run with the question;
//! - any other tool call runs, and its result goes back as a user message;
//! - a reply that is neither, or is not JSON, ends the run with a warning.
//!
//! The run is a state machine: [`JsonRun::run`] moves through [`State`]
//! values, one small function per state, until the run is over.

use super::environment::{environment_facts, project_has_build_manifest};
use super::execute::{absolute_paths, run_tool_call};
use super::quality::FinishGate;
use super::research::task_with_documentation;
use super::stream::stream_model_response;
use super::tool::{AgentResponse, Tool, ToolCall, parse_agent_response};
use super::{
    AgentConfig, AgentEvent, Confirmer, LoopKind, Reporter, RunContinuation, compact_context,
    earlier_turns_section, effective_context, ensure_workspace_exists, max_output_tokens,
};
use crate::constants::agent::{
    AGENT_TEMPERATURE, INITIAL_LIST_DEPTH, MAX_HISTORY_MESSAGES, SYSTEM_PROMPT, TOOL_RESULT_PREFIX,
};
use crate::constants::ansi::{COLOR_DIM, COLOR_RESET};
use crate::ollama::{ChatMessage, ChatRequest, OllamaClient, Role};
use crate::tools;
use anyhow::{Context, Result};
use std::path::Path;

/// The states of one JSON-loop run.
enum State {
    /// Start the next step, or give up when the step budget is spent.
    Ask,
    /// Stream the model's reply for the current step.
    Stream,
    /// Parse the streamed reply.
    Parse {
        /// The raw reply.
        content: String,
        /// Whether its answer field was streamed to the reporter.
        answer_was_streamed: bool,
    },
    /// Turn the reply's tool call into a typed tool.
    CallTool {
        /// The raw reply.
        content: String,
        /// The tool call in the reply.
        call: ToolCall,
    },
    /// Run a tool and send its result back.
    RunTool {
        /// The raw reply.
        content: String,
        /// Tool name, as the model wrote it.
        name: String,
        /// Tool arguments as JSON text.
        arguments: String,
        /// The typed tool.
        tool: Tool,
    },
    /// Send a malformed tool call back for a retry.
    Reject {
        /// The raw reply.
        content: String,
        /// Why the call was rejected.
        error: anyhow::Error,
    },
    /// The model answered or called `finish`.
    Finish {
        /// The answer or summary.
        summary: String,
        /// Whether the answer was streamed to the reporter.
        answer_was_streamed: bool,
    },
    /// The model asked the user this question.
    AskUser(String),
    /// The reply had neither an answer nor a tool call.
    Empty(String),
    /// The reply was not JSON.
    Unparsable {
        /// The raw reply.
        content: String,
        /// The parse error.
        error: anyhow::Error,
    },
    /// The step budget ran out before an answer.
    GiveUp,
    /// The run is over.
    Done,
}

/// One run of the JSON loop: its limits, its conversation, and its output.
struct JsonRun<'a, R, C> {
    /// Agent configuration (cwd, limits, safety flags).
    config: &'a AgentConfig,
    /// Ollama client for model API calls.
    client: &'a OllamaClient,
    /// Name of the model being used.
    model: &'a str,
    /// Context window size (tokens) sent to the model.
    num_ctx: u64,
    /// Whether the project has a build or test manifest, so changed files
    /// need a passing check before the run may finish.
    needs_check: bool,
    /// Number of the current step; 0 before the first.
    step: usize,
    /// Conversation history sent to the model.
    messages: Vec<ChatMessage>,
    /// Files written successfully during the task.
    changed_files: Vec<String>,
    /// Gate that rejects a finish before changed files are checked.
    finish_gate: FinishGate,
    /// Receives the run's events.
    reporter: &'a mut R,
    /// Human-in-the-loop gate for writes and commands.
    confirmer: &'a mut C,
}

/// Run the JSON loop until the model answers or the step budget is spent.
///
/// A conversation in `continuation` cannot be resumed here; the run reports
/// that and starts from `task`.
///
/// # Returns
///
/// `Ok(())` when the loop ends, or an error when the workspace is missing or
/// a model call fails.
pub(super) async fn run_json_agent<'a, R: Reporter, C: Confirmer>(
    config: &'a AgentConfig,
    client: &'a OllamaClient,
    model: &'a str,
    task: &str,
    continuation: Option<RunContinuation<'a>>,
    reporter: &'a mut R,
    confirmer: &'a mut C,
) -> Result<()> {
    ensure_workspace_exists(&config.cwd)?;
    warn_when_resume_is_unsupported(continuation.as_ref(), reporter);

    let num_ctx = effective_context(config.model_context, config.max_ctx);
    let earlier = continuation.map(|run| run.earlier).unwrap_or_default();
    let task_with_docs = task_with_documentation(task, config, reporter).await;
    let opening = Opening {
        task: &task_with_docs,
        earlier: &earlier_turns_section(&earlier, num_ctx),
    };

    let run = JsonRun {
        config,
        client,
        model,
        num_ctx,
        needs_check: project_has_build_manifest(&config.cwd),
        step: 0,
        messages: initial_messages(&config.cwd, &opening),
        changed_files: Vec::new(),
        finish_gate: FinishGate::default(),
        reporter,
        confirmer,
    };
    run.run().await
}

/// What the first user turn of a run is made of.
struct Opening<'a> {
    /// The task, with any documentation found for it.
    task: &'a str,
    /// The earlier turns of the session as a section of the first message;
    /// empty when there are none.
    earlier: &'a str,
}

/// Report that the interrupted conversation is dropped, when there is one:
/// the JSON loop always starts from the task.
fn warn_when_resume_is_unsupported<R: Reporter>(
    continuation: Option<&RunContinuation<'_>>,
    reporter: &mut R,
) {
    let has_conversation = matches!(
        continuation,
        Some(RunContinuation {
            messages: Some(..),
            ..
        })
    );

    if !has_conversation {
        return;
    }

    reporter.notice(
        "\n⚠️  The previous conversation cannot be resumed: this server uses the JSON tool loop."
            .to_string(),
    );
}

impl<R: Reporter, C: Confirmer> JsonRun<'_, R, C> {
    /// Move from state to state until the run is over.
    async fn run(mut self) -> Result<()> {
        let mut state = State::Ask;

        loop {
            state = match state {
                State::Ask => self.ask(),
                State::Stream => self.stream().await?,
                State::Parse {
                    content,
                    answer_was_streamed,
                } => parse_state(content, answer_was_streamed),
                State::CallTool { content, call } => call_state(content, call),
                State::RunTool {
                    content,
                    name,
                    arguments,
                    tool,
                } => self.run_tool(&content, name, arguments, tool).await,
                State::Reject { content, error } => self.reject_tool_call(&content, &error),
                State::Finish {
                    summary,
                    answer_was_streamed,
                } => self.finish(summary, answer_was_streamed),
                State::AskUser(question) => self.ask_user(question),
                State::Empty(content) => self.report_empty_reply(&content),
                State::Unparsable { content, error } => self.report_parse_error(&error, &content),
                State::GiveUp => self.give_up(),
                State::Done => return Ok(()),
            };
        }
    }

    /// Start the next step, or give up when the budget is spent.
    fn ask(&mut self) -> State {
        self.step += 1;

        if self.step > self.config.max_steps {
            return State::GiveUp;
        }

        self.reporter.report(AgentEvent::StepStarted {
            step: self.step,
            max_steps: self.config.max_steps,
            model: self.model.to_string(),
            loop_kind: LoopKind::Json,
        });

        // Auto rolling context: keep the system prompt, original task, and the
        // newest tool exchange verbatim; older tool calls/output are rolled into a
        // short note so the prompt always fits the safe per-device `num_ctx`.
        compact_context(&mut self.messages, self.num_ctx);

        State::Stream
    }

    /// Stream the model's reply, showing its thought and answer as they arrive.
    async fn stream(&mut self) -> Result<State> {
        let request = chat_request(self.model, &self.messages, self.num_ctx);
        let show_thinking = self.config.show_thinking;

        let (content, answer_was_streamed) =
            stream_model_response(self.client, request, self.reporter, show_thinking)
                .await
                .with_context(|| format!("model call failed at step {}", self.step))?;

        Ok(State::Parse {
            content,
            answer_was_streamed,
        })
    }

    /// End the run with `summary`, unless the finish gate rejects the finish.
    fn finish(&mut self, summary: String, answer_was_streamed: bool) -> State {
        if let Some(feedback) = self.finish_gate.reject_finish(self.needs_check) {
            return self.block_finish(feedback);
        }

        let changed_files = absolute_paths(&self.config.cwd, &self.changed_files);
        self.reporter.report(AgentEvent::Finished {
            answer: summary,
            answer_was_streamed,
            changed_files,
        });

        State::Done
    }

    /// Send the gate's feedback to the model and keep going.
    fn block_finish(&mut self, feedback: &str) -> State {
        self.messages.push(message(Role::User, feedback));
        self.reporter.notice(format!(
            "\n{COLOR_DIM}⚠️  Shallow finish blocked.{COLOR_RESET}"
        ));

        State::Ask
    }

    /// Show the question and end the run.
    fn ask_user(&mut self, question: String) -> State {
        self.reporter.report(AgentEvent::AskUser { question });

        State::Done
    }

    /// A malformed call does not end the run: report what the model sent,
    /// send the error back as a user message, and continue, so the next
    /// reply can correct the call.
    fn reject_tool_call(&mut self, content: &str, error: &anyhow::Error) -> State {
        self.reporter.notice(format!(
            "\n{COLOR_DIM}⚠️  Tool call rejected: {error}\n   model sent: {}{COLOR_RESET}",
            content.trim()
        ));

        self.messages.push(message(Role::Assistant, content));
        self.messages.push(message(
            Role::User,
            &format!(
                "That tool call was rejected: {error}. Reply again with one JSON object \
                 using the exact argument names, e.g. \
                 {{\"tool\":{{\"name\":\"read_file\",\"arguments\":{{\"path\":\"Cargo.toml\"}}}}}}"
            ),
        ));

        State::Ask
    }

    /// Push the assistant JSON, execute the tool, push its result, and
    /// continue to the next step.
    async fn run_tool(
        &mut self,
        content: &str,
        name: String,
        arguments: String,
        tool: Tool,
    ) -> State {
        let id = format!("call_{}", self.step);
        self.messages.push(message(Role::Assistant, content));
        self.reporter.report(AgentEvent::ToolCall {
            id: id.clone(),
            name,
            arguments,
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

        let tool_result = format!("{TOOL_RESULT_PREFIX}\n{}", report.output);
        self.reporter.report(AgentEvent::ToolResult {
            id,
            success: report.success,
            summary: report.summary,
            output: report.output,
        });
        self.messages.push(message(Role::User, &tool_result));
        trim_messages(&mut self.messages, MAX_HISTORY_MESSAGES);

        State::Ask
    }

    /// Neither field is a model failure; surface it and stop cleanly.
    fn report_empty_reply(&mut self, content: &str) -> State {
        self.reporter.notice(format!(
            "\n⚠️  Model returned neither an answer nor a tool call.\n   model sent: {}",
            content.trim()
        ));

        if matches!(content.trim(), "" | "{}") {
            self.reporter.notice(format!(
                "   An empty reply usually means the context window ran out while the \
                 model was thinking; the window was {} tokens.",
                self.num_ctx
            ));
        }

        State::Done
    }

    /// Report a JSON parse failure and show the raw model output as fallback.
    fn report_parse_error(&mut self, error: &anyhow::Error, content: &str) -> State {
        self.reporter
            .notice(format!("\n⚠️  Could not parse model JSON: {error}"));
        self.reporter
            .notice("Showing raw model output as the final response.".to_string());
        self.reporter.notice(format!("\n{content}"));

        State::Done
    }

    /// Warn that the step budget ran out without an answer.
    fn give_up(&mut self) -> State {
        self.reporter.notice(format!(
            "\n⚠️  Reached max steps ({}) without a final answer.",
            self.config.max_steps
        ));

        State::Done
    }
}

/// The state for a streamed reply: act on the parsed JSON, or show the raw
/// text when it is not JSON.
fn parse_state(content: String, answer_was_streamed: bool) -> State {
    match parse_agent_response(&content) {
        Ok(response) => response_state(content, response, answer_was_streamed),
        Err(error) => State::Unparsable { content, error },
    }
}

/// The model may answer, call exactly one tool, or do neither. Matching on
/// the pair keeps all three outcomes explicit.
fn response_state(content: String, response: AgentResponse, answer_was_streamed: bool) -> State {
    match (response.answer, response.tool) {
        (Some(summary), _) => State::Finish {
            summary,
            answer_was_streamed,
        },
        (None, Some(call)) => State::CallTool { content, call },
        (None, None) => State::Empty(content),
    }
}

/// The state for a tool call: `finish` and `ask_user` end the run, a
/// malformed call goes back to the model, any other tool runs.
fn call_state(content: String, call: ToolCall) -> State {
    let name = call.name.clone();
    let arguments = call.arguments.to_string();

    match Tool::from_call(call) {
        Ok(Tool::Finish { summary }) => State::Finish {
            summary,
            answer_was_streamed: false,
        },
        Ok(Tool::AskUser { question }) => State::AskUser(question),
        Ok(tool) => State::RunTool {
            content,
            name,
            arguments,
            tool,
        },
        Err(error) => State::Reject { content, error },
    }
}

/// A chat message with `role` and `content`.
fn message(role: Role, content: &str) -> ChatMessage {
    ChatMessage {
        role,
        content: content.to_string(),
    }
}

/// Phrases that signal the user wants information about the current workspace
/// or project structure. Only these tasks get an automatic top-level listing;
/// everything else starts with zero filesystem scanning.
const WORKSPACE_SCAN_HINTS: &[&str] = &[
    "current dir",
    "current directory",
    "this directory",
    "this folder",
    "this project",
    "this repo",
    "this repository",
    "workspace",
    "codebase",
    "repository",
    "file structure",
    "project structure",
    "list files",
    "what files",
    "explore",
];

/// True when the task explicitly asks about the project/workspace layout.
fn task_requests_workspace_scan(task: &str) -> bool {
    let task_lower = task.to_ascii_lowercase();

    WORKSPACE_SCAN_HINTS
        .iter()
        .any(|hint| task_lower.contains(hint))
}

/// Build the system + first user turn.
///
/// The workspace is only listed when the user actually asks about the current
/// directory/project structure. For any other task the agent starts with zero
/// filesystem scanning, which keeps agent mode fast and avoids flooding the
/// model with irrelevant files.
fn initial_messages(cwd: &Path, opening: &Opening) -> Vec<ChatMessage> {
    let user_content = format!(
        "{}\n\nEnvironment:\n{}{}\n\nTask:\n{}",
        workspace_note(cwd, opening.task),
        environment_facts(cwd),
        opening.earlier,
        opening.task
    );

    vec![
        message(Role::System, SYSTEM_PROMPT),
        message(Role::User, &user_content),
    ]
}

/// The workspace root, with a shallow top-level listing when the task asks
/// about the project structure. The model calls `list_files` on specific
/// directories instead of getting a recursive scan.
fn workspace_note(cwd: &Path, task: &str) -> String {
    if !task_requests_workspace_scan(task) {
        return format!(
            "Workspace root: {}\n\n(No files were listed because this task did not ask about the project structure. Call list_files only if you need to inspect the workspace.)",
            cwd.display()
        );
    }

    let initial_listing = tools::list_files(cwd, ".", INITIAL_LIST_DEPTH)
        .unwrap_or_else(|error| format!("(could not list files: {error})"));
    format!(
        "Workspace root: {}\n\nInitial top-level file listing:\n{}\n(Use list_files on specific subdirectories when you need deeper context.)",
        cwd.display(),
        initial_listing
    )
}

/// Build an Ollama chat request with JSON output format and agent temperature.
fn chat_request(model: &str, messages: &[ChatMessage], num_ctx: u64) -> ChatRequest {
    ChatRequest {
        model: model.to_string(),
        messages: messages.to_vec(),
        stream: false,
        keep_alive: Some(serde_json::json!(crate::constants::ollama::KEEP_ALIVE)),
        format: Some(serde_json::Value::String("json".to_string())),
        options: Some(serde_json::json!({
            "temperature": AGENT_TEMPERATURE,
            "num_ctx": num_ctx,
            // As many output tokens as fit: the value is computed from the
            // compacted prompt, so prompt and reply together fit num_ctx.
            "num_predict": max_output_tokens(num_ctx, messages),
        })),
    }
}

/// Trim conversation history to `max` messages, preserving system and first user turn.
fn trim_messages(messages: &mut Vec<ChatMessage>, max: usize) {
    if messages.len() <= max {
        return;
    }

    // Always keep the system message and the initial task message,
    // then the most recent exchanges.
    let system = messages.remove(0);
    let first_user = messages.remove(0);
    let tail = messages.split_off(messages.len().saturating_sub(max - 2));

    // A trimmed tail may start with a tool result whose assistant JSON was
    // cut off; drop the orphan so the conversation still makes sense.
    let tail = tail
        .into_iter()
        .skip_while(|message| message.content.starts_with(TOOL_RESULT_PREFIX));

    messages.clear();
    messages.push(system);
    messages.push(first_user);
    messages.extend(tail);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::fakes::{
        FixedConfirmer, RecordingReporter, RecordingSink, Reply, serve_replies, test_config,
    };
    use crate::constants::ansi::ANSI_GREEN_CHECK;
    use std::path::PathBuf;

    /// The opening of a run for `task`, with no earlier turns.
    fn opening(task: &str) -> Opening<'_> {
        Opening { task, earlier: "" }
    }

    /// The output of a JSON-loop run of `task` with `max_steps` steps against
    /// a server that sends `replies` in order, and the run's result.
    async fn run_with_replies(
        task: &str,
        max_steps: usize,
        replies: &[&str],
    ) -> Result<(String, Result<()>)> {
        let root = crate::test_support::unique_temp_dir("openbatrangs-json-loop");
        let replies = replies.iter().map(|reply| Reply::ollama_stream(reply));
        let url = serve_replies(replies.collect()).await?;
        let client = OllamaClient::new(&url)?;
        let config = test_config(&root, max_steps);
        let mut reporter = RecordingReporter::default();
        let mut confirmer = FixedConfirmer::new(true);

        let result = run_json_agent(
            &config,
            &client,
            "m",
            task,
            None,
            &mut reporter,
            &mut confirmer,
        )
        .await;

        std::fs::remove_dir_all(root)?;
        Ok((reporter.output, result))
    }

    /// A chat message with `role` and `content`.
    fn chat(role: Role, content: &str) -> ChatMessage {
        message(role, content)
    }

    #[tokio::test]
    async fn a_streamed_answer_is_printed_once() -> Result<()> {
        let (output, result) = run_with_replies("say hi", 3, &[r#"{"answer":"hello"}"#]).await?;

        result?;
        assert_eq!(output.matches("hello").count(), 1);
        assert!(!output.contains("Reached max steps"));

        Ok(())
    }

    #[tokio::test]
    async fn a_tool_call_runs_before_the_answer() -> Result<()> {
        let replies = [
            r#"{"tool":{"name":"list_files","arguments":{}}}"#,
            r#"{"tool":{"name":"finish","arguments":{"summary":"all set"}}}"#,
        ];

        let (output, result) = run_with_replies("look around", 3, &replies).await?;

        result?;
        assert!(output.contains("🔧 list_files → ."));
        assert!(output.contains(&format!("{ANSI_GREEN_CHECK} all set")));

        Ok(())
    }

    #[tokio::test]
    async fn a_malformed_tool_call_is_sent_back_for_a_retry() -> Result<()> {
        let replies = [
            r#"{"tool":{"name":"rm_rf","arguments":{}}}"#,
            r#"{"answer":"ok"}"#,
        ];

        let (output, result) = run_with_replies("clean up", 3, &replies).await?;

        result?;
        assert!(output.contains("Tool call rejected: unknown tool: rm_rf"));
        assert!(output.contains("🦇 Step 2/3"));

        Ok(())
    }

    #[tokio::test]
    async fn invalid_json_is_shown_as_the_final_response() -> Result<()> {
        let (output, result) = run_with_replies("say hi", 3, &["plain words"]).await?;

        result?;
        assert!(output.contains("Could not parse model JSON"));
        assert!(output.ends_with("\nplain words\n"));

        Ok(())
    }

    #[tokio::test]
    async fn an_empty_object_explains_the_context_window() -> Result<()> {
        let (output, result) = run_with_replies("say hi", 3, &["{}"]).await?;

        result?;
        assert!(output.contains("neither an answer nor a tool call"));
        assert!(output.contains("the window was 8192 tokens"));

        Ok(())
    }

    #[tokio::test]
    async fn a_reply_with_only_a_thought_ends_the_run() -> Result<()> {
        let (output, result) = run_with_replies("say hi", 3, &[r#"{"thought":"hmm"}"#]).await?;

        result?;
        assert!(output.contains("neither an answer nor a tool call"));
        assert!(!output.contains("the window was"));

        Ok(())
    }

    #[tokio::test]
    async fn a_change_in_a_cargo_project_is_checked_before_finishing() -> Result<()> {
        let root = crate::test_support::unique_temp_dir("openbatrangs-json-check");
        std::fs::write(root.join("Cargo.toml"), "[package]")?;
        let replies = [
            r#"{"tool":{"name":"write_file","arguments":{"path":"a.txt","content":"hi"}}}"#,
            r#"{"answer":"done"}"#,
        ];
        let replies = replies.iter().map(|reply| Reply::ollama_stream(reply));
        let url = serve_replies(replies.collect()).await?;
        let client = OllamaClient::new(&url)?;
        let config = test_config(&root, 2);
        let mut reporter = RecordingReporter::default();
        let mut confirmer = FixedConfirmer::new(true);

        run_json_agent(
            &config,
            &client,
            "m",
            "change a.txt",
            None,
            &mut reporter,
            &mut confirmer,
        )
        .await?;

        assert!(reporter.output.contains("Shallow finish blocked."));
        assert!(
            reporter
                .output
                .contains("Reached max steps (2) without a final answer.")
        );

        std::fs::remove_dir_all(root)?;
        Ok(())
    }

    #[tokio::test]
    async fn a_question_with_work_words_is_answered_at_once() -> Result<()> {
        let task = "write and add tests for a directory walker, just show the code";
        let (output, result) = run_with_replies(task, 3, &[r#"{"answer":"use walkdir"}"#]).await?;

        result?;
        assert!(output.contains("use walkdir"));
        assert!(!output.contains("Shallow finish blocked."));

        Ok(())
    }

    #[tokio::test]
    async fn ask_user_ends_the_run_with_the_question() -> Result<()> {
        let replies = [r#"{"tool":{"name":"ask_user","arguments":{"question":"Which file?"}}}"#];

        let (output, result) = run_with_replies("fix it", 3, &replies).await?;

        result?;
        assert!(output.contains("❓ Which file?"));
        assert!(!output.contains("Step 2/3"));

        Ok(())
    }

    #[tokio::test]
    async fn a_run_with_earlier_turns_answers() -> Result<()> {
        let root = crate::test_support::unique_temp_dir("openbatrangs-json-earlier");
        let url = serve_replies(vec![Reply::ollama_stream(
            r#"{"answer":"You asked me to rename it."}"#,
        )])
        .await?;
        let client = OllamaClient::new(&url)?;
        let config = test_config(&root, 2);
        let continuation = RunContinuation {
            earlier: vec![message(Role::User, "rename the parser")],
            ..RunContinuation::default()
        };
        let mut reporter = RecordingReporter::default();
        let mut confirmer = FixedConfirmer::new(true);

        run_json_agent(
            &config,
            &client,
            "m",
            "what did I ask?",
            Some(continuation),
            &mut reporter,
            &mut confirmer,
        )
        .await?;

        assert!(reporter.output.contains("You asked me to rename it."));

        std::fs::remove_dir_all(root)?;
        Ok(())
    }

    #[test]
    fn initial_messages_put_the_earlier_turns_before_the_task() {
        let opening = Opening {
            task: "what did I ask?",
            earlier: "\n\nEarlier in this conversation:\nUser: rename the parser",
        };

        let messages = initial_messages(Path::new("/definitely/not/a/workspace"), &opening);

        assert!(messages[1].content.ends_with(
            "Earlier in this conversation:\nUser: rename the parser\n\nTask:\nwhat did I ask?"
        ));
    }

    #[tokio::test]
    async fn a_failed_model_call_names_the_step() -> Result<()> {
        let root = crate::test_support::unique_temp_dir("openbatrangs-json-fail");
        let url = serve_replies(vec![Reply::server_error("no model")]).await?;
        let client = OllamaClient::new(&url)?;
        let config = test_config(&root, 3);
        let mut reporter = RecordingReporter::default();
        let mut confirmer = FixedConfirmer::new(true);

        let result = run_json_agent(
            &config,
            &client,
            "m",
            "hi",
            None,
            &mut reporter,
            &mut confirmer,
        )
        .await;

        assert!(format!("{result:?}").contains("model call failed at step 1"));

        std::fs::remove_dir_all(root)?;
        Ok(())
    }

    #[tokio::test]
    async fn a_missing_workspace_is_an_error() -> Result<()> {
        let client = OllamaClient::new("http://127.0.0.1:9")?;
        let config = test_config(Path::new("/definitely/not/a/workspace"), 1);
        let mut reporter = RecordingReporter::default();
        let mut confirmer = FixedConfirmer::new(true);

        let result = run_json_agent(
            &config,
            &client,
            "m",
            "hi",
            None,
            &mut reporter,
            &mut confirmer,
        )
        .await;

        assert!(result.is_err());
        Ok(())
    }

    #[test]
    fn a_stored_conversation_is_reported_as_dropped() {
        let mut sink = RecordingSink::default();
        let continuation = RunContinuation {
            messages: Some(Vec::new()),
            sink: Some(&mut sink),
            ..RunContinuation::default()
        };
        let fresh = RunContinuation::default();
        let mut reporter = RecordingReporter::default();

        warn_when_resume_is_unsupported(Some(&fresh), &mut reporter);
        warn_when_resume_is_unsupported(None, &mut reporter);
        assert!(reporter.output.is_empty());

        warn_when_resume_is_unsupported(Some(&continuation), &mut reporter);
        assert!(reporter.output.contains("cannot be resumed"));
    }

    #[test]
    fn trims_messages_but_keeps_system_and_first_user() {
        let mut messages = vec![
            chat(Role::System, "system"),
            chat(Role::User, "task"),
            chat(Role::Assistant, "a1"),
            chat(Role::User, "u1"),
            chat(Role::Assistant, "a2"),
        ];

        trim_messages(&mut messages, 3);

        assert_eq!(messages.len(), 3);
        assert_eq!(messages[0].role, Role::System);
        assert_eq!(messages[1].content, "task");
        assert_eq!(messages[2].content, "a2");
    }

    #[test]
    fn trim_messages_does_nothing_when_at_or_under_limit() {
        let mut messages = vec![
            chat(Role::System, "system"),
            chat(Role::User, "task"),
            chat(Role::Assistant, "answer"),
        ];

        trim_messages(&mut messages, 10);
        trim_messages(&mut messages, 3);

        assert_eq!(messages.len(), 3);
        assert_eq!(messages[2].content, "answer");
    }

    #[test]
    fn trim_messages_keeps_most_recent_exchanges() {
        let mut messages = vec![
            chat(Role::System, "system"),
            chat(Role::User, "task"),
            chat(Role::Assistant, "a1"),
            chat(Role::User, "u1"),
            chat(Role::Assistant, "a2"),
            chat(Role::User, "u2"),
            chat(Role::Assistant, "a3"),
        ];

        // max=4: system + first_user + 2 most recent
        trim_messages(&mut messages, 4);

        let contents: Vec<&str> = messages
            .iter()
            .map(|message| message.content.as_str())
            .collect();
        assert_eq!(contents, ["system", "task", "u2", "a3"]);
    }

    #[test]
    fn trim_drops_orphan_tool_results_from_the_tail() {
        let mut messages = vec![
            chat(Role::System, "system"),
            chat(Role::User, "task"),
            chat(Role::Assistant, r#"{"tool":{"name":"read_file"}}"#),
            chat(Role::User, "Tool result:\nold output"),
            chat(Role::User, "Tool result:\nolder output"),
            chat(Role::Assistant, r#"{"answer":"done"}"#),
        ];

        trim_messages(&mut messages, 4);

        assert_eq!(messages.len(), 3);
        assert_eq!(messages[1].content, "task");
        assert_eq!(messages[2].content, r#"{"answer":"done"}"#);
    }

    #[test]
    fn initial_messages_list_shallowly_when_task_asks_about_codebase() -> Result<()> {
        let root = crate::test_support::unique_temp_dir("openbatrangs-shallow-test");
        std::fs::create_dir_all(root.join("sub/nested"))?;
        std::fs::write(root.join("top.txt"), "top")?;
        std::fs::write(root.join("sub/nested/deep.txt"), "deep")?;

        let messages = initial_messages(&root, &opening("analyze this codebase"));

        let user_content = &messages[1].content;
        assert!(user_content.contains("top.txt"));
        assert!(!user_content.contains("deep.txt"));
        assert!(user_content.contains("Use list_files on specific subdirectories"));

        std::fs::remove_dir_all(root)?;
        Ok(())
    }

    #[test]
    fn initial_messages_report_a_listing_failure() {
        let missing = PathBuf::from("/definitely/not/a/workspace");

        let messages = initial_messages(&missing, &opening("explore this repo"));

        assert!(messages[1].content.contains("(could not list files:"));
    }

    #[test]
    fn initial_messages_do_not_scan_for_generic_task() -> Result<()> {
        let root = crate::test_support::unique_temp_dir("openbatrangs-no-scan-test");
        std::fs::write(root.join("top.txt"), "top")?;

        let messages = initial_messages(&root, &opening("write a quicksort function in Rust"));

        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].role, Role::System);
        assert_eq!(messages[1].role, Role::User);
        let user_content = &messages[1].content;
        assert!(user_content.ends_with("Task:\nwrite a quicksort function in Rust"));
        assert!(!user_content.contains("top.txt"));
        assert!(user_content.contains("No files were listed"));

        std::fs::remove_dir_all(root)?;
        Ok(())
    }

    #[test]
    fn task_workspace_scan_hints_are_case_insensitive() {
        assert!(task_requests_workspace_scan(
            "Look at the CURRENT DIRECTORY layout"
        ));
        assert!(task_requests_workspace_scan("explore this repo"));
        assert!(task_requests_workspace_scan(
            "what files are in this project?"
        ));
        assert!(!task_requests_workspace_scan("write a unit test"));
    }

    #[test]
    fn chat_request_sets_json_format_and_context() {
        let messages = vec![
            chat(Role::System, "system prompt"),
            chat(Role::User, "task"),
        ];

        let request = chat_request("test-model", &messages, 4096);

        assert_eq!(request.model, "test-model");
        assert_eq!(request.messages.len(), 2);
        assert!(!request.stream);
        assert!(request.format.is_some());
        assert!(
            request
                .options
                .is_some_and(|options| options["num_ctx"] == 4096)
        );
    }
}
