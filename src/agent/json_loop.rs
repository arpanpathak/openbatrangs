//! # JSON tool loop for Ollama servers
//!
//! Ollama's `/api/chat` has no tool-call channel the loop relies on, so the
//! model answers every step with one JSON object: a tool call, or the final
//! answer. Each step streams that object, prints its `thought` and `answer`
//! fields as they arrive, and then acts on it:
//!
//! - an answer, or a call to `finish`, ends the run unless the file-work gate
//!   rejects it as premature;
//! - any other tool call runs, and its result goes back as a user message;
//! - a reply that is neither, or is not JSON, ends the run with a warning.

use super::environment::environment_facts;
use super::execute::{execute_tool_or_report_error, print_changed_files};
use super::quality::{FileWorkGate, needs_verification, requires_file_work};
use super::research::task_with_documentation;
use super::stream::stream_model_response;
use super::tool::{AgentResponse, Tool, ToolCall, parse_agent_response};
use super::{
    AgentConfig, Confirmer, Reporter, RunContinuation, compact_context, effective_context,
    ensure_workspace_exists, max_output_tokens,
};
use crate::constants::agent::{
    AGENT_TEMPERATURE, INITIAL_LIST_DEPTH, MAX_HISTORY_MESSAGES, SYSTEM_PROMPT, TOOL_RESULT_PREFIX,
};
use crate::constants::ansi::{
    ANSI_GREEN_CHECK, COLOR_BOLD, COLOR_CYAN, COLOR_DIM, COLOR_MAGENTA, COLOR_RESET,
};
use crate::ollama::{ChatMessage, ChatRequest, OllamaClient, Role};
use crate::tools;
use anyhow::{Context, Result};
use std::path::Path;

/// The outcome of one agent iteration.
#[derive(Debug, PartialEq, Eq)]
enum AgentStepOutcome {
    /// The agent answered or hit a terminal parse error.
    Finished,
    /// The agent called a tool and should continue to the next step.
    Continue,
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
    /// Whether the original task asks for file output rather than just a chat answer.
    requires_file_work: bool,
    /// Whether the workspace has a build manifest, so a successful build or
    /// test is expected before the task counts as done.
    needs_verification: bool,
    /// Conversation history sent to the model.
    messages: Vec<ChatMessage>,
    /// Files written successfully during the task.
    changed_files: Vec<String>,
    /// Gate that rejects shallow finish calls.
    file_work_gate: FileWorkGate,
    /// Output sink for status and results.
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
    let task_with_docs = task_with_documentation(task, config, reporter).await;
    let requires_file_work = requires_file_work(task);

    let run = JsonRun {
        config,
        client,
        model,
        num_ctx,
        requires_file_work,
        needs_verification: needs_verification(requires_file_work, &config.cwd),
        messages: initial_messages(&config.cwd, &task_with_docs),
        changed_files: Vec::new(),
        file_work_gate: FileWorkGate::default(),
        reporter,
        confirmer,
    };
    run.run().await
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

    reporter.line(
        "\n⚠️  The previous conversation cannot be resumed: this server uses the JSON tool loop."
            .to_string(),
    );
}

impl<R: Reporter, C: Confirmer> JsonRun<'_, R, C> {
    /// Run steps until one finishes, and warn when the budget ran out first.
    async fn run(mut self) -> Result<()> {
        let outcome = self.run_steps().await?;

        self.report_unfinished(&outcome);
        Ok(())
    }

    /// Run steps until one finishes, fails, or the budget is spent; return
    /// the last outcome.
    async fn run_steps(&mut self) -> Result<AgentStepOutcome> {
        let mut outcome = Ok(AgentStepOutcome::Continue);
        let mut steps = 1..=self.config.max_steps;

        while let (Ok(AgentStepOutcome::Continue), Some(step)) = (&outcome, steps.next()) {
            outcome = self.step(step).await;
        }

        outcome
    }

    /// Warn that the step budget ran out when the last step did not finish.
    fn report_unfinished(&mut self, outcome: &AgentStepOutcome) {
        if *outcome == AgentStepOutcome::Finished {
            return;
        }

        self.reporter.line(format!(
            "\n⚠️  Reached max steps ({}) without a final answer.",
            self.config.max_steps
        ));
    }

    /// Run one agent iteration: stream a response, parse it, and act on it.
    async fn step(&mut self, step: usize) -> Result<AgentStepOutcome> {
        let max_steps = self.config.max_steps;
        self.reporter.line(step_header(step, max_steps, self.model));

        // Auto rolling context: keep the system prompt, original task, and the
        // newest tool exchange verbatim; older tool calls/output are rolled into a
        // short note so the prompt always fits the safe per-device `num_ctx`.
        compact_context(&mut self.messages, self.num_ctx);

        let request = chat_request(self.model, &self.messages, self.num_ctx);
        let show_thinking = self.config.show_thinking;
        let (content, answer_was_streamed) =
            stream_model_response(self.client, request, self.reporter, show_thinking)
                .await
                .with_context(|| format!("model call failed at step {step}"))?;

        Ok(self.act(&content, answer_was_streamed).await)
    }

    /// Parse the raw reply and act on it; a reply that is not valid JSON is
    /// shown as the final response.
    async fn act(&mut self, content: &str, answer_was_streamed: bool) -> AgentStepOutcome {
        match parse_agent_response(content) {
            Ok(response) => {
                self.act_on_response(content, response, answer_was_streamed)
                    .await
            }
            Err(error) => self.report_parse_error(&error, content),
        }
    }

    /// The model may answer, call exactly one tool, or do neither. Matching
    /// on the pair keeps all three outcomes explicit.
    async fn act_on_response(
        &mut self,
        content: &str,
        response: AgentResponse,
        answer_was_streamed: bool,
    ) -> AgentStepOutcome {
        match (response.answer, response.tool) {
            (Some(answer), _) => self.finish(&answer, answer_was_streamed),
            (None, Some(tool_call)) => self.handle_tool_call(content, tool_call).await,
            (None, None) => self.report_empty_reply(content),
        }
    }

    /// End the run with `summary`, unless the file-work gate rejects the
    /// finish. A summary that was already streamed live is not printed again.
    fn finish(&mut self, summary: &str, was_streamed: bool) -> AgentStepOutcome {
        if let Some(feedback) = self.rejected_finish() {
            return self.block_finish(feedback);
        }

        self.print_summary(summary, was_streamed);
        print_changed_files(&self.config.cwd, &self.changed_files, self.reporter);
        AgentStepOutcome::Finished
    }

    /// The gate's feedback when a finish now would be premature.
    fn rejected_finish(&mut self) -> Option<&'static str> {
        self.file_work_gate.reject_finish(
            self.requires_file_work,
            self.config.is_read_only,
            self.changed_files.len(),
            self.needs_verification,
        )
    }

    /// Send the gate's feedback to the model and keep going.
    fn block_finish(&mut self, feedback: &str) -> AgentStepOutcome {
        self.messages.push(message(Role::User, feedback));
        self.reporter.line(format!(
            "\n{COLOR_DIM}⚠️  Shallow finish blocked.{COLOR_RESET}"
        ));

        AgentStepOutcome::Continue
    }

    /// Print the final summary unless it was already streamed.
    fn print_summary(&mut self, summary: &str, was_streamed: bool) {
        if was_streamed {
            return;
        }

        self.reporter
            .line(format!("\n{ANSI_GREEN_CHECK} {summary}"));
    }

    /// Execute a parsed tool call; `finish` ends the run, a malformed call is
    /// sent back to the model to retry.
    async fn handle_tool_call(&mut self, content: &str, tool_call: ToolCall) -> AgentStepOutcome {
        match Tool::from_call(tool_call) {
            Ok(Tool::Finish { summary }) => self.finish(&summary, false),
            Ok(tool) => self.run_tool(content, tool).await,
            Err(error) => self.reject_tool_call(content, &error),
        }
    }

    /// A malformed call does not end the run: report what the model sent,
    /// send the error back as a user message, and continue, so the next
    /// reply can correct the call.
    fn reject_tool_call(&mut self, content: &str, error: &anyhow::Error) -> AgentStepOutcome {
        self.reporter.line(format!(
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

        AgentStepOutcome::Continue
    }

    /// Push the assistant JSON, execute the tool, push its result, and
    /// continue to the next step.
    async fn run_tool(&mut self, content: &str, tool: Tool) -> AgentStepOutcome {
        self.messages.push(message(Role::Assistant, content));
        self.reporter.line(format!(
            "\n{COLOR_MAGENTA}🔧 {}{COLOR_RESET}",
            tool.describe()
        ));

        let cwd = &self.config.cwd;
        let changed_files = &mut self.changed_files;
        let result_text =
            execute_tool_or_report_error(self.config, cwd, &tool, changed_files, self.confirmer)
                .await;
        self.file_work_gate.record_tool_result(&tool, &result_text);
        self.reporter
            .line(format!("{COLOR_DIM}{result_text}{COLOR_RESET}"));

        let tool_result = format!("{TOOL_RESULT_PREFIX}\n{result_text}");
        self.messages.push(message(Role::User, &tool_result));
        trim_messages(&mut self.messages, MAX_HISTORY_MESSAGES);

        AgentStepOutcome::Continue
    }

    /// Neither field is a model failure; surface it and stop cleanly.
    fn report_empty_reply(&mut self, content: &str) -> AgentStepOutcome {
        self.reporter.line(format!(
            "\n⚠️  Model returned neither an answer nor a tool call.\n   model sent: {}",
            content.trim()
        ));

        if matches!(content.trim(), "" | "{}") {
            self.reporter.line(format!(
                "   An empty reply usually means the context window ran out while the \
                 model was thinking; the window was {} tokens.",
                self.num_ctx
            ));
        }

        AgentStepOutcome::Finished
    }

    /// Report a JSON parse failure and show the raw model output as fallback.
    fn report_parse_error(&mut self, error: &anyhow::Error, content: &str) -> AgentStepOutcome {
        self.reporter
            .line(format!("\n⚠️  Could not parse model JSON: {error}"));
        self.reporter
            .line("Showing raw model output as the final response.".to_string());
        self.reporter.line(format!("\n{content}"));

        AgentStepOutcome::Finished
    }
}

/// A chat message with `role` and `content`.
fn message(role: Role, content: &str) -> ChatMessage {
    ChatMessage {
        role,
        content: content.to_string(),
    }
}

/// The step header with the step number, max steps, and model name.
fn step_header(step: usize, max_steps: usize, model: &str) -> String {
    format!(
        "\n{COLOR_BOLD}{COLOR_CYAN}🦇 Step {step}/{max_steps}{COLOR_RESET} — model: {COLOR_BOLD}{model}{COLOR_RESET}"
    )
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
fn initial_messages(cwd: &Path, task: &str) -> Vec<ChatMessage> {
    let user_content = format!(
        "{}\n\nEnvironment:\n{}\n\nTask:\n{task}",
        workspace_note(cwd, task),
        environment_facts(cwd)
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
    use std::path::PathBuf;

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
    async fn a_shallow_finish_is_blocked_until_steps_run_out() -> Result<()> {
        let (output, result) =
            run_with_replies("write a file", 1, &[r#"{"answer":"done"}"#]).await?;

        result?;
        assert!(output.contains("Shallow finish blocked."));
        assert!(output.contains("Reached max steps (1) without a final answer."));

        Ok(())
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
        };
        let fresh = RunContinuation {
            messages: None,
            sink: None,
        };
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

        let messages = initial_messages(&root, "analyze this codebase");

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

        let messages = initial_messages(&missing, "explore this repo");

        assert!(messages[1].content.contains("(could not list files:"));
    }

    #[test]
    fn initial_messages_do_not_scan_for_generic_task() -> Result<()> {
        let root = crate::test_support::unique_temp_dir("openbatrangs-no-scan-test");
        std::fs::write(root.join("top.txt"), "top")?;

        let messages = initial_messages(&root, "write a quicksort function in Rust");

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
