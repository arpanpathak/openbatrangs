//! # Agentic coding loop
//!
//! The agent alternates between two phases:
//!
//! 1. Ask the model for the next action as a strict JSON object.
//! 2. If the JSON contains a tool call, execute it and feed the result back;
//!    otherwise treat the `answer` field as the final response.
//!
//! The loop is bounded by `max_steps` so a misbehaving model cannot run
//! forever. All tool calls are parsed through an exhaustive [`Tool`] enum and
//! executed by the `execute` submodule.
//!
//! Responsibilities are split into focused submodules:
//! - `reporter`: output abstraction for stdout and TUI.
//! - `tool`: typed tool calls, argument parsing, and JSON response parsing.
//! - `stream`: incremental JSON field streaming.
//! - `execute`: tool execution, confirmation, and path formatting.
//!
//! ## References
//!
//! - ReAct: reasoning + acting: <https://arxiv.org/abs/2210.03629>
//! - Ollama `/api/chat`: <https://github.com/ollama/ollama/blob/main/docs/api.md#generate-a-chat-completion>
//! - JSON Schema for strict tool-call responses: <https://json-schema.org/>

mod confirm;
mod execute;
mod reporter;
mod stream;
mod tool;

use crate::constants::agent::{
    AGENT_TEMPERATURE, INITIAL_LIST_DEPTH, MAX_HISTORY_MESSAGES, MIN_CONTEXT_TOKENS, SYSTEM_PROMPT,
};
use crate::constants::ansi::{
    ANSI_GREEN_CHECK, COLOR_BOLD, COLOR_CYAN, COLOR_DIM, COLOR_MAGENTA, COLOR_RESET,
};
use crate::ollama::{ChatMessage, ChatRequest, OllamaClient};
use crate::tools;
use anyhow::{bail, Context, Result};
use execute::{execute_tool_or_report_error, print_changed_files};
use std::path::{Path, PathBuf};
use stream::stream_model_response;
use tool::{parse_agent_response, Tool, ToolCall};

pub use confirm::{Confirmer, StdioConfirmer};
pub use reporter::{Reporter, StdoutReporter};

/// Immutable configuration for the agentic coding loop.
#[derive(Debug, Clone)]
pub struct AgentConfig {
    /// Working directory where the agent reads and writes files.
    pub cwd: PathBuf,
    /// Maximum number of agent iterations before stopping.
    pub max_steps: usize,
    /// When `true`, mutating tools (write, run command) are disabled.
    pub is_read_only: bool,
    /// When `true`, the user must approve each write/command before execution.
    pub should_confirm: bool,
    /// When `true`, the model's reasoning/thinking text is shown to the user.
    pub show_thinking: bool,
    /// Maximum context window (tokens) sent to the model.
    pub max_ctx: u64,
}

/// The outcome of one agent iteration.
enum AgentStepOutcome {
    /// The agent answered or hit a terminal parse error.
    Finished,
    /// The agent called a tool and should continue to the next step.
    Continue,
}

/// Immutable inputs shared by every agent step.
struct AgentRunContext<'a> {
    /// Agent configuration (cwd, limits, safety flags).
    config: &'a AgentConfig,
    /// Ollama client for model API calls.
    client: &'a OllamaClient,
    /// Name of the model being used.
    model: &'a str,
    /// Context window size (tokens) sent to the model.
    num_ctx: u64,
}

/// Run the bounded agentic coding loop: plan, execute tools, iterate.
///
/// # Parameters
///
/// - `config`: agent runtime configuration.
/// - `client`: Ollama client for model calls.
/// - `model`: model name to use for completions.
/// - `model_context`: model's native context window size.
/// - `task`: user's task description.
/// - `reporter`: output sink for status and results.
/// - `confirmer`: human-in-the-loop confirmer for writes/commands.
///
/// # Returns
///
/// `Ok(())` when the agent finishes (with or without an answer), or an error
/// if the model call fails.
pub async fn run_agent<R: Reporter, C: Confirmer>(
    config: &AgentConfig,
    client: &OllamaClient,
    model: &str,
    model_context: u64,
    task: &str,
    reporter: &mut R,
    confirmer: &mut C,
) -> Result<()> {
    ensure_workspace_exists(&config.cwd)?;

    let max_ctx = config.max_ctx.max(MIN_CONTEXT_TOKENS);
    let num_ctx = model_context.clamp(MIN_CONTEXT_TOKENS, max_ctx);
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
            confirmer,
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
async fn run_agent_step<R: Reporter, C: Confirmer>(
    context: &AgentRunContext<'_>,
    messages: &mut Vec<ChatMessage>,
    changed_files: &mut Vec<String>,
    reporter: &mut R,
    confirmer: &mut C,
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

    // The model may answer, call exactly one tool, or do neither. Matching on
    // the tuple keeps all three outcomes explicit and exhaustive.
    match (response.answer, response.tool) {
        // A final answer ends the loop. If it was already streamed live, do
        // not print it a second time.
        (Some(answer), _) => {
            if !answer_was_streamed {
                reporter.line(format!("\n{ANSI_GREEN_CHECK} {answer}"));
            }
            print_changed_files(&context.config.cwd, changed_files, reporter);
            Ok(AgentStepOutcome::Finished)
        }
        // A tool call continues the loop: execute, append the result, repeat.
        (None, Some(tool_call)) => {
            handle_tool_call(
                context.config,
                messages,
                changed_files,
                reporter,
                confirmer,
                &content,
                tool_call,
            )
            .await
        }
        // Neither field is a model failure; surface it and stop cleanly.
        (None, None) => {
            reporter.line("\n⚠️  Model returned neither an answer nor a tool call.".to_string());
            Ok(AgentStepOutcome::Finished)
        }
    }
}

/// Execute a parsed tool call and append the assistant/tool messages.
async fn handle_tool_call<R: Reporter, C: Confirmer>(
    config: &AgentConfig,
    messages: &mut Vec<ChatMessage>,
    changed_files: &mut Vec<String>,
    reporter: &mut R,
    confirmer: &mut C,
    content: &str,
    tool_call: ToolCall,
) -> Result<AgentStepOutcome> {
    let tool = Tool::from_call(tool_call)?;
    match tool {
        // `finish` is the model's way of saying "done": report the summary,
        // list changed files, and stop the loop.
        Tool::Finish { summary } => {
            reporter.line(format!("\n{ANSI_GREEN_CHECK} {summary}"));
            print_changed_files(&config.cwd, changed_files, reporter);
            Ok(AgentStepOutcome::Finished)
        }
        // Any real tool (list/read/grep/write/run) follows the same contract:
        // push the assistant JSON, execute, push the result, and continue so
        // the model can decide what to do next.
        other => {
            messages.push(ChatMessage {
                role: crate::ollama::Role::Assistant,
                content: content.to_string(),
            });

            reporter.line(format!(
                "\n{COLOR_MAGENTA}🔧 {}{COLOR_RESET}",
                other.describe()
            ));
            let result_text =
                execute_tool_or_report_error(config, &config.cwd, &other, changed_files, confirmer)
                    .await;
            reporter.line(format!("{COLOR_DIM}{result_text}{COLOR_RESET}"));

            messages.push(ChatMessage {
                role: crate::ollama::Role::User,
                content: format!("Tool result:\n{result_text}"),
            });
            trim_messages(messages, MAX_HISTORY_MESSAGES);
            Ok(AgentStepOutcome::Continue)
        }
    }
}

/// Print a step header with the step number, max steps, and model name.
fn report_step_start<R: Reporter>(reporter: &mut R, step: usize, max_steps: usize, model: &str) {
    reporter.line(format!(
        "\n{COLOR_BOLD}{COLOR_CYAN}🦇 Step {step}/{max_steps}{COLOR_RESET} — model: {COLOR_BOLD}{model}{COLOR_RESET}"
    ));
}

/// Report a JSON parse failure and show the raw model output as fallback.
fn report_parse_error<R: Reporter>(reporter: &mut R, error: &anyhow::Error, content: &str) {
    reporter.line(format!("\n⚠️  Could not parse model JSON: {error}"));
    reporter.line("Showing raw model output as the final response.".to_string());
    reporter.line(format!("\n{content}"));
}

/// Verify the working directory exists before starting the agent.
fn ensure_workspace_exists(cwd: &Path) -> Result<()> {
    if !cwd.is_dir() {
        bail!("working directory does not exist: {}", cwd.display());
    }
    Ok(())
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
    let workspace_note = if task_requests_workspace_scan(task) {
        // Shallow top-level listing only. The model should call `list_files` on
        // specific directories instead of forcing a recursive scan.
        let initial_listing = tools::list_files(cwd, ".", INITIAL_LIST_DEPTH)
            .unwrap_or_else(|error| format!("(could not list files: {error})"));
        format!(
            "Workspace root: {}\n\nInitial top-level file listing:\n{}\n(Use list_files on specific subdirectories when you need deeper context.)",
            cwd.display(),
            initial_listing
        )
    } else {
        format!(
            "Workspace root: {}\n\n(No files were listed because this task did not ask about the project structure. Call list_files only if you need to inspect the workspace.)",
            cwd.display()
        )
    };
    let user_content = format!("{workspace_note}\n\nTask:\n{task}");
    vec![
        ChatMessage {
            role: crate::ollama::Role::System,
            content: SYSTEM_PROMPT.to_string(),
        },
        ChatMessage {
            role: crate::ollama::Role::User,
            content: user_content,
        },
    ]
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
    messages.clear();
    messages.push(system);
    messages.push(first_user);
    messages.extend(tail);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ollama::Role;

    #[test]
    fn trims_messages_but_keeps_system_and_first_user() {
        let mut messages = vec![
            ChatMessage {
                role: crate::ollama::Role::System,
                content: "system".to_string(),
            },
            ChatMessage {
                role: crate::ollama::Role::User,
                content: "task".to_string(),
            },
            ChatMessage {
                role: crate::ollama::Role::Assistant,
                content: "a1".to_string(),
            },
            ChatMessage {
                role: crate::ollama::Role::User,
                content: "u1".to_string(),
            },
            ChatMessage {
                role: crate::ollama::Role::Assistant,
                content: "a2".to_string(),
            },
        ];
        trim_messages(&mut messages, 3);
        assert_eq!(messages.len(), 3);
        assert_eq!(messages[0].role, Role::System);
        assert_eq!(messages[1].content, "task");
        assert_eq!(messages[2].content, "a2");
    }

    #[test]
    fn initial_messages_list_shallowly_when_task_asks_about_codebase() {
        let root = crate::test_support::unique_temp_dir("openbatrangs-shallow-test");
        std::fs::create_dir_all(root.join("sub/nested")).unwrap();
        std::fs::write(root.join("top.txt"), "top").unwrap();
        std::fs::write(root.join("sub/nested/deep.txt"), "deep").unwrap();

        let messages = initial_messages(&root, "analyze this codebase");
        let user_content = &messages[1].content;
        assert!(user_content.contains("top.txt"));
        assert!(!user_content.contains("deep.txt"));
        assert!(user_content.contains("Use list_files on specific subdirectories"));

        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn initial_messages_do_not_scan_for_generic_task() {
        let root = crate::test_support::unique_temp_dir("openbatrangs-no-scan-test");
        std::fs::write(root.join("top.txt"), "top").unwrap();

        let messages = initial_messages(&root, "write a quicksort function in Rust");
        let user_content = &messages[1].content;
        assert!(!user_content.contains("top.txt"));
        assert!(user_content.contains("No files were listed"));
        assert!(user_content.contains("Call list_files only if you need to inspect"));

        std::fs::remove_dir_all(root).unwrap();
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
    fn trim_messages_does_nothing_when_under_limit() {
        let mut messages = vec![
            ChatMessage {
                role: Role::System,
                content: "system".to_string(),
            },
            ChatMessage {
                role: Role::User,
                content: "task".to_string(),
            },
            ChatMessage {
                role: Role::Assistant,
                content: "answer".to_string(),
            },
        ];
        let original_len = messages.len();
        trim_messages(&mut messages, 10);
        assert_eq!(messages.len(), original_len);
        assert_eq!(messages[0].content, "system");
        assert_eq!(messages[1].content, "task");
        assert_eq!(messages[2].content, "answer");
    }

    #[test]
    fn trim_messages_keeps_exact_count_when_at_limit() {
        let mut messages = vec![
            ChatMessage {
                role: Role::System,
                content: "system".to_string(),
            },
            ChatMessage {
                role: Role::User,
                content: "task".to_string(),
            },
            ChatMessage {
                role: Role::Assistant,
                content: "a1".to_string(),
            },
        ];
        trim_messages(&mut messages, 3);
        assert_eq!(messages.len(), 3);
    }

    #[test]
    fn trim_messages_keeps_most_recent_exchanges() {
        let mut messages = vec![
            ChatMessage {
                role: Role::System,
                content: "system".to_string(),
            },
            ChatMessage {
                role: Role::User,
                content: "task".to_string(),
            },
            ChatMessage {
                role: Role::Assistant,
                content: "a1".to_string(),
            },
            ChatMessage {
                role: Role::User,
                content: "u1".to_string(),
            },
            ChatMessage {
                role: Role::Assistant,
                content: "a2".to_string(),
            },
            ChatMessage {
                role: Role::User,
                content: "u2".to_string(),
            },
            ChatMessage {
                role: Role::Assistant,
                content: "a3".to_string(),
            },
        ];
        // max=4: system + first_user + 2 most recent
        trim_messages(&mut messages, 4);
        assert_eq!(messages.len(), 4);
        assert_eq!(messages[0].role, Role::System);
        assert_eq!(messages[0].content, "system");
        assert_eq!(messages[1].role, Role::User);
        assert_eq!(messages[1].content, "task");
        assert_eq!(messages[2].content, "u2");
        assert_eq!(messages[3].content, "a3");
    }

    #[test]
    fn ensure_workspace_exists_succeeds_for_valid_directory() {
        let root = crate::test_support::unique_temp_dir("openbatrangs-workspace-test");
        assert!(ensure_workspace_exists(&root).is_ok());
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn ensure_workspace_exists_fails_for_missing_directory() {
        let missing = std::path::PathBuf::from("/tmp/openbatrangs-definitely-does-not-exist-12345");
        let result = ensure_workspace_exists(&missing);
        assert!(result.is_err());
        assert!(format!("{}", result.unwrap_err()).contains("does not exist"));
    }

    #[test]
    fn chat_request_sets_json_format_and_temperature() {
        let messages = vec![
            ChatMessage {
                role: Role::System,
                content: "system prompt".to_string(),
            },
            ChatMessage {
                role: Role::User,
                content: "task".to_string(),
            },
        ];
        let request = chat_request("test-model", &messages, 4096);
        assert_eq!(request.model, "test-model");
        assert_eq!(request.messages.len(), 2);
        assert!(!request.stream);
        assert!(request.format.is_some());
        let options = request.options.unwrap();
        assert_eq!(options["num_ctx"], 4096);
    }

    #[test]
    fn initial_messages_always_contain_system_prompt_and_task() {
        let root = crate::test_support::unique_temp_dir("openbatrangs-init-msg-test");
        std::fs::write(root.join("top.txt"), "hello").unwrap();

        let messages = initial_messages(&root, "fix the bug");
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].role, Role::System);
        assert!(!messages[0].content.is_empty());
        assert_eq!(messages[1].role, Role::User);
        assert!(messages[1].content.contains("fix the bug"));

        std::fs::remove_dir_all(root).unwrap();
    }
}
