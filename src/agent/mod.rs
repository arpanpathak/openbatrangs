//! Agentic coding loop: model conversation, tool-call parsing, and execution.
//!
//! Responsibilities are split into focused submodules:
//! - `reporter`: output abstraction for stdout and TUI.
//! - `tool`: typed tool calls, argument parsing, and JSON response parsing.
//! - `stream`: incremental JSON field streaming.
//! - `execute`: tool execution, confirmation, and path formatting.
//!
//! This module owns the orchestration: the run loop, step handling, and
//! conversation-history management.

mod execute;
mod reporter;
mod stream;
mod tool;

use crate::constants::agent::{
    AGENT_TEMPERATURE, MAX_CONTEXT_TOKENS, MAX_HISTORY_MESSAGES, MIN_CONTEXT_TOKENS, SYSTEM_PROMPT,
};
use crate::constants::ansi::{
    ANSI_GREEN_CHECK, COLOR_BOLD, COLOR_CYAN, COLOR_DIM, COLOR_MAGENTA, COLOR_RESET,
};
use crate::constants::tools::DEFAULT_LIST_DEPTH;
use crate::ollama::{ChatMessage, ChatRequest, OllamaClient};
use crate::tools;
use anyhow::{bail, Context, Result};
use execute::{execute_tool_or_report_error, print_changed_files};
use std::path::{Path, PathBuf};
use stream::stream_model_response;
use tool::{parse_agent_response, Tool, ToolCall};

pub use reporter::{Reporter, StdoutReporter};

#[derive(Debug, Clone)]
pub struct AgentConfig {
    pub cwd: PathBuf,
    pub max_steps: usize,
    pub is_read_only: bool,
    pub should_confirm: bool,
    pub show_thinking: bool,
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
            reporter.line(format!("\n{ANSI_GREEN_CHECK} {answer}"));
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
        reporter.line(format!("\n{ANSI_GREEN_CHECK} {summary}"));
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

fn ensure_workspace_exists(cwd: &Path) -> Result<()> {
    if !cwd.is_dir() {
        bail!("working directory does not exist: {}", cwd.display());
    }
    Ok(())
}

fn initial_messages(cwd: &Path, task: &str) -> Vec<ChatMessage> {
    let initial_listing = tools::list_files(cwd, ".", DEFAULT_LIST_DEPTH)
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
        format: Some(serde_json::Value::String("json".to_string())),
        options: Some(serde_json::json!({
            "temperature": AGENT_TEMPERATURE,
            "num_ctx": num_ctx,
        })),
    }
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

#[cfg(test)]
mod tests {
    use super::*;

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
}
