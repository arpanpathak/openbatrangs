//! # Agentic coding loop
//!
//! The agent alternates between two phases:
//!
//! 1. Ask the model for the next action.
//! 2. If the reply contains a tool call, execute it and feed the result back;
//!    otherwise treat the reply as the final answer.
//!
//! The loop is bounded by `max_steps` so a model that never answers cannot
//! keep it running forever. All tool calls are parsed through an exhaustive `Tool` enum and
//! executed by the `execute` submodule.
//!
//! An Ollama server gets the JSON loop in `json_loop`, where the model writes
//! each action as a strict JSON object. An OpenAI-compatible server gets the
//! native tool-calling loop in `native`.
//!
//! Responsibilities are split into focused submodules:
//! - `reporter`: structured agent events and the stdout reporter.
//! - `summary`: short summaries of tool results.
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
mod context;
mod environment;
mod execute;
#[cfg(test)]
mod fakes;
mod json_loop;
mod native;
mod quality;
mod reporter;
mod research;
mod stream;
mod summary;
mod tool;

use crate::constants::agent::MIN_CONTEXT_TOKENS;
use crate::ollama::{ChatMessage, OllamaClient};
use anyhow::{Result, ensure};
use serde_json::Value;
use std::path::{Path, PathBuf};

pub use confirm::{Confirmer, StdioConfirmer};
pub(crate) use context::{compact_context, earlier_turns_section, max_output_tokens};
pub use reporter::{AgentEvent, LoopKind, Reporter, StdoutReporter, TextKind};
pub(crate) use tool::describe_call;

/// Receives snapshots of the conversation as it grows.
///
/// The TUI keeps the latest snapshot so that when the user interrupts a run
/// and types a correction, the next run can resume the conversation instead
/// of restarting from the task text alone.
///
/// Implementations must be `Send`: the sink is held inside the spawned
/// background worker.
pub trait HistorySink: Send {
    /// Store the current conversation, in OpenAI `/chat/completions` shape.
    fn save(&mut self, messages: Vec<Value>);
}

/// What a run knows from earlier in the session: the interrupted run it
/// continues, the earlier tasks and answers, and the sink that keeps the
/// TUI's resume copy up to date for the next interrupt.
///
/// Bundling them keeps [`run_agent`]'s signature small.
#[derive(Default)]
pub struct RunContinuation<'a> {
    /// Prior conversation in OpenAI `/chat/completions` shape; `None` for a
    /// fresh start.
    pub messages: Option<Vec<Value>>,
    /// Earlier tasks and answers of this session, oldest first. A fresh run
    /// puts them in its first message, so the model knows what was said.
    pub earlier: Vec<ChatMessage>,
    /// Sink receiving conversation snapshots; `None` in one-shot mode.
    pub sink: Option<&'a mut dyn HistorySink>,
}

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
    /// The model's own context window, in tokens, from server metadata.
    pub model_context: u64,
    /// Upper limit on the context window (tokens); `None` uses the model's
    /// own context length in full.
    pub max_ctx: Option<u64>,
    /// Base URL of the SearXNG instance the web tools search.
    pub searxng_url: String,
    /// Bearer key sent with each search, for a search behind the Thor's agent.
    pub search_key: Option<String>,
}

/// Run the bounded agentic coding loop: plan, execute tools, iterate.
///
/// # Parameters
///
/// - `config`: agent runtime configuration.
/// - `client`: Ollama client for model calls.
/// - `model`: model name to use for completions.
/// - `task`: user's task description.
/// - `continuation`: the interrupted run to continue and the snapshot sink;
///   `None` in one-shot mode. Only the native tool loop consumes the
///   conversation.
/// - `reporter`: output sink for status and results.
/// - `confirmer`: human-in-the-loop confirmer for writes/commands.
///
/// # Returns
///
/// `Ok(())` when the agent finishes (with or without an answer), or an error
/// if the working directory is missing or the model call fails.
pub async fn run_agent<'a, R: Reporter, C: Confirmer>(
    config: &'a AgentConfig,
    client: &'a OllamaClient,
    model: &'a str,
    task: &str,
    continuation: Option<RunContinuation<'a>>,
    reporter: &'a mut R,
    confirmer: &'a mut C,
) -> Result<()> {
    if client.is_openai() {
        return native::run_native_agent(
            config,
            client,
            model,
            task,
            continuation,
            reporter,
            confirmer,
        )
        .await;
    }

    json_loop::run_json_agent(
        config,
        client,
        model,
        task,
        continuation,
        reporter,
        confirmer,
    )
    .await
}

/// The context window for a request: the model's own context length, lowered
/// to `cap` when one is given, and never below [`MIN_CONTEXT_TOKENS`].
pub fn effective_context(model_context: u64, cap: Option<u64>) -> u64 {
    let limited = cap.map_or(model_context, |cap| model_context.min(cap));

    limited.max(MIN_CONTEXT_TOKENS)
}

/// Verify the working directory exists before starting the agent.
fn ensure_workspace_exists(cwd: &Path) -> Result<()> {
    ensure!(
        cwd.is_dir(),
        "working directory does not exist: {}",
        cwd.display()
    );

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use fakes::{FixedConfirmer, RecordingReporter, Reply, serve_replies, test_config, text_delta};

    #[test]
    fn uses_the_full_model_context_unless_capped() {
        assert_eq!(effective_context(262_144, None), 262_144);
        assert_eq!(effective_context(262_144, Some(32_768)), 32_768);
        assert_eq!(effective_context(2_048, None), MIN_CONTEXT_TOKENS);
    }

    #[test]
    fn ensure_workspace_exists_checks_the_directory() -> Result<()> {
        let root = crate::test_support::unique_temp_dir("openbatrangs-workspace-test");
        assert!(ensure_workspace_exists(&root).is_ok());

        std::fs::remove_dir_all(&root)?;
        let missing = ensure_workspace_exists(&root);
        assert!(format!("{missing:?}").contains("does not exist"));

        Ok(())
    }

    #[tokio::test]
    async fn an_ollama_server_gets_the_json_loop() -> Result<()> {
        let root = crate::test_support::unique_temp_dir("openbatrangs-run-json");
        let url = serve_replies(vec![Reply::ollama_stream(r#"{"answer":"hello"}"#)]).await?;
        let client = OllamaClient::new(&url)?;
        let config = test_config(&root, 2);
        let mut reporter = RecordingReporter::default();
        let mut confirmer = FixedConfirmer::new(true);

        run_agent(
            &config,
            &client,
            "m",
            "say hi",
            None,
            &mut reporter,
            &mut confirmer,
        )
        .await?;

        assert!(reporter.output.contains("🦇 Step 1/2"));
        assert!(reporter.output.contains("hello"));

        std::fs::remove_dir_all(root)?;
        Ok(())
    }

    #[tokio::test]
    async fn an_openai_server_gets_the_native_loop() -> Result<()> {
        let root = crate::test_support::unique_temp_dir("openbatrangs-run-native");
        let url = serve_replies(vec![Reply::openai_stream(&[text_delta("hello")])]).await?;
        let client = OllamaClient::openai(&url, None, true)?;
        let config = test_config(&root, 2);
        let mut reporter = RecordingReporter::default();
        let mut confirmer = FixedConfirmer::new(true);

        run_agent(
            &config,
            &client,
            "m",
            "say hi",
            None,
            &mut reporter,
            &mut confirmer,
        )
        .await?;

        assert!(reporter.output.contains("── step 1/2 ── m"));
        assert!(reporter.output.contains("✅ hello"));

        std::fs::remove_dir_all(root)?;
        Ok(())
    }
}
