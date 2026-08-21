//! Background agent/chat workers for the TUI.

use super::text::strip_ansi;
use crate::agent::{AgentConfig, Reporter};
use crate::cli::{AgentMode, AgentRunConfig, ModelPrefs};
use crate::constants::agent::{MAX_CONTEXT_TOKENS, MIN_CONTEXT_TOKENS};
use crate::constants::tui::{CHAT_SYSTEM_PROMPT, CHAT_TEMPERATURE, MAX_CHAT_HISTORY_MESSAGES};
use crate::model_select::{calculate_memory_budget, resolve_model, resolve_model_context};
use crate::ollama::{ChatMessage, ChatRequest, OllamaClient};
use anyhow::Result;
use futures_util::StreamExt;
use tokio::sync::mpsc;

pub(crate) enum UiEvent {
    Log(String),
    Chunk(String),
    Done(Result<(), String>),
    PullDone {
        name: String,
        result: Result<(), String>,
    },
    SetupDone(Result<(), String>),
}

struct ChannelReporter {
    tx: mpsc::UnboundedSender<UiEvent>,
}

impl Reporter for ChannelReporter {
    fn line(&mut self, msg: String) {
        let _ = self.tx.send(UiEvent::Log(strip_ansi(&msg)));
    }

    fn chunk(&mut self, msg: &str) {
        let _ = self.tx.send(UiEvent::Chunk(strip_ansi(msg)));
    }
}

pub(crate) async fn run_agent_worker(
    client: OllamaClient,
    config: AgentRunConfig,
    model_slot: Option<String>,
    prefs: ModelPrefs,
    task: String,
    chat_history: Vec<ChatMessage>,
    tx: mpsc::UnboundedSender<UiEvent>,
) -> Result<()> {
    if config.mode == AgentMode::Chat {
        return run_chat_worker(client, config, model_slot, prefs, task, chat_history, tx).await;
    }

    let mem_budget = calculate_memory_budget();
    let progress_tx = tx.clone();
    let on_status = move |msg: &str| {
        let _ = progress_tx.send(UiEvent::Log(msg.to_string()));
    };
    let selected = resolve_model(&client, &model_slot, &prefs, mem_budget, &on_status).await?;
    let model_context = resolve_model_context(&client, &selected.name).await?;
    let agent_config = AgentConfig {
        cwd: config.cwd,
        max_steps: config.max_steps,
        is_read_only: config.is_read_only || config.mode == AgentMode::Plan,
        should_confirm: config.should_confirm,
        show_thinking: config.show_thinking,
    };
    let mut reporter = ChannelReporter { tx };
    crate::agent::run_agent(
        &agent_config,
        &client,
        &selected.name,
        model_context,
        &task,
        &mut reporter,
    )
    .await
}

/// Run a plain chat completion: no tools, just conversation and code.
async fn run_chat_worker(
    client: OllamaClient,
    _config: AgentRunConfig,
    model_slot: Option<String>,
    prefs: ModelPrefs,
    _task: String,
    history: Vec<ChatMessage>,
    tx: mpsc::UnboundedSender<UiEvent>,
) -> Result<()> {
    let mem_budget = calculate_memory_budget();
    let progress_tx = tx.clone();
    let on_status = move |msg: &str| {
        let _ = progress_tx.send(UiEvent::Log(msg.to_string()));
    };
    let selected = resolve_model(&client, &model_slot, &prefs, mem_budget, &on_status).await?;
    let num_ctx = resolve_model_context(&client, &selected.name)
        .await?
        .clamp(MIN_CONTEXT_TOKENS, MAX_CONTEXT_TOKENS);

    let mut messages = vec![ChatMessage {
        role: "system".to_string(),
        content: CHAT_SYSTEM_PROMPT.to_string(),
    }];
    messages.extend(history);
    messages.truncate(MAX_CHAT_HISTORY_MESSAGES + 1);

    let request = ChatRequest {
        model: selected.name.clone(),
        messages,
        stream: true,
        format: None,
        options: Some(serde_json::json!({
            "temperature": CHAT_TEMPERATURE,
            "num_ctx": num_ctx,
        })),
    };

    let mut stream = Box::pin(client.chat_stream(request).await?);
    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        let _ = tx.send(UiEvent::Chunk(chunk));
    }
    Ok(())
}

/// Pull a model in the background and stream progress to the TUI.
pub(crate) async fn run_pull_worker(
    client: OllamaClient,
    model: String,
    tx: mpsc::UnboundedSender<UiEvent>,
) {
    let progress_tx = tx.clone();
    let result = client
        .pull(&model, &|msg| {
            let _ = progress_tx.send(UiEvent::Log(msg.to_string()));
        })
        .await;
    let _ = tx.send(UiEvent::PullDone {
        name: model,
        result: result.map_err(|error| format!("{error:#}")),
    });
}

/// Run `/setup` in the background and stream progress to the TUI.
pub(crate) async fn run_setup_worker(client: OllamaClient, tx: mpsc::UnboundedSender<UiEvent>) {
    let progress_tx = tx.clone();
    let result = crate::commands::setup_with_status(&client, &|msg| {
        let _ = progress_tx.send(UiEvent::Log(msg.to_string()));
    })
    .await;
    let _ = tx.send(UiEvent::SetupDone(
        result.map_err(|error| format!("{error:#}")),
    ));
}
