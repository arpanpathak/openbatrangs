//! Background agent/chat workers for the TUI.

use super::text::strip_ansi;
use crate::agent::{AgentConfig, Confirmer, Reporter};
use crate::cli::{AgentMode, AgentRunConfig, ModelPrefs};
use crate::constants::agent::MIN_CONTEXT_TOKENS;
use crate::constants::tui::{CHAT_SYSTEM_PROMPT, CHAT_TEMPERATURE, MAX_CHAT_HISTORY_MESSAGES};
use crate::engine::InferenceBackend;
use crate::model_select::{calculate_memory_budget, resolve_model, resolve_model_context};
use crate::ollama::{ChatMessage, ChatRequest, OllamaClient};
use anyhow::Result;
use futures_util::StreamExt;
use std::sync::Arc;
use tokio::sync::mpsc;
use tokio::sync::oneshot;

pub(crate) enum UiEvent {
    Log(String),
    Chunk(String),
    Done(Result<(), String>),
    PullDone {
        name: String,
        result: Result<(), String>,
    },
    SetupDone(Result<(), String>),
    /// The agent worker needs a y/n decision from the TUI user.
    ConfirmRequest {
        prompt: String,
        response: oneshot::Sender<bool>,
    },
}

/// TUI-side confirmation: sends the prompt to the UI and waits for the keypress.
struct ChannelConfirmer {
    tx: mpsc::UnboundedSender<UiEvent>,
}

impl Confirmer for ChannelConfirmer {
    async fn confirm(&mut self, prompt: &str) -> Result<bool> {
        let (response_tx, response_rx) = oneshot::channel();
        let _ = self.tx.send(UiEvent::ConfirmRequest {
            prompt: prompt.to_string(),
            response: response_tx,
        });
        // If the UI is gone (quit/cancel), treat it as "no" so the agent
        // aborts the tool instead of hanging forever.
        Ok(response_rx.await.unwrap_or(false))
    }
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
    backend: Arc<dyn InferenceBackend>,
    config: AgentRunConfig,
    model_slot: Option<String>,
    prefs: ModelPrefs,
    task: String,
    chat_history: Vec<ChatMessage>,
    tx: mpsc::UnboundedSender<UiEvent>,
) -> Result<()> {
    if config.mode == AgentMode::Chat {
        return run_chat_worker(backend, config, model_slot, prefs, task, chat_history, tx).await;
    }

    let mem_budget = calculate_memory_budget();
    let progress_tx = tx.clone();
    let on_status = move |msg: &str| {
        let _ = progress_tx.send(UiEvent::Log(msg.to_string()));
    };
    let selected = resolve_model(
        backend.as_ref(),
        &model_slot,
        &prefs,
        mem_budget,
        &on_status,
    )
    .await?;
    let model_context = resolve_model_context(backend.as_ref(), &selected.name).await?;
    let agent_config = AgentConfig {
        cwd: config.cwd,
        max_steps: config.max_steps,
        is_read_only: config.is_read_only || config.mode == AgentMode::Plan,
        should_confirm: config.should_confirm,
        show_thinking: config.show_thinking,
        max_ctx: config.max_ctx,
    };
    let mut reporter = ChannelReporter { tx: tx.clone() };
    let mut confirmer = ChannelConfirmer { tx };
    crate::agent::run_agent(
        &agent_config,
        backend.as_ref(),
        &selected.name,
        model_context,
        &task,
        &mut reporter,
        &mut confirmer,
    )
    .await
}

/// Run a plain chat completion: no tools, just conversation and code.
async fn run_chat_worker(
    backend: Arc<dyn InferenceBackend>,
    config: AgentRunConfig,
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
    let selected = resolve_model(
        backend.as_ref(),
        &model_slot,
        &prefs,
        mem_budget,
        &on_status,
    )
    .await?;
    let max_ctx = config.max_ctx.max(MIN_CONTEXT_TOKENS);
    let num_ctx = resolve_model_context(backend.as_ref(), &selected.name)
        .await?
        .clamp(MIN_CONTEXT_TOKENS, max_ctx);

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

    let mut stream = Box::pin(backend.chat_stream(request).await?);
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

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn channel_confirmer_sends_prompt_and_waits_for_answer() {
        let (tx, mut rx) = mpsc::unbounded_channel();
        let mut confirmer = ChannelConfirmer { tx };
        let handle = tokio::spawn(async move { confirmer.confirm("write file 'a.txt'?").await });

        let event = rx.recv().await.expect("confirmer should send an event");
        let UiEvent::ConfirmRequest { prompt, response } = event else {
            panic!("expected ConfirmRequest");
        };
        assert_eq!(prompt, "write file 'a.txt'?");
        response.send(true).unwrap();
        assert!(handle.await.unwrap().unwrap());
    }

    #[tokio::test]
    async fn channel_confirmer_aborts_when_ui_is_gone() {
        // `_` (not `_rx`) drops the receiver immediately, so the confirmer's
        // prompt can never be answered and must resolve to "no".
        let (tx, _) = mpsc::unbounded_channel();
        let mut confirmer = ChannelConfirmer { tx };
        let result = confirmer.confirm("run command: make?").await.unwrap();
        assert!(!result, "dropped UI must be treated as 'no'");
    }
}
