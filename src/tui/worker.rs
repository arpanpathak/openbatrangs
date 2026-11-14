//! Background agent/chat workers for the TUI.

use crate::agent::{
    AgentConfig, AgentEvent, Confirmer, HistorySink, Reporter, RunContinuation, TextKind,
    compact_context, effective_context, max_output_tokens, run_agent,
};
use crate::cli::{AgentMode, AgentRunConfig, ModelPrefs};
use crate::constants::tui::{CHAT_SYSTEM_PROMPT, CHAT_TEMPERATURE};
use crate::model_select::{calculate_memory_budget, resolve_model, resolve_model_context};
use crate::models::ModelScore;
use crate::ollama::{ChatMessage, ChatRequest, Delta, OllamaClient, Role};
use crate::update::{Installer, NewRelease, UpdateCheck, daily_check};
use anyhow::Result;
use futures_util::{Stream, TryStreamExt};
use serde_json::Value;
use std::pin::Pin;
use tokio::sync::mpsc;
use tokio::sync::oneshot;

/// Messages sent from background workers to the TUI event loop.
pub(crate) enum UiEvent {
    /// A status line to show in the transcript (model selection, pulls).
    Log(String),
    /// An event from the agent loop, or reply text in chat mode.
    Agent(AgentEvent),
    /// The agent task finished, either successfully or with an error message.
    Done(Result<(), String>),
    /// A snapshot of the agent conversation, kept so an interrupted run can
    /// resume with its context instead of starting over.
    History(Vec<Value>),
    /// A background model pull completed with the given name and result.
    PullDone {
        /// The Ollama model tag that was being pulled.
        name: String,
        /// Pull outcome: `Ok(())` on success or `Err(message)` on failure.
        result: Result<(), String>,
    },
    /// The `/setup` background task completed with the given result.
    SetupDone(Result<(), String>),
    /// The agent worker needs a y/n decision from the TUI user.
    ConfirmRequest {
        /// Human-readable prompt describing the action to confirm.
        prompt: String,
        /// One-shot channel for sending the user's yes/no answer back.
        response: oneshot::Sender<bool>,
    },
    /// A newer release exists; the user answers whether to install it.
    UpdateAvailable {
        /// The question, naming the new and the running version.
        prompt: String,
        /// One-shot channel for the user's yes/no answer.
        response: oneshot::Sender<bool>,
    },
    /// An event from a background job (an agent task, a pull or a setup),
    /// tagged with the job's number so events from a cancelled job can be
    /// told apart from those of the job that runs now.
    Job {
        /// The number the app gave the job when it started it.
        job_id: u64,
        /// The event the job sent.
        event: Box<UiEvent>,
    },
}

/// A stream of reply deltas from [`OllamaClient::chat_stream`].
type DeltaStream = Pin<Box<dyn Stream<Item = Result<Delta>> + Send + 'static>>;

/// Send an event to the TUI event loop.
///
/// The receiver is gone only after the TUI has quit, and then nobody needs
/// the event, so a failed send is ignored.
pub(super) fn send_event(tx: &mpsc::UnboundedSender<UiEvent>, event: UiEvent) {
    tx.send(event).ok();
}

/// Pass every event of job `job_id` on to the event loop, tagged with
/// the job's number. Ends when the job and all its senders are gone.
pub(super) async fn forward_job_events(
    job_id: u64,
    mut job_rx: mpsc::UnboundedReceiver<UiEvent>,
    tx: mpsc::UnboundedSender<UiEvent>,
) {
    while let Some(event) = job_rx.recv().await {
        let event = Box::new(event);

        send_event(&tx, UiEvent::Job { job_id, event });
    }
}

/// A status callback that logs each message in the chat.
fn status_logger(tx: &mpsc::UnboundedSender<UiEvent>) -> impl Fn(&str) + Sync + use<> {
    let tx = tx.clone();

    move |message: &str| send_event(&tx, UiEvent::Log(message.to_string()))
}

/// A worker result as the event loop stores it: the error chain as text.
fn error_text(result: Result<()>) -> Result<(), String> {
    result.map_err(|error| format!("{error:#}"))
}

/// TUI-side confirmation: sends the prompt to the UI and waits for the keypress.
struct ChannelConfirmer {
    /// Channel for sending confirmation requests to the TUI event loop.
    tx: mpsc::UnboundedSender<UiEvent>,
}

impl Confirmer for ChannelConfirmer {
    /// Show `prompt` in the TUI and wait for the user's answer.
    async fn confirm(&mut self, prompt: &str) -> Result<bool> {
        let (response_tx, response_rx) = oneshot::channel();

        send_event(
            &self.tx,
            UiEvent::ConfirmRequest {
                prompt: prompt.to_string(),
                response: response_tx,
            },
        );

        // If the UI is gone (quit/cancel), treat it as "no" so the agent
        // aborts the tool instead of hanging forever.
        Ok(response_rx.await.unwrap_or(false))
    }
}

/// Forwards agent events to the TUI event channel.
struct ChannelReporter {
    /// Channel for forwarding agent events to the TUI event loop.
    tx: mpsc::UnboundedSender<UiEvent>,
}

impl Reporter for ChannelReporter {
    /// Send `event` to the TUI.
    fn report(&mut self, event: AgentEvent) {
        send_event(&self.tx, UiEvent::Agent(event));
    }
}

/// Sends conversation snapshots to the TUI for interrupt-safe resume.
struct ChannelHistorySink {
    /// Channel for forwarding conversation snapshots to the event loop.
    tx: mpsc::UnboundedSender<UiEvent>,
}

impl HistorySink for ChannelHistorySink {
    /// Send the conversation so far.
    fn save(&mut self, messages: Vec<Value>) {
        send_event(&self.tx, UiEvent::History(messages));
    }
}

/// Prior context handed to a worker: the session's turns and the agent-mode
/// resume snapshot.
///
/// Bundling the two keeps [`run_agent_worker`]'s signature small.
pub(crate) struct WorkerHistory {
    /// The session's tasks and answers, ending with the current task. Chat
    /// mode sends them as the conversation; agent and plan modes put the
    /// earlier ones in the first message.
    pub chat: Vec<ChatMessage>,
    /// Prior agent conversation (OpenAI message shape) to resume from.
    pub resume: Option<Vec<Value>>,
}

/// Run the agentic coding loop in the background, streaming events to the TUI.
///
/// Dispatches to [`run_chat_worker`] for chat mode or runs the full
/// tool-using agent for agent/plan modes.
///
/// # Parameters
///
/// - `client`: Ollama client for model calls.
/// - `config`: agent runtime configuration (mode, limits, safety flags).
/// - `model_slot`: explicit model tag or `None` for auto-selection.
/// - `prefs`: model preferences for auto-selection (min context, auto-pull).
/// - `task`: the user's task description.
/// - `prior`: prior chat messages or the interrupted agent conversation.
/// - `tx`: channel for sending UI events back to the event loop.
///
/// # Returns
///
/// `Ok(())` on success, or an error if the agent encounters an unrecoverable failure.
pub(crate) async fn run_agent_worker(
    client: OllamaClient,
    config: AgentRunConfig,
    model_slot: Option<String>,
    prefs: ModelPrefs,
    task: String,
    prior: WorkerHistory,
    tx: mpsc::UnboundedSender<UiEvent>,
) -> Result<()> {
    match config.mode {
        AgentMode::Chat => run_chat_worker(client, config, model_slot, prefs, prior.chat, tx).await,
        AgentMode::Agent | AgentMode::Plan => {
            run_tool_agent(client, config, model_slot, prefs, task, prior, tx).await
        }
    }
}

/// Run the tool-using agent (agent and plan modes) with the session's
/// earlier turns, or with the interrupted conversation it continues.
async fn run_tool_agent(
    client: OllamaClient,
    config: AgentRunConfig,
    model_slot: Option<String>,
    prefs: ModelPrefs,
    task: String,
    prior: WorkerHistory,
    tx: mpsc::UnboundedSender<UiEvent>,
) -> Result<()> {
    let selected = select_model(&client, &model_slot, &prefs, &tx).await?;
    let model_context = resolve_model_context(&client, &selected.name).await?;
    let agent_config = agent_config(config, model_context);

    let mut reporter = ChannelReporter { tx: tx.clone() };
    let mut confirmer = ChannelConfirmer { tx: tx.clone() };
    let mut history = ChannelHistorySink { tx };
    let continuation = RunContinuation {
        messages: prior.resume,
        earlier: earlier_turns(prior.chat),
        sink: Some(&mut history),
    };

    run_agent(
        &agent_config,
        &client,
        &selected.name,
        &task,
        Some(continuation),
        &mut reporter,
        &mut confirmer,
    )
    .await
}

/// The chat turns before the current task. The TUI records the task as
/// the last turn before the run starts; the run gets it as its task instead.
fn earlier_turns(mut chat: Vec<ChatMessage>) -> Vec<ChatMessage> {
    chat.pop();

    chat
}

/// Pick the model to run, logging pull and selection progress in the chat.
async fn select_model(
    client: &OllamaClient,
    model_slot: &Option<String>,
    prefs: &ModelPrefs,
    tx: &mpsc::UnboundedSender<UiEvent>,
) -> Result<ModelScore> {
    let on_status = status_logger(tx);

    resolve_model(
        client,
        model_slot,
        prefs,
        calculate_memory_budget(),
        &on_status,
    )
    .await
}

/// The agent loop's configuration for a TUI run. Plan mode is read-only.
fn agent_config(config: AgentRunConfig, model_context: u64) -> AgentConfig {
    let is_read_only = config.is_read_only || config.mode == AgentMode::Plan;

    AgentConfig {
        cwd: config.cwd,
        max_steps: config.max_steps,
        is_read_only,
        should_confirm: config.should_confirm,
        use_sandbox: config.use_sandbox,
        show_thinking: config.show_thinking,
        model_context,
        max_ctx: config.max_ctx,
        searxng_url: config.searxng_url,
        search_key: config.search_key,
    }
}

/// Run a plain chat completion: no tools, just conversation and code.
async fn run_chat_worker(
    client: OllamaClient,
    config: AgentRunConfig,
    model_slot: Option<String>,
    prefs: ModelPrefs,
    history: Vec<ChatMessage>,
    tx: mpsc::UnboundedSender<UiEvent>,
) -> Result<()> {
    let selected = select_model(&client, &model_slot, &prefs, &tx).await?;
    let model_context = resolve_model_context(&client, &selected.name).await?;
    let num_ctx = effective_context(model_context, config.max_ctx);
    let request = chat_request(selected.name, history, num_ctx);

    note_model_loading(&client, &tx);
    let stream = client.chat_stream(request).await?;

    forward_chat_stream(stream, &tx).await
}

/// The streamed chat request: the system prompt, then the conversation.
///
/// Old turns are paged out when the estimated prompt would overflow the
/// safe per-device window, as in agent mode.
fn chat_request(model: String, history: Vec<ChatMessage>, num_ctx: u64) -> ChatRequest {
    let system = ChatMessage {
        role: Role::System,
        content: CHAT_SYSTEM_PROMPT.to_string(),
    };
    let mut messages: Vec<ChatMessage> = std::iter::once(system).chain(history).collect();

    compact_context(&mut messages, num_ctx);

    let num_predict = max_output_tokens(num_ctx, &messages);

    ChatRequest {
        model,
        messages,
        stream: true,
        keep_alive: Some(serde_json::json!(crate::constants::ollama::KEEP_ALIVE)),
        format: None,
        options: Some(serde_json::json!({
            "temperature": CHAT_TEMPERATURE,
            "num_ctx": num_ctx,
            "num_predict": num_predict,
        })),
    }
}

/// Tell the user that Ollama may be loading the model.
///
/// The first request after an idle period can take a while because Ollama
/// has to load the model into memory; without this line the user only sees
/// "thinking...". An OpenAI-compatible server keeps its model loaded.
fn note_model_loading(client: &OllamaClient, tx: &mpsc::UnboundedSender<UiEvent>) {
    if client.is_openai() {
        return;
    }

    send_event(
        tx,
        UiEvent::Log("⏳ Loading model into Ollama (first reply may take a moment)...".to_string()),
    );
}

/// Forward every reply delta to the TUI until the stream ends or fails.
async fn forward_chat_stream(
    stream: DeltaStream,
    tx: &mpsc::UnboundedSender<UiEvent>,
) -> Result<()> {
    let mut has_noted_thinking = false;

    stream
        .try_for_each(|delta| {
            forward_delta(delta, &mut has_noted_thinking, tx);
            std::future::ready(Ok(()))
        })
        .await
}

/// Send reply text as answer text, and note the first reasoning delta once.
/// Chat mode has no tools, so tool-call deltas are dropped.
fn forward_delta(delta: Delta, has_noted_thinking: &mut bool, tx: &mpsc::UnboundedSender<UiEvent>) {
    match delta {
        Delta::Content(text) => send_event(tx, UiEvent::Agent(answer_text(text))),
        Delta::Thinking(_reasoning) => note_thinking(has_noted_thinking, tx),
        Delta::ToolCalls(_calls) => {}
    }
}

/// An answer-text event carrying `text`.
fn answer_text(text: String) -> AgentEvent {
    AgentEvent::Text {
        kind: TextKind::Answer,
        text,
    }
}

/// Log "thinking..." the first time the model reasons in a reply.
fn note_thinking(has_noted_thinking: &mut bool, tx: &mpsc::UnboundedSender<UiEvent>) {
    if *has_noted_thinking {
        return;
    }

    *has_noted_thinking = true;
    send_event(tx, UiEvent::Log("💭 thinking...".to_string()));
}

/// Pull a model in the background and stream progress to the TUI.
pub(crate) async fn run_pull_worker(
    client: OllamaClient,
    model: String,
    tx: mpsc::UnboundedSender<UiEvent>,
) {
    let result = client.pull(&model, &status_logger(&tx)).await;

    send_event(
        &tx,
        UiEvent::PullDone {
            name: model,
            result: error_text(result),
        },
    );
}

/// Run `/setup` in the background and stream progress to the TUI.
pub(crate) async fn run_setup_worker(client: OllamaClient, tx: mpsc::UnboundedSender<UiEvent>) {
    let result = crate::commands::setup_with_status(&client, &status_logger(&tx)).await;

    send_event(&tx, UiEvent::SetupDone(error_text(result)));
}

/// Check for a newer release once a day; when one exists and the user
/// accepts it, run the installer and show its output in the transcript.
pub(crate) async fn run_update_worker(check: UpdateCheck, tx: mpsc::UnboundedSender<UiEvent>) {
    let Some(release) = daily_check(&check).await else {
        return;
    };

    offer_update(&check.installer, &release, &tx).await;
}

/// Ask whether to install `release`, and install it on yes.
async fn offer_update(
    installer: &Installer,
    release: &NewRelease,
    tx: &mpsc::UnboundedSender<UiEvent>,
) {
    if !ask_to_install(release, tx).await {
        return;
    }

    install_update(installer, release, tx).await;
}

/// Show the install question and wait for the answer. A closed dialog or
/// a TUI that quit counts as no.
async fn ask_to_install(release: &NewRelease, tx: &mpsc::UnboundedSender<UiEvent>) -> bool {
    let (response_tx, response_rx) = oneshot::channel();

    send_event(
        tx,
        UiEvent::UpdateAvailable {
            prompt: release.install_question(),
            response: response_tx,
        },
    );

    response_rx.await.unwrap_or(false)
}

/// Run the installer, showing each line it prints, then the restart line
/// or the error.
async fn install_update(
    installer: &Installer,
    release: &NewRelease,
    tx: &mpsc::UnboundedSender<UiEvent>,
) {
    send_event(tx, UiEvent::Log(release.installing_line()));

    let outcome = installer
        .run(|line| send_event(tx, UiEvent::Log(line)))
        .await;

    let last_line = match outcome {
        Ok(()) => release.restart_line(),
        Err(error) => format!("⚠️ The update failed: {error:#}"),
    };

    send_event(tx, UiEvent::Log(last_line));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{MockResponse, spawn_mock_server};
    use anyhow::Context;

    /// A run configuration for `mode` in a workspace that does not exist.
    fn run_config(mode: AgentMode) -> AgentRunConfig {
        AgentRunConfig {
            cwd: std::path::PathBuf::from("/tmp/openbatrangs-tui-worker-missing-workspace"),
            max_steps: 1,
            is_read_only: false,
            should_confirm: true,
            use_sandbox: true,
            mode,
            show_thinking: true,
            max_ctx: None,
            searxng_url: String::new(),
            search_key: None,
        }
    }

    /// Model preferences that never pull a model.
    fn prefs() -> ModelPrefs {
        ModelPrefs {
            model: None,
            is_auto_pull_disabled: true,
            min_context: 0,
        }
    }

    /// Prior context with no chat turns and nothing to resume.
    fn no_history() -> WorkerHistory {
        WorkerHistory {
            chat: Vec::new(),
            resume: None,
        }
    }

    /// Every event already waiting on `rx`.
    fn sent_events(rx: &mut mpsc::UnboundedReceiver<UiEvent>) -> Vec<UiEvent> {
        std::iter::from_fn(|| rx.try_recv().ok()).collect()
    }

    /// The log lines and streamed answer text among `events`, in order.
    fn texts(events: &[UiEvent]) -> Vec<String> {
        events.iter().filter_map(event_text).collect()
    }

    /// The text of a log line or of streamed answer text.
    fn event_text(event: &UiEvent) -> Option<String> {
        match event {
            UiEvent::Log(line) => Some(line.clone()),
            UiEvent::Agent(AgentEvent::Text { text, .. }) => Some(text.clone()),
            _other => None,
        }
    }

    /// The base URL of a mock Ollama server with model `coder:1b` that
    /// replies `hello`.
    async fn ollama_server() -> String {
        spawn_mock_server(|path| match path {
            "/api/show" => MockResponse::json("200 OK", "{}"),
            "/api/chat" => MockResponse::json(
                "200 OK",
                "{\"message\":{\"content\":\"hello\"}}\n{\"done\":true}\n",
            ),
            _tags => MockResponse::json("200 OK", r#"{"models":[{"name":"coder:1b","size":1}]}"#),
        })
        .await
    }

    /// The base URL of a mock OpenAI-compatible server whose model reasons
    /// twice, then replies `hi`.
    async fn openai_server() -> String {
        let base_url = spawn_mock_server(|path| match path {
            "/v1/chat/completions" => MockResponse::json(
                "200 OK",
                concat!(
                    "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"hmm\"}}]}\n\n",
                    "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"more\"}}]}\n\n",
                    "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n",
                    "data: [DONE]\n\n",
                ),
            ),
            _models => MockResponse::json(
                "200 OK",
                r#"{"data":[{"id":"served","max_model_len":8192}]}"#,
            ),
        })
        .await;

        format!("{base_url}/v1")
    }

    /// The prompt and answer channel of a confirmation request.
    fn confirm_request(event: UiEvent) -> Option<(String, oneshot::Sender<bool>)> {
        match event {
            UiEvent::ConfirmRequest { prompt, response } => Some((prompt, response)),
            _other => None,
        }
    }

    #[tokio::test]
    async fn channel_confirmer_sends_prompt_and_waits_for_answer() -> anyhow::Result<()> {
        let (tx, mut rx) = mpsc::unbounded_channel();
        let mut confirmer = ChannelConfirmer { tx };
        let handle = tokio::spawn(async move { confirmer.confirm("write file 'a.txt'?").await });

        let event = rx.recv().await.context("confirmer should send an event")?;
        let (prompt, response) = confirm_request(event).context("expected ConfirmRequest")?;
        assert_eq!(prompt, "write file 'a.txt'?");

        assert!(response.send(true).is_ok());
        assert!(handle.await??);
        assert!(confirm_request(UiEvent::Log(String::new())).is_none());

        Ok(())
    }

    #[tokio::test]
    async fn a_failed_release_check_shows_nothing() {
        use crate::update::fakes::fake_check;

        let check = fake_check("http://127.0.0.1:1/releases/latest", "exit 1");
        let (tx, mut rx) = mpsc::unbounded_channel();

        run_update_worker(check, tx).await;

        assert!(rx.recv().await.is_none());
    }

    #[tokio::test]
    async fn channel_confirmer_aborts_when_ui_is_gone() -> anyhow::Result<()> {
        // `_` (not `_rx`) drops the receiver immediately, so the confirmer's
        // prompt can never be answered and must resolve to "no".
        let (tx, _) = mpsc::unbounded_channel();
        let mut confirmer = ChannelConfirmer { tx };

        let result = confirmer.confirm("run command: make?").await?;

        assert!(!result, "dropped UI must be treated as 'no'");

        Ok(())
    }

    #[test]
    fn the_current_task_is_not_an_earlier_turn() {
        let turn = |content: &str| ChatMessage {
            role: Role::User,
            content: content.to_string(),
        };

        let earlier = earlier_turns(vec![turn("first task"), turn("current task")]);

        assert_eq!(earlier.len(), 1);
        assert_eq!(earlier[0].content, "first task");
    }

    #[test]
    fn reporter_and_history_sink_forward_to_the_tui() {
        let (tx, mut rx) = mpsc::unbounded_channel();
        let mut reporter = ChannelReporter { tx: tx.clone() };
        let mut history = ChannelHistorySink { tx };

        reporter.notice("a notice".to_string());
        reporter.report(answer_text("bold".to_string()));
        history.save(vec![serde_json::json!({"role": "user"})]);

        let events = sent_events(&mut rx);

        assert!(
            matches!(&events[0], UiEvent::Agent(AgentEvent::Notice(text)) if text == "a notice")
        );
        assert_eq!(texts(&events), ["bold"]);
        assert!(matches!(events.last(), Some(UiEvent::History(_))));
    }

    #[tokio::test]
    async fn plan_mode_selects_the_model_then_needs_a_workspace() -> anyhow::Result<()> {
        let client = OllamaClient::new(&ollama_server().await)?;
        let (tx, _rx) = mpsc::unbounded_channel();
        let config = run_config(AgentMode::Plan);
        let model = Some("coder:1b".to_string());

        let result = run_agent_worker(
            client,
            config,
            model,
            prefs(),
            "task".to_string(),
            no_history(),
            tx,
        )
        .await;

        let error = result.err().context("a missing workspace must fail")?;
        assert!(format!("{error:#}").contains("does not exist"));

        Ok(())
    }

    #[tokio::test]
    async fn chat_mode_streams_the_reply_from_ollama() -> anyhow::Result<()> {
        let client = OllamaClient::new(&ollama_server().await)?;
        let (tx, mut rx) = mpsc::unbounded_channel();
        let config = run_config(AgentMode::Chat);
        let model = Some("coder:1b".to_string());

        run_agent_worker(
            client,
            config,
            model,
            prefs(),
            "hi".to_string(),
            no_history(),
            tx,
        )
        .await?;

        let texts = texts(&sent_events(&mut rx));
        assert!(texts[0].starts_with("⏳ Loading model"));
        assert_eq!(texts[1], "hello");

        Ok(())
    }

    #[tokio::test]
    async fn chat_mode_notes_thinking_once_on_an_openai_server() -> anyhow::Result<()> {
        let client = OllamaClient::openai(&openai_server().await, None, true)?;
        let (tx, mut rx) = mpsc::unbounded_channel();
        let config = run_config(AgentMode::Chat);

        run_agent_worker(
            client,
            config,
            None,
            prefs(),
            "hi".to_string(),
            no_history(),
            tx,
        )
        .await?;

        assert_eq!(texts(&sent_events(&mut rx)), ["💭 thinking...", "hi"]);

        Ok(())
    }

    #[test]
    fn tool_call_deltas_are_dropped_in_chat_mode() {
        let (tx, mut rx) = mpsc::unbounded_channel();
        let mut has_noted_thinking = false;

        forward_delta(Delta::ToolCalls(Vec::new()), &mut has_noted_thinking, &tx);

        assert!(sent_events(&mut rx).is_empty());
    }

    #[tokio::test]
    async fn pull_worker_reports_progress_and_success() -> anyhow::Result<()> {
        let base_url = spawn_mock_server(|_path| {
            MockResponse::json(
                "200 OK",
                "{\"status\":\"pulling manifest\"}\n{\"status\":\"success\"}\n",
            )
        })
        .await;
        let client = OllamaClient::new(&base_url)?;
        let (tx, mut rx) = mpsc::unbounded_channel();

        run_pull_worker(client, "coder:1b".to_string(), tx).await;

        let events = sent_events(&mut rx);
        assert!(
            texts(&events)
                .iter()
                .any(|line| line.contains("pulling manifest"))
        );
        assert!(matches!(
            events.last(),
            Some(UiEvent::PullDone { result: Ok(()), .. })
        ));

        Ok(())
    }

    #[tokio::test]
    async fn setup_worker_reports_the_error_as_text() -> anyhow::Result<()> {
        // The server answers, so setup never installs Ollama, but its model
        // list cannot be parsed.
        let base_url = spawn_mock_server(|_path| MockResponse::json("200 OK", "not json")).await;
        let client = OllamaClient::new(&base_url)?;
        let (tx, mut rx) = mpsc::unbounded_channel();

        run_setup_worker(client, tx).await;

        let events = sent_events(&mut rx);
        assert!(texts(&events)[0].starts_with("✅ Ollama already running"));
        assert!(matches!(events.last(), Some(UiEvent::SetupDone(Err(_)))));

        Ok(())
    }
}
