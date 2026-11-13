//! TUI application state and event handling.

use super::chat::ChatRenderCache;
use super::scroll::is_inside_track;
use super::session::SessionLog;
use super::worker::send_event;
use super::{
    ScrollMode, UiEvent, WorkerHistory, chat_visual_line_indices, extract_path_from_line,
    open_in_vim, run_agent_worker,
};
use crate::cli::{AgentMode, AgentRunConfig, Cli, ModelPrefs, search_key, searxng_url};
use crate::constants::models::BYTES_PER_GIGABYTE;
use crate::constants::perf::MIB_PER_GIB;
use crate::constants::tui::{
    CHARS_PER_TOKEN, CHAT_SCROLL_STEP, CHAT_SEPARATOR_LENGTH, MAX_CHAT_HISTORY_MESSAGES,
    MAX_LIVE_CHARS, MAX_LOG_LINES, SPINNER, STREAM_GAP, TOKEN_RATE_MIN_ELAPSED, VIM_TERMINALS,
};
use crate::ollama::{ChatMessage, OllamaClient, OllamaModel, Role};
use crate::perf::{PerfMonitor, SystemStats};
use crossterm::event::{
    KeyCode, KeyEvent, KeyEventKind, KeyModifiers, MouseButton, MouseEvent, MouseEventKind,
};
use ratatui::layout::Rect;
use serde_json::Value;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::time::Instant;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;

/// Model picker popup state.
pub(super) struct PickerState {
    /// Available model tags to choose from.
    pub(super) models: Vec<String>,
    /// Index of the currently highlighted model.
    pub(super) selected: usize,
}

impl PickerState {
    /// Highlight the model above the current one.
    fn select_previous(&mut self) {
        self.selected = self.selected.saturating_sub(1);
    }

    /// Highlight the model below the current one, stopping at the last.
    fn select_next(&mut self) {
        self.selected = (self.selected + 1).min(self.models.len().saturating_sub(1));
    }

    /// The highlighted model tag, or `None` when the list is empty.
    fn highlighted_model(&self) -> Option<String> {
        self.models.get(self.selected).cloned()
    }
}

/// Pending tool-confirmation dialog state.
pub(super) struct PendingConfirmation {
    /// Human-readable prompt shown to the user, e.g. `write file 'a.txt'?`.
    pub(super) prompt: String,
    /// Channel to send the user's yes/no answer back to the agent worker.
    pub(super) response: tokio::sync::oneshot::Sender<bool>,
}

/// Mutually exclusive modal overlays.
///
/// Using an enum instead of two independent `Option` fields prevents the picker
/// and the confirmation dialog from being open simultaneously — a state that
/// caused subtle key-dispatch bugs where confirmation keys leaked into the
/// picker or vice versa.
pub(super) enum ActiveModal {
    /// The model picker is open.
    Picker(PickerState),
    /// A tool-confirmation prompt is displayed.
    Confirm(PendingConfirmation),
}

/// Where a key press goes, decided by the open modal.
enum KeyTarget {
    /// A key release; terminals with the kitty protocol report one per key.
    Ignored,
    /// The tool-confirmation dialog.
    Confirmation,
    /// The model picker.
    Picker,
    /// The input line.
    Input,
}

/// What a key press does in the input line.
#[derive(Debug, PartialEq)]
enum InputAction {
    /// Cancel the running task and clear the input (Ctrl+C).
    Cancel,
    /// Insert a newline (Ctrl+J, or Enter with Shift, Ctrl or Alt).
    Newline,
    /// Insert a typed character.
    Insert(char),
    /// Delete the character before the cursor.
    Backspace,
    /// Delete the character at the cursor.
    Delete,
    /// Move the cursor one character left.
    Left,
    /// Move the cursor one character right.
    Right,
    /// Move the cursor to the start of the input.
    Home,
    /// Move the cursor to the end of the input.
    End,
    /// Previous suggestion or older history entry.
    Up,
    /// Next suggestion or newer history entry.
    Down,
    /// Scroll the chat up one page.
    PageUp,
    /// Scroll the chat down one page.
    PageDown,
    /// Accept the highlighted suggestion (Tab).
    Complete,
    /// Submit the input as a task or slash command.
    Submit,
    /// Do nothing. Esc lands here on purpose: it must neither quit the TUI
    /// nor destroy text the user is composing.
    Nothing,
}

/// What a key press does while the tool-confirmation dialog is open.
#[derive(Debug, PartialEq)]
enum ConfirmationKey {
    /// Cancel the whole task (Ctrl+C).
    CancelTask,
    /// Answer the prompt: `true` runs the tool, `false` aborts it.
    Answer(bool),
    /// Leave the dialog open.
    Ignore,
}

/// What a key press does while the model picker is open.
#[derive(Debug, PartialEq)]
enum PickerKey {
    /// Highlight the model above.
    Up,
    /// Highlight the model below.
    Down,
    /// Switch to the highlighted model and close the picker.
    Choose,
    /// Close the picker without changing the model.
    Close,
    /// Leave the picker as it is.
    Ignore,
}

/// Central TUI application state.
///
/// Owns every piece of mutable state the event loop and renderer need: the
/// chat log, input editing buffer, running agent handle, performance monitor,
/// and modal overlays. Keeping all state in one struct lets the event loop
/// borrow it mutably without interior-refcell gymnastics.
///
/// ## References
///
/// - Elm Architecture for UI state: <https://guide.elm-lang.org/architecture/>
pub(super) struct App {
    /// Bounded, disk-backed chat log that stores every message in the session.
    pub(super) log: SessionLog,
    /// In-progress streaming output not yet flushed to the log.
    pub(super) live: String,
    /// Current text in the input box being composed by the user.
    pub(super) input: String,
    /// Byte offset of the text cursor within `input`.
    pub(super) cursor: usize,
    /// Previously submitted input lines for Up/Down history navigation.
    pub(super) history: Vec<String>,
    /// Index into `history` during navigation, or `None` when not navigating.
    pub(super) history_idx: Option<usize>,
    /// Index of the currently highlighted suggestion in the autocomplete list.
    pub(super) selected: usize,
    /// Whether an agent task or background operation is currently running.
    pub(super) is_running: bool,
    /// Short status label shown in the status bar (e.g. "ready", "running").
    pub(super) status: String,
    /// Description of the current agent action (e.g. "write file 'a.txt'").
    pub(super) last_action: String,
    /// Whether the chat viewport auto-follows new output or preserves scroll.
    pub(super) scroll_mode: ScrollMode,
    /// Vertical scroll offset in wrapped visual rows when in manual scroll mode.
    pub(super) chat_scroll_offset: usize,
    /// Monotonically increasing counter used to cycle through spinner frames.
    pub(super) spinner_frame: u64,
    /// FIFO queue of user tasks waiting to run after the current task
    /// finishes, each with the conversation snapshot to resume from, if any.
    pub(super) task_queue: VecDeque<(String, Option<Vec<Value>>)>,
    /// Conversation history for chat-mode (no tools) interactions.
    pub(super) chat_history: Vec<ChatMessage>,
    /// The currently active modal overlay (picker or confirmation).
    /// Using `Option<ActiveModal>` ensures the two modals cannot overlap.
    pub(super) modal: Option<ActiveModal>,
    /// When `true`, the main event loop exits and the TUI shuts down.
    pub(super) should_quit: bool,
    /// Ollama model tag currently selected, or `None` for auto-selection.
    pub(super) model: Option<String>,
    /// URL of the Ollama server this app is connected to.
    pub(super) server_url: String,
    /// Pre-formatted model metadata line (size, params, quant, context).
    pub(super) model_info: Option<String>,
    /// The user's original task prompt, displayed in the banner area.
    pub(super) current_prompt: Option<String>,
    /// Agent task retained when Ctrl+C is followed by corrective steering.
    pub(super) interrupted_task: Option<String>,
    /// Last conversation snapshot from the native agent loop, kept so an
    /// interrupted run resumes with its context instead of starting over.
    pub(super) agent_conversation: Option<Vec<Value>>,

    /// Runtime configuration for the agent loop (mode, limits, safety flags).
    pub(super) run_config: AgentRunConfig,
    /// Minimum context window (tokens) the user requires for model selection.
    pub(super) min_context: u64,
    /// When `true`, the app will never auto-pull a model from the registry.
    pub(super) is_auto_pull_disabled: bool,
    /// Whether the live performance panel (GPU/CPU/RAM) is visible.
    pub(super) show_perf: bool,
    /// Whether crossterm mouse capture is active (wheel + scrollbar vs. native select).
    pub(super) mouse_capture: bool,
    /// System performance sampler that reads CPU, RAM, and GPU metrics.
    pub(super) perf: PerfMonitor,
    /// Most recently sampled system and GPU statistics.
    pub(super) system_stats: SystemStats,
    /// Number of characters received in the current streaming response.
    pub(super) stream_chars: u64,
    /// Timestamp of the first chunk in the current streaming response.
    pub(super) stream_started_at: Option<Instant>,
    /// Estimated tokens per second based on streaming character rate.
    pub(super) tokens_per_sec: f64,
    /// Handle to the currently spawned background Tokio task (agent/pull/setup).
    pub(super) current_task: Option<JoinHandle<()>>,
    /// Last measured chat area rectangle, used for scrollbar and click math.
    pub(super) last_chat_area: Option<Rect>,
    /// Characters accumulated in the current rate-measurement window.
    pub(super) rate_window_chars: u64,
    /// Start time of the current rate-measurement window.
    pub(super) rate_window_start: Option<Instant>,
    /// When the last chunk arrived; a longer gap than [`STREAM_GAP`] starts a
    /// new rate window, so time spent running tools is not counted as streaming.
    pub(super) last_chunk_at: Option<Instant>,
    /// Cache of the last syntax-highlighted chat rendering result.
    pub(super) chat_render: ChatRenderCache,
    /// Terminal emulators tried, in order, when a clicked file opens in vim.
    pub(super) editor_terminals: &'static [&'static str],
}

impl App {
    /// Create a new `App` with default state from CLI arguments.
    ///
    /// # Parameters
    ///
    /// - `cli`: parsed command-line arguments providing initial model, server URL, etc.
    /// - `tegrastats`: shared buffer for the background `tegrastats` reader thread.
    ///
    /// # Returns
    ///
    /// A fully initialized `App` ready for the event loop.
    pub(super) fn new(cli: &Cli, tegrastats: Arc<Mutex<Option<String>>>) -> Self {
        Self {
            log: SessionLog::new(MAX_LOG_LINES),
            live: String::new(),
            input: String::new(),
            cursor: 0,
            history: Vec::new(),
            history_idx: None,
            selected: 0,
            is_running: false,
            status: "ready".to_string(),
            last_action: String::new(),
            scroll_mode: ScrollMode::Manual,
            chat_scroll_offset: 0,
            spinner_frame: 0,
            task_queue: VecDeque::new(),
            chat_history: Vec::new(),
            modal: None,
            should_quit: false,
            model: cli.model.clone(),
            server_url: cli.ollama_url.clone(),
            model_info: None,
            current_prompt: None,
            interrupted_task: None,
            agent_conversation: None,
            run_config: AgentRunConfig {
                cwd: cli.cwd.clone(),
                max_steps: cli.max_steps,
                is_read_only: cli.is_read_only,
                should_confirm: true,
                mode: AgentMode::Agent,
                show_thinking: true,
                max_ctx: cli.max_ctx,
                searxng_url: searxng_url(cli),
                search_key: search_key(cli),
            },
            min_context: cli.min_context as u64,
            is_auto_pull_disabled: cli.is_auto_pull_disabled,
            show_perf: true,
            // Mouse capture is ON by default so wheel scrolling and the
            // scrollbar work out of the box. Use /mouse off for native
            // selection/copy without Shift.
            mouse_capture: true,
            perf: PerfMonitor::new(tegrastats),
            system_stats: SystemStats::default(),
            stream_chars: 0,
            stream_started_at: None,
            tokens_per_sec: 0.0,
            current_task: None,
            last_chat_area: None,
            rate_window_chars: 0,
            rate_window_start: None,
            last_chunk_at: None,
            chat_render: ChatRenderCache::new(),
            editor_terminals: VIM_TERMINALS,
        }
    }

    /// Add one chat turn, dropping the oldest turns beyond
    /// [`MAX_CHAT_HISTORY_MESSAGES`] so the newest question is always kept.
    pub(super) fn remember_chat(&mut self, message: ChatMessage) {
        self.chat_history.push(message);

        let excess = self
            .chat_history
            .len()
            .saturating_sub(MAX_CHAT_HISTORY_MESSAGES);
        self.chat_history.drain(..excess);
    }

    /// True when a tool-confirmation dialog is displayed.
    pub(super) fn is_confirming(&self) -> bool {
        matches!(self.modal, Some(ActiveModal::Confirm(_)))
    }

    /// True when the model picker popup is displayed.
    pub(super) fn is_showing_picker(&self) -> bool {
        matches!(self.modal, Some(ActiveModal::Picker(_)))
    }

    /// Close any active modal overlay without taking any action.
    pub(super) fn close_modal(&mut self) {
        self.modal = None;
    }

    /// Borrow the active picker state, if the model picker is open.
    pub(super) fn picker(&self) -> Option<&PickerState> {
        match &self.modal {
            Some(ActiveModal::Picker(picker)) => Some(picker),
            _ => None,
        }
    }

    /// Mutably borrow the active picker state, if the model picker is open.
    pub(super) fn picker_mut(&mut self) -> Option<&mut PickerState> {
        match &mut self.modal {
            Some(ActiveModal::Picker(picker)) => Some(picker),
            _ => None,
        }
    }

    /// Take the pending confirmation out of the modal, closing it.
    pub(super) fn take_confirmation(&mut self) -> Option<PendingConfirmation> {
        match self.modal.take() {
            Some(ActiveModal::Confirm(pending)) => Some(pending),
            other => {
                self.modal = other;
                None
            }
        }
    }

    /// Open the model picker, replacing any active modal.
    pub(super) fn open_picker(&mut self, models: Vec<String>) {
        self.modal = Some(ActiveModal::Picker(PickerState {
            models,
            selected: 0,
        }));
    }

    /// Open a confirmation dialog, replacing any active modal.
    pub(super) fn open_confirmation(
        &mut self,
        prompt: String,
        response: tokio::sync::oneshot::Sender<bool>,
    ) {
        self.modal = Some(ActiveModal::Confirm(PendingConfirmation {
            prompt,
            response,
        }));
    }

    /// Build a one-line summary of the current model, mode, and server for the banner.
    ///
    /// # Returns
    ///
    /// A formatted string like `"model: qwen2.5-coder:7b · 7.6 GB · agent · server http://localhost:11434"`.
    pub(super) fn model_info_line(&self) -> String {
        let mode = mode_name(self.run_config.mode);

        match &self.model_info {
            Some(info) => format!("{info} · mode: {mode} · server {}", self.server_url),
            None => format!(
                "model: {} · mode: {mode} · server {}",
                self.model.as_deref().unwrap_or("auto"),
                self.server_url
            ),
        }
    }

    /// Return the user's original task prompt for display in the banner.
    ///
    /// # Returns
    ///
    /// The stored prompt string, or `None` if no task has been submitted yet.
    pub(super) fn prompt_line(&self) -> Option<String> {
        self.current_prompt.clone()
    }

    /// Re-query Ollama for the current model's metadata and update `model_info`.
    ///
    /// # Parameters
    ///
    /// - `client`: Ollama client used to fetch the installed model tags.
    pub(super) async fn refresh_model_info(&mut self, client: &OllamaClient) {
        let tags = client.tags().await.unwrap_or_default();
        let selected = self.model.as_deref();

        if let Some(model) = tags
            .iter()
            .find(|model| Some(model.name.as_str()) == selected)
        {
            self.model_info = Some(model_summary(model));
        }
    }

    /// Move any in-progress streaming text from `live` into the persistent log.
    ///
    /// Called before appending a new log line so the streaming output and the
    /// new message appear in the correct order.
    pub(super) fn flush_live(&mut self) {
        if !self.live.is_empty() {
            self.log.push(std::mem::take(&mut self.live));
        }
    }

    /// Return the current spinner glyph based on the animation frame counter.
    ///
    /// # Returns
    ///
    /// A static string slice containing the spinner character for the current frame.
    pub(super) fn spinner(&self) -> &'static str {
        SPINNER
            .get((self.spinner_frame as usize) % SPINNER.len())
            .copied()
            .unwrap_or("")
    }

    /// Spawn a background agent task and reset streaming metrics.
    ///
    /// # Parameters
    ///
    /// - `task`: the user's task description to send to the agent.
    /// - `resume`: prior conversation to continue, when the task is a
    ///   correction of an interrupted run.
    /// - `client`: Ollama client for the background worker.
    /// - `tx`: channel for the worker to send UI events back to the event loop.
    pub(super) fn start_task(
        &mut self,
        task: String,
        resume: Option<Vec<Value>>,
        client: &OllamaClient,
        tx: &mpsc::UnboundedSender<UiEvent>,
    ) {
        self.is_running = true;
        self.status = "running".to_string();
        self.last_action.clear();
        self.scroll_mode = ScrollMode::Follow;
        self.reset_stream_metrics();
        self.remember_chat_turn(Role::User, task.clone());

        let prior = WorkerHistory {
            chat: self.chat_history.clone(),
            resume,
        };
        let prefs = ModelPrefs {
            model: self.model.clone(),
            is_auto_pull_disabled: self.is_auto_pull_disabled,
            min_context: self.min_context,
        };
        let worker = run_agent_worker(
            client.clone(),
            self.run_config.clone(),
            self.model.clone(),
            prefs,
            task,
            prior,
            tx.clone(),
        );
        let done_tx = tx.clone();

        let handle = tokio::spawn(async move {
            let result = worker.await;
            send_done(&done_tx, result);
        });

        self.current_task = Some(handle);
    }

    /// Zero the token-rate counters before a new task streams its reply.
    fn reset_stream_metrics(&mut self) {
        self.stream_chars = 0;
        self.stream_started_at = None;
        self.tokens_per_sec = 0.0;
        self.rate_window_chars = 0;
        self.rate_window_start = None;
        self.last_chunk_at = None;
    }

    /// Add a turn to the chat history in chat mode; other modes keep none.
    fn remember_chat_turn(&mut self, role: Role, content: String) {
        if self.run_config.mode != AgentMode::Chat {
            return;
        }

        self.remember_chat(ChatMessage { role, content });
    }

    /// Process a UI event from a background worker (log, chunk, done, pull, confirm).
    ///
    /// # Parameters
    ///
    /// - `event`: the incoming event to handle.
    /// - `client`: Ollama client, used to start queued tasks after completion.
    /// - `tx`: channel for spawning follow-up tasks.
    pub(super) fn handle_event(
        &mut self,
        event: UiEvent,
        client: &OllamaClient,
        tx: &mpsc::UnboundedSender<UiEvent>,
    ) {
        match event {
            UiEvent::Log(line) => self.on_log(line),
            UiEvent::Chunk(chunk) => self.on_chunk(&chunk),
            UiEvent::History(messages) => self.agent_conversation = Some(messages),
            UiEvent::Done(result) => self.on_done(result, client, tx),
            UiEvent::PullDone { name, result } => self.on_pull_done(&name, result),
            UiEvent::SetupDone(result) => self.on_setup_done(result),
            UiEvent::ConfirmRequest { prompt, response } => {
                self.on_confirm_request(prompt, response)
            }
        }
    }

    /// Append a worker log line after any streamed text, and remember tool
    /// calls (lines starting with 🔧) as the current action.
    fn on_log(&mut self, line: String) {
        self.note_tool_action(&line);
        self.flush_live();

        if !line.is_empty() {
            self.log.push(line);
        }
    }

    /// Show a tool-call line as the current action in the status bar.
    fn note_tool_action(&mut self, line: &str) {
        if line.starts_with("🔧") {
            self.last_action = line.trim().to_string();
        }
    }

    /// Add a streamed chunk to the live output and update the token rate.
    fn on_chunk(&mut self, chunk: &str) {
        let now = Instant::now();
        let chunk_chars = chunk.chars().count() as u64;

        self.stream_started_at = self.stream_started_at.or(Some(now));
        self.stream_chars += chunk_chars;

        let window_start = self.rate_window_start_at(now);
        self.rate_window_chars += chunk_chars;
        self.update_token_rate(now, window_start);

        self.append_live(chunk);
    }

    /// Start of the rate-measurement window that a chunk arriving at `now`
    /// belongs to. A first chunk, or one after a pause longer than
    /// [`STREAM_GAP`], opens a new window.
    fn rate_window_start_at(&mut self, now: Instant) -> Instant {
        let resumed = self
            .last_chunk_at
            .is_some_and(|last| now.duration_since(last) > STREAM_GAP);
        self.last_chunk_at = Some(now);

        match (resumed, self.rate_window_start) {
            (false, Some(window_start)) => window_start,
            (true, _) | (false, None) => self.open_rate_window(now),
        }
    }

    /// Open a new rate-measurement window at `now`.
    fn open_rate_window(&mut self, now: Instant) -> Instant {
        self.rate_window_start = Some(now);
        self.rate_window_chars = 0;

        now
    }

    /// Recompute tokens per second once the window is long enough to measure.
    fn update_token_rate(&mut self, now: Instant, window_start: Instant) {
        let elapsed = now.duration_since(window_start).as_secs_f64();

        if elapsed < TOKEN_RATE_MIN_ELAPSED {
            return;
        }

        self.tokens_per_sec = (self.rate_window_chars as f64 / CHARS_PER_TOKEN) / elapsed;
        self.open_rate_window(now);
    }

    /// Append streamed text to the live output.
    ///
    /// Overflow spills into the disk-backed log instead of being truncated,
    /// so long code is never silently lost.
    fn append_live(&mut self, chunk: &str) {
        self.live.push_str(chunk);

        if self.live.len() <= MAX_LIVE_CHARS {
            return;
        }

        let overflow = std::mem::take(&mut self.live);
        self.log.push(overflow);
        self.log.push("… (output continued)".to_string());
    }

    /// Finish an agent task: save the reply, log the result, and start the
    /// next queued task.
    fn on_done(
        &mut self,
        result: Result<(), String>,
        client: &OllamaClient,
        tx: &mpsc::UnboundedSender<UiEvent>,
    ) {
        self.current_task = None;
        self.agent_conversation = None;

        self.save_assistant_reply();
        self.log_task_result(result);

        self.is_running = false;
        self.last_action.clear();
        self.start_next_queued_task(client, tx);
    }

    /// Move the streamed reply into the log, and into the chat history in chat mode.
    fn save_assistant_reply(&mut self) {
        let reply = std::mem::take(&mut self.live);

        if reply.is_empty() {
            return;
        }

        self.log.push(reply.clone());
        self.remember_chat_turn(Role::Assistant, reply);
    }

    /// Log how an agent task ended.
    fn log_task_result(&mut self, result: Result<(), String>) {
        match result {
            Ok(()) => self
                .log
                .push_lines(vec![String::new(), "✅ Agent finished.".to_string()]),
            Err(error) => self.log.push(format!("⚠️ {error}")),
        }
    }

    /// Start the oldest queued task, or go idle when the queue is empty.
    fn start_next_queued_task(
        &mut self,
        client: &OllamaClient,
        tx: &mpsc::UnboundedSender<UiEvent>,
    ) {
        let Some((next, resume)) = self.task_queue.pop_front() else {
            self.is_running = false;
            self.status = "ready".to_string();
            return;
        };

        self.log.push("▶️  Starting next queued task.".to_string());
        self.start_task(next, resume, client, tx);
    }

    /// Finish a `/pull` and tell the user how to use the model.
    fn on_pull_done(&mut self, name: &str, result: Result<(), String>) {
        self.finish_background_job();

        let lines = match result {
            Ok(()) => vec![
                String::new(),
                format!("✅ Model ready: {name}"),
                format!("Now select it with /model {name}"),
            ],
            Err(error) => vec![format!("❌ Failed to pull model '{name}': {error}")],
        };

        self.log.push_lines(lines);
    }

    /// Finish a `/setup` and log its result.
    fn on_setup_done(&mut self, result: Result<(), String>) {
        self.finish_background_job();

        let lines = match result {
            Ok(()) => vec![String::new(), "✅ Setup finished.".to_string()],
            Err(error) => vec![format!("⚠️ Setup failed: {error}")],
        };

        self.log.push_lines(lines);
    }

    /// Mark a background pull or setup as finished and the app as ready.
    fn finish_background_job(&mut self) {
        self.current_task = None;
        self.is_running = false;
        self.status = "ready".to_string();
        self.last_action.clear();
    }

    /// Show the worker's yes/no question in the confirmation dialog.
    fn on_confirm_request(&mut self, prompt: String, response: tokio::sync::oneshot::Sender<bool>) {
        self.open_confirmation(prompt, response);
        self.status = "confirm".to_string();
    }

    /// Dispatch a keyboard event to the appropriate handler.
    ///
    /// Confirmation and picker modals intercept keys before the main input
    /// handler.
    ///
    /// # Parameters
    ///
    /// - `key`: the crossterm key event to handle.
    /// - `client`: Ollama client for submitting tasks.
    /// - `tx`: channel for background task communication.
    ///
    /// # Returns
    ///
    /// `true` if the TUI should exit, `false` otherwise.
    pub(super) async fn handle_key(
        &mut self,
        key: KeyEvent,
        client: &OllamaClient,
        tx: &mpsc::UnboundedSender<UiEvent>,
    ) -> bool {
        match self.key_target(&key) {
            KeyTarget::Ignored => {}
            KeyTarget::Confirmation => self.handle_confirmation_key(key, client, tx),
            KeyTarget::Picker => self.handle_picker_key(key, client).await,
            KeyTarget::Input => self.handle_input_key(key, client, tx).await,
        }

        self.should_quit
    }

    /// Where `key` goes: nowhere for a release, else the open modal or the input.
    ///
    /// `setup_terminal` pushes `REPORT_EVENT_TYPES`, so terminals that speak
    /// the kitty keyboard protocol deliver a Press *and* a Release event for
    /// every keystroke (plus Repeat while a key is held). Acting on all three
    /// types inserts each character twice, so only press-like events are
    /// dispatched. Terminals without the protocol always report Press.
    fn key_target(&self, key: &KeyEvent) -> KeyTarget {
        let is_press = matches!(key.kind, KeyEventKind::Press | KeyEventKind::Repeat);

        match (is_press, &self.modal) {
            (false, _) => KeyTarget::Ignored,
            (true, Some(ActiveModal::Confirm(_))) => KeyTarget::Confirmation,
            (true, Some(ActiveModal::Picker(_))) => KeyTarget::Picker,
            (true, None) => KeyTarget::Input,
        }
    }

    /// Edit the input line, scroll, or submit, depending on the key.
    async fn handle_input_key(
        &mut self,
        key: KeyEvent,
        client: &OllamaClient,
        tx: &mpsc::UnboundedSender<UiEvent>,
    ) {
        match input_action(&key) {
            InputAction::Cancel => self.cancel_and_clear_input(client, tx),
            InputAction::Newline => self.insert_newline(),
            InputAction::Insert(character) => self.insert_char(character),
            InputAction::Backspace => self.backspace(),
            InputAction::Delete => self.delete(),
            InputAction::Left => self.move_left(),
            InputAction::Right => self.move_right(),
            InputAction::Home => self.move_home(),
            InputAction::End => self.move_end(),
            InputAction::Up => self.move_up(),
            InputAction::Down => self.move_down(),
            InputAction::PageUp => self.scroll_chat_page(-1),
            InputAction::PageDown => self.scroll_chat_page(1),
            InputAction::Complete => self.accept_suggestion(),
            InputAction::Submit => self.submit_input(client, tx).await,
            InputAction::Nothing => {}
        }
    }

    /// Ctrl+C in the input line: cancel any running or queued work, then
    /// clear the input so half-typed text does not sit in the box.
    fn cancel_and_clear_input(
        &mut self,
        client: &OllamaClient,
        tx: &mpsc::UnboundedSender<UiEvent>,
    ) {
        if self.has_work() {
            self.cancel_task(client, tx);
        }

        self.input.clear();
        self.cursor = 0;
    }

    /// Whether a task is running or waiting in the queue.
    fn has_work(&self) -> bool {
        self.is_running || !self.task_queue.is_empty()
    }

    /// Abort the running task and resolve any pending confirmation.
    ///
    /// If tasks are queued behind the cancelled one, the next task starts
    /// immediately so the queue is not stranded.
    ///
    /// # Parameters
    ///
    /// - `client`: Ollama client for starting the next queued task.
    /// - `tx`: channel for the next queued task's events.
    pub(super) fn cancel_task(
        &mut self,
        client: &OllamaClient,
        tx: &mpsc::UnboundedSender<UiEvent>,
    ) {
        self.abort_current_task();
        self.reject_pending_confirmation();
        self.remember_interrupted_task();
        self.forget_unanswered_question();

        self.last_action.clear();
        self.scroll_mode = ScrollMode::Follow;
        self.flush_live();
        self.log.push("⛔ Cancelled.".to_string());

        // Keep the queue moving: cancelling the active task should not strand
        // tasks the user already queued behind it.
        self.start_next_queued_task(client, tx);
    }

    /// Stop the background task, if one is running.
    fn abort_current_task(&mut self) {
        if let Some(handle) = self.current_task.take() {
            handle.abort();
        }
    }

    /// Answer "no" to an open confirmation dialog and close it.
    fn reject_pending_confirmation(&mut self) {
        if let Some(pending) = self.take_confirmation() {
            send_answer(pending, false);
        }
    }

    /// Keep the prompt of a cancelled agent run, so the next message is
    /// treated as a correction of it.
    fn remember_interrupted_task(&mut self) {
        if self.is_running_agent() {
            self.interrupted_task = self.current_prompt.clone();
        }
    }

    /// Whether an agent or plan task is running (chat mode has no tools to resume).
    fn is_running_agent(&self) -> bool {
        self.is_running && self.run_config.mode != AgentMode::Chat
    }

    /// Drop a chat question whose answer was cancelled, so the history does
    /// not end with an unanswered turn.
    fn forget_unanswered_question(&mut self) {
        if self.is_awaiting_chat_reply() {
            self.chat_history.pop();
        }
    }

    /// Whether the chat history ends with a user turn in chat mode.
    fn is_awaiting_chat_reply(&self) -> bool {
        let ends_with_question = self
            .chat_history
            .last()
            .is_some_and(|message| message.role == Role::User);

        self.run_config.mode == AgentMode::Chat && ends_with_question
    }

    /// Handle a mouse event: scroll, scrollbar drag, or click-to-open file.
    ///
    /// Mouse events are ignored while a confirmation dialog is open to prevent
    /// accidental interactions with the chat behind the modal.
    ///
    /// # Parameters
    ///
    /// - `event`: the crossterm mouse event to process.
    pub(super) fn handle_mouse(&mut self, event: MouseEvent) {
        if self.is_confirming() {
            return;
        }

        self.apply_mouse_event(event);
    }

    /// Scroll on the wheel, drag the scrollbar, or open a clicked file.
    fn apply_mouse_event(&mut self, event: MouseEvent) {
        match event.kind {
            MouseEventKind::ScrollUp => self.scroll_chat(-(CHAT_SCROLL_STEP as i32)),
            MouseEventKind::ScrollDown => self.scroll_chat(CHAT_SCROLL_STEP as i32),
            MouseEventKind::Down(MouseButton::Left) => self.click_chat(event.column, event.row),
            MouseEventKind::Drag(MouseButton::Left) => self.drag_scrollbar(event.column, event.row),
            _ => {}
        }
    }

    /// A left click: jump on the scrollbar, or open the file on the clicked line.
    fn click_chat(&mut self, column: u16, row: u16) {
        if !self.handle_scrollbar_click(column, row) {
            self.open_clicked_file(column, row);
        }
    }

    /// A left drag moves the scrollbar; a drag elsewhere does nothing.
    fn drag_scrollbar(&mut self, column: u16, row: u16) {
        self.handle_scrollbar_click(column, row);
    }

    /// Open the file referenced in a clicked chat line in an external editor.
    ///
    /// Maps the mouse click coordinates to a visual line in the chat, extracts
    /// any file path from that line, and opens it in vim.
    ///
    /// # Parameters
    ///
    /// - `column`: x coordinate of the click in terminal columns. It is not
    ///   used: any file path on the clicked line opens, so the user does not
    ///   have to hit the path exactly.
    /// - `row`: y coordinate of the click in terminal rows.
    pub(super) fn open_clicked_file(&mut self, _column: u16, row: u16) {
        let path = self
            .clicked_chat_line(row)
            .and_then(|line| extract_path_from_line(&line, &self.run_config.cwd));

        let Some(path) = path else {
            return;
        };

        self.log
            .push(format!("📂 Opening {} in vim...", path.display()));
        open_in_vim(&path, self.editor_terminals);
    }

    /// The source chat line drawn at terminal row `row`, or `None` when the
    /// row is a border or below the text.
    fn clicked_chat_line(&self, row: u16) -> Option<String> {
        let area = self.last_chat_area?;
        let text_row = is_inside_track(area, row).then(|| (row - area.y - 1) as usize)?;

        let chat_text = self.chat_text();
        let max_text_width = (area.width as usize).saturating_sub(2).max(1);
        let visual_lines = chat_visual_line_indices(&chat_text, max_text_width);
        let scroll = self.chat_scroll_for(area, visual_lines.len().max(1));
        let source_index = *visual_lines.get(scroll + text_row)?;

        chat_text.split('\n').nth(source_index).map(str::to_string)
    }

    /// The first visible wrapped row of a chat `content_height` rows tall in `area`.
    fn chat_scroll_for(&self, area: Rect, content_height: usize) -> usize {
        let visible_height = area.height.saturating_sub(2) as usize;

        match self.scroll_mode {
            ScrollMode::Follow => content_height.saturating_sub(visible_height),
            ScrollMode::Manual => self.chat_scroll_offset.min(content_height),
        }
    }

    /// Whether the transcript has no messages yet, saved or still streaming.
    pub(super) fn transcript_is_empty(&self) -> bool {
        self.log.is_empty() && self.live.is_empty()
    }

    /// Assemble the full visible chat text from the log and any live streaming output.
    ///
    /// # Returns
    ///
    /// The complete chat text ready for rendering or scroll calculations.
    pub(super) fn chat_text(&self) -> String {
        let log_text = self.log.text();
        let parts: Vec<&str> = [log_text.as_str(), self.live.as_str()]
            .into_iter()
            .filter(|part| !part.is_empty())
            .collect();

        parts.join("\n")
    }

    /// Resolve a pending tool confirmation from a single keypress.
    ///
    /// `y`/`Y` confirms, `n`/`N`/`Esc`/`Enter` abort the tool, and unrelated
    /// keys are ignored so the modal stays open. Ctrl+C cancels the whole task.
    fn handle_confirmation_key(
        &mut self,
        key: KeyEvent,
        client: &OllamaClient,
        tx: &mpsc::UnboundedSender<UiEvent>,
    ) {
        match confirmation_key(&key) {
            ConfirmationKey::CancelTask => self.cancel_task(client, tx),
            ConfirmationKey::Answer(answer) => self.answer_confirmation(answer),
            ConfirmationKey::Ignore => {}
        }
    }

    /// Send `answer` to the waiting worker and close the dialog.
    fn answer_confirmation(&mut self, answer: bool) {
        let Some(pending) = self.take_confirmation() else {
            return;
        };

        self.log_aborted_tool(answer);
        send_answer(pending, answer);
    }

    /// Note in the chat that the user refused a tool.
    fn log_aborted_tool(&mut self, answer: bool) {
        if !answer {
            self.log.push("⛔ Tool aborted by user.".to_string());
        }
    }

    /// Handle keyboard navigation within the model picker popup.
    ///
    /// Up/Down arrows move the selection, Enter confirms, Esc/Ctrl+C closes
    /// the picker without changing the model.
    ///
    /// # Parameters
    ///
    /// - `key`: the key event to process.
    /// - `client`: Ollama client for refreshing model info after selection.
    async fn handle_picker_key(&mut self, key: KeyEvent, client: &OllamaClient) {
        match picker_key(&key) {
            PickerKey::Up => self.update_picker(PickerState::select_previous),
            PickerKey::Down => self.update_picker(PickerState::select_next),
            PickerKey::Choose => self.choose_highlighted_model(client).await,
            PickerKey::Close => self.close_modal(),
            PickerKey::Ignore => {}
        }
    }

    /// Apply `change` to the picker, if it is open.
    fn update_picker(&mut self, change: fn(&mut PickerState)) {
        if let Some(picker) = self.picker_mut() {
            change(picker);
        }
    }

    /// Close the picker and switch to the model it highlighted.
    async fn choose_highlighted_model(&mut self, client: &OllamaClient) {
        let chosen = self.picker().and_then(PickerState::highlighted_model);
        self.close_modal();

        if let Some(name) = chosen {
            self.activate_model(client, &name).await;
        }
    }

    /// Submit the current input line as a task or slash command.
    ///
    /// Slash commands are dispatched to `run_slash_command`; regular text
    /// starts a new agent task (or queues it if one is already running).
    ///
    /// # Parameters
    ///
    /// - `client`: Ollama client for network-dependent commands and tasks.
    /// - `tx`: channel for background task events.
    async fn submit_input(&mut self, client: &OllamaClient, tx: &mpsc::UnboundedSender<UiEvent>) {
        let submitted = self.take_input();

        let Some((task, resume)) = self.task_for_submission(submitted) else {
            return;
        };

        self.history.push(task.clone());
        self.scroll_mode = ScrollMode::Follow;
        self.run_submitted(task, resume, client, tx).await;
    }

    /// Take the trimmed input and reset the input line, history position,
    /// and suggestion highlight.
    fn take_input(&mut self) -> String {
        let submitted = self.input.trim().to_string();

        self.input.clear();
        self.cursor = 0;
        self.history_idx = None;
        self.selected = 0;

        submitted
    }

    /// The task to run for `submitted` text, with the conversation to resume.
    ///
    /// After Ctrl+C on an agent run, the next message corrects that run:
    /// it resumes the stored conversation as the next user turn, or without
    /// a stored conversation it is merged into the original prompt. A slash
    /// command leaves the interrupted run waiting for the next message.
    ///
    /// # Returns
    ///
    /// `None` when there is nothing to run (empty input, nothing interrupted).
    fn task_for_submission(&mut self, submitted: String) -> Option<(String, Option<Vec<Value>>)> {
        let Some(interrupted) = self.interrupted_task.take() else {
            return (!submitted.is_empty()).then_some((submitted, None));
        };

        Some(self.task_after_interrupt(interrupted, submitted))
    }

    /// The task for text submitted after an interrupt. A slash command runs
    /// as is and keeps the interrupted task waiting; other text corrects it.
    fn task_after_interrupt(
        &mut self,
        interrupted: String,
        submitted: String,
    ) -> (String, Option<Vec<Value>>) {
        if submitted.starts_with('/') {
            self.interrupted_task = Some(interrupted);
            return (submitted, None);
        }

        self.correct_interrupted_task(&interrupted, submitted)
    }

    /// Resume the interrupted conversation with `correction` as the next user
    /// turn, or merge the correction into the original prompt when no
    /// conversation was stored.
    fn correct_interrupted_task(
        &mut self,
        interrupted: &str,
        correction: String,
    ) -> (String, Option<Vec<Value>>) {
        match self.agent_conversation.take() {
            Some(conversation) => (correction, Some(conversation)),
            None => (corrected_task_prompt(interrupted, &correction), None),
        }
    }

    /// Run a slash command, or log the task and start or queue it.
    async fn run_submitted(
        &mut self,
        task: String,
        resume: Option<Vec<Value>>,
        client: &OllamaClient,
        tx: &mpsc::UnboundedSender<UiEvent>,
    ) {
        if task.starts_with('/') {
            self.run_logged_slash_command(client, &task, tx).await;
            return;
        }

        self.current_prompt = Some(task.clone());
        self.log.push("─".repeat(CHAT_SEPARATOR_LENGTH));
        self.log.push(format!("You: {task}"));
        self.start_or_queue(task, resume, client, tx);
    }

    /// Run a slash command and log its error in the chat.
    ///
    /// Slash commands can hit the network (models, doctor, pull, ...), and a
    /// failure must not tear down the whole TUI.
    async fn run_logged_slash_command(
        &mut self,
        client: &OllamaClient,
        command: &str,
        tx: &mpsc::UnboundedSender<UiEvent>,
    ) {
        if let Err(error) = self.run_slash_command(client, command, tx).await {
            self.log.push(format!("⚠️ {error:#}"));
        }
    }

    /// Start `task` now, or queue it while another task runs.
    fn start_or_queue(
        &mut self,
        task: String,
        resume: Option<Vec<Value>>,
        client: &OllamaClient,
        tx: &mpsc::UnboundedSender<UiEvent>,
    ) {
        if self.is_running {
            self.task_queue.push_back((task, resume));
            self.log
                .push("⏳ Queued — will run after the current task.".to_string());
            return;
        }

        self.start_task(task, resume, client, tx);
    }

    /// Format the current system stats into display lines for the perf panel.
    ///
    /// # Returns
    ///
    /// A `Vec<String>` with GPU, CPU/tokens, and RAM lines for rendering.
    pub(super) fn perf_lines(&self) -> Vec<String> {
        let stats = &self.system_stats;
        let cpu = stats
            .cpu_util_percent
            .map(|util| format!("{util:.0}%"))
            .unwrap_or_else(|| "--".to_string());

        let system = format!(
            "CPU {cpu} · {} cores · ⚡ {:.1} tok/s",
            stats.cpu_cores, self.tokens_per_sec
        );

        vec![gpu_line(stats), system, ram_line(stats)]
    }
}

/// The perf panel's GPU line: each metric the sampler reported, or `GPU n/a`.
fn gpu_line(stats: &SystemStats) -> String {
    let memory = stats
        .gpu_memory_used_mb
        .zip(stats.gpu_memory_total_mb)
        .map(|(used, total)| format!("{used}/{total} MB"));
    let reported = [
        stats.gpu_name.clone(),
        stats.gpu_util_percent.map(|util| format!("{util:.0}%")),
        memory,
        stats.gpu_power_watts.map(|power| format!("{power:.1}W")),
        stats.gpu_temp_c.map(|temp| format!("{temp:.0}°C")),
    ];
    let parts: Vec<String> = reported.into_iter().flatten().collect();

    if parts.is_empty() {
        return "GPU n/a".to_string();
    }

    format!("GPU {}", parts.join(" · "))
}

/// The perf panel's RAM line, in GiB.
fn ram_line(stats: &SystemStats) -> String {
    let gib = |mebibytes: u64| mebibytes as f64 / MIB_PER_GIB;

    format!(
        "RAM {:.1}/{:.1} GB · sh {:.1} · buf {:.1} · cache {:.1} · free {:.1}",
        gib(stats.memory_used_mb),
        gib(stats.memory_total_mb),
        gib(stats.memory_shared_mb),
        gib(stats.memory_buffers_mb),
        gib(stats.memory_cached_mb),
        gib(stats.memory_free_mb)
    )
}

/// The lowercase name of an agent mode, as the header shows it.
fn mode_name(mode: AgentMode) -> &'static str {
    match mode {
        AgentMode::Agent => "agent",
        AgentMode::Plan => "plan",
        AgentMode::Chat => "chat",
    }
}

/// What a key press does in the input line.
fn input_action(key: &KeyEvent) -> InputAction {
    let is_control = key.modifiers.contains(KeyModifiers::CONTROL);

    match (key.code, is_control) {
        (KeyCode::Char('c'), true) => InputAction::Cancel,
        (KeyCode::Char('j'), true) => InputAction::Newline,
        (KeyCode::Char('\r'), _) => InputAction::Nothing,
        (KeyCode::Char(character), _) => InputAction::Insert(character),
        (KeyCode::Backspace, _) => InputAction::Backspace,
        (KeyCode::Delete, _) => InputAction::Delete,
        (KeyCode::Left, _) => InputAction::Left,
        (KeyCode::Right, _) => InputAction::Right,
        (KeyCode::Home, _) => InputAction::Home,
        (KeyCode::End, _) => InputAction::End,
        (KeyCode::Up, _) => InputAction::Up,
        (KeyCode::Down, _) => InputAction::Down,
        (KeyCode::PageUp, _) => InputAction::PageUp,
        (KeyCode::PageDown, _) => InputAction::PageDown,
        (KeyCode::Tab, _) => InputAction::Complete,
        (KeyCode::Enter, _) => enter_action(key.modifiers),
        _ => InputAction::Nothing,
    }
}

/// Enter with Shift, Ctrl or Alt inserts a newline; plain Enter submits.
fn enter_action(modifiers: KeyModifiers) -> InputAction {
    let newline_modifiers = KeyModifiers::SHIFT | KeyModifiers::CONTROL | KeyModifiers::ALT;

    match modifiers.intersects(newline_modifiers) {
        true => InputAction::Newline,
        false => InputAction::Submit,
    }
}

/// What a key press does while the confirmation dialog is open.
fn confirmation_key(key: &KeyEvent) -> ConfirmationKey {
    let is_control = key.modifiers.contains(KeyModifiers::CONTROL);

    match (key.code, is_control) {
        (KeyCode::Char('c'), true) => ConfirmationKey::CancelTask,
        (KeyCode::Char('y' | 'Y'), _) => ConfirmationKey::Answer(true),
        (KeyCode::Char('n' | 'N') | KeyCode::Esc | KeyCode::Enter, _) => {
            ConfirmationKey::Answer(false)
        }
        _ => ConfirmationKey::Ignore,
    }
}

/// What a key press does while the model picker is open.
fn picker_key(key: &KeyEvent) -> PickerKey {
    let is_control = key.modifiers.contains(KeyModifiers::CONTROL);

    match (key.code, is_control) {
        (KeyCode::Up, _) => PickerKey::Up,
        (KeyCode::Down, _) => PickerKey::Down,
        (KeyCode::Enter, _) => PickerKey::Choose,
        (KeyCode::Esc, _) | (KeyCode::Char('c'), true) => PickerKey::Close,
        _ => PickerKey::Ignore,
    }
}

/// Send a yes/no answer to the worker waiting on a confirmation.
///
/// A worker that was cancelled no longer listens, and then the answer is not needed.
fn send_answer(pending: PendingConfirmation, answer: bool) {
    pending.response.send(answer).ok();
}

/// Report a finished agent task to the event loop.
fn send_done(tx: &mpsc::UnboundedSender<UiEvent>, result: anyhow::Result<()>) {
    send_event(
        tx,
        UiEvent::Done(result.map_err(|error| format!("{error:#}"))),
    );
}

/// The prompt for a correction of an interrupted task when no conversation
/// was stored: the correction must carry the original task text with it.
fn corrected_task_prompt(previous: &str, correction: &str) -> String {
    format!(
        "Continue the interrupted task below. Preserve its original goal and existing workspace changes; treat the new user message as corrective steering. Inspect the workspace before acting; do not restart from scratch or repeat rejected work.\n\nOriginal task: {previous}\n\nUser correction / next instruction: {correction}"
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Context;
    use crossterm::event;

    use clap::Parser;

    #[test]
    fn follow_up_after_ctrl_c_keeps_original_task_and_adds_correction() -> anyhow::Result<()> {
        let mut app = test_app();
        app.interrupted_task = Some("write a stable softmax kernel".to_string());

        let submission = app.task_for_submission("use CUDA Oxide, not CUDA C++".to_string());

        let (prompt, resume) = submission.context("a correction must produce a task")?;
        assert!(app.interrupted_task.is_none());
        assert!(prompt.contains("write a stable softmax kernel"));
        assert!(prompt.contains("use CUDA Oxide, not CUDA C++"));
        assert!(prompt.contains("Inspect the workspace"));
        assert_eq!(resume, None);

        Ok(())
    }

    #[test]
    fn slash_command_does_not_consume_interrupted_task() {
        let mut app = test_app();
        app.interrupted_task = Some("original task".to_string());

        let submission = app.task_for_submission("/help".to_string());

        assert_eq!(submission, Some(("/help".to_string(), None)));
        assert_eq!(app.interrupted_task.as_deref(), Some("original task"));
    }

    #[test]
    fn unicode_paste_does_not_panic() {
        let cli = Cli::parse_from(["openbatrangs"]);
        let mut app = App::new(&cli, Arc::new(Mutex::new(None)));
        let text = "NVIDIA — 3% → 😀\n| table |\n```rust\nfn main() {}\n```";
        for ch in text.chars() {
            app.insert_char(ch);
        }
        assert_eq!(app.input, text);
        assert!(app.cursor <= app.input.len());
        app.move_left();
        app.move_right();
        app.backspace();
        app.delete();
    }

    #[test]
    fn paste_multiline_crlf_unicode_keeps_cursor_valid() {
        let cli = Cli::parse_from(["openbatrangs"]);
        let mut app = App::new(&cli, Arc::new(Mutex::new(None)));
        let text = "line1 😀\r\nline2 🦇\r\n```rust\r\nfn main() {}\r\n```";
        app.insert_text(text);
        assert!(app.input.is_char_boundary(app.cursor));
        assert!(app.cursor <= app.input.len());
        app.move_left();
        app.move_right();
        app.backspace();
        app.delete();

        let backend = ratatui::backend::TestBackend::new(80, 24);
        let mut terminal = ratatui::Terminal::new(backend).expect("test backend");
        terminal
            .draw(|frame| super::super::ui::ui(frame, &mut app))
            .expect("draw with pasted multiline input should not panic");
    }

    fn test_app() -> App {
        let cli = Cli::parse_from(["openbatrangs"]);
        App::new(&cli, Arc::new(Mutex::new(None)))
    }

    /// Terminals with the kitty keyboard protocol send Press and Release for
    /// one keystroke; only the Press may reach the input line.
    #[tokio::test]
    async fn release_events_do_not_double_type() {
        let client = crate::ollama::OllamaClient::new("http://localhost:11434").unwrap();
        let (tx, _rx) = mpsc::unbounded_channel();
        let mut app = test_app();

        let mut press = event::KeyEvent::new(KeyCode::Char('a'), KeyModifiers::NONE);
        press.kind = event::KeyEventKind::Press;
        let mut release = press;
        release.kind = event::KeyEventKind::Release;

        app.handle_key(press, &client, &tx).await;
        app.handle_key(release, &client, &tx).await;

        assert_eq!(
            app.input, "a",
            "a single keystroke must insert one character"
        );
    }

    /// A held key repeats; the repeats are real input and must still land.
    #[tokio::test]
    async fn repeat_events_still_reach_the_input_line() {
        let client = crate::ollama::OllamaClient::new("http://localhost:11434").unwrap();
        let (tx, _rx) = mpsc::unbounded_channel();
        let mut app = test_app();

        let mut key = event::KeyEvent::new(KeyCode::Char('a'), KeyModifiers::NONE);
        key.kind = event::KeyEventKind::Repeat;
        app.handle_key(key, &client, &tx).await;

        assert_eq!(app.input, "a");
    }

    #[test]
    fn suggestions_return_slash_commands() {
        let mut app = test_app();
        app.input = "/h".to_string();
        let suggestions = app.suggestions();
        assert!(!suggestions.is_empty());
        assert!(
            suggestions
                .iter()
                .all(|suggestion| suggestion.starts_with("/h"))
        );
    }

    #[test]
    fn mode_suggestions_are_filtered() {
        let mut app = test_app();
        app.input = "/mode p".to_string();
        assert_eq!(app.suggestions(), vec!["/mode plan"]);
        app.input = "/mode ".to_string();
        assert_eq!(
            app.suggestions(),
            vec!["/mode agent", "/mode plan", "/mode chat"]
        );
    }

    #[test]
    fn scroll_chat_never_goes_below_zero() {
        let mut app = test_app();
        app.scroll_chat(-100);
        assert_eq!(app.chat_scroll_offset, 0);
        app.scroll_chat(10);
        assert_eq!(app.chat_scroll_offset, 10);
    }

    #[test]
    fn scroll_chat_keeps_large_offsets_in_usize_space() {
        let mut app = test_app();
        // Offsets larger than u16::MAX are possible with very long wrapped
        // output; they must never be truncated to a small u16 value.
        app.chat_scroll_offset = u16::MAX as usize + 100;
        app.scroll_chat(1);
        assert_eq!(app.chat_scroll_offset, u16::MAX as usize + 101);
        app.scroll_chat(-1);
        assert_eq!(app.chat_scroll_offset, u16::MAX as usize + 100);
    }

    #[test]
    fn tui_defaults_to_agent_mode_with_mouse_capture() {
        let app = test_app();
        assert_eq!(app.run_config.mode, AgentMode::Agent);
        assert!(
            app.mouse_capture,
            "mouse wheel/scrollbar should be on by default"
        );
    }

    #[test]
    fn scroll_chat_page_scrolls_by_viewport_height() {
        let mut app = test_app();
        app.last_chat_area = Some(Rect {
            x: 0,
            y: 0,
            width: 40,
            height: 22,
        });
        app.chat_scroll_offset = 100;
        app.scroll_chat_page(-1);
        assert_eq!(app.chat_scroll_offset, 80);
        app.scroll_chat_page(1);
        assert_eq!(app.chat_scroll_offset, 100);
    }

    #[test]
    fn scrollbar_click_jumps_to_scroll_position() {
        let mut app = test_app();
        for i in 0..100 {
            app.log.push(format!("line {i}"));
        }
        let area = Rect {
            x: 0,
            y: 0,
            width: 40,
            height: 20,
        };
        app.last_chat_area = Some(area);
        let consumed = app.handle_scrollbar_click(area.width - 2, area.y + area.height - 2);
        assert!(consumed);
        assert!(app.chat_scroll_offset > 0);
    }

    #[test]
    fn scrollbar_click_outside_scrollbar_is_not_consumed() {
        let mut app = test_app();
        let area = Rect {
            x: 0,
            y: 0,
            width: 40,
            height: 20,
        };
        app.last_chat_area = Some(area);
        assert!(!app.handle_scrollbar_click(area.width - 5, area.y + 5));
    }

    #[test]
    fn long_streamed_output_spills_to_log_without_loss() {
        let client = crate::ollama::OllamaClient::new("http://localhost:11434").unwrap();
        let (tx, _rx) = mpsc::unbounded_channel();
        let mut app = test_app();
        let part_a = "A".repeat(30_000);
        let part_b = "B".repeat(30_000);
        let part_c = "C".repeat(30_000);
        app.handle_event(UiEvent::Chunk(part_a.clone()), &client, &tx);
        app.handle_event(UiEvent::Chunk(part_b.clone()), &client, &tx);
        app.handle_event(UiEvent::Chunk(part_c.clone()), &client, &tx);
        assert!(app.log.text().contains(&part_a));
        assert!(app.log.text().contains(&part_b));
        assert_eq!(app.live, part_c);
    }

    #[test]
    fn done_event_appends_assistant_to_chat_history() {
        let client = crate::ollama::OllamaClient::new("http://localhost:11434").unwrap();
        let (tx, _rx) = mpsc::unbounded_channel();
        let mut app = test_app();
        app.run_config.mode = AgentMode::Chat;
        app.chat_history.push(ChatMessage {
            role: crate::ollama::Role::User,
            content: "hello".to_string(),
        });
        app.live.push_str("world");
        app.handle_event(UiEvent::Done(Ok(())), &client, &tx);
        assert_eq!(app.chat_history.len(), 2);
        assert_eq!(app.chat_history[1].role, crate::ollama::Role::Assistant);
        assert_eq!(app.chat_history[1].content, "world");
    }

    #[test]
    fn a_full_chat_history_keeps_the_newest_turn() {
        let mut app = test_app();
        let turn = |index: usize| ChatMessage {
            role: crate::ollama::Role::User,
            content: format!("turn {index}"),
        };

        (0..=MAX_CHAT_HISTORY_MESSAGES).for_each(|index| app.remember_chat(turn(index)));

        assert_eq!(app.chat_history.len(), MAX_CHAT_HISTORY_MESSAGES);
        assert_eq!(
            app.chat_history.first().map(|m| m.content.as_str()),
            Some("turn 1")
        );
        assert_eq!(
            app.chat_history.last().map(|m| m.content.clone()),
            Some(format!("turn {MAX_CHAT_HISTORY_MESSAGES}"))
        );
    }

    #[test]
    fn clear_chat_resets_history_and_prompt() {
        let mut app = test_app();
        app.current_prompt = Some("task".to_string());
        app.chat_history.push(ChatMessage {
            role: crate::ollama::Role::User,
            content: "task".to_string(),
        });
        app.scroll_mode = ScrollMode::Manual;
        app.chat_scroll_offset = 42;
        app.clear_chat();
        assert!(app.chat_history.is_empty());
        assert!(app.current_prompt.is_none());
        assert!(app.log.is_empty());
        assert_eq!(app.scroll_mode, ScrollMode::Follow);
        assert_eq!(app.chat_scroll_offset, 0);
    }

    #[test]
    fn cancel_task_removes_pending_chat_user_message() {
        let client = crate::ollama::OllamaClient::new("http://localhost:11434").unwrap();
        let (tx, _rx) = mpsc::unbounded_channel();
        let mut app = test_app();
        app.run_config.mode = AgentMode::Chat;
        app.chat_history.push(ChatMessage {
            role: crate::ollama::Role::User,
            content: "question".to_string(),
        });
        app.cancel_task(&client, &tx);
        assert!(app.chat_history.is_empty());
        assert!(app.log.iter().any(|line| line.contains("Cancelled")));
    }

    #[test]
    fn confirm_request_sets_pending_confirmation() {
        let client = crate::ollama::OllamaClient::new("http://localhost:11434").unwrap();
        let (tx, _rx) = mpsc::unbounded_channel();
        let mut app = test_app();
        let (response_tx, _response_rx) = tokio::sync::oneshot::channel();
        app.handle_event(
            UiEvent::ConfirmRequest {
                prompt: "write file 'a.txt'?".to_string(),
                response: response_tx,
            },
            &client,
            &tx,
        );
        assert!(app.is_confirming());
        assert_eq!(app.status, "confirm");
    }

    #[test]
    fn confirmation_key_yes_sends_true() {
        let client = crate::ollama::OllamaClient::new("http://localhost:11434").unwrap();
        let (tx, _rx) = mpsc::unbounded_channel();
        let mut app = test_app();
        let (response_tx, mut response_rx) = tokio::sync::oneshot::channel();
        app.open_confirmation("write file 'a.txt'?".to_string(), response_tx);
        app.handle_confirmation_key(
            event::KeyEvent::new(KeyCode::Char('y'), KeyModifiers::NONE),
            &client,
            &tx,
        );
        assert!(!app.is_confirming());
        assert!(response_rx.try_recv() == Ok(true));
    }

    #[test]
    fn confirmation_key_no_sends_false() {
        let client = crate::ollama::OllamaClient::new("http://localhost:11434").unwrap();
        let (tx, _rx) = mpsc::unbounded_channel();
        let mut app = test_app();
        let (response_tx, mut response_rx) = tokio::sync::oneshot::channel();
        app.open_confirmation("run command: rm -rf /?".to_string(), response_tx);
        app.handle_confirmation_key(
            event::KeyEvent::new(KeyCode::Char('n'), KeyModifiers::NONE),
            &client,
            &tx,
        );
        assert!(!app.is_confirming());
        assert!(response_rx.try_recv() == Ok(false));
        assert!(app.log.iter().any(|line| line.contains("aborted")));
    }

    #[test]
    fn confirmation_key_esc_aborts() {
        let client = crate::ollama::OllamaClient::new("http://localhost:11434").unwrap();
        let (tx, _rx) = mpsc::unbounded_channel();
        let mut app = test_app();
        let (response_tx, mut response_rx) = tokio::sync::oneshot::channel();
        app.open_confirmation("run command: make?".to_string(), response_tx);
        app.handle_confirmation_key(
            event::KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE),
            &client,
            &tx,
        );
        assert!(response_rx.try_recv() == Ok(false));
    }

    #[test]
    fn confirmation_key_ignores_unrelated_keys() {
        let client = crate::ollama::OllamaClient::new("http://localhost:11434").unwrap();
        let (tx, _rx) = mpsc::unbounded_channel();
        let mut app = test_app();
        let (response_tx, mut response_rx) = tokio::sync::oneshot::channel();
        app.open_confirmation("write file 'a.txt'?".to_string(), response_tx);
        app.handle_confirmation_key(
            event::KeyEvent::new(KeyCode::Char('x'), KeyModifiers::NONE),
            &client,
            &tx,
        );
        assert!(app.is_confirming());
        assert!(response_rx.try_recv().is_err());
    }

    #[test]
    fn cancel_task_resolves_pending_confirmation_with_false() {
        let client = crate::ollama::OllamaClient::new("http://localhost:11434").unwrap();
        let (tx, _rx) = mpsc::unbounded_channel();
        let mut app = test_app();
        let (response_tx, mut response_rx) = tokio::sync::oneshot::channel();
        app.open_confirmation("write file 'a.txt'?".to_string(), response_tx);
        app.cancel_task(&client, &tx);
        assert!(!app.is_confirming());
        assert!(response_rx.try_recv() == Ok(false));
    }

    #[tokio::test]
    async fn cancel_task_starts_next_queued_task() {
        let client = crate::ollama::OllamaClient::new("http://localhost:11434").unwrap();
        let (tx, _rx) = mpsc::unbounded_channel();
        let mut app = test_app();
        app.is_running = true;
        app.task_queue.push_back(("next task".to_string(), None));
        app.cancel_task(&client, &tx);
        assert!(app.task_queue.is_empty());
        assert!(app.is_running, "queued task should start immediately");
        assert!(app.current_task.is_some());
        assert!(
            app.log
                .iter()
                .any(|line| line.contains("Starting next queued task"))
        );
    }

    #[test]
    fn history_snapshot_is_stored_and_cleared_on_done() {
        let client = crate::ollama::OllamaClient::new("http://localhost:11434").unwrap();
        let (tx, _rx) = mpsc::unbounded_channel();
        let mut app = test_app();
        let snapshot = vec![serde_json::json!({"role": "user", "content": "task"})];

        app.handle_event(UiEvent::History(snapshot.clone()), &client, &tx);
        assert_eq!(app.agent_conversation, Some(snapshot));

        app.handle_event(UiEvent::Done(Ok(())), &client, &tx);
        assert_eq!(app.agent_conversation, None);
    }

    #[tokio::test]
    async fn a_correction_after_interrupt_carries_the_conversation() -> anyhow::Result<()> {
        let client = crate::ollama::OllamaClient::new("http://127.0.0.1:1")?;
        let (tx, _rx) = mpsc::unbounded_channel();
        let mut app = test_app();
        app.is_running = true;
        app.interrupted_task = Some("write a softmax kernel".to_string());
        app.current_prompt = Some("write a softmax kernel".to_string());
        app.agent_conversation = Some(vec![serde_json::json!({"role": "user", "content": "task"})]);
        app.input = "use cuda oxide instead".to_string();
        app.submit_input(&client, &tx).await;

        let (queued, resume) = app
            .task_queue
            .pop_front()
            .context("expected the correction to queue behind the running task")?;
        assert_eq!(queued, "use cuda oxide instead");
        assert!(
            resume.is_some(),
            "the interrupted conversation must ride with the correction"
        );
        assert_eq!(app.agent_conversation, None);

        Ok(())
    }

    #[tokio::test]
    async fn a_correction_without_a_snapshot_falls_back_to_the_merged_prompt() -> anyhow::Result<()>
    {
        let client = crate::ollama::OllamaClient::new("http://127.0.0.1:1")?;
        let (tx, _rx) = mpsc::unbounded_channel();
        let mut app = test_app();
        app.is_running = true;
        app.interrupted_task = Some("write a softmax kernel".to_string());
        app.current_prompt = Some("write a softmax kernel".to_string());
        app.input = "use cuda oxide instead".to_string();
        app.submit_input(&client, &tx).await;

        let (queued, resume) = app
            .task_queue
            .pop_front()
            .context("expected the correction to queue behind the running task")?;
        assert!(queued.contains("Original task: write a softmax kernel"));
        assert!(queued.contains("use cuda oxide instead"));
        assert_eq!(resume, None);

        Ok(())
    }

    #[test]
    fn confirmation_ctrl_c_cancels_task_and_resolves_false() {
        let client = crate::ollama::OllamaClient::new("http://localhost:11434").unwrap();
        let (tx, _rx) = mpsc::unbounded_channel();
        let mut app = test_app();
        app.is_running = true;
        let (response_tx, mut response_rx) = tokio::sync::oneshot::channel();
        app.open_confirmation("write file 'a.txt'?".to_string(), response_tx);
        let key = event::KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL);
        app.handle_confirmation_key(key, &client, &tx);
        assert!(!app.is_confirming());
        assert!(response_rx.try_recv() == Ok(false));
        assert!(!app.is_running);
        assert!(app.log.iter().any(|line| line.contains("Cancelled")));
    }

    #[tokio::test]
    async fn ctrl_c_at_idle_clears_input_without_cancel_log() {
        let client = crate::ollama::OllamaClient::new("http://localhost:11434").unwrap();
        let (tx, _rx) = mpsc::unbounded_channel();
        let mut app = test_app();
        app.input = "hello".to_string();
        app.cursor = 5;
        let key = event::KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL);
        let result = app.handle_key(key, &client, &tx).await;
        assert!(!result);
        assert!(app.input.is_empty());
        assert_eq!(app.cursor, 0);
        assert!(
            !app.log.text().contains("Cancelled"),
            "idle Ctrl+C must not log a fake cancel"
        );
    }

    #[tokio::test]
    async fn esc_preserves_input_and_never_quits() {
        let client = crate::ollama::OllamaClient::new("http://localhost:11434").unwrap();
        let (tx, _rx) = mpsc::unbounded_channel();
        let mut app = test_app();
        app.input = "hello".to_string();
        app.cursor = 5;
        let key = event::KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE);
        let result = app.handle_key(key, &client, &tx).await;
        assert!(!result, "Esc must not quit the TUI");
        assert_eq!(app.input, "hello", "Esc must not clear composed input");
        assert_eq!(app.cursor, 5);
        assert!(!app.should_quit);

        // Esc on an already-empty input is also a no-op quit-wise.
        let key = event::KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE);
        let result = app.handle_key(key, &client, &tx).await;
        assert!(!result, "Esc on empty input must not quit the TUI");
        assert!(!app.should_quit);
    }

    #[tokio::test]
    async fn picker_ctrl_c_closes_picker() {
        let client = crate::ollama::OllamaClient::new("http://localhost:11434").unwrap();
        let mut app = test_app();
        app.open_picker(vec!["qwen2.5-coder:7b".to_string()]);
        let key = event::KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL);
        app.handle_picker_key(key, &client).await;
        assert!(!app.is_showing_picker());
        assert!(
            app.model.is_none(),
            "closing picker must not change the model"
        );
    }

    #[test]
    fn accept_suggestion_clamps_out_of_range_selection() {
        let mut app = test_app();
        app.input = "/mode ".to_string();
        app.selected = 99;
        app.accept_suggestion();
        assert_eq!(app.input, "/mode chat");
    }

    #[test]
    fn pull_done_event_marks_ready_and_logs_success() {
        let client = crate::ollama::OllamaClient::new("http://localhost:11434").unwrap();
        let (tx, _rx) = mpsc::unbounded_channel();
        let mut app = test_app();
        app.is_running = true;
        app.handle_event(
            UiEvent::PullDone {
                name: "samantha-mistral:7b".to_string(),
                result: Ok(()),
            },
            &client,
            &tx,
        );
        assert!(!app.is_running);
        assert_eq!(app.status, "ready");
        assert!(app.log.iter().any(|line| line.contains("Model ready")));
    }

    #[test]
    fn pull_done_event_logs_failure_without_crashing() {
        let client = crate::ollama::OllamaClient::new("http://localhost:11434").unwrap();
        let (tx, _rx) = mpsc::unbounded_channel();
        let mut app = test_app();
        app.is_running = true;
        app.handle_event(
            UiEvent::PullDone {
                name: "bad-model".to_string(),
                result: Err("registry not found".to_string()),
            },
            &client,
            &tx,
        );
        assert!(!app.is_running);
        assert_eq!(app.status, "ready");
        assert!(
            app.log
                .iter()
                .any(|line| line.contains("Failed to pull model 'bad-model'"))
        );
    }

    #[test]
    fn setup_done_event_marks_ready_and_logs_success() {
        let client = crate::ollama::OllamaClient::new("http://localhost:11434").unwrap();
        let (tx, _rx) = mpsc::unbounded_channel();
        let mut app = test_app();
        app.is_running = true;
        app.handle_event(UiEvent::SetupDone(Ok(())), &client, &tx);
        assert!(!app.is_running);
        assert_eq!(app.status, "ready");
        assert!(app.log.iter().any(|line| line.contains("Setup finished")));
    }

    #[tokio::test]
    async fn slash_command_network_error_is_logged_not_fatal() -> anyhow::Result<()> {
        // Point at a port that refuses connections so the command fails fast.
        let client = crate::ollama::OllamaClient::new("http://127.0.0.1:1")?;
        let (tx, _rx) = mpsc::unbounded_channel();
        let mut app = test_app();
        app.input = "/model missing-model".to_string();
        app.submit_input(&client, &tx).await;
        assert!(
            app.log.iter().any(|line| line.contains('⚠')),
            "expected an error logged to chat"
        );

        Ok(())
    }

    /// A client for a port that refuses connections, so network calls fail fast.
    fn refused_client() -> anyhow::Result<OllamaClient> {
        OllamaClient::new("http://127.0.0.1:1")
    }

    /// A key press with no modifiers.
    fn press(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::NONE)
    }

    /// A key press with Ctrl held.
    fn ctrl(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::CONTROL)
    }

    /// Feed each key to the app in order.
    async fn type_keys(
        app: &mut App,
        keys: &[KeyEvent],
        client: &OllamaClient,
        tx: &mpsc::UnboundedSender<UiEvent>,
    ) {
        for key in keys {
            app.handle_key(*key, client, tx).await;
        }
    }

    /// A mouse event of `kind` at `column`, `row`.
    fn mouse(kind: MouseEventKind, column: u16, row: u16) -> MouseEvent {
        MouseEvent {
            kind,
            column,
            row,
            modifiers: KeyModifiers::NONE,
        }
    }

    #[test]
    fn input_keys_map_to_their_actions() {
        let shift_enter = KeyEvent::new(KeyCode::Enter, KeyModifiers::SHIFT);

        assert_eq!(input_action(&ctrl(KeyCode::Char('c'))), InputAction::Cancel);
        assert_eq!(
            input_action(&ctrl(KeyCode::Char('j'))),
            InputAction::Newline
        );
        assert_eq!(
            input_action(&ctrl(KeyCode::Char('a'))),
            InputAction::Insert('a')
        );
        assert_eq!(
            input_action(&press(KeyCode::Char('\r'))),
            InputAction::Nothing
        );
        assert_eq!(input_action(&press(KeyCode::Enter)), InputAction::Submit);
        assert_eq!(input_action(&shift_enter), InputAction::Newline);
        assert_eq!(input_action(&press(KeyCode::F(1))), InputAction::Nothing);
    }

    #[tokio::test]
    async fn editing_keys_change_the_input_line() -> anyhow::Result<()> {
        let client = refused_client()?;
        let (tx, _rx) = mpsc::unbounded_channel();
        let mut app = test_app();

        let keys = [
            press(KeyCode::Char('a')),
            press(KeyCode::Char('b')),
            press(KeyCode::Left),
            press(KeyCode::Backspace),
            press(KeyCode::Right),
            press(KeyCode::Home),
            press(KeyCode::Delete),
            press(KeyCode::End),
            ctrl(KeyCode::Char('j')),
            press(KeyCode::Char('c')),
        ];
        type_keys(&mut app, &keys, &client, &tx).await;

        assert_eq!(app.input, "\nc");

        Ok(())
    }

    #[tokio::test]
    async fn navigation_keys_scroll_complete_and_recall_history() -> anyhow::Result<()> {
        let client = refused_client()?;
        let (tx, _rx) = mpsc::unbounded_channel();
        let mut app = test_app();
        app.history = vec!["earlier task".to_string()];

        type_keys(&mut app, &[press(KeyCode::Up)], &client, &tx).await;
        assert_eq!(app.input, "earlier task");

        type_keys(&mut app, &[press(KeyCode::Down)], &client, &tx).await;
        assert!(app.input.is_empty());

        type_keys(&mut app, &[press(KeyCode::PageDown)], &client, &tx).await;
        assert_eq!(app.chat_scroll_offset, CHAT_SCROLL_STEP);

        type_keys(&mut app, &[press(KeyCode::PageUp)], &client, &tx).await;
        assert_eq!(app.chat_scroll_offset, 0);

        app.insert_text("/mod");
        type_keys(&mut app, &[press(KeyCode::Tab)], &client, &tx).await;
        assert_eq!(app.input, "/models");

        Ok(())
    }

    #[tokio::test]
    async fn enter_submits_a_slash_command_and_quit_ends_the_app() -> anyhow::Result<()> {
        let client = refused_client()?;
        let (tx, _rx) = mpsc::unbounded_channel();
        let mut app = test_app();

        app.insert_text("/help");
        let after_help = app.handle_key(press(KeyCode::Enter), &client, &tx).await;

        assert!(!after_help);
        assert!(app.log.iter().any(|line| line.contains("Commands")));

        app.insert_text("/quit");
        let after_quit = app.handle_key(press(KeyCode::Enter), &client, &tx).await;

        assert!(after_quit);

        Ok(())
    }

    #[tokio::test]
    async fn empty_input_submits_nothing() -> anyhow::Result<()> {
        let client = refused_client()?;
        let (tx, _rx) = mpsc::unbounded_channel();
        let mut app = test_app();

        app.handle_key(press(KeyCode::Enter), &client, &tx).await;

        assert!(app.history.is_empty());
        assert!(app.transcript_is_empty());

        Ok(())
    }

    #[tokio::test]
    async fn a_task_runs_in_the_background_and_reports_done() -> anyhow::Result<()> {
        let client = refused_client()?;
        let (tx, mut rx) = mpsc::unbounded_channel();
        let mut app = test_app();

        app.insert_text("write a kernel");
        app.handle_key(press(KeyCode::Enter), &client, &tx).await;

        assert!(app.is_running);
        assert_eq!(app.current_prompt.as_deref(), Some("write a kernel"));

        // The refused connection fails model selection before any log line.
        let event = rx.recv().await;

        assert!(matches!(event, Some(UiEvent::Done(Err(_)))));

        Ok(())
    }

    #[tokio::test]
    async fn ctrl_c_while_running_cancels_and_aborts_the_task() -> anyhow::Result<()> {
        let client = refused_client()?;
        let (tx, _rx) = mpsc::unbounded_channel();
        let mut app = test_app();
        app.is_running = true;
        app.current_prompt = Some("long task".to_string());
        app.current_task = Some(tokio::spawn(std::future::pending::<()>()));
        app.input = "half typed".to_string();

        app.handle_key(ctrl(KeyCode::Char('c')), &client, &tx).await;

        assert!(app.current_task.is_none());
        assert!(!app.is_running);
        assert!(app.input.is_empty());
        assert_eq!(app.interrupted_task.as_deref(), Some("long task"));

        Ok(())
    }

    #[tokio::test]
    async fn keys_reach_the_open_confirmation_dialog() -> anyhow::Result<()> {
        let client = refused_client()?;
        let (tx, _rx) = mpsc::unbounded_channel();
        let mut app = test_app();
        let (response_tx, mut response_rx) = tokio::sync::oneshot::channel();
        app.open_confirmation("write file 'a.txt'?".to_string(), response_tx);

        app.handle_key(press(KeyCode::Char('Y')), &client, &tx)
            .await;

        assert_eq!(response_rx.try_recv(), Ok(true));
        assert!(app.input.is_empty());

        Ok(())
    }

    #[test]
    fn an_answer_without_a_dialog_is_ignored() {
        let mut app = test_app();

        app.answer_confirmation(false);

        assert!(app.transcript_is_empty());
    }

    #[tokio::test]
    async fn the_picker_moves_chooses_and_closes() -> anyhow::Result<()> {
        let client = refused_client()?;
        let (tx, _rx) = mpsc::unbounded_channel();
        let mut app = test_app();
        app.open_picker(vec!["first".to_string(), "second".to_string()]);

        let keys = [
            press(KeyCode::Down),
            press(KeyCode::Down),
            press(KeyCode::Up),
            press(KeyCode::Down),
            press(KeyCode::Char('x')),
        ];
        type_keys(&mut app, &keys, &client, &tx).await;

        assert_eq!(app.picker().map(|picker| picker.selected), Some(1));

        type_keys(&mut app, &[press(KeyCode::Enter)], &client, &tx).await;

        assert!(!app.is_showing_picker());
        assert_eq!(app.model.as_deref(), Some("second"));

        app.open_picker(Vec::new());
        type_keys(&mut app, &[press(KeyCode::Enter)], &client, &tx).await;

        assert!(!app.is_showing_picker());

        app.open_picker(vec!["first".to_string()]);
        type_keys(&mut app, &[press(KeyCode::Esc)], &client, &tx).await;

        assert!(!app.is_showing_picker());
        assert_eq!(app.model.as_deref(), Some("second"));

        Ok(())
    }

    #[test]
    fn picker_changes_need_an_open_picker() {
        let mut app = test_app();

        app.update_picker(PickerState::select_next);

        assert!(app.picker().is_none());
    }

    #[tokio::test]
    async fn refresh_model_info_reads_the_selected_model() -> anyhow::Result<()> {
        let base_url = crate::test_support::spawn_mock_server(|_path| {
            crate::test_support::MockResponse::json(
                "200 OK",
                r#"{"models":[{"name":"other","size":1},{"name":"chosen","size":0}]}"#,
            )
        })
        .await;
        let client = OllamaClient::new(&base_url)?;
        let mut app = test_app();
        app.model = Some("chosen".to_string());
        app.server_url = "http://thor:11434".to_string();

        app.refresh_model_info(&client).await;

        assert_eq!(
            app.model_info_line(),
            "model: chosen · mode: agent · server http://thor:11434"
        );

        Ok(())
    }

    #[test]
    fn the_header_line_names_the_mode() {
        let mut app = test_app();

        app.run_config.mode = AgentMode::Plan;
        assert!(app.model_info_line().contains("mode: plan"));

        app.run_config.mode = AgentMode::Chat;
        assert!(app.model_info_line().contains("mode: chat"));
    }

    #[test]
    fn a_tool_log_line_becomes_the_current_action() -> anyhow::Result<()> {
        let client = refused_client()?;
        let (tx, _rx) = mpsc::unbounded_channel();
        let mut app = test_app();
        app.live = "partial reply".to_string();

        app.handle_event(
            UiEvent::Log("🔧 write file 'a.txt'".to_string()),
            &client,
            &tx,
        );
        app.handle_event(UiEvent::Log(String::new()), &client, &tx);

        assert_eq!(app.last_action, "🔧 write file 'a.txt'");
        assert_eq!(app.log.text(), "partial reply\n🔧 write file 'a.txt'");

        Ok(())
    }

    #[test]
    fn chunks_measure_the_token_rate_per_window() -> anyhow::Result<()> {
        let client = refused_client()?;
        let (tx, _rx) = mpsc::unbounded_channel();
        let mut app = test_app();
        let now = Instant::now();
        let one_second_ago = now
            .checked_sub(std::time::Duration::from_secs(1))
            .context("the clock must be past one second")?;

        app.rate_window_start = Some(one_second_ago);
        app.last_chunk_at = Some(now);
        app.handle_event(UiEvent::Chunk("a".repeat(400)), &client, &tx);

        assert!(app.tokens_per_sec > 0.0);
        assert_eq!(app.rate_window_chars, 0);

        let pause_start = now
            .checked_sub(STREAM_GAP * 2)
            .context("the clock must be past the stream gap")?;
        app.last_chunk_at = Some(pause_start);
        app.handle_event(UiEvent::Chunk("abcd".to_string()), &client, &tx);

        assert_eq!(app.rate_window_chars, 4);

        Ok(())
    }

    #[test]
    fn failed_tasks_and_setups_are_logged() -> anyhow::Result<()> {
        let client = refused_client()?;
        let (tx, _rx) = mpsc::unbounded_channel();
        let mut app = test_app();

        app.handle_event(
            UiEvent::Done(Err("model crashed".to_string())),
            &client,
            &tx,
        );
        app.handle_event(
            UiEvent::SetupDone(Err("no network".to_string())),
            &client,
            &tx,
        );

        assert!(app.log.text().contains("⚠️ model crashed"));
        assert!(app.log.text().contains("⚠️ Setup failed: no network"));

        Ok(())
    }

    #[tokio::test]
    async fn a_task_submitted_while_running_is_queued() -> anyhow::Result<()> {
        let client = refused_client()?;
        let (tx, _rx) = mpsc::unbounded_channel();
        let mut app = test_app();
        app.is_running = true;

        app.insert_text("second task");
        app.handle_key(press(KeyCode::Enter), &client, &tx).await;

        assert_eq!(app.task_queue.len(), 1);
        assert!(app.log.text().contains("Queued"));

        Ok(())
    }

    #[test]
    fn the_mouse_scrolls_drags_and_is_ignored_behind_a_dialog() {
        let mut app = test_app();
        for index in 0..100 {
            app.log.push(format!("line {index}"));
        }
        app.last_chat_area = Some(Rect::new(0, 0, 40, 20));

        app.handle_mouse(mouse(MouseEventKind::ScrollDown, 5, 5));
        assert_eq!(app.chat_scroll_offset, CHAT_SCROLL_STEP);

        app.handle_mouse(mouse(MouseEventKind::ScrollUp, 5, 5));
        assert_eq!(app.chat_scroll_offset, 0);

        app.handle_mouse(mouse(MouseEventKind::Drag(MouseButton::Left), 38, 18));
        assert_eq!(app.scroll_mode, ScrollMode::Follow);

        app.handle_mouse(mouse(MouseEventKind::Down(MouseButton::Left), 38, 1));
        assert_eq!(app.chat_scroll_offset, 0);

        app.handle_mouse(mouse(MouseEventKind::Moved, 5, 5));
        assert_eq!(app.chat_scroll_offset, 0);

        let (response_tx, _response_rx) = tokio::sync::oneshot::channel();
        app.open_confirmation("write?".to_string(), response_tx);
        app.handle_mouse(mouse(MouseEventKind::ScrollDown, 5, 5));
        assert_eq!(app.chat_scroll_offset, 0);
    }

    #[test]
    fn clicking_a_line_with_a_file_path_opens_it() -> anyhow::Result<()> {
        let workspace = crate::test_support::unique_temp_dir("openbatrangs-tui-click");
        std::fs::write(workspace.join("notes.txt"), "notes")?;

        let mut app = test_app();
        app.run_config.cwd = workspace.clone();
        app.editor_terminals = &["true"];
        app.log.push("no path here".to_string());
        app.log.push("wrote notes.txt".to_string());
        app.last_chat_area = Some(Rect::new(0, 0, 40, 10));
        app.scroll_mode = ScrollMode::Follow;

        app.handle_mouse(mouse(MouseEventKind::Down(MouseButton::Left), 5, 1));
        assert!(!app.log.text().contains("Opening"));

        app.handle_mouse(mouse(MouseEventKind::Down(MouseButton::Left), 5, 2));
        assert!(app.log.text().contains("📂 Opening"));

        app.scroll_mode = ScrollMode::Manual;
        app.chat_scroll_offset = 1;
        app.handle_mouse(mouse(MouseEventKind::Down(MouseButton::Left), 5, 0));
        app.handle_mouse(mouse(MouseEventKind::Down(MouseButton::Left), 5, 8));

        std::fs::remove_dir_all(workspace)?;

        Ok(())
    }

    #[test]
    fn a_click_before_the_first_draw_does_nothing() {
        let mut app = test_app();

        app.open_clicked_file(5, 5);

        assert!(app.transcript_is_empty());
    }

    #[test]
    fn perf_lines_list_every_reported_metric() {
        let mut app = test_app();
        app.system_stats = SystemStats {
            gpu_name: Some("Thor".to_string()),
            gpu_util_percent: Some(42.0),
            gpu_memory_used_mb: Some(1024),
            gpu_memory_total_mb: Some(2048),
            gpu_power_watts: Some(15.5),
            gpu_temp_c: Some(50.0),
            cpu_util_percent: Some(12.0),
            cpu_cores: 14,
            memory_used_mb: 2048,
            ..SystemStats::default()
        };

        let lines = app.perf_lines();

        assert_eq!(lines[0], "GPU Thor · 42% · 1024/2048 MB · 15.5W · 50°C");
        assert!(lines[1].starts_with("CPU 12% · 14 cores"));
        assert!(lines[2].starts_with("RAM 2.0/"));

        app.system_stats = SystemStats::default();

        assert_eq!(app.perf_lines()[0], "GPU n/a");
    }
}

/// The header's model line: the name, then each detail the server reported.
///
/// Ollama reports a size, parameter count, quantization and context length. An
/// OpenAI-compatible server often reports none of them, so a missing detail is
/// left out of the line in place of a placeholder.
fn model_summary(model: &OllamaModel) -> String {
    let details = model.details.clone().unwrap_or_default();
    let size =
        (model.size > 0).then(|| format!("{:.1} GB", model.size as f64 / BYTES_PER_GIGABYTE));
    let context = details
        .context_length
        .filter(|length| *length > 0)
        .map(|length| format!("{length} ctx"));

    let reported = [
        size,
        details.parameter_size,
        details.quantization_level,
        context,
    ];
    let name = format!("model: {}", model.name);

    std::iter::once(name)
        .chain(reported.into_iter().flatten())
        .collect::<Vec<_>>()
        .join(" · ")
}

#[cfg(test)]
mod model_summary_tests {
    use super::*;

    #[test]
    fn a_model_with_every_detail_lists_them_all() -> anyhow::Result<()> {
        let model: OllamaModel = serde_json::from_value(serde_json::json!({
            "name": "qwen2.5-coder:7b",
            "size": 4_700_000_000_u64,
            "details": {
                "parameter_size": "7.6B",
                "quantization_level": "Q4_K_M",
                "context_length": 32_768
            }
        }))?;

        let summary = model_summary(&model);

        assert!(summary.starts_with("model: qwen2.5-coder:7b · "));
        assert!(summary.ends_with(" · 7.6B · Q4_K_M · 32768 ctx"));

        Ok(())
    }

    #[test]
    fn a_model_without_details_shows_only_its_name() -> anyhow::Result<()> {
        let model: OllamaModel =
            serde_json::from_value(serde_json::json!({ "name": "Gemma-4-26B-A4B-NVFP4" }))?;

        assert_eq!(model_summary(&model), "model: Gemma-4-26B-A4B-NVFP4");

        Ok(())
    }
}
