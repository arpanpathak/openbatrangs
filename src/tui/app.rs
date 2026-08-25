//! TUI application state and event handling.

use super::chat::ChatRenderCache;
use super::session::SessionLog;
use super::{
    chat_visual_line_indices, open_in_vim, run_agent_worker, strip_ansi, ScrollMode, UiEvent,
};
use crate::cli::{AgentMode, AgentRunConfig, Cli, ModelPrefs};
use crate::constants::models::BYTES_PER_GIGABYTE;
use crate::constants::perf::MIB_PER_GIB;
use crate::constants::tui::{
    CHARS_PER_TOKEN, CHAT_SCROLL_STEP, CHAT_SEPARATOR_LENGTH, MAX_CHAT_HISTORY_MESSAGES,
    MAX_LIVE_CHARS, MAX_LOG_LINES, SPINNER, TOKEN_RATE_MIN_ELAPSED,
};
use crate::ollama::{ChatMessage, OllamaClient};
use crate::perf::{PerfMonitor, SystemStats};
use crossterm::event::{self, KeyCode, KeyModifiers, MouseButton, MouseEventKind};
use ratatui::layout::Rect;
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
    /// FIFO queue of user tasks waiting to run after the current task finishes.
    pub(super) task_queue: VecDeque<String>,
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
    /// Pre-split banner text lines, with ANSI codes stripped for rendering.
    pub(super) banner_lines: Vec<String>,
    /// Last measured chat area rectangle, used for scrollbar and click math.
    pub(super) last_chat_area: Option<Rect>,
    /// Characters accumulated in the current rate-measurement window.
    pub(super) rate_window_chars: u64,
    /// Start time of the current rate-measurement window.
    pub(super) rate_window_start: Option<Instant>,
    /// Cache of the last syntax-highlighted chat rendering result.
    pub(super) chat_render: ChatRenderCache,
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
        let banner = strip_ansi(&crate::banner::banner_text());
        let banner_lines = banner.lines().map(|s| s.to_string()).collect::<Vec<_>>();
        let mut log = SessionLog::new(MAX_LOG_LINES);
        log.push(String::new());
        log.push(
            "Agent mode active. Type a task to start coding, or /help for commands.".to_string(),
        );
        log.push(
            "Use /mode chat for plain conversation, /mode plan for read-only planning.".to_string(),
        );
        Self {
            log,
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
            run_config: AgentRunConfig {
                cwd: cli.cwd.clone(),
                max_steps: cli.max_steps,
                is_read_only: cli.is_read_only,
                should_confirm: true,
                mode: AgentMode::Agent,
                show_thinking: true,
                max_ctx: cli.max_ctx,
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
            banner_lines,
            last_chat_area: None,
            rate_window_chars: 0,
            rate_window_start: None,
            chat_render: ChatRenderCache::new(),
        }
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
        let mode = match self.run_config.mode {
            AgentMode::Agent => "agent",
            AgentMode::Plan => "plan",
            AgentMode::Chat => "chat",
        };
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
        let Ok(tags) = client.tags().await else {
            return;
        };
        let Some(model) = tags.iter().find(|model| {
            self.model
                .as_ref()
                .is_some_and(|selected| selected == &model.name)
        }) else {
            return;
        };
        let size_gb = model.size as f64 / BYTES_PER_GIGABYTE;
        let params = model
            .details
            .as_ref()
            .and_then(|details| details.parameter_size.clone())
            .unwrap_or_else(|| "?".to_string());
        let quant = model
            .details
            .as_ref()
            .and_then(|details| details.quantization_level.clone())
            .unwrap_or_else(|| "?".to_string());
        let context = model
            .details
            .as_ref()
            .and_then(|details| details.context_length)
            .unwrap_or(0);
        self.model_info = Some(format!(
            "model: {} · {size_gb:.1} GB · {params} · {quant} · {context} ctx",
            model.name
        ));
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
    /// - `client`: Ollama client for the background worker.
    /// - `tx`: channel for the worker to send UI events back to the event loop.
    pub(super) fn start_task(
        &mut self,
        task: String,
        client: &OllamaClient,
        tx: &mpsc::UnboundedSender<UiEvent>,
    ) {
        self.is_running = true;
        self.status = "running".to_string();
        self.last_action.clear();
        self.scroll_mode = ScrollMode::Follow;
        self.stream_chars = 0;
        self.stream_started_at = None;
        self.tokens_per_sec = 0.0;
        self.rate_window_chars = 0;
        self.rate_window_start = None;
        if self.run_config.mode == AgentMode::Chat {
            self.chat_history.push(ChatMessage {
                role: crate::ollama::Role::User,
                content: task.clone(),
            });
            self.chat_history.truncate(MAX_CHAT_HISTORY_MESSAGES);
        }
        let chat_history = self.chat_history.clone();
        let tx2 = tx.clone();
        let client2 = client.clone();
        let run_config = self.run_config.clone();
        let model = self.model.clone();
        let prefs = ModelPrefs {
            model: model.clone(),
            is_auto_pull_disabled: self.is_auto_pull_disabled,
            min_context: self.min_context,
        };
        let handle = tokio::spawn(async move {
            let result = run_agent_worker(
                client2,
                run_config,
                model,
                prefs,
                task,
                chat_history,
                tx2.clone(),
            )
            .await;
            let _ = tx2.send(UiEvent::Done(result.map_err(|e| format!("{e:#}"))));
        });
        self.current_task = Some(handle);
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
            UiEvent::Log(msg) => {
                if msg.starts_with("🔧") {
                    self.last_action = msg.trim().to_string();
                }
                self.flush_live();
                if !msg.is_empty() {
                    self.log.push(msg);
                }
            }
            UiEvent::Chunk(msg) => {
                let now = Instant::now();
                if self.stream_started_at.is_none() {
                    self.stream_started_at = Some(now);
                }
                self.stream_chars += msg.chars().count() as u64;
                if self.rate_window_start.is_none() {
                    self.rate_window_start = Some(now);
                }
                self.rate_window_chars += msg.chars().count() as u64;
                if let Some(window_start) = self.rate_window_start {
                    let elapsed = now.duration_since(window_start).as_secs_f64();
                    if elapsed >= TOKEN_RATE_MIN_ELAPSED {
                        self.tokens_per_sec =
                            (self.rate_window_chars as f64 / CHARS_PER_TOKEN) / elapsed;
                        self.rate_window_chars = 0;
                        self.rate_window_start = Some(now);
                    }
                }
                self.live.push_str(&msg);
                if self.live.len() > MAX_LIVE_CHARS {
                    // Spill the overflow into the disk-backed log instead of
                    // truncating, so long code is never silently lost.
                    let overflow = std::mem::take(&mut self.live);
                    self.log.push(overflow);
                    self.log.push("… (output continued)".to_string());
                }
            }
            UiEvent::Done(result) => {
                self.current_task = None;
                let assistant_text = std::mem::take(&mut self.live);
                if !assistant_text.is_empty() {
                    self.log.push(assistant_text.clone());
                    if self.run_config.mode == AgentMode::Chat {
                        self.chat_history.push(ChatMessage {
                            role: crate::ollama::Role::Assistant,
                            content: assistant_text,
                        });
                        self.chat_history.truncate(MAX_CHAT_HISTORY_MESSAGES);
                    }
                }
                match result {
                    Ok(()) => {
                        self.log.push(String::new());
                        self.log.push("✅ Agent finished.".to_string());
                    }
                    Err(error) => {
                        self.log.push(format!("⚠️ {error}"));
                    }
                }
                self.is_running = false;
                self.last_action.clear();
                if let Some(next) = self.task_queue.pop_front() {
                    self.log.push("▶️  Starting next queued task.".to_string());
                    self.start_task(next, client, tx);
                } else {
                    self.status = "ready".to_string();
                }
            }
            UiEvent::PullDone { name, result } => {
                self.current_task = None;
                self.is_running = false;
                self.status = "ready".to_string();
                self.last_action.clear();
                match result {
                    Ok(()) => {
                        self.log.push(String::new());
                        self.log.push(format!("✅ Model ready: {name}"));
                        self.log.push(format!("Now select it with /model {name}"));
                    }
                    Err(error) => {
                        self.log
                            .push(format!("❌ Failed to pull model '{name}': {error}"));
                    }
                }
            }
            UiEvent::SetupDone(result) => {
                self.current_task = None;
                self.is_running = false;
                self.status = "ready".to_string();
                self.last_action.clear();
                match result {
                    Ok(()) => {
                        self.log.push(String::new());
                        self.log.push("✅ Setup finished.".to_string());
                    }
                    Err(error) => {
                        self.log.push(format!("⚠️ Setup failed: {error}"));
                    }
                }
            }
            UiEvent::ConfirmRequest { prompt, response } => {
                self.open_confirmation(prompt, response);
                self.status = "confirm".to_string();
            }
        }
    }

    /// Dispatch a keyboard event to the appropriate handler.
    ///
    /// Confirmation and picker modals intercept keys before the main input
    /// handler. Returns `true` when the app should quit.
    ///
    /// # Parameters
    ///
    /// - `key`: the crossterm key event to handle.
    /// - `client`: Ollama client for submitting tasks.
    /// - `tx`: channel for background task communication.
    ///
    /// # Returns
    ///
    /// `Ok(true)` if the TUI should exit, `Ok(false)` otherwise.
    pub(super) async fn handle_key(
        &mut self,
        key: event::KeyEvent,
        client: &OllamaClient,
        tx: &mpsc::UnboundedSender<UiEvent>,
    ) -> anyhow::Result<bool> {
        if self.is_confirming() {
            self.handle_confirmation_key(key, client, tx);
            return Ok(false);
        }

        if self.is_showing_picker() {
            self.handle_picker_key(key, client).await;
            return Ok(false);
        }

        match key.code {
            KeyCode::Char(character)
                if character == 'c' && key.modifiers.contains(KeyModifiers::CONTROL) =>
            {
                if self.is_running || !self.task_queue.is_empty() {
                    self.cancel_task(client, tx);
                }
                // Always clear the input line: Ctrl+C should not leave
                // half-typed text sitting in the box after a cancel.
                self.input.clear();
                self.cursor = 0;
            }
            KeyCode::Char('j') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                self.insert_newline();
            }
            KeyCode::Char(character) if character != '\r' => self.insert_char(character),
            KeyCode::Backspace => self.backspace(),
            KeyCode::Delete => self.delete(),
            KeyCode::Left => self.move_left(),
            KeyCode::Right => self.move_right(),
            KeyCode::Home => self.cursor = 0,
            KeyCode::End => self.cursor = self.input.len(),
            KeyCode::Up => self.move_up(),
            KeyCode::Down => self.move_down(),
            KeyCode::PageUp => self.scroll_chat_page(-1),
            KeyCode::PageDown => self.scroll_chat_page(1),
            KeyCode::Tab => self.accept_suggestion(),
            KeyCode::Enter => {
                let modifiers = key.modifiers;
                if modifiers.contains(KeyModifiers::SHIFT)
                    || modifiers.contains(KeyModifiers::CONTROL)
                    || modifiers.contains(KeyModifiers::ALT)
                {
                    self.insert_newline();
                } else {
                    return self.submit_input(client, tx).await;
                }
            }
            KeyCode::Esc => {
                // Esc is intentionally a no-op in the main input area: it must
                // neither quit the TUI nor destroy text the user is composing.
                // Picker/confirmation popups handle Esc themselves before this
                // match is reached. Use /quit, /exit, or Ctrl+C at idle to
                // leave the app.
            }
            _ => {}
        }
        Ok(self.should_quit)
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
        if let Some(handle) = self.current_task.take() {
            handle.abort();
        }
        if let Some(pending) = self.take_confirmation() {
            let _ = pending.response.send(false);
        }
        if self.run_config.mode == AgentMode::Chat
            && self
                .chat_history
                .last()
                .is_some_and(|message| message.role == crate::ollama::Role::User)
        {
            self.chat_history.pop();
        }
        self.last_action.clear();
        self.scroll_mode = ScrollMode::Follow;
        self.flush_live();
        self.log.push("⛔ Cancelled.".to_string());
        // Keep the queue moving: cancelling the active task should not strand
        // tasks the user already queued behind it.
        if let Some(next) = self.task_queue.pop_front() {
            self.log.push("▶️  Starting next queued task.".to_string());
            self.start_task(next, client, tx);
        } else {
            self.is_running = false;
            self.status = "ready".to_string();
        }
    }

    /// Handle a mouse event: scroll, scrollbar drag, or click-to-open file.
    ///
    /// Mouse events are ignored while a confirmation dialog is open to prevent
    /// accidental interactions with the chat behind the modal.
    ///
    /// # Parameters
    ///
    /// - `event`: the crossterm mouse event to process.
    pub(super) fn handle_mouse(&mut self, event: crossterm::event::MouseEvent) {
        if self.is_confirming() {
            return;
        }
        match event.kind {
            MouseEventKind::ScrollUp => self.scroll_chat(-(CHAT_SCROLL_STEP as i32)),
            MouseEventKind::ScrollDown => self.scroll_chat(CHAT_SCROLL_STEP as i32),
            MouseEventKind::Down(MouseButton::Left)
                if !self.handle_scrollbar_click(event.column, event.row) =>
            {
                self.open_clicked_file(event.column, event.row);
            }
            MouseEventKind::Drag(MouseButton::Left) => {
                self.handle_scrollbar_click(event.column, event.row);
            }
            _ => {}
        }
    }

    /// Open the file referenced in a clicked chat line in an external editor.
    ///
    /// Maps the mouse click coordinates to a visual line in the chat, extracts
    /// any file path from that line, and opens it in vim.
    ///
    /// # Parameters
    ///
    /// - `column`: x coordinate of the click in terminal columns.
    /// - `row`: y coordinate of the click in terminal rows.
    pub(super) fn open_clicked_file(&mut self, _column: u16, row: u16) {
        let Some(area) = self.last_chat_area else {
            return;
        };
        if row <= area.y || row >= area.y + area.height.saturating_sub(1) {
            return;
        }
        let relative = (row - area.y - 1) as usize;
        let chat_text = self.chat_text();
        let max_text_width = (area.width as usize).saturating_sub(2).max(1);
        let visual_lines = chat_visual_line_indices(&chat_text, max_text_width);
        let content_height = visual_lines.len().max(1);
        let visible_height = area.height.saturating_sub(2) as usize;
        let scroll = match self.scroll_mode {
            ScrollMode::Follow => content_height.saturating_sub(visible_height),
            ScrollMode::Manual => self.chat_scroll_offset.min(content_height),
        };
        let visual_index = scroll + relative;
        let Some(source_index) = visual_lines.get(visual_index).copied() else {
            return;
        };
        let lines: Vec<&str> = chat_text.split('\n').collect();
        let Some(line) = lines.get(source_index) else {
            return;
        };
        // Column is intentionally unused: we open any file path found on the
        // clicked source line rather than requiring the user to precisely hit
        // the path token. Scrollbar clicks are already filtered by handle_mouse.
        if let Some(path) = super::extract_path_from_line(line, &self.run_config.cwd) {
            self.log
                .push(format!("📂 Opening {} in vim...", path.display()));
            open_in_vim(&path);
        }
    }

    /// Assemble the full visible chat text from the log and any live streaming output.
    ///
    /// # Returns
    ///
    /// The complete chat text ready for rendering or scroll calculations.
    pub(super) fn chat_text(&self) -> String {
        let mut chat_text = self.log.text();
        if !self.live.is_empty() {
            if !chat_text.is_empty() {
                chat_text.push('\n');
            }
            chat_text.push_str(&self.live);
        }
        chat_text
    }

    /// Resolve a pending tool confirmation from a single keypress.
    ///
    /// `y`/`Y` confirms, `n`/`N`/`Esc`/`Enter` abort the tool, and unrelated
    /// keys are ignored so the modal stays open. Ctrl+C cancels the whole task.
    fn handle_confirmation_key(
        &mut self,
        key: event::KeyEvent,
        client: &OllamaClient,
        tx: &mpsc::UnboundedSender<UiEvent>,
    ) {
        let Some(pending) = self.take_confirmation() else {
            return;
        };
        if key.code == KeyCode::Char('c') && key.modifiers.contains(KeyModifiers::CONTROL) {
            let _ = pending.response.send(false);
            self.cancel_task(client, tx);
            return;
        }
        let answer = match key.code {
            KeyCode::Char('y') | KeyCode::Char('Y') => Some(true),
            KeyCode::Char('n') | KeyCode::Char('N') | KeyCode::Esc | KeyCode::Enter => Some(false),
            _ => None,
        };
        let Some(answer) = answer else {
            // Unrelated key — put the confirmation back so the modal stays open.
            self.open_confirmation(pending.prompt, pending.response);
            return;
        };
        if !answer {
            self.log.push("⛔ Tool aborted by user.".to_string());
        }
        let _ = pending.response.send(answer);
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
    async fn handle_picker_key(&mut self, key: event::KeyEvent, client: &OllamaClient) {
        let Some(picker) = self.picker_mut() else {
            return;
        };
        match key.code {
            KeyCode::Up => picker.selected = picker.selected.saturating_sub(1),
            KeyCode::Down => {
                picker.selected = (picker.selected + 1).min(picker.models.len().saturating_sub(1));
            }
            KeyCode::Enter => {
                if let Some(name) = picker.models.get(picker.selected).cloned() {
                    self.model = Some(name.clone());
                    self.log.push(format!("✅ Model set to {name}"));
                    self.refresh_model_info(client).await;
                }
                self.close_modal();
            }
            KeyCode::Esc => self.close_modal(),
            KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                self.close_modal();
            }
            _ => {}
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
    ///
    /// # Returns
    ///
    /// `Ok(true)` if the TUI should quit, `Ok(false)` otherwise.
    async fn submit_input(
        &mut self,
        client: &OllamaClient,
        tx: &mpsc::UnboundedSender<UiEvent>,
    ) -> anyhow::Result<bool> {
        let task = self.input.trim().to_string();
        self.input.clear();
        self.cursor = 0;
        self.history_idx = None;
        self.selected = 0;
        if task.is_empty() {
            return Ok(false);
        }
        self.history.push(task.clone());
        self.scroll_mode = ScrollMode::Follow;
        if !task.starts_with('/') {
            self.current_prompt = Some(task.clone());
            self.log.push("─".repeat(CHAT_SEPARATOR_LENGTH));
            self.log.push(format!("You: {task}"));
        }
        if task.starts_with('/') {
            // Slash commands can hit the network (models, doctor, pull, ...).
            // Log failures into the chat instead of tearing down the whole TUI.
            if let Err(error) = self.run_slash_command(client, &task, tx).await {
                self.log.push(format!("⚠️ {error:#}"));
            }
        } else if self.is_running {
            self.task_queue.push_back(task);
            self.log
                .push("⏳ Queued — will run after the current task.".to_string());
        } else {
            self.start_task(task, client, tx);
        }
        Ok(self.should_quit)
    }

    /// Format the current system stats into display lines for the perf panel.
    ///
    /// # Returns
    ///
    /// A `Vec<String>` with GPU, CPU/tokens, and RAM lines for rendering.
    pub(super) fn perf_lines(&self) -> Vec<String> {
        let stats = &self.system_stats;
        let mut gpu_parts: Vec<String> = Vec::new();
        if let Some(name) = &stats.gpu_name {
            gpu_parts.push(name.clone());
        }
        if let Some(util) = stats.gpu_util_percent {
            gpu_parts.push(format!("{util:.0}%"));
        }
        if let (Some(used), Some(total)) = (stats.gpu_memory_used_mb, stats.gpu_memory_total_mb) {
            gpu_parts.push(format!("{used}/{total} MB"));
        }
        if let Some(power) = stats.gpu_power_watts {
            gpu_parts.push(format!("{power:.1}W"));
        }
        if let Some(temp) = stats.gpu_temp_c {
            gpu_parts.push(format!("{temp:.0}°C"));
        }
        let gpu = if gpu_parts.is_empty() {
            "GPU n/a".to_string()
        } else {
            format!("GPU {}", gpu_parts.join(" · "))
        };

        let cpu = stats
            .cpu_util_percent
            .map(|util| format!("{util:.0}%"))
            .unwrap_or_else(|| "--".to_string());
        let memory_used_gb = stats.memory_used_mb as f64 / MIB_PER_GIB;
        let memory_total_gb = stats.memory_total_mb as f64 / MIB_PER_GIB;
        let memory_shared_gb = stats.memory_shared_mb as f64 / MIB_PER_GIB;
        let memory_buffers_gb = stats.memory_buffers_mb as f64 / MIB_PER_GIB;
        let memory_cached_gb = stats.memory_cached_mb as f64 / MIB_PER_GIB;
        let memory_free_gb = stats.memory_free_mb as f64 / MIB_PER_GIB;
        let system = format!(
            "CPU {cpu} · {} cores · ⚡ {:.1} tok/s",
            stats.cpu_cores, self.tokens_per_sec
        );
        let ram = format!(
            "RAM {memory_used_gb:.1}/{memory_total_gb:.1} GB · sh {memory_shared_gb:.1} · buf {memory_buffers_gb:.1} · cache {memory_cached_gb:.1} · free {memory_free_gb:.1}"
        );
        vec![gpu, system, ram]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

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

    #[test]
    fn suggestions_return_slash_commands() {
        let mut app = test_app();
        app.input = "/h".to_string();
        let suggestions = app.suggestions();
        assert!(!suggestions.is_empty());
        assert!(suggestions
            .iter()
            .all(|suggestion| suggestion.starts_with("/h")));
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
        app.task_queue.push_back("next task".to_string());
        app.cancel_task(&client, &tx);
        assert!(app.task_queue.is_empty());
        assert!(app.is_running, "queued task should start immediately");
        assert!(app.current_task.is_some());
        assert!(app
            .log
            .iter()
            .any(|line| line.contains("Starting next queued task")));
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
        let result = app.handle_key(key, &client, &tx).await.unwrap();
        assert!(!result);
        assert!(app.input.is_empty());
        assert_eq!(app.cursor, 0);
        assert!(
            !app.log.iter().any(|line| line.contains("Cancelled")),
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
        let result = app.handle_key(key, &client, &tx).await.unwrap();
        assert!(!result, "Esc must not quit the TUI");
        assert_eq!(app.input, "hello", "Esc must not clear composed input");
        assert_eq!(app.cursor, 5);
        assert!(!app.should_quit);

        // Esc on an already-empty input is also a no-op quit-wise.
        let key = event::KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE);
        let result = app.handle_key(key, &client, &tx).await.unwrap();
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
        assert!(app
            .log
            .iter()
            .any(|line| line.contains("Failed to pull model 'bad-model'")));
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
    async fn slash_command_network_error_is_logged_not_fatal() {
        // Point at a port that refuses connections so the command fails fast.
        let client = crate::ollama::OllamaClient::new("http://127.0.0.1:1").unwrap();
        let (tx, _rx) = mpsc::unbounded_channel();
        let mut app = test_app();
        app.input = "/model missing-model".to_string();
        let result = app.submit_input(&client, &tx).await;
        assert!(result.is_ok(), "TUI must survive slash-command failures");
        assert!(
            app.log.iter().any(|line| line.contains('⚠')),
            "expected an error logged to chat"
        );
    }
}
