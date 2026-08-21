//! TUI application state and event handling.

use super::session::SessionLog;
use super::{
    chat_visual_line_indices, open_in_vim, run_agent_worker, strip_ansi, text_wrapped_height,
    UiEvent,
};
use crate::cli::{AgentMode, AgentRunConfig, Cli, ModelPrefs};
use crate::constants::models::BYTES_PER_GIGABYTE;
use crate::constants::perf::MIB_PER_GIB;
use crate::constants::tui::{
    CHARS_PER_TOKEN, CHAT_SCROLL_STEP, CHAT_SEPARATOR_LENGTH, COMMANDS, LOG_LOAD_CHUNK,
    MAX_CHAT_HISTORY_MESSAGES, MAX_LIVE_CHARS, MAX_LOG_LINES, PREFIXED_COMMANDS, SPINNER,
    TOKEN_RATE_MIN_ELAPSED,
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

pub(super) struct PickerState {
    pub(super) models: Vec<String>,
    pub(super) selected: usize,
}

pub(super) struct App {
    pub(super) log: SessionLog,
    pub(super) live: String,
    pub(super) input: String,
    pub(super) cursor: usize,
    pub(super) history: Vec<String>,
    pub(super) history_idx: Option<usize>,
    pub(super) selected: usize,
    pub(super) is_running: bool,
    pub(super) status: String,
    pub(super) last_action: String,
    pub(super) auto_scroll: bool,
    pub(super) chat_scroll_offset: usize,
    pub(super) spinner_frame: u64,
    pub(super) task_queue: VecDeque<String>,
    pub(super) chat_history: Vec<ChatMessage>,
    pub(super) picker: Option<PickerState>,
    pub(super) should_quit: bool,
    pub(super) model: Option<String>,
    pub(super) server_url: String,
    pub(super) model_info: Option<String>,
    pub(super) current_prompt: Option<String>,
    pub(super) run_config: AgentRunConfig,
    pub(super) min_context: u64,
    pub(super) is_auto_pull_disabled: bool,
    pub(super) show_perf: bool,
    pub(super) mouse_capture: bool,
    pub(super) perf: PerfMonitor,
    pub(super) system_stats: SystemStats,
    pub(super) stream_chars: u64,
    pub(super) stream_started_at: Option<Instant>,
    pub(super) tokens_per_sec: f64,
    pub(super) current_task: Option<JoinHandle<()>>,
    pub(super) banner_lines: Vec<String>,
    pub(super) last_chat_area: Option<Rect>,
    pub(super) rate_window_chars: u64,
    pub(super) rate_window_start: Option<Instant>,
}

impl App {
    pub(super) fn new(cli: &Cli, tegrastats: Arc<Mutex<Option<String>>>) -> Self {
        let banner = strip_ansi(&crate::banner::banner_text());
        let banner_lines = banner.lines().map(|s| s.to_string()).collect::<Vec<_>>();
        let mut log = SessionLog::new(MAX_LOG_LINES);
        log.push(String::new());
        log.push(
            "Type a task, or /help. Enter sends, Shift+Enter adds a new line. /models picks a model."
                .to_string(),
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
            auto_scroll: false,
            chat_scroll_offset: 0,
            spinner_frame: 0,
            task_queue: VecDeque::new(),
            chat_history: Vec::new(),
            picker: None,
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
            },
            min_context: cli.min_context as u64,
            is_auto_pull_disabled: cli.is_auto_pull_disabled,
            show_perf: true,
            mouse_capture: false,
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
        }
    }

    pub(super) fn suggestions(&self) -> Vec<String> {
        if !self.input.starts_with('/') {
            return vec![];
        }
        let query = &self.input[1..];
        for (prefix, options) in PREFIXED_COMMANDS {
            if query == prefix.trim_end() {
                return options
                    .iter()
                    .map(|option| format!("/{prefix}{option}"))
                    .collect();
            }
            if let Some(arg) = query.strip_prefix(prefix) {
                return options
                    .iter()
                    .filter(|option| option.starts_with(arg))
                    .map(|option| format!("/{prefix}{option}"))
                    .collect();
            }
        }
        COMMANDS
            .iter()
            .filter(|command| command.starts_with(query))
            .map(|command| format!("/{command}"))
            .collect()
    }

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

    pub(super) fn prompt_line(&self) -> Option<String> {
        self.current_prompt.clone()
    }

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

    pub(super) fn flush_live(&mut self) {
        if !self.live.is_empty() {
            self.log.push(std::mem::take(&mut self.live));
        }
    }

    pub(super) fn spinner(&self) -> &'static str {
        SPINNER
            .get((self.spinner_frame as usize) % SPINNER.len())
            .copied()
            .unwrap_or("")
    }

    pub(super) fn start_task(
        &mut self,
        task: String,
        client: &OllamaClient,
        tx: &mpsc::UnboundedSender<UiEvent>,
    ) {
        self.is_running = true;
        self.status = "running".to_string();
        self.last_action.clear();
        self.auto_scroll = true;
        self.stream_chars = 0;
        self.stream_started_at = None;
        self.tokens_per_sec = 0.0;
        self.rate_window_chars = 0;
        self.rate_window_start = None;
        if self.run_config.mode == AgentMode::Chat {
            self.chat_history.push(ChatMessage {
                role: "user".to_string(),
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
                    self.live.truncate(MAX_LIVE_CHARS);
                }
            }
            UiEvent::Done(result) => {
                self.current_task = None;
                let assistant_text = std::mem::take(&mut self.live);
                if !assistant_text.is_empty() {
                    self.log.push(assistant_text.clone());
                    if self.run_config.mode == AgentMode::Chat {
                        self.chat_history.push(ChatMessage {
                            role: "assistant".to_string(),
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
        }
    }

    pub(super) async fn handle_key(
        &mut self,
        key: event::KeyEvent,
        client: &OllamaClient,
        tx: &mpsc::UnboundedSender<UiEvent>,
    ) -> anyhow::Result<bool> {
        if self.picker.is_some() {
            self.handle_picker_key(key, client).await;
            return Ok(false);
        }

        match key.code {
            KeyCode::Char(character)
                if character == 'c' && key.modifiers.contains(KeyModifiers::CONTROL) =>
            {
                self.cancel_task();
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
            KeyCode::PageUp => self.scroll_chat(-(CHAT_SCROLL_STEP as i32)),
            KeyCode::PageDown => self.scroll_chat(CHAT_SCROLL_STEP as i32),
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
                if self.input.is_empty() {
                    self.should_quit = true;
                } else {
                    self.input.clear();
                    self.cursor = 0;
                }
            }
            _ => {}
        }
        Ok(self.should_quit)
    }

    pub(super) fn cancel_task(&mut self) {
        if let Some(handle) = self.current_task.take() {
            handle.abort();
        }
        if self.run_config.mode == AgentMode::Chat
            && self
                .chat_history
                .last()
                .is_some_and(|message| message.role == "user")
        {
            self.chat_history.pop();
        }
        self.is_running = false;
        self.status = "ready".to_string();
        self.last_action.clear();
        self.auto_scroll = true;
        self.flush_live();
        self.log.push("⛔ Cancelled.".to_string());
    }

    pub(super) fn scroll_chat(&mut self, delta: i32) {
        self.auto_scroll = false;
        let next = self.chat_scroll_offset as i32 + delta;
        if next <= 0 && delta < 0 && self.log.has_more_history() {
            self.log.load_more(LOG_LOAD_CHUNK);
            self.chat_scroll_offset = 0;
        } else {
            self.chat_scroll_offset = next.max(0) as usize;
        }
    }

    pub(super) fn handle_mouse(&mut self, event: crossterm::event::MouseEvent) {
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

    /// Handle clicks/drags on the chat scrollbar. Returns `true` when consumed.
    fn handle_scrollbar_click(&mut self, column: u16, row: u16) -> bool {
        let Some(area) = self.last_chat_area else {
            return false;
        };
        if column != area.x + area.width.saturating_sub(2) {
            return false;
        }
        if row <= area.y || row >= area.y + area.height.saturating_sub(1) {
            return false;
        }
        let visible_height = area.height.saturating_sub(2).max(1) as usize;
        let max_text_width = (area.width as usize).saturating_sub(2).max(1);
        let content_height = text_wrapped_height(&self.chat_text(), max_text_width);
        let max_scroll = content_height.saturating_sub(visible_height);
        let relative = (row - area.y - 1) as usize;
        let ratio = relative as f64 / visible_height.saturating_sub(1).max(1) as f64;
        let offset = (max_scroll as f64 * ratio).round() as usize;
        self.auto_scroll = offset >= max_scroll;
        self.chat_scroll_offset = offset.min(max_scroll);
        true
    }

    pub(super) fn open_clicked_file(&mut self, column: u16, row: u16) {
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
        let scroll = if self.auto_scroll {
            content_height.saturating_sub(visible_height)
        } else {
            self.chat_scroll_offset.min(content_height)
        };
        let visual_index = scroll + relative;
        let Some(source_index) = visual_lines.get(visual_index).copied() else {
            return;
        };
        let lines: Vec<&str> = chat_text.split('\n').collect();
        let Some(line) = lines.get(source_index) else {
            return;
        };
        let _ = column;
        if let Some(path) = super::extract_path_from_line(line, &self.run_config.cwd) {
            self.log
                .push(format!("📂 Opening {} in vim...", path.display()));
            open_in_vim(&path);
        }
    }

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

    async fn handle_picker_key(&mut self, key: event::KeyEvent, client: &OllamaClient) {
        let Some(picker) = &mut self.picker else {
            return;
        };
        match key.code {
            KeyCode::Up => picker.selected = picker.selected.saturating_sub(1),
            KeyCode::Down => {
                picker.selected = (picker.selected + 1).min(picker.models.len().saturating_sub(1));
            }
            KeyCode::Enter => {
                if let Some(name) = picker.models.get(picker.selected) {
                    self.model = Some(name.clone());
                    self.log.push(format!("✅ Model set to {name}"));
                    self.refresh_model_info(client).await;
                }
                self.picker = None;
            }
            KeyCode::Esc => self.picker = None,
            _ => {}
        }
    }

    /// Keep the byte cursor on a valid UTF-8 boundary and within bounds.
    fn clamp_cursor_to_boundary(&mut self) {
        self.cursor = self.cursor.min(self.input.len());
        while self.cursor > 0 && !self.input.is_char_boundary(self.cursor) {
            self.cursor -= 1;
        }
    }

    fn insert_char(&mut self, character: char) {
        self.clamp_cursor_to_boundary();
        self.input.insert(self.cursor, character);
        self.cursor += character.len_utf8();
    }

    pub(super) fn insert_text(&mut self, text: &str) {
        self.clamp_cursor_to_boundary();
        // Normalize CRLF/CR pastes to plain newlines so multiline paste behaves
        // identically across terminals and never leaves stray control chars.
        let text = text.replace("\r\n", "\n").replace('\r', "\n");
        self.input.insert_str(self.cursor, &text);
        self.cursor += text.len();
    }

    fn insert_newline(&mut self) {
        self.clamp_cursor_to_boundary();
        self.input.insert(self.cursor, '\n');
        self.cursor += 1;
    }

    fn backspace(&mut self) {
        self.clamp_cursor_to_boundary();
        if self.cursor > 0 {
            let mut index = self.cursor;
            while index > 0 && !self.input.is_char_boundary(index - 1) {
                index -= 1;
            }
            self.input.remove(index - 1);
            self.cursor = index - 1;
        }
    }

    fn delete(&mut self) {
        self.clamp_cursor_to_boundary();
        if self.cursor < self.input.len() {
            self.input.remove(self.cursor);
        }
    }

    fn move_left(&mut self) {
        self.clamp_cursor_to_boundary();
        if self.cursor > 0 {
            let mut index = self.cursor;
            while index > 0 && !self.input.is_char_boundary(index - 1) {
                index -= 1;
            }
            self.cursor = index - 1;
        }
    }

    fn move_right(&mut self) {
        self.clamp_cursor_to_boundary();
        if self.cursor < self.input.len() {
            let character = self.input[self.cursor..].chars().next().unwrap_or_default();
            self.cursor += character.len_utf8();
        }
    }

    fn move_up(&mut self) {
        let suggestions = self.suggestions();
        if !suggestions.is_empty() {
            self.selected = self
                .selected
                .min(suggestions.len().saturating_sub(1))
                .saturating_sub(1);
        } else if !self.history.is_empty() {
            let idx = match self.history_idx {
                Some(i) if i > 0 => i - 1,
                Some(_) => 0,
                None => self.history.len().saturating_sub(1),
            };
            self.history_idx = Some(idx);
            self.input = self.history[idx].clone();
            self.cursor = self.input.len();
        }
    }

    fn move_down(&mut self) {
        let suggestions = self.suggestions();
        if !suggestions.is_empty() {
            self.selected = (self.selected + 1).min(suggestions.len().saturating_sub(1));
        } else if let Some(idx) = self.history_idx {
            if idx + 1 < self.history.len() {
                self.history_idx = Some(idx + 1);
                self.input = self.history[idx + 1].clone();
                self.cursor = self.input.len();
            } else {
                self.history_idx = None;
                self.input.clear();
                self.cursor = 0;
            }
        }
    }

    fn accept_suggestion(&mut self) {
        let suggestions = self.suggestions();
        let selected = self.selected.min(suggestions.len().saturating_sub(1));
        if let Some(suggestion) = suggestions.get(selected) {
            self.input = suggestion.clone();
            self.cursor = self.input.len();
        }
    }

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
        self.auto_scroll = true;
        if !task.starts_with('/') {
            self.current_prompt = Some(task.clone());
            self.log.push("─".repeat(CHAT_SEPARATOR_LENGTH));
            self.log.push(format!("You: {task}"));
        }
        if task.starts_with('/') {
            self.run_slash_command(client, &task, tx).await?;
        } else if self.is_running {
            self.task_queue.push_back(task);
            self.log
                .push("⏳ Queued — will run after the current task.".to_string());
        } else {
            self.start_task(task, client, tx);
        }
        Ok(self.should_quit)
    }

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
    fn done_event_appends_assistant_to_chat_history() {
        let client = crate::ollama::OllamaClient::new("http://localhost:11434").unwrap();
        let (tx, _rx) = mpsc::unbounded_channel();
        let mut app = test_app();
        app.run_config.mode = AgentMode::Chat;
        app.chat_history.push(ChatMessage {
            role: "user".to_string(),
            content: "hello".to_string(),
        });
        app.live.push_str("world");
        app.handle_event(UiEvent::Done(Ok(())), &client, &tx);
        assert_eq!(app.chat_history.len(), 2);
        assert_eq!(app.chat_history[1].role, "assistant");
        assert_eq!(app.chat_history[1].content, "world");
    }

    #[test]
    fn clear_chat_resets_history_and_prompt() {
        let mut app = test_app();
        app.current_prompt = Some("task".to_string());
        app.chat_history.push(ChatMessage {
            role: "user".to_string(),
            content: "task".to_string(),
        });
        app.clear_chat();
        assert!(app.chat_history.is_empty());
        assert!(app.current_prompt.is_none());
        assert!(app.log.is_empty());
    }

    #[test]
    fn cancel_task_removes_pending_chat_user_message() {
        let mut app = test_app();
        app.run_config.mode = AgentMode::Chat;
        app.chat_history.push(ChatMessage {
            role: "user".to_string(),
            content: "question".to_string(),
        });
        app.cancel_task();
        assert!(app.chat_history.is_empty());
        assert!(app.log.iter().any(|line| line.contains("Cancelled")));
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
}
