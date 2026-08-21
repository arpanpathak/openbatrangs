//! TUI application state and event handling.

use super::{
    open_in_vim, run_agent_worker, split_command, strip_ansi, UiEvent, CHAT_SCROLL_STEP, COMMANDS,
    MAX_LIVE_CHARS, TOKEN_RATE_MIN_ELAPSED,
};
use crate::cli::{AgentMode, AgentRunConfig, Cli, ModelPrefs};
use crate::ollama::OllamaClient;
use crate::perf::{PerfMonitor, SystemStats};
use crossterm::event::{self, KeyCode, KeyModifiers, MouseButton, MouseEventKind};
use ratatui::layout::Rect;
use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Instant;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;

pub(super) struct PickerState {
    pub(super) models: Vec<String>,
    pub(super) selected: usize,
}

pub(super) struct App {
    pub(super) log: Vec<String>,
    pub(super) live: String,
    pub(super) input: String,
    pub(super) cursor: usize,
    history: Vec<String>,
    history_idx: Option<usize>,
    pub(super) selected: usize,
    pub(super) is_running: bool,
    pub(super) status: String,
    pub(super) last_action: String,
    pub(super) auto_scroll: bool,
    pub(super) chat_scroll_offset: usize,
    pub(super) spinner_frame: u64,
    pub(super) task_queue: VecDeque<String>,
    pub(super) picker: Option<PickerState>,
    pub(super) should_quit: bool,
    model: Option<String>,
    server_url: String,
    model_info: Option<String>,
    current_prompt: Option<String>,
    pub(super) run_config: AgentRunConfig,
    min_context: u64,
    is_auto_pull_disabled: bool,
    pub(super) show_perf: bool,
    pub(super) perf: PerfMonitor,
    pub(super) system_stats: SystemStats,
    stream_chars: u64,
    stream_started_at: Option<Instant>,
    pub(super) tokens_per_sec: f64,
    current_task: Option<JoinHandle<()>>,
    pub(super) banner_lines: Vec<String>,
    pub(super) last_chat_area: Option<Rect>,
    rate_window_chars: u64,
    rate_window_start: Option<Instant>,
}

impl App {
    pub(super) fn new(cli: &Cli, tegrastats: Arc<Mutex<Option<String>>>) -> Self {
        let banner = strip_ansi(&crate::banner::banner_text());
        let banner_lines = banner.lines().map(|s| s.to_string()).collect::<Vec<_>>();
        let log = vec![
            String::new(),
            "Type a task, or /help. Enter sends, Shift+Enter adds a new line. /models picks a model."
                .to_string(),
        ];
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
                should_confirm: cli.should_confirm,
                mode: AgentMode::Agent,
                show_thinking: true,
            },
            min_context: cli.min_context as u64,
            is_auto_pull_disabled: cli.is_auto_pull_disabled,
            show_perf: true,
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
        if let Some(arg) = query.strip_prefix("mode ") {
            return ["agent", "plan", "chat"]
                .iter()
                .filter(|option| option.starts_with(arg))
                .map(|option| format!("/mode {option}"))
                .collect();
        }
        if query == "mode" {
            return vec![
                "/mode agent".to_string(),
                "/mode plan".to_string(),
                "/mode chat".to_string(),
            ];
        }
        if let Some(arg) = query.strip_prefix("thinking ") {
            return ["on", "off"]
                .iter()
                .filter(|option| option.starts_with(arg))
                .map(|option| format!("/thinking {option}"))
                .collect();
        }
        if query == "thinking" {
            return vec!["/thinking on".to_string(), "/thinking off".to_string()];
        }
        COMMANDS
            .iter()
            .filter(|command| command.starts_with(query))
            .map(|command| format!("/{command}"))
            .collect()
    }

    pub(super) fn model_info_line(&self) -> String {
        match &self.model_info {
            Some(info) => format!("{info} · server {}", self.server_url),
            None => format!(
                "model: {} · server {}",
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
        let size_gb = model.size as f64 / 1e9;
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
        super::SPINNER
            .get((self.spinner_frame as usize) % super::SPINNER.len())
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
            let result =
                run_agent_worker(client2, run_config, model, prefs, task, tx2.clone()).await;
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
        self.auto_scroll = true;
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
                        // Rough estimate: ~4 characters per token for local models.
                        self.tokens_per_sec = (self.rate_window_chars as f64 / 4.0) / elapsed;
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
                self.flush_live();
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
                self.auto_scroll = true;
                if let Some(next) = self.task_queue.pop_front() {
                    self.log.push("▶️  Starting next queued task.".to_string());
                    self.start_task(next, client, tx);
                } else {
                    self.status = "ready".to_string();
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
                if key.modifiers.contains(KeyModifiers::SHIFT) {
                    self.insert_newline();
                } else {
                    return self.submit_input(client, tx).await;
                }
            }
            KeyCode::Esc => self.should_quit = true,
            _ => {}
        }
        Ok(self.should_quit)
    }

    pub(super) fn cancel_task(&mut self) {
        if let Some(handle) = self.current_task.take() {
            handle.abort();
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
        self.chat_scroll_offset = (self.chat_scroll_offset as i32 + delta).max(0) as usize;
    }

    pub(super) fn handle_mouse(&mut self, event: crossterm::event::MouseEvent) {
        match event.kind {
            MouseEventKind::ScrollUp => self.scroll_chat(-(CHAT_SCROLL_STEP as i32)),
            MouseEventKind::ScrollDown => self.scroll_chat(CHAT_SCROLL_STEP as i32),
            MouseEventKind::Down(MouseButton::Left) => {
                self.open_clicked_file(event.column, event.row);
            }
            _ => {}
        }
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
        let lines: Vec<&str> = chat_text.lines().collect();
        let content_height = lines.len().max(1);
        let visible_height = area.height.saturating_sub(2) as usize;
        let scroll = if self.auto_scroll {
            content_height.saturating_sub(visible_height)
        } else {
            self.chat_scroll_offset.min(content_height)
        };
        let line_index = scroll + relative;
        let Some(line) = lines.get(line_index) else {
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
        let mut chat_text = self.log.join("\n");
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

    fn insert_char(&mut self, character: char) {
        self.input.insert(self.cursor, character);
        self.cursor += 1;
    }

    fn insert_newline(&mut self) {
        self.input.insert(self.cursor, '\n');
        self.cursor += 1;
    }

    fn backspace(&mut self) {
        if self.cursor > 0 {
            self.input.remove(self.cursor - 1);
            self.cursor -= 1;
        }
    }

    fn delete(&mut self) {
        if self.cursor < self.input.len() {
            self.input.remove(self.cursor);
        }
    }

    fn move_left(&mut self) {
        self.cursor = self.cursor.saturating_sub(1);
    }

    fn move_right(&mut self) {
        if self.cursor < self.input.len() {
            self.cursor += 1;
        }
    }

    fn move_up(&mut self) {
        let suggestions = self.suggestions();
        if !suggestions.is_empty() {
            self.selected = self.selected.saturating_sub(1);
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
        if let Some(suggestion) = suggestions.get(self.selected) {
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
            self.log.push("─".repeat(60));
            self.log.push(format!("You: {task}"));
        }
        if task.starts_with('/') {
            self.run_slash_command(client, &task).await?;
        } else if self.is_running {
            self.task_queue.push_back(task);
            self.log
                .push("⏳ Queued — will run after the current task.".to_string());
        } else {
            self.start_task(task, client, tx);
        }
        Ok(self.should_quit)
    }

    async fn run_slash_command(&mut self, client: &OllamaClient, line: &str) -> anyhow::Result<()> {
        let (name, arg) = split_command(line);
        match name {
            "help" | "h" => self.log_help(),
            "exit" | "quit" => self.should_quit = true,
            "models" => self.show_model_picker(client).await?,
            "model" => self.handle_model_command(client, arg).await?,
            "read-only" => self.toggle_read_only(),
            "confirm" => self.toggle_confirm(),
            "steps" => self.handle_steps_command(arg),
            "cwd" => self.handle_cwd_command(arg),
            "doctor" => self.handle_doctor_command(client).await?,
            "setup" => self.handle_setup_command(client).await?,
            "clear" => self.clear_chat(),
            "perf" => self.toggle_perf(),
            "mode" => self.handle_mode_command(arg),
            "thinking" => self.handle_thinking_command(arg),
            _ => self.log_unknown_command(name),
        }
        Ok(())
    }

    fn log_help(&mut self) {
        self.log.push("Commands:".to_string());
        self.log
            .push("  /help, /exit, /quit, /setup, /models".to_string());
        self.log
            .push("  /model <tag>, /read-only, /confirm, /perf".to_string());
        self.log
            .push("  /mode agent|plan|chat, /thinking on|off, /steps <n>, /cwd <path>".to_string());
        self.log
            .push("  /doctor, /clear · Ctrl+C cancel · PgUp/PgDn scroll".to_string());
        self.log
            .push("  Shift+Enter = new line · Enter = send".to_string());
    }

    async fn show_model_picker(&mut self, client: &OllamaClient) -> anyhow::Result<()> {
        let tags = client.tags().await?;
        if tags.is_empty() {
            self.log
                .push("No models installed. Run /setup.".to_string());
        } else {
            self.picker = Some(PickerState {
                models: tags.into_iter().map(|model| model.name).collect(),
                selected: 0,
            });
            self.status = "select model".to_string();
        }
        Ok(())
    }

    async fn handle_model_command(
        &mut self,
        client: &OllamaClient,
        arg: &str,
    ) -> anyhow::Result<()> {
        if arg.is_empty() {
            match &self.model {
                Some(model) => self.log.push(format!("Current model: {model}")),
                None => self
                    .log
                    .push("Auto mode — best model will be selected on first task.".to_string()),
            }
        } else {
            let tags = client.tags().await?;
            if tags.iter().any(|model| model.name == arg) {
                self.model = Some(arg.to_string());
                self.log.push(format!("✅ Model set to {arg}"));
                self.refresh_model_info(client).await;
            } else {
                self.log.push(format!(
                    "❌ Model '{arg}' is not installed. Try /models or /setup."
                ));
            }
        }
        Ok(())
    }

    fn toggle_read_only(&mut self) {
        self.run_config.is_read_only = !self.run_config.is_read_only;
        let state = match self.run_config.is_read_only {
            true => "ON",
            false => "OFF",
        };
        self.log.push(format!("Read-only mode: {state}"));
    }

    fn toggle_confirm(&mut self) {
        self.run_config.should_confirm = !self.run_config.should_confirm;
        let state = match self.run_config.should_confirm {
            true => "ON",
            false => "OFF",
        };
        self.log.push(format!("Confirm mode: {state}"));
    }

    fn handle_mode_command(&mut self, arg: &str) {
        match arg {
            "agent" => {
                self.run_config.mode = AgentMode::Agent;
                self.log
                    .push("Mode: agent (full tools enabled)".to_string());
            }
            "plan" => {
                self.run_config.mode = AgentMode::Plan;
                self.log
                    .push("Mode: plan (read-only, explores and plans)".to_string());
            }
            "chat" => {
                self.run_config.mode = AgentMode::Chat;
                self.log
                    .push("Mode: chat (no tools, conversation + code)".to_string());
            }
            _ => self.log.push("Usage: /mode agent|plan|chat".to_string()),
        }
    }

    fn handle_thinking_command(&mut self, arg: &str) {
        match arg {
            "on" => {
                self.run_config.show_thinking = true;
                self.log.push("Thinking display: ON".to_string());
            }
            "off" => {
                self.run_config.show_thinking = false;
                self.log.push("Thinking display: OFF".to_string());
            }
            _ => {
                self.log.push(format!(
                    "Thinking display: {} · Usage: /thinking on|off",
                    if self.run_config.show_thinking {
                        "ON"
                    } else {
                        "OFF"
                    }
                ));
            }
        }
    }

    fn handle_steps_command(&mut self, arg: &str) {
        match arg.parse::<usize>() {
            Ok(steps) if steps > 0 => {
                self.run_config.max_steps = steps;
                self.log.push(format!("Max steps set to {steps}"));
            }
            _ => self.log.push("Usage: /steps <positive number>".to_string()),
        }
    }

    fn handle_cwd_command(&mut self, arg: &str) {
        match arg.is_empty() {
            true => self
                .log
                .push(format!("Workspace: {}", self.run_config.cwd.display())),
            false => {
                self.run_config.cwd = PathBuf::from(arg);
                self.log.push(format!(
                    "Workspace set to {}",
                    self.run_config.cwd.display()
                ));
            }
        }
    }

    async fn handle_doctor_command(&mut self, client: &OllamaClient) -> anyhow::Result<()> {
        for line in crate::commands::doctor_lines(client, self.min_context).await? {
            self.log.push(line);
        }
        Ok(())
    }

    async fn handle_setup_command(&mut self, client: &OllamaClient) -> anyhow::Result<()> {
        self.log.push("Running setup...".to_string());
        crate::commands::setup(client).await?;
        self.log.push("✅ Setup finished.".to_string());
        Ok(())
    }

    fn clear_chat(&mut self) {
        self.log.clear();
        self.live.clear();
        self.current_prompt = None;
    }

    fn toggle_perf(&mut self) {
        self.show_perf = !self.show_perf;
        self.log.push(format!(
            "Perf panel: {}",
            if self.show_perf { "ON" } else { "OFF" }
        ));
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
        let memory_used_gb = stats.memory_used_mb as f64 / 1024.0;
        let memory_total_gb = stats.memory_total_mb as f64 / 1024.0;
        let memory_shared_gb = stats.memory_shared_mb as f64 / 1024.0;
        let memory_buffers_gb = stats.memory_buffers_mb as f64 / 1024.0;
        let memory_cached_gb = stats.memory_cached_mb as f64 / 1024.0;
        let memory_free_gb = stats.memory_free_mb as f64 / 1024.0;
        let system = format!(
            "CPU {cpu} · {} cores · ⚡ {:.1} tok/s",
            stats.cpu_cores, self.tokens_per_sec
        );
        let ram = format!(
            "RAM {memory_used_gb:.1}/{memory_total_gb:.1} GB · sh {memory_shared_gb:.1} · buf {memory_buffers_gb:.1} · cache {memory_cached_gb:.1} · free {memory_free_gb:.1}"
        );
        vec![gpu, system, ram]
    }

    fn log_unknown_command(&mut self, name: &str) {
        self.log
            .push(format!("Unknown command: /{name}. Try /help"));
    }
}
