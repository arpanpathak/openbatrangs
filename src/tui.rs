//! Ratatui-based interactive terminal UI.
//!
//! The TUI provides:
//! - a chat scrollback area with wrapping,
//! - a fixed bottom input box,
//! - slash-command suggestions,
//! - a model picker,
//! - a live spinner and current agent action,
//! - background agent workers with a task queue,
//! - a live system/GPU performance panel.

use crate::agent::{AgentConfig, Reporter};
use crate::ollama::OllamaClient;
use crate::perf::{PerfMonitor, SystemStats, TegrastatsGuard};
use crate::{resolve_model, resolve_model_context, AgentRunConfig, Cli, ModelPrefs};
use anyhow::{Context, Result};
use crossterm::event::{self, Event, KeyCode, KeyModifiers};
use crossterm::execute;
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
use futures_util::StreamExt;
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, List, ListItem, Paragraph, Wrap};
use ratatui::Terminal;
use std::collections::VecDeque;
use std::io::stdout;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::sync::mpsc;

/// Slash commands recognized by the TUI.
const COMMANDS: &[&str] = &[
    "help",
    "exit",
    "quit",
    "setup",
    "models",
    "model",
    "read-only",
    "confirm",
    "steps",
    "cwd",
    "doctor",
    "clear",
    "perf",
];

/// Spinner frames shown while the agent is working.
const SPINNER: &[&str] = &["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

/// Maximum visible lines in the multi-line input box.
const MAX_INPUT_LINES: usize = 5;
/// Extra lines for the input box border.
const INPUT_BOX_PADDING: usize = 2;
/// Minimum input box height (border + one line).
const MIN_INPUT_BOX_HEIGHT: usize = 3;
/// Cap for the streaming live line to avoid unbounded memory.
const MAX_LIVE_CHARS: usize = 50_000;
/// Redraw interval in milliseconds (drives the spinner).
const TICK_MILLIS: u64 = 80;
/// Height of the suggestions list when visible.
const SUGGESTIONS_HEIGHT: u16 = 4;
/// Width of the model picker popup as a percentage of the screen.
const MODEL_PICKER_WIDTH_PERCENT: u16 = 60;
/// Height of the model picker popup as a percentage of the screen.
const MODEL_PICKER_HEIGHT_PERCENT: u16 = 40;
/// Height of the live performance panel (border + two content lines).
const PERF_PANEL_HEIGHT: u16 = 4;
/// Minimum elapsed time before the token-rate estimate refreshes.
const TOKEN_RATE_MIN_ELAPSED: f64 = 0.5;
/// Minimum terminal height for showing the performance panel automatically.
const PERF_MIN_TERMINAL_HEIGHT: u16 = 18;

enum UiEvent {
    Log(String),
    Chunk(String),
    Done(Result<(), String>),
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

fn strip_ansi(s: &str) -> String {
    let mut out = String::new();
    let mut chars = s.chars();
    while let Some(c) = chars.next() {
        if c == '\x1b' {
            for n in chars.by_ref() {
                if n.is_ascii_alphabetic() || n == '\\' {
                    break;
                }
            }
        } else {
            out.push(c);
        }
    }
    out
}

struct PickerState {
    models: Vec<String>,
    selected: usize,
}

struct App {
    log: Vec<String>,
    live: String,
    input: String,
    cursor: usize,
    history: Vec<String>,
    history_idx: Option<usize>,
    selected: usize,
    is_running: bool,
    status: String,
    last_action: String,
    auto_scroll: bool,
    spinner_frame: u64,
    task_queue: VecDeque<String>,
    picker: Option<PickerState>,
    should_quit: bool,
    model: Option<String>,
    run_config: AgentRunConfig,
    min_context: u64,
    is_auto_pull_disabled: bool,
    show_perf: bool,
    perf: PerfMonitor,
    system_stats: SystemStats,
    stream_chars: u64,
    stream_started_at: Option<Instant>,
    tokens_per_sec: f64,
}

impl App {
    fn new(cli: &Cli, tegrastats: Arc<Mutex<Option<String>>>) -> Self {
        let banner = strip_ansi(&crate::banner::banner_text());
        let mut log = banner.lines().map(|s| s.to_string()).collect::<Vec<_>>();
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
            spinner_frame: 0,
            task_queue: VecDeque::new(),
            picker: None,
            should_quit: false,
            model: cli.model.clone(),
            run_config: AgentRunConfig {
                cwd: cli.cwd.clone(),
                max_steps: cli.max_steps,
                is_read_only: cli.is_read_only,
                should_confirm: cli.should_confirm,
            },
            min_context: cli.min_context as u64,
            is_auto_pull_disabled: cli.is_auto_pull_disabled,
            show_perf: true,
            perf: PerfMonitor::new(tegrastats),
            system_stats: SystemStats::default(),
            stream_chars: 0,
            stream_started_at: None,
            tokens_per_sec: 0.0,
        }
    }

    fn suggestions(&self) -> Vec<String> {
        if self.input.starts_with('/') {
            let query = &self.input[1..];
            COMMANDS
                .iter()
                .filter(|command| command.starts_with(query))
                .map(|command| format!("/{command}"))
                .collect()
        } else {
            vec![]
        }
    }

    fn flush_live(&mut self) {
        if !self.live.is_empty() {
            self.log.push(std::mem::take(&mut self.live));
        }
    }

    fn spinner(&self) -> &'static str {
        SPINNER
            .get((self.spinner_frame as usize) % SPINNER.len())
            .copied()
            .unwrap_or("")
    }

    fn start_task(
        &mut self,
        task: String,
        client: &OllamaClient,
        tx: &mpsc::UnboundedSender<UiEvent>,
    ) {
        self.log.push(format!("🦇 {task}"));
        self.is_running = true;
        self.status = "running".to_string();
        self.last_action.clear();
        self.auto_scroll = true;
        self.stream_chars = 0;
        self.stream_started_at = None;
        self.tokens_per_sec = 0.0;
        let tx2 = tx.clone();
        let client2 = client.clone();
        let run_config = self.run_config.clone();
        let model = self.model.clone();
        let prefs = ModelPrefs {
            model: model.clone(),
            is_auto_pull_disabled: self.is_auto_pull_disabled,
            min_context: self.min_context,
        };
        tokio::spawn(async move {
            let result =
                run_agent_worker(client2, run_config, model, prefs, task, tx2.clone()).await;
            let _ = tx2.send(UiEvent::Done(result.map_err(|e| format!("{e:#}"))));
        });
    }

    fn handle_event(
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
                if let Some(started_at) = self.stream_started_at {
                    let elapsed = now.duration_since(started_at).as_secs_f64();
                    if elapsed >= TOKEN_RATE_MIN_ELAPSED {
                        // Rough estimate: ~4 characters per token for local models.
                        self.tokens_per_sec = (self.stream_chars as f64 / 4.0) / elapsed;
                    }
                }
                self.live.push_str(&msg);
                if self.live.len() > MAX_LIVE_CHARS {
                    self.live.truncate(MAX_LIVE_CHARS);
                }
            }
            UiEvent::Done(result) => {
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
                if let Some(next) = self.task_queue.pop_front() {
                    self.log.push("▶️  Starting next queued task.".to_string());
                    self.start_task(next, client, tx);
                } else {
                    self.status = "ready".to_string();
                }
            }
        }
    }

    async fn handle_key(
        &mut self,
        key: event::KeyEvent,
        client: &OllamaClient,
        tx: &mpsc::UnboundedSender<UiEvent>,
    ) -> Result<bool> {
        if self.picker.is_some() {
            self.handle_picker_key(key);
            return Ok(false);
        }

        match key.code {
            KeyCode::Char(character) if character != '\r' => self.insert_char(character),
            KeyCode::Backspace => self.backspace(),
            KeyCode::Delete => self.delete(),
            KeyCode::Left => self.move_left(),
            KeyCode::Right => self.move_right(),
            KeyCode::Home => self.cursor = 0,
            KeyCode::End => self.cursor = self.input.len(),
            KeyCode::Up => self.move_up(),
            KeyCode::Down => self.move_down(),
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

    fn handle_picker_key(&mut self, key: event::KeyEvent) {
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
    ) -> Result<bool> {
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

    async fn run_slash_command(&mut self, client: &OllamaClient, line: &str) -> Result<()> {
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
            .push("  /steps <n>, /cwd <path>, /doctor, /clear".to_string());
        self.log
            .push("  Shift+Enter = new line · Enter = send".to_string());
    }

    async fn show_model_picker(&mut self, client: &OllamaClient) -> Result<()> {
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

    async fn handle_model_command(&mut self, client: &OllamaClient, arg: &str) -> Result<()> {
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
        self.log.push(format!(
            "Read-only mode: {}",
            if self.run_config.is_read_only {
                "ON"
            } else {
                "OFF"
            }
        ));
    }

    fn toggle_confirm(&mut self) {
        self.run_config.should_confirm = !self.run_config.should_confirm;
        self.log.push(format!(
            "Confirm mode: {}",
            if self.run_config.should_confirm {
                "ON"
            } else {
                "OFF"
            }
        ));
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
        if arg.is_empty() {
            self.log
                .push(format!("Workspace: {}", self.run_config.cwd.display()));
        } else {
            self.run_config.cwd = PathBuf::from(arg);
            self.log.push(format!(
                "Workspace set to {}",
                self.run_config.cwd.display()
            ));
        }
    }

    async fn handle_doctor_command(&mut self, client: &OllamaClient) -> Result<()> {
        for line in crate::doctor_lines(client, self.min_context).await? {
            self.log.push(line);
        }
        Ok(())
    }

    async fn handle_setup_command(&mut self, client: &OllamaClient) -> Result<()> {
        self.log.push("Running setup...".to_string());
        crate::setup(client).await?;
        self.log.push("✅ Setup finished.".to_string());
        Ok(())
    }

    fn clear_chat(&mut self) {
        self.log.clear();
        self.live.clear();
    }

    fn toggle_perf(&mut self) {
        self.show_perf = !self.show_perf;
        self.log.push(format!(
            "Perf panel: {}",
            if self.show_perf { "ON" } else { "OFF" }
        ));
    }

    fn perf_lines(&self) -> Vec<String> {
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
        let system = format!(
            "CPU {cpu} · RAM {memory_used_gb:.1}/{memory_total_gb:.1} GB · {} cores · ⚡ {:.1} tok/s",
            stats.cpu_cores, self.tokens_per_sec
        );
        vec![gpu, system]
    }

    fn log_unknown_command(&mut self, name: &str) {
        self.log
            .push(format!("Unknown command: /{name}. Try /help"));
    }
}

fn split_command(line: &str) -> (&str, &str) {
    let mut parts = line[1..].splitn(2, char::is_whitespace);
    let name = parts.next().unwrap_or_default();
    let arg = parts.next().unwrap_or_default().trim();
    (name, arg)
}

async fn run_agent_worker(
    client: OllamaClient,
    config: AgentRunConfig,
    model_slot: Option<String>,
    prefs: ModelPrefs,
    task: String,
    tx: mpsc::UnboundedSender<UiEvent>,
) -> Result<()> {
    let mem_budget = crate::calculate_memory_budget();
    let selected = resolve_model(&client, &model_slot, &prefs, mem_budget).await?;
    let model_context = resolve_model_context(&client, &selected.name).await?;
    let agent_config = AgentConfig {
        cwd: config.cwd,
        max_steps: config.max_steps,
        is_read_only: config.is_read_only,
        should_confirm: config.should_confirm,
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

fn ui(f: &mut ratatui::Frame, app: &mut App) {
    let chunks = layout_chunks(f.area(), app);
    render_chat_area(f, app, chunks[0]);
    render_status_line(f, app, chunks[1]);
    if perf_visible(app, f.area().height) {
        render_perf_panel(f, app, chunks[2]);
    }
    render_suggestions(f, app, chunks[3]);
    render_input_box(f, app, chunks[4]);
    render_cursor(f, app, chunks[4]);
    render_model_picker(f, app);
}

fn perf_visible(app: &App, area_height: u16) -> bool {
    app.show_perf && area_height >= PERF_MIN_TERMINAL_HEIGHT
}

fn layout_chunks(area: Rect, app: &App) -> Vec<Rect> {
    Layout::default()
        .direction(Direction::Vertical)
        .margin(1)
        .constraints([
            Constraint::Min(3),
            Constraint::Length(1),
            Constraint::Length(if perf_visible(app, area.height) {
                PERF_PANEL_HEIGHT
            } else {
                0
            }),
            Constraint::Length(if app.suggestions().is_empty() {
                0
            } else {
                SUGGESTIONS_HEIGHT
            }),
            Constraint::Length(input_height(app) as u16),
        ])
        .split(area)
        .to_vec()
}

fn render_chat_area(f: &mut ratatui::Frame, app: &App, area: Rect) {
    let mut chat_text = app.log.join("\n");
    if !app.live.is_empty() {
        if !chat_text.is_empty() {
            chat_text.push('\n');
        }
        chat_text.push_str(&app.live);
    }
    let scroll_y = chat_scroll(app, &chat_text, area.height);
    let chat = Paragraph::new(chat_text)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title(" openBatarangs "),
        )
        .wrap(Wrap { trim: false })
        .scroll((scroll_y, 0));
    f.render_widget(chat, area);
}

/// Compute a safe vertical scroll offset that never overflows `u16`.
fn chat_scroll(app: &App, chat_text: &str, area_height: u16) -> u16 {
    if !app.auto_scroll {
        return 0;
    }
    let content_height = chat_text.lines().count().max(1) as u16;
    let visible_height = area_height.saturating_sub(2);
    content_height.saturating_sub(visible_height)
}

fn render_status_line(f: &mut ratatui::Frame, app: &App, area: Rect) {
    let status_line = if app.is_running {
        let spin = app.spinner();
        if app.last_action.is_empty() {
            format!("{spin} {} — thinking...", app.status)
        } else {
            format!("{spin} {} — {}", app.status, app.last_action)
        }
    } else if app.picker.is_some() {
        "↑↓ select · Enter confirm · Esc cancel".to_string()
    } else if !app.task_queue.is_empty() {
        format!("ready — {} queued", app.task_queue.len())
    } else {
        app.status.clone()
    };
    let status_style = if app.is_running {
        Style::default().fg(Color::Yellow)
    } else {
        Style::default().fg(Color::Green)
    };
    f.render_widget(Paragraph::new(status_line).style(status_style), area);
}

fn render_perf_panel(f: &mut ratatui::Frame, app: &App, area: Rect) {
    let content = app.perf_lines().join("\n");
    let panel = Paragraph::new(content)
        .block(Block::default().borders(Borders::ALL).title(" ⚡ perf "))
        .style(Style::default().fg(Color::Cyan));
    f.render_widget(panel, area);
}

fn render_suggestions(f: &mut ratatui::Frame, app: &App, area: Rect) {
    if area.height == 0 {
        return;
    }
    let suggestions = app.suggestions();
    let suggestion_items: Vec<ListItem> = suggestions
        .iter()
        .enumerate()
        .map(|(index, suggestion)| {
            let style = if index == app.selected {
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default()
            };
            ListItem::new(Line::from(Span::styled(suggestion.clone(), style)))
        })
        .collect();
    f.render_widget(
        List::new(suggestion_items).block(
            Block::default()
                .borders(Borders::ALL)
                .title(" suggestions "),
        ),
        area,
    );
}

fn render_input_box(f: &mut ratatui::Frame, app: &App, area: Rect) {
    let input_paragraph = Paragraph::new(app.input.clone())
        .block(Block::default().borders(Borders::ALL).title(" input "))
        .wrap(Wrap { trim: false });
    f.render_widget(input_paragraph, area);
}

fn render_cursor(f: &mut ratatui::Frame, app: &App, area: Rect) {
    let Some(prefix) = app.input.get(..app.cursor) else {
        return;
    };
    let line_idx = prefix.matches('\n').count();
    let column = prefix
        .rsplit('\n')
        .next()
        .map(|line| line.chars().count())
        .unwrap_or(0);
    let x = area.x + 1 + column as u16;
    let y = area.y + 1 + line_idx as u16;
    f.set_cursor_position((x, y.min(area.y + area.height.saturating_sub(1))));
}

fn render_model_picker(f: &mut ratatui::Frame, app: &App) {
    let Some(picker) = &app.picker else {
        return;
    };
    let area = centered_rect(
        MODEL_PICKER_WIDTH_PERCENT,
        MODEL_PICKER_HEIGHT_PERCENT,
        f.area(),
    );
    let items: Vec<ListItem> = picker
        .models
        .iter()
        .enumerate()
        .map(|(index, model)| {
            let style = if index == picker.selected {
                Style::default()
                    .fg(Color::Black)
                    .bg(Color::Cyan)
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default()
            };
            ListItem::new(Line::from(Span::styled(model.clone(), style)))
        })
        .collect();
    f.render_widget(
        List::new(items)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title(" Select model "),
            )
            .highlight_style(
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            ),
        area,
    );
}

fn centered_rect(
    percent_x: u16,
    percent_y: u16,
    area: ratatui::layout::Rect,
) -> ratatui::layout::Rect {
    let popup = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage((100 - percent_y) / 2),
            Constraint::Percentage(percent_y),
            Constraint::Percentage((100 - percent_y) / 2),
        ])
        .split(area);
    Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage((100 - percent_x) / 2),
            Constraint::Percentage(percent_x),
            Constraint::Percentage((100 - percent_x) / 2),
        ])
        .split(popup[1])[1]
}

fn input_height(app: &App) -> usize {
    let lines = app.input.split('\n').count();
    (lines.min(MAX_INPUT_LINES) + INPUT_BOX_PADDING).max(MIN_INPUT_BOX_HEIGHT)
}

pub async fn run(cli: &Cli, client: &OllamaClient) -> Result<()> {
    let mut terminal = setup_terminal()?;
    let result = run_loop(&mut terminal, cli, client).await;
    teardown_terminal(&mut terminal)?;
    result
}

fn setup_terminal() -> Result<Terminal<CrosstermBackend<std::io::Stdout>>> {
    enable_raw_mode()?;
    let mut stdout = stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend).context("failed to create terminal")?;
    terminal.hide_cursor()?;
    Ok(terminal)
}

fn teardown_terminal(terminal: &mut Terminal<CrosstermBackend<std::io::Stdout>>) -> Result<()> {
    terminal.show_cursor()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    disable_raw_mode()?;
    println!("👋 Bye!");
    Ok(())
}

async fn run_loop(
    terminal: &mut Terminal<CrosstermBackend<std::io::Stdout>>,
    cli: &Cli,
    client: &OllamaClient,
) -> Result<()> {
    let tegrastats_shared = Arc::new(Mutex::new(None));
    let _tegrastats_guard = TegrastatsGuard::start(tegrastats_shared.clone());
    let mut app = App::new(cli, tegrastats_shared);
    let (tx, mut rx) = mpsc::unbounded_channel::<UiEvent>();
    let mut events = event::EventStream::new();

    loop {
        terminal.draw(|f| ui(f, &mut app))?;

        if app.should_quit {
            break;
        }

        tokio::select! {
            maybe = rx.recv() => {
                if let Some(event) = maybe {
                    app.handle_event(event, client, &tx);
                }
            }
            maybe = events.next() => {
                match maybe {
                    Some(Ok(Event::Key(key))) => {
                        if app.handle_key(key, client, &tx).await? {
                            break;
                        }
                    }
                    Some(Ok(Event::Resize(_, _))) => {}
                    Some(Ok(_)) => {}
                    Some(Err(error)) => return Err(error.into()),
                    None => break,
                }
            }
            _ = tokio::time::sleep(Duration::from_millis(TICK_MILLIS)) => {
                app.spinner_frame += 1;
                if let Some(stats) = app.perf.sample_if_due(Instant::now()) {
                    app.system_stats = stats;
                }
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stripped_banner_has_content_for_tui() {
        let banner = strip_ansi(&crate::banner::banner_text());
        assert!(!banner.contains('\x1b'));
        assert!(banner.lines().count() > 5);
        assert!(banner.contains('█'));
    }

    #[test]
    fn ui_renders_banner_content() {
        use clap::Parser;
        use ratatui::backend::TestBackend;

        let backend = TestBackend::new(100, 40);
        let mut terminal = Terminal::new(backend).expect("test backend");
        let shared = Arc::new(Mutex::new(None));
        let cli = Cli::parse_from(["openbatrangs"]);
        let mut app = App::new(&cli, shared);
        terminal.draw(|frame| ui(frame, &mut app)).expect("draw ui");
        let buffer = terminal.backend().buffer();
        let content = buffer
            .content()
            .iter()
            .map(|cell| cell.symbol())
            .collect::<String>();
        assert!(content.contains('█'), "banner pixel art should render");
        assert!(content.contains("openBatarangs") || content.contains("perf"));
    }

    #[test]
    fn agent_events_are_rendered_in_chat() {
        use clap::Parser;
        use ratatui::backend::TestBackend;

        let client = crate::ollama::OllamaClient::new("http://localhost:11434")
            .expect("client should build");
        let (tx, mut rx) = mpsc::unbounded_channel::<UiEvent>();
        let shared = Arc::new(Mutex::new(None));
        let cli = Cli::parse_from(["openbatrangs"]);
        let mut app = App::new(&cli, shared);

        tx.send(UiEvent::Log("🦇 Step 1/1".to_string())).unwrap();
        tx.send(UiEvent::Chunk("HELLO from agent".to_string()))
            .unwrap();
        tx.send(UiEvent::Done(Ok(()))).unwrap();
        while let Ok(event) = rx.try_recv() {
            app.handle_event(event, &client, &tx);
        }

        let backend = TestBackend::new(100, 40);
        let mut terminal = Terminal::new(backend).expect("test backend");
        terminal.draw(|frame| ui(frame, &mut app)).expect("draw ui");
        let buffer = terminal.backend().buffer();
        let content = buffer
            .content()
            .iter()
            .map(|cell| cell.symbol())
            .collect::<String>();
        assert!(
            content.contains("HELLO from agent"),
            "agent output should render"
        );
        assert!(
            content.contains("Agent finished"),
            "done event should render"
        );
    }
}
