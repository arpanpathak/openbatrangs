use crate::agent::{AgentConfig, Reporter};
use crate::ollama::OllamaClient;
use crate::{resolve_model_context, select_model, Cli, ModelPrefs};
use anyhow::{Context, Result};
use crossterm::event::{self, Event, KeyCode, KeyModifiers};
use crossterm::execute;
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
use futures_util::StreamExt;
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Direction, Layout};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, List, ListItem, Paragraph, Wrap};
use ratatui::Terminal;
use std::io::stdout;
use std::path::PathBuf;
use std::time::Duration;
use tokio::sync::mpsc;

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
];

const SPINNER: &[&str] = &["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

const MAX_INPUT_LINES: usize = 5;
const INPUT_BOX_PADDING: usize = 2;
const MIN_INPUT_BOX_HEIGHT: usize = 3;
const MAX_LIVE_CHARS: usize = 50_000;
const TICK_MILLIS: u64 = 80;
const SUGGESTIONS_HEIGHT: u16 = 4;
const MODEL_PICKER_WIDTH_PERCENT: u16 = 60;
const MODEL_PICKER_HEIGHT_PERCENT: u16 = 40;

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

    fn chunk(&mut self, msg: String) {
        let _ = self.tx.send(UiEvent::Chunk(strip_ansi(&msg)));
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
    running: bool,
    status: String,
    last_action: String,
    spinner_frame: u64,
    task_queue: Vec<String>,
    picker: Option<PickerState>,
    quit: bool,
    model: Option<String>,
    read_only: bool,
    confirm: bool,
    max_steps: usize,
    cwd: PathBuf,
    min_context: u64,
    no_auto_pull: bool,
}

impl App {
    fn new(cli: &Cli) -> Self {
        let mut log = crate::banner::banner_text()
            .lines()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();
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
            running: false,
            status: "ready".to_string(),
            last_action: String::new(),
            spinner_frame: 0,
            task_queue: Vec::new(),
            picker: None,
            quit: false,
            model: cli.model.clone(),
            read_only: cli.read_only,
            confirm: cli.confirm,
            max_steps: cli.max_steps,
            cwd: cli.cwd.clone(),
            min_context: cli.min_context as u64,
            no_auto_pull: cli.no_auto_pull,
        }
    }

    fn suggestions(&self) -> Vec<String> {
        if self.input.starts_with('/') {
            let q = &self.input[1..];
            COMMANDS
                .iter()
                .filter(|c| c.starts_with(q))
                .map(|s| format!("/{s}"))
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
        SPINNER[(self.spinner_frame as usize) % SPINNER.len()]
    }

    fn start_task(
        &mut self,
        task: String,
        client: &OllamaClient,
        tx: &mpsc::UnboundedSender<UiEvent>,
    ) {
        self.log.push(format!("🦇 {task}"));
        self.running = true;
        self.status = "running".to_string();
        self.last_action.clear();
        let tx2 = tx.clone();
        let client2 = client.clone();
        let cwd = self.cwd.clone();
        let max_steps = self.max_steps;
        let read_only = self.read_only;
        let confirm = self.confirm;
        let model = self.model.clone();
        let prefs = ModelPrefs {
            model: model.clone(),
            no_auto_pull: self.no_auto_pull,
            min_context: self.min_context,
        };
        tokio::spawn(async move {
            let result = run_agent_worker(
                client2,
                cwd,
                max_steps,
                read_only,
                confirm,
                model,
                prefs,
                task,
                tx2.clone(),
            )
            .await;
            let _ = tx2.send(UiEvent::Done(result.map_err(|e| format!("{e:#}"))));
        });
    }

    fn handle_event(
        &mut self,
        ev: UiEvent,
        client: &OllamaClient,
        tx: &mpsc::UnboundedSender<UiEvent>,
    ) {
        match ev {
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
                    Err(e) => {
                        self.log.push(format!("⚠️ {e}"));
                    }
                }
                self.running = false;
                self.last_action.clear();
                if let Some(next) = self.task_queue.first().cloned() {
                    self.task_queue.remove(0);
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
        // Model picker takes priority.
        if let Some(picker) = &mut self.picker {
            match key.code {
                KeyCode::Up => picker.selected = picker.selected.saturating_sub(1),
                KeyCode::Down => {
                    picker.selected =
                        (picker.selected + 1).min(picker.models.len().saturating_sub(1));
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
            return Ok(false);
        }

        match key.code {
            KeyCode::Char(c) => {
                if c != '\r' {
                    self.input.insert(self.cursor, c);
                    self.cursor += 1;
                }
            }
            KeyCode::Backspace => {
                if self.cursor > 0 {
                    self.input.remove(self.cursor - 1);
                    self.cursor -= 1;
                }
            }
            KeyCode::Delete => {
                if self.cursor < self.input.len() {
                    self.input.remove(self.cursor);
                }
            }
            KeyCode::Left => self.cursor = self.cursor.saturating_sub(1),
            KeyCode::Right => {
                if self.cursor < self.input.len() {
                    self.cursor += 1;
                }
            }
            KeyCode::Home => self.cursor = 0,
            KeyCode::End => self.cursor = self.input.len(),
            KeyCode::Up => {
                let sugg = self.suggestions();
                if !sugg.is_empty() {
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
            KeyCode::Down => {
                let sugg = self.suggestions();
                if !sugg.is_empty() {
                    self.selected = (self.selected + 1).min(sugg.len().saturating_sub(1));
                } else if let Some(i) = self.history_idx {
                    if i + 1 < self.history.len() {
                        self.history_idx = Some(i + 1);
                        self.input = self.history[i + 1].clone();
                        self.cursor = self.input.len();
                    } else {
                        self.history_idx = None;
                        self.input.clear();
                        self.cursor = 0;
                    }
                }
            }
            KeyCode::Tab => {
                let sugg = self.suggestions();
                if !sugg.is_empty() {
                    self.input = sugg[self.selected].clone();
                    self.cursor = self.input.len();
                }
            }
            KeyCode::Enter => {
                if key.modifiers.contains(KeyModifiers::SHIFT) {
                    self.input.insert(self.cursor, '\n');
                    self.cursor += 1;
                } else {
                    let task = self.input.trim().to_string();
                    self.input.clear();
                    self.cursor = 0;
                    self.history_idx = None;
                    self.selected = 0;
                    if task.is_empty() {
                        return Ok(false);
                    }
                    self.history.push(task.clone());
                    if task.starts_with('/') {
                        self.run_slash_command(client, &task).await?;
                    } else if self.running {
                        self.task_queue.push(task);
                        self.log
                            .push("⏳ Queued — will run after the current task.".to_string());
                    } else {
                        self.start_task(task, client, tx);
                    }
                }
            }
            KeyCode::Esc => self.quit = true,
            _ => {}
        }
        Ok(self.quit)
    }

    async fn run_slash_command(&mut self, client: &OllamaClient, line: &str) -> Result<()> {
        let mut parts = line[1..].splitn(2, char::is_whitespace);
        let name = parts.next().unwrap_or("");
        let arg = parts.next().unwrap_or("").trim();

        match name {
            "help" | "h" => {
                self.log.push("Commands:".to_string());
                self.log
                    .push("  /help, /exit, /quit, /setup, /models".to_string());
                self.log
                    .push("  /model <tag>, /read-only, /confirm".to_string());
                self.log
                    .push("  /steps <n>, /cwd <path>, /doctor, /clear".to_string());
                self.log
                    .push("  Shift+Enter = new line · Enter = send".to_string());
            }
            "exit" | "quit" => self.quit = true,
            "models" => {
                let tags = client.tags().await?;
                if tags.is_empty() {
                    self.log
                        .push("No models installed. Run /setup.".to_string());
                } else {
                    self.picker = Some(PickerState {
                        models: tags.into_iter().map(|m| m.name).collect(),
                        selected: 0,
                    });
                    self.status = "select model".to_string();
                }
            }
            "model" => {
                if arg.is_empty() {
                    match &self.model {
                        Some(m) => self.log.push(format!("Current model: {m}")),
                        None => self.log.push(
                            "Auto mode — best model will be selected on first task.".to_string(),
                        ),
                    }
                } else {
                    let tags = client.tags().await?;
                    if tags.iter().any(|m| m.name == arg) {
                        self.model = Some(arg.to_string());
                        self.log.push(format!("✅ Model set to {arg}"));
                    } else {
                        self.log.push(format!(
                            "❌ Model '{arg}' is not installed. Try /models or /setup."
                        ));
                    }
                }
            }
            "read-only" => {
                self.read_only = !self.read_only;
                self.log.push(format!(
                    "Read-only mode: {}",
                    if self.read_only { "ON" } else { "OFF" }
                ));
            }
            "confirm" => {
                self.confirm = !self.confirm;
                self.log.push(format!(
                    "Confirm mode: {}",
                    if self.confirm { "ON" } else { "OFF" }
                ));
            }
            "steps" => match arg.parse::<usize>() {
                Ok(n) if n > 0 => {
                    self.max_steps = n;
                    self.log.push(format!("Max steps set to {n}"));
                }
                _ => self.log.push("Usage: /steps <positive number>".to_string()),
            },
            "cwd" => {
                if arg.is_empty() {
                    self.log.push(format!("Workspace: {}", self.cwd.display()));
                } else {
                    self.cwd = PathBuf::from(arg);
                    self.log
                        .push(format!("Workspace set to {}", self.cwd.display()));
                }
            }
            "doctor" => {
                for l in crate::doctor_lines(client, self.min_context).await? {
                    self.log.push(l);
                }
            }
            "setup" => {
                self.log.push("Running setup...".to_string());
                crate::setup(client).await?;
                self.log.push("✅ Setup finished.".to_string());
            }
            "clear" => {
                self.log.clear();
                self.live.clear();
            }
            _ => self
                .log
                .push(format!("Unknown command: /{name}. Try /help")),
        }
        Ok(())
    }
}

async fn run_agent_worker(
    client: OllamaClient,
    cwd: PathBuf,
    max_steps: usize,
    read_only: bool,
    confirm: bool,
    model_slot: Option<String>,
    prefs: ModelPrefs,
    task: String,
    tx: mpsc::UnboundedSender<UiEvent>,
) -> Result<()> {
    let mem_budget = crate::models::total_system_memory_bytes() * 3 / 4;
    let selected = match &model_slot {
        Some(name) => {
            let explicit = ModelPrefs {
                model: Some(name.clone()),
                ..prefs.clone()
            };
            select_model(&client, &explicit, mem_budget).await?
        }
        None => select_model(&client, &prefs, mem_budget).await?,
    };
    let model_context = resolve_model_context(&client, &selected.name).await?;
    let config = AgentConfig {
        cwd,
        max_steps,
        read_only,
        confirm,
    };
    let mut reporter = ChannelReporter { tx };
    crate::agent::run_agent(
        &config,
        &client,
        &selected.name,
        model_context,
        &task,
        &mut reporter,
    )
    .await
}

fn ui(f: &mut ratatui::Frame, app: &mut App) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .margin(1)
        .constraints([
            Constraint::Min(3),
            Constraint::Length(1),
            Constraint::Length(if app.suggestions().is_empty() {
                0
            } else {
                SUGGESTIONS_HEIGHT
            }),
            Constraint::Length(input_height(app) as u16),
        ])
        .split(f.area());

    // Chat area (wraps long lines)
    let mut chat_text = app.log.join("\n");
    if !app.live.is_empty() {
        if !chat_text.is_empty() {
            chat_text.push('\n');
        }
        chat_text.push_str(&app.live);
    }
    let chat = Paragraph::new(chat_text)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title(" openBatarangs "),
        )
        .wrap(Wrap { trim: false })
        .scroll((u16::MAX, 0));
    f.render_widget(chat, chunks[0]);

    // Status line with spinner + current action
    let status_line = if app.running {
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
    let status_style = if app.running {
        Style::default().fg(Color::Yellow)
    } else {
        Style::default().fg(Color::Green)
    };
    f.render_widget(Paragraph::new(status_line).style(status_style), chunks[1]);

    // Suggestions
    if chunks[2].height > 0 {
        let sugg = app.suggestions();
        let sugg_items: Vec<ListItem> = sugg
            .iter()
            .enumerate()
            .map(|(i, s)| {
                let style = if i == app.selected {
                    Style::default()
                        .fg(Color::Cyan)
                        .add_modifier(Modifier::BOLD)
                } else {
                    Style::default()
                };
                ListItem::new(Line::from(Span::styled(s.clone(), style)))
            })
            .collect();
        f.render_widget(
            List::new(sugg_items).block(
                Block::default()
                    .borders(Borders::ALL)
                    .title(" suggestions "),
            ),
            chunks[2],
        );
    }

    // Input box
    let input_para = Paragraph::new(app.input.clone())
        .block(Block::default().borders(Borders::ALL).title(" input "))
        .wrap(Wrap { trim: false });
    f.render_widget(input_para, chunks[3]);

    // Cursor
    let prefix = &app.input[..app.cursor];
    let line_idx = prefix.matches('\n').count();
    let col = prefix
        .rsplit('\n')
        .next()
        .map(|s| s.chars().count())
        .unwrap_or(0);
    let x = chunks[3].x + 1 + col as u16;
    let y = chunks[3].y + 1 + line_idx as u16;
    f.set_cursor_position((x, y.min(chunks[3].y + chunks[3].height.saturating_sub(1))));

    // Model picker overlay
    if let Some(picker) = &app.picker {
        let area = centered_rect(
            MODEL_PICKER_WIDTH_PERCENT,
            MODEL_PICKER_HEIGHT_PERCENT,
            f.area(),
        );
        let items: Vec<ListItem> = picker
            .models
            .iter()
            .enumerate()
            .map(|(i, m)| {
                let style = if i == picker.selected {
                    Style::default()
                        .fg(Color::Black)
                        .bg(Color::Cyan)
                        .add_modifier(Modifier::BOLD)
                } else {
                    Style::default()
                };
                ListItem::new(Line::from(Span::styled(m.clone(), style)))
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
    enable_raw_mode()?;
    let mut stdout = stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend).context("failed to create terminal")?;
    terminal.hide_cursor()?;

    let mut app = App::new(cli);
    let (tx, mut rx) = mpsc::unbounded_channel::<UiEvent>();
    let mut events = event::EventStream::new();

    loop {
        terminal.draw(|f| ui(f, &mut app))?;

        if app.quit {
            break;
        }

        tokio::select! {
            maybe = rx.recv() => {
                if let Some(ev) = maybe {
                    app.handle_event(ev, client, &tx);
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
                    Some(Err(e)) => return Err(e.into()),
                    None => break,
                }
            }
            _ = tokio::time::sleep(Duration::from_millis(TICK_MILLIS)) => {
                app.spinner_frame += 1;
            }
        }
    }

    terminal.show_cursor()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    disable_raw_mode()?;
    println!("👋 Bye!");
    Ok(())
}
