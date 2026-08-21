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

mod app;
mod ui;

use crate::agent::{AgentConfig, Reporter};
use crate::cli::{AgentMode, AgentRunConfig, Cli, ModelPrefs};
use crate::model_select::{calculate_memory_budget, resolve_model, resolve_model_context};
use crate::ollama::{ChatMessage, ChatRequest, OllamaClient};
use crate::perf::TegrastatsGuard;
use anyhow::{Context, Result};
use crossterm::event::{
    self, DisableMouseCapture, EnableMouseCapture, Event, KeyboardEnhancementFlags,
    PopKeyboardEnhancementFlags, PushKeyboardEnhancementFlags,
};
use crossterm::execute;
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
use futures_util::StreamExt;
use ratatui::backend::CrosstermBackend;
use ratatui::Terminal;
use std::io::stdout;
use std::path::{Path, PathBuf};
use std::process::Command;
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
    "mode",
    "thinking",
];

/// Spinner frames shown while the agent is working.
const SPINNER: &[&str] = &["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

/// Maximum visible lines in the multi-line input box.
const MAX_INPUT_LINES: usize = 20;
/// Extra lines for the input box border.
const INPUT_BOX_PADDING: usize = 2;
/// Minimum input box height (border + one line).
const MIN_INPUT_BOX_HEIGHT: usize = 3;
/// Cap for the streaming live line to avoid unbounded memory.
const MAX_LIVE_CHARS: usize = 50_000;
/// Redraw interval in milliseconds (drives the spinner).
const TICK_MILLIS: u64 = 80;
/// Width of the model picker popup as a percentage of the screen.
const MODEL_PICKER_WIDTH_PERCENT: u16 = 60;
/// Height of the model picker popup as a percentage of the screen.
const MODEL_PICKER_HEIGHT_PERCENT: u16 = 40;
/// Height of the live performance panel (border + three content lines).
const PERF_PANEL_HEIGHT: u16 = 5;
/// Minimum elapsed time before the token-rate estimate refreshes.
const TOKEN_RATE_MIN_ELAPSED: f64 = 0.5;
/// Minimum terminal height for showing the performance panel automatically.
const PERF_MIN_TERMINAL_HEIGHT: u16 = 18;
/// Number of lines scrolled per PageUp/PageDown in the chat area.
const CHAT_SCROLL_STEP: usize = 5;
/// Banner height: wordmark + Batman art + quote + model info + prompt.
const COMPACT_BANNER_HEIGHT: u16 = 18;

/// Chat-mode system prompt: no tools, direct conversation and code.
const CHAT_SYSTEM_PROMPT: &str =
    "You are openBatarangs in chat mode. Answer coding questions, explain ideas, and write code when asked. Be concise, practical, and do not mention tools.";

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

fn split_command(line: &str) -> (&str, &str) {
    let mut parts = line[1..].splitn(2, char::is_whitespace);
    let name = parts.next().unwrap_or_default();
    let arg = parts.next().unwrap_or_default().trim();
    (name, arg)
}

/// Find an existing file path inside a chat line, resolving relative to `cwd`.
fn extract_path_from_line(line: &str, cwd: &Path) -> Option<PathBuf> {
    line.split_whitespace().find_map(|token| {
        let candidate = match token.starts_with('/') {
            true => PathBuf::from(token),
            false => cwd.join(token),
        };
        candidate.is_file().then_some(candidate)
    })
}

/// Open a file in `vim` inside a new terminal window.
fn open_in_vim(path: &Path) {
    let path_str = path.to_string_lossy().to_string();
    let terminals = [
        "x-terminal-emulator",
        "gnome-terminal",
        "konsole",
        "xfce4-terminal",
        "alacritty",
        "kitty",
    ];
    for terminal in terminals {
        let spawned = match terminal {
            "gnome-terminal" => Command::new(terminal)
                .arg("--")
                .arg("vim")
                .arg(&path_str)
                .spawn(),
            _ => Command::new(terminal)
                .arg("-e")
                .arg("vim")
                .arg(&path_str)
                .spawn(),
        };
        if spawned.is_ok() {
            return;
        }
    }
}

async fn run_agent_worker(
    client: OllamaClient,
    config: AgentRunConfig,
    model_slot: Option<String>,
    prefs: ModelPrefs,
    task: String,
    tx: mpsc::UnboundedSender<UiEvent>,
) -> Result<()> {
    if config.mode == AgentMode::Chat {
        return run_chat_worker(client, config, model_slot, prefs, task, tx).await;
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
    task: String,
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
        .clamp(4_096, 16_384);

    let request = ChatRequest {
        model: selected.name.clone(),
        messages: vec![
            ChatMessage {
                role: "system".to_string(),
                content: CHAT_SYSTEM_PROMPT.to_string(),
            },
            ChatMessage {
                role: "user".to_string(),
                content: task,
            },
        ],
        stream: true,
        format: None,
        options: Some(serde_json::json!({
            "temperature": 0.7,
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

pub(crate) async fn run(cli: &Cli, client: &OllamaClient) -> Result<()> {
    let mut terminal = setup_terminal()?;
    let result = run_loop(&mut terminal, cli, client).await;
    teardown_terminal(&mut terminal)?;
    result
}

fn setup_terminal() -> Result<Terminal<CrosstermBackend<std::io::Stdout>>> {
    enable_raw_mode()?;
    let mut stdout = stdout();
    execute!(
        stdout,
        EnterAlternateScreen,
        EnableMouseCapture,
        PushKeyboardEnhancementFlags(
            KeyboardEnhancementFlags::DISAMBIGUATE_ESCAPE_CODES
                | KeyboardEnhancementFlags::REPORT_EVENT_TYPES
                | KeyboardEnhancementFlags::REPORT_ALL_KEYS_AS_ESCAPE_CODES
        )
    )?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend).context("failed to create terminal")?;
    terminal.hide_cursor()?;
    Ok(terminal)
}

fn teardown_terminal(terminal: &mut Terminal<CrosstermBackend<std::io::Stdout>>) -> Result<()> {
    terminal.show_cursor()?;
    execute!(
        terminal.backend_mut(),
        PopKeyboardEnhancementFlags,
        DisableMouseCapture,
        LeaveAlternateScreen
    )?;
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
    let mut app = app::App::new(cli, tegrastats_shared);
    app.refresh_model_info(client).await;
    let (tx, mut rx) = mpsc::unbounded_channel::<UiEvent>();
    let mut events = event::EventStream::new();

    loop {
        terminal.draw(|f| ui::ui(f, &mut app))?;

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
                    Some(Ok(Event::Mouse(mouse))) => app.handle_mouse(mouse),
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
    use crate::banner;

    #[test]
    fn stripped_banner_has_content_for_tui() {
        let banner = strip_ansi(&banner::banner_text());
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
        let mut app = app::App::new(&cli, shared);
        terminal
            .draw(|frame| ui::ui(frame, &mut app))
            .expect("draw ui");
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
        let mut app = app::App::new(&cli, shared);

        tx.send(UiEvent::Log("🦇 Step 1/1".to_string())).unwrap();
        tx.send(UiEvent::Chunk("HELLO from agent".to_string()))
            .unwrap();
        tx.send(UiEvent::Done(Ok(()))).unwrap();
        while let Ok(event) = rx.try_recv() {
            app.handle_event(event, &client, &tx);
        }

        let backend = TestBackend::new(100, 40);
        let mut terminal = Terminal::new(backend).expect("test backend");
        terminal
            .draw(|frame| ui::ui(frame, &mut app))
            .expect("draw ui");
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
