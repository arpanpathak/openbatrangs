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
//!
//! Responsibilities are split into:
//! - `app`: application state, editing, and event handling.
//! - `commands`: slash-command handlers.
//! - `text`: ANSI stripping, wrapping, path extraction, and editor launch.
//! - `worker`: background agent/chat workers.
//! - `ui`: ratatui rendering.

mod app;
mod chat;
mod commands;
mod input;
mod scroll;
mod session;
mod text;
mod ui;
mod worker;

pub(super) use text::{
    chat_visual_line_indices, extract_path_from_line, open_in_vim, split_command, strip_ansi,
    text_wrapped_height, wrap_text_to_lines,
};
pub(super) use worker::{run_agent_worker, run_pull_worker, run_setup_worker, UiEvent};

use crate::cli::Cli;
use crate::constants::tui::TICK_MILLIS;
use crate::engine::InferenceBackend;
use crate::ollama::OllamaClient;
use crate::perf::TegrastatsGuard;
use anyhow::{Context, Result};
use crossterm::event::{
    self, DisableBracketedPaste, DisableMouseCapture, EnableBracketedPaste, EnableMouseCapture,
    Event, KeyboardEnhancementFlags, PopKeyboardEnhancementFlags, PushKeyboardEnhancementFlags,
};
use crossterm::execute;
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
use futures_util::StreamExt;
use ratatui::backend::CrosstermBackend;
use ratatui::Terminal;
use std::io::stdout;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::sync::mpsc;

pub(crate) async fn run(
    cli: &Cli,
    client: &OllamaClient,
    backend: std::sync::Arc<dyn InferenceBackend>,
) -> Result<()> {
    let mut terminal = setup_terminal()?;
    let result = run_loop(&mut terminal, cli, client, backend).await;
    teardown_terminal(&mut terminal)?;
    result
}

fn setup_terminal() -> Result<Terminal<CrosstermBackend<std::io::Stdout>>> {
    enable_raw_mode()?;
    let mut stdout = stdout();
    execute!(
        stdout,
        EnterAlternateScreen,
        EnableBracketedPaste,
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
        DisableBracketedPaste,
        DisableMouseCapture,
        LeaveAlternateScreen
    )?;
    disable_raw_mode()?;
    println!("👋 Bye!");
    Ok(())
}

/// Keep the terminal's mouse-capture state in sync with the app preference.
/// Default is OFF so native text selection/copy always works.
fn sync_mouse_capture(
    terminal: &mut Terminal<CrosstermBackend<std::io::Stdout>>,
    enabled: bool,
    current: &mut bool,
) -> Result<()> {
    match (enabled, *current) {
        (true, false) => {
            execute!(terminal.backend_mut(), EnableMouseCapture)?;
            *current = true;
        }
        (false, true) => {
            execute!(terminal.backend_mut(), DisableMouseCapture)?;
            *current = false;
        }
        _ => {}
    }
    Ok(())
}

async fn run_loop(
    terminal: &mut Terminal<CrosstermBackend<std::io::Stdout>>,
    cli: &Cli,
    client: &OllamaClient,
    backend: std::sync::Arc<dyn InferenceBackend>,
) -> Result<()> {
    let tegrastats_shared = Arc::new(Mutex::new(None));
    let _tegrastats_guard = TegrastatsGuard::start(tegrastats_shared.clone());
    let mut app = app::App::new(cli, tegrastats_shared, backend);
    app.refresh_model_info().await;
    let mut mouse_capture = false;
    let (tx, mut rx) = mpsc::unbounded_channel::<UiEvent>();
    let mut events = event::EventStream::new();

    loop {
        sync_mouse_capture(terminal, app.mouse_capture, &mut mouse_capture)?;
        terminal.draw(|f| ui::ui(f, &mut app))?;

        if app.should_quit {
            break;
        }

        tokio::select! {
            maybe = rx.recv() => {
                if let Some(event) = maybe {
                    app.handle_event(event, &tx);
                }
            }
            maybe = events.next() => {
                match maybe {
                    Some(Ok(Event::Key(key))) => {
                        if app.handle_key(key, client, &tx).await? {
                            break;
                        }
                    }
                    Some(Ok(Event::Paste(text))) => app.insert_text(&text),
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

    fn test_backend() -> std::sync::Arc<dyn crate::engine::InferenceBackend> {
        let client = crate::ollama::OllamaClient::new("http://localhost:11434").expect("client");
        std::sync::Arc::new(crate::engine::OllamaBackend::new(client))
    }

    #[test]
    fn stripped_banner_has_content_for_tui() {
        let banner = strip_ansi(&banner::banner_text());
        assert!(!banner.contains('\x1b'));
        assert!(banner.lines().count() > 5);
        assert!(banner.contains('⣿') || banner.contains('█'));
    }

    #[test]
    fn ui_renders_banner_content() {
        use clap::Parser;
        use ratatui::backend::TestBackend;

        let backend = TestBackend::new(100, 40);
        let mut terminal = Terminal::new(backend).expect("test backend");
        let shared = Arc::new(Mutex::new(None));
        let cli = Cli::parse_from(["openbatrangs"]);
        let mut app = app::App::new(&cli, shared, test_backend());
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
            content.contains('⣿') || content.contains('█'),
            "banner pixel art should render"
        );
        assert!(content.contains("OPEN-BATARANGS") || content.contains("perf"));
    }

    #[test]
    fn agent_events_are_rendered_in_chat() {
        use clap::Parser;
        use ratatui::backend::TestBackend;

        let (tx, mut rx) = mpsc::unbounded_channel::<UiEvent>();
        let shared = Arc::new(Mutex::new(None));
        let cli = Cli::parse_from(["openbatrangs"]);
        let mut app = app::App::new(&cli, shared, test_backend());

        tx.send(UiEvent::Log("🦇 Step 1/1".to_string())).unwrap();
        tx.send(UiEvent::Chunk("HELLO from agent".to_string()))
            .unwrap();
        tx.send(UiEvent::Done(Ok(()))).unwrap();
        while let Ok(event) = rx.try_recv() {
            app.handle_event(event, &tx);
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

    #[test]
    fn small_terminal_large_paste_does_not_panic() {
        use clap::Parser;
        use ratatui::backend::TestBackend;

        let backend = TestBackend::new(80, 24);
        let mut terminal = Terminal::new(backend).expect("test backend");
        let shared = Arc::new(Mutex::new(None));
        let cli = Cli::parse_from(["openbatrangs"]);
        let mut app = app::App::new(&cli, shared, test_backend());
        app.input = "NVIDIA — 3% → 😀\n| table |\n```rust\nfn main() {}\n```".repeat(200);
        terminal
            .draw(|frame| ui::ui(frame, &mut app))
            .expect("draw with large pasted input should not panic");
    }

    #[test]
    fn tiny_terminal_does_not_panic() {
        use clap::Parser;
        use ratatui::backend::TestBackend;

        let backend = TestBackend::new(40, 8);
        let mut terminal = Terminal::new(backend).expect("test backend");
        let shared = Arc::new(Mutex::new(None));
        let cli = Cli::parse_from(["openbatrangs"]);
        let mut app = app::App::new(&cli, shared, test_backend());
        terminal
            .draw(|frame| ui::ui(frame, &mut app))
            .expect("draw on a tiny terminal should not panic");
    }
}
