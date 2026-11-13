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
    chat_visual_line_indices, extract_path_from_line, open_in_vim, split_command,
    text_wrapped_height, wrap_text_to_lines,
};
pub(super) use worker::{
    UiEvent, WorkerHistory, run_agent_worker, run_pull_worker, run_setup_worker,
};

/// How the chat viewport follows newly generated output.
///
/// Using an enum instead of a raw boolean makes the two states explicit and
/// impossible to confuse: `Follow` pins the view to the newest lines, `Manual`
/// leaves the user's scroll offset alone.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ScrollMode {
    /// Stay pinned to the bottom as new output arrives.
    Follow,
    /// Preserve the user's current scroll position.
    Manual,
}

use crate::cli::Cli;
use crate::constants::tui::{MAX_BACKGROUND_EVENTS_PER_FRAME, TICK_MILLIS};
use crate::ollama::OllamaClient;
use crate::perf::TegrastatsGuard;
use anyhow::{Context, Result};
use crossterm::event::{
    self, DisableBracketedPaste, DisableMouseCapture, EnableBracketedPaste, EnableMouseCapture,
    Event, KeyboardEnhancementFlags, PopKeyboardEnhancementFlags, PushKeyboardEnhancementFlags,
};
use crossterm::execute;
use crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use futures_util::{Stream, StreamExt};
use ratatui::Terminal;
use ratatui::backend::CrosstermBackend;
use std::io::{Write, stdout};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::sync::mpsc;

/// Whether the event loop keeps running.
#[derive(Debug, PartialEq)]
enum LoopControl {
    /// Draw the next frame and wait for the next input.
    Continue,
    /// Leave the event loop.
    Exit,
}

/// One thing the event loop woke up for.
enum LoopInput {
    /// A terminal event, an input error, or `None` when the input closed.
    Terminal(Option<std::io::Result<Event>>),
    /// An event from a background worker, or `None` when the channel closed.
    Background(Option<UiEvent>),
    /// The redraw timer fired.
    Tick,
}

/// Launch the interactive TUI: set up the terminal, run the event loop, restore.
///
/// # Parameters
///
/// - `cli`: parsed command-line arguments for initial app state.
/// - `client`: Ollama client for model operations.
///
/// # Returns
///
/// `Ok(())` when the user exits normally, or an error if setup/teardown fails.
pub(crate) async fn run(cli: &Cli, client: &OllamaClient) -> Result<()> {
    let mut terminal = setup_terminal()?;
    let mut events = event::EventStream::new();

    let result = run_app(&mut terminal, &mut events, cli, client).await;

    teardown_terminal(&mut terminal)?;
    result
}

/// Configure the terminal for TUI mode: raw input, alternate screen, enhanced keys.
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

/// Restore the terminal to its original state after the TUI exits.
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

/// Build the app state and run the event loop until the user quits.
///
/// The terminal writer and the event stream are parameters so tests can
/// drive the whole loop with an in-memory writer and scripted events.
async fn run_app<W: Write>(
    terminal: &mut Terminal<CrosstermBackend<W>>,
    events: &mut (impl Stream<Item = std::io::Result<Event>> + Unpin),
    cli: &Cli,
    client: &OllamaClient,
) -> Result<()> {
    let tegrastats_shared = Arc::new(Mutex::new(None));
    let _tegrastats_guard = TegrastatsGuard::start(tegrastats_shared.clone());

    let mut app = app::App::new(cli, tegrastats_shared);
    app.server_url = client.base_url.clone();
    app.refresh_model_info(client).await;

    run_loop(terminal, &mut app, events, client).await
}

/// Keep the terminal's mouse-capture state in sync with the app preference.
fn sync_mouse_capture<W: Write>(
    terminal: &mut Terminal<CrosstermBackend<W>>,
    enabled: bool,
    current: &mut bool,
) -> std::io::Result<()> {
    if enabled == *current {
        return Ok(());
    }

    *current = enabled;
    set_mouse_capture(terminal.backend_mut(), enabled)
}

/// Turn mouse capture on (wheel and scrollbar) or off (native text selection).
fn set_mouse_capture(writer: &mut impl Write, enabled: bool) -> std::io::Result<()> {
    match enabled {
        true => execute!(writer, EnableMouseCapture),
        false => execute!(writer, DisableMouseCapture),
    }
}

/// Main TUI event loop: render frames, process input, and dispatch background events.
///
/// Keyboard and mouse input is checked before background stream events (see
/// [`next_input`]), keeping scroll and Ctrl+C responsive during model generation.
async fn run_loop<W: Write>(
    terminal: &mut Terminal<CrosstermBackend<W>>,
    app: &mut app::App,
    events: &mut (impl Stream<Item = std::io::Result<Event>> + Unpin),
    client: &OllamaClient,
) -> Result<()> {
    let (tx, mut rx) = mpsc::unbounded_channel::<UiEvent>();
    let mut mouse_capture = false;
    let mut outcome = Ok(LoopControl::Continue);

    while let Ok(LoopControl::Continue) = outcome {
        outcome = step(
            terminal,
            app,
            events,
            &mut rx,
            client,
            &tx,
            &mut mouse_capture,
        )
        .await;
    }

    // `Ok(Exit)` ends the loop normally; an error ends it with that error.
    outcome.and(Ok(()))
}

/// Draw one frame, then handle the next input.
async fn step<W: Write>(
    terminal: &mut Terminal<CrosstermBackend<W>>,
    app: &mut app::App,
    events: &mut (impl Stream<Item = std::io::Result<Event>> + Unpin),
    rx: &mut mpsc::UnboundedReceiver<UiEvent>,
    client: &OllamaClient,
    tx: &mpsc::UnboundedSender<UiEvent>,
    mouse_capture: &mut bool,
) -> Result<LoopControl> {
    draw_frame(terminal, app, mouse_capture)?;

    next_control(app, events, rx, client, tx).await
}

/// Apply the mouse-capture setting and draw one frame.
fn draw_frame<W: Write>(
    terminal: &mut Terminal<CrosstermBackend<W>>,
    app: &mut app::App,
    mouse_capture: &mut bool,
) -> Result<()> {
    sync_mouse_capture(terminal, app.mouse_capture, mouse_capture)?;
    terminal.draw(|frame| ui::ui(frame, app))?;

    Ok(())
}

/// Exit once the app wants to quit; otherwise wait for one input and handle it.
async fn next_control(
    app: &mut app::App,
    events: &mut (impl Stream<Item = std::io::Result<Event>> + Unpin),
    rx: &mut mpsc::UnboundedReceiver<UiEvent>,
    client: &OllamaClient,
    tx: &mpsc::UnboundedSender<UiEvent>,
) -> Result<LoopControl> {
    if app.should_quit {
        return Ok(LoopControl::Exit);
    }

    let input = next_input(events, rx).await;

    handle_input(app, input, rx, client, tx).await
}

/// Wait for the next terminal event, background event, or timer tick.
///
/// Biased: keyboard/mouse input is checked before background stream events
/// so scrolling and Ctrl+C stay responsive while the model is generating.
/// Without this, a continuously-full event channel can starve input and make
/// scroll feel broken.
async fn next_input(
    events: &mut (impl Stream<Item = std::io::Result<Event>> + Unpin),
    rx: &mut mpsc::UnboundedReceiver<UiEvent>,
) -> LoopInput {
    tokio::select! {
        biased;
        event = events.next() => LoopInput::Terminal(event),
        event = rx.recv() => LoopInput::Background(event),
        () = tokio::time::sleep(Duration::from_millis(TICK_MILLIS)) => LoopInput::Tick,
    }
}

/// Handle one loop input.
async fn handle_input(
    app: &mut app::App,
    input: LoopInput,
    rx: &mut mpsc::UnboundedReceiver<UiEvent>,
    client: &OllamaClient,
    tx: &mpsc::UnboundedSender<UiEvent>,
) -> Result<LoopControl> {
    match input {
        LoopInput::Terminal(event) => handle_terminal_event(app, event, client, tx).await,
        LoopInput::Background(event) => Ok(handle_background_event(app, event, rx, client, tx)),
        LoopInput::Tick => Ok(tick(app)),
    }
}

/// Handle a terminal event. An input error ends the loop with that error.
async fn handle_terminal_event(
    app: &mut app::App,
    event: Option<std::io::Result<Event>>,
    client: &OllamaClient,
    tx: &mpsc::UnboundedSender<UiEvent>,
) -> Result<LoopControl> {
    dispatch_terminal_event(app, event.transpose()?, client, tx).await
}

/// Send a key, paste, or mouse event to the app. A closed input (`None`)
/// ends the loop. Resize and focus events need no handling: the next frame
/// is drawn at the new size.
async fn dispatch_terminal_event(
    app: &mut app::App,
    event: Option<Event>,
    client: &OllamaClient,
    tx: &mpsc::UnboundedSender<UiEvent>,
) -> Result<LoopControl> {
    match event {
        Some(Event::Key(key)) => Ok(handle_key_event(app, key, client, tx).await),
        Some(Event::Paste(text)) => Ok(paste(app, &text)),
        Some(Event::Mouse(mouse)) => Ok(click_or_scroll(app, mouse)),
        Some(Event::Resize(..) | Event::FocusGained | Event::FocusLost) => {
            Ok(LoopControl::Continue)
        }
        None => Ok(LoopControl::Exit),
    }
}

/// Handle a key press; exit when it quits the app.
async fn handle_key_event(
    app: &mut app::App,
    key: event::KeyEvent,
    client: &OllamaClient,
    tx: &mpsc::UnboundedSender<UiEvent>,
) -> LoopControl {
    let should_quit = app.handle_key(key, client, tx).await;

    loop_control(should_quit)
}

/// `Exit` when the app should quit, else `Continue`.
fn loop_control(should_quit: bool) -> LoopControl {
    match should_quit {
        true => LoopControl::Exit,
        false => LoopControl::Continue,
    }
}

/// Insert pasted text into the input line.
fn paste(app: &mut app::App, text: &str) -> LoopControl {
    app.insert_text(text);

    LoopControl::Continue
}

/// Handle a mouse event in the chat.
fn click_or_scroll(app: &mut app::App, mouse: event::MouseEvent) -> LoopControl {
    app.handle_mouse(mouse);

    LoopControl::Continue
}

/// Handle a background event, then any that queued up behind it.
fn handle_background_event(
    app: &mut app::App,
    event: Option<UiEvent>,
    rx: &mut mpsc::UnboundedReceiver<UiEvent>,
    client: &OllamaClient,
    tx: &mpsc::UnboundedSender<UiEvent>,
) -> LoopControl {
    if let Some(event) = event {
        app.handle_event(event, client, tx);
        handle_queued_events(app, rx, client, tx);
    }

    LoopControl::Continue
}

/// Handle up to [`MAX_BACKGROUND_EVENTS_PER_FRAME`] events that are already
/// waiting.
///
/// Coalescing bursts of background events (e.g. streaming tokens) redraws the
/// terminal once per batch instead of once per token. The biased select in
/// [`next_input`] still lets keyboard and mouse input through between batches.
fn handle_queued_events(
    app: &mut app::App,
    rx: &mut mpsc::UnboundedReceiver<UiEvent>,
    client: &OllamaClient,
    tx: &mpsc::UnboundedSender<UiEvent>,
) {
    let waiting = std::iter::from_fn(|| rx.try_recv().ok()).take(MAX_BACKGROUND_EVENTS_PER_FRAME);

    for event in waiting {
        app.handle_event(event, client, tx);
    }
}

/// Advance the spinner and refresh the perf panel when a sample is due.
fn tick(app: &mut app::App) -> LoopControl {
    app.spinner_frame += 1;

    if let Some(stats) = app.perf.sample_if_due(Instant::now()) {
        app.system_stats = stats;
    }

    LoopControl::Continue
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::banner;

    /// A terminal of 80 by 24 cells that writes its output into `output`.
    fn memory_terminal(
        output: &mut Vec<u8>,
    ) -> anyhow::Result<Terminal<CrosstermBackend<&mut Vec<u8>>>> {
        let options = ratatui::TerminalOptions {
            viewport: ratatui::Viewport::Fixed(ratatui::layout::Rect::new(0, 0, 80, 24)),
        };

        Ok(Terminal::with_options(
            CrosstermBackend::new(output),
            options,
        )?)
    }

    /// An app with default settings.
    fn test_app() -> app::App {
        use clap::Parser;

        let cli = Cli::parse_from(["openbatrangs"]);
        app::App::new(&cli, Arc::new(Mutex::new(None)))
    }

    /// A client for a port that refuses connections, so network calls fail fast.
    fn refused_client() -> anyhow::Result<OllamaClient> {
        OllamaClient::new("http://127.0.0.1:1")
    }

    /// Key presses that type `text`.
    fn typed(text: &str) -> Vec<std::io::Result<Event>> {
        text.chars()
            .map(|character| key(event::KeyCode::Char(character)))
            .collect()
    }

    /// A key press event.
    fn key(code: event::KeyCode) -> std::io::Result<Event> {
        Ok(Event::Key(event::KeyEvent::new(
            code,
            event::KeyModifiers::NONE,
        )))
    }

    /// A stream that yields `events`, then waits `pause`, then ends.
    fn events_then_pause(
        events: Vec<std::io::Result<Event>>,
        pause: Duration,
    ) -> impl Stream<Item = std::io::Result<Event>> + Unpin {
        let pause_then_end = futures_util::stream::once(tokio::time::sleep(pause))
            .filter_map(|()| std::future::ready(None));

        Box::pin(futures_util::stream::iter(events).chain(pause_then_end))
    }

    #[tokio::test]
    async fn the_app_runs_until_quit() -> anyhow::Result<()> {
        use clap::Parser;

        let mut output = Vec::new();
        let mut terminal = memory_terminal(&mut output)?;
        let mouse = event::MouseEvent {
            kind: event::MouseEventKind::ScrollDown,
            column: 1,
            row: 1,
            modifiers: event::KeyModifiers::NONE,
        };
        let mut script = vec![
            Ok(Event::Paste("/mouse".to_string())),
            Ok(Event::Mouse(mouse)),
            Ok(Event::Resize(80, 24)),
            Ok(Event::FocusGained),
            Ok(Event::FocusLost),
        ];
        script.extend(typed(" off"));
        script.push(key(event::KeyCode::Enter));
        script.extend(typed("/quit"));
        script.push(key(event::KeyCode::Enter));
        script.extend(typed("never typed"));
        let mut events = futures_util::stream::iter(script);
        let cli = Cli::parse_from(["openbatrangs"]);

        run_app(&mut terminal, &mut events, &cli, &refused_client()?).await?;

        drop(terminal);

        let output = String::from_utf8_lossy(&output);
        assert!(output.contains("\x1b[?1000h"), "mouse capture must turn on");
        assert!(
            output.contains("\x1b[?1000l"),
            "/mouse off must turn it off"
        );

        Ok(())
    }

    #[tokio::test]
    async fn the_loop_ends_when_the_input_closes() -> anyhow::Result<()> {
        let mut output = Vec::new();
        let mut terminal = memory_terminal(&mut output)?;
        let mut app = test_app();
        let mut events = futures_util::stream::iter(Vec::new());

        run_loop(&mut terminal, &mut app, &mut events, &refused_client()?).await?;

        assert!(!app.should_quit);

        Ok(())
    }

    #[tokio::test]
    async fn an_input_error_ends_the_loop_with_that_error() -> anyhow::Result<()> {
        let mut output = Vec::new();
        let mut terminal = memory_terminal(&mut output)?;
        let mut app = test_app();
        let failure = std::io::Error::other("terminal gone");
        let mut events = futures_util::stream::iter(vec![Err(failure)]);

        let result = run_loop(&mut terminal, &mut app, &mut events, &refused_client()?).await;

        assert!(result.is_err());

        Ok(())
    }

    #[tokio::test]
    async fn ticks_and_worker_events_arrive_while_input_waits() -> anyhow::Result<()> {
        let mut output = Vec::new();
        let mut terminal = memory_terminal(&mut output)?;
        let mut app = test_app();
        let mut script = typed("write a kernel");
        script.push(key(event::KeyCode::Enter));
        let mut events = events_then_pause(script, Duration::from_millis(500));

        run_loop(&mut terminal, &mut app, &mut events, &refused_client()?).await?;

        assert!(
            app.spinner_frame > 0,
            "the timer must tick while input waits"
        );
        assert!(!app.is_running, "the failed task must report back");
        assert!(app.log.iter().any(|line| line.starts_with("⚠️")));

        Ok(())
    }

    #[tokio::test]
    async fn queued_worker_events_are_handled_in_one_batch() -> anyhow::Result<()> {
        let client = refused_client()?;
        let (tx, mut rx) = mpsc::unbounded_channel();
        let mut app = test_app();
        tx.send(UiEvent::Log("second".to_string()))?;
        tx.send(UiEvent::Log("third".to_string()))?;

        let first = Some(UiEvent::Log("first".to_string()));
        let control = handle_background_event(&mut app, first, &mut rx, &client, &tx);
        let after_close = handle_background_event(&mut app, None, &mut rx, &client, &tx);

        assert_eq!(control, LoopControl::Continue);
        assert_eq!(after_close, LoopControl::Continue);
        assert_eq!(app.log.text(), "first\nsecond\nthird");

        Ok(())
    }

    #[tokio::test]
    async fn a_quitting_app_exits_without_waiting_for_input() -> anyhow::Result<()> {
        let client = refused_client()?;
        let (tx, mut rx) = mpsc::unbounded_channel();
        let mut app = test_app();
        app.should_quit = true;
        let mut events = futures_util::stream::pending();

        let control = next_control(&mut app, &mut events, &mut rx, &client, &tx).await?;

        assert_eq!(control, LoopControl::Exit);

        Ok(())
    }

    #[test]
    fn a_tick_advances_the_spinner_and_samples_stats() {
        let mut app = test_app();

        let control = tick(&mut app);

        assert_eq!(control, LoopControl::Continue);
        assert_eq!(app.spinner_frame, 1);
    }

    #[test]
    fn stripped_banner_has_content_for_tui() {
        let banner = text::strip_ansi(&banner::banner_text());
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
        assert!(content.contains('█'), "the banner art should render");
        assert!(content.contains("openbatrangs"));
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

    #[test]
    fn small_terminal_large_paste_does_not_panic() {
        use clap::Parser;
        use ratatui::backend::TestBackend;

        let backend = TestBackend::new(80, 24);
        let mut terminal = Terminal::new(backend).expect("test backend");
        let shared = Arc::new(Mutex::new(None));
        let cli = Cli::parse_from(["openbatrangs"]);
        let mut app = app::App::new(&cli, shared);
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
        let mut app = app::App::new(&cli, shared);
        terminal
            .draw(|frame| ui::ui(frame, &mut app))
            .expect("draw on a tiny terminal should not panic");
    }
}
