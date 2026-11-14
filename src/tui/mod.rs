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
//! - `transcript`: the transcript blocks, kept in a bounded window and drawn by viewport.
//! - `block`: one transcript block: a user message, a step, an answer, a question, a notice.
//! - `markdown`: Markdown to styled terminal rows.
//! - `session`: the session file that holds blocks outside the window.
//! - `commands`: slash-command handlers.
//! - `text`: ANSI stripping, wrapping, path extraction, and editor launch.
//! - `worker`: background agent/chat workers.
//! - `ui`: ratatui rendering.

mod app;
mod block;
mod chat;
mod commands;
mod input;
mod markdown;
mod scroll;
mod session;
mod text;
mod transcript;
mod ui;
mod worker;

pub(super) use text::{
    extract_path_from_line, open_in_vim, split_command, text_wrapped_height, wrap_text_to_lines,
};
pub(super) use worker::{
    UiEvent, WorkerHistory, run_agent_worker, run_pull_worker, run_setup_worker, run_update_worker,
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
use crate::constants::tui::{KEYBOARD_FLAGS, MAX_BACKGROUND_EVENTS_PER_FRAME, TICK_MILLIS};
use crate::ollama::OllamaClient;
use crate::perf::{ServerPoller, TegrastatsGuard};
use crate::update::UpdateCheck;
use anyhow::{Context, Result};
use crossterm::cursor::Show;
use crossterm::event::{
    self, DisableBracketedPaste, DisableMouseCapture, EnableBracketedPaste, EnableMouseCapture,
    Event, PopKeyboardEnhancementFlags, PushKeyboardEnhancementFlags,
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
use tokio::time::{Interval, MissedTickBehavior};

/// What the event loop keeps from one frame to the next.
struct LoopState {
    /// Whether the terminal reports mouse events now.
    mouse_capture: bool,
    /// The redraw timer, made by [`redraw_ticker`].
    ticker: Interval,
}

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
/// - `update_check`: the release check to run in the background, or `None`
///   when it is turned off.
///
/// # Returns
///
/// `Ok(())` when the user exits normally, or an error if setup/teardown fails.
pub(crate) async fn run(
    cli: &Cli,
    client: &OllamaClient,
    update_check: Option<UpdateCheck>,
) -> Result<()> {
    install_panic_hook();

    let mut terminal = setup_terminal().inspect_err(|_error| restore_terminal(&mut stdout()))?;
    let mut events = event::EventStream::new();

    let result = run_app(&mut terminal, &mut events, cli, client, update_check).await;

    teardown_terminal(&mut terminal)?;
    result
}

/// Restore the terminal before a panic message is printed, so a panic does
/// not leave the shell in raw mode on the alternate screen. The previous
/// hook then prints the message as usual.
fn install_panic_hook() {
    let previous_hook = std::panic::take_hook();

    std::panic::set_hook(Box::new(move |panic_info| {
        restore_terminal(&mut stdout());
        previous_hook(panic_info);
    }));
}

/// Undo every terminal setting the TUI turns on, as far as it can. Used when
/// the normal teardown cannot run: after a panic or a failed setup. Each
/// step that was not set up does nothing harmful, and errors are ignored
/// because nothing more can be done about them.
fn restore_terminal(writer: &mut impl Write) {
    let left = execute!(
        writer,
        PopKeyboardEnhancementFlags,
        DisableBracketedPaste,
        DisableMouseCapture,
        LeaveAlternateScreen,
        Show
    );

    left.ok();
    disable_raw_mode().ok();
}

/// Configure the terminal for TUI mode: raw input, alternate screen, enhanced keys.
fn setup_terminal() -> Result<Terminal<CrosstermBackend<std::io::Stdout>>> {
    enable_raw_mode()?;

    let mut stdout = stdout();
    let entered = execute!(
        stdout,
        EnterAlternateScreen,
        EnableBracketedPaste,
        PushKeyboardEnhancementFlags(KEYBOARD_FLAGS)
    );
    entered?;

    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend).context("failed to create terminal")?;
    terminal.hide_cursor()?;

    Ok(terminal)
}

/// Restore the terminal to its original state after the TUI exits.
fn teardown_terminal(terminal: &mut Terminal<CrosstermBackend<std::io::Stdout>>) -> Result<()> {
    terminal.show_cursor()?;
    let left = execute!(
        terminal.backend_mut(),
        PopKeyboardEnhancementFlags,
        DisableBracketedPaste,
        DisableMouseCapture,
        LeaveAlternateScreen
    );
    left?;
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
    update_check: Option<UpdateCheck>,
) -> Result<()> {
    let tegrastats_shared = Arc::new(Mutex::new(None));
    let _tegrastats_guard = TegrastatsGuard::start(tegrastats_shared.clone());

    let server_poller = ServerPoller::start_for_remote(client);

    let mut app = app::App::new(cli, tegrastats_shared);
    app.server_url = client.base_url.clone();
    app.server_feed = server_poller.as_ref().map(ServerPoller::feed);
    app.refresh_model_info(client).await;

    run_loop(terminal, &mut app, events, client, update_check).await
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
    update_check: Option<UpdateCheck>,
) -> Result<()> {
    let (tx, mut rx) = mpsc::unbounded_channel::<UiEvent>();
    let mut state = LoopState {
        mouse_capture: false,
        ticker: redraw_ticker(),
    };
    let mut outcome = Ok(LoopControl::Continue);

    start_update_worker(update_check, &tx);

    while let Ok(LoopControl::Continue) = outcome {
        outcome = step(terminal, app, events, &mut rx, client, &tx, &mut state).await;
    }

    // `Ok(Exit)` ends the loop normally; an error ends it with that error.
    outcome.and(Ok(()))
}

/// Start the release check in the background, unless it is turned off.
fn start_update_worker(update_check: Option<UpdateCheck>, tx: &mpsc::UnboundedSender<UiEvent>) {
    if let Some(check) = update_check {
        tokio::spawn(run_update_worker(check, tx.clone()));
    }
}

/// Draw one frame, then handle the next input.
async fn step<W: Write>(
    terminal: &mut Terminal<CrosstermBackend<W>>,
    app: &mut app::App,
    events: &mut (impl Stream<Item = std::io::Result<Event>> + Unpin),
    rx: &mut mpsc::UnboundedReceiver<UiEvent>,
    client: &OllamaClient,
    tx: &mpsc::UnboundedSender<UiEvent>,
    state: &mut LoopState,
) -> Result<LoopControl> {
    draw_frame(terminal, app, &mut state.mouse_capture)?;

    next_control(app, events, rx, &mut state.ticker, client, tx).await
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
    ticker: &mut Interval,
    client: &OllamaClient,
    tx: &mpsc::UnboundedSender<UiEvent>,
) -> Result<LoopControl> {
    if app.should_quit {
        return Ok(LoopControl::Exit);
    }

    let input = next_input(events, rx, ticker).await;

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
    ticker: &mut Interval,
) -> LoopInput {
    tokio::select! {
        biased;
        event = events.next() => LoopInput::Terminal(event),
        event = rx.recv() => LoopInput::Background(event),
        _instant = ticker.tick() => LoopInput::Tick,
    }
}

/// The redraw timer. One timer serves the whole loop, so a steady stream of
/// events cannot keep restarting it and starve the spinner and the stats.
fn redraw_ticker() -> Interval {
    let mut ticker = tokio::time::interval(Duration::from_millis(TICK_MILLIS));
    ticker.set_missed_tick_behavior(MissedTickBehavior::Skip);

    ticker
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

/// Advance the spinner, show an update question that waited for another
/// dialog to close, and refresh the perf panel when a sample is due.
fn tick(app: &mut app::App) -> LoopControl {
    app.spinner_frame += 1;
    app.open_waiting_update();

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

        Terminal::with_options(CrosstermBackend::new(output), options).map_err(anyhow::Error::from)
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

        run_app(&mut terminal, &mut events, &cli, &refused_client()?, None).await?;

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

        run_loop(
            &mut terminal,
            &mut app,
            &mut events,
            &refused_client()?,
            None,
        )
        .await?;

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

        let result = run_loop(
            &mut terminal,
            &mut app,
            &mut events,
            &refused_client()?,
            None,
        )
        .await;

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

        run_loop(
            &mut terminal,
            &mut app,
            &mut events,
            &refused_client()?,
            None,
        )
        .await?;

        assert!(
            app.spinner_frame > 0,
            "the timer must tick while input waits"
        );
        assert!(!app.is_running, "the failed task must report back");
        assert!(app.transcript.text().contains("⚠️"));

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
        assert_eq!(app.transcript.text(), "first\nsecond\nthird");

        Ok(())
    }

    /// Send `count` log lines, one every `gap`.
    async fn send_logs_every(gap: Duration, count: usize, tx: mpsc::UnboundedSender<UiEvent>) {
        for index in 0..count {
            tokio::time::sleep(gap).await;
            tx.send(UiEvent::Log(format!("chunk {index}"))).ok();
        }
    }

    #[tokio::test]
    async fn the_timer_ticks_while_events_stream() {
        let (tx, mut rx) = mpsc::unbounded_channel();
        let mut events = futures_util::stream::pending::<std::io::Result<Event>>();
        let mut ticker = redraw_ticker();
        let event_count = 25;
        tokio::spawn(send_logs_every(Duration::from_millis(40), event_count, tx));

        let mut ticks = 0;
        let mut received = 0;

        while received < event_count {
            let input = next_input(&mut events, &mut rx, &mut ticker).await;
            let is_tick = matches!(input, LoopInput::Tick);
            ticks += usize::from(is_tick);
            received += usize::from(!is_tick);
        }

        assert!(ticks >= 3, "only {ticks} ticks while events streamed");
    }

    #[test]
    fn restoring_the_terminal_leaves_every_mode_it_turned_on() {
        let mut output = Vec::new();

        restore_terminal(&mut output);
        install_panic_hook();

        let written = String::from_utf8_lossy(&output);
        assert!(
            written.contains("\u{1b}[?1049l"),
            "leaves the alternate screen"
        );
        assert!(
            written.contains("\u{1b}[?2004l"),
            "turns bracketed paste off"
        );
        assert!(written.contains("\u{1b}[?25h"), "shows the cursor");
    }

    #[tokio::test]
    async fn a_quitting_app_exits_without_waiting_for_input() -> anyhow::Result<()> {
        let client = refused_client()?;
        let (tx, mut rx) = mpsc::unbounded_channel();
        let mut app = test_app();
        app.should_quit = true;
        let mut events = futures_util::stream::pending();

        let mut ticker = redraw_ticker();

        let control =
            next_control(&mut app, &mut events, &mut rx, &mut ticker, &client, &tx).await?;

        assert_eq!(control, LoopControl::Exit);

        Ok(())
    }

    /// The rows of the screen after drawing `app` on a test terminal.
    fn screen_rows(app: &mut app::App, width: u16, height: u16) -> anyhow::Result<Vec<String>> {
        use ratatui::backend::TestBackend;

        let mut terminal = Terminal::new(TestBackend::new(width, height))?;
        terminal.draw(|frame| ui::ui(frame, app))?;

        let symbols: Vec<&str> = terminal
            .backend()
            .buffer()
            .content()
            .iter()
            .map(|cell| cell.symbol())
            .collect();

        Ok(symbols
            .chunks(width as usize)
            .map(|row| row.concat().trim_end().to_string())
            .collect())
    }

    /// Start the update worker for a release server with `v0.3.0` and an
    /// installer that runs `installer_command`, and show its question.
    async fn app_with_update_question(
        installer_command: &str,
    ) -> anyhow::Result<(
        app::App,
        mpsc::UnboundedSender<UiEvent>,
        mpsc::UnboundedReceiver<UiEvent>,
        tokio::task::JoinHandle<()>,
    )> {
        use crate::update::fakes::{fake_check, release_server};
        use anyhow::Context;

        let check = fake_check(&release_server("v0.3.0").await, installer_command);
        let (tx, mut rx) = mpsc::unbounded_channel();
        let mut app = test_app();
        let worker = tokio::spawn(run_update_worker(check, tx.clone()));

        let question = rx.recv().await.context("the worker should ask")?;
        app.handle_event(question, &refused_client()?, &tx);

        Ok((app, tx, rx, worker))
    }

    /// Press `answer`, wait for the worker to end, and show what it sent.
    async fn answer_update_question(
        app: &mut app::App,
        answer: char,
        tx: &mpsc::UnboundedSender<UiEvent>,
        rx: &mut mpsc::UnboundedReceiver<UiEvent>,
        worker: tokio::task::JoinHandle<()>,
    ) -> anyhow::Result<()> {
        let client = refused_client()?;
        let press = event::KeyEvent::new(event::KeyCode::Char(answer), event::KeyModifiers::NONE);

        app.handle_key(press, &client, tx).await;
        worker.await?;

        while let Ok(event) = rx.try_recv() {
            app.handle_event(event, &client, tx);
        }

        Ok(())
    }

    #[tokio::test]
    async fn the_update_question_is_drawn_and_y_runs_the_installer() -> anyhow::Result<()> {
        let (mut app, tx, mut rx, worker) = app_with_update_question("echo installed").await?;

        let screen = screen_rows(&mut app, 130, 24)?.join("\n");

        assert!(
            screen
                .contains("openbatrangs 0.3.0 is available (you have 0.2.2). Install it now? y/n")
        );
        assert!(screen.contains("Press y to confirm"));

        answer_update_question(&mut app, 'y', &tx, &mut rx, worker).await?;

        assert!(!app.is_confirming());
        assert_eq!(
            app.transcript.text(),
            "Installing openbatrangs 0.3.0.\ninstalled\nRestart openbatrangs to use 0.3.0."
        );

        let screen = screen_rows(&mut app, 130, 24)?.join("\n");

        assert!(screen.contains("Restart openbatrangs to use 0.3.0."));
        Ok(())
    }

    #[tokio::test]
    async fn n_closes_the_update_question_and_shows_nothing_more() -> anyhow::Result<()> {
        let (mut app, tx, mut rx, worker) = app_with_update_question("echo installed").await?;

        answer_update_question(&mut app, 'n', &tx, &mut rx, worker).await?;

        assert!(!app.is_confirming());
        assert_eq!(app.transcript.text(), "");
        Ok(())
    }

    #[tokio::test]
    async fn a_failed_update_is_shown() -> anyhow::Result<()> {
        let (mut app, tx, mut rx, worker) = app_with_update_question("exit 1").await?;

        answer_update_question(&mut app, 'y', &tx, &mut rx, worker).await?;

        assert!(app.transcript.text().contains("⚠️ The update failed"));
        Ok(())
    }

    #[test]
    fn the_update_question_waits_for_an_open_dialog_to_close() -> anyhow::Result<()> {
        let client = refused_client()?;
        let (tx, _rx) = mpsc::unbounded_channel();
        let (response, _answer) = tokio::sync::oneshot::channel();
        let mut app = test_app();
        app.open_picker(vec!["qwen".to_string()]);

        let question = UiEvent::UpdateAvailable {
            prompt: "Install it now? y/n".to_string(),
            response,
        };
        app.handle_event(question, &client, &tx);

        assert!(app.is_showing_picker());

        app.close_modal();
        tick(&mut app);

        assert!(app.is_confirming());
        Ok(())
    }

    #[tokio::test]
    async fn the_loop_starts_the_update_check_when_it_is_on() -> anyhow::Result<()> {
        use crate::update::fakes::fake_check;

        let mut output = Vec::new();
        let mut terminal = memory_terminal(&mut output)?;
        let mut app = test_app();
        let mut events = futures_util::stream::iter(Vec::new());
        let check = fake_check("http://127.0.0.1:1/releases/latest", "exit 1");

        run_loop(
            &mut terminal,
            &mut app,
            &mut events,
            &refused_client()?,
            Some(check),
        )
        .await?;

        assert!(!app.is_confirming());
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
        tx.send(UiEvent::Agent(crate::agent::AgentEvent::Text {
            kind: crate::agent::TextKind::Answer,
            text: "HELLO from agent".to_string(),
        }))
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
