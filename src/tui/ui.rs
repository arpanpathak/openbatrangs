//! # Ratatui rendering for the TUI
//!
//! This module owns the layout and widget construction. Keeping rendering out
//! of the event loop makes the TUI testable with ratatui's `TestBackend` and
//! keeps the state transitions in `App` free of layout concerns.
//!
//! Large chat logs are rendered with a raw paragraph (no syntect pass) so
//! streaming stays responsive on Jetson-class devices; smaller logs keep code
//! syntax highlighting.
//!
//! ## References
//!
//! - Ratatui layout API: <https://docs.rs/ratatui/latest/ratatui/layout/index.html>
//! - Ratatui `Paragraph`: <https://docs.rs/ratatui/latest/ratatui/widgets/struct.Paragraph.html>
//! - Ratatui `Scrollbar`: <https://docs.rs/ratatui/latest/ratatui/widgets/struct.Scrollbar.html>

use super::app::App;
use super::{ScrollMode, text_wrapped_height, wrap_text_to_lines};
use crate::banner;
use crate::cli::{AgentMode, AgentRunConfig};
use crate::constants::tui::{
    HEADER_HEIGHT, INPUT_BOX_PADDING, MAX_INPUT_LINES, MAX_SUGGESTION_ITEMS, MIN_INPUT_BOX_HEIGHT,
    MODEL_PICKER_HEIGHT_PERCENT, MODEL_PICKER_WIDTH_PERCENT, PERF_MAX_PANEL_HEIGHT,
    PERF_MIN_TERMINAL_HEIGHT, PERF_PANEL_HEIGHT, RAW_CHAT_RENDER_THRESHOLD, WELCOME_HINTS,
};
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{
    Block, BorderType, Borders, Clear, List, ListItem, Paragraph, Scrollbar, ScrollbarOrientation,
    ScrollbarState, Wrap,
};
use unicode_segmentation::UnicodeSegmentation;
use unicode_width::UnicodeWidthStr;

/// Top-level rendering function: compose and draw the full TUI frame.
///
/// Called by ratatui's `Terminal::draw` on every frame.
///
/// # Parameters
///
/// - `frame`: the ratatui frame to render into.
/// - `app`: mutable reference to app state (used for render caching).
pub(super) fn ui(frame: &mut ratatui::Frame, app: &mut App) {
    let chunks = layout_chunks(frame.area(), app);

    render_header(frame, app, chunks[0]);
    render_transcript(frame, app, chunks[1]);
    render_status_line(frame, app, chunks[2]);

    if perf_visible(app, frame.area().height) {
        render_perf_panel(frame, app, chunks[3]);
    }

    render_suggestions(frame, app, chunks[4]);
    render_input_box(frame, app, chunks[5]);
    render_cursor(frame, app, chunks[5]);
    render_model_picker(frame, app);
    render_confirmation(frame, app);
}

/// Whether the perf panel should be shown given current app state and terminal height.
fn perf_visible(app: &App, area_height: u16) -> bool {
    app.show_perf && area_height >= PERF_MIN_TERMINAL_HEIGHT
}

/// Height of the top header: one extra line while a task prompt is set.
fn header_height(app: &App) -> u16 {
    match app.prompt_line() {
        Some(_) => HEADER_HEIGHT + 1,
        None => HEADER_HEIGHT,
    }
}

/// Height of the suggestion list based on the number of matching commands.
fn suggestions_height(app: &App) -> u16 {
    let count = app.suggestions().len();

    match count {
        0 => 0,
        _ => count.min(MAX_SUGGESTION_ITEMS) as u16 + 2,
    }
}

/// Dynamic perf panel height based on wrapped text content.
fn perf_height(app: &App, width: u16) -> u16 {
    let max_text_width = (width as usize).saturating_sub(2).max(1);
    let rows = text_wrapped_height(&app.perf_lines().join("\n"), max_text_width);

    (rows as u16 + 2).clamp(PERF_PANEL_HEIGHT, PERF_MAX_PANEL_HEIGHT)
}

/// Split the terminal area into vertical chunks for each UI region.
fn layout_chunks(area: Rect, app: &App) -> Vec<Rect> {
    let perf_rows = match perf_visible(app, area.height) {
        true => perf_height(app, area.width),
        false => 0,
    };

    Layout::default()
        .direction(Direction::Vertical)
        .margin(1)
        .constraints([
            Constraint::Max(header_height(app)),
            Constraint::Min(3),
            Constraint::Length(1),
            Constraint::Max(perf_rows),
            Constraint::Max(suggestions_height(app)),
            Constraint::Max(input_height(app, area.width, area.height) as u16),
        ])
        .split(area)
        .to_vec()
}

/// Render the compact top header: identity, model, and the task prompt.
fn render_header(frame: &mut ratatui::Frame, app: &App, area: Rect) {
    if area.height == 0 {
        return;
    }

    let inner_width = area.width.saturating_sub(2) as usize;
    let prompt_line = app
        .prompt_line()
        .map(|prompt| header_prompt_line(&prompt, inner_width));
    let lines: Vec<Line> = std::iter::once(header_model_line(app, inner_width))
        .chain(prompt_line)
        .collect();

    frame.render_widget(
        Paragraph::new(lines).block(
            Block::default()
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .border_style(Style::default().fg(Color::DarkGray))
                .title(Span::styled(
                    " openbatrangs ",
                    Style::default()
                        .fg(Color::Magenta)
                        .add_modifier(Modifier::BOLD),
                )),
        ),
        area,
    );
}

/// The model, mode, and server summary line shown inside the header.
fn header_model_line(app: &App, inner_width: usize) -> Line<'static> {
    Line::from(Span::styled(
        truncate_to_width(&app.model_info_line(), inner_width),
        Style::default().fg(Color::DarkGray),
    ))
}

/// The task prompt line shown under the model line in the header.
fn header_prompt_line(prompt: &str, inner_width: usize) -> Line<'static> {
    Line::from(Span::styled(
        truncate_to_width(&format!("› {prompt}"), inner_width),
        Style::default().fg(Color::Yellow),
    ))
}

/// Shared frame for the transcript pane.
fn transcript_block() -> Block<'static> {
    Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(Color::DarkGray))
        .title(Span::styled(
            " transcript ",
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        ))
}

/// Draw the welcome banner while the transcript is empty, and the chat after that.
fn render_transcript(frame: &mut ratatui::Frame, app: &mut App, area: Rect) {
    if app.transcript_is_empty() {
        return render_welcome(frame, area);
    }

    render_chat_area(frame, app, area);
}

/// Draw the batarang banner and the starting hints in the middle of the empty
/// transcript pane.
fn render_welcome(frame: &mut ratatui::Frame, area: Rect) {
    let hint_style = Style::default().fg(Color::DarkGray);
    let mut lines = banner::banner_lines();
    lines.push(Line::default());
    lines.extend(
        WELCOME_HINTS
            .iter()
            .map(|hint| Line::from(Span::styled(format!(" {hint}"), hint_style))),
    );

    let widest = lines.iter().map(Line::width).max().unwrap_or(0);
    let width = widest.min(u16::MAX as usize) as u16;
    let height = lines.len().min(u16::MAX as usize) as u16;
    let inner = transcript_block().inner(area);

    frame.render_widget(transcript_block(), area);
    frame.render_widget(Paragraph::new(lines), middle_rect(inner, width, height));
}

/// A rectangle of `width` by `height` cells in the middle of `area`, cut to fit.
fn middle_rect(area: Rect, width: u16, height: u16) -> Rect {
    let width = width.min(area.width);
    let height = height.min(area.height);

    Rect {
        x: area.x + (area.width - width) / 2,
        y: area.y + (area.height - height) / 2,
        width,
        height,
    }
}

/// Render the chat scrollback area with optional syntax highlighting and scrollbar.
fn render_chat_area(frame: &mut ratatui::Frame, app: &mut App, area: Rect) {
    let chat_text = app.chat_text();
    app.last_chat_area = Some(area);

    let content_height = text_wrapped_height(&chat_text, chat_inner_width(area.width));
    let visible_height = area.height.saturating_sub(2) as usize;

    follow_when_at_bottom(app, content_height.saturating_sub(visible_height));

    let scroll_y = chat_scroll(app, area, content_height);
    let chat = chat_paragraph(app, chat_text)
        .block(transcript_block())
        .wrap(Wrap { trim: false })
        .scroll((scroll_y.min(u16::MAX as usize) as u16, 0));

    frame.render_widget(chat, area);
    render_chat_scrollbar(frame, area, content_height, scroll_y);
}

/// Re-stick to the bottom when the user scrolls down to the latest line, but
/// never yank them back down while they are reading older content.
fn follow_when_at_bottom(app: &mut App, max_scroll: usize) {
    if app.chat_scroll_offset >= max_scroll {
        app.scroll_mode = ScrollMode::Follow;
    }
}

/// The chat text as a paragraph, syntax-highlighted unless the log is huge.
///
/// Huge logs skip the syntect code-highlighting path entirely. Building
/// styled `Vec<Line>`s for every streaming token is what made generation
/// freeze on low-power devices once the chat grew long.
fn chat_paragraph(app: &mut App, chat_text: String) -> Paragraph<'static> {
    if chat_text.len() > RAW_CHAT_RENDER_THRESHOLD {
        return Paragraph::new(chat_text);
    }

    Paragraph::new(app.chat_render.lines(&chat_text).to_vec())
}

/// Draw the scrollbar on the right border when the chat is taller than the pane.
fn render_chat_scrollbar(
    frame: &mut ratatui::Frame,
    area: Rect,
    content_height: usize,
    scroll_y: usize,
) {
    let visible_height = area.height.saturating_sub(2) as usize;
    let max_scroll = content_height.saturating_sub(visible_height);

    if !needs_scrollbar(max_scroll, area.width) {
        return;
    }

    let scrollbar_area = Rect {
        x: area.x + area.width.saturating_sub(2),
        y: area.y + 1,
        width: 1,
        height: area.height.saturating_sub(2),
    };
    let scrollbar = Scrollbar::default()
        .orientation(ScrollbarOrientation::VerticalRight)
        .begin_symbol(Some("↑"))
        .end_symbol(Some("↓"))
        .track_symbol(Some("│"))
        .thumb_symbol("█");
    let mut scrollbar_state = ScrollbarState::new(content_height)
        .position(scroll_y.min(max_scroll))
        .viewport_content_length(visible_height);

    frame.render_stateful_widget(scrollbar, scrollbar_area, &mut scrollbar_state);
}

/// Whether the chat needs a scrollbar: there is content below the view and
/// the pane is wide enough to draw one.
fn needs_scrollbar(max_scroll: usize, area_width: u16) -> bool {
    max_scroll > 0 && area_width >= 3
}

/// Inner width of the chat area in character cells (excluding borders).
fn chat_inner_width(area_width: u16) -> usize {
    (area_width as usize).saturating_sub(2).max(1)
}

/// Compute a safe vertical scroll offset as `usize`.
///
/// Content can be much taller than `u16::MAX` after long streamed output wraps
/// into hundreds of thousands of visual rows. Keeping the math in `usize`
/// prevents the previous `as u16` truncation that made scrolling stop/glitch
/// once the chat grew past 65,535 rows. Ratatui only accepts `u16` offsets, so
/// callers cap the final value before passing it to `Paragraph::scroll`.
fn chat_scroll(app: &App, area: Rect, content_height: usize) -> usize {
    let visible_height = area.height.saturating_sub(2) as usize;
    let max_scroll = content_height.saturating_sub(visible_height);

    match app.scroll_mode {
        ScrollMode::Follow => max_scroll,
        ScrollMode::Manual => app.chat_scroll_offset.min(max_scroll),
    }
}

/// Render the status bar showing the current state, mode, and key hints.
fn render_status_line(frame: &mut ratatui::Frame, app: &App, area: Rect) {
    let status_line = truncate_to_width(&format!("▌ {}", status_text(app)), area.width as usize);

    frame.render_widget(Paragraph::new(status_line).style(status_style(app)), area);
}

/// The status bar text for the app's current state.
fn status_text(app: &App) -> String {
    let has_queued_tasks = !app.task_queue.is_empty();

    match (
        app.is_confirming(),
        app.is_running,
        app.is_showing_picker(),
        has_queued_tasks,
    ) {
        (true, _, _, _) => "❓ y = confirm · n / Esc = abort".to_string(),
        (false, true, _, _) => running_status(app),
        (false, false, true, _) => "↑↓ select · Enter confirm · Esc cancel".to_string(),
        (false, false, false, true) => queued_status(app),
        (false, false, false, false) => idle_status(app),
    }
}

/// Status while a task runs: spinner, status, mode, and the current action.
fn running_status(app: &App) -> String {
    let action = match app.last_action.is_empty() {
        true => "thinking...",
        false => app.last_action.as_str(),
    };

    format!(
        "{} {}{} — {action}",
        app.spinner(),
        app.status,
        mode_suffix(app.run_config.mode)
    )
}

/// Status while idle with tasks waiting in the queue.
fn queued_status(app: &App) -> String {
    format!(
        "ready{} — {} queued",
        safety_suffix(&app.run_config),
        app.task_queue.len()
    )
}

/// Status while idle with nothing queued.
fn idle_status(app: &App) -> String {
    format!(
        "{}{}{} · PgUp/PgDn scroll",
        app.status,
        mode_suffix(app.run_config.mode),
        safety_suffix(&app.run_config)
    )
}

/// The mode shown after the status; agent mode is the default and shows nothing.
fn mode_suffix(mode: AgentMode) -> &'static str {
    match mode {
        AgentMode::Plan => " · plan",
        AgentMode::Chat => " · chat",
        AgentMode::Agent => "",
    }
}

/// A warning shown when writes are blocked or run without confirmation.
fn safety_suffix(config: &AgentRunConfig) -> &'static str {
    match (config.is_read_only, config.should_confirm) {
        (true, _) => " · 🔒 read-only",
        (false, false) => " · ⚠️ yolo",
        (false, true) => "",
    }
}

/// Yellow while a task runs, green otherwise.
fn status_style(app: &App) -> Style {
    match app.is_running {
        true => Style::default().fg(Color::Yellow),
        false => Style::default().fg(Color::Green),
    }
}

/// Truncate a string to `max_width` display cells, adding an ellipsis when cut.
fn truncate_to_width(text: &str, max_width: usize) -> String {
    if text.width() <= max_width {
        return text.to_string();
    }

    // Keep whole graphemes while they leave one cell for the ellipsis.
    let mut width = 0;
    let kept: String = text
        .graphemes(true)
        .take_while(|grapheme| {
            width += grapheme.width();
            width < max_width
        })
        .collect();

    format!("{kept}…")
}

/// Render the GPU/CPU/RAM performance panel with live stats.
fn render_perf_panel(frame: &mut ratatui::Frame, app: &App, area: Rect) {
    let content = app.perf_lines().join("\n");

    let panel = Paragraph::new(content)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .border_style(Style::default().fg(Color::DarkGray))
                .title(" ⚡ perf "),
        )
        .wrap(Wrap { trim: false })
        .style(Style::default().fg(Color::Cyan));

    frame.render_widget(panel, area);
}

/// Render the slash-command autocomplete suggestion list.
fn render_suggestions(frame: &mut ratatui::Frame, app: &App, area: Rect) {
    if area.height == 0 {
        return;
    }

    let suggestions = app.suggestions();
    let selected = app.selected.min(suggestions.len().saturating_sub(1));
    let suggestion_items: Vec<ListItem> = suggestions
        .into_iter()
        .enumerate()
        .map(|(index, suggestion)| {
            ListItem::new(Line::from(Span::styled(
                suggestion,
                suggestion_style(index == selected),
            )))
        })
        .collect();

    frame.render_widget(
        List::new(suggestion_items).block(
            Block::default()
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .border_style(Style::default().fg(Color::DarkGray))
                .title(" suggestions "),
        ),
        area,
    );
}

/// Bold cyan for the highlighted suggestion, plain for the others.
fn suggestion_style(is_selected: bool) -> Style {
    match is_selected {
        true => Style::default()
            .fg(Color::Cyan)
            .add_modifier(Modifier::BOLD),
        false => Style::default(),
    }
}

/// Render the user input box with wrapped text and scrolling.
fn render_input_box(frame: &mut ratatui::Frame, app: &App, area: Rect) {
    let lines: Vec<Line> = wrap_text_to_lines(&app.input, input_inner_width(area.width))
        .into_iter()
        .map(Line::from)
        .collect();

    let input_paragraph = Paragraph::new(lines)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .border_style(Style::default().fg(Color::Cyan))
                .title(Span::styled(
                    " › ",
                    Style::default()
                        .fg(Color::Cyan)
                        .add_modifier(Modifier::BOLD),
                )),
        )
        .scroll((input_scroll_y(app, area), 0));

    frame.render_widget(input_paragraph, area);
}

/// Inner width of the input box in character cells (excluding borders).
fn input_inner_width(area_width: u16) -> usize {
    (area_width as usize).saturating_sub(2).max(1)
}

/// Vertical scroll offset for the input box so the cursor stays visible.
fn input_scroll_y(app: &App, area: Rect) -> u16 {
    let Some((cursor_line, _)) = input_cursor_position(app, area.width) else {
        return 0;
    };

    let inner_height = area.height.saturating_sub(2).max(1) as usize;

    cursor_line.saturating_sub(inner_height.saturating_sub(1)) as u16
}

/// Return the visual (row, column) of the text cursor within the wrapped input.
fn input_cursor_position(app: &App, area_width: u16) -> Option<(usize, u16)> {
    let prefix = app.input.get(..app.cursor)?;

    let rows = wrap_text_to_lines(prefix, input_inner_width(area_width));
    let line = rows.len().saturating_sub(1);
    let column = rows.last().map(|row| row.width() as u16).unwrap_or(0);

    Some((line, column))
}

/// Position the terminal cursor at the user's text insertion point in the input box.
fn render_cursor(frame: &mut ratatui::Frame, app: &App, area: Rect) {
    if let Some(position) = cursor_cell(app, area) {
        frame.set_cursor_position(position);
    }
}

/// The terminal cell of the text cursor, or `None` when it is outside the box.
fn cursor_cell(app: &App, area: Rect) -> Option<(u16, u16)> {
    let (cursor_line, column) = input_cursor_position(app, area.width)?;

    let scroll_y = input_scroll_y(app, area) as usize;
    let cursor_x = area.x + 1 + column;
    let cursor_y = area.y + 1 + cursor_line.saturating_sub(scroll_y) as u16;

    (cursor_y < area.y + area.height).then_some((cursor_x, cursor_y))
}

/// Render the model picker popup overlay when active.
fn render_model_picker(frame: &mut ratatui::Frame, app: &App) {
    let Some(picker) = app.picker() else {
        return;
    };

    let area = centered_rect(
        MODEL_PICKER_WIDTH_PERCENT,
        MODEL_PICKER_HEIGHT_PERCENT,
        frame.area(),
    );
    let items: Vec<ListItem> = picker
        .models
        .iter()
        .enumerate()
        .map(|(index, model)| {
            ListItem::new(Line::from(Span::styled(
                model.clone(),
                picker_style(index == picker.selected),
            )))
        })
        .collect();

    frame.render_widget(
        List::new(items)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_type(BorderType::Rounded)
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

/// Black on cyan for the highlighted model, plain for the others.
fn picker_style(is_selected: bool) -> Style {
    match is_selected {
        true => Style::default()
            .fg(Color::Black)
            .bg(Color::Cyan)
            .add_modifier(Modifier::BOLD),
        false => Style::default(),
    }
}

/// Render the tool-confirmation popup overlay when active.
fn render_confirmation(frame: &mut ratatui::Frame, app: &App) {
    let pending = match &app.modal {
        Some(super::app::ActiveModal::Confirm(pending)) => pending,
        _ => return,
    };

    let area = centered_rect(60, 35, frame.area());
    let lines = vec![
        Line::from(Span::styled(
            "❓ Confirmation required",
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        )),
        Line::from(""),
        Line::from(Span::raw(pending.prompt.clone())),
        Line::from(""),
        Line::from(Span::styled(
            "Press y to confirm · n / Esc to abort",
            Style::default().fg(Color::Cyan),
        )),
    ];

    frame.render_widget(Clear, area);
    frame.render_widget(
        Block::default()
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .title(" confirm ")
            .border_style(Style::default().fg(Color::Yellow)),
        area,
    );
    frame.render_widget(
        Paragraph::new(lines).wrap(Wrap { trim: false }),
        area.inner(ratatui::layout::Margin {
            horizontal: 1,
            vertical: 1,
        }),
    );
}

/// Compute a centered rectangle as a percentage of the parent area.
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

/// Compute the input box height based on wrapped text content and available space.
fn input_height(app: &App, width: u16, area_height: u16) -> usize {
    let text_lines = wrap_text_to_lines(&app.input, input_inner_width(width)).len();
    let desired = (text_lines.min(MAX_INPUT_LINES) + INPUT_BOX_PADDING).max(MIN_INPUT_BOX_HEIGHT);
    let available = (area_height as usize)
        .saturating_sub(4)
        .max(MIN_INPUT_BOX_HEIGHT);

    desired.min(available)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli::Cli;
    use clap::Parser;
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;
    use std::sync::{Arc, Mutex};

    #[test]
    fn input_height_grows_for_wrapped_long_lines() {
        let cli = Cli::parse_from(["openbatrangs"]);
        let shared = Arc::new(Mutex::new(None));
        let mut app = App::new(&cli, shared);
        let width = 20u16;
        let area_height = 40u16;

        let single_line_height = input_height(&app, width, area_height);
        app.input = "a".repeat(100);
        let wrapped_height = input_height(&app, width, area_height);
        assert!(wrapped_height > single_line_height);
        assert!(wrapped_height <= MAX_INPUT_LINES + INPUT_BOX_PADDING);
    }

    #[test]
    fn input_height_handles_wide_unicode() {
        let cli = Cli::parse_from(["openbatrangs"]);
        let shared = Arc::new(Mutex::new(None));
        let mut app = App::new(&cli, shared);
        let width = 20u16;
        let area_height = 40u16;
        app.input = "😀".repeat(30);
        let height = input_height(&app, width, area_height);
        assert!(height > MIN_INPUT_BOX_HEIGHT);
    }

    #[test]
    fn wrap_text_to_lines_never_loses_graphemes() {
        let text = "héllo 世界 abcdefghijklmnopqrstuvwxyz";
        let rows = wrap_text_to_lines(text, 10);
        assert_eq!(rows.concat(), text);
        assert!(rows.len() > 1);
    }

    #[test]
    fn chat_scroll_handles_content_taller_than_u16_max() {
        let cli = Cli::parse_from(["openbatrangs"]);
        let shared = Arc::new(Mutex::new(None));
        let mut app = App::new(&cli, shared);
        // One wrapped line with more visual rows than u16::MAX. Before the fix,
        // the `as u16` truncation made the scroll offset wrap to a tiny value.
        app.live = "x".repeat(70_000 * 80);
        let area = Rect {
            x: 0,
            y: 0,
            width: 80,
            height: 20,
        };
        let chat_text = app.chat_text();
        let content_height = text_wrapped_height(&chat_text, chat_inner_width(area.width));

        app.scroll_mode = ScrollMode::Follow;
        let bottom = chat_scroll(&app, area, content_height);
        assert!(
            bottom > u16::MAX as usize,
            "true bottom scroll must stay in usize space, got {bottom}"
        );

        app.scroll_mode = ScrollMode::Manual;
        app.chat_scroll_offset = 10_000;
        assert_eq!(chat_scroll(&app, area, content_height), 10_000);
    }

    #[test]
    fn scroll_offset_cap_never_wraps_for_ratatui() {
        let cli = Cli::parse_from(["openbatrangs"]);
        let shared = Arc::new(Mutex::new(None));
        let mut app = App::new(&cli, shared);
        app.live = "x".repeat(70_000 * 80);
        let area = Rect {
            x: 0,
            y: 0,
            width: 80,
            height: 20,
        };
        let chat_text = app.chat_text();
        let content_height = text_wrapped_height(&chat_text, chat_inner_width(area.width));
        app.scroll_mode = ScrollMode::Follow;
        let scroll_y = chat_scroll(&app, area, content_height);
        let ratatui_scroll = scroll_y.min(u16::MAX as usize) as u16;
        assert_eq!(ratatui_scroll, u16::MAX);
    }

    /// An app with default settings.
    fn test_app() -> App {
        let cli = Cli::parse_from(["openbatrangs"]);
        App::new(&cli, Arc::new(Mutex::new(None)))
    }

    /// Draw the whole UI on a `width` by `height` test terminal and return
    /// the screen text, row by row.
    fn draw(app: &mut App, width: u16, height: u16) -> anyhow::Result<String> {
        let mut terminal = Terminal::new(TestBackend::new(width, height))?;

        terminal.draw(|frame| super::ui(frame, app))?;

        let screen = terminal
            .backend()
            .buffer()
            .content()
            .iter()
            .map(|cell| cell.symbol())
            .collect();

        Ok(screen)
    }

    #[test]
    fn the_header_shows_the_task_prompt() -> anyhow::Result<()> {
        let mut app = test_app();
        app.current_prompt = Some("write a softmax kernel in CUDA Oxide please".to_string());
        app.log.push("You: write a softmax kernel".to_string());

        let screen = draw(&mut app, 40, 24)?;

        assert!(screen.contains("› write a softmax kernel"));
        assert!(screen.contains('…'));

        Ok(())
    }

    #[test]
    fn a_header_without_rows_draws_nothing() -> anyhow::Result<()> {
        let app = test_app();
        let mut terminal = Terminal::new(TestBackend::new(20, 5))?;

        terminal.draw(|frame| render_header(frame, &app, Rect::new(0, 0, 20, 0)))?;

        assert!(
            !terminal.backend().buffer().content()[0]
                .symbol()
                .contains('╭')
        );

        Ok(())
    }

    #[test]
    fn a_long_chat_gets_a_scrollbar_and_follows_at_the_bottom() -> anyhow::Result<()> {
        let mut app = test_app();
        app.log
            .push_lines((0..100).map(|index| format!("line {index}")).collect());
        app.scroll_mode = ScrollMode::Manual;
        app.chat_scroll_offset = 1_000;

        let screen = draw(&mut app, 60, 30)?;

        assert!(screen.contains('↑'));
        assert!(screen.contains("line 99"));
        assert_eq!(app.scroll_mode, ScrollMode::Follow);

        Ok(())
    }

    #[test]
    fn a_huge_chat_is_drawn_without_highlighting() -> anyhow::Result<()> {
        let mut app = test_app();
        app.log.push("x".repeat(RAW_CHAT_RENDER_THRESHOLD));
        app.live = "tail of the reply".to_string();
        app.scroll_mode = ScrollMode::Follow;

        let screen = draw(&mut app, 60, 30)?;

        assert!(screen.contains("tail of the reply"));

        Ok(())
    }

    #[test]
    fn suggestions_and_the_model_picker_are_drawn() -> anyhow::Result<()> {
        let mut app = test_app();
        app.input = "/mo".to_string();
        app.cursor = app.input.len();
        app.selected = 1;
        app.open_picker(vec![
            "qwen2.5-coder:7b".to_string(),
            "llama3.2:3b".to_string(),
        ]);

        let screen = draw(&mut app, 80, 30)?;

        assert!(screen.contains("/models"));
        assert!(screen.contains("/mode"));
        assert!(screen.contains("Select model"));
        assert!(screen.contains("llama3.2:3b"));

        Ok(())
    }

    #[test]
    fn the_status_line_describes_each_state() {
        let mut app = test_app();
        assert_eq!(status_text(&app), "ready · PgUp/PgDn scroll");

        app.run_config.mode = AgentMode::Plan;
        app.run_config.is_read_only = true;
        assert_eq!(
            status_text(&app),
            "ready · plan · 🔒 read-only · PgUp/PgDn scroll"
        );

        app.run_config.mode = AgentMode::Chat;
        app.run_config.is_read_only = false;
        app.run_config.should_confirm = false;
        app.task_queue.push_back(("next".to_string(), None));
        assert_eq!(status_text(&app), "ready · ⚠️ yolo — 1 queued");

        app.open_picker(vec!["model".to_string()]);
        assert_eq!(status_text(&app), "↑↓ select · Enter confirm · Esc cancel");

        app.is_running = true;
        app.status = "running".to_string();
        assert!(status_text(&app).ends_with("running · chat — thinking..."));

        app.last_action = "🔧 read file".to_string();
        assert!(status_text(&app).ends_with("— 🔧 read file"));
        assert_eq!(status_style(&app).fg, Some(Color::Yellow));

        let (response, _receiver) = tokio::sync::oneshot::channel();
        app.open_confirmation("write?".to_string(), response);
        assert_eq!(status_text(&app), "❓ y = confirm · n / Esc = abort");
    }

    #[test]
    fn the_cursor_is_hidden_when_it_cannot_be_placed() {
        let mut app = test_app();
        app.input = "abc".to_string();
        app.cursor = 3;

        assert_eq!(cursor_cell(&app, Rect::new(0, 0, 10, 1)), None);
        assert_eq!(cursor_cell(&app, Rect::new(0, 0, 10, 3)), Some((4, 1)));

        app.input = "😀".to_string();
        app.cursor = 1;

        assert_eq!(cursor_cell(&app, Rect::new(0, 0, 10, 3)), None);
        assert_eq!(input_scroll_y(&app, Rect::new(0, 0, 10, 3)), 0);
    }

    #[test]
    fn truncate_to_width_keeps_room_for_the_ellipsis() {
        assert_eq!(truncate_to_width("short", 10), "short");
        assert_eq!(truncate_to_width("abcdef", 4), "abc…");
        assert_eq!(truncate_to_width("😀😀😀", 4), "😀…");
    }

    #[test]
    fn confirmation_popup_renders_prompt() {
        let cli = Cli::parse_from(["openbatrangs"]);
        let shared = Arc::new(Mutex::new(None));
        let mut app = App::new(&cli, shared);
        let (response_tx, _response_rx) = tokio::sync::oneshot::channel();
        app.open_confirmation("write file 'a.txt'?".to_string(), response_tx);

        let backend = TestBackend::new(80, 24);
        let mut terminal = Terminal::new(backend).expect("test backend");
        terminal
            .draw(|frame| super::ui(frame, &mut app))
            .expect("draw with confirmation popup should not panic");
        let buffer = terminal.backend().buffer();
        let content = buffer
            .content()
            .iter()
            .map(|cell| cell.symbol())
            .collect::<String>();
        assert!(content.contains("Confirmation required"));
        assert!(content.contains("write file 'a.txt'?"));
        assert!(content.contains("Press y to confirm"));
    }
}
