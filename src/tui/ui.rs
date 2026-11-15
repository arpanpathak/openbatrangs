//! # Ratatui rendering for the TUI
//!
//! This module owns the layout and widget construction. Keeping rendering out
//! of the event loop makes the TUI testable with ratatui's `TestBackend` and
//! keeps the state transitions in `App` free of layout concerns.
//!
//! The transcript pane draws only the rows on screen: the transcript finds
//! the blocks that intersect the view and draws those, so syntax
//! highlighting stays on however long the session grows.
//!
//! ## References
//!
//! - Ratatui layout API: <https://docs.rs/ratatui/latest/ratatui/layout/index.html>
//! - Ratatui `Paragraph`: <https://docs.rs/ratatui/latest/ratatui/widgets/struct.Paragraph.html>
//! - Ratatui `Scrollbar`: <https://docs.rs/ratatui/latest/ratatui/widgets/struct.Scrollbar.html>

use super::app::{App, FocusArea};
use super::choice;
use super::scroll::{chat_text_width, chat_visible_height};
use super::selection;
use super::settings;
use super::{ScrollMode, wrap_text_to_lines};
use crate::agent::ContextUse;
use crate::banner;
use crate::cli::{AgentMode, AgentRunConfig};
use crate::constants::tui::{
    INPUT_BOX_PADDING, MAX_INPUT_LINES, MAX_SUGGESTION_ITEMS, MIN_INPUT_BOX_HEIGHT,
    MODEL_PICKER_HEIGHT_PERCENT, MODEL_PICKER_WIDTH_PERCENT, PERF_MIN_TERMINAL_HEIGHT,
    WELCOME_HINTS,
};
use ratatui::layout::{Alignment, Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{
    Block, BorderType, Borders, Clear, List, ListItem, Paragraph, Scrollbar, ScrollbarOrientation,
    ScrollbarState, Wrap,
};
use unicode_segmentation::UnicodeSegmentation;
use unicode_width::UnicodeWidthStr;

/// Cells of the context bar.
const CONTEXT_BAR_CELLS: usize = 30;

/// The keys shown on the input box's top border.
const INPUT_KEYS_HINT: &str = " Enter send · Alt+Enter or \\ Enter new line ";

/// Top-level rendering function: compose and draw the full TUI frame.
///
/// Called by ratatui's `Terminal::draw` on every frame.
///
/// # Parameters
///
/// - `frame`: the ratatui frame to render into.
/// - `app`: mutable reference to app state (used for render caching).
pub(super) fn ui(frame: &mut ratatui::Frame, app: &mut App) {
    let perf_on_screen = perf_visible(app, frame.area().height);
    app.set_perf_on_screen(perf_on_screen);

    let panel = panel_lines(app, perf_on_screen);
    let chunks = layout_chunks(frame.area(), app, panel_height(&panel, perf_on_screen));

    render_transcript(frame, app, chunks[0]);
    render_rule(frame, app.focus_area, chunks[1]);
    render_panel(frame, panel, perf_on_screen, chunks[2]);
    render_status_line(frame, app, chunks[3]);
    render_suggestions(frame, app, chunks[4]);
    render_input_box(frame, app, chunks[5]);
    render_cursor(frame, app, chunks[5]);
    render_model_picker(frame, app);
    render_confirmation(frame, app);
    settings::render_settings(frame, app);
    choice::render_choice(frame, app);
    blank_control_characters(frame);
}

/// Replace every cell that holds a control character with a space.
///
/// Widgets draw text from the model, tools and the server. A control
/// character in it would reach the terminal, which acts on it instead of
/// showing it, so none is let through, whichever widget drew it.
fn blank_control_characters(frame: &mut ratatui::Frame) {
    for cell in &mut frame.buffer_mut().content {
        if cell.symbol().contains(char::is_control) {
            cell.set_symbol(" ");
        }
    }
}

/// Whether the perf panel should be shown given current app state and terminal height.
fn perf_visible(app: &App, area_height: u16) -> bool {
    app.show_perf && area_height >= PERF_MIN_TERMINAL_HEIGHT
}

/// Height of the suggestion list based on the number of matching commands.
fn suggestions_height(app: &App) -> u16 {
    let count = app.suggestions().len();

    match count {
        0 => 0,
        _ => count.min(MAX_SUGGESTION_ITEMS) as u16 + 2,
    }
}

/// Split the terminal into the transcript, the rule under it, the stats
/// panel, the status line, the suggestions and the input box. The transcript
/// takes every row the others leave.
fn layout_chunks(area: Rect, app: &App, panel_rows: u16) -> Vec<Rect> {
    Layout::default()
        .direction(Direction::Vertical)
        .horizontal_margin(1)
        .constraints([
            Constraint::Min(3),
            Constraint::Length(1),
            Constraint::Length(panel_rows),
            Constraint::Length(1),
            Constraint::Max(suggestions_height(app)),
            Constraint::Max(input_height(app, area.width, area.height) as u16),
        ])
        .split(area)
        .to_vec()
}

/// The panel's lines: the model line, the context line once an agent step
/// has reported it, then the stats table while the stats are on screen.
fn panel_lines(app: &App, perf_on_screen: bool) -> Vec<Line<'static>> {
    let model = app.model_info_line();
    let table = match perf_on_screen {
        true => app.stats_table().lines(),
        false => Vec::new(),
    };

    std::iter::once(stat_line(&model))
        .chain(app.context.map(context_line))
        .chain(table)
        .collect()
}

/// The context line: a bar of the window, filled up to the prompt and marked
/// `│` where compaction starts, then the figures. The bar turns yellow once
/// the prompt reaches nine tenths of the compaction point.
fn context_line(context: ContextUse) -> Line<'static> {
    let cell = |tokens: u64| {
        let share = tokens as f64 / context.window.max(1) as f64;

        ((share * CONTEXT_BAR_CELLS as f64).round() as usize).min(CONTEXT_BAR_CELLS - 1)
    };
    let used = cell(context.used);
    let mark = cell(context.compact_at);
    let bar: String = (0..CONTEXT_BAR_CELLS)
        .map(|index| bar_cell(index, used, mark))
        .collect();
    let color = match context.used * 10 >= context.compact_at * 9 {
        true => Color::Yellow,
        false => Color::Cyan,
    };

    Line::from(vec![
        Span::styled("context: ", Style::default().fg(Color::Cyan)),
        Span::styled(bar, Style::default().fg(color)),
        Span::styled(
            format!(
                "  {} / {} tokens",
                thousands(context.used),
                thousands(context.window)
            ),
            Style::default().fg(Color::White),
        ),
        Span::styled(
            format!(" · compacts at {}", thousands(context.compact_at)),
            Style::default().fg(Color::DarkGray),
        ),
    ])
}

/// One cell of the context bar: the compaction mark, a filled cell up to the
/// prompt's size, or an empty one.
fn bar_cell(index: usize, used: usize, mark: usize) -> char {
    match (index == mark, index < used) {
        (true, _) => '│',
        (false, true) => '━',
        (false, false) => '─',
    }
}

/// `tokens` in thousands, padded to a fixed width so the line does not move.
fn thousands(tokens: u64) -> String {
    format!("{:>6.1}k", tokens as f64 / 1000.0)
}

/// Rows the panel needs: its lines, and the batarang's height while the
/// stats are on screen.
fn panel_height(lines: &[Line], perf_on_screen: bool) -> u16 {
    let logo_rows = match perf_on_screen {
        true => banner::batarang_height() as u16,
        false => 0,
    };

    (lines.len() as u16).max(logo_rows)
}

/// A thin rule under the transcript, cyan while the transcript has the keys.
fn render_rule(frame: &mut ratatui::Frame, focus_area: FocusArea, area: Rect) {
    let rule = "─".repeat(area.width as usize);
    let color = pane_border_color(focus_area, FocusArea::Transcript);

    frame.render_widget(Paragraph::new(rule).style(Style::default().fg(color)), area);
}

/// Draw the panel: the model line and the stats table on the left, and the
/// batarang in the bottom right corner while the stats are on screen. The
/// batarang is left out when the table would not fit beside it, so no stat is
/// cut; the model line, which starts with the model's name, may run under it.
fn render_panel(
    frame: &mut ratatui::Frame,
    lines: Vec<Line<'static>>,
    perf_on_screen: bool,
    area: Rect,
) {
    let widest = lines.iter().skip(1).map(Line::width).max().unwrap_or(0);
    let logo_columns = banner::batarang_width() + 1;
    let has_room = area.width as usize >= widest + logo_columns;
    let logo_width = match perf_on_screen && has_room {
        true => logo_columns as u16,
        false => 0,
    };
    let [text_area, logo_area] =
        Layout::horizontal([Constraint::Min(1), Constraint::Length(logo_width)]).areas(area);

    frame.render_widget(Paragraph::new(lines), text_area);
    render_logo(frame, logo_area);
}

/// One panel line styled by its parts: the figures are separated by ` · `,
/// a figure `label: value` shows its label in cyan and its value in white,
/// and a figure without a label, such as a host name, is bold magenta.
fn stat_line(line: &str) -> Line<'static> {
    let separator = Span::styled(" · ", Style::default().fg(Color::DarkGray));
    let mut spans = Vec::new();

    for (index, figure) in line.split(" · ").enumerate() {
        if index > 0 {
            spans.push(separator.clone());
        }

        spans.extend(figure_spans(figure));
    }

    Line::from(spans)
}

/// The spans of one figure: a cyan label and a white value, or the whole
/// figure in bold magenta when it has no label.
fn figure_spans(figure: &str) -> Vec<Span<'static>> {
    let Some((label, value)) = figure.split_once(": ") else {
        let heading = Style::default()
            .fg(Color::Magenta)
            .add_modifier(Modifier::BOLD);
        return vec![Span::styled(figure.to_string(), heading)];
    };

    vec![
        Span::styled(format!("{label}: "), Style::default().fg(Color::Cyan)),
        Span::styled(value.to_string(), Style::default().fg(Color::White)),
    ]
}

/// Draw the batarang at the bottom right of `area`; an area too small for it
/// stays empty.
fn render_logo(frame: &mut ratatui::Frame, area: Rect) {
    let logo_rows = banner::batarang_height() as u16;

    if area.width == 0 || area.height < logo_rows {
        return;
    }

    let logo_area = Rect {
        y: area.y + area.height - logo_rows,
        height: logo_rows,
        ..area
    };

    frame.render_widget(
        Paragraph::new(banner::batarang_lines()).alignment(Alignment::Right),
        logo_area,
    );
}

/// Draw the welcome banner while the transcript is empty, and the chat after that.
fn render_transcript(frame: &mut ratatui::Frame, app: &mut App, area: Rect) {
    if app.transcript.is_empty() {
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

    frame.render_widget(Paragraph::new(lines), middle_rect(area, width, height));
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

/// Draw the rows of the transcript that are on screen, and the scrollbar.
fn render_chat_area(frame: &mut ratatui::Frame, app: &mut App, area: Rect) {
    app.last_chat_area = Some(area);
    app.sync_transcript_scroll();

    let width = chat_text_width(area.width);
    let visible_height = chat_visible_height(area);
    let content_height = app.transcript.layout(width);

    follow_when_at_bottom(app, content_height.saturating_sub(visible_height));

    let scroll_y = chat_scroll(app, area, content_height);
    let show_focus = app.focus_area == FocusArea::Transcript;
    let lines = app
        .transcript
        .visible_lines(width, scroll_y, visible_height, show_focus);

    frame.render_widget(Paragraph::new(lines), area);
    render_chat_scrollbar(frame, area, content_height, scroll_y);
    render_selection(frame, app, area);
}

/// Draw the selected transcript text in reverse video and keep its text for
/// the clipboard.
fn render_selection(frame: &mut ratatui::Frame, app: &mut App, area: Rect) {
    let Some(selection) = app.selection else {
        return;
    };

    let text_area = Rect {
        width: chat_text_width(area.width) as u16,
        ..area
    };

    app.selected_text = selection::highlight(frame.buffer_mut(), text_area, selection);
}

/// Re-stick to the bottom when the user scrolls down to the latest line, but
/// never yank them back down while they are reading older content.
fn follow_when_at_bottom(app: &mut App, max_scroll: usize) {
    if app.chat_scroll_offset >= max_scroll {
        app.scroll_mode = ScrollMode::Follow;
    }
}

/// Draw the scrollbar in the last column when the chat is taller than the pane.
fn render_chat_scrollbar(
    frame: &mut ratatui::Frame,
    area: Rect,
    content_height: usize,
    scroll_y: usize,
) {
    let visible_height = chat_visible_height(area);
    let max_scroll = content_height.saturating_sub(visible_height);

    if !needs_scrollbar(max_scroll, area.width) {
        return;
    }

    let scrollbar_area = Rect {
        x: area.x + area.width.saturating_sub(1),
        width: 1,
        ..area
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

/// The first row on screen: the bottom while following, else the user's
/// offset, never past the bottom.
fn chat_scroll(app: &App, area: Rect, content_height: usize) -> usize {
    app.chat_scroll_for(area, content_height)
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
        "{}{}{} · {}",
        app.status,
        mode_suffix(app.run_config.mode),
        safety_suffix(&app.run_config),
        key_hint(app.focus_area)
    )
}

/// The keys that work in the focused pane.
fn key_hint(focus_area: FocusArea) -> &'static str {
    match focus_area {
        FocusArea::Input => "PgUp/PgDn scroll",
        FocusArea::Transcript => "↑↓ move · Enter fold · Tab back",
    }
}

/// Cyan for the pane that has the keys, gray for the other.
fn pane_border_color(focus_area: FocusArea, pane: FocusArea) -> Color {
    match focus_area == pane {
        true => Color::Cyan,
        false => Color::DarkGray,
    }
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
                .border_style(
                    Style::default().fg(pane_border_color(app.focus_area, FocusArea::Input)),
                )
                .title(Span::styled(
                    " › ",
                    Style::default()
                        .fg(Color::Cyan)
                        .add_modifier(Modifier::BOLD),
                ))
                .title_top(
                    Line::styled(INPUT_KEYS_HINT, Style::default().fg(Color::DarkGray))
                        .right_aligned(),
                ),
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
    let inner_width = input_inner_width(area_width);

    let rows = wrap_text_to_lines(prefix, inner_width);
    let line = rows.len().saturating_sub(1);
    let column = rows.last().map_or(0, |row| row.width());

    Some(cursor_inside_box(line, column, inner_width))
}

/// The cursor's (row, column), moved to the start of the next row when
/// `column` is at or past `inner_width`: that cell is the right border.
fn cursor_inside_box(line: usize, column: usize, inner_width: usize) -> (usize, u16) {
    if column >= inner_width {
        return (line + 1, 0);
    }

    (line, column as u16)
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
    use super::super::selection::Selection;
    use super::*;
    use crate::agent::AgentEvent;
    use crate::cli::Cli;
    use crate::perf::{ServerStats, ServerStatsFeed};
    use anyhow::Context;
    use clap::Parser;
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;
    use ratatui::layout::Position;
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
        let mut app = test_app();
        let area = Rect::new(0, 0, 80, 20);
        let content_height = 70_000 * 80;

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
    fn no_widget_draws_a_control_character() -> anyhow::Result<()> {
        let mut app = test_app();

        app.model = Some("served\x1b]0;renamed\x07model".to_string());

        app.open_choice(&crate::agent::Question {
            text: "Which?\x1b[2J".to_string(),
            options: vec!["first\r\x1b[1Afake".to_string(), "second".to_string()],
            is_multiple: false,
        });

        let screen = draw(&mut app, 100, 40)?;

        assert!(!screen.contains(char::is_control), "{screen:?}");
        assert!(
            screen.contains("served"),
            "the rest of the model name is drawn"
        );

        Ok(())
    }

    #[test]
    fn the_selection_is_read_from_the_screen() -> anyhow::Result<()> {
        let mut app = test_app();
        app.notice("first notice".to_string());
        app.notice("second notice".to_string());
        draw(&mut app, 60, 30)?;

        let area = app.last_chat_area.context("the chat was drawn")?;
        let row_of = |needle: &str, screen: &str| {
            screen
                .lines()
                .position(|line| line.contains(needle))
                .map(|row| row as u16)
        };
        let screen = draw_rows(&mut app, 60, 30)?;
        let first = row_of("first notice", &screen).context("first row")?;
        let second = row_of("second notice", &screen).context("second row")?;

        app.selection = Some(Selection {
            anchor: Position::new(area.x, first),
            cursor: Position::new(area.x + 5, second),
        });
        draw(&mut app, 60, 30)?;

        assert!(app.selected_text.starts_with("first notice"));
        assert!(app.selected_text.ends_with("\nsecond"));

        Ok(())
    }

    /// Draw the whole UI and return the screen as lines of `width` cells.
    fn draw_rows(app: &mut App, width: u16, height: u16) -> anyhow::Result<String> {
        let screen: Vec<String> = draw(app, width, height)?
            .chars()
            .collect::<Vec<char>>()
            .chunks(width as usize)
            .map(|row| row.iter().collect())
            .collect();

        Ok(screen.join("\n"))
    }

    #[test]
    fn the_panel_shows_the_model_the_stats_and_the_batarang() -> anyhow::Result<()> {
        let mut app = test_app();
        app.notice("hello".to_string());

        let screen = draw_rows(&mut app, 140, 30)?;
        let rows: Vec<&str> = screen.lines().collect();
        let logo_bottom = rows
            .iter()
            .rposition(|row| row.contains(SMALL_LOGO_BOTTOM))
            .context("the batarang is drawn")?;
        let input_top = rows
            .iter()
            .position(|row| row.contains('╭'))
            .context("the input box is drawn")?;

        assert!(screen.contains("mode: agent"));
        assert!(screen.contains("RAM free"));
        assert_eq!(
            logo_bottom + 2,
            input_top,
            "the status line sits between them"
        );

        Ok(())
    }

    #[test]
    fn the_context_bar_marks_the_prompt_and_the_compaction_point() {
        let early = context_line(ContextUse {
            used: 30_000,
            compact_at: 90_000,
            window: 120_000,
        });
        let near = context_line(ContextUse {
            used: 85_000,
            compact_at: 90_000,
            window: 120_000,
        });
        let bar = early.spans[1].content.to_string();

        assert_eq!(bar.chars().count(), CONTEXT_BAR_CELLS);
        assert_eq!(bar.chars().filter(|cell| *cell == '━').count(), 8);
        assert_eq!(bar.chars().position(|cell| cell == '│'), Some(23));
        assert_eq!(early.spans[1].style.fg, Some(Color::Cyan));
        assert_eq!(near.spans[1].style.fg, Some(Color::Yellow));
        assert_eq!(
            early.to_string().len(),
            near.to_string().len(),
            "the line keeps its width"
        );
        assert!(
            early
                .to_string()
                .ends_with("30.0k /  120.0k tokens · compacts at   90.0k")
        );
    }

    #[test]
    fn the_panel_shows_the_context_after_a_step() -> anyhow::Result<()> {
        let mut app = test_app();
        app.notice("hello".to_string());
        let before = draw(&mut app, 140, 30)?;

        app.context = Some(ContextUse {
            used: 1_000,
            compact_at: 7_000,
            window: 8_192,
        });
        let after = draw(&mut app, 140, 30)?;

        assert!(!before.contains("context:"));
        assert!(after.contains("context:"));
        Ok(())
    }

    #[test]
    fn stat_labels_values_and_hosts_have_their_own_colors() {
        let line = stat_line("thor · load: 37%");
        let styled: Vec<(&str, Option<Color>)> = line
            .spans
            .iter()
            .map(|span| (span.content.as_ref(), span.style.fg))
            .collect();

        assert_eq!(
            styled,
            [
                ("thor", Some(Color::Magenta)),
                (" · ", Some(Color::DarkGray)),
                ("load: ", Some(Color::Cyan)),
                ("37%", Some(Color::White)),
            ]
        );
    }

    #[test]
    fn a_narrow_screen_keeps_the_stats_and_leaves_the_batarang_out() -> anyhow::Result<()> {
        let mut app = test_app();
        app.notice("hello".to_string());

        let screen = draw_rows(&mut app, 60, 30)?;

        assert!(screen.contains("mode: agent"));
        assert!(!screen.contains(SMALL_LOGO_BOTTOM));

        Ok(())
    }

    #[test]
    fn the_input_box_names_the_keys_for_a_new_line() -> anyhow::Result<()> {
        let mut app = test_app();

        let screen = draw(&mut app, 120, 30)?;

        assert!(screen.contains("Enter send · Alt+Enter or \\ Enter new line"));
        assert!(!screen.contains("Shift+Enter"));
        Ok(())
    }

    #[test]
    fn only_the_input_has_a_box() -> anyhow::Result<()> {
        let mut app = test_app();
        app.notice("hello".to_string());

        let screen = draw(&mut app, 120, 30)?;

        assert_eq!(screen.matches('╭').count(), 1);
        assert!(!screen.contains(" transcript "));
        assert!(!screen.contains(" perf "));

        Ok(())
    }

    #[test]
    fn a_long_chat_gets_a_scrollbar_and_follows_at_the_bottom() -> anyhow::Result<()> {
        let mut app = test_app();
        app.notice_lines((0..100).map(|index| format!("line {index}")).collect());
        app.scroll_mode = ScrollMode::Manual;
        app.chat_scroll_offset = 1_000;

        let screen = draw(&mut app, 60, 30)?;

        assert!(screen.contains('↑'));
        assert!(screen.contains("line 99"));
        assert_eq!(app.scroll_mode, ScrollMode::Follow);

        Ok(())
    }

    #[test]
    fn code_on_screen_is_highlighted_however_long_the_session() -> anyhow::Result<()> {
        let mut app = test_app();
        app.notice("x".repeat(200_000));
        app.transcript.apply(AgentEvent::Finished {
            answer: "```rust\nfn main() {}\n```".to_string(),
            answer_was_streamed: false,
            changed_files: Vec::new(),
        });
        app.scroll_mode = ScrollMode::Follow;

        let mut terminal = Terminal::new(TestBackend::new(60, 30))?;
        terminal.draw(|frame| super::ui(frame, &mut app))?;

        let is_highlighted = terminal
            .backend()
            .buffer()
            .content()
            .iter()
            .any(|cell| matches!(cell.fg, Color::Rgb(..)));
        assert!(is_highlighted);
        assert_eq!(app.transcript.rendered_blocks(), 2);

        Ok(())
    }

    #[test]
    fn the_focused_pane_and_part_are_highlighted() -> anyhow::Result<()> {
        let mut app = test_app();
        app.notice("first".to_string());
        app.notice("second".to_string());
        app.focus_area = FocusArea::Transcript;
        app.transcript.move_focus(0);

        let mut terminal = Terminal::new(TestBackend::new(40, 24))?;
        terminal.draw(|frame| super::ui(frame, &mut app))?;

        let focused_rows = terminal
            .backend()
            .buffer()
            .content()
            .iter()
            .filter(|cell| cell.bg == Color::DarkGray)
            .count();
        assert!(focused_rows > 0);
        assert!(status_text(&app).ends_with("↑↓ move · Enter fold · Tab back"));

        let rule_color = |terminal: &Terminal<TestBackend>| {
            terminal
                .backend()
                .buffer()
                .content()
                .iter()
                .find(|cell| cell.symbol() == "─")
                .map(|cell| cell.fg)
        };
        assert_eq!(rule_color(&terminal), Some(Color::Cyan));

        app.focus_area = FocusArea::Input;
        terminal.draw(|frame| super::ui(frame, &mut app))?;
        assert_eq!(rule_color(&terminal), Some(Color::DarkGray));

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
    fn the_cursor_after_a_full_row_starts_the_next_row() {
        let mut app = test_app();
        app.input = "x".repeat(38);
        app.cursor = app.input.len();

        assert_eq!(cursor_cell(&app, Rect::new(0, 0, 40, 4)), Some((1, 2)));

        app.input = "😀".to_string();
        app.cursor = app.input.len();

        assert_eq!(cursor_cell(&app, Rect::new(0, 0, 3, 4)), Some((1, 2)));
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

    /// The `/v1/system` body thor-tigress-agent publishes.
    const THOR_SYSTEM: &str = r#"{"host":"thor","sampled_at_unix_ms":1794560000000,"gpu":{"name":"NVIDIA Thor","utilization_percent":37.0,"memory_used_bytes":90194313216,"memory_total_bytes":131072000000,"memory_is_unified":true,"power_watts":4.37,"temperature_celsius":41.0},"cpu":{"utilization_percent":12.0,"cores":14},"models":[{"id":"Gemma-4-26B-A4B-NVFP4","engine":"tensorrt-edgellm"}]}"#;

    /// vLLM metrics for two engines.
    const VLLM_METRICS: &str = "\
vllm:num_requests_running{engine=\"0\"} 2.0
vllm:num_requests_running{engine=\"1\"} 1.0
vllm:num_requests_waiting{engine=\"0\"} 1.0
vllm:kv_cache_usage_perc{engine=\"0\"} 0.34
";

    /// Width of the test terminal.
    const TERMINAL_WIDTH: u16 = 90;

    /// An app whose model server is at `server_url`, with `stats` published
    /// when the server is on another host.
    fn app_with_server(server_url: &str, stats: Option<ServerStats>) -> App {
        app_with_feed(server_url, stats.map(feed_holding))
    }

    /// An app whose model server is at `server_url`, reading server stats
    /// from `feed`.
    fn app_with_feed(server_url: &str, feed: Option<ServerStatsFeed>) -> App {
        let cli = Cli::parse_from(["openbatrangs"]);
        let mut app = App::new(&cli, Arc::new(Mutex::new(None)));
        app.server_url = server_url.to_string();
        app.server_feed = feed;

        app
    }

    /// A feed that holds `stats`, as if one poll had finished.
    fn feed_holding(stats: ServerStats) -> ServerStatsFeed {
        let feed = ServerStatsFeed::default();
        feed.store(stats);

        feed
    }

    /// The stats lines of the panel, one entry per row, drawn on a 90 by 40
    /// terminal.
    fn perf_panel_text(app: &mut App) -> anyhow::Result<Vec<String>> {
        let mut terminal = Terminal::new(TestBackend::new(TERMINAL_WIDTH, 40))?;
        terminal.draw(|frame| super::ui(frame, app))?;

        let rows: Vec<String> = terminal
            .backend()
            .buffer()
            .content()
            .chunks(usize::from(TERMINAL_WIDTH))
            .map(|row| row.iter().map(|cell| cell.symbol()).collect())
            .collect();

        Ok(panel_rows(&rows))
    }

    /// The stats table as drawn: the rows between the model line, which
    /// follows the rule under the transcript, and the status line, which
    /// starts with `▌`. The batarang's block characters, the trailing spaces
    /// and the empty rows are left out; the leading spaces stay, so the
    /// columns can be checked.
    fn panel_rows(rows: &[String]) -> Vec<String> {
        rows.iter()
            .skip_while(|row| !row.trim().starts_with('─'))
            .skip(2)
            .take_while(|row| !row.trim_start().starts_with('▌'))
            .map(|row| row.replace(is_block_character, "").trim_end().to_string())
            .filter(|row| !row.is_empty())
            .collect()
    }

    /// The bottom row of the panel's batarang, without its padding.
    const SMALL_LOGO_BOTTOM: &str = "▀ ▀▛▀▀█▛";

    /// Whether `character` is one of the block elements the batarang is
    /// drawn with, `U+2580` to `U+259F`.
    fn is_block_character(character: char) -> bool {
        ('\u{2580}'..='\u{259f}').contains(&character)
    }

    /// The column just past the end of the last `needle` in `row`.
    fn end_of(row: &str, needle: &str) -> Option<usize> {
        let start = row.rfind(needle)?;

        Some(row[..start].chars().count() + needle.chars().count())
    }

    #[test]
    fn each_figure_ends_under_the_end_of_its_heading() -> anyhow::Result<()> {
        let stats = ServerStats::from_bodies(Some(THOR_SYSTEM), None, None);
        let mut app = app_with_server("http://10.0.0.5:8079/v1", Some(stats));

        let panel = perf_panel_text(&mut app)?;
        println!("{}", panel.join("\n"));
        let [headings, thor, thor_note, local] = [&panel[0], &panel[1], &panel[2], &panel[3]];

        assert!(thor.trim_start().starts_with("thor"));
        assert!(local.trim_start().starts_with("local"));
        assert_eq!(end_of(headings, "load"), end_of(thor, "37%"));
        assert_eq!(end_of(headings, "temp"), end_of(thor, "41 °C"));
        assert_eq!(end_of(headings, "memory"), end_of(thor, "84.0/122.1 GB"));
        assert_eq!(end_of(headings, "cores"), end_of(thor, "14"));
        assert_eq!(end_of(headings, "cores"), end_of(local, "0"));
        assert_eq!(
            thor.find("NVIDIA Thor"),
            headings.find("GPU"),
            "text starts under its heading"
        );
        assert!(
            thor_note.ends_with("memory shared with the CPU"),
            "a note sits under its row"
        );
        Ok(())
    }

    #[test]
    fn engine_load_is_a_note_under_the_server_address() -> anyhow::Result<()> {
        let stats = ServerStats::from_bodies(None, Some(VLLM_METRICS), None);
        let mut app = app_with_server("http://10.0.0.5:8000/v1", Some(stats));

        let panel = perf_panel_text(&mut app)?;
        println!("{}", panel.join("\n"));

        let note = panel
            .iter()
            .find(|row| row.contains("running   3"))
            .context("the engine load is shown")?;
        assert!(
            note.trim_start().starts_with("10.0.0.5"),
            "a server without a row is named on its note"
        );
        assert!(note.ends_with("running   3 · waiting   1 · KV cache  34%"));
        Ok(())
    }

    #[test]
    fn remote_server_that_publishes_nothing_says_so() -> anyhow::Result<()> {
        let mut app = app_with_server("http://thor:8079/v1", Some(ServerStats::default()));

        let panel = perf_panel_text(&mut app)?;
        println!("{}", panel.join("\n"));

        assert!(
            panel
                .iter()
                .any(|row| row.ends_with("no server stats published"))
        );
        Ok(())
    }

    #[test]
    fn remote_server_before_the_first_poll_shows_waiting() -> anyhow::Result<()> {
        let feed = ServerStatsFeed::default();
        let mut app = app_with_feed("http://thor:8079/v1", Some(feed));

        let panel = perf_panel_text(&mut app)?;
        println!("{}", panel.join("\n"));

        assert!(
            panel
                .iter()
                .any(|row| row.ends_with("waiting for server stats"))
        );
        Ok(())
    }

    #[test]
    fn a_local_server_shows_only_the_local_row() -> anyhow::Result<()> {
        let mut app = app_with_server("http://127.0.0.1:11434", None);

        let panel = perf_panel_text(&mut app)?;
        println!("{}", panel.join("\n"));

        assert_eq!(panel.len(), 3, "headings, the local row and its note");
        assert!(panel[1].trim_start().starts_with("local"));
        assert!(
            !panel[2].trim_start().starts_with("local"),
            "the note continues the row"
        );
        assert!(panel[2].contains("RAM free"));
        Ok(())
    }
}
