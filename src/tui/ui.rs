//! Ratatui rendering for the TUI.

use super::app::App;
use super::{text_wrapped_height, wrap_text_to_lines};
use crate::cli::AgentMode;
use crate::constants::tui::{
    COMPACT_BANNER_HEIGHT, FULL_BANNER_MIN_TERMINAL_HEIGHT, FULL_BANNER_MIN_WIDTH,
    INPUT_BOX_PADDING, MAX_INPUT_LINES, MAX_SUGGESTION_ITEMS, MIN_INPUT_BOX_HEIGHT,
    MODEL_PICKER_HEIGHT_PERCENT, MODEL_PICKER_WIDTH_PERCENT, PERF_MAX_PANEL_HEIGHT,
    PERF_MIN_TERMINAL_HEIGHT, PERF_PANEL_HEIGHT, SMALL_BANNER_HEIGHT,
};
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{
    Block, Borders, Clear, List, ListItem, Paragraph, Scrollbar, ScrollbarOrientation,
    ScrollbarState, Wrap,
};
use unicode_segmentation::UnicodeSegmentation;
use unicode_width::UnicodeWidthStr;

pub(super) fn ui(f: &mut ratatui::Frame, app: &mut App) {
    let chunks = layout_chunks(f.area(), app);
    render_banner(f, app, chunks[0]);
    render_chat_area(f, app, chunks[1]);
    render_status_line(f, app, chunks[2]);
    if perf_visible(app, f.area().height) {
        render_perf_panel(f, app, chunks[3]);
    }
    render_suggestions(f, app, chunks[4]);
    render_input_box(f, app, chunks[5]);
    render_cursor(f, app, chunks[5]);
    render_model_picker(f, app);
    render_confirmation(f, app);
}

fn perf_visible(app: &App, area_height: u16) -> bool {
    app.show_perf && area_height >= PERF_MIN_TERMINAL_HEIGHT
}

fn banner_height(area_height: u16) -> u16 {
    if area_height >= FULL_BANNER_MIN_TERMINAL_HEIGHT {
        COMPACT_BANNER_HEIGHT
    } else {
        SMALL_BANNER_HEIGHT
    }
}

fn suggestions_height(app: &App) -> u16 {
    let count = app.suggestions().len();
    match count {
        0 => 0,
        _ => count.min(MAX_SUGGESTION_ITEMS) as u16 + 2,
    }
}

fn perf_height(app: &App, width: u16) -> u16 {
    let max_text_width = (width as usize).saturating_sub(2).max(1);
    let rows = text_wrapped_height(&app.perf_lines().join("\n"), max_text_width);
    (rows as u16 + 2).clamp(PERF_PANEL_HEIGHT, PERF_MAX_PANEL_HEIGHT)
}

fn layout_chunks(area: Rect, app: &App) -> Vec<Rect> {
    Layout::default()
        .direction(Direction::Vertical)
        .margin(1)
        .constraints([
            Constraint::Max(banner_height(area.height)),
            Constraint::Min(3),
            Constraint::Length(1),
            Constraint::Max(if perf_visible(app, area.height) {
                perf_height(app, area.width)
            } else {
                0
            }),
            Constraint::Max(suggestions_height(app)),
            Constraint::Max(input_height(app, area.width, area.height) as u16),
        ])
        .split(area)
        .to_vec()
}

fn render_banner(f: &mut ratatui::Frame, app: &App, area: Rect) {
    if area.height == 0 {
        return;
    }
    let full = area.height >= COMPACT_BANNER_HEIGHT && area.width >= FULL_BANNER_MIN_WIDTH;
    let mut lines: Vec<Line> = app
        .banner_lines
        .iter()
        .enumerate()
        .filter(|(index, _)| {
            if full {
                true
            } else {
                // Small terminal: title + quote only.
                *index == 0 || *index + 1 == app.banner_lines.len()
            }
        })
        .map(|(index, text)| {
            let style = if index == 0 {
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD)
            } else if index + 1 == app.banner_lines.len() {
                Style::default().fg(Color::Yellow)
            } else {
                Style::default().fg(Color::Magenta)
            };
            Line::from(Span::styled(
                truncate_to_width(text, area.width as usize),
                style,
            ))
        })
        .collect();
    lines.push(Line::from(Span::styled(
        truncate_to_width(&app.model_info_line(), area.width as usize),
        Style::default().fg(Color::Green),
    )));
    if full {
        if let Some(prompt) = app.prompt_line() {
            lines.push(Line::from(Span::styled(
                truncate_to_width(&format!("Prompt: {prompt}"), area.width as usize),
                Style::default().fg(Color::Yellow),
            )));
        }
    }
    f.render_widget(Paragraph::new(lines), area);
}

fn render_chat_area(f: &mut ratatui::Frame, app: &mut App, area: Rect) {
    let chat_text = app.chat_text();
    app.last_chat_area = Some(area);
    let inner_width = chat_inner_width(area.width);
    let content_height = text_wrapped_height(&chat_text, inner_width);
    let visible_height = area.height.saturating_sub(2) as usize;
    let max_scroll = content_height.saturating_sub(visible_height);
    // Re-stick to the bottom when the user scrolls down to the latest line, but
    // never yank them back down while they are reading older content.
    if app.auto_scroll {
        // Already following the latest output.
    } else if app.chat_scroll_offset >= max_scroll {
        app.auto_scroll = true;
    } else {
        app.chat_scroll_offset = app.chat_scroll_offset.min(max_scroll);
    }
    let scroll_y = chat_scroll(app, area, content_height);
    let chat_lines = app.chat_render.lines(&chat_text);
    let chat = Paragraph::new(chat_lines.to_vec())
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title(" openBatarangs "),
        )
        .wrap(Wrap { trim: false })
        .scroll((scroll_y.min(u16::MAX as usize) as u16, 0));
    f.render_widget(chat, area);

    if max_scroll > 0 && area.width >= 3 {
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
        let mut scrollbar_state =
            ScrollbarState::new(max_scroll).position(scroll_y.min(max_scroll));
        f.render_stateful_widget(scrollbar, scrollbar_area, &mut scrollbar_state);
    }
}

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
    match app.auto_scroll {
        true => max_scroll,
        false => app.chat_scroll_offset.min(max_scroll),
    }
}

fn render_status_line(f: &mut ratatui::Frame, app: &App, area: Rect) {
    let mode_suffix = match app.run_config.mode {
        AgentMode::Plan => " · plan",
        AgentMode::Chat => " · chat",
        AgentMode::Agent => "",
    };
    let status_line = if app.pending_confirmation.is_some() {
        "❓ y = confirm · n / Esc = abort".to_string()
    } else if app.is_running {
        let spin = app.spinner();
        if app.last_action.is_empty() {
            format!("{spin} {}{} — thinking...", app.status, mode_suffix)
        } else {
            format!("{spin} {}{} — {}", app.status, mode_suffix, app.last_action)
        }
    } else if app.picker.is_some() {
        "↑↓ select · Enter confirm · Esc cancel".to_string()
    } else if !app.task_queue.is_empty() {
        format!("ready — {} queued", app.task_queue.len())
    } else {
        format!("{}{} · PgUp/PgDn scroll", app.status, mode_suffix)
    };
    let status_style = if app.is_running {
        Style::default().fg(Color::Yellow)
    } else {
        Style::default().fg(Color::Green)
    };
    let status_line = truncate_to_width(&status_line, area.width as usize);
    f.render_widget(Paragraph::new(status_line).style(status_style), area);
}

/// Truncate a string to `max_width` display cells, adding an ellipsis when cut.
fn truncate_to_width(text: &str, max_width: usize) -> String {
    if text.width() <= max_width {
        return text.to_string();
    }
    let mut out = String::new();
    let mut width = 0usize;
    for grapheme in text.graphemes(true) {
        let grapheme_width = grapheme.width();
        if width + grapheme_width + 1 > max_width {
            out.push('…');
            return out;
        }
        out.push_str(grapheme);
        width += grapheme_width;
    }
    out
}

fn render_perf_panel(f: &mut ratatui::Frame, app: &App, area: Rect) {
    let content = app.perf_lines().join("\n");
    let panel = Paragraph::new(content)
        .block(Block::default().borders(Borders::ALL).title(" ⚡ perf "))
        .wrap(Wrap { trim: false })
        .style(Style::default().fg(Color::Cyan));
    f.render_widget(panel, area);
}

fn render_suggestions(f: &mut ratatui::Frame, app: &App, area: Rect) {
    if area.height == 0 {
        return;
    }
    let suggestions = app.suggestions();
    let selected = app.selected.min(suggestions.len().saturating_sub(1));
    let suggestion_items: Vec<ListItem> = suggestions
        .iter()
        .enumerate()
        .map(|(index, suggestion)| {
            let style = if index == selected {
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
    let lines: Vec<Line> = wrap_text_to_lines(&app.input, input_inner_width(area.width))
        .into_iter()
        .map(Line::from)
        .collect();
    let input_paragraph = Paragraph::new(lines)
        .block(Block::default().borders(Borders::ALL).title(" input "))
        .scroll((input_scroll_y(app, area), 0));
    f.render_widget(input_paragraph, area);
}

fn input_inner_width(area_width: u16) -> usize {
    (area_width as usize).saturating_sub(2).max(1)
}

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

fn render_cursor(f: &mut ratatui::Frame, app: &App, area: Rect) {
    let Some((cursor_line, column)) = input_cursor_position(app, area.width) else {
        return;
    };
    let scroll_y = input_scroll_y(app, area) as usize;
    let x = area.x + 1 + column;
    let y = area.y + 1 + cursor_line.saturating_sub(scroll_y) as u16;
    if y < area.y + area.height {
        f.set_cursor_position((x, y));
    }
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

fn render_confirmation(f: &mut ratatui::Frame, app: &App) {
    let Some(pending) = &app.pending_confirmation else {
        return;
    };
    let area = centered_rect(60, 35, f.area());
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
    f.render_widget(Clear, area);
    f.render_widget(
        Block::default()
            .borders(Borders::ALL)
            .title(" confirm ")
            .border_style(Style::default().fg(Color::Yellow)),
        area,
    );
    f.render_widget(
        Paragraph::new(lines).wrap(Wrap { trim: false }),
        area.inner(ratatui::layout::Margin {
            horizontal: 1,
            vertical: 1,
        }),
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
    use ratatui::backend::TestBackend;
    use ratatui::Terminal;
    use std::sync::{Arc, Mutex};

    fn test_backend() -> std::sync::Arc<dyn crate::engine::InferenceBackend> {
        let client = crate::ollama::OllamaClient::new("http://localhost:11434").unwrap();
        std::sync::Arc::new(crate::engine::OllamaBackend::new(client))
    }

    #[test]
    fn input_height_grows_for_wrapped_long_lines() {
        let cli = Cli::parse_from(["openbatrangs"]);
        let shared = Arc::new(Mutex::new(None));
        let mut app = App::new(&cli, shared, test_backend());
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
        let mut app = App::new(&cli, shared, test_backend());
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
        let mut app = App::new(&cli, shared, test_backend());
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

        app.auto_scroll = true;
        let bottom = chat_scroll(&app, area, content_height);
        assert!(
            bottom > u16::MAX as usize,
            "true bottom scroll must stay in usize space, got {bottom}"
        );

        app.auto_scroll = false;
        app.chat_scroll_offset = 10_000;
        assert_eq!(chat_scroll(&app, area, content_height), 10_000);
    }

    #[test]
    fn scroll_offset_cap_never_wraps_for_ratatui() {
        let cli = Cli::parse_from(["openbatrangs"]);
        let shared = Arc::new(Mutex::new(None));
        let mut app = App::new(&cli, shared, test_backend());
        app.live = "x".repeat(70_000 * 80);
        let area = Rect {
            x: 0,
            y: 0,
            width: 80,
            height: 20,
        };
        let chat_text = app.chat_text();
        let content_height = text_wrapped_height(&chat_text, chat_inner_width(area.width));
        app.auto_scroll = true;
        let scroll_y = chat_scroll(&app, area, content_height);
        let ratatui_scroll = scroll_y.min(u16::MAX as usize) as u16;
        assert_eq!(ratatui_scroll, u16::MAX);
    }

    #[test]
    fn confirmation_popup_renders_prompt() {
        use crate::tui::app::PendingConfirmation;

        let cli = Cli::parse_from(["openbatrangs"]);
        let shared = Arc::new(Mutex::new(None));
        let mut app = App::new(&cli, shared, test_backend());
        let (response_tx, _response_rx) = tokio::sync::oneshot::channel();
        app.pending_confirmation = Some(PendingConfirmation {
            prompt: "write file 'a.txt'?".to_string(),
            response: response_tx,
        });

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
