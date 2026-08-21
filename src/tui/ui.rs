//! Ratatui rendering for the TUI.

use super::app::App;
use super::{
    COMPACT_BANNER_HEIGHT, FULL_BANNER_HEIGHT, FULL_BANNER_MIN_HEIGHT, INPUT_BOX_PADDING,
    MAX_INPUT_LINES, MIN_INPUT_BOX_HEIGHT, MODEL_PICKER_HEIGHT_PERCENT, MODEL_PICKER_WIDTH_PERCENT,
    PERF_MIN_TERMINAL_HEIGHT, PERF_PANEL_HEIGHT, SUGGESTIONS_HEIGHT,
};
use crate::cli::AgentMode;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, List, ListItem, Paragraph, Wrap};

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
}

fn perf_visible(app: &App, area_height: u16) -> bool {
    app.show_perf && area_height >= PERF_MIN_TERMINAL_HEIGHT
}

fn banner_height(area_height: u16) -> u16 {
    if area_height >= FULL_BANNER_MIN_HEIGHT {
        FULL_BANNER_HEIGHT
    } else {
        COMPACT_BANNER_HEIGHT
    }
}

fn layout_chunks(area: Rect, app: &App) -> Vec<Rect> {
    Layout::default()
        .direction(Direction::Vertical)
        .margin(1)
        .constraints([
            Constraint::Length(banner_height(area.height)),
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
            Constraint::Length(input_height(app, area.width) as u16),
        ])
        .split(area)
        .to_vec()
}

fn render_banner(f: &mut ratatui::Frame, app: &App, area: Rect) {
    if area.height == 0 {
        return;
    }
    let full = area.height >= FULL_BANNER_HEIGHT;
    let lines: Vec<Line> = app
        .banner_lines
        .iter()
        .enumerate()
        .filter(|(index, _)| {
            if full {
                true
            } else {
                // Compact: wordmark (first 5 lines) + the quote (last line).
                *index < 5 || *index + 1 == app.banner_lines.len()
            }
        })
        .map(|(index, text)| {
            let style = if index < 5 {
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD)
            } else if index + 1 == app.banner_lines.len() {
                Style::default().fg(Color::Yellow)
            } else {
                Style::default().fg(Color::Magenta)
            };
            Line::from(Span::styled(text.clone(), style))
        })
        .collect();
    f.render_widget(Paragraph::new(lines), area);
}

fn render_chat_area(f: &mut ratatui::Frame, app: &mut App, area: Rect) {
    let chat_text = app.chat_text();
    app.last_chat_area = Some(area);
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
    let content_height = chat_text.lines().count().max(1) as u16;
    let visible_height = area_height.saturating_sub(2);
    if app.auto_scroll {
        content_height.saturating_sub(visible_height)
    } else {
        app.chat_scroll_offset.min(content_height as usize) as u16
    }
}

fn render_status_line(f: &mut ratatui::Frame, app: &App, area: Rect) {
    let mode_suffix = if app.run_config.mode == AgentMode::Plan {
        " · plan"
    } else {
        ""
    };
    let status_line = if app.is_running {
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

fn input_height(app: &App, width: u16) -> usize {
    let max_text_width = (width as usize)
        .saturating_sub(INPUT_BOX_PADDING + 2)
        .max(1);
    let mut text_lines = 0usize;
    for line in app.input.split('\n') {
        let char_count = line.chars().count();
        text_lines += (char_count / max_text_width).max(1);
    }
    (text_lines.min(MAX_INPUT_LINES) + INPUT_BOX_PADDING).max(MIN_INPUT_BOX_HEIGHT)
}
