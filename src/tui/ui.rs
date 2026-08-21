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
    Block, Borders, List, ListItem, Paragraph, Scrollbar, ScrollbarOrientation, ScrollbarState,
    Wrap,
};
use std::sync::OnceLock;
use syntect::easy::HighlightLines;
use syntect::highlighting::{Style as SynStyle, Theme, ThemeSet};
use syntect::parsing::SyntaxSet;
use syntect::util::LinesWithEndings;
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

struct Highlighter {
    syntaxes: SyntaxSet,
    theme: Theme,
}

fn highlighter() -> &'static Highlighter {
    static HIGHLIGHTER: OnceLock<Highlighter> = OnceLock::new();
    HIGHLIGHTER.get_or_init(|| {
        let mut theme_set = ThemeSet::load_defaults();
        Highlighter {
            syntaxes: SyntaxSet::load_defaults_newlines(),
            theme: theme_set
                .themes
                .remove("base16-ocean.dark")
                .unwrap_or_default(),
        }
    })
}

fn syn_style_to_ratatui(style: SynStyle) -> Style {
    let fg = style.foreground;
    Style::default().fg(Color::Rgb(fg.r, fg.g, fg.b))
}

/// Render chat text with syntax highlighting inside ``` code fences.
fn highlighted_chat_lines(text: &str) -> Vec<Line<'static>> {
    let h = highlighter();
    let mut out = Vec::new();
    let mut in_code = false;
    let mut lang = String::new();

    for raw in text.lines() {
        let trimmed = raw.trim_start();
        if trimmed.starts_with("```") {
            if in_code {
                in_code = false;
            } else {
                in_code = true;
                lang = trimmed.trim_start_matches("```").trim().to_string();
            }
            out.push(Line::from(Span::styled(
                raw.to_string(),
                Style::default().fg(Color::DarkGray),
            )));
            continue;
        }

        if in_code {
            let syntax = h
                .syntaxes
                .find_syntax_by_extension(&lang)
                .or_else(|| h.syntaxes.find_syntax_by_name(&lang))
                .or_else(|| h.syntaxes.find_syntax_by_token(&lang))
                .unwrap_or_else(|| h.syntaxes.find_syntax_plain_text());
            let mut line_highlighter = HighlightLines::new(syntax, &h.theme);
            let mut spans: Vec<Span<'static>> = Vec::new();
            for line in LinesWithEndings::from(raw) {
                match line_highlighter.highlight_line(line, &h.syntaxes) {
                    Ok(regions) => {
                        for (style, segment) in regions {
                            spans.push(Span::styled(
                                segment.to_string(),
                                syn_style_to_ratatui(style),
                            ));
                        }
                    }
                    Err(_) => spans.push(Span::raw(raw.to_string())),
                }
            }
            if spans.is_empty() {
                spans.push(Span::raw(raw.to_string()));
            }
            out.push(Line::from(spans));
        } else {
            out.push(Line::from(Span::raw(raw.to_string())));
        }
    }
    out
}

fn render_chat_area(f: &mut ratatui::Frame, app: &mut App, area: Rect) {
    let chat_text = app.chat_text();
    app.last_chat_area = Some(area);
    let content_height = text_wrapped_height(&chat_text, chat_inner_width(area.width)) as u16;
    let visible_height = area.height.saturating_sub(2);
    let max_scroll = content_height.saturating_sub(visible_height);
    // Re-stick to the bottom when the user scrolls down to the latest line, but
    // never yank them back down while they are reading older content.
    if app.auto_scroll {
        // Already following the latest output.
    } else if app.chat_scroll_offset >= max_scroll as usize {
        app.auto_scroll = true;
    } else {
        app.chat_scroll_offset = app.chat_scroll_offset.min(max_scroll as usize);
    }
    let scroll_y = chat_scroll(app, &chat_text, area);
    let chat_lines = highlighted_chat_lines(&chat_text);
    let chat = Paragraph::new(chat_lines)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title(" openBatarangs "),
        )
        .wrap(Wrap { trim: false })
        .scroll((scroll_y, 0));
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
            ScrollbarState::new(max_scroll as usize).position(scroll_y.min(max_scroll) as usize);
        f.render_stateful_widget(scrollbar, scrollbar_area, &mut scrollbar_state);
    }
}

fn chat_inner_width(area_width: u16) -> usize {
    (area_width as usize).saturating_sub(2).max(1)
}

/// Compute a safe vertical scroll offset that never overflows `u16`.
fn chat_scroll(app: &App, chat_text: &str, area: Rect) -> u16 {
    let content_height = text_wrapped_height(chat_text, chat_inner_width(area.width)) as u16;
    let visible_height = area.height.saturating_sub(2);
    let max_scroll = content_height.saturating_sub(visible_height);
    match app.auto_scroll {
        true => max_scroll,
        false => (app.chat_scroll_offset as u16).min(max_scroll),
    }
}

fn render_status_line(f: &mut ratatui::Frame, app: &App, area: Rect) {
    let mode_suffix = match app.run_config.mode {
        AgentMode::Plan => " · plan",
        AgentMode::Chat => " · chat",
        AgentMode::Agent => "",
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
}
