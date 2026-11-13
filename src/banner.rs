//! Startup banner: a batarang over the wordmark, colored with a gradient from
//! cyan through violet to pink.
//!
//! [`banner_rows`] lays the art out once. The command line prints it with
//! 24-bit ANSI colors through [`banner_text`], and the TUI draws the same rows
//! as styled ratatui lines through [`banner_lines`]. Both color a character by
//! its column, so the gradient runs straight down through the batarang and the
//! wordmark.

use crate::constants::ansi::{COLOR_DIM, COLOR_RESET};
use crate::constants::banner::{BATARANG_LEFT_HALF, GRADIENT_STOPS, TAGLINE, WORDMARK};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};

/// The art rows: the batarang centered over the wordmark, with a blank row between.
pub fn banner_rows() -> Vec<String> {
    let width = art_width();
    let mut rows: Vec<String> = BATARANG_LEFT_HALF
        .iter()
        .map(|half| centered(&mirrored(half), width))
        .collect();

    rows.push(String::new());
    rows.extend(WORDMARK.iter().map(|row| row.to_string()));

    rows
}

/// The banner with 24-bit ANSI colors, for printing to a terminal.
pub fn banner_text() -> String {
    let width = art_width();
    let mut text: String = banner_rows()
        .iter()
        .map(|row| format!("{}\n", ansi_row(row, width)))
        .collect();

    text.push_str(&format!("\n {COLOR_DIM}{}{COLOR_RESET}\n", tagline()));

    text
}

/// The banner as styled lines, for drawing inside the TUI.
pub fn banner_lines() -> Vec<Line<'static>> {
    let width = art_width();
    let mut lines: Vec<Line<'static>> = banner_rows()
        .iter()
        .map(|row| styled_row(row, width))
        .collect();

    let dim = Style::default().add_modifier(Modifier::DIM);
    lines.push(Line::default());
    lines.push(Line::from(Span::styled(format!(" {}", tagline()), dim)));

    lines
}

/// Print the startup banner to stdout.
pub fn print_banner() {
    print!("{}", banner_text());
}

/// The gradient color at `column` of art that is `width` columns wide.
///
/// The position runs from 0 at the left edge to the number of gradient
/// segments at the right edge. Its whole part picks the pair of stops, and its
/// fraction says how far to blend from the first stop to the second.
pub fn gradient_color(column: usize, width: usize) -> [u8; 3] {
    let segments = GRADIENT_STOPS.len() - 1;
    let last_column = width.saturating_sub(1).max(1);
    let position = column.min(last_column) as f32 / last_column as f32 * segments as f32;

    let segment = (position as usize).min(segments - 1);
    let fraction = position - segment as f32;
    let start = GRADIENT_STOPS[segment];
    let end = GRADIENT_STOPS[segment + 1];

    std::array::from_fn(|channel| blend(start[channel], end[channel], fraction))
}

/// The value `fraction` of the way from `start` to `end`.
fn blend(start: u8, end: u8, fraction: f32) -> u8 {
    let value = f32::from(start) + (f32::from(end) - f32::from(start)) * fraction;

    value.round().clamp(0.0, 255.0) as u8
}

/// A row followed by its mirror image.
fn mirrored(half: &str) -> String {
    let right_half: String = half.chars().rev().collect();

    format!("{half}{right_half}")
}

/// `row` with enough leading spaces to center it in `width` columns.
fn centered(row: &str, width: usize) -> String {
    let padding = width.saturating_sub(row.chars().count()) / 2;

    format!("{}{row}", " ".repeat(padding))
}

/// Width of the widest wordmark row, in characters.
pub fn art_width() -> usize {
    WORDMARK
        .iter()
        .map(|row| row.chars().count())
        .max()
        .unwrap_or(1)
}

/// The tagline with the version from `Cargo.toml`.
fn tagline() -> String {
    format!("{TAGLINE} · v{}", env!("CARGO_PKG_VERSION"))
}

/// One row with each character in its gradient color, as ANSI escapes.
fn ansi_row(row: &str, width: usize) -> String {
    row.chars()
        .enumerate()
        .map(|(column, character)| {
            let [red, green, blue] = gradient_color(column, width);
            format!("\x1b[1;38;2;{red};{green};{blue}m{character}{COLOR_RESET}")
        })
        .collect()
}

/// One row with each character in its gradient color, as ratatui spans.
fn styled_row(row: &str, width: usize) -> Line<'static> {
    let spans: Vec<Span<'static>> = row
        .chars()
        .enumerate()
        .map(|(column, character)| {
            let [red, green, blue] = gradient_color(column, width);
            let style = Style::default()
                .fg(Color::Rgb(red, green, blue))
                .add_modifier(Modifier::BOLD);
            Span::styled(character.to_string(), style)
        })
        .collect();

    Line::from(spans)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batarang_rows_are_symmetric() {
        let rows = banner_rows();

        for row in &rows[..BATARANG_LEFT_HALF.len()] {
            let shape = row.trim();
            let reversed: String = shape.chars().rev().collect();
            assert_eq!(shape, reversed);
        }
    }

    #[test]
    fn rows_hold_the_batarang_a_blank_row_and_the_wordmark() {
        let rows = banner_rows();

        assert_eq!(rows.len(), BATARANG_LEFT_HALF.len() + 1 + WORDMARK.len());
        assert_eq!(rows[BATARANG_LEFT_HALF.len()], "");
        assert_eq!(rows[rows.len() - 1], WORDMARK[WORDMARK.len() - 1]);
    }

    #[test]
    fn gradient_starts_cyan_and_ends_pink() {
        let width = art_width();

        assert_eq!(gradient_color(0, width), GRADIENT_STOPS[0]);
        assert_eq!(gradient_color(width - 1, width), GRADIENT_STOPS[2]);
        assert_eq!(gradient_color(width * 5, width), GRADIENT_STOPS[2]);
    }

    #[test]
    fn gradient_passes_violet_in_the_middle() {
        assert_eq!(gradient_color(50, 101), GRADIENT_STOPS[1]);
    }

    #[test]
    fn gradient_handles_art_one_column_wide() {
        assert_eq!(gradient_color(0, 1), GRADIENT_STOPS[0]);
        assert_eq!(gradient_color(0, 0), GRADIENT_STOPS[0]);
    }

    #[test]
    fn terminal_banner_is_colored_and_shows_the_version() {
        let text = banner_text();

        assert!(text.contains("\x1b[1;38;2;0;220;255m"));
        assert!(text.contains(env!("CARGO_PKG_VERSION")));
        assert!(text.contains('█'));
    }

    #[test]
    fn tui_banner_has_one_line_per_row_and_the_tagline() {
        let lines = banner_lines();
        let last = lines[lines.len() - 1].to_string();

        assert_eq!(lines.len(), banner_rows().len() + 2);
        assert!(last.contains(TAGLINE));
    }

    #[test]
    fn tui_banner_colors_the_first_column_cyan() {
        let lines = banner_lines();
        let wordmark_row = &lines[BATARANG_LEFT_HALF.len() + 1];

        assert_eq!(
            wordmark_row.spans[0].style.fg,
            Some(Color::Rgb(0, 220, 255))
        );
    }
}
