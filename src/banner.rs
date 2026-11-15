//! Startup banner: the logo over the wordmark, colored with a gradient from
//! cyan through violet to pink.
//!
//! The logo is `assets/logo.svg` drawn in quarter-block characters (see
//! [`LogoArt`]). Each of its columns takes the color the SVG's gradient has at
//! that column of the canvas, so the drawing shows the logo's own colors. The
//! wordmark spreads the same gradient over its own width.
//!
//! The command line prints the banner with 24-bit ANSI colors through
//! [`banner_text`], and the TUI draws it as styled ratatui lines through
//! [`banner_lines`]. The TUI's stats panel shows a smaller drawing of the logo
//! through [`batarang_lines`].

use crate::constants::ansi::{COLOR_DIM, COLOR_RESET};
use crate::constants::banner::{GRADIENT_STOPS, LOGO, LogoArt, SMALL_LOGO, TAGLINE, WORDMARK};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};

/// The banner with 24-bit ANSI colors, for printing to a terminal.
pub fn banner_text() -> String {
    let padding = " ".repeat(logo_padding());
    let wordmark_width = art_width();
    let logo_column_color = |column| logo_color(&LOGO, column);
    let wordmark_column_color = |column| gradient_color(column, wordmark_width);

    let logo: String = LOGO
        .rows
        .iter()
        .map(|row| format!("{padding}{}\n", ansi_row(row, logo_column_color)))
        .collect();
    let wordmark: String = WORDMARK
        .iter()
        .map(|row| format!("{}\n", ansi_row(row, wordmark_column_color)))
        .collect();
    let tagline = tagline();

    format!("{logo}\n{wordmark}\n {COLOR_DIM}{tagline}{COLOR_RESET}\n")
}

/// The banner as styled lines, for drawing inside the TUI.
pub fn banner_lines() -> Vec<Line<'static>> {
    let padding = logo_padding();
    let wordmark_width = art_width();
    let dim = Style::default().add_modifier(Modifier::DIM);

    let logo = LOGO
        .rows
        .iter()
        .map(|row| padded(styled_row(row, |column| logo_color(&LOGO, column)), padding));
    let wordmark = WORDMARK
        .iter()
        .map(|row| styled_row(row, |column| gradient_color(column, wordmark_width)));
    let tagline = Line::from(Span::styled(format!(" {}", tagline()), dim));

    logo.chain([Line::default()])
        .chain(wordmark)
        .chain([Line::default(), tagline])
        .collect()
}

/// The small logo, for the corner of the TUI's stats panel.
pub fn batarang_lines() -> Vec<Line<'static>> {
    SMALL_LOGO
        .rows
        .iter()
        .map(|row| styled_row(row, |column| logo_color(&SMALL_LOGO, column)))
        .collect()
}

/// Width of the small logo, in columns.
pub fn batarang_width() -> usize {
    logo_width(&SMALL_LOGO)
}

/// Height of the small logo, in rows.
pub fn batarang_height() -> usize {
    SMALL_LOGO.rows.len()
}

/// Print the startup banner to stdout.
pub fn print_banner() {
    print!("{}", banner_text());
}

/// Width of `art`, in columns.
fn logo_width(art: &LogoArt) -> usize {
    art.rows.first().map_or(0, |row| row.chars().count())
}

/// Spaces in front of the large logo that center it over the wordmark.
fn logo_padding() -> usize {
    art_width().saturating_sub(logo_width(&LOGO)) / 2
}

/// The color the SVG's gradient has at `column` of `art`.
fn logo_color(art: &LogoArt, column: usize) -> [u8; 3] {
    gradient_color(art.first_column + column, art.canvas_columns)
}

/// `line` with `padding` spaces in front.
fn padded(mut line: Line<'static>, padding: usize) -> Line<'static> {
    line.spans.insert(0, Span::raw(" ".repeat(padding)));

    line
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

/// One row with each character in the color `color` gives its column, as
/// ANSI escapes.
fn ansi_row(row: &str, color: impl Fn(usize) -> [u8; 3]) -> String {
    row.chars()
        .enumerate()
        .map(|(column, character)| {
            let [red, green, blue] = color(column);
            format!("\x1b[1;38;2;{red};{green};{blue}m{character}{COLOR_RESET}")
        })
        .collect()
}

/// One row with each character in the color `color` gives its column, as
/// ratatui spans.
fn styled_row(row: &str, color: impl Fn(usize) -> [u8; 3]) -> Line<'static> {
    let spans: Vec<Span<'static>> = row
        .chars()
        .enumerate()
        .map(|(column, character)| {
            let [red, green, blue] = color(column);
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
    fn each_logo_has_rows_of_one_width_inside_its_canvas() {
        for art in [&LOGO, &SMALL_LOGO] {
            let width = logo_width(art);

            assert!(art.rows.iter().all(|row| row.chars().count() == width));
            assert!(art.first_column + width <= art.canvas_columns);
        }
    }

    #[test]
    fn the_logo_takes_its_colors_from_the_svg_canvas() {
        let last = logo_width(&LOGO) - 1;

        assert_eq!(logo_color(&LOGO, 0), gradient_color(5, 60));
        assert_ne!(
            logo_color(&LOGO, 0),
            GRADIENT_STOPS[0],
            "the blade starts right of the canvas edge, so it is not pure cyan"
        );
        assert_eq!(logo_color(&LOGO, last), gradient_color(52, 60));
    }

    #[test]
    fn tui_banner_holds_the_logo_a_blank_row_and_the_wordmark() {
        let lines = banner_lines();
        let padding = " ".repeat(logo_padding());

        assert_eq!(lines.len(), LOGO.rows.len() + 1 + WORDMARK.len() + 2);
        assert_eq!(lines[0].to_string(), format!("{padding}{}", LOGO.rows[0]));
        assert_eq!(lines[LOGO.rows.len()].to_string(), "");
        assert_eq!(lines[LOGO.rows.len() + 1].to_string(), WORDMARK[0]);
    }

    #[test]
    fn the_panel_logo_is_the_small_drawing() {
        let lines = batarang_lines();

        assert_eq!(lines.len(), batarang_height());
        assert_eq!(lines[0].to_string(), SMALL_LOGO.rows[0]);
        assert_eq!(lines[0].width(), batarang_width());
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
    fn tui_banner_ends_with_the_tagline() {
        let lines = banner_lines();
        let last = lines[lines.len() - 1].to_string();

        assert!(last.contains(TAGLINE));
    }

    #[test]
    fn tui_banner_colors_the_first_column_cyan() {
        let lines = banner_lines();
        let wordmark_row = &lines[LOGO.rows.len() + 1];

        assert_eq!(
            wordmark_row.spans[0].style.fg,
            Some(Color::Rgb(0, 220, 255))
        );
    }
}
