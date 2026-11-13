//! # Code highlighting and line styles for the transcript
//!
//! Fenced code in answers, reasoning and tool output is highlighted with
//! [syntect]. Only blocks on screen are drawn, and each block's lines are
//! cached, so highlighting stays on however long the session grows.
//!
//! Notice lines get a style from their leading marker, so tool calls,
//! warnings and results each read differently.
//!
//! ## References
//!
//! - Syntect: <https://github.com/trishume/syntect>
//! - Ratatui `Line`/`Span`: <https://docs.rs/ratatui/latest/ratatui/text/index.html>

use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use std::sync::OnceLock;
use syntect::easy::HighlightLines;
use syntect::highlighting::{Style as SynStyle, Theme, ThemeSet};
use syntect::parsing::SyntaxSet;

/// Lazily-initialized syntect highlighter shared across all rendering calls.
struct Highlighter {
    /// Loaded syntax definitions used to resolve language hints to grammars.
    syntaxes: SyntaxSet,
    /// Color theme applied during syntax highlighting (base16-ocean.dark).
    theme: Theme,
}

/// Return the process-wide highlighter singleton, initializing it on first call.
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

/// Convert a syntect `Style` (foreground color) to a ratatui `Style`.
fn syn_style_to_ratatui(style: SynStyle) -> Style {
    let foreground = style.foreground;

    Style::default().fg(Color::Rgb(foreground.r, foreground.g, foreground.b))
}

/// Resolve the syntax definition for a code-fence language hint.
fn code_syntax<'a>(
    syntax_highlighter: &'a Highlighter,
    lang: &str,
) -> &'a syntect::parsing::SyntaxReference {
    syntax_highlighter
        .syntaxes
        .find_syntax_by_extension(lang)
        .or_else(|| syntax_highlighter.syntaxes.find_syntax_by_name(lang))
        .or_else(|| syntax_highlighter.syntaxes.find_syntax_by_token(lang))
        .unwrap_or_else(|| syntax_highlighter.syntaxes.find_syntax_plain_text())
}

/// The lines of `code`, one span list per line, highlighted for `lang` when
/// `highlight` is set. Both forms hold the same text, so they wrap to the
/// same number of rows. Tabs become four spaces.
pub(super) fn code_lines(code: &str, lang: &str, highlight: bool) -> Vec<Vec<Span<'static>>> {
    let code = code.replace('\t', "    ");

    match highlight {
        true => highlighted_code(&code, lang),
        false => code.lines().map(plain_code_line).collect(),
    }
}

/// One line of code as a single unstyled span.
fn plain_code_line(line: &str) -> Vec<Span<'static>> {
    vec![Span::raw(line.to_string())]
}

/// Every line of `code`, highlighted with the grammar for `lang`.
fn highlighted_code(code: &str, lang: &str) -> Vec<Vec<Span<'static>>> {
    let syntax_highlighter = highlighter();
    let mut lines = HighlightLines::new(
        code_syntax(syntax_highlighter, lang),
        &syntax_highlighter.theme,
    );

    code.lines()
        .map(|line| code_line(line, &mut lines, &syntax_highlighter.syntaxes))
        .collect()
}

/// Highlight one line of code. A line syntect cannot parse is shown as is.
fn code_line(
    line: &str,
    highlighter: &mut HighlightLines<'_>,
    syntaxes: &SyntaxSet,
) -> Vec<Span<'static>> {
    let with_newline = format!("{line}\n");
    let unhighlighted = plain_code_line(line);

    highlighter
        .highlight_line(&with_newline, syntaxes)
        .map(|regions| styled_spans(&regions))
        .unwrap_or(unhighlighted)
}

/// Turn syntect's styled regions into ratatui spans, without the newline.
fn styled_spans(regions: &[(SynStyle, &str)]) -> Vec<Span<'static>> {
    regions
        .iter()
        .map(|(style, segment)| (style, segment.trim_end_matches('\n')))
        .filter(|(_style, segment)| !segment.is_empty())
        .map(|(style, segment)| Span::styled(segment.to_string(), syn_style_to_ratatui(*style)))
        .collect()
}

/// Style of a notice line by its leading marker, so agent activity reads as a
/// transcript. The first matching marker wins; a line with no marker uses the
/// default style.
const MARKER_STYLES: &[(&str, Style)] = &[
    (
        "🔧",
        Style::new().fg(Color::Cyan).add_modifier(Modifier::BOLD),
    ),
    (
        "💭",
        Style::new()
            .fg(Color::DarkGray)
            .add_modifier(Modifier::ITALIC),
    ),
    ("✅", Style::new().fg(Color::Green)),
    ("⚠️", Style::new().fg(Color::Yellow)),
    ("⛔", Style::new().fg(Color::Yellow)),
    ("▶️", Style::new().fg(Color::Cyan)),
];

/// One notice line, styled by its leading marker (see [`MARKER_STYLES`]).
pub(super) fn notice_line(raw: &str) -> Line<'static> {
    let trimmed = raw.trim_start();
    let style = MARKER_STYLES
        .iter()
        .find(|(marker, _style)| trimmed.starts_with(marker))
        .map(|(_marker, style)| *style)
        .unwrap_or_default();

    Line::from(Span::styled(raw.to_string(), style))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tui::text::spans_text;

    #[test]
    fn highlighting_keeps_the_text_of_every_line() {
        let code = "// comment\nfn main() {\n\tprintln!(\"hi\");\n\n}";

        let highlighted = code_lines(code, "rust", true);
        let plain = code_lines(code, "rust", false);

        assert_eq!(highlighted.len(), 5);
        let texts: Vec<String> = highlighted.iter().map(|line| spans_text(line)).collect();
        let plain_texts: Vec<String> = plain.iter().map(|line| spans_text(line)).collect();
        assert_eq!(texts, plain_texts);
        assert_eq!(texts[2], "    println!(\"hi\");");
        assert!(highlighted[1].len() > 1, "a code line gets several colors");
        assert!(highlighted[3].is_empty());
    }

    #[test]
    fn an_unknown_language_is_plain_text() {
        let lines = code_lines("x = 1", "no-such-language", true);

        assert_eq!(spans_text(&lines[0]), "x = 1");
    }

    #[test]
    fn notice_lines_are_styled_by_their_marker() {
        let tool = notice_line("  🔧 write file");
        let warning = notice_line("⛔ Cancelled.");
        let plain = notice_line("just text");

        assert_eq!(tool.spans[0].style.fg, Some(Color::Cyan));
        assert_eq!(warning.spans[0].style.fg, Some(Color::Yellow));
        assert_eq!(plain.spans[0].style, Style::default());
    }
}
