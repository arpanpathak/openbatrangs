//! # TUI text helpers: ANSI stripping, wrapping, path extraction, editor launch
//!
//! The TUI renders arbitrary model/tool output, which may contain ANSI escape
//! sequences, very long lines, and file paths. This module keeps those concerns
//! in one place:
//!
//! - [`printable_line`] removes terminal escapes and control characters, so a
//!   drawn row cannot change the terminal's state.
//! - [`wrap_text_to_lines`] / [`wrap_spans`]
//!   compute visual wrapping for Unicode-safe layout.
//! - [`extract_path_from_line`] lets users click a file path in the chat.
//!
//! ## References
//!
//! - ANSI escape code standard: <https://en.wikipedia.org/wiki/ANSI_escape_code>
//! - Unicode grapheme clusters: <https://www.unicode.org/reports/tr29/>
//! - Ratatui wrapping: <https://docs.rs/ratatui/latest/ratatui/widgets/struct.Paragraph.html>

use crate::terminal_text::printable_text;
use ratatui::style::Style;
use ratatui::text::{Line, Span};
use std::path::{Path, PathBuf};
use std::process::Command;
use unicode_segmentation::UnicodeSegmentation;
use unicode_width::UnicodeWidthStr;

/// `text` without escape sequences or other control characters; a tab
/// becomes four spaces.
///
/// A terminal acts on a control character instead of showing it: `ESC[4m`
/// underlines every cell drawn after it, and a carriage return moves the
/// cursor. Text drawn in the TUI must hold none of them.
pub(crate) fn printable(text: &str) -> String {
    printable_text(text).replace('\t', "    ").replace('\n', "")
}

/// `line` with the text of every span made [`printable`].
pub(crate) fn printable_line(mut line: Line<'static>) -> Line<'static> {
    for span in &mut line.spans {
        if span.content.contains(char::is_control) {
            span.content = printable(&span.content).into();
        }
    }

    line
}

/// Wrap a multi-line string into rows that each fit within `max_text_width` cells.
///
/// Hard-wraps on grapheme boundaries so wide Unicode characters are preserved
/// and never silently clipped by the input box.
pub(crate) fn wrap_text_to_lines(text: &str, max_text_width: usize) -> Vec<String> {
    let max_text_width = max_text_width.max(1);

    text.split('\n')
        .flat_map(|line| wrap_line(line, max_text_width))
        .collect()
}

/// Wrap one source line (no newlines) into rows of at most `max_text_width`
/// cells. An empty line becomes one empty row.
fn wrap_line(line: &str, max_text_width: usize) -> Vec<String> {
    let mut rows = WrappedRows::default();

    for grapheme in line.graphemes(true) {
        rows.push(grapheme, max_text_width);
    }

    rows.finish()
}

/// Rows built so far while wrapping one line.
#[derive(Default)]
struct WrappedRows {
    /// Rows that are already full.
    full_rows: Vec<String>,
    /// The row being filled.
    row: String,
    /// Display width of `row` in cells.
    row_width: usize,
}

impl WrappedRows {
    /// Add one grapheme, starting a new row first when it does not fit.
    fn push(&mut self, grapheme: &str, max_text_width: usize) {
        let grapheme_width = grapheme.width();

        if !fits_on_row(self.row_width, grapheme_width, max_text_width) {
            self.full_rows.push(std::mem::take(&mut self.row));
            self.row_width = 0;
        }

        self.row.push_str(grapheme);
        self.row_width += grapheme_width;
    }

    /// All rows, including the last partly filled one.
    fn finish(mut self) -> Vec<String> {
        self.full_rows.push(self.row);
        self.full_rows
    }
}

/// Whether a grapheme of `grapheme_width` cells fits on a row that already
/// holds `row_width` cells. An empty row takes any grapheme, so a grapheme
/// wider than the row is never dropped.
fn fits_on_row(row_width: usize, grapheme_width: usize, max_text_width: usize) -> bool {
    row_width == 0 || row_width + grapheme_width <= max_text_width
}

/// Hard-wrap styled `spans` into rows of at most `max_text_width` cells,
/// keeping each grapheme's style. Empty input gives one empty row.
pub(crate) fn wrap_spans(
    spans: Vec<Span<'static>>,
    max_text_width: usize,
) -> Vec<Vec<Span<'static>>> {
    let mut rows = StyledRows::new(max_text_width.max(1));

    for span in spans {
        rows.push_span(&span);
    }

    rows.finish()
}

/// The display width of `spans`, in cells.
pub(crate) fn spans_width(spans: &[Span<'_>]) -> usize {
    spans.iter().map(|span| span.content.width()).sum()
}

/// The text of `spans`, without styles.
pub(crate) fn spans_text(spans: &[Span<'_>]) -> String {
    spans.iter().map(|span| span.content.as_ref()).collect()
}

/// Styled rows built so far while wrapping.
struct StyledRows {
    /// Width of a row, in cells.
    max_text_width: usize,
    /// Rows that are already full.
    full_rows: Vec<Vec<Span<'static>>>,
    /// The row being filled.
    row: Vec<Span<'static>>,
    /// Display width of `row` in cells.
    row_width: usize,
}

impl StyledRows {
    /// No rows yet.
    fn new(max_text_width: usize) -> Self {
        Self {
            max_text_width,
            full_rows: Vec::new(),
            row: Vec::new(),
            row_width: 0,
        }
    }

    /// Add every grapheme of `span`.
    fn push_span(&mut self, span: &Span<'_>) {
        for grapheme in span.content.graphemes(true) {
            self.push_grapheme(grapheme, span.style);
        }
    }

    /// Add one grapheme, starting a new row first when it does not fit.
    fn push_grapheme(&mut self, grapheme: &str, style: Style) {
        let grapheme_width = grapheme.width();

        if !fits_on_row(self.row_width, grapheme_width, self.max_text_width) {
            self.full_rows.push(std::mem::take(&mut self.row));
            self.row_width = 0;
        }

        self.append(grapheme, style);
        self.row_width += grapheme_width;
    }

    /// Append `grapheme` to the last span when it has the same style, else
    /// start a new span.
    fn append(&mut self, grapheme: &str, style: Style) {
        let last_span = self.row.last_mut().filter(|last| last.style == style);

        match last_span {
            Some(last) => last.content.to_mut().push_str(grapheme),
            None => self.row.push(Span::styled(grapheme.to_string(), style)),
        }
    }

    /// All rows, including the last partly filled one.
    fn finish(mut self) -> Vec<Vec<Span<'static>>> {
        self.full_rows.push(self.row);
        self.full_rows
    }
}

/// Split a slash-command line into its name and argument parts.
///
/// # Parameters
///
/// - `line`: raw input line, e.g. `"/model qwen2.5-coder:7b"`.
///
/// # Returns
///
/// `(name, arg)` — both empty strings when the line does not start with `/`.
pub(crate) fn split_command(line: &str) -> (&str, &str) {
    let Some(command) = line.strip_prefix('/') else {
        return ("", "");
    };

    let mut parts = command.splitn(2, char::is_whitespace);
    let name = parts.next().unwrap_or_default();
    let arg = parts.next().unwrap_or_default().trim();

    (name, arg)
}

/// Find an existing file path inside a chat line, resolving relative to `cwd`.
pub(crate) fn extract_path_from_line(line: &str, cwd: &Path) -> Option<PathBuf> {
    line.split_whitespace().find_map(|token| {
        let candidate = match token.starts_with('/') {
            true => PathBuf::from(token),
            false => cwd.join(token),
        };
        candidate.is_file().then_some(candidate)
    })
}

/// Open a file in `vim` inside a new terminal window.
///
/// # Parameters
///
/// - `path`: the file to open.
/// - `terminals`: terminal emulators to try, in order.
///
/// # Returns
///
/// The first terminal that started, or `None` when none of them could be started.
pub(crate) fn open_in_vim<'a>(path: &Path, terminals: &[&'a str]) -> Option<&'a str> {
    terminals
        .iter()
        .copied()
        .find(|terminal| vim_command(terminal, path).spawn().is_ok())
}

/// The command that runs `vim path` inside `terminal`.
fn vim_command(terminal: &str, path: &Path) -> Command {
    let mut command = Command::new(terminal);

    command
        .arg(command_separator(terminal))
        .arg("vim")
        .arg(path);

    command
}

/// The flag that introduces the program a terminal emulator should run.
fn command_separator(terminal: &str) -> &'static str {
    match terminal {
        "gnome-terminal" => "--",
        _ => "-e",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn styled_spans_wrap_and_keep_their_styles() {
        let bold = Style::new().add_modifier(ratatui::style::Modifier::BOLD);
        let spans = vec![Span::raw("ab"), Span::styled("cde", bold), Span::raw("f")];

        let rows = wrap_spans(spans, 3);

        assert_eq!(rows.len(), 2);
        assert_eq!(spans_text(&rows[0]), "abc");
        assert_eq!(rows[0][1].style, bold);
        assert_eq!(spans_text(&rows[1]), "def");
        assert_eq!(rows[1].len(), 2, "same-style graphemes share a span");
        assert_eq!(spans_width(&rows[1]), 3);
        assert_eq!(wrap_spans(Vec::new(), 0), vec![Vec::new()]);
    }

    #[test]
    fn printable_text_has_no_escapes_or_control_characters() {
        let underlined = Style::new().add_modifier(ratatui::style::Modifier::UNDERLINED);
        let line = Line::from(vec![
            Span::styled("\x1b[4mtitle\x1b[24m", underlined),
            Span::raw("50%\r100%\x07\u{9b}2J\tdone"),
            Span::raw("plain"),
        ]);

        let cleaned = printable_line(line);

        assert_eq!(spans_text(&cleaned.spans), "title50%100%2J    doneplain");
        assert_eq!(cleaned.spans[0].style, underlined);
        assert_eq!(printable("a\nb"), "ab");
    }

    #[test]
    fn wrap_text_to_lines_keeps_empty_lines_and_wide_graphemes() {
        assert_eq!(
            wrap_text_to_lines("ab\n\ncd", 1),
            vec!["a", "b", "", "c", "d"]
        );
        assert_eq!(wrap_text_to_lines("😀😀", 1), vec!["😀", "😀"]);
        assert_eq!(wrap_text_to_lines("", 0), vec![""]);
    }

    #[test]
    fn extract_path_from_line_finds_absolute_and_relative_files() -> anyhow::Result<()> {
        let workspace = crate::test_support::unique_temp_dir("openbatrangs-tui-text");
        let file = workspace.join("notes.txt");
        std::fs::write(&file, "notes")?;

        let absolute = format!("wrote {}", file.display());

        assert_eq!(
            extract_path_from_line(&absolute, &workspace),
            Some(file.clone())
        );
        assert_eq!(
            extract_path_from_line("wrote notes.txt", &workspace),
            Some(file)
        );
        assert_eq!(extract_path_from_line("no file here", &workspace), None);

        std::fs::remove_dir_all(workspace)?;

        Ok(())
    }

    #[test]
    fn open_in_vim_uses_the_first_terminal_that_starts() {
        let path = Path::new("/tmp/openbatrangs-unused.txt");
        let terminals = ["openbatrangs-missing-terminal", "true"];

        assert_eq!(open_in_vim(path, &terminals), Some("true"));
        assert_eq!(open_in_vim(path, &["openbatrangs-missing-terminal"]), None);
    }

    #[test]
    fn vim_command_uses_the_separator_each_terminal_expects() {
        let path = Path::new("/tmp/a.rs");
        let gnome = vim_command("gnome-terminal", path);
        let xterm = vim_command("xterm", path);

        let gnome_arguments: Vec<_> = gnome.get_args().collect();
        let xterm_arguments: Vec<_> = xterm.get_args().collect();

        assert_eq!(gnome_arguments, ["--", "vim", "/tmp/a.rs"]);
        assert_eq!(xterm_arguments, ["-e", "vim", "/tmp/a.rs"]);
    }

    #[test]
    fn split_command_handles_empty_and_non_slash_input() {
        assert_eq!(split_command(""), ("", ""));
        assert_eq!(split_command("plain text"), ("", ""));
        assert_eq!(split_command("/"), ("", ""));
        assert_eq!(
            split_command("/model qwen2.5-coder:7b"),
            ("model", "qwen2.5-coder:7b")
        );
    }
}
