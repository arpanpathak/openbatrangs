//! # TUI text helpers: ANSI stripping, wrapping, path extraction, editor launch
//!
//! The TUI renders arbitrary model/tool output, which may contain ANSI escape
//! sequences, very long lines, and file paths. This module keeps those concerns
//! in one place:
//!
//! - [`strip_ansi`] removes terminal escapes so the chat log stores plain text.
//! - [`text_wrapped_height`] / [`wrap_text_to_lines`] compute visual wrapping
//!   for Unicode-safe layout.
//! - [`extract_path_from_line`] lets users click a file path in the chat.
//!
//! ## References
//!
//! - ANSI escape code standard: <https://en.wikipedia.org/wiki/ANSI_escape_code>
//! - Unicode grapheme clusters: <https://www.unicode.org/reports/tr29/>
//! - Ratatui wrapping: <https://docs.rs/ratatui/latest/ratatui/widgets/struct.Paragraph.html>

use std::path::{Path, PathBuf};
use std::process::Command;
use std::str::Chars;
use unicode_segmentation::UnicodeSegmentation;
use unicode_width::UnicodeWidthStr;

/// Remove ANSI escape sequences (CSI, OSC) from a string, returning plain text.
///
/// # Parameters
///
/// - `text`: input string potentially containing terminal escape sequences.
///
/// # Returns
///
/// The input with all ANSI/OSC escape sequences removed.
pub(crate) fn strip_ansi(text: &str) -> String {
    let mut plain = String::new();
    let mut characters = text.chars();

    while let Some(character) = characters.next() {
        keep_unless_escape(&mut plain, character, &mut characters);
    }

    plain
}

/// Append `character` to `plain`, or skip the escape sequence it starts.
fn keep_unless_escape(plain: &mut String, character: char, characters: &mut Chars<'_>) {
    if character == '\x1b' {
        skip_escape_sequence(characters);
        return;
    }

    plain.push(character);
}

/// Consume and discard one ANSI escape sequence after the initial ESC character.
///
/// Other single-character escape sequences need no further characters.
fn skip_escape_sequence(characters: &mut Chars<'_>) {
    match characters.next() {
        Some(']') => skip_operating_system_command(characters),
        Some('[') => skip_control_sequence(characters),
        Some(_) | None => {}
    }
}

/// Skip an OSC sequence (`ESC ] ...`), which ends at BEL or at the string
/// terminator `ESC \`.
fn skip_operating_system_command(characters: &mut Chars<'_>) {
    let terminator = characters.find(|character| matches!(character, '\x07' | '\x1b'));

    if terminator == Some('\x1b') {
        characters.next();
    }
}

/// Skip a CSI sequence (`ESC [ ...`), which ends at a final byte in `0x40..=0x7E`.
fn skip_control_sequence(characters: &mut Chars<'_>) {
    characters.find(|character| ('\u{40}'..='\u{7e}').contains(character));
}

/// Number of visual rows a line of `display_width` cells occupies when wrapped
/// at `max_text_width` cells. Empty lines still occupy one row.
pub(crate) fn wrapped_line_count(display_width: usize, max_text_width: usize) -> usize {
    let max_text_width = max_text_width.max(1);

    match display_width {
        0 => 1,
        _ => display_width.div_ceil(max_text_width),
    }
}

/// Number of visual rows `text` occupies when each source line is wrapped at
/// `max_text_width` cells.
pub(crate) fn text_wrapped_height(text: &str, max_text_width: usize) -> usize {
    text.split('\n')
        .map(|line| wrapped_line_count(line.width(), max_text_width))
        .sum::<usize>()
        .max(1)
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

/// Map every visual row of `text` back to its source line index.
///
/// This keeps mouse hit-testing and scrollbar math correct when chat lines wrap.
pub(crate) fn chat_visual_line_indices(text: &str, max_text_width: usize) -> Vec<usize> {
    let max_text_width = max_text_width.max(1);
    let mut indices = Vec::new();

    for (index, line) in text.split('\n').enumerate() {
        let rows = wrapped_line_count(line.width(), max_text_width);
        indices.extend(std::iter::repeat_n(index, rows));
    }

    indices
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
    fn strip_ansi_removes_csi_color_codes() {
        assert_eq!(strip_ansi("\x1b[31mred\x1b[0m"), "red");
        assert_eq!(strip_ansi("\x1b[1;32mbold green\x1b[0m"), "bold green");
    }

    #[test]
    fn strip_ansi_removes_osc_hyperlinks() {
        let input = "\x1b]8;;file:///tmp/a\x1b\\/tmp/a\x1b]8;;\x1b\\";
        assert_eq!(strip_ansi(input), "/tmp/a");
    }

    #[test]
    fn strip_ansi_removes_osc_with_bel_terminator() {
        let input = "\x1b]0;title\x07visible";
        assert_eq!(strip_ansi(input), "visible");
    }

    #[test]
    fn strip_ansi_keeps_plain_text() {
        assert_eq!(strip_ansi("plain text 🦇"), "plain text 🦇");
    }

    #[test]
    fn wrapped_height_counts_visual_rows() {
        assert_eq!(text_wrapped_height("", 10), 1);
        assert_eq!(text_wrapped_height(&"a".repeat(100), 10), 10);
        assert_eq!(text_wrapped_height("a\nbb\n", 10), 3);
    }

    #[test]
    fn visual_line_indices_map_wrapped_rows_to_sources() {
        assert_eq!(chat_visual_line_indices("aaaaaa\nbb", 3), vec![0, 0, 1]);
    }

    #[test]
    fn strip_ansi_drops_single_character_escapes_and_a_trailing_escape() {
        assert_eq!(strip_ansi("a\x1b7b\x1b"), "ab");
    }

    #[test]
    fn strip_ansi_drops_an_unterminated_operating_system_command() {
        assert_eq!(strip_ansi("a\x1b]0;title"), "a");
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
