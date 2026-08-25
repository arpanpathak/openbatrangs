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

use crate::constants::tui::VIM_TERMINALS;
use std::path::{Path, PathBuf};
use std::process::Command;
use unicode_segmentation::UnicodeSegmentation;
use unicode_width::UnicodeWidthStr;

/// Remove ANSI escape sequences (CSI, OSC) from a string, returning plain text.
///
/// # Parameters
///
/// - `s`: input string potentially containing terminal escape sequences.
///
/// # Returns
///
/// The input with all ANSI/OSC escape sequences removed.
pub(crate) fn strip_ansi(s: &str) -> String {
    let mut out = String::new();
    let mut chars = s.chars();
    while let Some(c) = chars.next() {
        match c {
            '\x1b' => skip_escape_sequence(&mut chars),
            _ => out.push(c),
        }
    }
    out
}

/// Consume and discard one ANSI escape sequence after the initial ESC character.
fn skip_escape_sequence(chars: &mut std::str::Chars<'_>) {
    match chars.next() {
        // OSC: ESC ] ... BEL / ST
        Some(']') => {
            for n in chars.by_ref() {
                match n {
                    '\x07' => break,
                    '\x1b' => {
                        // Consume the backslash of ESC \ (ST) as well.
                        let _ = chars.next();
                        break;
                    }
                    _ => {}
                }
            }
        }
        // CSI: ESC [ params... final byte in 0x40..=0x7E
        Some('[') => {
            for n in chars.by_ref() {
                if n.is_ascii() && ('\u{40}'..='\u{7e}').contains(&n) {
                    break;
                }
            }
        }
        // Other single-character escape sequences need no further bytes.
        Some(_) | None => {}
    }
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
    let mut rows = Vec::new();
    for line in text.split('\n') {
        if line.is_empty() {
            rows.push(String::new());
            continue;
        }
        let mut current = String::new();
        let mut current_width = 0usize;
        for grapheme in line.graphemes(true) {
            let grapheme_width = grapheme.width();
            if current_width + grapheme_width > max_text_width && current_width > 0 {
                rows.push(std::mem::take(&mut current));
                current_width = 0;
            }
            current.push_str(grapheme);
            current_width += grapheme_width;
        }
        rows.push(current);
    }
    rows
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
    if !line.starts_with('/') {
        return ("", "");
    }
    let mut parts = line[1..].splitn(2, char::is_whitespace);
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
pub(crate) fn open_in_vim(path: &Path) {
    let path_str = path.to_string_lossy().to_string();
    for terminal in VIM_TERMINALS.iter().copied() {
        let spawned = match terminal {
            "gnome-terminal" => Command::new(terminal)
                .arg("--")
                .arg("vim")
                .arg(&path_str)
                .spawn(),
            _ => Command::new(terminal)
                .arg("-e")
                .arg("vim")
                .arg(&path_str)
                .spawn(),
        };
        if spawned.is_ok() {
            return;
        }
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
