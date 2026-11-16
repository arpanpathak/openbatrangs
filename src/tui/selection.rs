//! Text selection in the transcript and copying it to the clipboard.
//!
//! Mouse capture sends every drag to the app, so the terminal's own selection
//! does not work while it is on. A left-button drag in the transcript selects
//! text from the press to the pointer in reading order, the way a terminal
//! does, and the selected cells are drawn in reverse video. The ends of a
//! selection are rows of the transcript, not rows of the screen, so the
//! highlight moves with the text when the chat scrolls, and a selection can
//! be longer than one screen.
//!
//! On release, the text goes to the clipboard in up to three ways:
//!
//! - an OSC 52 escape sequence, which most terminals accept, also over SSH;
//!   VTE terminals such as GNOME Terminal ignore it;
//! - on a local desktop, the program `wl-copy` (Wayland) or `xclip` (X11);
//! - inside tmux, `tmux load-buffer -w -`, because tmux does not pass OSC 52
//!   on by default. tmux 3.2 and later keep the text as a paste buffer and
//!   send it to the outer terminal's clipboard themselves.
//!
//! ## References
//!
//! - OSC 52: <https://invisible-island.net/xterm/ctlseqs/ctlseqs.html#h3-Operating-System-Commands>
//! - tmux `load-buffer -w` and `set-clipboard`: <https://man7.org/linux/man-pages/man1/tmux.1.html>
//! - xclip: <https://github.com/astrand/xclip>

use base64::Engine;
use base64::engine::general_purpose::STANDARD;
use ratatui::buffer::Buffer;
use ratatui::layout::{Position, Rect};
use ratatui::style::Modifier;
use ratatui::text::Line;
use std::io::Write;
use std::process::{Command, Stdio};

/// A cell of the transcript: a wrapped row, counted from the first row in
/// memory, and a column of the text. Points compare in reading order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct TextPoint {
    /// Wrapped row of the transcript.
    pub(super) row: usize,
    /// Column within the text, 0 at the left edge of the chat.
    pub(super) column: u16,
}

impl TextPoint {
    /// The same cell after `rows` rows above it left memory.
    pub(super) fn moved_up(self, rows: usize) -> TextPoint {
        TextPoint {
            row: self.row.saturating_sub(rows),
            ..self
        }
    }
}

/// The text between two points of a left-button drag.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Selection {
    /// Point where the button was pressed.
    pub(super) anchor: TextPoint,
    /// Point under the pointer.
    pub(super) cursor: TextPoint,
}

impl Selection {
    /// Whether `point` lies between the two ends in reading order.
    fn contains(self, point: TextPoint) -> bool {
        self.anchor.min(self.cursor) <= point && point <= self.anchor.max(self.cursor)
    }

    /// The same selection after `rows` rows above it left memory.
    pub(super) fn moved_up(self, rows: usize) -> Selection {
        Selection {
            anchor: self.anchor.moved_up(rows),
            cursor: self.cursor.moved_up(rows),
        }
    }
}

/// A program that puts its standard input on the desktop clipboard.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct ClipboardProgram {
    /// Name or path of the program.
    pub(super) program: String,
    /// Arguments that make it read the clipboard text from standard input.
    pub(super) args: &'static [&'static str],
}

/// The clipboard program for a local desktop: `wl-copy` under Wayland,
/// `xclip` under X11, and none over SSH, where the desktop is on another
/// machine.
pub(super) fn desktop_clipboard(
    is_wayland: bool,
    is_x11: bool,
    is_remote: bool,
) -> Option<ClipboardProgram> {
    let (program, args): (&str, &'static [&'static str]) = match (is_remote, is_wayland, is_x11) {
        (false, true, _) => ("wl-copy", &[]),
        (false, false, true) => ("xclip", &["-selection", "clipboard"]),
        (true, _, _) | (false, false, false) => return None,
    };

    Some(ClipboardProgram {
        program: program.to_string(),
        args,
    })
}

/// Draws the selected cells of `area` in reverse video and returns their
/// text: one line per row, without trailing spaces. `top` is the transcript
/// row drawn on the first row of `area`.
pub(super) fn highlight(
    buffer: &mut Buffer,
    area: Rect,
    top: usize,
    selection: Selection,
) -> String {
    let rows = area.rows().enumerate();
    let lines: Vec<String> = rows
        .filter_map(|(offset, row)| select_row(buffer, row, top + offset, selection))
        .collect();

    lines.join("\n")
}

/// The selected text of `lines`, the transcript rows from row `top`, drawn
/// `width` cells wide.
pub(super) fn selected_text(
    lines: Vec<Line<'static>>,
    top: usize,
    width: u16,
    selection: Selection,
) -> String {
    let row = Rect::new(0, 0, width, 1);
    let mut texts = Vec::new();

    for (offset, line) in lines.into_iter().enumerate() {
        let mut buffer = Buffer::empty(row);
        buffer.set_line(0, 0, &line, width);
        texts.extend(select_row(&mut buffer, row, top + offset, selection));
    }

    texts.join("\n")
}

/// Marks the selected cells of one screen `row` in reverse video and returns
/// their text without trailing spaces, or `None` when none is selected.
/// `transcript_row` is the transcript row drawn on it.
fn select_row(
    buffer: &mut Buffer,
    row: Rect,
    transcript_row: usize,
    selection: Selection,
) -> Option<String> {
    let selected: Vec<Position> = row
        .positions()
        .filter(|cell| {
            selection.contains(TextPoint {
                row: transcript_row,
                column: cell.x - row.x,
            })
        })
        .collect();

    if selected.is_empty() {
        return None;
    }

    let text: String = selected
        .iter()
        .map(|cell| buffer[*cell].symbol().to_string())
        .collect();

    for cell in selected {
        buffer[cell].modifier.insert(Modifier::REVERSED);
    }

    Some(text.trim_end().to_string())
}

/// Puts `text`, if any, on the system clipboard with an OSC 52 sequence
/// written to `writer`, the terminal.
pub(super) fn copy_to_clipboard(
    writer: &mut impl Write,
    text: Option<String>,
) -> std::io::Result<()> {
    let Some(text) = text else {
        return Ok(());
    };

    let encoded = STANDARD.encode(text);

    write!(writer, "\x1b]52;c;{encoded}\x07")?;
    writer.flush()
}

/// Runs `program` with `args` and writes `text` to its standard input.
/// Returns whether the program took the text and exited successfully.
pub(super) fn pipe_text(program: &str, args: &[&str], text: &str) -> bool {
    let started = Command::new(program)
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn();

    let Ok(mut child) = started else {
        return false;
    };

    let is_written = child
        .stdin
        .take()
        .is_some_and(|mut stdin| stdin.write_all(text.as_bytes()).is_ok());
    let has_succeeded = child.wait().is_ok_and(|status| status.success());

    is_written && has_succeeded
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A selection from (`anchor_column`, `anchor_row`) to (`cursor_column`,
    /// `cursor_row`).
    fn selection(
        anchor_column: u16,
        anchor_row: usize,
        cursor_column: u16,
        cursor_row: usize,
    ) -> Selection {
        Selection {
            anchor: TextPoint {
                row: anchor_row,
                column: anchor_column,
            },
            cursor: TextPoint {
                row: cursor_row,
                column: cursor_column,
            },
        }
    }

    /// The point at `column` of transcript row `row`.
    fn point(column: u16, row: usize) -> TextPoint {
        TextPoint { row, column }
    }

    #[test]
    fn a_selection_runs_in_reading_order_across_rows() {
        let forward = selection(5, 0, 2, 1);
        let backward = selection(2, 1, 5, 0);

        for drag in [forward, backward] {
            assert!(drag.contains(point(9, 0)));
            assert!(drag.contains(point(0, 1)));
            assert!(!drag.contains(point(4, 0)));
            assert!(!drag.contains(point(3, 1)));
        }
    }

    #[test]
    fn a_selection_moves_up_with_rows_that_leave_memory() {
        assert_eq!(selection(1, 10, 2, 30).moved_up(8), selection(1, 2, 2, 22));
        assert_eq!(selection(1, 3, 2, 30).moved_up(8), selection(1, 0, 2, 22));
    }

    #[test]
    fn the_selected_text_keeps_rows_and_drops_trailing_spaces() {
        let mut screen = Buffer::with_lines(["first row ", "second    ", "third     "]);
        let area = Rect::new(0, 0, 10, 3);

        let text = highlight(&mut screen, area, 0, selection(6, 0, 5, 1));

        assert_eq!(text, "row\nsecond");
        assert!(screen[(6, 0)].modifier.contains(Modifier::REVERSED));
        assert!(!screen[(6, 1)].modifier.contains(Modifier::REVERSED));
    }

    #[test]
    fn the_highlight_follows_the_transcript_row_on_screen() {
        let mut screen = Buffer::with_lines(["row forty ", "row 41    "]);
        let area = Rect::new(0, 0, 10, 2);

        let text = highlight(&mut screen, area, 40, selection(0, 41, 5, 41));

        assert_eq!(text, "row 41");
        assert!(!screen[(0, 0)].modifier.contains(Modifier::REVERSED));
    }

    #[test]
    fn only_cells_inside_the_area_are_selected() {
        let mut screen = Buffer::with_lines(["│inside  │", "│text    │", "│        │"]);
        let area = Rect::new(1, 0, 8, 3);

        let text = highlight(&mut screen, area, 0, selection(0, 0, 9, 1));

        assert_eq!(text, "inside\ntext");
    }

    #[test]
    fn text_off_screen_is_read_from_the_transcript_lines() {
        let lines = vec![
            Line::from("row 7 is above"),
            Line::from(""),
            Line::from("row 9 is below"),
        ];

        let text = selected_text(lines, 7, 20, selection(6, 7, 4, 9));

        assert_eq!(text, "is above\n\nrow 9");
    }

    #[test]
    fn the_desktop_clipboard_is_wl_copy_or_xclip_and_none_over_ssh() {
        let wayland = desktop_clipboard(true, true, false).map(|clipboard| clipboard.program);
        let x11 = desktop_clipboard(false, true, false);

        assert_eq!(wayland.as_deref(), Some("wl-copy"));
        assert_eq!(
            x11,
            Some(ClipboardProgram {
                program: "xclip".to_string(),
                args: &["-selection", "clipboard"],
            })
        );
        assert_eq!(desktop_clipboard(true, true, true), None);
        assert_eq!(desktop_clipboard(false, false, false), None);
    }

    #[test]
    fn a_program_gets_the_text_on_its_standard_input() -> std::io::Result<()> {
        let directory = crate::test_support::unique_temp_dir("openbatrangs-fake-tmux");
        let received = directory.join("received.txt");
        let fake_tmux = directory.join("tmux");
        let script = format!(
            "#!/bin/sh\n[ \"$*\" = \"load-buffer -w -\" ] || exit 1\ncat > '{}'\n",
            received.display()
        );
        std::fs::write(&fake_tmux, script)?;
        std::fs::set_permissions(
            &fake_tmux,
            std::os::unix::fs::PermissionsExt::from_mode(0o755),
        )?;

        let is_loaded = pipe_text(
            &fake_tmux.to_string_lossy(),
            &["load-buffer", "-w", "-"],
            "copied text",
        );

        assert!(is_loaded);
        assert_eq!(std::fs::read_to_string(&received)?, "copied text");

        std::fs::remove_dir_all(directory)?;
        Ok(())
    }

    #[test]
    fn a_missing_or_failing_program_reports_that_it_took_nothing() {
        assert!(!pipe_text("/no/such/tmux", &[], "text"));
        assert!(!pipe_text("false", &[], "text"));
    }

    #[test]
    fn the_clipboard_sequence_carries_the_text_in_base64() -> std::io::Result<()> {
        let mut terminal = Vec::new();

        copy_to_clipboard(&mut terminal, None)?;
        copy_to_clipboard(&mut terminal, Some("hi".to_string()))?;

        assert_eq!(terminal, b"\x1b]52;c;aGk=\x07");
        Ok(())
    }
}
