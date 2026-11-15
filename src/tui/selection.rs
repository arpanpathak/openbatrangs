//! Text selection in the transcript and copying it to the clipboard.
//!
//! Mouse capture sends every drag to the app, so the terminal's own selection
//! does not work while it is on. A left-button drag in the transcript selects
//! the cells from the press to the pointer in reading order, the way a
//! terminal does, and the selected cells are drawn in reverse video. On
//! release, the text of those cells is sent to the system clipboard with an
//! OSC 52 escape sequence, which terminals also accept over SSH.
//!
//! ## References
//!
//! - OSC 52: <https://invisible-island.net/xterm/ctlseqs/ctlseqs.html#h3-Operating-System-Commands>

use base64::Engine;
use base64::engine::general_purpose::STANDARD;
use ratatui::buffer::Buffer;
use ratatui::layout::{Position, Rect};
use ratatui::style::Modifier;
use std::io::Write;

/// The cells between two points of a left-button drag.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Selection {
    /// Cell where the button was pressed.
    pub(super) anchor: Position,
    /// Cell under the pointer.
    pub(super) cursor: Position,
}

impl Selection {
    /// Whether `cell` lies between the two ends in reading order: row by row,
    /// left to right.
    fn contains(self, cell: Position) -> bool {
        let anchor = (self.anchor.y, self.anchor.x);
        let cursor = (self.cursor.y, self.cursor.x);
        let cell = (cell.y, cell.x);

        anchor.min(cursor) <= cell && cell <= anchor.max(cursor)
    }
}

/// Draws the selected cells of `area` in reverse video and returns their
/// text: one line per row, without trailing spaces.
pub(super) fn highlight(buffer: &mut Buffer, area: Rect, selection: Selection) -> String {
    let mut lines = Vec::new();

    for row in area.rows() {
        let selected: Vec<Position> = row
            .positions()
            .filter(|cell| selection.contains(*cell))
            .collect();

        if selected.is_empty() {
            continue;
        }

        let text: String = selected
            .iter()
            .map(|cell| buffer[*cell].symbol().to_string())
            .collect();
        lines.push(text.trim_end().to_string());

        for cell in selected {
            buffer[cell].modifier.insert(Modifier::REVERSED);
        }
    }

    lines.join("\n")
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

#[cfg(test)]
mod tests {
    use super::*;

    /// A selection from (`anchor_x`, `anchor_y`) to (`cursor_x`, `cursor_y`).
    fn selection(anchor_x: u16, anchor_y: u16, cursor_x: u16, cursor_y: u16) -> Selection {
        Selection {
            anchor: Position::new(anchor_x, anchor_y),
            cursor: Position::new(cursor_x, cursor_y),
        }
    }

    /// A 10 by 3 buffer holding `rows`.
    fn buffer(rows: [&str; 3]) -> Buffer {
        Buffer::with_lines(rows)
    }

    #[test]
    fn a_selection_runs_in_reading_order_across_rows() {
        let forward = selection(5, 0, 2, 1);
        let backward = selection(2, 1, 5, 0);

        for drag in [forward, backward] {
            assert!(drag.contains(Position::new(9, 0)));
            assert!(drag.contains(Position::new(0, 1)));
            assert!(!drag.contains(Position::new(4, 0)));
            assert!(!drag.contains(Position::new(3, 1)));
        }
    }

    #[test]
    fn the_selected_text_keeps_rows_and_drops_trailing_spaces() {
        let mut screen = buffer(["first row ", "second    ", "third     "]);
        let area = Rect::new(0, 0, 10, 3);

        let text = highlight(&mut screen, area, selection(6, 0, 5, 1));

        assert_eq!(text, "row\nsecond");
        assert!(screen[(6, 0)].modifier.contains(Modifier::REVERSED));
        assert!(!screen[(6, 1)].modifier.contains(Modifier::REVERSED));
    }

    #[test]
    fn only_cells_inside_the_area_are_selected() {
        let mut screen = buffer(["│inside  │", "│text    │", "│        │"]);
        let area = Rect::new(1, 0, 8, 3);

        let text = highlight(&mut screen, area, selection(0, 0, 9, 1));

        assert_eq!(text, "inside\ntext");
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
