//! # Runaway output
//!
//! A model sometimes falls into a loop and writes one short piece of text
//! over and over, such as `^^^^^^` or the same few words, until it reaches
//! the output limit. [`ends_in_a_loop`] finds such a loop at the end of the
//! streamed text, so the native loop can stop the request at once instead of
//! streaming thousands of copies into the transcript and the context.

/// Bytes of repeated text at the end of a reply that count as a loop.
const LOOP_BYTES: usize = 400;

/// Longest piece, in bytes, whose repetition counts as a loop.
const MAX_PIECE_BYTES: usize = 32;

/// Whether `text` ends in one piece of at most [`MAX_PIECE_BYTES`] bytes
/// repeated for the last [`LOOP_BYTES`] bytes.
pub(super) fn ends_in_a_loop(text: &str) -> bool {
    let Some(start) = text.len().checked_sub(LOOP_BYTES) else {
        return false;
    };

    let tail = &text.as_bytes()[start..];

    (1..=MAX_PIECE_BYTES).any(|piece| repeats_every(tail, piece))
}

/// Whether every byte of `tail` equals the byte `piece` places before it.
fn repeats_every(tail: &[u8], piece: usize) -> bool {
    tail.iter()
        .zip(&tail[piece..])
        .all(|(earlier, later)| earlier == later)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_short_piece_repeated_for_400_bytes_is_a_loop() {
        assert!(ends_in_a_loop(&"^".repeat(400)));
        assert!(ends_in_a_loop(&format!(
            "Plan:\n{}",
            "wait, I should ".repeat(30)
        )));
        assert!(ends_in_a_loop(&"é".repeat(200)));
    }

    #[test]
    fn ordinary_or_short_text_is_not_a_loop() {
        let prose = "The parser reads one line at a time and keeps the open block. "
            .chars()
            .cycle()
            .take(1_000)
            .collect::<String>()
            + " Then it writes the HTML for the block it closed.";
        let long_piece = "x".repeat(33) + "y";

        assert!(!ends_in_a_loop(&"^".repeat(399)));
        assert!(!ends_in_a_loop(&format!("{}!", "^".repeat(500))));
        assert!(!ends_in_a_loop(&prose));
        assert!(!ends_in_a_loop(&long_piece.repeat(20)));
    }
}
