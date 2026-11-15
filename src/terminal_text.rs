//! # Untrusted text on a terminal
//!
//! The model's replies, tool output, files from a repository and server
//! responses can all hold terminal control sequences. A terminal acts on
//! them instead of showing them: they can move the cursor and draw over the
//! screen, change the window title, or write to the clipboard. Text from
//! those sources is passed through [`printable_text`] before it is printed.
//!
//! ## References
//!
//! - ECMA-48, control functions for coded character sets:
//!   <https://ecma-international.org/publications-and-standards/standards/ecma-48/>
//! - ANSI escape codes: <https://en.wikipedia.org/wiki/ANSI_escape_code>

use std::str::Chars;

/// `text` without escape sequences (CSI, OSC and the other string
/// sequences).
pub(crate) fn strip_ansi(text: &str) -> String {
    let mut plain = String::new();
    let mut characters = text.chars();

    while let Some(character) = characters.next() {
        keep_unless_escape(&mut plain, character, &mut characters);
    }

    plain
}

/// `text` without escape sequences or control characters other than
/// newlines and tabs, so a terminal shows all of it and acts on none of it.
pub(crate) fn printable_text(text: &str) -> String {
    strip_ansi(text)
        .chars()
        .filter(|character| matches!(character, '\n' | '\t') || !character.is_control())
        .collect()
}

/// Append `character` to `plain`, or skip the escape sequence it starts.
fn keep_unless_escape(plain: &mut String, character: char, characters: &mut Chars<'_>) {
    if character == '\x1b' {
        skip_escape_sequence(characters);
        return;
    }

    plain.push(character);
}

/// Consume and discard one escape sequence after the initial ESC character.
///
/// Other single-character escape sequences need no further characters.
fn skip_escape_sequence(characters: &mut Chars<'_>) {
    match characters.next() {
        // OSC, DCS, APC, PM and SOS carry a string up to a terminator.
        Some(']' | 'P' | '_' | '^' | 'X') => skip_control_string(characters),
        Some('[') => skip_control_sequence(characters),
        Some(_) | None => {}
    }
}

/// Skip a control string, which ends at BEL or at the string terminator
/// `ESC \`.
fn skip_control_string(characters: &mut Chars<'_>) {
    let terminator = characters.find(|character| matches!(character, '\x07' | '\x1b'));

    if terminator == Some('\x1b') {
        characters.next();
    }
}

/// Skip a CSI sequence (`ESC [ ...`), which ends at a final byte in `0x40..=0x7E`.
fn skip_control_sequence(characters: &mut Chars<'_>) {
    characters.find(|character| ('\u{40}'..='\u{7e}').contains(character));
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
    fn strip_ansi_removes_every_kind_of_control_string() {
        let input = "a\x1bPq#0\x1b\\b\x1b_Gf=1\x1b\\c\x1b^pm\x07d\x1bXsos\x1b\\e";
        assert_eq!(strip_ansi(input), "abcde");
    }

    #[test]
    fn strip_ansi_keeps_plain_text() {
        assert_eq!(strip_ansi("plain text 🦇"), "plain text 🦇");
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
    fn printable_text_keeps_lines_and_tabs_and_drops_other_controls() {
        let input = "\x1b[2Jline 1\r\n\tline 2\x07\u{9b}31m\x08\x7f";
        assert_eq!(printable_text(input), "line 1\n\tline 231m");
    }
}
