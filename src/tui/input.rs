//! # Input editing, history navigation, and slash-command suggestions
//!
//! The input box is a single `String` plus a byte cursor. This module owns the
//! editing operations (insert, delete, move, history) and the slash-command
//! suggestion filter, keeping `App` free of string-manipulation details.
//!
//! ## References
//!
//! - UTF-8 and character boundaries: <https://doc.rust-lang.org/std/string/struct.String.html#method.is_char_boundary>
//! - Unicode segmentation: <https://unicode.org/reports/tr29/>

use super::app::App;
use crate::constants::tui::{COMMANDS, PREFIXED_COMMANDS};

impl App {
    /// Compute slash-command autocomplete suggestions for the current input.
    ///
    /// # Returns
    ///
    /// Matching command strings (e.g. `["/mode plan"]`), empty if no prefix match.
    pub(super) fn suggestions(&self) -> Vec<String> {
        let Some(query) = self.input.strip_prefix('/') else {
            return Vec::new();
        };

        PREFIXED_COMMANDS
            .iter()
            .find_map(|(prefix, options)| option_suggestions(query, prefix, options))
            .unwrap_or_else(|| command_suggestions(query))
    }

    /// Keep the byte cursor on a valid UTF-8 boundary and within bounds.
    pub(super) fn clamp_cursor_to_boundary(&mut self) {
        self.cursor = self.input.floor_char_boundary(self.cursor);
    }

    /// Insert a single character at the cursor position and advance the cursor.
    ///
    /// # Parameters
    ///
    /// - `character`: the character to insert.
    pub(super) fn insert_char(&mut self, character: char) {
        self.clamp_cursor_to_boundary();
        self.input.insert(self.cursor, character);
        self.cursor += character.len_utf8();
    }

    /// Insert a block of text (typically from a paste event) at the cursor.
    ///
    /// CRLF/CR line endings are normalized to LF so multiline paste behaves
    /// consistently across terminals.
    ///
    /// # Parameters
    ///
    /// - `text`: the pasted text to insert.
    pub(super) fn insert_text(&mut self, text: &str) {
        if self.is_confirming() {
            return;
        }

        self.clamp_cursor_to_boundary();

        // Normalize CRLF/CR pastes to plain newlines so multiline paste behaves
        // identically across terminals and never leaves stray control chars.
        let text = text.replace("\r\n", "\n").replace('\r', "\n");
        self.input.insert_str(self.cursor, &text);
        self.cursor += text.len();
    }

    /// Insert a newline character at the cursor (Shift+Enter or Ctrl+J).
    pub(super) fn insert_newline(&mut self) {
        self.clamp_cursor_to_boundary();
        self.input.insert(self.cursor, '\n');
        self.cursor += 1;
    }

    /// Delete the character before the cursor (Backspace key).
    pub(super) fn backspace(&mut self) {
        self.clamp_cursor_to_boundary();

        let Some(previous) = self.previous_char_start() else {
            return;
        };

        self.input.remove(previous);
        self.cursor = previous;
    }

    /// Delete the character at the cursor (Delete key).
    pub(super) fn delete(&mut self) {
        self.clamp_cursor_to_boundary();

        if self.cursor < self.input.len() {
            self.input.remove(self.cursor);
        }
    }

    /// Move the cursor one character to the left (Left arrow key).
    pub(super) fn move_left(&mut self) {
        self.clamp_cursor_to_boundary();

        if let Some(previous) = self.previous_char_start() {
            self.cursor = previous;
        }
    }

    /// Move the cursor one character to the right (Right arrow key).
    pub(super) fn move_right(&mut self) {
        self.clamp_cursor_to_boundary();

        let next_width = self.input[self.cursor..]
            .chars()
            .next()
            .map_or(0, char::len_utf8);

        self.cursor += next_width;
    }

    /// Move the cursor to the start of the input (Home key).
    pub(super) fn move_home(&mut self) {
        self.cursor = 0;
    }

    /// Move the cursor to the end of the input (End key).
    pub(super) fn move_end(&mut self) {
        self.cursor = self.input.len();
    }

    /// Byte index where the character before the cursor starts, or `None`
    /// at the start of the input.
    fn previous_char_start(&self) -> Option<usize> {
        self.input[..self.cursor]
            .char_indices()
            .next_back()
            .map(|(index, _character)| index)
    }

    /// Navigate up: cycle suggestions or recall older history entries.
    pub(super) fn move_up(&mut self) {
        let suggestion_count = self.suggestions().len();

        if suggestion_count > 0 {
            self.selected = self.selected.min(suggestion_count - 1).saturating_sub(1);
            return;
        }

        self.recall_older_history();
    }

    /// Navigate down: cycle suggestions or recall newer history entries.
    pub(super) fn move_down(&mut self) {
        let suggestion_count = self.suggestions().len();

        if suggestion_count > 0 {
            self.selected = (self.selected + 1).min(suggestion_count - 1);
            return;
        }

        self.recall_newer_history();
    }

    /// Show the history entry before the one shown now, or the newest entry
    /// when no entry is shown yet.
    fn recall_older_history(&mut self) {
        let Some(newest) = self.history.len().checked_sub(1) else {
            return;
        };

        let index = self
            .history_idx
            .map_or(newest, |shown| shown.saturating_sub(1));

        self.recall_history(index);
    }

    /// Show the history entry after the one shown now, or an empty input
    /// after the newest entry.
    fn recall_newer_history(&mut self) {
        let Some(shown) = self.history_idx else {
            return;
        };

        self.recall_history_or_clear(shown + 1);
    }

    /// Show history entry `index`, or clear the input when `index` is past
    /// the newest entry.
    fn recall_history_or_clear(&mut self, index: usize) {
        if index < self.history.len() {
            self.recall_history(index);
            return;
        }

        self.history_idx = None;
        self.input.clear();
        self.cursor = 0;
    }

    /// Put history entry `index` in the input with the cursor at its end.
    fn recall_history(&mut self, index: usize) {
        let entry = self.history.get(index).cloned().unwrap_or_default();

        self.history_idx = Some(index);
        self.input = entry;
        self.cursor = self.input.len();
    }

    /// Accept the currently highlighted autocomplete suggestion (Tab key).
    pub(super) fn accept_suggestion(&mut self) {
        let suggestions = self.suggestions();
        let selected = self.selected.min(suggestions.len().saturating_sub(1));

        if let Some(suggestion) = suggestions.get(selected) {
            self.input = suggestion.clone();
            self.cursor = self.input.len();
        }
    }
}

/// Suggestions for a command that takes one of a fixed set of options, such
/// as `/mode plan`, or `None` when `query` is not that command.
fn option_suggestions(query: &str, prefix: &str, options: &[&str]) -> Option<Vec<String>> {
    let typed_option = typed_option(query, prefix)?;

    let suggestions = options
        .iter()
        .filter(|option| option.starts_with(typed_option))
        .map(|option| format!("/{prefix}{option}"))
        .collect();

    Some(suggestions)
}

/// The part of an option typed after `prefix` (such as `"mode "`): empty for
/// the bare command name, `None` when `query` is another command.
fn typed_option<'a>(query: &'a str, prefix: &str) -> Option<&'a str> {
    if query == prefix.trim_end() {
        return Some("");
    }

    query.strip_prefix(prefix)
}

/// Slash commands whose name starts with `query`.
fn command_suggestions(query: &str) -> Vec<String> {
    COMMANDS
        .iter()
        .filter(|command| command.starts_with(query))
        .map(|command| format!("/{command}"))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;
    use std::sync::{Arc, Mutex};

    /// An app with default settings and an empty input.
    fn test_app() -> App {
        let cli = crate::cli::Cli::parse_from(["openbatrangs"]);
        App::new(&cli, Arc::new(Mutex::new(None)))
    }

    #[test]
    fn plain_text_has_no_suggestions() {
        let mut app = test_app();
        app.input = "hello".to_string();

        assert!(app.suggestions().is_empty());
    }

    #[test]
    fn a_bare_option_command_suggests_every_option() {
        let mut app = test_app();
        app.input = "/thinking".to_string();

        assert_eq!(app.suggestions(), vec!["/thinking on", "/thinking off"]);
    }

    #[test]
    fn editing_moves_over_whole_characters() {
        let mut app = test_app();
        app.insert_text("a😀b");

        app.move_left();
        app.move_left();
        app.backspace();

        assert_eq!(app.input, "😀b");
        assert_eq!(app.cursor, 0);

        app.backspace();
        app.move_left();
        app.delete();

        assert_eq!(app.input, "b");

        app.move_end();
        app.move_right();
        app.delete();

        assert_eq!(app.cursor, 1);

        app.move_home();

        assert_eq!(app.cursor, 0);
    }

    #[test]
    fn a_cursor_inside_a_character_moves_to_its_start() {
        let mut app = test_app();
        app.input = "😀".to_string();
        app.cursor = 2;

        app.insert_newline();

        assert_eq!(app.input, "\n😀");
    }

    #[test]
    fn paste_is_ignored_while_confirming() {
        let mut app = test_app();
        let (response, _receiver) = tokio::sync::oneshot::channel();
        app.open_confirmation("write?".to_string(), response);

        app.insert_text("pasted");

        assert!(app.input.is_empty());
    }

    #[test]
    fn up_and_down_walk_through_history() {
        let mut app = test_app();
        app.history = vec!["first".to_string(), "second".to_string()];

        app.move_up();
        assert_eq!(app.input, "second");

        app.move_up();
        app.move_up();
        assert_eq!(app.input, "first");

        app.move_down();
        assert_eq!(app.input, "second");

        app.move_down();
        assert!(app.input.is_empty());
        assert_eq!(app.history_idx, None);

        app.move_down();
        assert!(app.input.is_empty());
    }

    #[test]
    fn up_with_no_history_keeps_the_input() {
        let mut app = test_app();
        app.input = "draft".to_string();

        app.move_up();

        assert_eq!(app.input, "draft");
    }

    #[test]
    fn up_and_down_move_through_suggestions() {
        let mut app = test_app();
        app.input = "/mode ".to_string();

        app.move_down();
        app.move_down();
        app.move_down();
        assert_eq!(app.selected, 2);

        app.move_up();
        assert_eq!(app.selected, 1);
    }

    #[test]
    fn tab_without_suggestions_keeps_the_input() {
        let mut app = test_app();
        app.input = "hello".to_string();

        app.accept_suggestion();

        assert_eq!(app.input, "hello");
    }
}
