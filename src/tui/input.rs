//! Input editing, history navigation, and slash-command suggestions.
//!
//! These methods are inherent `impl App` blocks so the state stays in one place
//! while the file stays focused on one responsibility: text input.

use super::app::App;
use crate::constants::tui::{COMMANDS, PREFIXED_COMMANDS};

impl App {
    pub(super) fn suggestions(&self) -> Vec<String> {
        if !self.input.starts_with('/') {
            return vec![];
        }
        let query = &self.input[1..];
        for (prefix, options) in PREFIXED_COMMANDS {
            if query == prefix.trim_end() {
                return options
                    .iter()
                    .map(|option| format!("/{prefix}{option}"))
                    .collect();
            }
            if let Some(arg) = query.strip_prefix(prefix) {
                return options
                    .iter()
                    .filter(|option| option.starts_with(arg))
                    .map(|option| format!("/{prefix}{option}"))
                    .collect();
            }
        }
        COMMANDS
            .iter()
            .filter(|command| command.starts_with(query))
            .map(|command| format!("/{command}"))
            .collect()
    }

    /// Keep the byte cursor on a valid UTF-8 boundary and within bounds.
    pub(super) fn clamp_cursor_to_boundary(&mut self) {
        self.cursor = self.cursor.min(self.input.len());
        while self.cursor > 0 && !self.input.is_char_boundary(self.cursor) {
            self.cursor -= 1;
        }
    }

    pub(super) fn insert_char(&mut self, character: char) {
        self.clamp_cursor_to_boundary();
        self.input.insert(self.cursor, character);
        self.cursor += character.len_utf8();
    }

    pub(super) fn insert_text(&mut self, text: &str) {
        if self.pending_confirmation.is_some() {
            return;
        }
        self.clamp_cursor_to_boundary();
        // Normalize CRLF/CR pastes to plain newlines so multiline paste behaves
        // identically across terminals and never leaves stray control chars.
        let text = text.replace("\r\n", "\n").replace('\r', "\n");
        self.input.insert_str(self.cursor, &text);
        self.cursor += text.len();
    }

    pub(super) fn insert_newline(&mut self) {
        self.clamp_cursor_to_boundary();
        self.input.insert(self.cursor, '\n');
        self.cursor += 1;
    }

    pub(super) fn backspace(&mut self) {
        self.clamp_cursor_to_boundary();
        if self.cursor > 0 {
            let mut index = self.cursor;
            while index > 0 && !self.input.is_char_boundary(index - 1) {
                index -= 1;
            }
            self.input.remove(index - 1);
            self.cursor = index - 1;
        }
    }

    pub(super) fn delete(&mut self) {
        self.clamp_cursor_to_boundary();
        if self.cursor < self.input.len() {
            self.input.remove(self.cursor);
        }
    }

    pub(super) fn move_left(&mut self) {
        self.clamp_cursor_to_boundary();
        if self.cursor > 0 {
            let mut index = self.cursor;
            while index > 0 && !self.input.is_char_boundary(index - 1) {
                index -= 1;
            }
            self.cursor = index - 1;
        }
    }

    pub(super) fn move_right(&mut self) {
        self.clamp_cursor_to_boundary();
        if self.cursor < self.input.len() {
            let character = self.input[self.cursor..].chars().next().unwrap_or_default();
            self.cursor += character.len_utf8();
        }
    }

    pub(super) fn move_up(&mut self) {
        let suggestions = self.suggestions();
        if !suggestions.is_empty() {
            self.selected = self
                .selected
                .min(suggestions.len().saturating_sub(1))
                .saturating_sub(1);
        } else if !self.history.is_empty() {
            let idx = match self.history_idx {
                Some(i) if i > 0 => i - 1,
                Some(_) => 0,
                None => self.history.len().saturating_sub(1),
            };
            self.history_idx = Some(idx);
            self.input = self.history[idx].clone();
            self.cursor = self.input.len();
        }
    }

    pub(super) fn move_down(&mut self) {
        let suggestions = self.suggestions();
        if !suggestions.is_empty() {
            self.selected = (self.selected + 1).min(suggestions.len().saturating_sub(1));
        } else if let Some(idx) = self.history_idx {
            if idx + 1 < self.history.len() {
                self.history_idx = Some(idx + 1);
                self.input = self.history[idx + 1].clone();
                self.cursor = self.input.len();
            } else {
                self.history_idx = None;
                self.input.clear();
                self.cursor = 0;
            }
        }
    }

    pub(super) fn accept_suggestion(&mut self) {
        let suggestions = self.suggestions();
        let selected = self.selected.min(suggestions.len().saturating_sub(1));
        if let Some(suggestion) = suggestions.get(selected) {
            self.input = suggestion.clone();
            self.cursor = self.input.len();
        }
    }
}
