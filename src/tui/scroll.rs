//! Chat scrolling and scrollbar hit-testing.
//!
//! Kept separate from the rest of the app state so scroll math has one obvious
//! home and can be tested in isolation.

use super::app::App;
use super::text_wrapped_height;
use crate::constants::tui::{CHAT_SCROLL_STEP, LOG_LOAD_CHUNK};

impl App {
    pub(super) fn scroll_chat(&mut self, delta: i32) {
        self.auto_scroll = false;
        let next = self.chat_scroll_offset as i32 + delta;
        if next <= 0 && delta < 0 && self.log.has_more_history() {
            self.log.load_more(LOG_LOAD_CHUNK);
            self.chat_scroll_offset = 0;
        } else {
            self.chat_scroll_offset = next.max(0) as usize;
        }
    }

    /// Scroll by one viewport page (PageUp/PageDown).
    pub(super) fn scroll_chat_page(&mut self, direction: i32) {
        let visible_height = self
            .last_chat_area
            .map(|area| area.height.saturating_sub(2).max(1) as usize)
            .unwrap_or(CHAT_SCROLL_STEP);
        let delta = (visible_height as i32).saturating_mul(direction);
        self.scroll_chat(delta);
    }

    /// Handle clicks/drags on the chat scrollbar. Returns `true` when consumed.
    pub(super) fn handle_scrollbar_click(&mut self, column: u16, row: u16) -> bool {
        let Some(area) = self.last_chat_area else {
            return false;
        };
        if column != area.x + area.width.saturating_sub(2) {
            return false;
        }
        if row <= area.y || row >= area.y + area.height.saturating_sub(1) {
            return false;
        }
        let visible_height = area.height.saturating_sub(2).max(1) as usize;
        let max_text_width = (area.width as usize).saturating_sub(2).max(1);
        let content_height = text_wrapped_height(&self.chat_text(), max_text_width);
        let max_scroll = content_height.saturating_sub(visible_height);
        let relative = (row - area.y - 1) as usize;
        let ratio = relative as f64 / visible_height.saturating_sub(1).max(1) as f64;
        let offset = (max_scroll as f64 * ratio).round() as usize;
        self.auto_scroll = offset >= max_scroll;
        self.chat_scroll_offset = offset.min(max_scroll);
        true
    }
}
