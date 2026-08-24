//! # Chat scrolling and scrollbar hit-testing
//!
//! The chat area is a wrapped paragraph with its own vertical offset. Keeping
//! the scroll math in one small module means it can be tested in isolation and
//! the rest of the TUI does not need to reason about wrapped-row arithmetic.
//!
//! Two input paths feed this module:
//!
//! - Keyboard: `PageUp`/`PageDown` scroll by a full viewport; the mouse wheel
//!   scrolls by a smaller `CHAT_SCROLL_STEP`.
//! - Mouse: clicking or dragging the scrollbar maps the pointer row to a
//!   proportional offset in the wrapped content.
//!
//! ## References
//!
//! - Ratatui `Paragraph::scroll`: <https://docs.rs/ratatui/latest/ratatui/widgets/struct.Paragraph.html>
//! - Ratatui `Scrollbar`/`ScrollbarState`: <https://docs.rs/ratatui/latest/ratatui/widgets/struct.Scrollbar.html>

use super::app::App;
use super::text_wrapped_height;
use super::ScrollMode;
use crate::constants::tui::{CHAT_SCROLL_STEP, LOG_LOAD_CHUNK};

impl App {
    /// Scroll the chat by `delta` wrapped rows.
    ///
    /// Scrolling always switches to [`ScrollMode::Manual`]. When the user
    /// scrolls up past the top of the in-memory window, the next chunk of
    /// disk-backed history is loaded and the offset resets to the top of that
    /// chunk.
    pub(super) fn scroll_chat(&mut self, delta: i32) {
        self.scroll_mode = ScrollMode::Manual;
        let next = self.chat_scroll_offset as i32 + delta;
        if next <= 0 && delta < 0 && self.log.has_more_history() {
            self.log.load_more(LOG_LOAD_CHUNK);
            self.chat_scroll_offset = 0;
        } else {
            self.chat_scroll_offset = next.max(0) as usize;
        }
    }

    /// Scroll by one full viewport page (`PageUp`/`PageDown`).
    ///
    /// The page height is the visible chat area minus its two border rows. If
    /// the area has not been measured yet, fall back to the wheel step.
    pub(super) fn scroll_chat_page(&mut self, direction: i32) {
        let visible_height = self
            .last_chat_area
            .map(|area| area.height.saturating_sub(2).max(1) as usize)
            .unwrap_or(CHAT_SCROLL_STEP);
        let delta = (visible_height as i32).saturating_mul(direction);
        self.scroll_chat(delta);
    }

    /// Handle clicks/drags on the chat scrollbar.
    ///
    /// # Returns
    ///
    /// `true` when the event landed on the scrollbar and was consumed.
    pub(super) fn handle_scrollbar_click(&mut self, column: u16, row: u16) -> bool {
        let Some(area) = self.last_chat_area else {
            return false;
        };
        let is_on_scrollbar = column == area.x + area.width.saturating_sub(2);
        let is_inside_track = row > area.y && row < area.y + area.height.saturating_sub(1);
        if !is_on_scrollbar || !is_inside_track {
            return false;
        }

        let visible_height = area.height.saturating_sub(2).max(1) as usize;
        let max_text_width = (area.width as usize).saturating_sub(2).max(1);
        let content_height = text_wrapped_height(&self.chat_text(), max_text_width);
        let max_scroll = content_height.saturating_sub(visible_height);
        let relative = (row - area.y - 1) as usize;
        let track_height = visible_height.saturating_sub(1).max(1) as f64;
        let ratio = relative as f64 / track_height;
        let offset = (max_scroll as f64 * ratio).round() as usize;
        let offset = offset.min(max_scroll);

        self.scroll_mode = if offset >= max_scroll {
            ScrollMode::Follow
        } else {
            ScrollMode::Manual
        };
        self.chat_scroll_offset = offset;
        true
    }
}
