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

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;
    use ratatui::layout::Rect;
    use std::sync::{Arc, Mutex};

    fn test_app() -> App {
        let cli = crate::cli::Cli::parse_from(["openbatrangs"]);
        App::new(&cli, Arc::new(Mutex::new(None)))
    }

    #[test]
    fn scroll_chat_negative_delta_does_not_go_below_zero() {
        let mut app = test_app();
        app.chat_scroll_offset = 0;
        app.scroll_chat(-5);
        assert_eq!(app.chat_scroll_offset, 0);
        assert_eq!(app.scroll_mode, ScrollMode::Manual);
    }

    #[test]
    fn scroll_chat_positive_delta_increases_offset() {
        let mut app = test_app();
        app.chat_scroll_offset = 0;
        app.scroll_chat(10);
        assert_eq!(app.chat_scroll_offset, 10);
    }

    #[test]
    fn scroll_chat_always_sets_manual_mode() {
        let mut app = test_app();
        app.scroll_mode = ScrollMode::Follow;
        app.scroll_chat(5);
        assert_eq!(app.scroll_mode, ScrollMode::Manual);
    }

    #[test]
    fn scroll_chat_page_without_measured_area_uses_step() {
        let mut app = test_app();
        app.last_chat_area = None;
        app.scroll_chat_page(1);
        // Without measured area, falls back to CHAT_SCROLL_STEP
        assert_eq!(app.chat_scroll_offset, CHAT_SCROLL_STEP);
    }

    #[test]
    fn scroll_chat_page_with_measured_area() {
        let mut app = test_app();
        app.last_chat_area = Some(Rect::new(0, 0, 80, 20));
        app.scroll_chat_page(1);
        // Page height = 20 - 2 = 18
        assert_eq!(app.chat_scroll_offset, 18);
    }

    #[test]
    fn scroll_chat_page_up_goes_negative_but_clamped_to_zero() {
        let mut app = test_app();
        app.chat_scroll_offset = 0;
        app.last_chat_area = Some(Rect::new(0, 0, 80, 20));
        app.scroll_chat_page(-1);
        assert_eq!(app.chat_scroll_offset, 0);
    }

    #[test]
    fn handle_scrollbar_click_without_area_returns_false() {
        let mut app = test_app();
        app.last_chat_area = None;
        assert!(!app.handle_scrollbar_click(79, 10));
    }

    #[test]
    fn handle_scrollbar_click_outside_scrollbar_column_returns_false() {
        let mut app = test_app();
        app.last_chat_area = Some(Rect::new(0, 0, 80, 20));
        // Click in the middle of the content, not on the scrollbar
        assert!(!app.handle_scrollbar_click(40, 10));
    }

    #[test]
    fn handle_scrollbar_click_at_top_border_returns_false() {
        let mut app = test_app();
        app.last_chat_area = Some(Rect::new(0, 0, 80, 20));
        let scrollbar_col = 80 - 2;
        // Click on the top border row (y=0, which is not > area.y=0)
        assert!(!app.handle_scrollbar_click(scrollbar_col, 0));
    }

    #[test]
    fn handle_scrollbar_click_at_bottom_border_returns_false() {
        let mut app = test_app();
        app.last_chat_area = Some(Rect::new(0, 0, 80, 20));
        let scrollbar_col = 80 - 2;
        // Click on the bottom border (row 19, which is not < 0 + 20 - 1 = 19)
        assert!(!app.handle_scrollbar_click(scrollbar_col, 19));
    }

    #[test]
    fn handle_scrollbar_click_inside_track_returns_true() {
        let mut app = test_app();
        // Fill the log with enough content so max_scroll > 0
        for i in 0..50 {
            app.log.push(format!("line {i}"));
        }
        app.last_chat_area = Some(Rect::new(0, 0, 80, 20));
        let scrollbar_col = 80 - 2;
        assert!(app.handle_scrollbar_click(scrollbar_col, 10));
        assert_eq!(app.scroll_mode, ScrollMode::Manual);
    }

    #[test]
    fn handle_scrollbar_click_at_bottom_of_track_switches_to_follow() {
        let mut app = test_app();
        // Need content taller than viewport for Follow to kick in
        for i in 0..100 {
            app.log.push(format!("line {i}"));
        }
        app.last_chat_area = Some(Rect::new(0, 0, 80, 20));
        let scrollbar_col = 80 - 2;
        // Click at the last valid track row
        assert!(app.handle_scrollbar_click(scrollbar_col, 18));
        // At the bottom of the scrollbar, should be Follow mode
        assert_eq!(app.scroll_mode, ScrollMode::Follow);
    }

    #[test]
    fn handle_scrollbar_click_with_offset_area() {
        let mut app = test_app();
        for i in 0..50 {
            app.log.push(format!("line {i}"));
        }
        // Area offset from (0,0) to (5,3)
        app.last_chat_area = Some(Rect::new(5, 3, 75, 17));
        let scrollbar_col = 5 + 75 - 2;
        // Row 10 is within track: 10 > 3 and 10 < 3+17-1=19
        assert!(app.handle_scrollbar_click(scrollbar_col, 10));
    }
}
