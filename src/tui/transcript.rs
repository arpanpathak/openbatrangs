//! # The transcript: a bounded window of blocks, drawn by viewport
//!
//! Agent events become [`Block`]s. At most `max_blocks` blocks stay in
//! memory, as one contiguous window; older blocks go to the session file and
//! come back when the user scrolls to them. When the user scrolls back so far
//! that the newest blocks leave memory, the next change to the transcript
//! loads the newest blocks again.
//!
//! Drawing never renders the whole transcript. [`Transcript::layout`] keeps
//! a running total of block heights (each cached by content version and
//! width), so the blocks on screen are found by a binary search, and only
//! those are drawn with syntax highlighting. A streamed chunk changes one
//! block, so only that block is drawn again.

use super::block::{Answer, Block, BlockContent, Part, Step, ToolOutcome};
use super::session::BlockStore;
use super::text::{spans_text, strip_ansi};
use crate::agent::{AgentEvent, TextKind};
use ratatui::style::{Color, Style};
use ratatui::text::Line;
use std::collections::VecDeque;

/// The block part that has the keyboard focus.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Focus {
    /// Index of the block in the whole transcript.
    pub(super) block: usize,
    /// The part of that block.
    pub(super) part: Part,
}

/// What lies under one transcript row.
#[derive(Debug, PartialEq, Eq)]
pub(super) struct RowHit {
    /// Index of the block in the whole transcript.
    pub(super) block: usize,
    /// The part of the block the row belongs to.
    pub(super) part: Part,
    /// The row's text.
    pub(super) text: String,
}

/// Transcript blocks: a window in memory, the rest in the session file.
pub(super) struct Transcript {
    /// Blocks in memory, oldest first.
    blocks: VecDeque<Block>,
    /// Index of `blocks[0]` in the whole transcript.
    first: usize,
    /// Number of blocks ever added.
    total: usize,
    /// Blocks outside the window.
    store: BlockStore,
    /// Most blocks kept in memory.
    max_blocks: usize,
    /// Index of the step that is streaming, if any.
    running_step: Option<usize>,
    /// The focused block part, if any.
    focus: Option<Focus>,
    /// Width of the last layout, in cells.
    width: usize,
    /// First row of each block in memory, at `width`: the running total of
    /// the heights before it.
    row_starts: Vec<usize>,
    /// Height of all blocks in memory, at `width`.
    total_rows: usize,
    /// Rows that left the top of the window since the app last asked.
    dropped_rows: usize,
    /// Whether the window jumped back to the newest blocks since the app
    /// last asked.
    reattached: bool,
    /// Number of times a block was drawn with highlighting.
    rendered_blocks: usize,
}

impl Transcript {
    /// An empty transcript that keeps at most `max_blocks` blocks in memory.
    pub(super) fn new(max_blocks: usize) -> Self {
        Self {
            blocks: VecDeque::new(),
            first: 0,
            total: 0,
            store: BlockStore::new(),
            max_blocks: max_blocks.max(1),
            running_step: None,
            focus: None,
            width: 0,
            row_starts: Vec::new(),
            total_rows: 0,
            dropped_rows: 0,
            reattached: false,
            rendered_blocks: 0,
        }
    }

    /// Whether nothing was ever added.
    pub(super) fn is_empty(&self) -> bool {
        self.total == 0
    }

    /// Number of blocks in memory.
    #[cfg(test)]
    pub(super) fn blocks_in_memory(&self) -> usize {
        self.blocks.len()
    }

    /// Number of times a block was drawn with highlighting.
    #[cfg(test)]
    pub(super) fn rendered_blocks(&self) -> usize {
        self.rendered_blocks
    }

    /// Height of the blocks in memory at the width of the last layout.
    pub(super) fn total_rows(&self) -> usize {
        self.total_rows
    }

    /// Add a block at the end.
    pub(super) fn push(&mut self, content: BlockContent) {
        self.ensure_attached();

        self.blocks.push_back(Block::new(content));
        self.total += 1;
        self.evict_front_over_cap();
    }

    /// Add a notice block.
    pub(super) fn push_notice(&mut self, lines: Vec<String>) {
        self.push(BlockContent::Notice(lines));
    }

    /// Show one agent event.
    pub(super) fn apply(&mut self, event: AgentEvent) {
        match event {
            AgentEvent::StepStarted {
                step,
                max_steps,
                model,
                ..
            } => self.start_step(Step::new(step, max_steps, model)),
            AgentEvent::Text {
                kind: TextKind::Answer,
                text,
            } => self.add_answer_text(&text),
            AgentEvent::Text { text, .. } => {
                self.update_running_step(|step| step.reasoning.push_str(&text))
            }
            AgentEvent::TextEnded(_kind) => {}
            AgentEvent::ToolCall {
                id,
                name,
                arguments,
            } => self.update_running_step(|step| step.add_tool(id, name, arguments)),
            AgentEvent::ToolResult {
                id,
                success,
                summary,
                output,
            } => self.update_running_step(|step| {
                step.set_result(
                    &id,
                    ToolOutcome {
                        success,
                        summary,
                        output,
                    },
                )
            }),
            AgentEvent::Notice(text) => self.push_notice(notice_lines(&text)),
            AgentEvent::AskUser { question } => self.end_with(BlockContent::Question(question)),
            AgentEvent::Finished {
                answer,
                changed_files,
                ..
            } => self.end_with(BlockContent::Answer(Answer {
                text: answer,
                changed_files,
            })),
        }
    }

    /// Fold the running step and start a new one.
    fn start_step(&mut self, step: Step) {
        self.finish_running_step();
        self.push(BlockContent::Step(step));

        self.running_step = Some(self.total - 1);
    }

    /// Stop the running step's clock and fold it.
    pub(super) fn finish_running_step(&mut self) {
        self.update_running_step(Step::finish);

        self.running_step = None;
    }

    /// Finish the running step and add `content` after it.
    fn end_with(&mut self, content: BlockContent) {
        self.finish_running_step();
        self.push(content);
    }

    /// Apply `change` to the running step, if there is one.
    fn update_running_step(&mut self, change: impl FnOnce(&mut Step)) {
        self.ensure_attached();

        let running_step = self.running_step;
        let block = running_step.and_then(|index| self.block_mut(index));

        if let Some(block) = block {
            block.update_step(change);
        }
    }

    /// Answer text goes into the running step; without one (chat mode) it
    /// streams into an answer block.
    fn add_answer_text(&mut self, text: &str) {
        match self.running_step {
            Some(_index) => self.update_running_step(|step| step.reply.push_str(text)),
            None => self.add_chat_reply(text),
        }
    }

    /// Append `text` to the answer block at the end, or start one.
    fn add_chat_reply(&mut self, text: &str) {
        self.ensure_attached();

        let appended = self
            .blocks
            .back_mut()
            .is_some_and(|block| block.append_answer(text));

        if !appended {
            self.push(BlockContent::Answer(Answer {
                text: text.to_string(),
                changed_files: Vec::new(),
            }));
        }
    }

    /// The block with transcript index `index`, if it is in memory.
    fn block_mut(&mut self, index: usize) -> Option<&mut Block> {
        let offset = index.checked_sub(self.first)?;

        self.blocks.get_mut(offset)
    }

    /// Whether the window holds the newest blocks.
    fn is_attached(&self) -> bool {
        self.first + self.blocks.len() == self.total
    }

    /// Bring the newest blocks back into memory when the window has moved
    /// away from them.
    fn ensure_attached(&mut self) {
        if self.is_attached() {
            return;
        }

        self.store_window();
        self.first = self.total;
        self.load_older(self.max_blocks);
        self.reattached = true;
    }

    /// Write every block in memory to the session file and empty the window.
    fn store_window(&mut self) {
        let first = self.first;

        for (offset, block) in self.blocks.drain(..).enumerate() {
            self.store.write(first + offset, &block);
        }
    }

    /// Move the oldest blocks to the session file until the window fits.
    fn evict_front_over_cap(&mut self) {
        let excess = self.blocks.len().saturating_sub(self.max_blocks);
        let evicted: Vec<Block> = self.blocks.drain(..excess).collect();

        // Only rows of the last layout were on screen; a block never laid
        // out at this width moves nothing.
        for block in evicted {
            self.dropped_rows += block.cached_height(self.width).unwrap_or(0);
            self.store.write(self.first, &block);
            self.first += 1;
        }
    }

    /// Move the newest blocks to the session file until the window fits.
    fn evict_back_over_cap(&mut self) {
        let kept = self.blocks.len().min(self.max_blocks);
        let evicted: Vec<Block> = self.blocks.drain(kept..).collect();

        for (offset, block) in evicted.iter().enumerate() {
            self.store.write(self.first + kept + offset, block);
        }
    }

    /// Whether older blocks wait in the session file.
    pub(super) fn has_older(&self) -> bool {
        self.first
            .checked_sub(1)
            .is_some_and(|index| self.store.has(index))
    }

    /// Whether newer blocks wait in the session file.
    pub(super) fn has_newer(&self) -> bool {
        !self.is_attached()
    }

    /// Load up to `chunk` blocks before the window from the session file;
    /// the newest blocks leave memory when the window is full. Returns
    /// whether any block was loaded.
    pub(super) fn load_older(&mut self, chunk: usize) -> bool {
        let older: Vec<Block> = (self.first.saturating_sub(chunk)..self.first)
            .rev()
            .map_while(|index| self.store.read(index))
            .collect();
        let is_loaded = !older.is_empty();

        self.first -= older.len();
        for block in older {
            self.blocks.push_front(block);
        }

        self.evict_back_over_cap();
        is_loaded
    }

    /// Load up to `chunk` blocks after the window from the session file; the
    /// oldest blocks leave memory when the window is full. Returns whether
    /// any block was loaded.
    pub(super) fn load_newer(&mut self, chunk: usize) -> bool {
        let end = self.first + self.blocks.len();
        let newer: Vec<Block> = (end..(end + chunk).min(self.total))
            .map_while(|index| self.store.read(index))
            .collect();
        let is_loaded = !newer.is_empty();

        self.blocks.extend(newer);
        self.evict_front_over_cap();

        is_loaded
    }

    /// Rows that left the top of the window since the last call.
    pub(super) fn take_dropped_rows(&mut self) -> usize {
        std::mem::take(&mut self.dropped_rows)
    }

    /// Whether the window jumped back to the newest blocks since the last call.
    pub(super) fn take_reattached(&mut self) -> bool {
        std::mem::take(&mut self.reattached)
    }

    /// Remove every block, in memory and on disk.
    pub(super) fn clear(&mut self) {
        self.blocks.clear();
        self.store.clear();
        self.first = 0;
        self.total = 0;
        self.running_step = None;
        self.focus = None;
        self.row_starts.clear();
        self.total_rows = 0;
        self.dropped_rows = 0;
    }

    /// Measure the blocks in memory at `width` and keep the running total
    /// of their heights. Returns the height of them all.
    pub(super) fn layout(&mut self, width: usize) -> usize {
        let mut row_starts = Vec::with_capacity(self.blocks.len());
        let mut total_rows = 0;

        for block in &mut self.blocks {
            row_starts.push(total_rows);
            total_rows += block.height(width);
        }

        self.width = width;
        self.row_starts = row_starts;
        self.total_rows = total_rows;

        total_rows
    }

    /// Offset in the window of the block that holds `row`.
    fn block_at_row(&self, row: usize) -> usize {
        self.row_starts
            .partition_point(|start| *start <= row)
            .saturating_sub(1)
    }

    /// The `height` rows from row `top` at `width`. Only the blocks that
    /// intersect those rows are drawn. With `show_focus`, the focused part
    /// is highlighted.
    pub(super) fn visible_lines(
        &mut self,
        width: usize,
        top: usize,
        height: usize,
        show_focus: bool,
    ) -> Vec<Line<'static>> {
        self.layout(width);

        let bottom = top + height;
        let visible: Vec<usize> = (self.block_at_row(top)..self.blocks.len())
            .take_while(|offset| self.row_starts[*offset] < bottom)
            .collect();
        let mut lines = Vec::new();

        for offset in visible {
            lines.extend(self.visible_rows(offset, top..bottom, show_focus));
        }

        lines
    }

    /// The rows of block `offset` that fall inside `rows`.
    fn visible_rows(
        &mut self,
        offset: usize,
        rows: std::ops::Range<usize>,
        show_focus: bool,
    ) -> Vec<Line<'static>> {
        let block_top = self.row_starts[offset];
        let focused_part = self.focused_part(offset, show_focus);
        let skipped = rows.start.saturating_sub(block_top);
        let taken = rows.end - block_top.max(rows.start);

        let (drawn, is_new) = self.blocks[offset].render(self.width);
        let lines = drawn
            .lines
            .iter()
            .zip(&drawn.parts)
            .skip(skipped)
            .take(taken)
            .map(|(line, part)| focus_style(line, *part, focused_part))
            .collect();

        self.rendered_blocks += usize::from(is_new);
        lines
    }

    /// The focused part of block `offset`, when focus is shown.
    fn focused_part(&self, offset: usize, show_focus: bool) -> Option<Part> {
        self.focus
            .filter(|focus| show_focus && focus.block == self.first + offset)
            .map(|focus| focus.part)
    }

    /// What lies under row `row` at `width`, if a block does.
    pub(super) fn row_at(&mut self, width: usize, row: usize) -> Option<RowHit> {
        self.layout(width);

        let offset = self.block_at_row(row);
        let row_in_block = row.checked_sub(*self.row_starts.get(offset)?)?;
        let (drawn, _is_new) = self.blocks.get_mut(offset)?.render(width);
        let line = drawn.lines.get(row_in_block)?;

        Some(RowHit {
            block: self.first + offset,
            part: drawn.parts[row_in_block],
            text: spans_text(&line.spans),
        })
    }

    /// Fold or unfold `part` of block `index`; false when it does not fold.
    pub(super) fn toggle(&mut self, index: usize, part: Part) -> bool {
        self.block_mut(index)
            .is_some_and(|block| block.toggle(part))
    }

    /// The focused part, if any.
    #[cfg(test)]
    pub(super) fn focus(&self) -> Option<Focus> {
        self.focus
    }

    /// Focus `focus`.
    pub(super) fn set_focus(&mut self, focus: Focus) {
        self.focus = Some(focus);
    }

    /// Move the focus `step` parts down (or up, when negative), staying
    /// inside the blocks in memory. Without a focus, the last part gets it.
    pub(super) fn move_focus(&mut self, step: isize) {
        let targets = self.focus_targets();
        let current = self
            .focus
            .and_then(|focus| targets.iter().position(|target| *target == focus));
        let last = targets.len().saturating_sub(1);

        let next = current.map_or(last, |position| {
            position.saturating_add_signed(step).min(last)
        });

        self.focus = targets.get(next).copied();
    }

    /// Every part the keyboard can move to, in order.
    fn focus_targets(&self) -> Vec<Focus> {
        let first = self.first;

        self.blocks
            .iter()
            .enumerate()
            .flat_map(|(offset, block)| {
                block.focus_parts().into_iter().map(move |part| Focus {
                    block: first + offset,
                    part,
                })
            })
            .collect()
    }

    /// Fold or unfold the focused part; false when nothing folded.
    pub(super) fn toggle_focused(&mut self) -> bool {
        let focus = self.focus;

        focus.is_some_and(|focus| self.toggle(focus.block, focus.part))
    }

    /// The first row and the row after the last of the focused part, at `width`.
    pub(super) fn focus_rows(&mut self, width: usize) -> Option<(usize, usize)> {
        let focus = self.focus?;
        self.layout(width);

        let offset = focus.block.checked_sub(self.first)?;
        let block_top = *self.row_starts.get(offset)?;
        let (drawn, _is_new) = self.blocks.get_mut(offset)?.render(width);
        let first_row = drawn.parts.iter().position(|part| *part == focus.part)?;
        let last_row = drawn.parts.iter().rposition(|part| *part == focus.part)?;

        Some((block_top + first_row, block_top + last_row + 1))
    }

    /// The text of every block in memory, for tests.
    #[cfg(test)]
    pub(super) fn text(&self) -> String {
        let texts: Vec<String> = self.blocks.iter().map(Block::source_text).collect();

        texts.join("\n")
    }
}

/// The lines of a notice, without terminal escapes or leading blank lines.
fn notice_lines(text: &str) -> Vec<String> {
    strip_ansi(text)
        .trim_start_matches('\n')
        .lines()
        .map(str::to_string)
        .collect()
}

/// `line`, with a dark background when it belongs to the focused part.
fn focus_style(line: &Line<'static>, part: Part, focused_part: Option<Part>) -> Line<'static> {
    let line = line.clone();

    match focused_part == Some(part) {
        true => line.patch_style(Style::new().bg(Color::DarkGray)),
        false => line,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::LoopKind;
    use crate::constants::tui::MAX_TRANSCRIPT_BLOCKS;
    use std::path::PathBuf;

    /// A step-started event for step `step`.
    fn step(step: usize) -> AgentEvent {
        AgentEvent::StepStarted {
            step,
            max_steps: 5,
            model: "m".to_string(),
            loop_kind: LoopKind::Native,
        }
    }

    /// A streamed text event.
    fn text(kind: TextKind, text: &str) -> AgentEvent {
        AgentEvent::Text {
            kind,
            text: text.to_string(),
        }
    }

    /// The text of `lines`.
    fn texts(lines: &[Line<'static>]) -> Vec<String> {
        lines.iter().map(|line| spans_text(&line.spans)).collect()
    }

    /// A transcript with one notice per number in `numbers`.
    fn notices(max_blocks: usize, count: usize) -> Transcript {
        let mut transcript = Transcript::new(max_blocks);

        for number in 0..count {
            transcript.push_notice(vec![format!("notice {number}")]);
        }

        transcript
    }

    #[test]
    fn a_long_session_keeps_memory_bounded_and_draws_only_visible_blocks() {
        let mut transcript = notices(MAX_TRANSCRIPT_BLOCKS, 10_000);

        assert_eq!(transcript.blocks_in_memory(), MAX_TRANSCRIPT_BLOCKS);
        assert_eq!(transcript.first, 10_000 - MAX_TRANSCRIPT_BLOCKS);

        let total_rows = transcript.layout(80);
        let lines = transcript.visible_lines(80, total_rows - 20, 20, false);

        assert_eq!(total_rows, MAX_TRANSCRIPT_BLOCKS);
        assert_eq!(texts(&lines)[19], "notice 9999");
        assert_eq!(texts(&lines)[0], "notice 9980");
        assert_eq!(transcript.rendered_blocks(), 20);

        transcript.visible_lines(80, total_rows - 20, 20, false);
        assert_eq!(transcript.rendered_blocks(), 20, "drawings are cached");

        while transcript.load_older(100) {
            assert!(transcript.blocks_in_memory() <= MAX_TRANSCRIPT_BLOCKS);
        }

        assert_eq!(transcript.first, 0);
        assert!(!transcript.has_older());
        assert!(transcript.has_newer());
        assert!(transcript.text().starts_with("notice 0\nnotice 1"));
        assert_eq!(transcript.blocks_in_memory(), MAX_TRANSCRIPT_BLOCKS);
    }

    #[test]
    fn scrolling_back_down_loads_newer_blocks_and_drops_older_ones() {
        let mut transcript = notices(10, 40);
        transcript.layout(20);

        assert!(transcript.load_older(25));
        assert_eq!(transcript.first, 5);
        assert!(transcript.has_newer());
        transcript.layout(20);

        assert!(transcript.load_newer(10));
        assert_eq!(transcript.first, 15);
        assert_eq!(transcript.take_dropped_rows(), 10);
        assert_eq!(transcript.take_dropped_rows(), 0);
        assert!(transcript.has_newer());

        assert!(transcript.load_newer(30));
        assert_eq!(transcript.first, 30);
        assert!(!transcript.has_newer());
        assert!(!transcript.load_newer(30));
    }

    #[test]
    fn a_change_while_scrolled_back_brings_the_newest_blocks_back() {
        let mut transcript = notices(4, 12);
        transcript.load_older(8);
        assert!(transcript.has_newer());

        transcript.push_notice(vec!["latest".to_string()]);

        assert!(transcript.take_reattached());
        assert!(!transcript.take_reattached());
        assert!(!transcript.has_newer());
        assert!(transcript.text().ends_with("notice 11\nlatest"));
        assert_eq!(transcript.blocks_in_memory(), 4);
    }

    #[test]
    fn agent_events_build_steps_answers_and_questions() {
        let mut transcript = Transcript::new(50);

        transcript.apply(text(TextKind::Reasoning, "ignored before a step"));
        transcript.apply(AgentEvent::Notice(
            "\n\x1b[2m🔎 web_search → x\x1b[0m".to_string(),
        ));
        transcript.apply(step(1));
        transcript.apply(text(TextKind::Reasoning, "**Look**"));
        transcript.apply(AgentEvent::TextEnded(TextKind::Reasoning));
        transcript.apply(AgentEvent::ToolCall {
            id: "c1".to_string(),
            name: "list_files".to_string(),
            arguments: "{}".to_string(),
        });
        transcript.apply(AgentEvent::ToolResult {
            id: "c1".to_string(),
            success: true,
            summary: "2 lines".to_string(),
            output: "a\nb".to_string(),
        });
        transcript.apply(step(2));
        transcript.apply(text(TextKind::Answer, "done"));
        transcript.apply(AgentEvent::Finished {
            answer: "done".to_string(),
            answer_was_streamed: false,
            changed_files: vec![PathBuf::from("/w/a.rs")],
        });
        transcript.apply(AgentEvent::AskUser {
            question: "More?".to_string(),
        });

        let lines = texts(&transcript.visible_lines(60, 0, 40, false));

        assert_eq!(lines[0], "🔎 web_search → x");
        assert!(lines[1].starts_with("▸ Step 1/5 · Look · m · "));
        assert!(lines[2].starts_with("▸ Step 2/5 · done · m · "));
        assert!(lines.contains(&"✅ Answer".to_string()));
        assert!(lines.contains(&"   /w/a.rs".to_string()));
        assert!(lines.contains(&"❓ Question".to_string()));
    }

    #[test]
    fn the_running_step_stays_open() {
        let mut transcript = Transcript::new(50);

        transcript.apply(step(1));
        transcript.apply(text(TextKind::Thought, "Plan it."));

        let lines = texts(&transcript.visible_lines(60, 0, 10, false));

        assert_eq!(lines[0], "▾ Step 1/5 · Plan it · m · running");
        assert_eq!(lines[1], "  ▸ 💭 Plan it.");

        transcript.finish_running_step();
        let lines = texts(&transcript.visible_lines(60, 0, 10, false));
        assert_eq!(lines.len(), 1);
    }

    #[test]
    fn chat_replies_stream_into_one_answer_block() {
        let mut transcript = Transcript::new(50);
        transcript.push(BlockContent::User("hi".to_string()));

        transcript.apply(text(TextKind::Answer, "Hel"));
        transcript.apply(text(TextKind::Answer, "lo"));

        assert_eq!(transcript.text(), "You: hi\nHello");
    }

    #[test]
    fn clicks_and_keys_fold_and_focus_parts() {
        let mut transcript = Transcript::new(50);
        transcript.apply(step(1));
        transcript.apply(text(TextKind::Reasoning, "think"));
        transcript.apply(AgentEvent::ToolCall {
            id: "c1".to_string(),
            name: "list_files".to_string(),
            arguments: "{}".to_string(),
        });
        transcript.push_notice(vec!["after".to_string()]);

        let hit = transcript.row_at(60, 1);
        assert_eq!(
            hit,
            Some(RowHit {
                block: 0,
                part: Part::Thinking,
                text: "  ▸ 💭 think".to_string(),
            })
        );
        assert!(transcript.row_at(60, 99).is_none());
        assert!(transcript.toggle(0, Part::Thinking));
        assert!(!transcript.toggle(1, Part::Header));
        assert!(!transcript.toggle(7, Part::Header));

        transcript.move_focus(-1);
        assert_eq!(
            transcript.focus(),
            Some(Focus {
                block: 1,
                part: Part::Header
            })
        );
        transcript.move_focus(-1);
        transcript.move_focus(-1);
        assert_eq!(
            transcript.focus(),
            Some(Focus {
                block: 0,
                part: Part::Thinking
            })
        );
        assert_eq!(transcript.focus_rows(60), Some((1, 3)));
        assert!(transcript.toggle_focused());

        let lines = transcript.visible_lines(60, 0, 10, true);
        assert_eq!(lines[1].style.bg, Some(Color::DarkGray));
        assert_eq!(lines[0].style.bg, None);

        transcript.set_focus(Focus {
            block: 0,
            part: Part::Header,
        });
        transcript.move_focus(-1);
        assert_eq!(
            transcript.focus(),
            Some(Focus {
                block: 0,
                part: Part::Header
            })
        );
    }

    #[test]
    fn nothing_is_focused_or_toggled_in_an_empty_transcript() {
        let mut transcript = Transcript::new(5);

        transcript.move_focus(1);

        assert_eq!(transcript.focus(), None);
        assert!(!transcript.toggle_focused());
        assert_eq!(transcript.focus_rows(40), None);
        assert!(transcript.is_empty());
    }

    #[test]
    fn clear_forgets_blocks_in_memory_and_on_disk() {
        let mut transcript = notices(2, 5);
        transcript.apply(step(1));

        transcript.clear();

        assert!(transcript.is_empty());
        assert_eq!(transcript.blocks_in_memory(), 0);
        assert!(!transcript.has_older());
        assert_eq!(transcript.layout(40), 0);
        assert_eq!(transcript.total_rows(), 0);
    }
}
