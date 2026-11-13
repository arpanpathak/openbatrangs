//! # Gemma's channel markers
//!
//! Gemma 4 writes `<|channel>thought`, a line break and its reasoning, then
//! `<channel|>` and the reply. Some OpenAI-compatible servers pass these
//! markers through in the streamed text. `<|channel>` and the channel name
//! after it often arrive in separate deltas, so [`ChannelFilter`] remembers
//! whether the last delta ended with the marker. [`answer_without_channels`]
//! cleans a whole answer, such as the `finish` summary, in one piece.

use super::types::Delta;
use crate::constants::ollama::{GEMMA_CHANNEL_CLOSE, GEMMA_CHANNEL_OPEN};

/// Removes Gemma's channel markers and channel names from streamed deltas.
#[derive(Debug, Default)]
pub(super) struct ChannelFilter {
    /// Whether the last delta ended with `<|channel>`, so the next delta
    /// starts with the channel name.
    name_comes_next: bool,
}

impl ChannelFilter {
    /// `delta` without channel markers and channel names; `None` when no
    /// text is left.
    pub(super) fn shown(&mut self, delta: Delta) -> Option<Delta> {
        match delta {
            Delta::Content(text) => self.shown_text(&text).map(Delta::Content),
            Delta::Thinking(text) => self.shown_text(&text).map(Delta::Thinking),
            Delta::ToolCalls(calls) => Some(Delta::ToolCalls(calls)),
        }
    }

    /// `text` without channel markers and channel names; `None` when nothing
    /// is left.
    fn shown_text(&mut self, text: &str) -> Option<String> {
        let text = self.without_pending_name(text);
        self.name_comes_next = text.ends_with(GEMMA_CHANNEL_OPEN);

        let shown = without_channel_markers(text);

        (!shown.is_empty()).then_some(shown)
    }

    /// `text` without its leading word when the last delta ended with
    /// `<|channel>`, so the word is the channel name.
    fn without_pending_name<'text>(&self, text: &'text str) -> &'text str {
        if !self.name_comes_next {
            return text;
        }

        without_channel_name(text)
    }
}

/// An answer, such as the `finish` summary, without Gemma's channels: only
/// the text after the last `<channel|>` is the reply, and what comes before
/// it is the channel's reasoning. Markers and channel names are removed and
/// the result is trimmed.
pub(crate) fn answer_without_channels(text: &str) -> String {
    let reply = text
        .rsplit(GEMMA_CHANNEL_CLOSE)
        .next()
        .filter(|reply| !reply.trim().is_empty())
        .unwrap_or(text);

    without_channel_markers(reply).trim().to_string()
}

/// `text` without `<channel|>`, and without each `<|channel>` and the
/// channel name that follows it.
fn without_channel_markers(text: &str) -> String {
    let mut pieces = text.split(GEMMA_CHANNEL_OPEN);
    let before_first_open = pieces.next().unwrap_or_default();
    let after_each_open: String = pieces.map(without_channel_name).collect();

    format!("{before_first_open}{after_each_open}").replace(GEMMA_CHANNEL_CLOSE, "")
}

/// `text` without its leading channel name, a single word such as
/// `thought`, `analysis` or `final`.
fn without_channel_name(text: &str) -> &str {
    text.trim_start_matches(is_name_character)
}

/// True for a letter, digit or underscore, the characters of a channel name.
fn is_name_character(character: char) -> bool {
    character.is_alphanumeric() || character == '_'
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ollama::ToolCallDelta;

    /// The deltas `filter` shows for `texts`, sent in order as reply text.
    fn shown_replies(filter: &mut ChannelFilter, texts: &[&str]) -> Vec<Delta> {
        texts
            .iter()
            .filter_map(|text| filter.shown(Delta::Content(text.to_string())))
            .collect()
    }

    #[test]
    fn removes_markers_inside_one_delta() {
        let mut filter = ChannelFilter::default();

        let shown = shown_replies(
            &mut filter,
            &["to verify.<channel|>I will add", "<|channel>thought\nPlan"],
        );

        assert_eq!(
            shown,
            vec![
                Delta::Content("to verify.I will add".to_string()),
                Delta::Content("\nPlan".to_string()),
            ]
        );
    }

    #[test]
    fn removes_a_channel_name_sent_after_the_marker() {
        let mut filter = ChannelFilter::default();

        let shown = shown_replies(
            &mut filter,
            &["<|channel>", "thought", "\nThe user wants", " a median."],
        );

        assert_eq!(
            shown,
            vec![
                Delta::Content("\nThe user wants".to_string()),
                Delta::Content(" a median.".to_string()),
            ]
        );
    }

    #[test]
    fn removes_a_split_channel_name_from_reasoning() {
        let mut filter = ChannelFilter::default();
        let opener = Delta::Thinking("Done.<|channel>".to_string());
        let name = Delta::Thinking("analysis\nNext step".to_string());

        assert_eq!(
            filter.shown(opener),
            Some(Delta::Thinking("Done.".to_string()))
        );
        assert_eq!(
            filter.shown(name),
            Some(Delta::Thinking("\nNext step".to_string()))
        );
    }

    #[test]
    fn keeps_a_leading_word_when_no_marker_came_before() {
        let mut filter = ChannelFilter::default();

        let shown = shown_replies(&mut filter, &["<|channel>final", "thought about it"]);

        assert_eq!(shown, vec![Delta::Content("thought about it".to_string())]);
    }

    #[test]
    fn passes_tool_calls_through() {
        let mut filter = ChannelFilter::default();
        let calls = Delta::ToolCalls(vec![ToolCallDelta::default()]);

        assert_eq!(filter.shown(calls.clone()), Some(calls));
    }

    #[test]
    fn an_answer_keeps_only_the_text_after_the_last_channel() {
        let summary = "thought\n<channel|>I have added the median function.";

        assert_eq!(
            answer_without_channels(summary),
            "I have added the median function."
        );
        assert_eq!(
            answer_without_channels("<|channel>thought\nplan<channel|>\nDone.<|channel>final"),
            "Done."
        );
    }

    #[test]
    fn an_answer_that_ends_with_the_marker_keeps_its_text() {
        assert_eq!(
            answer_without_channels(" All tests pass.<channel|>\n"),
            "All tests pass."
        );
        assert_eq!(answer_without_channels("plain answer"), "plain answer");
    }
}
