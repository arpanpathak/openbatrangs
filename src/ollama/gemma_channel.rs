//! # Gemma's thought channel in the reply text
//!
//! Gemma 4 writes its reasoning as a channel before the reply:
//!
//! ```text
//! <|channel>thought
//! ...reasoning...
//! <channel|>The reply.
//! ```
//!
//! `<|channel>` and `<channel|>` are single tokens, but the channel name
//! `thought` is ordinary text, so a server streams it in its own delta. A
//! server that does not split the reasoning out passes all of it through as
//! reply text. [`split_thought_channel`] follows the channel across the
//! deltas of one reply and turns the text inside it into reasoning.
//!
//! ## References
//!
//! - Gemma 4 prompt formatting: <https://ai.google.dev/gemma/docs/core/prompt-formatting-gemma4>

use super::reader::DeltaStream;
use super::types::Delta;
use crate::constants::ollama::{GEMMA_CHANNEL_END, GEMMA_CHANNEL_START, GEMMA_THOUGHT_CHANNEL};
use anyhow::Result;
use futures_util::{StreamExt, future, stream};

/// Where the reply text of a Gemma turn currently is.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum Channel {
    /// The reply itself.
    #[default]
    Answer,
    /// Right after `<|channel>`, where the channel name comes.
    Name,
    /// Inside the channel, up to `<channel|>`.
    Thought,
}

impl Channel {
    /// The marker that ends this part of the reply text.
    fn end_marker(self) -> &'static str {
        match self {
            Self::Answer => GEMMA_CHANNEL_START,
            Self::Name | Self::Thought => GEMMA_CHANNEL_END,
        }
    }

    /// The part of the reply text that follows this part's end marker.
    fn next(self) -> Self {
        match self {
            Self::Answer => Self::Name,
            Self::Name | Self::Thought => Self::Answer,
        }
    }

    /// `text` as the delta this part of the reply text becomes.
    fn delta(self, text: &str) -> Delta {
        match self {
            Self::Answer => Delta::Content(text.to_string()),
            Self::Name | Self::Thought => Delta::Thinking(text.to_string()),
        }
    }
}

/// `deltas` with the text of Gemma's thought channel moved from the reply
/// into the reasoning, and the channel markers and name removed.
pub(super) fn split_thought_channel(deltas: DeltaStream) -> DeltaStream {
    let split = deltas
        .scan(Channel::default(), |channel, item| {
            future::ready(Some(stream::iter(route(channel, item))))
        })
        .flatten();

    Box::pin(split)
}

/// The deltas `item` becomes: reply text is split at the channel markers,
/// and anything else passes through.
fn route(channel: &mut Channel, item: Result<Delta>) -> Vec<Result<Delta>> {
    match item {
        Ok(Delta::Content(text)) => split_text(channel, &text),
        other => vec![other],
    }
}

/// The reply and reasoning deltas in `text`, which starts in `channel`.
fn split_text(channel: &mut Channel, text: &str) -> Vec<Result<Delta>> {
    let mut deltas = Vec::new();
    let mut rest = text;

    while !rest.is_empty() {
        rest = take_part(channel, rest, &mut deltas);
    }

    deltas
}

/// Push the delta for the start of `text` that lies in `channel`, move past
/// the marker that ends it, and return the text after the marker.
fn take_part<'a>(channel: &mut Channel, text: &'a str, deltas: &mut Vec<Result<Delta>>) -> &'a str {
    let text = skip_channel_name(channel, text);
    let marker_split = text.split_once(channel.end_marker());
    let (part, rest) = marker_split.unwrap_or((text, ""));

    if !part.is_empty() {
        deltas.push(Ok(channel.delta(part)));
    }

    if marker_split.is_some() {
        *channel = channel.next();
    }

    rest
}

/// `text` without the channel name and its newline when `channel` is at
/// the name; past the name, `channel` is the thought channel.
fn skip_channel_name<'a>(channel: &mut Channel, text: &'a str) -> &'a str {
    if *channel != Channel::Name {
        return text;
    }

    *channel = Channel::Thought;
    let after_name = text.strip_prefix(GEMMA_THOUGHT_CHANNEL).unwrap_or(text);

    after_name.strip_prefix('\n').unwrap_or(after_name)
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::anyhow;
    use futures_util::TryStreamExt;

    /// The deltas a stream of `items` becomes.
    async fn split(items: Vec<Result<Delta>>) -> Result<Vec<Delta>> {
        let deltas: DeltaStream = Box::pin(stream::iter(items));

        split_thought_channel(deltas).try_collect().await
    }

    /// A reply text delta.
    fn content(text: &str) -> Result<Delta> {
        Ok(Delta::Content(text.to_string()))
    }

    #[tokio::test]
    async fn a_channel_streamed_token_by_token_becomes_reasoning() -> Result<()> {
        let items = vec![
            content("<|channel>"),
            content("thought"),
            content("\nThe user wants a greeting."),
            content("<channel|>"),
            content("Hello!"),
        ];

        let deltas = split(items).await?;

        assert_eq!(
            deltas,
            [
                Delta::Thinking("\nThe user wants a greeting.".to_string()),
                Delta::Content("Hello!".to_string()),
            ]
        );
        Ok(())
    }

    #[tokio::test]
    async fn a_channel_inside_one_delta_is_split_out() -> Result<()> {
        let items = vec![content(
            "<|channel>thought\nCheck the file first.<channel|>I wrote src/lib.rs.",
        )];

        let deltas = split(items).await?;

        assert_eq!(
            deltas,
            [
                Delta::Thinking("Check the file first.".to_string()),
                Delta::Content("I wrote src/lib.rs.".to_string()),
            ]
        );
        Ok(())
    }

    #[tokio::test]
    async fn an_empty_channel_leaves_only_the_reply() -> Result<()> {
        let deltas = split(vec![content("<|channel><channel|>Done.")]).await?;

        assert_eq!(deltas, [Delta::Content("Done.".to_string())]);
        Ok(())
    }

    #[tokio::test]
    async fn other_deltas_pass_through() -> Result<()> {
        let items = vec![
            Ok(Delta::Thinking("hmm".to_string())),
            content("plain reply"),
            Ok(Delta::Cut),
        ];

        let deltas = split(items).await?;

        assert_eq!(
            deltas,
            [
                Delta::Thinking("hmm".to_string()),
                Delta::Content("plain reply".to_string()),
                Delta::Cut,
            ]
        );
        Ok(())
    }

    #[tokio::test]
    async fn errors_pass_through() {
        let split = split(vec![Err(anyhow!("stream error: reset"))]).await;

        assert!(format!("{split:?}").contains("stream error: reset"));
    }
}
