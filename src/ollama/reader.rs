//! # Reading streamed replies
//!
//! A streamed reply arrives in chunks that split lines and characters at any
//! byte. [`ResponseText`] collects the decoded text of a response;
//! [`delta_stream`] turns a chat reply into a stream of [`Delta`]s, and
//! [`follow_pull`] reports the progress of a model download.
//!
//! Each reading loop asks one step at a time for a [`ControlFlow`]: `Break`
//! carries what the loop returns, and `Continue` means one more chunk arrived
//! and the buffer is drained again.

use super::stream::{Drain, PullLine, TextBuffer, drain_pull_line};
use super::types::Delta;
use anyhow::{Result, anyhow};
use futures_util::Stream;
use reqwest::Response;
use std::ops::ControlFlow;
use std::pin::Pin;

/// A boxed stream of reply deltas, as the chat methods return it.
pub(super) type DeltaStream = Pin<Box<dyn Stream<Item = Result<Delta>> + Send + 'static>>;

/// A response body and the text decoded from it so far.
struct ResponseText {
    /// The response whose body is still arriving.
    response: Response,
    /// Decoded text not yet turned into events.
    buffer: TextBuffer,
}

impl ResponseText {
    /// Start reading the body of `response`.
    fn new(response: Response) -> Self {
        Self {
            response,
            buffer: TextBuffer::default(),
        }
    }

    /// Read the next chunk into the buffer; `Ok(false)` once the body has
    /// ended.
    async fn read_more(&mut self) -> reqwest::Result<bool> {
        let chunk = self.response.chunk().await?;
        let has_more = chunk.is_some();

        self.buffer.push(chunk.as_deref().unwrap_or_default());

        Ok(has_more)
    }
}

/// Reads a chat reply and hands out its deltas one at a time.
struct DeltaReader {
    /// The reply's text so far.
    text: ResponseText,
    /// Turns complete lines of text into the next event: NDJSON lines for
    /// Ollama, server-sent events for OpenAI-compatible servers.
    drain: fn(&mut String) -> Drain,
}

impl DeltaReader {
    /// The next delta, an error when reading fails, or `None` at the end of
    /// the reply.
    async fn next_delta(&mut self) -> Option<Result<Delta>> {
        loop {
            if let ControlFlow::Break(item) = self.step().await {
                return item;
            }
        }
    }

    /// A delta already in the buffer, the end of the reply, or the result of
    /// reading one more chunk.
    async fn step(&mut self) -> ControlFlow<Option<Result<Delta>>> {
        match (self.drain)(&mut self.text.buffer.text) {
            Drain::Delta(delta) => ControlFlow::Break(Some(Ok(delta))),
            Drain::Done => ControlFlow::Break(None),
            Drain::NeedMore => self.read().await,
            Drain::Error(message) => {
                ControlFlow::Break(Some(Err(anyhow!("model server error: {message}"))))
            }
        }
    }

    /// `Continue` when one more chunk arrived, `Break` at the end of the body
    /// or with the error that stopped reading.
    async fn read(&mut self) -> ControlFlow<Option<Result<Delta>>> {
        match self.text.read_more().await {
            Ok(true) => ControlFlow::Continue(()),
            Ok(false) => ControlFlow::Break(None),
            Err(error) => ControlFlow::Break(Some(Err(anyhow!("stream error: {error}")))),
        }
    }
}

/// The deltas of a streamed chat reply; `drain` parses the reply's lines.
pub(super) fn delta_stream(response: Response, drain: fn(&mut String) -> Drain) -> DeltaStream {
    let reader = DeltaReader {
        text: ResponseText::new(response),
        drain,
    };

    Box::pin(futures_util::stream::unfold(
        reader,
        |mut reader| async move {
            let item = reader.next_delta().await;

            item.map(|item| (item, reader))
        },
    ))
}

/// Read the progress lines of `/api/pull`, passing each status to
/// `on_status`, until the pull finishes or fails.
pub(super) async fn follow_pull(
    response: Response,
    on_status: &(dyn Fn(&str) + Sync),
) -> Result<()> {
    let mut text = ResponseText::new(response);

    loop {
        if let ControlFlow::Break(outcome) = pull_step(&mut text, on_status).await {
            return outcome;
        }
    }
}

/// The outcome of a progress line already in the buffer, or of reading one
/// more chunk.
async fn pull_step(
    text: &mut ResponseText,
    on_status: &(dyn Fn(&str) + Sync),
) -> ControlFlow<Result<()>> {
    match drain_pull_line(&mut text.buffer.text) {
        Some(line) => pull_line_outcome(line, on_status),
        None => read_pull(text).await,
    }
}

/// Report a status and go on, or stop with the pull's success or error.
fn pull_line_outcome(line: PullLine, on_status: &(dyn Fn(&str) + Sync)) -> ControlFlow<Result<()>> {
    match line {
        PullLine::Status(status) => report_status(&status, on_status),
        PullLine::Error(error) => {
            ControlFlow::Break(Err(anyhow!("Ollama /api/pull failed: {error}")))
        }
        PullLine::Done => finish_pull(on_status),
    }
}

/// Pass `status` to `on_status`; the pull goes on.
fn report_status(status: &str, on_status: &(dyn Fn(&str) + Sync)) -> ControlFlow<Result<()>> {
    on_status(status);

    ControlFlow::Continue(())
}

/// Report the finished pull; reading stops with success.
fn finish_pull(on_status: &(dyn Fn(&str) + Sync)) -> ControlFlow<Result<()>> {
    on_status("✅ Pull finished");

    ControlFlow::Break(Ok(()))
}

/// `Continue` when one more chunk arrived; an error when the body ended
/// before the success line or could not be read.
async fn read_pull(text: &mut ResponseText) -> ControlFlow<Result<()>> {
    match text.read_more().await {
        Ok(true) => ControlFlow::Continue(()),
        Ok(false) => ControlFlow::Break(Err(anyhow!(
            "Ollama /api/pull stream ended before reporting success"
        ))),
        Err(error) => ControlFlow::Break(Err(
            anyhow::Error::new(error).context("failed to read Ollama /api/pull stream")
        )),
    }
}
