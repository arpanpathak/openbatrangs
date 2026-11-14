//! # The `--transcript-out` file of a one-shot run
//!
//! When a one-shot run ends, `--transcript-out PATH` writes one JSON object:
//! the model, the tool definitions sent to it, the whole conversation in
//! OpenAI `/chat/completions` shape, the outcome and the number of steps.
//! Each file becomes one training example of the shape
//! `tokenizer.apply_chat_template(messages, tools=tools)` accepts.
//!
//! The pieces come from three places: [`TranscriptSink`] receives the
//! conversation from the agent loop, [`CountingReporter`] counts steps and
//! notices the step limit on the way to the stdout reporter, and the run's
//! result tells an error apart.
//!
//! ## References
//!
//! - OpenAI chat messages and tools: <https://platform.openai.com/docs/api-reference/chat/create>
//! - Chat templates with tools: <https://huggingface.co/docs/transformers/main/en/chat_extras>

use crate::agent::{AgentEvent, HistorySink, Reporter};
use anyhow::{Context, Result};
use serde::Serialize;
use serde_json::Value;
use std::path::Path;

/// How a run ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum Outcome {
    /// The run ended without an error or the step limit: an answer, a
    /// question for the user, or an empty reply.
    Answered,
    /// The run used every step without an answer.
    StepLimit,
    /// The run stopped with an error, such as a failed model call.
    Error,
}

impl Outcome {
    /// The outcome of a run that returned `result` and did or did not reach
    /// the step limit. An error wins over the step limit.
    pub(super) fn of(result: &Result<()>, reached_step_limit: bool) -> Self {
        match result {
            Ok(()) => Self::without_error(reached_step_limit),
            Err(_) => Self::Error,
        }
    }

    /// The outcome of a run that ended without an error.
    fn without_error(reached_step_limit: bool) -> Self {
        match reached_step_limit {
            true => Self::StepLimit,
            false => Self::Answered,
        }
    }
}

/// The JSON object written to the `--transcript-out` file.
#[derive(Debug, Serialize)]
pub(super) struct Transcript {
    /// Model that answered the run.
    pub(super) model: String,
    /// Tool definitions sent with every request, in OpenAI `tools` shape;
    /// empty for the JSON loop, which sends none.
    pub(super) tools: Value,
    /// The whole conversation, in OpenAI `/chat/completions` shape.
    pub(super) messages: Vec<Value>,
    /// How the run ended.
    pub(super) outcome: Outcome,
    /// Number of steps the run started.
    pub(super) steps: usize,
}

/// Keeps the conversation the agent loop hands over when the run ends.
#[derive(Debug, Default)]
pub(super) struct TranscriptSink {
    /// The newest conversation received; empty before the run ends.
    pub(super) messages: Vec<Value>,
}

impl HistorySink for TranscriptSink {
    /// Keep `messages` in place of any earlier conversation.
    fn save(&mut self, messages: Vec<Value>) {
        self.messages = messages;
    }
}

/// Passes every event on to `inner`, and counts the steps and notices the
/// step limit on the way.
#[derive(Debug, Default)]
pub(super) struct CountingReporter<R> {
    /// The reporter that shows the events.
    inner: R,
    /// Number of the last step started.
    pub(super) steps: usize,
    /// Whether the run used every step without an answer.
    pub(super) reached_step_limit: bool,
}

impl<R> CountingReporter<R> {
    /// A reporter that counts events and passes them on to `inner`.
    pub(super) fn new(inner: R) -> Self {
        Self {
            inner,
            steps: 0,
            reached_step_limit: false,
        }
    }

    /// Note the step number or the step limit when `event` carries one.
    fn count(&mut self, event: &AgentEvent) {
        if let AgentEvent::StepStarted { step, .. } = event {
            self.steps = *step;
        }

        self.reached_step_limit |= matches!(event, AgentEvent::StepLimitReached { .. });
    }
}

impl<R: Reporter> Reporter for CountingReporter<R> {
    /// Count `event`, then pass it on.
    fn report(&mut self, event: AgentEvent) {
        self.count(&event);
        self.inner.report(event);
    }
}

/// Write `transcript` to `path` when a path is given. A failed write prints
/// a warning and leaves the run's own result alone.
pub(super) fn save_transcript(path: Option<&Path>, transcript: &Transcript) {
    let written = path.map_or(Ok(()), |path| write_transcript(path, transcript));

    if let Err(error) = written {
        eprintln!("⚠️  Could not write the transcript: {error:#}");
    }
}

/// Write `transcript` to `path` as one JSON object followed by a newline.
fn write_transcript(path: &Path, transcript: &Transcript) -> Result<()> {
    let text = serde_json::to_string(transcript).context("encode the transcript")?;

    std::fs::write(path, format!("{text}\n")).with_context(|| format!("write {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::LoopKind;
    use crate::test_support::unique_temp_dir;
    use anyhow::anyhow;
    use serde_json::json;

    /// A reporter that drops every event.
    #[derive(Default)]
    struct SilentReporter;

    impl Reporter for SilentReporter {
        /// Drop `event`.
        fn report(&mut self, _event: AgentEvent) {}
    }

    /// The event that starts step `step` of 3.
    fn step_started(step: usize) -> AgentEvent {
        AgentEvent::StepStarted {
            step,
            max_steps: 3,
            model: "m".to_string(),
            loop_kind: LoopKind::Native,
        }
    }

    /// A small transcript of an answered one-step run.
    fn sample_transcript() -> Transcript {
        Transcript {
            model: "m".to_string(),
            tools: json!([{"type": "function", "function": {"name": "finish"}}]),
            messages: vec![json!({"role": "user", "content": "say hi"})],
            outcome: Outcome::Answered,
            steps: 1,
        }
    }

    #[test]
    fn an_error_wins_over_the_step_limit() {
        let failed: Result<()> = Err(anyhow!("model call failed"));

        assert_eq!(Outcome::of(&failed, true), Outcome::Error);
        assert_eq!(Outcome::of(&Ok(()), true), Outcome::StepLimit);
        assert_eq!(Outcome::of(&Ok(()), false), Outcome::Answered);
    }

    #[test]
    fn outcomes_are_written_in_snake_case() -> Result<()> {
        assert_eq!(serde_json::to_value(Outcome::StepLimit)?, "step_limit");
        assert_eq!(serde_json::to_value(Outcome::Answered)?, "answered");
        assert_eq!(serde_json::to_value(Outcome::Error)?, "error");
        Ok(())
    }

    #[test]
    fn the_reporter_counts_steps_and_the_step_limit() {
        let mut reporter = CountingReporter::new(SilentReporter);

        reporter.report(step_started(1));
        reporter.report(AgentEvent::Notice("working".to_string()));
        reporter.report(step_started(2));

        assert_eq!(reporter.steps, 2);
        assert!(!reporter.reached_step_limit);

        reporter.report(AgentEvent::StepLimitReached { max_steps: 2 });

        assert!(reporter.reached_step_limit);
    }

    #[test]
    fn the_sink_keeps_the_newest_conversation() {
        let mut sink = TranscriptSink::default();

        sink.save(vec![json!({"role": "user"})]);
        sink.save(vec![json!({"role": "user"}), json!({"role": "assistant"})]);

        assert_eq!(sink.messages.len(), 2);
    }

    #[test]
    fn the_transcript_is_one_json_object_with_five_keys() -> Result<()> {
        let directory = unique_temp_dir("openbatrangs-transcript");
        let path = directory.join("run.json");

        save_transcript(Some(&path), &sample_transcript());

        let written: Value = serde_json::from_str(&std::fs::read_to_string(&path)?)?;
        assert_eq!(written["model"], "m");
        assert_eq!(written["tools"][0]["function"]["name"], "finish");
        assert_eq!(written["messages"][0]["content"], "say hi");
        assert_eq!(written["outcome"], "answered");
        assert_eq!(written["steps"], 1);

        std::fs::remove_dir_all(directory)?;
        Ok(())
    }

    #[test]
    fn a_failed_write_is_an_error_that_names_the_path() -> Result<()> {
        let directory = unique_temp_dir("openbatrangs-transcript-missing");
        let path = directory.join("no/such/run.json");

        let written = write_transcript(&path, &sample_transcript());

        assert!(format!("{written:?}").contains("no/such/run.json"));

        std::fs::remove_dir_all(directory)?;
        Ok(())
    }
}
