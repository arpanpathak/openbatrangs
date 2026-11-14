//! # The `--transcript-out` file of a one-shot run
//!
//! `--transcript-out PATH` writes one JSON object per one-shot run: the
//! model, the tool definitions, the full conversation in OpenAI
//! `/chat/completions` format, the outcome and the step count. The `tools`
//! and `messages` fields can be passed directly to
//! `tokenizer.apply_chat_template(messages, tools=tools)`.
//!
//! [`TranscriptSink`] collects the conversation from the agent loop.
//! [`CountingReporter`] wraps the stdout reporter to count steps and detect
//! the step limit. The outcome is derived from these and the run's result.
//!
//! ## References
//!
//! - OpenAI chat messages and tools: <https://platform.openai.com/docs/api-reference/chat/create>
//! - Chat templates with tools: <https://huggingface.co/docs/transformers/main/en/chat_extras>

use crate::agent::{AgentEvent, HistorySink, Reporter};
use anyhow::Result;
use serde::Serialize;
use serde_json::Value;
use std::path::Path;

/// How a run ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum Outcome {
    /// Finished before the step limit without an error. Covers a final
    /// answer, a question to the user and an empty reply.
    Answered,
    /// Reached the step limit without a final answer.
    StepLimit,
    /// Stopped with an error, such as a failed model request.
    Error,
}

impl Outcome {
    /// Derives the outcome from the run's result and whether the step limit
    /// was reached. An error takes precedence over the step limit.
    pub(super) fn of(result: &Result<()>, reached_step_limit: bool) -> Self {
        match (result, reached_step_limit) {
            (Err(_), _) => Self::Error,
            (Ok(()), true) => Self::StepLimit,
            (Ok(()), false) => Self::Answered,
        }
    }
}

/// The JSON object written to the `--transcript-out` file.
#[derive(Debug, Serialize)]
pub(super) struct Transcript {
    /// Name of the model that served the run.
    pub(super) model: String,
    /// Tool definitions sent with every request, in OpenAI `tools` format.
    /// Empty for the JSON loop.
    pub(super) tools: Value,
    /// Full conversation in OpenAI `/chat/completions` format.
    pub(super) messages: Vec<Value>,
    /// How the run ended.
    pub(super) outcome: Outcome,
    /// Number of agent steps started.
    pub(super) steps: usize,
}

/// [`HistorySink`] that stores the final conversation of a run.
#[derive(Debug, Default)]
pub(super) struct TranscriptSink {
    /// Most recently saved conversation; empty until the run ends.
    pub(super) messages: Vec<Value>,
}

impl HistorySink for TranscriptSink {
    /// Replaces any stored conversation with `messages`.
    fn save(&mut self, messages: Vec<Value>) {
        self.messages = messages;
    }
}

/// [`Reporter`] wrapper that forwards every event to `inner` and records
/// the step count and whether the step limit was reached.
#[derive(Debug, Default)]
pub(super) struct CountingReporter<R> {
    /// Reporter that receives every forwarded event.
    inner: R,
    /// Number of the most recent step started.
    pub(super) steps: usize,
    /// `true` once a `StepLimitReached` event has been seen.
    pub(super) reached_step_limit: bool,
}

impl<R> CountingReporter<R> {
    /// Wraps `inner` with zeroed counters.
    pub(super) fn new(inner: R) -> Self {
        Self {
            inner,
            steps: 0,
            reached_step_limit: false,
        }
    }
}

impl<R: Reporter> Reporter for CountingReporter<R> {
    /// Records the step number and the step limit from `event`, then
    /// forwards it to the inner reporter.
    fn report(&mut self, event: AgentEvent) {
        if let AgentEvent::StepStarted { step, .. } = &event {
            self.steps = *step;
        }

        self.reached_step_limit |= matches!(event, AgentEvent::StepLimitReached { .. });
        self.inner.report(event);
    }
}

/// Writes `transcript` to `path` as one line of JSON, if a path is given. A
/// write failure is printed as a warning and does not change the run's exit
/// status.
pub(super) fn save_transcript(path: Option<&Path>, transcript: &Transcript) {
    let Some(path) = path else {
        return;
    };

    let text = serde_json::to_string(transcript).unwrap_or_default();

    if let Err(error) = std::fs::write(path, format!("{text}\n")) {
        eprintln!(
            "⚠️  Could not write the transcript to {}: {error}",
            path.display()
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::LoopKind;
    use crate::test_support::unique_temp_dir;
    use anyhow::anyhow;
    use serde_json::json;

    /// Reporter that discards every event.
    #[derive(Default)]
    struct SilentReporter;

    impl Reporter for SilentReporter {
        /// Discards `event`.
        fn report(&mut self, _event: AgentEvent) {}
    }

    /// `StepStarted` event for step `step` of 3.
    fn step_started(step: usize) -> AgentEvent {
        AgentEvent::StepStarted {
            step,
            max_steps: 3,
            model: "m".to_string(),
            loop_kind: LoopKind::Native,
        }
    }

    /// Transcript of a one-step run that answered.
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
    fn a_failed_write_leaves_no_file_and_does_not_fail_the_run() -> Result<()> {
        let directory = unique_temp_dir("openbatrangs-transcript-missing");
        let path = directory.join("no/such/run.json");

        save_transcript(Some(&path), &sample_transcript());

        assert!(!path.exists());

        std::fs::remove_dir_all(directory)?;
        Ok(())
    }
}
