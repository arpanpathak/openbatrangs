//! # Structured agent events and the stdout reporter
//!
//! The agent loops describe what happens as [`AgentEvent`] values: a step
//! starts, the model streams reasoning or reply text, a tool runs and returns,
//! the run finishes or asks the user a question. A [`Reporter`] receives the
//! events. The one-shot CLI prints them with [`StdoutReporter`]; the TUI turns
//! them into transcript blocks.
//!
//! ## References
//!
//! - Rust traits as abstraction boundaries: <https://doc.rust-lang.org/book/ch10-02-traits.html>

use super::execute::clickable_link;
use super::tool::describe_call;
use crate::constants::ansi::{
    ANSI_GREEN_CHECK, ANSI_YELLOW_FILES_HEADER, COLOR_BOLD, COLOR_CYAN, COLOR_DIM, COLOR_MAGENTA,
    COLOR_RESET,
};
use std::io::Write;
use std::path::PathBuf;

/// Which agent loop sent the events. The two loops print their one-shot
/// output in different styles.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoopKind {
    /// The native tool-calling loop for OpenAI-compatible servers.
    Native,
    /// The JSON loop for Ollama servers.
    Json,
}

/// What a piece of streamed text is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextKind {
    /// The model's own reasoning (thinking tokens).
    Reasoning,
    /// The `thought` field of a JSON-loop reply.
    Thought,
    /// Reply text: the answer, or text the model wrote outside a tool call.
    Answer,
}

/// One thing that happened during an agent run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AgentEvent {
    /// A model turn starts.
    StepStarted {
        /// Step number, counted from 1.
        step: usize,
        /// Step budget of the run.
        max_steps: usize,
        /// Model that answers this step.
        model: String,
        /// Loop that runs the step.
        loop_kind: LoopKind,
    },
    /// A piece of streamed text.
    Text {
        /// What the text is.
        kind: TextKind,
        /// The new text.
        text: String,
    },
    /// The streamed text of this kind has ended.
    TextEnded(TextKind),
    /// The model called a tool.
    ToolCall {
        /// Call id; the result carries the same id.
        id: String,
        /// Tool name.
        name: String,
        /// Arguments as JSON text.
        arguments: String,
    },
    /// A tool call returned.
    ToolResult {
        /// Id of the call this result answers.
        id: String,
        /// Whether the tool did its job (a command must exit with code 0).
        success: bool,
        /// A few words on the result, such as `4 passed` or `exit 101`.
        summary: String,
        /// The full result text handed back to the model.
        output: String,
    },
    /// A status or warning line, already formatted for the terminal.
    Notice(String),
    /// The model asked the user a question; the run stops for the answer.
    AskUser {
        /// The question to show.
        question: String,
    },
    /// The run used every step without a final answer.
    StepLimitReached {
        /// Step budget of the run.
        max_steps: usize,
    },
    /// The run ended with an answer.
    Finished {
        /// The final answer.
        answer: String,
        /// Whether the answer already arrived as streamed answer text.
        answer_was_streamed: bool,
        /// Absolute paths of the files the run changed.
        changed_files: Vec<PathBuf>,
    },
}

/// Receives agent events.
pub trait Reporter: Send {
    /// Handle one event.
    fn report(&mut self, event: AgentEvent);

    /// Report a notice line.
    fn notice(&mut self, text: String) {
        self.report(AgentEvent::Notice(text));
    }
}

/// Prints agent events to stdout in the one-shot CLI style.
#[derive(Debug, Default)]
pub struct StdoutReporter {
    /// Loop of the last step header; `None` before the first step.
    loop_kind: Option<LoopKind>,
    /// Kind of streamed text being printed right now.
    open_text: Option<TextKind>,
}

impl Reporter for StdoutReporter {
    /// Print `event`.
    fn report(&mut self, event: AgentEvent) {
        let printed = self.render(event);

        print!("{printed}");

        // A failed flush only delays the text until the next newline.
        std::io::stdout().flush().ok();
    }
}

/// The warning shown when a run used all `max_steps` steps without a final
/// answer.
pub fn step_limit_notice(max_steps: usize) -> String {
    format!("\n⚠️  Reached max steps ({max_steps}) without a final answer.")
}

impl StdoutReporter {
    /// The terminal text for `event`.
    pub(super) fn render(&mut self, event: AgentEvent) -> String {
        match event {
            AgentEvent::StepStarted {
                step,
                max_steps,
                model,
                loop_kind,
            } => self.step_header(step, max_steps, &model, loop_kind),
            AgentEvent::Text { kind, text } => self.text(kind, &text),
            AgentEvent::TextEnded(kind) => self.end_text(kind),
            AgentEvent::ToolCall {
                name, arguments, ..
            } => self.tool_call(&name, &arguments),
            AgentEvent::ToolResult { output, .. } => self.tool_result(&output),
            AgentEvent::Notice(text) => format!("{text}\n"),
            AgentEvent::AskUser { question } => format!("\n❓ {question}\n"),
            AgentEvent::StepLimitReached { max_steps } => {
                format!("{}\n", step_limit_notice(max_steps))
            }
            AgentEvent::Finished {
                answer,
                answer_was_streamed,
                changed_files,
            } => self.finished(&answer, answer_was_streamed, &changed_files),
        }
    }

    /// The loop style to print in; the JSON style before the first step.
    fn style(&self) -> LoopKind {
        self.loop_kind.unwrap_or(LoopKind::Json)
    }

    /// The step header line, in the loop's own style.
    fn step_header(
        &mut self,
        step: usize,
        max_steps: usize,
        model: &str,
        loop_kind: LoopKind,
    ) -> String {
        self.loop_kind = Some(loop_kind);

        match loop_kind {
            LoopKind::Native => format!("\n── step {step}/{max_steps} ── {model}\n"),
            LoopKind::Json => format!(
                "\n{COLOR_BOLD}{COLOR_CYAN}🦇 Step {step}/{max_steps}{COLOR_RESET} — model: {COLOR_BOLD}{model}{COLOR_RESET}\n"
            ),
        }
    }

    /// Streamed text, after the marker of its kind when the kind changes.
    fn text(&mut self, kind: TextKind, text: &str) -> String {
        if self.open_text == Some(kind) {
            return text.to_string();
        }

        self.open_text = Some(kind);
        format!("{}{text}", text_opening(self.style(), kind))
    }

    /// The text that ends streamed text of `kind`.
    fn end_text(&mut self, kind: TextKind) -> String {
        self.open_text = None;

        text_closing(self.style(), kind).to_string()
    }

    /// The tool-call line.
    fn tool_call(&self, name: &str, arguments: &str) -> String {
        let description = describe_call(name, arguments);

        match self.style() {
            LoopKind::Native => format!("\n🔧 {description}\n"),
            LoopKind::Json => format!("\n{COLOR_MAGENTA}🔧 {description}{COLOR_RESET}\n"),
        }
    }

    /// The tool's full output.
    fn tool_result(&self, output: &str) -> String {
        match self.style() {
            LoopKind::Native => format!("{output}\n"),
            LoopKind::Json => format!("{COLOR_DIM}{output}{COLOR_RESET}\n"),
        }
    }

    /// The final answer, unless it was streamed already, then the changed files.
    fn finished(
        &self,
        answer: &str,
        answer_was_streamed: bool,
        changed_files: &[PathBuf],
    ) -> String {
        let answer_line = match (self.style(), answer_was_streamed) {
            (LoopKind::Native, _) => format!("\n✅ {answer}\n"),
            (LoopKind::Json, false) => format!("\n{ANSI_GREEN_CHECK} {answer}\n"),
            (LoopKind::Json, true) => String::new(),
        };

        format!("{answer_line}{}", changed_files_text(changed_files))
    }
}

/// The marker printed before streamed text of `kind`.
fn text_opening(loop_kind: LoopKind, kind: TextKind) -> String {
    match (loop_kind, kind) {
        (LoopKind::Native, TextKind::Reasoning) => "💭 ".to_string(),
        (LoopKind::Json, TextKind::Reasoning) => format!("💭 {COLOR_DIM}"),
        (_, TextKind::Thought) => format!("{COLOR_CYAN}🧠 {COLOR_RESET}"),
        (LoopKind::Native, TextKind::Answer) => String::new(),
        (LoopKind::Json, TextKind::Answer) => format!("{ANSI_GREEN_CHECK} "),
    }
}

/// The text printed when streamed text of `kind` ends.
fn text_closing(loop_kind: LoopKind, kind: TextKind) -> String {
    match (loop_kind, kind) {
        (LoopKind::Native, TextKind::Answer) => String::new(),
        (LoopKind::Json, TextKind::Reasoning) => format!("{COLOR_RESET}\n"),
        (LoopKind::Native, _) | (LoopKind::Json, _) => "\n".to_string(),
    }
}

/// The "files changed" header and one clickable link per file; nothing when
/// no file changed.
fn changed_files_text(changed_files: &[PathBuf]) -> String {
    if changed_files.is_empty() {
        return String::new();
    }

    let links: String = changed_files
        .iter()
        .map(|path| format!("   {}\n", clickable_link(path)))
        .collect();

    format!("\n{ANSI_YELLOW_FILES_HEADER}\n{links}")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The text `events` print, in order.
    fn printed(events: Vec<AgentEvent>) -> String {
        let mut reporter = StdoutReporter::default();

        events
            .into_iter()
            .map(|event| reporter.render(event))
            .collect()
    }

    /// A streamed text event.
    fn text(kind: TextKind, text: &str) -> AgentEvent {
        AgentEvent::Text {
            kind,
            text: text.to_string(),
        }
    }

    /// A step header event for `loop_kind`.
    fn step(loop_kind: LoopKind) -> AgentEvent {
        AgentEvent::StepStarted {
            step: 1,
            max_steps: 3,
            model: "m".to_string(),
            loop_kind,
        }
    }

    /// A tool call and its result.
    fn tool_events() -> Vec<AgentEvent> {
        vec![
            AgentEvent::ToolCall {
                id: "c1".to_string(),
                name: "list_files".to_string(),
                arguments: "{}".to_string(),
            },
            AgentEvent::ToolResult {
                id: "c1".to_string(),
                success: true,
                summary: "1 line".to_string(),
                output: "a.rs".to_string(),
            },
        ]
    }

    #[test]
    fn native_events_print_the_native_style() {
        let mut events = vec![
            step(LoopKind::Native),
            text(TextKind::Reasoning, "plan"),
            text(TextKind::Reasoning, " more"),
            AgentEvent::TextEnded(TextKind::Reasoning),
            text(TextKind::Answer, "Hi"),
        ];
        events.extend(tool_events());
        events.push(AgentEvent::Finished {
            answer: "done".to_string(),
            answer_was_streamed: true,
            changed_files: vec![PathBuf::from("/w/a.rs")],
        });

        let output = printed(events);

        assert!(
            output.starts_with("\n── step 1/3 ── m\n💭 plan more\nHi\n🔧 list_files → .\na.rs\n")
        );
        assert!(output.contains("\n✅ done\n"));
        assert!(output.contains("📁 Files changed:"));
        assert!(output.contains("   \x1b]8;;file:///w/a.rs"));
    }

    #[test]
    fn json_events_print_the_json_style() {
        let mut events = vec![
            step(LoopKind::Json),
            text(TextKind::Reasoning, "hmm"),
            AgentEvent::TextEnded(TextKind::Reasoning),
            text(TextKind::Thought, "look"),
            AgentEvent::TextEnded(TextKind::Thought),
            text(TextKind::Answer, "hello"),
            AgentEvent::TextEnded(TextKind::Answer),
        ];
        events.extend(tool_events());
        events.push(AgentEvent::Finished {
            answer: "hello".to_string(),
            answer_was_streamed: true,
            changed_files: Vec::new(),
        });

        let output = printed(events);

        assert!(output.contains("🦇 Step 1/3"));
        assert!(output.contains(&format!("💭 {COLOR_DIM}hmm{COLOR_RESET}\n")));
        assert!(output.contains(&format!("{COLOR_CYAN}🧠 {COLOR_RESET}look\n")));
        assert!(output.contains(&format!("{ANSI_GREEN_CHECK} hello\n")));
        assert!(output.contains(&format!("{COLOR_MAGENTA}🔧 list_files → .{COLOR_RESET}")));
        assert!(output.contains(&format!("{COLOR_DIM}a.rs{COLOR_RESET}")));
        assert_eq!(output.matches("hello").count(), 1);
    }

    #[test]
    fn an_unstreamed_json_answer_notices_and_questions_are_printed() {
        let output = printed(vec![
            AgentEvent::Notice("⚠️ careful".to_string()),
            AgentEvent::AskUser {
                question: "Which file?".to_string(),
            },
            AgentEvent::Finished {
                answer: "all set".to_string(),
                answer_was_streamed: false,
                changed_files: Vec::new(),
            },
        ]);

        assert_eq!(
            output,
            format!("⚠️ careful\n\n❓ Which file?\n\n{ANSI_GREEN_CHECK} all set\n")
        );
    }

    #[test]
    fn a_native_reply_has_no_closing_text() {
        let output = printed(vec![
            step(LoopKind::Native),
            text(TextKind::Answer, "Hi"),
            AgentEvent::TextEnded(TextKind::Answer),
        ]);

        assert!(output.ends_with("Hi"));
    }

    #[test]
    fn the_stdout_reporter_prints_events() {
        let mut reporter = StdoutReporter::default();

        reporter.notice("a line".to_string());
    }
}
