//! Structural quality gates for model output.
//!
//! Small local models routinely emit stubs and then call `finish`. Rather than
//! adding more instructions to the system prompt, these gates reject the
//! output programmatically before it can be written or accepted.

use super::tool::Tool;
use crate::constants::tools::SUCCESS_EXIT_MARKER;

/// Number of times the agent may be asked to keep working before a shallow
/// finish is allowed to pass. This prevents an infinite loop with a weak model.
pub(crate) const MAX_FINISH_RETRIES: usize = 3;

/// Rejects a finish that comes after a file change but before a check.
///
/// A run that changed no files can always finish. A run that changed files in
/// a project with a build or test manifest must first run a command that
/// exits with code 0 after its last change.
#[derive(Debug, Default, PartialEq, Eq)]
pub(crate) struct FinishGate {
    /// Number of finishes already rejected.
    retries: usize,
    /// Files changed since the last command that exited with code 0.
    has_unchecked_changes: bool,
}

impl FinishGate {
    /// The feedback for the model when a finish now would come before the
    /// check, or `None` when the finish may pass.
    ///
    /// `needs_check` is true when the project has a build or test manifest.
    /// Each rejection uses one retry; after [`MAX_FINISH_RETRIES`] the finish
    /// passes, so a model that keeps finishing early cannot keep the loop
    /// running until the step budget is spent.
    pub(crate) fn reject_finish(&mut self, needs_check: bool) -> Option<&'static str> {
        let has_retries_left = self.retries < MAX_FINISH_RETRIES;
        let is_premature = matches!(
            (needs_check, self.has_unchecked_changes, has_retries_left),
            (true, true, true)
        );

        if !is_premature {
            return None;
        }

        self.retries += 1;
        Some(UNVERIFIED_FEEDBACK)
    }

    /// Record one step of work: `files_changed` when a tool changed files,
    /// `command_passed` when a command exited with code 0.
    pub(crate) fn record(&mut self, files_changed: bool, command_passed: bool) {
        let has_changes = self.has_unchecked_changes | files_changed;

        self.has_unchecked_changes = has_changes & !command_passed;
    }

    /// Record the result of one tool call, given whether it changed files.
    pub(crate) fn record_tool_result(&mut self, tool: &Tool, result: &str, files_changed: bool) {
        let is_command = matches!(tool, Tool::RunCommand { .. });
        let command_passed = is_command && command_succeeded(result);

        self.record(files_changed, command_passed);
    }
}

/// Feedback for a finish after a file change but before a passing check.
const UNVERIFIED_FEEDBACK: &str = "Finish rejected: the changed files are not verified. Run the \
     build or test command, read its output, and fix any errors. Only call finish after a \
     command exits with code 0.";

/// True when a `run_command` tool result reports a zero exit code.
pub(crate) fn command_succeeded(output: &str) -> bool {
    output.contains(SUCCESS_EXIT_MARKER)
}

/// Phrases that only a stub template contains, each with the reason shown to
/// the model. Single words such as "placeholder", "stub" or "hello world" are
/// not here: they appear in real files (an HTML `placeholder=` attribute, a
/// test that prints "hello world"), and refusing those writes left the agent
/// looping until its step budget ran out.
const PLACEHOLDER_MARKERS: &[(&str, &str)] = &[
    ("todo: implement", "TODO placeholder"),
    ("your code here", "code placeholder"),
    ("your code goes here", "code placeholder"),
    ("your implementation here", "implementation placeholder"),
    ("implement this here", "implementation placeholder"),
    ("add your code here", "code placeholder"),
    ("write your code here", "code placeholder"),
    ("... rest of code", "elided code"),
    ("rest of the code remains", "elided code"),
    ("lorem ipsum", "lorem ipsum filler"),
];

/// Return a reason when content is likely a placeholder that should not be
/// written to disk.
pub(crate) fn placeholder_reason(content: &str) -> Option<&'static str> {
    let content = content.to_ascii_lowercase();
    PLACEHOLDER_MARKERS
        .iter()
        .find_map(|(marker, reason)| content.contains(marker).then_some(*reason))
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Context;

    #[test]
    fn detects_placeholder_variants_case_insensitively() {
        assert_eq!(
            placeholder_reason("TODO: Implement binary search"),
            Some("TODO placeholder")
        );
        assert_eq!(
            placeholder_reason("Your Code Here"),
            Some("code placeholder")
        );
        assert_eq!(
            placeholder_reason("fn main() {\n    // ... rest of code\n}"),
            Some("elided code")
        );
    }

    #[test]
    fn real_files_that_mention_stub_words_are_written() {
        let form = r#"<input name="email" placeholder="you@example.com">"#;
        let test = "#[test]\nfn greets() {\n    assert_eq!(greet(), \"hello world\");\n}";
        let doc = "/// Elided bodies are written as `// ...` in the docs.\npub fn stub_free() {}";

        assert_eq!(placeholder_reason(form), None);
        assert_eq!(placeholder_reason(test), None);
        assert_eq!(placeholder_reason(doc), None);
    }

    #[test]
    fn allows_real_content() {
        let content = "\
            use std::collections::HashMap;\n\
            fn count<T: Eq + std::hash::Hash>(items: &[T]) -> HashMap<&T, usize> {\n\
                items.iter().fold(HashMap::new(), |mut counts, item| {\n\
                    *counts.entry(item).or_insert(0) += 1;\n\
                    counts\n\
                })\n\
            }\n";

        assert_eq!(placeholder_reason(content), None);
    }

    #[test]
    fn a_run_without_changes_always_finishes() {
        let mut gate = FinishGate::default();

        assert_eq!(gate.reject_finish(true), None);
        assert_eq!(gate.reject_finish(false), None);
    }

    #[test]
    fn a_change_needs_a_passing_check_after_it() -> anyhow::Result<()> {
        let mut gate = FinishGate::default();
        let command = Tool::RunCommand {
            command: "cargo test".to_string(),
        };
        let write = Tool::WriteFile {
            path: "a.rs".to_string(),
            content: "fn a() {}".to_string(),
        };

        gate.record_tool_result(&write, "wrote", true);
        assert_eq!(gate.reject_finish(false), None);
        let feedback = gate.reject_finish(true).context("an unchecked change")?;
        assert!(feedback.contains("not verified"), "{feedback}");

        gate.record_tool_result(&command, "[exit code 101]", false);
        assert!(gate.reject_finish(true).is_some());

        gate.record_tool_result(&command, "ok\n[exit code 0]\n", false);
        assert_eq!(gate.reject_finish(true), None);

        gate.record_tool_result(&write, "wrote", true);
        assert!(gate.reject_finish(true).is_some());
        assert_eq!(
            gate.reject_finish(true),
            None,
            "a weak model must not loop forever"
        );

        Ok(())
    }

    #[test]
    fn a_read_does_not_count_as_a_check() {
        let mut gate = FinishGate::default();
        let read = Tool::ReadFile {
            path: "log.txt".to_string(),
            max_chars: 100,
        };

        gate.record(true, false);
        gate.record_tool_result(&read, "[exit code 0]", false);

        assert!(gate.reject_finish(true).is_some());
    }

    #[test]
    fn command_success_reads_the_exit_code_marker() {
        assert!(command_succeeded("compiled!\n[exit code 0]\n"));
        assert!(!command_succeeded(
            "error: could not compile\n[exit code 101]\n"
        ));
    }
}
