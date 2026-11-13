//! Structural quality gates for model output.
//!
//! Small local models routinely emit stubs and then call `finish`. Rather than
//! adding more instructions to the system prompt, these gates reject the
//! output programmatically before it can be written or accepted.

use super::environment::project_has_build_manifest;
use super::tool::Tool;
use crate::constants::tools::SUCCESS_EXIT_MARKER;
use std::path::Path;

/// Number of times the agent may be asked to keep working before a shallow
/// finish is allowed to pass. This prevents an infinite loop with a weak model.
pub(crate) const MAX_FINISH_RETRIES: usize = 3;

/// Tracks how many times a finish was blocked and whether the work was
/// verified by a successful command.
#[derive(Debug, Default, PartialEq, Eq)]
pub(crate) struct FileWorkGate {
    /// Number of shallow-finish attempts already rejected.
    retries: usize,
    /// A `run_command` tool call exited with code 0 at least once.
    verified: bool,
}

impl FileWorkGate {
    /// Reject a finish that came too early, returning the feedback to send to
    /// the model, or `None` when the finish may pass.
    ///
    /// # Parameters
    ///
    /// - `requires_file_work`: the task asks for files to be written.
    /// - `is_read_only`: writes and commands are disabled entirely.
    /// - `changed_files`: number of files written so far.
    /// - `needs_verification`: the workspace has a build manifest, so a
    ///   successful build/test is expected before the task counts as done.
    ///
    /// Each rejection consumes one retry; after [`MAX_FINISH_RETRIES`] the
    /// gate lets the finish through, so a model that keeps finishing early
    /// cannot keep the loop running until the step budget is spent.
    pub(crate) fn reject_finish(
        &mut self,
        requires_file_work: bool,
        is_read_only: bool,
        changed_files: usize,
        needs_verification: bool,
    ) -> Option<&'static str> {
        if !self.applies(requires_file_work, is_read_only) {
            return None;
        }

        let feedback = self.missing_work(changed_files, needs_verification);
        self.count_rejection(feedback)
    }

    /// True when the gate checks this finish: the task asks for files, writes
    /// are allowed, and retries are left.
    fn applies(&self, requires_file_work: bool, is_read_only: bool) -> bool {
        let has_retries_left = self.retries < MAX_FINISH_RETRIES;

        matches!(
            (requires_file_work, is_read_only, has_retries_left),
            (true, false, true)
        )
    }

    /// The feedback for work that is still missing: no files written, or no
    /// successful command when one is expected. `None` when nothing is missing.
    fn missing_work(&self, changed_files: usize, needs_verification: bool) -> Option<&'static str> {
        if changed_files == 0 {
            return Some(NO_FILES_FEEDBACK);
        }

        let is_unverified = matches!((needs_verification, self.verified), (true, false));
        is_unverified.then_some(UNVERIFIED_FEEDBACK)
    }

    /// Use up one retry when `feedback` rejects the finish, and return it.
    fn count_rejection(&mut self, feedback: Option<&'static str>) -> Option<&'static str> {
        if feedback.is_some() {
            self.retries += 1;
        }

        feedback
    }

    /// Record a successful verification command (exit code 0).
    pub(crate) fn record_verification(&mut self) {
        self.verified = true;
    }

    /// Record the result of one tool call: a `run_command` that exited with
    /// code 0 counts as verification.
    pub(crate) fn record_tool_result(&mut self, tool: &Tool, result: &str) {
        let is_successful_command =
            matches!(tool, Tool::RunCommand { .. }) && command_succeeded(result);

        self.verified |= is_successful_command;
    }
}

/// Feedback for a finish before any file was written.
const NO_FILES_FEEDBACK: &str = "Finish rejected: the task requires file changes, but no files \
     were written. Write the complete implementation or document first; only then call finish.";

/// Feedback for a finish before a build or test command succeeded.
const UNVERIFIED_FEEDBACK: &str = "Finish rejected: the work is not verified. Run the build or \
     test command, read its output, and fix any errors. Only call finish after a command exits \
     with code 0.";

/// True when a `run_command` tool result reports a zero exit code.
pub(crate) fn command_succeeded(output: &str) -> bool {
    output.contains(SUCCESS_EXIT_MARKER)
}

/// True when a task that asks for files runs in a workspace with a build
/// manifest, so a successful build or test is expected before it counts as
/// done.
pub(crate) fn needs_verification(requires_file_work: bool, cwd: &Path) -> bool {
    requires_file_work && project_has_build_manifest(cwd)
}

/// True when a task asks the model to produce or change files.
pub(crate) fn requires_file_work(task: &str) -> bool {
    let task = task.to_ascii_lowercase();
    FILE_WORK_HINTS.iter().any(|hint| task.contains(hint))
}

/// Words that usually indicate an implementation/document task rather than a
/// question. Kept intentionally narrow to avoid blocking plain Q&A.
const FILE_WORK_HINTS: &[&str] = &[
    "implement",
    "implementation",
    "write ",
    "create ",
    "add ",
    "fix ",
    "refactor",
    "generate",
    "full ",
    "readme",
    "documentation",
    "design ",
    "tests",
];

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
    fn detects_file_work_tasks() {
        assert!(requires_file_work("create a full DSA implementation"));
        assert!(requires_file_work("write complete README for the project"));
        assert!(requires_file_work("fix the failing test"));
        assert!(!requires_file_work("explain what a mutex is"));
        assert!(!requires_file_work("why is this slow"));
    }

    #[test]
    fn gate_blocks_finish_until_files_exist() {
        let mut gate = FileWorkGate::default();

        assert!(gate.reject_finish(true, false, 0, false).is_some());
        assert_eq!(gate.reject_finish(false, false, 0, false), None);
        assert_eq!(gate.reject_finish(true, true, 0, false), None);
        assert_eq!(gate.reject_finish(true, false, 1, false), None);
    }

    #[test]
    fn gate_blocks_finish_until_a_command_succeeds() {
        let mut gate = FileWorkGate::default();

        assert!(gate.reject_finish(true, false, 1, true).is_some());
        gate.record_verification();
        assert_eq!(gate.reject_finish(true, false, 1, true), None);
    }

    #[test]
    fn gate_stops_after_max_retries() {
        let mut gate = FileWorkGate::default();

        for _ in 0..MAX_FINISH_RETRIES {
            assert!(gate.reject_finish(true, false, 0, false).is_some());
        }

        assert_eq!(
            gate.reject_finish(true, false, 0, false),
            None,
            "a weak model must not loop forever"
        );
    }

    #[test]
    fn feedback_names_the_rejection_reason() -> anyhow::Result<()> {
        let mut gate = FileWorkGate::default();

        let no_files = gate
            .reject_finish(true, false, 0, true)
            .context("no files written")?;
        assert!(no_files.contains("no files were written"), "{no_files}");

        let unverified = gate
            .reject_finish(true, false, 2, true)
            .context("no command succeeded")?;
        assert!(unverified.contains("not verified"), "{unverified}");

        Ok(())
    }

    #[test]
    fn only_a_successful_command_verifies_the_work() {
        let mut gate = FileWorkGate::default();
        let command = Tool::RunCommand {
            command: "cargo test".to_string(),
        };
        let read = Tool::ReadFile {
            path: "log.txt".to_string(),
            max_chars: 100,
        };

        gate.record_tool_result(&read, "[exit code 0]");
        gate.record_tool_result(&command, "[exit code 101]");
        assert!(gate.reject_finish(true, false, 1, true).is_some());

        gate.record_tool_result(&command, "ok\n[exit code 0]\n");
        assert_eq!(gate.reject_finish(true, false, 1, true), None);
    }

    #[test]
    fn command_success_reads_the_exit_code_marker() {
        assert!(command_succeeded("compiled!\n[exit code 0]\n"));
        assert!(!command_succeeded(
            "error: could not compile\n[exit code 101]\n"
        ));
    }

    #[test]
    fn build_manifest_detection_is_file_based() -> anyhow::Result<()> {
        let root = crate::test_support::unique_temp_dir("openbatrangs-quality-test");

        assert!(!project_has_build_manifest(&root));
        assert!(!needs_verification(true, &root));

        std::fs::write(root.join("Cargo.toml"), "[package]")?;
        assert!(project_has_build_manifest(&root));
        assert!(needs_verification(true, &root));
        assert!(!needs_verification(false, &root));

        std::fs::remove_dir_all(root)?;
        Ok(())
    }
}
