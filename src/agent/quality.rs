//! Structural quality gates for model output.
//!
//! Small local models routinely emit stubs and then call `finish`. Rather than
//! adding more instructions to the system prompt, these gates reject the
//! output programmatically before it can be written or accepted.

use crate::constants::tools::SUCCESS_EXIT_MARKER;

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
    /// gate lets the finish through so a weak model cannot loop forever.
    pub(crate) fn reject_finish(
        &mut self,
        requires_file_work: bool,
        is_read_only: bool,
        changed_files: usize,
        needs_verification: bool,
    ) -> Option<&'static str> {
        if !requires_file_work || is_read_only || self.retries >= MAX_FINISH_RETRIES {
            return None;
        }

        let feedback = if changed_files == 0 {
            "Finish rejected: the task requires file changes, but no files were written. \
             Write the complete implementation or document first; only then call finish."
        } else if needs_verification && !self.verified {
            "Finish rejected: the work is not verified. Run the build or test command, \
             read its output, and fix any errors. Only call finish after a command exits \
             with code 0."
        } else {
            return None;
        };

        self.retries += 1;
        Some(feedback)
    }

    /// Record a successful verification command (exit code 0).
    pub(crate) fn record_verification(&mut self) {
        self.verified = true;
    }
}

/// True when a `run_command` tool result reports a zero exit code.
pub(crate) fn command_succeeded(output: &str) -> bool {
    output.contains(SUCCESS_EXIT_MARKER)
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

/// Reasons why a write may look like a placeholder/stub.
const PLACEHOLDER_MARKERS: &[(&str, &str)] = &[
    ("todo: implement", "TODO placeholder"),
    ("your code here", "code placeholder"),
    ("your code goes here", "code placeholder"),
    ("your implementation here", "implementation placeholder"),
    ("implement this here", "implementation placeholder"),
    ("add your code here", "code placeholder"),
    ("write your code here", "code placeholder"),
    ("coming soon", "coming soon placeholder"),
    ("hello world", "hello-world stub"),
    ("lorem ipsum", "lorem ipsum filler"),
    ("not implemented", "unimplemented stub"),
    ("to be implemented", "unimplemented stub"),
    ("placeholder", "placeholder text"),
    ("... rest of code", "elided code"),
    ("// ...", "elided code"),
    ("# ...", "elided content"),
    ("stub", "stub content"),
    ("fixme", "unimplemented stub"),
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

    #[test]
    fn detects_placeholder_variants_case_insensitively() {
        assert_eq!(
            placeholder_reason("This is a Placeholder for the answer"),
            Some("placeholder text")
        );
        assert_eq!(
            placeholder_reason("TODO: Implement binary search"),
            Some("TODO placeholder")
        );
        assert_eq!(
            placeholder_reason("Your Code Here"),
            Some("code placeholder")
        );
        assert_eq!(placeholder_reason("hello world"), Some("hello-world stub"));
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
    fn feedback_names_the_rejection_reason() {
        let mut gate = FileWorkGate::default();

        let no_files = gate.reject_finish(true, false, 0, true).unwrap();
        assert!(no_files.contains("no files were written"), "{no_files}");

        let unverified = gate.reject_finish(true, false, 2, true).unwrap();
        assert!(unverified.contains("not verified"), "{unverified}");
    }

    #[test]
    fn command_success_reads_the_exit_code_marker() {
        assert!(command_succeeded("compiled!\n[exit code 0]\n"));
        assert!(!command_succeeded(
            "error: could not compile\n[exit code 101]\n"
        ));
    }

    #[test]
    fn build_manifest_detection_is_file_based() {
        let root = crate::test_support::unique_temp_dir("openbatrangs-quality-test");

        assert!(!super::super::environment::project_has_build_manifest(
            &root
        ));

        std::fs::write(root.join("Cargo.toml"), "[package]").unwrap();
        assert!(super::super::environment::project_has_build_manifest(&root));

        std::fs::remove_dir_all(root).unwrap();
    }
}
