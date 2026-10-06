//! Structural quality gates for model output.
//!
//! Small local models routinely emit stubs and then call `finish`. Rather than
//! adding more instructions to the system prompt, these gates reject the
//! output programmatically before it can be written or accepted.

/// Number of times the agent may be asked to keep working before we allow a
/// shallow finish to pass. This prevents an infinite loop with a weak model.
pub(crate) const MAX_FINISH_RETRIES: usize = 3;

/// Tracks how many times a finish was blocked because no file work happened.
#[derive(Debug, Default, PartialEq, Eq)]
pub(crate) struct FileWorkGate {
    /// Number of shallow-finish attempts already rejected.
    retries: usize,
}

impl FileWorkGate {
    /// Whether a finish should be rejected so the model can continue working.
    pub(crate) fn should_block_finish(
        &self,
        requires_file_work: bool,
        is_read_only: bool,
        changed_file_count: usize,
    ) -> bool {
        requires_file_work
            && !is_read_only
            && changed_file_count == 0
            && self.retries < MAX_FINISH_RETRIES
    }

    /// Record one rejected finish attempt.
    pub(crate) fn record_retry(&mut self) {
        self.retries += 1;
    }

    /// User-role feedback sent to the model after a blocked finish.
    pub(crate) fn feedback() -> &'static str {
        "Finish rejected: the task requires file changes, but no files were written. \
         Write the complete implementation or document first; only then call finish."
    }
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
    fn finish_gate_blocks_only_file_work_without_changes() {
        let gate = FileWorkGate::default();

        assert!(gate.should_block_finish(true, false, 0));
        assert!(!gate.should_block_finish(false, false, 0));
        assert!(!gate.should_block_finish(true, true, 0));
        assert!(!gate.should_block_finish(true, false, 1));
    }

    #[test]
    fn finish_gate_stops_after_max_retries() {
        let mut gate = FileWorkGate::default();

        for _ in 0..MAX_FINISH_RETRIES {
            assert!(gate.should_block_finish(true, false, 0));
            gate.record_retry();
        }

        assert!(!gate.should_block_finish(true, false, 0));
    }

    #[test]
    fn feedback_is_direct() {
        assert!(FileWorkGate::feedback().contains("Finish rejected"));
    }
}
