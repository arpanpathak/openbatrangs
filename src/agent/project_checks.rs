//! # A workspace's own checks before a finish
//!
//! The finish gate of [`super::quality`] asks only for some command that
//! exits with code 0 after the last change. A workspace can ask for more by
//! listing commands in `.openbatrangs/checks`, one per line: its tests, a
//! formatter in check mode, a linter that denies warnings, a complexity
//! limit. When the model finishes after changing files, openbatrangs runs
//! each command in the sandbox. A failing command rejects the finish, and
//! the model receives the command with the end of its output, so it fixes
//! what the checks report instead of declaring the task done.
//!
//! Blank lines and lines that start with `#` are ignored. After
//! [`MAX_CHECK_REJECTIONS`] rejections the finish passes, so a model that
//! cannot fix the code still ends its run.

use crate::agent::quality::command_succeeded;
use crate::constants::agent::COMMAND_TIMEOUT_SECONDS;
use crate::tools;
use crate::tools::sandbox::Sandbox;
use std::fs;
use std::path::Path;

/// The file that lists a workspace's checks, relative to its root.
pub(crate) const CHECKS_FILE: &str = ".openbatrangs/checks";

/// Finishes the checks may reject in one run.
const MAX_CHECK_REJECTIONS: usize = 5;

/// Characters of a failing command's output sent to the model.
const FAILURE_OUTPUT_CHARS: usize = 3000;

/// The checks of one workspace and how often they rejected a finish.
#[derive(Debug, Default)]
pub(crate) struct ProjectChecks {
    /// Commands from the checks file, in order.
    commands: Vec<String>,
    /// Finishes rejected so far.
    rejections: usize,
}

impl ProjectChecks {
    /// The checks listed in `cwd`'s checks file; none when it does not exist.
    pub(crate) fn load(cwd: &Path) -> Self {
        let text = fs::read_to_string(cwd.join(CHECKS_FILE)).unwrap_or_default();
        let commands = text
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty() && !line.starts_with('#'))
            .map(str::to_string)
            .collect();

        Self {
            commands,
            rejections: 0,
        }
    }

    /// The feedback for the model when a check fails, or `None` when the
    /// finish may pass: there are no checks, the rejections are used up, or
    /// every check passes.
    pub(crate) async fn feedback(&mut self, cwd: &Path, sandbox: Sandbox) -> Option<String> {
        let has_turn = !self.commands.is_empty() && self.rejections < MAX_CHECK_REJECTIONS;

        if !has_turn {
            return None;
        }

        let failures = self.failures(cwd, sandbox).await;
        self.rejections += usize::from(!failures.is_empty());

        (!failures.is_empty()).then(|| failure_message(&failures))
    }

    /// Each command that fails, with its output.
    async fn failures(&self, cwd: &Path, sandbox: Sandbox) -> Vec<(String, String)> {
        let mut failures = Vec::new();

        for command in &self.commands {
            let output = tools::run_command(cwd, command, COMMAND_TIMEOUT_SECONDS, sandbox)
                .await
                .unwrap_or_else(|error| format!("{error:#}"));

            if !command_succeeded(&output) {
                failures.push((command.to_string(), output));
            }
        }

        failures
    }
}

/// The message that tells the model which checks failed and what they said.
fn failure_message(failures: &[(String, String)]) -> String {
    let reports: Vec<String> = failures
        .iter()
        .map(|(command, output)| format!("$ {command}\n{}", tail(output, FAILURE_OUTPUT_CHARS)))
        .collect();

    format!(
        "The project's checks failed, so the task is not done. Fix what each check reports, \
         without changing the checks, then finish again.\n\n{}",
        reports.join("\n\n")
    )
}

/// The last `limit` characters of `text`.
fn tail(text: &str, limit: usize) -> &str {
    let skipped = text.chars().count().saturating_sub(limit);
    let start = text
        .char_indices()
        .nth(skipped)
        .map_or(text.len(), |(index, _character)| index);

    &text[start..]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::unique_temp_dir;

    /// A workspace whose checks file holds `checks`.
    fn workspace(name: &str, checks: &str) -> std::io::Result<std::path::PathBuf> {
        let root = unique_temp_dir(name);
        fs::create_dir_all(root.join(".openbatrangs"))?;
        fs::write(root.join(CHECKS_FILE), checks)?;

        Ok(root)
    }

    #[tokio::test]
    async fn a_failing_check_rejects_the_finish_with_its_output() -> anyhow::Result<()> {
        let root = workspace(
            "openbatrangs-checks-fail",
            "# the formatter\ntrue\n\necho too complex; exit 1\n",
        )?;
        let mut checks = ProjectChecks::load(&root);

        let feedback = checks
            .feedback(&root, Sandbox::Off)
            .await
            .unwrap_or_default();

        assert!(feedback.starts_with("The project's checks failed"));
        assert!(feedback.contains("$ echo too complex; exit 1\ntoo complex"));
        assert!(
            !feedback.contains("$ true"),
            "a passing check is not reported"
        );

        fs::remove_dir_all(root)?;
        Ok(())
    }

    #[tokio::test]
    async fn passing_checks_or_no_checks_file_let_the_finish_pass() -> anyhow::Result<()> {
        let root = workspace("openbatrangs-checks-pass", "true\n")?;
        let mut passing = ProjectChecks::load(&root);
        let mut missing = ProjectChecks::load(&root.join("no-such-folder"));

        assert_eq!(passing.feedback(&root, Sandbox::Off).await, None);
        assert_eq!(missing.feedback(&root, Sandbox::Off).await, None);

        fs::remove_dir_all(root)?;
        Ok(())
    }

    #[tokio::test]
    async fn the_checks_stop_rejecting_after_their_limit() -> anyhow::Result<()> {
        let root = workspace("openbatrangs-checks-limit", "false\n")?;
        let mut checks = ProjectChecks::load(&root);

        for _rejection in 0..MAX_CHECK_REJECTIONS {
            assert!(checks.feedback(&root, Sandbox::Off).await.is_some());
        }

        assert_eq!(checks.feedback(&root, Sandbox::Off).await, None);

        fs::remove_dir_all(root)?;
        Ok(())
    }

    #[tokio::test]
    async fn a_check_that_cannot_start_counts_as_failed() -> anyhow::Result<()> {
        let root = workspace("openbatrangs-checks-no-start", "true\n")?;
        let mut checks = ProjectChecks::load(&root);
        let not_a_folder = root.join(CHECKS_FILE);

        let feedback = checks
            .feedback(&not_a_folder, Sandbox::Off)
            .await
            .unwrap_or_default();

        assert!(feedback.contains("$ true"));

        fs::remove_dir_all(root)?;
        Ok(())
    }

    #[test]
    fn a_long_output_keeps_its_end() {
        assert_eq!(tail("abcdef", 3), "def");
        assert_eq!(tail("ab", 3), "ab");
    }
}
