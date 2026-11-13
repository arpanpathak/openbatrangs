//! # Short tool-result summaries
//!
//! The TUI shows one line per tool call with a few words on its result:
//! `4 passed` for a test run, `exit 101` for a failed command, `+32 −0` for a
//! written file. This module derives those words from the tool and its output.

use super::quality::command_succeeded;
use super::tool::Tool;
use crate::constants::tools::EXIT_CODE_PREFIX;
use regex::Regex;
use std::sync::OnceLock;

/// The summary of a tool call that returned `output`.
pub(super) fn result_summary(tool: &Tool, output: &str) -> String {
    match tool {
        Tool::RunCommand { .. } => command_summary(output),
        Tool::WriteFile { content, .. } => line_change_summary(content, ""),
        Tool::EditFile {
            old_string,
            new_string,
            ..
        } => line_change_summary(new_string, old_string),
        Tool::GrepFiles { .. } => counted(output.lines().count(), "match", "matches"),
        Tool::ListFiles { .. }
        | Tool::ReadFile { .. }
        | Tool::WebSearch { .. }
        | Tool::WebRead { .. }
        | Tool::Finish { .. }
        | Tool::AskUser { .. } => counted(output.lines().count(), "line", "lines"),
    }
}

/// `4 passed` for a successful test run, otherwise the exit code.
fn command_summary(output: &str) -> String {
    match (command_succeeded(output), passed_tests(output)) {
        (true, Some(passed)) => format!("{passed} passed"),
        (true, None) | (false, _) => exit_summary(output),
    }
}

/// The number of passed tests a test runner reported, summed over every
/// `N passed` it printed; `None` when it printed none.
fn passed_tests(output: &str) -> Option<usize> {
    static PASSED: OnceLock<Option<Regex>> = OnceLock::new();
    let pattern = PASSED.get_or_init(|| Regex::new(r"(\d+) passed").ok());

    let counts: Vec<usize> = pattern
        .iter()
        .flat_map(|pattern| pattern.captures_iter(output))
        .filter_map(|captures| captures.get(1)?.as_str().parse().ok())
        .collect();

    (!counts.is_empty()).then(|| counts.iter().sum())
}

/// `exit N` from the `[exit code N]` line of a command's output.
fn exit_summary(output: &str) -> String {
    let code = output
        .rsplit_once(EXIT_CODE_PREFIX)
        .and_then(|(_before, after)| after.split(']').next())
        .unwrap_or("?");

    format!("exit {code}")
}

/// `+added −removed`, counted in lines.
fn line_change_summary(added: &str, removed: &str) -> String {
    format!("+{} −{}", added.lines().count(), removed.lines().count())
}

/// `count` followed by the singular or plural noun.
fn counted(count: usize, singular: &str, plural: &str) -> String {
    match count {
        1 => format!("1 {singular}"),
        _ => format!("{count} {plural}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A `run_command` tool.
    fn command() -> Tool {
        Tool::RunCommand {
            command: "cargo test".to_string(),
        }
    }

    #[test]
    fn commands_report_tests_or_the_exit_code() {
        let tests = "test result: ok. 3 passed; 0 failed\ntest result: ok. 1 passed\n[exit code 0]";
        let failed = "test result: FAILED. 2 passed; 1 failed\n[exit code 101]";

        assert_eq!(result_summary(&command(), tests), "4 passed");
        assert_eq!(result_summary(&command(), failed), "exit 101");
        assert_eq!(result_summary(&command(), "built\n[exit code 0]"), "exit 0");
        assert_eq!(result_summary(&command(), "cut off"), "exit ?");
    }

    #[test]
    fn writes_and_edits_count_lines() {
        let write = Tool::WriteFile {
            path: "a.rs".to_string(),
            content: "a\nb\nc".to_string(),
        };
        let edit = Tool::EditFile {
            path: "a.rs".to_string(),
            old_string: "a".to_string(),
            new_string: "x\ny".to_string(),
            replace_all: false,
        };

        assert_eq!(result_summary(&write, "wrote"), "+3 −0");
        assert_eq!(result_summary(&edit, "edited"), "+2 −1");
    }

    #[test]
    fn other_tools_count_output_lines() {
        let grep = Tool::GrepFiles {
            pattern: "fn".to_string(),
            path: ".".to_string(),
            max_results: 5,
        };
        let read = Tool::ReadFile {
            path: "a.rs".to_string(),
            max_chars: 10,
        };

        assert_eq!(result_summary(&grep, "a.rs:1: fn a"), "1 match");
        assert_eq!(result_summary(&grep, "a\nb"), "2 matches");
        assert_eq!(result_summary(&read, "one"), "1 line");
        assert_eq!(result_summary(&read, ""), "0 lines");
    }
}
