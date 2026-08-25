//! # Tool execution, confirmation, and terminal-friendly result formatting
//!
//! Once a [`Tool`] has been parsed and typed, this module is responsible for
//! actually doing the work: listing files, reading/writing files, grepping,
//! running shell commands, and formatting results for the terminal.
//!
//! Mutating tools are protected by two guards:
//!
//! - [`AgentConfig::is_read_only`] disables writes/commands entirely.
//! - [`Confirmer`] asks the user before a write/command when confirm mode is on.
//!
//! Results are also made terminal-friendly: written files become OSC 8
//! hyperlinks and control characters are stripped so a model cannot inject
//! escape sequences through a crafted file name.
//!
//! ## References
//!
//! - OSC 8 hyperlinks: <https://gist.github.com/egmontkob/eb114294efbcd5adb1944c9f3cb5feda>
//! - Human-in-the-loop safety: <https://en.wikipedia.org/wiki/Human-in-the-loop>

use super::confirm::Confirmer;
use super::reporter::Reporter;
use super::tool::Tool;
use super::AgentConfig;
use crate::constants::agent::COMMAND_TIMEOUT_SECONDS;
use crate::constants::ansi::{ANSI_GREEN_CHECK, ANSI_YELLOW_FILES_HEADER};
use crate::constants::tools::{DEFAULT_LIST_DEPTH, MAX_TOOL_OUTPUT};
use crate::tools;
use anyhow::{bail, Result};
use std::path::Path;

/// Execute a tool call, returning its output or a formatted error message.
///
/// Wraps [`execute_tool`] so the caller always gets a displayable string.
pub(super) async fn execute_tool_or_report_error<C: Confirmer>(
    config: &AgentConfig,
    cwd: &Path,
    tool: &Tool,
    changed_files: &mut Vec<String>,
    confirmer: &mut C,
) -> String {
    match execute_tool(config, cwd, tool, changed_files, confirmer).await {
        Ok(text) => text,
        Err(error) => format!("Tool error: {error:#}"),
    }
}

/// Execute a typed tool call and return its text result.
///
/// Mutating tools (write, run command) are guarded by read-only mode and
/// user confirmation.
pub(super) async fn execute_tool<C: Confirmer>(
    config: &AgentConfig,
    cwd: &Path,
    tool: &Tool,
    changed_files: &mut Vec<String>,
    confirmer: &mut C,
) -> Result<String> {
    match tool {
        Tool::ListFiles { path } => tools::list_files(cwd, path, DEFAULT_LIST_DEPTH),
        Tool::ReadFile { path, max_chars } => {
            let max_chars = (*max_chars).min(MAX_TOOL_OUTPUT);
            tools::read_file(cwd, path, max_chars)
        }
        Tool::GrepFiles {
            pattern,
            path,
            max_results,
        } => tools::grep_files(cwd, pattern, path, *max_results),
        Tool::WriteFile { path, content } => {
            ensure_not_read_only(config)?;
            confirm_or_abort(config, confirmer, &format!("write file '{path}'?")).await?;
            let result = tools::write_file(cwd, path, content)?;
            changed_files.push(path.clone());
            Ok(format!("{result}\n📎 {}", clickable_path(cwd, path)))
        }
        Tool::RunCommand { command } => {
            ensure_not_read_only(config)?;
            confirm_or_abort(config, confirmer, &format!("run command: {command}")).await?;
            tools::run_command(cwd, command, COMMAND_TIMEOUT_SECONDS).await
        }
        Tool::Finish { summary } => Ok(format!("{ANSI_GREEN_CHECK} {summary}")),
    }
}

/// Bail if the agent is in read-only mode.
fn ensure_not_read_only(config: &AgentConfig) -> Result<()> {
    if config.is_read_only {
        bail!("this tool is disabled in read-only mode");
    }
    Ok(())
}

/// Ask the confirmer for approval, bailing if the user denies or confirm is off.
async fn confirm_or_abort<C: Confirmer>(
    config: &AgentConfig,
    confirmer: &mut C,
    prompt: &str,
) -> Result<()> {
    if !config.should_confirm {
        return Ok(());
    }
    if !confirmer.confirm(prompt).await? {
        bail!("aborted by user");
    }
    Ok(())
}

/// Terminal-clickable file path (OSC 8 hyperlink). Always emits an absolute path.
///
/// Control characters are stripped first so a malicious model cannot inject
/// ANSI escape sequences into the terminal through a crafted file name.
pub(super) fn clickable_path(cwd: &Path, path: &str) -> String {
    let joined = cwd.join(path);
    let full = std::path::absolute(&joined).unwrap_or(joined);
    let display = sanitize_terminal_text(&full.to_string_lossy());
    let encoded = display
        .replace('%', "%25")
        .replace(' ', "%20")
        .replace('#', "%23")
        .replace('?', "%3F");
    format!("\x1b]8;;file://{encoded}\x1b\\{display}\x1b]8;;\x1b\\")
}

/// Remove control characters that could be interpreted as terminal escape codes.
fn sanitize_terminal_text(text: &str) -> String {
    text.chars()
        .filter(|character| !character.is_control())
        .collect()
}

/// Print clickable OSC 8 links for each file changed during the agent session.
pub(super) fn print_changed_files<R: Reporter>(cwd: &Path, files: &[String], reporter: &mut R) {
    if files.is_empty() {
        return;
    }
    reporter.line(format!("\n{ANSI_YELLOW_FILES_HEADER}"));
    for file in files {
        reporter.line(format!("   {}", clickable_path(cwd, file)));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::tool::Tool;
    use crate::agent::{AgentConfig, StdioConfirmer};
    use std::path::PathBuf;

    #[test]
    fn clickable_path_encodes_special_characters() {
        let path = std::path::Path::new("/tmp/a b#c");
        let link = clickable_path(path, "x");
        assert!(link.contains("a%20b%23c"));
        assert!(link.contains("/tmp/a b#c"));
    }

    #[test]
    fn clickable_path_strips_terminal_escape_injection() {
        let path = std::path::Path::new("/tmp");
        let link = clickable_path(path, "evil\x1b]8;;http://evil\x1b\\file.txt");
        assert!(!link.contains("\x1b]8;;http://evil"));
        assert!(link.contains("file.txt"));
    }

    #[tokio::test]
    async fn execute_tool_reads_what_it_writes() {
        let root = crate::test_support::unique_temp_dir("openbatrangs-agent-test");
        let config = AgentConfig {
            cwd: root.clone(),
            max_steps: 1,
            is_read_only: false,
            should_confirm: false,
            show_thinking: true,
            max_ctx: 8_192,
        };
        let mut changed = Vec::new();
        let mut confirmer = StdioConfirmer;
        execute_tool(
            &config,
            &root,
            &Tool::WriteFile {
                path: "src/lib.rs".to_string(),
                content: "pub fn f() {}".to_string(),
            },
            &mut changed,
            &mut confirmer,
        )
        .await
        .unwrap();
        let output = execute_tool(
            &config,
            &root,
            &Tool::ReadFile {
                path: "src/lib.rs".to_string(),
                max_chars: 100,
            },
            &mut changed,
            &mut confirmer,
        )
        .await
        .unwrap();
        assert!(output.contains("pub fn f() {}"));
        std::fs::remove_dir_all(root).unwrap();
    }

    fn test_config(root: &std::path::Path, is_read_only: bool) -> AgentConfig {
        AgentConfig {
            cwd: root.to_path_buf(),
            max_steps: 1,
            is_read_only,
            should_confirm: false,
            show_thinking: true,
            max_ctx: 8_192,
        }
    }

    fn temp_root() -> PathBuf {
        crate::test_support::unique_temp_dir("openbatrangs-execute-test")
    }

    struct TestConfirmer {
        response: bool,
        prompts: Vec<String>,
    }

    impl Confirmer for TestConfirmer {
        async fn confirm(&mut self, prompt: &str) -> anyhow::Result<bool> {
            self.prompts.push(prompt.to_string());
            Ok(self.response)
        }
    }

    fn confirming_config(root: &std::path::Path) -> AgentConfig {
        AgentConfig {
            cwd: root.to_path_buf(),
            max_steps: 1,
            is_read_only: false,
            should_confirm: true,
            show_thinking: true,
            max_ctx: 8_192,
        }
    }

    #[tokio::test]
    async fn execute_tool_write_asks_confirmer_before_writing() {
        let root = temp_root();
        let config = confirming_config(&root);
        let mut changed = Vec::new();
        let mut confirmer = TestConfirmer {
            response: true,
            prompts: Vec::new(),
        };
        execute_tool(
            &config,
            &root,
            &Tool::WriteFile {
                path: "ok.txt".to_string(),
                content: "x".to_string(),
            },
            &mut changed,
            &mut confirmer,
        )
        .await
        .unwrap();
        assert_eq!(confirmer.prompts, vec!["write file 'ok.txt'?"]);
        assert!(root.join("ok.txt").exists());
        std::fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn execute_tool_write_aborts_when_confirmer_denies() {
        let root = temp_root();
        let config = confirming_config(&root);
        let mut changed = Vec::new();
        let mut confirmer = TestConfirmer {
            response: false,
            prompts: Vec::new(),
        };
        let result = execute_tool(
            &config,
            &root,
            &Tool::WriteFile {
                path: "nope.txt".to_string(),
                content: "x".to_string(),
            },
            &mut changed,
            &mut confirmer,
        )
        .await;
        assert!(result.is_err());
        let error = format!("{:#}", result.unwrap_err());
        assert!(error.contains("aborted by user"));
        assert!(!root.join("nope.txt").exists());
        std::fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn execute_tool_write_rejects_parent_traversal() {
        let root = temp_root();
        let config = test_config(&root, false);
        let mut changed = Vec::new();
        let mut confirmer = StdioConfirmer;
        let result = execute_tool(
            &config,
            &root,
            &Tool::WriteFile {
                path: "../evil.txt".to_string(),
                content: "x".to_string(),
            },
            &mut changed,
            &mut confirmer,
        )
        .await;
        assert!(result.is_err());
        assert!(!root.parent().unwrap().join("evil.txt").exists());
        std::fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn execute_tool_write_respects_read_only_mode() {
        let root = temp_root();
        let config = test_config(&root, true);
        let mut changed = Vec::new();
        let mut confirmer = StdioConfirmer;
        let result = execute_tool(
            &config,
            &root,
            &Tool::WriteFile {
                path: "a.txt".to_string(),
                content: "x".to_string(),
            },
            &mut changed,
            &mut confirmer,
        )
        .await;
        assert!(result.is_err());
        let error = format!("{:#}", result.unwrap_err());
        assert!(error.contains("read-only"));
        assert!(!root.join("a.txt").exists());
        std::fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn execute_tool_run_command_respects_read_only_mode() {
        let root = temp_root();
        let config = test_config(&root, true);
        let mut changed = Vec::new();
        let mut confirmer = StdioConfirmer;
        let result = execute_tool(
            &config,
            &root,
            &Tool::RunCommand {
                command: "touch nope.txt".to_string(),
            },
            &mut changed,
            &mut confirmer,
        )
        .await;
        assert!(result.is_err());
        let error = format!("{:#}", result.unwrap_err());
        assert!(error.contains("read-only"));
        assert!(!root.join("nope.txt").exists());
        std::fs::remove_dir_all(root).unwrap();
    }
}
