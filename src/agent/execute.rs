//! # Tool execution, confirmation, and terminal-friendly result formatting
//!
//! Once a [`Tool`] has been parsed and typed, this module is responsible for
//! actually doing the work: listing files, reading/writing files, grepping,
//! running shell commands, and formatting results for the terminal.
//!
//! Mutating tools are protected by two guards:
//!
//! - [`AgentConfig::is_read_only`] disables writes/commands entirely.
//! - [`Confirmer`] shows a yes/no prompt before a write/command when confirm
//!   mode is on.
//!
//! Results are also made terminal-friendly: written files become OSC 8
//! hyperlinks and control characters are stripped so a model cannot inject
//! escape sequences through a crafted file name.
//!
//! ## References
//!
//! - OSC 8 hyperlinks: <https://gist.github.com/egmontkob/eb114294efbcd5adb1944c9f3cb5feda>
//! - Human-in-the-loop safety: <https://en.wikipedia.org/wiki/Human-in-the-loop>

use super::AgentConfig;
use super::confirm::Confirmer;
use super::quality::command_succeeded;
use super::quality::placeholder_reason;
use super::summary::result_summary;
use super::tool::Tool;
use crate::constants::agent::COMMAND_TIMEOUT_SECONDS;
use crate::constants::ansi::ANSI_GREEN_CHECK;
use crate::constants::tools::{
    DEFAULT_LIST_DEPTH, DESTRUCTIVE_COMMAND_PATTERNS, MAX_TOOL_OUTPUT, UNSANDBOXED_COMMAND_WARNING,
};
use crate::tools;
use crate::tools::sandbox::{Sandbox, available_sandbox};
use anyhow::{Result, bail, ensure};
use std::path::{Path, PathBuf};

/// The outcome of one tool call, as the loops report and record it.
pub(super) struct ToolReport {
    /// Whether the tool did its job; a command must also exit with code 0.
    pub(super) success: bool,
    /// A few words on the result, such as `4 passed` or `exit 101`.
    pub(super) summary: String,
    /// The full result text handed back to the model.
    pub(super) output: String,
}

/// Execute a tool call and describe how it went.
///
/// An error becomes the output text `Tool error: ...`, so the model can read
/// it and retry.
pub(super) async fn run_tool_call<C: Confirmer>(
    config: &AgentConfig,
    cwd: &Path,
    tool: &Tool,
    changed_files: &mut Vec<String>,
    confirmer: &mut C,
) -> ToolReport {
    let result = execute_tool(config, cwd, tool, changed_files, confirmer).await;

    match result {
        Ok(output) => successful_call_report(tool, output),
        Err(error) => failed_call_report(&error),
    }
}

/// The report for a tool that returned output. A command only succeeds when
/// it exits with code 0.
fn successful_call_report(tool: &Tool, output: String) -> ToolReport {
    let is_failed_command = matches!(tool, Tool::RunCommand { .. }) && !command_succeeded(&output);

    ToolReport {
        success: !is_failed_command,
        summary: result_summary(tool, &output),
        output,
    }
}

/// The report for a tool that failed: the first line of the error is the summary.
fn failed_call_report(error: &anyhow::Error) -> ToolReport {
    let output = format!("Tool error: {error:#}");
    let summary = output.lines().next().unwrap_or_default().to_string();

    ToolReport {
        success: false,
        summary,
        output,
    }
}

/// Execute a typed tool call and return its text result.
///
/// Mutating tools (write, edit, run command) are guarded by read-only mode
/// and user confirmation.
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
            tools::read_file(cwd, path, (*max_chars).min(MAX_TOOL_OUTPUT))
        }
        Tool::GrepFiles {
            pattern,
            path,
            max_results,
        } => tools::grep_files(cwd, pattern, path, *max_results),
        Tool::WebSearch { query } => {
            tools::web_search(&config.searxng_url, config.search_key.as_deref(), query).await
        }
        Tool::WebRead { url } => tools::web_read(url).await,
        Tool::WriteFile { path, content } => {
            write_file(config, cwd, path, content, changed_files, confirmer).await
        }
        Tool::EditFile {
            path,
            old_string,
            new_string,
            replace_all,
        } => {
            edit_file(
                config,
                cwd,
                Edit {
                    path,
                    old_string,
                    new_string,
                    replace_all: *replace_all,
                },
                changed_files,
                confirmer,
            )
            .await
        }
        Tool::RunCommand { command } => run_command(config, cwd, command, confirmer).await,
        Tool::Finish { summary } => Ok(format!("{ANSI_GREEN_CHECK} {summary}")),
        Tool::AskUser { question } => Ok(question.clone()),
    }
}

/// The arguments of one `edit_file` call.
struct Edit<'a> {
    /// File path relative to the workspace.
    path: &'a str,
    /// Exact text to find.
    old_string: &'a str,
    /// Replacement text.
    new_string: &'a str,
    /// Replace every occurrence instead of requiring a unique match.
    replace_all: bool,
}

/// Write `content` to `path` after the read-only, placeholder, and
/// confirmation checks, and record the path as changed.
async fn write_file<C: Confirmer>(
    config: &AgentConfig,
    cwd: &Path,
    path: &str,
    content: &str,
    changed_files: &mut Vec<String>,
    confirmer: &mut C,
) -> Result<String> {
    ensure_not_read_only(config)?;
    ensure_not_placeholder(path, content)?;
    confirm_or_abort(config, confirmer, &format!("write file '{path}'?")).await?;

    let result = tools::write_file(cwd, path, content)?;
    Ok(record_change(cwd, path, &result, changed_files))
}

/// Apply `edit` after the read-only and confirmation checks, and record the
/// path as changed.
async fn edit_file<C: Confirmer>(
    config: &AgentConfig,
    cwd: &Path,
    edit: Edit<'_>,
    changed_files: &mut Vec<String>,
    confirmer: &mut C,
) -> Result<String> {
    let prompt = format!("edit file '{}'?", edit.path);
    approve(config, confirmer, &prompt).await?;

    let result = tools::edit_file(
        cwd,
        edit.path,
        edit.old_string,
        edit.new_string,
        edit.replace_all,
    )?;
    Ok(record_change(cwd, edit.path, &result, changed_files))
}

/// Run `command` in the sandbox after the read-only and approval checks.
async fn run_command<C: Confirmer>(
    config: &AgentConfig,
    cwd: &Path,
    command: &str,
    confirmer: &mut C,
) -> Result<String> {
    ensure_not_read_only(config)?;

    let sandbox = command_sandbox(config);
    approve_command(config, confirmer, command, sandbox).await?;

    tools::run_command(cwd, command, COMMAND_TIMEOUT_SECONDS, sandbox).await
}

/// The sandbox commands run in: the one this machine supports, or none
/// when `--no-sandbox` turned it off.
fn command_sandbox(config: &AgentConfig) -> Sandbox {
    if !config.use_sandbox {
        return Sandbox::Off;
    }

    available_sandbox()
}

/// Ask the user about `command` when [`command_needs_approval`] says so.
async fn approve_command<C: Confirmer>(
    config: &AgentConfig,
    confirmer: &mut C,
    command: &str,
    sandbox: Sandbox,
) -> Result<()> {
    if !command_needs_approval(config, command, sandbox) {
        return Ok(());
    }

    ask_user(confirmer, &command_prompt(config, command, sandbox)).await
}

/// Whether the user must approve `command` before it runs: in confirm mode,
/// when the sandbox is wanted but no sandbox works here, and for every
/// command that deletes or overwrites files in bulk.
fn command_needs_approval(config: &AgentConfig, command: &str, sandbox: Sandbox) -> bool {
    let reasons = [
        config.should_confirm,
        is_missing_sandbox(config, sandbox),
        is_destructive(command),
    ];

    reasons.contains(&true)
}

/// Whether the user wants commands sandboxed but this machine has no
/// working sandbox.
fn is_missing_sandbox(config: &AgentConfig, sandbox: Sandbox) -> bool {
    config.use_sandbox && !sandbox.is_active()
}

/// The approval question for `command`, which says when the command would
/// run without the sandbox it was meant to have.
fn command_prompt(config: &AgentConfig, command: &str, sandbox: Sandbox) -> String {
    let warning = if is_missing_sandbox(config, sandbox) {
        UNSANDBOXED_COMMAND_WARNING
    } else {
        ""
    };

    format!("run command{warning}: {command}")
}

/// Whether `command` deletes or overwrites files in bulk, such as `rm -rf`
/// or `git reset --hard`.
///
/// Shell separators become spaces first, so `cd src;rm -rf x` and
/// `(git clean -fd)` match as well. A match only means the user is asked;
/// the operating-system sandbox is what keeps commands inside the workspace.
fn is_destructive(command: &str) -> bool {
    let spaced = command.replace([';', '&', '|', '(', ')', '\n'], " ");
    let words: Vec<&str> = spaced.split_whitespace().collect();
    let normalized = format!(" {} ", words.join(" "));

    DESTRUCTIVE_COMMAND_PATTERNS
        .iter()
        .any(|pattern| normalized.contains(pattern))
}

/// Add `path` to the changed files and return the tool result followed by a
/// clickable link to the file.
fn record_change(cwd: &Path, path: &str, result: &str, changed_files: &mut Vec<String>) -> String {
    changed_files.push(path.to_string());

    format!("{result}\n📎 {}", clickable_link(&cwd.join(path)))
}

/// Allow a mutating tool: fail in read-only mode or when the user declines.
async fn approve<C: Confirmer>(
    config: &AgentConfig,
    confirmer: &mut C,
    prompt: &str,
) -> Result<()> {
    ensure_not_read_only(config)?;

    confirm_or_abort(config, confirmer, prompt).await
}

/// Bail if the agent is in read-only mode.
fn ensure_not_read_only(config: &AgentConfig) -> Result<()> {
    ensure!(
        !config.is_read_only,
        "this tool is disabled in read-only mode"
    );

    Ok(())
}

/// Bail if `content` looks like a placeholder instead of real file content.
fn ensure_not_placeholder(path: &str, content: &str) -> Result<()> {
    let Some(reason) = placeholder_reason(content) else {
        return Ok(());
    };

    bail!(
        "quality gate: refusing to write '{path}' because the content looks like a placeholder ({reason}). Write the complete implementation."
    )
}

/// Ask the confirmer for approval when confirm mode is on.
async fn confirm_or_abort<C: Confirmer>(
    config: &AgentConfig,
    confirmer: &mut C,
    prompt: &str,
) -> Result<()> {
    if !config.should_confirm {
        return Ok(());
    }

    ask_user(confirmer, prompt).await
}

/// Bail unless the user approves `prompt`.
async fn ask_user<C: Confirmer>(confirmer: &mut C, prompt: &str) -> Result<()> {
    let is_approved = confirmer.confirm(prompt).await?;
    ensure!(is_approved, "aborted by user");

    Ok(())
}

/// `files` as absolute paths under `cwd`.
pub(super) fn absolute_paths(cwd: &Path, files: &[String]) -> Vec<PathBuf> {
    files
        .iter()
        .map(|file| cwd.join(file))
        .map(|path| std::path::absolute(&path).unwrap_or(path))
        .collect()
}

/// Terminal-clickable file path (OSC 8 hyperlink). Always emits an absolute path.
///
/// Control characters are stripped first so a malicious model cannot inject
/// ANSI escape sequences into the terminal through a crafted file name.
pub(super) fn clickable_link(path: &Path) -> String {
    let given = path.to_path_buf();
    let full = std::path::absolute(path).unwrap_or(given);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::fakes::{FixedConfirmer, Reply, serve_replies, test_config};
    use serde_json::json;
    use std::path::PathBuf;

    /// A new empty workspace directory.
    fn temp_root() -> PathBuf {
        crate::test_support::unique_temp_dir("openbatrangs-execute-test")
    }

    /// Run `tool` in the workspace of `config`.
    async fn run(
        config: &AgentConfig,
        tool: Tool,
        changed_files: &mut Vec<String>,
        confirmer: &mut FixedConfirmer,
    ) -> Result<String> {
        execute_tool(config, &config.cwd, &tool, changed_files, confirmer).await
    }

    #[test]
    fn clickable_path_encodes_special_characters() {
        let link = clickable_link(&Path::new("/tmp/a b#c?d%").join("x"));

        assert!(link.contains("a%20b%23c%3Fd%25"));
        assert!(link.contains("/tmp/a b#c?d%"));
    }

    #[test]
    fn clickable_path_strips_terminal_escape_injection() {
        let link = clickable_link(&Path::new("/tmp").join("evil\x1b]8;;http://evil\x1b\\file.txt"));

        assert!(!link.contains("\x1b]8;;http://evil"));
        assert!(link.contains("file.txt"));
    }

    #[tokio::test]
    async fn execute_tool_reads_what_it_writes() -> Result<()> {
        let root = temp_root();
        let config = test_config(&root, 1);
        let mut changed = Vec::new();
        let mut confirmer = FixedConfirmer::new(true);

        let write = Tool::WriteFile {
            path: "src/lib.rs".to_string(),
            content: "pub fn f() {}".to_string(),
        };
        let written = run(&config, write, &mut changed, &mut confirmer).await?;
        assert!(written.contains("📎"));

        let read = Tool::ReadFile {
            path: "src/lib.rs".to_string(),
            max_chars: 100,
        };
        let output = run(&config, read, &mut changed, &mut confirmer).await?;
        assert!(output.contains("pub fn f() {}"));
        assert_eq!(changed, vec!["src/lib.rs"]);

        std::fs::remove_dir_all(root)?;
        Ok(())
    }

    #[tokio::test]
    async fn execute_tool_lists_greps_edits_and_runs() -> Result<()> {
        let root = temp_root();
        std::fs::write(root.join("notes.txt"), "alpha\nbeta\n")?;
        let config = test_config(&root, 1);
        let mut changed = Vec::new();
        let mut confirmer = FixedConfirmer::new(true);

        let list = Tool::ListFiles {
            path: ".".to_string(),
        };
        assert!(
            run(&config, list, &mut changed, &mut confirmer)
                .await?
                .contains("notes.txt")
        );

        let grep = Tool::GrepFiles {
            pattern: "be.a".to_string(),
            path: ".".to_string(),
            max_results: 5,
        };
        assert!(
            run(&config, grep, &mut changed, &mut confirmer)
                .await?
                .contains("beta")
        );

        let edit = Tool::EditFile {
            path: "notes.txt".to_string(),
            old_string: "alpha".to_string(),
            new_string: "gamma".to_string(),
            replace_all: false,
        };
        run(&config, edit, &mut changed, &mut confirmer).await?;
        assert_eq!(
            std::fs::read_to_string(root.join("notes.txt"))?,
            "gamma\nbeta\n"
        );

        let command = Tool::RunCommand {
            command: "echo ran".to_string(),
        };
        assert!(
            run(&config, command, &mut changed, &mut confirmer)
                .await?
                .contains("ran")
        );

        std::fs::remove_dir_all(root)?;
        Ok(())
    }

    #[tokio::test]
    async fn execute_tool_searches_and_refuses_local_pages() -> Result<()> {
        let results = json!({ "results": [
            { "title": "Tokio", "url": "https://docs.rs/tokio", "content": "async runtime" }
        ] });
        let root = temp_root();
        let mut config = test_config(&root, 1);
        config.searxng_url = serve_replies(vec![Reply::json(&results)]).await?;
        let mut changed = Vec::new();
        let mut confirmer = FixedConfirmer::new(true);

        let search = Tool::WebSearch {
            query: "tokio".to_string(),
        };
        let found = run(&config, search, &mut changed, &mut confirmer).await?;
        assert!(found.contains("https://docs.rs/tokio"));

        let read = Tool::WebRead {
            url: "http://127.0.0.1:9/page".to_string(),
        };
        assert!(
            run(&config, read, &mut changed, &mut confirmer)
                .await
                .is_err()
        );

        std::fs::remove_dir_all(root)?;
        Ok(())
    }

    /// The report of running `tool` in the workspace of `config`.
    async fn report_for(config: &AgentConfig, tool: Tool) -> ToolReport {
        let mut changed = Vec::new();
        let mut confirmer = FixedConfirmer::new(true);

        run_tool_call(config, &config.cwd, &tool, &mut changed, &mut confirmer).await
    }

    #[tokio::test]
    async fn tool_reports_carry_success_summary_and_output() -> Result<()> {
        let root = temp_root();
        let config = test_config(&root, 1);
        let report = |tool: Tool| report_for(&config, tool);

        let finished = report(Tool::Finish {
            summary: "all done".to_string(),
        })
        .await;
        assert_eq!(finished.output, format!("{ANSI_GREEN_CHECK} all done"));

        let asked = report(Tool::AskUser {
            question: "Which file?".to_string(),
        })
        .await;
        assert_eq!(asked.output, "Which file?");

        let missing = report(Tool::ReadFile {
            path: "missing.txt".to_string(),
            max_chars: 10,
        })
        .await;
        assert!(!missing.success);
        assert!(
            missing.output.starts_with("Tool error: "),
            "{}",
            missing.output
        );
        assert!(missing.summary.starts_with("Tool error: "));

        let failing = report(Tool::RunCommand {
            command: "exit 3".to_string(),
        })
        .await;
        assert!(!failing.success);
        assert_eq!(failing.summary, "exit 3");

        let passing = report(Tool::RunCommand {
            command: "true".to_string(),
        })
        .await;
        assert!(passing.success);
        assert_eq!(passing.summary, "exit 0");

        std::fs::remove_dir_all(root)?;
        Ok(())
    }

    #[tokio::test]
    async fn execute_tool_rejects_placeholder_write() -> Result<()> {
        let root = temp_root();
        let config = test_config(&root, 1);
        let mut changed = Vec::new();
        let mut confirmer = FixedConfirmer::new(true);

        let write = Tool::WriteFile {
            path: "README.md".to_string(),
            content: "# Design\n\nTODO: implement the full answer.".to_string(),
        };
        let result = run(&config, write, &mut changed, &mut confirmer).await;

        assert!(format!("{result:?}").contains("quality gate"));
        assert!(!root.join("README.md").exists());
        assert!(changed.is_empty());

        std::fs::remove_dir_all(root)?;
        Ok(())
    }

    #[tokio::test]
    async fn execute_tool_write_asks_confirmer_before_writing() -> Result<()> {
        let root = temp_root();
        let mut config = test_config(&root, 1);
        config.should_confirm = true;
        let mut changed = Vec::new();
        let mut confirmer = FixedConfirmer::new(true);

        let write = Tool::WriteFile {
            path: "ok.txt".to_string(),
            content: "x".to_string(),
        };
        run(&config, write, &mut changed, &mut confirmer).await?;

        assert_eq!(confirmer.prompts, vec!["write file 'ok.txt'?"]);
        assert!(root.join("ok.txt").exists());

        std::fs::remove_dir_all(root)?;
        Ok(())
    }

    #[tokio::test]
    async fn execute_tool_aborts_when_confirmer_denies() -> Result<()> {
        let root = temp_root();
        let mut config = test_config(&root, 1);
        config.should_confirm = true;
        let mut changed = Vec::new();
        let mut confirmer = FixedConfirmer::new(false);

        let write = Tool::WriteFile {
            path: "nope.txt".to_string(),
            content: "x".to_string(),
        };
        let written = run(&config, write, &mut changed, &mut confirmer).await;

        let command = Tool::RunCommand {
            command: "touch ran.txt".to_string(),
        };
        let ran = run(&config, command, &mut changed, &mut confirmer).await;

        assert!(format!("{written:?}").contains("aborted by user"));
        assert!(format!("{ran:?}").contains("aborted by user"));
        assert_eq!(
            confirmer.prompts,
            vec!["write file 'nope.txt'?", "run command: touch ran.txt"]
        );
        assert!(!root.join("nope.txt").exists());
        assert!(!root.join("ran.txt").exists());

        std::fs::remove_dir_all(root)?;
        Ok(())
    }

    #[tokio::test]
    async fn execute_tool_write_rejects_parent_traversal() -> Result<()> {
        let root = temp_root().join("inner");
        std::fs::create_dir_all(&root)?;
        let config = test_config(&root, 1);
        let mut changed = Vec::new();
        let mut confirmer = FixedConfirmer::new(true);

        let write = Tool::WriteFile {
            path: "../evil.txt".to_string(),
            content: "x".to_string(),
        };
        let result = run(&config, write, &mut changed, &mut confirmer).await;

        assert!(result.is_err());
        assert!(!root.join("../evil.txt").exists());

        std::fs::remove_dir_all(root.join(".."))?;
        Ok(())
    }

    #[tokio::test]
    async fn read_only_mode_blocks_every_mutating_tool() -> Result<()> {
        let root = temp_root();
        let mut config = test_config(&root, 1);
        config.is_read_only = true;
        let mut changed = Vec::new();
        let mut confirmer = FixedConfirmer::new(true);

        let write = Tool::WriteFile {
            path: "a.txt".to_string(),
            content: "x".to_string(),
        };
        let edit = Tool::EditFile {
            path: "a.txt".to_string(),
            old_string: "x".to_string(),
            new_string: "y".to_string(),
            replace_all: true,
        };
        let command = Tool::RunCommand {
            command: "touch nope.txt".to_string(),
        };

        let written = run(&config, write, &mut changed, &mut confirmer).await;
        let edited = run(&config, edit, &mut changed, &mut confirmer).await;
        let ran = run(&config, command, &mut changed, &mut confirmer).await;

        let errors = format!("{written:?} {edited:?} {ran:?}");
        assert_eq!(errors.matches("read-only").count(), 3);
        assert!(!root.join("a.txt").exists());
        assert!(!root.join("nope.txt").exists());

        std::fs::remove_dir_all(root)?;
        Ok(())
    }

    #[test]
    fn bulk_deletes_and_resets_are_destructive() {
        let destructive = [
            "rm -rf target",
            "cd src;rm -rf ../x",
            "cargo build && rm old.txt",
            "(git clean -fd)",
            "git reset --hard HEAD~1",
            "git push --force-with-lease",
            "find . -name '*.o' -delete",
            "sudo apt-get install zstd",
        ];

        for command in destructive {
            assert!(is_destructive(command), "{command}");
        }
    }

    #[test]
    fn ordinary_commands_are_not_destructive() {
        let ordinary = [
            "cargo test",
            "git add . && git commit -m 'add parser'",
            "echo address",
            "ls -la",
            "git init",
        ];

        for command in ordinary {
            assert!(!is_destructive(command), "{command}");
        }
    }

    #[test]
    fn commands_need_approval_in_confirm_mode_without_a_sandbox_or_when_destructive() {
        let mut config = test_config(Path::new("/work"), 1);

        assert!(!command_needs_approval(&config, "ls", Sandbox::Off));
        assert!(command_needs_approval(
            &config,
            "rm -rf x",
            Sandbox::Bubblewrap
        ));

        config.use_sandbox = true;
        assert!(!command_needs_approval(&config, "ls", Sandbox::Bubblewrap));
        assert!(command_needs_approval(&config, "ls", Sandbox::Off));

        config.use_sandbox = false;
        config.should_confirm = true;
        assert!(command_needs_approval(&config, "ls", Sandbox::Off));
    }

    #[test]
    fn the_prompt_says_when_a_command_runs_without_its_sandbox() {
        let mut config = test_config(Path::new("/work"), 1);
        config.use_sandbox = true;

        assert_eq!(
            command_prompt(&config, "ls", Sandbox::Off),
            "run command without a sandbox: ls"
        );
        assert_eq!(
            command_prompt(&config, "ls", Sandbox::Seatbelt),
            "run command: ls"
        );
    }

    #[test]
    fn no_sandbox_turns_the_command_sandbox_off() {
        let mut config = test_config(Path::new("/work"), 1);

        assert_eq!(command_sandbox(&config), Sandbox::Off);

        config.use_sandbox = true;
        assert_eq!(command_sandbox(&config), available_sandbox());
    }

    #[tokio::test]
    async fn a_destructive_command_asks_even_without_confirm_mode() -> Result<()> {
        let root = temp_root();
        std::fs::write(root.join("keep.txt"), "x")?;
        let config = test_config(&root, 1);
        let mut changed = Vec::new();
        let mut confirmer = FixedConfirmer::new(false);

        let command = Tool::RunCommand {
            command: "rm -rf keep.txt".to_string(),
        };
        let ran = run(&config, command, &mut changed, &mut confirmer).await;

        assert!(format!("{ran:?}").contains("aborted by user"));
        assert_eq!(confirmer.prompts, vec!["run command: rm -rf keep.txt"]);
        assert!(root.join("keep.txt").exists());

        std::fs::remove_dir_all(root)?;
        Ok(())
    }
}
