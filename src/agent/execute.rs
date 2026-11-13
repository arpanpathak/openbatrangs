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
use super::quality::placeholder_reason;
use super::reporter::Reporter;
use super::tool::Tool;
use crate::constants::agent::COMMAND_TIMEOUT_SECONDS;
use crate::constants::ansi::{ANSI_GREEN_CHECK, ANSI_YELLOW_FILES_HEADER};
use crate::constants::tools::{DEFAULT_LIST_DEPTH, MAX_TOOL_OUTPUT};
use crate::tools;
use anyhow::{Result, bail, ensure};
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

/// Run `command` after the read-only and confirmation checks.
async fn run_command<C: Confirmer>(
    config: &AgentConfig,
    cwd: &Path,
    command: &str,
    confirmer: &mut C,
) -> Result<String> {
    approve(config, confirmer, &format!("run command: {command}")).await?;

    tools::run_command(cwd, command, COMMAND_TIMEOUT_SECONDS).await
}

/// Add `path` to the changed files and return the tool result followed by a
/// clickable link to the file.
fn record_change(cwd: &Path, path: &str, result: &str, changed_files: &mut Vec<String>) -> String {
    changed_files.push(path.to_string());

    format!("{result}\n📎 {}", clickable_path(cwd, path))
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
    print_file_links(cwd, files, reporter);
}

/// Print one indented clickable link per file.
fn print_file_links<R: Reporter>(cwd: &Path, files: &[String], reporter: &mut R) {
    for file in files {
        reporter.line(format!("   {}", clickable_path(cwd, file)));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::fakes::{
        FixedConfirmer, RecordingReporter, Reply, serve_replies, test_config,
    };
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
        let link = clickable_path(Path::new("/tmp/a b#c?d%"), "x");

        assert!(link.contains("a%20b%23c%3Fd%25"));
        assert!(link.contains("/tmp/a b#c?d%"));
    }

    #[test]
    fn clickable_path_strips_terminal_escape_injection() {
        let link = clickable_path(Path::new("/tmp"), "evil\x1b]8;;http://evil\x1b\\file.txt");

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

    #[tokio::test]
    async fn finish_and_errors_become_text() -> Result<()> {
        let root = temp_root();
        let config = test_config(&root, 1);
        let mut changed = Vec::new();
        let mut confirmer = FixedConfirmer::new(true);

        let finish = Tool::Finish {
            summary: "all done".to_string(),
        };
        let finished =
            execute_tool_or_report_error(&config, &root, &finish, &mut changed, &mut confirmer)
                .await;
        assert_eq!(finished, format!("{ANSI_GREEN_CHECK} all done"));

        let missing = Tool::ReadFile {
            path: "missing.txt".to_string(),
            max_chars: 10,
        };
        let failed =
            execute_tool_or_report_error(&config, &root, &missing, &mut changed, &mut confirmer)
                .await;
        assert!(failed.starts_with("Tool error: "), "{failed}");

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
    fn changed_files_are_listed_only_when_there_are_some() {
        let mut reporter = RecordingReporter::default();

        print_changed_files(Path::new("/tmp"), &[], &mut reporter);
        assert!(reporter.output.is_empty());

        let files = ["a.rs".to_string(), "b.rs".to_string()];
        print_changed_files(Path::new("/tmp"), &files, &mut reporter);
        assert!(reporter.output.contains(ANSI_YELLOW_FILES_HEADER));
        assert!(reporter.output.contains("/tmp/a.rs"));
        assert!(reporter.output.contains("/tmp/b.rs"));
    }
}
