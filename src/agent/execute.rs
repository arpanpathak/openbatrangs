//! Tool execution, confirmation, and terminal-friendly result formatting.

use super::reporter::Reporter;
use super::tool::Tool;
use super::AgentConfig;
use crate::constants::agent::COMMAND_TIMEOUT_SECONDS;
use crate::constants::ansi::{ANSI_GREEN_CHECK, ANSI_YELLOW_FILES_HEADER};
use crate::constants::tools::{DEFAULT_LIST_DEPTH, MAX_TOOL_OUTPUT};
use crate::tools;
use anyhow::{bail, Result};
use std::io::Write;
use std::path::Path;

pub(super) async fn execute_tool_or_report_error(
    config: &AgentConfig,
    cwd: &Path,
    tool: &Tool,
    changed_files: &mut Vec<String>,
) -> String {
    match execute_tool(config, cwd, tool, changed_files).await {
        Ok(text) => text,
        Err(error) => format!("Tool error: {error:#}"),
    }
}

pub(super) async fn execute_tool(
    config: &AgentConfig,
    cwd: &Path,
    tool: &Tool,
    changed_files: &mut Vec<String>,
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
            confirm_or_abort(config, &format!("write file '{path}'?"))?;
            let result = tools::write_file(cwd, path, content)?;
            changed_files.push(path.clone());
            Ok(format!("{result}\n📎 {}", clickable_path(cwd, path)))
        }
        Tool::RunCommand { command } => {
            ensure_not_read_only(config)?;
            confirm_or_abort(config, &format!("run command: {command}"))?;
            tools::run_command(cwd, command, COMMAND_TIMEOUT_SECONDS).await
        }
        Tool::Finish { summary } => Ok(format!("{ANSI_GREEN_CHECK} {summary}")),
    }
}

fn ensure_not_read_only(config: &AgentConfig) -> Result<()> {
    if config.is_read_only {
        bail!("this tool is disabled in read-only mode");
    }
    Ok(())
}

fn confirm_or_abort(config: &AgentConfig, prompt: &str) -> Result<()> {
    if !config.should_confirm {
        return Ok(());
    }
    print!("❓ {prompt} [y/N] ");
    std::io::stdout().flush().ok();
    let mut input = String::new();
    std::io::stdin().read_line(&mut input).ok();
    if !input.trim().eq_ignore_ascii_case("y") {
        bail!("aborted by user");
    }
    Ok(())
}

/// Terminal-clickable file path (OSC 8 hyperlink). Always emits an absolute path.
pub(super) fn clickable_path(cwd: &Path, path: &str) -> String {
    let joined = cwd.join(path);
    let full = std::path::absolute(&joined).unwrap_or(joined);
    let display = full.to_string_lossy();
    let encoded = display
        .replace('%', "%25")
        .replace(' ', "%20")
        .replace('#', "%23")
        .replace('?', "%3F");
    format!("\x1b]8;;file://{encoded}\x1b\\{display}\x1b]8;;\x1b\\")
}

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
    use crate::agent::AgentConfig;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn clickable_path_encodes_special_characters() {
        let path = std::path::Path::new("/tmp/a b#c");
        let link = clickable_path(path, "x");
        assert!(link.contains("a%20b%23c"));
        assert!(link.contains("/tmp/a b#c"));
    }

    #[tokio::test]
    async fn execute_tool_reads_what_it_writes() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!("openbatrangs-agent-test-{unique}"));
        std::fs::create_dir_all(&root).unwrap();
        let config = AgentConfig {
            cwd: root.clone(),
            max_steps: 1,
            is_read_only: false,
            should_confirm: false,
            show_thinking: true,
        };
        let mut changed = Vec::new();
        execute_tool(
            &config,
            &root,
            &Tool::WriteFile {
                path: "src/lib.rs".to_string(),
                content: "pub fn f() {}".to_string(),
            },
            &mut changed,
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
        )
        .await
        .unwrap();
        assert!(output.contains("pub fn f() {}"));
        std::fs::remove_dir_all(root).unwrap();
    }
}
