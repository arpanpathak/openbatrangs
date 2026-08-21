//! Shell command execution for the agent.

use super::text::truncate;
use crate::constants::tools::MIN_COMMAND_TIMEOUT_SECONDS;
use anyhow::{anyhow, Context, Result};
use std::path::Path;
use std::time::Duration;

/// Run a shell command in the workspace and capture combined output.
///
/// # Arguments
/// - `root`: working directory for the command.
/// - `command`: shell command line.
/// - `timeout_secs`: maximum allowed runtime.
///
/// # Returns
/// Captured stdout/stderr plus exit status, truncated to `MAX_TOOL_OUTPUT`.
pub(crate) async fn run_command(root: &Path, command: &str, timeout_secs: u64) -> Result<String> {
    let timeout = Duration::from_secs(timeout_secs.max(MIN_COMMAND_TIMEOUT_SECONDS));
    let output = tokio::time::timeout(
        timeout,
        tokio::process::Command::new("bash")
            .arg("-lc")
            .arg(command)
            .current_dir(root)
            .output(),
    )
    .await
    .map_err(|_| anyhow!("command timed out after {timeout_secs}s"))?
    .context("failed to spawn shell")?;

    let mut text = String::from_utf8_lossy(&output.stdout).to_string();
    if !output.stderr.is_empty() {
        text.push_str("\n[stderr]\n");
        text.push_str(&String::from_utf8_lossy(&output.stderr));
    }

    let exit_code = output
        .status
        .code()
        .map(|code| code.to_string())
        .unwrap_or_else(|| "signal".to_string());
    text.push_str(&format!("\n[exit code {exit_code}]\n"));
    Ok(truncate(text))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_dir() -> PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("openbatrangs-command-test-{unique}"));
        fs::create_dir_all(&dir).expect("create temp dir");
        dir
    }

    #[tokio::test]
    async fn captures_stdout_and_exit_code() {
        let root = temp_dir();
        let output = run_command(&root, "echo hello", 5).await.unwrap();
        assert!(output.contains("hello"));
        assert!(output.contains("[exit code 0]"));
        fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn captures_stderr() {
        let root = temp_dir();
        let output = run_command(&root, "echo oops >&2", 5).await.unwrap();
        assert!(output.contains("[stderr]"));
        assert!(output.contains("oops"));
        fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn reports_nonzero_exit_code() {
        let root = temp_dir();
        let output = run_command(&root, "exit 3", 5).await.unwrap();
        assert!(output.contains("[exit code 3]"));
        fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn reports_timeout() {
        let root = temp_dir();
        let result = run_command(&root, "sleep 5", 1).await;
        assert!(result.is_err());
        let error = format!("{:#}", result.unwrap_err());
        assert!(error.contains("timed out"));
        fs::remove_dir_all(root).unwrap();
    }
}
