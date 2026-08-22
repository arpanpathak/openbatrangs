//! Shell command execution for the agent.

use super::text::truncate;
use crate::constants::tools::MIN_COMMAND_TIMEOUT_SECONDS;
use anyhow::{anyhow, Context, Result};
use std::path::{Path, PathBuf};
use std::time::Duration;

/// Run a shell command in the workspace and capture combined output.
///
/// The command is confined to the workspace for convenience: `$HOME` and common
/// tool caches are redirected under `root/.agent/` so builds and installs do not
/// pollute the real user home directory.
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
    create_sandbox_dirs(root);
    let mut process = tokio::process::Command::new("bash");
    process.arg("-lc").arg(command).current_dir(root);
    for (key, value) in agent_sandbox_env(root) {
        process.env(key, value);
    }
    let output = tokio::time::timeout(timeout, process.output())
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

/// Create every sandbox directory before a command runs.
///
/// Commands that `cd $HOME`, write to `$XDG_CONFIG_HOME`, or use caches expect
/// the target directories to exist. Creating them eagerly avoids spurious
/// failures for otherwise valid agent commands.
fn create_sandbox_dirs(root: &Path) {
    for (_, path) in agent_sandbox_env(root) {
        let _ = std::fs::create_dir_all(&path);
    }
}

/// Environment overrides that keep agent commands inside the workspace.
fn agent_sandbox_env(root: &Path) -> Vec<(&'static str, PathBuf)> {
    let agent_dir = root.join(".agent");
    vec![
        ("HOME", agent_dir.join("home")),
        ("XDG_CACHE_HOME", agent_dir.join("cache")),
        ("XDG_CONFIG_HOME", agent_dir.join("config")),
        ("XDG_DATA_HOME", agent_dir.join("data")),
        ("TMPDIR", agent_dir.join("tmp")),
        ("CARGO_HOME", agent_dir.join("cargo")),
        ("PIP_CACHE_DIR", agent_dir.join("cache/pip")),
        ("NPM_CONFIG_CACHE", agent_dir.join("cache/npm")),
        ("UV_CACHE_DIR", agent_dir.join("cache/uv")),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;

    fn temp_dir() -> PathBuf {
        crate::test_support::unique_temp_dir("openbatrangs-command-test")
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

    #[tokio::test]
    async fn runs_in_workspace_directory() {
        let root = temp_dir();
        let output = run_command(&root, "pwd", 5).await.unwrap();
        assert!(output.contains(&root.to_string_lossy().to_string()));
        fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn redirects_home_and_caches_into_workspace() {
        let root = temp_dir();
        let output = run_command(&root, "echo $HOME $XDG_CACHE_HOME $CARGO_HOME $TMPDIR", 5)
            .await
            .unwrap();
        let expected_home = root.join(".agent/home");
        let expected_cache = root.join(".agent/cache");
        let expected_cargo = root.join(".agent/cargo");
        let expected_tmp = root.join(".agent/tmp");
        assert!(output.contains(&expected_home.to_string_lossy().to_string()));
        assert!(output.contains(&expected_cache.to_string_lossy().to_string()));
        assert!(output.contains(&expected_cargo.to_string_lossy().to_string()));
        assert!(output.contains(&expected_tmp.to_string_lossy().to_string()));
        fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn cache_dirs_are_created_inside_workspace() {
        let root = temp_dir();
        let output = run_command(&root, "mkdir -p $HOME/.test && echo ok", 5)
            .await
            .unwrap();
        assert!(output.contains("ok"));
        assert!(root.join(".agent/home/.test").is_dir());
        fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn sandbox_dirs_exist_before_command_runs() {
        let root = temp_dir();
        run_command(&root, "true", 5).await.unwrap();
        for name in ["home", "cache", "config", "data", "tmp", "cargo"] {
            assert!(
                root.join(".agent").join(name).is_dir(),
                ".agent/{name} should exist before command execution"
            );
        }
        fs::remove_dir_all(root).unwrap();
    }
}
