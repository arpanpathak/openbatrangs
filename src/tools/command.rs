//! # Shell command execution for the agent
//!
//! The agent's `run_command` tool executes shell commands inside the workspace
//! and captures combined output. Commands run through `bash -lc` with a
//! timeout so a hung process cannot freeze the agent loop.
//!
//! The environment is sandboxed: `$HOME` and common tool caches are redirected
//! under `<workspace>/.agent/` so builds and installs do not pollute the real
//! user home directory.
//!
//! ## References
//!
//! - Shell injection / command injection: <https://owasp.org/www-community/attacks/Command_Injection>
//! - Tokio process + timeout: <https://docs.rs/tokio/latest/tokio/time/fn.timeout.html>
//! - Unix environment variables: <https://pubs.opengroup.org/onlinepubs/9699919799/basedefs/V1_chap08.html>

use super::text::truncate;
use crate::constants::tools::{EXIT_CODE_PREFIX, MIN_COMMAND_TIMEOUT_SECONDS};
use anyhow::{Context, Result, anyhow};
use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::process::Output;
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
    let mut shell = sandboxed_shell(root, command);

    let output = tokio::time::timeout(timeout, shell.output())
        .await
        .map_err(|_elapsed| anyhow!("command timed out after {timeout_secs}s"))?
        .context("failed to spawn shell")?;

    Ok(truncate(command_report(&output)))
}

/// A `bash -lc` process for `command`, run in `root` with the sandbox
/// environment.
fn sandboxed_shell(root: &Path, command: &str) -> tokio::process::Command {
    let mut shell = tokio::process::Command::new("bash");

    shell
        .arg("-lc")
        .arg(command)
        .current_dir(root)
        // A command that reads input, such as git asking for a username, gets
        // end-of-file at once. With the TUI's stdin, it would wait for keys
        // that the TUI keeps for itself, and the agent would freeze.
        .stdin(std::process::Stdio::null())
        // git fails with an error in place of asking for a username or password.
        .env("GIT_TERMINAL_PROMPT", "0")
        // If the timeout fires, the `Child` is dropped and must take the
        // shell (and its descendants) down with it instead of leaving an
        // orphaned command running in the background.
        .kill_on_drop(true)
        .envs(agent_sandbox_env(root))
        .envs(host_tool_env());

    shell
}

/// The text handed back to the model: stdout, then stderr under a
/// `[stderr]` heading when there is any, then the exit-status line.
fn command_report(output: &Output) -> String {
    let mut report = String::from_utf8_lossy(&output.stdout).to_string();
    report.push_str(&stderr_section(&output.stderr));

    let exit_code = output
        .status
        .code()
        .map(|code| code.to_string())
        .unwrap_or_else(|| "signal".to_string());
    report.push_str(&format!("\n{EXIT_CODE_PREFIX}{exit_code}]\n"));

    report
}

/// The `[stderr]` section of a command report; empty when the command wrote
/// nothing to stderr.
fn stderr_section(stderr: &[u8]) -> String {
    if stderr.is_empty() {
        return String::new();
    }

    format!("\n[stderr]\n{}", String::from_utf8_lossy(stderr))
}

/// The `PATH` and `RUSTUP_HOME` overrides that keep the user's installed
/// toolchains visible inside the sandbox.
///
/// `$HOME` points into the workspace, so a login shell would no longer find
/// `~/.cargo/bin`, and rustup would look for toolchains under the sandbox. The
/// model then tries to install Rust again (1.6 GB per workspace). This puts
/// the real `~/.cargo/bin` and `~/.local/bin` first on `PATH` and points
/// `RUSTUP_HOME` at the real `~/.rustup`, so installed tools just work while
/// caches and new installs still land in the sandbox. Without `$HOME` there is
/// nothing to add.
fn host_tool_env() -> Vec<(&'static str, OsString)> {
    std::env::var_os("HOME")
        .map(|home| tool_env_for_home(Path::new(&home)))
        .unwrap_or_default()
}

/// The `PATH` and `RUSTUP_HOME` overrides for the real home directory `home`.
fn tool_env_for_home(home: &Path) -> Vec<(&'static str, OsString)> {
    let path = host_path(home).map(|path| ("PATH", path));
    let rustup = rustup_home(home).map(|rustup| ("RUSTUP_HOME", rustup));

    path.into_iter().chain(rustup).collect()
}

/// The current `PATH` with `home`'s `.cargo/bin` and `.local/bin` in front,
/// for those of them that exist; `None` when the result cannot be joined.
fn host_path(home: &Path) -> Option<OsString> {
    let mut paths: Vec<PathBuf> = [".cargo/bin", ".local/bin"]
        .iter()
        .map(|dir| home.join(dir))
        .filter(|dir| dir.is_dir())
        .collect();

    let current = std::env::var_os("PATH");
    paths.extend(current.iter().flat_map(std::env::split_paths));

    std::env::join_paths(paths).ok()
}

/// `home/.rustup` when that directory exists.
fn rustup_home(home: &Path) -> Option<OsString> {
    let rustup = home.join(".rustup");

    rustup.is_dir().then(|| rustup.into_os_string())
}

/// Create every sandbox directory before a command runs.
///
/// Commands that `cd $HOME`, write to `$XDG_CONFIG_HOME`, or use caches expect
/// the target directories to exist. Creating them eagerly avoids spurious
/// failures for otherwise valid agent commands.
fn create_sandbox_dirs(root: &Path) {
    for (_variable, directory) in agent_sandbox_env(root) {
        // A directory that cannot be created fails only the commands that
        // use it, and their output shows the error to the model.
        std::fs::create_dir_all(&directory).ok();
    }
}

/// Environment overrides that keep agent commands inside the workspace.
///
/// The paths are absolute: with a relative workspace such as `.`, a command
/// that changes directory (`cd dsa && cargo test`) would otherwise point
/// `TMPDIR` at a folder that does not exist, and rustdoc fails with
/// "failed to create temporary directory".
fn agent_sandbox_env(root: &Path) -> Vec<(&'static str, PathBuf)> {
    let agent_dir = std::path::absolute(root)
        .unwrap_or_else(|_| root.to_path_buf())
        .join(".agent");

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

    /// A new empty directory to use as the workspace of one test.
    fn temp_dir() -> PathBuf {
        crate::test_support::unique_temp_dir("openbatrangs-command-test")
    }

    #[tokio::test]
    async fn captures_stdout_and_exit_code() -> Result<()> {
        let root = temp_dir();

        let output = run_command(&root, "echo hello", 5).await?;

        assert!(output.contains("hello"));
        assert!(output.contains("[exit code 0]"));
        assert!(!output.contains("[stderr]"));

        fs::remove_dir_all(root)?;
        Ok(())
    }

    #[tokio::test]
    async fn captures_stderr() -> Result<()> {
        let root = temp_dir();

        let output = run_command(&root, "echo oops >&2", 5).await?;

        assert!(output.contains("[stderr]"));
        assert!(output.contains("oops"));

        fs::remove_dir_all(root)?;
        Ok(())
    }

    #[tokio::test]
    async fn input_reads_get_end_of_file() -> Result<()> {
        let root = temp_dir();

        let output = run_command(&root, "read -r line; echo read-exit:$?", 5).await?;

        assert!(output.contains("read-exit:1"));

        fs::remove_dir_all(root)?;
        Ok(())
    }

    #[tokio::test]
    async fn git_never_prompts() -> Result<()> {
        let root = temp_dir();

        let output = run_command(&root, "echo prompt:$GIT_TERMINAL_PROMPT", 5).await?;

        assert!(output.contains("prompt:0"));

        fs::remove_dir_all(root)?;
        Ok(())
    }

    #[tokio::test]
    async fn reports_nonzero_exit_code() -> Result<()> {
        let root = temp_dir();

        let output = run_command(&root, "exit 3", 5).await?;

        assert!(output.contains("[exit code 3]"));

        fs::remove_dir_all(root)?;
        Ok(())
    }

    #[tokio::test]
    async fn reports_a_shell_killed_by_a_signal() -> Result<()> {
        let root = temp_dir();

        let output = run_command(&root, "kill -KILL $$", 5).await?;

        assert!(output.contains("[exit code signal]"), "{output}");

        fs::remove_dir_all(root)?;
        Ok(())
    }

    #[tokio::test]
    async fn reports_timeout() -> Result<()> {
        let root = temp_dir();

        let error = run_command(&root, "sleep 5", 1)
            .await
            .err()
            .context("a command longer than its timeout should fail")?;

        assert!(format!("{error:#}").contains("timed out"));

        fs::remove_dir_all(root)?;
        Ok(())
    }

    #[tokio::test]
    async fn timeout_kills_child_process() -> Result<()> {
        let root = temp_dir();

        // If the timed-out shell is not killed, it will eventually `touch done`
        // after the sleep finishes, proving the child leaked.
        let finished = run_command(&root, "sleep 2 && touch done", 1).await;
        tokio::time::sleep(Duration::from_millis(2_500)).await;

        assert!(finished.is_err());
        assert!(
            !root.join("done").exists(),
            "timed-out command must not keep running after the timeout"
        );

        fs::remove_dir_all(root)?;
        Ok(())
    }

    #[tokio::test]
    async fn runs_in_workspace_directory() -> Result<()> {
        let root = temp_dir();

        let output = run_command(&root, "pwd", 5).await?;

        assert!(output.contains(root.to_string_lossy().as_ref()));

        fs::remove_dir_all(root)?;
        Ok(())
    }

    #[tokio::test]
    async fn redirects_home_and_caches_into_workspace() -> Result<()> {
        let root = temp_dir();

        let output =
            run_command(&root, "echo $HOME $XDG_CACHE_HOME $CARGO_HOME $TMPDIR", 5).await?;

        let missing: Vec<PathBuf> = ["home", "cache", "cargo", "tmp"]
            .iter()
            .map(|directory| root.join(".agent").join(directory))
            .filter(|expected| !output.contains(expected.to_string_lossy().as_ref()))
            .collect();

        assert!(missing.is_empty(), "{missing:?} not in {output}");

        fs::remove_dir_all(root)?;
        Ok(())
    }

    #[tokio::test]
    async fn the_host_rustup_home_reaches_the_sandbox() -> Result<()> {
        let root = temp_dir();
        let rustup = host_tool_env()
            .into_iter()
            .find(|(variable, _value)| *variable == "RUSTUP_HOME")
            .map(|(_variable, value)| value.to_string_lossy().into_owned())
            .unwrap_or_default();

        let output = run_command(&root, "echo \"rustup=$RUSTUP_HOME\"", 5).await?;

        assert!(output.contains(&format!("rustup={rustup}\n")), "{output}");

        fs::remove_dir_all(root)?;
        Ok(())
    }

    #[test]
    fn installed_tools_come_first_on_the_path() -> Result<()> {
        let home = temp_dir();
        fs::create_dir_all(home.join(".cargo/bin"))?;
        fs::create_dir_all(home.join(".local/bin"))?;
        fs::create_dir_all(home.join(".rustup"))?;

        let env = tool_env_for_home(&home);
        let path = env.iter().find(|(variable, _value)| *variable == "PATH");
        let first_dirs: Vec<PathBuf> = path
            .map(|(_variable, value)| std::env::split_paths(value).take(2).collect())
            .unwrap_or_default();

        assert_eq!(
            first_dirs,
            [home.join(".cargo/bin"), home.join(".local/bin")]
        );
        assert!(env.contains(&("RUSTUP_HOME", home.join(".rustup").into_os_string())));

        fs::remove_dir_all(home)?;
        Ok(())
    }

    #[test]
    fn a_home_without_toolchains_adds_no_rustup_home() -> Result<()> {
        let home = temp_dir();

        let env = tool_env_for_home(&home);

        assert!(
            env.iter()
                .all(|(variable, _value)| *variable != "RUSTUP_HOME")
        );

        fs::remove_dir_all(home)?;
        Ok(())
    }

    #[tokio::test]
    async fn cache_dirs_are_created_inside_workspace() -> Result<()> {
        let root = temp_dir();

        let output = run_command(&root, "mkdir -p $HOME/.test && echo ok", 5).await?;

        assert!(output.contains("ok"));
        assert!(root.join(".agent/home/.test").is_dir());

        fs::remove_dir_all(root)?;
        Ok(())
    }

    #[tokio::test]
    async fn sandbox_dirs_exist_before_command_runs() -> Result<()> {
        let root = temp_dir();

        run_command(&root, "true", 5).await?;

        let missing: Vec<PathBuf> = ["home", "cache", "config", "data", "tmp", "cargo"]
            .iter()
            .map(|name| root.join(".agent").join(name))
            .filter(|directory| !directory.is_dir())
            .collect();

        assert!(
            missing.is_empty(),
            "{missing:?} should exist before the command runs"
        );

        fs::remove_dir_all(root)?;
        Ok(())
    }

    #[test]
    fn an_empty_root_keeps_the_sandbox_relative() {
        let env = agent_sandbox_env(Path::new(""));

        assert!(env.contains(&("HOME", PathBuf::from(".agent/home"))));
    }

    #[test]
    fn sandbox_paths_are_absolute_for_a_relative_root() {
        let env = agent_sandbox_env(&PathBuf::from("."));

        assert!(env.iter().all(|(_variable, path)| path.is_absolute()));
    }
}
