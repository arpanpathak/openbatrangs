//! # Operating-system sandbox for shell commands
//!
//! `write_file` and `edit_file` refuse paths outside the workspace, but a shell
//! command can name any path. This module wraps each command in the
//! operating system's sandbox, so the command can read the whole disk and use
//! the network but can write only inside the workspace:
//!
//! - On Linux, [bubblewrap](https://github.com/containers/bubblewrap) mounts
//!   the root file system read-only, gives the command a private `/tmp`, and
//!   mounts only the workspace writable.
//! - On macOS, `sandbox-exec` applies a Seatbelt profile that denies every
//!   file write except in the workspace, the temporary folders and `/dev`.
//!
//! When neither tool works, [`Sandbox::Off`] runs the command directly, and
//! the caller asks the user before each command.

use crate::constants::tools::{BUBBLEWRAP_PROGRAM, SEATBELT_PROFILE, SEATBELT_PROGRAM};
use std::ffi::OsString;
use std::path::Path;
use std::process::{Command, Stdio};
use std::sync::OnceLock;

/// How shell commands are kept inside the workspace.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sandbox {
    /// Linux: bubblewrap with a read-only root and a writable workspace.
    Bubblewrap,
    /// macOS: `sandbox-exec` with a profile that allows writes only to the
    /// workspace and the temporary folders.
    Seatbelt,
    /// No sandbox: the command can write anywhere the user can.
    Off,
}

impl Sandbox {
    /// Whether this sandbox limits writes to the workspace.
    pub fn is_active(self) -> bool {
        self != Sandbox::Off
    }
}

/// The sandbox this machine supports, checked once and then remembered.
pub fn available_sandbox() -> Sandbox {
    static AVAILABLE: OnceLock<Sandbox> = OnceLock::new();

    *AVAILABLE.get_or_init(detect_sandbox)
}

/// Try this platform's sandbox tool, and use it only when a trial command
/// succeeds. bubblewrap is often installed but blocked, for example inside a
/// container or where AppArmor restricts user namespaces.
fn detect_sandbox() -> Sandbox {
    let candidate = sandbox_for_os(std::env::consts::OS);
    let (program, arguments) = trial_command(candidate);

    first_working(candidate, runs_successfully(program, &arguments))
}

/// The sandbox tool to try on the operating system named `os`, as
/// [`std::env::consts::OS`] names it.
fn sandbox_for_os(os: &str) -> Sandbox {
    match os {
        "linux" => Sandbox::Bubblewrap,
        "macos" => Sandbox::Seatbelt,
        _other => Sandbox::Off,
    }
}

/// `candidate` when its trial command worked, else [`Sandbox::Off`].
fn first_working(candidate: Sandbox, trial_succeeded: bool) -> Sandbox {
    if !trial_succeeded {
        return Sandbox::Off;
    }

    candidate
}

/// A command that runs `true` inside `sandbox`, to check the tool works here.
fn trial_command(sandbox: Sandbox) -> (OsString, Vec<OsString>) {
    let root = std::env::temp_dir();

    wrapped_command(sandbox, &root, "true")
}

/// Whether `program` with `arguments` starts and exits with status 0.
fn runs_successfully(program: OsString, arguments: &[OsString]) -> bool {
    Command::new(program)
        .args(arguments)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|status| status.success())
}

/// The program and arguments that run `command` with `bash -lc` inside
/// `sandbox`, with `root` as the only writable folder.
pub fn wrapped_command(sandbox: Sandbox, root: &Path, command: &str) -> (OsString, Vec<OsString>) {
    let shell_arguments: Vec<OsString> = vec!["-lc".into(), command.into()];

    match sandbox {
        Sandbox::Bubblewrap => (
            BUBBLEWRAP_PROGRAM.into(),
            bubblewrap_arguments(root, bash_with(shell_arguments)),
        ),
        Sandbox::Seatbelt => (
            SEATBELT_PROGRAM.into(),
            seatbelt_arguments(root, bash_with(shell_arguments)),
        ),
        Sandbox::Off => ("bash".into(), shell_arguments),
    }
}

/// bubblewrap arguments for running `shell` with only `root` writable.
///
/// The root file system is mounted read-only and `/tmp` is a fresh, empty
/// tmpfs that disappears with the command. `/dev` stays the real one, so GPU
/// devices and `/dev/null` work. `--unshare-pid` with `--die-with-parent`
/// ends every process of the command when openbatrangs stops bubblewrap,
/// even though `--new-session` puts them in a session of their own.
/// `--new-session` keeps a command from typing into the user's terminal.
fn bubblewrap_arguments(root: &Path, shell: Vec<OsString>) -> Vec<OsString> {
    let workspace = absolute_path(root);
    let mut arguments: Vec<OsString> = [
        "--ro-bind",
        "/",
        "/",
        "--dev-bind",
        "/dev",
        "/dev",
        "--proc",
        "/proc",
        "--tmpfs",
        "/tmp",
        "--unshare-pid",
        "--die-with-parent",
        "--new-session",
    ]
    .iter()
    .map(OsString::from)
    .collect();

    // The workspace is mounted after `/tmp`, so a workspace under `/tmp`
    // stays the real folder instead of a path inside the empty tmpfs.
    arguments.extend(["--bind".into(), workspace.clone(), workspace.clone()]);
    arguments.extend(["--chdir".into(), workspace, "--".into()]);
    arguments.extend(shell);

    arguments
}

/// `sandbox-exec` arguments for running `shell` with only `root` and the
/// temporary folders writable.
///
/// Seatbelt compares real paths, so the workspace is resolved through
/// symbolic links first: on macOS, `/tmp` is a link to `/private/tmp`.
fn seatbelt_arguments(root: &Path, shell: Vec<OsString>) -> Vec<OsString> {
    let workspace = std::fs::canonicalize(root)
        .map(OsString::from)
        .unwrap_or_else(|_missing| absolute_path(root));

    let mut parameter = OsString::from("WORKSPACE=");
    parameter.push(workspace);

    let mut arguments: Vec<OsString> = vec!["-D".into(), parameter];
    arguments.extend(["-p".into(), SEATBELT_PROFILE.into()]);
    arguments.extend(shell);

    arguments
}

/// `root` as an absolute path, so the sandbox mounts the folder the command
/// runs in even when the workspace was given as `.`.
fn absolute_path(root: &Path) -> OsString {
    std::path::absolute(root)
        .unwrap_or_else(|_unresolved| root.to_path_buf())
        .into_os_string()
}

/// `bash` followed by `arguments`, for a sandbox tool to run.
fn bash_with(arguments: Vec<OsString>) -> Vec<OsString> {
    std::iter::once(OsString::from("bash"))
        .chain(arguments)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    /// `arguments` as plain strings, for comparing in assertions.
    fn strings(arguments: &[OsString]) -> Vec<String> {
        arguments
            .iter()
            .map(|argument| argument.to_string_lossy().to_string())
            .collect()
    }

    #[test]
    fn without_a_sandbox_the_command_runs_in_bash() {
        let (program, arguments) = wrapped_command(Sandbox::Off, Path::new("/work"), "ls");

        assert_eq!(program, "bash");
        assert_eq!(strings(&arguments), vec!["-lc", "ls"]);
    }

    #[test]
    fn bubblewrap_mounts_the_workspace_writable_after_a_private_tmp() {
        let root = PathBuf::from("/tmp/project");

        let (program, arguments) = wrapped_command(Sandbox::Bubblewrap, &root, "ls");
        let arguments = strings(&arguments);
        let joined = arguments.join(" ");

        assert_eq!(program, "bwrap");
        assert!(joined.starts_with("--ro-bind / / "));
        assert!(joined.contains("--tmpfs /tmp"));
        assert!(joined.ends_with("--chdir /tmp/project -- bash -lc ls"));

        let tmp_position = joined.find("--tmpfs /tmp");
        let bind_position = joined.find("--bind /tmp/project /tmp/project");
        assert!(tmp_position < bind_position);
    }

    #[test]
    fn seatbelt_passes_the_real_workspace_path_as_a_parameter() -> anyhow::Result<()> {
        let root = crate::test_support::unique_temp_dir("openbatrangs-seatbelt");
        let real_root = std::fs::canonicalize(&root)?;

        let (program, arguments) = wrapped_command(Sandbox::Seatbelt, &root, "ls");
        let expected = vec![
            "-D".to_string(),
            format!("WORKSPACE={}", real_root.display()),
            "-p".to_string(),
            SEATBELT_PROFILE.to_string(),
            "bash".to_string(),
            "-lc".to_string(),
            "ls".to_string(),
        ];

        assert_eq!(program, "sandbox-exec");
        assert_eq!(strings(&arguments), expected);

        std::fs::remove_dir_all(root)?;
        Ok(())
    }

    #[test]
    fn seatbelt_uses_the_absolute_path_of_a_missing_workspace() {
        let (_program, arguments) =
            wrapped_command(Sandbox::Seatbelt, Path::new("/no/such/folder"), "ls");

        assert!(strings(&arguments).contains(&"WORKSPACE=/no/such/folder".to_string()));
    }

    #[test]
    fn the_seatbelt_profile_denies_writes_outside_the_allowed_folders() {
        assert!(SEATBELT_PROFILE.contains("(deny file-write*)"));
        assert!(SEATBELT_PROFILE.contains("(subpath (param \"WORKSPACE\"))"));
    }

    #[test]
    fn a_sandbox_is_used_only_when_its_trial_command_works() {
        assert_eq!(
            first_working(Sandbox::Bubblewrap, true),
            Sandbox::Bubblewrap
        );
        assert_eq!(first_working(Sandbox::Bubblewrap, false), Sandbox::Off);
    }

    #[test]
    fn a_missing_program_does_not_run_successfully() {
        assert!(!runs_successfully(
            "openbatrangs-no-such-program".into(),
            &[]
        ));
        assert!(runs_successfully("true".into(), &[]));
    }

    #[test]
    fn only_off_is_an_inactive_sandbox() {
        assert!(Sandbox::Bubblewrap.is_active());
        assert!(Sandbox::Seatbelt.is_active());
        assert!(!Sandbox::Off.is_active());
    }

    #[test]
    fn linux_and_macos_have_a_sandbox_tool_to_try() {
        assert_eq!(sandbox_for_os("linux"), Sandbox::Bubblewrap);
        assert_eq!(sandbox_for_os("macos"), Sandbox::Seatbelt);
        assert_eq!(sandbox_for_os("windows"), Sandbox::Off);
        assert_eq!(available_sandbox(), detect_sandbox());
    }

    #[test]
    fn an_empty_workspace_path_stays_empty() {
        assert_eq!(absolute_path(Path::new("")), "");
    }
}
