//! Running the release installer and passing on its output line by line.

use crate::constants::update::{INSTALL_DIR_VARIABLE, INSTALLER_COMMAND};
use anyhow::{Context, Result, bail};
use std::path::{Path, PathBuf};
use std::process::{ExitStatus, Stdio};
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::{Child, ChildStdout, Command};

/// The release installer and the folder it installs into.
#[derive(Clone, Debug)]
pub(crate) struct Installer {
    /// Shell command that installs the release: [`INSTALLER_COMMAND`]
    /// outside tests.
    pub(crate) command: String,
    /// Folder for the new binary, passed to the installer as
    /// `OPENBATRANGS_INSTALL_DIR`. `None` leaves the choice to the installer.
    pub(crate) install_dir: Option<PathBuf>,
}

impl Installer {
    /// The release installer, set to replace the running executable.
    pub(crate) fn for_this_process() -> Self {
        Self {
            command: INSTALLER_COMMAND.to_string(),
            install_dir: executable_folder(std::env::current_exe().ok()),
        }
    }

    /// Run the installer and pass each line it prints, standard error
    /// included, to `on_line`. An installer that exits with an error status
    /// gives an error.
    pub(crate) async fn run(&self, on_line: impl FnMut(String)) -> Result<()> {
        let mut child = self.spawn().context("cannot start the installer")?;
        let output = child.stdout.take().context("the installer has no output")?;

        forward_lines(output, on_line).await;

        installer_outcome(child.wait().await?)
    }

    /// Start `sh -c` with the command, its standard error sent to standard
    /// output, and the install folder in the environment.
    fn spawn(&self) -> std::io::Result<Child> {
        let install_dir = self
            .install_dir
            .iter()
            .map(|folder| (INSTALL_DIR_VARIABLE, folder));

        Command::new("sh")
            .arg("-c")
            .arg(format!("exec 2>&1; {}", self.command))
            .envs(install_dir)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .kill_on_drop(true)
            .spawn()
    }
}

/// The folder that holds `executable`, or `None` when it is unknown.
fn executable_folder(executable: Option<PathBuf>) -> Option<PathBuf> {
    executable?.parent().map(Path::to_path_buf)
}

/// Pass each line of `output` to `on_line` until the output ends.
async fn forward_lines(output: ChildStdout, mut on_line: impl FnMut(String)) {
    let mut lines = BufReader::new(output).lines();

    while let Ok(Some(line)) = lines.next_line().await {
        on_line(line);
    }
}

/// An error when the installer exited with an error status.
fn installer_outcome(status: ExitStatus) -> Result<()> {
    if !status.success() {
        bail!("the installer failed ({status})");
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::unique_temp_dir;

    /// An installer that runs `command` into `install_dir`.
    fn fake_installer(command: &str, install_dir: Option<PathBuf>) -> Installer {
        Installer {
            command: command.to_string(),
            install_dir,
        }
    }

    #[test]
    fn the_real_installer_targets_the_executable_folder() -> anyhow::Result<()> {
        let installer = Installer::for_this_process();
        let executable = std::env::current_exe()?;

        assert_eq!(installer.command, INSTALLER_COMMAND);
        assert_eq!(installer.install_dir.as_deref(), executable.parent());
        Ok(())
    }

    #[test]
    fn an_unknown_executable_has_no_folder() {
        assert_eq!(executable_folder(None), None);
        assert_eq!(executable_folder(Some(PathBuf::from("/"))), None);
    }

    #[tokio::test]
    async fn every_output_line_and_the_install_folder_are_passed_on() -> anyhow::Result<()> {
        let folder = unique_temp_dir("openbatrangs-installer");
        let installer = fake_installer(
            "echo installed; echo \"into $OPENBATRANGS_INSTALL_DIR\" >&2",
            Some(folder.clone()),
        );
        let mut lines = Vec::new();

        installer.run(|line| lines.push(line)).await?;

        assert_eq!(
            lines,
            vec![
                "installed".to_string(),
                format!("into {}", folder.display())
            ]
        );
        Ok(())
    }

    #[tokio::test]
    async fn a_failing_installer_is_an_error() {
        let installer = fake_installer("echo broken; exit 3", None);
        let mut lines = Vec::new();

        let outcome = installer.run(|line| lines.push(line)).await;

        assert_eq!(lines, vec!["broken".to_string()]);
        assert!(format!("{outcome:?}").contains("the installer failed"));
    }
}
