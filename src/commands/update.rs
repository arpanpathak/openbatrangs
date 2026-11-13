//! `openbatrangs update`: check for a newer release now and install it.

use crate::update::{NewRelease, UpdateCheck, check_now, newer_release};
use anyhow::Result;

/// Check now, ignoring the once-a-day limit, print whether a newer release
/// exists, and run the installer when one does.
pub(crate) async fn update(check: &UpdateCheck) -> Result<()> {
    let latest_tag = check_now(check).await?;

    install_if_newer(check, &latest_tag).await
}

/// Install the release `latest_tag` names when it is newer than this binary;
/// otherwise say that this binary is up to date.
async fn install_if_newer(check: &UpdateCheck, latest_tag: &str) -> Result<()> {
    let Some(release) = newer_release(latest_tag, &check.current_version) else {
        println!(
            "openbatrangs {} is up to date (the latest release is {latest_tag}).",
            check.current_version
        );
        return Ok(());
    };

    install(check, &release).await
}

/// Run the installer, print its output, and ask for a restart.
async fn install(check: &UpdateCheck, release: &NewRelease) -> Result<()> {
    println!("{}", release.available_line());
    println!("{}", release.installing_line());

    check.installer.run(|line| println!("{line}")).await?;

    println!("{}", release.restart_line());

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::update::fakes::{fake_check, release_server};

    #[tokio::test]
    async fn a_newer_release_is_installed() -> Result<()> {
        let check = fake_check(&release_server("v0.3.0").await, "echo installed");

        update(&check).await
    }

    #[tokio::test]
    async fn a_failing_installer_is_reported() {
        let check = fake_check(&release_server("v0.3.0").await, "exit 1");

        let outcome = update(&check).await;

        assert!(format!("{outcome:?}").contains("the installer failed"));
    }

    #[tokio::test]
    async fn the_current_release_installs_nothing() -> Result<()> {
        let check = fake_check(&release_server("v0.2.2").await, "exit 1");

        update(&check).await
    }

    #[tokio::test]
    async fn an_unreachable_server_is_reported() {
        let check = fake_check("http://127.0.0.1:1/releases/latest", "exit 1");

        let outcome = update(&check).await;

        assert!(format!("{outcome:?}").contains("cannot read the latest release"));
    }
}
