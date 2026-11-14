//! # Release check and update
//!
//! Reads the latest release from GitHub, compares it with the running
//! version, and runs the release installer.
//!
//! - [`version`]: release versions and the newer-than comparison.
//! - [`cache`]: the record that keeps the automatic check to once a day.
//! - [`installer`]: running the installer and passing on its output.
//!
//! The network address, the cache folder, the clock and the installer are
//! fields of [`UpdateCheck`], so tests use a local server, a temporary
//! folder, a fixed time and a harmless command. Every failure of the
//! automatic check is silent: it gives no newer release.
//!
//! ## References
//!
//! - GitHub latest release API: <https://docs.github.com/en/rest/releases/releases#get-the-latest-release>
//! - XDG base directories: <https://specifications.freedesktop.org/basedir-spec/latest/>

mod cache;
mod installer;
mod version;

pub(crate) use installer::{Installer, installer_command};
pub(crate) use version::{NewRelease, newer_release};

use crate::cli::{Cli, home_directory};
use crate::constants::update::{
    LATEST_RELEASE_URL, RELEASE_TIMEOUT_SECONDS, RELEASE_URL_VARIABLE, RELEASE_USER_AGENT,
};
use anyhow::{Context, Result};
use cache::{CheckRecord, cache_folder, is_recent, write_record};
use serde::Deserialize;
use std::path::PathBuf;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// Everything a release check needs.
#[derive(Clone, Debug)]
pub(crate) struct UpdateCheck {
    /// Address of the latest-release endpoint of the GitHub API.
    pub(crate) release_url: String,
    /// Folder that holds the check record, or `None` when no cache folder
    /// is known; then the automatic check runs every time.
    pub(crate) cache_folder: Option<PathBuf>,
    /// Version of the running binary, such as `0.2.2`.
    pub(crate) current_version: String,
    /// Time of the check, in seconds since the Unix epoch.
    pub(crate) now_seconds: u64,
    /// The installer that runs when the user accepts the update.
    pub(crate) installer: Installer,
}

impl UpdateCheck {
    /// The check for this process: `OPENBATRANGS_RELEASE_URL` or the GitHub
    /// API, the user cache folder, this binary's version, the system clock
    /// and the release installer.
    pub(crate) fn for_this_process() -> Self {
        let xdg_cache_home = std::env::var_os("XDG_CACHE_HOME").map(PathBuf::from);
        let release_url = std::env::var(RELEASE_URL_VARIABLE)
            .unwrap_or_else(|_missing| LATEST_RELEASE_URL.to_string());

        Self {
            release_url,
            cache_folder: cache_folder(xdg_cache_home, home_directory()),
            current_version: env!("CARGO_PKG_VERSION").to_string(),
            now_seconds: unix_seconds(SystemTime::now()),
            installer: Installer::for_this_process(),
        }
    }
}

/// The body of the latest-release endpoint; only the tag is read.
#[derive(Deserialize)]
struct Release {
    /// Tag of the release, such as `v0.3.0`.
    tag_name: String,
}

/// The automatic check for this run, or `None` when `--no-update-check` or
/// `OPENBATRANGS_NO_UPDATE_CHECK` turns it off.
pub(crate) fn automatic_check(cli: &Cli) -> Option<UpdateCheck> {
    (!cli.is_update_check_disabled).then(UpdateCheck::for_this_process)
}

/// The automatic check: the newer release, or `None` when the last check
/// was less than a day ago, when no newer release exists, or when anything
/// fails.
pub(crate) async fn daily_check(check: &UpdateCheck) -> Option<NewRelease> {
    if is_recent(check.cache_folder.as_deref(), check.now_seconds) {
        return None;
    }

    compare_with_latest(check).await
}

/// The newer release, or `None` when there is none or the check fails.
async fn compare_with_latest(check: &UpdateCheck) -> Option<NewRelease> {
    let latest_tag = check_now(check).await.ok()?;

    newer_release(&latest_tag, &check.current_version)
}

/// Read the tag of the latest release and record the check. A record that
/// cannot be written is skipped: the next run checks again.
pub(crate) async fn check_now(check: &UpdateCheck) -> Result<String> {
    let latest_tag = latest_release_tag(&check.release_url)
        .await
        .with_context(|| format!("cannot read the latest release from {}", check.release_url))?;

    record_check(check, &latest_tag);

    Ok(latest_tag)
}

/// Write the time of the check and the tag it found into the cache folder.
fn record_check(check: &UpdateCheck, latest_tag: &str) {
    let record = CheckRecord {
        checked_at: check.now_seconds,
        latest_tag: latest_tag.to_string(),
    };

    if let Some(folder) = &check.cache_folder {
        write_record(folder, &record).ok();
    }
}

/// The `tag_name` of the release at `release_url`.
async fn latest_release_tag(release_url: &str) -> Result<String> {
    let client = reqwest::Client::builder()
        .user_agent(RELEASE_USER_AGENT)
        .timeout(Duration::from_secs(RELEASE_TIMEOUT_SECONDS))
        .build()?;

    let response = client.get(release_url).send().await?.error_for_status()?;
    let release: Release = response.json().await?;

    Ok(release.tag_name)
}

/// Seconds since the Unix epoch at `time`; 0 for a time before it.
fn unix_seconds(time: SystemTime) -> u64 {
    time.duration_since(UNIX_EPOCH)
        .map(|elapsed| elapsed.as_secs())
        .unwrap_or(0)
}

#[cfg(test)]
pub(crate) mod fakes {
    //! A fake release server and check settings for tests.

    use super::*;
    use crate::test_support::{MockResponse, spawn_mock_server, unique_temp_dir};

    /// A fixed time: 14 November 2026, in seconds since the Unix epoch.
    pub(crate) const NOW: u64 = 1_794_614_400;

    /// A local server whose latest release is `tag`.
    pub(crate) async fn release_server(tag: &'static str) -> String {
        let base_url = spawn_mock_server(move |_path| {
            MockResponse::json("200 OK", format!(r#"{{"tag_name":"{tag}"}}"#))
        })
        .await;

        format!("{base_url}/releases/latest")
    }

    /// A check against `release_url` at [`NOW`], for version 0.2.2, with an
    /// empty cache folder and an installer that runs `installer_command`.
    pub(crate) fn fake_check(release_url: &str, installer_command: &str) -> UpdateCheck {
        UpdateCheck {
            release_url: release_url.to_string(),
            cache_folder: Some(unique_temp_dir("openbatrangs-update-cache")),
            current_version: "0.2.2".to_string(),
            now_seconds: NOW,
            installer: Installer {
                command: installer_command.to_string(),
                install_dir: None,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::fakes::{NOW, fake_check, release_server};
    use super::*;
    use crate::constants::update::CHECK_INTERVAL_SECONDS;
    use crate::test_support::{MockResponse, spawn_mock_server};
    use clap::Parser;

    /// An address where nothing listens, so every request is refused at once.
    const CLOSED_URL: &str = "http://127.0.0.1:1/releases/latest";

    #[test]
    fn this_process_checks_its_own_version() {
        let check = UpdateCheck::for_this_process();

        assert_eq!(check.current_version, env!("CARGO_PKG_VERSION"));
        assert!(check.now_seconds > NOW - CHECK_INTERVAL_SECONDS * 365);
    }

    #[test]
    fn the_flag_turns_the_automatic_check_off() {
        let with_flag = Cli::parse_from(["openbatrangs", "--no-update-check"]);
        let without_flag = Cli::parse_from(["openbatrangs"]);

        assert!(automatic_check(&with_flag).is_none());
        assert_eq!(
            automatic_check(&without_flag).is_some(),
            std::env::var_os("OPENBATRANGS_NO_UPDATE_CHECK").is_none()
        );
    }

    #[test]
    fn a_time_before_the_epoch_is_zero() {
        let before_epoch = UNIX_EPOCH - Duration::from_secs(1);

        assert_eq!(unix_seconds(before_epoch), 0);
        assert_eq!(unix_seconds(UNIX_EPOCH + Duration::from_secs(7)), 7);
    }

    #[tokio::test]
    async fn a_newer_release_is_found_and_recorded() -> anyhow::Result<()> {
        let check = fake_check(&release_server("v0.3.0").await, "true");

        let release = daily_check(&check).await;
        let record = cache::read_record(&check.cache_folder.clone().unwrap_or_default());

        assert_eq!(release, newer_release("v0.3.0", "0.2.2"));
        assert_eq!(
            record,
            Some(CheckRecord {
                checked_at: NOW,
                latest_tag: "v0.3.0".to_string()
            })
        );
        Ok(())
    }

    #[tokio::test]
    async fn the_check_runs_at_most_once_a_day() {
        let mut check = fake_check(&release_server("v0.3.0").await, "true");

        assert!(daily_check(&check).await.is_some());
        assert!(daily_check(&check).await.is_none());

        check.now_seconds = NOW + CHECK_INTERVAL_SECONDS;

        assert!(daily_check(&check).await.is_some());
    }

    #[tokio::test]
    async fn the_same_release_is_not_newer() {
        let check = fake_check(&release_server("v0.2.2").await, "true");

        assert!(daily_check(&check).await.is_none());
    }

    #[tokio::test]
    async fn every_failure_is_silent() {
        let refused = fake_check(CLOSED_URL, "true");
        let server_error = spawn_mock_server(|_path| MockResponse::text("500 Oops", "down")).await;
        let broken_body = spawn_mock_server(|_path| MockResponse::json("200 OK", "{}")).await;

        assert!(daily_check(&refused).await.is_none());
        assert!(
            daily_check(&fake_check(&server_error, "true"))
                .await
                .is_none()
        );
        assert!(
            daily_check(&fake_check(&broken_body, "true"))
                .await
                .is_none()
        );
    }

    #[tokio::test]
    async fn a_failed_check_is_not_recorded() {
        let check = fake_check(CLOSED_URL, "true");

        daily_check(&check).await;

        assert!(!is_recent(check.cache_folder.as_deref(), NOW));
    }

    #[tokio::test]
    async fn a_cache_that_cannot_be_written_still_gives_the_release() -> anyhow::Result<()> {
        let mut check = fake_check(&release_server("v0.3.0").await, "true");
        let blocked = check
            .cache_folder
            .clone()
            .unwrap_or_default()
            .join("a-file");
        std::fs::write(&blocked, "")?;
        check.cache_folder = Some(blocked);

        assert!(daily_check(&check).await.is_some());

        check.cache_folder = None;

        assert!(daily_check(&check).await.is_some());
        Ok(())
    }
}
