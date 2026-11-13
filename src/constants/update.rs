//! Release check and update settings.

/// The GitHub API address that describes the latest openbatrangs release.
pub const LATEST_RELEASE_URL: &str =
    "https://api.github.com/repos/arpanpathak/openbatrangs/releases/latest";

/// Environment variable that replaces [`LATEST_RELEASE_URL`], so tests can
/// point the check at a local server.
pub const RELEASE_URL_VARIABLE: &str = "OPENBATRANGS_RELEASE_URL";

/// `User-Agent` header of the release request; the GitHub API refuses
/// requests without one.
pub const RELEASE_USER_AGENT: &str = concat!("openbatrangs/", env!("CARGO_PKG_VERSION"));

/// Seconds the release request may take before the check gives up.
pub const RELEASE_TIMEOUT_SECONDS: u64 = 3;

/// Seconds between two automatic checks: one day.
pub const CHECK_INTERVAL_SECONDS: u64 = 24 * 60 * 60;

/// Folder under the user cache folder that holds the check record.
pub const CACHE_FOLDER_NAME: &str = "openbatrangs";

/// File in that folder that records the time and result of the last check.
pub const CACHE_FILE_NAME: &str = "update-check.json";

/// Shell command that downloads the release installer and runs it.
pub const INSTALLER_COMMAND: &str = "curl -fsSL https://github.com/arpanpathak/openbatrangs/releases/latest/download/install.sh | sh";

/// Environment variable that tells the installer which folder to put the
/// binary in.
pub const INSTALL_DIR_VARIABLE: &str = "OPENBATRANGS_INSTALL_DIR";
