//! The record of the last release check. It lives in the user cache folder
//! and keeps the automatic check to once a day.

use crate::constants::update::{CACHE_FILE_NAME, CACHE_FOLDER_NAME, CHECK_INTERVAL_SECONDS};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// When the last check ran and what it found.
#[derive(Debug, PartialEq, Serialize, Deserialize)]
pub(crate) struct CheckRecord {
    /// When the check ran, in seconds since the Unix epoch.
    pub(crate) checked_at: u64,
    /// Tag of the latest release the check found, such as `v0.3.0`.
    pub(crate) latest_tag: String,
}

/// The openbatrangs folder in the user cache folder:
/// `$XDG_CACHE_HOME/openbatrangs` when that variable holds an absolute path,
/// else `~/.cache/openbatrangs`. `None` when neither is known.
pub(crate) fn cache_folder(
    xdg_cache_home: Option<PathBuf>,
    home: Option<PathBuf>,
) -> Option<PathBuf> {
    let cache_root = xdg_cache_home
        .filter(|path| path.is_absolute())
        .or_else(|| home.map(|home| home.join(".cache")));

    cache_root.map(|root| root.join(CACHE_FOLDER_NAME))
}

/// The record in `cache_folder`, or `None` when the file is missing or
/// cannot be read.
pub(crate) fn read_record(cache_folder: &Path) -> Option<CheckRecord> {
    let text = std::fs::read_to_string(cache_folder.join(CACHE_FILE_NAME)).ok();

    text.and_then(|text| serde_json::from_str(&text).ok())
}

/// Whether the record in `cache_folder` is less than a day old at
/// `now_seconds`. A missing or unreadable record is never recent.
pub(crate) fn is_recent(cache_folder: Option<&Path>, now_seconds: u64) -> bool {
    cache_folder.and_then(read_record).is_some_and(|record| {
        now_seconds.saturating_sub(record.checked_at) < CHECK_INTERVAL_SECONDS
    })
}

/// Write `record` into `cache_folder`, creating the folder when needed.
pub(crate) fn write_record(cache_folder: &Path, record: &CheckRecord) -> std::io::Result<()> {
    let text = serde_json::to_string(record)?;

    std::fs::create_dir_all(cache_folder)?;
    std::fs::write(cache_folder.join(CACHE_FILE_NAME), text)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::unique_temp_dir;

    /// A fixed time: 14 November 2026, in seconds since the Unix epoch.
    const NOW: u64 = 1_794_614_400;

    #[test]
    fn the_cache_folder_prefers_an_absolute_xdg_cache_home() {
        let xdg = Some(PathBuf::from("/var/cache/user"));
        let home = Some(PathBuf::from("/home/user"));

        assert_eq!(
            cache_folder(xdg, home.clone()),
            Some(PathBuf::from("/var/cache/user/openbatrangs"))
        );
        assert_eq!(
            cache_folder(Some(PathBuf::from("relative")), home.clone()),
            Some(PathBuf::from("/home/user/.cache/openbatrangs"))
        );
        assert_eq!(
            cache_folder(None, home),
            Some(PathBuf::from("/home/user/.cache/openbatrangs"))
        );
        assert_eq!(cache_folder(None, None), None);
    }

    #[test]
    fn a_written_record_reads_back() -> anyhow::Result<()> {
        let folder = unique_temp_dir("openbatrangs-update-record").join("nested");
        let record = CheckRecord {
            checked_at: NOW,
            latest_tag: "v0.3.0".to_string(),
        };

        write_record(&folder, &record)?;

        assert_eq!(read_record(&folder), Some(record));
        Ok(())
    }

    #[test]
    fn a_record_is_recent_for_one_day() -> anyhow::Result<()> {
        let folder = unique_temp_dir("openbatrangs-update-recent");
        let record = CheckRecord {
            checked_at: NOW,
            latest_tag: "v0.3.0".to_string(),
        };
        write_record(&folder, &record)?;

        assert!(is_recent(Some(&folder), NOW));
        assert!(is_recent(Some(&folder), NOW + CHECK_INTERVAL_SECONDS - 1));
        assert!(!is_recent(Some(&folder), NOW + CHECK_INTERVAL_SECONDS));
        Ok(())
    }

    #[test]
    fn a_missing_or_broken_record_is_not_recent() -> anyhow::Result<()> {
        let folder = unique_temp_dir("openbatrangs-update-broken");

        assert!(!is_recent(None, NOW));
        assert!(!is_recent(Some(&folder), NOW));

        std::fs::write(folder.join(CACHE_FILE_NAME), "not json")?;

        assert!(!is_recent(Some(&folder), NOW));
        Ok(())
    }

    #[test]
    fn a_record_in_a_file_path_cannot_be_written() -> anyhow::Result<()> {
        let folder = unique_temp_dir("openbatrangs-update-blocked");
        let blocked = folder.join("a-file");
        std::fs::write(&blocked, "")?;
        let record = CheckRecord {
            checked_at: NOW,
            latest_tag: "v0.3.0".to_string(),
        };

        assert!(write_record(&blocked, &record).is_err());
        Ok(())
    }
}
