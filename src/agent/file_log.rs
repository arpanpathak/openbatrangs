//! # The files a run has read and changed
//!
//! A model that edits or overwrites a file it has not read works from
//! memory, and on long tasks that memory is often wrong: an edit misses, or
//! a rewrite drops code the model forgot. [`FileLog`] records the version of
//! each file the run read or wrote, so `write_file` and `edit_file` can
//! refuse an existing file the run has not read, or one that changed since,
//! for example through a command or the user's own editor.
//!
//! A version is the file's size and modification time, which every write
//! changes, so checking it costs one `stat` call and no read.

use anyhow::{Result, ensure};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

/// The size and modification time of a file.
type Version = (u64, Option<SystemTime>);

/// The files one run changed, and the version of each file it last read or
/// wrote.
#[derive(Debug, Default)]
pub(super) struct FileLog {
    /// Paths the run wrote or edited, as the model gave them, in order.
    pub(super) changed: Vec<String>,
    /// The version the run last read or wrote, by canonical path.
    seen: HashMap<PathBuf, Version>,
}

impl FileLog {
    /// A log whose changed files are `changed`, as a resumed run finds
    /// them in its history; it has seen no file yet.
    pub(super) fn with_changed(changed: Vec<String>) -> Self {
        Self {
            changed,
            seen: HashMap::new(),
        }
    }

    /// Record that the run now knows `path` under `cwd` as it is on disk.
    pub(super) fn saw(&mut self, cwd: &Path, path: &str) {
        let Some((file, version)) = version_of(cwd, path) else {
            return;
        };

        self.seen.insert(file, version);
    }

    /// Record that the run wrote or edited `path`, which it now knows.
    pub(super) fn wrote(&mut self, cwd: &Path, path: &str) {
        self.changed.push(path.to_string());
        self.saw(cwd, path);
    }

    /// Fail when `path` exists and the run has not read or written it since
    /// it last changed; a file that does not exist yet may be written.
    pub(super) fn ensure_known(&self, cwd: &Path, path: &str) -> Result<()> {
        let Some((file, version)) = version_of(cwd, path) else {
            return Ok(());
        };

        ensure!(
            self.seen.get(&file) == Some(&version),
            "{path} already exists and you have not read it since it last changed; \
             read it with read_file, then change it with edit_file"
        );

        Ok(())
    }
}

/// The canonical path and the version of `path` under `cwd`; `None` when
/// it does not exist.
fn version_of(cwd: &Path, path: &str) -> Option<(PathBuf, Version)> {
    let file = cwd.join(path).canonicalize().ok()?;
    let metadata = fs::metadata(&file).ok()?;

    Some((file, (metadata.len(), metadata.modified().ok())))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::unique_temp_dir;

    #[test]
    fn a_new_file_may_be_written_and_an_unread_one_may_not() -> Result<()> {
        let root = unique_temp_dir("openbatrangs-file-log-new");
        fs::write(root.join("old.rs"), "fn old() {}")?;
        let log = FileLog::default();

        assert!(log.ensure_known(&root, "new.rs").is_ok());
        assert!(log.ensure_known(&root, "old.rs").is_err());

        fs::remove_dir_all(root)?;
        Ok(())
    }

    #[test]
    fn a_read_file_is_known_until_it_changes() -> Result<()> {
        let root = unique_temp_dir("openbatrangs-file-log-read");
        fs::write(root.join("a.rs"), "fn a() {}")?;
        let mut log = FileLog::default();

        log.saw(&root, "./a.rs");
        assert!(
            log.ensure_known(&root, "a.rs").is_ok(),
            "the same file by another path"
        );

        fs::write(root.join("a.rs"), "fn a() { changed_elsewhere(); }")?;
        let refused = log.ensure_known(&root, "a.rs");

        assert!(format!("{refused:?}").contains("read it with read_file"));

        log.saw(&root, "missing.rs");
        log.wrote(&root, "a.rs");
        assert!(log.ensure_known(&root, "a.rs").is_ok());
        assert_eq!(log.changed, ["a.rs"]);

        fs::remove_dir_all(root)?;
        Ok(())
    }
}
