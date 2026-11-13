//! # Bounded chat log with disk-backed history
//!
//! An agentic session can produce thousands of lines (tool output, file
//! listings, long code diffs). Keeping every line in memory is wasteful on
//! Jetson-class devices, so this log keeps only a recent window in RAM and
//! spills older lines to a temporary file.
//!
//! Scrolling up should feel like paging through a normal chat: the first
//! `load_more` returns the lines that were evicted most recently, i.e. the
//! lines immediately before the live window. Loading from the very beginning
//! of the session would make the first PageUp jump to the first message ever,
//! which is almost never what the user wants.
//!
//! The prefix window is also capped (`max_prefix`). When it grows too large,
//! the oldest lines are dropped from the front so they can be reloaded on
//! demand. Once the user reaches the very beginning, the cap stops dropping
//! lines so the oldest message stays visible.
//!
//! ## References
//!
//! - `std::fs::File` seek/read: <https://doc.rust-lang.org/std/fs/struct.File.html>
//! - `BufRead::lines`: <https://doc.rust-lang.org/std/io/trait.BufRead.html#method.lines>
//! - Temporary directories: <https://doc.rust-lang.org/std/env/fn.temp_dir.html>

use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

/// Bounded, disk-backed chat log that keeps only a recent window in RAM.
///
/// Older lines are evicted to a temporary file and reloaded on demand when the
/// user scrolls up, keeping memory usage constant regardless of session length.
pub(super) struct SessionLog {
    /// Path to the temporary file where evicted lines are appended.
    path: PathBuf,
    /// In-memory ring of the most recent chat lines (the "live window").
    memory: Vec<String>,
    /// Maximum number of lines kept in the in-memory window before eviction.
    max_memory: usize,
    /// Maximum number of disk-backed lines kept in the visible prefix window.
    /// Older lines are dropped from the front of the window and reloaded on
    /// demand, so memory stays bounded even when the user scrolls far back.
    max_prefix: usize,
    /// Disk-backed lines currently visible before the live memory window, in
    /// chronological order.
    prefix: Vec<String>,
    /// Index into `disk_offsets` of the first line in `prefix`.
    prefix_start: usize,
    /// Byte offset of each evicted line's start in `path`.
    disk_offsets: Vec<u64>,
    /// Current length of the disk file (end offset of the last evicted line).
    disk_len: u64,
}

impl SessionLog {
    /// Create a new session log with the given in-memory capacity.
    ///
    /// # Parameters
    ///
    /// - `max_memory`: maximum lines kept in RAM before evicting to disk.
    ///
    /// # Returns
    ///
    /// A new `SessionLog` backed by a unique temporary file.
    pub(super) fn new(max_memory: usize) -> Self {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "openbatrangs-session-{}-{unique}.log",
            std::process::id()
        ));
        Self {
            path,
            memory: Vec::new(),
            max_memory,
            max_prefix: max_memory.saturating_mul(2).max(1),
            prefix: Vec::new(),
            prefix_start: 0,
            disk_offsets: Vec::new(),
            disk_len: 0,
        }
    }

    /// Append a line, evicting the oldest in-memory line to disk when full.
    ///
    /// # Parameters
    ///
    /// - `line`: the chat line to store.
    pub(super) fn push(&mut self, line: String) {
        self.memory.push(line);

        if self.memory.len() <= self.max_memory {
            return;
        }

        self.evict_oldest_line();
    }

    /// Append several lines in order.
    pub(super) fn push_lines(&mut self, lines: Vec<String>) {
        for line in lines {
            self.push(line);
        }
    }

    /// Move the oldest in-memory line to the disk file.
    ///
    /// A line that cannot be written is dropped, so a full disk never stops
    /// the chat.
    fn evict_oldest_line(&mut self) {
        let evicted = self.memory.remove(0);

        let Some(offset) = append_line(&self.path, self.disk_len, &evicted) else {
            return;
        };

        self.disk_offsets.push(offset);
        self.disk_len = offset + evicted.len() as u64 + 1;

        self.extend_loaded_prefix(evicted);
    }

    /// Keep the loaded prefix contiguous with the live window: a line evicted
    /// while the user is reading older history still belongs between the
    /// prefix and the in-memory tail.
    fn extend_loaded_prefix(&mut self, evicted: String) {
        if self.prefix.is_empty() {
            return;
        }

        self.prefix.push(evicted);
        self.enforce_prefix_cap();
    }

    /// Discard all in-memory and disk-backed history, resetting the log.
    pub(super) fn clear(&mut self) {
        self.memory.clear();
        self.prefix.clear();
        self.prefix_start = 0;
        self.disk_offsets.clear();
        self.disk_len = 0;

        // A temp file that cannot be truncated only wastes disk space.
        std::fs::write(&self.path, "").ok();
    }

    /// True when older lines exist on disk that have not been loaded yet.
    pub(super) fn has_more_history(&self) -> bool {
        self.unloaded_line_count() > 0
    }

    /// Number of disk lines older than the loaded prefix. These are the lines
    /// at indices `0..count` of `disk_offsets`.
    fn unloaded_line_count(&self) -> usize {
        if self.prefix.is_empty() {
            return self.disk_offsets.len();
        }

        self.prefix_start
    }

    /// Load up to `chunk` older lines from disk into the visible prefix.
    ///
    /// Lines are read backwards from the live memory window, so the first
    /// scroll-up shows the most recently evicted lines instead of jumping to
    /// the very beginning of the session.
    ///
    /// # Returns
    /// `true` if at least one line was loaded.
    pub(super) fn load_more(&mut self, chunk: usize) -> bool {
        let end = self.unloaded_line_count();
        let start = end.saturating_sub(chunk);
        let loaded = self.read_disk_lines(start, end);

        if loaded.is_empty() {
            return false;
        }

        self.prefix.splice(0..0, loaded);
        self.prefix_start = start;
        self.enforce_prefix_cap();

        true
    }

    /// Read the evicted lines with indices `start..end` from the disk file.
    ///
    /// # Returns
    /// The lines in order; fewer (or none) when the file cannot be read.
    fn read_disk_lines(&self, start: usize, end: usize) -> Vec<String> {
        let reader = self
            .disk_offsets
            .get(start)
            .and_then(|offset| open_at(&self.path, *offset).ok());

        let Some(reader) = reader else {
            return Vec::new();
        };

        reader
            .lines()
            .take(end - start)
            .map_while(Result::ok)
            .collect()
    }

    /// Drop the oldest loaded lines when the prefix window grows too large.
    ///
    /// Dropping from the front keeps the window contiguous with the live
    /// memory tail and leaves `has_more_history` true, so those lines can be
    /// reloaded if the user keeps scrolling up. When the user has reached the
    /// very beginning of the session (`prefix_start == 0`) nothing is dropped;
    /// otherwise the oldest lines would never be displayable.
    fn enforce_prefix_cap(&mut self) {
        if self.prefix_start == 0 {
            return;
        }

        let excess = self.prefix.len().saturating_sub(self.max_prefix);

        self.prefix.drain(..excess);
        self.prefix_start += excess;
    }

    /// Full visible text: loaded history followed by the in-memory window.
    pub(super) fn text(&self) -> String {
        let lines: Vec<&str> = self
            .prefix
            .iter()
            .chain(&self.memory)
            .map(String::as_str)
            .collect();

        lines.join("\n")
    }

    /// The in-memory lines, oldest first.
    #[cfg(test)]
    pub(super) fn iter(&self) -> std::slice::Iter<'_, String> {
        self.memory.iter()
    }

    /// Whether the log holds no lines, on disk or in memory.
    pub(super) fn is_empty(&self) -> bool {
        self.memory.is_empty() && self.prefix.is_empty()
    }
}

impl Drop for SessionLog {
    /// Delete the temporary file. A file that is already gone needs no cleanup.
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

/// Append a line and return its starting byte offset, or `None` on failure.
fn append_line(path: &Path, offset: u64, line: &str) -> Option<u64> {
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .ok()?;

    writeln!(file, "{line}").ok()?;

    Some(offset)
}

/// Open `path` for reading, positioned at byte `offset`.
fn open_at(path: &Path, offset: u64) -> std::io::Result<BufReader<File>> {
    let mut reader = BufReader::new(File::open(path)?);

    reader.seek(SeekFrom::Start(offset))?;

    Ok(reader)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn evicts_oldest_lines_to_disk_and_loads_them_back() {
        let mut log = SessionLog::new(3);
        for i in 0..5 {
            log.push(format!("line {i}"));
        }
        assert_eq!(log.memory.len(), 3);
        assert_eq!(log.memory[0], "line 2");
        assert!(log.has_more_history());
        assert!(log.load_more(2));
        assert!(log.text().starts_with("line 0\nline 1\nline 2"));
        assert!(!log.has_more_history());
        assert!(!log.load_more(2));
    }

    #[test]
    fn clear_resets_memory_and_history() {
        let mut log = SessionLog::new(2);
        for i in 0..4 {
            log.push(format!("line {i}"));
        }
        log.clear();
        assert!(log.is_empty());
        assert!(!log.has_more_history());
    }

    #[test]
    fn text_joins_prefix_and_memory() {
        let mut log = SessionLog::new(10);
        log.push("a".to_string());
        log.push("b".to_string());
        assert_eq!(log.text(), "a\nb");
    }

    #[test]
    fn loads_history_backwards_from_live_window_in_chunks() {
        let mut log = SessionLog::new(2);
        // Raise the cap so this test exercises ordering, not the memory bound.
        log.max_prefix = 100;
        for i in 0..10 {
            log.push(format!("line {i}"));
        }
        assert_eq!(log.memory.len(), 2);
        assert!(log.has_more_history());

        // First scroll-up loads the newest evicted lines (7, 6, 5), not the
        // oldest ones, so the visible chat is contiguous with what was on
        // screen before scrolling.
        assert!(log.load_more(3));
        assert!(
            log.text()
                .starts_with("line 5\nline 6\nline 7\nline 8\nline 9")
        );

        assert!(log.load_more(3));
        assert!(
            log.text()
                .starts_with("line 2\nline 3\nline 4\nline 5\nline 6\nline 7")
        );

        assert!(log.load_more(100));
        assert!(!log.has_more_history());
        assert!(!log.load_more(100));
    }

    #[test]
    fn appending_after_loading_keeps_order() {
        let mut log = SessionLog::new(2);
        for i in 0..4 {
            log.push(format!("line {i}"));
        }
        assert!(log.load_more(10));
        assert!(!log.has_more_history());

        log.push("line 4".to_string());
        assert_eq!(log.memory.len(), 2);
        // The newly evicted line is folded into the loaded prefix so the
        // visible history stays contiguous with the live memory window.
        assert!(!log.has_more_history());
        assert!(log.text().starts_with("line 0\nline 1\nline 2"));
        assert!(log.text().ends_with("line 4"));
    }

    #[test]
    fn prefix_window_is_capped_and_older_lines_can_reload() {
        let mut log = SessionLog::new(2);
        log.max_prefix = 3;
        for i in 0..10 {
            log.push(format!("line {i}"));
        }

        // Loading in small chunks keeps the window bounded while there is
        // still older history to load.
        assert!(log.load_more(3));
        assert!(log.load_more(3));
        assert!(log.prefix.len() <= log.max_prefix);
        assert!(log.has_more_history());

        // Once the user reaches the very beginning, the oldest lines must
        // remain visible instead of being dropped by the cap.
        assert!(log.load_more(100));
        assert!(log.text().starts_with("line 0"));
        assert!(!log.has_more_history());
    }

    #[test]
    fn has_more_history_false_when_fresh() {
        let log = SessionLog::new(10);
        assert!(!log.has_more_history());
    }

    #[test]
    fn has_more_history_true_after_eviction() {
        let mut log = SessionLog::new(2);
        for i in 0..5 {
            log.push(format!("line {i}"));
        }
        assert!(log.has_more_history());
    }

    #[test]
    fn has_more_history_false_after_loading_all() {
        let mut log = SessionLog::new(2);
        for i in 0..5 {
            log.push(format!("line {i}"));
        }
        log.load_more(100);
        assert!(!log.has_more_history());
    }

    #[test]
    fn clear_after_loading_resets_everything() {
        let mut log = SessionLog::new(2);
        for i in 0..10 {
            log.push(format!("line {i}"));
        }
        log.load_more(5);
        assert!(!log.prefix.is_empty());
        log.clear();
        assert!(log.is_empty());
        assert!(!log.has_more_history());
        assert_eq!(log.text(), "");
    }

    #[test]
    fn text_output_with_empty_log() {
        let log = SessionLog::new(10);
        assert_eq!(log.text(), "");
    }

    #[test]
    fn text_output_with_only_memory() {
        let mut log = SessionLog::new(10);
        log.push("first".to_string());
        log.push("second".to_string());
        assert_eq!(log.text(), "first\nsecond");
    }

    #[test]
    fn large_number_of_lines_causes_eviction() {
        let mut log = SessionLog::new(5);
        for i in 0..100 {
            log.push(format!("line {i}"));
        }
        // Memory should be bounded at max_memory
        assert_eq!(log.memory.len(), 5);
        // The last 5 lines should be in memory
        assert_eq!(log.memory[0], "line 95");
        assert_eq!(log.memory[4], "line 99");
        // History should exist
        assert!(log.has_more_history());
    }

    #[test]
    fn load_more_returns_false_when_nothing_to_load() {
        let mut log = SessionLog::new(10);
        assert!(!log.load_more(5));
    }

    #[test]
    fn load_more_returns_false_when_the_disk_file_is_gone() -> anyhow::Result<()> {
        let mut log = SessionLog::new(1);
        log.push("line 0".to_string());
        log.push("line 1".to_string());

        std::fs::remove_file(&log.path)?;

        assert!(!log.load_more(5));

        Ok(())
    }

    #[test]
    fn a_line_that_cannot_be_written_is_dropped() {
        let mut log = SessionLog::new(1);
        log.path = std::env::temp_dir()
            .join("openbatrangs-missing-dir")
            .join("session.log");

        log.push("line 0".to_string());
        log.push("line 1".to_string());

        assert!(!log.has_more_history());
        assert_eq!(log.text(), "line 1");
    }

    #[test]
    fn push_lines_appends_in_order() {
        let mut log = SessionLog::new(10);

        log.push_lines(vec!["a".to_string(), "b".to_string()]);

        assert_eq!(log.text(), "a\nb");
    }

    #[test]
    fn push_after_clear_starts_fresh() {
        let mut log = SessionLog::new(3);
        for i in 0..5 {
            log.push(format!("line {i}"));
        }
        log.clear();
        log.push("new line".to_string());
        assert_eq!(log.memory.len(), 1);
        assert_eq!(log.memory[0], "new line");
        assert!(!log.has_more_history());
    }
}
