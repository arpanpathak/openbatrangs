//! Bounded chat log with disk-backed history.
//!
//! Only the most recent lines are kept in memory. Older lines are appended to a
//! temporary session file and loaded back on demand when the user scrolls up,
//! so long agent sessions do not grow memory without bound.
//!
//! The on-disk history is accessed with a byte cursor: each evicted line's
//! offset is recorded when it is appended, and `load_more` seeks straight to the
//! next unread offset. This keeps repeated scroll-up loads O(chunk) instead of
//! O(total history) per call.

use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

pub(super) struct SessionLog {
    path: PathBuf,
    memory: Vec<String>,
    max_memory: usize,
    prefix: Vec<String>,
    /// Byte offset of each evicted line's start in `path`.
    disk_offsets: Vec<u64>,
    /// Current length of the disk file (end offset of the last evicted line).
    disk_len: u64,
    loaded_prefix_count: usize,
}

impl SessionLog {
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
            prefix: Vec::new(),
            disk_offsets: Vec::new(),
            disk_len: 0,
            loaded_prefix_count: 0,
        }
    }

    pub(super) fn push(&mut self, line: String) {
        self.memory.push(line);
        if self.memory.len() > self.max_memory {
            let evicted = self.memory.remove(0);
            if let Some(offset) = append_line(&self.path, self.disk_len, &evicted) {
                self.disk_offsets.push(offset);
                self.disk_len = offset + evicted.len() as u64 + 1;
            }
        }
    }

    pub(super) fn clear(&mut self) {
        self.memory.clear();
        self.prefix.clear();
        self.disk_offsets.clear();
        self.disk_len = 0;
        self.loaded_prefix_count = 0;
        let _ = std::fs::write(&self.path, "");
    }

    /// True when older lines exist on disk that have not been loaded yet.
    pub(super) fn has_more_history(&self) -> bool {
        self.loaded_prefix_count < self.disk_offsets.len()
    }

    /// Load up to `chunk` older lines from disk into the visible prefix.
    ///
    /// # Returns
    /// `true` if at least one line was loaded.
    pub(super) fn load_more(&mut self, chunk: usize) -> bool {
        let start = self.loaded_prefix_count;
        if start >= self.disk_offsets.len() {
            return false;
        }
        let end = (start + chunk).min(self.disk_offsets.len());
        let Some(offset) = self.disk_offsets.get(start).copied() else {
            return false;
        };
        let Ok(file) = File::open(&self.path) else {
            return false;
        };
        let mut reader = BufReader::new(file);
        if reader.seek(SeekFrom::Start(offset)).is_err() {
            return false;
        }

        let mut loaded = 0usize;
        for line in reader.lines().take(end - start) {
            match line {
                Ok(line) => {
                    self.prefix.push(line);
                    loaded += 1;
                }
                Err(_) => break,
            }
        }
        if loaded == 0 {
            return false;
        }
        self.loaded_prefix_count = start + loaded;
        true
    }

    /// Full visible text: loaded history followed by the in-memory window.
    pub(super) fn text(&self) -> String {
        let mut parts = Vec::new();
        if !self.prefix.is_empty() {
            parts.push(self.prefix.join("\n"));
        }
        if !self.memory.is_empty() {
            parts.push(self.memory.join("\n"));
        }
        parts.join("\n")
    }

    #[cfg(test)]
    pub(super) fn iter(&self) -> std::slice::Iter<'_, String> {
        self.memory.iter()
    }

    #[cfg(test)]
    pub(super) fn is_empty(&self) -> bool {
        self.memory.is_empty() && self.prefix.is_empty()
    }
}

impl Drop for SessionLog {
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
    fn loads_history_in_chunks_in_order() {
        let mut log = SessionLog::new(2);
        for i in 0..10 {
            log.push(format!("line {i}"));
        }
        assert_eq!(log.memory.len(), 2);
        assert!(log.has_more_history());

        assert!(log.load_more(3));
        assert!(log.text().starts_with("line 0\nline 1\nline 2"));

        assert!(log.load_more(3));
        assert!(log
            .text()
            .starts_with("line 0\nline 1\nline 2\nline 3\nline 4\nline 5"));

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
        // The newly evicted line (line 2) is now on disk and can be loaded.
        assert!(log.has_more_history());
        assert!(log.load_more(10));
        assert!(log.text().starts_with("line 0\nline 1\nline 2"));
        assert!(log.text().ends_with("line 4"));
    }
}
