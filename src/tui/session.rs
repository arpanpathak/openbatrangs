//! Bounded chat log with disk-backed history.
//!
//! Only the most recent lines are kept in memory. Older lines are appended to a
//! temporary session file and loaded back on demand when the user scrolls up,
//! so long agent sessions do not grow memory without bound.

use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

pub(super) struct SessionLog {
    path: PathBuf,
    memory: Vec<String>,
    max_memory: usize,
    prefix: Vec<String>,
    evicted_count: usize,
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
            evicted_count: 0,
            loaded_prefix_count: 0,
        }
    }

    pub(super) fn push(&mut self, line: String) {
        self.memory.push(line);
        if self.memory.len() > self.max_memory {
            let evicted = self.memory.remove(0);
            if append_line(&self.path, &evicted) {
                self.evicted_count += 1;
            }
        }
    }

    pub(super) fn clear(&mut self) {
        self.memory.clear();
        self.prefix.clear();
        self.evicted_count = 0;
        self.loaded_prefix_count = 0;
        let _ = std::fs::write(&self.path, "");
    }

    /// True when older lines exist on disk that have not been loaded yet.
    pub(super) fn has_more_history(&self) -> bool {
        self.evicted_count > self.loaded_prefix_count
    }

    /// Load up to `chunk` older lines from disk into the visible prefix.
    ///
    /// # Returns
    /// `true` if at least one line was loaded.
    pub(super) fn load_more(&mut self, chunk: usize) -> bool {
        let Ok(lines) = read_lines(&self.path) else {
            return false;
        };
        let start = self.loaded_prefix_count.min(lines.len());
        let end = (start + chunk).min(lines.len());
        if start >= end {
            return false;
        }
        self.prefix.extend(lines[start..end].iter().cloned());
        self.loaded_prefix_count = end;
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

fn append_line(path: &Path, line: &str) -> bool {
    let Ok(mut file) = OpenOptions::new().create(true).append(true).open(path) else {
        return false;
    };
    writeln!(file, "{line}").is_ok()
}

fn read_lines(path: &Path) -> std::io::Result<Vec<String>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    reader.lines().collect()
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
}
