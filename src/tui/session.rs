//! # Session file for transcript blocks that left memory
//!
//! An agentic session can produce thousands of transcript blocks. Only a
//! bounded window of them stays in memory; the rest are written to a
//! temporary session file, one JSON line per block, and read back when the
//! user scrolls to them.
//!
//! The file is append-only. A block written a second time (after it was
//! loaded, folded or unfolded, and moved out again) is appended once more,
//! and its index now points at the newer copy.
//!
//! ## References
//!
//! - `std::fs::File` seek/read: <https://doc.rust-lang.org/std/fs/struct.File.html>
//! - Temporary directories: <https://doc.rust-lang.org/std/env/fn.temp_dir.html>

use super::block::Block;
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

/// Transcript blocks stored on disk, by their index in the transcript.
pub(super) struct BlockStore {
    /// Path of the session file.
    path: PathBuf,
    /// Byte offset of each stored block's line, by block index; `None` for a
    /// block that was never written, or could not be.
    offsets: Vec<Option<u64>>,
    /// Current length of the file.
    file_len: u64,
}

impl BlockStore {
    /// An empty store backed by a new temporary file.
    pub(super) fn new() -> Self {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "openbatrangs-session-{}-{unique}.jsonl",
            std::process::id()
        ));

        Self {
            path,
            offsets: Vec::new(),
            file_len: 0,
        }
    }

    /// Write `block` as block number `index`.
    ///
    /// A block that cannot be written is dropped, so a full disk never stops
    /// the chat; reading it back then finds nothing.
    pub(super) fn write(&mut self, index: usize, block: &Block) {
        // Serializing plain data cannot fail; an empty line would read back as nothing.
        let line = serde_json::to_string(block).unwrap_or_default();

        let offset = append_line(&self.path, self.file_len, &line);
        self.file_len += offset.map_or(0, |_offset| line.len() as u64 + 1);

        self.set_offset(index, offset);
    }

    /// Remember where block `index` starts.
    fn set_offset(&mut self, index: usize, offset: Option<u64>) {
        if self.offsets.len() <= index {
            self.offsets.resize(index + 1, None);
        }

        self.offsets[index] = offset;
    }

    /// Whether block `index` can be read back.
    pub(super) fn has(&self, index: usize) -> bool {
        self.offsets.get(index).is_some_and(Option::is_some)
    }

    /// Read block `index` back, or `None` when it was never stored or the
    /// file cannot be read.
    pub(super) fn read(&self, index: usize) -> Option<Block> {
        let offset = self.offsets.get(index).copied().flatten()?;
        let mut line = String::new();

        let mut reader = open_at(&self.path, offset).ok()?;
        reader.read_line(&mut line).ok()?;

        serde_json::from_str(&line).ok()
    }

    /// Forget every stored block and empty the file.
    pub(super) fn clear(&mut self) {
        self.offsets.clear();
        self.file_len = 0;

        // A temp file that cannot be truncated only wastes disk space.
        std::fs::write(&self.path, "").ok();
    }
}

impl Drop for BlockStore {
    /// Delete the temporary file. A file that is already gone needs no cleanup.
    fn drop(&mut self) {
        std::fs::remove_file(&self.path).ok();
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
    use crate::tui::block::BlockContent;

    /// A notice block with one line.
    fn notice(text: &str) -> Block {
        Block::new(BlockContent::Notice(vec![text.to_string()]))
    }

    #[test]
    fn blocks_are_read_back_by_index() {
        let mut store = BlockStore::new();

        store.write(0, &notice("first"));
        store.write(3, &notice("fourth"));
        store.write(0, &notice("first again"));

        assert!(store.has(0));
        assert!(!store.has(1));
        assert!(!store.has(9));
        assert_eq!(
            store.read(0).map(|block| block.source_text()).as_deref(),
            Some("first again")
        );
        assert_eq!(
            store.read(3).map(|block| block.source_text()).as_deref(),
            Some("fourth")
        );
        assert!(store.read(1).is_none());
    }

    #[test]
    fn clear_forgets_every_block() {
        let mut store = BlockStore::new();
        store.write(0, &notice("gone"));

        store.clear();

        assert!(!store.has(0));
        assert!(store.read(0).is_none());
    }

    #[test]
    fn a_missing_file_reads_nothing() -> anyhow::Result<()> {
        let mut store = BlockStore::new();
        store.write(0, &notice("lost"));

        std::fs::remove_file(&store.path)?;

        assert!(store.read(0).is_none());

        Ok(())
    }

    #[test]
    fn a_block_that_cannot_be_written_is_dropped() {
        let mut store = BlockStore::new();
        store.path = std::env::temp_dir()
            .join("openbatrangs-missing-dir")
            .join("session.jsonl");

        store.write(0, &notice("dropped"));

        assert!(!store.has(0));
    }
}
