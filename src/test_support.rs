//! Test-only helpers shared across unit-test modules.
//!
//! Temp-dir helpers must be unique even when tests run in parallel on multiple
//! threads: a bare nanosecond timestamp can collide, causing one test to delete
//! another test's directory mid-write.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

static TEMP_DIR_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Create a new empty temporary directory with a process-unique name.
pub(crate) fn unique_temp_dir(prefix: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let counter = TEMP_DIR_COUNTER.fetch_add(1, Ordering::Relaxed);
    let path =
        std::env::temp_dir().join(format!("{prefix}-{}-{nanos}-{counter}", std::process::id()));
    std::fs::create_dir_all(&path).expect("create temp dir");
    path
}
