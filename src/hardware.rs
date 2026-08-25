//! Hardware capability probing.
//!
//! Keeping `/proc` parsing here (instead of inside model scoring) separates
//! platform concerns from domain logic and makes the memory source mockable in
//! tests.

use crate::constants::models::{BYTES_PER_KIB, FALLBACK_SYSTEM_MEMORY_BYTES};

/// Source of total physical memory for model-budget calculations.
pub trait MemoryInfo {
    /// Return total physical memory in bytes.
    ///
    /// # Returns
    ///
    /// Total system RAM in bytes, used to compute the model memory budget.
    fn total_memory_bytes(&self) -> u64;
}

/// Reads total memory from `/proc/meminfo` (Linux).
pub struct ProcMemoryInfo;

impl MemoryInfo for ProcMemoryInfo {
    fn total_memory_bytes(&self) -> u64 {
        read_memtotal_from_proc()
    }
}

/// Convenience accessor using the default `/proc` implementation.
pub fn total_system_memory_bytes() -> u64 {
    ProcMemoryInfo.total_memory_bytes()
}

/// Parse `MemTotal` from `/proc/meminfo` and return total RAM in bytes.
///
/// Falls back to [`FALLBACK_SYSTEM_MEMORY_BYTES`] when the file is missing or
/// the value cannot be parsed, so model selection always has a usable budget.
///
/// # Returns
///
/// Total physical memory in bytes.
fn read_memtotal_from_proc() -> u64 {
    let content = std::fs::read_to_string("/proc/meminfo").unwrap_or_default();
    for line in content.lines() {
        if let Some(rest) = line.strip_prefix("MemTotal:") {
            let kilobytes: u64 = rest
                .split_whitespace()
                .next()
                .and_then(|value| value.parse().ok())
                .unwrap_or(FALLBACK_SYSTEM_MEMORY_BYTES / BYTES_PER_KIB);
            return kilobytes * BYTES_PER_KIB;
        }
    }
    FALLBACK_SYSTEM_MEMORY_BYTES
}

#[cfg(test)]
mod tests {
    use super::*;

    struct FakeMemory(u64);

    impl MemoryInfo for FakeMemory {
        fn total_memory_bytes(&self) -> u64 {
            self.0
        }
    }

    #[test]
    fn memory_info_trait_is_mockable() {
        let fake = FakeMemory(123);
        assert_eq!(fake.total_memory_bytes(), 123);
    }

    #[test]
    fn proc_memory_returns_positive_total() {
        assert!(ProcMemoryInfo.total_memory_bytes() > 0);
    }

    #[test]
    fn convenience_accessor_matches_proc_source() {
        assert_eq!(
            total_system_memory_bytes(),
            ProcMemoryInfo.total_memory_bytes()
        );
    }
}
