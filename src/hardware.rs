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

/// Reads total memory from `/proc/meminfo` (Linux), or `sysctl` (macOS).
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

/// Total RAM in bytes from `/proc/meminfo`, else from `sysctl`.
///
/// Falls back to [`FALLBACK_SYSTEM_MEMORY_BYTES`] when neither source gives a
/// value, so model selection always has a usable budget.
fn read_memtotal_from_proc() -> u64 {
    let meminfo = std::fs::read_to_string("/proc/meminfo").unwrap_or_default();

    memtotal_bytes(&meminfo).unwrap_or_else(memory_from_sysctl)
}

/// `MemTotal` from the text of `/proc/meminfo`, in bytes, or `None` when the
/// text has no `MemTotal` line.
fn memtotal_bytes(meminfo: &str) -> Option<u64> {
    let value = meminfo
        .lines()
        .find_map(|line| line.strip_prefix("MemTotal:"))?;

    Some(kibibytes_or_fallback(value) * BYTES_PER_KIB)
}

/// The number of kibibytes in a value such as ` 16384 kB`; an unreadable
/// value counts as [`FALLBACK_SYSTEM_MEMORY_BYTES`].
fn kibibytes_or_fallback(value: &str) -> u64 {
    value
        .split_whitespace()
        .next()
        .and_then(|number| number.parse().ok())
        .unwrap_or(FALLBACK_SYSTEM_MEMORY_BYTES / BYTES_PER_KIB)
}

/// Total RAM from `sysctl`, or [`FALLBACK_SYSTEM_MEMORY_BYTES`] without it.
fn memory_from_sysctl() -> u64 {
    read_memsize_from_sysctl().unwrap_or(FALLBACK_SYSTEM_MEMORY_BYTES)
}

/// Total RAM on macOS, which has no `/proc`: `sysctl -n hw.memsize`.
fn read_memsize_from_sysctl() -> Option<u64> {
    let output = std::process::Command::new("sysctl")
        .args(["-n", "hw.memsize"])
        .output()
        .ok()?;

    String::from_utf8_lossy(&output.stdout).trim().parse().ok()
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
    fn memtotal_is_read_in_kibibytes() {
        assert_eq!(
            memtotal_bytes("MemFree: 1 kB\nMemTotal:  16 kB\n"),
            Some(16 * BYTES_PER_KIB)
        );
    }

    #[test]
    fn unreadable_memtotal_uses_the_fallback() {
        assert_eq!(
            memtotal_bytes("MemTotal: many kB"),
            Some(FALLBACK_SYSTEM_MEMORY_BYTES)
        );
    }

    #[test]
    fn meminfo_without_memtotal_has_no_total() {
        assert_eq!(memtotal_bytes("MemFree: 1 kB"), None);
    }

    #[test]
    fn sysctl_or_fallback_is_positive() {
        assert!(memory_from_sysctl() > 0);
    }

    #[test]
    fn convenience_accessor_matches_proc_source() {
        assert_eq!(
            total_system_memory_bytes(),
            ProcMemoryInfo.total_memory_bytes()
        );
    }
}
