//! Performance sampling and formatting constants.

use std::time::Duration;

/// How often the TUI refreshes the performance panel.
pub const SAMPLE_INTERVAL: Duration = Duration::from_secs(1);

/// `tegrastats` sampling interval in milliseconds.
pub const TEGRASTATS_INTERVAL_MILLIS: &str = "1000";

/// Milliwatts per watt, used to normalize `VDD_IN` readings.
pub const MILLIWATTS_PER_WATT: f64 = 1_000.0;

/// Mebibytes per gibibyte for human-readable RAM formatting.
pub const MIB_PER_GIB: f64 = 1_024.0;

/// Kibibytes per mebibyte for `/proc/meminfo` conversions.
pub const KIB_PER_MIB: u64 = 1_024;
