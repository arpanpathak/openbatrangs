//! # GPU metric parsing from `tegrastats` (Jetson) and `nvidia-smi`
//!
//! The performance panel needs GPU utilization, memory, power, and temperature.
//! On Jetson devices that data comes from `tegrastats`; on desktop systems it
//! comes from `nvidia-smi`. Parsing is kept pure (a string goes in, a struct
//! comes out) so every branch is testable without hardware.
//!
//! ## References
//!
//! - NVIDIA `nvidia-smi` query format: <https://developer.nvidia.com/nvidia-system-management-interface>
//! - Jetson `tegrastats`: <https://docs.nvidia.com/jetson/archives/r36.2.0/developer-guide/sd/ApplicationNotes/Tegrastats.html>

use crate::constants::perf::MILLIWATTS_PER_WATT;
use regex::{Captures, Regex};
use std::process::Command;
use std::sync::OnceLock;

/// GPU-only fields parsed from either `tegrastats` or `nvidia-smi`.
#[derive(Clone, Debug, Default)]
pub(super) struct GpuStats {
    /// GPU model name (e.g. "NVIDIA GeForce RTX 4090"), or `None` for Jetson.
    pub(super) name: Option<String>,
    /// GPU utilization percentage (0–100), or `None` if unavailable.
    pub(super) util_percent: Option<f64>,
    /// Used GPU memory in MiB, or `None` for Jetson (shared memory).
    pub(super) memory_used_mb: Option<u64>,
    /// Total GPU memory in MiB, or `None` for Jetson.
    pub(super) memory_total_mb: Option<u64>,
    /// GPU power draw in watts, or `None` if unavailable.
    pub(super) power_watts: Option<f64>,
    /// GPU temperature in Celsius, or `None` if unavailable.
    pub(super) temp_c: Option<f64>,
}

/// Parse GPU metrics from a single `tegrastats` output line.
///
/// # Parameters
///
/// - `line`: raw tegrastats line containing RAM, GR3D_FREQ, gpu@, VDD_IN fields.
///
/// # Returns
///
/// Parsed [`GpuStats`], or `None` if the RAM pattern does not match.
pub(super) fn parse_tegrastats(line: &str) -> Option<GpuStats> {
    let patterns = tegra_patterns()?;
    let ram = patterns.ram.captures(line)?;
    let power_milliwatts = first_number(&patterns.vdd_in, line);

    Some(GpuStats {
        name: None,
        util_percent: first_number(&patterns.gr3d, line),
        memory_used_mb: captured_integer(&ram, 1),
        memory_total_mb: captured_integer(&ram, 2),
        power_watts: power_milliwatts.map(|milliwatts| milliwatts / MILLIWATTS_PER_WATT),
        temp_c: first_number(&patterns.gpu_temp, line),
    })
}

/// The first capture group of `pattern` in `line`, as a number.
fn first_number(pattern: &Regex, line: &str) -> Option<f64> {
    pattern
        .captures(line)
        .and_then(|captures| captures.get(1))
        .and_then(|value| value.as_str().parse().ok())
}

/// Capture group `index` as a whole number.
fn captured_integer(captures: &Captures, index: usize) -> Option<u64> {
    captures
        .get(index)
        .and_then(|value| value.as_str().parse().ok())
}

/// Query GPU metrics by running `nvidia-smi` as a subprocess.
///
/// # Returns
///
/// Parsed [`GpuStats`], or `None` if `nvidia-smi` is not available or fails.
pub(super) fn parse_nvidia_smi() -> Option<GpuStats> {
    let output = Command::new("nvidia-smi")
        .args([
            "--query-gpu=name,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()?;

    stats_from_nvidia_smi(output.status.success(), &output.stdout)
}

/// GPU metrics from the output of a finished `nvidia-smi`, or `None` when it
/// failed.
fn stats_from_nvidia_smi(succeeded: bool, stdout: &[u8]) -> Option<GpuStats> {
    if !succeeded {
        return None;
    }

    first_line_stats(&String::from_utf8_lossy(stdout))
}

/// GPU metrics from the first line of `nvidia-smi` output, for the first GPU.
fn first_line_stats(text: &str) -> Option<GpuStats> {
    parse_nvidia_smi_line(text.lines().next()?)
}

/// Parse one `nvidia-smi --format=csv,noheader,nounits` line.
///
/// # Parameters
///
/// - `line`: raw CSV line, e.g.
///   `NVIDIA GeForce RTX 4090, 100, 12345, 24564, 250.00, 65`.
///
/// # Returns
///
/// Parsed [`GpuStats`], or `None` when the line has fewer than six fields.
fn parse_nvidia_smi_line(line: &str) -> Option<GpuStats> {
    let fields: Vec<&str> = line.split(',').map(str::trim).collect();

    if fields.len() < 6 {
        return None;
    }

    Some(GpuStats {
        name: Some(fields[0].to_string()),
        util_percent: parse_float(fields[1]),
        memory_used_mb: fields[2].parse().ok(),
        memory_total_mb: fields[3].parse().ok(),
        power_watts: parse_float(fields[4]),
        temp_c: parse_float(fields[5]),
    })
}

/// Parse a numeric CSV field, treating `[N/A]` as missing.
fn parse_float(field: &str) -> Option<f64> {
    if field.contains("[N/A]") {
        return None;
    }
    field.parse().ok()
}

/// Compiled regex patterns for parsing `tegrastats` output lines.
struct TegrastatsPatterns {
    /// Matches `RAM used/totalMB` for system memory usage.
    ram: Regex,
    /// Matches `GR3D_FREQ percent%` for GPU utilization.
    gr3d: Regex,
    /// Matches `gpu@tempC` for GPU temperature.
    gpu_temp: Regex,
    /// Matches `VDD_IN milliwattsmW/` for board power draw.
    vdd_in: Regex,
}

/// The compiled `tegrastats` patterns, built once per process.
fn tegra_patterns() -> Option<&'static TegrastatsPatterns> {
    static PATTERNS: OnceLock<Option<TegrastatsPatterns>> = OnceLock::new();

    PATTERNS
        .get_or_init(|| compile_tegra_patterns().ok())
        .as_ref()
}

/// Compile the `tegrastats` patterns.
fn compile_tegra_patterns() -> Result<TegrastatsPatterns, regex::Error> {
    Ok(TegrastatsPatterns {
        ram: Regex::new(r"RAM (\d+)/(\d+)MB")?,
        gr3d: Regex::new(r"GR3D_FREQ (\d+)%")?,
        gpu_temp: Regex::new(r"gpu@([\d.]+)C")?,
        vdd_in: Regex::new(r"VDD_IN (\d+)mW/")?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::{Context, Result};

    #[test]
    fn parses_tegrastats_gpu_fields() -> Result<()> {
        let line = "08-20-2026 20:27:02 RAM 9982/15656MB (lfb 55x4MB) SWAP 6301/16020MB (cached 45MB) CPU [8%@729,8%@729] GR3D_FREQ 0% cv0@51.812C cpu@55.906C gpu@54.437C tj@55.906C VDD_IN 7716mW/7716mW VDD_CPU_GPU_CV 908mW/908mW";
        let stats = parse_tegrastats(line).context("the line should parse")?;

        assert_eq!(stats.memory_used_mb, Some(9982));
        assert_eq!(stats.memory_total_mb, Some(15656));
        assert_eq!(stats.util_percent, Some(0.0));
        assert_eq!(stats.temp_c, Some(54.437));
        assert_eq!(stats.power_watts, Some(7.716));
        Ok(())
    }

    #[test]
    fn parses_float_na_as_missing() {
        assert!(parse_float("[N/A]").is_none());
        assert_eq!(parse_float("42.5"), Some(42.5));
    }

    #[test]
    fn parses_nvidia_smi_line() -> Result<()> {
        let line = "NVIDIA GeForce RTX 4090, 100, 12345, 24564, 250.00, 65";
        let stats = parse_nvidia_smi_line(line).context("the line should parse")?;

        assert_eq!(stats.name.as_deref(), Some("NVIDIA GeForce RTX 4090"));
        assert_eq!(stats.util_percent, Some(100.0));
        assert_eq!(stats.memory_used_mb, Some(12345));
        assert_eq!(stats.memory_total_mb, Some(24564));
        assert_eq!(stats.power_watts, Some(250.0));
        assert_eq!(stats.temp_c, Some(65.0));
        Ok(())
    }

    #[test]
    fn nvidia_smi_line_with_na_fields_parses_partially() -> Result<()> {
        let line = "Tesla T4, [N/A], 100, 15109, [N/A], [N/A]";
        let stats = parse_nvidia_smi_line(line).context("the line should parse")?;

        assert_eq!(stats.util_percent, None);
        assert_eq!(stats.memory_used_mb, Some(100));
        assert_eq!(stats.power_watts, None);
        assert_eq!(stats.temp_c, None);
        Ok(())
    }

    #[test]
    fn nvidia_smi_line_with_too_few_fields_is_none() {
        assert!(parse_nvidia_smi_line("GPU 0, 50").is_none());
    }

    #[test]
    fn parse_tegrastats_returns_none_for_empty_string() {
        assert!(parse_tegrastats("").is_none());
    }

    #[test]
    fn parse_tegrastats_returns_none_for_random_text() {
        assert!(parse_tegrastats("this is not tegrastats output").is_none());
    }

    #[test]
    fn parse_tegrastats_with_partial_fields() -> Result<()> {
        // Only RAM field present (no GR3D_FREQ, no gpu@, no VDD_IN)
        let line = "RAM 4096/8192MB";
        let stats = parse_tegrastats(line).context("the line should parse")?;

        assert_eq!(stats.memory_used_mb, Some(4096));
        assert_eq!(stats.memory_total_mb, Some(8192));
        assert_eq!(stats.util_percent, None);
        assert_eq!(stats.temp_c, None);
        assert_eq!(stats.power_watts, None);
        Ok(())
    }

    #[test]
    fn parse_tegrastats_without_ram_returns_none() {
        // GR3D_FREQ present but no RAM — should return None because RAM is required
        let line = "GR3D_FREQ 50% gpu@45.0C VDD_IN 5000mW/5000mW";

        assert!(parse_tegrastats(line).is_none());
    }

    #[test]
    fn parse_tegrastats_with_high_gpu_utilization() -> Result<()> {
        let line = "RAM 2000/16000MB GR3D_FREQ 99% gpu@80.5C VDD_IN 15000mW/15000mW";
        let stats = parse_tegrastats(line).context("the line should parse")?;

        assert_eq!(stats.memory_used_mb, Some(2000));
        assert_eq!(stats.memory_total_mb, Some(16000));
        assert_eq!(stats.util_percent, Some(99.0));
        assert_eq!(stats.temp_c, Some(80.5));
        assert_eq!(stats.power_watts, Some(15.0));
        Ok(())
    }

    #[test]
    fn parse_nvidia_smi_line_with_extra_fields_parses_first_six() -> Result<()> {
        // nvidia-smi may return extra trailing fields; we only care about the first 6
        let line = "RTX 3090, 75, 8000, 24576, 300.50, 70, extra_field, another";
        let stats = parse_nvidia_smi_line(line).context("the line should parse")?;

        assert_eq!(stats.name.as_deref(), Some("RTX 3090"));
        assert_eq!(stats.util_percent, Some(75.0));
        assert_eq!(stats.memory_used_mb, Some(8000));
        assert_eq!(stats.memory_total_mb, Some(24576));
        assert_eq!(stats.power_watts, Some(300.5));
        assert_eq!(stats.temp_c, Some(70.0));
        Ok(())
    }

    #[test]
    fn parse_nvidia_smi_line_with_zero_values() -> Result<()> {
        let line = "GPU, 0, 0, 24576, 0.00, 0";
        let stats = parse_nvidia_smi_line(line).context("the line should parse")?;

        assert_eq!(stats.util_percent, Some(0.0));
        assert_eq!(stats.memory_used_mb, Some(0));
        assert_eq!(stats.power_watts, Some(0.0));
        assert_eq!(stats.temp_c, Some(0.0));
        Ok(())
    }

    #[test]
    fn parse_float_handles_empty_string() {
        assert!(parse_float("").is_none());
    }

    #[test]
    fn nvidia_smi_output_is_read_only_after_success() {
        let stdout = b"RTX 3090, 75, 8000, 24576, 300.50, 70\n";

        assert!(stats_from_nvidia_smi(false, stdout).is_none());
        assert!(stats_from_nvidia_smi(true, b"").is_none());
        assert!(stats_from_nvidia_smi(true, stdout).is_some());
    }

    #[test]
    fn nvidia_smi_query_never_panics() {
        // The tool is missing on most machines; when it runs, it names the GPU.
        let stats = parse_nvidia_smi();

        assert!(stats.is_none_or(|stats| stats.name.is_some()));
    }

    #[test]
    fn parse_float_handles_negative_values() {
        // Negative values aren't expected but should parse
        assert_eq!(parse_float("-5.0"), Some(-5.0));
    }
}
