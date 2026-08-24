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
use regex::Regex;
use std::process::Command;
use std::sync::OnceLock;

/// GPU-only fields parsed from either `tegrastats` or `nvidia-smi`.
#[derive(Clone, Debug, Default)]
pub(super) struct GpuStats {
    pub(super) name: Option<String>,
    pub(super) util_percent: Option<f64>,
    pub(super) memory_used_mb: Option<u64>,
    pub(super) memory_total_mb: Option<u64>,
    pub(super) power_watts: Option<f64>,
    pub(super) temp_c: Option<f64>,
}

pub(super) fn parse_tegrastats(line: &str) -> Option<GpuStats> {
    let patterns = tegra_patterns();

    let ram = patterns.ram.captures(line)?;
    let memory_used_mb = ram.get(1)?.as_str().parse().ok();
    let memory_total_mb = ram.get(2)?.as_str().parse().ok();

    let util_percent = patterns
        .gr3d
        .captures(line)
        .and_then(|captures| captures.get(1))
        .and_then(|value| value.as_str().parse().ok());

    let temp_c = patterns
        .gpu_temp
        .captures(line)
        .and_then(|captures| captures.get(1))
        .and_then(|value| value.as_str().parse().ok());

    let power_watts = patterns
        .vdd_in
        .captures(line)
        .and_then(|captures| captures.get(1))
        .and_then(|value| value.as_str().parse::<f64>().ok())
        .map(|milliwatts| milliwatts / MILLIWATTS_PER_WATT);

    Some(GpuStats {
        name: None,
        util_percent,
        memory_used_mb,
        memory_total_mb,
        power_watts,
        temp_c,
    })
}

pub(super) fn parse_nvidia_smi() -> Option<GpuStats> {
    let output = Command::new("nvidia-smi")
        .args([
            "--query-gpu=name,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }

    let text = String::from_utf8_lossy(&output.stdout);
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

struct TegrastatsPatterns {
    ram: Regex,
    gr3d: Regex,
    gpu_temp: Regex,
    vdd_in: Regex,
}

fn tegra_patterns() -> &'static TegrastatsPatterns {
    static PATTERNS: OnceLock<TegrastatsPatterns> = OnceLock::new();
    PATTERNS.get_or_init(|| TegrastatsPatterns {
        ram: Regex::new(r"RAM (\d+)/(\d+)MB").expect("static RAM regex is valid"),
        gr3d: Regex::new(r"GR3D_FREQ (\d+)%").expect("static GR3D regex is valid"),
        gpu_temp: Regex::new(r"gpu@([\d.]+)C").expect("static GPU temperature regex is valid"),
        vdd_in: Regex::new(r"VDD_IN (\d+)mW/").expect("static VDD_IN regex is valid"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_tegrastats_gpu_fields() {
        let line = "08-20-2026 20:27:02 RAM 9982/15656MB (lfb 55x4MB) SWAP 6301/16020MB (cached 45MB) CPU [8%@729,8%@729] GR3D_FREQ 0% cv0@51.812C cpu@55.906C gpu@54.437C tj@55.906C VDD_IN 7716mW/7716mW VDD_CPU_GPU_CV 908mW/908mW";
        let stats = parse_tegrastats(line).unwrap();
        assert_eq!(stats.memory_used_mb, Some(9982));
        assert_eq!(stats.memory_total_mb, Some(15656));
        assert_eq!(stats.util_percent, Some(0.0));
        assert_eq!(stats.temp_c, Some(54.437));
        assert_eq!(stats.power_watts, Some(7.716));
    }

    #[test]
    fn parses_float_na_as_missing() {
        assert!(parse_float("[N/A]").is_none());
        assert_eq!(parse_float("42.5"), Some(42.5));
    }

    #[test]
    fn parses_nvidia_smi_line() {
        let line = "NVIDIA GeForce RTX 4090, 100, 12345, 24564, 250.00, 65";
        let stats = parse_nvidia_smi_line(line).unwrap();
        assert_eq!(stats.name.as_deref(), Some("NVIDIA GeForce RTX 4090"));
        assert_eq!(stats.util_percent, Some(100.0));
        assert_eq!(stats.memory_used_mb, Some(12345));
        assert_eq!(stats.memory_total_mb, Some(24564));
        assert_eq!(stats.power_watts, Some(250.0));
        assert_eq!(stats.temp_c, Some(65.0));
    }

    #[test]
    fn nvidia_smi_line_with_na_fields_parses_partially() {
        let line = "Tesla T4, [N/A], 100, 15109, [N/A], [N/A]";
        let stats = parse_nvidia_smi_line(line).unwrap();
        assert_eq!(stats.util_percent, None);
        assert_eq!(stats.memory_used_mb, Some(100));
        assert_eq!(stats.power_watts, None);
        assert_eq!(stats.temp_c, None);
    }

    #[test]
    fn nvidia_smi_line_with_too_few_fields_is_none() {
        assert!(parse_nvidia_smi_line("GPU 0, 50").is_none());
    }
}
