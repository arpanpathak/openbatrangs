//! Lightweight system/GPU performance sampling for the TUI task manager.
//!
//! On Jetson-class devices this reads live `tegrastats` output. On systems with
//! discrete NVIDIA GPUs it falls back to `nvidia-smi`. CPU and RAM are always
//! read from `/proc`.

mod gpu;

use crate::constants::perf::{
    GPU_CACHE_TTL, KIB_PER_MIB, SAMPLE_INTERVAL, TEGRASTATS_INTERVAL_MILLIS,
};
use gpu::{parse_nvidia_smi, parse_tegrastats, GpuStats};
use std::io::{BufRead, BufReader};
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Instant;

/// A point-in-time snapshot of system and GPU health.
#[derive(Clone, Debug, Default)]
pub struct SystemStats {
    /// GPU model name, when known.
    pub gpu_name: Option<String>,
    /// GPU utilization percentage.
    pub gpu_util_percent: Option<f64>,
    /// Used GPU memory in MiB.
    pub gpu_memory_used_mb: Option<u64>,
    /// Total GPU memory in MiB.
    pub gpu_memory_total_mb: Option<u64>,
    /// GPU power draw in watts.
    pub gpu_power_watts: Option<f64>,
    /// GPU temperature in Celsius.
    pub gpu_temp_c: Option<f64>,
    /// Overall CPU utilization percentage.
    pub cpu_util_percent: Option<f64>,
    /// Number of logical CPU cores.
    pub cpu_cores: usize,
    /// Green "used" system memory in MiB (matches jtop, excludes shared GPU).
    pub memory_used_mb: u64,
    /// Total system memory in MiB.
    pub memory_total_mb: u64,
    /// GPU-shared memory in MiB (NvMapMemUsed on Jetson).
    pub memory_shared_mb: u64,
    /// Buffers in MiB.
    pub memory_buffers_mb: u64,
    /// Cached (+ SReclaimable) in MiB.
    pub memory_cached_mb: u64,
    /// Free memory in MiB.
    pub memory_free_mb: u64,
}

/// Tracks the previous `/proc/stat` sample so CPU utilization is a delta.
struct CpuSampler {
    previous: Option<(u64, u64)>,
}

impl CpuSampler {
    fn new() -> Self {
        let mut sampler = Self { previous: None };
        // Prime the delta so the first visible sample is meaningful.
        let _ = sampler.sample();
        sampler
    }

    fn sample(&mut self) -> Option<f64> {
        let current = read_cpu_times()?;
        let utilization = match self.previous {
            Some((previous_idle, previous_total)) => {
                let idle_delta = current.0.saturating_sub(previous_idle);
                let total_delta = current.1.saturating_sub(previous_total);
                if total_delta == 0 {
                    None
                } else {
                    Some(100.0 * (1.0 - idle_delta as f64 / total_delta as f64))
                }
            }
            None => None,
        };
        self.previous = Some(current);
        utilization
    }
}

/// Monitors system stats and the latest `tegrastats` line.
pub struct PerfMonitor {
    tegrastats: Arc<Mutex<Option<String>>>,
    cpu: CpuSampler,
    last_sample: Option<Instant>,
    gpu_cache: Option<(Instant, GpuStats)>,
}

impl PerfMonitor {
    pub fn new(tegrastats: Arc<Mutex<Option<String>>>) -> Self {
        Self {
            tegrastats,
            cpu: CpuSampler::new(),
            last_sample: None,
            gpu_cache: None,
        }
    }

    /// Sample at most once per `SAMPLE_INTERVAL`.
    pub fn sample_if_due(&mut self, now: Instant) -> Option<SystemStats> {
        let is_due = self
            .last_sample
            .map(|last| now.duration_since(last) >= SAMPLE_INTERVAL)
            .unwrap_or(true);
        if !is_due {
            return None;
        }
        self.last_sample = Some(now);
        Some(self.sample())
    }

    fn sample(&mut self) -> SystemStats {
        let latest_tegrastats = self
            .tegrastats
            .lock()
            .map(|guard| guard.clone())
            .unwrap_or_default();
        let tegrastats_stats = latest_tegrastats.as_deref().and_then(parse_tegrastats);
        let gpu = match &tegrastats_stats {
            // On Jetson, tegrastats RAM is the shared system memory, not a
            // dedicated VRAM pool; leave GPU memory fields for nvidia-smi only.
            Some(stats) => GpuStats {
                name: None,
                util_percent: stats.util_percent,
                memory_used_mb: None,
                memory_total_mb: None,
                power_watts: stats.power_watts,
                temp_c: stats.temp_c,
            },
            None => self.cached_nvidia_smi(),
        };
        // Match jtop: green used = (Total - Free - Buffers - Cached) - shared.
        let memory = read_memory_mb();

        SystemStats {
            gpu_name: gpu.name,
            gpu_util_percent: gpu.util_percent,
            gpu_memory_used_mb: gpu.memory_used_mb,
            gpu_memory_total_mb: gpu.memory_total_mb,
            gpu_power_watts: gpu.power_watts,
            gpu_temp_c: gpu.temp_c,
            cpu_util_percent: self.cpu.sample(),
            cpu_cores: std::thread::available_parallelism()
                .map(|count| count.get())
                .unwrap_or(0),
            memory_used_mb: memory.used_mb,
            memory_total_mb: memory.total_mb,
            memory_shared_mb: memory.shared_mb,
            memory_buffers_mb: memory.buffers_mb,
            memory_cached_mb: memory.cached_mb,
            memory_free_mb: memory.free_mb,
        }
    }

    /// Query `nvidia-smi`, reusing the last result within `GPU_CACHE_TTL`.
    fn cached_nvidia_smi(&mut self) -> GpuStats {
        let now = Instant::now();
        if let Some((cached_at, cached)) = &self.gpu_cache {
            if now.duration_since(*cached_at) < GPU_CACHE_TTL {
                return cached.clone();
            }
        }
        let fresh = parse_nvidia_smi().unwrap_or_default();
        self.gpu_cache = Some((now, fresh.clone()));
        fresh
    }
}

/// Spawn `tegrastats` in the background and keep its latest line in `shared`.
pub fn start_tegrastats(shared: Arc<Mutex<Option<String>>>) -> Option<std::process::Child> {
    let mut child = Command::new("tegrastats")
        .arg("--interval")
        .arg(TEGRASTATS_INTERVAL_MILLIS)
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .ok()?;
    let stdout = child.stdout.take()?;
    thread::spawn(move || {
        let reader = BufReader::new(stdout);
        for line in reader.lines().map_while(Result::ok) {
            if let Ok(mut guard) = shared.lock() {
                *guard = Some(line);
            }
        }
    });
    Some(child)
}

/// Kills the background `tegrastats` process when dropped.
pub struct TegrastatsGuard(Option<std::process::Child>);

impl TegrastatsGuard {
    pub fn start(shared: Arc<Mutex<Option<String>>>) -> Self {
        Self(start_tegrastats(shared))
    }
}

impl Drop for TegrastatsGuard {
    fn drop(&mut self) {
        if let Some(child) = &mut self.0 {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

/// System memory breakdown matching jtop's RAM page.
struct MemoryStats {
    used_mb: u64,
    total_mb: u64,
    shared_mb: u64,
    buffers_mb: u64,
    cached_mb: u64,
    free_mb: u64,
}

fn read_memory_mb() -> MemoryStats {
    let content = std::fs::read_to_string("/proc/meminfo").unwrap_or_default();
    let mut total_kb = 0u64;
    let mut free_kb = 0u64;
    let mut buffers_kb = 0u64;
    let mut cached_kb = 0u64;
    let mut reclaimable_kb = 0u64;
    let mut shared_kb = 0u64;
    for line in content.lines() {
        let Some((key, value)) = line.split_once(':') else {
            continue;
        };
        match key.trim() {
            "MemTotal" => total_kb = parse_kibibytes(value),
            "MemFree" => free_kb = parse_kibibytes(value),
            "Buffers" => buffers_kb = parse_kibibytes(value),
            "Cached" => cached_kb = parse_kibibytes(value),
            "SReclaimable" => reclaimable_kb = parse_kibibytes(value),
            "NvMapMemUsed" => shared_kb = parse_kibibytes(value),
            _ => {}
        }
    }
    let total_mb = total_kb / KIB_PER_MIB;
    let free_mb = free_kb / KIB_PER_MIB;
    let buffers_mb = buffers_kb / KIB_PER_MIB;
    let cached_mb = (cached_kb + reclaimable_kb) / KIB_PER_MIB;
    let shared_mb = shared_kb / KIB_PER_MIB;
    let used_kb = total_kb.saturating_sub(free_kb + buffers_kb + cached_kb + reclaimable_kb);
    let used_mb = used_kb.saturating_sub(shared_kb) / KIB_PER_MIB;
    MemoryStats {
        used_mb,
        total_mb,
        shared_mb,
        buffers_mb,
        cached_mb,
        free_mb,
    }
}

fn parse_kibibytes(rest: &str) -> u64 {
    rest.split_whitespace()
        .next()
        .and_then(|value| value.parse().ok())
        .unwrap_or(0)
}

fn read_cpu_times() -> Option<(u64, u64)> {
    let content = std::fs::read_to_string("/proc/stat").ok()?;
    let line = content.lines().next()?;
    let mut fields = line.split_whitespace().skip(1);
    let user: u64 = fields.next()?.parse().ok()?;
    let nice: u64 = fields.next()?.parse().ok()?;
    let system: u64 = fields.next()?.parse().ok()?;
    let idle: u64 = fields.next()?.parse().ok()?;
    let iowait: u64 = fields.next()?.parse().ok()?;
    let irq: u64 = fields.next()?.parse().ok()?;
    let softirq: u64 = fields.next()?.parse().ok()?;
    let steal: u64 = fields.next()?.parse().ok()?;

    let idle_total = idle + iowait;
    let total = user + nice + system + idle + iowait + irq + softirq + steal;
    Some((idle_total, total))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_kibibytes_value() {
        assert_eq!(parse_kibibytes("12345 kB"), 12_345);
        assert_eq!(parse_kibibytes("0 kB"), 0);
        assert_eq!(parse_kibibytes("garbage"), 0);
    }

    #[test]
    fn reads_memory_from_proc_meminfo() {
        let memory = read_memory_mb();
        assert!(memory.total_mb > 0);
        assert!(memory.free_mb <= memory.total_mb);
        assert!(memory.used_mb <= memory.total_mb);
    }

    #[test]
    fn reads_cpu_times_from_proc_stat() {
        let times = read_cpu_times();
        assert!(times.is_some());
        let (idle, total) = times.unwrap();
        assert!(idle <= total);
        assert!(total > 0);
    }

    #[test]
    fn cpu_sampler_primes_without_panicking() {
        let mut sampler = CpuSampler::new();
        // The second read can legitimately be `None` when `/proc/stat` counters
        // did not advance between samples; the point of this test is that
        // sampling never panics.
        let _ = sampler.sample();
        let _ = sampler.sample();
    }
}
