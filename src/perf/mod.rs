//! Lightweight system/GPU performance sampling for the TUI task manager.
//!
//! On Jetson-class devices this reads live `tegrastats` output. On systems with
//! discrete NVIDIA GPUs it falls back to `nvidia-smi`. CPU and RAM are always
//! read from `/proc`.

mod gpu;

use crate::constants::perf::{
    CPU_TICK_FIELDS, GPU_CACHE_TTL, KIB_PER_MIB, SAMPLE_INTERVAL, TEGRASTATS_INTERVAL_MILLIS,
};
use gpu::{GpuStats, parse_nvidia_smi, parse_tegrastats};
use std::collections::HashMap;
use std::io::{BufRead, BufReader};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex, PoisonError};
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
    /// Previous sample's (idle_total, total) tick counts from `/proc/stat`.
    previous: Option<(u64, u64)>,
}

impl CpuSampler {
    /// Create a sampler that holds a first reading, so the first real sample
    /// already has a delta to measure.
    fn new() -> Self {
        Self {
            previous: read_cpu_times(),
        }
    }

    /// Take a new CPU utilization sample as a percentage (0–100).
    ///
    /// # Returns
    ///
    /// CPU utilization percentage, or `None` if `/proc/stat` is unreadable.
    fn sample(&mut self) -> Option<f64> {
        let current = read_cpu_times()?;
        let utilization = cpu_utilization(self.previous, current);

        self.previous = Some(current);

        utilization
    }
}

/// Busy percentage between two `(idle_total, total)` tick readings, or `None`
/// without an earlier reading.
fn cpu_utilization(previous: Option<(u64, u64)>, current: (u64, u64)) -> Option<f64> {
    let (previous_idle, previous_total) = previous?;
    let idle_delta = current.0.saturating_sub(previous_idle);
    let total_delta = current.1.saturating_sub(previous_total);

    busy_percent(idle_delta, total_delta)
}

/// The share of `total_delta` ticks that were not idle, as a percentage, or
/// `None` when no ticks passed.
fn busy_percent(idle_delta: u64, total_delta: u64) -> Option<f64> {
    if total_delta == 0 {
        return None;
    }

    Some(100.0 * (1.0 - idle_delta as f64 / total_delta as f64))
}

/// Monitors system stats and the latest `tegrastats` line.
pub struct PerfMonitor {
    /// Shared buffer containing the latest `tegrastats` output line.
    tegrastats: Arc<Mutex<Option<String>>>,
    /// Delta-based CPU utilization sampler.
    cpu: CpuSampler,
    /// Timestamp of the last completed sample, for rate-limiting.
    last_sample: Option<Instant>,
    /// Cached `nvidia-smi` result with its timestamp, to avoid frequent subprocess calls.
    gpu_cache: Option<(Instant, GpuStats)>,
}

impl PerfMonitor {
    /// Create a new performance monitor backed by the given `tegrastats` buffer.
    ///
    /// # Parameters
    ///
    /// - `tegrastats`: shared buffer for the background `tegrastats` reader.
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

    /// Collect a full system snapshot from tegrastats/nvidia-smi and /proc.
    fn sample(&mut self) -> SystemStats {
        let gpu = self.gpu_stats();
        let memory = read_memory_mb();

        SystemStats {
            gpu_name: gpu.name,
            gpu_util_percent: gpu.util_percent,
            gpu_memory_used_mb: gpu.memory_used_mb,
            gpu_memory_total_mb: gpu.memory_total_mb,
            gpu_power_watts: gpu.power_watts,
            gpu_temp_c: gpu.temp_c,
            cpu_util_percent: self.cpu.sample(),
            cpu_cores: logical_cores(),
            memory_used_mb: memory.used_mb,
            memory_total_mb: memory.total_mb,
            memory_shared_mb: memory.shared_mb,
            memory_buffers_mb: memory.buffers_mb,
            memory_cached_mb: memory.cached_mb,
            memory_free_mb: memory.free_mb,
        }
    }

    /// GPU metrics from the latest `tegrastats` line, else from `nvidia-smi`.
    fn gpu_stats(&mut self) -> GpuStats {
        let latest_line = self.latest_tegrastats_line();

        match latest_line.as_deref().and_then(parse_tegrastats) {
            Some(stats) => jetson_gpu_stats(stats),
            None => self.cached_nvidia_smi(),
        }
    }

    /// The newest `tegrastats` line, or `None` before the first one arrives.
    fn latest_tegrastats_line(&self) -> Option<String> {
        self.tegrastats
            .lock()
            .map(|guard| guard.clone())
            .unwrap_or_default()
    }

    /// Query `nvidia-smi`, reusing the last result within `GPU_CACHE_TTL`.
    fn cached_nvidia_smi(&mut self) -> GpuStats {
        let now = Instant::now();

        if let Some(cached) = self.fresh_gpu_cache(now) {
            return cached;
        }

        let fresh = parse_nvidia_smi().unwrap_or_default();
        self.gpu_cache = Some((now, fresh.clone()));

        fresh
    }

    /// The cached `nvidia-smi` result when it is younger than `GPU_CACHE_TTL`.
    fn fresh_gpu_cache(&self, now: Instant) -> Option<GpuStats> {
        let (cached_at, cached) = self.gpu_cache.as_ref()?;
        let is_fresh = now.duration_since(*cached_at) < GPU_CACHE_TTL;

        is_fresh.then(|| cached.clone())
    }
}

/// GPU metrics on Jetson. The `tegrastats` RAM figure is the shared system
/// memory, not a dedicated VRAM pool, so the GPU memory fields stay empty.
fn jetson_gpu_stats(stats: GpuStats) -> GpuStats {
    GpuStats {
        name: None,
        memory_used_mb: None,
        memory_total_mb: None,
        ..stats
    }
}

/// Number of logical CPU cores, or 1 when the count is unknown.
fn logical_cores() -> usize {
    std::thread::available_parallelism()
        .map(|count| count.get())
        .unwrap_or(1)
}

/// Spawn `tegrastats` in the background and keep its latest line in `shared`.
pub fn start_tegrastats(shared: Arc<Mutex<Option<String>>>) -> Option<Child> {
    let mut child = Command::new("tegrastats")
        .arg("--interval")
        .arg(TEGRASTATS_INTERVAL_MILLIS)
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .ok()?;
    let stdout = child.stdout.take()?;

    thread::spawn(move || keep_latest_line(BufReader::new(stdout), &shared));

    Some(child)
}

/// Store each line `reader` produces in `shared`, replacing the one before.
fn keep_latest_line(reader: impl BufRead, shared: &Mutex<Option<String>>) {
    for line in reader.lines().map_while(Result::ok) {
        store_line(shared, line);
    }
}

/// Replace the stored line. A poisoned lock reads as no line at all, so the
/// line is stored even then.
fn store_line(shared: &Mutex<Option<String>>, line: String) {
    let mut latest = shared.lock().unwrap_or_else(PoisonError::into_inner);

    *latest = Some(line);
}

/// Kills the background `tegrastats` process when dropped.
pub struct TegrastatsGuard(Option<Child>);

impl TegrastatsGuard {
    /// Spawn `tegrastats` and return a guard that kills it on drop.
    ///
    /// # Parameters
    ///
    /// - `shared`: buffer where the latest `tegrastats` line is stored.
    pub fn start(shared: Arc<Mutex<Option<String>>>) -> Self {
        Self(start_tegrastats(shared))
    }
}

impl Drop for TegrastatsGuard {
    fn drop(&mut self) {
        let Some(child) = &mut self.0 else {
            return;
        };

        stop_child(child);
    }
}

/// Kill `child` and wait for it to exit.
fn stop_child(child: &mut Child) {
    // An error here means the process has already exited; nothing is left to stop.
    child.kill().ok();
    child.wait().ok();
}

/// System memory breakdown matching jtop's RAM page.
struct MemoryStats {
    /// Green "used" memory in MiB (excludes shared GPU memory).
    used_mb: u64,
    /// Total physical memory in MiB.
    total_mb: u64,
    /// GPU-shared memory in MiB (NvMapMemUsed on Jetson).
    shared_mb: u64,
    /// Kernel buffers in MiB.
    buffers_mb: u64,
    /// Page cache + SReclaimable in MiB.
    cached_mb: u64,
    /// Completely unused memory in MiB.
    free_mb: u64,
}

/// Read `/proc/meminfo` into a structured memory breakdown.
fn read_memory_mb() -> MemoryStats {
    let meminfo = std::fs::read_to_string("/proc/meminfo").unwrap_or_default();

    memory_stats(&meminfo)
}

/// The memory breakdown from the text of `/proc/meminfo`.
///
/// "Used" matches jtop: `Total - Free - Buffers - Cached - SReclaimable`,
/// minus the memory shared with the GPU.
fn memory_stats(meminfo: &str) -> MemoryStats {
    let fields = meminfo_fields(meminfo);
    let field = |name: &str| fields.get(name).copied().unwrap_or(0);

    let total_kb = field("MemTotal");
    let free_kb = field("MemFree");
    let buffers_kb = field("Buffers");
    let cached_kb = field("Cached") + field("SReclaimable");
    let shared_kb = field("NvMapMemUsed");

    let used_kb = total_kb.saturating_sub(free_kb + buffers_kb + cached_kb);

    MemoryStats {
        used_mb: used_kb.saturating_sub(shared_kb) / KIB_PER_MIB,
        total_mb: total_kb / KIB_PER_MIB,
        shared_mb: shared_kb / KIB_PER_MIB,
        buffers_mb: buffers_kb / KIB_PER_MIB,
        cached_mb: cached_kb / KIB_PER_MIB,
        free_mb: free_kb / KIB_PER_MIB,
    }
}

/// Every `Key: value kB` line of `/proc/meminfo`, as kibibytes by key.
fn meminfo_fields(meminfo: &str) -> HashMap<&str, u64> {
    meminfo
        .lines()
        .filter_map(|line| line.split_once(':'))
        .map(|(key, value)| (key.trim(), parse_kibibytes(value)))
        .collect()
}

/// Parse a numeric value followed by a unit from a `/proc` line (e.g. "12345 kB").
fn parse_kibibytes(rest: &str) -> u64 {
    rest.split_whitespace()
        .next()
        .and_then(|value| value.parse().ok())
        .unwrap_or(0)
}

/// Read aggregate CPU tick counts from `/proc/stat`.
///
/// # Returns
///
/// `(idle_total, total)` tick counts, or `None` if the file is unreadable.
fn read_cpu_times() -> Option<(u64, u64)> {
    let stat = std::fs::read_to_string("/proc/stat").ok()?;

    cpu_times(&stat)
}

/// `(idle_total, total)` from the aggregate `cpu` line, the first line of
/// `/proc/stat`, or `None` when it lacks eight numeric tick counts.
fn cpu_times(stat: &str) -> Option<(u64, u64)> {
    let first_line = stat.lines().next().unwrap_or_default();
    let ticks: Option<Vec<u64>> = first_line
        .split_whitespace()
        .skip(1)
        .take(CPU_TICK_FIELDS)
        .map(|field| field.parse().ok())
        .collect();

    tick_totals(&ticks?)
}

/// `(idle + iowait, sum of all eight)` from the ticks user, nice, system,
/// idle, iowait, irq, softirq and steal.
fn tick_totals(ticks: &[u64]) -> Option<(u64, u64)> {
    let [user, nice, system, idle, iowait, irq, softirq, steal] = ticks else {
        return None;
    };

    let idle_total = idle + iowait;
    let total = user + nice + system + idle + iowait + irq + softirq + steal;

    Some((idle_total, total))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    /// A `/proc/meminfo` excerpt with every field the breakdown reads.
    const MEMINFO: &str = "MemTotal: 16384 kB\nMemFree: 2048 kB\nBuffers: 1024 kB\n\
        Cached: 3072 kB\nSReclaimable: 1024 kB\nNvMapMemUsed: 2048 kB\nno colon here\n";

    #[test]
    fn parses_kibibytes_value() {
        assert_eq!(parse_kibibytes("12345 kB"), 12_345);
        assert_eq!(parse_kibibytes("0 kB"), 0);
        assert_eq!(parse_kibibytes("garbage"), 0);
    }

    #[test]
    fn memory_breakdown_matches_jtop() {
        let memory = memory_stats(MEMINFO);

        assert_eq!(memory.total_mb, 16);
        assert_eq!(memory.free_mb, 2);
        assert_eq!(memory.buffers_mb, 1);
        assert_eq!(memory.cached_mb, 4);
        assert_eq!(memory.shared_mb, 2);
        assert_eq!(memory.used_mb, 7);
    }

    #[test]
    fn cpu_times_sum_the_first_eight_ticks() {
        let stat = "cpu  1 2 3 4 5 6 7 8 9 10\ncpu0 1 1 1 1 1 1 1 1\n";

        assert_eq!(cpu_times(stat), Some((9, 36)));
    }

    #[test]
    fn cpu_times_need_eight_numbers() {
        assert_eq!(cpu_times(""), None);
        assert_eq!(cpu_times("cpu 1 2 3"), None);
        assert_eq!(cpu_times("cpu 1 2 x 4 5 6 7 8"), None);
    }

    #[test]
    fn cpu_utilization_is_the_busy_share_of_new_ticks() {
        assert_eq!(cpu_utilization(None, (10, 100)), None);
        assert_eq!(cpu_utilization(Some((10, 100)), (10, 100)), None);
        assert_eq!(cpu_utilization(Some((10, 100)), (35, 200)), Some(75.0));
    }

    // `/proc` exists only on Linux. These two read the real kernel files, so
    // they are meaningless (and fail) on macOS and Windows builds.
    #[cfg(target_os = "linux")]
    #[test]
    fn reads_memory_from_proc_meminfo() {
        let memory = read_memory_mb();

        assert!(memory.total_mb > 0);
        assert!(memory.free_mb <= memory.total_mb);
        assert!(memory.used_mb <= memory.total_mb);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn cpu_sampler_reads_proc_stat() {
        let mut sampler = CpuSampler::new();

        assert!(sampler.previous.is_some());

        // The share is `None` when no ticks passed between the two readings.
        let utilization = sampler.sample().unwrap_or(0.0);

        assert!((0.0..=100.0).contains(&utilization));
    }

    #[test]
    fn perf_monitor_samples_when_due_and_skips_early_samples() {
        let shared = Arc::new(Mutex::new(Some(
            "RAM 100/1000MB GR3D_FREQ 10% gpu@40C VDD_IN 5000mW/".to_string(),
        )));
        let mut monitor = PerfMonitor::new(shared);
        let now = Instant::now();

        let first = monitor.sample_if_due(now);
        let not_due = monitor.sample_if_due(now + Duration::from_millis(1));
        let stats = first.unwrap_or_default();

        assert_eq!(stats.gpu_util_percent, Some(10.0));
        assert_eq!(stats.gpu_temp_c, Some(40.0));
        assert_eq!(stats.gpu_power_watts, Some(5.0));
        assert_eq!(stats.gpu_memory_total_mb, None);
        assert!(not_due.is_none());
    }

    #[test]
    fn nvidia_smi_result_is_reused_within_the_cache_time() {
        let mut monitor = PerfMonitor::new(Arc::new(Mutex::new(None)));

        let first = monitor.gpu_stats();
        let second = monitor.gpu_stats();

        assert_eq!(first.name, second.name);
    }

    #[test]
    fn latest_line_replaces_the_one_before() {
        let shared = Mutex::new(None);

        keep_latest_line("first\nsecond\n".as_bytes(), &shared);

        let latest = shared.lock().map(|guard| guard.clone()).unwrap_or_default();

        assert_eq!(latest.as_deref(), Some("second"));
    }

    #[test]
    fn tegrastats_guard_handles_missing_binary() {
        let shared = Arc::new(Mutex::new(None));
        let guard = TegrastatsGuard::start(shared);

        drop(guard);
    }

    #[test]
    fn guard_without_a_process_does_nothing() {
        drop(TegrastatsGuard(None));
    }

    #[test]
    fn guard_stops_its_process() -> anyhow::Result<()> {
        let child = Command::new("sleep").arg("30").spawn()?;

        drop(TegrastatsGuard(Some(child)));

        Ok(())
    }
}
