//! Performance sampling and formatting constants.

use std::time::Duration;

/// How often the TUI refreshes the performance panel.
pub const SAMPLE_INTERVAL: Duration = Duration::from_secs(1);

/// How long a cached `nvidia-smi` result is reused before re-querying.
///
/// `nvidia-smi` is a subprocess spawn; on non-Jetson systems with no
/// `tegrastats`, sampling it every frame would be wasteful.
pub const GPU_CACHE_TTL: Duration = Duration::from_secs(2);

/// `tegrastats` sampling interval in milliseconds.
pub const TEGRASTATS_INTERVAL_MILLIS: &str = "1000";

/// Milliwatts per watt, used to normalize `VDD_IN` readings.
pub const MILLIWATTS_PER_WATT: f64 = 1_000.0;

/// Mebibytes per gibibyte for human-readable RAM formatting.
pub const MIB_PER_GIB: f64 = 1_024.0;

/// Kibibytes per mebibyte for `/proc/meminfo` conversions.
pub const KIB_PER_MIB: u64 = 1_024;

/// Number of tick counts summed from the `cpu` line of `/proc/stat`: user,
/// nice, system, idle, iowait, irq, softirq and steal.
pub const CPU_TICK_FIELDS: usize = 8;

/// How often the performance panel asks a model server on another host for
/// its stats.
pub const SERVER_STATS_INTERVAL: Duration = Duration::from_secs(2);

/// How long one stats request may take before it counts as unpublished, so a
/// server that accepts the connection and never answers cannot stall polling.
pub const SERVER_STATS_TIMEOUT: Duration = Duration::from_secs(2);

/// Path, under the server root, of the JSON hardware report that
/// thor-tigress-agent publishes.
pub const SERVER_SYSTEM_PATH: &str = "/v1/system";

/// Path, under the server root, of the Prometheus metrics that vLLM, SGLang,
/// llama.cpp and TensorRT-LLM publish.
pub const SERVER_METRICS_PATH: &str = "/metrics";

/// Path of Ollama's list of the models loaded in memory.
pub const OLLAMA_PS_PATH: &str = "/api/ps";

/// Bytes per gibibyte, for the server's memory figures.
pub const BYTES_PER_GIB: f64 = 1_073_741_824.0;

/// The panel's only server line when the server answers none of the stats
/// requests.
pub const NO_SERVER_STATS: &str = "no server stats published";

/// The panel's only server line before the first poll has finished.
pub const WAITING_FOR_SERVER_STATS: &str = "waiting for server stats";

/// The heading above the local figures when the model server is on another
/// host.
pub const THIS_MACHINE_HEADING: &str = "this machine";
