//! # Stats published by a model server on another host
//!
//! When the model server runs on another machine, the local `nvidia-smi`,
//! `tegrastats` and `/proc` figures describe the client, not the GPU that
//! serves the model. This module reads what the server publishes about
//! itself:
//!
//! - `GET /v1/system`: a JSON hardware report (GPU, memory, CPU), from
//!   servers such as thor-tigress-agent.
//! - `GET /metrics`: Prometheus text from vLLM, SGLang and llama.cpp, with the
//!   requests running and waiting, KV-cache use and generation throughput.
//!   TensorRT-LLM answers `/metrics` with JSON, which is ignored.
//! - `GET /api/ps`: the models Ollama holds in memory and their GPU memory.
//!
//! Each parser takes the response text and returns `None` for text it does
//! not recognize, so every case is testable with fixed input.
//!
//! ## References
//!
//! - Prometheus text format: <https://prometheus.io/docs/instrumenting/exposition_formats/>
//! - vLLM metrics: <https://docs.vllm.ai/en/latest/design/metrics.html>
//! - SGLang metrics: <https://docs.sglang.ai/references/production_metrics.html>
//! - llama.cpp server metrics: <https://github.com/ggml-org/llama.cpp/tree/master/tools/server>
//! - Ollama `/api/ps`: <https://github.com/ollama/ollama/blob/main/docs/api.md#list-running-models>

use crate::constants::perf::{BYTES_PER_GIB, NO_SERVER_STATS, WAITING_FOR_SERVER_STATS};
use serde::Deserialize;
use std::net::IpAddr;

/// Everything the model server published in one poll. A field is `None` when
/// its request failed or its answer was not recognized.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ServerStats {
    /// The hardware report from `/v1/system`.
    pub system: Option<ServerSystem>,
    /// The engine load from Prometheus `/metrics`.
    pub engine_load: Option<EngineLoad>,
    /// The models Ollama holds in memory, from `/api/ps`.
    pub loaded_models: Option<Vec<LoadedModel>>,
}

impl ServerStats {
    /// Parse the bodies of the three stats requests; `None` stands for a
    /// request that failed.
    pub fn from_bodies(
        system: Option<&str>,
        metrics: Option<&str>,
        processes: Option<&str>,
    ) -> Self {
        Self {
            system: system.and_then(server_system),
            engine_load: metrics.and_then(engine_load),
            loaded_models: processes.and_then(loaded_models),
        }
    }
}

/// The JSON body of `GET /v1/system`. Every figure may be `null`.
#[derive(Clone, Debug, Default, Deserialize, PartialEq)]
pub struct ServerSystem {
    /// The server's host name, such as `thor`.
    pub host: Option<String>,
    /// The GPU that serves the model.
    pub gpu: Option<ServerGpu>,
    /// The server's CPU.
    pub cpu: Option<ServerCpu>,
}

/// The `gpu` object of `/v1/system`.
#[derive(Clone, Debug, Default, Deserialize, PartialEq)]
pub struct ServerGpu {
    /// The GPU's name, such as `NVIDIA Thor`.
    pub name: Option<String>,
    /// GPU utilization, 0 to 100.
    pub utilization_percent: Option<f64>,
    /// GPU memory in use, in bytes.
    pub memory_used_bytes: Option<u64>,
    /// All GPU memory, in bytes.
    pub memory_total_bytes: Option<u64>,
    /// Whether the GPU shares its memory with the CPU, as on Jetson Thor.
    pub memory_is_unified: Option<bool>,
    /// GPU power draw in watts.
    pub power_watts: Option<f64>,
    /// GPU temperature in degrees Celsius.
    pub temperature_celsius: Option<f64>,
}

/// The `cpu` object of `/v1/system`.
#[derive(Clone, Debug, Default, Deserialize, PartialEq)]
pub struct ServerCpu {
    /// CPU utilization over all cores, 0 to 100.
    pub utilization_percent: Option<f64>,
    /// Number of CPU cores.
    pub cores: Option<u64>,
}

/// The engine load read from Prometheus metrics, each figure summed over
/// every label set.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct EngineLoad {
    /// Requests being generated now.
    pub running: Option<f64>,
    /// Requests waiting in the queue.
    pub waiting: Option<f64>,
    /// The share of the KV cache in use, 0 to 1.
    pub kv_cache_fraction: Option<f64>,
    /// Generated tokens per second.
    pub tokens_per_second: Option<f64>,
}

/// One figure of [`EngineLoad`].
#[derive(Clone, Copy, Debug, PartialEq)]
enum LoadFigure {
    /// [`EngineLoad::running`].
    Running,
    /// [`EngineLoad::waiting`].
    Waiting,
    /// [`EngineLoad::kv_cache_fraction`].
    KvCache,
    /// [`EngineLoad::tokens_per_second`].
    Throughput,
}

/// The metric names each engine publishes, and the figure each one gives.
const METRIC_FIGURES: [(&str, LoadFigure); 12] = [
    ("vllm:num_requests_running", LoadFigure::Running),
    ("vllm:num_requests_waiting", LoadFigure::Waiting),
    ("vllm:kv_cache_usage_perc", LoadFigure::KvCache),
    ("vllm:gpu_cache_usage_perc", LoadFigure::KvCache),
    ("sglang:num_running_reqs", LoadFigure::Running),
    ("sglang:num_queue_reqs", LoadFigure::Waiting),
    ("sglang:token_usage", LoadFigure::KvCache),
    ("sglang:gen_throughput", LoadFigure::Throughput),
    ("llamacpp:requests_processing", LoadFigure::Running),
    ("llamacpp:requests_deferred", LoadFigure::Waiting),
    ("llamacpp:kv_cache_usage_ratio", LoadFigure::KvCache),
    ("llamacpp:predicted_tokens_seconds", LoadFigure::Throughput),
];

impl EngineLoad {
    /// Add `value` to `figure`.
    fn add(&mut self, figure: LoadFigure, value: f64) {
        let slot = self.slot(figure);

        *slot = Some(slot.unwrap_or(0.0) + value);
    }

    /// The field that holds `figure`.
    fn slot(&mut self, figure: LoadFigure) -> &mut Option<f64> {
        match figure {
            LoadFigure::Running => &mut self.running,
            LoadFigure::Waiting => &mut self.waiting,
            LoadFigure::KvCache => &mut self.kv_cache_fraction,
            LoadFigure::Throughput => &mut self.tokens_per_second,
        }
    }
}

/// One model Ollama holds in memory, from `/api/ps`.
#[derive(Clone, Debug, Default, Deserialize, PartialEq)]
pub struct LoadedModel {
    /// The model's tag, such as `qwen2.5-coder:7b`.
    pub name: String,
    /// The bytes of the model held in GPU memory.
    pub size_vram: Option<u64>,
}

/// The JSON body of Ollama's `/api/ps`.
#[derive(Deserialize)]
struct ProcessList {
    /// The models held in memory.
    models: Vec<LoadedModel>,
}

/// The hardware report in a `/v1/system` body, or `None` when it is not
/// that JSON.
pub fn server_system(body: &str) -> Option<ServerSystem> {
    serde_json::from_str(body).ok()
}

/// The engine load in Prometheus text, or `None` when the text holds none of
/// the metrics in [`METRIC_FIGURES`], as with text that is not Prometheus.
pub fn engine_load(metrics_text: &str) -> Option<EngineLoad> {
    let samples: Vec<(LoadFigure, f64)> = metrics_text.lines().filter_map(known_sample).collect();

    if samples.is_empty() {
        return None;
    }

    Some(summed_load(&samples))
}

/// The sum of `samples` for each figure.
fn summed_load(samples: &[(LoadFigure, f64)]) -> EngineLoad {
    let mut load = EngineLoad::default();

    for &(figure, value) in samples {
        load.add(figure, value);
    }

    load
}

/// The figure and value of a Prometheus sample line with a known metric name.
fn known_sample(line: &str) -> Option<(LoadFigure, f64)> {
    let (name, value) = prometheus_sample(line)?;
    let figure = metric_figure(name)?;

    Some((figure, value))
}

/// The figure that metric `name` gives, or `None` for any other metric.
fn metric_figure(name: &str) -> Option<LoadFigure> {
    METRIC_FIGURES
        .iter()
        .find(|(metric, _figure)| *metric == name)
        .map(|(_metric, figure)| *figure)
}

/// The metric name and value of one Prometheus sample line, such as
/// `vllm:num_requests_running{model_name="qwen"} 2.0`, or `None` for a line
/// that is not one.
fn prometheus_sample(line: &str) -> Option<(&str, f64)> {
    let line = line.trim();
    let name_end = line.find(['{', ' ', '\t'])?;
    let (name, after_name) = line.split_at(name_end);

    let value = text_after_labels(after_name)
        .split_whitespace()
        .next()?
        .parse()
        .ok()?;

    Some((name, value))
}

/// The text after a sample's `{...}` labels, or `after_name` itself when the
/// sample has no labels. A label value may hold spaces, so the labels end at
/// the last `}`.
fn text_after_labels(after_name: &str) -> &str {
    after_name
        .strip_prefix('{')
        .and_then(|labels| labels.rsplit_once('}'))
        .map_or(after_name, |(_labels, rest)| rest)
}

/// The loaded models in an `/api/ps` body, or `None` when it is not that JSON.
pub fn loaded_models(body: &str) -> Option<Vec<LoadedModel>> {
    serde_json::from_str::<ProcessList>(body)
        .ok()
        .map(|list| list.models)
}

/// Whether the server at `base_url` runs on another host: a host that is not
/// `localhost`, `127.0.0.0/8` or `::1`. An address that does not parse counts
/// as local.
pub fn is_remote_server(base_url: &str) -> bool {
    server_host(base_url).is_some_and(|host| !is_loopback(&host))
}

/// The host in `base_url`, such as `thor` or `[::1]`.
pub fn server_host(base_url: &str) -> Option<String> {
    let url = reqwest::Url::parse(base_url).ok()?;

    url.host_str().map(str::to_string)
}

/// Whether `host` names this machine. An IPv6 address keeps its brackets in
/// a URL, so they are removed before it is read.
fn is_loopback(host: &str) -> bool {
    let address = host.trim_start_matches('[').trim_end_matches(']');

    address
        .parse::<IpAddr>()
        .map_or(address == "localhost", |ip| ip.is_loopback())
}

/// The panel lines for the server: [`WAITING_FOR_SERVER_STATS`] alone while
/// `stats` is `None` because no poll has finished, else [`server_lines`].
pub fn server_section(stats: Option<&ServerStats>, fallback_host: &str) -> Vec<String> {
    let Some(stats) = stats else {
        return vec![WAITING_FOR_SERVER_STATS.to_string()];
    };

    server_lines(stats, fallback_host)
}

/// The panel lines for the server's stats. The first line names the server:
/// the host `/v1/system` reports, else `fallback_host`. When the server
/// published nothing, the only line is [`NO_SERVER_STATS`].
fn server_lines(stats: &ServerStats, fallback_host: &str) -> Vec<String> {
    let system = stats.system.as_ref();
    let published: Vec<String> = [
        system.and_then(gpu_line),
        system.and_then(memory_and_cpu_line),
        stats.engine_load.as_ref().and_then(load_line),
        stats.loaded_models.as_deref().map(loaded_models_line),
    ]
    .into_iter()
    .flatten()
    .collect();

    let host = system
        .and_then(|system| system.host.clone())
        .unwrap_or_else(|| fallback_host.to_string());

    labeled_lines(published, &host)
}

/// `lines` with `server {host} · ` before the first, or [`NO_SERVER_STATS`]
/// alone when there are no lines.
fn labeled_lines(mut lines: Vec<String>, host: &str) -> Vec<String> {
    let Some(first) = lines.first_mut() else {
        return vec![NO_SERVER_STATS.to_string()];
    };

    *first = format!("server {host} · {first}");

    lines
}

/// The GPU's name, utilization, power and temperature, such as
/// `NVIDIA Thor · GPU 37% · 4.4 W · 41 °C`.
fn gpu_line(system: &ServerSystem) -> Option<String> {
    let gpu = system.gpu.as_ref()?;

    joined([
        gpu.name.clone(),
        gpu.utilization_percent
            .map(|util| format!("GPU {util:.0}%")),
        gpu.power_watts.map(|power| format!("{power:.1} W")),
        gpu.temperature_celsius.map(|temp| format!("{temp:.0} °C")),
    ])
}

/// The GPU memory and the CPU figures, such as
/// `memory 84.0/122.1 GB (shared with CPU) · CPU 12% · 14 cores`.
fn memory_and_cpu_line(system: &ServerSystem) -> Option<String> {
    let cpu = system.cpu.as_ref();

    joined([
        system.gpu.as_ref().and_then(memory_figure),
        cpu.and_then(|cpu| cpu.utilization_percent)
            .map(|util| format!("CPU {util:.0}%")),
        cpu.and_then(|cpu| cpu.cores)
            .map(|cores| format!("{cores} cores")),
    ])
}

/// The GPU memory in use out of all of it, or `None` unless both are known.
fn memory_figure(gpu: &ServerGpu) -> Option<String> {
    let used = gibibytes(gpu.memory_used_bytes?);
    let total = gibibytes(gpu.memory_total_bytes?);

    Some(memory_text(
        used,
        total,
        gpu.memory_is_unified == Some(true),
    ))
}

/// `used` out of `total` gibibytes, named as memory shared with the CPU when
/// `is_unified`, else as VRAM.
fn memory_text(used: f64, total: f64, is_unified: bool) -> String {
    if is_unified {
        return format!("memory {used:.1}/{total:.1} GB (shared with CPU)");
    }

    format!("VRAM {used:.1}/{total:.1} GB")
}

/// The engine load, such as
/// `requests 2 running · 1 waiting · KV cache 34% · 52.1 tok/s`.
fn load_line(load: &EngineLoad) -> Option<String> {
    let kv_cache_percent = load.kv_cache_fraction.map(|fraction| fraction * 100.0);

    joined([
        load.running
            .map(|running| format!("requests {running:.0} running")),
        load.waiting.map(|waiting| format!("{waiting:.0} waiting")),
        kv_cache_percent.map(|percent| format!("KV cache {percent:.0}%")),
        load.tokens_per_second
            .map(|rate| format!("{rate:.1} tok/s")),
    ])
}

/// The models Ollama holds in memory, such as
/// `loaded qwen2.5-coder:7b · 5.1 GB VRAM`.
fn loaded_models_line(models: &[LoadedModel]) -> String {
    if models.is_empty() {
        return "no model loaded".to_string();
    }

    let described: Vec<String> = models.iter().map(loaded_model_text).collect();

    format!("loaded {}", described.join(", "))
}

/// One loaded model's name and the GPU memory it holds.
fn loaded_model_text(model: &LoadedModel) -> String {
    match model.size_vram {
        Some(bytes) => format!("{} · {:.1} GB VRAM", model.name, gibibytes(bytes)),
        None => model.name.clone(),
    }
}

/// The figures that are present, joined with ` · `, or `None` when none is.
fn joined<const COUNT: usize>(figures: [Option<String>; COUNT]) -> Option<String> {
    let present: Vec<String> = figures.into_iter().flatten().collect();

    if present.is_empty() {
        return None;
    }

    Some(present.join(" · "))
}

/// `bytes` in gibibytes.
fn gibibytes(bytes: u64) -> f64 {
    bytes as f64 / BYTES_PER_GIB
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The `/v1/system` body thor-tigress-agent publishes.
    const THOR_SYSTEM: &str = r#"{"host":"thor","sampled_at_unix_ms":1794560000000,"gpu":{"name":"NVIDIA Thor","utilization_percent":37.0,"memory_used_bytes":90194313216,"memory_total_bytes":131072000000,"memory_is_unified":true,"power_watts":4.37,"temperature_celsius":41.0},"cpu":{"utilization_percent":12.0,"cores":14},"models":[{"id":"Gemma-4-26B-A4B-NVFP4","engine":"tensorrt-edgellm"}]}"#;

    /// vLLM metrics with two engines, a histogram and the older cache name.
    const VLLM_METRICS: &str = "\
# HELP vllm:num_requests_running Number of requests in model execution batches.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{engine=\"0\",model_name=\"qwen\"} 2.0
vllm:num_requests_running{engine=\"1\",model_name=\"qwen\"} 1.0
vllm:num_requests_waiting{engine=\"0\",model_name=\"qwen\"} 1.0
vllm:gpu_cache_usage_perc{engine=\"0\",model_name=\"qwen\"} 0.34
vllm:time_to_first_token_seconds_bucket{le=\"0.001\",model_name=\"qwen\"} 0.0
";

    /// SGLang metrics, with a label value that holds a space and a brace.
    const SGLANG_METRICS: &str = "\
# TYPE sglang:num_running_reqs gauge
sglang:num_running_reqs{model_name=\"meta llama {8B}\"} 4.0
sglang:num_queue_reqs{model_name=\"meta llama {8B}\"} 0.0
sglang:token_usage{model_name=\"meta llama {8B}\"} 0.5
sglang:gen_throughput{model_name=\"meta llama {8B}\"} 85.25
";

    /// llama.cpp metrics, which carry no labels.
    const LLAMACPP_METRICS: &str = "\
# HELP llamacpp:predicted_tokens_seconds Average generation throughput in tokens/s.
llamacpp:predicted_tokens_seconds 52.1
llamacpp:requests_processing 1
llamacpp:requests_deferred 3
llamacpp:kv_cache_usage_ratio 0.25
";

    /// An Ollama `/api/ps` body with two models, one without `size_vram`.
    const OLLAMA_PS: &str = r#"{"models":[{"name":"qwen2.5-coder:7b","model":"qwen2.5-coder:7b","size":6000000000,"size_vram":5476083712},{"name":"nomic-embed-text:latest"}]}"#;

    #[test]
    fn system_report_parses_every_figure() {
        let system = server_system(THOR_SYSTEM).unwrap_or_default();
        let gpu = system.gpu.clone().unwrap_or_default();

        assert_eq!(system.host.as_deref(), Some("thor"));
        assert_eq!(gpu.memory_used_bytes, Some(90_194_313_216));
        assert_eq!(gpu.memory_is_unified, Some(true));
        assert_eq!(system.cpu.and_then(|cpu| cpu.cores), Some(14));
    }

    #[test]
    fn system_report_takes_nulls_and_refuses_other_text() {
        let nulls = r#"{"host":null,"sampled_at_unix_ms":1,"gpu":null,"cpu":null,"models":null}"#;

        assert_eq!(server_system(nulls), Some(ServerSystem::default()));
        assert_eq!(server_system("<html>404</html>"), None);
    }

    #[test]
    fn vllm_metrics_sum_every_label_set() {
        let expected = EngineLoad {
            running: Some(3.0),
            waiting: Some(1.0),
            kv_cache_fraction: Some(0.34),
            tokens_per_second: None,
        };

        assert_eq!(engine_load(VLLM_METRICS), Some(expected));
    }

    #[test]
    fn sglang_metrics_read_labels_with_spaces_and_braces() {
        let expected = EngineLoad {
            running: Some(4.0),
            waiting: Some(0.0),
            kv_cache_fraction: Some(0.5),
            tokens_per_second: Some(85.25),
        };

        assert_eq!(engine_load(SGLANG_METRICS), Some(expected));
    }

    #[test]
    fn llamacpp_metrics_without_labels_parse() {
        let expected = EngineLoad {
            running: Some(1.0),
            waiting: Some(3.0),
            kv_cache_fraction: Some(0.25),
            tokens_per_second: Some(52.1),
        };

        assert_eq!(engine_load(LLAMACPP_METRICS), Some(expected));
    }

    #[test]
    fn newer_vllm_cache_name_is_read() {
        let metrics = "vllm:kv_cache_usage_perc{engine=\"0\"} 0.2\n";
        let load = engine_load(metrics).unwrap_or_default();

        assert_eq!(load.kv_cache_fraction, Some(0.2));
    }

    #[test]
    fn text_that_is_not_prometheus_gives_no_load() {
        let json = r#"[{"iterCounter": 3, "numActiveRequests": 1}]"#;
        let broken = "vllm:num_requests_running{engine=\"0\" 2\nvllm:num_requests_waiting x\n";

        assert_eq!(engine_load(json), None);
        assert_eq!(engine_load("<html>Not Found</html>"), None);
        assert_eq!(engine_load(broken), None);
        assert_eq!(engine_load(""), None);
    }

    #[test]
    fn ollama_process_list_parses() {
        let models = loaded_models(OLLAMA_PS).unwrap_or_default();

        assert_eq!(models.len(), 2);
        assert_eq!(models[0].size_vram, Some(5_476_083_712));
        assert_eq!(models[1].size_vram, None);
        assert_eq!(loaded_models("{\"error\":\"not found\"}"), None);
    }

    #[test]
    fn loopback_hosts_are_local() {
        assert!(!is_remote_server("http://localhost:11434"));
        assert!(!is_remote_server("http://127.0.0.1:8079/v1"));
        assert!(!is_remote_server("http://127.8.9.10:8079/v1"));
        assert!(!is_remote_server("http://[::1]:8079/v1"));
        assert!(!is_remote_server("not a url"));
    }

    #[test]
    fn other_hosts_are_remote() {
        assert!(is_remote_server("http://thor:8079/v1"));
        assert!(is_remote_server("http://192.168.1.20:11434"));
        assert!(is_remote_server("http://[fd00::2]:8000/v1"));
        assert_eq!(server_host("http://thor:8079/v1").as_deref(), Some("thor"));
    }

    #[test]
    fn thor_report_gives_the_gpu_and_memory_lines() {
        let stats = ServerStats::from_bodies(Some(THOR_SYSTEM), None, None);

        assert_eq!(
            server_lines(&stats, "10.0.0.5"),
            [
                "server thor · NVIDIA Thor · GPU 37% · 4.4 W · 41 °C",
                "memory 84.0/122.1 GB (shared with CPU) · CPU 12% · 14 cores",
            ]
        );
    }

    #[test]
    fn a_discrete_gpu_shows_vram_and_leaves_out_null_figures() {
        let body = r#"{"sampled_at_unix_ms":1,"gpu":{"name":null,"utilization_percent":5.0,"memory_used_bytes":12884901888,"memory_total_bytes":25769803776,"memory_is_unified":false,"power_watts":null,"temperature_celsius":null},"cpu":{"utilization_percent":null,"cores":null}}"#;
        let stats = ServerStats::from_bodies(Some(body), None, None);

        assert_eq!(
            server_lines(&stats, "gpu-box"),
            ["server gpu-box · GPU 5%", "VRAM 12.0/24.0 GB"]
        );
    }

    #[test]
    fn metrics_alone_name_the_server_by_its_address() {
        let stats = ServerStats::from_bodies(None, Some(LLAMACPP_METRICS), None);

        assert_eq!(
            server_lines(&stats, "10.0.0.5"),
            ["server 10.0.0.5 · requests 1 running · 3 waiting · KV cache 25% · 52.1 tok/s"]
        );
    }

    #[test]
    fn ollama_models_are_listed_with_their_vram() {
        let stats = ServerStats::from_bodies(None, None, Some(OLLAMA_PS));
        let empty = ServerStats::from_bodies(None, None, Some(r#"{"models":[]}"#));

        assert_eq!(
            server_lines(&stats, "box"),
            ["server box · loaded qwen2.5-coder:7b · 5.1 GB VRAM, nomic-embed-text:latest"]
        );
        assert_eq!(
            server_lines(&empty, "box"),
            ["server box · no model loaded"]
        );
    }

    #[test]
    fn a_server_that_publishes_nothing_gets_one_line() {
        let stats = ServerStats::from_bodies(Some("{}"), Some("<html></html>"), None);

        assert_eq!(server_lines(&stats, "box"), [NO_SERVER_STATS]);
    }

    #[test]
    fn the_section_waits_until_a_poll_has_finished() {
        let finished = ServerStats::default();

        assert_eq!(server_section(None, "box"), [WAITING_FOR_SERVER_STATS]);
        assert_eq!(server_section(Some(&finished), "box"), [NO_SERVER_STATS]);
    }
}
