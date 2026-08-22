//! TensorRT backend adapter (experimental).
//!
//! [`TensorRtBackend`] benchmarks the installed NVIDIA `trtexec` binary against
//! an ONNX model. It is intentionally benchmark-only: `trtexec` is not a chat
//! server, so [`InferenceBackend::chat_stream`] reports an unsupported error.

use super::{BenchSample, BoxStreamText, EngineConfig, EngineKind, InferenceBackend};
use crate::constants::engine::{TRTEXEC_CANDIDATES, TRTEXEC_DEFAULT_WARMUP_MILLIS};
use crate::ollama::ChatRequest;
use anyhow::{anyhow, bail, Context, Result};
use async_trait::async_trait;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::Duration;

/// Experimental backend that shells out to `trtexec`.
#[derive(Clone, Debug)]
pub struct TensorRtBackend {
    onnx_model: Option<PathBuf>,
    seq_len: usize,
    avg_runs: usize,
    trt_shapes: Option<String>,
}

impl TensorRtBackend {
    /// Build a TensorRT backend from a config.
    pub fn new(config: &EngineConfig) -> Self {
        Self {
            onnx_model: config.model.as_ref().map(PathBuf::from),
            seq_len: config.trtexec_seq_len,
            avg_runs: config.trtexec_avg_runs,
            trt_shapes: config.trt_shapes.clone(),
        }
    }

    /// Locate the `trtexec` executable on this machine.
    pub fn find_trtexec() -> Option<PathBuf> {
        TRTEXEC_CANDIDATES
            .iter()
            .find(|candidate| trtexec_exists(candidate))
            .map(PathBuf::from)
    }

    async fn run_trtexec(&self) -> Result<String> {
        let Some(trtexec) = Self::find_trtexec() else {
            bail!("trtexec not found; install NVIDIA TensorRT or pass a custom path");
        };
        let Some(model) = &self.onnx_model else {
            bail!("TensorRT benchmark requires --model pointing to an .onnx file");
        };
        if !model.is_file() {
            bail!("ONNX model does not exist: {}", model.display());
        }

        let mut command = tokio::process::Command::new(&trtexec);
        command
            .arg(format!("--onnx={}", model.display()))
            .arg("--fp16")
            .arg(format!("--avgRuns={}", self.avg_runs))
            .arg(format!("--warmUp={}", TRTEXEC_DEFAULT_WARMUP_MILLIS));
        if let Some(shapes) = &self.trt_shapes {
            command.arg(format!("--shapes={shapes}"));
        }
        command.stdout(Stdio::piped()).stderr(Stdio::piped());

        let output = tokio::time::timeout(Duration::from_secs(600), command.output())
            .await
            .context("trtexec timed out after 600s")?
            .context("failed to launch trtexec")?;

        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();
        if !output.status.success() {
            return Err(anyhow!(
                "trtexec failed for {}:\n{}",
                model.display(),
                stderr.trim()
            ));
        }
        Ok(stdout)
    }
}

/// Whether a candidate `trtexec` path is executable.
fn trtexec_exists(candidate: &str) -> bool {
    if candidate.contains('/') {
        return Path::new(candidate).is_file();
    }
    std::process::Command::new("sh")
        .arg("-lc")
        .arg(format!("command -v {candidate}"))
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|status| status.success())
        .unwrap_or(false)
}

/// Extract throughput (queries per second) from `trtexec` output.
fn parse_trtexec_throughput(output: &str) -> Option<f64> {
    output.lines().find_map(|line| {
        let marker = "Throughput:";
        let rest = line.split(marker).nth(1)?.trim();
        rest.split_whitespace().next()?.parse().ok()
    })
}

#[async_trait]
impl InferenceBackend for TensorRtBackend {
    fn kind(&self) -> EngineKind {
        EngineKind::TensorRt
    }

    async fn is_available(&self) -> bool {
        Self::find_trtexec().is_some()
    }

    async fn chat_stream(&self, _request: ChatRequest) -> Result<BoxStreamText> {
        bail!("TensorRT (trtexec) is a benchmark-only experimental backend; it cannot serve chat")
    }

    async fn bench_generate(&self, _prompt: &str, _max_tokens: usize) -> Result<BenchSample> {
        let output = self.run_trtexec().await?;
        let throughput = parse_trtexec_throughput(&output)
            .ok_or_else(|| anyhow!("could not parse 'Throughput' from trtexec output"))?;
        let elapsed_per_inference = 1.0 / throughput;
        Ok(BenchSample {
            prompt_tokens: self.seq_len as u64,
            generated_tokens: 0,
            elapsed_seconds: elapsed_per_inference,
            notes: format!(
                "prefill-only via trtexec; {} inferences/s × {} seq_len",
                throughput, self.seq_len
            ),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_throughput_from_trtexec_output() {
        let output = "\
[08/21/2026-23:59:59] [I] Throughput: 1234.56 qps
[08/21/2026-23:59:59] [I] Latency: min = 0.123 ms
";
        assert_eq!(parse_trtexec_throughput(output), Some(1234.56));
    }

    #[test]
    fn parses_no_throughput_as_none() {
        assert_eq!(parse_trtexec_throughput("no throughput here"), None);
    }

    #[test]
    fn trtexec_defaults_are_sane() {
        use crate::constants::engine::{
            TRTEXEC_DEFAULT_AVG_RUNS, TRTEXEC_DEFAULT_SEQ_LEN, TRTEXEC_DEFAULT_WARMUP_MILLIS,
        };
        const _: () = {
            assert!(TRTEXEC_DEFAULT_SEQ_LEN > 0);
            assert!(TRTEXEC_DEFAULT_AVG_RUNS > 0);
            assert!(TRTEXEC_DEFAULT_WARMUP_MILLIS > 0);
        };
    }
}
