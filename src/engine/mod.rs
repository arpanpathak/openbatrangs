//! Pluggable inference backends.
//!
//! This module is the single extension point for inference engines. Adding a
//! new engine means:
//!
//! 1. implementing [`InferenceBackend`] (Rust-side adapter), and
//! 2. registering it in [`create_backend`] with one `match` arm.
//!
//! Existing agent, TUI, and command code only talks to [`InferenceBackend`], so
//! it stays **open for extension but closed for modification**: adding TensorRT,
//! vLLM, or another engine does not require changing the agent loop.

mod ollama;
mod tensorrt;

pub use ollama::OllamaBackend;
pub use tensorrt::TensorRtBackend;

use crate::constants::engine::CHARS_PER_TOKEN;
use crate::ollama::{ChatMessage, ChatRequest, OllamaModel};
use anyhow::{bail, Context, Result};
use async_trait::async_trait;
use futures_util::StreamExt;
use serde_json::Value;
use std::time::Instant;

/// Identifier for every inference engine openBatarangs can talk to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EngineKind {
    /// Ollama's local HTTP server (production default).
    Ollama,
    /// NVIDIA TensorRT via the installed `trtexec` binary.
    TensorRt,
}

impl EngineKind {
    /// Machine-readable engine name used on the CLI and in reports.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Ollama => "ollama",
            Self::TensorRt => "tensorrt",
        }
    }

    /// Human-friendly display name.
    pub fn display_name(self) -> &'static str {
        match self {
            Self::Ollama => "Ollama",
            Self::TensorRt => "TensorRT (trtexec)",
        }
    }

    /// Parse a user-supplied engine name (case-insensitive).
    pub fn parse(name: &str) -> Option<Self> {
        match name.trim().to_ascii_lowercase().as_str() {
            "ollama" => Some(Self::Ollama),
            "tensorrt" | "trt" | "trtexec" => Some(Self::TensorRt),
            _ => None,
        }
    }

    /// Every known engine, in benchmark display order.
    pub fn all() -> &'static [Self] {
        &[Self::Ollama, Self::TensorRt]
    }
}

/// Configuration needed to construct a backend.
#[derive(Clone, Debug)]
pub struct EngineConfig {
    /// Which engine to build.
    pub kind: EngineKind,
    /// Ollama base URL (only used by the Ollama backend).
    pub ollama_url: String,
    /// Optional explicit model tag (Ollama) or ONNX model path (TensorRT).
    pub model: Option<String>,
    /// TensorRT input sequence length for prefill-equivalent throughput.
    pub trtexec_seq_len: usize,
    /// TensorRT repetitions per benchmark run.
    pub trtexec_avg_runs: usize,
    /// Optional `--shapes` string forwarded to trtexec (e.g.
    /// `input_ids:1x128,attention_mask:1x128`).
    pub trt_shapes: Option<String>,
}

impl EngineConfig {
    /// Build a config for the given kind with project defaults.
    pub fn new(kind: EngineKind, ollama_url: &str, model: Option<String>) -> Self {
        Self {
            kind,
            ollama_url: ollama_url.to_string(),
            model,
            trtexec_seq_len: crate::constants::engine::TRTEXEC_DEFAULT_SEQ_LEN,
            trtexec_avg_runs: crate::constants::engine::TRTEXEC_DEFAULT_AVG_RUNS,
            trt_shapes: None,
        }
    }
}

/// One measured generation sample.
#[derive(Clone, Debug)]
pub struct BenchSample {
    /// Prompt tokens consumed (0 when unknown).
    pub prompt_tokens: u64,
    /// Generated tokens (0 for engine-level microbenchmarks).
    pub generated_tokens: u64,
    /// Wall-clock (or engine-reported) elapsed seconds for the measured work.
    pub elapsed_seconds: f64,
    /// Extra notes, e.g. prefill-only or unavailable.
    pub notes: String,
}

impl BenchSample {
    /// Tokens per second.
    ///
    /// Prefers engine-reported generated tokens, then prompt tokens (TensorRT
    /// prefill microbenchmarks), then a character-count estimate.
    pub fn tokens_per_second(&self, chars_generated: usize) -> f64 {
        let tokens = if self.generated_tokens > 0 {
            self.generated_tokens as f64
        } else if self.prompt_tokens > 0 {
            self.prompt_tokens as f64
        } else if chars_generated > 0 {
            chars_generated as f64 / CHARS_PER_TOKEN
        } else {
            0.0
        };
        if self.elapsed_seconds > 0.0 {
            tokens / self.elapsed_seconds
        } else {
            0.0
        }
    }
}

/// Common interface for talking to an inference engine.
///
/// This is intentionally small: every method maps to an operation the agent or
/// benchmark harness needs. Default implementations let experimental backends
/// avoid implementing Ollama-only operations.
#[async_trait]
pub trait InferenceBackend: Send + Sync {
    /// Which engine this backend talks to.
    fn kind(&self) -> EngineKind;

    /// Whether the engine is installed and usable on this machine.
    async fn is_available(&self) -> bool;

    /// List installed models (empty for engines without a model registry).
    async fn tags(&self) -> Result<Vec<OllamaModel>> {
        Ok(Vec::new())
    }

    /// Fetch engine-specific model metadata.
    async fn show(&self, _model: &str) -> Result<Value> {
        Ok(Value::Null)
    }

    /// Stream a chat completion as text deltas.
    async fn chat_stream(&self, request: ChatRequest) -> Result<BoxStreamText>;

    /// Download/install a model (Ollama-only today).
    async fn pull(&self, _name: &str, _on_status: &(dyn for<'a> Fn(&'a str) + Sync)) -> Result<()> {
        bail!("this engine does not support pulling models")
    }

    /// Run one benchmark generation and return token/latency metrics.
    ///
    /// The default implementation streams [`Self::chat_stream`] and estimates
    /// token count from characters; engines with native metrics override this
    /// for accurate counts.
    async fn bench_generate(&self, prompt: &str, max_tokens: usize) -> Result<BenchSample> {
        let request = ChatRequest {
            model: String::new(),
            messages: vec![ChatMessage {
                role: "user".to_string(),
                content: prompt.to_string(),
            }],
            stream: true,
            format: None,
            options: Some(serde_json::json!({ "num_predict": max_tokens, "temperature": 0.0 })),
        };
        let started = Instant::now();
        let mut stream = self.chat_stream(request).await?;
        let mut chars = 0usize;
        while let Some(delta) = stream.next().await {
            chars += delta?.chars().count();
        }
        let estimated_tokens = (chars as f64 / CHARS_PER_TOKEN).round() as u64;
        Ok(BenchSample {
            prompt_tokens: 0,
            generated_tokens: estimated_tokens,
            elapsed_seconds: started.elapsed().as_secs_f64(),
            notes: "estimated from character count".to_string(),
        })
    }
}

/// Boxed stream of text deltas used by [`InferenceBackend::chat_stream`].
pub type BoxStreamText = futures_util::stream::BoxStream<'static, Result<String>>;

/// Create a backend from a config.
///
/// This is the **only** place that knows how to construct a backend from an
/// [`EngineKind`]; adding a new engine adds one arm here.
pub fn create_backend(config: &EngineConfig) -> Result<std::sync::Arc<dyn InferenceBackend>> {
    match config.kind {
        EngineKind::Ollama => {
            let client = crate::ollama::OllamaClient::new(&config.ollama_url)?;
            Ok(std::sync::Arc::new(
                OllamaBackend::new(client).with_model(config.model.clone()),
            ))
        }
        EngineKind::TensorRt => Ok(std::sync::Arc::new(TensorRtBackend::new(config))),
    }
}

/// Resolve a user-supplied engine list; empty means "all available engines".
pub fn resolve_engine_kinds(names: &[String]) -> Result<Vec<EngineKind>> {
    if names.is_empty() {
        return Ok(EngineKind::all().to_vec());
    }
    names
        .iter()
        .map(|name| {
            EngineKind::parse(name)
                .with_context(|| format!("unknown engine '{name}' (try: ollama, tensorrt)"))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_engine_names_case_insensitively() {
        assert_eq!(EngineKind::parse("Ollama"), Some(EngineKind::Ollama));
        assert_eq!(EngineKind::parse("trt"), Some(EngineKind::TensorRt));
        assert_eq!(EngineKind::parse("trtexec"), Some(EngineKind::TensorRt));
        assert_eq!(EngineKind::parse("vllm"), None);
        assert_eq!(EngineKind::parse("unknown"), None);
    }

    #[test]
    fn all_engine_names_round_trip() {
        for kind in EngineKind::all() {
            assert_eq!(EngineKind::parse(kind.as_str()), Some(*kind));
        }
    }

    #[test]
    fn empty_engine_list_resolves_to_all() {
        assert_eq!(
            resolve_engine_kinds(&[]).unwrap().len(),
            EngineKind::all().len()
        );
    }

    #[test]
    fn unknown_engine_list_is_rejected() {
        assert!(resolve_engine_kinds(&["nope".to_string()]).is_err());
    }

    #[test]
    fn bench_sample_tokens_per_second_uses_reported_tokens() {
        let sample = BenchSample {
            prompt_tokens: 1,
            generated_tokens: 100,
            elapsed_seconds: 2.0,
            notes: String::new(),
        };
        assert!((sample.tokens_per_second(0) - 50.0).abs() < f64::EPSILON);
    }

    #[test]
    fn bench_sample_tokens_per_second_uses_prompt_tokens_for_prefill() {
        let sample = BenchSample {
            prompt_tokens: 128,
            generated_tokens: 0,
            elapsed_seconds: 0.5,
            notes: "prefill-only".to_string(),
        };
        assert!((sample.tokens_per_second(0) - 256.0).abs() < f64::EPSILON);
    }

    #[test]
    fn bench_sample_tokens_per_second_estimates_from_chars() {
        let sample = BenchSample {
            prompt_tokens: 0,
            generated_tokens: 0,
            elapsed_seconds: 1.0,
            notes: String::new(),
        };
        assert!((sample.tokens_per_second(40) - 10.0).abs() < f64::EPSILON);
    }
}
