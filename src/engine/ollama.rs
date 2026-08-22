//! Ollama backend adapter.
//!
//! [`OllamaBackend`] implements [`InferenceBackend`] by delegating to the
//! existing [`OllamaClient`]. Keeping the HTTP client concrete here means the
//! agent/TUI layers never see Ollama-specific code.

use super::{BenchSample, BoxStreamText, EngineKind, InferenceBackend};
use crate::ollama::{ChatRequest, OllamaClient, OllamaModel};
use anyhow::{bail, Result};
use async_trait::async_trait;
use serde_json::Value;

/// Production backend backed by a local Ollama server.
#[derive(Clone)]
pub struct OllamaBackend {
    client: OllamaClient,
    model: Option<String>,
}

impl OllamaBackend {
    /// Wrap an existing Ollama client with no fixed model.
    pub fn new(client: OllamaClient) -> Self {
        Self {
            client,
            model: None,
        }
    }

    /// Attach an explicit model tag (used by the benchmark harness).
    pub fn with_model(mut self, model: Option<String>) -> Self {
        self.model = model;
        self
    }
}

#[async_trait]
impl InferenceBackend for OllamaBackend {
    fn kind(&self) -> EngineKind {
        EngineKind::Ollama
    }

    async fn is_available(&self) -> bool {
        self.client.is_available().await
    }

    async fn tags(&self) -> Result<Vec<OllamaModel>> {
        self.client.tags().await
    }

    async fn show(&self, model: &str) -> Result<Value> {
        self.client.show(model).await
    }

    async fn chat_stream(&self, request: ChatRequest) -> Result<BoxStreamText> {
        Ok(Box::pin(self.client.chat_stream(request).await?))
    }

    async fn pull(&self, name: &str, on_status: &(dyn for<'a> Fn(&'a str) + Sync)) -> Result<()> {
        self.client.pull(name, on_status).await
    }

    async fn bench_generate(&self, prompt: &str, max_tokens: usize) -> Result<BenchSample> {
        let Some(model) = &self.model else {
            bail!("Ollama benchmark requires a model; pass --model or let the harness select one");
        };
        let (response, prompt_tokens, generated_tokens, total_seconds) = self
            .client
            .generate_bench(model, prompt, max_tokens)
            .await?;
        let _ = response;
        Ok(BenchSample {
            prompt_tokens,
            generated_tokens,
            elapsed_seconds: total_seconds,
            notes: "Ollama /api/generate counters".to_string(),
        })
    }
}
