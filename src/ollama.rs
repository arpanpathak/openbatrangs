//! Minimal async HTTP client for the Ollama local model server.
//!
//! This module wraps the endpoints used by openBatarangs:
//! - `GET  /api/tags`    — list installed models
//! - `POST /api/show`    — inspect model metadata
//! - `POST /api/chat`    — chat completion (streaming and non-streaming)
//! - `POST /api/pull`    — download a model from the Ollama registry

use anyhow::{anyhow, Context, Result};
use futures_util::{Stream, StreamExt};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::time::Duration;

/// How long a full HTTP request may take before it is aborted.
const HTTP_TIMEOUT_SECONDS: u64 = 600;

/// How long to wait for the initial TCP connection to the Ollama server.
const CONNECT_TIMEOUT_SECONDS: u64 = 5;

/// Client for talking to a local Ollama server.
#[derive(Clone)]
pub struct OllamaClient {
    /// Base URL, e.g. `http://localhost:11434` (trailing slash stripped).
    pub base_url: String,
    /// Reusable HTTP client with sensible timeouts.
    http: reqwest::Client,
}

/// Response body of `GET /api/tags`.
#[derive(Deserialize, Clone, Debug)]
pub struct TagsResponse {
    /// All models currently installed on the Ollama server.
    pub models: Vec<OllamaModel>,
}

/// A single installed model as returned by `/api/tags`.
#[derive(Deserialize, Clone, Debug)]
pub struct OllamaModel {
    /// Model tag, e.g. `qwen2.5-coder:7b`.
    pub name: String,
    /// Size of the model file on disk, in bytes.
    #[serde(default)]
    pub size: u64,
    /// Optional human-readable metadata about the model.
    #[serde(default)]
    pub details: Option<ModelDetails>,
}

/// Optional metadata for an installed model.
#[derive(Deserialize, Clone, Debug, Default)]
pub struct ModelDetails {
    /// Parameter count label, e.g. `7.6B` or `873.44M`.
    #[serde(default)]
    pub parameter_size: Option<String>,
    /// Quantization level, e.g. `Q4_K_M`.
    #[serde(default)]
    pub quantization_level: Option<String>,
    /// Maximum context length the model supports.
    #[serde(default)]
    pub context_length: Option<u64>,
}

/// A single chat message sent to the model.
#[derive(Serialize, Clone, Debug)]
pub struct ChatMessage {
    /// One of `system`, `user`, `assistant`, or `tool`.
    pub role: String,
    /// Message body.
    pub content: String,
}

/// Payload for `POST /api/chat`.
#[derive(Serialize, Debug)]
pub struct ChatRequest {
    /// Model tag to use.
    pub model: String,
    /// Ordered conversation history.
    pub messages: Vec<ChatMessage>,
    /// Whether to stream the response as NDJSON.
    pub stream: bool,
    /// Optional response format hint (e.g. `json`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub format: Option<Value>,
    /// Optional sampling options (`temperature`, `num_ctx`, ...).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub options: Option<Value>,
}

/// Non-streaming chat response body.
#[derive(Deserialize, Debug)]
#[allow(dead_code)]
pub struct ChatResponse {
    /// The assistant's reply message.
    pub message: ChatResponseMessage,
}

/// The assistant's reply message in a non-streaming response.
#[derive(Deserialize, Debug)]
#[allow(dead_code)]
pub struct ChatResponseMessage {
    /// Reply text.
    pub content: String,
}

/// Payload for `POST /api/pull`.
#[derive(Serialize, Debug)]
struct PullRequest {
    /// Model tag to download, e.g. `qwen2.5-coder:3b`.
    name: String,
    /// Whether to stream pull progress. We use `false` for simplicity.
    stream: bool,
}

impl OllamaClient {
    /// Create a client for the given Ollama server URL.
    ///
    /// # Arguments
    /// - `base_url`: server address, e.g. `http://localhost:11434`.
    ///
    /// # Returns
    /// A configured client, or an error if the HTTP client could not be built.
    pub fn new(base_url: &str) -> Result<Self> {
        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(HTTP_TIMEOUT_SECONDS))
            .connect_timeout(Duration::from_secs(CONNECT_TIMEOUT_SECONDS))
            .build()
            .context("failed to build HTTP client")?;
        Ok(Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            http,
        })
    }

    /// Check whether the Ollama server is reachable.
    ///
    /// # Returns
    /// `true` if `GET /api/tags` succeeds, otherwise `false`.
    pub async fn is_available(&self) -> bool {
        self.http
            .get(format!("{}/api/tags", self.base_url))
            .send()
            .await
            .map(|response| response.status().is_success())
            .unwrap_or(false)
    }

    /// List all installed models.
    ///
    /// # Returns
    /// A vector of installed models, or an error if the server is unreachable
    /// or returns a non-success status.
    pub async fn tags(&self) -> Result<Vec<OllamaModel>> {
        let response = self
            .http
            .get(format!("{}/api/tags", self.base_url))
            .send()
            .await
            .context("failed to reach Ollama server; is `ollama serve` running?")?;

        if !response.status().is_success() {
            return Err(anyhow!(
                "Ollama /api/tags returned HTTP {}",
                response.status()
            ));
        }

        let body: TagsResponse = response
            .json()
            .await
            .context("failed to parse Ollama /api/tags response")?;
        Ok(body.models)
    }

    /// Fetch detailed metadata for a single model.
    ///
    /// # Arguments
    /// - `name`: model tag, e.g. `qwen2.5-coder:7b`.
    ///
    /// # Returns
    /// Raw JSON metadata from `/api/show`.
    pub async fn show(&self, name: &str) -> Result<Value> {
        let response = self
            .http
            .post(format!("{}/api/show", self.base_url))
            .json(&serde_json::json!({ "name": name }))
            .send()
            .await
            .context("failed to call Ollama /api/show")?;

        if !response.status().is_success() {
            return Err(anyhow!(
                "Ollama /api/show returned HTTP {} for model '{}'",
                response.status(),
                name
            ));
        }

        Ok(response
            .json()
            .await
            .context("failed to parse Ollama /api/show response")?)
    }

    /// Perform a non-streaming chat completion.
    ///
    /// Kept for callers that want the whole response at once.
    #[allow(dead_code)]
    pub async fn chat(&self, request: ChatRequest) -> Result<ChatResponse> {
        let response = self
            .http
            .post(format!("{}/api/chat", self.base_url))
            .json(&request)
            .send()
            .await
            .context("failed to call Ollama /api/chat")?;

        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            return Err(anyhow!("Ollama /api/chat returned HTTP {status}: {text}"));
        }

        let body: ChatResponse = response
            .json()
            .await
            .context("failed to parse Ollama /api/chat response")?;
        Ok(body)
    }

    /// Stream a chat completion as a sequence of content deltas.
    ///
    /// # Arguments
    /// - `request`: chat request; `stream` is forced to `true`.
    ///
    /// # Returns
    /// A stream of text deltas. Each item is `Ok(String)` with the next piece
    /// of generated text, or `Err` if the stream fails.
    pub async fn chat_stream(
        &self,
        mut request: ChatRequest,
    ) -> Result<impl Stream<Item = Result<String>> + Send + 'static> {
        request.stream = true;
        let response = self
            .http
            .post(format!("{}/api/chat", self.base_url))
            .json(&request)
            .send()
            .await
            .context("failed to call Ollama /api/chat (stream)")?;

        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            return Err(anyhow!("Ollama /api/chat returned HTTP {status}: {text}"));
        }

        let byte_stream = response.bytes_stream();
        let stream = futures_util::stream::unfold(
            (byte_stream, String::new()),
            |(mut byte_stream, mut buffer)| async move {
                loop {
                    match byte_stream.next().await {
                        Some(Ok(bytes)) => {
                            buffer.push_str(&String::from_utf8_lossy(&bytes));
                            while let Some(newline_pos) = buffer.find('\n') {
                                let line = buffer[..newline_pos].trim().to_string();
                                buffer = buffer[newline_pos + 1..].to_string();
                                if line.is_empty() {
                                    continue;
                                }
                                if let Ok(value) = serde_json::from_str::<Value>(&line) {
                                    if let Some(content) = value
                                        .pointer("/message/content")
                                        .and_then(|content| content.as_str())
                                    {
                                        return Some((
                                            Ok(content.to_string()),
                                            (byte_stream, buffer),
                                        ));
                                    }
                                    if value
                                        .get("done")
                                        .and_then(|done| done.as_bool())
                                        .unwrap_or(false)
                                    {
                                        return None;
                                    }
                                }
                            }
                        }
                        Some(Err(error)) => {
                            return Some((
                                Err(anyhow!("stream error: {error}")),
                                (byte_stream, buffer),
                            ));
                        }
                        None => return None,
                    }
                }
            },
        );
        Ok(stream)
    }

    /// Download a model from the Ollama registry.
    ///
    /// # Arguments
    /// - `name`: model tag to pull, e.g. `qwen2.5-coder:3b`.
    ///
    /// # Returns
    /// `Ok(())` once the pull finishes successfully.
    pub async fn pull(&self, name: &str) -> Result<()> {
        println!("⬇️  Pulling model '{name}' from Ollama registry...");
        let response = self
            .http
            .post(format!("{}/api/pull", self.base_url))
            .json(&PullRequest {
                name: name.to_string(),
                stream: false,
            })
            .send()
            .await
            .context("failed to call Ollama /api/pull")?;

        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            return Err(anyhow!("Ollama /api/pull returned HTTP {status}: {text}"));
        }

        let body: Value = response
            .json()
            .await
            .context("failed to parse Ollama /api/pull response")?;
        let status = body
            .get("status")
            .and_then(|value| value.as_str())
            .unwrap_or("done");
        println!("✅ Pull finished: {status}");
        Ok(())
    }
}
