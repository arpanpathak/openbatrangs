//! Minimal async HTTP client for the Ollama local model server.
//!
//! This module wraps the endpoints used by openBatarangs:
//! - `GET  /api/tags`    — list installed models
//! - `POST /api/show`    — inspect model metadata
//! - `POST /api/chat`    — chat completion (streaming and non-streaming)
//! - `POST /api/pull`    — download a model from the Ollama registry

mod stream;
mod types;

pub(crate) use types::{ChatMessage, ChatRequest, OllamaModel, PullRequest, TagsResponse};

use crate::constants::ollama::{
    API_CHAT_PATH, API_PULL_PATH, API_SHOW_PATH, API_TAGS_PATH, CONNECT_TIMEOUT_SECONDS,
    HTTP_TIMEOUT_SECONDS,
};
use anyhow::{anyhow, Context, Result};
use futures_util::{Stream, StreamExt};
use serde_json::Value;
use std::time::Duration;

/// Client for talking to a local Ollama server.
#[derive(Clone)]
pub struct OllamaClient {
    /// Base URL, e.g. `http://localhost:11434` (trailing slash stripped).
    pub base_url: String,
    /// Reusable HTTP client with sensible timeouts.
    http: reqwest::Client,
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
            .get(format!("{}{API_TAGS_PATH}", self.base_url))
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
            .get(format!("{}{API_TAGS_PATH}", self.base_url))
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
            .post(format!("{}{API_SHOW_PATH}", self.base_url))
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

        response
            .json()
            .await
            .context("failed to parse Ollama /api/show response")
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
            .post(format!("{}{API_CHAT_PATH}", self.base_url))
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
                    match stream::drain_complete_lines(&mut buffer) {
                        stream::LineDrain::Content(content) => {
                            return Some((Ok(content), (byte_stream, buffer)));
                        }
                        stream::LineDrain::Done => return None,
                        stream::LineDrain::NeedMore => {}
                    }
                    match byte_stream.next().await {
                        Some(Ok(bytes)) => {
                            buffer.push_str(&String::from_utf8_lossy(&bytes));
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
    /// - `on_status`: callback invoked with progress messages.
    ///
    /// # Returns
    /// `Ok(())` once the pull finishes successfully.
    pub async fn pull(&self, name: &str, on_status: &(dyn Fn(&str) + Sync)) -> Result<()> {
        on_status(&format!(
            "⬇️  Pulling model '{name}' from Ollama registry..."
        ));
        let response = self
            .http
            .post(format!("{}{API_PULL_PATH}", self.base_url))
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
        on_status(&format!("✅ Pull finished: {status}"));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn client_strips_trailing_slashes_from_base_url() {
        let client = OllamaClient::new("http://localhost:11434///").unwrap();
        assert_eq!(client.base_url, "http://localhost:11434");
    }

    #[test]
    fn client_accepts_plain_url() {
        let client = OllamaClient::new("http://127.0.0.1:11434").unwrap();
        assert_eq!(client.base_url, "http://127.0.0.1:11434");
    }

    #[test]
    fn chat_request_serializes_expected_shape() {
        let request = ChatRequest {
            model: "qwen2.5-coder:3b".to_string(),
            messages: vec![ChatMessage {
                role: "user".to_string(),
                content: "hello".to_string(),
            }],
            stream: true,
            format: None,
            options: Some(serde_json::json!({"temperature": 0.7})),
        };
        let json = serde_json::to_value(&request).unwrap();
        assert_eq!(json["model"], "qwen2.5-coder:3b");
        assert_eq!(json["stream"], true);
        assert_eq!(json["messages"][0]["role"], "user");
        assert_eq!(json["options"]["temperature"], 0.7);
    }

    #[test]
    fn tags_response_deserializes_with_missing_details() {
        let json = r#"{"models":[{"name":"qwen2.5-coder:3b","size":123}]}"#;
        let response: TagsResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.models.len(), 1);
        assert_eq!(response.models[0].name, "qwen2.5-coder:3b");
        assert_eq!(response.models[0].size, 123);
        assert!(response.models[0].details.is_none());
    }

    #[test]
    fn tags_response_deserializes_with_details() {
        let json = r#"{"models":[{"name":"qwen2.5-coder:7b","size":1,"details":{"parameter_size":"7.6B","quantization_level":"Q4_K_M","context_length":32768}}]}"#;
        let response: TagsResponse = serde_json::from_str(json).unwrap();
        let details = response.models[0].details.as_ref().unwrap();
        assert_eq!(details.parameter_size.as_deref(), Some("7.6B"));
        assert_eq!(details.quantization_level.as_deref(), Some("Q4_K_M"));
        assert_eq!(details.context_length, Some(32_768));
    }
}
