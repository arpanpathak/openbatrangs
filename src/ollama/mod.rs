//! # Minimal async HTTP client for the Ollama local model server
//!
//! This module wraps the endpoints used by openBatarangs:
//! - `GET  /api/tags`    — list installed models
//! - `POST /api/show`    — inspect model metadata
//! - `POST /api/chat`    — chat completion (streaming and non-streaming)
//! - `POST /api/pull`    — download a model from the Ollama registry
//!
//! It can also talk to an OpenAI-compatible server such as llama-server
//! (`/models`, `/props`, `/chat/completions`); see [`OllamaClient::openai`].
//!
//! The client uses `reqwest` with conservative timeouts so a hung local server
//! cannot freeze the TUI or the one-shot CLI.
//!
//! ## References
//!
//! - Ollama API reference: <https://github.com/ollama/ollama/blob/main/docs/api.md>
//! - `reqwest` streaming responses: <https://docs.rs/reqwest/latest/reqwest/struct.Response.html#method.bytes_stream>
//! - NDJSON (newline-delimited JSON): <https://ndjson.org/>

mod openai;
mod stream;
mod types;

pub(crate) use types::{
    ChatMessage, ChatRequest, Delta, OllamaModel, PullRequest, Role, TagsResponse,
};

use crate::constants::ollama::{
    API_CHAT_PATH, API_PULL_PATH, API_SHOW_PATH, API_TAGS_PATH, CONNECT_TIMEOUT_SECONDS,
    HTTP_TIMEOUT_SECONDS,
};
use anyhow::{anyhow, bail, Context, Result};
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
    /// Which HTTP API the server speaks.
    api: Api,
}

/// The HTTP API spoken by the model server.
#[derive(Clone)]
enum Api {
    /// Ollama's `/api/*` endpoints.
    Ollama,
    /// An OpenAI-compatible server such as llama-server.
    OpenAi {
        /// Bearer token sent with every request, if the server needs one.
        key: Option<String>,
        /// Ask the model to think before answering.
        think: bool,
    },
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
            api: Api::Ollama,
        })
    }

    /// Create a client for an OpenAI-compatible server such as llama-server.
    ///
    /// # Arguments
    /// - `base_url`: API address including `/v1`, e.g. `http://127.0.0.1:8079/v1`.
    /// - `key`: bearer token, if the server requires one.
    /// - `think`: ask thinking models to reason before answering.
    ///
    /// Idle connections are not reused: through an SSH tunnel a kept-alive
    /// connection can be closed between agent steps, and the next request
    /// would fail with "connection closed before message completed".
    pub fn openai(base_url: &str, key: Option<String>, think: bool) -> Result<Self> {
        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(HTTP_TIMEOUT_SECONDS))
            .connect_timeout(Duration::from_secs(CONNECT_TIMEOUT_SECONDS))
            .pool_max_idle_per_host(0)
            .build()
            .context("failed to build HTTP client")?;
        Ok(Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            http,
            api: Api::OpenAi { key, think },
        })
    }

    /// Whether this client talks to an OpenAI-compatible server.
    pub fn is_openai(&self) -> bool {
        matches!(self.api, Api::OpenAi { .. })
    }

    /// Attach the bearer token, if any.
    fn authorize(&self, request: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        match &self.api {
            Api::OpenAi { key: Some(key), .. } => request.bearer_auth(key),
            _ => request,
        }
    }

    /// `GET` a JSON body from an OpenAI-compatible server.
    async fn get_openai_json(&self, url: String) -> Result<Value> {
        let response = self
            .authorize(self.http.get(&url))
            .send()
            .await
            .with_context(|| format!("failed to reach {url}"))?;
        if !response.status().is_success() {
            return Err(anyhow!("{url} returned HTTP {}", response.status()));
        }
        response
            .json()
            .await
            .with_context(|| format!("failed to parse the response from {url}"))
    }

    /// Check whether the Ollama server is reachable.
    ///
    /// # Returns
    /// `true` if `GET /api/tags` succeeds, otherwise `false`.
    pub async fn is_available(&self) -> bool {
        let path = if self.is_openai() {
            "/models"
        } else {
            API_TAGS_PATH
        };
        self.authorize(self.http.get(format!("{}{path}", self.base_url)))
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
        if self.is_openai() {
            let body = self
                .get_openai_json(format!("{}/models", self.base_url))
                .await?;
            return Ok(openai::models_from_list(&body));
        }
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
        if self.is_openai() {
            let root = openai::server_root(&self.base_url);
            return match self.get_openai_json(format!("{root}/props")).await {
                Ok(props) => Ok(openai::show_from_props(&props)),
                Err(_) => Ok(serde_json::json!({})),
            };
        }
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

    /// Stream a chat completion as a sequence of deltas.
    ///
    /// # Arguments
    /// - `request`: chat request; `stream` is forced to `true`.
    ///
    /// # Returns
    /// A stream of deltas: reply text, or reasoning from thinking models on
    /// an OpenAI-compatible server. `Err` if the stream fails.
    pub async fn chat_stream(
        &self,
        mut request: ChatRequest,
    ) -> Result<std::pin::Pin<Box<dyn Stream<Item = Result<Delta>> + Send + 'static>>> {
        if let Api::OpenAi { think, .. } = self.api {
            return self.openai_chat_stream(&request, think).await;
        }
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
                            return Some((Ok(Delta::Content(content)), (byte_stream, buffer)));
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
        Ok(Box::pin(stream))
    }

    /// Stream a chat completion from an OpenAI-compatible server.
    async fn openai_chat_stream(
        &self,
        request: &ChatRequest,
        think: bool,
    ) -> Result<std::pin::Pin<Box<dyn Stream<Item = Result<Delta>> + Send + 'static>>> {
        let url = format!("{}/chat/completions", self.base_url);
        let response = self
            .authorize(self.http.post(&url))
            .json(&openai::chat_body(request, think))
            .send()
            .await
            .with_context(|| format!("failed to call {url}"))?;

        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            return Err(anyhow!("{url} returned HTTP {status}: {text}"));
        }

        let byte_stream = response.bytes_stream();
        let stream = futures_util::stream::unfold(
            (byte_stream, String::new()),
            |(mut byte_stream, mut buffer)| async move {
                loop {
                    match openai::drain_events(&mut buffer) {
                        openai::EventDrain::Delta(delta) => {
                            return Some((Ok(delta), (byte_stream, buffer)));
                        }
                        openai::EventDrain::Done => return None,
                        openai::EventDrain::NeedMore => {}
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
        Ok(Box::pin(stream))
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
        if self.is_openai() {
            bail!(
                "'{name}' can't be pulled: {} is not an Ollama server; load models on that server instead",
                self.base_url
            );
        }
        on_status(&format!(
            "⬇️  Pulling model '{name}' from Ollama registry..."
        ));
        let response = self
            .http
            .post(format!("{}{API_PULL_PATH}", self.base_url))
            .json(&PullRequest {
                name: name.to_string(),
                stream: true,
            })
            .send()
            .await
            .context("failed to call Ollama /api/pull")?;

        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            return Err(anyhow!("Ollama /api/pull returned HTTP {status}: {text}"));
        }

        let mut byte_stream = Box::pin(response.bytes_stream());
        let mut buffer = String::new();
        while let Some(bytes) = byte_stream.next().await {
            let bytes = bytes.context("failed to read Ollama /api/pull stream")?;
            buffer.push_str(&String::from_utf8_lossy(&bytes));
            while let Some(event) = stream::drain_pull_line(&mut buffer) {
                match event {
                    stream::PullLine::Status(status) => on_status(&status),
                    stream::PullLine::Error(error) => {
                        return Err(anyhow!("Ollama /api/pull failed: {error}"));
                    }
                    stream::PullLine::Done => {
                        on_status("✅ Pull finished");
                        return Ok(());
                    }
                }
            }
        }

        Err(anyhow!(
            "Ollama /api/pull stream ended before reporting success"
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{spawn_mock_server, MockResponse};

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
                role: Role::User,
                content: "hello".to_string(),
            }],
            stream: true,
            keep_alive: None,
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

    // ========================================================================
    // Local mock Ollama HTTP server
    //
    // `spawn_mock_server` and `MockResponse` live in `crate::test_support` so
    // the same real-HTTP seam is reusable by command/model tests.
    // ========================================================================

    fn sample_request() -> ChatRequest {
        ChatRequest {
            model: "qwen2.5-coder:3b".to_string(),
            messages: vec![ChatMessage {
                role: Role::User,
                content: "hello".to_string(),
            }],
            stream: false,
            keep_alive: None,
            format: None,
            options: None,
        }
    }

    #[tokio::test]
    async fn is_available_true_on_success() {
        let base_url = spawn_mock_server(|path| {
            assert_eq!(path, "/api/tags");
            MockResponse::json("200 OK", r#"{"models":[]}"#)
        })
        .await;
        let client = OllamaClient::new(&base_url).unwrap();
        assert!(client.is_available().await);
    }

    #[tokio::test]
    async fn is_available_false_on_error() {
        let base_url =
            spawn_mock_server(|_| MockResponse::text("503 Service Unavailable", "down")).await;
        let client = OllamaClient::new(&base_url).unwrap();
        assert!(!client.is_available().await);
    }

    #[tokio::test]
    async fn tags_parses_models_from_mock_server() {
        let base_url = spawn_mock_server(|path| {
            assert_eq!(path, "/api/tags");
            MockResponse::json(
                "200 OK",
                r#"{"models":[{"name":"qwen2.5-coder:3b","size":123}]}"#,
            )
        })
        .await;
        let client = OllamaClient::new(&base_url).unwrap();
        let tags = client.tags().await.unwrap();
        assert_eq!(tags.len(), 1);
        assert_eq!(tags[0].name, "qwen2.5-coder:3b");
        assert_eq!(tags[0].size, 123);
    }

    #[tokio::test]
    async fn tags_returns_error_on_http_failure() {
        let base_url =
            spawn_mock_server(|_| MockResponse::text("500 Internal Server Error", "boom")).await;
        let client = OllamaClient::new(&base_url).unwrap();
        assert!(client.tags().await.is_err());
    }

    #[tokio::test]
    async fn show_returns_model_metadata() {
        let base_url = spawn_mock_server(|path| {
            assert_eq!(path, "/api/show");
            MockResponse::json("200 OK", r#"{"model_info":{"llama.context_length":32768}}"#)
        })
        .await;
        let client = OllamaClient::new(&base_url).unwrap();
        let value = client.show("qwen2.5-coder:7b").await.unwrap();
        assert_eq!(value["model_info"]["llama.context_length"], 32_768);
    }

    #[tokio::test]
    async fn chat_stream_yields_content_deltas() {
        use futures_util::StreamExt;

        let base_url = spawn_mock_server(|path| {
            assert_eq!(path, "/api/chat");
            MockResponse::json(
                "200 OK",
                "{\"message\":{\"content\":\"hello\"}}\n{\"message\":{\"content\":\" world\"}}\n{\"done\":true}\n",
            )
        })
        .await;
        let client = OllamaClient::new(&base_url).unwrap();
        let mut stream = client.chat_stream(sample_request()).await.unwrap();
        let mut parts = Vec::new();
        while let Some(chunk) = stream.next().await {
            parts.push(chunk.unwrap());
        }
        assert_eq!(
            parts,
            vec![
                Delta::Content("hello".to_string()),
                Delta::Content(" world".to_string())
            ]
        );
    }

    #[tokio::test]
    async fn chat_stream_returns_error_on_http_failure() {
        let base_url =
            spawn_mock_server(|_| MockResponse::text("400 Bad Request", "bad model")).await;
        let client = OllamaClient::new(&base_url).unwrap();
        assert!(client.chat_stream(sample_request()).await.is_err());
    }

    #[tokio::test]
    async fn pull_reports_statuses_and_finishes() {
        use std::sync::{Arc, Mutex};

        let base_url = spawn_mock_server(|path| {
            assert_eq!(path, "/api/pull");
            MockResponse::json(
                "200 OK",
                "{\"status\":\"pulling manifest\"}\n{\"status\":\"downloading\",\"total\":100,\"completed\":50}\n{\"status\":\"success\"}\n",
            )
        })
        .await;
        let client = OllamaClient::new(&base_url).unwrap();
        let statuses = Arc::new(Mutex::new(Vec::new()));
        let captured = Arc::clone(&statuses);
        client
            .pull("qwen2.5-coder:3b", &move |msg| {
                captured.lock().unwrap().push(msg.to_string());
            })
            .await
            .unwrap();
        let statuses = statuses.lock().unwrap();
        assert!(statuses.iter().any(|s| s.contains("pulling manifest")));
        assert!(statuses.iter().any(|s| s.contains("50%")));
        assert!(statuses.iter().any(|s| s.contains("Pull finished")));
    }

    #[tokio::test]
    async fn openai_chat_stream_yields_thinking_then_content() {
        use futures_util::StreamExt;

        let base_url = spawn_mock_server(|path| {
            assert_eq!(path, "/chat/completions");
            MockResponse::json(
                "200 OK",
                concat!(
                    "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"hmm\"}}]}\n\n",
                    "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n",
                    "data: [DONE]\n\n",
                ),
            )
        })
        .await;
        let client = OllamaClient::openai(&base_url, Some("key".to_string()), true).unwrap();
        let mut stream = client.chat_stream(sample_request()).await.unwrap();
        let mut parts = Vec::new();
        while let Some(chunk) = stream.next().await {
            parts.push(chunk.unwrap());
        }
        assert_eq!(
            parts,
            vec![
                Delta::Thinking("hmm".to_string()),
                Delta::Content("hi".to_string())
            ]
        );
    }

    #[tokio::test]
    async fn openai_tags_and_pull() {
        let base_url = spawn_mock_server(|path| {
            assert_eq!(path, "/models");
            MockResponse::json("200 OK", r#"{"data":[{"id":"/m/Nemotron-Q8_0.gguf"}]}"#)
        })
        .await;
        let client = OllamaClient::openai(&base_url, None, false).unwrap();
        assert!(client.is_openai());
        assert!(client.is_available().await);
        let tags = client.tags().await.unwrap();
        assert_eq!(tags[0].name, "Nemotron-Q8_0");
        assert!(client.pull("x", &|_| {}).await.is_err());
    }
}
