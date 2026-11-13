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

mod channel;
mod openai;
mod reader;
mod stream;
mod types;

pub(crate) use channel::answer_without_channels;
pub(crate) use types::{
    ChatMessage, ChatRequest, Delta, OllamaModel, PullRequest, Role, TagsResponse, ToolCallDelta,
};

use crate::constants::ollama::{
    API_CHAT_PATH, API_PULL_PATH, API_SHOW_PATH, API_TAGS_PATH, CONNECT_TIMEOUT_SECONDS,
    HTTP_TIMEOUT_SECONDS,
};
use anyhow::{Context, Result, anyhow, bail};
use reader::DeltaStream;
use reqwest::{Response, StatusCode};
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
        /// Send `enable_thinking: true` with each chat request, so thinking
        /// models reason before the reply.
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
        let http = http_client(reqwest::Client::builder())?;

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
    /// - `think`: send `enable_thinking: true`, so thinking models reason
    ///   before the reply.
    ///
    /// Idle connections are not reused: through an SSH tunnel a kept-alive
    /// connection can be closed between agent steps, and the next request
    /// would fail with "connection closed before message completed".
    pub fn openai(base_url: &str, key: Option<String>, think: bool) -> Result<Self> {
        let http = http_client(reqwest::Client::builder().pool_max_idle_per_host(0))?;

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

    /// Whether the model is asked to think before answering.
    ///
    /// Only an OpenAI-compatible server takes this flag, and only `--no-think`
    /// clears it. Showing or hiding the reasoning is a separate concern: the
    /// model must keep thinking for its instructions to hold, so a display
    /// toggle never changes what is sent.
    pub fn requests_thinking(&self) -> bool {
        matches!(self.api, Api::OpenAi { think: true, .. })
    }

    /// Attach the bearer token, if any.
    fn authorize(&self, request: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        match &self.api {
            Api::OpenAi { key: Some(key), .. } => request.bearer_auth(key),
            Api::Ollama | Api::OpenAi { key: None, .. } => request,
        }
    }

    /// `GET` a JSON body from an OpenAI-compatible server.
    async fn get_openai_json(&self, url: String) -> Result<Value> {
        let response = self
            .authorize(self.http.get(&url))
            .send()
            .await
            .with_context(|| format!("failed to reach {url}"))?;
        let response =
            require_success(response).map_err(|status| anyhow!("{url} returned HTTP {status}"))?;

        response
            .json()
            .await
            .with_context(|| format!("failed to parse the response from {url}"))
    }

    /// The body of `GET {server root}{path}`, where the server root is the
    /// base URL without `/v1`, sent with the bearer key, or `None` when the
    /// server cannot be reached or answers with an error status.
    ///
    /// # Arguments
    /// - `path`: a path such as `/metrics` or `/api/ps`.
    pub async fn server_text(&self, path: &str) -> Option<String> {
        let url = format!("{}{path}", openai::server_root(&self.base_url));
        let response = self.authorize(self.http.get(url)).send().await.ok()?;
        let response = require_success(response).ok()?;

        response.text().await.ok()
    }

    /// Check whether the Ollama server is reachable.
    ///
    /// # Returns
    /// `true` if `GET /api/tags` (or `/models` on an OpenAI-compatible
    /// server) succeeds, otherwise `false`.
    pub async fn is_available(&self) -> bool {
        let path = if self.is_openai() {
            "/models"
        } else {
            API_TAGS_PATH
        };

        self.authorize(self.http.get(format!("{}{path}", self.base_url)))
            .send()
            .await
            .is_ok_and(|response| response.status().is_success())
    }

    /// List all installed models.
    ///
    /// # Returns
    /// A vector of installed models, or an error if the server is unreachable
    /// or returns a non-success status.
    pub async fn tags(&self) -> Result<Vec<OllamaModel>> {
        if self.is_openai() {
            return self.openai_tags().await;
        }

        self.ollama_tags().await
    }

    /// The models listed by an OpenAI-compatible server's `/models`.
    async fn openai_tags(&self) -> Result<Vec<OllamaModel>> {
        let body = self
            .get_openai_json(format!("{}/models", self.base_url))
            .await?;

        Ok(openai::models_from_list(&body))
    }

    /// The models listed by Ollama's `/api/tags`.
    async fn ollama_tags(&self) -> Result<Vec<OllamaModel>> {
        let response = self
            .http
            .get(format!("{}{API_TAGS_PATH}", self.base_url))
            .send()
            .await
            .context("failed to reach Ollama server; is `ollama serve` running?")?;
        let response = require_success(response)
            .map_err(|status| anyhow!("Ollama /api/tags returned HTTP {status}"))?;

        let body: TagsResponse = response
            .json()
            .await
            .context("failed to parse Ollama /api/tags response")?;

        Ok(body.models)
    }

    /// Metadata for a model on an OpenAI-compatible server: its entry in
    /// `/models` when that carries the context window (TensorRT Edge-LLM),
    /// else `/props` (llama-server), else nothing.
    async fn show_openai(&self, name: &str) -> Value {
        let listed = self
            .get_openai_json(format!("{}/models", self.base_url))
            .await
            .ok()
            .and_then(|list| openai::show_from_models(&list, name));

        match listed {
            Some(show) => show,
            None => self.show_from_props().await,
        }
    }

    /// The context window from llama-server's `/props`, shaped like
    /// `/api/show`; an empty object when the server has no `/props`.
    async fn show_from_props(&self) -> Value {
        let root = openai::server_root(&self.base_url);

        self.get_openai_json(format!("{root}/props"))
            .await
            .ok()
            .map_or_else(
                || serde_json::json!({}),
                |props| openai::show_from_props(&props),
            )
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
            return Ok(self.show_openai(name).await);
        }

        self.show_ollama(name).await
    }

    /// The metadata Ollama's `/api/show` returns for `name`.
    async fn show_ollama(&self, name: &str) -> Result<Value> {
        let response = self
            .http
            .post(format!("{}{API_SHOW_PATH}", self.base_url))
            .json(&serde_json::json!({ "name": name }))
            .send()
            .await
            .context("failed to call Ollama /api/show")?;
        let response = require_success(response).map_err(|status| {
            anyhow!("Ollama /api/show returned HTTP {status} for model '{name}'")
        })?;

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
    pub async fn chat_stream(&self, mut request: ChatRequest) -> Result<DeltaStream> {
        if let Api::OpenAi { think, .. } = self.api {
            return self.openai_chat_stream(&request, think).await;
        }

        request.stream = true;

        self.ollama_chat_stream(&request).await
    }

    /// Stream a chat completion from Ollama's `/api/chat`.
    async fn ollama_chat_stream(&self, request: &ChatRequest) -> Result<DeltaStream> {
        let response = self
            .http
            .post(format!("{}{API_CHAT_PATH}", self.base_url))
            .json(request)
            .send()
            .await
            .context("failed to call Ollama /api/chat (stream)")?;
        let response = require_success_with_body(response, "Ollama /api/chat").await?;

        Ok(reader::delta_stream(response, stream::drain_complete_lines))
    }

    /// Stream a chat completion from an OpenAI-compatible server.
    async fn openai_chat_stream(&self, request: &ChatRequest, think: bool) -> Result<DeltaStream> {
        self.chat_stream_body(openai::chat_body(request, think))
            .await
    }

    /// Stream a chat completion from an OpenAI-compatible server using a
    /// fully-formed request body.
    ///
    /// The native tool-calling loop builds its own body (a `tools` array,
    /// `tool_choice`, and `tool`/`tool_call_id` messages) instead of going
    /// through [`ChatRequest`], so this is the shared transport for both paths.
    ///
    /// # Arguments
    /// - `body`: the complete `/chat/completions` JSON body, `stream: true`.
    ///
    /// # Returns
    /// A stream of [`Delta`]s, or an error if the server cannot be reached or
    /// answers with a non-success status.
    pub async fn chat_stream_body(&self, body: Value) -> Result<DeltaStream> {
        let url = format!("{}/chat/completions", self.base_url);
        let response = self
            .authorize(self.http.post(&url))
            .json(&body)
            .send()
            .await
            .with_context(|| format!("failed to call {url}"))?;
        let response = require_success_with_body(response, &url).await?;

        Ok(reader::delta_stream(response, openai::drain_events))
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
        self.refuse_openai_pull(name)?;
        on_status(&format!(
            "⬇️  Pulling model '{name}' from Ollama registry..."
        ));
        let response = self.start_pull(name).await?;

        reader::follow_pull(response, on_status).await
    }

    /// An error when this client talks to an OpenAI-compatible server, which
    /// cannot download models.
    fn refuse_openai_pull(&self, name: &str) -> Result<()> {
        if self.is_openai() {
            bail!(
                "'{name}' can't be pulled: {} is not an Ollama server; load models on that server instead",
                self.base_url
            );
        }

        Ok(())
    }

    /// Send the pull request for `name` to `/api/pull` and return the
    /// response that streams its progress.
    async fn start_pull(&self, name: &str) -> Result<Response> {
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

        require_success_with_body(response, "Ollama /api/pull").await
    }
}

/// The HTTP client built from `builder` with the request and connect
/// timeouts every model server call uses.
fn http_client(builder: reqwest::ClientBuilder) -> Result<reqwest::Client> {
    builder
        .timeout(Duration::from_secs(HTTP_TIMEOUT_SECONDS))
        .connect_timeout(Duration::from_secs(CONNECT_TIMEOUT_SECONDS))
        .build()
        .context("failed to build HTTP client")
}

/// `response` when its status is a success, else the status, so each caller
/// words its own error.
fn require_success(response: Response) -> std::result::Result<Response, StatusCode> {
    let status = response.status();

    if !status.is_success() {
        return Err(status);
    }

    Ok(response)
}

/// `response` when its status is a success, else an error that holds the
/// status and the body the server sent: `{endpoint} returned HTTP {status}:
/// {body}`.
async fn require_success_with_body(response: Response, endpoint: &str) -> Result<Response> {
    let status = response.status();

    if status.is_success() {
        return Ok(response);
    }

    let text = response.text().await.unwrap_or_default();

    Err(anyhow!("{endpoint} returned HTTP {status}: {text}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{MockResponse, spawn_mock_server};
    use crate::tools::test_server::{cut_short, spawn_raw_server};
    use futures_util::{StreamExt, TryStreamExt};
    use std::sync::{Mutex, PoisonError};

    /// A one-message chat request with no options.
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

    /// Add `status` to the statuses a pull reported.
    fn record(statuses: &Mutex<Vec<String>>, status: &str) {
        let mut seen = statuses.lock().unwrap_or_else(PoisonError::into_inner);

        seen.push(status.to_string());
    }

    /// Pull a model from the server at `base_url`, returning the outcome and
    /// every status line reported along the way.
    async fn pull_from(base_url: &str) -> Result<(Result<()>, Vec<String>)> {
        let client = OllamaClient::new(base_url)?;
        let statuses = Mutex::new(Vec::new());

        let outcome = client
            .pull("qwen2.5-coder:3b", &|status: &str| {
                record(&statuses, status)
            })
            .await;

        Ok((outcome, statuses.into_inner()?))
    }

    #[test]
    fn client_strips_trailing_slashes_from_base_url() -> Result<()> {
        let client = OllamaClient::new("http://localhost:11434///")?;

        assert_eq!(client.base_url, "http://localhost:11434");
        Ok(())
    }

    #[test]
    fn client_accepts_plain_url() -> Result<()> {
        let client = OllamaClient::new("http://127.0.0.1:11434")?;

        assert_eq!(client.base_url, "http://127.0.0.1:11434");
        Ok(())
    }

    #[test]
    fn chat_request_serializes_expected_shape() -> Result<()> {
        let request = ChatRequest {
            stream: true,
            options: Some(serde_json::json!({"temperature": 0.7})),
            ..sample_request()
        };

        let json = serde_json::to_value(&request)?;

        assert_eq!(json["model"], "qwen2.5-coder:3b");
        assert_eq!(json["stream"], true);
        assert_eq!(json["messages"][0]["role"], "user");
        assert_eq!(json["options"]["temperature"], 0.7);
        Ok(())
    }

    #[test]
    fn tags_response_deserializes_with_missing_details() -> Result<()> {
        let json = r#"{"models":[{"name":"qwen2.5-coder:3b","size":123}]}"#;

        let response: TagsResponse = serde_json::from_str(json)?;

        assert_eq!(response.models.len(), 1);
        assert_eq!(response.models[0].name, "qwen2.5-coder:3b");
        assert_eq!(response.models[0].size, 123);
        assert!(response.models[0].details.is_none());
        Ok(())
    }

    #[test]
    fn tags_response_deserializes_with_details() -> Result<()> {
        let json = r#"{"models":[{"name":"qwen2.5-coder:7b","size":1,"details":{"parameter_size":"7.6B","quantization_level":"Q4_K_M","context_length":32768}}]}"#;

        let response: TagsResponse = serde_json::from_str(json)?;
        let details = response.models[0].details.clone().unwrap_or_default();

        assert_eq!(details.parameter_size.as_deref(), Some("7.6B"));
        assert_eq!(details.quantization_level.as_deref(), Some("Q4_K_M"));
        assert_eq!(details.context_length, Some(32_768));
        Ok(())
    }

    #[tokio::test]
    async fn is_available_true_on_success() -> Result<()> {
        let base_url = spawn_mock_server(|path| {
            assert_eq!(path, "/api/tags");
            MockResponse::json("200 OK", r#"{"models":[]}"#)
        })
        .await;
        let client = OllamaClient::new(&base_url)?;

        assert!(client.is_available().await);
        Ok(())
    }

    #[tokio::test]
    async fn is_available_false_on_error() -> Result<()> {
        let base_url =
            spawn_mock_server(|_path| MockResponse::text("503 Service Unavailable", "down")).await;
        let client = OllamaClient::new(&base_url)?;

        assert!(!client.is_available().await);
        Ok(())
    }

    #[tokio::test]
    async fn tags_parses_models_from_mock_server() -> Result<()> {
        let base_url = spawn_mock_server(|path| {
            assert_eq!(path, "/api/tags");
            MockResponse::json(
                "200 OK",
                r#"{"models":[{"name":"qwen2.5-coder:3b","size":123}]}"#,
            )
        })
        .await;
        let client = OllamaClient::new(&base_url)?;

        let tags = client.tags().await?;

        assert_eq!(tags.len(), 1);
        assert_eq!(tags[0].name, "qwen2.5-coder:3b");
        assert_eq!(tags[0].size, 123);
        Ok(())
    }

    #[tokio::test]
    async fn tags_returns_error_on_http_failure() -> Result<()> {
        let base_url =
            spawn_mock_server(|_path| MockResponse::text("500 Internal Server Error", "boom"))
                .await;
        let client = OllamaClient::new(&base_url)?;

        let tags = client.tags().await;

        assert!(format!("{tags:?}").contains("Ollama /api/tags returned HTTP 500"));
        Ok(())
    }

    #[tokio::test]
    async fn show_returns_model_metadata() -> Result<()> {
        let base_url = spawn_mock_server(|path| {
            assert_eq!(path, "/api/show");
            MockResponse::json("200 OK", r#"{"model_info":{"llama.context_length":32768}}"#)
        })
        .await;
        let client = OllamaClient::new(&base_url)?;

        let value = client.show("qwen2.5-coder:7b").await?;

        assert_eq!(value["model_info"]["llama.context_length"], 32_768);
        Ok(())
    }

    #[tokio::test]
    async fn show_names_the_model_on_http_failure() -> Result<()> {
        let base_url = spawn_mock_server(|_path| MockResponse::text("404 Not Found", "")).await;
        let client = OllamaClient::new(&base_url)?;

        let shown = client.show("missing:7b").await;

        assert!(
            format!("{shown:?}").contains("returned HTTP 404 Not Found for model 'missing:7b'")
        );
        Ok(())
    }

    #[tokio::test]
    async fn chat_stream_yields_content_deltas() -> Result<()> {
        let base_url = spawn_mock_server(|path| {
            assert_eq!(path, "/api/chat");
            MockResponse::json(
                "200 OK",
                "{\"message\":{\"content\":\"hello\"}}\n{\"message\":{\"content\":\" world\"}}\n{\"done\":true}\n",
            )
        })
        .await;
        let client = OllamaClient::new(&base_url)?;

        let parts: Vec<Delta> = client
            .chat_stream(sample_request())
            .await?
            .try_collect()
            .await?;

        assert_eq!(
            parts,
            vec![
                Delta::Content("hello".to_string()),
                Delta::Content(" world".to_string())
            ]
        );
        Ok(())
    }

    #[tokio::test]
    async fn chat_stream_ends_with_the_body_when_done_never_arrives() -> Result<()> {
        let base_url = spawn_mock_server(|_path| {
            MockResponse::json("200 OK", "{\"message\":{\"content\":\"cut\"}}\n")
        })
        .await;
        let client = OllamaClient::new(&base_url)?;

        let parts: Vec<Delta> = client
            .chat_stream(sample_request())
            .await?
            .try_collect()
            .await?;

        assert_eq!(parts, vec![Delta::Content("cut".to_string())]);
        Ok(())
    }

    #[tokio::test]
    async fn chat_stream_reports_a_connection_lost_mid_reply() -> Result<()> {
        let base_url =
            spawn_raw_server(|_path| cut_short("{\"message\":{\"content\":\"hi\"}}\n")).await?;
        let client = OllamaClient::new(&base_url)?;
        let mut stream = client.chat_stream(sample_request()).await?;

        let first = stream.next().await.transpose()?;
        let second = stream.next().await;

        assert_eq!(first, Some(Delta::Content("hi".to_string())));
        assert!(format!("{second:?}").contains("stream error"));
        Ok(())
    }

    #[tokio::test]
    async fn chat_stream_returns_error_on_http_failure() -> Result<()> {
        let base_url =
            spawn_mock_server(|_path| MockResponse::text("400 Bad Request", "bad model")).await;
        let client = OllamaClient::new(&base_url)?;

        let error = client.chat_stream(sample_request()).await.err();

        assert!(
            format!("{error:?}")
                .contains("Ollama /api/chat returned HTTP 400 Bad Request: bad model")
        );
        Ok(())
    }

    #[tokio::test]
    async fn pull_reports_statuses_and_finishes() -> Result<()> {
        let base_url = spawn_mock_server(|path| {
            assert_eq!(path, "/api/pull");
            MockResponse::json(
                "200 OK",
                "{\"status\":\"pulling manifest\"}\n{\"status\":\"downloading\",\"total\":100,\"completed\":50}\n{\"status\":\"success\"}\n",
            )
        })
        .await;

        let (outcome, statuses) = pull_from(&base_url).await?;

        outcome?;
        assert_eq!(
            statuses,
            [
                "⬇️  Pulling model 'qwen2.5-coder:3b' from Ollama registry...",
                "pulling manifest",
                "downloading 50%",
                "✅ Pull finished",
            ]
        );
        Ok(())
    }

    #[tokio::test]
    async fn pull_fails_on_an_error_line_or_an_early_end() -> Result<()> {
        let refused = spawn_mock_server(|_path| {
            MockResponse::json("200 OK", "{\"error\":\"model not found\"}\n")
        })
        .await;
        let ended = spawn_mock_server(|_path| {
            MockResponse::json("200 OK", "{\"status\":\"pulling manifest\"}\n")
        })
        .await;

        let (refused, _refused_statuses) = pull_from(&refused).await?;
        let (ended, _ended_statuses) = pull_from(&ended).await?;

        assert!(format!("{refused:?}").contains("Ollama /api/pull failed: model not found"));
        assert!(format!("{ended:?}").contains("ended before reporting success"));
        Ok(())
    }

    #[tokio::test]
    async fn pull_fails_on_http_and_read_errors() -> Result<()> {
        let rejected =
            spawn_mock_server(|_path| MockResponse::text("500 Internal Server Error", "disk full"))
                .await;
        let broken = spawn_raw_server(|_path| cut_short("{\"status\":\"pulling\"}\n")).await?;

        let (rejected, _rejected_statuses) = pull_from(&rejected).await?;
        let (broken, _broken_statuses) = pull_from(&broken).await?;

        assert!(format!("{rejected:?}").contains("Ollama /api/pull returned HTTP 500"));
        assert!(format!("{broken:?}").contains("failed to read Ollama /api/pull stream"));
        Ok(())
    }

    #[tokio::test]
    async fn openai_chat_stream_yields_thinking_then_content() -> Result<()> {
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
        let client = OllamaClient::openai(&base_url, Some("key".to_string()), true)?;

        let parts: Vec<Delta> = client
            .chat_stream(sample_request())
            .await?
            .try_collect()
            .await?;

        assert_eq!(
            parts,
            vec![
                Delta::Thinking("hmm".to_string()),
                Delta::Content("hi".to_string())
            ]
        );
        Ok(())
    }

    #[tokio::test]
    async fn openai_chat_stream_drops_channel_markers_split_across_deltas() -> Result<()> {
        let base_url = spawn_mock_server(|_path| {
            MockResponse::json(
                "200 OK",
                concat!(
                    "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"<|channel>\"}}]}\n\n",
                    "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"thought\"}}]}\n\n",
                    "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"\\nPlan\"}}]}\n\n",
                    "data: {\"choices\":[{\"delta\":{\"content\":\"<channel|>\"}}]}\n\n",
                    "data: {\"choices\":[{\"delta\":{\"content\":\"Done\"}}]}\n\n",
                    "data: [DONE]\n\n",
                ),
            )
        })
        .await;
        let client = OllamaClient::openai(&base_url, None, true)?;

        let parts: Vec<Delta> = client
            .chat_stream(sample_request())
            .await?
            .try_collect()
            .await?;

        assert_eq!(
            parts,
            vec![
                Delta::Thinking("\nPlan".to_string()),
                Delta::Content("Done".to_string())
            ]
        );
        Ok(())
    }

    #[tokio::test]
    async fn openai_chat_stream_returns_error_on_http_failure() -> Result<()> {
        let base_url =
            spawn_mock_server(|_path| MockResponse::text("503 Service Unavailable", "loading"))
                .await;
        let client = OllamaClient::openai(&base_url, None, false)?;

        let error = client.chat_stream(sample_request()).await.err();

        assert!(
            format!("{error:?}")
                .contains("/chat/completions returned HTTP 503 Service Unavailable: loading")
        );
        Ok(())
    }

    #[tokio::test]
    async fn openai_tags_and_pull() -> Result<()> {
        let base_url = spawn_mock_server(|path| {
            assert_eq!(path, "/models");
            MockResponse::json("200 OK", r#"{"data":[{"id":"/m/Nemotron-Q8_0.gguf"}]}"#)
        })
        .await;
        let client = OllamaClient::openai(&base_url, None, false)?;

        let tags = client.tags().await?;

        assert!(client.is_openai());
        assert!(client.is_available().await);
        assert_eq!(tags[0].name, "Nemotron-Q8_0");
        assert!(client.refuse_openai_pull("x").is_err());
        Ok(())
    }

    #[tokio::test]
    async fn openai_tags_report_an_http_failure() -> Result<()> {
        let base_url = spawn_mock_server(|_path| MockResponse::text("401 Unauthorized", "")).await;
        let client = OllamaClient::openai(&base_url, Some("wrong".to_string()), false)?;

        let tags = client.tags().await;

        assert!(format!("{tags:?}").contains("/models returned HTTP 401 Unauthorized"));
        Ok(())
    }

    #[tokio::test]
    async fn an_unreachable_server_is_named_in_the_error() -> Result<()> {
        let client = OllamaClient::openai("http://127.0.0.1:1/v1", None, false)?;

        let tags = client.tags().await;
        let chat = client.chat_stream(sample_request()).await.err();

        assert!(format!("{tags:?}").contains("failed to reach http://127.0.0.1:1/v1/models"));
        assert!(
            format!("{chat:?}").contains("failed to call http://127.0.0.1:1/v1/chat/completions")
        );
        Ok(())
    }

    #[tokio::test]
    async fn openai_tags_report_a_body_that_is_not_json() -> Result<()> {
        let base_url = spawn_mock_server(|_path| MockResponse::text("200 OK", "<html>")).await;
        let client = OllamaClient::openai(&base_url, None, false)?;

        let tags = client.tags().await;

        assert!(format!("{tags:?}").contains("failed to parse the response from"));
        Ok(())
    }

    #[tokio::test]
    async fn openai_show_reads_the_context_from_the_list_then_from_props() -> Result<()> {
        let base_url = spawn_mock_server(|path| match path {
            "/v1/models" => MockResponse::json(
                "200 OK",
                r#"{"data":[{"id":"qwen","max_model_len":65536},{"id":"gguf"}]}"#,
            ),
            _ => MockResponse::json(
                "200 OK",
                r#"{"default_generation_settings":{"n_ctx":4096}}"#,
            ),
        })
        .await;
        let client = OllamaClient::openai(&format!("{base_url}/v1"), None, false)?;

        let listed = client.show("qwen").await?;
        let from_props = client.show("gguf").await?;

        assert_eq!(listed["details"]["context_length"], 65_536);
        assert_eq!(from_props["details"]["context_length"], 4_096);
        Ok(())
    }

    #[tokio::test]
    async fn openai_show_is_empty_when_the_server_says_nothing() -> Result<()> {
        let base_url = spawn_mock_server(|_path| MockResponse::text("404 Not Found", "no")).await;
        let client = OllamaClient::openai(&format!("{base_url}/v1"), None, false)?;

        assert_eq!(client.show("qwen").await?, serde_json::json!({}));
        Ok(())
    }

    #[test]
    fn only_the_openai_flag_decides_whether_the_model_thinks() -> Result<()> {
        let thinking = OllamaClient::openai("http://127.0.0.1:8079/v1", None, true)?;
        let not_thinking = OllamaClient::openai("http://127.0.0.1:8079/v1", None, false)?;
        let ollama = OllamaClient::new("http://127.0.0.1:11434")?;

        assert!(thinking.requests_thinking());
        assert!(!not_thinking.requests_thinking());
        assert!(!ollama.requests_thinking());
        Ok(())
    }

    /// Answer one connection on `listener` with the request it received as
    /// the body, so a test can read the headers the client sent.
    async fn echo_one_request(listener: tokio::net::TcpListener) -> Result<()> {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        let (mut socket, _peer) = listener.accept().await?;
        let mut request = [0u8; 4096];
        let read = socket.read(&mut request).await?;
        let request = String::from_utf8_lossy(&request[..read]).to_string();

        let reply = crate::tools::test_server::response("200 OK", "text/plain", &request);
        socket.write_all(reply.as_bytes()).await?;

        Ok(())
    }

    #[tokio::test]
    async fn server_text_reads_the_server_root_with_the_key() -> Result<()> {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
        let base_url = format!("http://{}/v1", listener.local_addr()?);
        tokio::spawn(echo_one_request(listener));
        let client = OllamaClient::openai(&base_url, Some("secret".to_string()), true)?;

        let request = client.server_text("/metrics").await.unwrap_or_default();

        assert!(request.starts_with("GET /metrics HTTP/1.1"));
        assert!(
            request
                .to_lowercase()
                .contains("authorization: bearer secret")
        );
        Ok(())
    }

    #[tokio::test]
    async fn server_text_is_none_on_an_error_status_or_no_server() -> Result<()> {
        let base_url = spawn_mock_server(|_path| MockResponse::text("404 Not Found", "no")).await;
        let answering = OllamaClient::new(&base_url)?;
        let closed = OllamaClient::new("http://127.0.0.1:1")?;

        assert_eq!(answering.server_text("/api/ps").await, None);
        assert_eq!(closed.server_text("/api/ps").await, None);
        Ok(())
    }
}
