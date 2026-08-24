//! Data types exchanged with the Ollama HTTP API.

use serde::{Deserialize, Serialize};
use serde_json::Value;

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
    /// How long to keep the model loaded after this request (Ollama `keep_alive`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub keep_alive: Option<Value>,
    /// Optional response format hint (e.g. `json`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub format: Option<Value>,
    /// Optional sampling options (`temperature`, `num_ctx`, ...).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub options: Option<Value>,
}

/// Payload for `POST /api/pull`.
#[derive(Serialize, Debug)]
pub struct PullRequest {
    /// Model tag to download, e.g. `qwen2.5-coder:3b`.
    pub name: String,
    /// Whether to stream pull progress as NDJSON status events.
    pub stream: bool,
}

/// A meaningful event extracted from one Ollama NDJSON stream line.
pub enum StreamLine {
    /// Assistant content delta.
    Content(String),
    /// Terminal `done: true` marker.
    Done,
}
