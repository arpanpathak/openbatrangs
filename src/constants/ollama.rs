//! Ollama HTTP API constants.

use std::time::Duration;

/// Longest time (seconds) the model server may stay silent: before its
/// response starts, or between two chunks of a streamed reply.
///
/// It limits silence, not the whole request, so a slow reply that keeps
/// streaming is never cut off. It is high (10 min) because loading a large
/// model can take minutes before the first byte.
pub const READ_TIMEOUT: Duration = Duration::from_secs(600);

/// Context window asked of an Ollama server when `--max-ctx` is not given
/// and the model's own window is larger.
///
/// Ollama reserves GPU memory for the whole window it is asked for, and
/// newer models have windows of 128K tokens or more. A window that large can
/// push the model off a 16 GB GPU onto the CPU. 32K tokens fits a mid-size
/// model on such a GPU and is enough for most coding tasks. OpenAI-compatible
/// servers set their window when they start, so they get no default cap.
pub const DEFAULT_OLLAMA_CONTEXT_CAP: u64 = 32_768;

/// Maximum time (seconds) to wait for the TCP handshake with the Ollama server.
/// A short timeout ensures the CLI fails fast when Ollama is not running.
pub const CONNECT_TIMEOUT_SECONDS: u64 = 5;

/// How long Ollama should keep the model resident after a request.
///
/// A long keep-alive avoids the cold model load that makes the first reply
/// after an idle period appear stuck on "thinking...".
pub const KEEP_ALIVE: &str = "30m";

/// API path for listing installed model tags (`GET /api/tags`).
pub const API_TAGS_PATH: &str = "/api/tags";
/// API path for retrieving model metadata (`POST /api/show`).
pub const API_SHOW_PATH: &str = "/api/show";
/// API path for chat completions (`POST /api/chat`).
pub const API_CHAT_PATH: &str = "/api/chat";
/// API path for pulling a model from the Ollama registry (`POST /api/pull`).
pub const API_PULL_PATH: &str = "/api/pull";

/// Gemma's channel markers. Gemma wraps its reasoning in `<|channel>thought` and
/// `<channel|>`, and some servers pass the markers through in the reply text
/// when they do not split the reasoning out. Each marker is one token, so it
/// always arrives whole inside a single streamed delta.
pub const GEMMA_CHANNEL_MARKERS: [&str; 3] = ["<|channel>thought", "<|channel>", "<channel|>"];
