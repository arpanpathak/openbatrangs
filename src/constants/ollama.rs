//! Ollama HTTP API constants.

/// Maximum duration (seconds) for a complete HTTP request to the Ollama server.
/// Set high (10 min) because model generation and pull operations can be slow.
pub const HTTP_TIMEOUT_SECONDS: u64 = 600;

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

/// Opens one of Gemma's channels. Gemma writes `<|channel>thought`, a line
/// break and its reasoning, then [`GEMMA_CHANNEL_CLOSE`] before the reply.
/// The channel name (`thought`, `analysis`, `final`) often arrives in the
/// next streamed delta, apart from the marker.
pub const GEMMA_CHANNEL_OPEN: &str = "<|channel>";

/// Closes one of Gemma's channels; the reply follows it.
pub const GEMMA_CHANNEL_CLOSE: &str = "<channel|>";
