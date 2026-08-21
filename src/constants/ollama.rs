//! Ollama HTTP API constants.

/// How long a full HTTP request may take before it is aborted.
pub const HTTP_TIMEOUT_SECONDS: u64 = 600;

/// How long to wait for the initial TCP connection to the Ollama server.
pub const CONNECT_TIMEOUT_SECONDS: u64 = 5;

/// Ollama API endpoint paths.
pub const API_TAGS_PATH: &str = "/api/tags";
pub const API_SHOW_PATH: &str = "/api/show";
pub const API_CHAT_PATH: &str = "/api/chat";
pub const API_PULL_PATH: &str = "/api/pull";
