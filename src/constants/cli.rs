//! Command-line interface defaults.

/// Default Ollama server address.
pub const DEFAULT_OLLAMA_URL: &str = "http://localhost:11434";

/// Nemotron's llama-server on the Thor, as seen from the Thor itself or
/// through the `thor-model-tunnel` SSH tunnel.
pub const THOR_OPENAI_URL: &str = "http://127.0.0.1:8079/v1";

/// Access key file for the Thor's model server, relative to `$HOME`.
pub const THOR_KEY_FILE: &str = ".config/thor-chat/api-key";

/// Default maximum agent iterations.
pub const DEFAULT_MAX_STEPS: usize = 40;

/// Default minimum acceptable context window for auto model selection.
pub const DEFAULT_MIN_CONTEXT: usize = 8_192;
