//! Command-line interface defaults.

/// Default Ollama server address.
pub const DEFAULT_OLLAMA_URL: &str = "http://localhost:11434";

/// Default maximum agent iterations.
pub const DEFAULT_MAX_STEPS: usize = 12;

/// Default minimum acceptable context window for auto model selection.
pub const DEFAULT_MIN_CONTEXT: usize = 8_192;
