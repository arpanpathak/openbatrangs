//! Command-line interface defaults.

/// Default Ollama server address.
pub const DEFAULT_OLLAMA_URL: &str = "http://localhost:11434";

/// The Thor's chat server on its Tailscale address. It fronts every TensorRT
/// Edge-LLM engine on the Thor, and the access key is the gate.
pub const THOR_OPENAI_URL: &str = "https://arpanpathak.taildb9a39.ts.net/v1";

/// Access key file for the Thor's model server, relative to `$HOME`.
pub const THOR_KEY_FILE: &str = ".config/thor-chat/api-key";

/// Default maximum agent iterations.
pub const DEFAULT_MAX_STEPS: usize = 40;

/// Default minimum acceptable context window for auto model selection.
pub const DEFAULT_MIN_CONTEXT: usize = 8_192;

/// Default SearXNG endpoint the web tools search when neither `--searxng-url`
/// nor `OPENBATRANGS_SEARXNG_URL` is set.
pub const DEFAULT_SEARXNG_URL: &str = "http://127.0.0.1:8888";
