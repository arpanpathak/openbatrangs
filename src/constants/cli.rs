//! Command-line interface defaults.

/// Default Ollama server address.
pub const DEFAULT_OLLAMA_URL: &str = "http://localhost:11434";

/// The Thor's chat server on its Tailscale address. It fronts every TensorRT
/// Edge-LLM engine on the Thor, and the access key is the gate.
pub const THOR_OPENAI_URL: &str = "https://arpanpathak.taildb9a39.ts.net/v1";

/// Access key file for the Thor's model server, relative to `$HOME`.
pub const THOR_KEY_FILE: &str = ".config/thor-chat/api-key";

/// Default minimum acceptable context window for auto model selection.
pub const DEFAULT_MIN_CONTEXT: usize = 8_192;

/// Default SearXNG endpoint the web tools search when neither `--searxng-url`
/// nor `OPENBATRANGS_SEARXNG_URL` is set.
pub const DEFAULT_SEARXNG_URL: &str = "http://127.0.0.1:8888";

/// Name of the openbatrangs folder in the user config folder.
pub const CONFIG_FOLDER_NAME: &str = "openbatrangs";

/// Name of the config file that holds the providers.
pub const CONFIG_FILE_NAME: &str = "config.toml";

/// Top-level key of the config file that names the SearXNG instance.
pub const SEARCH_ENGINE_URL_KEY: &str = "search_engine_url";

/// Path suffix of an OpenAI-compatible API. A provider URL that ends with it
/// is an OpenAI-compatible server; any other URL is an Ollama server.
pub const OPENAI_API_SUFFIX: &str = "/v1";

/// Hosted providers whose address openbatrangs knows: the name of the block
/// in the config file, the name shown in the TUI, and the API address.
/// Anthropic is reached through its OpenAI-compatible endpoint.
pub const BUILT_IN_PROVIDERS: &[(&str, &str, &str)] = &[
    ("anthropic", "Anthropic", "https://api.anthropic.com/v1"),
    ("deepseek", "DeepSeek", "https://api.deepseek.com/v1"),
    ("openai", "OpenAI", "https://api.openai.com/v1"),
];
