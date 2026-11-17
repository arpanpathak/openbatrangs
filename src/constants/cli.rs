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

/// A hosted provider whose address openbatrangs knows.
pub struct BuiltInProvider {
    /// Name of its block in the config file.
    pub name: &'static str,
    /// Name shown in the TUI.
    pub label: &'static str,
    /// API address.
    pub url: &'static str,
    /// Context window in tokens when the provider's model list reports
    /// none, or `None` when it reports one.
    pub context: Option<u64>,
}

/// The built-in providers. Anthropic is reached through its
/// OpenAI-compatible endpoint, and its model list reports each window.
/// DeepSeek's list reports none; its documentation gives 1M tokens for
/// every model it serves.
pub const BUILT_IN_PROVIDERS: &[BuiltInProvider] = &[
    BuiltInProvider {
        name: "anthropic",
        label: "Anthropic",
        url: "https://api.anthropic.com/v1",
        context: None,
    },
    BuiltInProvider {
        name: "deepseek",
        label: "DeepSeek",
        url: "https://api.deepseek.com/v1",
        context: Some(1_000_000),
    },
    BuiltInProvider {
        name: "openai",
        label: "OpenAI",
        url: "https://api.openai.com/v1",
        context: None,
    },
];
