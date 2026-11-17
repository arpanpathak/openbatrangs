//! Providers from `config.toml`.
//!
//! Each provider is one table named after the provider, holding everything
//! about it: the server address, the API key, and the thinking and
//! temperature settings. A built-in name such as `deepseek` knows its
//! address, so its table needs only a key. The only top-level setting that
//! belongs to no provider is `search_engine_url`.
//!
//! ```toml
//! search_engine_url = "http://localhost:8888"
//!
//! [deepseek]
//! api_key = "sk-..."
//! think = false
//!
//! [home-gpu]
//! url = "http://192.168.1.20:8000/v1"
//! temperature = 0.7
//! ```
//!
//! ## References
//!
//! - TOML: <https://toml.io/en/v1.0.0>
//! - XDG base directories: <https://specifications.freedesktop.org/basedir-spec/latest/>

use super::args::{Cli, Commands};
use super::config::{home_directory, names_server};
use crate::constants::cli::{
    BUILT_IN_PROVIDERS, BuiltInProvider, CONFIG_FILE_NAME, CONFIG_FOLDER_NAME, OPENAI_API_SUFFIX,
    SEARCH_ENGINE_URL_KEY,
};
use crate::constants::ollama::ANTHROPIC_API_HOST;
use crate::ollama::OllamaClient;
use anyhow::{Context, Result, anyhow, bail};
use serde::Deserialize;
use std::collections::BTreeMap;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};

/// The contents of `config.toml`.
#[derive(Debug, Default)]
pub(crate) struct ConfigFile {
    /// Where the file was read from, named in listings and errors.
    pub(crate) path: PathBuf,
    /// The SearXNG instance the web tools search when no flag names one.
    pub(crate) search_engine_url: Option<String>,
    /// Providers by name, in name order.
    pub(crate) providers: BTreeMap<String, Provider>,
}

/// One provider: a model server and the settings used with it.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Provider {
    /// Server address. One that ends in `/v1` is an OpenAI-compatible
    /// server; any other is an Ollama server.
    pub(crate) url: String,
    /// Bearer key sent with every request.
    pub(crate) api_key: Option<String>,
    /// Whether thinking models reason before they answer; on when unset.
    pub(crate) think: Option<bool>,
    /// Sampling temperature, or `None` for the model card's.
    pub(crate) temperature: Option<f64>,
    /// Context window in tokens when the server reports none.
    pub(crate) context: Option<u64>,
}

/// A provider table as written in the file.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ProviderBlock {
    /// Server address; optional for a built-in name.
    url: Option<String>,
    /// Bearer key.
    api_key: Option<String>,
    /// Thinking on or off.
    think: Option<bool>,
    /// A number, or `"auto"` for the model card's temperature.
    temperature: Option<Temperature>,
    /// Context window in tokens when the server reports none.
    context: Option<u64>,
}

/// A temperature setting: a number, or `None` for `"auto"`.
#[derive(Deserialize)]
#[serde(try_from = "toml::Value")]
struct Temperature(Option<f64>);

impl TryFrom<toml::Value> for Temperature {
    type Error = String;

    fn try_from(value: toml::Value) -> Result<Self, Self::Error> {
        match value {
            toml::Value::Float(number) => Ok(Self(Some(number))),
            toml::Value::Integer(number) => Ok(Self(Some(number as f64))),
            toml::Value::String(word) if word == "auto" => Ok(Self(None)),
            other => Err(format!("temperature is a number or \"auto\", not {other}")),
        }
    }
}

/// `config.toml` in the user config folder: `$XDG_CONFIG_HOME/openbatrangs`
/// when that variable holds an absolute path, else `~/.config/openbatrangs`.
/// When neither is known, the path is relative to the working directory.
pub(crate) fn config_path(xdg_config_home: Option<PathBuf>, home: Option<&Path>) -> PathBuf {
    let config_root = xdg_config_home
        .filter(|path| path.is_absolute())
        .or_else(|| home.map(|home| home.join(".config")))
        .unwrap_or_default();

    config_root.join(CONFIG_FOLDER_NAME).join(CONFIG_FILE_NAME)
}

/// The config file of this process, from `XDG_CONFIG_HOME` and `HOME`.
pub(crate) fn user_config_path() -> PathBuf {
    let xdg_config_home = std::env::var_os("XDG_CONFIG_HOME").map(PathBuf::from);

    config_path(xdg_config_home, home_directory().as_deref())
}

/// The built-in provider named `name`.
fn built_in(name: &str) -> Option<&'static BuiltInProvider> {
    BUILT_IN_PROVIDERS
        .iter()
        .find(|provider| provider.name == name)
}

/// The address openbatrangs knows for the built-in provider `name`.
pub(crate) fn built_in_url(name: &str) -> Option<&'static str> {
    built_in(name).map(|provider| provider.url)
}

/// A warning when the file at `path` can be read by other users, as it
/// holds API keys; `None` when it cannot or does not exist.
pub(crate) fn permission_warning(path: &Path) -> Option<String> {
    let mode = std::fs::metadata(path).ok()?.permissions().mode();

    (mode & 0o077 != 0).then(|| {
        format!(
            "⚠️  {} holds API keys and other users can read it; run `chmod 600 {}`",
            path.display(),
            path.display()
        )
    })
}

/// The provider for this run, or `None` when a flag names the server, or
/// nothing names a provider and the file holds none or several.
///
/// `--provider` or `OPENBATRANGS_PROVIDER` names one; otherwise the only
/// provider in the file is used. A task run with several providers and none
/// named is an error, as there is nobody to ask.
pub(crate) fn chosen_provider(cli: &Cli, config: ConfigFile) -> Result<Option<(String, Provider)>> {
    if names_server(cli) {
        return Ok(None);
    }

    if cli.provider.is_none() && is_task_run(cli) && config.providers.len() > 1 {
        let names: Vec<&str> = config.providers.keys().map(String::as_str).collect();
        bail!(
            "{} has several providers ({}); choose one with --provider NAME",
            config.path.display(),
            names.join(", ")
        );
    }

    config.into_chosen(cli.provider.as_deref())
}

/// Whether the command line runs one task without the TUI.
fn is_task_run(cli: &Cli) -> bool {
    match &cli.command {
        Some(Commands::Agent { task }) => !task.is_empty(),
        Some(_other) => false,
        None => !cli.task.is_empty(),
    }
}

/// The text of the config file at `path`, or an empty text when the file
/// does not exist.
pub(crate) fn read_config_text(path: &Path) -> Result<String> {
    match path.exists() {
        true => {
            std::fs::read_to_string(path).with_context(|| format!("cannot read {}", path.display()))
        }
        false => Ok(String::new()),
    }
}

impl ConfigFile {
    /// Read the config file at `path`. A missing file holds no providers.
    pub(crate) fn load(path: &Path) -> Result<Self> {
        let text = read_config_text(path)?;
        let table: toml::Table = toml::from_str(&text)
            .with_context(|| format!("{} is not a valid config file", path.display()))?;

        let mut config = Self {
            path: path.to_path_buf(),
            ..Self::default()
        };

        for (key, value) in table {
            config.add_entry(key, value)?;
        }

        Ok(config)
    }

    /// Add one top-level entry: `search_engine_url`, or a provider table.
    fn add_entry(&mut self, key: String, value: toml::Value) -> Result<()> {
        let place = format!("`{key}` in {}", self.path.display());

        if key == SEARCH_ENGINE_URL_KEY {
            self.search_engine_url = Some(value.try_into().context(place)?);
            return Ok(());
        }

        let block: ProviderBlock = value.try_into().context(place)?;
        let provider = Provider::from_block(&key, block, &self.path)?;
        self.providers.insert(key, provider);

        Ok(())
    }

    /// The provider named `name`, else the only provider in the file, or
    /// `None` when nothing is named and the file holds none or several.
    pub(crate) fn into_chosen(mut self, name: Option<&str>) -> Result<Option<(String, Provider)>> {
        match (name, self.providers.len()) {
            (Some(name), _) => {
                let provider = self.into_provider(name)?;
                Ok(Some((name.to_string(), provider)))
            }
            (None, 1) => Ok(self.providers.pop_first()),
            (None, _) => Ok(None),
        }
    }

    /// The provider named `name`, or an error naming the file it is missing from.
    pub(crate) fn into_provider(mut self, name: &str) -> Result<Provider> {
        self.providers.remove(name).ok_or_else(|| {
            anyhow!(
                "no provider named `{name}` in {}; add it with `openbatrangs provider add {name}`",
                self.path.display()
            )
        })
    }

    /// The listing `openbatrangs provider list` prints: the file, then one
    /// line per provider with its URL and whether it has a key. The key
    /// itself is never shown.
    pub(crate) fn lines(&self) -> Vec<String> {
        let width = self.providers.keys().map(String::len).max().unwrap_or(0);
        let mut lines = vec![format!("Providers in {}:", self.path.display())];

        for (name, provider) in &self.providers {
            let key = match &provider.api_key {
                Some(_key) => "key set",
                None => "no key",
            };

            lines.push(format!("  {name:width$}  {}  {key}", provider.url));
        }

        lines
    }
}

impl Provider {
    /// The provider named `name` from its table in the file at `path`. A
    /// built-in name may leave out `url` and `context`; any other name needs
    /// `url`.
    fn from_block(name: &str, block: ProviderBlock, path: &Path) -> Result<Self> {
        let built_in = built_in(name);
        let url = block
            .url
            .or_else(|| built_in.map(|provider| provider.url.to_string()))
            .ok_or_else(|| anyhow!("[{name}] in {} needs a `url`", path.display()))?;

        Ok(Self {
            url,
            api_key: block.api_key,
            think: block.think,
            temperature: block.temperature.and_then(|temperature| temperature.0),
            context: block
                .context
                .or(built_in.and_then(|provider| provider.context)),
        })
    }

    /// A client for this provider's server with `key`. A URL ending in `/v1`
    /// gets an OpenAI-compatible client that sends `think`, with the key in
    /// Anthropic's headers when the URL is Anthropic's API; any other URL
    /// gets an Ollama client.
    /// The client falls back to this provider's `context` when the server
    /// reports no context window.
    pub(crate) fn client(&self, key: Option<String>, think: bool) -> Result<OllamaClient> {
        let url = self.url.trim_end_matches('/');
        let is_anthropic = reqwest::Url::parse(url)
            .is_ok_and(|parsed| parsed.host_str() == Some(ANTHROPIC_API_HOST));

        let client = match (url.ends_with(OPENAI_API_SUFFIX), is_anthropic) {
            (true, true) => OllamaClient::anthropic(url, key, think),
            (true, false) => OllamaClient::openai(url, key, think),
            (false, _) => OllamaClient::new(url),
        }?;

        Ok(client.with_context_window(self.context))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{config_file_with, unique_temp_dir};
    use clap::Parser;

    /// The example file from the README.
    const EXAMPLE: &str = r#"
search_engine_url = "http://localhost:8888"

[anthropic]
api_key = "sk-ant-test"
think = true
temperature = "auto"

[deepseek]
api_key = "sk-test"
think = false

[openai]
api_key = "sk-test"

[home-gpu]
url = "http://192.168.1.20:8000/v1"
temperature = 0.7

[colab]
url = "http://localhost:8000/v1"
api_key = "colab-test"
"#;

    /// The config file holding `text`.
    fn config(text: &str) -> Result<ConfigFile> {
        ConfigFile::load(&config_file_with(text)?)
    }

    /// The error of loading a config file holding `text`, as text.
    fn load_error(text: &str) -> Result<String> {
        let path = config_file_with(text)?;
        let message = format!("{:?}", ConfigFile::load(&path).err());

        assert!(message.contains(&path.display().to_string()), "{message}");
        Ok(message)
    }

    /// The name of the provider `arguments` choose from a file holding `text`.
    fn chosen_name(arguments: &[&str], text: &str) -> Result<Option<String>> {
        let cli = Cli::parse_from(arguments);
        let chosen = chosen_provider(&cli, config(text)?)?;

        Ok(chosen.map(|(name, _provider)| name))
    }

    #[test]
    fn the_file_lives_in_the_xdg_config_folder_or_under_home() {
        let home = Some(Path::new("/home/u"));

        let xdg = config_path(Some(PathBuf::from("/xdg")), home);
        let relative_xdg = config_path(Some(PathBuf::from("xdg")), home);
        let nothing_known = config_path(None, None);

        assert_eq!(xdg, PathBuf::from("/xdg/openbatrangs/config.toml"));
        assert_eq!(
            relative_xdg,
            PathBuf::from("/home/u/.config/openbatrangs/config.toml")
        );
        assert_eq!(nothing_known, PathBuf::from("openbatrangs/config.toml"));
        assert!(user_config_path().ends_with("openbatrangs/config.toml"));
    }

    #[test]
    fn the_example_file_reads_every_setting() -> Result<()> {
        let file = config(EXAMPLE)?;
        let anthropic = &file.providers["anthropic"];
        let deepseek = &file.providers["deepseek"];
        let home_gpu = &file.providers["home-gpu"];

        assert_eq!(
            file.search_engine_url.as_deref(),
            Some("http://localhost:8888")
        );
        assert_eq!(file.providers.len(), 5);
        assert_eq!(anthropic.url, "https://api.anthropic.com/v1");
        assert_eq!(anthropic.api_key.as_deref(), Some("sk-ant-test"));
        assert_eq!(anthropic.think, Some(true));
        assert_eq!(anthropic.temperature, None);
        assert_eq!(deepseek.url, "https://api.deepseek.com/v1");
        assert_eq!(deepseek.think, Some(false));
        assert_eq!(file.providers["openai"].url, "https://api.openai.com/v1");
        assert_eq!(home_gpu.url, "http://192.168.1.20:8000/v1");
        assert_eq!(home_gpu.api_key, None);
        assert_eq!(home_gpu.temperature, Some(0.7));
        Ok(())
    }

    #[test]
    fn a_built_in_name_may_set_its_own_url_and_a_whole_number_temperature() -> Result<()> {
        let file = config("[openai]\nurl = \"http://proxy/v1\"\ntemperature = 1")?;

        assert_eq!(file.providers["openai"].url, "http://proxy/v1");
        assert_eq!(file.providers["openai"].temperature, Some(1.0));
        Ok(())
    }

    #[test]
    fn a_missing_file_holds_no_providers() -> Result<()> {
        let path = unique_temp_dir("openbatrangs-config").join("config.toml");

        let file = ConfigFile::load(&path)?;

        assert_eq!(file.path, path);
        assert!(file.providers.is_empty());
        assert_eq!(file.search_engine_url, None);
        Ok(())
    }

    #[test]
    fn errors_name_the_file_the_table_and_the_key() -> Result<()> {
        let malformed = load_error("[deepseek\napi_key = 1")?;
        let unknown_key = load_error("[deepseek]\nmodel = \"x\"")?;
        let missing_url = load_error("[home-gpu]\napi_key = \"k\"")?;
        let loose_value = load_error("default = \"deepseek\"")?;
        let temperature = load_error("[deepseek]\ntemperature = \"hot\"")?;
        let search_url = load_error("search_engine_url = 3")?;

        assert!(malformed.contains("is not a valid config file"));
        assert!(
            unknown_key.contains("`deepseek`") && unknown_key.contains("unknown field `model`")
        );
        assert!(missing_url.contains("[home-gpu]") && missing_url.contains("needs a `url`"));
        assert!(loose_value.contains("`default`"));
        assert!(temperature.contains("temperature is a number or \"auto\""));
        assert!(search_url.contains("`search_engine_url`"));
        Ok(())
    }

    #[test]
    fn an_unreadable_file_is_an_error_that_names_it() {
        let path = unique_temp_dir("openbatrangs-config");

        let message = format!("{:?}", ConfigFile::load(&path).err());

        assert!(message.contains(&format!("cannot read {}", path.display())));
    }

    #[test]
    fn the_named_provider_or_the_only_one_is_chosen() -> Result<()> {
        let one = "[colab]\nurl = \"http://localhost:8000/v1\"";

        let named = chosen_name(&["openbatrangs", "--provider", "deepseek"], EXAMPLE)?;
        let only = chosen_name(&["openbatrangs"], one)?;
        let several = chosen_name(&["openbatrangs"], EXAMPLE)?;
        let none = chosen_name(&["openbatrangs"], "")?;

        assert_eq!(named.as_deref(), Some("deepseek"));
        assert_eq!(only.as_deref(), Some("colab"));
        assert_eq!(several, None);
        assert_eq!(none, None);
        Ok(())
    }

    #[test]
    fn a_flag_that_names_the_server_chooses_no_provider() -> Result<()> {
        let ollama = chosen_name(
            &[
                "openbatrangs",
                "--provider",
                "deepseek",
                "--ollama-url",
                "http://box",
            ],
            EXAMPLE,
        )?;
        let openai = chosen_name(&["openbatrangs", "--openai-url", "http://box/v1"], EXAMPLE)?;
        let thor = chosen_name(&["openbatrangs", "--thor"], EXAMPLE)?;

        assert_eq!(ollama, None);
        assert_eq!(openai, None);
        assert_eq!(thor, None);
        Ok(())
    }

    #[test]
    fn a_task_run_with_several_providers_needs_a_name() -> Result<()> {
        let task = chosen_name(&["openbatrangs", "fix", "it"], EXAMPLE);
        let agent = chosen_name(&["openbatrangs", "agent", "fix", "it"], EXAMPLE);
        let tui = chosen_name(&["openbatrangs", "agent"], EXAMPLE)?;
        let other_command = chosen_name(&["openbatrangs", "list-models"], EXAMPLE)?;

        let message = format!("{:?}", task.err());
        assert!(message.contains("anthropic, colab, deepseek, home-gpu, openai"));
        assert!(message.contains("--provider NAME"));
        assert!(agent.is_err());
        assert_eq!(tui, None);
        assert_eq!(other_command, None);
        Ok(())
    }

    #[test]
    fn an_unknown_provider_names_itself_the_file_and_how_to_add_it() -> Result<()> {
        let file = config(EXAMPLE)?;
        let path = file.path.display().to_string();

        let message = format!("{:?}", file.into_provider("nope").err());

        assert!(message.contains("no provider named `nope`"));
        assert!(message.contains(&path));
        assert!(message.contains("openbatrangs provider add nope"));
        Ok(())
    }

    #[test]
    fn a_file_readable_by_others_gets_a_warning() -> Result<()> {
        let path = config_file_with("")?;

        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o644))?;
        let shared = permission_warning(&path);
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600))?;
        let private = permission_warning(&path);
        let missing = permission_warning(&path.with_file_name("missing.toml"));

        assert!(shared.is_some_and(|warning| warning.contains("chmod 600")));
        assert_eq!(private, None);
        assert_eq!(missing, None);
        Ok(())
    }

    #[test]
    fn the_listing_shows_urls_and_never_a_key() -> Result<()> {
        let file = config(EXAMPLE)?;

        let lines = file.lines();

        assert_eq!(lines[0], format!("Providers in {}:", file.path.display()));
        assert_eq!(
            lines[1],
            "  anthropic  https://api.anthropic.com/v1  key set"
        );
        assert_eq!(lines[4], "  home-gpu   http://192.168.1.20:8000/v1  no key");
        assert!(!lines.join("\n").contains("sk-"));
        Ok(())
    }

    #[test]
    fn deepseek_knows_its_window_and_a_block_can_set_one() -> Result<()> {
        let file = config(
            "[deepseek]\n[anthropic]\n[home-gpu]\nurl = \"http://box:8000/v1\"\ncontext = 131072",
        )?;
        let shorter = config("[deepseek]\ncontext = 65536")?;

        assert_eq!(file.providers["deepseek"].context, Some(1_000_000));
        assert_eq!(file.providers["anthropic"].context, None);
        assert_eq!(file.providers["home-gpu"].context, Some(131_072));
        assert_eq!(shorter.providers["deepseek"].context, Some(65_536));

        let client = file.providers["deepseek"].client(None, true)?;
        assert_eq!(client.context_window, Some(1_000_000));
        Ok(())
    }

    #[test]
    fn a_v1_url_is_openai_compatible_and_any_other_is_ollama() -> Result<()> {
        let file =
            config("[box]\nurl = \"http://box:8000/v1/\"\n[local]\nurl = \"http://box:11434\"")?;

        let openai = file.providers["box"].client(None, true)?;
        let ollama = file.providers["local"].client(None, true)?;

        assert!(openai.is_openai());
        assert_eq!(openai.base_url, "http://box:8000/v1");
        assert!(!ollama.is_openai());
        assert_eq!(ollama.base_url, "http://box:11434");
        Ok(())
    }

    #[test]
    fn only_anthropics_api_gets_anthropics_key_headers() -> Result<()> {
        let file = config("[anthropic]\n[claude-proxy]\nurl = \"http://box:8000/v1\"")?;

        let anthropic = file.providers["anthropic"].client(None, true)?;
        let proxy = file.providers["claude-proxy"].client(None, true)?;

        assert!(anthropic.sends_anthropic_headers());
        assert!(!proxy.sends_anthropic_headers());
        Ok(())
    }
}
