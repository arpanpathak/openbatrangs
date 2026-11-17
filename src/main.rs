//! openBatarangs CLI entry point.
//!
//! Parses arguments, creates the Ollama client, and dispatches to the
//! command/TUI implementations in `commands` and `tui`.
//!
//! ## References
//!
//! - Clap derive API: <https://docs.rs/clap/latest/clap/_derive/index.html>
//! - Tokio async runtime: <https://docs.rs/tokio/latest/tokio/>

#![warn(missing_docs)]
#![warn(clippy::missing_docs_in_private_items)]

mod agent;
mod banner;
mod cli;
mod commands;
mod constants;
mod hardware;
mod model_select;
mod models;
mod ollama;
mod perf;
mod sampling;
mod terminal_text;
mod tools;
mod tui;
mod update;

#[cfg(test)]
mod test_support;

use anyhow::{Context, Result};
use clap::Parser;
use cli::{Cli, Commands, ConfigFile};
use constants::cli::DEFAULT_OLLAMA_URL;
use ollama::OllamaClient;
use std::path::{Path, PathBuf};

/// Parse the arguments, read the config file, connect to the model server
/// and run the command. `provider` only reads or changes the config file,
/// so it runs before any client is built.
#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    let config_path = cli::user_config_path();

    if let Some(warning) = cli::permission_warning(&config_path) {
        eprintln!("{warning}");
    }

    if let Some(Commands::Provider { action }) = &cli.command {
        return commands::provider(action, &config_path);
    }

    let config = ConfigFile::load(&config_path)?;
    let (cli, client) = connect(cli, config)?;

    run_command(&cli, &client).await
}

/// The model client for this run, and the arguments with the chosen
/// provider's settings filled in where no flag sets them. After this,
/// `cli.provider` names the provider in use, or is `None` when none is.
fn connect(mut cli: Cli, mut config: ConfigFile) -> Result<(Cli, OllamaClient)> {
    // `--thor` searches through the Thor unless a flag or the environment
    // names a SearXNG instance, as before the config file existed.
    let file_search_url = config.search_engine_url.take().filter(|_url| !cli.thor);
    cli.searxng_url = cli.searxng_url.or(file_search_url);
    let chosen = cli::chosen_provider(&cli, config)?;
    cli.provider = None;

    let Some((name, mut provider)) = chosen else {
        let client = build_client(&cli)?;
        return Ok((cli, client));
    };

    let key = match &cli.api_key_file {
        Some(key_file) => Some(read_key(key_file)?),
        None => provider.api_key.take(),
    };
    cli.temperature = cli.temperature.or(provider.temperature);
    cli.is_thinking_disabled = cli.is_thinking_disabled || provider.think == Some(false);
    cli.provider = Some(name);

    let client = provider.client(key, !cli.is_thinking_disabled)?;

    Ok((cli, client))
}

/// Run the subcommand named on the command line, or the agent when none is named.
async fn run_command(cli: &Cli, client: &OllamaClient) -> Result<()> {
    match &cli.command {
        Some(Commands::Setup) => commands::setup(client).await,
        Some(Commands::ListModels) => list_models(cli, client).await,
        Some(Commands::Doctor) => doctor(cli, client).await,
        Some(Commands::Agent { task }) => run_agent(cli, client, task).await,
        Some(Commands::Pull { model }) => pull(client, model).await,
        Some(Commands::Update) => commands::update(&update::UpdateCheck::for_this_process()).await,
        // `main` runs `provider` before a client exists.
        Some(Commands::Provider { .. }) => Ok(()),
        None => run_agent(cli, client, &cli.task).await,
    }
}

/// `list-models`: print the installed models with their scores.
async fn list_models(cli: &Cli, client: &OllamaClient) -> Result<()> {
    commands::ensure_ollama(client).await?;

    commands::list_models(client, cli.min_context as u64).await
}

/// `doctor`: print the health report.
async fn doctor(cli: &Cli, client: &OllamaClient) -> Result<()> {
    commands::ensure_ollama(client).await?;

    commands::doctor(client, cli.min_context as u64).await
}

/// `agent`, or a bare task: run the agent once, or start the TUI without a task.
async fn run_agent(cli: &Cli, client: &OllamaClient, task: &[String]) -> Result<()> {
    commands::ensure_ollama(client).await?;

    commands::run_agent_or_tui(cli, client, task, update::automatic_check(cli)).await
}

/// `pull`: download a model.
async fn pull(client: &OllamaClient, model: &str) -> Result<()> {
    commands::ensure_ollama(client).await?;

    commands::pull(client, model).await
}

/// Create the model client: an OpenAI-compatible server when `--thor` or
/// `--openai-url` is given, Ollama otherwise.
fn build_client(cli: &Cli) -> Result<OllamaClient> {
    match openai_server(cli, crate::cli::home_directory()) {
        Some((url, key_file)) => openai_client(cli, &url, key_file),
        None => OllamaClient::new(cli.ollama_url.as_deref().unwrap_or(DEFAULT_OLLAMA_URL)),
    }
}

/// A client for the OpenAI-compatible server at `url`, with the key read
/// from `key_file` when one is given.
fn openai_client(cli: &Cli, url: &str, key_file: Option<PathBuf>) -> Result<OllamaClient> {
    let key = key_file.as_deref().map(read_key).transpose()?;

    OllamaClient::openai(url, key, !cli.is_thinking_disabled)
}

/// The address of the OpenAI-compatible server and the file holding its key,
/// or `None` for Ollama. `--openai-url` wins over `--thor`.
fn openai_server(cli: &Cli, home: Option<PathBuf>) -> Option<(String, Option<PathBuf>)> {
    match (&cli.openai_url, cli.thor) {
        (Some(url), _) => Some((url.clone(), cli.api_key_file.clone())),
        (None, true) => Some((
            constants::cli::THOR_OPENAI_URL.to_string(),
            crate::cli::thor_key_file(cli, home),
        )),
        (None, false) => None,
    }
}

/// The access key in `path`, without surrounding whitespace.
fn read_key(path: &Path) -> Result<String> {
    let key = std::fs::read_to_string(path)
        .with_context(|| format!("cannot read the access key from {}", path.display()))?;

    Ok(key.trim().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plain_arguments_use_ollama() -> Result<()> {
        let cli = Cli::parse_from(["openbatrangs"]);

        assert_eq!(openai_server(&cli, None), None);
        assert!(!build_client(&cli)?.is_openai());
        Ok(())
    }

    #[test]
    fn openai_url_keeps_its_own_key_file() {
        let cli = Cli::parse_from(["openbatrangs", "--thor", "--openai-url", "http://box/v1"]);

        assert_eq!(
            openai_server(&cli, Some(PathBuf::from("/home/user"))),
            Some(("http://box/v1".to_string(), None))
        );
    }

    #[test]
    fn thor_reads_the_key_from_home() {
        let cli = Cli::parse_from(["openbatrangs", "--thor"]);
        let home_key_file = PathBuf::from("/home/user").join(constants::cli::THOR_KEY_FILE);

        assert_eq!(
            openai_server(&cli, Some(PathBuf::from("/home/user"))),
            Some((
                constants::cli::THOR_OPENAI_URL.to_string(),
                Some(home_key_file)
            ))
        );
    }

    #[test]
    fn key_is_trimmed() -> Result<()> {
        let directory = test_support::unique_temp_dir("openbatrangs-key");
        let key_file = directory.join("api-key");
        std::fs::write(&key_file, "secret\n")?;

        assert_eq!(read_key(&key_file)?, "secret");
        Ok(())
    }

    #[test]
    fn missing_key_file_is_an_error() {
        let outcome = read_key(Path::new("/nonexistent/openbatrangs/api-key"));

        assert!(format!("{outcome:?}").contains("cannot read the access key"));
    }

    #[test]
    fn openai_client_is_built_with_the_key() -> Result<()> {
        let directory = test_support::unique_temp_dir("openbatrangs-client-key");
        let key_file = directory.join("api-key");
        std::fs::write(&key_file, "secret")?;
        let key_arg = key_file.display().to_string();
        let cli = Cli::parse_from([
            "openbatrangs",
            "--openai-url",
            "http://box/v1",
            "--api-key-file",
            &key_arg,
        ]);

        assert!(build_client(&cli)?.is_openai());
        Ok(())
    }

    /// A config file with two providers: an OpenAI-compatible server with
    /// settings, and an Ollama server.
    const TWO_PROVIDERS: &str = "search_engine_url = \"http://search:8888\"\n\
        [box]\nurl = \"http://box:8000/v1\"\napi_key = \"box-key\"\nthink = false\ntemperature = 0.4\n\
        [local]\nurl = \"http://localhost:11434\"\n";

    /// The config file holding `text`.
    fn config(text: &str) -> Result<ConfigFile> {
        ConfigFile::load(&test_support::config_file_with(text)?)
    }

    /// A key file holding `secret`, and its path as an argument.
    fn key_file_argument() -> Result<String> {
        let key_file = test_support::unique_temp_dir("openbatrangs-key").join("api-key");
        std::fs::write(&key_file, "secret")?;

        Ok(key_file.display().to_string())
    }

    #[test]
    fn without_providers_the_local_ollama_is_used() -> Result<()> {
        let (cli, client) = connect(Cli::parse_from(["openbatrangs"]), config("")?)?;

        assert!(!client.is_openai());
        assert_eq!(client.base_url, DEFAULT_OLLAMA_URL);
        assert_eq!(cli.provider, None);
        Ok(())
    }

    #[test]
    fn the_named_provider_brings_its_settings() -> Result<()> {
        let cli = Cli::parse_from(["openbatrangs", "--provider", "box"]);

        let (cli, client) = connect(cli, config(TWO_PROVIDERS)?)?;

        assert!(client.is_openai());
        assert_eq!(client.base_url, "http://box:8000/v1");
        assert_eq!(cli.provider.as_deref(), Some("box"));
        assert_eq!(cli.temperature, Some(0.4));
        assert!(cli.is_thinking_disabled);
        assert_eq!(cli.searxng_url.as_deref(), Some("http://search:8888"));
        Ok(())
    }

    #[test]
    fn flags_win_over_the_provider_settings() -> Result<()> {
        let cli = Cli::parse_from([
            "openbatrangs",
            "--provider",
            "box",
            "--temperature",
            "1.2",
            "--searxng-url",
            "http://other:8888",
        ]);

        let (cli, _client) = connect(cli, config(TWO_PROVIDERS)?)?;

        assert_eq!(cli.temperature, Some(1.2));
        assert_eq!(cli.searxng_url.as_deref(), Some("http://other:8888"));
        Ok(())
    }

    #[test]
    fn no_think_wins_over_a_provider_that_thinks() -> Result<()> {
        let cli = Cli::parse_from(["openbatrangs", "--no-think"]);

        let (cli, _client) = connect(cli, config("[box]\nurl = \"http://box/v1\"\nthink = true")?)?;

        assert!(cli.is_thinking_disabled);
        assert_eq!(cli.provider.as_deref(), Some("box"));
        Ok(())
    }

    #[test]
    fn the_api_key_file_replaces_the_provider_key() -> Result<()> {
        let key_arg = key_file_argument()?;
        let flagged = Cli::parse_from([
            "openbatrangs",
            "--provider",
            "box",
            "--api-key-file",
            &key_arg,
        ]);
        let missing = Cli::parse_from([
            "openbatrangs",
            "--provider",
            "box",
            "--api-key-file",
            "/nonexistent/openbatrangs/api-key",
        ]);

        let (_cli, client) = connect(flagged, config(TWO_PROVIDERS)?)?;
        let outcome = connect(missing, config(TWO_PROVIDERS)?);

        assert_eq!(client.base_url, "http://box:8000/v1");
        assert!(format!("{:?}", outcome.err()).contains("cannot read the access key"));
        Ok(())
    }

    #[test]
    fn several_providers_and_no_name_leave_the_choice_to_the_tui() -> Result<()> {
        let (tui, client) = connect(Cli::parse_from(["openbatrangs"]), config(TWO_PROVIDERS)?)?;
        let task = connect(
            Cli::parse_from(["openbatrangs", "fix", "it"]),
            config(TWO_PROVIDERS)?,
        );

        assert_eq!(tui.provider, None);
        assert_eq!(client.base_url, DEFAULT_OLLAMA_URL);
        assert!(format!("{:?}", task.err()).contains("box, local"));
        Ok(())
    }

    #[test]
    fn server_flags_and_thor_win_over_the_providers() -> Result<()> {
        let key_arg = key_file_argument()?;
        let ollama = Cli::parse_from([
            "openbatrangs",
            "--provider",
            "box",
            "--ollama-url",
            "http://other:11434",
        ]);
        let thor = Cli::parse_from([
            "openbatrangs",
            "--provider",
            "box",
            "--thor",
            "--api-key-file",
            &key_arg,
        ]);

        let (ollama_cli, ollama_client) = connect(ollama, config(TWO_PROVIDERS)?)?;
        let (_thor_cli, thor_client) = connect(thor, config(TWO_PROVIDERS)?)?;

        assert_eq!(ollama_client.base_url, "http://other:11434");
        assert_eq!(ollama_cli.provider, None);
        assert_eq!(thor_client.base_url, constants::cli::THOR_OPENAI_URL);
        Ok(())
    }

    #[test]
    fn thor_searches_through_the_thor_despite_the_file() -> Result<()> {
        let key_arg = key_file_argument()?;
        let cli = Cli::parse_from(["openbatrangs", "--thor", "--api-key-file", &key_arg]);

        let (cli, _client) = connect(cli, config(TWO_PROVIDERS)?)?;
        let run_config = cli::agent_run_config(&cli);

        assert_eq!(cli.searxng_url, None);
        assert_eq!(run_config.searxng_url, constants::cli::THOR_OPENAI_URL);
        assert_eq!(run_config.search_key.as_deref(), Some("secret"));
        Ok(())
    }

    #[tokio::test]
    async fn run_command_leaves_the_provider_command_to_main() -> Result<()> {
        // Every other command would fail on a server that refuses connections.
        let cli = Cli::parse_from(["openbatrangs", "provider", "list"]);
        let client = OllamaClient::new("http://127.0.0.1:1")?;

        run_command(&cli, &client).await
    }
}
