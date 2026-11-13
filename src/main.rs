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
mod tools;
mod tui;
mod update;

#[cfg(test)]
mod test_support;

use anyhow::{Context, Result};
use clap::Parser;
use cli::{Cli, Commands};
use ollama::OllamaClient;
use std::path::{Path, PathBuf};

/// Parse the arguments, connect to the model server and run the command.
#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    let client = build_client(&cli)?;

    run_command(&cli, &client).await
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
        None => OllamaClient::new(&cli.ollama_url),
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
}
