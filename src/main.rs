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

#[cfg(test)]
mod test_support;

use anyhow::{Context, Result};
use clap::Parser;
use cli::{Cli, Commands};
use ollama::OllamaClient;

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    let client = build_client(&cli)?;

    match &cli.command {
        Some(Commands::Setup) => commands::setup(&client).await?,
        Some(Commands::ListModels) => {
            commands::ensure_ollama(&client).await?;
            commands::list_models(&client, cli.min_context as u64).await?;
        }
        Some(Commands::Doctor) => {
            commands::ensure_ollama(&client).await?;
            commands::doctor(&client, cli.min_context as u64).await?;
        }
        Some(Commands::Agent { task }) => {
            commands::ensure_ollama(&client).await?;
            commands::run_agent_or_tui(&cli, &client, task).await?;
        }
        Some(Commands::Pull { model }) => {
            commands::ensure_ollama(&client).await?;
            commands::pull(&client, model).await?;
        }
        None => {
            commands::ensure_ollama(&client).await?;
            commands::run_agent_or_tui(&cli, &client, &cli.task).await?;
        }
    }

    Ok(())
}

/// Create the model client: an OpenAI-compatible server when `--thor` or
/// `--openai-url` is given, Ollama otherwise.
fn build_client(cli: &Cli) -> Result<OllamaClient> {
    let home = std::env::var_os("HOME").map(std::path::PathBuf::from);
    let (url, key_file) = match (&cli.openai_url, cli.thor) {
        (Some(url), _) => (url.clone(), cli.api_key_file.clone()),
        (None, true) => (
            constants::cli::THOR_OPENAI_URL.to_string(),
            cli.api_key_file
                .clone()
                .or_else(|| home.map(|home| home.join(constants::cli::THOR_KEY_FILE))),
        ),
        (None, false) => return OllamaClient::new(&cli.ollama_url),
    };
    let key = key_file
        .map(|path| {
            std::fs::read_to_string(&path)
                .map(|key| key.trim().to_string())
                .with_context(|| format!("cannot read the access key from {}", path.display()))
        })
        .transpose()?;
    OllamaClient::openai(&url, key, !cli.is_thinking_disabled)
}
