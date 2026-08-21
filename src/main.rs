//! openBatarangs CLI entry point.
//!
//! Parses arguments, creates the Ollama client, and dispatches to the
//! command/TUI implementations in `commands` and `tui`.

mod agent;
mod banner;
mod cli;
mod commands;
mod model_select;
mod models;
mod ollama;
mod perf;
mod tools;
mod tui;

use anyhow::Result;
use clap::Parser;
use cli::{Cli, Commands};
use ollama::OllamaClient;

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    let client = OllamaClient::new(&cli.ollama_url)?;

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
        None => {
            commands::ensure_ollama(&client).await?;
            commands::run_agent_or_tui(&cli, &client, &cli.task).await?;
        }
    }

    Ok(())
}
