//! openBatarangs CLI entry point.
//!
//! Parses arguments, creates the inference backend, and dispatches to the
//! command/TUI implementations in `commands` and `tui`.

mod agent;
mod banner;
mod cli;
mod commands;
mod constants;
mod engine;
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
use cli::{Cli, Commands, ExperimentalCommand};
use engine::{create_backend, EngineConfig, EngineKind};
use ollama::OllamaClient;

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    let client = OllamaClient::new(&cli.ollama_url)?;
    let engine_kind = EngineKind::parse(&cli.engine)
        .with_context(|| format!("unknown --engine '{}' (try: ollama, tensorrt)", cli.engine))?;
    let backend = create_backend(&EngineConfig::new(
        engine_kind,
        &cli.ollama_url,
        cli.model.clone(),
    ))?;

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
            commands::run_agent_or_tui(&cli, &client, backend.clone(), task).await?;
        }
        Some(Commands::Pull { model }) => {
            commands::ensure_ollama(&client).await?;
            commands::pull(&client, model).await?;
        }
        Some(Commands::Experimental { command }) => match command {
            ExperimentalCommand::Doctor => commands::experimental_doctor(&client).await?,
            ExperimentalCommand::Bench(args) => {
                commands::experimental_bench(&client, args.clone()).await?
            }
        },
        None => {
            commands::ensure_ollama(&client).await?;
            commands::run_agent_or_tui(&cli, &client, backend.clone(), &cli.task).await?;
        }
    }

    Ok(())
}
