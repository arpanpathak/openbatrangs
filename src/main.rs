mod agent;
mod models;
mod ollama;
mod tools;

use agent::AgentConfig;
use anyhow::{anyhow, bail, Result};
use clap::{Parser, Subcommand};
use models::ModelScore;
use ollama::OllamaClient;
use std::path::PathBuf;

#[derive(Parser)]
#[command(
    name = "openbatrangs",
    version,
    about = "🦇 openBatarangs — agentic coding CLI for local models via Ollama",
    arg_required_else_help = true
)]
struct Cli {
    /// Task for the coding agent, e.g. "fix the Rust build errors"
    #[arg(value_name = "TASK")]
    task: Vec<String>,

    /// Ollama server URL
    #[arg(long, global = true, default_value = "http://localhost:11434")]
    ollama_url: String,

    /// Model to use; auto-discovered when omitted
    #[arg(short, long, global = true)]
    model: Option<String>,

    /// Working directory for the agent
    #[arg(long, global = true, default_value = ".")]
    cwd: PathBuf,

    /// Maximum agent steps
    #[arg(long, global = true, default_value_t = 12)]
    max_steps: usize,

    /// Read-only mode: no file writes or shell commands
    #[arg(long, global = true)]
    read_only: bool,

    /// Ask before each file write or shell command
    #[arg(long, global = true)]
    confirm: bool,

    /// Do not auto-pull a recommended model when none is suitable
    #[arg(long, global = true)]
    no_auto_pull: bool,

    /// Minimum context window for auto model selection
    #[arg(long, global = true, default_value_t = 8192)]
    min_context: usize,

    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// Run the coding agent (same as passing a task directly)
    Agent {
        /// Task description
        #[arg(value_name = "TASK")]
        task: Vec<String>,
    },
    /// List locally installed Ollama models with agent scores
    ListModels,
    /// Check Ollama connectivity and recommend the best model
    Doctor,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    let client = OllamaClient::new(&cli.ollama_url)?;

    if !client.is_available().await {
        bail!(
            "Cannot reach Ollama at {}. Start it with `ollama serve`, or pass --ollama-url.",
            cli.ollama_url
        );
    }

    match &cli.command {
        Some(Commands::ListModels) => list_models(&client, cli.min_context as u64).await?,
        Some(Commands::Doctor) => doctor(&client, cli.min_context as u64).await?,
        Some(Commands::Agent { task }) => run_agent_cli(&cli, &client, task.clone()).await?,
        None => run_agent_cli(&cli, &client, cli.task.clone()).await?,
    }

    Ok(())
}

async fn run_agent_cli(cli: &Cli, client: &OllamaClient, task: Vec<String>) -> Result<()> {
    let task = if task.is_empty() {
        // Interactive prompt.
        print!("🦇 Task: ");
        std::io::Write::flush(&mut std::io::stdout())?;
        let mut input = String::new();
        std::io::stdin().read_line(&mut input)?;
        input.trim().to_string()
    } else {
        task.join(" ")
    };

    if task.trim().is_empty() {
        bail!("no task provided");
    }

    let mem_budget = models::total_system_memory_bytes() * 3 / 4;
    let model = select_model(client, cli, mem_budget).await?;

    let model_context = resolve_model_context(client, &model.name).await?;

    println!(
        "\n🚀 Starting agent\n   model: {}\n   cwd:   {}\n   steps: {}\n   context: {}",
        model.name,
        cli.cwd.display(),
        cli.max_steps,
        model_context
    );

    let config = AgentConfig {
        cwd: cli.cwd.clone(),
        max_steps: cli.max_steps,
        read_only: cli.read_only,
        confirm: cli.confirm,
    };

    agent::run_agent(&config, client, &model.name, model_context, &task).await
}

async fn select_model(client: &OllamaClient, cli: &Cli, mem_budget: u64) -> Result<ModelScore> {
    let mut attempts = 0usize;
    loop {
        attempts += 1;
        if attempts > 3 {
            bail!("could not find a suitable model after auto-pulling");
        }

        if let Some(explicit) = &cli.model {
            if models::looks_like_path(explicit) {
                bail!(
                    "'{explicit}' looks like a file path. openBatarangs now uses Ollama model tags.\n\
                     Install a model first: ollama pull qwen2.5-coder:7b\n\
                     Then pass --model qwen2.5-coder:7b"
                );
            }
            let tags = client.tags().await?;
            if !tags.iter().any(|m| m.name == *explicit) {
                bail!(
                    "Model '{explicit}' is not installed. Install it with: ollama pull {explicit}"
                );
            }
            return models::score_model(
                &tags.iter().find(|m| m.name == *explicit).unwrap(),
                mem_budget,
                cli.min_context as u64,
            )
            .ok_or_else(|| anyhow!("model '{explicit}' has context below --min-context"));
        }

        let tags = client.tags().await?;
        if tags.is_empty() {
            if cli.no_auto_pull {
                bail!("no models installed and --no-auto-pull is set");
            }
            let fallback = models::recommended_fallback_model(mem_budget);
            client.pull(fallback).await?;
            continue;
        }

        let scored = models::score_models(&tags, mem_budget, cli.min_context as u64);
        if let Some(best) = scored.first() {
            // If the best candidate is not a coding model and scores poorly, try to auto-pull.
            if best.score < 60.0 && !cli.no_auto_pull {
                println!(
                    "⚠️  Best local model '{}' scores {:.0}/100 for agentic coding.",
                    best.name, best.score
                );
                let fallback = models::recommended_fallback_model(mem_budget);
                println!("⬇️  Pulling a better default: {fallback}");
                client.pull(fallback).await?;
                continue;
            }
            return Ok(best.clone());
        }

        if cli.no_auto_pull {
            bail!("no installed model meets --min-context={}", cli.min_context);
        }
        let fallback = models::recommended_fallback_model(mem_budget);
        println!("⬇️  No suitable model found. Pulling {fallback}");
        client.pull(fallback).await?;
    }
}

async fn resolve_model_context(client: &OllamaClient, model: &str) -> Result<u64> {
    let show = client.show(model).await?;
    let context = show
        .get("model_info")
        .and_then(|m| m.as_object())
        .and_then(|obj| {
            obj.iter().find_map(|(k, v)| {
                if k.ends_with(".context_length") {
                    v.as_u64()
                } else {
                    None
                }
            })
        })
        .or_else(|| {
            show.get("details")
                .and_then(|d| d.get("context_length"))
                .and_then(|v| v.as_u64())
        })
        .unwrap_or(8192);
    Ok(context)
}

async fn list_models(client: &OllamaClient, min_context: u64) -> Result<()> {
    let tags = client.tags().await?;
    if tags.is_empty() {
        println!("No models installed. Run: ollama pull qwen2.5-coder:7b");
        return Ok(());
    }
    let mem_budget = models::total_system_memory_bytes() * 3 / 4;
    let scored = models::score_models(&tags, mem_budget, min_context);
    println!(
        "{:<28} {:>9} {:>8} {:>8} {:>8} {:>6}  Notes",
        "MODEL", "SIZE", "PARAMS", "CTX", "QUANT", "SCORE"
    );
    for m in scored {
        println!(
            "{:<28} {:>7.1}G {:>8} {:>7}K {:>8} {:>6.0}  {}",
            m.name,
            m.size_bytes as f64 / 1e9,
            m.parameter_size,
            m.context_length / 1000,
            m.quantization,
            m.score,
            m.reasons.join("; ")
        );
    }
    Ok(())
}

async fn doctor(client: &OllamaClient, min_context: u64) -> Result<()> {
    println!("✅ Ollama reachable at {}", client.base_url);
    let tags = client.tags().await?;
    println!("Installed models: {}", tags.len());
    let mem_budget = models::total_system_memory_bytes() * 3 / 4;
    let scored = models::score_models(&tags, mem_budget, min_context);
    match scored.first() {
        Some(best) => {
            println!(
                "🏆 Best model for agentic coding: {} (score {:.0}/100)",
                best.name, best.score
            );
            println!(
                "   {:.1} GB, {} params, {} context, {}",
                best.size_bytes as f64 / 1e9,
                best.parameter_size,
                best.context_length,
                best.quantization
            );
        }
        None => {
            println!("⚠️  No model meets the minimum context of {min_context}.");
            println!(
                "   Try: ollama pull {}",
                models::recommended_fallback_model(mem_budget)
            );
        }
    }
    Ok(())
}
