//! openBatarangs CLI entry point.
//!
//! Responsibilities:
//! - Parse command-line arguments and subcommands.
//! - Ensure Ollama is available (auto-start or one-time setup).
//! - Dispatch to the interactive TUI, one-shot agent, model listing, doctor,
//!   or setup flows.

mod agent;
mod banner;
mod models;
mod ollama;
mod perf;
mod tools;
mod tui;

use agent::AgentConfig;
use anyhow::{anyhow, bail, Context, Result};
use clap::{Parser, Subcommand};
use models::ModelScore;
use ollama::OllamaClient;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::Duration;

/// Default Ollama server address.
const DEFAULT_OLLAMA_URL: &str = "http://localhost:11434";

/// Default maximum agent iterations.
const DEFAULT_MAX_STEPS: usize = 12;

/// Default minimum acceptable context window for auto model selection.
const DEFAULT_MIN_CONTEXT: usize = 8_192;

/// Fraction of total system memory treated as the model memory budget.
const MEMORY_BUDGET_NUMERATOR: u64 = 3;
const MEMORY_BUDGET_DENOMINATOR: u64 = 4;

/// How many times to poll Ollama while waiting for it to start.
const OLLAMA_START_POLL_ATTEMPTS: u32 = 20;
const SETUP_START_POLL_ATTEMPTS: u32 = 40;
const OLLAMA_POLL_INTERVAL_MILLIS: u64 = 500;

/// Maximum number of auto-pull retries while selecting a model.
const MODEL_SELECT_MAX_ATTEMPTS: u32 = 3;

/// Models scoring below this are considered unsuitable for agentic coding.
const GOOD_MODEL_SCORE_THRESHOLD: f64 = 60.0;

/// Fallback context length when Ollama metadata is missing.
const FALLBACK_CONTEXT_LENGTH: u64 = 8_192;

/// Suffix of Ollama metadata keys that contain the context length.
const CONTEXT_LENGTH_KEY_SUFFIX: &str = ".context_length";

/// Command-line interface definition.
#[derive(Parser)]
#[command(
    name = "openbatrangs",
    version,
    about = "🦇 openBatarangs — interactive agentic coding CLI for local models via Ollama"
)]
struct Cli {
    /// Task for the coding agent, e.g. "fix the Rust build errors".
    /// With no task, openBatarangs starts an interactive TUI.
    #[arg(value_name = "TASK")]
    task: Vec<String>,

    /// Ollama server URL.
    #[arg(long, global = true, default_value = DEFAULT_OLLAMA_URL)]
    ollama_url: String,

    /// Model to use; auto-discovered when omitted.
    #[arg(short, long, global = true)]
    model: Option<String>,

    /// Working directory for the agent.
    #[arg(long, global = true, default_value = ".")]
    cwd: PathBuf,

    /// Maximum agent steps.
    #[arg(long, global = true, default_value_t = DEFAULT_MAX_STEPS)]
    max_steps: usize,

    /// Read-only mode: no file writes or shell commands.
    #[arg(long = "read-only", global = true)]
    is_read_only: bool,

    /// Ask before each file write or shell command.
    #[arg(long = "confirm", global = true)]
    should_confirm: bool,

    /// Do not auto-pull a recommended model when none is suitable.
    #[arg(long = "no-auto-pull", global = true)]
    is_auto_pull_disabled: bool,

    /// Minimum context window for auto model selection.
    #[arg(long, global = true, default_value_t = DEFAULT_MIN_CONTEXT)]
    min_context: usize,

    #[command(subcommand)]
    command: Option<Commands>,
}

/// Available subcommands.
#[derive(Subcommand)]
enum Commands {
    /// Run the coding agent (same as passing a task directly).
    Agent {
        /// Task description.
        #[arg(value_name = "TASK")]
        task: Vec<String>,
    },
    /// List locally installed Ollama models with agent scores.
    ListModels,
    /// Check Ollama connectivity and recommend the best model.
    Doctor,
    /// Install/start Ollama and pull a recommended coding model.
    Setup,
}

/// Preferences used for automatic model selection.
#[derive(Clone)]
pub(crate) struct ModelPrefs {
    /// Explicit model tag; `None` means auto-select.
    pub(crate) model: Option<String>,
    /// If `true`, never auto-pull a model.
    pub(crate) is_auto_pull_disabled: bool,
    /// Minimum acceptable context window.
    pub(crate) min_context: u64,
}

/// Agent execution mode selected in the TUI.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum AgentMode {
    /// Full agentic loop with tools.
    Agent,
    /// Planning-only: read-only, no writes or shell commands.
    Plan,
}

/// Runtime settings shared by the one-shot agent and the TUI worker.
#[derive(Clone)]
pub(crate) struct AgentRunConfig {
    /// Workspace directory for the agent.
    pub(crate) cwd: PathBuf,
    /// Maximum agent iterations.
    pub(crate) max_steps: usize,
    /// Disable file writes and shell commands.
    pub(crate) is_read_only: bool,
    /// Ask before mutating actions.
    pub(crate) should_confirm: bool,
    /// Agent vs planning mode.
    pub(crate) mode: AgentMode,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    let client = OllamaClient::new(&cli.ollama_url)?;

    match &cli.command {
        Some(Commands::Setup) => setup(&client).await?,
        Some(Commands::ListModels) => {
            ensure_ollama(&client).await?;
            list_models(&client, cli.min_context as u64).await?;
        }
        Some(Commands::Doctor) => {
            ensure_ollama(&client).await?;
            doctor(&client, cli.min_context as u64).await?;
        }
        Some(Commands::Agent { task }) => {
            ensure_ollama(&client).await?;
            run_agent_or_tui(&cli, &client, task).await?;
        }
        None => {
            ensure_ollama(&client).await?;
            run_agent_or_tui(&cli, &client, &cli.task).await?;
        }
    }

    Ok(())
}

/// Build model-selection preferences from parsed CLI arguments.
fn model_prefs_from_cli(cli: &Cli) -> ModelPrefs {
    ModelPrefs {
        model: cli.model.clone(),
        is_auto_pull_disabled: cli.is_auto_pull_disabled,
        min_context: cli.min_context as u64,
    }
}

/// Build the shared agent runtime configuration from parsed CLI arguments.
fn agent_run_config(cli: &Cli) -> AgentRunConfig {
    AgentRunConfig {
        cwd: cli.cwd.clone(),
        max_steps: cli.max_steps,
        is_read_only: cli.is_read_only,
        should_confirm: cli.should_confirm,
        mode: AgentMode::Agent,
    }
}

/// Run the agent for a task, or start the TUI when the task is empty.
async fn run_agent_or_tui(cli: &Cli, client: &OllamaClient, task: &[String]) -> Result<()> {
    if task.is_empty() {
        return tui::run(cli, client).await;
    }

    banner::print_banner();
    let prefs = model_prefs_from_cli(cli);
    let config = agent_run_config(cli);
    run_agent_task(client, &config, &mut None, &prefs, &task.join(" ")).await
}

/// Make sure Ollama is reachable.
///
/// If the `ollama` binary exists but the server is not running, this starts
/// `ollama serve` automatically and polls until it responds.
///
/// # Arguments
/// - `client`: Ollama HTTP client.
///
/// # Returns
/// `Ok(())` when the server is reachable; otherwise a helpful error.
async fn ensure_ollama(client: &OllamaClient) -> Result<()> {
    if client.is_available().await {
        return Ok(());
    }

    if has_ollama_binary() {
        println!("🔄 Ollama server is not running — starting `ollama serve`...");
        let _child = std::process::Command::new("ollama")
            .arg("serve")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn();

        for _ in 0..OLLAMA_START_POLL_ATTEMPTS {
            tokio::time::sleep(Duration::from_millis(OLLAMA_POLL_INTERVAL_MILLIS)).await;
            if client.is_available().await {
                println!("✅ Ollama started at {}", client.base_url);
                return Ok(());
            }
        }
    }

    bail!(
        "Cannot reach Ollama at {}.\n\
         Run `openbatrangs setup` to install/start it automatically, or start it manually with `ollama serve`.",
        client.base_url
    )
}

/// Check whether the `ollama` executable is present on `PATH`.
fn has_ollama_binary() -> bool {
    std::process::Command::new("sh")
        .arg("-lc")
        .arg("command -v ollama")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|status| status.success())
        .unwrap_or(false)
}

/// One-command onboarding: install/start Ollama and pull a recommended model.
///
/// # Arguments
/// - `client`: Ollama HTTP client.
///
/// # Returns
/// `Ok(())` once Ollama is running and a default coding model is installed.
async fn setup(client: &OllamaClient) -> Result<()> {
    if client.is_available().await {
        println!("✅ Ollama already running at {}", client.base_url);
    } else {
        install_and_start_ollama(client).await?;
    }

    let mem_budget = calculate_memory_budget();
    let fallback = models::recommended_fallback_model(mem_budget);
    let tags = client.tags().await?;
    if !tags.iter().any(|model| model.name == fallback) {
        client.pull(fallback).await?;
    }
    println!("✅ Setup complete. Default coding model: {fallback}");
    println!("   Start chatting: openbatrangs");
    Ok(())
}

/// Install Ollama if missing, then start `ollama serve` and wait for it.
async fn install_and_start_ollama(client: &OllamaClient) -> Result<()> {
    if !has_ollama_binary() {
        println!("⬇️  Ollama not found. Installing with the official script (may ask for sudo)...");
        let status = std::process::Command::new("sh")
            .arg("-c")
            .arg("curl -fsSL https://ollama.com/install.sh | sh")
            .status()
            .context("failed to run Ollama installer")?;
        if !status.success() {
            bail!("Ollama install failed. Install manually from https://ollama.com/download/linux");
        }
    }

    println!("🔄 Starting Ollama...");
    let _child = std::process::Command::new("ollama")
        .arg("serve")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn();

    for _ in 0..SETUP_START_POLL_ATTEMPTS {
        tokio::time::sleep(Duration::from_millis(OLLAMA_POLL_INTERVAL_MILLIS)).await;
        if client.is_available().await {
            println!("✅ Ollama started at {}", client.base_url);
            return Ok(());
        }
    }

    bail!("Ollama was installed but did not start. Try `ollama serve` manually.")
}

/// Run the agent once for a single task using stdout output.
///
/// # Arguments
/// - `client`: Ollama HTTP client.
/// - `config`: shared agent runtime configuration.
/// - `model_slot`: current model; updated after selection.
/// - `prefs`: model selection preferences.
/// - `task`: task text.
///
/// # Returns
/// `Ok(())` when the agent finishes.
async fn run_agent_task(
    client: &OllamaClient,
    config: &AgentRunConfig,
    model_slot: &mut Option<String>,
    prefs: &ModelPrefs,
    task: &str,
) -> Result<()> {
    let task = task.trim();
    if task.is_empty() {
        bail!("empty task");
    }

    let mem_budget = calculate_memory_budget();
    let selected = resolve_model(client, model_slot, prefs, mem_budget).await?;
    *model_slot = Some(selected.name.clone());

    let model_context = resolve_model_context(client, &selected.name).await?;
    println!(
        "\n🚀 Agent\n   model:   {}\n   cwd:     {}\n   steps:   {}\n   context: {}",
        selected.name,
        config.cwd.display(),
        config.max_steps,
        model_context
    );

    let agent_config = AgentConfig {
        cwd: config.cwd.clone(),
        max_steps: config.max_steps,
        is_read_only: config.is_read_only || config.mode == AgentMode::Plan,
        should_confirm: config.should_confirm,
    };

    let mut reporter = agent::StdoutReporter;
    agent::run_agent(
        &agent_config,
        client,
        &selected.name,
        model_context,
        task,
        &mut reporter,
    )
    .await
}

/// Resolve a model from an optional explicit slot or auto-selection.
///
/// # Arguments
/// - `client`: Ollama HTTP client.
/// - `model_slot`: explicit model tag, if any.
/// - `prefs`: model selection preferences.
/// - `mem_budget`: usable memory in bytes.
///
/// # Returns
/// The selected model score.
pub(crate) async fn resolve_model(
    client: &OllamaClient,
    model_slot: &Option<String>,
    prefs: &ModelPrefs,
    mem_budget: u64,
) -> Result<ModelScore> {
    match model_slot {
        Some(name) => {
            let explicit = ModelPrefs {
                model: Some(name.clone()),
                ..prefs.clone()
            };
            select_model(client, &explicit, mem_budget).await
        }
        None => select_model(client, prefs, mem_budget).await,
    }
}

/// Select the best model, auto-pulling a fallback when needed.
///
/// # Arguments
/// - `client`: Ollama HTTP client.
/// - `prefs`: model selection preferences.
/// - `mem_budget`: usable memory in bytes.
///
/// # Returns
/// The chosen model score.
async fn select_model(
    client: &OllamaClient,
    prefs: &ModelPrefs,
    mem_budget: u64,
) -> Result<ModelScore> {
    let mut attempts = 0u32;
    loop {
        attempts += 1;
        if attempts > MODEL_SELECT_MAX_ATTEMPTS {
            bail!("could not find a suitable model after auto-pulling");
        }

        if let Some(explicit) = &prefs.model {
            return select_explicit_model(client, explicit, prefs.min_context, mem_budget).await;
        }

        let tags = client.tags().await?;
        if tags.is_empty() {
            if prefs.is_auto_pull_disabled {
                bail!("no models installed and --no-auto-pull is set");
            }
            pull_fallback(client, mem_budget).await?;
            continue;
        }

        let scored = models::score_models(&tags, mem_budget, prefs.min_context);
        if let Some(best) = scored.first() {
            if best.score < GOOD_MODEL_SCORE_THRESHOLD && !prefs.is_auto_pull_disabled {
                println!(
                    "⚠️  Best local model '{}' scores {:.0}/100 for agentic coding.",
                    best.name, best.score
                );
                pull_fallback(client, mem_budget).await?;
                continue;
            }
            return Ok(best.clone());
        }

        if prefs.is_auto_pull_disabled {
            bail!(
                "no installed model meets --min-context={}",
                prefs.min_context
            );
        }
        pull_fallback(client, mem_budget).await?;
    }
}

/// Validate and score an explicitly requested model tag.
async fn select_explicit_model(
    client: &OllamaClient,
    explicit: &str,
    min_context: u64,
    mem_budget: u64,
) -> Result<ModelScore> {
    if models::looks_like_path(explicit) {
        bail!(
            "'{explicit}' looks like a file path. openBatarangs now uses Ollama model tags.\n\
             Just run `openbatrangs setup` and it will pull a coding model for you."
        );
    }
    let tags = client.tags().await?;
    let model = tags
        .iter()
        .find(|model| model.name == explicit)
        .ok_or_else(|| {
            anyhow!("Model '{explicit}' is not installed. Run `openbatrangs setup` to auto-install a model.")
        })?;
    models::score_model(model, mem_budget, min_context)
        .ok_or_else(|| anyhow!("model '{explicit}' has context below --min-context"))
}

/// Pull the recommended fallback model and print progress.
async fn pull_fallback(client: &OllamaClient, mem_budget: u64) -> Result<()> {
    let fallback = models::recommended_fallback_model(mem_budget);
    println!("⬇️  Pulling a better default: {fallback}");
    client.pull(fallback).await
}

/// Resolve the context length of a model from Ollama metadata.
///
/// # Arguments
/// - `client`: Ollama HTTP client.
/// - `model`: model tag.
///
/// # Returns
/// Context length in tokens, or `FALLBACK_CONTEXT_LENGTH` if unknown.
pub(crate) async fn resolve_model_context(client: &OllamaClient, model: &str) -> Result<u64> {
    let show = client.show(model).await?;
    let context = show
        .get("model_info")
        .and_then(|info| info.as_object())
        .and_then(|object| {
            object.iter().find_map(|(key, value)| {
                if key.ends_with(CONTEXT_LENGTH_KEY_SUFFIX) {
                    value.as_u64()
                } else {
                    None
                }
            })
        })
        .or_else(|| {
            show.get("details")
                .and_then(|details| details.get("context_length"))
                .and_then(|value| value.as_u64())
        })
        .unwrap_or(FALLBACK_CONTEXT_LENGTH);
    Ok(context)
}

/// Compute the usable model memory budget from total system memory.
pub(crate) fn calculate_memory_budget() -> u64 {
    models::total_system_memory_bytes() * MEMORY_BUDGET_NUMERATOR / MEMORY_BUDGET_DENOMINATOR
}

/// Print the model table to stdout.
async fn list_models(client: &OllamaClient, min_context: u64) -> Result<()> {
    for line in list_models_lines(client, min_context).await? {
        println!("{line}");
    }
    Ok(())
}

/// Build the model table as lines, used by both stdout and the TUI.
async fn list_models_lines(client: &OllamaClient, min_context: u64) -> Result<Vec<String>> {
    let tags = client.tags().await?;
    if tags.is_empty() {
        return Ok(vec![
            "No models installed. Run `openbatrangs setup` to auto-pull one.".to_string(),
        ]);
    }
    let mem_budget = calculate_memory_budget();
    let scored = models::score_models(&tags, mem_budget, min_context);
    let mut lines = vec![format!(
        "{:<28} {:>9} {:>8} {:>8} {:>8} {:>6}  Notes",
        "MODEL", "SIZE", "PARAMS", "CTX", "QUANT", "SCORE"
    )];
    for model in scored {
        lines.push(format!(
            "{:<28} {:>7.1}G {:>8} {:>7}K {:>8} {:>6.0}  {}",
            model.name,
            model.size_bytes as f64 / 1e9,
            model.parameter_size,
            model.context_length / 1000,
            model.quantization,
            model.score,
            model.reasons.join("; ")
        ));
    }
    Ok(lines)
}

/// Print the doctor report to stdout.
async fn doctor(client: &OllamaClient, min_context: u64) -> Result<()> {
    for line in doctor_lines(client, min_context).await? {
        println!("{line}");
    }
    Ok(())
}

/// Build the doctor report as lines, used by both stdout and the TUI.
async fn doctor_lines(client: &OllamaClient, min_context: u64) -> Result<Vec<String>> {
    let mut lines = vec![format!("✅ Ollama reachable at {}", client.base_url)];
    let tags = client.tags().await?;
    lines.push(format!("Installed models: {}", tags.len()));
    let mem_budget = calculate_memory_budget();
    let scored = models::score_models(&tags, mem_budget, min_context);
    match scored.first() {
        Some(best) => {
            lines.push(format!(
                "🏆 Best model for agentic coding: {} (score {:.0}/100)",
                best.name, best.score
            ));
            lines.push(format!(
                "   {:.1} GB, {} params, {} context, {}",
                best.size_bytes as f64 / 1e9,
                best.parameter_size,
                best.context_length,
                best.quantization
            ));
        }
        None => {
            lines.push(format!(
                "⚠️  No model meets the minimum context of {min_context}."
            ));
            lines.push("   Run `openbatrangs setup` to pull a recommended model.".to_string());
        }
    }
    lines.push(String::new());
    lines.push("⚡ Performance per watt per dollar:".to_string());
    lines.push("   - Local GPU inference on unified memory, no cloud API fees".to_string());
    lines.push(
        "   - Auto-picker deliberately chooses models that fit memory (Q4_K_M 3B-8B on 16GB Jetson)"
            .to_string(),
    );
    lines.push(
        "   - Keeps latency and power low while preserving a 32K+ context window".to_string(),
    );
    Ok(lines)
}
