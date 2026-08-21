//! High-level command implementations: setup, doctor, listing, one-shot agent.

use crate::agent::{self, AgentConfig};
use crate::banner;
use crate::cli::{agent_run_config, model_prefs_from_cli, AgentRunConfig, Cli, ModelPrefs};
use crate::model_select::{calculate_memory_budget, resolve_model, resolve_model_context};
use crate::models;
use crate::ollama::OllamaClient;
use anyhow::{bail, Context, Result};
use std::process::Stdio;
use std::time::Duration;

/// How many times to poll Ollama while waiting for it to start.
const OLLAMA_START_POLL_ATTEMPTS: u32 = 20;
const SETUP_START_POLL_ATTEMPTS: u32 = 40;
const OLLAMA_POLL_INTERVAL_MILLIS: u64 = 500;

/// Make sure Ollama is reachable.
///
/// If the `ollama` binary exists but the server is not running, this starts
/// `ollama serve` automatically and polls until it responds.
pub(crate) async fn ensure_ollama(client: &OllamaClient) -> Result<()> {
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
pub(crate) async fn setup(client: &OllamaClient) -> Result<()> {
    if client.is_available().await {
        println!("✅ Ollama already running at {}", client.base_url);
    } else {
        install_and_start_ollama(client).await?;
    }

    let mem_budget = calculate_memory_budget();
    let fallback = models::recommended_fallback_model(mem_budget);
    let tags = client.tags().await?;
    if !tags.iter().any(|model| model.name == fallback) {
        client.pull(fallback, &|msg| println!("{msg}")).await?;
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

/// Run the agent for a task, or start the TUI when the task is empty.
pub(crate) async fn run_agent_or_tui(
    cli: &Cli,
    client: &OllamaClient,
    task: &[String],
) -> Result<()> {
    if task.is_empty() {
        return crate::tui::run(cli, client).await;
    }

    banner::print_banner();
    let prefs = model_prefs_from_cli(cli);
    let config = agent_run_config(cli);
    run_agent_task(client, &config, &mut None, &prefs, &task.join(" ")).await
}

/// Run the agent once for a single task using stdout output.
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
    let selected = resolve_model(client, model_slot, prefs, mem_budget, &|msg| {
        println!("{msg}")
    })
    .await?;
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
        is_read_only: config.is_read_only || config.mode == crate::cli::AgentMode::Plan,
        should_confirm: config.should_confirm,
        show_thinking: true,
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

/// Print the model table to stdout.
pub(crate) async fn list_models(client: &OllamaClient, min_context: u64) -> Result<()> {
    for line in list_models_lines(client, min_context).await? {
        println!("{line}");
    }
    Ok(())
}

/// Build the model table as lines, used by both stdout and the TUI.
pub(crate) async fn list_models_lines(
    client: &OllamaClient,
    min_context: u64,
) -> Result<Vec<String>> {
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
pub(crate) async fn doctor(client: &OllamaClient, min_context: u64) -> Result<()> {
    for line in doctor_lines(client, min_context).await? {
        println!("{line}");
    }
    Ok(())
}

/// Build the doctor report as lines, used by both stdout and the TUI.
pub(crate) async fn doctor_lines(client: &OllamaClient, min_context: u64) -> Result<Vec<String>> {
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
