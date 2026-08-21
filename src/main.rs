mod agent;
mod banner;
mod models;
mod ollama;
mod tools;

use agent::AgentConfig;
use anyhow::{anyhow, bail, Context, Result};
use clap::{Parser, Subcommand};
use models::ModelScore;
use ollama::OllamaClient;
use rustyline::error::ReadlineError;
use rustyline::DefaultEditor;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::Duration;

#[derive(Parser)]
#[command(
    name = "openbatrangs",
    version,
    about = "🦇 openBatarangs — interactive agentic coding CLI for local models via Ollama"
)]
struct Cli {
    /// Task for the coding agent, e.g. "fix the Rust build errors".
    /// With no task, openBatarangs starts an interactive REPL.
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
    /// Install/start Ollama and pull a recommended coding model
    Setup,
}

#[derive(Clone)]
struct ModelPrefs {
    model: Option<String>,
    no_auto_pull: bool,
    min_context: u64,
}

struct ReplState {
    model: Option<String>,
    read_only: bool,
    confirm: bool,
    max_steps: usize,
    cwd: PathBuf,
    min_context: u64,
    no_auto_pull: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    let client = OllamaClient::new(&cli.ollama_url)?;

    if matches!(&cli.command, Some(Commands::Setup)) {
        setup(&client).await?;
        return Ok(());
    }

    ensure_ollama(&client).await?;

    if matches!(&cli.command, None) || matches!(&cli.command, Some(Commands::Agent { .. })) {
        banner::print_banner();
    }

    match &cli.command {
        Some(Commands::ListModels) => list_models(&client, cli.min_context as u64).await?,
        Some(Commands::Doctor) => doctor(&client, cli.min_context as u64).await?,
        Some(Commands::Setup) => unreachable!("setup is handled before ensure_ollama"),
        Some(Commands::Agent { task }) => {
            if task.is_empty() {
                run_repl(&cli, &client).await?;
            } else {
                let prefs = ModelPrefs {
                    model: cli.model.clone(),
                    no_auto_pull: cli.no_auto_pull,
                    min_context: cli.min_context as u64,
                };
                run_agent_task(
                    &client,
                    &cli.cwd,
                    cli.max_steps,
                    cli.read_only,
                    cli.confirm,
                    &mut None,
                    &prefs,
                    &task.join(" "),
                )
                .await?;
            }
        }
        None => {
            if cli.task.is_empty() {
                run_repl(&cli, &client).await?;
            } else {
                let prefs = ModelPrefs {
                    model: cli.model.clone(),
                    no_auto_pull: cli.no_auto_pull,
                    min_context: cli.min_context as u64,
                };
                run_agent_task(
                    &client,
                    &cli.cwd,
                    cli.max_steps,
                    cli.read_only,
                    cli.confirm,
                    &mut None,
                    &prefs,
                    &cli.task.join(" "),
                )
                .await?;
            }
        }
    }

    Ok(())
}

/// Make sure Ollama is reachable. If the binary exists but the server is not
/// running, start `ollama serve` automatically.
async fn ensure_ollama(client: &OllamaClient) -> Result<()> {
    if client.is_available().await {
        return Ok(());
    }

    let has_ollama = std::process::Command::new("sh")
        .arg("-lc")
        .arg("command -v ollama")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false);

    if has_ollama {
        println!("🔄 Ollama server is not running — starting `ollama serve`...");
        let _child = std::process::Command::new("ollama")
            .arg("serve")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn();

        for _ in 0..20 {
            tokio::time::sleep(Duration::from_millis(500)).await;
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

/// One-command onboarding: install/start Ollama and pull a recommended model.
async fn setup(client: &OllamaClient) -> Result<()> {
    if client.is_available().await {
        println!("✅ Ollama already running at {}", client.base_url);
    } else {
        let has_ollama = std::process::Command::new("sh")
            .arg("-lc")
            .arg("command -v ollama")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .map(|s| s.success())
            .unwrap_or(false);

        if !has_ollama {
            println!(
                "⬇️  Ollama not found. Installing with the official script (may ask for sudo)..."
            );
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

        let mut ok = false;
        for _ in 0..40 {
            tokio::time::sleep(Duration::from_millis(500)).await;
            if client.is_available().await {
                ok = true;
                break;
            }
        }
        if !ok {
            bail!("Ollama was installed but did not start. Try `ollama serve` manually.");
        }
        println!("✅ Ollama started at {}", client.base_url);
    }

    let mem_budget = models::total_system_memory_bytes() * 3 / 4;
    let fallback = models::recommended_fallback_model(mem_budget);
    let tags = client.tags().await?;
    if !tags.iter().any(|m| m.name == fallback) {
        client.pull(fallback).await?;
    }
    println!("✅ Setup complete. Default coding model: {fallback}");
    println!("   Start chatting: openbatrangs");
    Ok(())
}

/// Interactive DeepCode-style REPL.
async fn run_repl(cli: &Cli, client: &OllamaClient) -> Result<()> {
    let mut rl = DefaultEditor::new().context("failed to initialize line editor")?;
    let mut state = ReplState {
        model: cli.model.clone(),
        read_only: cli.read_only,
        confirm: cli.confirm,
        max_steps: cli.max_steps,
        cwd: cli.cwd.clone(),
        min_context: cli.min_context as u64,
        no_auto_pull: cli.no_auto_pull,
    };

    println!(
        "🦇 openBatarangs interactive agent\n\
         ⚡ local models via Ollama · auto model discovery\n\
         💡 type a task, or /help for commands\n"
    );

    loop {
        match rl.readline("openBatarangs> ") {
            Ok(line) => {
                let _ = rl.add_history_entry(line.as_str());
                let line = line.trim().to_string();
                if line.is_empty() {
                    continue;
                }
                if let Some(cmd) = line.strip_prefix('/') {
                    if handle_slash_command(&mut state, client, cmd).await? {
                        break;
                    }
                    continue;
                }
                let prefs = ModelPrefs {
                    model: state.model.clone(),
                    no_auto_pull: state.no_auto_pull,
                    min_context: state.min_context,
                };
                run_agent_task(
                    client,
                    &state.cwd,
                    state.max_steps,
                    state.read_only,
                    state.confirm,
                    &mut state.model,
                    &prefs,
                    &line,
                )
                .await?;
            }
            Err(ReadlineError::Interrupted) => {
                println!("\n👋 Bye!");
                break;
            }
            Err(ReadlineError::Eof) => {
                println!("\n👋 Bye!");
                break;
            }
            Err(err) => return Err(err.into()),
        }
    }
    Ok(())
}

/// Returns `true` if the REPL should exit.
async fn handle_slash_command(
    state: &mut ReplState,
    client: &OllamaClient,
    cmd: &str,
) -> Result<bool> {
    let mut parts = cmd.splitn(2, char::is_whitespace);
    let name = parts.next().unwrap_or("");
    let arg = parts.next().unwrap_or("").trim();

    match name {
        "help" | "h" => {
            println!(
                "Commands:\n\
                 \x20 /help          show this help\n\
                 \x20 /exit, /quit   leave the REPL\n\
                 \x20 /setup         install/start Ollama + pull a model\n\
                 \x20 /models        list installed models + scores\n\
                 \x20 /model <tag>   switch model (e.g. /model qwen2.5-coder:7b)\n\
                 \x20 /read-only     toggle read-only mode\n\
                 \x20 /confirm       toggle confirm-before-write/command\n\
                 \x20 /steps <n>     set max agent steps\n\
                 \x20 /cwd <path>    change workspace\n\
                 \x20 /doctor        check Ollama + best model\n\
                 \nAnything else is sent to the coding agent as a task."
            );
        }
        "exit" | "quit" => return Ok(true),
        "models" => list_models(client, state.min_context).await?,
        "model" => {
            if arg.is_empty() {
                match &state.model {
                    Some(m) => println!("Current model: {m}"),
                    None => println!("Auto mode — best model will be selected on first task."),
                }
            } else {
                let tags = client.tags().await?;
                if tags.iter().any(|m| m.name == arg) {
                    state.model = Some(arg.to_string());
                    println!("✅ Model set to {arg}");
                } else {
                    println!("❌ Model '{arg}' is not installed. Try /models or /setup.");
                }
            }
        }
        "read-only" => {
            state.read_only = !state.read_only;
            println!(
                "Read-only mode: {}",
                if state.read_only { "ON" } else { "OFF" }
            );
        }
        "confirm" => {
            state.confirm = !state.confirm;
            println!("Confirm mode: {}", if state.confirm { "ON" } else { "OFF" });
        }
        "steps" => match arg.parse::<usize>() {
            Ok(n) if n > 0 => {
                state.max_steps = n;
                println!("Max steps set to {n}");
            }
            _ => println!("Usage: /steps <positive number>"),
        },
        "cwd" => {
            if arg.is_empty() {
                println!("Workspace: {}", state.cwd.display());
            } else {
                state.cwd = PathBuf::from(arg);
                println!("Workspace set to {}", state.cwd.display());
            }
        }
        "doctor" => doctor(client, state.min_context).await?,
        "setup" => setup(client).await?,
        "clear" => print!("\x1b[2J\x1b[H"),
        _ => println!("Unknown command: /{name}. Try /help"),
    }
    Ok(false)
}

async fn run_agent_task(
    client: &OllamaClient,
    cwd: &std::path::Path,
    max_steps: usize,
    read_only: bool,
    confirm: bool,
    model_slot: &mut Option<String>,
    prefs: &ModelPrefs,
    task: &str,
) -> Result<()> {
    let task = task.trim();
    if task.is_empty() {
        bail!("empty task");
    }

    let mem_budget = models::total_system_memory_bytes() * 3 / 4;
    let selected = match model_slot {
        Some(name) => {
            let explicit = ModelPrefs {
                model: Some(name.clone()),
                ..prefs.clone()
            };
            select_model(client, &explicit, mem_budget).await?
        }
        None => select_model(client, prefs, mem_budget).await?,
    };
    *model_slot = Some(selected.name.clone());

    let model_context = resolve_model_context(client, &selected.name).await?;
    println!(
        "\n🚀 Agent\n   model:   {}\n   cwd:     {}\n   steps:   {}\n   context: {}",
        selected.name,
        cwd.display(),
        max_steps,
        model_context
    );

    let config = AgentConfig {
        cwd: cwd.to_path_buf(),
        max_steps,
        read_only,
        confirm,
    };

    agent::run_agent(&config, client, &selected.name, model_context, task).await
}

async fn select_model(
    client: &OllamaClient,
    prefs: &ModelPrefs,
    mem_budget: u64,
) -> Result<ModelScore> {
    let mut attempts = 0usize;
    loop {
        attempts += 1;
        if attempts > 3 {
            bail!("could not find a suitable model after auto-pulling");
        }

        if let Some(explicit) = &prefs.model {
            if models::looks_like_path(explicit) {
                bail!(
                    "'{explicit}' looks like a file path. openBatarangs now uses Ollama model tags.\n\
                     Just run `openbatrangs setup` and it will pull a coding model for you."
                );
            }
            let tags = client.tags().await?;
            if !tags.iter().any(|m| m.name == *explicit) {
                bail!(
                    "Model '{explicit}' is not installed. Run `openbatrangs setup` to auto-install a model."
                );
            }
            return models::score_model(
                &tags.iter().find(|m| m.name == *explicit).unwrap(),
                mem_budget,
                prefs.min_context,
            )
            .ok_or_else(|| anyhow!("model '{explicit}' has context below --min-context"));
        }

        let tags = client.tags().await?;
        if tags.is_empty() {
            if prefs.no_auto_pull {
                bail!("no models installed and --no-auto-pull is set");
            }
            let fallback = models::recommended_fallback_model(mem_budget);
            client.pull(fallback).await?;
            continue;
        }

        let scored = models::score_models(&tags, mem_budget, prefs.min_context);
        if let Some(best) = scored.first() {
            if best.score < 60.0 && !prefs.no_auto_pull {
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

        if prefs.no_auto_pull {
            bail!(
                "no installed model meets --min-context={}",
                prefs.min_context
            );
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
        println!("No models installed. Run `openbatrangs setup` to auto-pull one.");
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
            println!("   Run `openbatrangs setup` to pull a recommended model.");
        }
    }
    println!("\n⚡ Performance per watt per dollar:");
    println!("   - Local GPU inference on unified memory, no cloud API fees");
    println!("   - Auto-picker deliberately chooses models that fit memory (Q4_K_M 3B-8B on 16GB Jetson)");
    println!("   - Keeps latency and power low while preserving a 32K+ context window");
    Ok(())
}
