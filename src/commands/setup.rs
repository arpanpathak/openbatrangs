//! Ollama onboarding: install/start Ollama and pull a recommended model.

use super::{has_ollama_binary, SETUP_START_POLL_ATTEMPTS};
use crate::model_select::calculate_memory_budget;
use crate::models;
use crate::ollama::OllamaClient;
use anyhow::{bail, Context, Result};
use std::process::Stdio;
use std::time::Duration;

/// One-command onboarding: install/start Ollama and pull a recommended model.
pub(crate) async fn setup(client: &OllamaClient) -> Result<()> {
    setup_with_status(client, &|msg| println!("{msg}")).await
}

/// Onboarding with a status callback instead of printing directly to stdout.
///
/// The TUI uses this variant so `/setup` can stream progress into the chat log
/// instead of corrupting the alternate screen with `println!`.
pub(crate) async fn setup_with_status(
    client: &OllamaClient,
    on_status: &(dyn Fn(&str) + Sync),
) -> Result<()> {
    if client.is_available().await {
        on_status(&format!("✅ Ollama already running at {}", client.base_url));
    } else {
        install_and_start_ollama(client, on_status).await?;
    }

    let mem_budget = calculate_memory_budget();
    let fallback = models::recommended_fallback_model(mem_budget);
    let tags = client.tags().await?;
    if !tags.iter().any(|model| model.name == fallback) {
        client.pull(fallback, on_status).await?;
    }
    on_status(&format!(
        "✅ Setup complete. Default coding model: {fallback}"
    ));
    on_status("   Start chatting: openbatrangs");
    Ok(())
}

/// Install Ollama if missing, then start `ollama serve` and wait for it.
async fn install_and_start_ollama(
    client: &OllamaClient,
    on_status: &(dyn Fn(&str) + Sync),
) -> Result<()> {
    if !has_ollama_binary() {
        on_status(
            "⬇️  Ollama not found. Installing with the official script (may ask for sudo)...",
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

    on_status("🔄 Starting Ollama...");
    let _child = std::process::Command::new("ollama")
        .arg("serve")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn();

    for _ in 0..SETUP_START_POLL_ATTEMPTS {
        tokio::time::sleep(Duration::from_millis(super::OLLAMA_POLL_INTERVAL_MILLIS)).await;
        if client.is_available().await {
            on_status(&format!("✅ Ollama started at {}", client.base_url));
            return Ok(());
        }
    }

    bail!("Ollama was installed but did not start. Try `ollama serve` manually.")
}
