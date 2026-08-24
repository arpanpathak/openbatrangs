//! # Ollama onboarding: install/start Ollama and pull a recommended model
//!
//! `setup` is the zero-knowledge entry point: it makes sure Ollama is running
//! (installing it if necessary) and pulls a coding model that fits the device's
//! memory. The status callback variant keeps the TUI's alternate screen intact.
//!
//! ## References
//!
//! - Ollama install script: <https://ollama.com/download/linux>
//! - Ollama pull API: <https://github.com/ollama/ollama/blob/main/docs/api.md#pull-a-model>

use super::{has_ollama_binary, start_ollama_server, wait_until_available};
use crate::constants::commands::{OLLAMA_INSTALL_SCRIPT_URL, SETUP_START_POLL_ATTEMPTS};
use crate::model_select::calculate_memory_budget;
use crate::models;
use crate::ollama::OllamaClient;
use anyhow::{bail, Context, Result};
use std::process::Stdio;

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
        install_ollama(on_status)?;
    }

    on_status("🔄 Starting Ollama...");
    start_ollama_server();
    if wait_until_available(client, SETUP_START_POLL_ATTEMPTS).await? {
        on_status(&format!("✅ Ollama started at {}", client.base_url));
        return Ok(());
    }

    bail!("Ollama was installed but did not start. Try `ollama serve` manually.")
}

/// Install Ollama with the official script (may ask for sudo).
fn install_ollama(on_status: &(dyn Fn(&str) + Sync)) -> Result<()> {
    on_status("⬇️  Ollama not found. Installing with the official script (may ask for sudo)...");
    let status = std::process::Command::new("sh")
        .arg("-c")
        .arg(format!("curl -fsSL {OLLAMA_INSTALL_SCRIPT_URL} | sh"))
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .status()
        .context("failed to run Ollama installer")?;
    if !status.success() {
        bail!("Ollama install failed. Install manually from https://ollama.com/download/linux");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn setup_with_status_pulls_fallback_when_missing() {
        use crate::test_support::{spawn_mock_server, MockResponse};
        use std::sync::{Arc, Mutex};

        let base_url = spawn_mock_server(|path| match path {
            "/api/tags" => MockResponse::json("200 OK", r#"{"models":[]}"#),
            "/api/pull" => MockResponse::json("200 OK", "{\"status\":\"success\"}\n"),
            _ => MockResponse::text("404 Not Found", "no route"),
        })
        .await;
        let client = OllamaClient::new(&base_url).unwrap();
        let statuses = Arc::new(Mutex::new(Vec::new()));
        let captured = Arc::clone(&statuses);
        setup_with_status(&client, &move |msg| {
            captured.lock().unwrap().push(msg.to_string());
        })
        .await
        .unwrap();

        let statuses = statuses.lock().unwrap();
        assert!(statuses.iter().any(|s| s.contains("Setup complete")));
        assert!(statuses.iter().any(|s| s.contains("Pulling")));
    }
}
