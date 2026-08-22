//! High-level command implementations: setup, doctor, listing, one-shot agent.
//!
//! The module is split by concern:
//! - [`setup`]: Ollama onboarding (`/setup`).
//! - [`models`]: model listing (`list-models`).
//! - [`doctor`]: connectivity/health report (`doctor`).
//! - [`agent`]: one-shot agent/TUI dispatch.

mod agent;
mod doctor;
mod experimental;
mod models;
mod pull;
mod setup;

pub(crate) use agent::run_agent_or_tui;
pub(crate) use doctor::{doctor, doctor_lines};
pub(crate) use experimental::{experimental_bench, experimental_doctor};
pub(crate) use models::list_models;
pub(crate) use pull::pull;
pub(crate) use setup::{setup, setup_with_status};

use crate::constants::commands::{
    OLLAMA_BINARY, OLLAMA_POLL_INTERVAL_MILLIS, OLLAMA_START_POLL_ATTEMPTS,
};
use crate::ollama::OllamaClient;
use anyhow::{bail, Result};
use std::process::Stdio;
use std::time::Duration;

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
        start_ollama_server();
        if wait_until_available(client, OLLAMA_START_POLL_ATTEMPTS).await? {
            println!("✅ Ollama started at {}", client.base_url);
            return Ok(());
        }
    }

    bail!(
        "Cannot reach Ollama at {}.\nRun `openbatrangs setup` to install/start it automatically, or start it manually with `ollama serve`.",
        client.base_url
    )
}

/// Start `ollama serve` in the background without blocking.
pub(super) fn start_ollama_server() {
    let _ = std::process::Command::new(OLLAMA_BINARY)
        .arg("serve")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn();
}

/// Poll the Ollama server until it responds or the attempt budget runs out.
pub(super) async fn wait_until_available(client: &OllamaClient, attempts: u32) -> Result<bool> {
    for _ in 0..attempts {
        tokio::time::sleep(Duration::from_millis(OLLAMA_POLL_INTERVAL_MILLIS)).await;
        if client.is_available().await {
            return Ok(true);
        }
    }
    Ok(false)
}

/// Check whether the `ollama` executable is present on `PATH`.
pub(crate) fn has_ollama_binary() -> bool {
    std::process::Command::new("sh")
        .arg("-lc")
        .arg(format!("command -v {OLLAMA_BINARY}"))
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|status| status.success())
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use crate::constants::commands::{
        OLLAMA_POLL_INTERVAL_MILLIS, OLLAMA_START_POLL_ATTEMPTS, SETUP_START_POLL_ATTEMPTS,
    };

    #[test]
    fn poll_constants_are_sane() {
        const _: () = {
            assert!(OLLAMA_START_POLL_ATTEMPTS > 0);
            assert!(SETUP_START_POLL_ATTEMPTS >= OLLAMA_START_POLL_ATTEMPTS);
            assert!(OLLAMA_POLL_INTERVAL_MILLIS > 0);
        };
    }
}
