//! High-level command implementations: setup, doctor, listing, one-shot agent.
//!
//! The module is split by concern:
//! - [`setup`]: Ollama onboarding (`/setup`).
//! - [`models`]: model listing (`list-models`).
//! - [`doctor`]: connectivity/health report (`doctor`).
//! - [`agent`]: one-shot agent/TUI dispatch.

mod agent;
mod doctor;
mod models;
mod setup;

pub(crate) use agent::run_agent_or_tui;
pub(crate) use doctor::{doctor, doctor_lines};
pub(crate) use models::list_models;
pub(crate) use setup::{setup, setup_with_status};

use crate::ollama::OllamaClient;
use anyhow::{bail, Result};
use std::process::Stdio;
use std::time::Duration;

/// How many times to poll Ollama while waiting for it to start.
pub(crate) const OLLAMA_START_POLL_ATTEMPTS: u32 = 20;
pub(crate) const SETUP_START_POLL_ATTEMPTS: u32 = 40;
pub(crate) const OLLAMA_POLL_INTERVAL_MILLIS: u64 = 500;

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
        "Cannot reach Ollama at {}.\\n\\\
         Run `openbatrangs setup` to install/start it automatically, or start it manually with `ollama serve`.",
        client.base_url
    )
}

/// Check whether the `ollama` executable is present on `PATH`.
pub(crate) fn has_ollama_binary() -> bool {
    std::process::Command::new("sh")
        .arg("-lc")
        .arg("command -v ollama")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|status| status.success())
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn poll_constants_are_sane() {
        const _: () = {
            assert!(OLLAMA_START_POLL_ATTEMPTS > 0);
            assert!(SETUP_START_POLL_ATTEMPTS >= OLLAMA_START_POLL_ATTEMPTS);
            assert!(OLLAMA_POLL_INTERVAL_MILLIS > 0);
        };
    }
}
