//! # High-level command implementations: setup, doctor, listing, one-shot agent
//!
//! The module is split by concern:
//! - [`setup`]: Ollama onboarding (`/setup`).
//! - [`models`]: model listing (`list-models`).
//! - [`doctor`]: connectivity/health report (`doctor`).
//! - [`agent`]: one-shot agent/TUI dispatch.
//! - [`update`]: release check and install (`update`).
//! - [`transcript`]: the `--transcript-out` file of a one-shot run.
//!
//! [`ensure_ollama`] is the shared bootstrap: it probes the server, starts
//! `ollama serve` when the binary exists, and polls until the server responds.
//!
//! ## References
//!
//! - Ollama server CLI: <https://github.com/ollama/ollama>
//! - Polling pattern: <https://en.wikipedia.org/wiki/Polling_(computer_science)>

mod agent;
mod doctor;
mod models;
mod pull;
mod setup;
mod transcript;
mod update;

pub(crate) use agent::run_agent_or_tui;
pub(crate) use doctor::{doctor, doctor_lines};
pub(crate) use models::list_models;
pub(crate) use pull::pull;
pub(crate) use setup::{setup, setup_with_status};
pub(crate) use update::update;

use crate::constants::commands::{
    OLLAMA_BINARY, OLLAMA_POLL_INTERVAL_MILLIS, OLLAMA_START_POLL_ATTEMPTS,
};
use crate::ollama::OllamaClient;
use anyhow::{Result, anyhow, bail};
use std::process::{Command, Stdio};
use std::time::Duration;

/// Make sure the model server is reachable.
///
/// When a local Ollama does not answer but its binary is on `PATH`, this
/// starts `ollama serve` and polls until the server answers.
pub(crate) async fn ensure_ollama(client: &OllamaClient) -> Result<()> {
    if client.is_available().await {
        return Ok(());
    }

    start_unreachable_server(client, OLLAMA_BINARY).await
}

/// Start a server that did not answer. Only a local Ollama can be started; an
/// OpenAI-compatible server runs elsewhere, so its failure is reported as is.
async fn start_unreachable_server(client: &OllamaClient, binary: &str) -> Result<()> {
    if client.is_openai() {
        bail!(
            "Cannot reach the model server at {}, or the access key was refused.\n\
             Check the address and the key file given with `--api-key-file`.",
            client.base_url
        );
    }

    start_installed_ollama(client, binary).await
}

/// Run `<binary> serve` when `binary` is on `PATH` and wait for the server.
async fn start_installed_ollama(client: &OllamaClient, binary: &str) -> Result<()> {
    if !binary_on_path(binary) {
        return Err(ollama_unreachable(client));
    }

    println!("🔄 Ollama server is not running — starting `ollama serve`...");
    let has_started = start_and_wait(client, binary, OLLAMA_START_POLL_ATTEMPTS).await;

    report_start(client, has_started)
}

/// Print that the server started, or return the unreachable error.
fn report_start(client: &OllamaClient, has_started: bool) -> Result<()> {
    if !has_started {
        return Err(ollama_unreachable(client));
    }

    println!("✅ Ollama started at {}", client.base_url);

    Ok(())
}

/// The error for a local Ollama that does not answer, with the ways to start it.
fn ollama_unreachable(client: &OllamaClient) -> anyhow::Error {
    anyhow!(
        "Cannot reach Ollama at {}.\nRun `openbatrangs setup` to install/start it automatically, or start it manually with `ollama serve`.",
        client.base_url
    )
}

/// Start `<binary> serve` in the background and wait up to `attempts` poll
/// intervals for the server to answer. Returns `false` when the process
/// cannot be started or the server does not answer in time.
pub(super) async fn start_and_wait(client: &OllamaClient, binary: &str, attempts: u32) -> bool {
    start_server(binary) && wait_until_available(client, attempts).await
}

/// Spawn `<binary> serve` with its output discarded. Returns whether the
/// process was spawned; the server inside it may still take time to answer.
fn start_server(binary: &str) -> bool {
    Command::new(binary)
        .arg("serve")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .is_ok()
}

/// Wait up to `attempts` poll intervals for the server to answer.
pub(super) async fn wait_until_available(client: &OllamaClient, attempts: u32) -> bool {
    let wait_limit = poll_interval() * attempts;

    tokio::time::timeout(wait_limit, poll_until_available(client))
        .await
        .is_ok()
}

/// Send an availability request once per poll interval until one succeeds.
async fn poll_until_available(client: &OllamaClient) {
    while !client.is_available().await {
        tokio::time::sleep(poll_interval()).await;
    }
}

/// Time between two availability checks.
fn poll_interval() -> Duration {
    Duration::from_millis(OLLAMA_POLL_INTERVAL_MILLIS)
}

/// Whether `binary` is an executable on `PATH`, as a login shell sees it.
pub(super) fn binary_on_path(binary: &str) -> bool {
    Command::new("sh")
        .arg("-lc")
        .arg(format!("command -v {binary}"))
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|status| status.success())
        .unwrap_or(false)
}

/// Print one status message to stdout.
fn print_status(message: &str) {
    println!("{message}");
}

/// Print each line of a report to stdout.
fn print_lines(lines: &[String]) {
    for line in lines {
        println!("{line}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants::commands::SETUP_START_POLL_ATTEMPTS;
    use crate::test_support::{MockResponse, spawn_mock_server};

    /// A binary name that no system has on `PATH`.
    const MISSING_BINARY: &str = "openbatrangs-missing-binary";

    /// An address where nothing listens, so every request is refused at once.
    const CLOSED_URL: &str = "http://127.0.0.1:1";

    /// A client for a mock server that answers every request with an empty model list.
    async fn running_server() -> Result<OllamaClient> {
        let base_url =
            spawn_mock_server(|_path| MockResponse::json("200 OK", r#"{"models":[]}"#)).await;

        OllamaClient::new(&base_url)
    }

    #[test]
    fn poll_constants_are_sane() {
        const _: () = {
            assert!(OLLAMA_START_POLL_ATTEMPTS > 0);
            assert!(SETUP_START_POLL_ATTEMPTS >= OLLAMA_START_POLL_ATTEMPTS);
            assert!(OLLAMA_POLL_INTERVAL_MILLIS > 0);
        };
    }

    #[tokio::test]
    async fn running_server_needs_no_start() -> Result<()> {
        let client = running_server().await?;

        ensure_ollama(&client).await
    }

    #[tokio::test]
    async fn unreachable_openai_server_is_reported() -> Result<()> {
        let client = OllamaClient::openai(CLOSED_URL, None, true)?;

        let error = ensure_ollama(&client).await.err();

        assert!(format!("{error:?}").contains("--api-key-file"));
        Ok(())
    }

    #[tokio::test]
    async fn ollama_without_binary_is_reported() -> Result<()> {
        let client = OllamaClient::new(CLOSED_URL)?;

        let error = start_unreachable_server(&client, MISSING_BINARY)
            .await
            .err();

        assert!(format!("{error:?}").contains("openbatrangs setup"));
        Ok(())
    }

    #[tokio::test]
    async fn installed_ollama_is_started() -> Result<()> {
        let client = running_server().await?;

        start_installed_ollama(&client, "true").await
    }

    #[tokio::test]
    async fn server_that_never_answers_is_reported() -> Result<()> {
        let client = OllamaClient::new(CLOSED_URL)?;

        let has_started = start_and_wait(&client, "true", 1).await;

        assert!(!has_started);
        assert!(report_start(&client, has_started).is_err());
        Ok(())
    }

    #[tokio::test]
    async fn binary_that_cannot_start_is_reported() -> Result<()> {
        let client = running_server().await?;

        assert!(!start_and_wait(&client, MISSING_BINARY, 1).await);
        Ok(())
    }

    #[test]
    fn binary_on_path_finds_sh_only() {
        assert!(binary_on_path("sh"));
        assert!(!binary_on_path(MISSING_BINARY));
    }

    #[test]
    fn print_lines_prints_every_line() {
        print_lines(&["first".to_string(), "second".to_string()]);
    }
}
