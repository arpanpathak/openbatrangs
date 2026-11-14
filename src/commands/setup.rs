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

use super::{binary_on_path, print_status, start_and_wait};
use crate::constants::commands::{
    OLLAMA_BINARY, OLLAMA_INSTALL_SCRIPT_URL, SETUP_START_POLL_ATTEMPTS,
};
use crate::model_select::calculate_memory_budget;
use crate::models;
use crate::ollama::{OllamaClient, OllamaModel};
use crate::update::installer_command;
use anyhow::{Context, Result, bail};
use std::process::{Command, ExitStatus, Stdio};

/// One-command onboarding: install/start Ollama and pull a recommended model.
pub(crate) async fn setup(client: &OllamaClient) -> Result<()> {
    setup_with_status(client, &print_status).await
}

/// Onboarding with a status callback instead of printing directly to stdout.
///
/// The TUI uses this variant so `/setup` can stream progress into the chat log
/// instead of corrupting the alternate screen with `println!`.
pub(crate) async fn setup_with_status(
    client: &OllamaClient,
    on_status: &(dyn Fn(&str) + Sync),
) -> Result<()> {
    let install_command = installer_command(OLLAMA_INSTALL_SCRIPT_URL);
    ensure_running(client, OLLAMA_BINARY, &install_command, on_status).await?;

    let fallback = models::recommended_fallback_model(calculate_memory_budget());
    pull_if_missing(client, fallback, on_status).await?;

    on_status(&format!(
        "✅ Setup complete. Default coding model: {fallback}"
    ));
    on_status("   Start chatting: openbatrangs");

    Ok(())
}

/// Report a server that already answers, or install and start Ollama.
async fn ensure_running(
    client: &OllamaClient,
    binary: &str,
    install_command: &str,
    on_status: &(dyn Fn(&str) + Sync),
) -> Result<()> {
    if client.is_available().await {
        on_status(&format!("✅ Ollama already running at {}", client.base_url));

        return Ok(());
    }

    install_and_start(client, binary, install_command, on_status).await
}

/// Install Ollama when `binary` is missing, then start it.
async fn install_and_start(
    client: &OllamaClient,
    binary: &str,
    install_command: &str,
    on_status: &(dyn Fn(&str) + Sync),
) -> Result<()> {
    install_if_missing(binary, install_command, on_status)?;

    start_ollama(client, binary, on_status).await
}

/// Run `install_command` when `binary` is not on `PATH`.
fn install_if_missing(
    binary: &str,
    install_command: &str,
    on_status: &(dyn Fn(&str) + Sync),
) -> Result<()> {
    if binary_on_path(binary) {
        return Ok(());
    }

    on_status("⬇️  Ollama not found. Installing with the official script (may ask for sudo)...");

    run_installer(install_command)
}

/// Run `install_command` with `sh -c`, its output shown on the terminal.
fn run_installer(install_command: &str) -> Result<()> {
    let status = Command::new("sh")
        .arg("-c")
        .arg(install_command)
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .status()
        .context("failed to run Ollama installer")?;

    installer_outcome(status)
}

/// `Ok` when the installer exited successfully.
fn installer_outcome(status: ExitStatus) -> Result<()> {
    if !status.success() {
        bail!("Ollama install failed. Install manually from https://ollama.com/download/linux");
    }

    Ok(())
}

/// Start `<binary> serve` and wait for the server to answer.
async fn start_ollama(
    client: &OllamaClient,
    binary: &str,
    on_status: &(dyn Fn(&str) + Sync),
) -> Result<()> {
    on_status("🔄 Starting Ollama...");

    if !start_and_wait(client, binary, SETUP_START_POLL_ATTEMPTS).await {
        bail!("Ollama was installed but did not start. Try `ollama serve` manually.");
    }

    on_status(&format!("✅ Ollama started at {}", client.base_url));

    Ok(())
}

/// Pull `model` unless the server already has it.
async fn pull_if_missing(
    client: &OllamaClient,
    model: &str,
    on_status: &(dyn Fn(&str) + Sync),
) -> Result<()> {
    let installed = client.tags().await?;

    pull_unless_installed(client, &installed, model, on_status).await
}

/// Pull `model` unless it is in `installed`.
async fn pull_unless_installed(
    client: &OllamaClient,
    installed: &[OllamaModel],
    model: &str,
    on_status: &(dyn Fn(&str) + Sync),
) -> Result<()> {
    if installed
        .iter()
        .any(|installed_model| installed_model.name == model)
    {
        return Ok(());
    }

    client.pull(model, on_status).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{
        MockResponse, StatusLog, ignore_status, sample_model, spawn_mock_server,
    };
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// A binary name that no system has on `PATH`.
    const MISSING_BINARY: &str = "openbatrangs-missing-binary";

    /// Answer `/api/pull` with success and `/api/tags` with `tags_body`.
    fn ollama_route(path: &str, tags_body: &'static str) -> MockResponse {
        match path {
            "/api/pull" => MockResponse::json("200 OK", "{\"status\":\"success\"}\n"),
            _ => MockResponse::json("200 OK", tags_body),
        }
    }

    /// A client for a mock Ollama that has no models and pulls successfully.
    async fn empty_ollama() -> Result<OllamaClient> {
        let base_url = spawn_mock_server(|path| ollama_route(path, r#"{"models":[]}"#)).await;

        OllamaClient::new(&base_url)
    }

    /// A mock Ollama whose first answer is an error, so it looks stopped until
    /// it is started.
    async fn stopped_ollama() -> Result<OllamaClient> {
        let requests = AtomicUsize::new(0);
        let base_url = spawn_mock_server(move |_path| {
            let earlier_requests = requests.fetch_add(1, Ordering::SeqCst);
            stopped_then_running(earlier_requests)
        })
        .await;

        OllamaClient::new(&base_url)
    }

    /// An error for the first request and a running server after it.
    fn stopped_then_running(earlier_requests: usize) -> MockResponse {
        if earlier_requests == 0 {
            return MockResponse::text("503 Service Unavailable", "starting");
        }

        MockResponse::json("200 OK", r#"{"models":[]}"#)
    }

    #[tokio::test]
    async fn setup_with_status_pulls_fallback_when_missing() -> Result<()> {
        let client = empty_ollama().await?;
        let log = StatusLog::default();

        setup_with_status(&client, &|message| log.record(message)).await?;

        assert!(log.contains("Setup complete"));
        assert!(log.contains("Pulling"));
        Ok(())
    }

    #[tokio::test]
    async fn setup_skips_the_pull_when_the_fallback_is_installed() -> Result<()> {
        // A pull from this address would fail, so success means no pull ran.
        let client = OllamaClient::new("http://127.0.0.1:1")?;
        let installed = [sample_model("qwen2.5-coder:3b", 32_768, "3B", "Q4_K_M", 1)?];

        pull_unless_installed(&client, &installed, "qwen2.5-coder:3b", &ignore_status).await
    }

    #[tokio::test]
    async fn stopped_ollama_is_started() -> Result<()> {
        let client = stopped_ollama().await?;
        let log = StatusLog::default();

        ensure_running(&client, "true", "false", &|message| log.record(message)).await?;

        assert!(log.contains("Ollama started"));
        Ok(())
    }

    #[tokio::test]
    async fn missing_ollama_is_installed_before_the_start() -> Result<()> {
        let client = stopped_ollama().await?;
        let log = StatusLog::default();

        let outcome = ensure_running(&client, MISSING_BINARY, "true", &|message| {
            log.record(message)
        })
        .await;

        assert!(log.contains("Installing"));
        assert!(format!("{outcome:?}").contains("did not start"));
        Ok(())
    }

    #[test]
    fn failed_installer_is_reported() {
        let outcome = install_if_missing(MISSING_BINARY, "false", &ignore_status);

        assert!(format!("{outcome:?}").contains("Ollama install failed"));
    }

    #[test]
    fn failed_installer_download_is_reported() {
        let install_command = installer_command("file:///nonexistent/install.sh");

        let outcome = install_if_missing(MISSING_BINARY, &install_command, &ignore_status);

        assert!(format!("{outcome:?}").contains("Ollama install failed"));
    }

    #[test]
    fn installed_binary_skips_the_installer() -> Result<()> {
        install_if_missing("sh", "false", &ignore_status)
    }

    #[tokio::test]
    async fn setup_prints_to_stdout() -> Result<()> {
        let client = empty_ollama().await?;

        setup(&client).await
    }
}
