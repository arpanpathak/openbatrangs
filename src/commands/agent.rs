//! One-shot agent dispatch and stdout reporting.

use super::print_status;
use crate::agent::{self, AgentConfig};
use crate::banner;
use crate::cli::{
    AgentMode, AgentRunConfig, Cli, ModelPrefs, agent_run_config, model_prefs_from_cli,
};
use crate::model_select::{calculate_memory_budget, resolve_model, resolve_model_context};
use crate::ollama::OllamaClient;
use crate::update::{NewRelease, UpdateCheck, daily_check};
use anyhow::{Result, bail};
use tokio::task::JoinHandle;

/// Run the agent for a task, or start the TUI when the task is empty.
///
/// `update_check` is the automatic release check, or `None` when it is
/// turned off. After a task, a newer release is named in one line.
pub(crate) async fn run_agent_or_tui(
    cli: &Cli,
    client: &OllamaClient,
    task: &[String],
    update_check: Option<UpdateCheck>,
) -> Result<()> {
    if task.is_empty() {
        return crate::tui::run(cli, client, update_check).await;
    }

    let release_check = start_release_check(update_check);

    banner::print_banner();
    let prefs = model_prefs_from_cli(cli);
    let config = agent_run_config(cli);

    let outcome = run_agent_task(client, &config, &prefs, &task.join(" ")).await;

    print_new_release(release_check).await;

    outcome
}

/// Run the once-a-day release check in the background while the task runs.
fn start_release_check(update_check: Option<UpdateCheck>) -> JoinHandle<Option<NewRelease>> {
    tokio::spawn(async move {
        match update_check {
            Some(check) => daily_check(&check).await,
            None => None,
        }
    })
}

/// Print one line about a newer release once the background check ends.
/// The request times out after a few seconds, so the wait is short.
async fn print_new_release(release_check: JoinHandle<Option<NewRelease>>) {
    if let Some(release) = release_check.await.ok().flatten() {
        println!("{}", release.one_shot_line());
    }
}

/// Trim a task and reject empty input.
fn validate_task(task: &str) -> Result<String> {
    let task = task.trim().to_string();

    if task.is_empty() {
        bail!("empty task");
    }

    Ok(task)
}

/// Run the agent once for a single task, reporting to stdout.
async fn run_agent_task(
    client: &OllamaClient,
    config: &AgentRunConfig,
    prefs: &ModelPrefs,
    task: &str,
) -> Result<()> {
    let task = validate_task(task)?;
    let model = select_task_model(client, prefs).await?;
    let model_context = resolve_model_context(client, &model).await?;

    println!("{}", run_summary(&model, config, model_context));

    let agent_config = agent_config(config, model_context);
    let mut reporter = agent::StdoutReporter::default();
    let mut confirmer = agent::StdioConfirmer;

    agent::run_agent(
        &agent_config,
        client,
        &model,
        &task,
        None,
        &mut reporter,
        &mut confirmer,
    )
    .await
}

/// Name of the model chosen for the task, with selection progress printed.
async fn select_task_model(client: &OllamaClient, prefs: &ModelPrefs) -> Result<String> {
    let selected = resolve_model(
        client,
        &None,
        prefs,
        calculate_memory_budget(),
        &print_status,
    )
    .await?;

    Ok(selected.name)
}

/// The block printed before the run: model, workspace, step limit and context.
fn run_summary(model: &str, config: &AgentRunConfig, model_context: u64) -> String {
    format!(
        "\n🚀 Agent\n   model:   {model}\n   cwd:     {}\n   steps:   {}\n   context: {model_context}",
        config.cwd.display(),
        config.max_steps,
    )
}

/// Agent settings for one run. Plan mode never writes, so it runs read-only.
fn agent_config(config: &AgentRunConfig, model_context: u64) -> AgentConfig {
    AgentConfig {
        cwd: config.cwd.clone(),
        max_steps: config.max_steps,
        is_read_only: config.is_read_only || config.mode == AgentMode::Plan,
        should_confirm: config.should_confirm,
        use_sandbox: config.use_sandbox,
        show_thinking: true,
        model_context,
        max_ctx: config.max_ctx,
        searxng_url: config.searxng_url.clone(),
        search_key: config.search_key.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{MockResponse, spawn_mock_server, unique_temp_dir};
    use crate::update::fakes::{fake_check, release_server};
    use crate::update::newer_release;
    use clap::Parser;

    /// One installed coding model, as `/api/tags` lists it.
    const TAGS_BODY: &str = r#"{"models":[{"name":"qwen2.5-coder:7b","size":4700000000,"details":{"parameter_size":"7.6B","quantization_level":"Q4_K_M","context_length":32768}}]}"#;

    /// Answer tags and show requests, and fail every chat request.
    fn ollama_route(path: &str) -> MockResponse {
        match path {
            "/api/tags" => MockResponse::json("200 OK", TAGS_BODY),
            "/api/show" => MockResponse::json("200 OK", r#"{"details":{"context_length":32768}}"#),
            _ => MockResponse::text("500 Internal Server Error", "chat is down"),
        }
    }

    #[test]
    fn empty_task_is_rejected() {
        assert!(validate_task("").is_err());
        assert!(validate_task("   ").is_err());
        assert!(validate_task("\n\t").is_err());
    }

    #[test]
    fn task_is_trimmed() -> Result<()> {
        assert_eq!(validate_task("  fix build  ")?, "fix build");
        Ok(())
    }

    #[test]
    fn valid_task_is_accepted() -> Result<()> {
        assert_eq!(validate_task("write tests")?, "write tests");
        Ok(())
    }

    #[test]
    fn plan_mode_runs_read_only() {
        let cli = Cli::parse_from(["openbatrangs"]);
        let mut config = agent_run_config(&cli);

        assert!(!agent_config(&config, 4_096).is_read_only);

        config.mode = AgentMode::Plan;

        assert!(agent_config(&config, 4_096).is_read_only);
    }

    #[test]
    fn run_summary_names_the_model_and_context() {
        let cli = Cli::parse_from(["openbatrangs", "--max-steps", "7"]);
        let summary = run_summary("qwen2.5-coder:7b", &agent_run_config(&cli), 32_768);

        assert!(summary.contains("model:   qwen2.5-coder:7b"));
        assert!(summary.contains("steps:   7"));
        assert!(summary.contains("context: 32768"));
    }

    #[tokio::test]
    async fn task_runs_the_agent_with_the_selected_model() -> Result<()> {
        let base_url = spawn_mock_server(ollama_route).await;
        let client = OllamaClient::new(&base_url)?;
        let workspace = unique_temp_dir("openbatrangs-one-shot-agent");
        let workspace_arg = workspace.display().to_string();
        let cli = Cli::parse_from(["openbatrangs", "--cwd", &workspace_arg, "say", "hello"]);

        let outcome = run_agent_or_tui(&cli, &client, &cli.task, None).await;

        assert!(format!("{outcome:?}").contains("chat is down"));
        Ok(())
    }

    #[tokio::test]
    async fn a_newer_release_is_found_while_the_task_runs() -> Result<()> {
        let check = fake_check(&release_server("v0.3.0").await, "exit 1");

        let release = start_release_check(Some(check)).await?;

        assert_eq!(release, newer_release("v0.3.0", "0.2.2"));
        Ok(())
    }

    #[tokio::test]
    async fn a_turned_off_check_finds_nothing() -> Result<()> {
        assert_eq!(start_release_check(None).await?, None);
        Ok(())
    }

    #[tokio::test]
    async fn a_task_with_a_newer_release_prints_one_line() -> Result<()> {
        let base_url = spawn_mock_server(ollama_route).await;
        let client = OllamaClient::new(&base_url)?;
        let check = fake_check(&release_server("v0.3.0").await, "exit 1");
        let workspace = unique_temp_dir("openbatrangs-one-shot-release");
        let workspace_arg = workspace.display().to_string();
        let cli = Cli::parse_from(["openbatrangs", "--cwd", &workspace_arg, "say", "hello"]);

        let outcome = run_agent_or_tui(&cli, &client, &cli.task, Some(check)).await;

        assert!(outcome.is_err());
        Ok(())
    }
}
