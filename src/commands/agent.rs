//! One-shot agent dispatch and stdout reporting.

use crate::agent::{self, AgentConfig};
use crate::banner;
use crate::cli::{agent_run_config, model_prefs_from_cli, AgentRunConfig, Cli, ModelPrefs};
use crate::model_select::{calculate_memory_budget, resolve_model, resolve_model_context};
use crate::ollama::OllamaClient;
use anyhow::{bail, Result};

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

/// Trim a task and reject empty input.
fn validate_task(task: &str) -> Result<String> {
    let task = task.trim().to_string();
    if task.is_empty() {
        bail!("empty task");
    }
    Ok(task)
}

/// Run the agent once for a single task using stdout output.
async fn run_agent_task(
    client: &OllamaClient,
    config: &AgentRunConfig,
    model_slot: &mut Option<String>,
    prefs: &ModelPrefs,
    task: &str,
) -> Result<()> {
    let task = validate_task(task)?;

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
        max_ctx: config.max_ctx,
    };

    let mut reporter = agent::StdoutReporter;
    let mut confirmer = agent::StdioConfirmer;
    agent::run_agent(
        &agent_config,
        client,
        &selected.name,
        model_context,
        &task,
        &mut reporter,
        &mut confirmer,
    )
    .await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_task_is_rejected() {
        assert!(validate_task("").is_err());
        assert!(validate_task("   ").is_err());
        assert!(validate_task("\n\t").is_err());
    }

    #[test]
    fn task_is_trimmed() {
        assert_eq!(validate_task("  fix build  ").unwrap(), "fix build");
    }

    #[test]
    fn valid_task_is_accepted() {
        assert_eq!(validate_task("write tests").unwrap(), "write tests");
    }
}
