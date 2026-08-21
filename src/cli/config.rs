//! Runtime configuration types shared by the one-shot CLI and the TUI.

use super::args::Cli;
use std::path::PathBuf;

/// Agent execution mode selected in the TUI.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum AgentMode {
    /// Full agentic loop with tools.
    Agent,
    /// Planning-only: read-only, no writes or shell commands.
    Plan,
    /// Plain chat: no tools, just conversation and code.
    Chat,
}

/// Preferences used for automatic model selection.
#[derive(Clone)]
pub(crate) struct ModelPrefs {
    /// Explicit model tag; `None` means auto-select.
    pub(crate) model: Option<String>,
    /// If `true`, never auto-pull a model.
    pub(crate) is_auto_pull_disabled: bool,
    /// Minimum acceptable context window.
    pub(crate) min_context: u64,
}

/// Runtime settings shared by the one-shot agent and the TUI worker.
#[derive(Clone)]
pub(crate) struct AgentRunConfig {
    /// Workspace directory for the agent.
    pub(crate) cwd: PathBuf,
    /// Maximum agent iterations.
    pub(crate) max_steps: usize,
    /// Disable file writes and shell commands.
    pub(crate) is_read_only: bool,
    /// Ask before mutating actions.
    pub(crate) should_confirm: bool,
    /// Agent vs planning vs chat mode.
    pub(crate) mode: AgentMode,
    /// Stream the model's internal reasoning in agent mode.
    pub(crate) show_thinking: bool,
}

/// Build model-selection preferences from parsed CLI arguments.
pub(crate) fn model_prefs_from_cli(cli: &Cli) -> ModelPrefs {
    ModelPrefs {
        model: cli.model.clone(),
        is_auto_pull_disabled: cli.is_auto_pull_disabled,
        min_context: cli.min_context as u64,
    }
}

/// Build the shared agent runtime configuration from parsed CLI arguments.
pub(crate) fn agent_run_config(cli: &Cli) -> AgentRunConfig {
    AgentRunConfig {
        cwd: cli.cwd.clone(),
        max_steps: cli.max_steps,
        is_read_only: cli.is_read_only,
        should_confirm: cli.should_confirm,
        mode: AgentMode::Agent,
        show_thinking: true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    fn sample_cli() -> Cli {
        Cli::parse_from([
            "openbatrangs",
            "--model",
            "qwen2.5-coder:3b",
            "--max-steps",
            "4",
            "--read-only",
            "--confirm",
            "--no-auto-pull",
            "--min-context",
            "4096",
        ])
    }

    #[test]
    fn model_prefs_reflect_cli_options() {
        let prefs = model_prefs_from_cli(&sample_cli());
        assert_eq!(prefs.model.as_deref(), Some("qwen2.5-coder:3b"));
        assert!(prefs.is_auto_pull_disabled);
        assert_eq!(prefs.min_context, 4_096);
    }

    #[test]
    fn agent_run_config_uses_agent_mode_and_thinking() {
        let config = agent_run_config(&sample_cli());
        assert_eq!(config.mode, AgentMode::Agent);
        assert!(config.show_thinking);
        assert!(config.is_read_only);
        assert!(config.should_confirm);
        assert_eq!(config.max_steps, 4);
    }

    #[test]
    fn agent_mode_variants_are_distinct() {
        assert_ne!(AgentMode::Agent, AgentMode::Plan);
        assert_ne!(AgentMode::Plan, AgentMode::Chat);
        assert_ne!(AgentMode::Chat, AgentMode::Agent);
    }

    #[test]
    fn model_prefs_default_when_model_omitted() {
        let cli = Cli::parse_from(["openbatrangs"]);
        let prefs = model_prefs_from_cli(&cli);
        assert!(prefs.model.is_none());
        assert!(!prefs.is_auto_pull_disabled);
        assert_eq!(prefs.min_context, 8_192);
    }
}
