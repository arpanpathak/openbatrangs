//! Runtime configuration types shared by the one-shot CLI and the TUI.

use super::args::Cli;
use crate::constants::cli::{DEFAULT_SEARXNG_URL, THOR_KEY_FILE, THOR_OPENAI_URL};
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
    /// Upper limit on the context window; `None` uses the model's own length.
    pub(crate) max_ctx: Option<u64>,
    /// Base URL of the SearXNG instance the web tools search.
    pub(crate) searxng_url: String,
    /// Bearer key sent with each search, for a search behind the Thor's agent.
    pub(crate) search_key: Option<String>,
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
        max_ctx: cli.max_ctx,
        searxng_url: searxng_url(cli),
        search_key: search_key(cli),
    }
}

/// The SearXNG endpoint: the flag first, the environment second, the Thor's
/// agent with `--thor`, localhost last.
pub(crate) fn searxng_url(cli: &Cli) -> String {
    let fallback = if cli.thor {
        THOR_OPENAI_URL
    } else {
        DEFAULT_SEARXNG_URL
    };
    cli.searxng_url
        .as_deref()
        .unwrap_or(fallback)
        .trim_end_matches('/')
        .to_string()
}

/// The key a search sends: the Thor's access key when `--thor` searches through
/// the Thor's agent, nothing for a SearXNG named by flag or environment.
pub(crate) fn search_key(cli: &Cli) -> Option<String> {
    if cli.searxng_url.is_some() || !cli.thor {
        return None;
    }
    let path = cli
        .api_key_file
        .clone()
        .or_else(|| std::env::var_os("HOME").map(|home| PathBuf::from(home).join(THOR_KEY_FILE)))?;
    std::fs::read_to_string(path)
        .ok()
        .map(|key| key.trim().to_string())
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
        assert_eq!(config.max_ctx, None);
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

    #[test]
    fn searxng_url_defaults_to_localhost_and_trims_the_trailing_slash() {
        let cli = Cli::parse_from(["openbatrangs"]);
        assert_eq!(agent_run_config(&cli).searxng_url, DEFAULT_SEARXNG_URL);

        let flagged = Cli::parse_from(["openbatrangs", "--searxng-url", "http://thor:8080/"]);
        assert_eq!(agent_run_config(&flagged).searxng_url, "http://thor:8080");
    }

    #[test]
    fn thor_searches_through_the_agent() {
        let cli = Cli::parse_from(["openbatrangs", "--thor"]);
        assert_eq!(agent_run_config(&cli).searxng_url, THOR_OPENAI_URL);

        let named = Cli::parse_from([
            "openbatrangs",
            "--thor",
            "--searxng-url",
            "http://other:8888",
        ]);
        assert_eq!(agent_run_config(&named).searxng_url, "http://other:8888");
        assert_eq!(agent_run_config(&named).search_key, None);
    }
}
