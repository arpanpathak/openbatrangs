//! Command-line interface definitions and shared agent configuration.

use clap::{Parser, Subcommand};
use std::path::PathBuf;

/// Default Ollama server address.
pub(crate) const DEFAULT_OLLAMA_URL: &str = "http://localhost:11434";

/// Default maximum agent iterations.
pub(crate) const DEFAULT_MAX_STEPS: usize = 12;

/// Default minimum acceptable context window for auto model selection.
pub(crate) const DEFAULT_MIN_CONTEXT: usize = 8_192;

/// Command-line interface definition.
#[derive(Parser)]
#[command(
    name = "openbatrangs",
    version,
    about = "🦇 openBatarangs — interactive agentic coding CLI for local models via Ollama"
)]
pub(crate) struct Cli {
    /// Task for the coding agent, e.g. "fix the Rust build errors".
    /// With no task, openBatarangs starts an interactive TUI.
    #[arg(value_name = "TASK")]
    pub(crate) task: Vec<String>,

    /// Ollama server URL.
    #[arg(long, global = true, default_value = DEFAULT_OLLAMA_URL)]
    pub(crate) ollama_url: String,

    /// Model to use; auto-discovered when omitted.
    #[arg(short, long, global = true)]
    pub(crate) model: Option<String>,

    /// Working directory for the agent.
    #[arg(long, global = true, default_value = ".")]
    pub(crate) cwd: PathBuf,

    /// Maximum agent steps.
    #[arg(long, global = true, default_value_t = DEFAULT_MAX_STEPS)]
    pub(crate) max_steps: usize,

    /// Read-only mode: no file writes or shell commands.
    #[arg(long = "read-only", global = true)]
    pub(crate) is_read_only: bool,

    /// Ask before each file write or shell command.
    #[arg(long = "confirm", global = true)]
    pub(crate) should_confirm: bool,

    /// Do not auto-pull a recommended model when none is suitable.
    #[arg(long = "no-auto-pull", global = true)]
    pub(crate) is_auto_pull_disabled: bool,

    /// Minimum context window for auto model selection.
    #[arg(long, global = true, default_value_t = DEFAULT_MIN_CONTEXT)]
    pub(crate) min_context: usize,

    #[command(subcommand)]
    pub(crate) command: Option<Commands>,
}

/// Available subcommands.
#[derive(Subcommand)]
pub(crate) enum Commands {
    /// Run the coding agent (same as passing a task directly).
    Agent {
        /// Task description.
        #[arg(value_name = "TASK")]
        task: Vec<String>,
    },
    /// List locally installed Ollama models with agent scores.
    ListModels,
    /// Check Ollama connectivity and recommend the best model.
    Doctor,
    /// Install/start Ollama and pull a recommended coding model.
    Setup,
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

/// Agent execution mode selected in the TUI.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum AgentMode {
    /// Full agentic loop with tools.
    Agent,
    /// Planning-only: read-only, no writes or shell commands.
    Plan,
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
    /// Agent vs planning mode.
    pub(crate) mode: AgentMode,
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
    }
}
