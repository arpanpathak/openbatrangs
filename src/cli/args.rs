//! Clap argument definitions and default values.

use crate::constants::cli::DEFAULT_MIN_CONTEXT;
use clap::builder::FalseyValueParser;
use clap::{Parser, Subcommand};
use std::path::PathBuf;

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

    /// Ollama server URL. Without it, a provider from the config file or the
    /// local Ollama at http://localhost:11434 is used.
    #[arg(long, global = true, value_name = "URL")]
    pub(crate) ollama_url: Option<String>,

    /// Use the models on the Thor over Tailscale: same as
    /// `--openai-url https://arpanpathak.taildb9a39.ts.net/v1 --api-key-file ~/.config/thor-chat/api-key`.
    #[arg(long, global = true)]
    pub(crate) thor: bool,

    /// Use the provider named NAME in `~/.config/openbatrangs/config.toml`.
    #[arg(
        long,
        global = true,
        env = "OPENBATRANGS_PROVIDER",
        value_name = "NAME"
    )]
    pub(crate) provider: Option<String>,

    /// Use an OpenAI-compatible server (llama-server, vLLM, trtllm-serve)
    /// instead of Ollama, e.g. `http://127.0.0.1:8079/v1`.
    #[arg(long, global = true, value_name = "URL")]
    pub(crate) openai_url: Option<String>,

    /// File holding the bearer key for `--openai-url`, or for the provider
    /// in place of its `api_key`.
    #[arg(long, global = true, value_name = "PATH")]
    pub(crate) api_key_file: Option<PathBuf>,

    /// Sampling temperature. Without it, each model family gets the
    /// temperature, top-p and top-k from its model card.
    #[arg(long, global = true, value_name = "T")]
    pub(crate) temperature: Option<f64>,

    /// Turn the model's thinking off (OpenAI-compatible servers only).
    #[arg(long = "no-think", global = true)]
    pub(crate) is_thinking_disabled: bool,

    /// Model to use; auto-discovered when omitted.
    #[arg(short, long, global = true)]
    pub(crate) model: Option<String>,

    /// Working directory for the agent.
    #[arg(long, global = true, default_value = ".")]
    pub(crate) cwd: PathBuf,

    /// Stop the agent after this many steps. There is no limit by default.
    #[arg(long, global = true)]
    pub(crate) max_steps: Option<usize>,

    /// Read-only mode: no file writes or shell commands.
    #[arg(long = "read-only", global = true)]
    pub(crate) is_read_only: bool,

    /// Ask before each file write or shell command.
    #[arg(long = "confirm", global = true)]
    pub(crate) should_confirm: bool,

    /// Run shell commands without the operating-system sandbox, which
    /// otherwise lets them write only inside the workspace. Use it only on a
    /// machine you can throw away, such as a Colab runtime.
    #[arg(long = "no-sandbox", global = true)]
    pub(crate) is_sandbox_disabled: bool,

    /// Do not auto-pull a recommended model when none is suitable.
    #[arg(long = "no-auto-pull", global = true)]
    pub(crate) is_auto_pull_disabled: bool,

    /// Minimum context window for auto model selection.
    #[arg(long, global = true, default_value_t = DEFAULT_MIN_CONTEXT)]
    pub(crate) min_context: usize,

    /// Upper limit on the context window sent to the model. Without it,
    /// Ollama gets at most 32768 tokens and an OpenAI-compatible server gets
    /// the model's whole window (lower uses less memory).
    #[arg(long = "max-ctx", global = true)]
    pub(crate) max_ctx: Option<u64>,

    /// Base URL of the SearXNG instance the web tools search; falls back to
    /// OPENBATRANGS_SEARXNG_URL, then to a localhost instance.
    #[arg(long = "searxng-url", global = true, env = "OPENBATRANGS_SEARXNG_URL")]
    pub(crate) searxng_url: Option<String>,

    /// Write the run's model, tool definitions, full conversation, outcome
    /// and step count to PATH as JSON (one-shot runs only).
    #[arg(long = "transcript-out", global = true, value_name = "PATH")]
    pub(crate) transcript_out: Option<PathBuf>,

    /// Do not check for a newer release when openBatarangs starts.
    /// `OPENBATRANGS_NO_UPDATE_CHECK=1` does the same; `openbatrangs update`
    /// still checks.
    #[arg(
        long = "no-update-check",
        global = true,
        env = "OPENBATRANGS_NO_UPDATE_CHECK",
        value_parser = FalseyValueParser::new()
    )]
    pub(crate) is_update_check_disabled: bool,

    /// Optional subcommand (agent, list-models, doctor, setup, pull, update,
    /// provider).
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
    /// Pull a model tag from the Ollama registry.
    Pull {
        /// Model tag, e.g. `samantha-mistral:7b`.
        #[arg(value_name = "MODEL")]
        model: String,
    },
    /// Check for a newer release now and install it when there is one.
    Update,
    /// List, add or remove the providers in the config file.
    Provider {
        /// What to do with the providers.
        #[command(subcommand)]
        action: ProviderAction,
    },
}

/// What `openbatrangs provider` does.
#[derive(Subcommand)]
pub(crate) enum ProviderAction {
    /// List the providers and whether each has a key.
    List,
    /// Add a provider, or change its URL and key. The key is read from the
    /// terminal without echo, or from standard input when it is piped.
    Add {
        /// Provider name: anthropic, deepseek, openai, or a name of your own.
        #[arg(value_name = "NAME")]
        name: String,
        /// Server address; needed for a name of your own.
        #[arg(long, value_name = "URL")]
        url: Option<String>,
    },
    /// Remove a provider.
    Remove {
        /// Provider name.
        #[arg(value_name = "NAME")]
        name: String,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants::cli::DEFAULT_MIN_CONTEXT;
    use clap::CommandFactory;

    #[test]
    fn parses_defaults_without_arguments() {
        let cli = Cli::parse_from(["openbatrangs"]);

        assert!(cli.task.is_empty());
        assert_eq!(cli.ollama_url, None);
        assert_eq!(cli.provider, None);
        assert_eq!(cli.max_steps, None);
        assert_eq!(cli.min_context, DEFAULT_MIN_CONTEXT);
        assert_eq!(cli.max_ctx, None);
        assert_eq!(cli.cwd, PathBuf::from("."));
        assert!(!cli.is_read_only);
        assert!(!cli.should_confirm);
        assert!(!cli.is_auto_pull_disabled);
        assert!(cli.transcript_out.is_none());
        assert!(cli.command.is_none());
    }

    #[test]
    fn parses_positional_task() {
        let cli = Cli::parse_from(["openbatrangs", "fix", "the", "bug"]);

        assert_eq!(cli.task, vec!["fix", "the", "bug"]);
    }

    #[test]
    fn parses_global_options_before_subcommand() {
        let cli = Cli::parse_from([
            "openbatrangs",
            "--model",
            "qwen2.5-coder:7b",
            "--max-steps",
            "5",
            "--read-only",
            "list-models",
        ]);

        assert_eq!(cli.model.as_deref(), Some("qwen2.5-coder:7b"));
        assert_eq!(cli.max_steps, Some(5));
        assert!(cli.is_read_only);
        assert!(matches!(cli.command, Some(Commands::ListModels)));
    }

    #[test]
    fn parses_agent_subcommand_with_task() {
        let cli = Cli::parse_from(["openbatrangs", "agent", "write", "tests"]);

        assert!(matches!(
            cli.command,
            Some(Commands::Agent { task }) if task == ["write", "tests"]
        ));
    }

    #[test]
    fn parses_the_transcript_path() {
        let cli = Cli::parse_from(["openbatrangs", "--transcript-out", "run.json", "say", "hi"]);

        assert_eq!(cli.transcript_out, Some(PathBuf::from("run.json")));
        assert_eq!(cli.task, vec!["say", "hi"]);
    }

    #[test]
    fn parses_setup_subcommand() {
        let cli = Cli::parse_from(["openbatrangs", "setup"]);

        assert!(matches!(cli.command, Some(Commands::Setup)));
    }

    #[test]
    fn parses_doctor_subcommand() {
        let cli = Cli::parse_from(["openbatrangs", "doctor"]);

        assert!(matches!(cli.command, Some(Commands::Doctor)));
    }

    #[test]
    fn parses_update_subcommand_and_the_opt_out_flag() {
        let cli = Cli::parse_from(["openbatrangs", "update", "--no-update-check"]);

        assert!(matches!(cli.command, Some(Commands::Update)));
        assert!(cli.is_update_check_disabled);
    }

    #[test]
    fn parses_the_provider_flag_and_subcommands() {
        let flagged = Cli::parse_from(["openbatrangs", "--provider", "deepseek"]);
        let add = Cli::parse_from([
            "openbatrangs",
            "provider",
            "add",
            "box",
            "--url",
            "http://box/v1",
        ]);
        let remove = Cli::parse_from(["openbatrangs", "provider", "remove", "box"]);
        let list = Cli::parse_from(["openbatrangs", "provider", "list"]);

        assert_eq!(flagged.provider.as_deref(), Some("deepseek"));
        assert!(matches!(
            add.command,
            Some(Commands::Provider { action: ProviderAction::Add { name, url } })
                if name == "box" && url.as_deref() == Some("http://box/v1")
        ));
        assert!(matches!(
            remove.command,
            Some(Commands::Provider { action: ProviderAction::Remove { name } }) if name == "box"
        ));
        assert!(matches!(
            list.command,
            Some(Commands::Provider {
                action: ProviderAction::List
            })
        ));
    }

    #[test]
    fn provider_reads_its_environment_variable() {
        let command = Cli::command();
        let provider = command
            .get_arguments()
            .find(|argument| argument.get_id() == "provider");

        assert_eq!(
            provider.and_then(|argument| argument.get_env()),
            Some(std::ffi::OsStr::new("OPENBATRANGS_PROVIDER"))
        );
    }

    #[test]
    fn parses_pull_subcommand_with_model() {
        let cli = Cli::parse_from(["openbatrangs", "pull", "samantha-mistral:7b"]);

        assert!(matches!(
            cli.command,
            Some(Commands::Pull { model }) if model == "samantha-mistral:7b"
        ));
    }
}
