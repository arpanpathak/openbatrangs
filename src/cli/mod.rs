//! Command-line interface definitions and shared agent configuration.
//!
//! The module is split into:
//! - [`args`]: clap argument parsing (`Cli`, `Commands`) and defaults.
//! - [`config`]: runtime configuration types shared by the CLI and TUI.

mod args;
mod config;

pub(crate) use args::{BenchArgs, Cli, Commands, ExperimentalCommand};
pub(crate) use config::{
    agent_run_config, model_prefs_from_cli, AgentMode, AgentRunConfig, ModelPrefs,
};
