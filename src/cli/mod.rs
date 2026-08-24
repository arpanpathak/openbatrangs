//! # Command-line interface definitions and shared agent configuration
//!
//! The module is split into:
//! - [`args`]: clap argument parsing (`Cli`, `Commands`) and defaults.
//! - [`config`]: runtime configuration types shared by the CLI and TUI.
//!
//! Keeping CLI parsing and runtime config separate means the one-shot CLI and
//! the interactive TUI consume the same [`AgentRunConfig`] shape, so behavior
//! cannot drift between the two entry points.
//!
//! ## References
//!
//! - Clap derive API: <https://docs.rs/clap/latest/clap/>
//! - Configuration-over-convention: <https://en.wikipedia.org/wiki/Convention_over_configuration>

mod args;
mod config;

pub(crate) use args::{Cli, Commands};
pub(crate) use config::{
    agent_run_config, model_prefs_from_cli, AgentMode, AgentRunConfig, ModelPrefs,
};
