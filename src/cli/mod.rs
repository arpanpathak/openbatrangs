//! # Command-line interface definitions and shared agent configuration
//!
//! The module is split into:
//! - [`args`]: clap argument parsing (`Cli`, `Commands`) and defaults.
//! - [`config`]: runtime configuration types shared by the CLI and TUI.
//! - [`providers`]: the providers in `config.toml`.
//! - [`provider_edit`]: changes to `config.toml` that keep its comments.
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
mod provider_edit;
mod providers;

pub(crate) use args::{Cli, Commands, ProviderAction};
pub(crate) use config::{
    AgentMode, AgentRunConfig, ModelPrefs, agent_run_config, home_directory, model_prefs_from_cli,
    names_server, search_key, searxng_url, thor_key_file,
};
pub(crate) use provider_edit::{add_provider, check_new_provider, remove_provider, save_settings};
pub(crate) use providers::{ConfigFile, chosen_provider, permission_warning, user_config_path};
