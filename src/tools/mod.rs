//! Safe filesystem and shell tools available to the coding agent.
//!
//! All file tools operate on paths relative to the workspace root. Absolute
//! paths and `..` traversal are rejected so the model cannot escape the project.
//!
//! The module is split into:
//! - [`path`]: path safety (`resolve_in_root`).
//! - [`files`]: listing, reading, writing, grepping files.
//! - [`command`]: running shell commands.
//! - [`text`]: output truncation helpers.

mod command;
mod files;
mod path;
mod text;

pub(crate) use command::run_command;
pub(crate) use files::{grep_files, list_files, read_file, write_file};
