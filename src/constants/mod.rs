//! Central home for all named constants used across openBatarangs.
//!
//! Keeping tuning values, limits, timeouts, ANSI codes, prompt text, and other
//! literals in one modular directory avoids magic numbers scattered across
//! implementation files and makes maintenance easier.

pub mod agent;
pub mod ansi;
pub mod banner;
pub mod cli;
pub mod commands;
pub mod models;
pub mod ollama;
pub mod perf;
pub mod tools;
pub mod tui;
