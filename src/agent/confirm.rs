//! Confirmation prompts for mutating agent tools.
//!
//! One-shot CLI mode confirms on stdin/stdout. The TUI cannot do that while raw
//! mode and the alternate screen are active, so it provides a channel-based
//! confirmer that asks the UI to display a modal prompt and waits for the user's
//! keypress.

use anyhow::{Context, Result};
use std::io::Write;

/// Decides whether a mutating tool call may proceed.
pub trait Confirmer: Send {
    /// Ask the user to confirm `prompt`.
    ///
    /// # Returns
    /// `Ok(true)` to allow the tool, `Ok(false)` to abort it.
    async fn confirm(&mut self, prompt: &str) -> Result<bool>;
}

/// Reads `y`/`N` from stdin; used by one-shot CLI mode.
pub struct StdioConfirmer;

impl Confirmer for StdioConfirmer {
    async fn confirm(&mut self, prompt: &str) -> Result<bool> {
        let prompt = prompt.to_string();
        tokio::task::spawn_blocking(move || {
            print!("❓ {prompt} [y/N] ");
            std::io::stdout().flush().ok();
            let mut input = String::new();
            std::io::stdin().read_line(&mut input).ok();
            Ok(input.trim().eq_ignore_ascii_case("y"))
        })
        .await
        .context("failed to read confirmation input")?
    }
}
