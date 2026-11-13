//! # Confirmation prompts for mutating agent tools
//!
//! Mutating tools (`write_file`, `run_command`) are gated behind a
//! [`Confirmer`]. One-shot CLI mode reads `y`/`N` from stdin. The TUI cannot
//! do that while raw mode and the alternate screen are active, so it provides
//! a channel-based confirmer that shows a modal prompt and returns the key the
//! user pressed.
//!
//! ## References
//!
//! - Human-in-the-loop safety for agents: <https://en.wikipedia.org/wiki/Human-in-the-loop>
//! - Rust `tokio::task::spawn_blocking`: <https://docs.rs/tokio/latest/tokio/task/fn.spawn_blocking.html>

use anyhow::{Context, Result};
use std::io::{BufRead, Write};

/// Decides whether a mutating tool call may proceed.
pub trait Confirmer: Send {
    /// Ask the user to confirm `prompt`.
    ///
    /// # Parameters
    ///
    /// - `prompt`: human-readable description of the mutation, e.g.
    ///   `write file 'a.txt'?`.
    ///
    /// # Returns
    ///
    /// `Ok(true)` to allow the tool, `Ok(false)` to abort it.
    async fn confirm(&mut self, prompt: &str) -> Result<bool>;
}

/// Reads `y`/`N` from stdin; used by one-shot CLI mode.
pub struct StdioConfirmer;

impl Confirmer for StdioConfirmer {
    /// Ask on stdin and stdout.
    async fn confirm(&mut self, prompt: &str) -> Result<bool> {
        ask_on_blocking_thread(prompt, ask_on_terminal).await
    }
}

/// Run `ask` with `prompt` on a blocking thread, so the async runtime keeps
/// running while the user types.
async fn ask_on_blocking_thread(prompt: &str, ask: fn(&str) -> bool) -> Result<bool> {
    let prompt = prompt.to_string();

    tokio::task::spawn_blocking(move || ask(&prompt))
        .await
        .context("failed to read confirmation input")
}

/// Print `prompt` to stdout and read the answer from stdin.
fn ask_on_terminal(prompt: &str) -> bool {
    read_answer(prompt, &mut std::io::stdin().lock(), &mut std::io::stdout())
}

/// Write `prompt` to `output`, read one line from `input`, and return true
/// only when that line is `y` or `Y`. A failed read counts as `N`.
fn read_answer(prompt: &str, input: &mut impl BufRead, output: &mut impl Write) -> bool {
    write!(output, "❓ {prompt} [y/N] ").ok();
    output.flush().ok();

    let mut answer = String::new();
    input.read_line(&mut answer).ok();

    answer.trim().eq_ignore_ascii_case("y")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn the_answer_is_read_on_a_blocking_thread() -> anyhow::Result<()> {
        let approves = |prompt: &str| prompt == "write?";

        assert!(ask_on_blocking_thread("write?", approves).await?);
        assert!(!ask_on_blocking_thread("delete?", approves).await?);

        Ok(())
    }

    #[test]
    fn only_y_approves() -> anyhow::Result<()> {
        let mut output = Vec::new();

        assert!(read_answer("write?", &mut "y\n".as_bytes(), &mut output));
        assert!(read_answer("write?", &mut " Y \n".as_bytes(), &mut output));
        assert!(!read_answer("write?", &mut "yes\n".as_bytes(), &mut output));
        assert!(!read_answer("write?", &mut "".as_bytes(), &mut output));

        let printed = String::from_utf8(output)?;
        assert!(printed.starts_with("❓ write? [y/N] "));

        Ok(())
    }
}
