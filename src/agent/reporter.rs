//! # Output reporter abstraction
//!
//! The agent loop produces two kinds of output: complete lines (step headers,
//! tool results, errors) and streaming chunks (incremental JSON field text).
//! Instead of hard-coding stdout, the loop accepts a [`Reporter`] so the same
//! orchestration can drive both the one-shot CLI and the TUI.
//!
//! ## References
//!
//! - Rust traits as abstraction boundaries: <https://doc.rust-lang.org/book/ch10-02-traits.html>
//! - SOLID interface segregation principle: <https://en.wikipedia.org/wiki/Interface_segregation_principle>

use std::io::Write;

/// Receives agent output.
///
/// `line` is a complete line; `chunk` is streaming text that should be
/// appended to the current live line.
pub trait Reporter: Send {
    /// Emit one complete line of output.
    fn line(&mut self, msg: String);

    /// Emit a streaming chunk that continues the current line.
    fn chunk(&mut self, msg: &str);
}

/// Prints agent output directly to stdout (used by one-shot mode).
pub struct StdoutReporter;

impl Reporter for StdoutReporter {
    /// Print `msg` followed by a newline.
    fn line(&mut self, msg: String) {
        println!("{msg}");
    }

    /// Print `msg` without a newline and flush, so streamed text shows at once.
    fn chunk(&mut self, msg: &str) {
        print!("{msg}");

        // A failed flush only delays the text until the next newline.
        std::io::stdout().flush().ok();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stdout_reporter_prints_lines_and_chunks() -> anyhow::Result<()> {
        let mut reporter = StdoutReporter;

        reporter.line(String::new());
        reporter.line("a line".to_string());
        reporter.chunk("a chunk");

        Ok(())
    }
}
