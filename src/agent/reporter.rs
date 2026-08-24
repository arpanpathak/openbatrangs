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
    fn line(&mut self, msg: String) {
        if msg.is_empty() {
            println!();
        } else {
            println!("{msg}");
        }
    }

    fn chunk(&mut self, msg: &str) {
        print!("{msg}");
        let _ = std::io::stdout().flush();
    }
}
