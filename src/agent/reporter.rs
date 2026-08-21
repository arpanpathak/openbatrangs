//! Output reporter abstraction for agent progress.

use std::io::Write;

/// Receives agent output. `line` is a complete line; `chunk` is streaming text
/// that should be appended to the current live line.
pub trait Reporter: Send {
    fn line(&mut self, msg: String);
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
