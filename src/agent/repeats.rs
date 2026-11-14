//! Detects a model that repeats the same tool call with the same result.
//!
//! Both agent loops pass every tool result through a [`RepeatGuard`]. From
//! the third identical call in a row, the result the model sees carries a
//! note telling it to read the file again or change its approach. The fifth
//! identical call stops the run, so a stuck model does not spend every
//! remaining step.

use crate::constants::agent::{REPEATED_CALL_LIMIT, REPEATED_CALL_WARNING};
use anyhow::{Result, ensure};

/// One tool call and the result it returned.
#[derive(PartialEq)]
struct CallRecord {
    /// Tool name.
    name: String,
    /// Tool arguments as JSON text.
    arguments: String,
    /// Result text the tool returned.
    result: String,
}

/// Counts identical tool calls in a row.
#[derive(Default)]
pub(super) struct RepeatGuard {
    /// The most recent tool call.
    last_call: Option<CallRecord>,
    /// Number of identical calls in a row, ending with `last_call`.
    repeats: usize,
    /// Whether some call reached the limit; stays set once it does.
    reached_limit: bool,
}

impl RepeatGuard {
    /// Record a tool call and return the result text the model should see:
    /// `result`, with a note from the warning count on.
    pub(super) fn model_result(&mut self, name: &str, arguments: &str, result: &str) -> String {
        let call = CallRecord {
            name: name.to_string(),
            arguments: arguments.to_string(),
            result: result.to_string(),
        };

        self.count(call);

        with_repeat_note(result, self.repeats)
    }

    /// `next` while no call reached the limit; an error that stops the run
    /// once one did.
    pub(super) fn allow<T>(&self, next: T) -> Result<T> {
        ensure!(
            !self.reached_limit,
            "stopped: the model repeated the same tool call {REPEATED_CALL_LIMIT} times"
        );

        Ok(next)
    }

    /// Count `call` as a repeat of the last call, or start a new count.
    fn count(&mut self, call: CallRecord) {
        let is_repeat = self.last_call.as_ref() == Some(&call);

        self.repeats = match is_repeat {
            true => self.repeats + 1,
            false => 1,
        };
        self.reached_limit |= self.repeats >= REPEATED_CALL_LIMIT;
        self.last_call = Some(call);
    }
}

/// `result` with a note about the repeats once `repeats` reaches the warning
/// count.
fn with_repeat_note(result: &str, repeats: usize) -> String {
    if repeats < REPEATED_CALL_WARNING {
        return result.to_string();
    }

    format!(
        "{result}\n\nNote: this same tool call has returned the same result {repeats} times \
         in a row. Read the file again or change your approach instead of repeating it."
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The result text the model sees for `count` identical calls in a row.
    fn results_of_identical_calls(guard: &mut RepeatGuard, count: usize) -> Vec<String> {
        (0..count)
            .map(|_call| guard.model_result("edit_file", r#"{"path":"a.rs"}"#, "no match"))
            .collect()
    }

    #[test]
    fn the_third_identical_call_carries_a_note() {
        let mut guard = RepeatGuard::default();

        let results = results_of_identical_calls(&mut guard, 3);

        assert_eq!(results[0], "no match");
        assert_eq!(results[1], "no match");
        assert!(results[2].starts_with("no match\n\nNote:"));
        assert!(results[2].contains("same result 3 times"));
        assert!(results[2].contains("Read the file again"));
        assert!(guard.allow(()).is_ok());
    }

    #[test]
    fn the_fifth_identical_call_stops_the_run() {
        let mut guard = RepeatGuard::default();

        results_of_identical_calls(&mut guard, 5);

        let error = guard.allow(()).err().map(|error| error.to_string());
        assert_eq!(
            error.as_deref(),
            Some("stopped: the model repeated the same tool call 5 times")
        );
    }

    #[test]
    fn a_different_call_resets_the_count() {
        let mut guard = RepeatGuard::default();
        results_of_identical_calls(&mut guard, 4);

        let other = guard.model_result("read_file", r#"{"path":"a.rs"}"#, "fn main() {}");
        let results = results_of_identical_calls(&mut guard, 2);

        assert_eq!(other, "fn main() {}");
        assert_eq!(results, ["no match", "no match"]);
        assert!(guard.allow(()).is_ok());
    }

    #[test]
    fn a_different_result_resets_the_count() {
        let mut guard = RepeatGuard::default();
        results_of_identical_calls(&mut guard, 2);

        let changed = guard.model_result("edit_file", r#"{"path":"a.rs"}"#, "edited");
        let results = results_of_identical_calls(&mut guard, 2);

        assert_eq!(changed, "edited");
        assert_eq!(results, ["no match", "no match"]);
    }

    #[test]
    fn the_stop_stays_after_a_later_different_call() {
        let mut guard = RepeatGuard::default();
        results_of_identical_calls(&mut guard, 5);

        guard.model_result("read_file", "{}", "text");

        assert!(guard.allow(()).is_err());
    }
}
