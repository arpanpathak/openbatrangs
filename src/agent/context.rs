//! Rolling context for small local-model context windows.
//!
//! Ollama does not expose a tokenizer, so prompts are budgeted with a
//! conservative characters-per-token estimate. When history exceeds the safe
//! per-device window, older turns are rolled into a short structured note while
//! the system message, original task, and newest tool exchanges stay verbatim.

use crate::constants::agent::{
    CHARS_PER_TOKEN, COMPACT_NOTE_MAX_CHARS, CONTEXT_RESERVE_TOKENS, KEEP_RECENT_MESSAGES,
    MAX_OUTPUT_TOKENS, MAX_RECENT_MESSAGE_CHARS, MIN_OUTPUT_TOKENS,
};
use crate::ollama::{ChatMessage, Role};
use serde_json::Value;
use std::borrow::Cow;

/// Marker at the beginning of the compacted history note.
const NOTE_HEADER: &str = "--- Prior context ---";
/// Marker at the end of the compacted history note.
const NOTE_FOOTER: &str = "--- End prior context ---";
/// Maximum number of history entries included in one note.
const NOTE_LINE_LIMIT: usize = 60;

/// Shrink `messages` in place so the estimated prompt fits `num_ctx`.
pub(crate) fn compact_context(messages: &mut Vec<ChatMessage>, num_ctx: u64) {
    let budget = max_prompt_chars(num_ctx);

    if messages.len() == 2 {
        let task = messages.pop().expect("length checked");
        messages.push(truncate_message(task, budget));
    } else if messages.len() > 2 && total_chars(messages) > budget {
        *messages = rolled_context(std::mem::take(messages), budget);
    }
}

/// Largest `num_predict` that leaves room for the prompt inside `num_ctx`.
pub(crate) fn max_output_tokens(num_ctx: u64, messages: &[ChatMessage]) -> u64 {
    let available = num_ctx
        .saturating_sub(estimate_prompt_tokens(messages))
        .saturating_sub(CONTEXT_RESERVE_TOKENS);
    available.clamp(MIN_OUTPUT_TOKENS, MAX_OUTPUT_TOKENS)
}

/// Estimated token count of a message list.
pub(crate) fn estimate_prompt_tokens(messages: &[ChatMessage]) -> u64 {
    messages
        .iter()
        .map(|message| estimate_tokens(&message.content))
        .sum()
}

/// Rebuild an owned message list inside the budget, moving instead of cloning.
fn rolled_context(mut messages: Vec<ChatMessage>, budget: usize) -> Vec<ChatMessage> {
    let tail_len = recent_tail_len(messages.len());
    let tail_start = messages.len() - tail_len;
    let middle_end = tail_start - 2;
    let fixed_chars = messages[0].content.len() + messages[1].content.len();

    let system = messages.remove(0);
    let mut task = messages.remove(0);

    let note = compact_note(&messages[..middle_end], note_budget(fixed_chars, budget));
    append_note(&mut task, &note);

    let tail = messages.split_off(middle_end);
    let mut compacted = vec![system, task];
    append_tail(&mut compacted, tail, budget);
    shrink_task_if_needed(&mut compacted, budget);
    compacted
}

/// Preserve the original task and append the compacted older history to it.
fn append_note(task: &mut ChatMessage, note: &str) {
    if note.is_empty() {
        return;
    }

    task.content.push_str("\n\n");
    task.content.push_str(NOTE_HEADER);
    task.content.push('\n');
    task.content.push_str(note);
    task.content.push('\n');
    task.content.push_str(NOTE_FOOTER);
}

/// Budget reserved for the compacted note after the fixed messages.
fn note_budget(fixed_chars: usize, budget: usize) -> usize {
    budget
        .saturating_sub(fixed_chars)
        .clamp(256, COMPACT_NOTE_MAX_CHARS)
}

/// Number of recent messages to keep verbatim, rounded down to an even number.
fn recent_tail_len(message_count: usize) -> usize {
    let tail = (message_count - 2).min(KEEP_RECENT_MESSAGES);
    tail & !1
}

/// Move the recent tail into place, distributing the remaining budget evenly.
fn append_tail(compacted: &mut Vec<ChatMessage>, tail: Vec<ChatMessage>, budget: usize) {
    if tail.is_empty() {
        return;
    }

    let remaining = budget.saturating_sub(total_chars(compacted));
    let per_message = (remaining / tail.len()).clamp(256, MAX_RECENT_MESSAGE_CHARS);
    compacted.extend(
        tail.into_iter()
            .map(|message| truncate_message(message, per_message)),
    );
}

/// Truncate the task only when system + task already exceed the budget.
fn shrink_task_if_needed(compacted: &mut Vec<ChatMessage>, budget: usize) {
    if compacted.len() < 2 || total_chars(compacted) <= budget {
        return;
    }

    let system_len = compacted[0].content.len();
    let task = compacted.remove(1);
    compacted.insert(
        1,
        truncate_message(task, budget.saturating_sub(system_len).max(256)),
    );
}

/// Number of prompt characters that leave room for [`MIN_OUTPUT_TOKENS`] output.
fn max_prompt_chars(num_ctx: u64) -> usize {
    let budget_tokens = num_ctx
        .saturating_sub(MIN_OUTPUT_TOKENS)
        .saturating_sub(CONTEXT_RESERVE_TOKENS)
        .max(256);
    (budget_tokens as f64 * CHARS_PER_TOKEN) as usize
}

/// One-line summaries of older history.
fn compact_note(messages: &[ChatMessage], max_chars: usize) -> String {
    let mut note = messages
        .iter()
        .take(NOTE_LINE_LIMIT)
        .filter_map(|message| {
            let line = message_line(message);
            (!line.is_empty()).then(|| format!("- {line}\n"))
        })
        .collect::<String>();

    if messages.len() > NOTE_LINE_LIMIT {
        note.push_str("- ...\n");
    }

    truncate_chars(&note, max_chars)
}

/// One terse line for a compacted message.
fn message_line(message: &ChatMessage) -> String {
    match message.role {
        Role::User if message.content.starts_with("Tool result:") => {
            let result = message.content.trim_start_matches("Tool result:").trim();
            format!("tool result: {}", head(result, 200))
        }
        Role::User => format!("user: {}", head(&message.content, 200)),
        Role::Assistant => assistant_line(&message.content),
        Role::System | Role::Tool => String::new(),
    }
}

/// Summarize an assistant JSON message without repeating file contents.
fn assistant_line(content: &str) -> String {
    let Ok(value) = serde_json::from_str::<Value>(content) else {
        return format!("assistant: {}", head(content, 160));
    };

    if let Some(answer) = value.get("answer").and_then(Value::as_str) {
        return format!("answer: {}", head(answer, 240));
    }

    let Some(tool) = value.get("tool") else {
        return format!("assistant: {}", head(content, 160));
    };
    let Some(name) = tool.get("name").and_then(Value::as_str) else {
        return format!("assistant: {}", head(content, 160));
    };
    let Some(args) = tool.get("arguments") else {
        return format!("assistant: {}", head(content, 160));
    };

    match name {
        "list_files" => format!("list_files(path={})", arg(args, "path")),
        "read_file" => format!(
            "read_file(path={}, max_chars={})",
            arg(args, "path"),
            arg(args, "max_chars")
        ),
        "grep_files" => format!(
            "grep_files(pattern={:?}, path={}, max_results={})",
            arg(args, "pattern"),
            arg(args, "path"),
            arg(args, "max_results")
        ),
        // Written content lives on disk; repeating it would waste the window.
        "write_file" => format!("write_file(path={})", arg(args, "path")),
        "run_command" => format!("run_command({})", arg(args, "command")),
        "finish" => format!(
            "finish(summary={})",
            head(arg(args, "summary").as_ref(), 120)
        ),
        other => format!("tool({other}): {}", head(content, 160)),
    }
}

/// Read a string or displayable argument from a tool-call JSON object.
fn arg<'a>(args: &'a Value, key: &str) -> Cow<'a, str> {
    match args.get(key) {
        Some(Value::String(value)) => Cow::Borrowed(value),
        Some(value) => Cow::Owned(value.to_string()),
        None => Cow::Borrowed(""),
    }
}

/// First line of text, truncated.
fn head(text: &str, max_chars: usize) -> String {
    let line = text.lines().next().unwrap_or("").trim();
    truncate_chars(line, max_chars)
}

/// Truncate a string and append an omission marker.
fn truncate_chars(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        return text.to_string();
    }

    let mut truncated = text.chars().take(max_chars).collect::<String>();
    truncated.push_str(&format!(
        "\n... (paged, {} chars omitted)",
        text.chars().count().saturating_sub(max_chars)
    ));
    truncated
}

/// Truncate a chat message while preserving its role.
fn truncate_message(mut message: ChatMessage, max_chars: usize) -> ChatMessage {
    message.content = truncate_chars(&message.content, max_chars);
    message
}

/// Estimated token count for one string.
fn estimate_tokens(text: &str) -> u64 {
    ((text.chars().count() as f64) / CHARS_PER_TOKEN).ceil() as u64
}

/// Combined character length of all message content.
fn total_chars(messages: &[ChatMessage]) -> usize {
    messages.iter().map(|message| message.content.len()).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn message(role: Role, content: impl Into<String>) -> ChatMessage {
        ChatMessage {
            role,
            content: content.into(),
        }
    }

    #[test]
    fn rolls_old_tool_output_into_note() {
        let mut messages = vec![
            message(Role::System, "system".repeat(100)),
            message(Role::User, "Fix the regression in src/lib.rs"),
        ];
        for index in 0..20 {
            messages.push(message(
                Role::Assistant,
                format!(
                    r#"{{"tool":{{"name":"read_file","arguments":{{"path":"src/file_{index}.rs","max_chars":8000}}}}}}"#
                ),
            ));
            messages.push(message(
                Role::User,
                format!("Tool result:\n{}", "x".repeat(6_000)),
            ));
        }
        let original_len = messages.len();

        compact_context(&mut messages, 4096);

        assert!(messages.len() < original_len);
        assert_eq!(messages[0].role, Role::System);
        assert_eq!(messages[1].role, Role::User);
        assert!(messages[1].content.contains(NOTE_HEADER));
        assert_eq!(messages.last().unwrap().role, Role::User);
        assert!(
            total_chars(&messages) <= max_prompt_chars(4096) + 512,
            "context pager should stay near the token budget"
        );
    }

    #[test]
    fn leaves_short_history_untouched() {
        let mut messages = vec![
            message(Role::System, "system"),
            message(Role::User, "task"),
            message(Role::Assistant, r#"{"answer":"done"}"#),
        ];
        compact_context(&mut messages, 4096);

        assert_eq!(messages.len(), 3);
        assert_eq!(messages[1].content, "task");
        assert_eq!(messages[2].content, r#"{"answer":"done"}"#);
    }

    #[test]
    fn note_does_not_repeat_written_file_contents() {
        let messages = vec![message(
            Role::Assistant,
            serde_json::json!({
                "tool": {
                    "name": "write_file",
                    "arguments": {
                        "path": "src/lib.rs",
                        "content": "pub fn real_code() {}\n".repeat(2_000),
                    }
                }
            })
            .to_string(),
        )];

        let note = compact_note(&messages, 1_000);

        assert!(note.contains("write_file(path=src/lib.rs)"));
        assert!(!note.contains("real_code"));
    }

    #[test]
    fn output_tokens_are_bounded() {
        let messages = vec![
            message(Role::System, "s".repeat(8_000)),
            message(Role::User, "t".repeat(8_000)),
        ];

        let output = max_output_tokens(4096, &messages);

        assert!(output >= MIN_OUTPUT_TOKENS);
        assert!(output <= MAX_OUTPUT_TOKENS);
        assert!(output + estimate_prompt_tokens(&messages) + CONTEXT_RESERVE_TOKENS >= 4096);
    }

    #[test]
    fn truncates_single_user_message_when_only_two_messages() {
        let mut messages = vec![
            message(Role::System, "system"),
            message(Role::User, "x".repeat(20_000)),
        ];

        compact_context(&mut messages, 2048);

        assert_eq!(messages.len(), 2);
        assert!(messages[1].content.len() < 20_000);
        assert!(messages[1].content.contains("paged"));
    }

    #[test]
    fn empty_note_is_not_appended() {
        let mut task = message(Role::User, "task");

        append_note(&mut task, "");

        assert_eq!(task.content, "task");
    }

    #[test]
    fn empty_tail_is_a_noop() {
        let mut compacted = vec![message(Role::System, "system")];
        let original_len = compacted.len();

        append_tail(&mut compacted, Vec::new(), 4_096);

        assert_eq!(compacted.len(), original_len);
    }

    #[test]
    fn shrinks_task_when_fixed_messages_exceed_budget() {
        let mut messages = vec![
            message(Role::System, "s".repeat(2_000)),
            message(Role::User, "t".repeat(2_000)),
            message(Role::Assistant, r#"{"answer":"done"}"#),
        ];

        compact_context(&mut messages, 256);

        assert_eq!(messages.len(), 2);
        assert!(messages[1].content.len() < 2_000);
    }

    #[test]
    fn note_marks_overflowing_history() {
        let messages = (0..NOTE_LINE_LIMIT + 10)
            .map(|index| message(Role::User, format!("message {index}")))
            .collect::<Vec<_>>();

        let note = compact_note(&messages, 10_000);

        assert!(note.contains("- ..."));
    }

    #[test]
    fn message_line_covers_all_roles() {
        let user = message(Role::User, "plain user text");
        let system = message(Role::System, "system text");
        let tool = message(Role::Tool, "tool text");

        assert!(message_line(&user).starts_with("user:"));
        assert_eq!(message_line(&system), "");
        assert_eq!(message_line(&tool), "");
    }

    #[test]
    fn assistant_line_covers_parse_edge_cases() {
        assert!(assistant_line("not json").starts_with("assistant:"));
        assert!(assistant_line(r#"{"answer":"complete"}"#).contains("complete"));
        assert!(assistant_line(r#"{"thought":"none"}"#).starts_with("assistant:"));
        assert!(assistant_line(r#"{"tool":{"arguments":{}}}"#).starts_with("assistant:"));
        assert!(assistant_line(r#"{"tool":{"name":"write_file"}}"#).starts_with("assistant:"));
    }

    #[test]
    fn assistant_line_summarizes_tool_families() {
        let tools = [
            (
                "list_files",
                r#"{"tool":{"name":"list_files","arguments":{"path":"src"}}}"#,
                "list_files(path=src)",
            ),
            (
                "read_file",
                r#"{"tool":{"name":"read_file","arguments":{"path":"src/main.rs","max_chars":4096}}}"#,
                "read_file(path=src/main.rs, max_chars=4096)",
            ),
            (
                "grep_files",
                r#"{"tool":{"name":"grep_files","arguments":{"pattern":"todo","path":"src","max_results":10}}}"#,
                "grep_files(pattern=\"todo\", path=src, max_results=10)",
            ),
            (
                "run_command",
                r#"{"tool":{"name":"run_command","arguments":{"command":"cargo test"}}}"#,
                "run_command(cargo test)",
            ),
            (
                "finish",
                r#"{"tool":{"name":"finish","arguments":{"summary":"done"}}}"#,
                "finish(summary=done)",
            ),
            (
                "unknown_tool",
                r#"{"tool":{"name":"unknown_tool","arguments":{}}}"#,
                "tool(unknown_tool):",
            ),
        ];

        for (_, json, expected) in tools {
            let line = assistant_line(json);
            assert!(line.contains(expected), "{line} should contain {expected}");
        }
    }

    #[test]
    fn arg_handles_string_number_and_missing() {
        let args = serde_json::json!({"text": "value", "count": 3});

        assert_eq!(arg(&args, "text").as_ref(), "value");
        assert_eq!(arg(&args, "count").as_ref(), "3");
        assert_eq!(arg(&args, "missing").as_ref(), "");
    }
}
