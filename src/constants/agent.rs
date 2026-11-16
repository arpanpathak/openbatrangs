//! Agent loop tuning constants and the agent system prompt.

/// Minimum context window used for agent requests.
pub const MIN_CONTEXT_TOKENS: u64 = 4_096;

/// Depth of the initial workspace listing injected into the first agent turn.
///
/// Kept shallow so starting a task never recursively scans the whole repo;
/// the model can call `list_files` on specific directories when needed.
pub const INITIAL_LIST_DEPTH: usize = 1;

/// Default number of characters read by the `read_file` tool: as many as
/// one tool result may hold, so a file of a few hundred lines is read whole.
pub const DEFAULT_READ_CHARS: usize = 20_000;

/// Default maximum results for the `grep_files` tool.
pub const DEFAULT_GREP_MAX_RESULTS: usize = 200;

/// Timeout for shell commands launched by the agent, in seconds.
pub const COMMAND_TIMEOUT_SECONDS: u64 = 120;

/// Conversation trimming keeps the system message, the initial task, and this
/// many recent messages.
pub const MAX_HISTORY_MESSAGES: usize = 40;

/// Number of the most recent messages kept verbatim before older tool
/// exchanges are rolled into the compacted context note.
pub const KEEP_RECENT_MESSAGES: usize = 4;

/// Maximum characters retained for one message in the recent verbatim window.
pub const MAX_RECENT_MESSAGE_CHARS: usize = 6_000;

/// Maximum characters in the automatically compacted older-context note.
pub const COMPACT_NOTE_MAX_CHARS: usize = 3_000;

/// Rough characters-per-token heuristic used for prompt budgeting.
pub const CHARS_PER_TOKEN: f64 = 4.0;

/// Minimum output tokens reserved for a model response.
pub const MIN_OUTPUT_TOKENS: u64 = 1_024;

/// Upper cap for `num_predict` and `max_tokens`; the server also stops at
/// the context window. A thinking model writing a whole file needs room for
/// its reasoning and the file in one reply: a 60 KB chapter is about 15,000
/// tokens before any thinking.
pub const MAX_OUTPUT_TOKENS: u64 = 32_768;

/// Longest existing file, in lines, that `write_file` may overwrite. A
/// model that rewrites a long file from memory drops most of it, so longer
/// files are changed with `edit_file`.
pub const MAX_OVERWRITE_LINES: usize = 100;

/// Share of the prompt budget, in percent, that the earlier turns of the
/// conversation may use in a new task's first message.
pub const EARLIER_TURNS_BUDGET_PERCENT: usize = 25;

/// Maximum characters kept from one earlier turn of the conversation.
pub const MAX_EARLIER_TURN_CHARS: usize = 2_000;

/// Heading of the earlier turns in a new task's first message.
pub const EARLIER_TURNS_HEADING: &str = "Earlier in this conversation:";

/// Fewest characters a tool result is shortened to when the prompt is over
/// its budget.
pub const MIN_TOOL_RESULT_CHARS: usize = 512;

/// Prefix of the user messages that carry a tool's output in the JSON loop.
pub const TOOL_RESULT_PREFIX: &str = "Tool result:";

/// Feedback injected when the model produced an empty turn in the native loop.
pub const EMPTY_TURN_NUDGE: &str = "Your previous turn produced no answer or tool call. \
     Continue the task now: call the necessary tools to inspect or change files; \
     do not repeat your plan or ask the user to do the work.";

/// Feedback injected when the server cut the model's reply off at the output
/// limit; the cut-off tool call is not run.
pub const CUT_REPLY_NOTE: &str = "Your previous reply reached the output limit and was cut off, \
     so its tool call was not run and nothing was written. Write a long file in parts of at \
     most about 200 lines: call write_file with the first part, then append_file with each \
     next part, one call per reply. Do not repeat parts already written.";

/// Feedback injected when the model's reply holds a tool call written as text
/// that cannot be read, such as Gemma's own call syntax cut off partway.
pub const BROKEN_CALL_NOTE: &str = "Your previous reply wrote a tool call as text, and it could \
     not be read, so nothing ran. Call tools only through the tool-calling interface, with one \
     complete call per reply; write a long file in parts.";

/// Feedback injected when the model's reply fell into repeating one short
/// piece of text and was stopped.
pub const RUNAWAY_NOTE: &str = "Your previous reply kept repeating the same text, so it was \
     stopped and dropped. Continue the task with one tool call, and do not repeat text.";

/// Small safety margin left unused so prompt + output fit inside `num_ctx`.
pub const CONTEXT_RESERVE_TOKENS: u64 = 128;

/// System prompt for the JSON tool loop.
pub const SYSTEM_PROMPT: &str = r#"You are a coding agent for the current workspace.

Respond with exactly one JSON object:

1. Tool call:
{"tool": {"name": "tool_name", "arguments": {...}}}

2. Answer:
{"answer": "the complete answer"}

Tools:
- list_files(path)
- read_file(path, offset=1, limit, max_chars=20000) — whole lines from line offset; a long file is read in parts, and the result says the offset to read on from
- grep_files(pattern, path, max_results=200)
- web_search(query) — search the web for documentation and return source URLs and snippets; search once or twice, then read the best result
- web_read(url) — read a public web page, such as documentation found by web_search
- write_file(path, content)
- append_file(path, content) — add text to the end of a file; write a long file in parts with write_file, then append_file
- run_command(command)
- ask_user(question, options, multiple) — ask the user one short question when the request is unclear and the workspace cannot settle it; list known choices in options (multiple: true to allow several); the user's reply arrives as the next message
- finish(summary)

Behavior:
- Decide first whether the request needs this workspace. A general question, an explanation, or a request for example code does not: answer at once with {"answer": ...}, put any code in the answer, and call no tool.
- When the request is about this project, use the tools: read the relevant files before editing them.
- Write complete, working files. No stubs or placeholders unless explicitly requested.
- After changing files in a project with a build or test setup, run a check command; the change is done when it exits with code 0.
- When you are not sure about an API, a library or crate version, a command-line flag, a configuration key or an error message, call web_search and read the best result with web_read before you write code; do not write it from memory. Prefer official documentation and cite the URLs in your answer.
- Use paths relative to the workspace. Do not read build, cache, data, or generated directories."#;

/// System prompt for the native tool-calling loop used with OpenAI-compatible
/// servers. The tools are declared to the server as JSON schemas, so the prompt
/// describes behavior instead of the JSON envelope.
pub const NATIVE_SYSTEM_PROMPT: &str = r#"You are a coding agent working in the user's workspace. You work one step at a time: each reply is a short thought followed by tool calls, and the next message brings their results.

How to think:
- Keep thinking short. Decide the next step, then call the tool; its result shows whether the step was right.
- Do not reopen a decision you have made. If you start a sentence with "Wait" or write the same idea twice, stop thinking and make the call you decided on.
- Plan only the next step, not the whole task.

Questions and work:
- A general question, an explanation, or a request for example code does not need this workspace: call finish at once with the complete answer as its summary, with any code in it, and call no other tool.
- A request about this project needs the tools. List and read the relevant files before you change anything; never guess what a file contains.

Changing files:
- Change an existing file with edit_file. Copy old_string exactly from what read_file returned, including indentation, and keep it to the lines that change plus enough context to be unique. Make one edit per change.
- Never rewrite an existing file with write_file; files longer than 100 lines are refused.
- Create a new file with write_file. One reply holds at most about 200 lines of a file, so write a longer file in parts: write_file with the first part, then append_file with each next part, one part per reply, never repeating a part.
- When edit_file cannot find old_string, read the file again and copy the text exactly; do not switch to write_file.
- Write complete, working code. No stubs or placeholders unless asked.
- Use paths relative to the workspace root. Never read build, cache, data, or generated directories.

Checking:
- After changing files in a project with a build or test setup, run a check with run_command (build, tests, linters). The change is done when the check exits with code 0. When it fails, read the error, fix its cause, and run the check again.

Facts:
- When you are not sure about an API, a library or crate version, a command-line flag, a configuration key or an error message, call web_search and read the best result with web_read before you write code; do not write it from memory. Prefer official documentation and cite the URLs in your final answer.

When you are stuck:
- When the request is unclear and the workspace cannot settle it, call ask_user with one short question; the user's reply arrives as the next message.
- When a tool fails the same way twice, change your approach instead of repeating the call.

Finishing:
- Finish only when every part of the request is done and checked; a plan or a partial change is not done.
- End with finish. Its summary is the only text the user reads: for a question, the complete answer, with its code, quotes and URLs; for a change, what was done and what the checks reported."#;

/// Number of identical tool calls in a row, each with the same result, after
/// which the model is told to read the file again or change its approach.
pub const REPEATED_CALL_WARNING: usize = 3;

/// Number of identical tool calls in a row, each with the same result, that
/// stops the run.
pub const REPEATED_CALL_LIMIT: usize = 5;
