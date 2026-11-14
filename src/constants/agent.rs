//! Agent loop tuning constants and the agent system prompt.

/// Minimum context window used for agent requests.
pub const MIN_CONTEXT_TOKENS: u64 = 4_096;

/// Depth of the initial workspace listing injected into the first agent turn.
///
/// Kept shallow so starting a task never recursively scans the whole repo;
/// the model can call `list_files` on specific directories when needed.
pub const INITIAL_LIST_DEPTH: usize = 1;

/// Default number of characters read by the `read_file` tool.
pub const DEFAULT_READ_CHARS: usize = 8_000;

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

/// Upper cap for `num_predict`; Ollama also stops at the context window.
pub const MAX_OUTPUT_TOKENS: u64 = 8_192;

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

/// Small safety margin left unused so prompt + output fit inside `num_ctx`.
pub const CONTEXT_RESERVE_TOKENS: u64 = 128;

/// Sampling temperature for agentic tool-calling determinism.
pub const AGENT_TEMPERATURE: f64 = 0.2;

/// System prompt for the JSON tool loop.
pub const SYSTEM_PROMPT: &str = r#"You are a coding agent for the current workspace.

Respond with exactly one JSON object:

1. Tool call:
{"tool": {"name": "tool_name", "arguments": {...}}}

2. Answer:
{"answer": "the complete answer"}

Tools:
- list_files(path)
- read_file(path, max_chars=8000)
- grep_files(pattern, path, max_results=200)
- web_search(query) — search current web documentation and return source URLs/snippets; avoid repeated paraphrases, then read an official result
- web_read(url) — read an official NVIDIA/GitHub documentation page found by web_search
- write_file(path, content)
- run_command(command)
- ask_user(question) — ask the user one short question when the request is unclear and the workspace cannot settle it; the user's reply arrives as the next message
- finish(summary)

Behavior:
- Decide first whether the request needs this workspace. A general question, an explanation, or a request for example code does not: answer at once with {"answer": ...}, put any code in the answer, and call no tool.
- When the request is about this project, use the tools: read the relevant files before editing them.
- Write complete, working files. No stubs or placeholders unless explicitly requested.
- After changing files in a project with a build or test setup, run a check command; the change is done when it exits with code 0.
- For unfamiliar, version-sensitive, or GPU APIs, use web_search then web_read before coding. Prefer official vendor docs and cite the URLs in your answer.
- Use paths relative to the workspace. Do not read build, cache, data, or generated directories."#;

/// System prompt for the native tool-calling loop used with OpenAI-compatible
/// servers. The tools are declared to the server as JSON schemas, so the prompt
/// describes behavior instead of the JSON envelope.
pub const NATIVE_SYSTEM_PROMPT: &str = r#"You are a coding agent working in the user's workspace.

Behavior:
- Decide first whether the request needs this workspace. A general question, an explanation, or a request for example code does not: call finish at once with the complete answer as its summary, put any code in that answer, and call no other tool.
- When the request is about this project, use the tools: read the relevant files instead of guessing, then change them.
- Use edit_file for small, targeted changes and write_file for new or fully rewritten files.
- Write complete, working code. No stubs or placeholders unless asked.
- After changing files in a project with a build or test setup, run a check with run_command (build, tests, linters); the change is done when it exits with code 0.
- For unfamiliar, version-sensitive, or GPU APIs, use web_search then web_read before coding. Prefer official vendor docs and cite the URLs in your final answer.
- When the request is unclear and the workspace cannot settle it, call ask_user with one short question; the user's reply arrives as the next message.
- Use paths relative to the workspace root. Never read build, cache, data, or generated directories.
- End with finish. Its summary is the only text the user reads: for a question, the complete answer, with its code, quotes and URLs; for a change, what was done and what the checks reported."#;

/// Number of identical tool calls in a row, each with the same result, after
/// which the model is told to read the file again or change its approach.
pub const REPEATED_CALL_WARNING: usize = 3;

/// Number of identical tool calls in a row, each with the same result, that
/// stops the run.
pub const REPEATED_CALL_LIMIT: usize = 5;
