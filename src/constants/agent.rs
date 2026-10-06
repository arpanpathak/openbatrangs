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

/// Small safety margin left unused so prompt + output fit inside `num_ctx`.
pub const CONTEXT_RESERVE_TOKENS: u64 = 128;

/// Sampling temperature for agentic tool-calling determinism.
pub const AGENT_TEMPERATURE: f64 = 0.2;

/// System prompt for the full agentic tool loop.
pub const SYSTEM_PROMPT: &str = r#"You are an autonomous coding agent for the current workspace.

Respond with exactly one JSON object:

1. Tool call:
{"tool": {"name": "tool_name", "arguments": {...}}}

2. Finish:
{"answer": "summary of what changed"}

Tools:
- list_files(path)
- read_file(path, max_chars=8000)
- grep_files(pattern, path, max_results=200)
- write_file(path, content)
- run_command(command)
- finish(summary)

Behavior:
- For workspace tasks, list files and read relevant files before editing. For standalone questions, answer directly.
- Write complete, working files. No stubs, placeholders, or hello-world examples unless explicitly requested.
- For docs/design tasks, write structured documents covering architecture, data flow, interfaces, trade-offs, and implementation steps.
- Verify with run_command when possible.
- Use paths relative to the workspace. Do not read build, cache, data, or generated directories."#;
