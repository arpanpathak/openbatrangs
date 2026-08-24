//! TUI constants and the chat-mode system prompt.

/// Slash commands recognized by the TUI.
pub const COMMANDS: &[&str] = &[
    "help",
    "exit",
    "quit",
    "setup",
    "models",
    "model",
    "engine",
    "pull",
    "read-only",
    "confirm",
    "steps",
    "cwd",
    "doctor",
    "clear",
    "perf",
    "mode",
    "thinking",
    "mouse",
    "yolo",
];

/// Commands with fixed value options, used for slash-command suggestions.
pub const PREFIXED_COMMANDS: &[(&str, &[&str])] = &[
    ("mode ", &["agent", "plan", "chat"]),
    ("thinking ", &["on", "off"]),
    ("mouse ", &["on", "off"]),
    ("engine ", &["ollama", "tensorrt"]),
];

/// Spinner frames shown while the agent is working.
pub const SPINNER: &[&str] = &["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

/// Maximum visible lines in the multi-line input box.
pub const MAX_INPUT_LINES: usize = 20;

/// Extra lines for the input box border.
pub const INPUT_BOX_PADDING: usize = 2;

/// Minimum input box height (border + one line).
pub const MIN_INPUT_BOX_HEIGHT: usize = 3;

/// Cap for the streaming live line to avoid unbounded memory.
pub const MAX_LIVE_CHARS: usize = 50_000;

/// Maximum chat messages kept in memory for conversational context.
pub const MAX_CHAT_HISTORY_MESSAGES: usize = 40;

/// Redraw interval in milliseconds (drives the spinner).
pub const TICK_MILLIS: u64 = 80;

/// Width of the model picker popup as a percentage of the screen.
pub const MODEL_PICKER_WIDTH_PERCENT: u16 = 60;

/// Height of the model picker popup as a percentage of the screen.
pub const MODEL_PICKER_HEIGHT_PERCENT: u16 = 40;

/// Height of the live performance panel (border + three content lines).
pub const PERF_PANEL_HEIGHT: u16 = 5;

/// Maximum height of the performance panel once wrapped lines are accounted for.
pub const PERF_MAX_PANEL_HEIGHT: u16 = 8;

/// Minimum elapsed time before the token-rate estimate refreshes.
pub const TOKEN_RATE_MIN_ELAPSED: f64 = 0.5;

/// Minimum terminal height for showing the performance panel automatically.
pub const PERF_MIN_TERMINAL_HEIGHT: u16 = 18;

/// Number of lines scrolled per PageUp/PageDown in the chat area.
pub const CHAT_SCROLL_STEP: usize = 5;

/// Banner height: title + compact Batman art + quote + model info + prompt.
pub const COMPACT_BANNER_HEIGHT: u16 = 9;

/// Banner height on small terminals (title + quote only).
pub const SMALL_BANNER_HEIGHT: u16 = 7;

/// Minimum terminal height for showing the full banner.
pub const FULL_BANNER_MIN_TERMINAL_HEIGHT: u16 = 30;

/// Minimum terminal width for showing the full banner.
pub const FULL_BANNER_MIN_WIDTH: u16 = 20;

/// Maximum number of suggestion rows rendered before scrolling.
pub const MAX_SUGGESTION_ITEMS: usize = 6;

/// Rough estimate of characters per token for local models.
pub const CHARS_PER_TOKEN: f64 = 4.0;

/// Separator line length used before a user task in the TUI log.
pub const CHAT_SEPARATOR_LENGTH: usize = 60;

/// Maximum chat lines kept in memory before older lines spill to disk.
pub const MAX_LOG_LINES: usize = 1_000;

/// How many older lines to load from disk at a time when scrolling up.
pub const LOG_LOAD_CHUNK: usize = 500;

/// Chat logs larger than this switch to the raw renderer (no syntax
/// highlighting) so generation stays responsive on low-power devices.
pub const RAW_CHAT_RENDER_THRESHOLD: usize = 50_000;

/// Chat-mode system prompt: no tools, direct conversation and code.
pub const CHAT_SYSTEM_PROMPT: &str = "You are openBatarangs, an expert coding assistant in chat mode. Answer coding questions and write complete, production-quality code when asked. Never give hello-world stubs, placeholders, or toy examples: implement the requested feature in full with real logic, proper error handling, and idiomatic code. Write clean code: use guard clauses and early returns, avoid deeply nested conditionals, keep functions focused, and use descriptive names. Match the user's language and project context, be practical and concise, and do not mention tools.";

/// Sampling temperature for plain chat completions.
pub const CHAT_TEMPERATURE: f64 = 0.7;

/// Terminal emulators tried, in order, when opening a file in `vim`.
pub const VIM_TERMINALS: &[&str] = &[
    "x-terminal-emulator",
    "gnome-terminal",
    "konsole",
    "xfce4-terminal",
    "alacritty",
    "kitty",
];
