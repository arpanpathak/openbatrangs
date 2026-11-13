//! TUI constants and the chat-mode system prompt.

use crossterm::event::KeyboardEnhancementFlags;

/// Slash commands recognized by the TUI.
pub const COMMANDS: &[&str] = &[
    "help",
    "status",
    "exit",
    "quit",
    "setup",
    "models",
    "model",
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
];

/// Lines printed by `/help`: every slash command, then the key bindings.
pub const HELP_LINES: &[&str] = &[
    "═══ Commands ═══",
    "  /help              — show this help",
    "  /exit, /quit       — leave the TUI",
    "  /status            — show current settings and model",
    "  /mode agent|plan|chat — agent=tools, plan=read-only, chat=no tools",
    "  /model <tag>       — switch to a specific installed model",
    "  /models            — open model picker",
    "  /pull <tag>        — download a model from Ollama",
    "  /setup             — install/start Ollama + pull a model",
    "  /doctor            — check Ollama + best model recommendation",
    "  /read-only         — toggle read-only mode (no writes/commands)",
    "  /confirm           — toggle confirm before writes/commands",
    "  /yolo              — disable confirmations (careful!)",
    "  /thinking on|off   — show/hide model reasoning",
    "  /steps <n>         — set max agent iterations",
    "  /cwd <path>        — change workspace directory",
    "  /perf              — toggle performance panel",
    "  /mouse on|off      — wheel/scrollbar (off = native select/copy)",
    "  /clear             — clear chat log",
    "",
    "═══ Keys ═══",
    "  Enter = send · Ctrl+J / Alt+Enter = newline · Ctrl+C = cancel",
    "  PgUp/PgDn = scroll · Mouse wheel = scroll · Tab = complete, or switch to the transcript",
    "  In the transcript: ↑↓ = move · Enter / Space = fold or unfold · Tab / Esc = back",
    "  Click a step, thinking line or tool line to fold or unfold it",
];

/// Spinner frames shown while the agent is working.
pub const SPINNER: &[&str] = &["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

/// Maximum visible lines in the multi-line input box.
pub const MAX_INPUT_LINES: usize = 20;

/// Extra lines for the input box border.
pub const INPUT_BOX_PADDING: usize = 2;

/// Minimum input box height (border + one line).
pub const MIN_INPUT_BOX_HEIGHT: usize = 3;

/// Maximum chat messages kept in memory for conversational context.
pub const MAX_CHAT_HISTORY_MESSAGES: usize = 40;

/// Redraw interval in milliseconds (drives the spinner).
pub const TICK_MILLIS: u64 = 80;

/// Most background events handled between two redraws. Streaming tokens
/// arrive in bursts, and one redraw per batch keeps the terminal responsive.
pub const MAX_BACKGROUND_EVENTS_PER_FRAME: usize = 32;

/// Width of the model picker popup as a percentage of the screen.
pub const MODEL_PICKER_WIDTH_PERCENT: u16 = 60;

/// Height of the model picker popup as a percentage of the screen.
pub const MODEL_PICKER_HEIGHT_PERCENT: u16 = 40;

/// Height of the live performance panel (border + three content lines).
pub const PERF_PANEL_HEIGHT: u16 = 5;

/// Maximum height of the performance panel once wrapped lines are accounted
/// for: room for four server lines, the `this machine` heading and the three
/// local lines.
pub const PERF_MAX_PANEL_HEIGHT: u16 = 10;

/// Minimum elapsed time before the token-rate estimate refreshes.
pub const TOKEN_RATE_MIN_ELAPSED: f64 = 0.5;

/// A pause between streamed chunks longer than this, such as a tool running,
/// starts a new tokens-per-second window.
pub const STREAM_GAP: std::time::Duration = std::time::Duration::from_millis(1500);

/// Minimum terminal height for showing the performance panel automatically.
pub const PERF_MIN_TERMINAL_HEIGHT: u16 = 18;

/// Number of lines scrolled per PageUp/PageDown in the chat area.
pub const CHAT_SCROLL_STEP: usize = 5;

/// Height of the compact top header (border, model line, border).
pub const HEADER_HEIGHT: u16 = 3;

/// Maximum number of suggestion rows rendered before scrolling.
pub const MAX_SUGGESTION_ITEMS: usize = 6;

/// Rough estimate of characters per token for local models.
pub const CHARS_PER_TOKEN: f64 = 4.0;

/// Separator line length used before a user task in the TUI log.
pub const CHAT_SEPARATOR_LENGTH: usize = 60;

/// Maximum transcript blocks kept in memory before older blocks spill to disk.
pub const MAX_TRANSCRIPT_BLOCKS: usize = 300;

/// How many blocks to load from disk at a time when scrolling past the
/// blocks in memory.
pub const BLOCK_LOAD_CHUNK: usize = 50;

/// Chat-mode system prompt: no tools, direct conversation and code.
pub const CHAT_SYSTEM_PROMPT: &str = r#"You are a coding assistant.

- Implement complete code; no stubs, placeholders, or toy examples unless explicitly requested.
- For design/docs requests, write structured documents covering architecture, data flow, interfaces, trade-offs, and implementation steps.
- Use any project context the user provides.
"#;

/// Sampling temperature for plain chat completions.
///
/// Lower than the old default so small local models produce longer, more
/// coherent code and documentation instead of drifting after a few hundred
/// tokens.
pub const CHAT_TEMPERATURE: f64 = 0.3;

/// Terminal emulators tried, in order, when opening a file in `vim`.
pub const VIM_TERMINALS: &[&str] = &[
    "x-terminal-emulator",
    "gnome-terminal",
    "konsole",
    "xfce4-terminal",
    "alacritty",
    "kitty",
];

/// Hints shown under the banner until the first message.
pub const WELCOME_HINTS: &[&str] = &[
    "Type a task to start coding, or /help for commands.",
    "/mode chat for plain conversation, /mode plan for read-only planning.",
];

/// Kitty keyboard protocol flags the TUI asks the terminal for.
///
/// Escape codes are disambiguated, so Esc and Alt or Ctrl combinations arrive
/// as distinct keys, and release events are reported so they can be ignored.
/// Alternate keys make a shifted key arrive as the character it types, `<` for
/// Shift+`,`. The flag that reports every key as an escape code is left off:
/// crossterm cannot read the text a key produces in that mode, so shifted
/// symbols, Option characters and non-US layouts would be lost.
pub const KEYBOARD_FLAGS: KeyboardEnhancementFlags =
    KeyboardEnhancementFlags::DISAMBIGUATE_ESCAPE_CODES
        .union(KeyboardEnhancementFlags::REPORT_EVENT_TYPES)
        .union(KeyboardEnhancementFlags::REPORT_ALTERNATE_KEYS);
