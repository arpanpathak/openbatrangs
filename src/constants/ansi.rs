//! ANSI escape sequences and styled terminal markers.

/// SGR escape for cyan foreground, used for informational messages and highlights.
pub const COLOR_CYAN: &str = "\x1b[36m";
/// SGR escape for magenta foreground, used for tool call descriptions.
pub const COLOR_MAGENTA: &str = "\x1b[35m";
/// SGR escape for yellow foreground, used for warnings and file change headers.
pub const COLOR_YELLOW: &str = "\x1b[33m";
/// SGR escape for green foreground, used for success messages.
pub const COLOR_GREEN: &str = "\x1b[32m";
/// SGR escape for bold/intense text, used for emphasis in status lines.
pub const COLOR_BOLD: &str = "\x1b[1m";
/// SGR escape for dim/faint text, used for de-emphasized tool output.
pub const COLOR_DIM: &str = "\x1b[2m";
/// SGR escape that resets all text attributes to terminal defaults.
pub const COLOR_RESET: &str = "\x1b[0m";

/// Green checkmark used in terminal messages.
pub const ANSI_GREEN_CHECK: &str = "\x1b[32m✅\x1b[0m";

/// Yellow "Files changed" header.
pub const ANSI_YELLOW_FILES_HEADER: &str = "\x1b[1;33m📁 Files changed:\x1b[0m";
