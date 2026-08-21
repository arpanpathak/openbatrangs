//! ANSI escape sequences and styled terminal markers.

pub const COLOR_CYAN: &str = "\x1b[36m";
pub const COLOR_MAGENTA: &str = "\x1b[35m";
pub const COLOR_YELLOW: &str = "\x1b[33m";
pub const COLOR_GREEN: &str = "\x1b[32m";
pub const COLOR_BOLD: &str = "\x1b[1m";
pub const COLOR_DIM: &str = "\x1b[2m";
pub const COLOR_RESET: &str = "\x1b[0m";

/// Green checkmark used in terminal messages.
pub const ANSI_GREEN_CHECK: &str = "\x1b[32m✅\x1b[0m";

/// Yellow "Files changed" header.
pub const ANSI_YELLOW_FILES_HEADER: &str = "\x1b[1;33m📁 Files changed:\x1b[0m";
