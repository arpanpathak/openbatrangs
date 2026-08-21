//! Filesystem and shell tool constants.

/// Directories skipped during recursive walks (build caches, VCS, dependencies).
pub const HEAVY_DIRS: &[&str] = &[
    ".git",
    "target",
    "node_modules",
    "dist",
    "build",
    ".venv",
    "venv",
    "__pycache__",
    ".cache",
    ".next",
    ".nuxt",
    ".idea",
    ".vscode",
    ".agent",
];

/// Default depth for `list_files` in the agent's initial context.
pub const DEFAULT_LIST_DEPTH: usize = 5;

/// Depth used by `grep_files` when scanning the workspace.
pub const GREP_MAX_DEPTH: usize = 12;

/// Maximum number of files listed before truncating.
pub const MAX_LISTED_FILES: usize = 5_000;

/// Number of leading bytes inspected to detect binary files.
pub const BINARY_SNIFF_BYTES: usize = 8_192;

/// Maximum number of characters returned by any tool to protect context memory.
pub const MAX_TOOL_OUTPUT: usize = 20_000;

/// Minimum shell timeout in seconds (avoids instant timeout bugs).
pub const MIN_COMMAND_TIMEOUT_SECONDS: u64 = 1;
