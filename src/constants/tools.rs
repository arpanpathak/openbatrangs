//! Filesystem and shell tool constants.

/// Directories skipped during recursive walks (build caches, VCS, dependencies,
/// and large data directories that are almost never relevant to coding tasks).
pub const HEAVY_DIRS: &[&str] = &[
    ".git",
    "target",
    "node_modules",
    "dist",
    "build",
    "out",
    "output",
    "outputs",
    ".venv",
    "venv",
    "__pycache__",
    ".cache",
    "cache",
    "caches",
    ".next",
    ".nuxt",
    ".idea",
    ".vscode",
    ".agent",
    "data",
    "datasets",
    "dataset",
    "books",
    "models",
    "weights",
    "checkpoints",
    "logs",
    "tmp",
    "temp",
    "artifacts",
    "generated",
    "coverage",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".gradle",
    "Pods",
    "DerivedData",
];

/// Default depth for `list_files` tool calls made by the agent.
pub const DEFAULT_LIST_DEPTH: usize = 2;

/// Depth used by `grep_files` when scanning the workspace.
pub const GREP_MAX_DEPTH: usize = 12;

/// Maximum number of files listed before truncating.
///
/// Kept low enough that a listing fits comfortably inside the tool-output cap
/// instead of flooding the model with hundreds of irrelevant paths.
pub const MAX_LISTED_FILES: usize = 500;

/// Number of leading bytes inspected to detect binary files.
pub const BINARY_SNIFF_BYTES: usize = 8_192;

/// Maximum number of characters returned by any tool to protect context memory.
pub const MAX_TOOL_OUTPUT: usize = 20_000;

/// Minimum shell timeout in seconds (avoids instant timeout bugs).
pub const MIN_COMMAND_TIMEOUT_SECONDS: u64 = 1;
