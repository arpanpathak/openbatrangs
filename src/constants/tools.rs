//! Filesystem and shell tool constants.

/// Directories skipped at any depth: version control, dependency trees and
/// tool caches, which never hold the project's own source.
pub const SKIPPED_DIRS: &[&str] = &[
    ".git",
    "target",
    "node_modules",
    ".venv",
    "venv",
    "__pycache__",
    ".cache",
    ".next",
    ".nuxt",
    ".idea",
    ".vscode",
    ".agent",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".gradle",
    "Pods",
    "DerivedData",
];

/// Directories skipped only directly under the workspace root, where they
/// usually hold build output or large data. Deeper down the same names are
/// often source code (`src/models`, `src/data`), so they are listed there.
pub const SKIPPED_ROOT_DIRS: &[&str] = &[
    "dist",
    "build",
    "out",
    "output",
    "outputs",
    "cache",
    "caches",
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

/// Prefix of the exit-status line `run_command` appends to its output; the
/// finished line reads `[exit code 0]`, `[exit code 101]`, and so on.
pub const EXIT_CODE_PREFIX: &str = "[exit code ";

/// The zero-exit marker the quality gate accepts as a verified command.
pub const SUCCESS_EXIT_MARKER: &str = "[exit code 0]";
