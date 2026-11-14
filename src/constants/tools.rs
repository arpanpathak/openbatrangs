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

/// How long `run_command` keeps reading a pipe after the command's process
/// group has been stopped. The pipes normally close at once; only a process
/// that left the group, such as one started with `setsid`, keeps one open.
pub const COMMAND_OUTPUT_GRACE_MILLISECONDS: u64 = 500;

/// Prefix of the exit-status line `run_command` appends to its output; the
/// finished line reads `[exit code 0]`, `[exit code 101]`, and so on.
pub const EXIT_CODE_PREFIX: &str = "[exit code ";

/// The zero-exit marker the quality gate accepts as a verified command.
pub const SUCCESS_EXIT_MARKER: &str = "[exit code 0]";

/// The bubblewrap program that sandboxes shell commands on Linux.
pub const BUBBLEWRAP_PROGRAM: &str = "bwrap";

/// The program that sandboxes shell commands on macOS.
pub const SEATBELT_PROGRAM: &str = "sandbox-exec";

/// The Seatbelt profile for shell commands on macOS. It allows everything
/// except file writes, then allows writes in the workspace (passed as the
/// `WORKSPACE` parameter), the temporary folders and `/dev`.
pub const SEATBELT_PROFILE: &str = r#"(version 1)
(allow default)
(deny file-write*)
(allow file-write*
    (subpath (param "WORKSPACE"))
    (subpath "/private/tmp")
    (subpath "/private/var/folders")
    (subpath "/dev"))"#;

/// Command fragments that delete or overwrite files in bulk. A command that
/// contains one, with words separated by single spaces and a space at each
/// end, needs the user's approval even without `--confirm`.
pub const DESTRUCTIVE_COMMAND_PATTERNS: &[&str] = &[
    " rm ",
    " rmdir ",
    " unlink ",
    " shred ",
    " dd ",
    " mkfs",
    " truncate ",
    " sudo ",
    " chmod -R ",
    " chown -R ",
    " -delete ",
    " git reset --hard",
    " git clean ",
    " git push --force",
    " git push -f ",
    " git checkout -- ",
    " git checkout . ",
    " git restore ",
    " git branch -D ",
    " git stash drop",
    " git stash clear",
];

/// Added to the approval question when a command would run without a
/// sandbox, so the user knows it can change files outside the project.
pub const UNSANDBOXED_COMMAND_WARNING: &str = " without a sandbox";
