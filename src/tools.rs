//! Safe filesystem tools available to the coding agent.
//!
//! All tools operate on paths relative to the workspace root. Absolute paths
//! and `..` traversal are rejected so the model cannot escape the project.

use anyhow::{anyhow, bail, Context, Result};
use regex::Regex;
use std::path::{Component, Path, PathBuf};
use std::time::Duration;
use walkdir::WalkDir;

/// Maximum number of characters returned by any tool to protect context memory.
pub const MAX_TOOL_OUTPUT: usize = 20_000;

/// Directories skipped during recursive walks (build caches, VCS, dependencies).
const HEAVY_DIRS: &[&str] = &[
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
const GREP_MAX_DEPTH: usize = 12;

/// Maximum number of files listed before truncating.
const MAX_LISTED_FILES: usize = 5_000;

/// Number of leading bytes inspected to detect binary files.
const BINARY_SNIFF_BYTES: usize = 8_192;

/// Minimum shell timeout in seconds (avoids instant timeout bugs).
const MIN_COMMAND_TIMEOUT_SECONDS: u64 = 1;

/// Returns true when a directory name should be skipped by recursive walks.
fn is_heavy_dir(name: &str) -> bool {
    HEAVY_DIRS.contains(&name)
}

/// Resolve a model-supplied path safely inside `root`.
///
/// # Arguments
/// - `root`: workspace root directory.
/// - `path`: relative path supplied by the model.
///
/// # Returns
/// The joined `PathBuf`, or an error if the path is absolute or contains `..`.
pub fn resolve_in_root(root: &Path, path: &str) -> Result<PathBuf> {
    let requested = Path::new(path);
    if requested.is_absolute() {
        bail!("absolute paths are not allowed; use paths relative to the workspace");
    }
    for component in requested.components() {
        if matches!(component, Component::ParentDir) {
            bail!("'..' is not allowed in tool paths");
        }
    }
    Ok(root.join(requested))
}

/// Build a recursive directory walker that skips heavy directories.
///
/// # Arguments
/// - `root`: directory to walk.
/// - `max_depth`: maximum recursion depth.
///
/// # Returns
/// An iterator over directory entries.
fn walker(
    root: &Path,
    max_depth: usize,
) -> impl Iterator<Item = Result<walkdir::DirEntry, walkdir::Error>> + '_ {
    WalkDir::new(root)
        .max_depth(max_depth)
        .follow_links(false)
        .into_iter()
        .filter_entry(|entry| {
            if entry.depth() == 0 {
                return true;
            }
            if entry.file_type().is_dir() {
                let name = entry.file_name().to_string_lossy();
                return !is_heavy_dir(&name);
            }
            true
        })
}

/// List files in a directory as a human-readable string.
///
/// # Arguments
/// - `root`: workspace root used for path safety.
/// - `path`: relative directory to list.
/// - `max_depth`: maximum recursion depth.
///
/// # Returns
/// One line per file: relative path and byte size.
pub fn list_files(root: &Path, path: &str, max_depth: usize) -> Result<String> {
    let directory = resolve_in_root(root, path)?;
    if !directory.is_dir() {
        bail!("not a directory: {}", directory.display());
    }

    let mut output = String::new();
    let mut count = 0usize;
    for entry in walker(&directory, max_depth) {
        let entry = entry.context("walkdir error")?;
        if entry.file_type().is_file() {
            let relative = entry
                .path()
                .strip_prefix(root)
                .unwrap_or(entry.path())
                .to_string_lossy()
                .to_string();
            let size = entry.metadata().map(|metadata| metadata.len()).unwrap_or(0);
            output.push_str(&format!("{relative} ({size} bytes)\n"));
            count += 1;
            if count >= MAX_LISTED_FILES {
                output.push_str(&format!("... (truncated at {MAX_LISTED_FILES} files)\n"));
                break;
            }
        }
    }
    if count == 0 {
        output.push_str("(no files found)\n");
    }
    Ok(truncate(output))
}

/// Read a text file with a size cap.
///
/// # Arguments
/// - `root`: workspace root used for path safety.
/// - `path`: relative file path.
/// - `max_chars`: maximum characters to include.
///
/// # Returns
/// A header line plus the (possibly truncated) file contents.
pub fn read_file(root: &Path, path: &str, max_chars: usize) -> Result<String> {
    let file = resolve_in_root(root, path)?;
    if !file.is_file() {
        bail!("not a file: {}", file.display());
    }
    let content = std::fs::read_to_string(&file).context("failed to read file (may be binary)")?;
    let truncated = truncate_to(content, max_chars);
    Ok(format!(
        "--- {} ---\n{}",
        file.strip_prefix(root).unwrap_or(&file).display(),
        truncated
    ))
}

/// Write (or overwrite) a file, creating parent directories as needed.
///
/// # Arguments
/// - `root`: workspace root used for path safety.
/// - `path`: relative file path.
/// - `content`: full file contents.
///
/// # Returns
/// A confirmation string with the absolute path and byte count.
pub fn write_file(root: &Path, path: &str, content: &str) -> Result<String> {
    let file = resolve_in_root(root, path)?;
    if let Some(parent) = file.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("failed to create parent dirs for {}", file.display()))?;
    }
    std::fs::write(&file, content)
        .with_context(|| format!("failed to write {}", file.display()))?;
    Ok(format!(
        "✅ wrote {} ({} bytes)",
        file.display(),
        content.len()
    ))
}

/// Regex-search file contents in a directory.
///
/// # Arguments
/// - `root`: workspace root used for path safety.
/// - `pattern`: regular expression to search for.
/// - `path`: relative directory to search.
/// - `max_results`: maximum number of matches to return.
///
/// # Returns
/// Lines of the form `file:line: matched text`.
pub fn grep_files(root: &Path, pattern: &str, path: &str, max_results: usize) -> Result<String> {
    let directory = resolve_in_root(root, path)?;
    if !directory.is_dir() {
        bail!("not a directory: {}", directory.display());
    }
    let regex = Regex::new(pattern).context("invalid regex pattern")?;
    let mut output = String::new();
    let mut count = 0usize;

    for entry in walker(&directory, GREP_MAX_DEPTH) {
        let entry = entry.context("walkdir error")?;
        if !entry.file_type().is_file() {
            continue;
        }
        // Skip obvious binary files by sniffing for NUL bytes.
        if let Ok(data) = std::fs::read(entry.path()) {
            if data.iter().take(BINARY_SNIFF_BYTES).any(|&byte| byte == 0) {
                continue;
            }
        }
        let Ok(content) = std::fs::read_to_string(entry.path()) else {
            continue;
        };
        let relative = entry
            .path()
            .strip_prefix(root)
            .unwrap_or(entry.path())
            .to_string_lossy()
            .to_string();
        for (line_index, line) in content.lines().enumerate() {
            if regex.is_match(line) {
                let trimmed = line.trim();
                output.push_str(&format!("{relative}:{}: {trimmed}\n", line_index + 1));
                count += 1;
                if count >= max_results {
                    output.push_str(&format!("... (truncated at {max_results} matches)\n"));
                    return Ok(truncate(output));
                }
            }
        }
    }
    if count == 0 {
        output.push_str("(no matches)\n");
    }
    Ok(truncate(output))
}

/// Run a shell command in the workspace and capture combined output.
///
/// # Arguments
/// - `root`: working directory for the command.
/// - `command`: shell command line.
/// - `timeout_secs`: maximum allowed runtime.
///
/// # Returns
/// Captured stdout/stderr plus exit status, truncated to `MAX_TOOL_OUTPUT`.
pub async fn run_command(root: &Path, command: &str, timeout_secs: u64) -> Result<String> {
    let timeout = Duration::from_secs(timeout_secs.max(MIN_COMMAND_TIMEOUT_SECONDS));
    let output = tokio::time::timeout(
        timeout,
        tokio::process::Command::new("bash")
            .arg("-lc")
            .arg(command)
            .current_dir(root)
            .output(),
    )
    .await
    .map_err(|_| anyhow!("command timed out after {timeout_secs}s"))?
    .context("failed to spawn shell")?;

    let mut text = String::from_utf8_lossy(&output.stdout).to_string();
    if !output.stderr.is_empty() {
        text.push_str("\n[stderr]\n");
        text.push_str(&String::from_utf8_lossy(&output.stderr));
    }

    let exit_code = output
        .status
        .code()
        .map(|code| code.to_string())
        .unwrap_or_else(|| "signal".to_string());
    text.push_str(&format!("\n[exit code {exit_code}]\n"));
    Ok(truncate(text))
}

/// Truncate a string to `MAX_TOOL_OUTPUT` characters.
pub fn truncate(text: String) -> String {
    truncate_to(text, MAX_TOOL_OUTPUT)
}

/// Truncate a string to at most `max` characters, appending an omission note.
fn truncate_to(text: String, max: usize) -> String {
    if text.len() <= max {
        return text;
    }
    let mut result = text.chars().take(max).collect::<String>();
    result.push_str(&format!(
        "\n... (truncated, {} chars omitted)",
        text.chars().count().saturating_sub(max)
    ));
    result
}
