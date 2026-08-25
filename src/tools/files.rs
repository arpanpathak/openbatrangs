//! # Filesystem tools: list, read, write, grep
//!
//! These are the safe file operations exposed to the agent. Every function
//! validates paths through [`super::path`], skips heavy directories, and caps
//! output so a huge repository cannot flood the model's context window.
//!
//! ## References
//!
//! - Regex in Rust: <https://docs.rs/regex/latest/regex/>
//! - `walkdir` for recursive traversal: <https://docs.rs/walkdir/latest/walkdir/>
//! - Path traversal attack: <https://owasp.org/www-community/attacks/Path_Traversal>

use super::path::{canonical_root, ensure_canonical_within_root, resolve_in_root};
use super::text::{truncate, truncate_to};
use crate::constants::tools::{BINARY_SNIFF_BYTES, GREP_MAX_DEPTH, HEAVY_DIRS, MAX_LISTED_FILES};
use anyhow::{bail, Context, Result};
use regex::Regex;
use std::path::Path;
use walkdir::WalkDir;

/// Returns true when a directory name should be skipped by recursive walks.
fn is_heavy_dir(name: &str) -> bool {
    HEAVY_DIRS.contains(&name)
}

/// Return `path` relative to the workspace root for display.
///
/// `entry.path()` comes from a walk rooted at the canonical directory, so it is
/// absolute even when the caller supplied a relative `root` like `"."`. Always
/// strip against the canonical root first, then fall back to the raw root.
fn relative_to_root<'a>(path: &'a Path, root: &Path, canonical_root: &Path) -> &'a Path {
    path.strip_prefix(canonical_root)
        .or_else(|_| path.strip_prefix(root))
        .unwrap_or(path)
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
pub(crate) fn list_files(root: &Path, path: &str, max_depth: usize) -> Result<String> {
    let directory = resolve_in_root(root, path)?;
    if !directory.is_dir() {
        bail!("not a directory: {}", directory.display());
    }
    let canonical_root = canonical_root(root)?;
    let directory = ensure_canonical_within_root(root, &directory)?;

    let mut output = String::new();
    let mut count = 0usize;
    for entry in walker(&directory, max_depth) {
        let entry = entry.context("walkdir error")?;
        if entry.file_type().is_file() {
            let relative = relative_to_root(entry.path(), root, &canonical_root)
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
pub(crate) fn read_file(root: &Path, path: &str, max_chars: usize) -> Result<String> {
    let file = resolve_in_root(root, path)?;
    if !file.is_file() {
        bail!("not a file: {}", file.display());
    }
    let canonical_root = canonical_root(root)?;
    let file = ensure_canonical_within_root(root, &file)?;
    let content = std::fs::read_to_string(&file).context("failed to read file (may be binary)")?;
    let truncated = truncate_to(content, max_chars);
    Ok(format!(
        "--- {} ---\n{}",
        relative_to_root(&file, root, &canonical_root).display(),
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
pub(crate) fn write_file(root: &Path, path: &str, content: &str) -> Result<String> {
    let file = resolve_in_root(root, path)?;
    let file = if let Some(parent) = file.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("failed to create parent dirs for {}", file.display()))?;
        let canonical_parent = ensure_canonical_within_root(root, parent)?;
        let file = canonical_parent.join(file.file_name().unwrap_or_default());
        // Parent-directory symlinks are already rejected by canonicalization,
        // but a symlink in the final component would still be followed by
        // `std::fs::write`. Refuse to write through any existing symlink so the
        // agent can never modify a file outside the workspace.
        if let Ok(metadata) = std::fs::symlink_metadata(&file) {
            if metadata.file_type().is_symlink() {
                bail!("refusing to write through symlink: {}", file.display());
            }
        }
        file
    } else {
        file
    };
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
pub(crate) fn grep_files(
    root: &Path,
    pattern: &str,
    path: &str,
    max_results: usize,
) -> Result<String> {
    let directory = resolve_in_root(root, path)?;
    if !directory.is_dir() {
        bail!("not a directory: {}", directory.display());
    }
    let canonical_root = canonical_root(root)?;
    let directory = ensure_canonical_within_root(root, &directory)?;
    let regex = Regex::new(pattern).context("invalid regex pattern")?;
    let max_results = max_results.max(1);
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
        let relative = relative_to_root(entry.path(), root, &canonical_root)
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;

    fn temp_dir() -> PathBuf {
        crate::test_support::unique_temp_dir("openbatrangs-tools-test")
    }

    #[test]
    fn list_files_returns_relative_paths_and_sizes() {
        let root = temp_dir();
        fs::write(root.join("a.txt"), "hello").unwrap();
        fs::create_dir_all(root.join("sub")).unwrap();
        fs::write(root.join("sub/b.txt"), "world").unwrap();

        let output = list_files(&root, ".", 5).unwrap();
        assert!(output.contains("a.txt (5 bytes)"));
        assert!(output.contains("sub/b.txt (5 bytes)"));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn list_files_rejects_missing_directory() {
        let root = temp_dir();
        assert!(list_files(&root, "missing", 5).is_err());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn list_files_skips_heavy_dirs() {
        let root = temp_dir();
        fs::create_dir_all(root.join("target")).unwrap();
        fs::create_dir_all(root.join(".agent/cache")).unwrap();
        fs::create_dir_all(root.join("data/text")).unwrap();
        fs::write(root.join("target/ignored.txt"), "x").unwrap();
        fs::write(root.join(".agent/cache/secret.txt"), "x").unwrap();
        fs::write(root.join("data/text/book.txt"), "x").unwrap();
        fs::write(root.join("keep.txt"), "y").unwrap();

        let output = list_files(&root, ".", 5).unwrap();
        assert!(output.contains("keep.txt"));
        assert!(!output.contains("target/ignored.txt"));
        assert!(!output.contains(".agent/cache/secret.txt"));
        assert!(!output.contains("data/text/book.txt"));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn read_file_returns_header_and_content() {
        let root = temp_dir();
        fs::write(root.join("main.rs"), "fn main() {}").unwrap();
        let output = read_file(&root, "main.rs", 100).unwrap();
        assert!(output.contains("--- main.rs ---"));
        assert!(output.contains("fn main() {}"));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn read_file_rejects_missing_file() {
        let root = temp_dir();
        assert!(read_file(&root, "nope.rs", 100).is_err());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn write_file_creates_parent_dirs() {
        let root = temp_dir();
        let result = write_file(&root, "src/deep/lib.rs", "pub fn f() {}").unwrap();
        assert!(result.contains("lib.rs"));
        assert!(root.join("src/deep/lib.rs").exists());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn grep_files_finds_matches_and_reports_missing() {
        let root = temp_dir();
        fs::write(root.join("a.txt"), "hello world\nfoo bar\n").unwrap();
        fs::write(root.join("b.rs"), "fn hello() {}\n").unwrap();

        let output = grep_files(&root, "hello", ".", 10).unwrap();
        assert!(output.contains("a.txt:1: hello world"));
        assert!(output.contains("b.rs:1: fn hello() {}"));

        let none = grep_files(&root, "zzz", ".", 10).unwrap();
        assert!(none.contains("(no matches)"));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn grep_files_rejects_invalid_regex() {
        let root = temp_dir();
        assert!(grep_files(&root, "(", ".", 10).is_err());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn grep_files_honors_max_results() {
        let root = temp_dir();
        let content = "match\n".repeat(100);
        fs::write(root.join("many.txt"), content).unwrap();
        let output = grep_files(&root, "match", ".", 3).unwrap();
        assert!(output.contains("truncated at 3 matches"));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn grep_files_zero_max_results_is_clamped_to_one() {
        let root = temp_dir();
        fs::write(root.join("a.txt"), "match\n").unwrap();
        let output = grep_files(&root, "match", ".", 0).unwrap();
        assert!(output.contains("a.txt:1: match"));
        assert!(output.contains("truncated at 1 match"));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn grep_files_skips_binary_files() {
        let root = temp_dir();
        fs::write(root.join("bin.dat"), b"\x00\x01\x02match").unwrap();
        let output = grep_files(&root, "match", ".", 10).unwrap();
        assert!(output.contains("(no matches)"));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn write_file_rejects_parent_dir_traversal() {
        let root = temp_dir();
        assert!(write_file(&root, "../evil.txt", "x").is_err());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn read_file_rejects_parent_dir_traversal() {
        let root = temp_dir();
        assert!(read_file(&root, "../etc/passwd", 100).is_err());
        fs::remove_dir_all(root).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn file_tools_reject_symlink_escape() {
        use std::os::unix::fs::symlink;

        let root = temp_dir();
        let outside = temp_dir();
        fs::write(outside.join("secret.txt"), "secret").unwrap();
        symlink(&outside, root.join("escape")).unwrap();

        assert!(read_file(&root, "escape/secret.txt", 100).is_err());
        assert!(list_files(&root, "escape", 5).is_err());
        assert!(grep_files(&root, "secret", "escape", 10).is_err());
        assert!(write_file(&root, "escape/new.txt", "x").is_err());

        fs::remove_dir_all(root).unwrap();
        fs::remove_dir_all(outside).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn write_file_rejects_symlink_file_escape() {
        use std::os::unix::fs::symlink;

        let root = temp_dir();
        let outside = temp_dir();
        fs::write(outside.join("secret.txt"), "original").unwrap();
        symlink(outside.join("secret.txt"), root.join("link.txt")).unwrap();

        let result = write_file(&root, "link.txt", "pwned");
        assert!(result.is_err());
        assert_eq!(
            fs::read_to_string(outside.join("secret.txt")).unwrap(),
            "original",
            "writing through a symlink must not touch the target"
        );

        fs::remove_dir_all(root).unwrap();
        fs::remove_dir_all(outside).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn write_file_rejects_dangling_symlink_file() {
        use std::os::unix::fs::symlink;

        let root = temp_dir();
        let outside = temp_dir();
        symlink(outside.join("new.txt"), root.join("dangling.txt")).unwrap();

        assert!(write_file(&root, "dangling.txt", "x").is_err());
        assert!(
            !outside.join("new.txt").exists(),
            "dangling symlink target must not be created outside the workspace"
        );

        fs::remove_dir_all(root).unwrap();
        fs::remove_dir_all(outside).unwrap();
    }

    #[test]
    fn relative_to_root_prefers_canonical_root() {
        let root = Path::new(".");
        let canonical_root = Path::new("/tmp/project");
        assert_eq!(
            relative_to_root(Path::new("/tmp/project/src/main.rs"), root, canonical_root),
            Path::new("src/main.rs")
        );
        assert_eq!(
            relative_to_root(Path::new("/tmp/project"), root, canonical_root),
            Path::new("")
        );
    }

    #[test]
    fn relative_to_root_falls_back_to_raw_root() {
        let root = Path::new(".");
        let canonical_root = Path::new("/tmp/project");
        // A non-canonical relative entry (rare) still resolves against `root`.
        assert_eq!(
            relative_to_root(Path::new("./src/main.rs"), root, canonical_root),
            Path::new("src/main.rs")
        );
    }

    #[test]
    fn grep_files_with_anchored_regex() {
        let root = temp_dir();
        fs::write(root.join("a.txt"), "hello world\ngoodbye world\nhello again\n").unwrap();
        let output = grep_files(&root, r"^hello", ".", 10).unwrap();
        assert!(output.contains("a.txt:1: hello world"));
        assert!(output.contains("a.txt:3: hello again"));
        assert!(!output.contains("goodbye"));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn grep_files_with_complex_regex_pattern() {
        let root = temp_dir();
        fs::write(
            root.join("code.rs"),
            "fn foo() {}\nfn bar(x: i32) -> bool {}\nlet y = 42;\nfn baz()\n",
        )
        .unwrap();
        let output = grep_files(&root, r"fn \w+\(", ".", 10).unwrap();
        assert!(output.contains("code.rs:1: fn foo() {}"));
        assert!(output.contains("fn bar(x: i32) -> bool {}"));
        assert!(output.contains("fn baz()"));
        assert!(!output.contains("let y = 42"));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn grep_files_max_results_one_returns_single_match() {
        let root = temp_dir();
        fs::write(
            root.join("lines.txt"),
            "aaa\nbbb\naaa\nccc\naaa\n",
        )
        .unwrap();
        let output = grep_files(&root, "aaa", ".", 1).unwrap();
        let match_lines: Vec<&str> = output
            .lines()
            .filter(|l| l.contains("lines.txt"))
            .collect();
        assert_eq!(match_lines.len(), 1);
        assert!(output.contains("truncated at 1 match"));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn read_file_detects_binary_via_read_error() {
        let root = temp_dir();
        // Write bytes that are valid UTF-8 but the file is large and has
        // non-text content. read_file uses read_to_string which will succeed
        // on valid UTF-8, so we test that max_chars truncation works.
        let long_text = "x".repeat(500);
        fs::write(root.join("big.txt"), &long_text).unwrap();
        let output = read_file(&root, "big.txt", 50).unwrap();
        assert!(output.contains("--- big.txt ---"));
        // Output should be truncated
        assert!(output.len() < 500);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn read_file_max_chars_truncates_content() {
        let root = temp_dir();
        let content = "line1\nline2\nline3\nline4\nline5\n";
        fs::write(root.join("data.txt"), content).unwrap();
        let output = read_file(&root, "data.txt", 10).unwrap();
        // The header is always present; content after it should be truncated
        assert!(output.contains("--- data.txt ---"));
        // The truncated output should contain the truncation marker
        assert!(output.contains("truncated"));
        // The output should not contain all 5 lines (would be present without truncation)
        assert!(!output.contains("line5"));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn write_file_creates_deeply_nested_parent_directories() {
        let root = temp_dir();
        let result = write_file(&root, "a/b/c/d/e.txt", "deep").unwrap();
        assert!(result.contains("e.txt"));
        assert!(root.join("a/b/c/d/e.txt").exists());
        assert_eq!(fs::read_to_string(root.join("a/b/c/d/e.txt")).unwrap(), "deep");
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn list_files_depth_one_shows_only_top_level() {
        let root = temp_dir();
        fs::create_dir_all(root.join("sub/nested")).unwrap();
        fs::write(root.join("top.txt"), "top").unwrap();
        fs::write(root.join("sub/mid.txt"), "mid").unwrap();
        fs::write(root.join("sub/nested/deep.txt"), "deep").unwrap();

        let output = list_files(&root, ".", 1).unwrap();
        assert!(output.contains("top.txt"));
        // depth=1 means only the root itself; no subdirectory contents
        assert!(!output.contains("sub/mid.txt"));
        assert!(!output.contains("deep.txt"));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn list_files_depth_two_shows_one_level_of_subdirectories() {
        let root = temp_dir();
        fs::create_dir_all(root.join("sub/nested")).unwrap();
        fs::write(root.join("top.txt"), "top").unwrap();
        fs::write(root.join("sub/mid.txt"), "mid").unwrap();
        fs::write(root.join("sub/nested/deep.txt"), "deep").unwrap();

        let output = list_files(&root, ".", 2).unwrap();
        assert!(output.contains("top.txt"));
        assert!(output.contains("sub/mid.txt"));
        assert!(!output.contains("deep.txt"));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn list_files_empty_directory_shows_no_files_message() {
        let root = temp_dir();
        let output = list_files(&root, ".", 5).unwrap();
        assert!(output.contains("(no files found)"));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn write_file_overwrites_existing_file() {
        let root = temp_dir();
        fs::write(root.join("file.txt"), "original").unwrap();
        let result = write_file(&root, "file.txt", "replaced").unwrap();
        assert!(result.contains("file.txt"));
        assert_eq!(fs::read_to_string(root.join("file.txt")).unwrap(), "replaced");
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn list_files_with_relative_root_returns_relative_paths() {
        // `cargo test` runs with the crate root as the working directory, so
        // `"."` is a valid relative workspace root. Regression: the walker uses
        // the canonical absolute path, which used to leak absolute paths into
        // the agent's file listing when the root was relative.
        let output = list_files(Path::new("."), "src", 1).unwrap();
        let first = output.lines().next().unwrap_or_default();
        assert!(
            !first.starts_with('/'),
            "expected relative path, got: {first}"
        );
        assert!(
            first.starts_with("src/"),
            "expected src/ prefix, got: {first}"
        );
    }
}
