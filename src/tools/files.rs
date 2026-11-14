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
use crate::constants::tools::{
    BINARY_SNIFF_BYTES, GREP_MAX_DEPTH, MAX_LISTED_FILES, SKIPPED_DIRS, SKIPPED_ROOT_DIRS,
};
use anyhow::{Context, Result, bail};
use regex::Regex;
use std::fs;
use std::path::{Path, PathBuf};
use walkdir::{DirEntry, WalkDir};

/// Whether a recursive walk skips this directory: a cache or dependency tree
/// anywhere, or a build or data directory directly under the workspace root.
fn is_skipped_dir(directory: &Path, workspace: &Path) -> bool {
    let name = directory
        .file_name()
        .map(|name| name.to_string_lossy())
        .unwrap_or_default();

    SKIPPED_DIRS.contains(&name.as_ref()) || is_skipped_root_dir(directory, workspace, &name)
}

/// Whether `directory` sits directly under the workspace root and is named
/// like build output or data.
fn is_skipped_root_dir(directory: &Path, workspace: &Path, name: &str) -> bool {
    directory.parent() == Some(workspace) && SKIPPED_ROOT_DIRS.contains(&name)
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

/// Build a recursive directory walker that skips the directories
/// [`is_skipped_dir`] names. The directory the walk starts in is never
/// skipped, so asking for `data` explicitly still lists it.
///
/// # Arguments
/// - `start`: canonical directory to walk.
/// - `workspace`: canonical workspace root.
/// - `max_depth`: maximum recursion depth.
///
/// # Returns
/// An iterator over directory entries.
fn walker<'a>(
    start: &'a Path,
    workspace: &'a Path,
    max_depth: usize,
) -> impl Iterator<Item = Result<DirEntry, walkdir::Error>> + 'a {
    WalkDir::new(start)
        .max_depth(max_depth)
        .follow_links(false)
        .into_iter()
        .filter_entry(move |entry| is_walked(entry, workspace))
}

/// Whether the walk enters or yields `entry`: everything except a skipped
/// directory below the start of the walk.
fn is_walked(entry: &DirEntry, workspace: &Path) -> bool {
    !is_nested_directory(entry) || !is_skipped_dir(entry.path(), workspace)
}

/// Whether `entry` is a directory below the start of the walk.
fn is_nested_directory(entry: &DirEntry) -> bool {
    entry.depth() > 0 && entry.file_type().is_dir()
}

/// Whether a walk item is a file, or an error that must reach the caller.
fn is_file_or_error(entry: &Result<DirEntry>) -> bool {
    entry
        .as_ref()
        .map_or(true, |entry| entry.file_type().is_file())
}

/// The model's `path` resolved inside the workspace in canonical form, or an
/// error when it is not a directory or escapes the workspace.
fn canonical_directory(root: &Path, path: &str) -> Result<PathBuf> {
    let directory = resolve_in_root(root, path)?;
    let directory = directory_only(directory)?;

    ensure_canonical_within_root(root, &directory)
}

/// `directory` unchanged when it is a directory, else an error.
fn directory_only(directory: PathBuf) -> Result<PathBuf> {
    if !directory.is_dir() {
        bail!("not a directory: {}", directory.display());
    }

    Ok(directory)
}

/// The model's `path` resolved inside the workspace in canonical form, or an
/// error when it is not a file or escapes the workspace.
fn canonical_file(root: &Path, path: &str) -> Result<PathBuf> {
    let file = resolve_in_root(root, path)?;
    let file = file_only(file)?;

    ensure_canonical_within_root(root, &file)
}

/// `file` unchanged when it is a file, else an error.
fn file_only(file: PathBuf) -> Result<PathBuf> {
    if !file.is_file() {
        bail!("not a file: {}", file.display());
    }

    Ok(file)
}

/// `file` unchanged unless it is a symlink. `action` names the refused
/// operation in the error, such as `write` or `edit`.
fn refuse_symlink(file: PathBuf, action: &str) -> Result<PathBuf> {
    let is_symlink =
        fs::symlink_metadata(&file).is_ok_and(|metadata| metadata.file_type().is_symlink());

    if is_symlink {
        bail!("refusing to {action} through symlink: {}", file.display());
    }

    Ok(file)
}

/// The note after a listing or a search: what was not found when `count` is
/// zero, a truncation note when `count` reached `limit`, else nothing.
fn closing_note(count: usize, limit: usize, nothing_found: &str, unit: &str) -> String {
    match (count, count == limit) {
        (0, _) => format!("({nothing_found})\n"),
        (_, true) => format!("... (truncated at {limit} {unit})\n"),
        (_, false) => String::new(),
    }
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
    let directory = canonical_directory(root, path)?;
    let canonical_root = canonical_root(root)?;

    // The walk is lazy, so it stops at the first error or at the listing limit.
    let files: Vec<DirEntry> = walker(&directory, &canonical_root, max_depth)
        .map(|entry| entry.context("walkdir error"))
        .filter(is_file_or_error)
        .take(MAX_LISTED_FILES)
        .collect::<Result<_>>()?;

    Ok(truncate(listing_text(&files, root, &canonical_root)))
}

/// One line per file with its relative path and size, then the closing note.
fn listing_text(files: &[DirEntry], root: &Path, canonical_root: &Path) -> String {
    let mut listing: String = files
        .iter()
        .map(|file| listing_line(file, root, canonical_root))
        .collect();

    listing.push_str(&closing_note(
        files.len(),
        MAX_LISTED_FILES,
        "no files found",
        "files",
    ));

    listing
}

/// `path (size bytes)` for one listed file; the size is 0 when it cannot be read.
fn listing_line(file: &DirEntry, root: &Path, canonical_root: &Path) -> String {
    let relative = relative_to_root(file.path(), root, canonical_root).to_string_lossy();
    let size = file.metadata().map(|metadata| metadata.len()).unwrap_or(0);

    format!("{relative} ({size} bytes)\n")
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
    let file = canonical_file(root, path)?;
    let canonical_root = canonical_root(root)?;
    let content = fs::read_to_string(&file).context("failed to read file (may be binary)")?;

    let shown_path = relative_to_root(&file, root, &canonical_root).display();
    let truncated = truncate_to(content, max_chars);

    Ok(format!("--- {shown_path} ---\n{truncated}"))
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
    let file = write_target(root, file)?;

    fs::write(&file, content).with_context(|| format!("failed to write {}", file.display()))?;

    Ok(format!(
        "✅ wrote {} ({} bytes)",
        file.display(),
        content.len()
    ))
}

/// The path a write goes to: inside the canonical parent directory, which is
/// created first. A path with no parent, such as `/`, is returned unchanged.
fn write_target(root: &Path, file: PathBuf) -> Result<PathBuf> {
    match file.parent() {
        Some(parent) => target_in_parent(root, parent, &file),
        None => Ok(file),
    }
}

/// `file`'s name joined to the canonical form of `parent`, after creating
/// `parent` and checking that it lies inside the workspace.
fn target_in_parent(root: &Path, parent: &Path, file: &Path) -> Result<PathBuf> {
    create_parent_inside_root(root, parent, file)?;
    let canonical_parent = ensure_canonical_within_root(root, parent)?;

    // Parent-directory symlinks are already rejected by canonicalization,
    // but a symlink in the final component would still be followed by
    // `fs::write`. Refuse to write through any existing symlink so the
    // agent can never modify a file outside the workspace.
    let target = canonical_parent.join(file.file_name().unwrap_or_default());

    refuse_symlink(target, "write")
}

/// Create `parent` and its missing ancestors for writing `file`, after
/// checking that the part of `parent` that already exists lies inside the
/// workspace. A symlink that leads outside is refused before any folder is
/// made behind it.
fn create_parent_inside_root(root: &Path, parent: &Path, file: &Path) -> Result<()> {
    ensure_canonical_within_root(root, deepest_existing_ancestor(parent))?;

    fs::create_dir_all(parent)
        .with_context(|| format!("failed to create parent dirs for {}", file.display()))
}

/// The deepest of `path` and its ancestors that exists, a symlink counted
/// as existing even when its target is missing; `path` itself when none do.
fn deepest_existing_ancestor(path: &Path) -> &Path {
    path.ancestors()
        .find(|ancestor| fs::symlink_metadata(ancestor).is_ok())
        .unwrap_or(path)
}

/// Replace a literal string inside an existing file.
///
/// A coding agent makes most edits by changing a few lines, not by rewriting a
/// whole file, so this is the tool models reach for most. The match is literal
/// (no regex) and must be unique unless `replace_all` is set, which prevents an
/// ambiguous `old_string` from silently editing the wrong place.
///
/// # Arguments
/// - `root`: workspace root used for path safety.
/// - `path`: relative file path.
/// - `old_string`: exact text to find.
/// - `new_string`: replacement text.
/// - `replace_all`: replace every occurrence instead of requiring a unique match.
///
/// # Returns
/// A confirmation string with how many replacements were made.
pub(crate) fn edit_file(
    root: &Path,
    path: &str,
    old_string: &str,
    new_string: &str,
    replace_all: bool,
) -> Result<String> {
    refuse_empty_old_string(old_string)?;
    let file = editable_file(root, path)?;
    let content = fs::read_to_string(&file).context("failed to read file (may be binary)")?;
    let (old_string, new_string) = with_file_line_endings(&content, old_string, new_string);
    let replacements = replacement_count(&content, &old_string, path, replace_all)?;

    let updated = replaced_text(&content, &old_string, &new_string, replace_all);
    fs::write(&file, updated).with_context(|| format!("failed to write {}", file.display()))?;

    Ok(edit_summary(path, replacements))
}

/// The canonical path of the file an edit changes, or an error when it is not
/// a file inside the workspace or is a symlink.
fn editable_file(root: &Path, path: &str) -> Result<PathBuf> {
    let file = canonical_file(root, path)?;

    refuse_symlink(file, "edit")
}

/// An error when `old_string` is empty, which would match everywhere.
fn refuse_empty_old_string(old_string: &str) -> Result<()> {
    if old_string.is_empty() {
        bail!("old_string must not be empty; use write_file to create or rewrite a file");
    }

    Ok(())
}

/// `old_string` and `new_string` written with CRLF line endings when
/// `content` uses CRLF and does not contain `old_string` as given, else both
/// unchanged. Models write `\n`, so without this a multi-line edit of a
/// CRLF file is never found.
fn with_file_line_endings(content: &str, old_string: &str, new_string: &str) -> (String, String) {
    if !needs_crlf(content, old_string) {
        return (old_string.to_string(), new_string.to_string());
    }

    (with_crlf(old_string), with_crlf(new_string))
}

/// Whether an edit tries CRLF line endings: `content` uses CRLF and does not
/// contain `old_string` as given.
fn needs_crlf(content: &str, old_string: &str) -> bool {
    !content.contains(old_string) && content.contains("\r\n")
}

/// `text` with every line ending written as CRLF.
fn with_crlf(text: &str) -> String {
    text.replace("\r\n", "\n").replace('\n', "\r\n")
}

/// How many places an edit replaces, or an error when `old_string` is
/// missing, or appears more than once without `replace_all`.
fn replacement_count(
    content: &str,
    old_string: &str,
    path: &str,
    replace_all: bool,
) -> Result<usize> {
    let occurrences = overlapping_count(content, old_string);

    match (occurrences, replace_all) {
        (0, _) => {
            bail!("old_string was not found in {path}; re-read the file and copy the exact text")
        }
        (1, _) => Ok(1),
        // `str::replace` replaces only matches that do not overlap.
        (_, true) => Ok(content.matches(old_string).count()),
        (_, false) => bail!(
            "old_string appears {occurrences} times in {path}; add surrounding context or pass replace_all=true"
        ),
    }
}

/// How many places in `content` start a match of `old_string`, counting
/// matches that overlap. `"}\n}\n"` occurs twice in `"}\n}\n}\n"`, so an edit
/// of it is ambiguous, although `str::matches` finds it only once.
fn overlapping_count(content: &str, old_string: &str) -> usize {
    content
        .char_indices()
        .filter(|(start, _character)| content[*start..].starts_with(old_string))
        .count()
}

/// `content` with every occurrence of `old_string` replaced when
/// `replace_all` is set, else only the first.
fn replaced_text(content: &str, old_string: &str, new_string: &str, replace_all: bool) -> String {
    if replace_all {
        return content.replace(old_string, new_string);
    }

    content.replacen(old_string, new_string, 1)
}

/// The confirmation line of an edit, such as `edited a.rs (2 replacements)`.
fn edit_summary(path: &str, replacements: usize) -> String {
    let plural = if replacements == 1 { "" } else { "s" };

    format!("✅ edited {path} ({replacements} replacement{plural})")
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
    let directory = canonical_directory(root, path)?;
    let canonical_root = canonical_root(root)?;
    let regex = Regex::new(pattern).context("invalid regex pattern")?;
    let max_results = max_results.max(1);

    // The walk is lazy, so no file after the last needed match is read.
    let matches: Vec<String> = walker(&directory, &canonical_root, GREP_MAX_DEPTH)
        .map(|entry| entry.context("walkdir error"))
        .filter(is_file_or_error)
        .flat_map(|entry| entry_matches(entry, &regex, root, &canonical_root))
        .take(max_results)
        .collect::<Result<_>>()?;

    let mut output = matches.concat();
    output.push_str(&closing_note(
        matches.len(),
        max_results,
        "no matches",
        "matches",
    ));

    Ok(truncate(output))
}

/// The matching lines of one walked file, or the walk error in its place.
fn entry_matches(
    entry: Result<DirEntry>,
    regex: &Regex,
    root: &Path,
    canonical_root: &Path,
) -> Vec<Result<String>> {
    match entry {
        Ok(file) => file_matches(&file, regex, root, canonical_root),
        Err(error) => vec![Err(error)],
    }
}

/// `path:line: text` for every line of `file` that matches `regex`. Binary
/// files and files that are not UTF-8 text have no matches.
fn file_matches(
    file: &DirEntry,
    regex: &Regex,
    root: &Path,
    canonical_root: &Path,
) -> Vec<Result<String>> {
    let Some(content) = text_content(file.path()) else {
        return Vec::new();
    };

    let relative = relative_to_root(file.path(), root, canonical_root).to_string_lossy();

    content
        .lines()
        .enumerate()
        .filter(|(_line_index, line)| regex.is_match(line))
        .map(|(line_index, line)| Ok(format!("{relative}:{}: {}\n", line_index + 1, line.trim())))
        .collect()
}

/// The text of the file at `path`, or `None` when it cannot be read, holds a
/// NUL byte near its start (a binary file), or is not UTF-8.
fn text_content(path: &Path) -> Option<String> {
    fs::read(path)
        .ok()
        .filter(|bytes| !is_binary(bytes))
        .and_then(|bytes| String::from_utf8(bytes).ok())
}

/// Whether the first [`BINARY_SNIFF_BYTES`] bytes hold a NUL byte.
fn is_binary(bytes: &[u8]) -> bool {
    bytes.iter().take(BINARY_SNIFF_BYTES).any(|&byte| byte == 0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;

    /// A new empty directory to use as the workspace of one test.
    fn temp_dir() -> PathBuf {
        crate::test_support::unique_temp_dir("openbatrangs-tools-test")
    }

    #[test]
    fn list_files_returns_relative_paths_and_sizes() -> Result<()> {
        let root = temp_dir();
        fs::write(root.join("a.txt"), "hello")?;
        fs::create_dir_all(root.join("sub"))?;
        fs::write(root.join("sub/b.txt"), "world")?;

        let output = list_files(&root, ".", 5)?;

        assert!(output.contains("a.txt (5 bytes)"));
        assert!(output.contains("sub/b.txt (5 bytes)"));

        fs::remove_dir_all(root)?;
        Ok(())
    }

    #[test]
    fn list_files_rejects_missing_directory() -> Result<()> {
        let root = temp_dir();

        assert!(list_files(&root, "missing", 5).is_err());

        fs::remove_dir_all(root)?;
        Ok(())
    }

    #[test]
    fn source_folders_named_like_build_output_are_listed_below_the_root() -> Result<()> {
        let root = temp_dir();
        fs::create_dir_all(root.join("src/models"))?;
        fs::create_dir_all(root.join("src/data"))?;
        fs::create_dir_all(root.join("data"))?;
        fs::write(root.join("src/models/mod.rs"), "pub struct Model;\n")?;
        fs::write(root.join("src/data/mod.rs"), "pub struct Row;\n")?;
        fs::write(root.join("data/rows.csv"), "1,2\n")?;

        let listed = list_files(&root, ".", 5)?;
        let only_data = list_files(&root, "data", 5)?;
        let found = grep_files(&root, "pub struct", ".", 10)?;

        assert!(listed.contains("src/models/mod.rs"), "{listed}");
        assert!(listed.contains("src/data/mod.rs"), "{listed}");
        assert!(!listed.contains("data/rows.csv"), "{listed}");
        assert!(only_data.contains("data/rows.csv"), "{only_data}");
        assert!(
            found.contains("src/models/mod.rs:1: pub struct Model;"),
            "{found}"
        );

        fs::remove_dir_all(root)?;
        Ok(())
    }

    #[test]
    fn list_files_skips_heavy_dirs() -> Result<()> {
        let root = temp_dir();
        fs::create_dir_all(root.join("target"))?;
        fs::create_dir_all(root.join(".agent/cache"))?;
        fs::create_dir_all(root.join("data/text"))?;
        fs::write(root.join("target/ignored.txt"), "x")?;
        fs::write(root.join(".agent/cache/secret.txt"), "x")?;
        fs::write(root.join("data/text/book.txt"), "x")?;
        fs::write(root.join("keep.txt"), "y")?;

        let output = list_files(&root, ".", 5)?;

        assert!(output.contains("keep.txt"));
        assert!(!output.contains("target/ignored.txt"));
        assert!(!output.contains(".agent/cache/secret.txt"));
        assert!(!output.contains("data/text/book.txt"));

        fs::remove_dir_all(root)?;
        Ok(())
    }

    #[test]
    fn list_files_stops_at_the_listing_limit() -> Result<()> {
        let root = temp_dir();

        let write_file_number = |index| fs::write(root.join(format!("file-{index}.txt")), "x");
        (0..=MAX_LISTED_FILES).try_for_each(write_file_number)?;

        let output = list_files(&root, ".", 1)?;
        let listed_count = output
            .lines()
            .filter(|line| line.ends_with("bytes)"))
            .count();

        assert_eq!(listed_count, MAX_LISTED_FILES);
        assert!(output.contains(&format!("truncated at {MAX_LISTED_FILES} files")));

        fs::remove_dir_all(root)?;
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn an_unreadable_directory_fails_the_walk() -> Result<()> {
        use std::os::unix::fs::PermissionsExt;

        let root = temp_dir();
        let locked = root.join("locked");
        fs::create_dir_all(&locked)?;
        fs::set_permissions(&locked, fs::Permissions::from_mode(0o000))?;

        // root can read any directory, so the walk only fails for other users.
        let lock_holds = fs::read_dir(&locked).is_err();
        let listed = list_files(&root, ".", 5);
        let found = grep_files(&root, "x", ".", 10);

        fs::set_permissions(&locked, fs::Permissions::from_mode(0o755))?;
        fs::remove_dir_all(&root)?;

        assert_eq!(listed.is_err(), lock_holds);
        assert_eq!(found.is_err(), lock_holds);
        Ok(())
    }

    #[test]
    fn read_file_returns_header_and_content() -> Result<()> {
        let root = temp_dir();
        fs::write(root.join("main.rs"), "fn main() {}")?;

        let output = read_file(&root, "main.rs", 100)?;

        assert!(output.contains("--- main.rs ---"));
        assert!(output.contains("fn main() {}"));

        fs::remove_dir_all(root)?;
        Ok(())
    }

    #[test]
    fn read_file_rejects_missing_file() -> Result<()> {
        let root = temp_dir();

        assert!(read_file(&root, "nope.rs", 100).is_err());

        fs::remove_dir_all(root)?;
        Ok(())
    }

    #[test]
    fn write_file_creates_parent_dirs() -> Result<()> {
        let root = temp_dir();

        let confirmation = write_file(&root, "src/deep/lib.rs", "pub fn f() {}")?;

        assert!(confirmation.contains("lib.rs"));
        assert!(root.join("src/deep/lib.rs").exists());

        fs::remove_dir_all(root)?;
        Ok(())
    }

    #[test]
    fn write_file_fails_when_a_parent_is_a_file() -> Result<()> {
        let root = temp_dir();
        fs::write(root.join("a.txt"), "x")?;

        let written = write_file(&root, "a.txt/b.txt", "y");

        assert!(format!("{written:?}").contains("failed to create parent dirs"));

        fs::remove_dir_all(root)?;
        Ok(())
    }

    #[test]
    fn write_file_to_a_path_with_no_parent_fails_to_write() {
        let written = write_file(Path::new("/"), "", "x");

        assert!(written.is_err());
    }

    #[test]
    fn edit_file_replaces_a_unique_match() -> Result<()> {
        let root = temp_dir();
        fs::write(root.join("lib.rs"), "fn old() {}\n")?;

        let confirmation = edit_file(&root, "lib.rs", "old", "new", false)?;

        assert_eq!(confirmation, "✅ edited lib.rs (1 replacement)");
        assert_eq!(fs::read_to_string(root.join("lib.rs"))?, "fn new() {}\n");

        fs::remove_dir_all(root)?;
        Ok(())
    }

    #[test]
    fn edit_file_replaces_every_match_with_replace_all() -> Result<()> {
        let root = temp_dir();
        fs::write(root.join("lib.rs"), "a a a")?;

        let confirmation = edit_file(&root, "lib.rs", "a", "b", true)?;

        assert_eq!(confirmation, "✅ edited lib.rs (3 replacements)");
        assert_eq!(fs::read_to_string(root.join("lib.rs"))?, "b b b");

        fs::remove_dir_all(root)?;
        Ok(())
    }

    #[test]
    fn edit_file_refuses_missing_ambiguous_and_empty_matches() -> Result<()> {
        let root = temp_dir();
        fs::write(root.join("lib.rs"), "a a")?;

        let missing = edit_file(&root, "lib.rs", "zzz", "b", false);
        let ambiguous = edit_file(&root, "lib.rs", "a", "b", false);
        let empty = edit_file(&root, "lib.rs", "", "b", false);
        let not_a_file = edit_file(&root, "nope.rs", "a", "b", false);

        assert!(format!("{missing:?}").contains("was not found"));
        assert!(format!("{ambiguous:?}").contains("appears 2 times"));
        assert!(format!("{empty:?}").contains("must not be empty"));
        assert!(format!("{not_a_file:?}").contains("not a file"));
        assert_eq!(fs::read_to_string(root.join("lib.rs"))?, "a a");

        fs::remove_dir_all(root)?;
        Ok(())
    }

    #[test]
    fn edit_file_counts_overlapping_matches() -> Result<()> {
        let root = temp_dir();
        let content = "fn f() {\n    }\n}\n}\n";
        fs::write(root.join("lib.rs"), content)?;

        let ambiguous = edit_file(&root, "lib.rs", "}\n}\n", "", false);

        assert!(format!("{ambiguous:?}").contains("appears 2 times"));
        assert_eq!(fs::read_to_string(root.join("lib.rs"))?, content);

        fs::remove_dir_all(root)?;
        Ok(())
    }

    #[test]
    fn replace_all_reports_the_replacements_it_makes() -> Result<()> {
        let root = temp_dir();
        fs::write(root.join("lib.rs"), "aaa")?;

        let confirmation = edit_file(&root, "lib.rs", "aa", "b", true)?;

        assert_eq!(confirmation, "✅ edited lib.rs (1 replacement)");
        assert_eq!(fs::read_to_string(root.join("lib.rs"))?, "ba");

        fs::remove_dir_all(root)?;
        Ok(())
    }

    #[test]
    fn edit_file_edits_a_crlf_file_with_a_multi_line_old_string() -> Result<()> {
        let root = temp_dir();
        fs::write(root.join("lib.rs"), "fn a() {\r\n    old();\r\n}\r\n")?;

        edit_file(
            &root,
            "lib.rs",
            "{\n    old();\n}",
            "{\n    new();\n}",
            false,
        )?;

        assert_eq!(
            fs::read_to_string(root.join("lib.rs"))?,
            "fn a() {\r\n    new();\r\n}\r\n"
        );

        fs::remove_dir_all(root)?;
        Ok(())
    }

    #[test]
    fn edit_file_keeps_an_exact_match_in_a_mixed_line_ending_file() -> Result<()> {
        let root = temp_dir();
        fs::write(root.join("lib.rs"), "a\r\nold\nline\n")?;

        edit_file(&root, "lib.rs", "old\nline", "new\nline", false)?;

        assert_eq!(fs::read_to_string(root.join("lib.rs"))?, "a\r\nnew\nline\n");

        fs::remove_dir_all(root)?;
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn edit_file_reports_a_file_it_cannot_write() -> Result<()> {
        use std::os::unix::fs::PermissionsExt;

        let root = temp_dir();
        let file = root.join("locked.rs");
        fs::write(&file, "old")?;
        fs::set_permissions(&file, fs::Permissions::from_mode(0o444))?;

        // root can write any file, so the edit only fails for other users.
        let lock_holds = fs::OpenOptions::new().write(true).open(&file).is_err();
        let edited = edit_file(&root, "locked.rs", "old", "new", false);

        fs::remove_dir_all(&root)?;

        assert_eq!(
            format!("{edited:?}").contains("failed to write"),
            lock_holds
        );
        Ok(())
    }

    #[test]
    fn grep_files_finds_matches_and_reports_missing() -> Result<()> {
        let root = temp_dir();
        fs::write(root.join("a.txt"), "hello world\nfoo bar\n")?;
        fs::write(root.join("b.rs"), "fn hello() {}\n")?;

        let output = grep_files(&root, "hello", ".", 10)?;
        let none = grep_files(&root, "zzz", ".", 10)?;

        assert!(output.contains("a.txt:1: hello world"));
        assert!(output.contains("b.rs:1: fn hello() {}"));
        assert!(none.contains("(no matches)"));

        fs::remove_dir_all(root)?;
        Ok(())
    }

    #[test]
    fn grep_files_rejects_invalid_regex() -> Result<()> {
        let root = temp_dir();

        assert!(grep_files(&root, "(", ".", 10).is_err());

        fs::remove_dir_all(root)?;
        Ok(())
    }

    #[test]
    fn grep_files_honors_max_results() -> Result<()> {
        let root = temp_dir();
        fs::write(root.join("many.txt"), "match\n".repeat(100))?;

        let output = grep_files(&root, "match", ".", 3)?;

        assert!(output.contains("truncated at 3 matches"));

        fs::remove_dir_all(root)?;
        Ok(())
    }

    #[test]
    fn grep_files_zero_max_results_is_clamped_to_one() -> Result<()> {
        let root = temp_dir();
        fs::write(root.join("a.txt"), "match\n")?;

        let output = grep_files(&root, "match", ".", 0)?;

        assert!(output.contains("a.txt:1: match"));
        assert!(output.contains("truncated at 1 match"));

        fs::remove_dir_all(root)?;
        Ok(())
    }

    #[test]
    fn grep_files_skips_binary_and_non_utf8_files() -> Result<()> {
        let root = temp_dir();
        fs::write(root.join("bin.dat"), b"\x00\x01\x02match")?;
        fs::write(root.join("latin1.txt"), b"match \xe9t\xe9")?;

        let output = grep_files(&root, "match", ".", 10)?;

        assert!(output.contains("(no matches)"));

        fs::remove_dir_all(root)?;
        Ok(())
    }

    #[test]
    fn write_file_rejects_parent_dir_traversal() -> Result<()> {
        let root = temp_dir();

        assert!(write_file(&root, "../evil.txt", "x").is_err());

        fs::remove_dir_all(root)?;
        Ok(())
    }

    #[test]
    fn read_file_rejects_parent_dir_traversal() -> Result<()> {
        let root = temp_dir();

        assert!(read_file(&root, "../etc/passwd", 100).is_err());

        fs::remove_dir_all(root)?;
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn file_tools_reject_symlink_escape() -> Result<()> {
        use std::os::unix::fs::symlink;

        let root = temp_dir();
        let outside = temp_dir();
        fs::write(outside.join("secret.txt"), "secret")?;
        symlink(&outside, root.join("escape"))?;

        assert!(read_file(&root, "escape/secret.txt", 100).is_err());
        assert!(list_files(&root, "escape", 5).is_err());
        assert!(grep_files(&root, "secret", "escape", 10).is_err());
        assert!(write_file(&root, "escape/new.txt", "x").is_err());
        assert!(edit_file(&root, "escape/secret.txt", "secret", "x", false).is_err());

        fs::remove_dir_all(root)?;
        fs::remove_dir_all(outside)?;
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn a_refused_write_creates_no_folders_outside_the_workspace() -> Result<()> {
        use std::os::unix::fs::symlink;

        let root = temp_dir();
        let outside = temp_dir();
        symlink(&outside, root.join("link"))?;

        let written = write_file(&root, "link/made/by/agent/x.txt", "x");

        assert!(format!("{written:?}").contains("escapes the workspace"));
        assert!(!outside.join("made").exists());

        fs::remove_dir_all(root)?;
        fs::remove_dir_all(outside)?;
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn write_file_rejects_symlink_file_escape() -> Result<()> {
        use std::os::unix::fs::symlink;

        let root = temp_dir();
        let outside = temp_dir();
        fs::write(outside.join("secret.txt"), "original")?;
        symlink(outside.join("secret.txt"), root.join("link.txt"))?;

        let written = write_file(&root, "link.txt", "pwned");

        assert!(written.is_err());
        assert_eq!(
            fs::read_to_string(outside.join("secret.txt"))?,
            "original",
            "writing through a symlink must not touch the target"
        );

        fs::remove_dir_all(root)?;
        fs::remove_dir_all(outside)?;
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn write_file_rejects_dangling_symlink_file() -> Result<()> {
        use std::os::unix::fs::symlink;

        let root = temp_dir();
        let outside = temp_dir();
        symlink(outside.join("new.txt"), root.join("dangling.txt"))?;

        assert!(write_file(&root, "dangling.txt", "x").is_err());
        assert!(
            !outside.join("new.txt").exists(),
            "dangling symlink target must not be created outside the workspace"
        );

        fs::remove_dir_all(root)?;
        fs::remove_dir_all(outside)?;
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn refuse_symlink_names_the_refused_action() -> Result<()> {
        use std::os::unix::fs::symlink;

        let root = temp_dir();
        fs::write(root.join("real.txt"), "x")?;
        symlink(root.join("real.txt"), root.join("link.txt"))?;

        let refused = refuse_symlink(root.join("link.txt"), "edit");
        let allowed = refuse_symlink(root.join("real.txt"), "edit")?;

        assert!(format!("{refused:?}").contains("refusing to edit through symlink"));
        assert_eq!(allowed, root.join("real.txt"));

        fs::remove_dir_all(root)?;
        Ok(())
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
    fn grep_files_with_anchored_regex() -> Result<()> {
        let root = temp_dir();
        let content = "hello world\ngoodbye world\nhello again\n";
        fs::write(root.join("a.txt"), content)?;

        let output = grep_files(&root, r"^hello", ".", 10)?;

        assert!(output.contains("a.txt:1: hello world"));
        assert!(output.contains("a.txt:3: hello again"));
        assert!(!output.contains("goodbye"));

        fs::remove_dir_all(root)?;
        Ok(())
    }

    #[test]
    fn grep_files_with_complex_regex_pattern() -> Result<()> {
        let root = temp_dir();
        let code = "fn foo() {}\nfn bar(x: i32) -> bool {}\nlet y = 42;\nfn baz()\n";
        fs::write(root.join("code.rs"), code)?;

        let output = grep_files(&root, r"fn \w+\(", ".", 10)?;

        assert!(output.contains("code.rs:1: fn foo() {}"));
        assert!(output.contains("fn bar(x: i32) -> bool {}"));
        assert!(output.contains("fn baz()"));
        assert!(!output.contains("let y = 42"));

        fs::remove_dir_all(root)?;
        Ok(())
    }

    #[test]
    fn grep_files_max_results_one_returns_single_match() -> Result<()> {
        let root = temp_dir();
        fs::write(root.join("lines.txt"), "aaa\nbbb\naaa\nccc\naaa\n")?;

        let output = grep_files(&root, "aaa", ".", 1)?;
        let match_lines = output
            .lines()
            .filter(|line| line.contains("lines.txt"))
            .count();

        assert_eq!(match_lines, 1);
        assert!(output.contains("truncated at 1 match"));

        fs::remove_dir_all(root)?;
        Ok(())
    }

    #[test]
    fn read_file_detects_binary_via_read_error() -> Result<()> {
        let root = temp_dir();
        // read_file uses read_to_string, which succeeds on valid UTF-8, so
        // this checks that max_chars truncation works on a long file.
        fs::write(root.join("big.txt"), "x".repeat(500))?;

        let output = read_file(&root, "big.txt", 50)?;

        assert!(output.contains("--- big.txt ---"));
        assert!(output.len() < 500);

        fs::remove_dir_all(root)?;
        Ok(())
    }

    #[test]
    fn read_file_max_chars_truncates_content() -> Result<()> {
        let root = temp_dir();
        fs::write(root.join("data.txt"), "line1\nline2\nline3\nline4\nline5\n")?;

        let output = read_file(&root, "data.txt", 10)?;

        // The header is always present; the content after it is truncated.
        assert!(output.contains("--- data.txt ---"));
        assert!(output.contains("truncated"));
        assert!(!output.contains("line5"));

        fs::remove_dir_all(root)?;
        Ok(())
    }

    #[test]
    fn write_file_creates_deeply_nested_parent_directories() -> Result<()> {
        let root = temp_dir();

        let confirmation = write_file(&root, "a/b/c/d/e.txt", "deep")?;

        assert!(confirmation.contains("e.txt"));
        assert_eq!(fs::read_to_string(root.join("a/b/c/d/e.txt"))?, "deep");

        fs::remove_dir_all(root)?;
        Ok(())
    }

    #[test]
    fn list_files_depth_one_shows_only_top_level() -> Result<()> {
        let root = temp_dir();
        fs::create_dir_all(root.join("sub/nested"))?;
        fs::write(root.join("top.txt"), "top")?;
        fs::write(root.join("sub/mid.txt"), "mid")?;
        fs::write(root.join("sub/nested/deep.txt"), "deep")?;

        let output = list_files(&root, ".", 1)?;

        // depth=1 means only the root itself; no subdirectory contents
        assert!(output.contains("top.txt"));
        assert!(!output.contains("sub/mid.txt"));
        assert!(!output.contains("deep.txt"));

        fs::remove_dir_all(root)?;
        Ok(())
    }

    #[test]
    fn list_files_depth_two_shows_one_level_of_subdirectories() -> Result<()> {
        let root = temp_dir();
        fs::create_dir_all(root.join("sub/nested"))?;
        fs::write(root.join("top.txt"), "top")?;
        fs::write(root.join("sub/mid.txt"), "mid")?;
        fs::write(root.join("sub/nested/deep.txt"), "deep")?;

        let output = list_files(&root, ".", 2)?;

        assert!(output.contains("top.txt"));
        assert!(output.contains("sub/mid.txt"));
        assert!(!output.contains("deep.txt"));

        fs::remove_dir_all(root)?;
        Ok(())
    }

    #[test]
    fn list_files_empty_directory_shows_no_files_message() -> Result<()> {
        let root = temp_dir();

        let output = list_files(&root, ".", 5)?;

        assert!(output.contains("(no files found)"));

        fs::remove_dir_all(root)?;
        Ok(())
    }

    #[test]
    fn write_file_overwrites_existing_file() -> Result<()> {
        let root = temp_dir();
        fs::write(root.join("file.txt"), "original")?;

        let confirmation = write_file(&root, "file.txt", "replaced")?;

        assert!(confirmation.contains("file.txt"));
        assert_eq!(fs::read_to_string(root.join("file.txt"))?, "replaced");

        fs::remove_dir_all(root)?;
        Ok(())
    }

    #[test]
    fn list_files_with_relative_root_returns_relative_paths() -> Result<()> {
        // `cargo test` runs with the crate root as the working directory, so
        // `"."` is a valid relative workspace root. Regression: the walker uses
        // the canonical absolute path, which used to leak absolute paths into
        // the agent's file listing when the root was relative.
        let output = list_files(Path::new("."), "src", 1)?;
        let first = output.lines().next().unwrap_or_default();

        assert!(
            !first.starts_with('/'),
            "expected relative path, got: {first}"
        );
        assert!(
            first.starts_with("src/"),
            "expected src/ prefix, got: {first}"
        );

        Ok(())
    }
}
