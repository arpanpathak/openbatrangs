//! # Path-safety helpers for agent-supplied paths
//!
//! The model can ask the agent to read, write, list, or grep files. Because
//! the model is an untrusted input, every path must be validated before it
//! touches the filesystem: paths outside the workspace and `..` traversal are
//! rejected, and symlinks are canonicalized so they cannot escape the workspace.
//!
//! ## References
//!
//! - Path traversal attack: <https://owasp.org/www-community/attacks/Path_Traversal>
//! - Symlink attack: <https://en.wikipedia.org/wiki/Symlink_attack>
//! - Rust `Path`/`Component`: <https://doc.rust-lang.org/std/path/index.html>

use anyhow::{Context, Result, bail};
use std::path::{Component, Path, PathBuf};

/// Resolve a model-supplied path safely inside `root`.
///
/// The prompt tells the model the workspace root, and models hand that
/// absolute path straight back, so an absolute path inside the workspace is
/// read as the relative path it stands for.
///
/// # Arguments
/// - `root`: workspace root directory.
/// - `path`: path supplied by the model.
///
/// # Returns
/// The joined `PathBuf`, or an error if the path is outside the workspace or
/// contains `..`.
pub(super) fn resolve_in_root(root: &Path, path: &str) -> Result<PathBuf> {
    let relative = workspace_relative(root, Path::new(path))?;

    refuse_parent_dirs(relative)?;

    Ok(root.join(relative))
}

/// `requested` as a path relative to the workspace: an absolute path inside
/// the workspace loses its workspace prefix, a relative path is kept, and any
/// other absolute path is an error.
fn workspace_relative<'a>(root: &Path, requested: &'a Path) -> Result<&'a Path> {
    match without_workspace_prefix(root, requested) {
        Some(inside) => Ok(inside),
        None => relative_only(requested),
    }
}

/// `requested` with the workspace root removed from its front, written either
/// as given or in canonical form; `None` when it does not start with either.
fn without_workspace_prefix<'a>(root: &Path, requested: &'a Path) -> Option<&'a Path> {
    let canonical = canonical_root(root).ok();

    requested
        .strip_prefix(root)
        .ok()
        .or_else(|| requested.strip_prefix(canonical.as_deref()?).ok())
}

/// `requested` unchanged when it is relative, else an error.
fn relative_only(requested: &Path) -> Result<&Path> {
    if requested.is_absolute() {
        bail!("that path is outside the workspace; use paths relative to the workspace");
    }

    Ok(requested)
}

/// An error when `relative` contains a `..` component.
fn refuse_parent_dirs(relative: &Path) -> Result<()> {
    let has_parent_dir = relative
        .components()
        .any(|component| matches!(component, Component::ParentDir));

    if has_parent_dir {
        bail!("'..' is not allowed in tool paths");
    }

    Ok(())
}

/// Canonicalize the workspace root once so relative roots (`"."`, `"./src"`)
/// still produce stable absolute paths for containment checks and display.
pub(super) fn canonical_root(root: &Path) -> Result<PathBuf> {
    std::fs::canonicalize(root)
        .with_context(|| format!("failed to resolve workspace root: {}", root.display()))
}

/// Canonicalize `resolved` and verify it still lives under `root`.
///
/// This closes symlink-escape holes: a symlink inside the workspace that points
/// outside must not let the agent read or write files elsewhere on the system.
pub(super) fn ensure_canonical_within_root(root: &Path, resolved: &Path) -> Result<PathBuf> {
    let canonical_root = canonical_root(root)?;
    let canonical_path = std::fs::canonicalize(resolved)
        .with_context(|| format!("failed to resolve path: {}", resolved.display()))?;

    contained_path(canonical_path, &canonical_root, resolved)
}

/// `canonical_path` when it lies under `canonical_root`, else an error naming
/// the path the model asked for.
fn contained_path(
    canonical_path: PathBuf,
    canonical_root: &Path,
    resolved: &Path,
) -> Result<PathBuf> {
    if !canonical_path.starts_with(canonical_root) {
        bail!("path escapes the workspace: {}", resolved.display());
    }

    Ok(canonical_path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn rejects_absolute_paths() {
        let root = Path::new("/tmp/project");
        assert!(resolve_in_root(root, "/etc/passwd").is_err());
    }

    #[test]
    fn reads_an_absolute_path_inside_the_workspace_as_relative() -> Result<()> {
        let root = Path::new("/tmp/project");

        assert_eq!(
            resolve_in_root(root, "/tmp/project/src/main.rs")?,
            Path::new("/tmp/project/src/main.rs")
        );
        assert_eq!(resolve_in_root(root, "/tmp/project")?, root);
        assert!(resolve_in_root(root, "/tmp/project/../secret").is_err());
        assert!(resolve_in_root(root, "/tmp/project-other/x").is_err());

        Ok(())
    }

    #[test]
    fn reads_the_canonical_form_of_a_relative_root() -> Result<()> {
        let base = crate::test_support::unique_temp_dir("openbatrangs-path-canonical");
        fs::create_dir_all(base.join("src"))?;

        let canonical = fs::canonicalize(&base)?;
        let dotted = base.join("src").join("..");
        let absolute = canonical.join("a.rs");
        let resolved = resolve_in_root(&dotted, &absolute.to_string_lossy())?;

        assert_eq!(resolved, dotted.join("a.rs"));

        fs::remove_dir_all(&base)?;
        Ok(())
    }

    #[test]
    fn rejects_parent_dir_traversal() {
        let root = Path::new("/tmp/project");
        assert!(resolve_in_root(root, "../secret").is_err());
        assert!(resolve_in_root(root, "a/../../secret").is_err());
    }

    #[test]
    fn accepts_relative_paths() -> Result<()> {
        let root = Path::new("/tmp/project");

        assert_eq!(
            resolve_in_root(root, "src/main.rs")?,
            Path::new("/tmp/project/src/main.rs")
        );

        Ok(())
    }

    #[test]
    fn accepts_dot_and_cur_dir() -> Result<()> {
        let root = Path::new("/tmp/project");

        assert_eq!(resolve_in_root(root, ".")?, root.to_path_buf());
        assert_eq!(
            resolve_in_root(root, "./src")?,
            Path::new("/tmp/project/src")
        );

        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn rejects_symlink_escape_via_canonicalization() -> Result<()> {
        let base = crate::test_support::unique_temp_dir("openbatrangs-path-test");
        let root = base.join("workspace");
        let outside = base.join("outside");
        fs::create_dir_all(&root)?;
        fs::create_dir_all(&outside)?;
        fs::write(outside.join("secret.txt"), "secret")?;
        std::os::unix::fs::symlink(&outside, root.join("escape"))?;

        let escaped = root.join("escape");
        assert!(ensure_canonical_within_root(&root, &escaped).is_err());
        assert!(ensure_canonical_within_root(&root, &root).is_ok());
        assert!(ensure_canonical_within_root(&root, &root.join("missing")).is_err());

        fs::remove_dir_all(&base)?;
        Ok(())
    }
}
