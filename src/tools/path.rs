//! Path-safety helpers for agent-supplied paths.

use anyhow::{bail, Context, Result};
use std::path::{Component, Path, PathBuf};

/// Resolve a model-supplied path safely inside `root`.
///
/// # Arguments
/// - `root`: workspace root directory.
/// - `path`: relative path supplied by the model.
///
/// # Returns
/// The joined `PathBuf`, or an error if the path is absolute or contains `..`.
pub(super) fn resolve_in_root(root: &Path, path: &str) -> Result<PathBuf> {
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
    if !canonical_path.starts_with(&canonical_root) {
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
    fn rejects_parent_dir_traversal() {
        let root = Path::new("/tmp/project");
        assert!(resolve_in_root(root, "../secret").is_err());
        assert!(resolve_in_root(root, "a/../../secret").is_err());
    }

    #[test]
    fn accepts_relative_paths() {
        let root = Path::new("/tmp/project");
        assert_eq!(
            resolve_in_root(root, "src/main.rs").expect("relative path should resolve"),
            Path::new("/tmp/project/src/main.rs")
        );
    }

    #[test]
    fn accepts_dot_and_cur_dir() {
        let root = Path::new("/tmp/project");
        assert_eq!(resolve_in_root(root, ".").unwrap(), root.to_path_buf());
        assert_eq!(
            resolve_in_root(root, "./src").unwrap(),
            Path::new("/tmp/project/src")
        );
    }

    #[cfg(unix)]
    #[test]
    fn rejects_symlink_escape_via_canonicalization() {
        let base = crate::test_support::unique_temp_dir("openbatrangs-path-test");
        let root = base.join("workspace");
        let outside = base.join("outside");
        fs::create_dir_all(&root).unwrap();
        fs::create_dir_all(&outside).unwrap();
        fs::write(outside.join("secret.txt"), "secret").unwrap();
        std::os::unix::fs::symlink(&outside, root.join("escape")).unwrap();

        let escaped = root.join("escape");
        assert!(ensure_canonical_within_root(&root, &escaped).is_err());
        assert!(ensure_canonical_within_root(&root, &root).is_ok());
        fs::remove_dir_all(&base).unwrap();
    }
}
