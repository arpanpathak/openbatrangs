//! Path-safety helpers for agent-supplied paths.

use anyhow::{bail, Result};
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

#[cfg(test)]
mod tests {
    use super::*;

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
}
