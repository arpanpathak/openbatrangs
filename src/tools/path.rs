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

use anyhow::{bail, Context, Result};
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
    let requested = Path::new(path);
    let canonical = canonical_root(root).ok();

    let inside = requested
        .strip_prefix(root)
        .ok()
        .or_else(|| requested.strip_prefix(canonical.as_deref()?).ok());

    let relative = match (inside, requested.is_absolute()) {
        (Some(inside), _) => inside,
        (None, false) => requested,
        (None, true) => {
            bail!("that path is outside the workspace; use paths relative to the workspace")
        }
    };

    if relative
        .components()
        .any(|component| matches!(component, Component::ParentDir))
    {
        bail!("'..' is not allowed in tool paths");
    }

    Ok(root.join(relative))
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
    fn reads_an_absolute_path_inside_the_workspace_as_relative() {
        let root = Path::new("/tmp/project");

        assert_eq!(
            resolve_in_root(root, "/tmp/project/src/main.rs").unwrap(),
            Path::new("/tmp/project/src/main.rs")
        );
        assert_eq!(resolve_in_root(root, "/tmp/project").unwrap(), root);
        assert!(resolve_in_root(root, "/tmp/project/../secret").is_err());
        assert!(resolve_in_root(root, "/tmp/project-other/x").is_err());
    }

    #[test]
    fn reads_the_canonical_form_of_a_relative_root() {
        let base = crate::test_support::unique_temp_dir("openbatrangs-path-canonical");
        fs::create_dir_all(base.join("src")).unwrap();

        let canonical = fs::canonicalize(&base).unwrap();
        let dotted = base.join("src").join("..");
        let resolved = resolve_in_root(&dotted, canonical.join("a.rs").to_str().unwrap());

        assert_eq!(resolved.unwrap(), dotted.join("a.rs"));

        fs::remove_dir_all(&base).unwrap();
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
