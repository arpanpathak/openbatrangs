//! One-shot environment facts injected into the first agent turn.
//!
//! The model cannot see the machine it runs on, so GPU and build tasks
//! otherwise start blind: it guesses toolkit versions and architecture names,
//! and the code it writes fails to build. This module gathers the few facts
//! that matter from files on disk — no subprocesses, no network — so the
//! first turn carries them at zero cost.

use std::path::Path;

/// Build manifests whose presence tells the agent how the project is built
/// and, for the finish gate, whether "it builds" is the natural acceptance test.
pub(crate) const BUILD_MANIFESTS: &[(&str, &str)] = &[
    ("Cargo.toml", "Cargo (Rust)"),
    ("package.json", "npm (Node)"),
    ("pyproject.toml", "Python"),
    ("go.mod", "Go"),
    ("Makefile", "Make"),
    ("CMakeLists.txt", "CMake (C/C++)"),
    ("build.gradle", "Gradle"),
];

/// Short, stable facts about the host and the workspace, one per line.
pub(crate) fn environment_facts(cwd: &Path) -> String {
    let mut facts = vec![format!("platform: {}", platform_name())];

    if let Some(version) = cuda_version() {
        facts.push(format!("CUDA toolkit: {version}"));
    }

    let build_systems = present_build_systems(cwd);
    if !build_systems.is_empty() {
        facts.push(format!(
            "workspace build system: {}",
            build_systems.join(", ")
        ));
    }

    facts.join("\n")
}

/// True when the workspace root holds a build manifest, so the natural
/// acceptance test for the task is "the project builds or tests pass".
pub(crate) fn project_has_build_manifest(cwd: &Path) -> bool {
    BUILD_MANIFESTS
        .iter()
        .any(|(file, _)| cwd.join(file).is_file())
}

/// Names of the build manifests present directly in the workspace root.
fn present_build_systems(cwd: &Path) -> Vec<&'static str> {
    BUILD_MANIFESTS
        .iter()
        .filter(|(file, _)| cwd.join(file).is_file())
        .map(|(_, name)| *name)
        .collect()
}

/// The SoC/board name from the device tree, or the CPU architecture when the
/// device tree is unavailable (non-embedded Linux, containers).
fn platform_name() -> String {
    std::fs::read_to_string("/proc/device-tree/model")
        .ok()
        .map(|model| model.trim_end_matches('\0').trim().to_string())
        .filter(|model| !model.is_empty())
        .unwrap_or_else(|| format!("{} {}", std::env::consts::ARCH, std::env::consts::OS))
}

/// The installed CUDA toolkit version, read from `/usr/local/cuda`.
fn cuda_version() -> Option<String> {
    cuda_version_json().or_else(cuda_version_txt)
}

/// Read the version from JetPack-style `/usr/local/cuda/version.json`.
fn cuda_version_json() -> Option<String> {
    let text = std::fs::read_to_string("/usr/local/cuda/version.json").ok()?;
    let value: serde_json::Value = serde_json::from_str(&text).ok()?;
    value
        .pointer("/cuda/version")
        .and_then(serde_json::Value::as_str)
        .map(str::to_string)
}

/// Read the version from classic `/usr/local/cuda/version.txt`.
fn cuda_version_txt() -> Option<String> {
    std::fs::read_to_string("/usr/local/cuda/version.txt")
        .ok()?
        .lines()
        .next()
        .map(|line| line.trim().to_string())
        .filter(|line| !line.is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_build_manifests_in_the_workspace_root() {
        let root = crate::test_support::unique_temp_dir("openbatrangs-env-test");
        std::fs::write(root.join("Cargo.toml"), "[package]").unwrap();

        assert!(project_has_build_manifest(&root));
        assert_eq!(present_build_systems(&root), vec!["Cargo (Rust)"]);
        assert!(environment_facts(&root).contains("workspace build system: Cargo (Rust)"));

        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn reports_no_build_system_for_an_empty_workspace() {
        let root = crate::test_support::unique_temp_dir("openbatrangs-env-test-empty");

        assert!(!project_has_build_manifest(&root));
        assert!(present_build_systems(&root).is_empty());

        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn facts_always_name_the_platform() {
        let facts = environment_facts(std::path::Path::new("."));
        assert!(facts.starts_with("platform: "), "{facts}");
    }
}
