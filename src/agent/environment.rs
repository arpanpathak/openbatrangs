//! One-shot environment facts injected into the first agent turn.
//!
//! The model has no information about the machine it runs on, so without
//! these facts its replies to GPU and build tasks contain guessed toolkit
//! versions and architecture names, and the code they contain fails to build. This module gathers the few facts
//! that matter from files on disk — no subprocesses, no network — so the
//! first turn carries them at zero cost.

use serde_json::Value;
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

/// Device-tree file that names the board on embedded Linux.
const DEVICE_TREE_MODEL: &str = "/proc/device-tree/model";

/// Directory where the CUDA toolkit is installed.
const CUDA_ROOT: &str = "/usr/local/cuda";

/// Short, stable facts about the host and the workspace, one per line.
pub(crate) fn environment_facts(cwd: &Path) -> String {
    let platform = platform_name(Path::new(DEVICE_TREE_MODEL));
    let cuda_version = cuda_version(Path::new(CUDA_ROOT));
    let build_systems = present_build_systems(cwd);

    facts_text(&platform, cuda_version, &build_systems)
}

/// The fact lines: the platform, then the CUDA version and the build systems
/// when they are known.
fn facts_text(platform: &str, cuda_version: Option<String>, build_systems: &[&str]) -> String {
    let mut facts = vec![format!("platform: {platform}")];

    facts.extend(cuda_version.map(|version| format!("CUDA toolkit: {version}")));
    facts.extend(build_system_fact(build_systems));

    facts.join("\n")
}

/// The build-system line, or `None` when the workspace has no manifest.
fn build_system_fact(build_systems: &[&str]) -> Option<String> {
    if build_systems.is_empty() {
        return None;
    }

    Some(format!(
        "workspace build system: {}",
        build_systems.join(", ")
    ))
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

/// The board name in the device-tree file `model_path`, or the CPU
/// architecture and OS when that file is missing or empty (non-embedded
/// Linux, containers).
fn platform_name(model_path: &Path) -> String {
    std::fs::read_to_string(model_path)
        .ok()
        .map(|model| model.trim_end_matches('\0').trim().to_string())
        .filter(|model| !model.is_empty())
        .unwrap_or_else(|| format!("{} {}", std::env::consts::ARCH, std::env::consts::OS))
}

/// The CUDA toolkit version installed under `cuda_root`.
fn cuda_version(cuda_root: &Path) -> Option<String> {
    cuda_version_json(cuda_root).or_else(|| cuda_version_txt(cuda_root))
}

/// The version in JetPack-style `version.json`.
fn cuda_version_json(cuda_root: &Path) -> Option<String> {
    let text = std::fs::read_to_string(cuda_root.join("version.json")).ok()?;

    serde_json::from_str::<Value>(&text)
        .ok()
        .and_then(|value| version_in_json(&value))
}

/// The `cuda.version` string of a parsed `version.json`.
fn version_in_json(value: &Value) -> Option<String> {
    value
        .pointer("/cuda/version")
        .and_then(Value::as_str)
        .map(str::to_string)
}

/// The first line of classic `version.txt`.
fn cuda_version_txt(cuda_root: &Path) -> Option<String> {
    std::fs::read_to_string(cuda_root.join("version.txt"))
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
    fn detects_build_manifests_in_the_workspace_root() -> anyhow::Result<()> {
        let root = crate::test_support::unique_temp_dir("openbatrangs-env-test");
        std::fs::write(root.join("Cargo.toml"), "[package]")?;

        assert!(project_has_build_manifest(&root));
        assert_eq!(present_build_systems(&root), vec!["Cargo (Rust)"]);
        assert!(environment_facts(&root).contains("workspace build system: Cargo (Rust)"));

        std::fs::remove_dir_all(root)?;
        Ok(())
    }

    #[test]
    fn reports_no_build_system_for_an_empty_workspace() -> anyhow::Result<()> {
        let root = crate::test_support::unique_temp_dir("openbatrangs-env-test-empty");

        assert!(!project_has_build_manifest(&root));
        assert!(present_build_systems(&root).is_empty());
        assert_eq!(build_system_fact(&[]), None);

        std::fs::remove_dir_all(root)?;
        Ok(())
    }

    #[test]
    fn facts_always_name_the_platform() {
        let facts = environment_facts(Path::new("."));

        assert!(facts.starts_with("platform: "), "{facts}");
    }

    #[test]
    fn facts_list_cuda_and_build_systems_when_present() {
        let facts = facts_text("Thor", Some("13.0".to_string()), &["Cargo (Rust)", "Make"]);

        assert_eq!(
            facts,
            "platform: Thor\nCUDA toolkit: 13.0\nworkspace build system: Cargo (Rust), Make"
        );
    }

    #[test]
    fn platform_name_reads_the_device_tree_or_falls_back() -> anyhow::Result<()> {
        let root = crate::test_support::unique_temp_dir("openbatrangs-env-model");
        let model_path = root.join("model");
        std::fs::write(&model_path, "NVIDIA Jetson AGX Thor\0")?;

        assert_eq!(platform_name(&model_path), "NVIDIA Jetson AGX Thor");
        assert!(platform_name(&root.join("missing")).contains(std::env::consts::OS));

        std::fs::remove_dir_all(root)?;
        Ok(())
    }

    #[test]
    fn cuda_version_prefers_version_json_then_version_txt() -> anyhow::Result<()> {
        let root = crate::test_support::unique_temp_dir("openbatrangs-env-cuda");
        assert_eq!(cuda_version(&root), None);

        std::fs::write(root.join("version.txt"), "CUDA Version 12.6.68\n")?;
        assert_eq!(cuda_version(&root).as_deref(), Some("CUDA Version 12.6.68"));

        std::fs::write(
            root.join("version.json"),
            r#"{"cuda":{"version":"13.0.48"}}"#,
        )?;
        assert_eq!(cuda_version(&root).as_deref(), Some("13.0.48"));

        std::fs::remove_dir_all(root)?;
        Ok(())
    }
}
