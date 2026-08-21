use anyhow::{anyhow, bail, Context, Result};
use regex::Regex;
use std::path::{Component, Path, PathBuf};
use std::time::Duration;
use walkdir::WalkDir;

pub const MAX_TOOL_OUTPUT: usize = 20_000;
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

fn is_heavy_dir(name: &str) -> bool {
    HEAVY_DIRS.contains(&name)
}

/// Resolve a model-supplied path safely inside `root`. Only relative paths without `..` are allowed.
pub fn resolve_in_root(root: &Path, path: &str) -> Result<PathBuf> {
    let p = Path::new(path);
    if p.is_absolute() {
        bail!("absolute paths are not allowed; use paths relative to the workspace");
    }
    for comp in p.components() {
        if matches!(comp, Component::ParentDir) {
            bail!("'..' is not allowed in tool paths");
        }
    }
    Ok(root.join(p))
}

fn walker(
    root: &Path,
    max_depth: usize,
) -> impl Iterator<Item = Result<walkdir::DirEntry, walkdir::Error>> + '_ {
    WalkDir::new(root)
        .max_depth(max_depth)
        .follow_links(false)
        .into_iter()
        .filter_entry(|e| {
            if e.depth() == 0 {
                return true;
            }
            if e.file_type().is_dir() {
                let name = e.file_name().to_string_lossy();
                return !is_heavy_dir(&name);
            }
            true
        })
}

pub fn list_files(root: &Path, path: &str, max_depth: usize) -> Result<String> {
    let dir = resolve_in_root(root, path)?;
    if !dir.is_dir() {
        bail!("not a directory: {}", dir.display());
    }

    let mut out = String::new();
    let mut count = 0usize;
    for entry in walker(&dir, max_depth) {
        let entry = entry.context("walkdir error")?;
        if entry.file_type().is_file() {
            let rel = entry
                .path()
                .strip_prefix(root)
                .unwrap_or(entry.path())
                .to_string_lossy()
                .to_string();
            let size = entry.metadata().map(|m| m.len()).unwrap_or(0);
            out.push_str(&format!("{} ({} bytes)\n", rel, size));
            count += 1;
            if count >= 5000 {
                out.push_str("... (truncated at 5000 files)\n");
                break;
            }
        }
    }
    if count == 0 {
        out.push_str("(no files found)\n");
    }
    Ok(truncate(out))
}

pub fn read_file(root: &Path, path: &str, max_chars: usize) -> Result<String> {
    let file = resolve_in_root(root, path)?;
    if !file.is_file() {
        bail!("not a file: {}", file.display());
    }
    let content = std::fs::read_to_string(&file).context("failed to read file (may be binary)")?;
    let content = truncate_to(content, max_chars);
    Ok(format!(
        "--- {} ---\n{}",
        file.strip_prefix(root).unwrap_or(&file).display(),
        content
    ))
}

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

pub fn grep_files(root: &Path, pattern: &str, path: &str, max_results: usize) -> Result<String> {
    let dir = resolve_in_root(root, path)?;
    if !dir.is_dir() {
        bail!("not a directory: {}", dir.display());
    }
    let re = Regex::new(pattern).context("invalid regex pattern")?;
    let mut out = String::new();
    let mut count = 0usize;

    for entry in walker(&dir, 12) {
        let entry = entry.context("walkdir error")?;
        if !entry.file_type().is_file() {
            continue;
        }
        // Skip obvious binary files.
        if let Ok(data) = std::fs::read(entry.path()) {
            if data.iter().take(8192).any(|&b| b == 0) {
                continue;
            }
        }
        let Ok(content) = std::fs::read_to_string(entry.path()) else {
            continue;
        };
        let rel = entry
            .path()
            .strip_prefix(root)
            .unwrap_or(entry.path())
            .to_string_lossy()
            .to_string();
        for (idx, line) in content.lines().enumerate() {
            if re.is_match(line) {
                let line = line.trim();
                out.push_str(&format!("{}:{}: {}\n", rel, idx + 1, line));
                count += 1;
                if count >= max_results {
                    out.push_str(&format!("... (truncated at {max_results} matches)\n"));
                    return Ok(truncate(out));
                }
            }
        }
    }
    if count == 0 {
        out.push_str("(no matches)\n");
    }
    Ok(truncate(out))
}

pub async fn run_command(root: &Path, command: &str, timeout_secs: u64) -> Result<String> {
    let output = tokio::time::timeout(
        Duration::from_secs(timeout_secs.max(1)),
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

    let status = output.status;
    let exit = status
        .code()
        .map(|c| c.to_string())
        .unwrap_or_else(|| "signal".into());
    if !status.success() {
        text.push_str(&format!("\n[exit code {exit}]\n"));
    } else {
        text.push_str(&format!("\n[exit code 0]\n"));
    }
    Ok(truncate(text))
}

pub fn truncate(s: String) -> String {
    truncate_to(s, MAX_TOOL_OUTPUT)
}

fn truncate_to(s: String, max: usize) -> String {
    if s.len() <= max {
        return s;
    }
    let mut result = s.chars().take(max).collect::<String>();
    result.push_str(&format!(
        "\n... (truncated, {} chars omitted)",
        s.chars().count().saturating_sub(max)
    ));
    result
}
