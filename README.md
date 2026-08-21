<div align="center">

# 🦇 openBatarangs

> *Agentic coding CLI for local models.*
>
> *Auto-discovers the best model on your hardware,*
> *thinks, reads, edits, and verifies —*
> *no manual model ops, no cloud required.*

**A DeepCode-style autonomous coding agent that runs on Jetson-class edge devices and anywhere Ollama runs.**

[![License: AGPL v3](https://img.shields.io/badge/License-AGPLv3-blue?style=flat-square)](LICENSE) [![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen?style=flat-square)](CONTRIBUTING.md) [![Rust](https://img.shields.io/badge/Rust-1.85+-orange?style=flat-square&logo=rust&logoColor=white)](https://www.rust-lang.org) [![Ollama](https://img.shields.io/badge/Ollama-Local%20LLMs-000?style=flat-square&logo=ollama&logoColor=white)](https://ollama.com) [![NVIDIA Jetson](https://img.shields.io/badge/NVIDIA-Jetson%20Orin%20Nano%20Super-76B900?style=flat-square&logo=nvidia&logoColor=white)](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/) [![Local First](https://img.shields.io/badge/Local-First-2E8B57?style=flat-square)](https://github.com/arpanpathak/openbatrangs) [![Agentic](https://img.shields.io/badge/Agentic-Tools-8A2BE2?style=flat-square)](https://github.com/arpanpathak/openbatrangs) [![Context Aware](https://img.shields.io/badge/Context-32K%2B-FF6B6B?style=flat-square)](https://github.com/arpanpathak/openbatrangs) [![Cargo Install](https://img.shields.io/badge/Install-Cargo-4B8BBE?style=flat-square&logo=rust&logoColor=white)](https://crates.io) [![GitHub](https://img.shields.io/badge/GitHub-arpanpathak%2Fopenbatrangs-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/arpanpathak/openbatrangs)

</div>

---

openBatarangs is an **agentic coding CLI** for local LLMs. It talks to an
[Ollama](https://ollama.com) server, auto-discovers the best installed model for
your hardware, and iterates with tools to explore, edit, and verify code —
no manual model selection required.

It is designed to run well on Jetson-class edge devices (Orin Nano/NX Super,
16 GB unified memory) and on any desktop/laptop that can run Ollama.

## Why Ollama?

- Same `llama.cpp` + CUDA/ROCm/Metal engine used by most local LLM tools.
- Auto-discovers installed models through the Ollama API.
- No heavy Rust/CUDA compile per machine.
- Best performance-per-watt-per-dollar for local agentic coding on edge devices.

## Requirements

- [Ollama](https://ollama.com) installed and running (`ollama serve`).
- At least one model installed, e.g.:
  ```sh
  ollama pull qwen2.5-coder:7b
  ```
  If no suitable model is installed, openBatarangs can auto-pull a recommended
  coding model unless you pass `--no-auto-pull`.

## Install / Build

```sh
git clone https://github.com/arpanpathak/openbatrangs.git
cd openbatrangs

# Build release
cargo build --release

# Or install the binary into ~/.cargo/bin
cargo install --path .
```

The binary is at `target/release/openbatrangs` (or `~/.cargo/bin/openbatrangs`).

## Quick start

```sh
# Agent mode with a task (auto-discovers the best model)
./target/release/openbatrangs "fix the Rust build errors"

# Agent mode in a specific directory
./target/release/openbatrangs --cwd /path/to/project "add a --dry-run flag"

# See what models are installed and which is best
./target/release/openbatrangs list-models

# Check Ollama and get a recommendation
./target/release/openbatrangs doctor

# Force a specific model
./target/release/openbatrangs --model qwen2.5-coder:7b "explain this repo"

# Read-only mode (no file writes or shell commands)
./target/release/openbatrangs --read-only "suggest a refactor plan"

# Ask before every write/command
./target/release/openbatrangs --confirm "update the CLI docs"
```

## Agent tools

The agent can use these tools during a task:

| Tool | Description |
| --- | --- |
| `list_files` | List files in the workspace (skips `target`, `.git`, `node_modules`, etc.) |
| `read_file` | Read a text file with a size cap |
| `grep_files` | Regex search across workspace files |
| `write_file` | Write or overwrite a file (relative paths only) |
| `run_command` | Run a shell command in the workspace (e.g. `cargo check`) |
| `finish` | Signal the task is complete |

For safety:
- Tool paths must be relative; absolute paths and `..` are rejected.
- `--read-only` disables `write_file` and `run_command`.
- `--confirm` asks before each write/command.

## Model auto-discovery

`openBatarangs` scores installed models by:

- Memory fit (model file size vs. system memory)
- Parameter size sweet spot for agentic coding (roughly 3B–8B on 16 GB devices)
- Coding-model name bonus (`qwen2.5-coder`, `deepseek-coder`, etc.)
- Context window (target ~32K)
- Quantization quality

If nothing suitable is installed, it can automatically pull a recommended model
(`qwen2.5-coder:7b` on >=12 GB systems, otherwise `qwen2.5-coder:3b`).

## Common flags

```
--ollama-url <URL>   Ollama server URL (default http://localhost:11434)
--model <TAG>        Use a specific Ollama model tag
--cwd <DIR>          Workspace directory (default .)
--max-steps <N>      Max agent iterations (default 12)
--min-context <N>    Minimum context window for auto-selection (default 8192)
--read-only          Disable writes and shell commands
--confirm            Ask before writes/commands
--no-auto-pull       Never auto-pull models
```

## Roadmap

This is the first working version. Future steps for standalone distribution:

- Prebuilt binaries for `aarch64` and `x86_64`
- `cargo install` from crates.io
- Support OpenAI-compatible remote endpoints in addition to Ollama
- Optional direct GGUF fallback via `mistralrs` or `llama.cpp`
- Better token budgeting / context compression for long agent sessions
- Installer script that checks for Ollama and installs a recommended model

## License

[AGPL-3.0](LICENSE) — same license as the CivicSense project. Contributions are welcome; see [CONTRIBUTING.md](CONTRIBUTING.md).
