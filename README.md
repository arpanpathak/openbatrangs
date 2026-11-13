<div align="center">

<img src="assets/logo.svg" alt="openbatrangs logo" width="160">

# openbatrangs

A terminal coding agent that runs on local language models.

[![CI](https://img.shields.io/github/actions/workflow/status/arpanpathak/openbatrangs/ci.yml?branch=main&style=flat-square&logo=github&label=CI)](https://github.com/arpanpathak/openbatrangs/actions/workflows/ci.yml)
[![Coverage](https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Farpanpathak%2Fopenbatrangs%2Fcoverage%2Fcoverage.json&style=flat-square)](https://github.com/arpanpathak/openbatrangs/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/arpanpathak/openbatrangs?style=flat-square&color=9664ff)](https://github.com/arpanpathak/openbatrangs/releases)
[![Rust 2024](https://img.shields.io/badge/Rust-2024%20edition-dea584?style=flat-square&logo=rust&logoColor=white)](https://doc.rust-lang.org/edition-guide/rust-2024/)
[![License: AGPL-3.0](https://img.shields.io/badge/license-AGPL--3.0-2e8b57?style=flat-square)](LICENSE)

</div>

With openbatrangs, a language model on your own hardware works on a project
through file, search, shell and web tools. It connects to TensorRT Edge-LLM,
TensorRT-LLM, vLLM, SGLang, llama.cpp, Ollama, and any other server with
OpenAI's chat completions API. It runs on Linux and macOS, from a Jetson board
to a multi-GPU server.

- **Agent loop.** In each step, the model calls one tool to list, read, search,
  write or edit files, run a shell command, or search the web. The loop ends
  when the model calls `finish`, or after 40 steps.
- **Modes.** `agent` allows every tool, `plan` allows only reading, and `chat`
  allows no tools.
- **Web search.** `web_search` and `web_read` look up documentation through a
  [SearXNG](https://github.com/searxng/searxng) instance.
- **Safety controls.** `--confirm` adds a prompt before each file write and
  shell command, and `--read-only` allows neither. Commands run with `$HOME`
  and tool caches inside the project's `.agent/` folder. They get no keyboard
  input, and they end at a timeout.
- **Terminal UI.** A transcript with syntax-highlighted code, slash commands, a
  model picker, and a panel with GPU, CPU and memory use and tokens per second.
- **Reasoning.** For thinking models such as Qwen3 and Nemotron, the reasoning
  appears dimmed before each step. `--no-think` turns thinking off.
- **Model selection.** `/models` lists what the server serves. With Ollama, the
  installed models are also scored for agentic coding on the current hardware,
  and the best fit is used first.

## Engines

| | Engine | Connect with | Status |
|---|---|---|---|
| <img src="assets/engines/nvidia.png" alt="" height="28"> | [TensorRT Edge-LLM](https://github.com/NVIDIA/TensorRT-Edge-LLM) | `--openai-url` | tested on Jetson AGX Thor |
| <img src="assets/engines/nvidia.png" alt="" height="28"> | [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) (`trtllm-serve`) | `--openai-url` | OpenAI-compatible, not tested |
| <picture><source media="(prefers-color-scheme: dark)" srcset="assets/engines/vllm-dark.png"><img src="assets/engines/vllm-light.png" alt="" height="24"></picture> | [vLLM](https://github.com/vllm-project/vllm) | `--openai-url` | OpenAI-compatible, not tested |
| <img src="assets/engines/sglang.svg" alt="" height="28"> | [SGLang](https://github.com/sgl-project/sglang) | `--openai-url` | OpenAI-compatible, not tested |
| <img src="assets/engines/llamacpp.jpg" alt="" height="28"> | [llama.cpp](https://github.com/ggml-org/llama.cpp) (`llama-server`) | `--openai-url` | tested |
| <picture><source media="(prefers-color-scheme: dark)" srcset="assets/engines/ollama-dark.png"><img src="assets/engines/ollama.svg" alt="" height="28"></picture> | [Ollama](https://ollama.com) | `--ollama-url` | tested |

Each logo belongs to its owner. [`assets/engines/README.md`](assets/engines/README.md)
lists where each one comes from.

## Install

The installer downloads the release binary for your system and architecture.

```sh
curl -fsSL https://github.com/arpanpathak/openbatrangs/releases/latest/download/install.sh | sh
```

Release binaries are also available on their own.

| System | File |
|---|---|
| Linux, x86_64 | `openbatrangs-x86_64-unknown-linux-gnu` |
| Linux, ARM64 (Jetson, Raspberry Pi 5) | `openbatrangs-aarch64-unknown-linux-gnu` |
| macOS, Apple silicon | `openbatrangs-aarch64-apple-darwin` |
| macOS, Intel | `openbatrangs-x86_64-apple-darwin` |

```sh
curl -fsSL -o openbatrangs \
  https://github.com/arpanpathak/openbatrangs/releases/latest/download/openbatrangs-aarch64-unknown-linux-gnu
chmod +x openbatrangs
sudo mv openbatrangs /usr/local/bin/
```

To build from source, install Rust with [rustup](https://rustup.rs). The
repository pins Rust 1.100.0 in `rust-toolchain.toml`, and rustup installs it
on the first build.

```sh
cargo install --git https://github.com/arpanpathak/openbatrangs --locked
```

## Usage

`--openai-url` takes the base URL of a TensorRT Edge-LLM, TensorRT-LLM, vLLM,
SGLang or llama.cpp server, including `/v1`. `--api-key-file` sends the
contents of a file as a bearer token.

```sh
openbatrangs --openai-url http://127.0.0.1:8000/v1                          # start the terminal UI
openbatrangs --openai-url http://127.0.0.1:8000/v1 "fix the failing test"   # run one task and exit
openbatrangs --openai-url https://gpu-box.example.com/v1 --api-key-file ~/.config/gpu-box/key
openbatrangs --openai-url http://127.0.0.1:8000/v1 --model Qwen3.6-35B-A3B-NVFP4 --confirm "update the CLI docs"
openbatrangs --openai-url http://127.0.0.1:8000/v1 --read-only --cwd ~/code/app "suggest a refactoring plan"
```

Without `--model`, openbatrangs uses the first model the server lists, and
`/models` and `list-models` show them all. `--max-ctx` caps the context window
that openbatrangs uses, and `--thor` is a shortcut for the author's Jetson AGX
Thor.

### Ollama

Without `--openai-url`, openbatrangs connects to Ollama at
`http://localhost:11434`. It starts `ollama serve` when Ollama is installed but
not running, and it downloads a recommended coding model when no installed
model is suitable. `--no-auto-pull` turns that download off.

```sh
openbatrangs setup                          # install and start Ollama, download a coding model
openbatrangs list-models                    # installed models and their agent scores
openbatrangs doctor                         # check Ollama and recommend a model
openbatrangs --model qwen2.5-coder:7b "explain this repository"
```

### Web search

The web tools query SearXNG at `http://127.0.0.1:8888` by default. To use
another instance, pass `--searxng-url` or set `OPENBATRANGS_SEARXNG_URL`.

```sh
ssh -fN -L 18888:127.0.0.1:8888 gpu-box     # reach SearXNG on a remote machine
openbatrangs --searxng-url http://127.0.0.1:18888
```

## Terminal UI commands

| Command | Effect |
|---|---|
| `/help` | list the commands |
| `/status` | show the current settings and model |
| `/mode agent\|plan\|chat` | switch between every tool, read-only planning and plain chat |
| `/model <tag>` | switch to an installed model |
| `/models` | open the model picker |
| `/pull <tag>` | download a model from Ollama |
| `/setup` | install and start Ollama and download a model |
| `/doctor` | check Ollama and recommend a model |
| `/read-only` | toggle read-only mode |
| `/confirm` | toggle confirmation before writes and commands |
| `/yolo` | turn off every confirmation |
| `/thinking on\|off` | show or hide the model's reasoning |
| `/steps <n>` | set the step limit |
| `/cwd <path>` | change the project folder |
| `/perf` | toggle the performance panel |
| `/mouse on\|off` | use the mouse wheel and scrollbar, or the terminal's own selection |
| `/clear` | clear the transcript |
| `/exit`, `/quit` | leave openbatrangs |

Ctrl+C stops a running task. A correction typed after that continues the same
conversation.

On macOS, the performance panel shows no GPU data, because it reads
`tegrastats` or `nvidia-smi`, and its CPU and memory figures stay at zero,
because they come from `/proc`.

## Development

```sh
cargo test
cargo clippy --all-targets
cargo llvm-cov --summary-only               # line coverage
```

[CONTRIBUTING.md](CONTRIBUTING.md) describes the development setup and the
guidelines for pull requests.

## License

openbatrangs is licensed under the [GNU Affero General Public License v3.0](LICENSE).
