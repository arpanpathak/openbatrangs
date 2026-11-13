<div align="center">

<img src="assets/logo.svg" alt="openbatrangs logo" width="160">

# openbatrangs

A terminal coding agent for local language models, on any hardware.

[![CI](https://img.shields.io/github/actions/workflow/status/arpanpathak/openbatrangs/ci.yml?branch=main&style=for-the-badge&logo=githubactions&logoColor=white&label=CI)](https://github.com/arpanpathak/openbatrangs/actions/workflows/ci.yml)
[![Coverage](https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Farpanpathak%2Fopenbatrangs%2Fcoverage%2Fcoverage.json&style=for-the-badge&logo=codecov&logoColor=white)](https://github.com/arpanpathak/openbatrangs/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/arpanpathak/openbatrangs?style=for-the-badge&logo=github&logoColor=white&color=9664ff)](https://github.com/arpanpathak/openbatrangs/releases)
[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL--3.0-2e8b57?style=for-the-badge&logo=gnu&logoColor=white)](LICENSE)

[![Rust 2024](https://img.shields.io/badge/Rust-2024%20edition-ce422b?style=for-the-badge&logo=rust&logoColor=white)](https://doc.rust-lang.org/edition-guide/rust-2024/)
[![Linux](https://img.shields.io/badge/Linux-x86__64%20%7C%20ARM64-fcc624?style=for-the-badge&logo=linux&logoColor=black)](#install)
[![macOS](https://img.shields.io/badge/macOS-Apple%20silicon%20%7C%20Intel-000000?style=for-the-badge&logo=apple&logoColor=white)](#install)

</div>

With openbatrangs, you run a coding agent in your terminal on a model you host
yourself, on a laptop, a workstation, a GPU server or an edge device.

- **Engines:** TensorRT Edge-LLM, TensorRT-LLM, vLLM, SGLang, llama.cpp, Ollama,
  and any server with OpenAI's chat completions API.
- **Hardware:** Linux on x86_64 and ARM64, and macOS on Apple silicon and Intel.
- **Agent loop:** one tool call per step, up to 40 steps. Tools for listing,
  reading, searching, writing and editing files, running shell commands, and
  searching the web.
- **Modes:** `agent` with every tool, `plan` with read-only tools, and `chat`
  with none.
- **Safety:** `--confirm` for a prompt before each file write and command, and
  `--read-only` for no writes or commands at all. Every command runs with
  `$HOME` and tool caches inside the project's `.agent/` folder, without
  keyboard input, and under a timeout.
- **Web search:** `web_search` and `web_read` through a
  [SearXNG](https://github.com/searxng/searxng) instance.
- **Terminal UI:** a transcript with syntax-highlighted code, slash commands, a
  model picker, and a live panel with GPU, CPU and memory use and tokens per
  second.
- **Reasoning display:** the reasoning of thinking models such as Qwen3 and
  Nemotron, dimmed before each step, or hidden with `--no-think`.

## Engines

| | Engine | Option |
|---|---|---|
| <img src="assets/engines/nvidia.png" alt="" height="28"> | [TensorRT Edge-LLM](https://github.com/NVIDIA/TensorRT-Edge-LLM) | `--openai-url` |
| <img src="assets/engines/nvidia.png" alt="" height="28"> | [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) (`trtllm-serve`) | `--openai-url` |
| <picture><source media="(prefers-color-scheme: dark)" srcset="assets/engines/vllm-dark.png"><img src="assets/engines/vllm-light.png" alt="" height="24"></picture> | [vLLM](https://github.com/vllm-project/vllm) | `--openai-url` |
| <img src="assets/engines/sglang.svg" alt="" height="28"> | [SGLang](https://github.com/sgl-project/sglang) | `--openai-url` |
| <img src="assets/engines/llamacpp.jpg" alt="" height="28"> | [llama.cpp](https://github.com/ggml-org/llama.cpp) (`llama-server`) | `--openai-url` |
| <picture><source media="(prefers-color-scheme: dark)" srcset="assets/engines/ollama-dark.png"><img src="assets/engines/ollama.svg" alt="" height="28"></picture> | [Ollama](https://ollama.com) | `--ollama-url` |

The logos belong to their owners. Their sources are listed in
[`assets/engines/README.md`](assets/engines/README.md).

## Install

Run the installer to get the release binary for your system and architecture.

```sh
curl -fsSL https://github.com/arpanpathak/openbatrangs/releases/latest/download/install.sh | sh
```

You can also download a binary yourself.

| System | File |
|---|---|
| Linux, x86_64 | `openbatrangs-x86_64-unknown-linux-gnu` |
| Linux, ARM64 | `openbatrangs-aarch64-unknown-linux-gnu` |
| macOS, Apple silicon | `openbatrangs-aarch64-apple-darwin` |
| macOS, Intel | `openbatrangs-x86_64-apple-darwin` |

```sh
curl -fsSL -o openbatrangs \
  https://github.com/arpanpathak/openbatrangs/releases/latest/download/openbatrangs-x86_64-unknown-linux-gnu
chmod +x openbatrangs
sudo mv openbatrangs /usr/local/bin/
```

To build from source, install Rust with [rustup](https://rustup.rs) and run the
command below. `rust-toolchain.toml` pins Rust 1.100.0, and you get that
version through rustup on the first build.

```sh
cargo install --git https://github.com/arpanpathak/openbatrangs --locked
```

## Usage

Pass the base URL of your server, including `/v1`, to `--openai-url`. If the
server needs a key, save the key in a file and pass that file to
`--api-key-file`.

```sh
openbatrangs --openai-url http://127.0.0.1:8000/v1                          # open the terminal UI
openbatrangs --openai-url http://127.0.0.1:8000/v1 "fix the failing test"   # run one task and exit
openbatrangs --openai-url https://gpu-box.example.com/v1 --api-key-file ~/.config/gpu-box/key
openbatrangs --openai-url http://127.0.0.1:8000/v1 --model Qwen3.6-35B-A3B-NVFP4 --confirm "update the CLI docs"
openbatrangs --openai-url http://127.0.0.1:8000/v1 --read-only --cwd ~/code/app "suggest a refactoring plan"
```

If you leave out `--model`, you get the first model your server lists. To see
the others, type `/models` in the terminal UI or run `openbatrangs list-models`.
To limit the context window, pass `--max-ctx`.

### Ollama

Without `--openai-url`, you connect to Ollama at `http://localhost:11434`. If
Ollama is installed but not running, `ollama serve` is started for you. If none
of your installed models suits coding, a recommended model is downloaded, unless
you pass `--no-auto-pull`.

```sh
openbatrangs setup                          # install and start Ollama, download a coding model
openbatrangs list-models                    # installed models and their agent scores
openbatrangs doctor                         # check Ollama and get a model recommendation
openbatrangs --model qwen2.5-coder:7b "explain this repository"
```

### Web search

For web search, run a SearXNG instance at `http://127.0.0.1:8888`, or point
`--searxng-url` or `OPENBATRANGS_SEARXNG_URL` at another one.

```sh
ssh -fN -L 18888:127.0.0.1:8888 gpu-box     # forward SearXNG from a remote machine
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
| `/doctor` | check Ollama and get a model recommendation |
| `/read-only` | turn read-only mode on or off |
| `/confirm` | turn the prompt before writes and commands on or off |
| `/yolo` | turn off every prompt |
| `/thinking on\|off` | show or hide the model's reasoning |
| `/steps <n>` | set the step limit |
| `/cwd <path>` | change the project folder |
| `/perf` | show or hide the performance panel |
| `/mouse on\|off` | use the mouse wheel and scrollbar, or the terminal's own text selection |
| `/clear` | clear the transcript |
| `/exit`, `/quit` | leave openbatrangs |

Press Ctrl+C to stop a running task, then type a correction to continue the same
conversation.

On macOS, the performance panel stays empty. Its GPU figures are read from
`tegrastats` or `nvidia-smi`, and its CPU and memory figures from `/proc`.

## Development

```sh
cargo test
cargo clippy --all-targets
cargo llvm-cov --summary-only               # line coverage
```

For the development setup and the pull request guidelines, see
[CONTRIBUTING.md](CONTRIBUTING.md).

## License

openbatrangs is licensed under the [GNU Affero General Public License v3.0](LICENSE).
