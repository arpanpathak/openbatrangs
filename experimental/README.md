# openBatarangs Experimental Inference Engines

> **Branch:** `experimental/tensorrt-bench`
>
> This directory is intentionally experimental. It exists on a dedicated branch
> so it never pollutes the stable `main` codebase.

## What this branch adds

A Rust-only experimental harness that benchmarks local inference engines on
Jetson-class hardware and reports **tokens/second**, **watts**,
**tokens/second per watt**, and an estimated **USD per million tokens**.

Currently implemented engines:

| Engine | How it runs | Status |
| --- | --- | --- |
| `ollama` | Rust `OllamaClient` → `/api/generate` | Production baseline |
| `tensorrt` | Rust shell-out to installed `trtexec` binary | Experimental (prefill-only microbenchmark) |

No Python is used. The experimental code is modular Rust that follows the
same conventions as the rest of the project: focused modules, constants in
`src/constants`, pattern matching instead of nested `if/else`, and an
open-for-extension/closed-for-modification backend trait.

## Using engines directly in the CLI

The engine is selectable at startup and switchable at runtime:

```sh
# Start the agent/TUI with a specific engine
openbatrangs --engine ollama "explain this repo"
openbatrangs --engine tensorrt "task..."   # benchmark-only; see note below

# Inside the TUI, switch engines without restarting
/engine ollama
/engine tensorrt
```

- `/engine` with no argument shows the current engine.
- Switching requires the engine to be available (`experimental doctor` checks this).
- `tensorrt` is currently **benchmark-only**: `trtexec` cannot serve interactive
  chat/tool-calling, so agent tasks on TensorRT will report an unsupported error
  until a real TensorRT chat server (e.g. TensorRT-LLM) is added.

## Commands

```sh
# Show which engines are available on this machine
cargo run -- experimental doctor

# Benchmark every available engine (defaults)
cargo run -- experimental bench

# Benchmark a specific engine/model
cargo run -- experimental bench --engines ollama --model qwen2.5-coder:7b --iterations 3

# TensorRT requires an ONNX model file and, for dynamic inputs, explicit shapes
cargo run -- experimental bench --engines tensorrt \
  --model experimental/models/gpt2-tiny-decoder-plain.onnx \
  --trt-shapes "input_ids:1x128,attention_mask:1x128"

# Write the markdown report to a file
cargo run -- experimental bench --output experimental/RESULTS.md
```

### Useful flags

| Flag | Meaning |
| --- | --- |
| `--engines ollama,tensorrt` | Comma-separated engine list; default is all |
| `--model <tag-or-path>` | Ollama model tag or ONNX model path |
| `--prompt "..."` | Prompt used for Ollama generation |
| `--max-tokens N` | Max generated tokens for Ollama (default 128) |
| `--iterations N` | Benchmark repetitions per engine (default 3) |
| `--seq-len N` | TensorRT input sequence length for prefill tok/s (default 128) |
| `--avg-runs N` | `trtexec --avgRuns` count (default 20) |
| `--trt-shapes "..."` | Optional `trtexec --shapes` for dynamic ONNX inputs |
| `--device-cost-usd N` | Hardware price used in amortization math (default 699) |
| `--electricity-usd-per-kwh N` | Electricity price (default 0.15) |
| `--output PATH` | Write the markdown report to a file |

## Architecture

```
src/engine/
├── mod.rs        EngineKind, InferenceBackend trait, BenchSample, create_backend factory
├── ollama.rs     OllamaBackend (wraps existing OllamaClient)
└── tensorrt.rs   TensorRtBackend (locates/executes trtexec, parses throughput)

src/commands/experimental.rs   `experimental doctor` + `experimental bench`
src/perf/power.rs              PowerSampler (tegrastats VDD_IN averaging)
```

### Adding a new engine (open for extension)

1. Implement `InferenceBackend` for your engine in `src/engine/<name>.rs`.
2. Add one `match` arm in `create_backend` in `src/engine/mod.rs`.
3. Add the engine name to `EngineKind` and `EngineKind::all()`.

The agent, TUI, and benchmark harness only talk to `dyn InferenceBackend`, so
they do **not** need to change.

### Why `--engine tensorrt` is benchmark-only

`trtexec` is an engine build/benchmark tool, not a chat server. It cannot
stream JSON tool-call responses, so `TensorRtBackend::chat_stream` returns an
unsupported error. The main agent therefore stays on Ollama for interactive
use; TensorRT is used for head-to-head engine throughput experiments.

## Performance-per-watt-per-dollar methodology

For each benchmark iteration:

1. `PowerSampler` starts `tegrastats --interval 500` and records every `VDD_IN`
   reading in watts.
2. The engine generates a fixed prompt with up to `--max-tokens` tokens
   (Ollama) or runs `trtexec` on the ONNX graph (TensorRT).
3. The sampler is stopped and the average power is computed.

Reported metrics:

- **Tokens/sec**
  - Ollama: `eval_count / total_duration` from `/api/generate` (native counters).
  - TensorRT: `trtexec` throughput (inferences/sec) × `--seq-len`.
    This is a **prefill-only** microbenchmark, not end-to-end decoding.
- **Tokens/sec/W**: tokens/sec ÷ average watts.
- **USD / 1M tokens**:
  - Energy: `(1e6 / tps) × watts / 1000 / 3600 × $/kWh`
  - Hardware amortization: `device_cost / (tps × seconds_per_year × lifetime_years × duty_cycle) × 1e6`
  - Defaults: `$699` device, `$0.15/kWh`, 5-year life, 50% duty cycle.

Treat the dollar figures as **rough estimates** for comparing engines on the
same device, not as cloud-pricing equivalents.

## vLLM status

vLLM is intentionally **not** implemented on this branch.

> vLLM does not provide official aarch64/Jetson wheels and is not viable on
> Orin-class unified memory today.

The `experimental doctor` command prints this rationale so users know it was
considered and rejected for Jetson, rather than silently omitted.

## Reproducing the TensorRT benchmark

1. Download a small ONNX GPT-2 decoder that uses standard ONNX ops
   (no Python required):

```sh
mkdir -p experimental/models
curl -L -o experimental/models/gpt2-tiny-decoder-plain.onnx \
  "https://huggingface.co/fxmarty/onnx-tiny-random-gpt2-with-merge/resolve/main/decoder_model.onnx?download=true"
```

2. Run the benchmark (dynamic inputs need `--trt-shapes`):

```sh
cargo run --release -- experimental bench --engines tensorrt \
  --model experimental/models/gpt2-tiny-decoder-plain.onnx \
  --avg-runs 20 --seq-len 128 \
  --trt-shapes "input_ids:1x128,attention_mask:1x128" \
  --output experimental/RESULTS.md
```

3. Compare with the Ollama baseline:

```sh
cargo run --release -- experimental bench --engines ollama \
  --model qwen2.5-coder:3b --max-tokens 128 --iterations 3
```

> Why not the Qwen ONNX files? The `onnx-community/Qwen2.5-Coder-0.5B-Instruct`
> exports use `com.microsoft` fused ops or asymmetric INT8 quantization that the
> TensorRT parser cannot import without extra plugins. See RESULTS.md for details.

## Results

See [RESULTS.md](RESULTS.md) for the latest numbers measured on this
Jetson Orin NX Super (JetPack 6.2, TensorRT 10.7).
