# openBatarangs experimental benchmark results

Measured on: **NVIDIA Jetson Orin NX Super (16GB)**, JetPack 6.2, CUDA 12.6,
TensorRT 10.7, Ollama 0.32.15, MAXN_SUPER power mode.

Defaults: `$699` device, `$0.15/kWh`, 5-year life, 50% duty cycle.
Ollama runs use 128 generated tokens; TensorRT uses `trtexec --avgRuns=20`
with a 128-token prefill shape.

## Ollama (end-to-end generation, `/api/generate` counters)

| Model | Tokens/s | Avg W | Tokens/s/W | USD / 1M tokens |
| --- | ---: | ---: | ---: | ---: |
| qwen2.5-coder:3b | 20.79 | 24.07 | 0.864 | 0.475 |
| qwen2.5-coder:7b-instruct-q4_K_M | 6.20 | 18.65 | 0.333 | 1.55 |

Notes:
- 3 iterations for 3B; 7B was a 64-token quick probe.
- These are end-to-end generation rates including prompt processing overhead.
- Power is average board `VDD_IN` while the model is generating.

## TensorRT via trtexec (prefill-only microbenchmark)

| Model | Tokens/s | Avg W | Tokens/s/W | USD / 1M tokens |
| --- | ---: | ---: | ---: | ---: |
| gpt2-tiny-decoder (random, ~1M params) | 223 556 | 15.45 | 14 474 | 0.0000 |

Notes:
- Model: `fxmarty/onnx-tiny-random-gpt2-with-merge` `decoder_model.onnx`,
  128-token prefill, `--shapes=input_ids:1x128,attention_mask:1x128`.
- This is a **prefill-only engine microbenchmark on a tiny random decoder**.
  It is **not comparable** to Ollama's 3B/7B end-to-end coding models.
- It proves the TensorRT backend + harness works and shows how fast the
  Jetson's TensorRT engine can push a small transformer graph.

## Why the Qwen ONNX exports did not work with TensorRT

`onnx-community/Qwen2.5-Coder-0.5B-Instruct` ONNX files (int8, fp16, and base)
use `com.microsoft` fused ops (`MultiHeadAttention`,
`SkipSimplifiedLayerNormalization`) or asymmetric INT8 quantization that the
TensorRT parser cannot import without extra plugins. A future experimental
step would be to export a Qwen model with standard ONNX ops (e.g. via
`torch.onnx.export` with `attn_implementation="eager"`) and benchmark that.

## How to reproduce

```sh
# Ollama baseline
cargo run --release -- experimental bench --engines ollama \
  --model qwen2.5-coder:3b --max-tokens 128 --iterations 3

# TensorRT microbenchmark
mkdir -p experimental/models
curl -L -o experimental/models/gpt2-tiny-decoder-plain.onnx \
  "https://huggingface.co/fxmarty/onnx-tiny-random-gpt2-with-merge/resolve/main/decoder_model.onnx?download=true"

cargo run --release -- experimental bench --engines tensorrt \
  --model experimental/models/gpt2-tiny-decoder-plain.onnx \
  --seq-len 128 --avg-runs 20 \
  --trt-shapes "input_ids:1x128,attention_mask:1x128"
```
