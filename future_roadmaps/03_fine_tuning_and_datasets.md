# 03 - Fine-tuning and Datasets

**Status:** idea / draft

## Vision

Turn openBatarangs into a complete local-AI training playground: collect data
from real agent sessions, curate datasets, fine-tune small models, and evaluate
the results — all on edge hardware.

## Why this attracts people

- Fine-tuning is the next step after "chat with a local model".
- Real agent logs are a unique data source nobody else has.
- The repo already has a `fine_tuning_datasets/` directory — this formalizes it.
- "Train your own coding agent on your own machine" is a strong hook.

## Cool drafts

### 1. Session-to-dataset pipeline

```bash
openbatrangs dataset export --from-session "last 50"
openbatrangs dataset stats
```

- Automatically convert successful agent trajectories into instruction pairs
- Filter low-quality turns with score thresholds
- Deduplicate and tokenize locally

### 2. Curated coding datasets

- Qwen/Llama/Gemma-compatible chat templates
- Edge-device-focused tasks: CUDA kernels, embedded C, Python optimization
- Hard-negative examples from failed agent runs

### 3. Local fine-tuning command

```bash
openbatrangs train --model qwen2.5-coder:3b --dataset ./data
```

- Uses Ollama's training support or a bundled LoRA tool
- Memory-aware defaults for 8/16 GB devices
- Live loss/perplexity dashboard in the TUI

### 4. Evaluation leaderboard

- Run the same benchmark suite across models before/after fine-tuning
- Track pass@1, edit distance, token efficiency
- Publish results as a markdown report: `openbatrangs eval report`

## Roadmap sketch

| Phase | Deliverable |
|---|---|
| 1 | Export agent sessions to JSONL |
| 2 | Dataset stats, filtering, and dedup |
| 3 | LoRA fine-tune command for small models |
| 4 | Benchmark and eval report generator |
| 5 | Community dataset hub + leaderboard |

## Open questions

- Which training stack should we integrate (Ollama, llama.cpp, axolotl)?
- How much data is enough to make a visible improvement on a 3B model?
- Should evaluation benchmarks be public for community contributions?
