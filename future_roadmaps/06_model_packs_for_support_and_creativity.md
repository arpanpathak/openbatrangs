# 06 - Model Packs: Support, Therapy, SVG, and ASCII Art

**Status:** draft

## Vision

openBatarangs should not be coding-only. Give users one-command model packs for
emotional support, therapy-style conversations, SVG generation, and ASCII art so
the same edge device becomes a friendly creative companion too.

## Why this attracts people

- The current local models are all coding-focused; creative/support models widen the audience.
- Privacy is a huge selling point for therapy-style chats — everything stays on-device.
- SVG + ASCII art generation is a fun, demo-friendly feature that people share.
- Model packs make "pull the right model" effortless.

## Emotional support / therapy models

### Dedicated community models (pull directly from Ollama)

| Model | Pull command | Notes |
|---|---|---|
| Samantha | `ollama run samantha-mistral:7b` | Most popular companion-style model; warm, philosophical, relationship-aware |
| self | `ollama run gurubot/self` | Active listening, thoughtful questions, consistent emotional mood |
| psychologist | `ollama run ALIENTELLIGENCE/psychologist` | Empathetic psychologist roleplay (Llama-3 based) |
| mentalwellness | `ollama run ALIENTELLIGENCE/mentalwellness` | Mindfulness tips, calming activities, emotional support |
| mindpal | `ollama run ALIENTELLIGENCE/mindpal` | Mental-health screening / risk-assessment style responses |
| emotional_llama | `ollama run sebdg/emotional_llama` | Small community fine-tune for empathetic responses |

### Strong general-purpose alternatives

Sometimes a base model plus a good system prompt beats a niche fine-tune:

| Model | Pull command | Why |
|---|---|---|
| Llama 3.1 / 3.2 | `ollama run llama3.1:8b` or `llama3.2:3b` | Reliable, balanced, easy to prompt |
| Gemma 2 / 3 | `ollama run gemma2:9b` or `gemma3` | Warm and natural instruction-following |
| Mistral | `ollama run mistral:7b` | Fast on low-power devices |
| Qwen 2.5 / 3 | `ollama run qwen2.5:7b` or `qwen3` | Strong multilingual support |
| Dolphin Mistral | `ollama run dolphin-mistral` | Friendly, less filtered conversational tone |

### Hugging Face GGUF imports (specialized therapy fine-tunes)

These are not all in the default Ollama library, but GGUF versions can be pulled
directly with:

```bash
ollama run hf.co/<namespace>/<model>
```

- Thera-Space (Llama-3-8B) — reflective counseling style
- CBT-Copilot (Llama-3.2-3B) — CBT-focused local assistant
- EmoLLM / SoulChat — academic mental-health LLMs (strong in Chinese)
- Claria 1.7B — lightweight, Qwen-3 based companion
- phi-4-therapy — structured therapist-style conversations
- SQPsychLLM-8b-mistral — CBT-informed counselor roleplay

## SVG generation models

SVG is code, so **coder models** are the best starting point:

| Model | Pull command | Notes |
|---|---|---|
| Qwen 2.5 Coder | `ollama run qwen2.5-coder:7b` | Best all-round for SVG on edge hardware |
| DeepSeek Coder | `ollama run deepseek-coder:6.7b` | Strong structural code output |
| CodeStral | `ollama run codestral` | Great at long, structured generations |
| Stable Code | `ollama run stable-code:3b` | Tiny and fast for simple SVGs |
| Llama 3.1 / 3.3 | `ollama run llama3.1:8b` | Fine with a precise system prompt |
| Mistral Nemo | `ollama run mistral-nemo` | Good balance of size and creativity |

Example prompt:

```text
Generate a complete standalone SVG for a neon bat logo.
Rules:
- Only output valid SVG code inside a ```svg code block.
- Use viewBox="0 0 400 200".
- Use gradients, rounded paths, and no external assets.
- Keep the file under 40 lines.
```

## ASCII art generation models

Most general-purpose models can do ASCII art; smaller creative models work well
for quick, fun output:

| Model | Pull command | Notes |
|---|---|---|
| Qwen 2.5 | `ollama run qwen2.5:7b` | Good at precise character layouts |
| Gemma 2 / 3 | `ollama run gemma2:9b` | Creative and follows constraints well |
| Llama 3.2 | `ollama run llama3.2:3b` | Tiny enough to run alongside other packs |
| Phi 4 | `ollama run phi4` | Strong instruction following for exact art |
| Mistral | `ollama run mistral:7b` | Reliable all-rounder |

Example prompt:

```text
Draw a cat using ASCII art, exactly 20 characters wide.
Use only these characters: . o O @ # space and newline.
Do not add explanation text.
```

## Hardware fit for Jetson-class devices

| Device memory | Recommended packs |
|---|---|
| 8 GB | `samantha-mistral:7b`, `llama3.2:3b`, `stable-code:3b`, `phi4:14b` too big — use `phi3:3.8b` or `qwen2.5:3b` |
| 16 GB | `llama3.1:8b`, `gemma2:9b`, `qwen2.5-coder:7b`, `mistral-nemo`, `ALIENTELLIGENCE/psychologist` |
| 32 GB+ | larger quants (Q5/Q6), 13B+ models, multiple packs side-by-side |

## Jetson power throttling ("throttled due to overcurrent")

When a larger model like `samantha-mistral:7b` loads, power draw can spike and
the Jetson's PMIC may throttle the GPU with `throttled due to overcurrent`.
This is a hardware power limit, not a model bug.

Mitigations:

- Use a proper power supply: original 5V/4A+ USB-C (or the OEM barrel adapter).
  Cheap chargers cause overcurrent throttling.
- Check current power mode: `sudo nvpmodel -q`. Try a lower mode (e.g. 15W)
  to stay within the supply budget, or MAXN if the supply is adequate.
- Prefer smaller/quanted models: `qwen2.5-coder:3b`, `llama3.2:3b`,
  `samantha-mistral:7b` at Q4_K_M instead of higher quants.
- Lower `num_ctx` (e.g. 4096 instead of 8192) to reduce sustained memory
  bandwidth and power.
- Watch live stats: `sudo tegrastats` shows GPU throttling reason in real time.
- Avoid heavy CPU+GPU work at the same time (builds while a model is loaded).

## Ready-to-use therapist Modelfile

```dockerfile
FROM llama3.1:8b

SYSTEM """
You are a warm, non-judgmental listener. Validate feelings first,
reflect back what the user said, ask open-ended questions, and only
offer gentle reframes. Keep responses concise and caring.

Important: you are not a licensed therapist. If the user mentions
self-harm or a crisis, encourage them to contact a local helpline
immediately.
"""
```

Build and run:

```bash
ollama create therapist -f Modelfile.therapist
ollama run therapist
```

## Roadmap sketch

| Phase | Deliverable |
|---|---|
| 1 | Add `openbatrangs model pack list` and `model pack install support` |
| 2 | Bundle therapist Modelfile + crisis-response safety prompt |
| 3 | Add SVG/ASCII art example prompts to the TUI |
| 4 | Community-submitted model packs with per-device memory tags |

## Open questions

- Should support/therapy models be gated behind an explicit opt-in?
- Which ASCII art style should the default prompt target (blocky vs shaded)?
- How many models can safely stay loaded on an 8 GB device at once?
