# Model Registy

> Model registry will help you to pull models from hugging face and modidy its parameters, which then can be used to server your own variant.

# Steps to create custom model

1. Create your model directory `model_name:param_size-additional_info`. Example `mkdir qwen2.5-coder:14b-nonlazyaf`
2. Go to the directory :- `cd 
3. Create a model file and add these

```dockerfile
FROM qwen2.5-coder:14b

# Force long outputs
PARAMETER num_predict 8192
PARAMETER num_ctx 8192
PARAMETER temperature 0.2

# Bake the anti-lazy prompt directly into the model's system prompt
SYSTEM """You are an expert software architect and systems programmer. 
You write complete, production-ready code. 
NEVER use placeholders like '// ... rest of code', '// TODO', or 'implement this here'. 
Always write the full, generous, and complete implementation. 
Explain OS-level and CS concepts deeply."""

# Build & Run

```sh
ollama create qwen2.5-coder:14b-nonlazy -f Modelfile
ollama run qwen2.5-coder:14b-nonlazy
```

