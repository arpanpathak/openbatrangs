use crate::ollama::OllamaModel;
use std::path::Path;

#[derive(Debug, Clone)]
pub struct ModelScore {
    pub name: String,
    pub size_bytes: u64,
    pub parameter_size: String,
    pub context_length: u64,
    pub quantization: String,
    pub score: f64,
    pub reasons: Vec<String>,
}

/// Read total system memory from /proc/meminfo (Linux). Falls back to 8 GiB.
pub fn total_system_memory_bytes() -> u64 {
    let content = std::fs::read_to_string("/proc/meminfo").unwrap_or_default();
    for line in content.lines() {
        if let Some(rest) = line.strip_prefix("MemTotal:") {
            let kb: u64 = rest
                .trim()
                .split_whitespace()
                .next()
                .and_then(|v| v.parse().ok())
                .unwrap_or(8 * 1024 * 1024);
            return kb * 1024;
        }
    }
    8 * 1024 * 1024 * 1024
}

fn parse_parameter_size(s: &str) -> f64 {
    let s = s.trim().to_ascii_lowercase();
    let number: f64 = s
        .chars()
        .take_while(|c| c.is_ascii_digit() || *c == '.')
        .collect::<String>()
        .parse()
        .unwrap_or(0.0);
    if s.contains('m') {
        number / 1000.0
    } else {
        number
    }
}

fn coding_bonus(name: &str) -> f64 {
    let n = name.to_ascii_lowercase();
    if n.contains("coder") || n.contains("deepseek") || n.contains("starcoder") {
        1.0
    } else if n.contains("code") || n.contains("devstral") || n.contains("gpt-oss") {
        0.9
    } else if n.contains("qwen3") || n.contains("qwen2.5") || n.contains("llama3.3") {
        0.6
    } else if n.contains("phi") || n.contains("gemma") || n.contains("mistral") {
        0.5
    } else {
        0.3
    }
}

fn quant_bonus(q: &str) -> f64 {
    let q = q.to_ascii_uppercase();
    if q.contains("Q4_K_M") || q.contains("Q4_K_S") {
        1.0
    } else if q.contains("Q5") {
        0.9
    } else if q.contains("Q4_0") {
        0.85
    } else if q.contains("Q6") {
        0.8
    } else if q.contains("Q8") {
        0.75
    } else if q.contains("F16") || q.contains("FP16") {
        0.5
    } else {
        0.8
    }
}

pub fn score_model(model: &OllamaModel, mem_budget: u64, min_context: u64) -> Option<ModelScore> {
    let details = model.details.as_ref()?;
    let context = details.context_length.unwrap_or(4096).max(2048);
    let params_str = details
        .parameter_size
        .clone()
        .unwrap_or_else(|| "unknown".to_string());
    let quant = details
        .quantization_level
        .clone()
        .unwrap_or_else(|| "unknown".to_string());

    if context < min_context {
        return None;
    }

    let params_b = parse_parameter_size(&params_str);
    let mut reasons = Vec::new();

    // Memory fit: prefer models that comfortably fit in the memory budget.
    let mem_factor = if model.size == 0 {
        0.5
    } else if model.size <= mem_budget / 2 {
        reasons.push(format!(
            "fits comfortably ({:.1} GB / {:.1} GB budget)",
            model.size as f64 / 1e9,
            mem_budget as f64 / 1e9
        ));
        1.0
    } else if model.size <= mem_budget {
        reasons.push("fits but leaves little headroom".to_string());
        0.6
    } else {
        reasons.push(format!(
            "model file ({:.1} GB) exceeds memory budget ({:.1} GB)",
            model.size as f64 / 1e9,
            mem_budget as f64 / 1e9
        ));
        0.0
    };

    // Parameter sweet spot for agentic coding on edge devices.
    let size_factor = match params_b {
        p if p >= 1.0 && p < 4.0 => 0.7,
        p if p >= 4.0 && p < 8.0 => 1.0,
        p if p >= 8.0 && p < 14.0 => 0.8,
        p if p >= 14.0 => 0.4,
        _ => 0.5,
    };

    // Context: 32k is a great target for agentic coding.
    let ctx_factor = (context as f64 / 32768.0).min(1.0);

    let coding = coding_bonus(&model.name);
    if coding >= 0.9 {
        reasons.push("strong coding model name".to_string());
    }
    let quant_label = quant.clone();
    let quant = quant_bonus(&quant_label);

    let score =
        mem_factor * 0.45 + size_factor * 0.2 + coding * 0.2 + ctx_factor * 0.1 + quant * 0.05;

    Some(ModelScore {
        name: model.name.clone(),
        size_bytes: model.size,
        parameter_size: params_str,
        context_length: context,
        quantization: quant_label,
        score: score * 100.0,
        reasons,
    })
}

pub fn score_models(models: &[OllamaModel], mem_budget: u64, min_context: u64) -> Vec<ModelScore> {
    let mut scored: Vec<ModelScore> = models
        .iter()
        .filter_map(|m| score_model(m, mem_budget, min_context))
        .collect();
    scored.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    scored
}

/// A safe default coding model to auto-pull when nothing good is installed.
/// Prefers the faster 3B model on Jetson-class 16GB devices for snappy UX.
pub fn recommended_fallback_model(mem_budget: u64) -> &'static str {
    if mem_budget >= 20 * 1024 * 1024 * 1024 {
        "qwen2.5-coder:7b"
    } else {
        "qwen2.5-coder:3b"
    }
}

/// True if a model name looks like a local file path rather than an Ollama tag.
pub fn looks_like_path(name: &str) -> bool {
    name.ends_with(".gguf") || name.contains('/') || name.contains('\\') || Path::new(name).exists()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_parameter_sizes() {
        assert_eq!(parse_parameter_size("7.6B"), 7.6);
        assert!((parse_parameter_size("873.44M") - 0.87344).abs() < 1e-9);
        assert_eq!(parse_parameter_size("unknown"), 0.0);
    }

    #[test]
    fn coding_bonus_recognizes_coders() {
        assert_eq!(coding_bonus("qwen2.5-coder:7b"), 1.0);
        assert_eq!(coding_bonus("deepseek-coder:6.7b"), 1.0);
        assert!(coding_bonus("llama3.2:3b") < 1.0);
    }
}
