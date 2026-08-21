//! Model auto-discovery and scoring.
//!
//! This module decides which installed Ollama model is best for agentic coding
//! on the current hardware. It balances memory fit, coding suitability,
//! parameter count, context window, and quantization quality.

use crate::ollama::OllamaModel;
use std::path::Path;

/// Fallback memory size used when `/proc/meminfo` is unavailable (8 GiB).
const FALLBACK_SYSTEM_MEMORY_BYTES: u64 = 8 * 1024 * 1024 * 1024;

/// Minimum context length assumed for a model with unknown context.
const DEFAULT_CONTEXT_LENGTH: u64 = 4_096;
/// Context length clamped to at least this value.
const MIN_CONTEXT_LENGTH: u64 = 2_048;

/// Context length considered "ideal" for agentic coding (32K).
const IDEAL_CONTEXT_LENGTH: f64 = 32_768.0;

/// Parameter-size sweet spots (in billions of parameters).
const MIN_SMALL_MODEL_B: f64 = 1.0;
const MAX_SMALL_MODEL_B: f64 = 4.0;
const MAX_MEDIUM_MODEL_B: f64 = 8.0;
const MAX_LARGE_MODEL_B: f64 = 14.0;

/// Scoring weights for each model quality factor. Weights sum to 1.0.
const WEIGHT_MEMORY: f64 = 0.45;
const WEIGHT_SIZE: f64 = 0.20;
const WEIGHT_CODING: f64 = 0.20;
const WEIGHT_CONTEXT: f64 = 0.10;
const WEIGHT_QUANTIZATION: f64 = 0.05;

/// Score multipliers for memory fit.
const MEMORY_FACTOR_UNKNOWN: f64 = 0.5;
const MEMORY_FACTOR_COMFORTABLE: f64 = 1.0;
const MEMORY_FACTOR_TIGHT: f64 = 0.6;
const MEMORY_FACTOR_OVERFLOW: f64 = 0.0;

/// Threshold for "comfortably fits": model size must be at most half the budget.
const MEMORY_COMFORT_DIVISOR: u64 = 2;

/// Parameter-size score multipliers.
const SIZE_FACTOR_SMALL: f64 = 0.7;
const SIZE_FACTOR_MEDIUM: f64 = 1.0;
const SIZE_FACTOR_LARGE: f64 = 0.8;
const SIZE_FACTOR_HUGE: f64 = 0.4;
const SIZE_FACTOR_UNKNOWN: f64 = 0.5;

/// Score threshold for considering a model "strongly coding".
const STRONG_CODING_SCORE: f64 = 0.9;

/// Score is stored as 0..100 instead of 0..1.
const SCORE_SCALE: f64 = 100.0;

/// Memory threshold (bytes) above which the 7B fallback model is preferred.
const FALLBACK_7B_MEMORY_THRESHOLD_BYTES: u64 = 20 * 1024 * 1024 * 1024;

/// `ModelScore` summarizes why a model is (or is not) a good agentic-coding pick.
#[derive(Debug, Clone)]
pub struct ModelScore {
    /// Model tag, e.g. `qwen2.5-coder:7b`.
    pub name: String,
    /// Model file size in bytes.
    pub size_bytes: u64,
    /// Human-readable parameter count, e.g. `7.6B`.
    pub parameter_size: String,
    /// Maximum context window supported by the model.
    pub context_length: u64,
    /// Quantization label, e.g. `Q4_K_M`.
    pub quantization: String,
    /// Final score from 0 to 100. Higher is better.
    pub score: f64,
    /// Human-readable reasons for the score, shown to the user.
    pub reasons: Vec<String>,
}

/// Read total system memory from `/proc/meminfo` (Linux).
///
/// # Returns
/// Total physical memory in bytes, or `8 GiB` if the file cannot be read.
pub fn total_system_memory_bytes() -> u64 {
    let content = std::fs::read_to_string("/proc/meminfo").unwrap_or_default();
    for line in content.lines() {
        if let Some(rest) = line.strip_prefix("MemTotal:") {
            let kilobytes: u64 = rest
                .split_whitespace()
                .next()
                .and_then(|value| value.parse().ok())
                .unwrap_or(FALLBACK_SYSTEM_MEMORY_BYTES / 1024);
            return kilobytes * 1024;
        }
    }
    FALLBACK_SYSTEM_MEMORY_BYTES
}

/// Parse a parameter-size label like `"7.6B"` or `"873.44M"` into billions.
///
/// # Arguments
/// - `label`: raw label from Ollama metadata.
///
/// # Returns
/// Parameter count in billions of parameters (e.g. `7.6`, `0.87344`).
fn parse_parameter_size(label: &str) -> f64 {
    let normalized = label.trim().to_ascii_lowercase();
    let number: f64 = normalized
        .chars()
        .take_while(|character| character.is_ascii_digit() || *character == '.')
        .collect::<String>()
        .parse()
        .unwrap_or(0.0);
    if normalized.contains('m') {
        number / 1000.0
    } else {
        number
    }
}

/// Heuristic bonus for model names that indicate coding specialization.
///
/// # Arguments
/// - `name`: model tag, e.g. `deepseek-coder:6.7b`.
///
/// # Returns
/// A value from 0.3 to 1.0; higher means more coding-oriented.
fn coding_bonus(name: &str) -> f64 {
    let normalized = name.to_ascii_lowercase();
    if normalized.contains("coder")
        || normalized.contains("deepseek")
        || normalized.contains("starcoder")
    {
        1.0
    } else if normalized.contains("code")
        || normalized.contains("devstral")
        || normalized.contains("gpt-oss")
    {
        0.9
    } else if normalized.contains("qwen3")
        || normalized.contains("qwen2.5")
        || normalized.contains("llama3.3")
    {
        0.6
    } else if normalized.contains("phi")
        || normalized.contains("gemma")
        || normalized.contains("mistral")
    {
        0.5
    } else {
        0.3
    }
}

/// Heuristic bonus for quantization quality.
///
/// # Arguments
/// - `quantization`: quantization label, e.g. `Q4_K_M`.
///
/// # Returns
/// A value from 0.5 to 1.0; higher means better quality-to-size tradeoff.
fn quant_bonus(quantization: &str) -> f64 {
    let normalized = quantization.to_ascii_uppercase();
    if normalized.contains("Q4_K_M") || normalized.contains("Q4_K_S") {
        1.0
    } else if normalized.contains("Q5") {
        0.9
    } else if normalized.contains("Q4_0") {
        0.85
    } else if normalized.contains("Q6") {
        0.8
    } else if normalized.contains("Q8") {
        0.75
    } else if normalized.contains("F16") || normalized.contains("FP16") {
        0.5
    } else {
        0.8
    }
}

/// Score a single model against a memory budget and minimum context window.
///
/// # Arguments
/// - `model`: installed model metadata from Ollama.
/// - `mem_budget`: usable memory in bytes.
/// - `min_context`: minimum acceptable context length.
///
/// # Returns
/// `Some(ModelScore)` if the model meets the context requirement,
/// otherwise `None`.
pub fn score_model(model: &OllamaModel, mem_budget: u64, min_context: u64) -> Option<ModelScore> {
    let details = model.details.as_ref()?;
    let context = details
        .context_length
        .unwrap_or(DEFAULT_CONTEXT_LENGTH)
        .max(MIN_CONTEXT_LENGTH);
    let parameter_label = details
        .parameter_size
        .clone()
        .unwrap_or_else(|| "unknown".to_string());
    let quantization_label = details
        .quantization_level
        .clone()
        .unwrap_or_else(|| "unknown".to_string());

    if context < min_context {
        return None;
    }

    let parameters_b = parse_parameter_size(&parameter_label);
    let mut reasons: Vec<String> = Vec::new();

    let memory_factor = memory_fit_factor(model, mem_budget, &mut reasons);
    let size_factor = parameter_size_factor(parameters_b);
    let coding = coding_bonus(&model.name);
    if coding >= STRONG_CODING_SCORE {
        reasons.push("strong coding model name".to_string());
    }
    let context_factor = (context as f64 / IDEAL_CONTEXT_LENGTH).min(1.0);
    let quantization_factor = quant_bonus(&quantization_label);

    let score = memory_factor * WEIGHT_MEMORY
        + size_factor * WEIGHT_SIZE
        + coding * WEIGHT_CODING
        + context_factor * WEIGHT_CONTEXT
        + quantization_factor * WEIGHT_QUANTIZATION;

    Some(ModelScore {
        name: model.name.clone(),
        size_bytes: model.size,
        parameter_size: parameter_label,
        context_length: context,
        quantization: quantization_label,
        score: score * SCORE_SCALE,
        reasons,
    })
}

/// Compute the memory-fit factor for a model and append human-readable reasons.
fn memory_fit_factor(model: &OllamaModel, mem_budget: u64, reasons: &mut Vec<String>) -> f64 {
    if model.size == 0 {
        return MEMORY_FACTOR_UNKNOWN;
    }
    if model.size <= mem_budget / MEMORY_COMFORT_DIVISOR {
        reasons.push(format!(
            "fits comfortably ({:.1} GB / {:.1} GB budget)",
            model.size as f64 / 1e9,
            mem_budget as f64 / 1e9
        ));
        MEMORY_FACTOR_COMFORTABLE
    } else if model.size <= mem_budget {
        reasons.push("fits but leaves little headroom".to_string());
        MEMORY_FACTOR_TIGHT
    } else {
        reasons.push(format!(
            "model file ({:.1} GB) exceeds memory budget ({:.1} GB)",
            model.size as f64 / 1e9,
            mem_budget as f64 / 1e9
        ));
        MEMORY_FACTOR_OVERFLOW
    }
}

/// Parameter-size factor favoring 4B-8B coding models on edge devices.
fn parameter_size_factor(parameters_b: f64) -> f64 {
    match parameters_b {
        p if (MIN_SMALL_MODEL_B..MAX_SMALL_MODEL_B).contains(&p) => SIZE_FACTOR_SMALL,
        p if (MAX_SMALL_MODEL_B..MAX_MEDIUM_MODEL_B).contains(&p) => SIZE_FACTOR_MEDIUM,
        p if (MAX_MEDIUM_MODEL_B..MAX_LARGE_MODEL_B).contains(&p) => SIZE_FACTOR_LARGE,
        p if p >= MAX_LARGE_MODEL_B => SIZE_FACTOR_HUGE,
        _ => SIZE_FACTOR_UNKNOWN,
    }
}

/// Score and sort all models, best first.
///
/// # Arguments
/// - `models`: installed models from Ollama.
/// - `mem_budget`: usable memory in bytes.
/// - `min_context`: minimum acceptable context length.
///
/// # Returns
/// Models that meet the context requirement, sorted descending by score.
pub fn score_models(models: &[OllamaModel], mem_budget: u64, min_context: u64) -> Vec<ModelScore> {
    let mut scored: Vec<ModelScore> = models
        .iter()
        .filter_map(|model| score_model(model, mem_budget, min_context))
        .collect();
    scored.sort_by(|left, right| {
        right
            .score
            .partial_cmp(&left.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    scored
}

/// Pick the default coding model to auto-pull when nothing suitable is installed.
///
/// # Arguments
/// - `mem_budget`: usable memory in bytes.
///
/// # Returns
/// A well-known Ollama model tag. Prefers the faster 3B model on Jetson-class
/// 16GB devices for snappy UX, and the 7B model on larger machines.
pub fn recommended_fallback_model(mem_budget: u64) -> &'static str {
    if mem_budget >= FALLBACK_7B_MEMORY_THRESHOLD_BYTES {
        "qwen2.5-coder:7b"
    } else {
        "qwen2.5-coder:3b"
    }
}

/// True if a model name looks like a local file path rather than an Ollama tag.
///
/// # Arguments
/// - `name`: user-supplied model identifier.
///
/// # Returns
/// `true` if the name ends with `.gguf`, contains a path separator, or exists
/// as a file on disk.
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

    fn model(name: &str, context: u64, params: &str, quant: &str, size: u64) -> OllamaModel {
        serde_json::from_value(serde_json::json!({
            "name": name,
            "size": size,
            "details": {
                "parameter_size": params,
                "quantization_level": quant,
                "context_length": context,
            },
        }))
        .expect("valid model JSON")
    }

    #[test]
    fn score_model_returns_none_below_min_context() {
        let model = model("tiny:1b", 2_048, "1B", "Q4_0", 500_000_000);
        assert!(score_model(&model, 8_000_000_000, 8_192).is_none());
    }

    #[test]
    fn score_model_returns_some_when_context_meets_minimum() {
        let model = model("qwen2.5-coder:7b", 32_768, "7.6B", "Q4_K_M", 4_700_000_000);
        let scored = score_model(&model, 16_000_000_000, 8_192).unwrap();
        assert_eq!(scored.name, "qwen2.5-coder:7b");
        assert!(scored.score > 0.0);
        assert!(scored.score <= 100.0);
    }

    #[test]
    fn score_models_sorts_best_first() {
        let good = model("good:7b", 32_768, "7B", "Q4_K_M", 4_000_000_000);
        let meh = model("meh:3b", 8_192, "3B", "Q4_0", 2_000_000_000);
        let scored = score_models(&[meh.clone(), good.clone()], 16_000_000_000, 8_192);
        assert_eq!(scored[0].name, "good:7b");
        assert_eq!(scored[1].name, "meh:3b");
    }

    #[test]
    fn looks_like_path_detects_paths() {
        assert!(looks_like_path("./model.gguf"));
        assert!(looks_like_path("models/foo"));
        assert!(looks_like_path("C:\\models\\foo"));
        assert!(!looks_like_path("qwen2.5-coder:7b"));
    }

    #[test]
    fn recommended_fallback_model_chooses_bigger_model_on_large_memory() {
        let small = recommended_fallback_model(8 * 1024 * 1024 * 1024);
        let large = recommended_fallback_model(24 * 1024 * 1024 * 1024);
        assert_eq!(small, "qwen2.5-coder:3b");
        assert_eq!(large, "qwen2.5-coder:7b");
    }

    #[test]
    fn quant_bonus_prefers_q4_k_m() {
        assert!(quant_bonus("Q4_K_M") > quant_bonus("Q8_0"));
        assert!(quant_bonus("Q5_K_S") > quant_bonus("F16"));
    }
}
