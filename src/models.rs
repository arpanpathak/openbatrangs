//! Model auto-discovery and scoring.
//!
//! This module decides which installed Ollama model is best for agentic coding
//! on the current hardware. It balances memory fit, coding suitability,
//! parameter count, context window, and quantization quality.

use crate::constants::models::{
    BYTES_PER_GIGABYTE, CODING_FALLBACK_BONUS, CODING_KEYWORDS, DEFAULT_CONTEXT_LENGTH,
    FALLBACK_7B_MEMORY_THRESHOLD_BYTES, FALLBACK_MODEL_3B, FALLBACK_MODEL_7B, IDEAL_CONTEXT_LENGTH,
    MAX_LARGE_MODEL_B, MAX_MEDIUM_MODEL_B, MAX_SMALL_MODEL_B, MEMORY_COMFORT_DIVISOR,
    MEMORY_FACTOR_COMFORTABLE, MEMORY_FACTOR_OVERFLOW, MEMORY_FACTOR_TIGHT, MEMORY_FACTOR_UNKNOWN,
    MIN_CONTEXT_LENGTH, MIN_SMALL_MODEL_B, PARAMETERS_MILLION_DIVISOR, QUANT_FALLBACK_BONUS,
    QUANT_KEYWORDS, SCORE_SCALE, SIZE_FACTOR_HUGE, SIZE_FACTOR_LARGE, SIZE_FACTOR_MEDIUM,
    SIZE_FACTOR_SMALL, SIZE_FACTOR_UNKNOWN, STRONG_CODING_SCORE, WEIGHT_CODING, WEIGHT_CONTEXT,
    WEIGHT_MEMORY, WEIGHT_QUANTIZATION, WEIGHT_SIZE,
};
use crate::ollama::OllamaModel;
use std::path::Path;

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

/// Read total system memory from the hardware probe (defaults to `/proc`).
///
/// # Returns
/// Total physical memory in bytes, or `8 GiB` if the file cannot be read.
pub fn total_system_memory_bytes() -> u64 {
    crate::hardware::total_system_memory_bytes()
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
        number / PARAMETERS_MILLION_DIVISOR
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
    CODING_KEYWORDS
        .iter()
        .find(|(needle, _)| normalized.contains(needle))
        .map(|(_, bonus)| *bonus)
        .unwrap_or(CODING_FALLBACK_BONUS)
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
    QUANT_KEYWORDS
        .iter()
        .find(|(needle, _)| normalized.contains(needle))
        .map(|(_, bonus)| *bonus)
        .unwrap_or(QUANT_FALLBACK_BONUS)
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
    // Older Ollama tags can omit `details` entirely. Treat missing metadata as
    // "unknown" instead of silently dropping the model from scoring/listing.
    let details = model.details.as_ref();
    let context = details
        .and_then(|details| details.context_length)
        .unwrap_or(DEFAULT_CONTEXT_LENGTH)
        .max(MIN_CONTEXT_LENGTH);
    let parameter_label = details
        .and_then(|details| details.parameter_size.clone())
        .unwrap_or_else(|| "unknown".to_string());
    let quantization_label = details
        .and_then(|details| details.quantization_level.clone())
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
            model.size as f64 / BYTES_PER_GIGABYTE,
            mem_budget as f64 / BYTES_PER_GIGABYTE
        ));
        MEMORY_FACTOR_COMFORTABLE
    } else if model.size <= mem_budget {
        reasons.push("fits but leaves little headroom".to_string());
        MEMORY_FACTOR_TIGHT
    } else {
        reasons.push(format!(
            "model file ({:.1} GB) exceeds memory budget ({:.1} GB)",
            model.size as f64 / BYTES_PER_GIGABYTE,
            mem_budget as f64 / BYTES_PER_GIGABYTE
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
        FALLBACK_MODEL_7B
    } else {
        FALLBACK_MODEL_3B
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
    name.ends_with(".gguf")
        || Path::new(name).is_absolute()
        || name.starts_with('.')
        || name.contains("..")
        || name.contains('\\')
        || Path::new(name).exists()
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
    fn score_model_includes_models_without_details() {
        let model = OllamaModel {
            name: "legacy:latest".to_string(),
            size: 1_000_000_000,
            details: None,
        };
        let scored =
            score_model(&model, 16_000_000_000, 4_096).expect("missing details should score");
        assert_eq!(scored.name, "legacy:latest");
        assert_eq!(scored.parameter_size, "unknown");
        assert_eq!(scored.quantization, "unknown");
        assert!(scored.score > 0.0);
    }

    #[test]
    fn score_models_keeps_models_without_details() {
        let legacy = OllamaModel {
            name: "legacy:latest".to_string(),
            size: 1_000_000_000,
            details: None,
        };
        let modern = model("qwen2.5-coder:7b", 32_768, "7.6B", "Q4_K_M", 4_700_000_000);
        let scored = score_models(&[legacy, modern], 16_000_000_000, 4_096);
        assert_eq!(scored.len(), 2);
        assert_eq!(scored[0].name, "qwen2.5-coder:7b");
        assert_eq!(scored[1].name, "legacy:latest");
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
        assert!(looks_like_path("/tmp/foo"));
        assert!(looks_like_path("C:\\models\\foo"));
        assert!(looks_like_path("../secret"));
        assert!(!looks_like_path("qwen2.5-coder:7b"));
    }

    #[test]
    fn namespaced_ollama_tags_are_not_paths() {
        assert!(!looks_like_path("sebdg/emotional_llama:latest"));
        assert!(!looks_like_path("ALIENTELLIGENCE/psychologist"));
        assert!(!looks_like_path("models/foo"));
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
