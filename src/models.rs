//! Model auto-discovery and scoring.
//!
//! This module decides which installed Ollama model is best for agentic coding
//! on the current hardware. It balances memory fit, coding suitability,
//! parameter count, context window, and quantization quality.

use crate::constants::models::{
    BYTES_PER_GIGABYTE, CODING_FALLBACK_BONUS, CODING_KEYWORDS, DEFAULT_CONTEXT_LENGTH,
    FALLBACK_7B_MEMORY_THRESHOLD_BYTES, FALLBACK_MODEL_3B, FALLBACK_MODEL_7B, IDEAL_CONTEXT_LENGTH,
    MAX_LARGE_MODEL_B, MEMORY_COMFORT_DIVISOR, MEMORY_FACTOR_COMFORTABLE, MEMORY_FACTOR_OVERFLOW,
    MEMORY_FACTOR_TIGHT, MEMORY_FACTOR_UNKNOWN, MIN_CONTEXT_LENGTH, PARAMETER_SIZE_TIERS,
    PARAMETERS_MILLION_DIVISOR, QUANT_FALLBACK_BONUS, QUANT_KEYWORDS, SCORE_SCALE,
    SIZE_FACTOR_HUGE, SIZE_FACTOR_UNKNOWN, STRONG_CODING_SCORE, WEIGHT_CODING, WEIGHT_CONTEXT,
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
    let number = leading_number(&normalized);

    if normalized.contains('m') {
        return number / PARAMETERS_MILLION_DIVISOR;
    }

    number
}

/// The number at the start of `text`, or 0 when it does not start with one.
fn leading_number(text: &str) -> f64 {
    let digits: String = text.chars().take_while(is_number_character).collect();

    digits.parse().unwrap_or(0.0)
}

/// Whether `character` can be part of a decimal number: a digit or a dot.
fn is_number_character(character: &char) -> bool {
    character.is_ascii_digit() || *character == '.'
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
    let context = context_length(model);

    if context < min_context {
        return None;
    }

    Some(scored_model(model, context, mem_budget))
}

/// The model's context window, with defaults for missing metadata.
///
/// Older Ollama tags can omit `details` entirely. Missing metadata counts as
/// unknown instead of dropping the model from scoring and listing.
fn context_length(model: &OllamaModel) -> u64 {
    model
        .details
        .as_ref()
        .and_then(|details| details.context_length)
        .unwrap_or(DEFAULT_CONTEXT_LENGTH)
        .max(MIN_CONTEXT_LENGTH)
}

/// The parameter-count label, or `unknown`.
fn parameter_label(model: &OllamaModel) -> String {
    model
        .details
        .as_ref()
        .and_then(|details| details.parameter_size.clone())
        .unwrap_or_else(|| "unknown".to_string())
}

/// The quantization label, or `unknown`.
fn quantization_label(model: &OllamaModel) -> String {
    model
        .details
        .as_ref()
        .and_then(|details| details.quantization_level.clone())
        .unwrap_or_else(|| "unknown".to_string())
}

/// The weighted score of a model whose context window is large enough.
fn scored_model(model: &OllamaModel, context: u64, mem_budget: u64) -> ModelScore {
    let parameter_size = parameter_label(model);
    let quantization = quantization_label(model);
    let mut reasons: Vec<String> = Vec::new();

    let memory_factor = memory_fit_factor(model, mem_budget, &mut reasons);
    let size_factor = parameter_size_factor(parse_parameter_size(&parameter_size));
    let coding = coding_bonus(&model.name);
    reasons.extend(coding_reason(coding));
    let context_factor = (context as f64 / IDEAL_CONTEXT_LENGTH).min(1.0);
    let quantization_factor = quant_bonus(&quantization);

    let score = memory_factor * WEIGHT_MEMORY
        + size_factor * WEIGHT_SIZE
        + coding * WEIGHT_CODING
        + context_factor * WEIGHT_CONTEXT
        + quantization_factor * WEIGHT_QUANTIZATION;

    ModelScore {
        name: model.name.clone(),
        size_bytes: model.size,
        parameter_size,
        context_length: context,
        quantization,
        score: score * SCORE_SCALE,
        reasons,
    }
}

/// The reason shown for a model whose name marks it as a coding model.
fn coding_reason(coding: f64) -> Option<String> {
    let is_strong_coder = coding >= STRONG_CODING_SCORE;

    is_strong_coder.then(|| "strong coding model name".to_string())
}

/// Compute the memory-fit factor for a model and append human-readable reasons.
fn memory_fit_factor(model: &OllamaModel, mem_budget: u64, reasons: &mut Vec<String>) -> f64 {
    if model.size == 0 {
        return MEMORY_FACTOR_UNKNOWN;
    }

    let (factor, reason) = memory_fit(model.size, mem_budget);
    reasons.push(reason);

    factor
}

/// The memory-fit factor for a model file of `size` bytes, and its reason.
fn memory_fit(size: u64, mem_budget: u64) -> (f64, String) {
    let fits_comfortably = size <= mem_budget / MEMORY_COMFORT_DIVISOR;
    let fits = size <= mem_budget;

    match (fits_comfortably, fits) {
        (true, _) => (
            MEMORY_FACTOR_COMFORTABLE,
            comfortable_reason(size, mem_budget),
        ),
        (false, true) => (
            MEMORY_FACTOR_TIGHT,
            "fits but leaves little headroom".to_string(),
        ),
        (false, false) => (MEMORY_FACTOR_OVERFLOW, overflow_reason(size, mem_budget)),
    }
}

/// The reason for a model that fits in half the memory budget.
fn comfortable_reason(size: u64, mem_budget: u64) -> String {
    format!(
        "fits comfortably ({:.1} GB / {:.1} GB budget)",
        size as f64 / BYTES_PER_GIGABYTE,
        mem_budget as f64 / BYTES_PER_GIGABYTE
    )
}

/// The reason for a model larger than the memory budget.
fn overflow_reason(size: u64, mem_budget: u64) -> String {
    format!(
        "model file ({:.1} GB) exceeds memory budget ({:.1} GB)",
        size as f64 / BYTES_PER_GIGABYTE,
        mem_budget as f64 / BYTES_PER_GIGABYTE
    )
}

/// Parameter-size factor favoring 4B-8B coding models on edge devices.
fn parameter_size_factor(parameters_b: f64) -> f64 {
    if parameters_b >= MAX_LARGE_MODEL_B {
        return SIZE_FACTOR_HUGE;
    }

    PARAMETER_SIZE_TIERS
        .iter()
        .find(|(range, _factor)| range.contains(&parameters_b))
        .map(|(_range, factor)| *factor)
        .unwrap_or(SIZE_FACTOR_UNKNOWN)
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
/// `true` if the name ends with `.gguf`, is absolute, starts with a dot,
/// contains `..` or a backslash, or exists as a file on disk.
pub fn looks_like_path(name: &str) -> bool {
    let path = Path::new(name);
    let path_signs = [
        name.ends_with(".gguf"),
        path.is_absolute(),
        name.starts_with('.'),
        name.contains(".."),
        name.contains('\\'),
    ];

    path_signs.contains(&true) || path.exists()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::sample_model;
    use anyhow::{Context, Result};

    /// A model as old Ollama versions list it: a name and a size, no details.
    fn legacy_model() -> OllamaModel {
        OllamaModel {
            name: "legacy:latest".to_string(),
            size: 1_000_000_000,
            details: None,
        }
    }

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

    #[test]
    fn score_model_returns_none_below_min_context() -> Result<()> {
        let model = sample_model("tiny:1b", 2_048, "1B", "Q4_0", 500_000_000)?;

        assert!(score_model(&model, 8_000_000_000, 8_192).is_none());
        Ok(())
    }

    #[test]
    fn score_model_includes_models_without_details() -> Result<()> {
        let scored = score_model(&legacy_model(), 16_000_000_000, 4_096)
            .context("missing details should score")?;

        assert_eq!(scored.name, "legacy:latest");
        assert_eq!(scored.parameter_size, "unknown");
        assert_eq!(scored.quantization, "unknown");
        assert!(scored.score > 0.0);
        Ok(())
    }

    #[test]
    fn score_models_keeps_models_without_details() -> Result<()> {
        let modern = sample_model("qwen2.5-coder:7b", 32_768, "7.6B", "Q4_K_M", 4_700_000_000)?;

        let scored = score_models(&[legacy_model(), modern], 16_000_000_000, 4_096);

        assert_eq!(scored.len(), 2);
        assert_eq!(scored[0].name, "qwen2.5-coder:7b");
        assert_eq!(scored[1].name, "legacy:latest");
        Ok(())
    }

    #[test]
    fn score_model_returns_some_when_context_meets_minimum() -> Result<()> {
        let model = sample_model("qwen2.5-coder:7b", 32_768, "7.6B", "Q4_K_M", 4_700_000_000)?;

        let scored = score_model(&model, 16_000_000_000, 8_192).context("model should score")?;

        assert_eq!(scored.name, "qwen2.5-coder:7b");
        assert!(scored.score > 0.0);
        assert!(scored.score <= 100.0);
        assert!(
            scored
                .reasons
                .iter()
                .any(|reason| reason.contains("strong coding"))
        );
        Ok(())
    }

    #[test]
    fn score_models_sorts_best_first() -> Result<()> {
        let good = sample_model("good:7b", 32_768, "7B", "Q4_K_M", 4_000_000_000)?;
        let meh = sample_model("meh:3b", 8_192, "3B", "Q4_0", 2_000_000_000)?;

        let scored = score_models(&[meh, good], 16_000_000_000, 8_192);

        assert_eq!(scored[0].name, "good:7b");
        assert_eq!(scored[1].name, "meh:3b");
        Ok(())
    }

    #[test]
    fn memory_fit_depends_on_the_share_of_the_budget() {
        assert_eq!(memory_fit(4, 10).0, MEMORY_FACTOR_COMFORTABLE);
        assert_eq!(memory_fit(8, 10).0, MEMORY_FACTOR_TIGHT);
        assert_eq!(memory_fit(12, 10).0, MEMORY_FACTOR_OVERFLOW);
        assert!(memory_fit(12, 10).1.contains("exceeds memory budget"));
    }

    #[test]
    fn model_without_a_size_has_unknown_memory_fit() {
        let model = OllamaModel {
            size: 0,
            ..legacy_model()
        };
        let mut reasons = Vec::new();

        assert_eq!(
            memory_fit_factor(&model, 10, &mut reasons),
            MEMORY_FACTOR_UNKNOWN
        );
        assert!(reasons.is_empty());
    }

    #[test]
    fn parameter_size_factor_prefers_medium_models() {
        assert_eq!(parameter_size_factor(0.5), SIZE_FACTOR_UNKNOWN);
        assert_eq!(parameter_size_factor(3.0), 0.7);
        assert_eq!(parameter_size_factor(7.0), 1.0);
        assert_eq!(parameter_size_factor(13.0), 0.8);
        assert_eq!(parameter_size_factor(70.0), SIZE_FACTOR_HUGE);
    }

    #[test]
    fn looks_like_path_detects_paths() {
        assert!(looks_like_path("./model.gguf"));
        assert!(looks_like_path("/tmp/foo"));
        assert!(looks_like_path("C:\\models\\foo"));
        assert!(looks_like_path("../secret"));
        assert!(looks_like_path("Cargo.toml"));
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
