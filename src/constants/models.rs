//! Model scoring and auto-selection tuning constants.

/// Fallback memory size used when `/proc/meminfo` is unavailable (8 GiB).
pub const FALLBACK_SYSTEM_MEMORY_BYTES: u64 = 8 * 1024 * 1024 * 1024;

/// Minimum context length assumed for a model with unknown context.
pub const DEFAULT_CONTEXT_LENGTH: u64 = 4_096;

/// Context length clamped to at least this value.
pub const MIN_CONTEXT_LENGTH: u64 = 2_048;

/// Context length considered "ideal" for agentic coding (32K).
pub const IDEAL_CONTEXT_LENGTH: f64 = 32_768.0;

/// Lower bound (billions of parameters) for the "small" model size tier.
pub const MIN_SMALL_MODEL_B: f64 = 1.0;
/// Upper bound (billions of parameters) for the "small" model size tier.
pub const MAX_SMALL_MODEL_B: f64 = 4.0;
/// Upper bound (billions of parameters) for the "medium" model size tier.
pub const MAX_MEDIUM_MODEL_B: f64 = 8.0;
/// Upper bound (billions of parameters) for the "large" model size tier.
pub const MAX_LARGE_MODEL_B: f64 = 14.0;

/// Weight for memory-fit scoring: how well the model fits in available RAM/VRAM.
pub const WEIGHT_MEMORY: f64 = 0.45;
/// Weight for parameter-size scoring: prefers mid-range models for balanced speed/quality.
pub const WEIGHT_SIZE: f64 = 0.20;
/// Weight for coding-ability scoring based on model name heuristics.
pub const WEIGHT_CODING: f64 = 0.20;
/// Weight for context-length scoring: prefers models with longer context windows.
pub const WEIGHT_CONTEXT: f64 = 0.10;
/// Weight for quantization-quality scoring based on the GGUF quant label.
pub const WEIGHT_QUANTIZATION: f64 = 0.05;

/// Score multiplier when the model's memory footprint cannot be determined.
pub const MEMORY_FACTOR_UNKNOWN: f64 = 0.5;
/// Score multiplier when the model fits comfortably within the memory budget.
pub const MEMORY_FACTOR_COMFORTABLE: f64 = 1.0;
/// Score multiplier when the model fits but leaves little headroom.
pub const MEMORY_FACTOR_TIGHT: f64 = 0.6;
/// Score multiplier when the model exceeds the available memory budget.
pub const MEMORY_FACTOR_OVERFLOW: f64 = 0.0;

/// Threshold for "comfortably fits": model size must be at most half the budget.
pub const MEMORY_COMFORT_DIVISOR: u64 = 2;

/// Score multiplier for small models (1B–4B params): fast but limited capability.
pub const SIZE_FACTOR_SMALL: f64 = 0.7;
/// Score multiplier for medium models (4B–8B params): best speed/quality balance.
pub const SIZE_FACTOR_MEDIUM: f64 = 1.0;
/// Score multiplier for large models (8B–14B params): capable but slower.
pub const SIZE_FACTOR_LARGE: f64 = 0.8;
/// Score multiplier for huge models (>14B params): often too slow for interactive use.
pub const SIZE_FACTOR_HUGE: f64 = 0.4;
/// Score multiplier when the model's parameter count cannot be determined.
pub const SIZE_FACTOR_UNKNOWN: f64 = 0.5;

/// Score threshold for considering a model "strongly coding".
pub const STRONG_CODING_SCORE: f64 = 0.9;

/// Score is stored as 0..100 instead of 0..1.
pub const SCORE_SCALE: f64 = 100.0;

/// Memory threshold (bytes) above which the 7B fallback model is preferred.
pub const FALLBACK_7B_MEMORY_THRESHOLD_BYTES: u64 = 20 * 1024 * 1024 * 1024;

/// Coding-keyword heuristics: `(substring, bonus)`.
pub const CODING_KEYWORDS: &[(&str, f64)] = &[
    ("coder", 1.0),
    ("deepseek", 1.0),
    ("starcoder", 1.0),
    ("code", 0.9),
    ("devstral", 0.9),
    ("gpt-oss", 0.9),
    ("qwen3", 0.6),
    ("qwen2.5", 0.6),
    ("llama3.3", 0.6),
    ("phi", 0.5),
    ("gemma", 0.5),
    ("mistral", 0.5),
];

/// Bonus for model names that do not match any known coding keyword.
pub const CODING_FALLBACK_BONUS: f64 = 0.3;

/// Quantization-quality heuristics: `(substring, bonus)`.
pub const QUANT_KEYWORDS: &[(&str, f64)] = &[
    ("Q4_K_M", 1.0),
    ("Q4_K_S", 1.0),
    ("Q5", 0.9),
    ("Q4_0", 0.85),
    ("Q6", 0.8),
    ("Q8", 0.75),
    ("F16", 0.5),
    ("FP16", 0.5),
];

/// Bonus for quantization labels that do not match any known quality keyword.
pub const QUANT_FALLBACK_BONUS: f64 = 0.8;

/// 7B fallback model tag, preferred when enough memory is available.
pub const FALLBACK_MODEL_7B: &str = "qwen2.5-coder:7b";
/// 3B fallback model tag, used on memory-constrained devices.
pub const FALLBACK_MODEL_3B: &str = "qwen2.5-coder:3b";

/// Maximum number of auto-pull retries while selecting a model.
pub const MODEL_SELECT_MAX_ATTEMPTS: u32 = 3;

/// Models scoring below this are considered unsuitable for agentic coding.
pub const GOOD_MODEL_SCORE_THRESHOLD: f64 = 60.0;

/// Fallback context length when Ollama metadata is missing.
pub const FALLBACK_CONTEXT_LENGTH: u64 = 8_192;

/// Suffix of Ollama metadata keys that contain the context length.
pub const CONTEXT_LENGTH_KEY_SUFFIX: &str = ".context_length";

/// Numerator of the memory budget fraction (3/4 of total RAM is available for models).
pub const MEMORY_BUDGET_NUMERATOR: u64 = 3;
/// Denominator of the memory budget fraction (3/4 of total RAM is available for models).
pub const MEMORY_BUDGET_DENOMINATOR: u64 = 4;

/// Number of bytes in a gigabyte, used for human-readable formatting.
pub const BYTES_PER_GIGABYTE: f64 = 1_000_000_000.0;

/// Number of bytes in a kibibyte, used when parsing `/proc/meminfo`.
pub const BYTES_PER_KIB: u64 = 1_024;

/// Conversion factor from millions of parameters to billions (873.44M -> 0.87344B).
pub const PARAMETERS_MILLION_DIVISOR: f64 = 1_000.0;

/// Divisor used to render a context length in thousands (32768 -> 32K).
pub const CONTEXT_LENGTH_KILO_DIVISOR: u64 = 1_000;
