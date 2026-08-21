//! Model scoring and auto-selection tuning constants.

/// Fallback memory size used when `/proc/meminfo` is unavailable (8 GiB).
pub const FALLBACK_SYSTEM_MEMORY_BYTES: u64 = 8 * 1024 * 1024 * 1024;

/// Minimum context length assumed for a model with unknown context.
pub const DEFAULT_CONTEXT_LENGTH: u64 = 4_096;

/// Context length clamped to at least this value.
pub const MIN_CONTEXT_LENGTH: u64 = 2_048;

/// Context length considered "ideal" for agentic coding (32K).
pub const IDEAL_CONTEXT_LENGTH: f64 = 32_768.0;

/// Parameter-size sweet spots (in billions of parameters).
pub const MIN_SMALL_MODEL_B: f64 = 1.0;
pub const MAX_SMALL_MODEL_B: f64 = 4.0;
pub const MAX_MEDIUM_MODEL_B: f64 = 8.0;
pub const MAX_LARGE_MODEL_B: f64 = 14.0;

/// Scoring weights for each model quality factor. Weights sum to 1.0.
pub const WEIGHT_MEMORY: f64 = 0.45;
pub const WEIGHT_SIZE: f64 = 0.20;
pub const WEIGHT_CODING: f64 = 0.20;
pub const WEIGHT_CONTEXT: f64 = 0.10;
pub const WEIGHT_QUANTIZATION: f64 = 0.05;

/// Score multipliers for memory fit.
pub const MEMORY_FACTOR_UNKNOWN: f64 = 0.5;
pub const MEMORY_FACTOR_COMFORTABLE: f64 = 1.0;
pub const MEMORY_FACTOR_TIGHT: f64 = 0.6;
pub const MEMORY_FACTOR_OVERFLOW: f64 = 0.0;

/// Threshold for "comfortably fits": model size must be at most half the budget.
pub const MEMORY_COMFORT_DIVISOR: u64 = 2;

/// Parameter-size score multipliers.
pub const SIZE_FACTOR_SMALL: f64 = 0.7;
pub const SIZE_FACTOR_MEDIUM: f64 = 1.0;
pub const SIZE_FACTOR_LARGE: f64 = 0.8;
pub const SIZE_FACTOR_HUGE: f64 = 0.4;
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

/// Well-known Ollama tags used as auto-pull fallbacks.
pub const FALLBACK_MODEL_7B: &str = "qwen2.5-coder:7b";
pub const FALLBACK_MODEL_3B: &str = "qwen2.5-coder:3b";

/// Maximum number of auto-pull retries while selecting a model.
pub const MODEL_SELECT_MAX_ATTEMPTS: u32 = 3;

/// Models scoring below this are considered unsuitable for agentic coding.
pub const GOOD_MODEL_SCORE_THRESHOLD: f64 = 60.0;

/// Fallback context length when Ollama metadata is missing.
pub const FALLBACK_CONTEXT_LENGTH: u64 = 8_192;

/// Suffix of Ollama metadata keys that contain the context length.
pub const CONTEXT_LENGTH_KEY_SUFFIX: &str = ".context_length";

/// Fraction of total system memory treated as the model memory budget.
pub const MEMORY_BUDGET_NUMERATOR: u64 = 3;
pub const MEMORY_BUDGET_DENOMINATOR: u64 = 4;

/// Number of bytes in a gigabyte, used for human-readable formatting.
pub const BYTES_PER_GIGABYTE: f64 = 1_000_000_000.0;

/// Number of bytes in a kibibyte, used when parsing `/proc/meminfo`.
pub const BYTES_PER_KIB: u64 = 1_024;

/// Conversion factor from millions of parameters to billions (873.44M -> 0.87344B).
pub const PARAMETERS_MILLION_DIVISOR: f64 = 1_000.0;

/// Divisor used to render a context length in thousands (32768 -> 32K).
pub const CONTEXT_LENGTH_KILO_DIVISOR: u64 = 1_000;
