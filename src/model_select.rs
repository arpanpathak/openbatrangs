//! Model auto-selection, validation, and fallback pulling.

use crate::cli::ModelPrefs;
use crate::models;
use crate::models::ModelScore;
use crate::ollama::OllamaClient;
use anyhow::{anyhow, bail, Result};

/// Maximum number of auto-pull retries while selecting a model.
const MODEL_SELECT_MAX_ATTEMPTS: u32 = 3;

/// Models scoring below this are considered unsuitable for agentic coding.
const GOOD_MODEL_SCORE_THRESHOLD: f64 = 60.0;

/// Fallback context length when Ollama metadata is missing.
const FALLBACK_CONTEXT_LENGTH: u64 = 8_192;

/// Suffix of Ollama metadata keys that contain the context length.
const CONTEXT_LENGTH_KEY_SUFFIX: &str = ".context_length";

/// Fraction of total system memory treated as the model memory budget.
const MEMORY_BUDGET_NUMERATOR: u64 = 3;
const MEMORY_BUDGET_DENOMINATOR: u64 = 4;

/// Resolve a model from an optional explicit slot or auto-selection.
///
/// # Arguments
/// - `client`: Ollama HTTP client.
/// - `model_slot`: explicit model tag, if any.
/// - `prefs`: model selection preferences.
/// - `mem_budget`: usable memory in bytes.
/// - `on_status`: callback for progress messages (pull/selection).
///
/// # Returns
/// The selected model score.
pub(crate) async fn resolve_model(
    client: &OllamaClient,
    model_slot: &Option<String>,
    prefs: &ModelPrefs,
    mem_budget: u64,
    on_status: &(dyn Fn(&str) + Sync),
) -> Result<ModelScore> {
    match model_slot {
        Some(name) => {
            let explicit = ModelPrefs {
                model: Some(name.clone()),
                ..prefs.clone()
            };
            select_model(client, &explicit, mem_budget, on_status).await
        }
        None => select_model(client, prefs, mem_budget, on_status).await,
    }
}

/// Select the best model, auto-pulling a fallback when needed.
async fn select_model(
    client: &OllamaClient,
    prefs: &ModelPrefs,
    mem_budget: u64,
    on_status: &(dyn Fn(&str) + Sync),
) -> Result<ModelScore> {
    let mut attempts = 0u32;
    loop {
        attempts += 1;
        if attempts > MODEL_SELECT_MAX_ATTEMPTS {
            bail!("could not find a suitable model after auto-pulling");
        }

        if let Some(explicit) = &prefs.model {
            return select_explicit_model(client, explicit, prefs.min_context, mem_budget).await;
        }

        let tags = client.tags().await?;
        if tags.is_empty() {
            match prefs.is_auto_pull_disabled {
                true => bail!("no models installed and --no-auto-pull is set"),
                false => pull_fallback(client, mem_budget, on_status).await?,
            }
            continue;
        }

        let scored = models::score_models(&tags, mem_budget, prefs.min_context);
        match scored.first() {
            Some(best)
                if best.score < GOOD_MODEL_SCORE_THRESHOLD && !prefs.is_auto_pull_disabled =>
            {
                on_status(&format!(
                    "⚠️  Best local model '{}' scores {:.0}/100 for agentic coding.",
                    best.name, best.score
                ));
                pull_fallback(client, mem_budget, on_status).await?;
                continue;
            }
            Some(best) => return Ok(best.clone()),
            None if prefs.is_auto_pull_disabled => {
                bail!(
                    "no installed model meets --min-context={}",
                    prefs.min_context
                );
            }
            None => pull_fallback(client, mem_budget, on_status).await?,
        }
    }
}

/// Validate and score an explicitly requested model tag.
async fn select_explicit_model(
    client: &OllamaClient,
    explicit: &str,
    min_context: u64,
    mem_budget: u64,
) -> Result<ModelScore> {
    if models::looks_like_path(explicit) {
        bail!(
            "'{explicit}' looks like a file path. openBatarangs now uses Ollama model tags.\n\
             Just run `openbatrangs setup` and it will pull a coding model for you."
        );
    }
    let tags = client.tags().await?;
    let model = tags
        .iter()
        .find(|model| model.name == explicit)
        .ok_or_else(|| {
            anyhow!("Model '{explicit}' is not installed. Run `openbatrangs setup` to auto-install a model.")
        })?;
    models::score_model(model, mem_budget, min_context)
        .ok_or_else(|| anyhow!("model '{explicit}' has context below --min-context"))
}

/// Pull the recommended fallback model and report progress.
async fn pull_fallback(
    client: &OllamaClient,
    mem_budget: u64,
    on_status: &(dyn Fn(&str) + Sync),
) -> Result<()> {
    let fallback = models::recommended_fallback_model(mem_budget);
    on_status(&format!("⬇️  Pulling a better default: {fallback}"));
    client.pull(fallback, on_status).await
}

/// Resolve the context length of a model from Ollama metadata.
///
/// # Returns
/// Context length in tokens, or `FALLBACK_CONTEXT_LENGTH` if unknown.
pub(crate) async fn resolve_model_context(client: &OllamaClient, model: &str) -> Result<u64> {
    let show = client.show(model).await?;
    let context = show
        .get("model_info")
        .and_then(|info| info.as_object())
        .and_then(|object| {
            object.iter().find_map(|(key, value)| {
                if key.ends_with(CONTEXT_LENGTH_KEY_SUFFIX) {
                    value.as_u64()
                } else {
                    None
                }
            })
        })
        .or_else(|| {
            show.get("details")
                .and_then(|details| details.get("context_length"))
                .and_then(|value| value.as_u64())
        })
        .unwrap_or(FALLBACK_CONTEXT_LENGTH);
    Ok(context)
}

/// Compute the usable model memory budget from total system memory.
pub(crate) fn calculate_memory_budget() -> u64 {
    models::total_system_memory_bytes() * MEMORY_BUDGET_NUMERATOR / MEMORY_BUDGET_DENOMINATOR
}
