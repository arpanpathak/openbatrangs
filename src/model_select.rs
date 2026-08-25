//! Model auto-selection, validation, and fallback pulling.

use crate::cli::ModelPrefs;
use crate::constants::models::{
    CONTEXT_LENGTH_KEY_SUFFIX, FALLBACK_CONTEXT_LENGTH, GOOD_MODEL_SCORE_THRESHOLD,
    MEMORY_BUDGET_DENOMINATOR, MEMORY_BUDGET_NUMERATOR, MODEL_SELECT_MAX_ATTEMPTS,
};
use crate::models;
use crate::models::ModelScore;
use crate::ollama::OllamaClient;
use anyhow::{anyhow, bail, Result};

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
            if prefs.is_auto_pull_disabled {
                bail!("no models installed and --no-auto-pull is set");
            }
            pull_fallback(client, mem_budget, on_status).await?;
            continue;
        }

        match models::score_models(&tags, mem_budget, prefs.min_context).first() {
            Some(best) if needs_fallback(best, prefs) => {
                on_status(&format!(
                    "⚠️  Best local model '{}' scores {:.0}/100 for agentic coding.",
                    best.name, best.score
                ));
                pull_fallback(client, mem_budget, on_status).await?;
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

/// True when the best local model is weak enough to justify pulling a fallback.
fn needs_fallback(best: &ModelScore, prefs: &ModelPrefs) -> bool {
    best.score < GOOD_MODEL_SCORE_THRESHOLD && !prefs.is_auto_pull_disabled
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
            anyhow!("Model '{explicit}' is not installed. Run `openbatrangs pull {explicit}` to download it.")
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
    Ok(context_length_from_show(&show))
}

/// Extract the model context length from a raw `/api/show` JSON body.
fn context_length_from_show(show: &serde_json::Value) -> u64 {
    show.get("model_info")
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
        .unwrap_or(FALLBACK_CONTEXT_LENGTH)
}

/// Compute the usable model memory budget from total system memory.
pub(crate) fn calculate_memory_budget() -> u64 {
    models::total_system_memory_bytes() * MEMORY_BUDGET_NUMERATOR / MEMORY_BUDGET_DENOMINATOR
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn extracts_context_from_model_info_key() {
        let show = json!({
            "model_info": {
                "llama.context_length": 32768,
                "other": 1,
            }
        });
        assert_eq!(context_length_from_show(&show), 32_768);
    }

    #[test]
    fn extracts_context_from_details_fallback() {
        let show = json!({"details": {"context_length": 8192}});
        assert_eq!(context_length_from_show(&show), 8_192);
    }

    #[test]
    fn falls_back_when_context_unknown() {
        assert_eq!(
            context_length_from_show(&json!({})),
            FALLBACK_CONTEXT_LENGTH
        );
        assert_eq!(
            context_length_from_show(&json!({"model_info": {}})),
            FALLBACK_CONTEXT_LENGTH
        );
    }

    #[test]
    fn model_info_takes_precedence_over_details() {
        let show = json!({
            "model_info": {"llama.context_length": 16384},
            "details": {"context_length": 4096},
        });
        assert_eq!(context_length_from_show(&show), 16_384);
    }

    #[test]
    fn memory_budget_is_positive_on_linux() {
        assert!(calculate_memory_budget() > 0);
    }

    fn prefs(model: Option<&str>, auto_pull_disabled: bool) -> ModelPrefs {
        ModelPrefs {
            model: model.map(str::to_string),
            is_auto_pull_disabled: auto_pull_disabled,
            min_context: 8_192,
        }
    }

    fn model_json() -> &'static str {
        r#"{"models":[{"name":"qwen2.5-coder:7b","size":4700000000,"details":{"parameter_size":"7.6B","quantization_level":"Q4_K_M","context_length":32768}}]}"#
    }

    #[tokio::test]
    async fn resolve_model_returns_explicit_installed_model() {
        use crate::test_support::{spawn_mock_server, MockResponse};

        let base_url = spawn_mock_server(|path| {
            assert_eq!(path, "/api/tags");
            MockResponse::json("200 OK", model_json())
        })
        .await;
        let client = OllamaClient::new(&base_url).unwrap();
        let statuses = std::sync::Mutex::new(Vec::new());
        let scored = resolve_model(
            &client,
            &Some("qwen2.5-coder:7b".to_string()),
            &prefs(None, false),
            16_000_000_000,
            &|msg| statuses.lock().unwrap().push(msg.to_string()),
        )
        .await
        .unwrap();
        assert_eq!(scored.name, "qwen2.5-coder:7b");
    }

    #[tokio::test]
    async fn resolve_model_errors_when_explicit_model_missing() {
        use crate::test_support::{spawn_mock_server, MockResponse};

        let base_url =
            spawn_mock_server(|_| MockResponse::json("200 OK", r#"{"models":[]}"#)).await;
        let client = OllamaClient::new(&base_url).unwrap();
        let result = resolve_model(
            &client,
            &Some("missing:7b".to_string()),
            &prefs(None, false),
            16_000_000_000,
            &|_| {},
        )
        .await;
        assert!(result.is_err());
        assert!(format!("{:#}", result.unwrap_err()).contains("not installed"));
    }

    #[tokio::test]
    async fn resolve_model_rejects_path_like_explicit_slot() {
        let client = OllamaClient::new("http://127.0.0.1:1").unwrap();
        let result = resolve_model(
            &client,
            &Some("./model.gguf".to_string()),
            &prefs(None, false),
            16_000_000_000,
            &|_| {},
        )
        .await;
        assert!(result.is_err());
        assert!(format!("{:#}", result.unwrap_err()).contains("looks like a file path"));
    }

    #[tokio::test]
    async fn resolve_model_auto_selects_best_installed_model() {
        use crate::test_support::{spawn_mock_server, MockResponse};

        let base_url = spawn_mock_server(|_| MockResponse::json("200 OK", model_json())).await;
        let client = OllamaClient::new(&base_url).unwrap();
        let scored = resolve_model(&client, &None, &prefs(None, false), 16_000_000_000, &|_| {})
            .await
            .unwrap();
        assert_eq!(scored.name, "qwen2.5-coder:7b");
    }

    #[tokio::test]
    async fn resolve_model_auto_errors_when_no_models_and_pull_disabled() {
        use crate::test_support::{spawn_mock_server, MockResponse};

        let base_url =
            spawn_mock_server(|_| MockResponse::json("200 OK", r#"{"models":[]}"#)).await;
        let client = OllamaClient::new(&base_url).unwrap();
        let result =
            resolve_model(&client, &None, &prefs(None, true), 16_000_000_000, &|_| {}).await;
        assert!(result.is_err());
        assert!(format!("{:#}", result.unwrap_err()).contains("no models installed"));
    }

    #[tokio::test]
    async fn resolve_model_auto_pulls_fallback_when_empty() {
        use crate::test_support::{spawn_mock_server, MockResponse};
        use std::sync::atomic::{AtomicUsize, Ordering};

        let calls = AtomicUsize::new(0);
        let base_url = spawn_mock_server(move |path| match path {
            "/api/tags" if calls.load(Ordering::SeqCst) == 0 => {
                calls.fetch_add(1, Ordering::SeqCst);
                MockResponse::json("200 OK", r#"{"models":[]}"#)
            }
            "/api/pull" => {
                calls.fetch_add(1, Ordering::SeqCst);
                MockResponse::json("200 OK", "{\"status\":\"success\"}\n")
            }
            "/api/tags" => {
                calls.fetch_add(1, Ordering::SeqCst);
                MockResponse::json("200 OK", model_json())
            }
            _ => MockResponse::text("404 Not Found", "no route"),
        })
        .await;
        let client = OllamaClient::new(&base_url).unwrap();
        let statuses = std::sync::Mutex::new(Vec::new());
        let scored = resolve_model(
            &client,
            &None,
            &prefs(None, false),
            16_000_000_000,
            &|msg| statuses.lock().unwrap().push(msg.to_string()),
        )
        .await
        .unwrap();
        assert_eq!(scored.name, "qwen2.5-coder:7b");
        assert!(statuses
            .lock()
            .unwrap()
            .iter()
            .any(|msg| msg.contains("Pulling a better default")));
    }

    #[tokio::test]
    async fn resolve_model_context_reads_show_endpoint() {
        use crate::test_support::{spawn_mock_server, MockResponse};

        let base_url = spawn_mock_server(|path| {
            assert_eq!(path, "/api/show");
            MockResponse::json("200 OK", r#"{"model_info":{"llama.context_length":32768}}"#)
        })
        .await;
        let client = OllamaClient::new(&base_url).unwrap();
        assert_eq!(
            resolve_model_context(&client, "qwen2.5-coder:7b")
                .await
                .unwrap(),
            32_768
        );
    }

    #[test]
    fn needs_fallback_true_when_score_below_threshold() {
        let best = crate::models::ModelScore {
            name: "weak-model".to_string(),
            score: 40.0,
            size_bytes: 1_000_000_000,
            parameter_size: "7B".to_string(),
            context_length: 8192,
            quantization: "Q4_K_M".to_string(),
            reasons: Vec::new(),
        };
        let p = prefs(None, false);
        assert!(needs_fallback(&best, &p));
    }

    #[test]
    fn needs_fallback_false_when_score_above_threshold() {
        let best = crate::models::ModelScore {
            name: "strong-model".to_string(),
            score: 80.0,
            size_bytes: 1_000_000_000,
            parameter_size: "7B".to_string(),
            context_length: 32768,
            quantization: "Q4_K_M".to_string(),
            reasons: Vec::new(),
        };
        let p = prefs(None, false);
        assert!(!needs_fallback(&best, &p));
    }

    #[test]
    fn needs_fallback_false_when_auto_pull_disabled() {
        let best = crate::models::ModelScore {
            name: "weak-model".to_string(),
            score: 40.0,
            size_bytes: 1_000_000_000,
            parameter_size: "7B".to_string(),
            context_length: 8192,
            quantization: "Q4_K_M".to_string(),
            reasons: Vec::new(),
        };
        let p = prefs(None, true);
        assert!(!needs_fallback(&best, &p));
    }

    #[test]
    fn needs_fallback_at_threshold_boundary() {
        // At exactly the threshold, should not need fallback
        let best = crate::models::ModelScore {
            name: "boundary-model".to_string(),
            score: GOOD_MODEL_SCORE_THRESHOLD,
            size_bytes: 1_000_000_000,
            parameter_size: "7B".to_string(),
            context_length: 16384,
            quantization: "Q4_K_M".to_string(),
            reasons: Vec::new(),
        };
        let p = prefs(None, false);
        assert!(!needs_fallback(&best, &p));

        // Just below threshold should need fallback
        let below = crate::models::ModelScore {
            name: "below-model".to_string(),
            score: GOOD_MODEL_SCORE_THRESHOLD - 0.1,
            size_bytes: 1_000_000_000,
            parameter_size: "7B".to_string(),
            context_length: 16384,
            quantization: "Q4_K_M".to_string(),
            reasons: Vec::new(),
        };
        assert!(needs_fallback(&below, &p));
    }
}
