//! Model auto-selection, validation, and fallback pulling.

use crate::cli::ModelPrefs;
use crate::constants::models::{
    CONTEXT_LENGTH_KEY_SUFFIX, FALLBACK_CONTEXT_LENGTH, GOOD_MODEL_SCORE_THRESHOLD,
    MEMORY_BUDGET_DENOMINATOR, MEMORY_BUDGET_NUMERATOR, MODEL_SELECT_MAX_ATTEMPTS,
};
use crate::models;
use crate::models::ModelScore;
use crate::ollama::{OllamaClient, OllamaModel};
use anyhow::{Result, anyhow, bail};

/// Resolve a model from an optional explicit slot or auto-selection.
///
/// # Arguments
/// - `client`: Ollama HTTP client.
/// - `model_slot`: explicit model tag, if any; it wins over `prefs.model`.
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
    let wanted = model_slot.clone().or_else(|| prefs.model.clone());

    if client.is_openai() {
        return select_served_model(client, wanted.as_ref()).await;
    }

    let effective_prefs = ModelPrefs {
        model: wanted,
        ..prefs.clone()
    };

    select_model(client, &effective_prefs, mem_budget, on_status).await
}

/// Pick a model on an OpenAI-compatible server: the requested one, or the
/// first the server lists. The server owns its models, so nothing is scored
/// or pulled.
async fn select_served_model(client: &OllamaClient, wanted: Option<&String>) -> Result<ModelScore> {
    let served = client.tags().await?;
    let name = served_model_name(&served, wanted, &client.base_url)?;
    let context_length = resolve_model_context(client, &name).await?;

    Ok(ModelScore {
        name,
        size_bytes: 0,
        parameter_size: "served".to_string(),
        context_length,
        quantization: "served".to_string(),
        score: 100.0,
        reasons: vec![format!("served by {}", client.base_url)],
    })
}

/// The name of the wanted model when the server serves it, or of the first
/// served model when none is wanted.
fn served_model_name(
    served: &[OllamaModel],
    wanted: Option<&String>,
    base_url: &str,
) -> Result<String> {
    match wanted {
        Some(name) => wanted_served_model(served, name, base_url),
        None => first_served_model(served, base_url),
    }
}

/// `name` when the server serves it; otherwise an error listing what it serves.
fn wanted_served_model(served: &[OllamaModel], name: &str, base_url: &str) -> Result<String> {
    let names: Vec<&str> = served.iter().map(|model| model.name.as_str()).collect();

    if !names.contains(&name) {
        bail!(
            "'{name}' is not served at {base_url}; it serves: {}",
            names.join(", ")
        );
    }

    Ok(name.to_string())
}

/// The first model the server lists.
fn first_served_model(served: &[OllamaModel], base_url: &str) -> Result<String> {
    served
        .first()
        .map(|model| model.name.clone())
        .ok_or_else(|| anyhow!("{base_url} serves no models"))
}

/// Select the requested model, or the best installed one with a fallback
/// pulled when needed.
async fn select_model(
    client: &OllamaClient,
    prefs: &ModelPrefs,
    mem_budget: u64,
    on_status: &(dyn Fn(&str) + Sync),
) -> Result<ModelScore> {
    if let Some(explicit) = &prefs.model {
        return select_explicit_model(client, explicit, prefs.min_context, mem_budget).await;
    }

    select_auto_model(client, prefs, mem_budget, on_status).await
}

/// Pick the best installed model. Each round either picks a model or pulls
/// the fallback; after `MODEL_SELECT_MAX_ATTEMPTS` rounds without a pick the
/// selection fails.
async fn select_auto_model(
    client: &OllamaClient,
    prefs: &ModelPrefs,
    mem_budget: u64,
    on_status: &(dyn Fn(&str) + Sync),
) -> Result<ModelScore> {
    let mut rounds = 0;
    let mut selection = Ok(None);

    while is_still_searching(&selection, rounds) {
        rounds += 1;
        selection = select_or_pull(client, prefs, mem_budget, on_status).await;
    }

    selected_model(selection)
}

/// Whether another round may run: no model picked, no error, rounds left.
fn is_still_searching(selection: &Result<Option<ModelScore>>, rounds: u32) -> bool {
    matches!(selection, Ok(None)) && rounds < MODEL_SELECT_MAX_ATTEMPTS
}

/// The picked model, the error that stopped the search, or an error when
/// every round ended in a pull.
fn selected_model(selection: Result<Option<ModelScore>>) -> Result<ModelScore> {
    selection?.ok_or_else(|| anyhow!("could not find a suitable model after auto-pulling"))
}

/// One selection round: the best installed model when it is good enough,
/// otherwise `None` after pulling the fallback.
async fn select_or_pull(
    client: &OllamaClient,
    prefs: &ModelPrefs,
    mem_budget: u64,
    on_status: &(dyn Fn(&str) + Sync),
) -> Result<Option<ModelScore>> {
    let installed = client.tags().await?;
    let choice = auto_choice(&installed, prefs, mem_budget)?;

    act_on_choice(client, choice, mem_budget, on_status).await
}

/// The outcome of one round of automatic selection.
enum AutoChoice {
    /// An installed model is good enough.
    Use(ModelScore),
    /// Pull the fallback model, reporting this warning first when there is one.
    PullFallback(Option<String>),
}

/// The outcome for the installed models; an error when a pull is needed but
/// `--no-auto-pull` forbids it.
fn auto_choice(
    installed: &[OllamaModel],
    prefs: &ModelPrefs,
    mem_budget: u64,
) -> Result<AutoChoice> {
    if installed.is_empty() {
        return pull_unless_disabled(prefs, "no models installed and --no-auto-pull is set");
    }

    let best = models::score_models(installed, mem_budget, prefs.min_context)
        .into_iter()
        .next();

    choice_for_best(best, prefs)
}

/// The outcome for the best scored model, or for none when no model has a
/// large enough context window.
fn choice_for_best(best: Option<ModelScore>, prefs: &ModelPrefs) -> Result<AutoChoice> {
    match best {
        Some(best) => Ok(choice_for_scored(best, prefs)),
        None => pull_unless_disabled(
            prefs,
            &format!(
                "no installed model meets --min-context={}",
                prefs.min_context
            ),
        ),
    }
}

/// Use `best`, or pull the fallback with a warning when `best` is weak.
fn choice_for_scored(best: ModelScore, prefs: &ModelPrefs) -> AutoChoice {
    if !needs_fallback(&best, prefs) {
        return AutoChoice::Use(best);
    }

    AutoChoice::PullFallback(Some(format!(
        "⚠️  Best local model '{}' scores {:.0}/100 for agentic coding.",
        best.name, best.score
    )))
}

/// Pull the fallback, or fail with `refusal` when auto-pull is disabled.
fn pull_unless_disabled(prefs: &ModelPrefs, refusal: &str) -> Result<AutoChoice> {
    if prefs.is_auto_pull_disabled {
        bail!("{refusal}");
    }

    Ok(AutoChoice::PullFallback(None))
}

/// Return the chosen model, or pull the fallback.
async fn act_on_choice(
    client: &OllamaClient,
    choice: AutoChoice,
    mem_budget: u64,
    on_status: &(dyn Fn(&str) + Sync),
) -> Result<Option<ModelScore>> {
    match choice {
        AutoChoice::Use(best) => Ok(Some(best)),
        AutoChoice::PullFallback(warning) => {
            pull_with_warning(client, warning, mem_budget, on_status).await
        }
    }
}

/// Report the warning, if any, then pull the fallback. Returns no model, so
/// the next round scores the models again.
async fn pull_with_warning(
    client: &OllamaClient,
    warning: Option<String>,
    mem_budget: u64,
    on_status: &(dyn Fn(&str) + Sync),
) -> Result<Option<ModelScore>> {
    if let Some(warning) = warning {
        on_status(&warning);
    }

    pull_fallback(client, mem_budget, on_status)
        .await
        .map(|()| None)
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
    reject_path(explicit)?;
    let installed = client.tags().await?;

    score_installed(&installed, explicit, min_context, mem_budget)
}

/// An error for a model name that looks like a file path, which Ollama cannot load.
fn reject_path(explicit: &str) -> Result<()> {
    if models::looks_like_path(explicit) {
        bail!(
            "'{explicit}' looks like a file path. openBatarangs now uses Ollama model tags.\n\
             Just run `openbatrangs setup` and it will pull a coding model for you."
        );
    }

    Ok(())
}

/// The score of the installed model named `explicit`.
fn score_installed(
    installed: &[OllamaModel],
    explicit: &str,
    min_context: u64,
    mem_budget: u64,
) -> Result<ModelScore> {
    let model = installed
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
    use crate::test_support::{MockResponse, StatusLog, ignore_status, spawn_mock_server};
    use serde_json::json;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// One installed coding model, as `/api/tags` lists it.
    const MODEL_JSON: &str = r#"{"models":[{"name":"qwen2.5-coder:7b","size":4700000000,"details":{"parameter_size":"7.6B","quantization_level":"Q4_K_M","context_length":32768}}]}"#;

    /// One installed model too weak for agentic coding.
    const WEAK_MODEL_JSON: &str = r#"{"models":[{"name":"tiny-chat:1b","size":90000000000,"details":{"parameter_size":"1B","quantization_level":"F16","context_length":8192}}]}"#;

    /// One installed model with a context window below the minimum.
    const SHORT_CONTEXT_JSON: &str =
        r#"{"models":[{"name":"short:1b","size":900000000,"details":{"context_length":2048}}]}"#;

    /// No installed models.
    const NO_MODELS_JSON: &str = r#"{"models":[]}"#;

    /// Two models on an OpenAI-compatible server.
    const SERVED_JSON: &str = r#"{"data":[{"id":"gemma","max_model_len":16384},{"id":"qwen"}]}"#;

    /// An address where nothing listens, so every request is refused at once.
    const CLOSED_URL: &str = "http://127.0.0.1:1";

    fn prefs(model: Option<&str>, auto_pull_disabled: bool) -> ModelPrefs {
        ModelPrefs {
            model: model.map(str::to_string),
            is_auto_pull_disabled: auto_pull_disabled,
            min_context: 8_192,
        }
    }

    /// A model score with the given name and score; the other fields are filler.
    fn scored(name: &str, score: f64) -> ModelScore {
        ModelScore {
            name: name.to_string(),
            score,
            size_bytes: 1_000_000_000,
            parameter_size: "7B".to_string(),
            context_length: 8_192,
            quantization: "Q4_K_M".to_string(),
            reasons: Vec::new(),
        }
    }

    /// A client for a mock Ollama that lists `tags_body` and pulls successfully.
    async fn ollama_with(tags_body: &'static str) -> Result<OllamaClient> {
        let base_url = spawn_mock_server(move |path| ollama_route(path, tags_body)).await;

        OllamaClient::new(&base_url)
    }

    /// Answer `/api/pull` with success and every other path with `tags_body`.
    fn ollama_route(path: &str, tags_body: &'static str) -> MockResponse {
        match path {
            "/api/pull" => MockResponse::json("200 OK", "{\"status\":\"success\"}\n"),
            _ => MockResponse::json("200 OK", tags_body),
        }
    }

    /// A mock Ollama with no models until the first pull: `/api/tags` lists
    /// nothing on its first request and one coding model after it.
    fn models_after_pull(path: &str, tags_requests: &AtomicUsize) -> MockResponse {
        match path {
            "/api/tags" => tags_after_pull(tags_requests.fetch_add(1, Ordering::SeqCst)),
            _ => ollama_route(path, NO_MODELS_JSON),
        }
    }

    /// No models on the first `/api/tags` request, one coding model after it.
    fn tags_after_pull(earlier_tags_requests: usize) -> MockResponse {
        if earlier_tags_requests == 0 {
            return MockResponse::json("200 OK", NO_MODELS_JSON);
        }

        MockResponse::json("200 OK", MODEL_JSON)
    }

    /// A client for a mock OpenAI-compatible server that serves two models.
    async fn served_models() -> Result<OllamaClient> {
        let base_url = spawn_mock_server(|_path| MockResponse::json("200 OK", SERVED_JSON)).await;

        OllamaClient::openai(&format!("{base_url}/v1"), None, true)
    }

    #[test]
    fn extracts_context_from_model_info_key() {
        let show = json!({
            "model_info": {
                "general.architecture": "llama",
                "llama.context_length": 32768,
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

    #[tokio::test]
    async fn resolve_model_returns_explicit_installed_model() -> Result<()> {
        let client = ollama_with(MODEL_JSON).await?;
        let slot = Some("qwen2.5-coder:7b".to_string());

        let selected = resolve_model(
            &client,
            &slot,
            &prefs(None, false),
            16_000_000_000,
            &ignore_status,
        )
        .await?;

        assert_eq!(selected.name, "qwen2.5-coder:7b");
        Ok(())
    }

    #[tokio::test]
    async fn resolve_model_errors_when_explicit_model_missing() -> Result<()> {
        let client = ollama_with(NO_MODELS_JSON).await?;
        let slot = Some("missing:7b".to_string());

        let outcome = resolve_model(
            &client,
            &slot,
            &prefs(None, false),
            16_000_000_000,
            &ignore_status,
        )
        .await;

        assert!(format!("{outcome:?}").contains("not installed"));
        Ok(())
    }

    #[test]
    fn explicit_model_below_min_context_is_rejected() -> Result<()> {
        let installed = [crate::test_support::sample_model(
            "short:1b", 2_048, "1B", "Q4_0", 1,
        )?];

        let outcome = score_installed(&installed, "short:1b", 8_192, 16_000_000_000);

        assert!(format!("{outcome:?}").contains("below --min-context"));
        Ok(())
    }

    #[tokio::test]
    async fn resolve_model_rejects_path_like_explicit_slot() -> Result<()> {
        let client = OllamaClient::new(CLOSED_URL)?;
        let slot = Some("./model.gguf".to_string());

        let outcome = resolve_model(
            &client,
            &slot,
            &prefs(None, false),
            16_000_000_000,
            &ignore_status,
        )
        .await;

        assert!(format!("{outcome:?}").contains("looks like a file path"));
        Ok(())
    }

    #[tokio::test]
    async fn resolve_model_auto_selects_best_installed_model() -> Result<()> {
        let client = ollama_with(MODEL_JSON).await?;

        let selected = resolve_model(
            &client,
            &None,
            &prefs(None, false),
            16_000_000_000,
            &ignore_status,
        )
        .await?;

        assert_eq!(selected.name, "qwen2.5-coder:7b");
        Ok(())
    }

    #[tokio::test]
    async fn resolve_model_auto_errors_when_no_models_and_pull_disabled() -> Result<()> {
        let client = ollama_with(NO_MODELS_JSON).await?;

        let outcome = resolve_model(
            &client,
            &None,
            &prefs(None, true),
            16_000_000_000,
            &ignore_status,
        )
        .await;

        assert!(format!("{outcome:?}").contains("no models installed"));
        Ok(())
    }

    #[tokio::test]
    async fn no_model_with_enough_context_and_pull_disabled_is_an_error() -> Result<()> {
        let client = ollama_with(SHORT_CONTEXT_JSON).await?;

        let outcome = resolve_model(
            &client,
            &None,
            &prefs(None, true),
            16_000_000_000,
            &ignore_status,
        )
        .await;

        assert!(format!("{outcome:?}").contains("--min-context=8192"));
        Ok(())
    }

    #[tokio::test]
    async fn resolve_model_auto_pulls_fallback_when_empty() -> Result<()> {
        let tags_requests = AtomicUsize::new(0);
        let base_url = spawn_mock_server(move |path| models_after_pull(path, &tags_requests)).await;
        let client = OllamaClient::new(&base_url)?;
        let log = StatusLog::default();

        let selected = resolve_model(
            &client,
            &None,
            &prefs(None, false),
            16_000_000_000,
            &|message| log.record(message),
        )
        .await?;

        assert_eq!(selected.name, "qwen2.5-coder:7b");
        assert!(log.contains("Pulling a better default"));
        Ok(())
    }

    #[tokio::test]
    async fn weak_model_is_replaced_until_the_rounds_run_out() -> Result<()> {
        let client = ollama_with(WEAK_MODEL_JSON).await?;
        let log = StatusLog::default();

        let outcome = resolve_model(
            &client,
            &None,
            &prefs(None, false),
            16_000_000_000,
            &|message| log.record(message),
        )
        .await;

        assert!(log.contains("for agentic coding"));
        assert!(format!("{outcome:?}").contains("after auto-pulling"));
        Ok(())
    }

    #[tokio::test]
    async fn short_context_models_trigger_a_pull() -> Result<()> {
        let client = ollama_with(SHORT_CONTEXT_JSON).await?;
        let log = StatusLog::default();

        let outcome = resolve_model(
            &client,
            &None,
            &prefs(None, false),
            16_000_000_000,
            &|message| log.record(message),
        )
        .await;

        assert!(log.contains("Pulling a better default"));
        assert!(outcome.is_err());
        Ok(())
    }

    #[tokio::test]
    async fn resolve_model_context_reads_show_endpoint() -> Result<()> {
        let base_url = spawn_mock_server(|_path| {
            MockResponse::json("200 OK", r#"{"model_info":{"llama.context_length":32768}}"#)
        })
        .await;
        let client = OllamaClient::new(&base_url)?;

        assert_eq!(
            resolve_model_context(&client, "qwen2.5-coder:7b").await?,
            32_768
        );
        Ok(())
    }

    #[tokio::test]
    async fn served_model_defaults_to_the_first_listed() -> Result<()> {
        let client = served_models().await?;

        let selected =
            resolve_model(&client, &None, &prefs(None, false), 0, &ignore_status).await?;

        assert_eq!(selected.name, "gemma");
        assert_eq!(selected.context_length, 16_384);
        Ok(())
    }

    #[tokio::test]
    async fn served_model_must_be_listed() -> Result<()> {
        let client = served_models().await?;

        let outcome = resolve_model(
            &client,
            &None,
            &prefs(Some("llama"), false),
            0,
            &ignore_status,
        )
        .await;

        assert!(format!("{outcome:?}").contains("it serves: gemma, qwen"));
        Ok(())
    }

    #[test]
    fn wanted_served_model_is_chosen() -> Result<()> {
        let served = [crate::test_support::sample_model(
            "qwen", 8_192, "7B", "Q4_0", 0,
        )?];

        assert_eq!(
            served_model_name(&served, Some(&"qwen".to_string()), "url")?,
            "qwen"
        );
        Ok(())
    }

    #[test]
    fn server_without_models_is_an_error() {
        let outcome = served_model_name(&[], None, "http://thor/v1");

        assert!(format!("{outcome:?}").contains("serves no models"));
    }

    #[test]
    fn needs_fallback_true_when_score_below_threshold() {
        assert!(needs_fallback(
            &scored("weak-model", 40.0),
            &prefs(None, false)
        ));
    }

    #[test]
    fn needs_fallback_false_when_score_above_threshold() {
        assert!(!needs_fallback(
            &scored("strong-model", 80.0),
            &prefs(None, false)
        ));
    }

    #[test]
    fn needs_fallback_false_when_auto_pull_disabled() {
        assert!(!needs_fallback(
            &scored("weak-model", 40.0),
            &prefs(None, true)
        ));
    }

    #[test]
    fn needs_fallback_at_threshold_boundary() {
        let at_threshold = scored("boundary-model", GOOD_MODEL_SCORE_THRESHOLD);
        let below = scored("below-model", GOOD_MODEL_SCORE_THRESHOLD - 0.1);

        assert!(!needs_fallback(&at_threshold, &prefs(None, false)));
        assert!(needs_fallback(&below, &prefs(None, false)));
    }
}
