//! # Model listing command
//!
//! `list-models` prints a table of installed models sorted by agentic-coding
//! score. The pure table builder is tested without network; the async wrapper
//! is tested against the shared mock Ollama server.
//!
//! ## References
//!
//! - Ollama `/api/tags`: <https://github.com/ollama/ollama/blob/main/docs/api.md#list-local-models>
//! - Model scoring heuristics: [`crate::models`]

use super::print_lines;
use crate::constants::models::{BYTES_PER_GIGABYTE, CONTEXT_LENGTH_KILO_DIVISOR};
use crate::model_select::calculate_memory_budget;
use crate::models::{self, ModelScore};
use crate::ollama::{OllamaClient, OllamaModel};
use anyhow::Result;

/// Print the model table to stdout.
pub(crate) async fn list_models(client: &OllamaClient, min_context: u64) -> Result<()> {
    let lines = list_models_lines(client, min_context).await?;

    print_lines(&lines);

    Ok(())
}

/// Build the model table as lines, used by both stdout and the TUI.
pub(crate) async fn list_models_lines(
    client: &OllamaClient,
    min_context: u64,
) -> Result<Vec<String>> {
    let tags = client.tags().await?;

    Ok(model_lines(client, &tags, min_context))
}

/// The scored table for Ollama, or one line per model for an
/// OpenAI-compatible server, which reports no sizes to score.
fn model_lines(client: &OllamaClient, tags: &[OllamaModel], min_context: u64) -> Vec<String> {
    if client.is_openai() {
        return served_model_lines(&client.base_url, tags);
    }

    list_models_lines_from_tags(tags, min_context)
}

/// One line per model, naming the server that serves it.
fn served_model_lines(base_url: &str, tags: &[OllamaModel]) -> Vec<String> {
    tags.iter()
        .map(|model| format!("{}  (served by {base_url})", model.name))
        .collect()
}

/// Build the model table from already-fetched tags (pure, testable).
fn list_models_lines_from_tags(tags: &[OllamaModel], min_context: u64) -> Vec<String> {
    if tags.is_empty() {
        return vec!["No models installed. Run `openbatrangs setup` to auto-pull one.".to_string()];
    }

    let scored = models::score_models(tags, calculate_memory_budget(), min_context);
    let mut lines = vec![table_header()];
    lines.extend(scored.iter().map(table_row));

    lines
}

/// The column titles of the model table.
fn table_header() -> String {
    format!(
        "{:<28} {:>9} {:>8} {:>8} {:>8} {:>6}  Notes",
        "MODEL", "SIZE", "PARAMS", "CTX", "QUANT", "SCORE"
    )
}

/// One row of the model table.
fn table_row(model: &ModelScore) -> String {
    format!(
        "{:<28} {:>7.1}G {:>8} {:>7}K {:>8} {:>6.0}  {}",
        model.name,
        model.size_bytes as f64 / BYTES_PER_GIGABYTE,
        model.parameter_size,
        model.context_length / CONTEXT_LENGTH_KILO_DIVISOR,
        model.quantization,
        model.score,
        model.reasons.join("; ")
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{MockResponse, sample_model, spawn_mock_server};

    /// One installed model with only a name and a size.
    const ONE_MODEL_BODY: &str = r#"{"models":[{"name":"qwen2.5-coder:3b","size":123}]}"#;

    #[test]
    fn empty_tags_returns_friendly_message() {
        let lines = list_models_lines_from_tags(&[], 8_192);

        assert_eq!(lines.len(), 1);
        assert!(lines[0].contains("No models installed"));
    }

    #[test]
    fn model_table_contains_header_and_rows() -> Result<()> {
        let tags = vec![
            sample_model("qwen2.5-coder:7b", 32_768, "7.6B", "Q4_K_M", 4_700_000_000)?,
            sample_model("llama3.2:3b", 8_192, "3.2B", "Q4_0", 2_000_000_000)?,
        ];

        let lines = list_models_lines_from_tags(&tags, 8_192);

        assert!(lines.len() >= 3);
        assert!(lines[0].contains("MODEL"));
        assert!(lines[1].contains("qwen2.5-coder:7b"));
        assert!(lines[2].contains("llama3.2:3b"));
        Ok(())
    }

    #[test]
    fn models_below_min_context_are_excluded_from_table() -> Result<()> {
        let tags = vec![sample_model("tiny:1b", 2_048, "1B", "Q4_0", 500_000_000)?];

        let lines = list_models_lines_from_tags(&tags, 8_192);

        // Header only, plus no rows for the excluded model.
        assert_eq!(lines.len(), 1);
        Ok(())
    }

    #[test]
    fn served_models_name_their_server() -> Result<()> {
        let client = OllamaClient::openai("http://thor:8079/v1", None, true)?;
        let tags = vec![sample_model("gemma", 8_192, "4B", "Q4_0", 0)?];

        let lines = model_lines(&client, &tags, 8_192);

        assert_eq!(lines, vec!["gemma  (served by http://thor:8079/v1)"]);
        Ok(())
    }

    #[tokio::test]
    async fn list_models_lines_fetch_tags_from_server() -> Result<()> {
        let base_url =
            spawn_mock_server(|_path| MockResponse::json("200 OK", ONE_MODEL_BODY)).await;
        let client = OllamaClient::new(&base_url)?;

        let lines = list_models_lines(&client, 4_096).await?;

        assert!(lines[0].contains("MODEL"));
        assert!(lines.iter().any(|line| line.contains("qwen2.5-coder:3b")));
        Ok(())
    }

    #[tokio::test]
    async fn list_models_prints_the_table() -> Result<()> {
        let base_url =
            spawn_mock_server(|_path| MockResponse::json("200 OK", ONE_MODEL_BODY)).await;
        let client = OllamaClient::new(&base_url)?;

        list_models(&client, 4_096).await
    }

    #[tokio::test]
    async fn list_models_lines_propagate_tags_error() -> Result<()> {
        let base_url =
            spawn_mock_server(|_path| MockResponse::text("500 Internal Server Error", "boom"))
                .await;
        let client = OllamaClient::new(&base_url)?;

        assert!(list_models_lines(&client, 8_192).await.is_err());
        Ok(())
    }
}
