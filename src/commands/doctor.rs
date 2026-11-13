//! # Doctor/health-check command
//!
//! `doctor` verifies Ollama connectivity, scores installed models, and prints a
//! human-readable recommendation. The pure report builder
//! [`doctor_lines_from_tags`] is tested without network; the async wrapper is
//! tested against the shared mock Ollama server.
//!
//! ## References
//!
//! - Ollama `/api/tags`: <https://github.com/ollama/ollama/blob/main/docs/api.md#list-local-models>

use super::print_lines;
use crate::constants::models::BYTES_PER_GIGABYTE;
use crate::model_select::calculate_memory_budget;
use crate::models::{self, ModelScore};
use crate::ollama::{OllamaClient, OllamaModel};
use anyhow::Result;

/// Print the doctor report to stdout.
pub(crate) async fn doctor(client: &OllamaClient, min_context: u64) -> Result<()> {
    let lines = doctor_lines(client, min_context).await?;

    print_lines(&lines);

    Ok(())
}

/// Build the doctor report as lines, used by both stdout and the TUI.
pub(crate) async fn doctor_lines(client: &OllamaClient, min_context: u64) -> Result<Vec<String>> {
    let tags = client.tags().await?;

    Ok(doctor_lines_from_tags(&client.base_url, &tags, min_context))
}

/// Build the doctor report from already-fetched tags (pure, testable).
fn doctor_lines_from_tags(base_url: &str, tags: &[OllamaModel], min_context: u64) -> Vec<String> {
    let scored = models::score_models(tags, calculate_memory_budget(), min_context);
    let mut lines = vec![
        format!("✅ Ollama reachable at {base_url}"),
        format!("Installed models: {}", tags.len()),
    ];

    match scored.first() {
        Some(best) => lines.extend(best_model_lines(best)),
        None => lines.extend(no_model_lines(min_context)),
    }

    lines.extend(performance_lines());

    lines
}

/// The two lines that name the best model and describe it.
fn best_model_lines(best: &ModelScore) -> [String; 2] {
    [
        format!(
            "🏆 Best model for agentic coding: {} (score {:.0}/100)",
            best.name, best.score
        ),
        format!(
            "   {:.1} GB, {} params, {} context, {}",
            best.size_bytes as f64 / BYTES_PER_GIGABYTE,
            best.parameter_size,
            best.context_length,
            best.quantization
        ),
    ]
}

/// The two lines shown when no installed model has a large enough context.
fn no_model_lines(min_context: u64) -> [String; 2] {
    [
        format!("⚠️  No model meets the minimum context of {min_context}."),
        "   Run `openbatrangs setup` to pull a recommended model.".to_string(),
    ]
}

/// The closing notes on performance per watt per dollar, after a blank line.
fn performance_lines() -> [String; 5] {
    [
        String::new(),
        "⚡ Performance per watt per dollar:".to_string(),
        "   - Local GPU inference on unified memory, no cloud API fees".to_string(),
        "   - Auto-picker deliberately chooses models that fit memory (Q4_K_M 3B-8B on 16GB Jetson)"
            .to_string(),
        "   - Keeps latency and power low while preserving a 32K+ context window".to_string(),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{MockResponse, sample_model, spawn_mock_server};

    #[test]
    fn empty_tags_report_mentions_no_models() {
        let lines = doctor_lines_from_tags("http://localhost:11434", &[], 8_192);

        assert!(lines[0].contains("localhost"));
        assert!(lines[1].contains("0"));
        assert!(lines.iter().any(|line| line.contains("No model meets")));
    }

    #[test]
    fn best_model_is_reported() -> Result<()> {
        let tags = vec![
            sample_model("small:3b", 8_192, "3B", "Q4_0", 2_000_000_000)?,
            sample_model("best:7b", 32_768, "7.6B", "Q4_K_M", 4_700_000_000)?,
        ];

        let lines = doctor_lines_from_tags("http://localhost:11434", &tags, 8_192);

        assert!(lines.iter().any(|line| line.contains("best:7b")));
        assert!(lines.iter().any(|line| line.contains("score")));
        Ok(())
    }

    #[test]
    fn all_models_below_min_context_are_reported_as_missing() -> Result<()> {
        let tags = vec![sample_model("tiny:1b", 2_048, "1B", "Q4_0", 500_000_000)?];

        let lines = doctor_lines_from_tags("http://localhost:11434", &tags, 8_192);

        assert!(lines.iter().any(|line| line.contains("No model meets")));
        Ok(())
    }

    #[tokio::test]
    async fn doctor_lines_fetch_tags_from_server() -> Result<()> {
        let base_url = spawn_mock_server(|_path| {
            MockResponse::json(
                "200 OK",
                r#"{"models":[{"name":"qwen2.5-coder:3b","size":123}]}"#,
            )
        })
        .await;
        let client = OllamaClient::new(&base_url)?;

        let lines = doctor_lines(&client, 8_192).await?;

        assert!(lines[0].contains("Ollama reachable"));
        assert!(lines[1].contains("Installed models: 1"));
        Ok(())
    }

    #[tokio::test]
    async fn doctor_prints_the_report() -> Result<()> {
        let base_url =
            spawn_mock_server(|_path| MockResponse::json("200 OK", r#"{"models":[]}"#)).await;
        let client = OllamaClient::new(&base_url)?;

        doctor(&client, 8_192).await
    }

    #[tokio::test]
    async fn doctor_lines_propagate_tags_error() -> Result<()> {
        let base_url =
            spawn_mock_server(|_path| MockResponse::text("500 Internal Server Error", "boom"))
                .await;
        let client = OllamaClient::new(&base_url)?;

        assert!(doctor_lines(&client, 8_192).await.is_err());
        Ok(())
    }
}
