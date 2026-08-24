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

use crate::constants::models::BYTES_PER_GIGABYTE;
use crate::model_select::calculate_memory_budget;
use crate::models;
use crate::ollama::{OllamaClient, OllamaModel};
use anyhow::Result;

/// Print the doctor report to stdout.
pub(crate) async fn doctor(client: &OllamaClient, min_context: u64) -> Result<()> {
    for line in doctor_lines(client, min_context).await? {
        println!("{line}");
    }
    Ok(())
}

/// Build the doctor report as lines, used by both stdout and the TUI.
pub(crate) async fn doctor_lines(client: &OllamaClient, min_context: u64) -> Result<Vec<String>> {
    let tags = client.tags().await?;
    Ok(doctor_lines_from_tags(&client.base_url, &tags, min_context))
}

/// Build the doctor report from already-fetched tags (pure, testable).
fn doctor_lines_from_tags(base_url: &str, tags: &[OllamaModel], min_context: u64) -> Vec<String> {
    let mut lines = vec![format!("✅ Ollama reachable at {base_url}")];
    lines.push(format!("Installed models: {}", tags.len()));
    let mem_budget = calculate_memory_budget();
    let scored = models::score_models(tags, mem_budget, min_context);
    match scored.first() {
        Some(best) => {
            lines.push(format!(
                "🏆 Best model for agentic coding: {} (score {:.0}/100)",
                best.name, best.score
            ));
            lines.push(format!(
                "   {:.1} GB, {} params, {} context, {}",
                best.size_bytes as f64 / BYTES_PER_GIGABYTE,
                best.parameter_size,
                best.context_length,
                best.quantization
            ));
        }
        None => {
            lines.push(format!(
                "⚠️  No model meets the minimum context of {min_context}."
            ));
            lines.push("   Run `openbatrangs setup` to pull a recommended model.".to_string());
        }
    }
    lines.push(String::new());
    lines.push("⚡ Performance per watt per dollar:".to_string());
    lines.push("   - Local GPU inference on unified memory, no cloud API fees".to_string());
    lines.push(
        "   - Auto-picker deliberately chooses models that fit memory (Q4_K_M 3B-8B on 16GB Jetson)"
            .to_string(),
    );
    lines.push(
        "   - Keeps latency and power low while preserving a 32K+ context window".to_string(),
    );
    lines
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn empty_tags_report_mentions_no_models() {
        let lines = doctor_lines_from_tags("http://localhost:11434", &[], 8_192);
        assert!(lines[0].contains("localhost"));
        assert!(lines[1].contains("0"));
        assert!(lines.iter().any(|line| line.contains("No model meets")));
    }

    #[test]
    fn best_model_is_reported() {
        let tags = vec![
            model("small:3b", 8_192, "3B", "Q4_0", 2_000_000_000),
            model("best:7b", 32_768, "7.6B", "Q4_K_M", 4_700_000_000),
        ];
        let lines = doctor_lines_from_tags("http://localhost:11434", &tags, 8_192);
        assert!(lines.iter().any(|line| line.contains("best:7b")));
        assert!(lines.iter().any(|line| line.contains("score")));
    }

    #[test]
    fn all_models_below_min_context_are_reported_as_missing() {
        let tags = vec![model("tiny:1b", 2_048, "1B", "Q4_0", 500_000_000)];
        let lines = doctor_lines_from_tags("http://localhost:11434", &tags, 8_192);
        assert!(lines.iter().any(|line| line.contains("No model meets")));
    }

    #[tokio::test]
    async fn doctor_lines_fetch_tags_from_server() {
        use crate::test_support::{spawn_mock_server, MockResponse};

        let base_url = spawn_mock_server(|path| {
            assert_eq!(path, "/api/tags");
            MockResponse::json(
                "200 OK",
                r#"{"models":[{"name":"qwen2.5-coder:3b","size":123}]}"#,
            )
        })
        .await;
        let client = OllamaClient::new(&base_url).unwrap();
        let lines = doctor_lines(&client, 8_192).await.unwrap();
        assert!(lines[0].contains("Ollama reachable"));
        assert!(lines[1].contains("Installed models: 1"));
    }

    #[tokio::test]
    async fn doctor_lines_propagate_tags_error() {
        use crate::test_support::{spawn_mock_server, MockResponse};

        let base_url =
            spawn_mock_server(|_| MockResponse::text("500 Internal Server Error", "boom")).await;
        let client = OllamaClient::new(&base_url).unwrap();
        assert!(doctor_lines(&client, 8_192).await.is_err());
    }
}
