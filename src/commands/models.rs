//! Model listing command.

use crate::constants::models::{BYTES_PER_GIGABYTE, CONTEXT_LENGTH_KILO_DIVISOR};
use crate::model_select::calculate_memory_budget;
use crate::models;
use crate::ollama::{OllamaClient, OllamaModel};
use anyhow::Result;

/// Print the model table to stdout.
pub(crate) async fn list_models(client: &OllamaClient, min_context: u64) -> Result<()> {
    for line in list_models_lines(client, min_context).await? {
        println!("{line}");
    }
    Ok(())
}

/// Build the model table as lines, used by both stdout and the TUI.
pub(crate) async fn list_models_lines(
    client: &OllamaClient,
    min_context: u64,
) -> Result<Vec<String>> {
    let tags = client.tags().await?;
    Ok(list_models_lines_from_tags(&tags, min_context))
}

/// Build the model table from already-fetched tags (pure, testable).
fn list_models_lines_from_tags(tags: &[OllamaModel], min_context: u64) -> Vec<String> {
    if tags.is_empty() {
        return vec!["No models installed. Run `openbatrangs setup` to auto-pull one.".to_string()];
    }
    let mem_budget = calculate_memory_budget();
    let scored = models::score_models(tags, mem_budget, min_context);
    let mut lines = vec![format!(
        "{:<28} {:>9} {:>8} {:>8} {:>8} {:>6}  Notes",
        "MODEL", "SIZE", "PARAMS", "CTX", "QUANT", "SCORE"
    )];
    for model in scored {
        lines.push(format!(
            "{:<28} {:>7.1}G {:>8} {:>7}K {:>8} {:>6.0}  {}",
            model.name,
            model.size_bytes as f64 / BYTES_PER_GIGABYTE,
            model.parameter_size,
            model.context_length / CONTEXT_LENGTH_KILO_DIVISOR,
            model.quantization,
            model.score,
            model.reasons.join("; ")
        ));
    }
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
    fn empty_tags_returns_friendly_message() {
        let lines = list_models_lines_from_tags(&[], 8_192);
        assert_eq!(lines.len(), 1);
        assert!(lines[0].contains("No models installed"));
    }

    #[test]
    fn model_table_contains_header_and_rows() {
        let tags = vec![
            model("qwen2.5-coder:7b", 32_768, "7.6B", "Q4_K_M", 4_700_000_000),
            model("llama3.2:3b", 8_192, "3.2B", "Q4_0", 2_000_000_000),
        ];
        let lines = list_models_lines_from_tags(&tags, 8_192);
        assert!(lines.len() >= 3);
        assert!(lines[0].contains("MODEL"));
        assert!(lines[1].contains("qwen2.5-coder:7b"));
        assert!(lines[2].contains("llama3.2:3b"));
    }

    #[test]
    fn models_below_min_context_are_excluded_from_table() {
        let tags = vec![model("tiny:1b", 2_048, "1B", "Q4_0", 500_000_000)];
        let lines = list_models_lines_from_tags(&tags, 8_192);
        // Header only, plus no rows for the excluded model.
        assert_eq!(lines.len(), 1);
    }
}
