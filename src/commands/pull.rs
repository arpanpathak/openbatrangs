//! Model pulling command.

use super::print_status;
use crate::ollama::OllamaClient;
use anyhow::Result;

/// Pull a model tag from the Ollama registry and print progress.
pub(crate) async fn pull(client: &OllamaClient, model: &str) -> Result<()> {
    client.pull(model, &print_status).await?;

    println!("✅ Model '{model}' is ready.");

    Ok(())
}
