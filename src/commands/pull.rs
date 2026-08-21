//! Model pulling command.

use crate::ollama::OllamaClient;
use anyhow::Result;

/// Pull a model tag from the Ollama registry and print progress.
pub(crate) async fn pull(client: &OllamaClient, model: &str) -> Result<()> {
    client.pull(model, &|msg| println!("{msg}")).await?;
    println!("✅ Model '{model}' is ready.");
    Ok(())
}
