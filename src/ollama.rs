use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::time::Duration;

#[derive(Clone)]
pub struct OllamaClient {
    pub base_url: String,
    http: reqwest::Client,
}

#[derive(Deserialize, Clone, Debug)]
pub struct TagsResponse {
    pub models: Vec<OllamaModel>,
}

#[derive(Deserialize, Clone, Debug)]
pub struct OllamaModel {
    pub name: String,
    #[serde(default)]
    pub size: u64,
    #[serde(default)]
    pub details: Option<ModelDetails>,
}

#[derive(Deserialize, Clone, Debug, Default)]
pub struct ModelDetails {
    #[serde(default)]
    pub parameter_size: Option<String>,
    #[serde(default)]
    pub quantization_level: Option<String>,
    #[serde(default)]
    pub context_length: Option<u64>,
}

#[derive(Serialize, Clone, Debug)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Serialize, Debug)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    pub stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub format: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub options: Option<Value>,
}

#[derive(Deserialize, Debug)]
pub struct ChatResponse {
    pub message: ChatResponseMessage,
}

#[derive(Deserialize, Debug)]
pub struct ChatResponseMessage {
    pub content: String,
}

#[derive(Serialize, Debug)]
struct PullRequest {
    name: String,
    stream: bool,
}

impl OllamaClient {
    pub fn new(base_url: &str) -> Result<Self> {
        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(600))
            .connect_timeout(Duration::from_secs(5))
            .build()
            .context("failed to build HTTP client")?;
        Ok(Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            http,
        })
    }

    pub async fn is_available(&self) -> bool {
        self.http
            .get(format!("{}/api/tags", self.base_url))
            .send()
            .await
            .map(|r| r.status().is_success())
            .unwrap_or(false)
    }

    pub async fn tags(&self) -> Result<Vec<OllamaModel>> {
        let resp = self
            .http
            .get(format!("{}/api/tags", self.base_url))
            .send()
            .await
            .context("failed to reach Ollama server; is `ollama serve` running?")?;

        if !resp.status().is_success() {
            return Err(anyhow!("Ollama /api/tags returned HTTP {}", resp.status()));
        }

        let body: TagsResponse = resp
            .json()
            .await
            .context("failed to parse Ollama /api/tags response")?;
        Ok(body.models)
    }

    pub async fn show(&self, name: &str) -> Result<Value> {
        let resp = self
            .http
            .post(format!("{}/api/show", self.base_url))
            .json(&serde_json::json!({ "name": name }))
            .send()
            .await
            .context("failed to call Ollama /api/show")?;

        if !resp.status().is_success() {
            return Err(anyhow!(
                "Ollama /api/show returned HTTP {} for model '{}'",
                resp.status(),
                name
            ));
        }

        Ok(resp
            .json()
            .await
            .context("failed to parse Ollama /api/show response")?)
    }

    pub async fn chat(&self, req: ChatRequest) -> Result<ChatResponse> {
        let resp = self
            .http
            .post(format!("{}/api/chat", self.base_url))
            .json(&req)
            .send()
            .await
            .context("failed to call Ollama /api/chat")?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(anyhow!("Ollama /api/chat returned HTTP {status}: {text}"));
        }

        let body: ChatResponse = resp
            .json()
            .await
            .context("failed to parse Ollama /api/chat response")?;
        Ok(body)
    }

    pub async fn pull(&self, name: &str) -> Result<()> {
        println!("⬇️  Pulling model '{name}' from Ollama registry...");
        let resp = self
            .http
            .post(format!("{}/api/pull", self.base_url))
            .json(&PullRequest {
                name: name.to_string(),
                stream: false,
            })
            .send()
            .await
            .context("failed to call Ollama /api/pull")?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(anyhow!("Ollama /api/pull returned HTTP {status}: {text}"));
        }

        let body: Value = resp
            .json()
            .await
            .context("failed to parse Ollama /api/pull response")?;
        let status = body
            .get("status")
            .and_then(|v| v.as_str())
            .unwrap_or("done");
        println!("✅ Pull finished: {status}");
        Ok(())
    }
}
