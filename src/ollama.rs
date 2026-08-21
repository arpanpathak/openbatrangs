use anyhow::{anyhow, Context, Result};
use futures_util::{Stream, StreamExt};
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
#[allow(dead_code)]
pub struct ChatResponse {
    pub message: ChatResponseMessage,
}

#[derive(Deserialize, Debug)]
#[allow(dead_code)]
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

    #[allow(dead_code)]
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

    /// Stream Ollama chat completions as NDJSON content deltas.
    pub async fn chat_stream(
        &self,
        mut req: ChatRequest,
    ) -> Result<impl Stream<Item = Result<String>> + Send + 'static> {
        req.stream = true;
        let resp = self
            .http
            .post(format!("{}/api/chat", self.base_url))
            .json(&req)
            .send()
            .await
            .context("failed to call Ollama /api/chat (stream)")?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(anyhow!("Ollama /api/chat returned HTTP {status}: {text}"));
        }

        let byte_stream = resp.bytes_stream();
        let stream = futures_util::stream::unfold(
            (byte_stream, String::new()),
            |(mut byte_stream, mut buf)| async move {
                loop {
                    match byte_stream.next().await {
                        Some(Ok(bytes)) => {
                            buf.push_str(&String::from_utf8_lossy(&bytes));
                            while let Some(pos) = buf.find('\n') {
                                let line = buf[..pos].trim().to_string();
                                buf = buf[pos + 1..].to_string();
                                if line.is_empty() {
                                    continue;
                                }
                                if let Ok(v) = serde_json::from_str::<Value>(&line) {
                                    if let Some(content) =
                                        v.pointer("/message/content").and_then(|c| c.as_str())
                                    {
                                        return Some((Ok(content.to_string()), (byte_stream, buf)));
                                    }
                                    if v.get("done").and_then(|d| d.as_bool()).unwrap_or(false) {
                                        return None;
                                    }
                                }
                            }
                        }
                        Some(Err(e)) => {
                            return Some((Err(anyhow!("stream error: {e}")), (byte_stream, buf)));
                        }
                        None => return None,
                    }
                }
            },
        );
        Ok(stream)
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
