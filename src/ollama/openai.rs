//! # OpenAI-compatible servers (llama-server, vLLM, trtllm-serve)
//!
//! The same [`ChatRequest`] used for Ollama is translated into a
//! `POST /chat/completions` body, and the server-sent event stream is parsed
//! back into [`Delta`]s. llama-server streams a thinking model's reasoning in
//! `delta.reasoning_content` and the reply in `delta.content`.
//!
//! ## References
//!
//! - OpenAI chat completions streaming: <https://platform.openai.com/docs/api-reference/chat-streaming>
//! - llama-server API: <https://github.com/ggml-org/llama.cpp/tree/master/tools/server>

use super::types::{ChatRequest, Delta, OllamaModel, Role};
use serde_json::{json, Value};

/// Result of draining complete server-sent event lines from the buffer.
pub(super) enum EventDrain {
    /// A delta is ready to emit.
    Delta(Delta),
    /// The `data: [DONE]` marker was seen.
    Done,
    /// No complete event line is available yet.
    NeedMore,
}

/// Build the `/chat/completions` body for a chat request.
///
/// Tool results are sent with the `user` role: the OpenAI `tool` role needs a
/// `tool_call_id`, and openBatarangs passes tool calls as JSON text instead.
pub(super) fn chat_body(request: &ChatRequest, think: bool) -> Value {
    let messages: Vec<Value> = request
        .messages
        .iter()
        .map(|message| {
            let role = match message.role {
                Role::System => "system",
                Role::User | Role::Tool => "user",
                Role::Assistant => "assistant",
            };
            json!({ "role": role, "content": message.content })
        })
        .collect();
    let mut body = json!({
        "model": request.model,
        "messages": messages,
        "stream": true,
        "chat_template_kwargs": { "enable_thinking": think },
    });
    if request.format.is_some() {
        body["response_format"] = json!({ "type": "json_object" });
    }
    if let Some(options) = &request.options {
        if let Some(temperature) = options.get("temperature") {
            body["temperature"] = temperature.clone();
        }
        if let Some(max_tokens) = options.get("num_predict") {
            body["max_tokens"] = max_tokens.clone();
        }
    }
    body
}

/// Consume complete lines from `buffer`, returning the first meaningful event.
pub(super) fn drain_events(buffer: &mut String) -> EventDrain {
    while let Some(newline_pos) = buffer.find('\n') {
        let line = buffer[..newline_pos].trim().to_string();
        *buffer = buffer[newline_pos + 1..].to_string();
        let Some(data) = line.strip_prefix("data:").map(str::trim) else {
            continue;
        };
        if data == "[DONE]" {
            return EventDrain::Done;
        }
        if let Some(delta) = parse_event(data) {
            return EventDrain::Delta(delta);
        }
    }
    EventDrain::NeedMore
}

/// Parse one `data:` payload into a content or reasoning delta.
fn parse_event(data: &str) -> Option<Delta> {
    let value = serde_json::from_str::<Value>(data).ok()?;
    let delta = value.pointer("/choices/0/delta")?;
    let text = |key: &str| {
        delta
            .get(key)
            .and_then(Value::as_str)
            .filter(|text| !text.is_empty())
            .map(str::to_string)
    };
    text("content")
        .map(Delta::Content)
        .or_else(|| text("reasoning_content").map(Delta::Thinking))
}

/// Turn a `GET /models` body into model entries.
///
/// llama-server reports the model file path as the id; only the file name is
/// kept so the model list stays readable. The server ignores the name in
/// requests, so nothing is lost.
pub(super) fn models_from_list(body: &Value) -> Vec<OllamaModel> {
    body.get("data")
        .and_then(Value::as_array)
        .map(|models| {
            models
                .iter()
                .filter_map(|model| model.get("id").and_then(Value::as_str))
                .map(|id| OllamaModel {
                    name: short_model_name(id),
                    size: 0,
                    details: None,
                })
                .collect()
        })
        .unwrap_or_default()
}

/// File name of a model path without the `.gguf` extension.
fn short_model_name(id: &str) -> String {
    let file = id.rsplit('/').next().unwrap_or(id);
    file.strip_suffix(".gguf").unwrap_or(file).to_string()
}

/// Server address without the `/v1` suffix, where llama-server serves `/props`.
pub(super) fn server_root(base_url: &str) -> &str {
    base_url.strip_suffix("/v1").unwrap_or(base_url)
}

/// Shape a `/props` body like Ollama's `/api/show`, so context lookup works
/// for both servers.
pub(super) fn show_from_props(props: &Value) -> Value {
    match props
        .pointer("/default_generation_settings/n_ctx")
        .and_then(Value::as_u64)
    {
        Some(context) => json!({ "details": { "context_length": context } }),
        None => json!({}),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ollama::ChatMessage;

    fn request(format: Option<Value>) -> ChatRequest {
        ChatRequest {
            model: "nemotron".to_string(),
            messages: vec![
                ChatMessage {
                    role: Role::System,
                    content: "rules".to_string(),
                },
                ChatMessage {
                    role: Role::Tool,
                    content: "tool output".to_string(),
                },
            ],
            stream: false,
            keep_alive: None,
            format,
            options: Some(json!({"temperature": 0.2, "num_ctx": 8192, "num_predict": 4000})),
        }
    }

    #[test]
    fn chat_body_maps_roles_options_and_thinking() {
        let body = chat_body(&request(Some(json!("json"))), true);
        assert_eq!(body["messages"][0]["role"], "system");
        assert_eq!(body["messages"][1]["role"], "user");
        assert_eq!(body["stream"], true);
        assert_eq!(body["temperature"], 0.2);
        assert_eq!(body["max_tokens"], 4000);
        assert_eq!(body["response_format"]["type"], "json_object");
        assert_eq!(body["chat_template_kwargs"]["enable_thinking"], true);
    }

    #[test]
    fn chat_body_without_format_has_no_response_format() {
        let body = chat_body(&request(None), false);
        assert!(body.get("response_format").is_none());
        assert_eq!(body["chat_template_kwargs"]["enable_thinking"], false);
    }

    #[test]
    fn drains_reasoning_content_and_done_in_order() {
        let mut buffer = String::from(concat!(
            "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"hmm\"}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\"{}\"}}]}\n\n",
            "data: [DONE]\n\n",
        ));
        assert!(matches!(drain_events(&mut buffer), EventDrain::Delta(Delta::Thinking(text)) if text == "hmm"));
        assert!(matches!(drain_events(&mut buffer), EventDrain::Delta(Delta::Content(text)) if text == "{}"));
        assert!(matches!(drain_events(&mut buffer), EventDrain::Done));
    }

    #[test]
    fn waits_for_a_complete_line() {
        let mut buffer = String::from("data: {\"choices\":[{\"delta\":{\"content\":\"pa");
        assert!(matches!(drain_events(&mut buffer), EventDrain::NeedMore));
    }

    #[test]
    fn skips_empty_deltas() {
        let mut buffer = String::from("data: {\"choices\":[{\"delta\":{\"content\":\"\"}}]}\n");
        assert!(matches!(drain_events(&mut buffer), EventDrain::NeedMore));
    }

    #[test]
    fn model_list_keeps_file_names() {
        let body = json!({"data": [{"id": "/home/a/models/Nemotron-Q8_0.gguf"}, {"id": "plain"}]});
        let names: Vec<String> = models_from_list(&body).into_iter().map(|m| m.name).collect();
        assert_eq!(names, vec!["Nemotron-Q8_0", "plain"]);
    }

    #[test]
    fn props_become_context_length() {
        let props = json!({"default_generation_settings": {"n_ctx": 1_048_576}});
        assert_eq!(show_from_props(&props)["details"]["context_length"], 1_048_576);
        assert_eq!(show_from_props(&json!({})), json!({}));
    }

    #[test]
    fn root_drops_v1() {
        assert_eq!(server_root("http://127.0.0.1:8079/v1"), "http://127.0.0.1:8079");
        assert_eq!(server_root("http://127.0.0.1:8079"), "http://127.0.0.1:8079");
    }
}
