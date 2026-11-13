//! # OpenAI-compatible servers (llama-server, vLLM, trtllm-serve)
//!
//! The same [`ChatRequest`] used for Ollama is translated into a
//! `POST /chat/completions` body, and the server-sent event stream is parsed
//! back into [`Delta`]s. llama-server streams a thinking model's reasoning in
//! `delta.reasoning_content`, vLLM in `delta.reasoning`, and both the reply in
//! `delta.content`.
//!
//! ## References
//!
//! - OpenAI chat completions streaming: <https://platform.openai.com/docs/api-reference/chat-streaming>
//! - llama-server API: <https://github.com/ggml-org/llama.cpp/tree/master/tools/server>

use super::stream::{Drain, take_line};
use super::types::{ChatMessage, ChatRequest, Delta, OllamaModel, Role, ToolCallDelta};
use serde_json::{Value, json};

/// Build the `/chat/completions` body for a chat request.
///
/// Tool results are sent with the `user` role: the OpenAI `tool` role needs a
/// `tool_call_id`, and openBatarangs passes tool calls as JSON text instead.
pub(super) fn chat_body(request: &ChatRequest, think: bool) -> Value {
    let messages: Vec<Value> = request.messages.iter().map(message_json).collect();
    let mut body = json!({
        "model": request.model,
        "messages": messages,
        "stream": true,
        "chat_template_kwargs": { "enable_thinking": think },
    });

    if request.format.is_some() {
        body["response_format"] = json!({ "type": "json_object" });
    }

    add_sampling_options(&mut body, request.options.as_ref());

    body
}

/// One chat message in the OpenAI format.
fn message_json(message: &ChatMessage) -> Value {
    json!({ "role": openai_role(&message.role), "content": message.content })
}

/// The OpenAI role name for `role`; tool results travel as `user` messages.
fn openai_role(role: &Role) -> &'static str {
    match role {
        Role::System => "system",
        Role::User | Role::Tool => "user",
        Role::Assistant => "assistant",
    }
}

/// Copy Ollama's `temperature` and `num_predict` options into `body` under
/// their OpenAI names, `temperature` and `max_tokens`.
fn add_sampling_options(body: &mut Value, options: Option<&Value>) {
    let Some(options) = options else {
        return;
    };

    copy_field(options, "temperature", body, "temperature");
    copy_field(options, "num_predict", body, "max_tokens");
}

/// Set `target[target_key]` to `source[source_key]` when the source has it.
fn copy_field(source: &Value, source_key: &str, target: &mut Value, target_key: &str) {
    let Some(value) = source.get(source_key) else {
        return;
    };

    target[target_key] = value.clone();
}

/// Consume complete lines from `buffer`, returning the first meaningful event.
///
/// Lines after that event stay in `buffer` for the next call.
pub(super) fn drain_events(buffer: &mut String) -> Drain {
    std::iter::from_fn(|| take_line(buffer))
        .find_map(|line| event_drain(&line))
        .unwrap_or(Drain::NeedMore)
}

/// The event on one server-sent event line; `None` for lines that are not
/// `data:` lines or carry nothing to show.
fn event_drain(line: &str) -> Option<Drain> {
    line.strip_prefix("data:")
        .map(str::trim)
        .and_then(data_drain)
}

/// [`Drain::Done`] for the `[DONE]` marker, else the delta in the payload.
fn data_drain(data: &str) -> Option<Drain> {
    if data == "[DONE]" {
        return Some(Drain::Done);
    }

    parse_event(data).map(Drain::Delta)
}

/// Parse one `data:` payload into a content, reasoning, or tool-call delta.
fn parse_event(data: &str) -> Option<Delta> {
    let value = serde_json::from_str::<Value>(data).ok()?;

    value.pointer("/choices/0/delta").and_then(parse_delta)
}

/// The delta in one streamed choice: its tool calls when it has any, else
/// its reply text, else its reasoning.
fn parse_delta(delta: &Value) -> Option<Delta> {
    tool_calls_delta(delta).or_else(|| text_delta(delta))
}

/// The tool calls of a streamed delta, `None` when it has none.
fn tool_calls_delta(delta: &Value) -> Option<Delta> {
    let calls = parse_tool_calls(delta);

    if calls.is_empty() {
        return None;
    }

    Some(Delta::ToolCalls(calls))
}

/// The reply text of a streamed delta, or else its reasoning: llama-server
/// sends `reasoning_content` and vLLM sends `reasoning`.
fn text_delta(delta: &Value) -> Option<Delta> {
    shown_text(delta, "content")
        .map(Delta::Content)
        .or_else(|| shown_text(delta, "reasoning_content").map(Delta::Thinking))
        .or_else(|| shown_text(delta, "reasoning").map(Delta::Thinking))
}

/// The string at `key`; `None` when it is missing or empty.
fn shown_text(delta: &Value, key: &str) -> Option<String> {
    delta
        .get(key)
        .and_then(Value::as_str)
        .filter(|text| !text.is_empty())
        .map(str::to_string)
}

/// Extract the incremental `tool_calls` array of one streamed delta.
fn parse_tool_calls(delta: &Value) -> Vec<ToolCallDelta> {
    delta
        .get("tool_calls")
        .and_then(Value::as_array)
        .map(|calls| calls.iter().map(parse_tool_call).collect())
        .unwrap_or_default()
}

/// Map one `tool_calls` entry to its incremental delta.
fn parse_tool_call(call: &Value) -> ToolCallDelta {
    let function = call.get("function");

    ToolCallDelta {
        index: call.get("index").and_then(Value::as_u64).unwrap_or(0) as u32,
        id: string_at(call, "id"),
        name: function.and_then(|function| string_at(function, "name")),
        arguments: function
            .and_then(|function| string_at(function, "arguments"))
            .unwrap_or_default(),
    }
}

/// Read an optional string field from a JSON object.
fn string_at(value: &Value, key: &str) -> Option<String> {
    value.get(key).and_then(Value::as_str).map(str::to_string)
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

/// Shape one model's entry in a `/models` body like Ollama's `/api/show`.
///
/// TensorRT Edge-LLM has no `/props`; it reports each model's context window
/// in the list, as `max_model_len`.
pub(super) fn show_from_models(list: &Value, name: &str) -> Option<Value> {
    list.get("data")
        .and_then(Value::as_array)
        .and_then(|models| listed_model(models, name))
        .and_then(|model| model.get("max_model_len"))
        .and_then(Value::as_u64)
        .map(context_show)
}

/// The entry of `models` whose `id` is `name`.
fn listed_model<'a>(models: &'a [Value], name: &str) -> Option<&'a Value> {
    models
        .iter()
        .find(|model| model.get("id").and_then(Value::as_str) == Some(name))
}

/// Shape a `/props` body like Ollama's `/api/show`, so context lookup works
/// for both servers.
pub(super) fn show_from_props(props: &Value) -> Value {
    props
        .pointer("/default_generation_settings/n_ctx")
        .and_then(Value::as_u64)
        .map_or_else(|| json!({}), context_show)
}

/// An `/api/show`-shaped body that reports a context window of `context`
/// tokens.
fn context_show(context: u64) -> Value {
    json!({ "details": { "context_length": context } })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A request with a system, a tool and an assistant message, and the
    /// given response format and sampling options.
    fn request(format: Option<Value>, options: Option<Value>) -> ChatRequest {
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
                ChatMessage {
                    role: Role::Assistant,
                    content: "reply".to_string(),
                },
            ],
            stream: false,
            keep_alive: None,
            format,
            options,
        }
    }

    /// The text of a drained delta of either kind, `None` for anything else.
    fn delta_text(drain: Drain) -> Option<String> {
        match drain {
            Drain::Delta(Delta::Content(text) | Delta::Thinking(text)) => Some(text),
            Drain::Delta(Delta::ToolCalls(_)) | Drain::Done | Drain::NeedMore => None,
        }
    }

    #[test]
    fn chat_body_maps_roles_options_and_thinking() {
        let options = json!({"temperature": 0.2, "num_ctx": 8192, "num_predict": 4000});

        let body = chat_body(&request(Some(json!("json")), Some(options)), true);

        assert_eq!(body["messages"][0]["role"], "system");
        assert_eq!(body["messages"][1]["role"], "user");
        assert_eq!(body["messages"][2]["role"], "assistant");
        assert_eq!(body["stream"], true);
        assert_eq!(body["temperature"], 0.2);
        assert_eq!(body["max_tokens"], 4000);
        assert_eq!(body["response_format"]["type"], "json_object");
        assert_eq!(body["chat_template_kwargs"]["enable_thinking"], true);
    }

    #[test]
    fn chat_body_without_format_has_no_response_format() {
        let body = chat_body(&request(None, None), false);

        assert!(body.get("response_format").is_none());
        assert!(body.get("temperature").is_none());
        assert_eq!(body["chat_template_kwargs"]["enable_thinking"], false);
    }

    #[test]
    fn chat_body_copies_only_the_options_that_are_set() {
        let body = chat_body(&request(None, Some(json!({"temperature": 0.5}))), false);

        assert_eq!(body["temperature"], 0.5);
        assert!(body.get("max_tokens").is_none());
    }

    #[test]
    fn reads_vllm_reasoning_as_thinking() {
        let delta = parse_event(r#"{"choices":[{"delta":{"reasoning":"hmm"}}]}"#);

        assert_eq!(delta, Some(Delta::Thinking("hmm".to_string())));
    }

    #[test]
    fn reads_streamed_tool_calls() {
        let first = parse_event(
            r#"{"choices":[{"delta":{"tool_calls":[
                {"index":1,"id":"call_7","function":{"name":"read_file","arguments":"{\"pa"}}
            ]}}]}"#,
        );
        let later = parse_event(r#"{"choices":[{"delta":{"tool_calls":[{}]}}]}"#);

        assert_eq!(
            first,
            Some(Delta::ToolCalls(vec![ToolCallDelta {
                index: 1,
                id: Some("call_7".to_string()),
                name: Some("read_file".to_string()),
                arguments: "{\"pa".to_string(),
            }]))
        );
        assert_eq!(
            later,
            Some(Delta::ToolCalls(vec![ToolCallDelta::default()]))
        );
    }

    #[test]
    fn an_empty_tool_call_list_falls_back_to_the_text() {
        let delta = parse_event(r#"{"choices":[{"delta":{"tool_calls":[],"content":"hi"}}]}"#);

        assert_eq!(delta, Some(Delta::Content("hi".to_string())));
    }

    #[test]
    fn drains_reasoning_content_and_done_in_order() {
        let mut buffer = String::from(concat!(
            ": keep-alive comment\n",
            "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"hmm\"}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\"{}\"}}]}\n\n",
            "data: [DONE]\n\n",
        ));

        assert_eq!(
            delta_text(drain_events(&mut buffer)).as_deref(),
            Some("hmm")
        );
        assert_eq!(delta_text(drain_events(&mut buffer)).as_deref(), Some("{}"));
        assert!(matches!(drain_events(&mut buffer), Drain::Done));
        assert_eq!(delta_text(Drain::Done), None);
    }

    #[test]
    fn waits_for_a_complete_line() {
        let mut buffer = String::from("data: {\"choices\":[{\"delta\":{\"content\":\"pa");

        assert!(matches!(drain_events(&mut buffer), Drain::NeedMore));
    }

    #[test]
    fn skips_empty_deltas() {
        let mut buffer = String::from("data: {\"choices\":[{\"delta\":{\"content\":\"\"}}]}\n");

        assert!(matches!(drain_events(&mut buffer), Drain::NeedMore));
    }

    #[test]
    fn model_list_keeps_file_names() {
        let body = json!({"data": [{"id": "/home/a/models/Nemotron-Q8_0.gguf"}, {"id": "plain"}]});

        let names: Vec<String> = models_from_list(&body)
            .into_iter()
            .map(|model| model.name)
            .collect();

        assert_eq!(names, vec!["Nemotron-Q8_0", "plain"]);
    }

    #[test]
    fn a_listed_model_reports_its_context_length() {
        let list = json!({"data": [
            {"id": "qwen", "max_model_len": 65_536},
            {"id": "plain"}
        ]});

        assert_eq!(
            show_from_models(&list, "qwen"),
            Some(json!({"details": {"context_length": 65_536}}))
        );
        assert_eq!(show_from_models(&list, "plain"), None);
        assert_eq!(show_from_models(&list, "other"), None);
        assert_eq!(show_from_models(&json!({}), "qwen"), None);
    }

    #[test]
    fn props_become_context_length() {
        let props = json!({"default_generation_settings": {"n_ctx": 1_048_576}});

        assert_eq!(
            show_from_props(&props)["details"]["context_length"],
            1_048_576
        );
        assert_eq!(show_from_props(&json!({})), json!({}));
    }

    #[test]
    fn root_drops_v1() {
        assert_eq!(
            server_root("http://127.0.0.1:8079/v1"),
            "http://127.0.0.1:8079"
        );
        assert_eq!(
            server_root("http://127.0.0.1:8079"),
            "http://127.0.0.1:8079"
        );
    }
}
