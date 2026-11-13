//! Test doubles for the agent loops: a local HTTP server that replays canned
//! model replies, a reporter that records its events, a confirmer with a
//! fixed answer, and a history sink that keeps every snapshot.

use super::{AgentConfig, AgentEvent, Confirmer, HistorySink, Reporter, StdoutReporter};
use anyhow::{Context, Result};
use serde_json::{Value, json};
use std::path::Path;
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::net::{TcpListener, TcpStream};

/// One canned HTTP reply.
pub(super) struct Reply {
    /// Status line after `HTTP/1.1`, such as `200 OK`.
    status: &'static str,
    /// Value of the `Content-Type` header.
    content_type: &'static str,
    /// Response body.
    body: String,
}

impl Reply {
    /// A successful reply with a JSON body.
    pub(super) fn json(body: &Value) -> Self {
        Self {
            status: "200 OK",
            content_type: "application/json",
            body: body.to_string(),
        }
    }

    /// A server error with a plain-text body.
    pub(super) fn server_error(body: &str) -> Self {
        Self {
            status: "500 Internal Server Error",
            content_type: "text/plain",
            body: body.to_string(),
        }
    }

    /// An Ollama `/api/chat` stream that sends `content` as one message and
    /// then the `done` line.
    pub(super) fn ollama_stream(content: &str) -> Self {
        let message = json!({ "message": { "role": "assistant", "content": content } });
        let done = json!({ "done": true });

        Self {
            status: "200 OK",
            content_type: "application/x-ndjson",
            body: format!("{message}\n{done}\n"),
        }
    }

    /// An OpenAI server-sent-event stream with one event per delta, then
    /// `[DONE]`.
    pub(super) fn openai_stream(deltas: &[Value]) -> Self {
        let events: String = deltas
            .iter()
            .map(|delta| format!("data: {}\n\n", json!({ "choices": [{ "delta": delta }] })))
            .collect();

        Self {
            status: "200 OK",
            content_type: "text/event-stream",
            body: format!("{events}data: [DONE]\n\n"),
        }
    }
}

/// An OpenAI delta carrying reply text.
pub(super) fn text_delta(text: &str) -> Value {
    json!({ "content": text })
}

/// An OpenAI delta carrying reasoning text.
pub(super) fn thinking_delta(text: &str) -> Value {
    json!({ "reasoning_content": text })
}

/// An OpenAI delta carrying one complete tool call.
pub(super) fn tool_call_delta(id: &str, name: &str, arguments: &Value) -> Value {
    json!({ "tool_calls": [{
        "index": 0,
        "id": id,
        "type": "function",
        "function": { "name": name, "arguments": arguments.to_string() },
    }] })
}

/// Start a server on a free local port that answers the n-th request with
/// `replies[n]` and stops after the last one. Returns the base URL.
pub(super) async fn serve_replies(replies: Vec<Reply>) -> Result<String> {
    let listener = TcpListener::bind("127.0.0.1:0").await?;
    let base_url = format!("http://{}", listener.local_addr()?);

    tokio::spawn(answer_in_order(listener, replies));

    Ok(base_url)
}

/// Answer one connection per reply, stopping at the first failure.
async fn answer_in_order(listener: TcpListener, replies: Vec<Reply>) -> Result<()> {
    let mut served = Ok(());
    let mut replies = replies.into_iter();

    while let (Ok(()), Some(reply)) = (&served, replies.next()) {
        served = serve_one(&listener, &reply).await;
    }

    served
}

/// Accept one connection, read the whole request, and send `reply`.
async fn serve_one(listener: &TcpListener, reply: &Reply) -> Result<()> {
    let (socket, _peer) = listener.accept().await?;
    let mut reader = BufReader::new(socket);

    let head = read_head(&mut reader).await;
    let mut body = vec![0; content_length(&head)];
    reader.read_exact(&mut body).await?;

    write_reply(reader.into_inner(), reply).await
}

/// The request line and headers, read up to the blank line that ends them.
async fn read_head(reader: &mut BufReader<TcpStream>) -> String {
    let mut head = String::new();
    let mut read = reader.read_line(&mut head).await;

    // A header line is longer than the bare "\r\n" that ends the head.
    while let Ok(3..) = read {
        read = reader.read_line(&mut head).await;
    }

    head
}

/// The `Content-Length` value in a request head; 0 when it has none.
fn content_length(head: &str) -> usize {
    head.lines()
        .find_map(|line| {
            line.to_ascii_lowercase()
                .strip_prefix("content-length:")
                .and_then(|length| length.trim().parse().ok())
        })
        .unwrap_or(0)
}

/// Send `reply` and close the connection.
async fn write_reply(mut socket: TcpStream, reply: &Reply) -> Result<()> {
    let response = format!(
        "HTTP/1.1 {}\r\nContent-Type: {}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        reply.status,
        reply.content_type,
        reply.body.len(),
        reply.body
    );

    socket.write_all(response.as_bytes()).await?;
    socket.shutdown().await.context("close the fake connection")
}

/// Records every event, and the text the stdout reporter would print for it.
#[derive(Default)]
pub(super) struct RecordingReporter {
    /// Every event, in order.
    pub(super) events: Vec<AgentEvent>,
    /// The one-shot CLI output for the events, in order.
    pub(super) output: String,
    /// Renders the events in the one-shot CLI style.
    printer: StdoutReporter,
}

impl Reporter for RecordingReporter {
    /// Keep `event` and its printed text.
    fn report(&mut self, event: AgentEvent) {
        let printed = self.printer.render(event.clone());

        self.output.push_str(&printed);
        self.events.push(event);
    }
}

/// Answers every confirmation with `answer` and records the prompts.
pub(super) struct FixedConfirmer {
    /// The answer given to every prompt.
    pub(super) answer: bool,
    /// Every prompt received, in order.
    pub(super) prompts: Vec<String>,
}

impl FixedConfirmer {
    /// A confirmer that answers `answer` and has seen no prompt yet.
    pub(super) fn new(answer: bool) -> Self {
        Self {
            answer,
            prompts: Vec::new(),
        }
    }
}

impl Confirmer for FixedConfirmer {
    /// Record `prompt` and return the fixed answer.
    async fn confirm(&mut self, prompt: &str) -> Result<bool> {
        self.prompts.push(prompt.to_string());
        Ok(self.answer)
    }
}

/// Keeps every conversation snapshot it receives.
#[derive(Default)]
pub(super) struct RecordingSink {
    /// Snapshots in the order they arrived.
    pub(super) snapshots: Vec<Vec<Value>>,
}

impl HistorySink for RecordingSink {
    /// Keep `messages` as the newest snapshot.
    fn save(&mut self, messages: Vec<Value>) {
        self.snapshots.push(messages);
    }
}

/// A configuration for a writable workspace at `cwd` with no confirmation,
/// `max_steps` steps, and a search URL where nothing listens.
pub(super) fn test_config(cwd: &Path, max_steps: usize) -> AgentConfig {
    AgentConfig {
        cwd: cwd.to_path_buf(),
        max_steps,
        is_read_only: false,
        should_confirm: false,
        show_thinking: true,
        model_context: 8_192,
        max_ctx: None,
        searxng_url: "http://127.0.0.1:9".to_string(),
        search_key: None,
    }
}
