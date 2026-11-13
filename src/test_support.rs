//! Test-only helpers shared across unit-test modules.
//!
//! Temp-dir helpers must be unique even when tests run in parallel on multiple
//! threads: a bare nanosecond timestamp can collide, causing one test to delete
//! another test's directory mid-write.
//!
//! The mock HTTP server lets tests exercise the real `reqwest` client (and
//! any code built on top of it) against a local TCP listener, without needing
//! a real Ollama installation or network access.
//!
//! ## References
//!
//! - Tokio TCP listener: <https://docs.rs/tokio/latest/tokio/net/struct.TcpListener.html>
//! - Hypertext Transfer Protocol basics: <https://developer.mozilla.org/en-US/docs/Web/HTTP>

use crate::ollama::OllamaModel;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

/// Counter that makes every temporary directory name unique within the process.
static TEMP_DIR_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Size of the buffer that holds one request; every test request fits in it.
const REQUEST_BUFFER_BYTES: usize = 8192;

/// The request handler of a mock server: request path in, canned response out.
type MockHandler = Arc<dyn Fn(&str) -> MockResponse + Send + Sync>;

/// Create a new empty temporary directory with a process-unique name.
pub(crate) fn unique_temp_dir(prefix: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let counter = TEMP_DIR_COUNTER.fetch_add(1, Ordering::Relaxed);
    let path =
        std::env::temp_dir().join(format!("{prefix}-{}-{nanos}-{counter}", std::process::id()));

    std::fs::create_dir_all(&path).expect("create temp dir");

    path
}

/// A canned HTTP response served by [`spawn_mock_server`].
pub(crate) struct MockResponse {
    /// Status line after `HTTP/1.1`, e.g. `200 OK`.
    status: &'static str,
    /// Value of the `Content-Type` header.
    content_type: &'static str,
    /// Response body.
    body: String,
}

impl MockResponse {
    /// Build a JSON response with the given status line.
    pub(crate) fn json(status: &'static str, body: impl Into<String>) -> Self {
        Self {
            status,
            content_type: "application/json",
            body: body.into(),
        }
    }

    /// Build a plain-text response with the given status line.
    pub(crate) fn text(status: &'static str, body: impl Into<String>) -> Self {
        Self {
            status,
            content_type: "text/plain",
            body: body.into(),
        }
    }

    /// The full HTTP/1.1 response: status line, headers and body.
    fn http_text(&self) -> String {
        format!(
            "HTTP/1.1 {}\r\nContent-Type: {}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            self.status,
            self.content_type,
            self.body.len(),
            self.body
        )
    }
}

/// Spawn a minimal HTTP server that routes requests by request path.
///
/// # Parameters
///
/// - `handler`: closure receiving the request path (e.g. `/api/tags`) and
///   returning a canned [`MockResponse`].
///
/// # Returns
///
/// Base URL like `http://127.0.0.1:54321` that can be passed to
/// [`crate::ollama::OllamaClient::new`].
pub(crate) async fn spawn_mock_server(
    handler: impl Fn(&str) -> MockResponse + Send + Sync + 'static,
) -> String {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("mock server should bind");
    let address = listener
        .local_addr()
        .expect("a bound listener has an address");

    tokio::spawn(serve_connections(listener, Arc::new(handler)));

    format!("http://{address}")
}

/// Answer every connection the listener accepts, each on its own task.
async fn serve_connections(listener: TcpListener, handler: MockHandler) {
    loop {
        let accepted = listener.accept().await;

        tokio::spawn(answer_connection(accepted, Arc::clone(&handler)));
    }
}

/// Answer one accepted connection; a failed accept has nothing to answer.
async fn answer_connection(
    accepted: std::io::Result<(TcpStream, SocketAddr)>,
    handler: MockHandler,
) {
    let Ok((socket, _peer)) = accepted else {
        return;
    };

    answer_request(socket, handler).await;
}

/// Read one request, build the response with the handler and send it back.
async fn answer_request(mut socket: TcpStream, handler: MockHandler) {
    let mut buffer = [0u8; REQUEST_BUFFER_BYTES];
    let read = socket.read(&mut buffer).await.unwrap_or(0);
    let request = String::from_utf8_lossy(&buffer[..read]);

    let response = handler(request_path(&request));

    // A client that hung up early does not need its response.
    write_response(&mut socket, &response).await.ok();
}

/// The path in the request line, e.g. `/api/tags`, or `/` when there is none.
fn request_path(request: &str) -> &str {
    request
        .lines()
        .next()
        .and_then(|line| line.split_whitespace().nth(1))
        .unwrap_or("/")
}

/// Write the whole response and close the sending half of the socket.
async fn write_response(socket: &mut TcpStream, response: &MockResponse) -> std::io::Result<()> {
    socket.write_all(response.http_text().as_bytes()).await?;
    socket.shutdown().await
}

/// An installed model with full details, as `/api/tags` reports it.
pub(crate) fn sample_model(
    name: &str,
    context_length: u64,
    parameter_size: &str,
    quantization: &str,
    size_bytes: u64,
) -> anyhow::Result<OllamaModel> {
    let model = serde_json::from_value(serde_json::json!({
        "name": name,
        "size": size_bytes,
        "details": {
            "parameter_size": parameter_size,
            "quantization_level": quantization,
            "context_length": context_length,
        },
    }))?;

    Ok(model)
}

/// A status callback for tests that do not check the messages.
pub(crate) fn ignore_status(_message: &str) {}

/// Status messages collected from a status callback, so a test can check them.
#[derive(Default)]
pub(crate) struct StatusLog {
    /// Every message received, in order.
    messages: Mutex<Vec<String>>,
}

impl StatusLog {
    /// Store one status message. A poisoned lock still holds the messages
    /// stored before the panic, so recording continues.
    pub(crate) fn record(&self, message: &str) {
        let mut messages = self.messages.lock().unwrap_or_else(PoisonError::into_inner);

        messages.push(message.to_string());
    }

    /// Whether any stored message contains `text`.
    pub(crate) fn contains(&self, text: &str) -> bool {
        self.messages
            .lock()
            .map(|messages| messages.iter().any(|message| message.contains(text)))
            .unwrap_or(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_path_is_the_second_word_of_the_request_line() {
        assert_eq!(
            request_path("GET /api/tags HTTP/1.1\r\nHost: x"),
            "/api/tags"
        );

        assert_eq!(request_path(""), "/");
    }

    /// A route that answers with the request path as the body.
    fn echo_path(path: &str) -> MockResponse {
        MockResponse::text("200 OK", path.to_string())
    }

    #[tokio::test]
    async fn mock_server_answers_with_the_route() -> anyhow::Result<()> {
        let base_url = spawn_mock_server(echo_path).await;

        let body = reqwest::get(format!("{base_url}/api/tags"))
            .await?
            .text()
            .await?;

        assert_eq!(body, "/api/tags");
        Ok(())
    }

    #[tokio::test]
    async fn failed_accept_is_ignored() {
        let accepted = Err(std::io::Error::other("accept failed"));

        answer_connection(accepted, Arc::new(echo_path)).await;
    }

    #[test]
    fn status_log_finds_recorded_messages() {
        let log = StatusLog::default();

        log.record("pulling manifest");

        assert!(log.contains("manifest"));
        assert!(!log.contains("success"));
    }

    #[test]
    fn ignore_status_accepts_any_message() {
        ignore_status("anything");
    }
}
