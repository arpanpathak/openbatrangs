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

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

static TEMP_DIR_COUNTER: AtomicU64 = AtomicU64::new(0);

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
    status: &'static str,
    content_type: &'static str,
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
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("mock server should bind");
    let base_url = format!("http://{}", listener.local_addr().unwrap());
    let handler = Arc::new(handler);

    tokio::spawn(async move {
        loop {
            let Ok((mut socket, _)) = listener.accept().await else {
                break;
            };
            let handler = Arc::clone(&handler);
            tokio::spawn(async move {
                let mut buffer = [0u8; 8192];
                let Ok(read) = socket.read(&mut buffer).await else {
                    return;
                };
                let request = String::from_utf8_lossy(&buffer[..read]);
                let path = request
                    .lines()
                    .next()
                    .and_then(|line| line.split_whitespace().nth(1))
                    .unwrap_or("/");
                let response = handler(path);
                let head = format!(
                    "HTTP/1.1 {}\r\nContent-Type: {}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    response.status,
                    response.content_type,
                    response.body.len()
                );
                let _ = socket.write_all(head.as_bytes()).await;
                let _ = socket.write_all(response.body.as_bytes()).await;
                let _ = socket.shutdown().await;
            });
        }
    });

    base_url
}
