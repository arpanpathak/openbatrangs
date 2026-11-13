//! # Background polling of a model server's stats
//!
//! A task asks the model server for its stats every
//! [`SERVER_STATS_INTERVAL`], only while the performance panel is on screen,
//! and keeps the latest answer in a [`ServerStatsFeed`] that the TUI reads
//! when it draws. The requests run on their own task, so a slow or failed
//! request never holds up drawing or input; a failed request shows as a
//! missing figure.

use super::server::{ServerStats, is_remote_server};
use crate::constants::perf::{
    OLLAMA_PS_PATH, SERVER_METRICS_PATH, SERVER_STATS_INTERVAL, SERVER_STATS_TIMEOUT,
    SERVER_SYSTEM_PATH,
};
use crate::ollama::OllamaClient;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, PoisonError};
use tokio::task::JoinHandle;
use tokio::time::MissedTickBehavior;

/// The latest server stats, shared between the polling task and the TUI.
#[derive(Clone, Default)]
pub struct ServerStatsFeed {
    /// Set while the performance panel is on screen; the task sends no
    /// requests while it is clear.
    panel_visible: Arc<AtomicBool>,
    /// The stats from the latest finished poll, or `None` before the first
    /// one finishes.
    latest: Arc<Mutex<Option<ServerStats>>>,
}

impl ServerStatsFeed {
    /// Record whether the performance panel is on screen.
    pub fn set_panel_visible(&self, visible: bool) {
        self.panel_visible.store(visible, Ordering::Relaxed);
    }

    /// Whether the performance panel is on screen.
    fn is_panel_visible(&self) -> bool {
        self.panel_visible.load(Ordering::Relaxed)
    }

    /// The stats from the latest finished poll, or `None` before the first
    /// one finishes. A poisoned lock still holds the stats stored before the
    /// panic.
    pub fn latest(&self) -> Option<ServerStats> {
        let latest = self.latest.lock().unwrap_or_else(PoisonError::into_inner);

        latest.clone()
    }

    /// Replace the stored stats with `stats`.
    pub fn store(&self, stats: ServerStats) {
        let mut latest = self.latest.lock().unwrap_or_else(PoisonError::into_inner);

        *latest = Some(stats);
    }
}

/// The polling task. Dropping it stops the task.
pub struct ServerPoller {
    /// Where the task stores what it reads.
    feed: ServerStatsFeed,
    /// The running task.
    task: JoinHandle<()>,
}

impl ServerPoller {
    /// Start polling when `client` talks to a server on another host, or
    /// `None` when the server runs on this machine.
    pub fn start_for_remote(client: &OllamaClient) -> Option<Self> {
        if !is_remote_server(&client.base_url) {
            return None;
        }

        Some(Self::start(client.clone()))
    }

    /// Start polling the server `client` talks to.
    pub fn start(client: OllamaClient) -> Self {
        let feed = ServerStatsFeed::default();
        let task = tokio::spawn(poll_forever(client, feed.clone()));

        Self { feed, task }
    }

    /// The feed the task stores its stats in.
    pub fn feed(&self) -> ServerStatsFeed {
        self.feed.clone()
    }
}

impl Drop for ServerPoller {
    fn drop(&mut self) {
        self.task.abort();
    }
}

/// Poll once per [`SERVER_STATS_INTERVAL`] until the task is stopped. A poll
/// that takes longer than the interval delays the next one instead of
/// starting a second one beside it.
async fn poll_forever(client: OllamaClient, feed: ServerStatsFeed) {
    let mut interval = tokio::time::interval(SERVER_STATS_INTERVAL);
    interval.set_missed_tick_behavior(MissedTickBehavior::Delay);

    loop {
        interval.tick().await;
        poll_when_visible(&client, &feed).await;
    }
}

/// Read the server's stats into `feed` while the panel is on screen.
async fn poll_when_visible(client: &OllamaClient, feed: &ServerStatsFeed) {
    if !feed.is_panel_visible() {
        return;
    }

    feed.store(fetch_server_stats(client).await);
}

/// Ask the server for `/v1/system`, `/metrics` and `/api/ps` at once and
/// parse the answers.
pub async fn fetch_server_stats(client: &OllamaClient) -> ServerStats {
    let (system, metrics, processes) = tokio::join!(
        published_text(client, SERVER_SYSTEM_PATH),
        published_text(client, SERVER_METRICS_PATH),
        published_text(client, OLLAMA_PS_PATH),
    );

    ServerStats::from_bodies(system.as_deref(), metrics.as_deref(), processes.as_deref())
}

/// The body of `GET {server root}{path}`, or `None` when the request fails or
/// takes longer than [`SERVER_STATS_TIMEOUT`].
async fn published_text(client: &OllamaClient, path: &str) -> Option<String> {
    tokio::time::timeout(SERVER_STATS_TIMEOUT, client.server_text(path))
        .await
        .ok()
        .flatten()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{MockResponse, spawn_mock_server};
    use std::sync::atomic::AtomicUsize;
    use std::time::Duration;

    /// A `/v1/system` body with only the host and the GPU utilization.
    const SYSTEM: &str =
        r#"{"host":"thor","sampled_at_unix_ms":1,"gpu":{"utilization_percent":37.0}}"#;

    /// The answer of a server that publishes `/v1/system` and Prometheus
    /// metrics but is not Ollama.
    fn stats_route(path: &str) -> MockResponse {
        match path {
            "/v1/system" => MockResponse::json("200 OK", SYSTEM),
            "/metrics" => MockResponse::text("200 OK", "vllm:num_requests_running 2\n"),
            _ => MockResponse::text("404 Not Found", "not found"),
        }
    }

    #[tokio::test]
    async fn fetch_reads_every_published_endpoint() -> anyhow::Result<()> {
        let base_url = spawn_mock_server(stats_route).await;
        let client = OllamaClient::openai(&format!("{base_url}/v1"), None, true)?;

        let stats = fetch_server_stats(&client).await;

        assert_eq!(
            stats.system.and_then(|system| system.host).as_deref(),
            Some("thor")
        );
        assert_eq!(stats.engine_load.and_then(|load| load.running), Some(2.0));
        assert_eq!(stats.loaded_models, None);
        Ok(())
    }

    #[tokio::test]
    async fn poller_asks_only_while_the_panel_is_visible() -> anyhow::Result<()> {
        let requests = Arc::new(AtomicUsize::new(0));
        let counted = Arc::clone(&requests);
        let base_url = spawn_mock_server(move |path| {
            counted.fetch_add(1, Ordering::Relaxed);
            stats_route(path)
        })
        .await;
        let poller = ServerPoller::start(OllamaClient::new(&base_url)?);
        let feed = poller.feed();

        tokio::time::sleep(Duration::from_millis(300)).await;
        let requests_while_hidden = requests.load(Ordering::Relaxed);

        feed.set_panel_visible(true);
        tokio::time::sleep(SERVER_STATS_INTERVAL + Duration::from_millis(500)).await;

        assert_eq!(requests_while_hidden, 0);
        assert_eq!(requests.load(Ordering::Relaxed), 3);
        assert!(feed.latest().and_then(|stats| stats.system).is_some());
        Ok(())
    }

    /// Accept `count` connections on `listener` and return them still open
    /// and unanswered, like a server that hangs.
    async fn accept_without_answering(
        listener: tokio::net::TcpListener,
        count: usize,
    ) -> std::io::Result<Vec<tokio::net::TcpStream>> {
        let mut held_connections = Vec::new();

        for _ in 0..count {
            let (socket, _peer) = listener.accept().await?;
            held_connections.push(socket);
        }

        Ok(held_connections)
    }

    #[tokio::test]
    async fn a_server_that_never_answers_counts_as_unpublished() -> anyhow::Result<()> {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
        let base_url = format!("http://{}/v1", listener.local_addr()?);
        let holder = tokio::spawn(accept_without_answering(listener, 3));
        let client = OllamaClient::openai(&base_url, None, true)?;
        let started = std::time::Instant::now();

        let stats = fetch_server_stats(&client).await;

        assert!(started.elapsed() < SERVER_STATS_TIMEOUT + Duration::from_secs(1));
        assert_eq!(stats, ServerStats::default());
        assert_eq!(holder.await??.len(), 3);
        Ok(())
    }

    #[test]
    fn a_new_feed_has_no_stats_until_one_is_stored() {
        let feed = ServerStatsFeed::default();

        assert_eq!(feed.latest(), None);

        feed.store(ServerStats::default());

        assert_eq!(feed.latest(), Some(ServerStats::default()));
    }

    #[tokio::test]
    async fn only_a_server_on_another_host_is_polled() -> anyhow::Result<()> {
        let local = OllamaClient::new("http://127.0.0.1:11434")?;
        let remote = OllamaClient::openai("http://192.0.2.1:8079/v1", None, true)?;

        assert!(ServerPoller::start_for_remote(&local).is_none());
        assert!(ServerPoller::start_for_remote(&remote).is_some());
        Ok(())
    }
}
