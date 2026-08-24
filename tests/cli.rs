//! Process-level integration tests for the CLI binary.
//!
//! These tests exercise the real `main()` dispatch by spawning the compiled
//! binary. A tiny blocking HTTP server stands in for Ollama so the network
//! paths are covered end-to-end without a real server.
//!
//! ## References
//!
//! - Cargo integration tests: <https://doc.rust-lang.org/cargo/reference/writing-tests.html>
//! - `CARGO_BIN_EXE_<name>` environment variable: <https://doc.rust-lang.org/cargo/reference/environment-variables.html>

use std::io::{Read, Write};
use std::net::{Shutdown, TcpListener};
use std::process::Command;
use std::thread;

/// Tiny blocking HTTP server used to stand in for Ollama.
struct MockServer {
    addr: String,
}

impl MockServer {
    fn start(mut handler: impl FnMut(&str) -> (u16, String) + Send + 'static) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind mock server");
        let addr = listener.local_addr().unwrap();
        thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(mut stream) = stream else {
                    break;
                };
                let mut buffer = [0u8; 8192];
                let Ok(read) = stream.read(&mut buffer) else {
                    continue;
                };
                let request = String::from_utf8_lossy(&buffer[..read]);
                let path = request
                    .lines()
                    .next()
                    .and_then(|line| line.split_whitespace().nth(1))
                    .unwrap_or("/");
                let (status, body) = handler(path);
                let response = format!(
                    "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                );
                let _ = stream.write_all(response.as_bytes());
                let _ = stream.shutdown(Shutdown::Write);
            }
        });
        Self {
            addr: format!("http://{addr}"),
        }
    }

    fn url(&self) -> &str {
        &self.addr
    }
}

fn bin() -> &'static str {
    env!("CARGO_BIN_EXE_openbatrangs")
}

#[test]
fn version_flag_prints_version() {
    let output = Command::new(bin()).arg("--version").output().unwrap();
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("openbatrangs"));
}

#[test]
fn help_flag_prints_usage() {
    let output = Command::new(bin()).arg("--help").output().unwrap();
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Usage"));
    assert!(stdout.contains("doctor"));
}

#[test]
fn list_models_subcommand_prints_model_table() {
    let server = MockServer::start(|path| {
        assert_eq!(path, "/api/tags");
        (
            200,
            r#"{"models":[{"name":"qwen2.5-coder:3b","size":123,"details":{"parameter_size":"3B","quantization_level":"Q4_0","context_length":8192}}]}"#
                .to_string(),
        )
    });
    let output = Command::new(bin())
        .args(["--ollama-url", server.url(), "list-models"])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("MODEL"));
    assert!(stdout.contains("qwen2.5-coder:3b"));
}

#[test]
fn doctor_subcommand_prints_report() {
    let server = MockServer::start(|path| {
        assert_eq!(path, "/api/tags");
        (
            200,
            r#"{"models":[{"name":"qwen2.5-coder:3b","size":123,"details":{"parameter_size":"3B","quantization_level":"Q4_0","context_length":8192}}]}"#
                .to_string(),
        )
    });
    let output = Command::new(bin())
        .args(["--ollama-url", server.url(), "doctor"])
        .output()
        .unwrap();
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Ollama reachable"));
    assert!(stdout.contains("Installed models: 1"));
}

#[test]
fn pull_subcommand_reports_progress_and_success() {
    let server = MockServer::start(|path| match path {
        "/api/tags" => (200, r#"{"models":[]}"#.to_string()),
        "/api/pull" => (
            200,
            "{\"status\":\"pulling manifest\"}\n{\"status\":\"success\"}\n".to_string(),
        ),
        _ => (404, "no route".to_string()),
    });
    let output = Command::new(bin())
        .args(["--ollama-url", server.url(), "pull", "qwen2.5-coder:3b"])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("pulling manifest"));
    assert!(stdout.contains("Model 'qwen2.5-coder:3b' is ready"));
}
