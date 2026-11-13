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

use anyhow::Result;
use std::io::{Read, Write};
use std::net::{Shutdown, TcpListener, TcpStream};
use std::process::{Command, Output, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;

/// One installed coding model, as `/api/tags` lists it.
const ONE_MODEL_BODY: &str = r#"{"models":[{"name":"qwen2.5-coder:3b","size":123,"details":{"parameter_size":"3B","quantization_level":"Q4_0","context_length":8192}}]}"#;

/// An address where nothing listens, so every request is refused at once.
const CLOSED_URL: &str = "http://127.0.0.1:1";

/// Size of the buffer that holds one request; every test request fits in it.
const REQUEST_BUFFER_BYTES: usize = 8192;

/// The status code and body a mock server sends for a request path.
type Route = fn(&str) -> (u16, String);

/// Tiny blocking HTTP server used to stand in for Ollama.
struct MockServer {
    /// Base URL of the server, e.g. `http://127.0.0.1:54321`.
    addr: String,
}

impl MockServer {
    /// Listen on a free local port and answer each request with `route`.
    fn start(route: Route) -> Result<Self> {
        let listener = TcpListener::bind("127.0.0.1:0")?;
        let addr = format!("http://{}", listener.local_addr()?);

        thread::spawn(move || serve(&listener, route));

        Ok(Self { addr })
    }

    /// Base URL of the server.
    fn url(&self) -> &str {
        &self.addr
    }
}

/// Answer connections until the listener fails.
fn serve(listener: &TcpListener, route: Route) {
    for stream in listener.incoming().map_while(Result::ok) {
        answer(stream, route);
    }
}

/// Read one request and write the routed response; errors drop the connection.
fn answer(mut stream: TcpStream, route: Route) {
    let mut buffer = [0u8; REQUEST_BUFFER_BYTES];
    let read = stream.read(&mut buffer).unwrap_or(0);
    let request = String::from_utf8_lossy(&buffer[..read]);
    let (status, body) = route(request_path(&request));
    let response = format!(
        "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );

    // A client that hung up early does not need its response.
    stream.write_all(response.as_bytes()).ok();
    stream.shutdown(Shutdown::Write).ok();
}

/// The path in the request line, e.g. `/api/tags`, or `/` when there is none.
fn request_path(request: &str) -> &str {
    request
        .lines()
        .next()
        .and_then(|line| line.split_whitespace().nth(1))
        .unwrap_or("/")
}

/// Ollama with one installed model: tags and show succeed, chat fails.
fn ollama_route(path: &str) -> (u16, String) {
    match path {
        "/api/tags" => (200, ONE_MODEL_BODY.to_string()),
        "/api/show" => (200, r#"{"details":{"context_length":8192}}"#.to_string()),
        "/api/pull" => (
            200,
            "{\"status\":\"pulling manifest\"}\n{\"status\":\"success\"}\n".to_string(),
        ),
        _ => (500, "chat is down".to_string()),
    }
}

/// An OpenAI-compatible server that lists one model.
fn openai_route(_path: &str) -> (u16, String) {
    (200, r#"{"data":[{"id":"gemma"}]}"#.to_string())
}

/// The first chat reply of [`tool_call_route`]: one `run_command` call, as an
/// OpenAI server streams it.
const RUN_COMMAND_STREAM: &str = concat!(
    "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",",
    "\"type\":\"function\",\"function\":{\"name\":\"run_command\",",
    "\"arguments\":\"{\\\"command\\\":\\\"echo confirmed-by-y\\\"}\"}}]}}]}\n\n",
    "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n",
    "data: [DONE]\n\n",
);

/// Chat requests [`tool_call_route`] has answered so far.
static TOOL_CALL_CHATS: AtomicUsize = AtomicUsize::new(0);

/// An OpenAI server whose first chat reply asks to run a command and whose
/// later chat replies fail, so the run stops after one confirmed tool call.
fn tool_call_route(path: &str) -> (u16, String) {
    match path {
        "/v1/models" => (200, r#"{"data":[{"id":"gemma"}]}"#.to_string()),
        "/v1/chat/completions" => tool_call_chat(),
        _ => (404, "not found".to_string()),
    }
}

/// The reply to one chat request of [`tool_call_route`].
fn tool_call_chat() -> (u16, String) {
    match TOOL_CALL_CHATS.fetch_add(1, Ordering::SeqCst) {
        0 => (200, RUN_COMMAND_STREAM.to_string()),
        _ => (500, "chat is down".to_string()),
    }
}

/// Path of the compiled binary under test.
fn bin() -> &'static str {
    env!("CARGO_BIN_EXE_openbatrangs")
}

/// Run the binary with `args` and collect its output.
fn run(args: &[&str]) -> Result<Output> {
    Ok(Command::new(bin()).args(args).output()?)
}

/// A new empty directory for the agent to work in, as a `--cwd` argument.
fn empty_workspace(name: &str) -> Result<String> {
    let workspace =
        std::env::temp_dir().join(format!("openbatrangs-{name}-{}", std::process::id()));

    std::fs::create_dir_all(&workspace)?;

    Ok(workspace.display().to_string())
}

/// Standard output of a finished run, as text.
fn stdout(output: &Output) -> String {
    String::from_utf8_lossy(&output.stdout).to_string()
}

/// Standard error of a finished run, as text.
fn stderr(output: &Output) -> String {
    String::from_utf8_lossy(&output.stderr).to_string()
}

#[test]
fn version_flag_prints_version() -> Result<()> {
    let output = run(&["--version"])?;

    assert!(output.status.success());
    assert!(stdout(&output).contains("openbatrangs"));
    Ok(())
}

#[test]
fn help_flag_prints_usage() -> Result<()> {
    let output = run(&["--help"])?;

    assert!(output.status.success());
    assert!(stdout(&output).contains("Usage"));
    assert!(stdout(&output).contains("doctor"));
    Ok(())
}

#[test]
fn list_models_subcommand_prints_model_table() -> Result<()> {
    let server = MockServer::start(ollama_route)?;

    let output = run(&["--ollama-url", server.url(), "list-models"])?;

    assert!(output.status.success(), "stderr: {}", stderr(&output));
    assert!(stdout(&output).contains("MODEL"));
    assert!(stdout(&output).contains("qwen2.5-coder:3b"));
    Ok(())
}

#[test]
fn doctor_subcommand_prints_report() -> Result<()> {
    let server = MockServer::start(ollama_route)?;

    let output = run(&["--ollama-url", server.url(), "doctor"])?;

    assert!(output.status.success());
    assert!(stdout(&output).contains("Ollama reachable"));
    assert!(stdout(&output).contains("Installed models: 1"));
    Ok(())
}

#[test]
fn pull_subcommand_reports_progress_and_success() -> Result<()> {
    let server = MockServer::start(ollama_route)?;

    let output = run(&["--ollama-url", server.url(), "pull", "qwen2.5-coder:3b"])?;

    assert!(output.status.success(), "stderr: {}", stderr(&output));
    assert!(stdout(&output).contains("pulling manifest"));
    assert!(stdout(&output).contains("Model 'qwen2.5-coder:3b' is ready"));
    Ok(())
}

#[test]
fn setup_subcommand_finds_the_running_server() -> Result<()> {
    let server = MockServer::start(ollama_route)?;

    let output = run(&["--ollama-url", server.url(), "setup"])?;

    assert!(output.status.success(), "stderr: {}", stderr(&output));
    assert!(stdout(&output).contains("Ollama already running"));
    assert!(stdout(&output).contains("Setup complete"));
    Ok(())
}

#[test]
fn agent_subcommand_reports_a_failed_model_call() -> Result<()> {
    let server = MockServer::start(ollama_route)?;
    let workspace_arg = empty_workspace("agent-subcommand")?;

    let output = run(&[
        "--ollama-url",
        server.url(),
        "--cwd",
        &workspace_arg,
        "agent",
        "say",
        "hello",
    ])?;

    assert!(!output.status.success());
    assert!(stdout(&output).contains("model:   qwen2.5-coder:3b"));
    assert!(stderr(&output).contains("chat is down"));
    Ok(())
}

#[test]
fn bare_task_runs_the_agent() -> Result<()> {
    let server = MockServer::start(ollama_route)?;
    let workspace_arg = empty_workspace("bare-task")?;

    let output = run(&[
        "--ollama-url",
        server.url(),
        "--cwd",
        &workspace_arg,
        "say",
        "hello",
    ])?;

    assert!(stderr(&output).contains("chat is down"));
    Ok(())
}

/// Without a terminal the TUI cannot start, so the run fails instead of
/// waiting for keys. `setsid` detaches the binary from the terminal that
/// runs the tests.
#[cfg(target_os = "linux")]
#[test]
fn no_task_without_a_terminal_fails_to_start_the_tui() -> Result<()> {
    let server = MockServer::start(ollama_route)?;

    let output = Command::new("setsid")
        .args(["--wait", bin(), "--ollama-url", server.url()])
        .output()?;

    assert!(!output.status.success());
    Ok(())
}

#[test]
fn openai_server_lists_its_models() -> Result<()> {
    let server = MockServer::start(openai_route)?;
    let openai_url = format!("{}/v1", server.url());

    let output = run(&["--openai-url", &openai_url, "list-models"])?;

    assert!(output.status.success(), "stderr: {}", stderr(&output));
    assert!(stdout(&output).contains("gemma  (served by"));
    Ok(())
}

#[test]
fn unreachable_openai_server_is_reported() -> Result<()> {
    let output = run(&["--openai-url", CLOSED_URL, "doctor"])?;

    assert!(!output.status.success());
    assert!(stderr(&output).contains("--api-key-file"));
    Ok(())
}

#[test]
fn missing_key_file_is_reported() -> Result<()> {
    let output = run(&[
        "--thor",
        "--api-key-file",
        "/nonexistent/openbatrangs/key",
        "doctor",
    ])?;

    assert!(!output.status.success());
    assert!(stderr(&output).contains("cannot read the access key"));
    Ok(())
}

/// With `--confirm`, a tool call waits for an answer on standard input, and
/// `y` lets the command run.
#[test]
fn confirm_runs_the_command_after_y() -> Result<()> {
    let server = MockServer::start(tool_call_route)?;
    let openai_url = format!("{}/v1", server.url());
    let workspace_arg = empty_workspace("confirm")?;

    let mut child = Command::new(bin())
        .args([
            "--openai-url",
            &openai_url,
            "--confirm",
            "--cwd",
            &workspace_arg,
        ])
        .args(["--max-steps", "2", "print", "a", "word"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;

    child
        .stdin
        .take()
        .ok_or_else(|| anyhow::anyhow!("no stdin pipe"))?
        .write_all(b"y\n")?;
    let output = child.wait_with_output()?;
    let printed = stdout(&output);

    assert!(printed.contains("[y/N]"), "stdout: {printed}");
    assert!(printed.contains("confirmed-by-y"), "stdout: {printed}");
    Ok(())
}

/// The TUI starts in a pseudo-terminal, quits on `/exit`, and gives the
/// terminal back. `script` from util-linux provides the pseudo-terminal.
#[cfg(target_os = "linux")]
#[test]
fn tui_starts_in_a_terminal_and_quits_on_exit() -> Result<()> {
    let server = MockServer::start(ollama_route)?;
    let tui_command = format!("{} --ollama-url {}", bin(), server.url());
    let session = format!(
        "(sleep 2; printf '/exit\\r'; sleep 2) | timeout 20 script -qec '{tui_command}' /dev/null"
    );

    let output = Command::new("sh").args(["-c", &session]).output()?;

    assert!(output.status.success(), "stderr: {}", stderr(&output));
    Ok(())
}
