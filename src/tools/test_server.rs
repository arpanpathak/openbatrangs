//! A local HTTP server for tests that need control of the whole response:
//! any status line, headers such as `Location` or `Content-Type`, and bodies
//! that end early to simulate a dropped connection.
//!
//! `crate::test_support::spawn_mock_server` covers JSON and plain-text
//! replies; this server covers the rest.

use anyhow::Result;
use std::net::SocketAddr;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

/// Serve on a free local port until the test's runtime shuts down.
///
/// `respond` maps the request path, such as `/search?q=rust`, to the whole raw
/// HTTP response. Returns the base URL, such as `http://127.0.0.1:41234`.
pub(crate) async fn spawn_raw_server(respond: fn(&str) -> String) -> Result<String> {
    let listener = TcpListener::bind("127.0.0.1:0").await?;
    let base_url = format!("http://{}", listener.local_addr()?);

    tokio::spawn(serve(listener, respond));

    Ok(base_url)
}

/// Answer every connection that arrives on `listener`, each in its own task.
async fn serve(listener: TcpListener, respond: fn(&str) -> String) {
    loop {
        let accepted = listener.accept().await;

        tokio::spawn(answer(accepted, respond));
    }
}

/// Read one request from an accepted connection and write the response for
/// its path.
async fn answer(
    accepted: std::io::Result<(TcpStream, SocketAddr)>,
    respond: fn(&str) -> String,
) -> Result<()> {
    let (mut socket, _peer) = accepted?;
    let mut request = [0u8; 8192];
    let read = socket.read(&mut request).await?;
    let request = String::from_utf8_lossy(&request[..read]);
    let path = request.split_whitespace().nth(1).unwrap_or("/");

    socket.write_all(respond(path).as_bytes()).await?;
    socket.shutdown().await?;

    Ok(())
}

/// A complete response with `status` (such as `200 OK`), a `Content-Type`
/// header and `body`.
pub(crate) fn response(status: &str, content_type: &str, body: &str) -> String {
    format!(
        "HTTP/1.1 {status}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\n\
         Connection: close\r\n\r\n{body}",
        body.len()
    )
}

/// A `200 OK` response whose `Content-Length` promises 100 more bytes than
/// `body`, so reading the body fails once the connection closes.
pub(crate) fn cut_short(body: &str) -> String {
    format!(
        "HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: {}\r\n\
         Connection: close\r\n\r\n{body}",
        body.len() + 100
    )
}

/// A `302 Found` response that redirects to `location`.
pub(crate) fn redirect(location: &str) -> String {
    format!(
        "HTTP/1.1 302 Found\r\nLocation: {location}\r\nContent-Length: 0\r\n\
         Connection: close\r\n\r\n"
    )
}
