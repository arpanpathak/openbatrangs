//! Compact web research tools backed by a SearXNG instance.
//!
//! The instance's base URL comes from the agent configuration (`--searxng-url`
//! or `OPENBATRANGS_SEARXNG_URL`), because SearXNG often runs on a different
//! machine than openBatarangs (e.g. next to a remote model server).
use anyhow::{Context, Result, bail};
use regex::Regex;
use reqwest::dns::{Addrs, Name, Resolve, Resolving};
use reqwest::redirect::{Action, Attempt, Policy};
use reqwest::{Client, ClientBuilder, RequestBuilder, Response, Url};
use serde_json::Value;
use std::error::Error;
use std::io::ErrorKind;
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};
use std::sync::Arc;
use std::time::Duration;

/// The most of a page that is read; a longer page is cut here, not refused.
const MAX_PAGE_BYTES: usize = 1024 * 1024;

/// Hard cap on the cleaned text handed back to the model.
const MAX_PAGE_CHARS: usize = 6_000;

/// Number of search results handed back to the model.
const MAX_SEARCH_RESULTS: usize = 6;

/// Characters kept from each search result's snippet.
const MAX_SNIPPET_CHARS: usize = 500;

/// The most redirects one `web_read` follows.
const MAX_REDIRECTS: usize = 5;

/// Hosts of official NVIDIA, GitHub and Rust documentation. Their subdomains
/// count as official too.
const OFFICIAL_DOC_HOSTS: &[&str] = &[
    "nvidia.github.io",
    "developer.nvidia.com",
    "docs.nvidia.com",
    "github.com",
    "raw.githubusercontent.com",
    "docs.rs",
    "doc.rust-lang.org",
];

/// Search the SearXNG instance at `base_url` and return cited results. `key`
/// is sent as a bearer token when the search sits behind the Thor's agent.
pub(crate) async fn web_search(base_url: &str, key: Option<&str>, query: &str) -> Result<String> {
    let request = http_client(is_public_web_url)?
        .get(format!("{base_url}/search"))
        .query(&[("format", "json"), ("q", query)]);
    let body = searxng_json(with_bearer(request, key)).await?;
    let results = body
        .get("results")
        .and_then(Value::as_array)
        .context("SearXNG response has no results array")?;

    Ok(rendered_results(results))
}

/// `request` with `key` attached as a bearer token, when there is a key.
fn with_bearer(request: RequestBuilder, key: Option<&str>) -> RequestBuilder {
    match key {
        Some(key) => request.bearer_auth(key),
        None => request,
    }
}

/// Send a SearXNG search request and parse its JSON body.
async fn searxng_json(request: RequestBuilder) -> Result<Value> {
    let response = request
        .send()
        .await
        .context("request to local SearXNG failed")?
        .error_for_status()
        .context("SearXNG returned an error")?;

    response
        .json()
        .await
        .context("SearXNG did not return JSON; check that JSON output is enabled")
}

/// The first [`MAX_SEARCH_RESULTS`] results as numbered entries, or a note
/// when there are none.
fn rendered_results(results: &[Value]) -> String {
    let rendered = results
        .iter()
        .take(MAX_SEARCH_RESULTS)
        .enumerate()
        .map(|(index, result)| search_result_text(index + 1, result))
        .collect::<Vec<_>>()
        .join("\n");

    if rendered.is_empty() {
        return "No search results returned.".to_string();
    }

    rendered
}

/// One numbered search result: title, URL, and the start of its snippet.
fn search_result_text(number: usize, result: &Value) -> String {
    let title = string_field(result, "title").unwrap_or("(untitled)");
    let url = string_field(result, "url").unwrap_or_default();
    let snippet: String = string_field(result, "content")
        .unwrap_or_default()
        .chars()
        .take(MAX_SNIPPET_CHARS)
        .collect();

    format!("{number}. {title}\n   {url}\n   {snippet}")
}

/// The string at `key` in a JSON object, if there is one.
fn string_field<'a>(value: &'a Value, key: &str) -> Option<&'a str> {
    value.get(key).and_then(Value::as_str)
}

/// Read a public web page, usually one returned by search.
pub(crate) async fn web_read(url: &str) -> Result<String> {
    let page_url = public_page_url(url)?;
    let client = web_read_client()?;

    read_page(&client, page_url).await
}

/// `url` parsed and rewritten by [`raw_source`], or an error when it is not
/// a public web URL.
fn public_page_url(url: &str) -> Result<Url> {
    let parsed = Url::parse(url).context("invalid URL")?;

    refuse_local(raw_source(parsed))
}

/// `url` unchanged when it is a public web URL, else an error.
fn refuse_local(url: Url) -> Result<Url> {
    if !is_public_web_url(&url) {
        bail!(
            "web_read reads public http and https pages only, not addresses on this machine or its network"
        );
    }

    Ok(url)
}

/// Fetch `page_url` with `client` and return its readable text, at most
/// [`MAX_PAGE_CHARS`] characters.
async fn read_page(client: &Client, page_url: Url) -> Result<String> {
    let response = fetch(client, page_url).await?;
    let is_html = is_html_response(&response);
    let body = capped_body(response).await?;
    let text = readable_text(&String::from_utf8_lossy(&body), is_html)?;

    Ok(text.chars().take(MAX_PAGE_CHARS).collect())
}

/// The response for `url`, or an error when the request fails or the status
/// is an error.
async fn fetch(client: &Client, url: Url) -> Result<Response> {
    client
        .get(url)
        .send()
        .await
        .context("page request failed")?
        .error_for_status()
        .context("page returned an error")
}

/// Whether the response's `Content-Type` names HTML.
fn is_html_response(response: &Response) -> bool {
    response
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .is_some_and(|kind| kind.contains("html"))
}

/// The response body, read only up to [`MAX_PAGE_BYTES`] so a huge page is
/// not downloaded in full.
async fn capped_body(mut response: Response) -> Result<Vec<u8>> {
    let mut body = Vec::new();
    let mut next = next_chunk(&mut response, &body).await;

    while let Ok(Some(chunk)) = &next {
        append_capped(&mut body, chunk);
        next = next_chunk(&mut response, &body).await;
    }

    // The loop stops at the end of the body, at the cap, or on a read error.
    next.map(|_end_of_body| body)
}

/// The next chunk of the body; `None` at its end, or once `body` is full.
async fn next_chunk(response: &mut Response, body: &[u8]) -> Result<Option<Vec<u8>>> {
    if body.len() >= MAX_PAGE_BYTES {
        return Ok(None);
    }

    response
        .chunk()
        .await
        .map(|chunk| chunk.map(Vec::from))
        .context("failed while reading documentation page")
}

/// Append as much of `chunk` to `body` as fits under [`MAX_PAGE_BYTES`].
fn append_capped(body: &mut Vec<u8>, chunk: &[u8]) {
    let room = MAX_PAGE_BYTES - body.len();

    body.extend_from_slice(&chunk[..chunk.len().min(room)]);
}

/// The page as text: an HTML page without its markup, any other page as is.
fn readable_text(page: &str, is_html: bool) -> Result<String> {
    if !is_html {
        return Ok(page.to_string());
    }

    html_text(page)
}

/// The words of an HTML page's main part, without scripts, styles and tags,
/// separated by single spaces.
fn html_text(page: &str) -> Result<String> {
    let html = main_part(page);
    let without_scripts = Regex::new(r"(?is)<script\b[^>]*>.*?</script>")?.replace_all(&html, " ");
    let without_styles =
        Regex::new(r"(?is)<style\b[^>]*>.*?</style>")?.replace_all(&without_scripts, " ");
    let text = Regex::new(r"(?s)<[^>]*>")?.replace_all(&without_styles, " ");

    Ok(text.split_whitespace().collect::<Vec<_>>().join(" "))
}

/// A GitHub file page as its raw source on raw.githubusercontent.com, so the
/// read is the code instead of the page around it; any other URL unchanged.
fn raw_source(url: Url) -> Url {
    if !is_github_file_page(&url) {
        return url;
    }

    let path = url.path().replacen("/blob/", "/", 1);

    Url::parse(&format!("https://raw.githubusercontent.com{path}")).unwrap_or(url)
}

/// Whether `url` is a file page on github.com (`/owner/repo/blob/...`).
fn is_github_file_page(url: &Url) -> bool {
    url.host_str() == Some("github.com")
        && url
            .path_segments()
            .is_some_and(|mut parts| parts.nth(2) == Some("blob"))
}

/// The page's `<main>` or `<article>` when it has one, else the whole page, so
/// navigation and footers do not crowd out the text.
fn main_part(html: &str) -> String {
    ["main", "article"]
        .iter()
        .find_map(|tag| {
            Regex::new(&format!(r"(?is)<{tag}\b[^>]*>(.*)</{tag}>"))
                .ok()?
                .captures(html)
                .and_then(|found| found.get(1))
                .map(|part| part.as_str().to_string())
        })
        .unwrap_or_else(|| html.to_string())
}

/// True when `url` is on an official documentation host. Research prefers
/// these among search results; `web_read` reads any public page.
pub(crate) fn is_official_doc(url: &str) -> bool {
    Url::parse(url).is_ok_and(|url| approved_doc_url(&raw_source(url)))
}

/// True for HTTPS pages on official NVIDIA or GitHub documentation hosts.
fn approved_doc_url(url: &Url) -> bool {
    url.scheme() == "https" && is_official_doc_host(url.host_str().unwrap_or_default())
}

/// Whether `host` is one of [`OFFICIAL_DOC_HOSTS`] or a subdomain of one.
fn is_official_doc_host(host: &str) -> bool {
    OFFICIAL_DOC_HOSTS
        .iter()
        .any(|domain| is_same_or_subdomain(host, domain))
}

/// Whether `host` is `domain` itself or a name under it.
fn is_same_or_subdomain(host: &str, domain: &str) -> bool {
    host == domain || host.ends_with(&format!(".{domain}"))
}

/// True for an http or https URL whose host is not this machine or a private
/// network: loopback, private, link-local and Tailscale (100.64.0.0/10)
/// addresses, and `localhost`, `.local` and `.internal` names, are refused, so
/// a model cannot use `web_read` to reach the model servers next to it.
fn is_public_web_url(url: &Url) -> bool {
    let is_web = matches!(url.scheme(), "http" | "https");

    is_web && !is_local_host(&bare_host(url))
}

/// The URL's host in lower case, without the brackets around an IPv6 address.
fn bare_host(url: &Url) -> String {
    url.host_str()
        .unwrap_or_default()
        .trim_start_matches('[')
        .trim_end_matches(']')
        .to_ascii_lowercase()
}

/// Whether `host`, an IP address or a name, points at this machine or a
/// private network.
fn is_local_host(host: &str) -> bool {
    match host.parse::<IpAddr>().ok() {
        Some(address) => is_local_address(address),
        None => is_local_name(host),
    }
}

/// Whether `address` belongs to this machine or a private network.
fn is_local_address(address: IpAddr) -> bool {
    match address {
        IpAddr::V4(address) => is_local_v4(address),
        IpAddr::V6(address) => is_local_v6(address),
    }
}

/// Whether an IPv4 address is loopback, private, link-local, unspecified,
/// broadcast, or in Tailscale's 100.64.0.0/10 range.
fn is_local_v4(address: Ipv4Addr) -> bool {
    [
        address.is_loopback(),
        address.is_private(),
        address.is_link_local(),
        address.is_unspecified(),
        address.is_broadcast(),
        is_tailscale(address),
    ]
    .contains(&true)
}

/// Whether an IPv4 address is in 100.64.0.0/10, the range Tailscale uses.
fn is_tailscale(address: Ipv4Addr) -> bool {
    let [first, second, ..] = address.octets();

    first == 100 && (64..128).contains(&second)
}

/// Whether an IPv6 address is loopback, unspecified, unique-local
/// (`fc00::/7`), link-local (`fe80::/10`), or an IPv4-mapped form
/// (`::ffff:a.b.c.d`) of a local IPv4 address.
fn is_local_v6(address: Ipv6Addr) -> bool {
    let first = address.segments()[0];
    let is_mapped_local = address.to_ipv4_mapped().is_some_and(is_local_v4);

    [
        address.is_loopback(),
        address.is_unspecified(),
        (first & 0xfe00) == 0xfc00,
        (first & 0xffc0) == 0xfe80,
        is_mapped_local,
    ]
    .contains(&true)
}

/// Whether a host name is empty, `localhost`, or under `.localhost`,
/// `.local` or `.internal`.
fn is_local_name(host: &str) -> bool {
    let has_local_suffix = [".localhost", ".local", ".internal"]
        .iter()
        .any(|suffix| host.ends_with(suffix));

    [host.is_empty(), host == "localhost", has_local_suffix].contains(&true)
}

/// HTTP client for research requests, built by [`client_builder`].
fn http_client(allows: fn(&Url) -> bool) -> Result<Client> {
    Ok(client_builder(allows).build()?)
}

/// HTTP client for `web_read`: it resolves host names with
/// [`PublicOnlyResolver`], so the page and every redirect target are fetched
/// only from public addresses.
fn web_read_client() -> Result<Client> {
    Ok(client_builder(is_public_web_url)
        .dns_resolver(Arc::new(PublicOnlyResolver))
        .build()?)
}

/// Client settings for research requests: a short timeout, and redirects
/// followed only to URLs that `allows` accepts, which is [`is_public_web_url`]
/// outside tests. Documentation sites often redirect (`/latest/` to a
/// version), and refusing every redirect failed those reads.
fn client_builder(allows: fn(&Url) -> bool) -> ClientBuilder {
    let redirects = Policy::custom(move |attempt| redirect_action(attempt, allows));

    Client::builder()
        .timeout(Duration::from_secs(20))
        .redirect(redirects)
}

/// DNS resolver for `web_read`. It resolves a host name once and refuses it
/// when any address is on this machine or a private network. reqwest connects
/// only to the addresses it returns, so a second DNS answer cannot send the
/// request somewhere else (DNS rebinding).
struct PublicOnlyResolver;

impl Resolve for PublicOnlyResolver {
    fn resolve(&self, name: Name) -> Resolving {
        Box::pin(public_addresses(name.as_str().to_string()))
    }
}

/// The addresses of `host`, or an error when any of them is local.
async fn public_addresses(host: String) -> Result<Addrs, Box<dyn Error + Send + Sync>> {
    let addresses = tokio::net::lookup_host((host.as_str(), 0)).await?.collect();
    let checked = only_public(&host, addresses)?;

    Ok(Box::new(checked.into_iter()))
}

/// `addresses` unchanged when none of them is local, else an error that
/// names `host`.
fn only_public(host: &str, addresses: Vec<SocketAddr>) -> std::io::Result<Vec<SocketAddr>> {
    if addresses
        .iter()
        .any(|address| is_local_address(address.ip()))
    {
        return Err(std::io::Error::new(
            ErrorKind::PermissionDenied,
            format!("{host} resolves to a local address"),
        ));
    }

    Ok(addresses)
}

/// An error after [`MAX_REDIRECTS`] redirects, else follow the redirect
/// when `allows` accepts its target.
fn redirect_action(attempt: Attempt, allows: fn(&Url) -> bool) -> Action {
    if attempt.previous().len() >= MAX_REDIRECTS {
        return attempt.error("too many redirects");
    }

    follow_if_allowed(attempt, allows)
}

/// Follow the redirect when `allows` accepts its target, else stop and keep
/// the redirect response.
fn follow_if_allowed(attempt: Attempt, allows: fn(&Url) -> bool) -> Action {
    if allows(attempt.url()) {
        return attempt.follow();
    }

    attempt.stop()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{MockResponse, spawn_mock_server};
    use crate::tools::test_server::{cut_short, redirect, response, spawn_raw_server};

    /// A client that follows every redirect, so tests can redirect between
    /// pages of a server on this machine.
    fn local_client() -> Result<Client> {
        http_client(|_target| true)
    }

    #[test]
    fn github_file_pages_are_read_as_raw_source() -> Result<()> {
        let page =
            Url::parse("https://github.com/tokio-rs/tokio/blob/master/tokio/src/macros/select.rs")?;
        let repo = Url::parse("https://github.com/tokio-rs/tokio")?;

        assert_eq!(
            raw_source(page).as_str(),
            "https://raw.githubusercontent.com/tokio-rs/tokio/master/tokio/src/macros/select.rs"
        );
        assert_eq!(
            raw_source(repo).as_str(),
            "https://github.com/tokio-rs/tokio"
        );

        Ok(())
    }

    #[test]
    fn main_part_drops_navigation_around_the_content() {
        let html = "<nav>menu</nav><main><p>the docs</p></main><footer>links</footer>";

        assert_eq!(main_part(html), "<p>the docs</p>");
        assert_eq!(main_part("<p>no main</p>"), "<p>no main</p>");
    }

    #[test]
    fn reads_public_pages_and_refuses_local_ones() {
        let public = [
            "https://docs.python.org/3/library/asyncio.html",
            "http://example.com/",
            "https://kubernetes.io/docs/home/",
            "http://[2606:4700::1111]/",
        ];
        let local = [
            "http://127.0.0.1:8080/v1/models",
            "http://localhost:8000/",
            "http://192.168.0.189/",
            "http://100.101.102.103/",
            "http://thor.local/",
            "http://[::1]:8079/",
            "http://[fd00::1]/",
            "file:///etc/passwd",
        ];

        let public_urls = public.map(Url::parse);
        let local_urls = local.map(Url::parse);

        assert!(public_urls.iter().flatten().all(is_public_web_url));
        assert!(!local_urls.iter().flatten().any(is_public_web_url));
        assert!(public_urls.iter().chain(&local_urls).all(Result::is_ok));
    }

    #[test]
    fn ipv4_mapped_ipv6_addresses_of_local_hosts_are_refused() -> Result<()> {
        let loopback = Url::parse("http://[::ffff:127.0.0.1]:8079/v1/models")?;
        let private = Url::parse("http://[::ffff:192.168.0.189]/")?;
        let public = Url::parse("http://[::ffff:93.184.216.34]/")?;

        assert!(!is_public_web_url(&loopback));
        assert!(!is_public_web_url(&private));
        assert!(is_public_web_url(&public));

        Ok(())
    }

    #[test]
    fn only_allows_https_official_documentation_hosts() -> Result<()> {
        assert!(approved_doc_url(&Url::parse(
            "https://nvidia.github.io/cuda-rust/"
        )?));
        assert!(approved_doc_url(&Url::parse(
            "https://docs.nvidia.com/cuda/"
        )?));
        assert!(!approved_doc_url(&Url::parse("http://nvidia.github.io/")?));
        assert!(!approved_doc_url(&Url::parse("https://evil.example/")?));
        assert!(!approved_doc_url(&Url::parse(
            "https://evilnvidia.github.io/"
        )?));

        Ok(())
    }

    #[test]
    fn official_docs_include_github_files_read_as_raw_source() {
        assert!(is_official_doc(
            "https://github.com/rust-lang/rust/blob/master/README.md"
        ));
        assert!(!is_official_doc("not a url"));
    }

    #[tokio::test]
    async fn web_search_numbers_results_and_fills_missing_fields() -> Result<()> {
        let base_url = spawn_mock_server(|_path| {
            MockResponse::json(
                "200 OK",
                r#"{"results":[
                    {"title":"Tokio","url":"https://tokio.rs","content":"async runtime"},
                    {"url":"https://docs.rs/tokio"}
                ]}"#,
            )
        })
        .await;

        let found = web_search(&base_url, Some("key"), "tokio").await?;

        assert_eq!(
            found,
            "1. Tokio\n   https://tokio.rs\n   async runtime\n\
             2. (untitled)\n   https://docs.rs/tokio\n   "
        );
        Ok(())
    }

    #[tokio::test]
    async fn web_search_says_when_nothing_was_found() -> Result<()> {
        let base_url =
            spawn_mock_server(|_path| MockResponse::json("200 OK", r#"{"results":[]}"#)).await;

        let found = web_search(&base_url, None, "nothing").await?;

        assert_eq!(found, "No search results returned.");
        Ok(())
    }

    #[tokio::test]
    async fn web_search_reports_bad_searxng_replies() -> Result<()> {
        let no_results = spawn_mock_server(|_path| MockResponse::json("200 OK", "{}")).await;
        let not_json = spawn_mock_server(|_path| MockResponse::text("200 OK", "<html>")).await;
        let failing =
            spawn_mock_server(|_path| MockResponse::text("500 Internal Server Error", "")).await;

        let no_results = web_search(&no_results, None, "q").await;
        let not_json = web_search(&not_json, None, "q").await;
        let failing = web_search(&failing, None, "q").await;
        let unreachable = web_search("http://127.0.0.1:1", None, "q").await;

        assert!(format!("{no_results:?}").contains("no results array"));
        assert!(format!("{not_json:?}").contains("did not return JSON"));
        assert!(format!("{failing:?}").contains("SearXNG returned an error"));
        assert!(format!("{unreachable:?}").contains("request to local SearXNG failed"));
        Ok(())
    }

    #[tokio::test]
    async fn web_read_refuses_local_and_invalid_urls() {
        let local = web_read("http://127.0.0.1:8079/v1/models").await;
        let invalid = web_read("not a url").await;

        assert!(format!("{local:?}").contains("public http and https pages only"));
        assert!(format!("{invalid:?}").contains("invalid URL"));
    }

    #[tokio::test]
    async fn web_read_requests_a_public_url() {
        // `.invalid` names never resolve, so the request fails without network.
        let unresolvable = web_read("http://openbatrangs-test.invalid/").await;

        assert!(format!("{unresolvable:?}").contains("page request failed"));
    }

    #[tokio::test]
    async fn read_page_keeps_the_text_of_the_main_part() -> Result<()> {
        let base_url = spawn_raw_server(|_path| {
            response(
                "200 OK",
                "text/html; charset=utf-8",
                "<nav>menu</nav><main><script>let x;</script><style>p{}</style>\
                 <h1>Title</h1>\n<p>Body   text</p></main>",
            )
        })
        .await?;

        let text = read_page(&local_client()?, Url::parse(&base_url)?).await?;

        assert_eq!(text, "Title Body text");
        Ok(())
    }

    #[tokio::test]
    async fn read_page_returns_other_pages_as_is_up_to_the_cap() -> Result<()> {
        let base_url = spawn_raw_server(|path| match path {
            "/short" => response("200 OK", "text/plain", "plain <b>text</b>"),
            _ => response("200 OK", "text/plain", &"x".repeat(MAX_PAGE_BYTES + 10)),
        })
        .await?;

        let short = read_page(&local_client()?, Url::parse(&format!("{base_url}/short"))?).await?;
        let long = read_page(&local_client()?, Url::parse(&format!("{base_url}/long"))?).await?;

        assert_eq!(short, "plain <b>text</b>");
        assert_eq!(long.len(), MAX_PAGE_CHARS);
        Ok(())
    }

    #[tokio::test]
    async fn read_page_reports_error_statuses_and_broken_bodies() -> Result<()> {
        let base_url = spawn_raw_server(|path| match path {
            "/missing" => response("404 Not Found", "text/plain", "gone"),
            _ => cut_short("partial"),
        })
        .await?;

        let missing = read_page(
            &local_client()?,
            Url::parse(&format!("{base_url}/missing"))?,
        )
        .await;
        let broken = read_page(&local_client()?, Url::parse(&format!("{base_url}/broken"))?).await;

        assert!(format!("{missing:?}").contains("page returned an error"));
        assert!(format!("{broken:?}").contains("failed while reading documentation page"));
        Ok(())
    }

    #[tokio::test]
    async fn redirects_are_followed_only_to_allowed_targets() -> Result<()> {
        let base_url = spawn_raw_server(|path| match path {
            "/start" => redirect("/final"),
            _ => response("200 OK", "text/plain", "arrived"),
        })
        .await?;
        let start = Url::parse(&format!("{base_url}/start"))?;

        let followed = read_page(&local_client()?, start.clone()).await?;
        let stopped = read_page(&http_client(is_public_web_url)?, start).await?;

        assert_eq!(followed, "arrived");
        assert_eq!(stopped, "", "a redirect to this machine is not followed");
        Ok(())
    }

    #[tokio::test]
    async fn a_name_that_resolves_to_this_machine_is_refused() -> Result<()> {
        let base_url =
            spawn_raw_server(|_path| response("200 OK", "text/plain", "private")).await?;
        let mut page = Url::parse(&base_url)?;
        page.set_host(Some("localhost"))?;

        let outcome = read_page(&web_read_client()?, page).await;

        assert!(
            format!("{outcome:?}").contains("localhost resolves to a local address"),
            "{outcome:?}"
        );
        Ok(())
    }

    #[tokio::test]
    async fn a_public_host_resolves_to_its_addresses() -> Result<()> {
        // An address literal resolves to itself without a DNS server.
        let resolved = public_addresses("93.184.216.34".to_string()).await;

        let addresses: Vec<SocketAddr> = resolved.map_err(anyhow::Error::from_boxed)?.collect();

        assert_eq!(
            addresses,
            vec![on_port_zero(Ipv4Addr::new(93, 184, 216, 34))]
        );
        Ok(())
    }

    /// `address` on port 0, the port the resolver asks for.
    fn on_port_zero(address: impl Into<IpAddr>) -> SocketAddr {
        SocketAddr::new(address.into(), 0)
    }

    #[test]
    fn resolved_addresses_are_kept_only_when_all_are_public() -> Result<()> {
        let public_v4 = on_port_zero(Ipv4Addr::new(93, 184, 216, 34));
        let public_v6 = on_port_zero(Ipv6Addr::new(0x2606, 0x4700, 0, 0, 0, 0, 0, 0x1111));
        let local_sets = [
            vec![public_v4, on_port_zero(Ipv4Addr::new(10, 0, 0, 5))],
            vec![on_port_zero(Ipv4Addr::UNSPECIFIED)],
            vec![on_port_zero(Ipv6Addr::LOCALHOST)],
            vec![on_port_zero(Ipv6Addr::UNSPECIFIED)],
            vec![on_port_zero(Ipv6Addr::new(0xfd12, 0, 0, 0, 0, 0, 0, 1))],
            vec![on_port_zero(Ipv6Addr::new(0xfe80, 0, 0, 0, 0, 0, 0, 1))],
            vec![on_port_zero(
                Ipv4Addr::new(169, 254, 169, 254).to_ipv6_mapped(),
            )],
        ];

        let kept = only_public("example.com", vec![public_v4, public_v6])?;

        assert_eq!(kept, vec![public_v4, public_v6]);
        assert!(
            local_sets
                .into_iter()
                .all(|addresses| only_public("example.com", addresses).is_err())
        );
        Ok(())
    }

    #[tokio::test]
    async fn a_redirect_chain_stops_after_the_limit() -> Result<()> {
        let base_url = spawn_raw_server(|path| redirect(&format!("{path}x"))).await?;

        let endless = read_page(&local_client()?, Url::parse(&format!("{base_url}/hop"))?).await;

        assert!(format!("{endless:?}").contains("too many redirects"));
        Ok(())
    }
}
