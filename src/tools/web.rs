//! Compact web research tools backed by a SearXNG instance.
//!
//! The instance's base URL comes from the agent configuration (`--searxng-url`
//! or `OPENBATRANGS_SEARXNG_URL`), because SearXNG often runs on a different
//! machine than openBatarangs (e.g. next to a remote model server).
use anyhow::{bail, Context, Result};
use futures_util::StreamExt;
use serde_json::Value;
use std::net::IpAddr;
use std::time::Duration;

/// The most of a page that is read; a longer page is cut here, not refused.
const MAX_PAGE_BYTES: usize = 1024 * 1024;

/// Hard cap on the cleaned text handed back to the model.
const MAX_PAGE_CHARS: usize = 6_000;

/// Search the SearXNG instance at `base_url` and return cited results. `key`
/// is sent as a bearer token when the search sits behind the Thor's agent.
pub(crate) async fn web_search(base_url: &str, key: Option<&str>, query: &str) -> Result<String> {
    let client = http_client()?;
    let request = client
        .get(format!("{base_url}/search"))
        .query(&[("format", "json"), ("q", query)]);
    let request = match key {
        Some(key) => request.bearer_auth(key),
        None => request,
    };
    let response = request
        .send()
        .await
        .context("request to local SearXNG failed")?
        .error_for_status()
        .context("SearXNG returned an error")?;
    let body: Value = response
        .json()
        .await
        .context("SearXNG did not return JSON; check that JSON output is enabled")?;
    let results = body
        .get("results")
        .and_then(Value::as_array)
        .context("SearXNG response has no results array")?;
    let rendered = results
        .iter()
        .take(6)
        .enumerate()
        .map(|(index, result)| {
            let title = result
                .get("title")
                .and_then(Value::as_str)
                .unwrap_or("(untitled)");
            let url = result.get("url").and_then(Value::as_str).unwrap_or("");
            let snippet = result.get("content").and_then(Value::as_str).unwrap_or("");
            format!(
                "{}. {}\n   {}\n   {}",
                index + 1,
                title,
                url,
                snippet.chars().take(500).collect::<String>()
            )
        })
        .collect::<Vec<_>>()
        .join("\n");
    Ok(if rendered.is_empty() {
        "No search results returned.".to_string()
    } else {
        rendered
    })
}

/// Read a public web page, usually one returned by search.
pub(crate) async fn web_read(url: &str) -> Result<String> {
    let parsed = raw_source(reqwest::Url::parse(url).context("invalid URL")?);
    if !is_public_web_url(&parsed) {
        bail!("web_read reads public http and https pages only, not addresses on this machine or its network");
    }

    let response = http_client()?
        .get(parsed)
        .send()
        .await
        .context("page request failed")?
        .error_for_status()
        .context("page returned an error")?;
    let is_html = response
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .is_some_and(|kind| kind.contains("html"));
    let mut stream = response.bytes_stream();
    let mut body = Vec::new();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.context("failed while reading documentation page")?;
        let room = MAX_PAGE_BYTES - body.len();
        body.extend_from_slice(&chunk[..chunk.len().min(room)]);
        if body.len() == MAX_PAGE_BYTES {
            break;
        }
    }
    let page = String::from_utf8_lossy(&body);
    if !is_html {
        return Ok(page.chars().take(MAX_PAGE_CHARS).collect());
    }
    let html = main_part(&page);
    let without_scripts =
        regex::Regex::new(r"(?is)<script\b[^>]*>.*?</script>")?.replace_all(&html, " ");
    let without_code =
        regex::Regex::new(r"(?is)<style\b[^>]*>.*?</style>")?.replace_all(&without_scripts, " ");
    let text = regex::Regex::new(r"(?s)<[^>]*>")?.replace_all(&without_code, " ");
    let cleaned = text.split_whitespace().collect::<Vec<_>>().join(" ");
    Ok(cleaned.chars().take(MAX_PAGE_CHARS).collect())
}

/// A GitHub file page as its raw source on raw.githubusercontent.com, so the
/// read is the code instead of the page around it; any other URL unchanged.
fn raw_source(url: reqwest::Url) -> reqwest::Url {
    let is_blob = url.host_str() == Some("github.com")
        && url
            .path_segments()
            .is_some_and(|mut parts| parts.nth(2) == Some("blob"));
    if !is_blob {
        return url;
    }
    let path = url.path().replacen("/blob/", "/", 1);
    reqwest::Url::parse(&format!("https://raw.githubusercontent.com{path}")).unwrap_or(url)
}

/// The page's `<main>` or `<article>` when it has one, else the whole page, so
/// navigation and footers do not crowd out the text.
fn main_part(html: &str) -> String {
    ["main", "article"]
        .iter()
        .find_map(|tag| {
            regex::Regex::new(&format!(r"(?is)<{tag}\b[^>]*>(.*)</{tag}>"))
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
    reqwest::Url::parse(url).is_ok_and(|url| approved_doc_url(&raw_source(url)))
}

/// True for HTTPS pages on official NVIDIA or GitHub documentation hosts.
fn approved_doc_url(url: &reqwest::Url) -> bool {
    let host = url.host_str().unwrap_or_default();
    url.scheme() == "https"
        && [
            "nvidia.github.io",
            "developer.nvidia.com",
            "docs.nvidia.com",
            "github.com",
            "raw.githubusercontent.com",
            "docs.rs",
            "doc.rust-lang.org",
        ]
        .iter()
        .any(|domain| host == *domain || host.ends_with(&format!(".{domain}")))
}

/// True for an http or https URL whose host is not this machine or a private
/// network: loopback, private, link-local and Tailscale (100.64.0.0/10)
/// addresses, and `localhost`, `.local` and `.internal` names, are refused, so
/// a model cannot use `web_read` to reach the model servers next to it.
fn is_public_web_url(url: &reqwest::Url) -> bool {
    let is_web = matches!(url.scheme(), "http" | "https");

    let host = url
        .host_str()
        .unwrap_or_default()
        .trim_start_matches('[')
        .trim_end_matches(']')
        .to_ascii_lowercase();

    let is_local = match host.parse::<IpAddr>() {
        Ok(IpAddr::V4(ip)) => {
            let [first, second, ..] = ip.octets();
            ip.is_loopback()
                || ip.is_private()
                || ip.is_link_local()
                || ip.is_unspecified()
                || ip.is_broadcast()
                || (first == 100 && (64..128).contains(&second))
        }
        Ok(IpAddr::V6(ip)) => {
            let first = ip.segments()[0];
            ip.is_loopback()
                || ip.is_unspecified()
                || (first & 0xfe00) == 0xfc00
                || (first & 0xffc0) == 0xfe80
        }
        Err(_) => {
            host.is_empty()
                || host == "localhost"
                || [".localhost", ".local", ".internal"]
                    .iter()
                    .any(|suffix| host.ends_with(suffix))
        }
    };

    is_web && !is_local
}

/// The most redirects one `web_read` follows.
const MAX_REDIRECTS: usize = 5;

/// HTTP client for research requests: a short timeout, and redirects followed
/// only while each hop is still a public web URL. Documentation sites often
/// redirect (`/latest/` to a version), and refusing every redirect failed
/// those reads.
fn http_client() -> Result<reqwest::Client> {
    let redirects = reqwest::redirect::Policy::custom(|attempt| {
        if attempt.previous().len() >= MAX_REDIRECTS {
            attempt.error("too many redirects")
        } else if is_public_web_url(attempt.url()) {
            attempt.follow()
        } else {
            attempt.stop()
        }
    });

    Ok(reqwest::Client::builder()
        .timeout(Duration::from_secs(20))
        .redirect(redirects)
        .build()?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn github_file_pages_are_read_as_raw_source() {
        let page = reqwest::Url::parse(
            "https://github.com/tokio-rs/tokio/blob/master/tokio/src/macros/select.rs",
        )
        .unwrap();
        assert_eq!(
            raw_source(page).as_str(),
            "https://raw.githubusercontent.com/tokio-rs/tokio/master/tokio/src/macros/select.rs"
        );

        let repo = reqwest::Url::parse("https://github.com/tokio-rs/tokio").unwrap();
        assert_eq!(
            raw_source(repo).as_str(),
            "https://github.com/tokio-rs/tokio"
        );
    }

    #[test]
    fn main_part_drops_navigation_around_the_content() {
        let html = "<nav>menu</nav><main><p>the docs</p></main><footer>links</footer>";
        assert_eq!(main_part(html), "<p>the docs</p>");
        assert_eq!(main_part("<p>no main</p>"), "<p>no main</p>");
    }

    #[test]
    fn reads_public_pages_and_refuses_local_ones() -> Result<()> {
        let public = [
            "https://docs.python.org/3/library/asyncio.html",
            "http://example.com/",
            "https://kubernetes.io/docs/home/",
        ];
        let local = [
            "http://127.0.0.1:8080/v1/models",
            "http://localhost:8000/",
            "http://192.168.0.189/",
            "http://100.101.102.103/",
            "http://thor.local/",
            "http://[::1]:8079/",
            "file:///etc/passwd",
        ];

        for url in public {
            assert!(is_public_web_url(&reqwest::Url::parse(url)?), "{url}");
        }
        for url in local {
            assert!(!is_public_web_url(&reqwest::Url::parse(url)?), "{url}");
        }

        Ok(())
    }

    #[test]
    fn only_allows_https_official_documentation_hosts() {
        assert!(approved_doc_url(
            &reqwest::Url::parse("https://nvidia.github.io/cuda-rust/").unwrap()
        ));
        assert!(approved_doc_url(
            &reqwest::Url::parse("https://docs.nvidia.com/cuda/").unwrap()
        ));
        assert!(!approved_doc_url(
            &reqwest::Url::parse("http://nvidia.github.io/").unwrap()
        ));
        assert!(!approved_doc_url(
            &reqwest::Url::parse("https://evil.example/").unwrap()
        ));
        assert!(!approved_doc_url(
            &reqwest::Url::parse("https://evilnvidia.github.io/").unwrap()
        ));
    }
}
