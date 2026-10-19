//! Compact web research tools backed by a SearXNG instance.
//!
//! The instance's base URL comes from the agent configuration (`--searxng-url`
//! or `OPENBATRANGS_SEARXNG_URL`), because SearXNG often runs on a different
//! machine than openBatarangs (e.g. next to a remote model server).
use anyhow::{bail, Context, Result};
use futures_util::StreamExt;
use serde_json::Value;
use std::time::Duration;

/// Hard cap on the raw page size read from a documentation host.
const MAX_PAGE_BYTES: usize = 128 * 1024;

/// Hard cap on the cleaned text handed back to the model.
const MAX_PAGE_CHARS: usize = 6_000;

/// Search the SearXNG instance at `base_url` and return cited results.
pub(crate) async fn web_search(base_url: &str, query: &str) -> Result<String> {
    let client = http_client()?;
    let response = client
        .get(format!("{base_url}/search"))
        .query(&[("format", "json"), ("q", query)])
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

/// Read an official NVIDIA or GitHub documentation page returned by search.
pub(crate) async fn web_read(url: &str) -> Result<String> {
    let parsed = reqwest::Url::parse(url).context("invalid documentation URL")?;
    if !approved_doc_url(&parsed) {
        bail!("web_read only accepts HTTPS pages on official NVIDIA or GitHub documentation hosts");
    }
    let response = http_client()?
        .get(parsed)
        .send()
        .await
        .context("documentation page request failed")?
        .error_for_status()
        .context("documentation page returned an error")?;
    if response
        .content_length()
        .is_some_and(|length| length > MAX_PAGE_BYTES as u64)
    {
        bail!("documentation page exceeds the 128 KiB read limit");
    }
    let mut stream = response.bytes_stream();
    let mut body = Vec::new();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.context("failed while reading documentation page")?;
        if body.len() + chunk.len() > MAX_PAGE_BYTES {
            bail!("documentation page exceeds the 128 KiB read limit");
        }
        body.extend_from_slice(&chunk);
    }
    let html = String::from_utf8_lossy(&body);
    let without_scripts =
        regex::Regex::new(r"(?is)<script\b[^>]*>.*?</script>")?.replace_all(&html, " ");
    let without_code =
        regex::Regex::new(r"(?is)<style\b[^>]*>.*?</style>")?.replace_all(&without_scripts, " ");
    let text = regex::Regex::new(r"(?s)<[^>]*>")?.replace_all(&without_code, " ");
    let cleaned = text.split_whitespace().collect::<Vec<_>>().join(" ");
    Ok(cleaned.chars().take(MAX_PAGE_CHARS).collect())
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
        ]
        .iter()
        .any(|domain| host == *domain || host.ends_with(&format!(".{domain}")))
}

/// HTTP client for research requests: short timeout, no redirect following.
fn http_client() -> Result<reqwest::Client> {
    Ok(reqwest::Client::builder()
        .timeout(Duration::from_secs(20))
        .redirect(reqwest::redirect::Policy::none())
        .build()?)
}

#[cfg(test)]
mod tests {
    use super::*;

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
