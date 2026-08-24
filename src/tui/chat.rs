//! # Chat text rendering: syntax highlighting and a cheap render cache
//!
//! The full chat log is re-highlighted by [syntect] on every TUI frame when the
//! text changes. On idle frames (spinner ticks) the log is unchanged, so we
//! cache the last rendered `Vec<Line>` and only rebuild when the underlying
//! text actually differs.
//!
//! Code fences are tracked with an explicit [`CodeFenceState`] enum instead of
//! a pair of booleans/options; the highlighter lives inside the enum variant so
//! multi-line constructs (comments, strings, doc blocks) keep their state
//! across lines.
//!
//! ## References
//!
//! - Syntect: <https://github.com/trishume/syntect>
//! - Ratatui `Line`/`Span`: <https://docs.rs/ratatui/latest/ratatui/text/index.html>
//! - CommonMark fenced code blocks: <https://spec.commonmark.org/0.31.2/#fenced-code-blocks>

use ratatui::style::{Color, Style};
use ratatui::text::{Line, Span};
use std::sync::OnceLock;
use syntect::easy::HighlightLines;
use syntect::highlighting::{Style as SynStyle, Theme, ThemeSet};
use syntect::parsing::SyntaxSet;
use syntect::util::LinesWithEndings;

/// State of the markdown code-fence scanner.
///
/// The highlighter is boxed because `HighlightLines` is a large type; boxing
/// keeps the enum small and satisfies `clippy::large_enum_variant`.
enum CodeFenceState<'a> {
    /// Outside a fenced code block; lines are rendered as plain text.
    Plain,
    /// Inside a fenced code block with an active syntect highlighter.
    InCode(Box<HighlightLines<'a>>),
}

struct Highlighter {
    syntaxes: SyntaxSet,
    theme: Theme,
}

fn highlighter() -> &'static Highlighter {
    static HIGHLIGHTER: OnceLock<Highlighter> = OnceLock::new();
    HIGHLIGHTER.get_or_init(|| {
        let mut theme_set = ThemeSet::load_defaults();
        Highlighter {
            syntaxes: SyntaxSet::load_defaults_newlines(),
            theme: theme_set
                .themes
                .remove("base16-ocean.dark")
                .unwrap_or_default(),
        }
    })
}

fn syn_style_to_ratatui(style: SynStyle) -> Style {
    let fg = style.foreground;
    Style::default().fg(Color::Rgb(fg.r, fg.g, fg.b))
}

/// Resolve the syntax definition for a code-fence language hint.
fn code_syntax<'a>(h: &'a Highlighter, lang: &str) -> &'a syntect::parsing::SyntaxReference {
    h.syntaxes
        .find_syntax_by_extension(lang)
        .or_else(|| h.syntaxes.find_syntax_by_name(lang))
        .or_else(|| h.syntaxes.find_syntax_by_token(lang))
        .unwrap_or_else(|| h.syntaxes.find_syntax_plain_text())
}

/// Above this size, skip syntect highlighting entirely and render plain lines.
/// This keeps long streaming output usable on Jetson-class devices.
const FAST_PATH_MAX_CHARS: usize = 50_000;
const FAST_PATH_MAX_LINES: usize = 2_000;

/// Render chat text with syntax highlighting inside ``` code fences.
fn highlighted_chat_lines(text: &str) -> Vec<Line<'static>> {
    if text.len() > FAST_PATH_MAX_CHARS || text.lines().count() > FAST_PATH_MAX_LINES {
        return text
            .lines()
            .map(|line| Line::from(Span::raw(line.to_string())))
            .collect();
    }

    let h = highlighter();
    let mut out = Vec::new();
    let mut fence_state = CodeFenceState::Plain;

    for raw in text.lines() {
        let trimmed = raw.trim_start();
        if trimmed.starts_with("```") {
            fence_state = toggle_fence(fence_state, trimmed, h);
            out.push(Line::from(Span::styled(
                raw.to_string(),
                Style::default().fg(Color::DarkGray),
            )));
            continue;
        }
        out.push(render_chat_line(raw, &mut fence_state, h));
    }
    out
}

/// Toggle between plain text and an active code fence.
fn toggle_fence<'a>(
    state: CodeFenceState<'a>,
    trimmed: &str,
    h: &'a Highlighter,
) -> CodeFenceState<'a> {
    match state {
        CodeFenceState::Plain => {
            let lang = trimmed.trim_start_matches("```").trim().to_string();
            let highlighter = HighlightLines::new(code_syntax(h, &lang), &h.theme);
            CodeFenceState::InCode(Box::new(highlighter))
        }
        CodeFenceState::InCode(_) => CodeFenceState::Plain,
    }
}

/// Render one chat line, highlighting it when inside a code fence.
fn render_chat_line<'a>(
    raw: &str,
    fence_state: &mut CodeFenceState<'a>,
    h: &'a Highlighter,
) -> Line<'static> {
    match fence_state {
        CodeFenceState::Plain => Line::from(Span::raw(raw.to_string())),
        CodeFenceState::InCode(highlighter) => {
            let mut spans: Vec<Span<'static>> = Vec::new();
            for line in LinesWithEndings::from(raw) {
                match highlighter.highlight_line(line, &h.syntaxes) {
                    Ok(regions) => spans.extend(regions.into_iter().map(|(style, segment)| {
                        Span::styled(segment.to_string(), syn_style_to_ratatui(style))
                    })),
                    Err(_) => spans.push(Span::raw(raw.to_string())),
                }
            }
            if spans.is_empty() {
                spans.push(Span::raw(raw.to_string()));
            }
            Line::from(spans)
        }
    }
}

/// Cache of the last highlighted chat log.
pub(super) struct ChatRenderCache {
    last_text: String,
    lines: Option<Vec<Line<'static>>>,
}

impl ChatRenderCache {
    pub(super) fn new() -> Self {
        Self {
            last_text: String::new(),
            lines: None,
        }
    }

    /// Return highlighted lines for `text`, rebuilding only when it changed.
    pub(super) fn lines(&mut self, text: &str) -> &[Line<'static>] {
        if self.lines.is_none() || self.last_text != text {
            self.last_text = text.to_string();
            self.lines = Some(highlighted_chat_lines(text));
        }
        self.lines.as_deref().unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn highlighted_chat_lines_preserves_multiline_code_blocks() {
        let input = "```rust\n// comment\nfn main() {\n    println!(\"hi\");\n}\n```";
        let lines = highlighted_chat_lines(input);
        assert_eq!(lines.len(), input.lines().count());
        let joined: String = lines
            .iter()
            .map(|line| {
                line.spans
                    .iter()
                    .map(|span| span.content.as_ref())
                    .collect::<String>()
            })
            .collect::<Vec<_>>()
            .join("\n");
        assert!(joined.contains("fn main()"));
        assert!(joined.contains("println!"));
        assert!(joined.contains("// comment"));
    }

    #[test]
    fn cache_reuses_lines_when_text_is_unchanged() {
        let mut cache = ChatRenderCache::new();
        let first = cache.lines("hello");
        let first_ptr = first.as_ptr();
        let second = cache.lines("hello");
        assert_eq!(
            first_ptr,
            second.as_ptr(),
            "unchanged text must reuse cache"
        );
        let changed = cache.lines("hello\nworld");
        assert_ne!(first_ptr, changed.as_ptr(), "changed text must rebuild");
    }
}
