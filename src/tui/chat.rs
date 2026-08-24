//! Chat text rendering: syntax highlighting and a cheap render cache.
//!
//! The full chat log is re-highlighted by syntect on every TUI frame when the
//! text changes. On idle frames (spinner ticks) the log is unchanged, so we
//! cache the last rendered `Vec<Line>` and only rebuild when the underlying
//! text actually differs.

use ratatui::style::{Color, Style};
use ratatui::text::{Line, Span};
use std::sync::OnceLock;
use syntect::easy::HighlightLines;
use syntect::highlighting::{Style as SynStyle, Theme, ThemeSet};
use syntect::parsing::SyntaxSet;
use syntect::util::LinesWithEndings;

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
    let mut in_code = false;
    // One highlighter per code block so multi-line constructs (comments,
    // strings, doc blocks) keep their highlighting state across lines.
    let mut line_highlighter: Option<HighlightLines> = None;

    for raw in text.lines() {
        let trimmed = raw.trim_start();
        if trimmed.starts_with("```") {
            if in_code {
                in_code = false;
                line_highlighter = None;
            } else {
                in_code = true;
                let lang = trimmed.trim_start_matches("```").trim().to_string();
                line_highlighter = Some(HighlightLines::new(code_syntax(h, &lang), &h.theme));
            }
            out.push(Line::from(Span::styled(
                raw.to_string(),
                Style::default().fg(Color::DarkGray),
            )));
            continue;
        }

        if in_code {
            let mut spans: Vec<Span<'static>> = Vec::new();
            if let Some(line_highlighter) = &mut line_highlighter {
                for line in LinesWithEndings::from(raw) {
                    match line_highlighter.highlight_line(line, &h.syntaxes) {
                        Ok(regions) => {
                            for (style, segment) in regions {
                                spans.push(Span::styled(
                                    segment.to_string(),
                                    syn_style_to_ratatui(style),
                                ));
                            }
                        }
                        Err(_) => spans.push(Span::raw(raw.to_string())),
                    }
                }
            }
            if spans.is_empty() {
                spans.push(Span::raw(raw.to_string()));
            }
            out.push(Line::from(spans));
        } else {
            out.push(Line::from(Span::raw(raw.to_string())));
        }
    }
    out
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
