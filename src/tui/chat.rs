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

use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use std::sync::OnceLock;
use syntect::easy::HighlightLines;
use syntect::highlighting::{Style as SynStyle, Theme, ThemeSet};
use syntect::parsing::SyntaxSet;

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

/// Lazily-initialized syntect highlighter shared across all chat rendering calls.
struct Highlighter {
    /// Loaded syntax definitions used to resolve language hints to grammars.
    syntaxes: SyntaxSet,
    /// Color theme applied during syntax highlighting (base16-ocean.dark).
    theme: Theme,
}

/// Return the process-wide highlighter singleton, initializing it on first call.
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

/// Convert a syntect `Style` (foreground color) to a ratatui `Style`.
fn syn_style_to_ratatui(style: SynStyle) -> Style {
    let fg = style.foreground;
    Style::default().fg(Color::Rgb(fg.r, fg.g, fg.b))
}

/// Resolve the syntax definition for a code-fence language hint.
fn code_syntax<'a>(
    syntax_highlighter: &'a Highlighter,
    lang: &str,
) -> &'a syntect::parsing::SyntaxReference {
    syntax_highlighter
        .syntaxes
        .find_syntax_by_extension(lang)
        .or_else(|| syntax_highlighter.syntaxes.find_syntax_by_name(lang))
        .or_else(|| syntax_highlighter.syntaxes.find_syntax_by_token(lang))
        .unwrap_or_else(|| syntax_highlighter.syntaxes.find_syntax_plain_text())
}

/// Maximum character count before syntect highlighting is skipped entirely.
/// This keeps long streaming output responsive on Jetson-class devices.
const FAST_PATH_MAX_CHARS: usize = 50_000;
/// Maximum line count before syntect highlighting is skipped entirely.
/// Prevents the highlighter from stalling on very long outputs.
const FAST_PATH_MAX_LINES: usize = 2_000;

/// Style of a plain chat line by its leading marker, so agent activity reads
/// as a transcript: tool calls, thinking, and results each get their own look.
/// The first matching marker wins; a line with no marker uses the default style.
const MARKER_STYLES: &[(&str, Style)] = &[
    (
        "🔧",
        Style::new().fg(Color::Cyan).add_modifier(Modifier::BOLD),
    ),
    (
        "💭",
        Style::new()
            .fg(Color::DarkGray)
            .add_modifier(Modifier::ITALIC),
    ),
    ("✅", Style::new().fg(Color::Green)),
    ("⚠️", Style::new().fg(Color::Yellow)),
    ("⛔", Style::new().fg(Color::Yellow)),
    (
        "── ",
        Style::new().fg(Color::Magenta).add_modifier(Modifier::BOLD),
    ),
    (
        "🦇",
        Style::new().fg(Color::Magenta).add_modifier(Modifier::BOLD),
    ),
    ("▶️", Style::new().fg(Color::Cyan)),
    ("You:", Style::new().add_modifier(Modifier::BOLD)),
];

/// Render chat text with syntax highlighting inside ``` code fences.
fn highlighted_chat_lines(text: &str) -> Vec<Line<'static>> {
    if is_too_long_to_highlight(text) {
        return text.lines().map(plain_chat_line).collect();
    }

    let highlighter = highlighter();
    let mut fence_state = CodeFenceState::Plain;

    text.lines()
        .map(|raw| chat_line(raw, &mut fence_state, highlighter))
        .collect()
}

/// Whether `text` is long enough that highlighting it would stall rendering.
fn is_too_long_to_highlight(text: &str) -> bool {
    text.len() > FAST_PATH_MAX_CHARS || text.lines().count() > FAST_PATH_MAX_LINES
}

/// Render one chat line. A fence line (```) toggles the fence state and is
/// drawn dim; other lines are highlighted when inside a fence.
fn chat_line<'a>(
    raw: &str,
    fence_state: &mut CodeFenceState<'a>,
    syntax_highlighter: &'a Highlighter,
) -> Line<'static> {
    let trimmed = raw.trim_start();

    if !trimmed.starts_with("```") {
        return render_chat_line(raw, fence_state, syntax_highlighter);
    }

    let previous = std::mem::replace(fence_state, CodeFenceState::Plain);
    *fence_state = toggle_fence(previous, trimmed, syntax_highlighter);

    Line::from(Span::styled(
        raw.to_string(),
        Style::default().fg(Color::DarkGray),
    ))
}

/// Toggle between plain text and an active code fence.
fn toggle_fence<'a>(
    state: CodeFenceState<'a>,
    trimmed: &str,
    syntax_highlighter: &'a Highlighter,
) -> CodeFenceState<'a> {
    match state {
        CodeFenceState::Plain => {
            let lang = trimmed.trim_start_matches("```").trim().to_string();
            let highlighter = HighlightLines::new(
                code_syntax(syntax_highlighter, &lang),
                &syntax_highlighter.theme,
            );
            CodeFenceState::InCode(Box::new(highlighter))
        }
        CodeFenceState::InCode(_) => CodeFenceState::Plain,
    }
}

/// Render one chat line, highlighting it when inside a code fence.
fn render_chat_line<'a>(
    raw: &str,
    fence_state: &mut CodeFenceState<'a>,
    syntax_highlighter: &'a Highlighter,
) -> Line<'static> {
    match fence_state {
        CodeFenceState::Plain => plain_chat_line(raw),
        CodeFenceState::InCode(highlighter) => {
            code_line(raw, highlighter, &syntax_highlighter.syntaxes)
        }
    }
}

/// Highlight one line of code. A line syntect cannot parse, or one that
/// yields no spans (an empty line), is shown as is.
fn code_line(
    raw: &str,
    highlighter: &mut HighlightLines<'_>,
    syntaxes: &SyntaxSet,
) -> Line<'static> {
    let spans = highlighter
        .highlight_line(raw, syntaxes)
        .map(|regions| styled_spans(&regions))
        .unwrap_or_default();

    if spans.is_empty() {
        return Line::raw(raw.to_string());
    }

    Line::from(spans)
}

/// Turn syntect's styled regions into ratatui spans.
fn styled_spans(regions: &[(SynStyle, &str)]) -> Vec<Span<'static>> {
    regions
        .iter()
        .map(|(style, segment)| Span::styled(segment.to_string(), syn_style_to_ratatui(*style)))
        .collect()
}

/// Style one plain chat line by its leading marker (see [`MARKER_STYLES`]).
fn plain_chat_line(raw: &str) -> Line<'static> {
    let trimmed = raw.trim_start();
    let style = MARKER_STYLES
        .iter()
        .find(|(marker, _style)| trimmed.starts_with(marker))
        .map(|(_marker, style)| *style)
        .unwrap_or_default();

    Line::from(Span::styled(raw.to_string(), style))
}

/// Cache of the last highlighted chat log.
pub(super) struct ChatRenderCache {
    /// Raw text that produced the cached `lines`, or `None` before the first render.
    last_text: Option<String>,
    /// Cached highlighted lines for `last_text`.
    lines: Vec<Line<'static>>,
}

impl ChatRenderCache {
    /// Create an empty cache that will build on the first `lines` call.
    pub(super) fn new() -> Self {
        Self {
            last_text: None,
            lines: Vec::new(),
        }
    }

    /// Return highlighted lines for `text`, rebuilding only when it changed.
    pub(super) fn lines(&mut self, text: &str) -> &[Line<'static>] {
        if self.last_text.as_deref() != Some(text) {
            self.last_text = Some(text.to_string());
            self.lines = highlighted_chat_lines(text);
        }

        &self.lines
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

    /// The text of each span in `line`, joined.
    fn line_text(line: &Line<'_>) -> String {
        line.spans
            .iter()
            .map(|span| span.content.as_ref())
            .collect()
    }

    #[test]
    fn plain_lines_are_styled_by_their_marker() {
        let tool = plain_chat_line("  🔧 write file");
        let warning = plain_chat_line("⛔ Cancelled.");
        let plain = plain_chat_line("just text");

        assert_eq!(tool.spans[0].style.fg, Some(Color::Cyan));
        assert_eq!(warning.spans[0].style.fg, Some(Color::Yellow));
        assert_eq!(plain.spans[0].style, Style::default());
    }

    #[test]
    fn very_long_text_skips_highlighting() {
        let text = "```rust\nfn main() {}\n```\n".repeat(1_000);

        let lines = highlighted_chat_lines(&text);

        assert_eq!(lines.len(), 3_000);
        assert_eq!(line_text(&lines[1]), "fn main() {}");
        assert_eq!(lines[1].spans.len(), 1);
    }

    #[test]
    fn an_empty_code_line_stays_one_empty_line() {
        let lines = highlighted_chat_lines("```rust\n\n```");

        assert_eq!(lines.len(), 3);
        assert_eq!(line_text(&lines[1]), "");
    }

    #[test]
    fn a_closed_fence_returns_to_plain_lines() {
        let lines = highlighted_chat_lines("```\ncode\n```\n✅ done");

        assert_eq!(lines[3].spans[0].style.fg, Some(Color::Green));
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
