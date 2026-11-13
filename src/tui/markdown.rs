//! # Markdown to terminal rows
//!
//! Answers, reasoning and tool output are Markdown. [`markdown_lines`] parses
//! the text with pulldown-cmark and draws it as styled rows that fit a given
//! width: headings, bold, italics, strikethrough, inline code, bullet and
//! numbered lists (nested too), block quotes, links (text underlined, the URL
//! dimmed after it), horizontal rules, tables, and fenced code highlighted by
//! syntect.
//!
//! Every row is already wrapped, so the number of rows is the height on
//! screen. With `highlight` off, code is left plain; the rows hold the same
//! text and wrap the same way, which makes a cheap height measurement.
//!
//! ## References
//!
//! - CommonMark: <https://spec.commonmark.org/0.31.2/>
//! - pulldown-cmark: <https://docs.rs/pulldown-cmark/>

use super::chat::code_lines;
use super::text::{spans_text, spans_width, wrap_spans};
use pulldown_cmark::{CodeBlockKind, Event, HeadingLevel, Options, Parser, Tag, TagEnd};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};

/// Draw `text` as Markdown rows at most `width` cells wide.
pub(super) fn markdown_lines(text: &str, width: usize, highlight: bool) -> Vec<Line<'static>> {
    let options = Options::ENABLE_TABLES
        | Options::ENABLE_STRIKETHROUGH
        | Options::ENABLE_TASKLISTS
        | Options::ENABLE_FOOTNOTES;
    let mut writer = MarkdownWriter::new(width, highlight);

    for event in Parser::new_ext(text, options) {
        writer.event(event);
    }

    writer.finish()
}

/// One level of an open list.
struct ListLevel {
    /// Number of the next item; `None` for a bullet list.
    next_number: Option<u64>,
    /// Width of this level's item marker, so wrapped rows line up.
    marker_width: usize,
}

/// A fenced or indented code block being collected.
#[derive(Default)]
struct CodeBuffer {
    /// Language hint after the opening fence.
    lang: String,
    /// The code so far.
    text: String,
}

/// A table being collected: rows of cells of styled spans.
#[derive(Default)]
struct TableBuffer {
    /// Finished rows; the first is the header.
    rows: Vec<Vec<Vec<Span<'static>>>>,
    /// Cells of the row being read.
    row: Vec<Vec<Span<'static>>>,
}

/// A link being read.
#[derive(Default)]
struct OpenLink {
    /// Where the link points.
    url: String,
    /// Number of spans on the current line before the link text began.
    first_span: usize,
}

/// Turns pulldown-cmark events into wrapped, styled rows.
struct MarkdownWriter {
    /// Width of a row, in cells.
    width: usize,
    /// Whether code is highlighted.
    highlight: bool,
    /// Finished rows.
    rows: Vec<Line<'static>>,
    /// Spans of the line being written.
    spans: Vec<Span<'static>>,
    /// Inline styles in effect, innermost last.
    styles: Vec<Style>,
    /// Number of open block quotes.
    quote_depth: usize,
    /// Open lists, outermost first.
    lists: Vec<ListLevel>,
    /// Marker for the first row of the current list item, until it is written.
    item_marker: Option<String>,
    /// The code block being collected.
    code: Option<CodeBuffer>,
    /// The table being collected; empty outside a table.
    table: TableBuffer,
    /// Open links, innermost last.
    links: Vec<OpenLink>,
}

impl MarkdownWriter {
    /// A writer with no rows yet.
    fn new(width: usize, highlight: bool) -> Self {
        Self {
            width: width.max(1),
            highlight,
            rows: Vec::new(),
            spans: Vec::new(),
            styles: Vec::new(),
            quote_depth: 0,
            lists: Vec::new(),
            item_marker: None,
            code: None,
            table: TableBuffer::default(),
            links: Vec::new(),
        }
    }

    /// Handle one parser event.
    fn event(&mut self, event: Event<'_>) {
        match event {
            Event::Start(tag) => self.start(tag),
            Event::End(tag) => self.end(tag),
            Event::Text(text)
            | Event::Html(text)
            | Event::InlineHtml(text)
            | Event::InlineMath(text)
            | Event::DisplayMath(text) => self.text(&text),
            Event::Code(code) => self.inline_code(&code),
            Event::FootnoteReference(name) => self.text(&format!("[{name}]")),
            Event::SoftBreak | Event::HardBreak => self.flush_line(),
            Event::Rule => self.rule(),
            Event::TaskListMarker(checked) => self.task_marker(checked),
        }
    }

    /// Handle the start of a block or an inline style.
    fn start(&mut self, tag: Tag<'_>) {
        match tag {
            Tag::Heading { level, .. } => self.styles.push(heading_style(level)),
            Tag::BlockQuote(_kind) => self.start_quote(),
            Tag::CodeBlock(kind) => self.start_code(kind),
            Tag::List(first_number) => self.start_list(first_number),
            Tag::Item => self.start_item(),
            Tag::Emphasis => self
                .styles
                .push(Style::new().add_modifier(Modifier::ITALIC)),
            Tag::Strong => self.styles.push(Style::new().add_modifier(Modifier::BOLD)),
            Tag::Strikethrough => self
                .styles
                .push(Style::new().add_modifier(Modifier::CROSSED_OUT)),
            Tag::Link { dest_url, .. } => self.start_link(dest_url.to_string()),
            Tag::Table(_alignments) => self.start_table(),
            Tag::TableHead | Tag::TableRow => self.start_row(),
            // Paragraphs, images, footnotes, HTML and other containers need no
            // setup: their text arrives as text events.
            _other => {}
        }
    }

    /// Handle the end of a block or an inline style.
    fn end(&mut self, tag: TagEnd) {
        match tag {
            TagEnd::Paragraph => self.end_block(),
            TagEnd::Heading(_level) => self.end_heading(),
            TagEnd::BlockQuote(_kind) => self.end_quote(),
            TagEnd::CodeBlock => self.end_code(),
            TagEnd::List(_ordered) => self.end_list(),
            TagEnd::Item => self.flush_line(),
            TagEnd::Emphasis | TagEnd::Strong | TagEnd::Strikethrough => self.end_style(),
            TagEnd::Link => self.end_link(),
            TagEnd::Table => self.end_table(),
            TagEnd::TableHead | TagEnd::TableRow => self.end_row(),
            TagEnd::TableCell => self.end_cell(),
            // Other containers end without extra rows.
            _other => {}
        }
    }

    /// The combined inline style in effect.
    fn style(&self) -> Style {
        self.styles
            .iter()
            .fold(Style::default(), |combined, style| combined.patch(*style))
    }

    /// Add text: into the open code block, else as styled spans, one line per
    /// newline.
    fn text(&mut self, text: &str) {
        if let Some(code) = &mut self.code {
            code.text.push_str(text);
            return;
        }

        self.styled_text(text);
    }

    /// Add text in the current style; a newline ends the line.
    fn styled_text(&mut self, text: &str) {
        let style = self.style();
        let text = text.replace('\t', "    ");

        for (index, piece) in text.split('\n').enumerate() {
            self.break_line_before(index);
            self.spans.push(Span::styled(piece.to_string(), style));
        }
    }

    /// End the line before every piece of a text but the first.
    fn break_line_before(&mut self, piece_index: usize) {
        if piece_index > 0 {
            self.flush_line();
        }
    }

    /// Add inline code.
    fn inline_code(&mut self, code: &str) {
        let style = self.style().patch(Style::new().fg(Color::Yellow));

        self.spans.push(Span::styled(code.to_string(), style));
    }

    /// Add a task-list checkbox.
    fn task_marker(&mut self, checked: bool) {
        let marker = TASK_MARKERS[usize::from(checked)];

        self.spans.push(Span::raw(marker));
    }

    /// A horizontal rule across the whole width.
    fn rule(&mut self) {
        self.flush_line();

        let rule = "─".repeat(self.width);
        self.rows.push(Line::from(Span::styled(
            rule,
            Style::new().fg(Color::DarkGray),
        )));
        self.blank_row();
    }

    /// End a paragraph: write its last line and leave a blank row.
    fn end_block(&mut self) {
        self.flush_line();
        self.blank_row();
    }

    /// End a heading: drop its style and leave a blank row.
    fn end_heading(&mut self) {
        self.flush_line();
        self.styles.pop();
        self.blank_row();
    }

    /// Drop the innermost inline style.
    fn end_style(&mut self) {
        self.styles.pop();
    }

    /// Open a block quote.
    fn start_quote(&mut self) {
        self.flush_line();
        self.quote_depth += 1;
    }

    /// Close a block quote.
    fn end_quote(&mut self) {
        self.flush_line();
        self.quote_depth = self.quote_depth.saturating_sub(1);
        self.blank_row();
    }

    /// Start collecting a code block.
    fn start_code(&mut self, kind: CodeBlockKind<'_>) {
        self.flush_line();

        self.code = Some(CodeBuffer {
            lang: code_language(&kind),
            text: String::new(),
        });
    }

    /// Write the collected code block, indented by two cells.
    fn end_code(&mut self) {
        let code = self.code.take().unwrap_or_default();

        for line in code_lines(&code.text, &code.lang, self.highlight) {
            let prefix = self.line_prefix(false);
            self.push_wrapped(prefix.clone(), prefix, indented(line));
        }

        self.blank_row();
    }

    /// Open a list.
    fn start_list(&mut self, first_number: Option<u64>) {
        self.flush_line();

        self.lists.push(ListLevel {
            next_number: first_number,
            marker_width: 2,
        });
    }

    /// Close a list; the outermost list leaves a blank row.
    fn end_list(&mut self) {
        self.flush_line();
        self.lists.pop();

        if self.lists.is_empty() {
            self.blank_row();
        }
    }

    /// Start a list item: its first row gets a bullet or a number.
    fn start_item(&mut self) {
        self.flush_line();

        self.item_marker = self.lists.last_mut().map(item_marker);
    }

    /// Open a link: its text is underlined.
    fn start_link(&mut self, url: String) {
        self.styles
            .push(Style::new().add_modifier(Modifier::UNDERLINED));
        self.links.push(OpenLink {
            url,
            first_span: self.spans.len(),
        });
    }

    /// Close a link: the URL follows its text, dimmed, unless the text is
    /// the URL itself.
    fn end_link(&mut self) {
        self.styles.pop();

        let link = self.links.pop().unwrap_or_default();
        self.add_link_url(link);
    }

    /// Add the URL after the link text, unless the text is the URL itself.
    fn add_link_url(&mut self, link: OpenLink) {
        let link_text = self.spans.get(link.first_span..).map(spans_text);
        if link_text.as_deref() == Some(link.url.as_str()) {
            return;
        }

        let url_style = Style::new().fg(Color::DarkGray);
        self.spans
            .push(Span::styled(format!(" ({})", link.url), url_style));
    }

    /// Start collecting a table.
    fn start_table(&mut self) {
        self.flush_line();
        self.table = TableBuffer::default();
    }

    /// Start a table row.
    fn start_row(&mut self) {
        self.table.row.clear();
    }

    /// Move the spans written since the cell began into the row.
    fn end_cell(&mut self) {
        let cell = std::mem::take(&mut self.spans);

        self.table.row.push(cell);
    }

    /// Finish a table row.
    fn end_row(&mut self) {
        let row = std::mem::take(&mut self.table.row);

        self.table.rows.push(row);
    }

    /// Write the collected table with box-drawing borders.
    fn end_table(&mut self) {
        let table = std::mem::take(&mut self.table);

        let widths = column_widths(&table.rows);
        for line in table_rows(&table.rows, &widths) {
            let prefix = self.line_prefix(false);
            self.push_wrapped(prefix.clone(), prefix, line);
        }

        self.blank_row();
    }

    /// Write the current line, wrapped, with the quote and list prefixes.
    fn flush_line(&mut self) {
        if self.spans.is_empty() {
            return;
        }

        let spans = std::mem::take(&mut self.spans);
        let first_prefix = self.line_prefix(true);
        let rest_prefix = self.line_prefix(false);

        self.push_wrapped(first_prefix, rest_prefix, spans);
    }

    /// The prefix of a row: quote bars, list indentation, and the item
    /// marker when `is_first_row` of an item.
    fn line_prefix(&mut self, is_first_row: bool) -> Vec<Span<'static>> {
        let quote = Span::styled(
            "│ ".repeat(self.quote_depth),
            Style::new().fg(Color::DarkGray),
        );
        let outer_indent: usize = self
            .lists
            .iter()
            .rev()
            .skip(1)
            .map(|level| level.marker_width)
            .sum();
        let marker = self.marker_for_row(is_first_row);

        vec![
            quote,
            Span::raw(" ".repeat(outer_indent)),
            Span::styled(marker, Style::new().fg(Color::Cyan)),
        ]
    }

    /// The item marker for the first row of an item, spaces of the same
    /// width for any other row inside a list.
    fn marker_for_row(&mut self, is_first_row: bool) -> String {
        let marker = self.item_marker.take().filter(|_marker| is_first_row);
        let current_width = self.lists.last().map_or(0, |level| level.marker_width);

        marker.unwrap_or_else(|| " ".repeat(current_width))
    }

    /// Wrap `spans` to the room left after the prefix and add the rows.
    fn push_wrapped(
        &mut self,
        first_prefix: Vec<Span<'static>>,
        rest_prefix: Vec<Span<'static>>,
        spans: Vec<Span<'static>>,
    ) {
        let room = self.width.saturating_sub(spans_width(&rest_prefix));

        for (index, row) in wrap_spans(spans, room).into_iter().enumerate() {
            let prefix = row_prefix(index, &first_prefix, &rest_prefix);
            self.rows.push(Line::from([prefix, row].concat()));
        }
    }

    /// Add a blank row, unless there are no rows yet or the last is blank.
    fn blank_row(&mut self) {
        let last_is_blank = self.rows.last().is_none_or(|row| row.width() == 0);

        if !last_is_blank {
            self.rows.push(Line::default());
        }
    }

    /// Every row, without trailing blank rows.
    fn finish(mut self) -> Vec<Line<'static>> {
        self.flush_line();

        while self.rows.last().is_some_and(|row| row.width() == 0) {
            self.rows.pop();
        }

        self.rows
    }
}

/// Checkbox text for an open task (`false`) and a done task (`true`).
const TASK_MARKERS: [&str; 2] = ["[ ] ", "[x] "];

/// The prefix for row `index` of a wrapped line.
fn row_prefix(
    index: usize,
    first_prefix: &[Span<'static>],
    rest_prefix: &[Span<'static>],
) -> Vec<Span<'static>> {
    match index {
        0 => first_prefix.to_vec(),
        _ => rest_prefix.to_vec(),
    }
}

/// A code line indented by two cells.
fn indented(line: Vec<Span<'static>>) -> Vec<Span<'static>> {
    [vec![Span::raw("  ")], line].concat()
}

/// The style of a heading: the top levels stand out the most.
fn heading_style(level: HeadingLevel) -> Style {
    let bold = Style::new().add_modifier(Modifier::BOLD);

    match level {
        HeadingLevel::H1 => bold.fg(Color::Magenta).add_modifier(Modifier::UNDERLINED),
        HeadingLevel::H2 => bold.fg(Color::Cyan),
        HeadingLevel::H3 | HeadingLevel::H4 | HeadingLevel::H5 | HeadingLevel::H6 => bold,
    }
}

/// The language hint of a code block; none for an indented block.
fn code_language(kind: &CodeBlockKind<'_>) -> String {
    match kind {
        CodeBlockKind::Fenced(lang) => lang.split_whitespace().next().unwrap_or("").to_string(),
        CodeBlockKind::Indented => String::new(),
    }
}

/// The marker of the next item at `level`: a bullet, or the next number.
/// The level remembers the marker's width, so wrapped rows line up.
fn item_marker(level: &mut ListLevel) -> String {
    let marker = match level.next_number {
        Some(number) => format!("{number}. "),
        None => "• ".to_string(),
    };

    level.next_number = level.next_number.map(|number| number + 1);
    level.marker_width = marker.chars().count();

    marker
}

/// The width of each table column: its widest cell.
fn column_widths(rows: &[Vec<Vec<Span<'static>>>]) -> Vec<usize> {
    let columns = rows.iter().map(Vec::len).max().unwrap_or(0);

    (0..columns)
        .map(|column| column_width(rows, column))
        .collect()
}

/// The width of the widest cell in `column`.
fn column_width(rows: &[Vec<Vec<Span<'static>>>], column: usize) -> usize {
    rows.iter()
        .filter_map(|row| row.get(column))
        .map(|cell| spans_width(cell))
        .max()
        .unwrap_or(0)
}

/// The table as lines: a top border, the header, a rule, the body rows, and
/// a bottom border.
fn table_rows(rows: &[Vec<Vec<Span<'static>>>], widths: &[usize]) -> Vec<Vec<Span<'static>>> {
    let mut lines = vec![border(widths, '┌', '┬', '┐')];

    for (index, row) in rows.iter().enumerate() {
        lines.push(table_row(row, widths));
        lines.extend(header_rule(index, widths));
    }

    lines.push(border(widths, '└', '┴', '┘'));
    lines
}

/// The rule under the header row; nothing under the other rows.
fn header_rule(row_index: usize, widths: &[usize]) -> Option<Vec<Span<'static>>> {
    (row_index == 0).then(|| border(widths, '├', '┼', '┤'))
}

/// A horizontal table border with the given corner and joint characters.
fn border(widths: &[usize], left: char, joint: char, right: char) -> Vec<Span<'static>> {
    let segments: Vec<String> = widths.iter().map(|width| "─".repeat(width + 2)).collect();
    let line = format!("{left}{}{right}", segments.join(&joint.to_string()));

    vec![Span::styled(line, Style::new().fg(Color::DarkGray))]
}

/// One table row: each cell padded to its column's width.
fn table_row(row: &[Vec<Span<'static>>], widths: &[usize]) -> Vec<Span<'static>> {
    let bar = Span::styled("│", Style::new().fg(Color::DarkGray));
    let mut spans = vec![bar.clone()];

    for (column, width) in widths.iter().enumerate() {
        let cell = row.get(column).cloned().unwrap_or_default();
        let padding = width.saturating_sub(spans_width(&cell));

        spans.push(Span::raw(" "));
        spans.extend(cell);
        spans.push(Span::raw(" ".repeat(padding + 1)));
        spans.push(bar.clone());
    }

    spans
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Context;

    /// The text of every row.
    fn texts(lines: &[Line<'static>]) -> Vec<String> {
        lines.iter().map(|line| spans_text(&line.spans)).collect()
    }

    /// The style of the first span whose text contains `needle`.
    fn style_of(lines: &[Line<'static>], needle: &str) -> Option<Style> {
        lines
            .iter()
            .flat_map(|line| line.spans.iter())
            .find(|span| span.content.contains(needle))
            .map(|span| span.style)
    }

    #[test]
    fn headings_and_inline_styles_are_styled() -> anyhow::Result<()> {
        let lines = markdown_lines(
            "# Title\n\n## Part\n\n### Small\n\n**bold** *italic* ~~gone~~ `code`",
            40,
            true,
        );

        assert_eq!(texts(&lines)[0], "Title");
        assert_eq!(texts(&lines)[1], "");
        let title = style_of(&lines, "Title").context("title")?;
        assert_eq!(title.fg, Some(Color::Magenta));
        assert_eq!(
            style_of(&lines, "Part").context("part")?.fg,
            Some(Color::Cyan)
        );
        assert!(
            style_of(&lines, "Small")
                .context("small")?
                .add_modifier
                .contains(Modifier::BOLD)
        );
        assert!(
            style_of(&lines, "bold")
                .context("bold")?
                .add_modifier
                .contains(Modifier::BOLD)
        );
        assert!(
            style_of(&lines, "italic")
                .context("it")?
                .add_modifier
                .contains(Modifier::ITALIC)
        );
        assert!(
            style_of(&lines, "gone")
                .context("gone")?
                .add_modifier
                .contains(Modifier::CROSSED_OUT)
        );
        assert_eq!(
            style_of(&lines, "code").context("code")?.fg,
            Some(Color::Yellow)
        );

        Ok(())
    }

    #[test]
    fn nested_lists_are_indented_and_numbered() {
        let lines = markdown_lines(
            "1. one\n2. two\n   - inner\n   - [x] done\n\n- last",
            40,
            true,
        );

        assert_eq!(
            texts(&lines),
            [
                "1. one",
                "2. two",
                "   • inner",
                "   • [x] done",
                "",
                "• last"
            ]
        );
    }

    #[test]
    fn wrapped_list_rows_line_up_under_the_text() {
        let lines = markdown_lines("- abcdefgh", 6, true);

        assert_eq!(texts(&lines), ["• abcd", "  efgh"]);
    }

    #[test]
    fn quotes_rules_links_and_breaks() {
        let text = "> quoted\n> more\n\n---\n\n[docs](https://a.b) and <https://c.d>\n\n<b>raw</b>\n\nx[^1]\n\n[^1]: note";
        let lines = markdown_lines(text, 40, true);
        let rows = texts(&lines);

        assert_eq!(rows[0], "│ quoted");
        assert_eq!(rows[1], "│ more");
        assert_eq!(rows[3], "─".repeat(40));
        assert!(
            rows.iter()
                .any(|row| row == "docs (https://a.b) and https://c.d")
        );
        assert!(rows.iter().any(|row| row.contains("<b>raw</b>")));
        assert!(rows.iter().any(|row| row.contains("x[1]")));
        assert!(
            style_of(&lines, "docs")
                .is_some_and(|style| style.add_modifier.contains(Modifier::UNDERLINED))
        );
    }

    #[test]
    fn fenced_code_is_highlighted_and_indented() {
        let lines = markdown_lines("```rust\nfn main() {}\n```\n\n    indented", 40, true);
        let rows = texts(&lines);

        assert_eq!(rows[0], "  fn main() {}");
        assert!(lines[0].spans.len() > 3, "the code line is highlighted");
        assert_eq!(rows[2], "  indented");
    }

    #[test]
    fn plain_and_highlighted_rows_hold_the_same_text() {
        let text = "Intro\n\n```python\ndef f(x):\n\treturn x * 2  # a long comment here\n```";

        let highlighted = markdown_lines(text, 16, true);
        let plain = markdown_lines(text, 16, false);

        assert_eq!(texts(&highlighted), texts(&plain));
    }

    #[test]
    fn tables_get_borders_and_padded_cells() {
        let lines = markdown_lines("| a | bb |\n|---|---|\n| ccc | d |", 40, true);

        assert_eq!(
            texts(&lines),
            [
                "┌─────┬────┐",
                "│ a   │ bb │",
                "├─────┼────┤",
                "│ ccc │ d  │",
                "└─────┴────┘",
            ]
        );
    }

    #[test]
    fn an_html_block_keeps_its_lines() {
        let lines = markdown_lines("<div>\nhi\n</div>", 40, true);

        assert_eq!(texts(&lines)[..3], ["<div>", "hi", "</div>"]);
    }

    #[test]
    fn soft_breaks_keep_lines_and_empty_text_has_no_rows() {
        let lines = markdown_lines("one\ntwo", 40, true);

        assert_eq!(texts(&lines), ["one", "two"]);
        assert!(markdown_lines("", 40, true).is_empty());
        assert!(markdown_lines("$x$", 40, true).len() == 1);
    }
}
