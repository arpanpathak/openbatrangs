//! # Transcript blocks
//!
//! The transcript is a list of blocks: a user message, an agent step, a
//! final answer, a question from the model, or a notice. A step folds to one
//! line (number, title, model, time taken); an open step shows a thinking
//! line, a decision line, its reply, and one line per tool call. The thinking
//! line and each tool line open again to show the full reasoning, or the
//! arguments and full output.
//!
//! A block draws itself as rows that already fit the width, and caches them
//! by its content version and the width. Every change to a block gives it a
//! new version, so a cached drawing is never stale.

use super::chat::{code_lines, notice_line};
use super::markdown::markdown_lines;
use super::text::{spans_width, wrap_spans};
use crate::agent::describe_call;
use crate::constants::tui::CHAT_SEPARATOR_LENGTH;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

/// The part of a block a click or Enter acts on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Part {
    /// The block itself; for a step, its header line.
    Header,
    /// The thinking line of a step.
    Thinking,
    /// The tool line at this index of a step.
    Tool(usize),
}

/// How a tool call ended.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(super) struct ToolOutcome {
    /// Whether the tool did its job.
    pub(super) success: bool,
    /// A few words on the result.
    pub(super) summary: String,
    /// The full result text.
    pub(super) output: String,
}

/// One tool call inside a step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct ToolRun {
    /// Call id, matched by the result.
    id: String,
    /// Tool name.
    name: String,
    /// One-line description, such as `read_file → src/main.rs`.
    title: String,
    /// Arguments as JSON text.
    arguments: String,
    /// The result, once it arrived.
    outcome: Option<ToolOutcome>,
    /// Whether the arguments and output are shown.
    is_open: bool,
}

/// One agent step: a model turn and the tools it ran.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct Step {
    /// Step number, counted from 1.
    number: usize,
    /// Step budget of the run.
    max_steps: usize,
    /// Model that answered.
    model: String,
    /// Streamed reasoning.
    pub(super) reasoning: String,
    /// Streamed reply text.
    pub(super) reply: String,
    /// Tool calls, in order.
    tools: Vec<ToolRun>,
    /// Time from start to finish, once finished.
    elapsed_ms: Option<u64>,
    /// When the step started; not kept on disk.
    #[serde(skip)]
    started_at: Option<Instant>,
    /// Whether the step is open.
    is_open: bool,
    /// Whether the full reasoning is shown.
    is_thinking_open: bool,
}

impl Step {
    /// A running step, open.
    pub(super) fn new(number: usize, max_steps: usize, model: String) -> Self {
        Self {
            number,
            max_steps,
            model,
            reasoning: String::new(),
            reply: String::new(),
            tools: Vec::new(),
            elapsed_ms: None,
            started_at: Some(Instant::now()),
            is_open: true,
            is_thinking_open: false,
        }
    }

    /// Add a tool call.
    pub(super) fn add_tool(&mut self, id: String, name: String, arguments: String) {
        let title = describe_call(&name, &arguments);

        self.tools.push(ToolRun {
            id,
            name,
            title,
            arguments,
            outcome: None,
            is_open: false,
        });
    }

    /// Attach the result to the tool call with id `id`.
    pub(super) fn set_result(&mut self, id: &str, outcome: ToolOutcome) {
        if let Some(tool) = self.tools.iter_mut().find(|tool| tool.id == id) {
            tool.outcome = Some(outcome);
        }
    }

    /// Stop the clock and fold the step.
    pub(super) fn finish(&mut self) {
        let elapsed = self
            .started_at
            .map_or(0, |started_at| started_at.elapsed().as_millis() as u64);

        self.elapsed_ms = self.elapsed_ms.or(Some(elapsed));
        self.is_open = false;
    }

    /// Fold or unfold `part`; false when the step has no such part.
    fn toggle(&mut self, part: Part) -> bool {
        match part {
            Part::Header => flip(&mut self.is_open),
            Part::Thinking => flip(&mut self.is_thinking_open),
            Part::Tool(index) => self
                .tools
                .get_mut(index)
                .is_some_and(|tool| flip(&mut tool.is_open)),
        }
    }

    /// The parts the keyboard can move to: the header, and inside an open
    /// step the thinking line and each tool line.
    fn focus_parts(&self) -> Vec<Part> {
        if !self.is_open {
            return vec![Part::Header];
        }

        let thinking = has_text(&self.reasoning).then_some(Part::Thinking);

        std::iter::once(Part::Header)
            .chain(thinking)
            .chain((0..self.tools.len()).map(Part::Tool))
            .collect()
    }
}

/// Flip `flag` and report that something changed.
fn flip(flag: &mut bool) -> bool {
    *flag = !*flag;

    true
}

/// The final answer of a run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct Answer {
    /// The answer, in Markdown.
    pub(super) text: String,
    /// Absolute paths of the files the run changed.
    pub(super) changed_files: Vec<PathBuf>,
}

/// What a block shows.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) enum BlockContent {
    /// A message the user sent.
    User(String),
    /// One agent step.
    Step(Step),
    /// A final answer, always open.
    Answer(Answer),
    /// A question the model asked the user.
    Question(String),
    /// Status lines.
    Notice(Vec<String>),
}

/// The rows of a drawn block and the part each row belongs to.
#[derive(Debug, Default)]
pub(super) struct BlockRows {
    /// One line per terminal row.
    pub(super) lines: Vec<Line<'static>>,
    /// The part of the block each row belongs to.
    pub(super) parts: Vec<Part>,
}

/// A cached drawing of a block.
#[derive(Debug)]
struct Rendered {
    /// Block version it was drawn from.
    version: u64,
    /// Width it was drawn at.
    width: usize,
    /// The rows.
    rows: BlockRows,
}

/// A cached height of a block.
#[derive(Debug)]
struct Measured {
    /// Block version it was measured from.
    version: u64,
    /// Width it was measured at.
    width: usize,
    /// Height in rows.
    height: usize,
}

/// Source of block versions; every change takes the next one.
static NEXT_VERSION: AtomicU64 = AtomicU64::new(1);

/// A version no block has had yet.
fn new_version() -> u64 {
    NEXT_VERSION.fetch_add(1, Ordering::Relaxed)
}

/// One transcript block with its cached drawing.
#[derive(Debug, Serialize, Deserialize)]
pub(super) struct Block {
    /// What the block shows.
    pub(super) content: BlockContent,
    /// Content version; a new one after every change.
    #[serde(skip, default = "new_version")]
    version: u64,
    /// The last highlighted drawing.
    #[serde(skip)]
    rendered: Option<Rendered>,
    /// The last measured height.
    #[serde(skip)]
    measured: Option<Measured>,
}

impl Block {
    /// A new block showing `content`.
    pub(super) fn new(content: BlockContent) -> Self {
        Self {
            content,
            version: new_version(),
            rendered: None,
            measured: None,
        }
    }

    /// Mark the content as changed, so it is drawn again.
    pub(super) fn changed(&mut self) {
        self.version = new_version();
    }

    /// The step this block shows, if it is a step.
    pub(super) fn step_mut(&mut self) -> Option<&mut Step> {
        match &mut self.content {
            BlockContent::Step(step) => Some(step),
            _ => None,
        }
    }

    /// Apply `change` to the step this block shows; other blocks stay as
    /// they are.
    pub(super) fn update_step(&mut self, change: impl FnOnce(&mut Step)) {
        if let Some(step) = self.step_mut() {
            change(step);
            self.changed();
        }
    }

    /// Append `text` when this block is an answer; false for other blocks.
    pub(super) fn append_answer(&mut self, text: &str) -> bool {
        let BlockContent::Answer(answer) = &mut self.content else {
            return false;
        };

        answer.text.push_str(text);
        self.changed();

        true
    }

    /// Fold or unfold `part`; false when the block does not fold.
    pub(super) fn toggle(&mut self, part: Part) -> bool {
        let toggled = self.step_mut().is_some_and(|step| step.toggle(part));

        if toggled {
            self.changed();
        }

        toggled
    }

    /// The parts the keyboard can move to, in order.
    pub(super) fn focus_parts(&self) -> Vec<Part> {
        match &self.content {
            BlockContent::Step(step) => step.focus_parts(),
            _ => vec![Part::Header],
        }
    }

    /// The height in rows at `width`, measured without highlighting when no
    /// drawing for this version and width is cached.
    pub(super) fn height(&mut self, width: usize) -> usize {
        if let Some(height) = self.cached_height(width) {
            return height;
        }

        let height = content_rows(&self.content, width, false).lines.len();
        self.measured = Some(Measured {
            version: self.version,
            width,
            height,
        });

        height
    }

    /// The cached height for this version at `width`, if any.
    pub(super) fn cached_height(&self, width: usize) -> Option<usize> {
        let is_current =
            |version: u64, cached_width: usize| (version, cached_width) == (self.version, width);
        let drawn = self
            .rendered
            .as_ref()
            .filter(|rendered| is_current(rendered.version, rendered.width))
            .map(|rendered| rendered.rows.lines.len());
        let measured = self
            .measured
            .as_ref()
            .filter(|measured| is_current(measured.version, measured.width))
            .map(|measured| measured.height);

        drawn.or(measured)
    }

    /// The highlighted rows at `width`, and whether they were drawn now
    /// rather than taken from the cache.
    pub(super) fn render(&mut self, width: usize) -> (&BlockRows, bool) {
        let version = self.version;
        let is_stale = !self
            .rendered
            .as_ref()
            .is_some_and(|rendered| (rendered.version, rendered.width) == (version, width));

        if is_stale {
            self.rendered = None;
        }

        let content = &self.content;
        let rendered = self.rendered.get_or_insert_with(|| Rendered {
            version,
            width,
            rows: content_rows(content, width, true),
        });

        (&rendered.rows, is_stale)
    }

    /// The block's text without styles, for tests.
    #[cfg(test)]
    pub(super) fn source_text(&self) -> String {
        match &self.content {
            BlockContent::User(task) => format!("You: {task}"),
            BlockContent::Step(step) => format!("Step {} {}", step.number, step.reply),
            BlockContent::Answer(answer) => answer.text.clone(),
            BlockContent::Question(question) => question.clone(),
            BlockContent::Notice(lines) => lines.join("\n"),
        }
    }
}

/// True when `text` holds more than whitespace.
fn has_text(text: &str) -> bool {
    !text.trim().is_empty()
}

impl BlockRows {
    /// Add one row that already fits.
    fn push_line(&mut self, line: Line<'static>, part: Part) {
        self.lines.push(line);
        self.parts.push(part);
    }

    /// Add `spans`, wrapped to `width`.
    fn push_wrapped(&mut self, spans: Vec<Span<'static>>, width: usize, part: Part) {
        for row in wrap_spans(spans, width) {
            self.push_line(Line::from(row), part);
        }
    }

    /// Add `spans` as one row, cut to `width` with an ellipsis.
    fn push_fitted(&mut self, spans: Vec<Span<'static>>, width: usize, part: Part) {
        self.push_line(fit_to_width(spans, width), part);
    }

    /// Add rows drawn at a narrower width, each indented by `indent` cells.
    fn push_indented(&mut self, lines: Vec<Line<'static>>, indent: usize, part: Part) {
        for line in lines {
            let indented = [vec![Span::raw(" ".repeat(indent))], line.spans].concat();
            self.push_line(Line::from(indented).style(line.style), part);
        }
    }
}

/// `spans` as one row of at most `width` cells; a cut row ends in `…`.
fn fit_to_width(spans: Vec<Span<'static>>, width: usize) -> Line<'static> {
    if spans_width(&spans) <= width {
        return Line::from(spans);
    }

    let mut first_row = wrap_spans(spans, width.saturating_sub(1))
        .into_iter()
        .next()
        .unwrap_or_default();
    first_row.push(Span::raw("…"));

    Line::from(first_row)
}

/// Draw `content` at `width`, highlighting code when `highlight` is set.
fn content_rows(content: &BlockContent, width: usize, highlight: bool) -> BlockRows {
    let width = width.max(1);

    match content {
        BlockContent::User(task) => user_rows(task, width),
        BlockContent::Step(step) => step_rows(step, width, highlight),
        BlockContent::Answer(answer) => answer_rows(answer, width, highlight),
        BlockContent::Question(question) => question_rows(question, width, highlight),
        BlockContent::Notice(lines) => notice_rows(lines, width),
    }
}

/// A blank row, a rule, and the user's message.
fn user_rows(task: &str, width: usize) -> BlockRows {
    let mut rows = BlockRows::default();
    let rule = "─".repeat(CHAT_SEPARATOR_LENGTH.min(width));
    let bold = Style::new().add_modifier(Modifier::BOLD);

    rows.push_line(Line::default(), Part::Header);
    rows.push_line(
        Line::styled(rule, Style::new().fg(Color::DarkGray)),
        Part::Header,
    );

    for line in format!("You: {task}").lines() {
        rows.push_wrapped(
            vec![Span::styled(line.to_string(), bold)],
            width,
            Part::Header,
        );
    }

    rows
}

/// Each notice line, styled by its marker.
fn notice_rows(lines: &[String], width: usize) -> BlockRows {
    let mut rows = BlockRows::default();

    for line in lines {
        rows.push_wrapped(notice_line(line).spans, width, Part::Header);
    }

    rows
}

/// A blank row, a heading row, then `markdown` drawn at `width`.
fn titled_markdown_rows(
    title: Span<'static>,
    markdown: &str,
    width: usize,
    highlight: bool,
) -> BlockRows {
    let mut rows = BlockRows::default();

    rows.push_line(Line::default(), Part::Header);
    rows.push_line(Line::from(title), Part::Header);

    for line in markdown_lines(markdown, width, highlight) {
        rows.push_line(line, Part::Header);
    }

    rows
}

/// The final answer, then the changed files.
fn answer_rows(answer: &Answer, width: usize, highlight: bool) -> BlockRows {
    let title = Span::styled(
        "✅ Answer",
        Style::new().fg(Color::Green).add_modifier(Modifier::BOLD),
    );
    let mut rows = titled_markdown_rows(title, &answer.text, width, highlight);

    changed_files_rows(&answer.changed_files, width, &mut rows);

    rows
}

/// The "files changed" heading and one path per row; nothing without files.
fn changed_files_rows(files: &[PathBuf], width: usize, rows: &mut BlockRows) {
    if files.is_empty() {
        return;
    }

    let heading = Style::new().fg(Color::Yellow).add_modifier(Modifier::BOLD);
    rows.push_line(Line::default(), Part::Header);
    rows.push_line(Line::styled("📁 Files changed:", heading), Part::Header);

    file_rows(files, width, rows);
}

/// One indented path per file.
fn file_rows(files: &[PathBuf], width: usize, rows: &mut BlockRows) {
    for file in files {
        let path = format!("   {}", file.display());
        rows.push_wrapped(vec![Span::raw(path)], width, Part::Header);
    }
}

/// The model's question.
fn question_rows(question: &str, width: usize, highlight: bool) -> BlockRows {
    let title = Span::styled(
        "❓ Question",
        Style::new().fg(Color::Yellow).add_modifier(Modifier::BOLD),
    );

    titled_markdown_rows(title, question, width, highlight)
}

/// The header row, then the inside of the step when it is open.
fn step_rows(step: &Step, width: usize, highlight: bool) -> BlockRows {
    let mut rows = BlockRows::default();

    rows.push_fitted(step_header(step), width, Part::Header);

    if step.is_open {
        open_step_rows(step, width, highlight, &mut rows);
    }

    rows
}

/// The thinking line, the decision line, the reply, and the tool lines.
fn open_step_rows(step: &Step, width: usize, highlight: bool, rows: &mut BlockRows) {
    thinking_rows(step, width, highlight, rows);
    decision_rows(step, width, rows);
    reply_rows(step, width, highlight, rows);

    for (index, tool) in step.tools.iter().enumerate() {
        tool_rows(index, tool, width, highlight, rows);
    }
}

/// Cells that indent the inside of an open step.
const STEP_INDENT: usize = 4;

/// Cells that indent the arguments and output of an open tool line.
const TOOL_INDENT: usize = 6;

/// The thinking line, and the full reasoning when it is open.
fn thinking_rows(step: &Step, width: usize, highlight: bool, rows: &mut BlockRows) {
    if !has_text(&step.reasoning) {
        return;
    }

    let first_line = step.reasoning.trim().lines().next().unwrap_or_default();
    let line = vec![
        Span::raw(format!("  {} ", fold_arrow(step.is_thinking_open))),
        Span::styled(
            format!("💭 {first_line}"),
            Style::new()
                .fg(Color::DarkGray)
                .add_modifier(Modifier::ITALIC),
        ),
    ];
    rows.push_fitted(line, width, Part::Thinking);

    full_reasoning_rows(step, width, highlight, rows);
}

/// The whole reasoning as Markdown, when the thinking line is open.
fn full_reasoning_rows(step: &Step, width: usize, highlight: bool, rows: &mut BlockRows) {
    if !step.is_thinking_open {
        return;
    }

    let room = width.saturating_sub(STEP_INDENT);
    let italic = Style::new().add_modifier(Modifier::ITALIC);
    let lines = markdown_lines(&step.reasoning, room, highlight)
        .into_iter()
        .map(|line| line.patch_style(italic))
        .collect();

    rows.push_indented(lines, STEP_INDENT, Part::Thinking);
}

/// What the step chose: its tool calls, or to answer.
fn decision(step: &Step) -> Option<String> {
    let names: Vec<&str> = step.tools.iter().map(|tool| tool.name.as_str()).collect();

    match (names.is_empty(), has_text(&step.reply)) {
        (false, _) => Some(names.join(", ")),
        (true, true) => Some("answer".to_string()),
        (true, false) => None,
    }
}

/// The decision line, when the step decided something.
fn decision_rows(step: &Step, width: usize, rows: &mut BlockRows) {
    if let Some(decision) = decision(step) {
        let line = Span::styled(format!("  → {decision}"), Style::new().fg(Color::DarkGray));
        rows.push_fitted(vec![line], width, Part::Header);
    }
}

/// The reply text, when there is any.
fn reply_rows(step: &Step, width: usize, highlight: bool, rows: &mut BlockRows) {
    if !has_text(&step.reply) {
        return;
    }

    let room = width.saturating_sub(STEP_INDENT);
    rows.push_indented(
        markdown_lines(&step.reply, room, highlight),
        STEP_INDENT,
        Part::Header,
    );
}

/// The tool line, and its arguments and output when it is open.
fn tool_rows(index: usize, tool: &ToolRun, width: usize, highlight: bool, rows: &mut BlockRows) {
    rows.push_fitted(tool_line(tool), width, Part::Tool(index));

    if tool.is_open {
        tool_detail_rows(index, tool, width, highlight, rows);
    }
}

/// The arguments as highlighted JSON, then the full output as Markdown.
fn tool_detail_rows(
    index: usize,
    tool: &ToolRun,
    width: usize,
    highlight: bool,
    rows: &mut BlockRows,
) {
    let room = width.saturating_sub(TOOL_INDENT);
    let label = Style::new().fg(Color::DarkGray);
    let arguments = pretty_json(&tool.arguments);

    rows.push_indented(
        vec![Line::styled("arguments:", label)],
        STEP_INDENT,
        Part::Tool(index),
    );
    for line in code_lines(&arguments, "json", highlight) {
        let wrapped = wrap_spans(line, room).into_iter().map(Line::from).collect();
        rows.push_indented(wrapped, TOOL_INDENT, Part::Tool(index));
    }

    let output = tool
        .outcome
        .as_ref()
        .map_or("", |outcome| outcome.output.as_str());
    rows.push_indented(
        vec![Line::styled("output:", label)],
        STEP_INDENT,
        Part::Tool(index),
    );
    rows.push_indented(
        markdown_lines(output, room, highlight),
        TOOL_INDENT,
        Part::Tool(index),
    );
}

/// `arguments` laid out on several lines; as given when it is not JSON.
fn pretty_json(arguments: &str) -> String {
    serde_json::from_str::<serde_json::Value>(arguments)
        .ok()
        .and_then(|value| serde_json::to_string_pretty(&value).ok())
        .unwrap_or_else(|| arguments.to_string())
}

/// `▾` for an open part, `▸` for a folded one.
fn fold_arrow(is_open: bool) -> &'static str {
    match is_open {
        true => "▾",
        false => "▸",
    }
}

/// The tool line: arrow, a mark for the result, the call, and its summary.
fn tool_line(tool: &ToolRun) -> Vec<Span<'static>> {
    let (mark, color) = result_mark(tool.outcome.as_ref());
    let summary = tool
        .outcome
        .as_ref()
        .map_or("running", |outcome| outcome.summary.as_str());

    vec![
        Span::raw(format!("  {} ", fold_arrow(tool.is_open))),
        Span::styled(mark, Style::new().fg(color)),
        Span::raw(format!(" {}", tool.title)),
        Span::styled(
            format!(" · {mark} {summary}"),
            Style::new().fg(Color::DarkGray),
        ),
    ]
}

/// `✓` for a tool that did its job, `✗` for one that failed, `…` while it runs.
fn result_mark(outcome: Option<&ToolOutcome>) -> (&'static str, Color) {
    match outcome.map(|outcome| outcome.success) {
        Some(true) => ("✓", Color::Green),
        Some(false) => ("✗", Color::Red),
        None => ("…", Color::Yellow),
    }
}

/// The step header: arrow, number, title, model, and time taken.
fn step_header(step: &Step) -> Vec<Span<'static>> {
    let number = format!(
        "{} Step {}/{}",
        fold_arrow(step.is_open),
        step.number,
        step.max_steps
    );
    let details = format!(" · {} · {}", step.model, step_time(step));

    vec![
        Span::styled(
            number,
            Style::new().fg(Color::Magenta).add_modifier(Modifier::BOLD),
        ),
        Span::styled(
            format!(" · {}", step_title(step)),
            Style::new().add_modifier(Modifier::BOLD),
        ),
        Span::styled(details, Style::new().fg(Color::DarkGray)),
    ]
}

/// The time a finished step took, or `running`.
fn step_time(step: &Step) -> String {
    match step.elapsed_ms {
        Some(elapsed_ms) => format_elapsed(elapsed_ms),
        None => "running".to_string(),
    }
}

/// `4.2s` under a minute, `2m 5s` from a minute on.
fn format_elapsed(elapsed_ms: u64) -> String {
    let seconds = elapsed_ms / 1_000;

    match seconds < 60 {
        true => format!("{:.1}s", elapsed_ms as f64 / 1_000.0),
        false => format!("{}m {}s", seconds / 60, seconds % 60),
    }
}

/// The step title: from the reasoning, else the reply, else the first tool
/// call; `working` before any of them arrived.
fn step_title(step: &Step) -> String {
    let text = [&step.reasoning, &step.reply]
        .into_iter()
        .find(|text| has_text(text));
    let first_tool = step.tools.first().map(|tool| tool.title.clone());

    match (text, first_tool) {
        (Some(text), _) => title_from_text(text),
        (None, Some(title)) => title,
        (None, None) => "working".to_string(),
    }
}

/// The model's own heading when the text starts with one (`**Checking the
/// tests**` or `# Checking the tests`), otherwise its first sentence.
pub(super) fn title_from_text(text: &str) -> String {
    heading_title(text).unwrap_or_else(|| first_sentence(text))
}

/// The heading on the first line of `text`, if it is a bold line or a
/// Markdown heading.
fn heading_title(text: &str) -> Option<String> {
    let first_line = text.trim_start().lines().next()?.trim();
    let bold = first_line
        .strip_prefix("**")
        .and_then(|rest| rest.strip_suffix("**"));
    let heading = first_line
        .strip_prefix('#')
        .map(|rest| rest.trim_start_matches('#'));

    bold.or(heading)
        .map(str::trim)
        .filter(|title| !title.is_empty())
        .map(str::to_string)
}

/// The text up to the first sentence end or line break.
fn first_sentence(text: &str) -> String {
    let trimmed = text.trim_start();
    let end = trimmed.find(['.', '!', '?', '\n']).unwrap_or(trimmed.len());

    trimmed[..end].trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tui::text::spans_text;

    /// The text of every row of `block` at `width`.
    fn row_texts(block: &mut Block, width: usize) -> Vec<String> {
        let (rows, _is_new) = block.render(width);

        rows.lines
            .iter()
            .map(|line| spans_text(&line.spans))
            .collect()
    }

    /// A step with reasoning, a reply and two tool calls, one finished.
    fn busy_step() -> Step {
        let mut step = Step::new(3, 10, "qwen".to_string());
        step.reasoning = "**Checking the tests**\nI should run them.".to_string();
        step.reply = "Running `cargo test`.".to_string();
        step.add_tool(
            "c1".to_string(),
            "run_command".to_string(),
            r#"{"command":"cargo test"}"#.to_string(),
        );
        step.add_tool(
            "c2".to_string(),
            "read_file".to_string(),
            r#"{"path":"a.rs"}"#.to_string(),
        );
        step.set_result(
            "c1",
            ToolOutcome {
                success: true,
                summary: "4 passed".to_string(),
                output: "test result: ok".to_string(),
            },
        );
        step.set_result(
            "missing",
            ToolOutcome {
                success: false,
                summary: String::new(),
                output: String::new(),
            },
        );

        step
    }

    #[test]
    fn titles_come_from_headings_or_the_first_sentence() {
        assert_eq!(
            title_from_text("**Checking the tests**\nmore"),
            "Checking the tests"
        );
        assert_eq!(title_from_text("## Reading files"), "Reading files");
        assert_eq!(title_from_text("  I will look. Then act."), "I will look");
        assert_eq!(title_from_text("**  **\nplain line"), "**  **");
        assert_eq!(title_from_text(""), "");
    }

    #[test]
    fn an_open_step_shows_thinking_decision_reply_and_tools() {
        let mut block = Block::new(BlockContent::Step(busy_step()));

        let rows = row_texts(&mut block, 80);

        assert_eq!(rows[0], "▾ Step 3/10 · Checking the tests · qwen · running");
        assert_eq!(rows[1], "  ▸ 💭 **Checking the tests**");
        assert_eq!(rows[2], "  → run_command, read_file");
        assert_eq!(rows[3], "    Running cargo test.");
        assert_eq!(rows[4], "  ▸ ✓ run_command → cargo test · ✓ 4 passed");
        assert_eq!(rows[5], "  ▸ … read_file → a.rs · … running");
        assert_eq!(rows.len(), 6);
    }

    #[test]
    fn opening_the_thinking_and_a_tool_shows_their_full_text() {
        let mut block = Block::new(BlockContent::Step(busy_step()));

        assert!(block.toggle(Part::Thinking));
        assert!(block.toggle(Part::Tool(0)));
        assert!(block.toggle(Part::Tool(1)));
        assert!(!block.toggle(Part::Tool(7)));
        let rows = row_texts(&mut block, 80);

        assert!(rows.contains(&"    I should run them.".to_string()));
        assert!(rows.contains(&"        \"command\": \"cargo test\"".to_string()));
        assert!(rows.contains(&"      test result: ok".to_string()));
        assert_eq!(
            block.focus_parts(),
            [Part::Header, Part::Thinking, Part::Tool(0), Part::Tool(1)]
        );
    }

    #[test]
    fn a_finished_step_folds_to_its_header() {
        let mut step = busy_step();
        step.finish();
        step.elapsed_ms = Some(4_200);
        let mut block = Block::new(BlockContent::Step(step));

        let rows = row_texts(&mut block, 80);

        assert_eq!(rows, ["▸ Step 3/10 · Checking the tests · qwen · 4.2s"]);
        assert_eq!(block.focus_parts(), [Part::Header]);
        assert_eq!(row_texts(&mut block, 20), ["▸ Step 3/10 · Check…"]);
    }

    #[test]
    fn step_titles_fall_back_to_the_tool_or_working() {
        let mut step = Step::new(1, 2, "m".to_string());
        assert_eq!(step_title(&step), "working");

        step.add_tool("c1".to_string(), "list_files".to_string(), "{}".to_string());
        assert_eq!(step_title(&step), "list_files → .");
        assert_eq!(decision(&step).as_deref(), Some("list_files"));

        step.tools.clear();
        step.reply = "Done. Bye.".to_string();
        assert_eq!(step_title(&step), "Done");
        assert_eq!(decision(&step).as_deref(), Some("answer"));
    }

    #[test]
    fn the_header_folds_and_other_blocks_ignore_step_changes() {
        let mut block = Block::new(BlockContent::Step(busy_step()));
        let mut notice = Block::new(BlockContent::Notice(vec!["x".to_string()]));
        let question = Block::new(BlockContent::Question("Which?".to_string()));

        assert!(block.toggle(Part::Header));
        assert_eq!(row_texts(&mut block, 80).len(), 1);

        notice.update_step(Step::finish);
        assert_eq!(notice.source_text(), "x");
        assert!(!notice.append_answer("never"));
        assert_eq!(question.source_text(), "Which?");
    }

    #[test]
    fn elapsed_times_read_as_seconds_or_minutes() {
        assert_eq!(format_elapsed(4_210), "4.2s");
        assert_eq!(format_elapsed(125_000), "2m 5s");
    }

    #[test]
    fn a_failed_tool_has_a_cross_and_bad_arguments_stay_as_text() {
        let mut step = Step::new(1, 1, "m".to_string());
        step.add_tool(
            "c1".to_string(),
            "run_command".to_string(),
            "not json".to_string(),
        );
        step.set_result(
            "c1",
            ToolOutcome {
                success: false,
                summary: "exit 101".to_string(),
                output: String::new(),
            },
        );
        let mut block = Block::new(BlockContent::Step(step));
        block.toggle(Part::Tool(0));

        let rows = row_texts(&mut block, 60);

        assert_eq!(rows[1], "  → run_command");
        assert_eq!(rows[2], "  ▾ ✗ run_command · ✗ exit 101");
        assert_eq!(rows[4], "      not json");
    }

    #[test]
    fn answers_questions_users_and_notices_have_their_own_rows() {
        let answer = Answer {
            text: "**Done**".to_string(),
            changed_files: vec![PathBuf::from("/w/a.rs")],
        };
        let mut answer = Block::new(BlockContent::Answer(answer));
        let mut question = Block::new(BlockContent::Question("Which file?".to_string()));
        let mut user = Block::new(BlockContent::User("fix it\nplease".to_string()));
        let mut notice = Block::new(BlockContent::Notice(vec![
            "⚠️ careful".to_string(),
            String::new(),
        ]));

        assert_eq!(
            row_texts(&mut answer, 40),
            [
                "",
                "✅ Answer",
                "Done",
                "",
                "📁 Files changed:",
                "   /w/a.rs"
            ]
        );
        assert_eq!(
            row_texts(&mut question, 40),
            ["", "❓ Question", "Which file?"]
        );
        assert_eq!(
            row_texts(&mut user, 4),
            ["", "────", "You:", " fix", " it", "plea", "se"]
        );
        assert_eq!(row_texts(&mut notice, 40), ["⚠️ careful", ""]);
        assert!(!notice.toggle(Part::Header));
        assert_eq!(notice.focus_parts(), [Part::Header]);
    }

    #[test]
    fn drawings_and_heights_are_cached_by_version_and_width() {
        let mut block = Block::new(BlockContent::Notice(vec!["abcdef".to_string()]));

        assert_eq!(block.height(3), 2);
        assert_eq!(block.height(3), 2);
        assert!(block.render(3).1);
        assert!(!block.render(3).1);
        assert_eq!(block.height(3), 2);
        assert!(block.render(6).1);

        block.changed();
        assert!(block.render(6).1);
    }

    #[test]
    fn blocks_survive_a_round_trip_through_json() -> anyhow::Result<()> {
        let block = Block::new(BlockContent::Step(busy_step()));

        let json = serde_json::to_string(&block)?;
        let mut restored: Block = serde_json::from_str(&json)?;

        assert_ne!(restored.version, block.version);
        assert_eq!(restored.source_text(), "Step 3 Running `cargo test`.");
        assert!(
            restored
                .step_mut()
                .is_some_and(|step| step.started_at.is_none())
        );

        Ok(())
    }
}
