//! # The stats table in the panel
//!
//! The panel shows one row per machine: the model server, when it runs on
//! another host, and this machine. Every row has the same columns, so each
//! figure sits under its heading and the numbers line up. A figure a machine
//! did not report is shown as `–`. Figures that only one machine has, such as
//! the engine load of a vLLM server or the page cache of this machine, follow
//! the rows as notes, each under the name of its machine.

use crate::constants::perf::{
    BYTES_PER_GIB, MIB_PER_GIB, NO_SERVER_STATS, WAITING_FOR_SERVER_STATS,
};
use crate::perf::{EngineLoad, LoadedModel, ServerGpu, ServerStats, ServerSystem, SystemStats};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use unicode_width::UnicodeWidthStr;

/// Column headings, after the machine name.
pub(super) const HEADINGS: [&str; 7] = ["GPU", "load", "power", "temp", "memory", "CPU", "cores"];

/// The narrowest each column may be: room for the widest value a figure can
/// take, such as `100%`, `999.9 W`, `100 °C` or `999.9/999.9 GB`, so a column
/// keeps its width while its figures change. The GPU name column has no
/// minimum; a machine's GPU name does not change.
const MIN_WIDTHS: [usize; HEADINGS.len()] = [0, 4, 7, 6, 14, 4, 5];

/// Index of the GPU name column, the one column of text that is aligned left.
pub(super) const TEXT_COLUMN: usize = 0;

/// Shown for a figure a machine did not report.
const MISSING: &str = "–";

/// Spaces between two columns.
const COLUMN_GAP: &str = "   ";

/// Name of the row for the machine openbatrangs runs on.
const LOCAL_NAME: &str = "local";

/// One machine and its figures, in the order of [`HEADINGS`].
#[derive(Debug, PartialEq)]
pub(super) struct MachineRow {
    /// The machine's name: the server's host, or `local`.
    pub(super) name: String,
    /// The figures, one per heading.
    pub(super) figures: [String; HEADINGS.len()],
}

/// A line of figures that only one machine has, shown under the rows.
#[derive(Debug, PartialEq)]
pub(super) struct Note {
    /// The machine the figures belong to.
    pub(super) name: String,
    /// The figures, as `label value` pairs joined with ` · `.
    pub(super) text: String,
}

/// The rows and notes the panel draws under the model line.
#[derive(Debug, Default, PartialEq)]
pub(super) struct StatsTable {
    /// One row per machine.
    pub(super) rows: Vec<MachineRow>,
    /// The notes, in the order of the rows they belong to.
    pub(super) notes: Vec<Note>,
}

/// What the panel knows about a model server on another host.
pub(super) struct RemoteServer<'a> {
    /// The host to name the server by when its report has no host.
    pub(super) fallback_name: &'a str,
    /// The latest poll, or `None` before the first one finishes.
    pub(super) stats: Option<&'a ServerStats>,
}

impl StatsTable {
    /// The table for `remote`, when the server runs on another host, and for
    /// this machine's `local` figures.
    pub(super) fn new(remote: Option<RemoteServer>, local: &SystemStats) -> Self {
        let mut table = Self::default();

        if let Some(remote) = remote {
            table.add_server(remote);
        }

        table.add_local(local);
        table
    }

    /// Add the server's row and notes, or a note that it has not reported.
    fn add_server(&mut self, remote: RemoteServer) {
        let Some(stats) = remote.stats else {
            return self.note(remote.fallback_name, WAITING_FOR_SERVER_STATS);
        };

        let system = stats.system.as_ref();
        let name = system
            .and_then(|system| system.host.as_deref())
            .unwrap_or(remote.fallback_name);
        let lines_before = self.rows.len() + self.notes.len();
        let notes = [
            system.and_then(unified_memory_note),
            stats.engine_load.as_ref().and_then(load_note),
            stats.loaded_models.as_deref().map(models_note),
        ];

        self.rows
            .extend(system.map(|system| server_row(name, system)));

        for text in notes.into_iter().flatten() {
            self.note(name, &text);
        }

        if self.rows.len() + self.notes.len() == lines_before {
            self.note(name, NO_SERVER_STATS);
        }
    }

    /// Add this machine's row, and a note with its GPU memory and the parts
    /// of its RAM.
    fn add_local(&mut self, stats: &SystemStats) {
        self.rows.push(local_row(stats));
        self.note(LOCAL_NAME, &local_memory_note(stats));
    }

    /// Add a note for machine `name`.
    fn note(&mut self, name: &str, text: &str) {
        self.notes.push(Note {
            name: name.to_string(),
            text: text.to_string(),
        });
    }

    /// The table as aligned lines: the headings, then each row followed by
    /// its notes, then the notes of machines without a row. A note's text
    /// starts under the first figure column; a note under its row leaves the
    /// name out and reads as the row's continuation.
    pub(super) fn lines(&self) -> Vec<Line<'static>> {
        let name_width = self.name_width();
        let widths = self.column_widths();
        let mut lines = vec![heading_line(name_width, &widths)];

        for row in &self.rows {
            let notes = self.notes.iter().filter(|note| note.name == row.name);

            lines.push(row_line(row, name_width, &widths));
            lines.extend(notes.map(|note| note_line("", &note.text, name_width)));
        }

        let orphans = self.notes.iter().filter(|note| !self.has_row(&note.name));
        lines.extend(orphans.map(|note| note_line(&note.name, &note.text, name_width)));

        lines
    }

    /// Whether machine `name` has a row.
    fn has_row(&self, name: &str) -> bool {
        self.rows.iter().any(|row| row.name == name)
    }

    /// Width of the name column: the longest machine name.
    fn name_width(&self) -> usize {
        let row_names = self.rows.iter().map(|row| row.name.width());
        let note_names = self.notes.iter().map(|note| note.name.width());

        row_names.chain(note_names).max().unwrap_or(0)
    }

    /// Width of each figure column: its minimum, its heading or its widest
    /// figure, whichever is widest.
    fn column_widths(&self) -> [usize; HEADINGS.len()] {
        std::array::from_fn(|column| {
            let fixed = MIN_WIDTHS[column].max(HEADINGS[column].width());

            self.rows
                .iter()
                .map(|row| row.figures[column].width())
                .fold(fixed, usize::max)
        })
    }
}

/// The headings, dimmed, each over its column.
fn heading_line(name_width: usize, widths: &[usize; HEADINGS.len()]) -> Line<'static> {
    let style = Style::default().fg(Color::DarkGray);
    let mut spans = vec![Span::raw(" ".repeat(name_width))];

    for (column, heading) in HEADINGS.iter().enumerate() {
        spans.push(Span::raw(COLUMN_GAP));
        spans.push(Span::styled(
            aligned(heading, widths[column], column),
            style,
        ));
    }

    Line::from(spans)
}

/// One machine: its name in bold magenta, then its figures in white, with a
/// missing figure in gray.
fn row_line(
    row: &MachineRow,
    name_width: usize,
    widths: &[usize; HEADINGS.len()],
) -> Line<'static> {
    let mut spans = vec![name_span(&row.name, name_width)];

    for (column, figure) in row.figures.iter().enumerate() {
        let color = match figure.as_str() {
            MISSING => Color::DarkGray,
            _ => Color::White,
        };

        spans.push(Span::raw(COLUMN_GAP));
        spans.push(Span::styled(
            aligned(figure, widths[column], column),
            Style::default().fg(color),
        ));
    }

    Line::from(spans)
}

/// A note: `name`, which may be empty, then `text` in cyan.
fn note_line(name: &str, text: &str, name_width: usize) -> Line<'static> {
    Line::from(vec![
        name_span(name, name_width),
        Span::raw(COLUMN_GAP),
        Span::styled(text.to_string(), Style::default().fg(Color::Cyan)),
    ])
}

/// A machine name in bold magenta, padded to `width` cells.
fn name_span(name: &str, width: usize) -> Span<'static> {
    let style = Style::default()
        .fg(Color::Magenta)
        .add_modifier(Modifier::BOLD);

    Span::styled(aligned(name, width, TEXT_COLUMN), style)
}

/// `text` padded with spaces to `width` cells: after the text in the text
/// column, before it in the number columns.
fn aligned(text: &str, width: usize, column: usize) -> String {
    let padding = " ".repeat(width.saturating_sub(text.width()));

    match column {
        TEXT_COLUMN => format!("{text}{padding}"),
        _ => format!("{padding}{text}"),
    }
}

/// The server's row from its `/v1/system` report.
fn server_row(name: &str, system: &ServerSystem) -> MachineRow {
    let gpu = system.gpu.as_ref();
    let cpu = system.cpu.as_ref();

    MachineRow {
        name: name.to_string(),
        figures: [
            gpu.and_then(|gpu| gpu.name.clone()),
            gpu.and_then(|gpu| gpu.utilization_percent).map(percent),
            gpu.and_then(|gpu| gpu.power_watts).map(watts),
            gpu.and_then(|gpu| gpu.temperature_celsius).map(celsius),
            gpu.and_then(gpu_memory),
            cpu.and_then(|cpu| cpu.utilization_percent).map(percent),
            cpu.and_then(|cpu| cpu.cores).map(|cores| cores.to_string()),
        ]
        .map(|figure| figure.unwrap_or_else(|| MISSING.to_string())),
    }
}

/// The server GPU's memory in use out of all of it, or `None` unless both
/// figures are known.
fn gpu_memory(gpu: &ServerGpu) -> Option<String> {
    let used = gpu.memory_used_bytes? as f64 / BYTES_PER_GIB;
    let total = gpu.memory_total_bytes? as f64 / BYTES_PER_GIB;

    Some(format!("{used:.1}/{total:.1} GB"))
}

/// The note for a server GPU that shares its memory with the CPU, as on
/// Jetson Thor.
fn unified_memory_note(system: &ServerSystem) -> Option<String> {
    let is_unified = system.gpu.as_ref()?.memory_is_unified?;

    is_unified.then(|| "memory shared with the CPU".to_string())
}

/// This machine's row: its GPU, and its RAM in the memory column.
fn local_row(stats: &SystemStats) -> MachineRow {
    let memory = format!(
        "{:.1}/{:.1} GB",
        gibibytes(stats.memory_used_mb),
        gibibytes(stats.memory_total_mb)
    );

    MachineRow {
        name: LOCAL_NAME.to_string(),
        figures: [
            stats.gpu_name.clone(),
            stats.gpu_util_percent.map(percent),
            stats.gpu_power_watts.map(watts),
            stats.gpu_temp_c.map(celsius),
            Some(memory),
            stats.cpu_util_percent.map(percent),
            Some(stats.cpu_cores.to_string()),
        ]
        .map(|figure| figure.unwrap_or_else(|| MISSING.to_string())),
    }
}

/// The engine load, such as `running 2 · waiting 1 · KV cache 34%`, or
/// `None` when the metrics held none of its figures.
fn load_note(load: &EngineLoad) -> Option<String> {
    let figures = [
        load.running
            .map(|running| format!("running {running:>3.0}")),
        load.waiting
            .map(|waiting| format!("waiting {waiting:>3.0}")),
        load.kv_cache_fraction
            .map(|fraction| format!("KV cache {:>3.0}%", fraction * 100.0)),
        load.tokens_per_second
            .map(|rate| format!("speed {rate:>5.1} tok/s")),
    ];
    let present: Vec<String> = figures.into_iter().flatten().collect();

    (!present.is_empty()).then(|| present.join(" · "))
}

/// The models Ollama holds in memory, such as
/// `loaded qwen2.5-coder:7b 5.1 GB · nomic-embed-text:latest`.
fn models_note(models: &[LoadedModel]) -> String {
    let described: Vec<String> = models
        .iter()
        .map(|model| match model.size_vram {
            Some(bytes) => format!("{} {:.1} GB", model.name, bytes as f64 / BYTES_PER_GIB),
            None => model.name.clone(),
        })
        .collect();

    match described.is_empty() {
        true => "loaded none".to_string(),
        false => format!("loaded {}", described.join(" · ")),
    }
}

/// This machine's GPU memory, when it has its own, and the parts of its RAM.
fn local_memory_note(stats: &SystemStats) -> String {
    let vram = stats
        .gpu_memory_used_mb
        .zip(stats.gpu_memory_total_mb)
        .map(|(used, total)| {
            format!(
                "VRAM {:>5.1}/{:>5.1} GB · ",
                gibibytes(used),
                gibibytes(total)
            )
        })
        .unwrap_or_default();

    format!(
        "{vram}RAM free {:>5.1} GB · cache {:>5.1} GB · buffers {:>5.1} GB · shared {:>5.1} GB",
        gibibytes(stats.memory_free_mb),
        gibibytes(stats.memory_cached_mb),
        gibibytes(stats.memory_buffers_mb),
        gibibytes(stats.memory_shared_mb),
    )
}

/// `value` as a whole percentage.
fn percent(value: f64) -> String {
    format!("{value:.0}%")
}

/// `value` in watts, to one decimal.
fn watts(value: f64) -> String {
    format!("{value:.1} W")
}

/// `value` in degrees Celsius.
fn celsius(value: f64) -> String {
    format!("{value:.0} °C")
}

/// `mebibytes` in gibibytes.
fn gibibytes(mebibytes: u64) -> f64 {
    mebibytes as f64 / MIB_PER_GIB
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The `/v1/system` body thor-tigress-agent publishes.
    const THOR_SYSTEM: &str = r#"{"host":"thor","gpu":{"name":"NVIDIA Thor","utilization_percent":37.0,"memory_used_bytes":90194313216,"memory_total_bytes":131072000000,"memory_is_unified":true,"power_watts":4.37,"temperature_celsius":41.0},"cpu":{"utilization_percent":12.0,"cores":14}}"#;

    /// A discrete GPU that reports no power or temperature.
    const DISCRETE_SYSTEM: &str = r#"{"host":"gpu-box","gpu":{"utilization_percent":5.0,"memory_used_bytes":12884901888,"memory_total_bytes":25769803776,"memory_is_unified":false,"power_watts":null,"temperature_celsius":null}}"#;

    /// llama.cpp metrics.
    const LLAMACPP_METRICS: &str = "llamacpp:requests_processing 1\nllamacpp:requests_deferred 3\nllamacpp:kv_cache_usage_ratio 0.25\nllamacpp:predicted_tokens_seconds 52.1\n";

    /// An Ollama `/api/ps` body with two models, one without `size_vram`.
    const OLLAMA_PS: &str = r#"{"models":[{"name":"qwen2.5-coder:7b","size_vram":5476083712},{"name":"nomic-embed-text:latest"}]}"#;

    /// The table for a server on another host that published `stats`.
    fn remote_table(stats: Option<&ServerStats>) -> StatsTable {
        let remote = RemoteServer {
            fallback_name: "10.0.0.5",
            stats,
        };

        StatsTable::new(Some(remote), &SystemStats::default())
    }

    #[test]
    fn the_thor_report_fills_every_column() {
        let stats = ServerStats::from_bodies(Some(THOR_SYSTEM), None, None);

        let table = remote_table(Some(&stats));

        assert_eq!(table.rows[0].name, "thor");
        assert_eq!(
            table.rows[0].figures,
            [
                "NVIDIA Thor",
                "37%",
                "4.4 W",
                "41 °C",
                "84.0/122.1 GB",
                "12%",
                "14"
            ]
        );
        assert_eq!(table.notes[0].text, "memory shared with the CPU");
        assert_eq!(table.rows[1].name, "local");
    }

    #[test]
    fn a_discrete_gpu_shows_its_memory_and_dashes_for_null_figures() {
        let stats = ServerStats::from_bodies(Some(DISCRETE_SYSTEM), None, None);

        let table = remote_table(Some(&stats));

        assert_eq!(
            table.rows[0].figures,
            ["–", "5%", "–", "–", "12.0/24.0 GB", "–", "–"]
        );
    }

    #[test]
    fn metrics_alone_give_a_note_under_the_server_address() {
        let stats = ServerStats::from_bodies(None, Some(LLAMACPP_METRICS), None);

        let table = remote_table(Some(&stats));

        assert_eq!(table.rows.len(), 1, "only the local row");
        assert_eq!(
            table.notes[0],
            Note {
                name: "10.0.0.5".to_string(),
                text: "running   1 · waiting   3 · KV cache  25% · speed  52.1 tok/s".to_string(),
            }
        );
    }

    #[test]
    fn ollama_models_are_listed_with_their_memory() {
        let stats = ServerStats::from_bodies(None, None, Some(OLLAMA_PS));
        let empty = ServerStats::from_bodies(None, None, Some(r#"{"models":[]}"#));

        assert_eq!(
            remote_table(Some(&stats)).notes[0].text,
            "loaded qwen2.5-coder:7b 5.1 GB · nomic-embed-text:latest"
        );
        assert_eq!(remote_table(Some(&empty)).notes[0].text, "loaded none");
    }

    #[test]
    fn a_server_that_published_nothing_says_so() {
        let table = remote_table(Some(&ServerStats::default()));

        assert_eq!(table.notes[0].name, "10.0.0.5");
        assert_eq!(table.notes[0].text, NO_SERVER_STATS);
    }

    #[test]
    fn the_server_waits_until_a_poll_has_finished() {
        let table = remote_table(None);

        assert_eq!(table.notes[0].text, WAITING_FOR_SERVER_STATS);
        assert_eq!(table.rows.len(), 1, "only the local row");
    }

    #[test]
    fn a_local_server_gives_only_the_local_row_and_note() {
        let stats = SystemStats {
            gpu_name: Some("Thor".to_string()),
            gpu_util_percent: Some(42.0),
            gpu_memory_used_mb: Some(1024),
            gpu_memory_total_mb: Some(2048),
            gpu_power_watts: Some(15.5),
            gpu_temp_c: Some(50.0),
            cpu_util_percent: Some(12.0),
            cpu_cores: 14,
            memory_used_mb: 2048,
            memory_total_mb: 4096,
            memory_free_mb: 1024,
            ..SystemStats::default()
        };

        let table = StatsTable::new(None, &stats);

        assert_eq!(
            table.rows,
            [MachineRow {
                name: "local".to_string(),
                figures: ["Thor", "42%", "15.5 W", "50 °C", "2.0/4.0 GB", "12%", "14"]
                    .map(str::to_string),
            }]
        );
        assert_eq!(
            table.notes[0].text,
            "VRAM   1.0/  2.0 GB · RAM free   1.0 GB · cache   0.0 GB · buffers   0.0 GB · shared   0.0 GB"
        );
    }

    #[test]
    fn the_columns_keep_their_place_when_the_figures_change() {
        let quiet = SystemStats {
            gpu_util_percent: Some(5.0),
            gpu_power_watts: Some(4.8),
            gpu_temp_c: Some(9.0),
            cpu_util_percent: Some(1.0),
            memory_used_mb: 1024,
            memory_total_mb: 2048,
            ..SystemStats::default()
        };
        let busy = SystemStats {
            gpu_util_percent: Some(100.0),
            gpu_power_watts: Some(123.4),
            gpu_temp_c: Some(88.0),
            cpu_util_percent: Some(100.0),
            memory_used_mb: 900_000,
            memory_total_mb: 1_000_000,
            ..SystemStats::default()
        };
        let drawn = |stats: &SystemStats| -> Vec<String> {
            StatsTable::new(None, stats)
                .lines()
                .iter()
                .map(|line| line.to_string())
                .collect()
        };

        let quiet_lines = drawn(&quiet);
        let busy_lines = drawn(&busy);

        assert_eq!(quiet_lines[0], busy_lines[0], "the headings stay put");
        assert_eq!(quiet_lines[1].len(), busy_lines[1].len());
        assert_eq!(quiet_lines[2].len(), busy_lines[2].len());
    }

    #[test]
    fn a_machine_without_a_gpu_reading_gets_dashes() {
        let table = StatsTable::new(None, &SystemStats::default());

        assert_eq!(
            table.rows[0].figures,
            ["–", "–", "–", "–", "0.0/0.0 GB", "–", "0"]
        );
        assert!(table.notes[0].text.starts_with("RAM free"));
    }
}
