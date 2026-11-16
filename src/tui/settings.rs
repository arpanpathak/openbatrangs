//! # The settings dialog
//!
//! `/settings` opens a dialog with the two settings worth changing during a
//! session: the sampling temperature, as a slider from 0 to 2 in steps of
//! 0.1, and whether the model thinks before it answers. ↑ and ↓ choose a
//! setting, ← and → change it, `a` sets the temperature back to the model
//! card's, and Esc or Enter closes the dialog. A change applies from the next
//! request on.
//!
//! The mouse works too. A left click on the thinking row chooses it and turns
//! thinking off or on; a left click on the temperature row chooses it, and on
//! the slider track also moves the knob to the clicked cell. The wheel over
//! the dialog chooses the row above or below. Mouse events outside the dialog
//! do nothing while it is open.

use super::app::{ActiveModal, App};
use crate::sampling::Sampling;
use crossterm::event::{KeyCode, KeyEvent, MouseButton, MouseEvent, MouseEventKind};
use ratatui::layout::{Position, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, BorderType, Borders, Clear, Paragraph};

/// A row of the settings dialog.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum SettingsRow {
    /// The sampling temperature slider.
    Temperature,
    /// Thinking on or off.
    Thinking,
}

impl SettingsRow {
    /// The other row; the dialog has two.
    fn other(self) -> Self {
        match self {
            Self::Temperature => Self::Thinking,
            Self::Thinking => Self::Temperature,
        }
    }
}

/// What a key does in the settings dialog.
#[derive(Debug, PartialEq, Eq)]
enum SettingsKey {
    /// ↑ or ↓: choose the other setting.
    Switch,
    /// ←: lower the temperature, or turn thinking off or on.
    Lower,
    /// →: raise the temperature, or turn thinking off or on.
    Raise,
    /// `a`: use the model card's temperature.
    Auto,
    /// Esc or Enter: close the dialog.
    Close,
    /// Any other key.
    Nothing,
}

/// Highest temperature the slider reaches, the most OpenAI-compatible servers
/// accept.
const MAX_TEMPERATURE: f64 = 2.0;

/// Change of one slider step.
const TEMPERATURE_STEP: f64 = 0.1;

/// Where the slider starts for a model whose card is unknown: 1.0, the
/// default of OpenAI-compatible servers when a request sends no temperature.
const SERVER_TEMPERATURE: f64 = 1.0;

/// Cells of the slider track, one per step.
const SLIDER_CELLS: usize = 20;

/// Width and height of the dialog, borders included.
const DIALOG_SIZE: (u16, u16) = (60, 8);

/// Line of the temperature row, counted from the dialog's top border.
const TEMPERATURE_LINE: u16 = 2;

/// Line of the thinking row, counted from the dialog's top border.
const THINKING_LINE: u16 = 3;

/// Column of the slider's first cell, counted from the dialog's left border:
/// the border, then the 16-cell label.
const SLIDER_COLUMN: u16 = 17;

/// The action of `key` in the dialog.
fn settings_key(key: &KeyEvent) -> SettingsKey {
    match key.code {
        KeyCode::Up | KeyCode::Down | KeyCode::Tab => SettingsKey::Switch,
        KeyCode::Left => SettingsKey::Lower,
        KeyCode::Right => SettingsKey::Raise,
        KeyCode::Char('a') => SettingsKey::Auto,
        KeyCode::Esc | KeyCode::Enter => SettingsKey::Close,
        _ => SettingsKey::Nothing,
    }
}

impl App {
    /// Open the settings dialog on the temperature row.
    pub(super) fn open_settings(&mut self) {
        self.modal = Some(ActiveModal::Settings(SettingsRow::Temperature));
    }

    /// Apply a key pressed while the settings dialog is open.
    pub(super) fn handle_settings_key(&mut self, key: KeyEvent) {
        match settings_key(&key) {
            SettingsKey::Switch => self.switch_settings_row(),
            SettingsKey::Lower => self.change_setting(-TEMPERATURE_STEP),
            SettingsKey::Raise => self.change_setting(TEMPERATURE_STEP),
            SettingsKey::Auto => self.run_config.temperature = None,
            SettingsKey::Close => self.modal = None,
            SettingsKey::Nothing => {}
        }
    }

    /// Apply a mouse event while the settings dialog is open: a left click
    /// chooses and changes the row under it, and the wheel chooses the row
    /// above or below. Events outside the dialog do nothing.
    pub(super) fn handle_settings_mouse(&mut self, event: MouseEvent) {
        let area = self.last_settings_area.unwrap_or_default();
        let cell = Position::new(event.column, event.row);

        match (area.contains(cell), event.kind) {
            (false, _) => {}
            (true, MouseEventKind::ScrollUp) => self.choose_settings_row(SettingsRow::Temperature),
            (true, MouseEventKind::ScrollDown) => self.choose_settings_row(SettingsRow::Thinking),
            (true, MouseEventKind::Down(MouseButton::Left)) => self.click_settings(area, cell),
            (true, _) => {}
        }
    }

    /// Choose and change the row under `cell` of the dialog drawn in `area`:
    /// flip thinking, or move the temperature to the slider cell clicked.
    fn click_settings(&mut self, area: Rect, cell: Position) {
        match cell.y - area.y {
            TEMPERATURE_LINE => self.click_temperature(cell.x - area.x),
            THINKING_LINE => {
                self.choose_settings_row(SettingsRow::Thinking);
                self.run_config.think = !self.run_config.think;
            }
            _ => {}
        }
    }

    /// Choose the temperature row and, when `column` of the dialog is on the
    /// slider track, set the temperature that puts the knob there.
    fn click_temperature(&mut self, column: u16) {
        self.choose_settings_row(SettingsRow::Temperature);

        let Some(knob) = column.checked_sub(SLIDER_COLUMN) else {
            return;
        };

        if usize::from(knob) <= SLIDER_CELLS {
            self.run_config.temperature =
                Some(f64::from(knob) * MAX_TEMPERATURE / SLIDER_CELLS as f64);
        }
    }

    /// Choose `row` in the open dialog.
    fn choose_settings_row(&mut self, row: SettingsRow) {
        self.modal = Some(ActiveModal::Settings(row));
    }

    /// The row the dialog has chosen, or `None` when it is closed.
    fn settings_row(&self) -> Option<SettingsRow> {
        match self.modal {
            Some(ActiveModal::Settings(row)) => Some(row),
            _ => None,
        }
    }

    /// Choose the other row.
    fn switch_settings_row(&mut self) {
        if let Some(row) = self.settings_row() {
            self.modal = Some(ActiveModal::Settings(row.other()));
        }
    }

    /// Move the temperature by `step`, or flip thinking.
    fn change_setting(&mut self, step: f64) {
        match self.settings_row() {
            Some(SettingsRow::Temperature) => self.step_temperature(step),
            Some(SettingsRow::Thinking) => self.run_config.think = !self.run_config.think,
            None => {}
        }
    }

    /// Move the temperature by `step`, starting from the one in use, within
    /// 0 to [`MAX_TEMPERATURE`] and rounded to a tenth.
    fn step_temperature(&mut self, step: f64) {
        let start = self.temperature_in_use().unwrap_or(SERVER_TEMPERATURE);
        let next = (start + step).clamp(0.0, MAX_TEMPERATURE);

        self.run_config.temperature = Some((next * 10.0).round() / 10.0);
    }

    /// The temperature the next request sends: the user's, else the model
    /// card's for the selected model, or `None` when it sends none and the
    /// server uses the model's default.
    pub(super) fn temperature_in_use(&self) -> Option<f64> {
        let model = self.model.as_deref().unwrap_or_default();

        Sampling::for_model(model, self.run_config.temperature).temperature
    }
}

/// Draw the settings dialog in the middle of the screen while it is open,
/// and record where it went for mouse clicks.
pub(super) fn render_settings(frame: &mut ratatui::Frame, app: &mut App) {
    let Some(row) = app.settings_row() else {
        return;
    };

    let area = centered(frame.area(), DIALOG_SIZE);
    app.last_settings_area = Some(area);
    let lines = vec![
        Line::default(),
        temperature_line(app, row == SettingsRow::Temperature),
        thinking_line(app.run_config.think, row == SettingsRow::Thinking),
        Line::default(),
        Line::styled(
            "  ↑↓ choose · ←→ change · a model card · Esc close",
            Style::default().fg(Color::DarkGray),
        ),
    ];
    let block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(Color::Cyan))
        .title(" settings ");

    frame.render_widget(Clear, area);
    frame.render_widget(Paragraph::new(lines).block(block), area);
}

/// The temperature row: the slider, then the value, marked `model card`
/// while the user has not set one. A model without a known card shows
/// `server default` and the slider at [`SERVER_TEMPERATURE`].
fn temperature_line(app: &App, is_chosen: bool) -> Line<'static> {
    let in_use = app.temperature_in_use();
    let value = in_use.map_or_else(String::new, |temperature| format!("  {temperature:.1}"));
    let source = match (app.run_config.temperature, in_use) {
        (Some(_), _) => "",
        (None, Some(_)) => "  model card",
        (None, None) => "  server default",
    };
    let knob = in_use.unwrap_or(SERVER_TEMPERATURE);

    Line::from(vec![
        label("Temperature", is_chosen),
        Span::styled(slider(knob), Style::default().fg(Color::Cyan)),
        Span::styled(value, Style::default().fg(Color::White)),
        Span::styled(source, Style::default().fg(Color::DarkGray)),
    ])
}

/// The thinking row: `● on  ○ off`, with the chosen state filled.
fn thinking_line(think: bool, is_chosen: bool) -> Line<'static> {
    let (on, off) = match think {
        true => ("● on", "○ off"),
        false => ("○ on", "● off"),
    };

    Line::from(vec![
        label("Thinking", is_chosen),
        Span::styled(format!("{on}   {off}"), Style::default().fg(Color::White)),
    ])
}

/// A row's label, after a `›` marker when the row is chosen.
fn label(name: &str, is_chosen: bool) -> Span<'static> {
    let (marker, style) = match is_chosen {
        true => (
            "›",
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        ),
        false => (" ", Style::default().fg(Color::Gray)),
    };

    Span::styled(format!(" {marker} {name:<13}"), style)
}

/// The slider for `temperature`: a filled track up to the knob, then an
/// empty one, [`SLIDER_CELLS`] cells long.
fn slider(temperature: f64) -> String {
    let knob = (temperature / MAX_TEMPERATURE * SLIDER_CELLS as f64).round() as usize;
    let knob = knob.min(SLIDER_CELLS);

    format!("{}●{}", "━".repeat(knob), "─".repeat(SLIDER_CELLS - knob))
}

/// A rectangle of `size` in the middle of `area`, cut to fit.
fn centered(area: Rect, (width, height): (u16, u16)) -> Rect {
    let width = width.min(area.width);
    let height = height.min(area.height);

    Rect {
        x: area.x + (area.width - width) / 2,
        y: area.y + (area.height - height) / 2,
        width,
        height,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli::Cli;
    use clap::Parser;
    use crossterm::event::KeyModifiers;
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;
    use std::sync::{Arc, Mutex};

    /// An app with default settings and the model `model`.
    fn app_with_model(model: &str) -> App {
        let cli = Cli::parse_from(["openbatrangs"]);
        let mut app = App::new(&cli, Arc::new(Mutex::new(None)));
        app.model = Some(model.to_string());

        app
    }

    /// Press `code` in the settings dialog.
    fn press(app: &mut App, code: KeyCode) {
        app.handle_settings_key(KeyEvent::new(code, KeyModifiers::NONE));
    }

    #[test]
    fn the_slider_starts_at_the_model_card_and_moves_in_tenths() {
        let mut app = app_with_model("gemma-4-26b");
        app.open_settings();

        assert_eq!(app.temperature_in_use(), Some(1.0));

        press(&mut app, KeyCode::Left);
        press(&mut app, KeyCode::Left);

        assert_eq!(app.run_config.temperature, Some(0.8));

        press(&mut app, KeyCode::Char('a'));

        assert_eq!(app.run_config.temperature, None);
    }

    #[test]
    fn the_temperature_stays_between_zero_and_two() {
        let mut app = app_with_model("gemma-4-26b");
        app.open_settings();

        for _ in 0..30 {
            press(&mut app, KeyCode::Right);
        }
        assert_eq!(app.run_config.temperature, Some(2.0));

        for _ in 0..30 {
            press(&mut app, KeyCode::Left);
        }
        assert_eq!(app.run_config.temperature, Some(0.0));
    }

    #[test]
    fn the_thinking_row_flips_thinking_and_esc_closes() {
        let mut app = app_with_model("gemma-4-26b");
        app.open_settings();
        let was_thinking = app.run_config.think;

        press(&mut app, KeyCode::Down);
        press(&mut app, KeyCode::Right);

        assert_eq!(app.run_config.think, !was_thinking);
        assert_eq!(
            app.run_config.temperature, None,
            "the temperature is untouched"
        );

        press(&mut app, KeyCode::Up);
        press(&mut app, KeyCode::F(1));
        assert_eq!(app.settings_row(), Some(SettingsRow::Temperature));

        press(&mut app, KeyCode::Esc);
        assert_eq!(app.settings_row(), None);

        press(&mut app, KeyCode::Right);
        press(&mut app, KeyCode::Down);
        assert_eq!(
            app.run_config.temperature, None,
            "a closed dialog changes nothing"
        );
        assert_eq!(app.settings_row(), None, "and stays closed");
    }

    #[test]
    fn the_slider_knob_sits_at_the_temperature() {
        assert_eq!(slider(0.0), format!("●{}", "─".repeat(20)));
        assert_eq!(
            slider(1.0),
            format!("{}●{}", "━".repeat(10), "─".repeat(10))
        );
        assert_eq!(slider(2.0), format!("{}●", "━".repeat(20)));
    }

    #[test]
    fn the_dialog_shows_the_slider_the_value_and_thinking() -> anyhow::Result<()> {
        let mut app = app_with_model("gemma-4-26b");
        let mut terminal = Terminal::new(TestBackend::new(80, 24))?;

        terminal.draw(|frame| render_settings(frame, &mut app))?;
        let closed: String = screen(&terminal);

        app.open_settings();
        terminal.draw(|frame| render_settings(frame, &mut app))?;
        let open = screen(&terminal);

        assert!(!closed.contains("settings"));
        assert!(open.contains("› Temperature"));
        assert!(open.contains("1.0  model card"));
        assert!(open.contains("● on   ○ off"));

        app.run_config.think = false;
        app.run_config.temperature = Some(0.4);
        terminal.draw(|frame| render_settings(frame, &mut app))?;
        let changed = screen(&terminal);

        assert!(changed.contains("○ on   ● off"));
        assert!(changed.contains("0.4"));
        assert!(
            !changed.contains("0.4  model card"),
            "a value the user set is not labeled as the model card's"
        );

        Ok(())
    }

    #[test]
    fn a_model_without_a_card_shows_the_server_default() -> anyhow::Result<()> {
        let mut app = app_with_model("llama3.2");
        let mut terminal = Terminal::new(TestBackend::new(80, 24))?;
        app.open_settings();

        terminal.draw(|frame| render_settings(frame, &mut app))?;
        let open = screen(&terminal);

        assert_eq!(app.temperature_in_use(), None);
        assert!(open.contains(&format!("{}  server default", slider(1.0))));

        press(&mut app, KeyCode::Right);

        assert_eq!(app.run_config.temperature, Some(1.1), "steps from 1.0");
        Ok(())
    }

    /// A mouse event of `kind` at `column`, `row` of the screen.
    fn mouse(app: &mut App, kind: MouseEventKind, column: u16, row: u16) {
        app.handle_mouse(MouseEvent {
            kind,
            column,
            row,
            modifiers: KeyModifiers::NONE,
        });
    }

    /// An app with the settings dialog open and drawn on an 80 by 24
    /// screen, where the dialog spans columns 10 to 69 and rows 8 to 15:
    /// the temperature row is screen row 10, the thinking row 11, and the
    /// slider track starts at column 27.
    fn app_with_drawn_dialog() -> anyhow::Result<App> {
        let mut app = app_with_model("gemma-4-26b");
        let mut terminal = Terminal::new(TestBackend::new(80, 24))?;

        app.open_settings();
        terminal.draw(|frame| render_settings(frame, &mut app))?;

        Ok(app)
    }

    #[test]
    fn a_click_on_the_thinking_row_chooses_it_and_flips_thinking() -> anyhow::Result<()> {
        let mut app = app_with_drawn_dialog()?;
        let was_thinking = app.run_config.think;

        mouse(&mut app, MouseEventKind::Down(MouseButton::Left), 40, 11);

        assert_eq!(app.settings_row(), Some(SettingsRow::Thinking));
        assert_eq!(app.run_config.think, !was_thinking);

        mouse(&mut app, MouseEventKind::Up(MouseButton::Left), 40, 11);
        mouse(&mut app, MouseEventKind::Down(MouseButton::Left), 12, 11);

        assert_eq!(
            app.run_config.think, was_thinking,
            "a second click flips it back"
        );
        assert_eq!(
            app.run_config.temperature, None,
            "the temperature is untouched"
        );

        Ok(())
    }

    #[test]
    fn a_click_on_the_slider_track_moves_the_knob_there() -> anyhow::Result<()> {
        let mut app = app_with_drawn_dialog()?;
        app.modal = Some(ActiveModal::Settings(SettingsRow::Thinking));

        mouse(&mut app, MouseEventKind::Down(MouseButton::Left), 14, 10);

        assert_eq!(app.settings_row(), Some(SettingsRow::Temperature));
        assert_eq!(
            app.run_config.temperature, None,
            "a click on the label only chooses the row"
        );

        mouse(&mut app, MouseEventKind::Down(MouseButton::Left), 27, 10);
        assert_eq!(app.run_config.temperature, Some(0.0));

        mouse(&mut app, MouseEventKind::Down(MouseButton::Left), 30, 10);
        assert_eq!(app.run_config.temperature, Some(0.3));

        mouse(&mut app, MouseEventKind::Down(MouseButton::Left), 47, 10);
        assert_eq!(app.run_config.temperature, Some(2.0));

        mouse(&mut app, MouseEventKind::Down(MouseButton::Left), 50, 10);
        assert_eq!(
            app.run_config.temperature,
            Some(2.0),
            "a click right of the track changes nothing"
        );

        Ok(())
    }

    #[test]
    fn the_wheel_chooses_a_row_and_the_mouse_outside_does_nothing() -> anyhow::Result<()> {
        let mut app = app_with_drawn_dialog()?;
        for index in 0..100 {
            app.notice(format!("line {index}"));
        }
        app.last_chat_area = Some(Rect::new(0, 0, 80, 20));
        let was_thinking = app.run_config.think;

        mouse(&mut app, MouseEventKind::ScrollDown, 40, 9);
        assert_eq!(app.settings_row(), Some(SettingsRow::Thinking));

        mouse(&mut app, MouseEventKind::ScrollUp, 40, 14);
        assert_eq!(app.settings_row(), Some(SettingsRow::Temperature));

        mouse(&mut app, MouseEventKind::Down(MouseButton::Left), 40, 8);
        mouse(&mut app, MouseEventKind::Moved, 40, 11);
        mouse(&mut app, MouseEventKind::ScrollDown, 5, 5);
        mouse(&mut app, MouseEventKind::Down(MouseButton::Left), 5, 11);

        assert_eq!(app.settings_row(), Some(SettingsRow::Temperature));
        assert_eq!(app.run_config.think, was_thinking);
        assert_eq!(app.run_config.temperature, None);
        assert_eq!(app.chat_scroll_offset, 0, "the chat behind does not scroll");
        assert!(app.press.is_none(), "and no click starts in it");

        Ok(())
    }

    /// The text of every cell on `terminal`, row after row.
    fn screen(terminal: &Terminal<TestBackend>) -> String {
        terminal
            .backend()
            .buffer()
            .content()
            .iter()
            .map(|cell| cell.symbol())
            .collect()
    }
}
