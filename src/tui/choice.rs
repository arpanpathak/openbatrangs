//! # Questions with answers to choose from
//!
//! When the model asks a question with options, a dialog lists them: radio
//! buttons when one answer is wanted, checkboxes when several may be picked.
//! ↑ and ↓ move, Space ticks a checkbox, and Enter sends the choice as the
//! user's reply, which resumes the run. Esc closes the dialog so the user can
//! type an answer of their own in the input box.

use super::UiEvent;
use super::app::{ActiveModal, App};
use crate::agent::Question;
use crate::ollama::OllamaClient;
use crossterm::event::{KeyCode, KeyEvent};
use ratatui::layout::Rect;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, BorderType, Borders, Clear, Paragraph, Wrap};
use tokio::sync::mpsc;
use unicode_width::UnicodeWidthStr;

/// The open dialog: the question, the highlighted option, and the ticked
/// ones.
pub(super) struct ChoiceState {
    /// The question and its options.
    question: Question,
    /// Index of the highlighted option.
    cursor: usize,
    /// Whether each option is ticked; used when several may be picked.
    ticked: Vec<bool>,
}

/// What a key does in the dialog.
#[derive(Debug, PartialEq, Eq)]
enum ChoiceKey {
    /// ↑: highlight the option above.
    Up,
    /// ↓: highlight the option below.
    Down,
    /// Space: tick or untick the highlighted option.
    Tick,
    /// Enter: send the choice.
    Send,
    /// Esc: close the dialog to type an answer instead.
    Dismiss,
    /// Any other key.
    Nothing,
}

/// Widest the dialog gets, borders included.
const DIALOG_WIDTH: u16 = 72;

/// The action of `key` in the dialog.
fn choice_key(key: &KeyEvent) -> ChoiceKey {
    match key.code {
        KeyCode::Up => ChoiceKey::Up,
        KeyCode::Down => ChoiceKey::Down,
        KeyCode::Char(' ') => ChoiceKey::Tick,
        KeyCode::Enter => ChoiceKey::Send,
        KeyCode::Esc => ChoiceKey::Dismiss,
        _ => ChoiceKey::Nothing,
    }
}

impl ChoiceState {
    /// A dialog for `question` with the first option highlighted.
    pub(super) fn new(question: Question) -> Self {
        let ticked = vec![false; question.options.len()];

        Self {
            question,
            cursor: 0,
            ticked,
        }
    }

    /// Highlight the option `step` places away, within the list.
    fn move_by(&mut self, step: isize) {
        let last = self.question.options.len().saturating_sub(1);

        self.cursor = self.cursor.saturating_add_signed(step).min(last);
    }

    /// Tick or untick the highlighted option, when several may be picked.
    fn tick(&mut self) {
        if !self.question.is_multiple {
            return;
        }

        if let Some(ticked) = self.ticked.get_mut(self.cursor) {
            *ticked = !*ticked;
        }
    }

    /// The reply: the highlighted option, or the ticked options joined with
    /// commas, or the highlighted one when none is ticked.
    fn answer(&self) -> String {
        let ticked: Vec<&str> = self
            .question
            .options
            .iter()
            .zip(&self.ticked)
            .filter(|(_option, is_ticked)| **is_ticked)
            .map(|(option, _is_ticked)| option.as_str())
            .collect();

        match (self.question.is_multiple, ticked.is_empty()) {
            (true, false) => ticked.join(", "),
            (true, true) | (false, _) => self.highlighted().to_string(),
        }
    }

    /// The highlighted option.
    fn highlighted(&self) -> &str {
        self.question
            .options
            .get(self.cursor)
            .map_or("", String::as_str)
    }
}

impl App {
    /// Open the dialog for `question` when it has options to choose from.
    pub(super) fn open_choice(&mut self, question: &Question) {
        if question.options.is_empty() {
            return;
        }

        self.modal = Some(ActiveModal::Choice(ChoiceState::new(question.clone())));
    }

    /// The open dialog, or `None` when it is closed.
    fn choice_mut(&mut self) -> Option<&mut ChoiceState> {
        match &mut self.modal {
            Some(ActiveModal::Choice(state)) => Some(state),
            _ => None,
        }
    }

    /// Apply a key pressed while the dialog is open.
    pub(super) async fn handle_choice_key(
        &mut self,
        key: KeyEvent,
        client: &OllamaClient,
        tx: &mpsc::UnboundedSender<UiEvent>,
    ) {
        match choice_key(&key) {
            ChoiceKey::Up => self.update_choice(|state| state.move_by(-1)),
            ChoiceKey::Down => self.update_choice(|state| state.move_by(1)),
            ChoiceKey::Tick => self.update_choice(ChoiceState::tick),
            ChoiceKey::Send => self.send_choice(client, tx).await,
            ChoiceKey::Dismiss => self.modal = None,
            ChoiceKey::Nothing => {}
        }
    }

    /// Apply `change` to the open dialog.
    fn update_choice(&mut self, change: impl FnOnce(&mut ChoiceState)) {
        if let Some(state) = self.choice_mut() {
            change(state);
        }
    }

    /// Close the dialog and send the choice as the user's reply.
    async fn send_choice(&mut self, client: &OllamaClient, tx: &mpsc::UnboundedSender<UiEvent>) {
        let Some(answer) = self.choice_mut().map(|state| state.answer()) else {
            return;
        };

        self.modal = None;
        self.input = answer;
        self.cursor = self.input.len();
        self.submit_input(client, tx).await;
    }
}

/// Draw the dialog in the middle of the screen while it is open.
pub(super) fn render_choice(frame: &mut ratatui::Frame, app: &App) {
    let Some(ActiveModal::Choice(state)) = &app.modal else {
        return;
    };

    let width = DIALOG_WIDTH.min(frame.area().width);
    let inner_width = usize::from(width.saturating_sub(4)).max(1);
    let question_rows = state.question.text.width() / inner_width + 1;
    let height = (question_rows + state.question.options.len() + 6) as u16;
    let area = centered(frame.area(), width, height);

    let mut lines = vec![Line::styled(
        state.question.text.clone(),
        Style::default().add_modifier(Modifier::BOLD),
    )];
    lines.push(Line::default());
    lines.extend(
        state
            .question
            .options
            .iter()
            .enumerate()
            .map(|(index, option)| option_line(state, index, option)),
    );
    lines.push(Line::default());
    lines.push(Line::styled(
        keys_hint(state.question.is_multiple),
        Style::default().fg(Color::DarkGray),
    ));

    let block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(Color::Cyan))
        .title(" ❓ question ");

    frame.render_widget(Clear, area);
    frame.render_widget(
        Paragraph::new(lines)
            .wrap(Wrap { trim: false })
            .block(block),
        area,
    );
}

/// One option: a `›` on the highlighted one, then a radio button or a
/// checkbox, then the option.
fn option_line(state: &ChoiceState, index: usize, option: &str) -> Line<'static> {
    let is_highlighted = index == state.cursor;
    let is_ticked = state.ticked.get(index).copied().unwrap_or(false);
    let mark = match (state.question.is_multiple, is_ticked, is_highlighted) {
        (true, true, _) => "[x]",
        (true, false, _) => "[ ]",
        (false, _, true) => "(•)",
        (false, _, false) => "( )",
    };
    let (pointer, style) = match is_highlighted {
        true => (
            "›",
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        ),
        false => (" ", Style::default().fg(Color::White)),
    };

    Line::from(vec![Span::styled(
        format!("{pointer} {mark} {option}"),
        style,
    )])
}

/// The keys that work in the dialog.
fn keys_hint(is_multiple: bool) -> &'static str {
    match is_multiple {
        true => "↑↓ move · Space tick · Enter answer · Esc type your own",
        false => "↑↓ choose · Enter answer · Esc type your own",
    }
}

/// A rectangle of `width` by `height` in the middle of `area`, cut to fit.
fn centered(area: Rect, width: u16, height: u16) -> Rect {
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

    /// A question with three options; several may be picked when
    /// `is_multiple`.
    fn question(is_multiple: bool) -> Question {
        Question {
            text: "Which crates should the example use?".to_string(),
            options: ["tokio", "serde", "anyhow"].map(str::to_string).to_vec(),
            is_multiple,
        }
    }

    /// An app with the dialog open for `question`.
    fn app_asking(question: &Question) -> App {
        let cli = Cli::parse_from(["openbatrangs"]);
        let mut app = App::new(&cli, Arc::new(Mutex::new(None)));
        app.open_choice(question);

        app
    }

    /// Press `code` in the dialog.
    async fn press(app: &mut App, code: KeyCode) -> anyhow::Result<()> {
        let client = OllamaClient::new("http://127.0.0.1:9")?;
        let (tx, _rx) = mpsc::unbounded_channel();

        app.handle_choice_key(KeyEvent::new(code, KeyModifiers::NONE), &client, &tx)
            .await;
        Ok(())
    }

    #[test]
    fn a_question_without_options_opens_no_dialog() {
        let free_text = Question {
            text: "What should the file be called?".to_string(),
            ..Question::default()
        };

        let app = app_asking(&free_text);

        assert!(app.modal.is_none());
    }

    #[test]
    fn one_choice_answers_with_the_highlighted_option() {
        let mut state = ChoiceState::new(question(false));

        state.move_by(1);
        state.tick();
        state.move_by(5);

        assert_eq!(state.answer(), "anyhow");

        state.move_by(-9);

        assert_eq!(state.answer(), "tokio");
    }

    #[test]
    fn several_choices_answer_with_the_ticked_options() {
        let mut state = ChoiceState::new(question(true));

        assert_eq!(state.answer(), "tokio", "nothing ticked yet");

        state.tick();
        state.move_by(2);
        state.tick();

        assert_eq!(state.answer(), "tokio, anyhow");
    }

    #[tokio::test]
    async fn enter_sends_the_choice_as_the_reply() -> anyhow::Result<()> {
        let mut app = app_asking(&question(false));

        press(&mut app, KeyCode::Down).await?;
        press(&mut app, KeyCode::F(2)).await?;
        press(&mut app, KeyCode::Enter).await?;

        assert!(app.modal.is_none());
        assert_eq!(app.history.last().map(String::as_str), Some("serde"));
        Ok(())
    }

    #[tokio::test]
    async fn esc_closes_the_dialog_without_answering() -> anyhow::Result<()> {
        let mut app = app_asking(&question(true));

        press(&mut app, KeyCode::Char(' ')).await?;
        press(&mut app, KeyCode::Up).await?;
        press(&mut app, KeyCode::Esc).await?;
        press(&mut app, KeyCode::Enter).await?;

        assert!(app.modal.is_none());
        assert!(app.history.is_empty());
        Ok(())
    }

    #[test]
    fn the_dialog_shows_radio_buttons_or_checkboxes() -> anyhow::Result<()> {
        let mut terminal = Terminal::new(TestBackend::new(80, 24))?;
        let draw = |terminal: &mut Terminal<TestBackend>, app: &App| -> anyhow::Result<String> {
            terminal.draw(|frame| render_choice(frame, app))?;
            Ok(terminal
                .backend()
                .buffer()
                .content()
                .iter()
                .map(|cell| cell.symbol())
                .collect())
        };

        let one = draw(&mut terminal, &app_asking(&question(false)))?;
        let mut several = app_asking(&question(true));
        several.update_choice(ChoiceState::tick);
        let many = draw(&mut terminal, &several)?;

        assert!(one.contains("› (•) tokio"));
        assert!(one.contains("( ) serde"));
        assert!(one.contains("Enter answer"));
        assert!(many.contains("› [x] tokio"));
        assert!(many.contains("[ ] serde"));
        assert!(many.contains("Space tick"));
        Ok(())
    }
}
