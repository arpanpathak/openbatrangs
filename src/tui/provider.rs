//! # Choosing and adding providers
//!
//! The provider picker lists the providers in the config file, then
//! "Add a provider…", and "Local Ollama" when the file holds none. It opens
//! at launch when no flag names the server and no provider is chosen, and
//! with `/provider`. It is the model picker with other rows.
//!
//! Adding a provider asks for its kind in the same picker: a built-in
//! provider, or a server of your own. A server of your own then needs a
//! name and a URL, typed on the input line. Last comes the API key, which
//! the input line shows as dots. The new table is written to the config
//! file, and the provider is used at once.

use super::app::{App, PickerPurpose, ProviderChoice};
use crate::cli::{Cli, ConfigFile, add_provider, names_server, save_settings};
use crate::constants::cli::{BUILT_IN_PROVIDERS, DEFAULT_OLLAMA_URL};
use crate::ollama::OllamaClient;
use std::borrow::Cow;

/// The step of adding a provider that the next input line answers.
#[derive(Debug, PartialEq, Eq)]
pub(super) enum ProviderForm {
    /// The name of a server of your own.
    Name,
    /// The URL of the server called `name`.
    Url {
        /// The name typed in the step before.
        name: String,
    },
    /// The API key of the provider `name`.
    Key {
        /// Provider name.
        name: String,
        /// Server address; `None` for a built-in provider.
        url: Option<String>,
    },
}

impl App {
    /// At launch, open the provider picker unless a flag names the server
    /// or a provider is already chosen.
    pub(super) fn offer_providers(&mut self, cli: &Cli) {
        if self.provider.is_some() || names_server(cli) {
            return;
        }

        self.open_provider_picker();
    }

    /// Open the provider picker on the providers in the config file, or
    /// show why the file cannot be read.
    pub(super) fn open_provider_picker(&mut self) {
        match ConfigFile::load(&self.config_path) {
            Ok(config) => self.open_provider_rows(config),
            Err(error) => self.notice(format!("⚠️ {error:#}")),
        }
    }

    /// Open the picker with a row per provider in `config`, then the row
    /// that adds one, and the local Ollama when `config` holds none.
    fn open_provider_rows(&mut self, config: ConfigFile) {
        let width = config.providers.keys().map(String::len).max().unwrap_or(0);
        let has_providers = !config.providers.is_empty();
        let mut rows = Vec::new();
        let mut choices = Vec::new();

        for (name, provider) in config.providers {
            rows.push(format!("{name:width$}  {}", provider.url));
            choices.push(ProviderChoice::Named(name));
        }

        rows.push("Add a provider…".to_string());
        choices.push(ProviderChoice::Add);

        if !has_providers {
            rows.push(format!("Local Ollama ({DEFAULT_OLLAMA_URL})"));
            choices.push(ProviderChoice::LocalOllama);
        }

        self.open_picker_for(rows, PickerPurpose::Provider(choices));
    }

    /// Open the picker of what to add: each built-in provider, then a
    /// server of your own.
    fn open_provider_kinds(&mut self) {
        let mut rows = Vec::new();
        let mut choices = Vec::new();

        for (name, label, _url) in BUILT_IN_PROVIDERS {
            rows.push(label.to_string());
            choices.push(ProviderChoice::BuiltIn(name));
        }

        rows.push("Your own server".to_string());
        choices.push(ProviderChoice::Own);

        self.open_picker_for(rows, PickerPurpose::Provider(choices));
    }

    /// Act on the row `selected` of the provider picker with `choices`.
    pub(super) async fn choose_provider(&mut self, choices: Vec<ProviderChoice>, selected: usize) {
        let Some(choice) = choices.into_iter().nth(selected) else {
            return;
        };

        match choice {
            ProviderChoice::Named(name) => {
                let outcome = self.switch_provider(&name).await;
                self.report(outcome);
            }
            ProviderChoice::Add => self.open_provider_kinds(),
            ProviderChoice::LocalOllama => {
                let outcome = self.use_local_ollama().await;
                self.report(outcome);
            }
            ProviderChoice::BuiltIn(name) => self.ask_provider_form(ProviderForm::Key {
                name: name.to_string(),
                url: None,
            }),
            ProviderChoice::Own => self.ask_provider_form(ProviderForm::Name),
        }
    }

    /// Show the question of `step`; the next input line answers it.
    fn ask_provider_form(&mut self, step: ProviderForm) {
        let question = match &step {
            ProviderForm::Name => "Name for the server, then Enter (Ctrl+C cancels):".to_string(),
            ProviderForm::Url { name } => {
                format!("URL of {name}, such as http://192.168.1.20:8000/v1:")
            }
            ProviderForm::Key { name, .. } => {
                format!("API key for {name}, shown as dots (Enter for none):")
            }
        };

        self.notice(question);
        self.provider_form = Some(step);
    }

    /// Take `answer` as the reply to `step`: ask the next question, or after
    /// the key, add the provider and use it.
    pub(super) async fn answer_provider_form(&mut self, step: ProviderForm, answer: String) {
        match step {
            ProviderForm::Name => self.ask_provider_form(ProviderForm::Url { name: answer }),
            ProviderForm::Url { name } => self.ask_provider_form(ProviderForm::Key {
                name,
                url: (!answer.is_empty()).then_some(answer),
            }),
            ProviderForm::Key { name, url } => {
                let outcome = self.add_and_switch(&name, url.as_deref(), &answer).await;
                self.report(outcome);
            }
        }
    }

    /// Write the provider `name` into the config file, then use it. An empty
    /// `key` writes no key.
    async fn add_and_switch(
        &mut self,
        name: &str,
        url: Option<&str>,
        key: &str,
    ) -> anyhow::Result<()> {
        let key = (!key.is_empty()).then_some(key);
        add_provider(&self.config_path, name, url, key)?;

        self.notice(format!("✅ Added {name} to {}", self.config_path.display()));

        self.switch_provider(name).await
    }

    /// Show the error of a failed provider action in the transcript.
    fn report(&mut self, outcome: anyhow::Result<()>) {
        if let Err(error) = outcome {
            self.notice(format!("⚠️ {error:#}"));
        }
    }

    /// Use the provider `name` from the config file for every later task and
    /// command, with its key and its thinking and temperature settings. The
    /// model is chosen again for the new server.
    pub(super) async fn switch_provider(&mut self, name: &str) -> anyhow::Result<()> {
        let mut provider = ConfigFile::load(&self.config_path)?.into_provider(name)?;
        let think = provider.think.unwrap_or(true);
        let key = provider.api_key.take();
        let client = provider.client(key, think)?;

        self.run_config.think = think;
        self.run_config.temperature = provider.temperature;
        self.provider = Some(name.to_string());
        self.notice(format!("✅ Provider {name} ({})", client.base_url));
        self.use_client(client).await;

        Ok(())
    }

    /// Use the local Ollama for every later task and command, without
    /// writing the config file.
    async fn use_local_ollama(&mut self) -> anyhow::Result<()> {
        let client = OllamaClient::new(DEFAULT_OLLAMA_URL)?;

        self.provider = None;
        self.notice(format!("✅ Local Ollama ({DEFAULT_OLLAMA_URL})"));
        self.use_client(client).await;

        Ok(())
    }

    /// Make `client` the client of later tasks and commands, and choose the
    /// model again from its server.
    async fn use_client(&mut self, client: OllamaClient) {
        self.model = None;
        self.model_info = None;
        self.server_url = client.base_url.clone();
        self.refresh_model_info(&client).await;
        self.switched_client = Some(client);
    }

    /// Save the thinking and temperature settings to the table of the
    /// provider in use, so they hold after a restart. Without a provider
    /// they last for the session only.
    pub(super) fn save_provider_settings(&mut self) {
        let Some(name) = &self.provider else {
            return;
        };

        let outcome = save_settings(
            &self.config_path,
            name,
            self.run_config.think,
            self.run_config.temperature,
        );

        self.report(outcome);
    }

    /// The input line as drawn: dots in place of an API key being typed.
    pub(super) fn shown_input(&self) -> Cow<'_, str> {
        let is_secret = matches!(self.provider_form, Some(ProviderForm::Key { .. }));

        match is_secret {
            true => Cow::Owned("•".repeat(self.input.chars().count())),
            false => Cow::Borrowed(&self.input),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{config_file_with, served_model_server};
    use clap::Parser;
    use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
    use std::sync::{Arc, Mutex};
    use tokio::sync::mpsc;

    /// An app whose config file holds `text`.
    fn app_with_config(text: &str) -> anyhow::Result<App> {
        let cli = Cli::parse_from(["openbatrangs"]);
        let mut app = App::new(&cli, Arc::new(Mutex::new(None)));
        app.config_path = config_file_with(text)?;

        Ok(app)
    }

    /// Press each key in `codes`, with `modifiers` held.
    async fn press_with(
        app: &mut App,
        codes: &[KeyCode],
        modifiers: KeyModifiers,
    ) -> anyhow::Result<()> {
        let client = OllamaClient::new("http://127.0.0.1:1")?;
        let (tx, _rx) = mpsc::unbounded_channel();

        for code in codes {
            app.handle_key(KeyEvent::new(*code, modifiers), &client, &tx)
                .await;
        }

        Ok(())
    }

    /// Press each key in `codes`.
    async fn press(app: &mut App, codes: &[KeyCode]) -> anyhow::Result<()> {
        press_with(app, codes, KeyModifiers::NONE).await
    }

    /// Type `text` without pressing Enter.
    async fn type_text(app: &mut App, text: &str) -> anyhow::Result<()> {
        let codes: Vec<KeyCode> = text.chars().map(KeyCode::Char).collect();

        press(app, &codes).await
    }

    /// The rows of the open picker; empty when none is open.
    fn rows(app: &App) -> Vec<String> {
        app.picker()
            .map(|picker| picker.rows.clone())
            .unwrap_or_default()
    }

    #[test]
    fn launch_offers_the_picker_unless_a_server_or_provider_is_named() -> anyhow::Result<()> {
        let mut plain = app_with_config("")?;
        let mut chosen = app_with_config("")?;
        chosen.provider = Some("box".to_string());
        let mut flagged = app_with_config("")?;
        let flag = Cli::parse_from(["openbatrangs", "--ollama-url", "http://box:11434"]);

        plain.offer_providers(&Cli::parse_from(["openbatrangs"]));
        chosen.offer_providers(&Cli::parse_from(["openbatrangs"]));
        flagged.offer_providers(&flag);

        assert_eq!(
            rows(&plain),
            vec![
                "Add a provider…".to_string(),
                format!("Local Ollama ({DEFAULT_OLLAMA_URL})")
            ]
        );
        assert!(!chosen.is_showing_picker());
        assert!(!flagged.is_showing_picker());
        Ok(())
    }

    #[test]
    fn the_picker_lists_the_providers_then_the_add_row() -> anyhow::Result<()> {
        let mut app = app_with_config(
            "[deepseek]\napi_key = \"k\"\n[home-gpu]\nurl = \"http://192.168.1.20:8000/v1\"\n",
        )?;

        app.open_provider_picker();

        assert_eq!(
            rows(&app),
            vec![
                "deepseek  https://api.deepseek.com/v1",
                "home-gpu  http://192.168.1.20:8000/v1",
                "Add a provider…",
            ]
        );
        Ok(())
    }

    #[test]
    fn an_unreadable_config_file_is_shown_instead_of_the_picker() -> anyhow::Result<()> {
        let mut app = app_with_config("[deepseek\n")?;

        app.open_provider_picker();

        assert!(!app.is_showing_picker());
        assert!(app.transcript.text().contains("is not a valid config file"));
        Ok(())
    }

    #[tokio::test]
    async fn a_server_of_your_own_is_added_and_used() -> anyhow::Result<()> {
        let server_url = served_model_server().await;
        let mut app = app_with_config("# my servers\nsearch_engine_url = \"http://search\"\n")?;
        app.open_provider_picker();

        press(&mut app, &[KeyCode::Enter]).await?;
        let kinds = rows(&app);
        press(
            &mut app,
            &[KeyCode::Down, KeyCode::Down, KeyCode::Down, KeyCode::Enter],
        )
        .await?;
        type_text(&mut app, "home-gpu").await?;
        press(&mut app, &[KeyCode::Enter]).await?;
        type_text(&mut app, &server_url).await?;
        press(&mut app, &[KeyCode::Enter]).await?;
        type_text(&mut app, "secret").await?;
        let shown = app.shown_input().to_string();
        press(&mut app, &[KeyCode::Enter]).await?;

        let text = std::fs::read_to_string(&app.config_path)?;
        assert_eq!(
            kinds,
            vec!["Anthropic", "DeepSeek", "OpenAI", "Your own server"]
        );
        assert_eq!(shown, "••••••");
        assert!(text.starts_with("# my servers\nsearch_engine_url"));
        assert!(text.contains(&format!(
            "[home-gpu]\nurl = \"{server_url}\"\napi_key = \"secret\""
        )));
        assert_eq!(app.provider.as_deref(), Some("home-gpu"));
        assert_eq!(app.model.as_deref(), Some("served"));
        assert_eq!(
            app.switched_client.map(|client| client.base_url),
            Some(server_url)
        );
        assert!(!app.transcript.text().contains("secret"));
        assert!(app.history.is_empty());
        Ok(())
    }

    #[tokio::test]
    async fn a_built_in_provider_asks_only_for_the_key() -> anyhow::Result<()> {
        // The table sets a local URL, so the test never calls the real API.
        let server_url = served_model_server().await;
        let mut app = app_with_config(&format!("[anthropic]\nurl = \"{server_url}\"\n"))?;
        app.open_provider_picker();

        press(&mut app, &[KeyCode::Down, KeyCode::Enter, KeyCode::Enter]).await?;
        type_text(&mut app, "sk-ant").await?;
        press(&mut app, &[KeyCode::Enter]).await?;

        let config = ConfigFile::load(&app.config_path)?;
        assert_eq!(
            config.providers["anthropic"].api_key.as_deref(),
            Some("sk-ant")
        );
        assert_eq!(config.providers["anthropic"].url, server_url);
        assert_eq!(app.provider.as_deref(), Some("anthropic"));
        assert!(app.switched_client.is_some());
        Ok(())
    }

    #[tokio::test]
    async fn ctrl_c_cancels_adding_a_provider() -> anyhow::Result<()> {
        let mut app = app_with_config("")?;
        app.open_provider_picker();
        press(
            &mut app,
            &[
                KeyCode::Enter,
                KeyCode::Down,
                KeyCode::Down,
                KeyCode::Down,
                KeyCode::Enter,
            ],
        )
        .await?;
        let asked = app.provider_form.is_some();

        press_with(&mut app, &[KeyCode::Char('c')], KeyModifiers::CONTROL).await?;

        assert!(asked);
        assert_eq!(app.provider_form, None);
        Ok(())
    }

    #[tokio::test]
    async fn a_server_without_a_url_is_not_added() -> anyhow::Result<()> {
        let mut app = app_with_config("")?;

        let url_step = ProviderForm::Url {
            name: "home-gpu".to_string(),
        };

        app.answer_provider_form(url_step, String::new()).await;
        let key_step = app.provider_form.take();
        app.answer_provider_form(ProviderForm::Name, "home-gpu".to_string())
            .await;
        app.answer_provider_form(
            ProviderForm::Key {
                name: "home-gpu".to_string(),
                url: None,
            },
            String::new(),
        )
        .await;

        assert_eq!(
            key_step,
            Some(ProviderForm::Key {
                name: "home-gpu".to_string(),
                url: None
            })
        );

        assert!(
            app.transcript
                .text()
                .contains("provider `home-gpu` needs a URL")
        );
        assert_eq!(std::fs::read_to_string(&app.config_path)?, "");
        assert_eq!(app.provider, None);
        Ok(())
    }

    #[tokio::test]
    async fn choosing_a_provider_uses_its_settings() -> anyhow::Result<()> {
        let server_url = served_model_server().await;
        let mut app = app_with_config(&format!(
            "[box]\nurl = \"{server_url}\"\nthink = false\ntemperature = 0.5\n"
        ))?;
        app.open_provider_picker();

        press(&mut app, &[KeyCode::Enter]).await?;

        assert_eq!(app.provider.as_deref(), Some("box"));
        assert!(!app.run_config.think);
        assert_eq!(app.run_config.temperature, Some(0.5));
        assert_eq!(app.server_url, server_url);
        assert!(app.transcript.text().contains("Provider box"));
        Ok(())
    }

    #[tokio::test]
    async fn a_provider_removed_meanwhile_is_reported() -> anyhow::Result<()> {
        let mut app = app_with_config("[box]\nurl = \"http://box/v1\"\n")?;
        app.open_provider_picker();
        std::fs::write(&app.config_path, "")?;

        press(&mut app, &[KeyCode::Enter]).await?;

        assert!(app.transcript.text().contains("no provider named `box`"));
        assert_eq!(app.provider, None);
        Ok(())
    }

    #[tokio::test]
    async fn the_local_ollama_is_one_enter_away_and_writes_nothing() -> anyhow::Result<()> {
        let mut app = app_with_config("")?;
        app.provider = Some("old".to_string());
        app.open_provider_picker();

        press(&mut app, &[KeyCode::Down, KeyCode::Enter]).await?;

        assert_eq!(app.provider, None);
        assert_eq!(
            app.switched_client.map(|client| client.base_url),
            Some(DEFAULT_OLLAMA_URL.to_string())
        );
        assert_eq!(std::fs::read_to_string(&app.config_path)?, "");
        Ok(())
    }

    #[tokio::test]
    async fn a_row_past_the_choices_does_nothing() -> anyhow::Result<()> {
        let mut app = app_with_config("")?;

        app.choose_provider(Vec::new(), 0).await;

        assert!(app.transcript.is_empty());
        assert!(app.modal.is_none());
        Ok(())
    }

    #[tokio::test]
    async fn closing_settings_saves_them_to_the_provider() -> anyhow::Result<()> {
        let mut app = app_with_config("[box] # mine\nurl = \"http://box/v1\"\n")?;
        app.provider = Some("box".to_string());
        app.run_config.temperature = Some(0.6);
        app.open_settings();

        press(&mut app, &[KeyCode::Down, KeyCode::Right, KeyCode::Esc]).await?;

        let text = std::fs::read_to_string(&app.config_path)?;
        assert!(text.contains("[box] # mine"));
        assert!(text.contains("think = false"));
        assert!(text.contains("temperature = 0.6"));
        Ok(())
    }

    #[tokio::test]
    async fn without_a_provider_settings_last_for_the_session() -> anyhow::Result<()> {
        let mut app = app_with_config("[box]\nurl = \"http://box/v1\"\n")?;
        let mut orphan = app_with_config("")?;
        orphan.provider = Some("box".to_string());
        app.open_settings();
        orphan.open_settings();

        press(&mut app, &[KeyCode::Esc]).await?;
        press(&mut orphan, &[KeyCode::Esc]).await?;

        assert_eq!(
            std::fs::read_to_string(&app.config_path)?,
            "[box]\nurl = \"http://box/v1\"\n"
        );
        assert!(orphan.transcript.text().contains("no provider named `box`"));
        Ok(())
    }

    #[tokio::test]
    async fn the_provider_command_opens_the_picker_or_switches() -> anyhow::Result<()> {
        let server_url = served_model_server().await;
        let mut app = app_with_config(&format!("[box]\nurl = \"{server_url}\"\n"))?;
        let client = OllamaClient::new("http://127.0.0.1:1")?;
        let (tx, _rx) = mpsc::unbounded_channel();

        app.run_slash_command(&client, "/provider", &tx).await?;
        let opened = app.is_showing_picker();
        app.close_modal();
        app.run_slash_command(&client, "/provider box", &tx).await?;
        let unknown = app.run_slash_command(&client, "/provider nope", &tx).await;

        assert!(opened);
        assert_eq!(app.provider.as_deref(), Some("box"));
        assert!(format!("{:?}", unknown.err()).contains("no provider named `nope`"));
        Ok(())
    }

    #[test]
    fn help_and_completion_name_the_provider_command() {
        assert!(crate::constants::tui::COMMANDS.contains(&"provider"));
        assert!(
            crate::constants::tui::HELP_LINES
                .iter()
                .any(|line| line.contains("/provider [name]"))
        );
    }
}
