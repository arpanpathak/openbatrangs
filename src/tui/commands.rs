//! TUI slash-command handlers.
//!
//! Keeping command parsing and state mutations separate from the main event
//! loop keeps `App` focused on navigation/editing while this module owns the
//! `/command` behaviors.

use super::app::{App, JobKind};
use super::split_command;
use super::{ScrollMode, UiEvent, run_pull_worker, run_setup_worker};
use crate::cli::AgentMode;
use crate::constants::tui::HELP_LINES;
use crate::ollama::{OllamaClient, OllamaModel};
use std::path::PathBuf;
use tokio::sync::mpsc;

/// True when a model tag is already installed locally (pure and testable).
fn model_installed(tags: &[OllamaModel], requested: &str) -> bool {
    tags.iter().any(|model| model.name == requested)
}

impl App {
    /// Parse and execute a slash command from user input.
    ///
    /// # Parameters
    ///
    /// - `client`: Ollama client for network-dependent commands.
    /// - `line`: raw input line starting with `/`.
    /// - `tx`: channel for background task events (pull, setup).
    ///
    /// # Returns
    ///
    /// `Ok(())` on success, or the network error of `/models`, `/model` or `/doctor`.
    pub(super) async fn run_slash_command(
        &mut self,
        client: &OllamaClient,
        line: &str,
        tx: &mpsc::UnboundedSender<UiEvent>,
    ) -> anyhow::Result<()> {
        let (name, arg) = split_command(line);

        match name {
            "help" | "h" => self.log_help(),
            "status" => self.log_status(),
            "exit" | "quit" => self.should_quit = true,
            "models" => self.show_model_picker(client).await?,
            "model" => self.handle_model_command(client, arg).await?,
            "pull" => self.handle_pull_command(client, arg, tx),
            "read-only" => self.toggle_read_only(),
            "confirm" => self.toggle_confirm(),
            "steps" => self.handle_steps_command(arg),
            "cwd" => self.handle_cwd_command(arg),
            "doctor" => self.handle_doctor_command(client).await?,
            "setup" => self.handle_setup_command(client, tx),
            "clear" => self.clear_chat(),
            "perf" => self.toggle_perf(),
            "mode" => self.handle_mode_command(arg),
            "thinking" => self.handle_thinking_command(arg),
            "mouse" => self.handle_mouse_command(arg),
            "yolo" => self.handle_yolo_command(),
            _ => self.log_unknown_command(name),
        }

        Ok(())
    }

    /// Print the full list of slash commands and key bindings to the chat log.
    fn log_help(&mut self) {
        let lines = HELP_LINES.iter().map(|line| line.to_string()).collect();

        self.notice_lines(lines);
    }

    /// Print the current configuration and model status to the chat log.
    fn log_status(&mut self) {
        let lines = self.status_lines();

        self.notice_lines(lines);
    }

    /// The lines `/status` prints: mode, model, server, and every setting.
    fn status_lines(&self) -> Vec<String> {
        let model = self.model.as_deref().unwrap_or("auto (best available)");
        let model_info = self
            .model_info
            .as_ref()
            .map(|info| format!("  Model info: {info}"));
        let config = &self.run_config;

        let head = vec![
            "═══ Current Status ═══".to_string(),
            format!("  Mode:       {}", mode_description(config.mode)),
            format!("  Model:      {model}"),
        ];
        let settings = vec![
            format!("  Server:     {}", self.server_url),
            format!("  Workspace:  {}", config.cwd.display()),
            format!("  Max steps:  {}", config.max_steps),
            format!("  Read-only:  {}", on_off(config.is_read_only)),
            format!("  Confirm:    {}", confirm_label(config.should_confirm)),
            format!("  Thinking:   {}", on_off(config.show_thinking)),
            format!("  Mouse:      {}", on_off(self.mouse_capture)),
            format!("  Perf panel: {}", on_off(self.show_perf)),
        ];

        head.into_iter().chain(model_info).chain(settings).collect()
    }

    /// Open the model picker popup, populated from installed Ollama tags.
    ///
    /// # Parameters
    ///
    /// - `client`: Ollama client for fetching installed model tags.
    async fn show_model_picker(&mut self, client: &OllamaClient) -> anyhow::Result<()> {
        let tags = client.tags().await?;

        self.open_model_picker(tags.into_iter().map(|model| model.name).collect());

        Ok(())
    }

    /// Open the picker on `models`, or point to `/setup` when there are none.
    fn open_model_picker(&mut self, models: Vec<String>) {
        if models.is_empty() {
            self.notice("No models installed. Run /setup.".to_string());
            return;
        }

        self.open_picker(models);
        self.status = "select model".to_string();
    }

    /// Switch to a specific model by tag, or show the current model if no arg.
    ///
    /// # Parameters
    ///
    /// - `client`: Ollama client for checking installed tags.
    /// - `arg`: model tag to switch to, or empty to show the current model.
    async fn handle_model_command(
        &mut self,
        client: &OllamaClient,
        arg: &str,
    ) -> anyhow::Result<()> {
        if arg.is_empty() {
            self.log_current_model();
            return Ok(());
        }

        self.switch_to_installed_model(client, arg).await
    }

    /// Activate `name` when the server has it installed, or explain how to pull it.
    async fn switch_to_installed_model(
        &mut self,
        client: &OllamaClient,
        name: &str,
    ) -> anyhow::Result<()> {
        let tags = client.tags().await?;

        self.activate_if_installed(client, &tags, name).await;

        Ok(())
    }

    /// Activate `name` when it is one of `tags`, or explain how to pull it.
    async fn activate_if_installed(
        &mut self,
        client: &OllamaClient,
        tags: &[OllamaModel],
        name: &str,
    ) {
        if !model_installed(tags, name) {
            self.notice(format!(
                "❌ Model '{name}' is not installed. Pull it first with /pull {name}."
            ));
            return;
        }

        self.activate_model(client, name).await;
    }

    /// Print the currently selected model name to the chat log.
    fn log_current_model(&mut self) {
        match &self.model {
            Some(model) => self.notice(format!("Current model: {model}")),
            None => {
                self.notice("Auto mode — best model will be selected on first task.".to_string())
            }
        }
    }

    /// Set the active model and refresh its metadata from the Ollama server.
    ///
    /// # Parameters
    ///
    /// - `client`: Ollama client for refreshing model info.
    /// - `name`: the model tag to activate.
    pub(super) async fn activate_model(&mut self, client: &OllamaClient, name: &str) {
        self.model = Some(name.to_string());
        self.notice(format!("✅ Model set to {name}"));
        self.refresh_model_info(client).await;
    }

    /// Start downloading a model from the Ollama registry in the background.
    ///
    /// # Parameters
    ///
    /// - `client`: Ollama client for the pull operation.
    /// - `arg`: model tag to pull.
    /// - `tx`: channel for progress events.
    fn handle_pull_command(
        &mut self,
        client: &OllamaClient,
        arg: &str,
        tx: &mpsc::UnboundedSender<UiEvent>,
    ) {
        let name = arg.trim();

        if name.is_empty() {
            self.notice("Usage: /pull <model-tag>".to_string());
            return;
        }

        self.pull_unless_busy(client, name.to_string(), tx);
    }

    /// Start pulling `name`, or refuse while another task runs.
    fn pull_unless_busy(
        &mut self,
        client: &OllamaClient,
        name: String,
        tx: &mpsc::UnboundedSender<UiEvent>,
    ) {
        if self.is_running {
            self.notice("⚠️  Busy — finish the current task before pulling a model.".to_string());
            return;
        }

        self.notice(format!("⬇️  Pulling model '{name}' in background..."));
        self.start_pull(client, name, tx);
    }

    /// Spawn a background task to pull a model and track it as the current task.
    ///
    /// # Parameters
    ///
    /// - `client`: Ollama client for the pull operation.
    /// - `name`: model tag to download.
    /// - `tx`: channel for progress events.
    fn start_pull(
        &mut self,
        client: &OllamaClient,
        name: String,
        tx: &mpsc::UnboundedSender<UiEvent>,
    ) {
        self.is_running = true;
        self.status = "pulling model".to_string();
        self.last_action = format!("pulling {name}");
        self.scroll_mode = ScrollMode::Follow;

        let client = client.clone();
        let job_tx = self.job_sender(tx);
        let handle = tokio::spawn(run_pull_worker(client, name, job_tx));

        self.track_job(JobKind::Command, handle);
    }

    /// Toggle read-only mode on/off, preventing or allowing file writes and commands.
    fn toggle_read_only(&mut self) {
        self.run_config.is_read_only = !self.run_config.is_read_only;

        let state = on_off(self.run_config.is_read_only);
        self.notice(format!("Read-only mode: {state}"));
    }

    /// Toggle confirmation prompts on/off for writes and shell commands.
    fn toggle_confirm(&mut self) {
        self.run_config.should_confirm = !self.run_config.should_confirm;

        let state = on_off(self.run_config.should_confirm);
        self.notice(format!("Confirm mode: {state}"));
    }

    /// Switch the agent mode: agent (full tools), plan (read-only), or chat (no tools).
    ///
    /// # Parameters
    ///
    /// - `arg`: one of "agent", "plan", or "chat".
    fn handle_mode_command(&mut self, arg: &str) {
        let Some(mode) = parse_mode(arg) else {
            self.notice("Usage: /mode agent|plan|chat".to_string());
            return;
        };

        self.run_config.mode = mode;
        self.notice(mode_switch_message(mode).to_string());
    }

    /// Toggle or display the model reasoning/thinking display.
    ///
    /// # Parameters
    ///
    /// - `arg`: "on", "off", or empty to show current state.
    fn handle_thinking_command(&mut self, arg: &str) {
        let Some(show_thinking) = parse_on_off(arg) else {
            let state = on_off(self.run_config.show_thinking);
            self.notice(format!(
                "Thinking display: {state} · Usage: /thinking on|off"
            ));
            return;
        };

        self.run_config.show_thinking = show_thinking;
        self.notice(format!("Thinking display: {}", on_off(show_thinking)));
    }

    /// Toggle mouse capture: on for wheel/scrollbar, off for native select/copy.
    ///
    /// # Parameters
    ///
    /// - `arg`: "on", "off", or empty to show current state.
    fn handle_mouse_command(&mut self, arg: &str) {
        let Some(mouse_capture) = parse_on_off(arg) else {
            let state = on_off(self.mouse_capture);
            self.notice(format!("Mouse mode: {state} · Usage: /mouse on|off"));
            return;
        };

        self.mouse_capture = mouse_capture;
        self.notice(mouse_mode_message(mouse_capture).to_string());
    }

    /// Disable confirmation prompts ("YOLO mode") for writes and commands.
    fn handle_yolo_command(&mut self) {
        self.run_config.should_confirm = false;
        self.notice(
            "⚠️  YOLO mode ON — agent writes/commands run without confirmation. Be careful!"
                .to_string(),
        );
    }

    /// Set the maximum number of agent iterations per task.
    ///
    /// # Parameters
    ///
    /// - `arg`: positive integer string, or empty/invalid to show usage.
    fn handle_steps_command(&mut self, arg: &str) {
        let Some(steps) = positive_number(arg) else {
            self.notice("Usage: /steps <positive number>".to_string());
            return;
        };

        self.run_config.max_steps = steps;
        self.notice(format!("Max steps set to {steps}"));
    }

    /// Show or change the agent's working directory.
    ///
    /// # Parameters
    ///
    /// - `arg`: new directory path, or empty to show the current workspace.
    fn handle_cwd_command(&mut self, arg: &str) {
        if arg.is_empty() {
            self.notice(format!("Workspace: {}", self.run_config.cwd.display()));
            return;
        }

        self.run_config.cwd = PathBuf::from(arg);
        self.notice(format!(
            "Workspace set to {}",
            self.run_config.cwd.display()
        ));
    }

    /// Run connectivity and model-recommendation diagnostics, logging results.
    ///
    /// # Parameters
    ///
    /// - `client`: Ollama client for checking server status and installed models.
    async fn handle_doctor_command(&mut self, client: &OllamaClient) -> anyhow::Result<()> {
        let lines = crate::commands::doctor_lines(client, self.min_context).await?;

        self.notice_lines(lines);

        Ok(())
    }

    /// Run `/setup` (install Ollama + pull recommended model) in the background.
    ///
    /// # Parameters
    ///
    /// - `client`: Ollama client for setup operations.
    /// - `tx`: channel for progress events.
    fn handle_setup_command(&mut self, client: &OllamaClient, tx: &mpsc::UnboundedSender<UiEvent>) {
        if self.is_running {
            self.notice("⚠️  Busy — finish the current task before running setup.".to_string());
            return;
        }

        self.notice("🔄 Running setup in background...".to_string());
        self.is_running = true;
        self.status = "running setup".to_string();
        self.last_action = "setup".to_string();
        self.scroll_mode = ScrollMode::Follow;

        let job_tx = self.job_sender(tx);
        let handle = tokio::spawn(run_setup_worker(client.clone(), job_tx));

        self.track_job(JobKind::Command, handle);
    }

    /// Clear the transcript, the streamed reply, the prompt, and the
    /// conversation history.
    pub(super) fn clear_chat(&mut self) {
        self.transcript.clear();
        self.reply.clear();
        self.current_prompt = None;
        self.interrupted_task = None;
        self.agent_conversation = None;
        self.chat_history.clear();

        self.scroll_mode = ScrollMode::Follow;
        self.chat_scroll_offset = 0;
    }

    /// Toggle the live GPU/CPU/RAM performance panel on/off.
    fn toggle_perf(&mut self) {
        self.show_perf = !self.show_perf;

        self.notice(format!("Perf panel: {}", on_off(self.show_perf)));
    }

    /// Log a "command not found" message with a hint to use `/help`.
    ///
    /// # Parameters
    ///
    /// - `name`: the unrecognized command name.
    fn log_unknown_command(&mut self, name: &str) {
        self.notice(format!("Unknown command: /{name}. Try /help"));
    }
}

/// `"ON"` or `"OFF"` for a setting.
fn on_off(enabled: bool) -> &'static str {
    match enabled {
        true => "ON",
        false => "OFF",
    }
}

/// The `/status` label for the confirmation setting; off means yolo mode.
fn confirm_label(should_confirm: bool) -> &'static str {
    match should_confirm {
        true => "ON",
        false => "OFF (yolo)",
    }
}

/// What each agent mode allows, as shown by `/status`.
fn mode_description(mode: AgentMode) -> &'static str {
    match mode {
        AgentMode::Agent => "agent (full tools)",
        AgentMode::Plan => "plan (read-only, no writes)",
        AgentMode::Chat => "chat (no tools, conversation only)",
    }
}

/// The agent mode named by a `/mode` argument, or `None` for an unknown name.
fn parse_mode(arg: &str) -> Option<AgentMode> {
    match arg {
        "agent" => Some(AgentMode::Agent),
        "plan" => Some(AgentMode::Plan),
        "chat" => Some(AgentMode::Chat),
        _ => None,
    }
}

/// The message logged after `/mode` switches to `mode`.
fn mode_switch_message(mode: AgentMode) -> &'static str {
    match mode {
        AgentMode::Agent => "Mode: agent (full tools enabled)",
        AgentMode::Plan => "Mode: plan (read-only, explores and plans)",
        AgentMode::Chat => "Mode: chat (no tools, conversation + code)",
    }
}

/// `Some(true)` for `"on"`, `Some(false)` for `"off"`, `None` for anything else.
fn parse_on_off(arg: &str) -> Option<bool> {
    match arg {
        "on" => Some(true),
        "off" => Some(false),
        _ => None,
    }
}

/// The message logged after `/mouse on` or `/mouse off`.
fn mouse_mode_message(mouse_capture: bool) -> &'static str {
    match mouse_capture {
        true => "Mouse mode: ON (wheel scroll + click-to-open, selection via Shift+drag)",
        false => "Mouse mode: OFF (native select/copy, scroll with PgUp/PgDn)",
    }
}

/// `arg` as a number above zero, or `None`.
fn positive_number(arg: &str) -> Option<usize> {
    arg.parse::<usize>().ok().filter(|number| *number > 0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;
    use std::sync::{Arc, Mutex};

    fn installed_model(name: &str) -> OllamaModel {
        serde_json::from_value(serde_json::json!({
            "name": name,
            "size": 123,
        }))
        .expect("valid model JSON")
    }

    #[test]
    fn model_installed_detects_existing_tags() {
        let tags = vec![installed_model("qwen2.5-coder:7b")];
        assert!(model_installed(&tags, "qwen2.5-coder:7b"));
    }

    #[test]
    fn model_installed_false_for_missing_tags() {
        let tags = vec![installed_model("qwen2.5-coder:7b")];
        assert!(!model_installed(&tags, "llama3.2:3b"));
    }

    #[test]
    fn model_installed_false_with_no_models() {
        assert!(!model_installed(&[], "qwen2.5-coder:7b"));
    }

    #[tokio::test]
    async fn pull_command_without_model_shows_usage_and_does_not_run() {
        let cli = crate::cli::Cli::parse_from(["openbatrangs"]);
        let mut app = App::new(&cli, Arc::new(Mutex::new(None)));
        let client = OllamaClient::new("http://localhost:11434").unwrap();
        let (tx, _rx) = mpsc::unbounded_channel();
        app.handle_pull_command(&client, "", &tx);
        assert!(app.transcript.text().contains("Usage: /pull"));
        assert!(!app.is_running);
        assert!(app.current_task.is_none());
    }

    #[test]
    fn log_help_pushes_command_list() {
        let cli = crate::cli::Cli::parse_from(["openbatrangs"]);
        let mut app = App::new(&cli, Arc::new(Mutex::new(None)));
        app.transcript.clear();
        app.log_help();
        let text = app.transcript.text();
        assert!(text.contains("Commands"));
        assert!(text.contains("/help"));
        assert!(text.contains("/exit"));
        assert!(text.contains("/models"));
        assert!(text.contains("Keys"));
    }

    #[test]
    fn log_status_shows_current_settings() {
        let cli = crate::cli::Cli::parse_from(["openbatrangs"]);
        let mut app = App::new(&cli, Arc::new(Mutex::new(None)));
        app.transcript.clear();
        app.log_status();
        let text = app.transcript.text();
        assert!(text.contains("Status"));
        assert!(text.contains("Mode:"));
        assert!(text.contains("Model:"));
        assert!(text.contains("Workspace:"));
    }

    #[test]
    fn toggle_read_only_flips_state() {
        let cli = crate::cli::Cli::parse_from(["openbatrangs"]);
        let mut app = App::new(&cli, Arc::new(Mutex::new(None)));
        let initial = app.run_config.is_read_only;
        app.toggle_read_only();
        assert_eq!(app.run_config.is_read_only, !initial);
        assert!(app.transcript.text().contains("Read-only mode:"));
        app.toggle_read_only();
        assert_eq!(app.run_config.is_read_only, initial);
    }

    #[test]
    fn toggle_confirm_flips_state() {
        let cli = crate::cli::Cli::parse_from(["openbatrangs"]);
        let mut app = App::new(&cli, Arc::new(Mutex::new(None)));
        let initial = app.run_config.should_confirm;
        app.toggle_confirm();
        assert_eq!(app.run_config.should_confirm, !initial);
        assert!(app.transcript.text().contains("Confirm mode:"));
    }

    #[test]
    fn handle_mode_command_sets_agent() {
        let cli = crate::cli::Cli::parse_from(["openbatrangs"]);
        let mut app = App::new(&cli, Arc::new(Mutex::new(None)));
        app.transcript.clear();
        app.handle_mode_command("agent");
        assert_eq!(app.run_config.mode, AgentMode::Agent);
        assert!(app.transcript.text().contains("agent (full tools enabled)"));
    }

    #[test]
    fn handle_mode_command_sets_plan() {
        let cli = crate::cli::Cli::parse_from(["openbatrangs"]);
        let mut app = App::new(&cli, Arc::new(Mutex::new(None)));
        app.transcript.clear();
        app.handle_mode_command("plan");
        assert_eq!(app.run_config.mode, AgentMode::Plan);
        assert!(app.transcript.text().contains("plan (read-only"));
    }

    #[test]
    fn handle_mode_command_sets_chat() {
        let cli = crate::cli::Cli::parse_from(["openbatrangs"]);
        let mut app = App::new(&cli, Arc::new(Mutex::new(None)));
        app.transcript.clear();
        app.handle_mode_command("chat");
        assert_eq!(app.run_config.mode, AgentMode::Chat);
        assert!(app.transcript.text().contains("chat (no tools"));
    }

    #[test]
    fn handle_mode_command_invalid_shows_usage() {
        let cli = crate::cli::Cli::parse_from(["openbatrangs"]);
        let mut app = App::new(&cli, Arc::new(Mutex::new(None)));
        app.transcript.clear();
        app.handle_mode_command("invalid");
        assert!(
            app.transcript
                .text()
                .contains("Usage: /mode agent|plan|chat")
        );
    }

    #[test]
    fn handle_steps_command_sets_valid_steps() {
        let cli = crate::cli::Cli::parse_from(["openbatrangs"]);
        let mut app = App::new(&cli, Arc::new(Mutex::new(None)));
        app.transcript.clear();
        app.handle_steps_command("25");
        assert_eq!(app.run_config.max_steps, 25);
        assert!(app.transcript.text().contains("Max steps set to 25"));
    }

    #[test]
    fn handle_steps_command_rejects_zero() {
        let cli = crate::cli::Cli::parse_from(["openbatrangs"]);
        let mut app = App::new(&cli, Arc::new(Mutex::new(None)));
        app.transcript.clear();
        app.handle_steps_command("0");
        assert!(
            app.transcript
                .text()
                .contains("Usage: /steps <positive number>")
        );
    }

    #[test]
    fn handle_steps_command_rejects_non_numeric() {
        let cli = crate::cli::Cli::parse_from(["openbatrangs"]);
        let mut app = App::new(&cli, Arc::new(Mutex::new(None)));
        app.transcript.clear();
        app.handle_steps_command("abc");
        assert!(
            app.transcript
                .text()
                .contains("Usage: /steps <positive number>")
        );
    }

    #[test]
    fn handle_thinking_command_on() {
        let cli = crate::cli::Cli::parse_from(["openbatrangs"]);
        let mut app = App::new(&cli, Arc::new(Mutex::new(None)));
        app.transcript.clear();
        app.handle_thinking_command("on");
        assert!(app.run_config.show_thinking);
        assert!(app.transcript.text().contains("Thinking display: ON"));
    }

    #[test]
    fn handle_thinking_command_off() {
        let cli = crate::cli::Cli::parse_from(["openbatrangs"]);
        let mut app = App::new(&cli, Arc::new(Mutex::new(None)));
        app.transcript.clear();
        app.handle_thinking_command("off");
        assert!(!app.run_config.show_thinking);
        assert!(app.transcript.text().contains("Thinking display: OFF"));
    }

    #[test]
    fn handle_thinking_command_invalid_shows_current_state() {
        let cli = crate::cli::Cli::parse_from(["openbatrangs"]);
        let mut app = App::new(&cli, Arc::new(Mutex::new(None)));
        app.transcript.clear();
        app.handle_thinking_command("maybe");
        assert!(app.transcript.text().contains("Usage: /thinking on|off"));
    }

    #[test]
    fn handle_mouse_command_on() {
        let cli = crate::cli::Cli::parse_from(["openbatrangs"]);
        let mut app = App::new(&cli, Arc::new(Mutex::new(None)));
        app.transcript.clear();
        app.handle_mouse_command("on");
        assert!(app.mouse_capture);
        assert!(app.transcript.text().contains("Mouse mode: ON"));
    }

    #[test]
    fn handle_mouse_command_off() {
        let cli = crate::cli::Cli::parse_from(["openbatrangs"]);
        let mut app = App::new(&cli, Arc::new(Mutex::new(None)));
        app.transcript.clear();
        app.handle_mouse_command("off");
        assert!(!app.mouse_capture);
        assert!(app.transcript.text().contains("Mouse mode: OFF"));
    }

    #[test]
    fn handle_yolo_disables_confirmation() {
        let cli = crate::cli::Cli::parse_from(["openbatrangs"]);
        let mut app = App::new(&cli, Arc::new(Mutex::new(None)));
        app.run_config.should_confirm = true;
        app.transcript.clear();
        app.handle_yolo_command();
        assert!(!app.run_config.should_confirm);
        assert!(app.transcript.text().contains("YOLO mode ON"));
    }

    #[test]
    fn log_unknown_command_shows_message() {
        let cli = crate::cli::Cli::parse_from(["openbatrangs"]);
        let mut app = App::new(&cli, Arc::new(Mutex::new(None)));
        app.transcript.clear();
        app.log_unknown_command("foobar");
        assert!(app.transcript.text().contains("Unknown command: /foobar"));
        assert!(app.transcript.text().contains("/help"));
    }

    #[test]
    fn handle_cwd_command_with_empty_arg_shows_current() {
        let cli = crate::cli::Cli::parse_from(["openbatrangs"]);
        let mut app = App::new(&cli, Arc::new(Mutex::new(None)));
        let original_cwd = app.run_config.cwd.clone();
        app.transcript.clear();
        app.handle_cwd_command("");
        assert!(
            app.transcript
                .text()
                .contains(&format!("Workspace: {}", original_cwd.display()))
        );
    }

    #[test]
    fn handle_cwd_command_with_path_changes_workspace() {
        let cli = crate::cli::Cli::parse_from(["openbatrangs"]);
        let mut app = App::new(&cli, Arc::new(Mutex::new(None)));
        app.transcript.clear();
        app.handle_cwd_command("/tmp/new_workspace");
        assert_eq!(
            app.run_config.cwd,
            std::path::PathBuf::from("/tmp/new_workspace")
        );
        assert!(
            app.transcript
                .text()
                .contains("Workspace set to /tmp/new_workspace")
        );
    }

    #[test]
    fn clear_chat_resets_all_state() {
        let cli = crate::cli::Cli::parse_from(["openbatrangs"]);
        let mut app = App::new(&cli, Arc::new(Mutex::new(None)));
        app.reply = "streaming text".to_string();
        app.chat_scroll_offset = 42;
        app.scroll_mode = ScrollMode::Manual;
        app.clear_chat();
        assert_eq!(app.reply, "");
        assert_eq!(app.chat_scroll_offset, 0);
        assert_eq!(app.scroll_mode, ScrollMode::Follow);
    }

    /// A mock Ollama server with one installed model, `qwen2.5-coder:7b`.
    async fn server_with_one_model() -> anyhow::Result<OllamaClient> {
        let base_url = crate::test_support::spawn_mock_server(|_path| {
            crate::test_support::MockResponse::json(
                "200 OK",
                r#"{"models":[{"name":"qwen2.5-coder:7b","size":4700000000}]}"#,
            )
        })
        .await;

        OllamaClient::new(&base_url)
    }

    /// An app with default settings.
    fn test_app() -> App {
        let cli = crate::cli::Cli::parse_from(["openbatrangs"]);
        App::new(&cli, Arc::new(Mutex::new(None)))
    }

    #[tokio::test]
    async fn every_slash_command_runs_through_the_dispatcher() -> anyhow::Result<()> {
        let client = server_with_one_model().await?;
        let (tx, _rx) = mpsc::unbounded_channel();
        let mut app = test_app();
        let commands = [
            "/h",
            "/model",
            "/model qwen2.5-coder:7b",
            "/model",
            "/model missing:1b",
            "/status",
            "/read-only",
            "/confirm",
            "/steps 3",
            "/cwd",
            "/doctor",
            "/perf",
            "/mode chat",
            "/thinking",
            "/thinking off",
            "/mouse",
            "/mouse on",
            "/yolo",
            "/nope",
            "/models",
            "/exit",
        ];

        let results = run_commands(&mut app, &client, &commands, &tx).await;

        let text = app.transcript.text();
        assert!(results.iter().all(Result::is_ok));
        assert!(text.contains("Auto mode"));
        assert!(text.contains("Current model: qwen2.5-coder:7b"));
        assert!(text.contains("  Model info: model: qwen2.5-coder:7b · 4.7 GB"));
        assert!(text.contains("❌ Model 'missing:1b' is not installed."));
        assert!(text.contains("Ollama reachable"));
        assert!(text.contains("Thinking display: ON · Usage"));
        assert!(text.contains("Mouse mode: ON · Usage"));
        assert!(text.contains("Mouse mode: ON (wheel scroll"));
        assert!(app.is_showing_picker());
        assert_eq!(app.status, "select model");
        assert!(app.should_quit);

        Ok(())
    }

    /// Run each slash command in order and collect the results.
    async fn run_commands(
        app: &mut App,
        client: &OllamaClient,
        commands: &[&str],
        tx: &mpsc::UnboundedSender<UiEvent>,
    ) -> Vec<anyhow::Result<()>> {
        let mut results = Vec::new();

        for command in commands {
            results.push(app.run_slash_command(client, command, tx).await);
        }

        results
    }

    #[test]
    fn status_shows_every_setting() {
        let mut app = test_app();
        app.run_config.mode = AgentMode::Plan;
        app.run_config.should_confirm = false;
        app.mouse_capture = false;

        let mut lines = app.status_lines();

        assert!(lines.contains(&"  Mode:       plan (read-only, no writes)".to_string()));
        assert!(lines.contains(&"  Confirm:    OFF (yolo)".to_string()));
        assert!(lines.contains(&"  Mouse:      OFF".to_string()));

        app.run_config.mode = AgentMode::Chat;
        lines = app.status_lines();

        assert!(lines.contains(&"  Mode:       chat (no tools, conversation only)".to_string()));
    }

    #[tokio::test]
    async fn the_picker_needs_installed_models() -> anyhow::Result<()> {
        let base_url = crate::test_support::spawn_mock_server(|_path| {
            crate::test_support::MockResponse::json("200 OK", r#"{"models":[]}"#)
        })
        .await;
        let client = OllamaClient::new(&base_url)?;
        let (tx, _rx) = mpsc::unbounded_channel();
        let mut app = test_app();

        app.run_slash_command(&client, "/models", &tx).await?;

        assert!(!app.is_showing_picker());
        assert!(app.transcript.text().contains("No models installed"));

        Ok(())
    }

    #[tokio::test]
    async fn pull_and_setup_start_in_the_background_unless_busy() -> anyhow::Result<()> {
        let client = server_with_one_model().await?;
        let (tx, _rx) = mpsc::unbounded_channel();
        let mut app = test_app();

        app.run_slash_command(&client, "/pull qwen2.5-coder:3b", &tx)
            .await?;

        assert!(app.is_running);
        assert_eq!(app.last_action, "pulling qwen2.5-coder:3b");

        app.run_slash_command(&client, "/pull llama3.2:3b", &tx)
            .await?;
        app.run_slash_command(&client, "/setup", &tx).await?;

        assert!(
            app.transcript
                .text()
                .contains("Busy — finish the current task before pulling")
        );
        assert!(
            app.transcript
                .text()
                .contains("Busy — finish the current task before running setup")
        );

        let mut idle_app = test_app();
        idle_app.run_slash_command(&client, "/setup", &tx).await?;

        assert!(idle_app.is_running);
        assert_eq!(idle_app.status, "running setup");

        Ok(())
    }

    #[tokio::test]
    async fn clear_runs_through_the_dispatcher() -> anyhow::Result<()> {
        let client = OllamaClient::new("http://127.0.0.1:1")?;
        let (tx, _rx) = mpsc::unbounded_channel();
        let mut app = test_app();
        app.notice("old line".to_string());

        app.run_slash_command(&client, "/clear", &tx).await?;

        assert!(app.transcript.is_empty());

        Ok(())
    }

    #[test]
    fn the_mode_switch_messages_name_each_mode() {
        let mut app = test_app();

        app.handle_mode_command("plan");
        app.handle_mode_command("agent");

        assert!(
            app.transcript
                .text()
                .contains("Mode: agent (full tools enabled)")
        );
    }

    #[test]
    fn toggle_perf_flips_display() {
        let cli = crate::cli::Cli::parse_from(["openbatrangs"]);
        let mut app = App::new(&cli, Arc::new(Mutex::new(None)));
        let initial = app.show_perf;
        app.toggle_perf();
        assert_eq!(app.show_perf, !initial);
        assert!(app.transcript.text().contains("Perf panel:"));
    }
}
