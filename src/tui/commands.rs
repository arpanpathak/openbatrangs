//! TUI slash-command handlers.
//!
//! Keeping command parsing and state mutations separate from the main event
//! loop keeps `App` focused on navigation/editing while this module owns the
//! `/command` behaviors.

use super::app::App;
use super::split_command;
use super::{run_pull_worker, run_setup_worker, ScrollMode, UiEvent};
use crate::cli::AgentMode;
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
    /// `Ok(())` on success; network errors are logged to chat instead of propagated.
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
            "pull" => self.handle_pull_command(client, arg, tx).await?,
            "read-only" => self.toggle_read_only(),
            "confirm" => self.toggle_confirm(),
            "steps" => self.handle_steps_command(arg),
            "cwd" => self.handle_cwd_command(arg),
            "doctor" => self.handle_doctor_command(client).await?,
            "setup" => self.handle_setup_command(client, tx).await?,
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
        self.log.push("═══ Commands ═══".to_string());
        self.log
            .push("  /help              — show this help".to_string());
        self.log
            .push("  /exit, /quit       — leave the TUI".to_string());
        self.log.push(
            "  /status            — show current settings and model".to_string(),
        );
        self.log.push(
            "  /mode agent|plan|chat — agent=tools, plan=read-only, chat=no tools"
                .to_string(),
        );
        self.log.push(
            "  /model <tag>       — switch to a specific installed model"
                .to_string(),
        );
        self.log
            .push("  /models            — open model picker".to_string());
        self.log
            .push("  /pull <tag>        — download a model from Ollama".to_string());
        self.log.push(
            "  /setup             — install/start Ollama + pull a model"
                .to_string(),
        );
        self.log.push(
            "  /doctor            — check Ollama + best model recommendation"
                .to_string(),
        );
        self.log.push(
            "  /read-only         — toggle read-only mode (no writes/commands)"
                .to_string(),
        );
        self.log.push(
            "  /confirm           — toggle confirm before writes/commands"
                .to_string(),
        );
        self.log.push(
            "  /yolo              — disable confirmations (careful!)"
                .to_string(),
        );
        self.log
            .push("  /thinking on|off   — show/hide model reasoning".to_string());
        self.log
            .push("  /steps <n>         — set max agent iterations".to_string());
        self.log
            .push("  /cwd <path>        — change workspace directory".to_string());
        self.log
            .push("  /perf              — toggle performance panel".to_string());
        self.log.push(
            "  /mouse on|off      — wheel/scrollbar (off = native select/copy)"
                .to_string(),
        );
        self.log
            .push("  /clear             — clear chat log".to_string());
        self.log.push("".to_string());
        self.log.push("═══ Keys ═══".to_string());
        self.log.push(
            "  Enter = send · Shift+Enter / Ctrl+J = newline · Ctrl+C = cancel"
                .to_string(),
        );
        self.log
            .push("  PgUp/PgDn = scroll · Mouse wheel = scroll · Tab = complete"
                .to_string());
    }

    /// Print the current configuration and model status to the chat log.
    fn log_status(&mut self) {
        let mode = match self.run_config.mode {
            AgentMode::Agent => "agent (full tools)",
            AgentMode::Plan => "plan (read-only, no writes)",
            AgentMode::Chat => "chat (no tools, conversation only)",
        };
        self.log.push("═══ Current Status ═══".to_string());
        self.log.push(format!("  Mode:       {mode}"));
        self.log.push(format!(
            "  Model:      {}",
            self.model.as_deref().unwrap_or("auto (best available)")
        ));
        if let Some(info) = &self.model_info {
            self.log.push(format!("  Model info: {info}"));
        }
        self.log.push(format!("  Server:     {}", self.server_url));
        self.log.push(format!(
            "  Workspace:  {}",
            self.run_config.cwd.display()
        ));
        self.log
            .push(format!("  Max steps:  {}", self.run_config.max_steps));
        self.log.push(format!(
            "  Read-only:  {}",
            if self.run_config.is_read_only {
                "ON"
            } else {
                "OFF"
            }
        ));
        self.log.push(format!(
            "  Confirm:    {}",
            if self.run_config.should_confirm {
                "ON"
            } else {
                "OFF (yolo)"
            }
        ));
        self.log.push(format!(
            "  Thinking:   {}",
            if self.run_config.show_thinking {
                "ON"
            } else {
                "OFF"
            }
        ));
        self.log.push(format!(
            "  Mouse:      {}",
            if self.mouse_capture { "ON" } else { "OFF" }
        ));
        self.log.push(format!(
            "  Perf panel: {}",
            if self.show_perf { "ON" } else { "OFF" }
        ));
    }

    /// Open the model picker popup, populated from installed Ollama tags.
    ///
    /// # Parameters
    ///
    /// - `client`: Ollama client for fetching installed model tags.
    async fn show_model_picker(&mut self, client: &OllamaClient) -> anyhow::Result<()> {
        let tags = client.tags().await?;
        if tags.is_empty() {
            self.log
                .push("No models installed. Run /setup.".to_string());
        } else {
            self.open_picker(tags.into_iter().map(|model| model.name).collect());
            self.status = "select model".to_string();
        }
        Ok(())
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

        let tags = client.tags().await?;
        if model_installed(&tags, arg) {
            self.activate_model(client, arg).await;
        } else {
            self.log.push(format!(
                "❌ Model '{arg}' is not installed. Pull it first with /pull {arg}."
            ));
        }
        Ok(())
    }

    /// Print the currently selected model name to the chat log.
    fn log_current_model(&mut self) {
        match &self.model {
            Some(model) => self.log.push(format!("Current model: {model}")),
            None => self
                .log
                .push("Auto mode — best model will be selected on first task.".to_string()),
        }
    }

    /// Set the active model and refresh its metadata from the Ollama server.
    ///
    /// # Parameters
    ///
    /// - `client`: Ollama client for refreshing model info.
    /// - `name`: the model tag to activate.
    async fn activate_model(&mut self, client: &OllamaClient, name: &str) {
        self.model = Some(name.to_string());
        self.log.push(format!("✅ Model set to {name}"));
        self.refresh_model_info(client).await;
    }

    /// Start downloading a model from the Ollama registry in the background.
    ///
    /// # Parameters
    ///
    /// - `client`: Ollama client for the pull operation.
    /// - `arg`: model tag to pull.
    /// - `tx`: channel for progress events.
    async fn handle_pull_command(
        &mut self,
        client: &OllamaClient,
        arg: &str,
        tx: &mpsc::UnboundedSender<UiEvent>,
    ) -> anyhow::Result<()> {
        let name = arg.trim().to_string();
        if name.is_empty() {
            self.log.push("Usage: /pull <model-tag>".to_string());
            return Ok(());
        }
        if self.is_running {
            self.log
                .push("⚠️  Busy — finish the current task before pulling a model.".to_string());
            return Ok(());
        }

        self.log
            .push(format!("⬇️  Pulling model '{name}' in background..."));
        self.start_pull(client, name, tx);
        Ok(())
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
        let tx = tx.clone();
        let handle = tokio::spawn(async move {
            run_pull_worker(client, name, tx).await;
        });
        self.current_task = Some(handle);
    }

    /// Toggle read-only mode on/off, preventing or allowing file writes and commands.
    fn toggle_read_only(&mut self) {
        self.run_config.is_read_only = !self.run_config.is_read_only;
        let state = if self.run_config.is_read_only {
            "ON"
        } else {
            "OFF"
        };
        self.log.push(format!("Read-only mode: {state}"));
    }

    /// Toggle confirmation prompts on/off for writes and shell commands.
    fn toggle_confirm(&mut self) {
        self.run_config.should_confirm = !self.run_config.should_confirm;
        let state = if self.run_config.should_confirm {
            "ON"
        } else {
            "OFF"
        };
        self.log.push(format!("Confirm mode: {state}"));
    }

    /// Switch the agent mode: agent (full tools), plan (read-only), or chat (no tools).
    ///
    /// # Parameters
    ///
    /// - `arg`: one of "agent", "plan", or "chat".
    fn handle_mode_command(&mut self, arg: &str) {
        match arg {
            "agent" => {
                self.run_config.mode = AgentMode::Agent;
                self.log
                    .push("Mode: agent (full tools enabled)".to_string());
            }
            "plan" => {
                self.run_config.mode = AgentMode::Plan;
                self.log
                    .push("Mode: plan (read-only, explores and plans)".to_string());
            }
            "chat" => {
                self.run_config.mode = AgentMode::Chat;
                self.log
                    .push("Mode: chat (no tools, conversation + code)".to_string());
            }
            _ => self.log.push("Usage: /mode agent|plan|chat".to_string()),
        }
    }

    /// Toggle or display the model reasoning/thinking display.
    ///
    /// # Parameters
    ///
    /// - `arg`: "on", "off", or empty to show current state.
    fn handle_thinking_command(&mut self, arg: &str) {
        match arg {
            "on" => {
                self.run_config.show_thinking = true;
                self.log.push("Thinking display: ON".to_string());
            }
            "off" => {
                self.run_config.show_thinking = false;
                self.log.push("Thinking display: OFF".to_string());
            }
            _ => {
                let state = if self.run_config.show_thinking {
                    "ON"
                } else {
                    "OFF"
                };
                self.log.push(format!(
                    "Thinking display: {state} · Usage: /thinking on|off"
                ));
            }
        }
    }

    /// Toggle mouse capture: on for wheel/scrollbar, off for native select/copy.
    ///
    /// # Parameters
    ///
    /// - `arg`: "on", "off", or empty to show current state.
    fn handle_mouse_command(&mut self, arg: &str) {
        match arg {
            "on" => {
                self.mouse_capture = true;
                self.log.push(
                    "Mouse mode: ON (wheel scroll + click-to-open, selection via Shift+drag)"
                        .to_string(),
                );
            }
            "off" => {
                self.mouse_capture = false;
                self.log.push(
                    "Mouse mode: OFF (native select/copy, scroll with PgUp/PgDn)".to_string(),
                );
            }
            _ => {
                let state = if self.mouse_capture { "ON" } else { "OFF" };
                self.log
                    .push(format!("Mouse mode: {state} · Usage: /mouse on|off"));
            }
        }
    }

    /// Disable confirmation prompts ("YOLO mode") for writes and commands.
    fn handle_yolo_command(&mut self) {
        self.run_config.should_confirm = false;
        self.log.push(
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
        match arg.parse::<usize>() {
            Ok(steps) if steps > 0 => {
                self.run_config.max_steps = steps;
                self.log.push(format!("Max steps set to {steps}"));
            }
            _ => self.log.push("Usage: /steps <positive number>".to_string()),
        }
    }

    /// Show or change the agent's working directory.
    ///
    /// # Parameters
    ///
    /// - `arg`: new directory path, or empty to show the current workspace.
    fn handle_cwd_command(&mut self, arg: &str) {
        if arg.is_empty() {
            self.log
                .push(format!("Workspace: {}", self.run_config.cwd.display()));
        } else {
            self.run_config.cwd = PathBuf::from(arg);
            self.log.push(format!(
                "Workspace set to {}",
                self.run_config.cwd.display()
            ));
        }
    }

    /// Run connectivity and model-recommendation diagnostics, logging results.
    ///
    /// # Parameters
    ///
    /// - `client`: Ollama client for checking server status and installed models.
    async fn handle_doctor_command(&mut self, client: &OllamaClient) -> anyhow::Result<()> {
        for line in crate::commands::doctor_lines(client, self.min_context).await? {
            self.log.push(line);
        }
        Ok(())
    }

    /// Run `/setup` (install Ollama + pull recommended model) in the background.
    ///
    /// # Parameters
    ///
    /// - `client`: Ollama client for setup operations.
    /// - `tx`: channel for progress events.
    async fn handle_setup_command(
        &mut self,
        client: &OllamaClient,
        tx: &mpsc::UnboundedSender<UiEvent>,
    ) -> anyhow::Result<()> {
        if self.is_running {
            self.log
                .push("⚠️  Busy — finish the current task before running setup.".to_string());
            return Ok(());
        }

        self.log
            .push("🔄 Running setup in background...".to_string());
        self.is_running = true;
        self.status = "running setup".to_string();
        self.last_action = "setup".to_string();
        self.scroll_mode = ScrollMode::Follow;

        let client = client.clone();
        let tx = tx.clone();
        let handle = tokio::spawn(async move {
            run_setup_worker(client, tx).await;
        });
        self.current_task = Some(handle);
        Ok(())
    }

    /// Clear the chat log, live output, prompt, and conversation history.
    pub(super) fn clear_chat(&mut self) {
        self.log.clear();
        self.live.clear();
        self.current_prompt = None;
        self.chat_history.clear();
        self.scroll_mode = ScrollMode::Follow;
        self.chat_scroll_offset = 0;
    }

    /// Toggle the live GPU/CPU/RAM performance panel on/off.
    fn toggle_perf(&mut self) {
        self.show_perf = !self.show_perf;
        self.log.push(format!(
            "Perf panel: {}",
            if self.show_perf { "ON" } else { "OFF" }
        ));
    }

    /// Log a "command not found" message with a hint to use `/help`.
    ///
    /// # Parameters
    ///
    /// - `name`: the unrecognized command name.
    fn log_unknown_command(&mut self, name: &str) {
        self.log
            .push(format!("Unknown command: /{name}. Try /help"));
    }
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
        app.handle_pull_command(&client, "", &tx).await.unwrap();
        assert!(app.log.iter().any(|line| line.contains("Usage: /pull")));
        assert!(!app.is_running);
        assert!(app.current_task.is_none());
    }

    #[test]
    fn log_help_pushes_command_list() {
        let cli = crate::cli::Cli::parse_from(["openbatrangs"]);
        let mut app = App::new(&cli, Arc::new(Mutex::new(None)));
        app.log.clear();
        app.log_help();
        let text = app.log.text();
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
        app.log.clear();
        app.log_status();
        let text = app.log.text();
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
        assert!(app.log.text().contains("Read-only mode:"));
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
        assert!(app.log.text().contains("Confirm mode:"));
    }

    #[test]
    fn handle_mode_command_sets_agent() {
        let cli = crate::cli::Cli::parse_from(["openbatrangs"]);
        let mut app = App::new(&cli, Arc::new(Mutex::new(None)));
        app.log.clear();
        app.handle_mode_command("agent");
        assert_eq!(app.run_config.mode, AgentMode::Agent);
        assert!(app.log.text().contains("agent (full tools enabled)"));
    }

    #[test]
    fn handle_mode_command_sets_plan() {
        let cli = crate::cli::Cli::parse_from(["openbatrangs"]);
        let mut app = App::new(&cli, Arc::new(Mutex::new(None)));
        app.log.clear();
        app.handle_mode_command("plan");
        assert_eq!(app.run_config.mode, AgentMode::Plan);
        assert!(app.log.text().contains("plan (read-only"));
    }

    #[test]
    fn handle_mode_command_sets_chat() {
        let cli = crate::cli::Cli::parse_from(["openbatrangs"]);
        let mut app = App::new(&cli, Arc::new(Mutex::new(None)));
        app.log.clear();
        app.handle_mode_command("chat");
        assert_eq!(app.run_config.mode, AgentMode::Chat);
        assert!(app.log.text().contains("chat (no tools"));
    }

    #[test]
    fn handle_mode_command_invalid_shows_usage() {
        let cli = crate::cli::Cli::parse_from(["openbatrangs"]);
        let mut app = App::new(&cli, Arc::new(Mutex::new(None)));
        app.log.clear();
        app.handle_mode_command("invalid");
        assert!(app.log.text().contains("Usage: /mode agent|plan|chat"));
    }

    #[test]
    fn handle_steps_command_sets_valid_steps() {
        let cli = crate::cli::Cli::parse_from(["openbatrangs"]);
        let mut app = App::new(&cli, Arc::new(Mutex::new(None)));
        app.log.clear();
        app.handle_steps_command("25");
        assert_eq!(app.run_config.max_steps, 25);
        assert!(app.log.text().contains("Max steps set to 25"));
    }

    #[test]
    fn handle_steps_command_rejects_zero() {
        let cli = crate::cli::Cli::parse_from(["openbatrangs"]);
        let mut app = App::new(&cli, Arc::new(Mutex::new(None)));
        app.log.clear();
        app.handle_steps_command("0");
        assert!(app.log.text().contains("Usage: /steps <positive number>"));
    }

    #[test]
    fn handle_steps_command_rejects_non_numeric() {
        let cli = crate::cli::Cli::parse_from(["openbatrangs"]);
        let mut app = App::new(&cli, Arc::new(Mutex::new(None)));
        app.log.clear();
        app.handle_steps_command("abc");
        assert!(app.log.text().contains("Usage: /steps <positive number>"));
    }

    #[test]
    fn handle_thinking_command_on() {
        let cli = crate::cli::Cli::parse_from(["openbatrangs"]);
        let mut app = App::new(&cli, Arc::new(Mutex::new(None)));
        app.log.clear();
        app.handle_thinking_command("on");
        assert!(app.run_config.show_thinking);
        assert!(app.log.text().contains("Thinking display: ON"));
    }

    #[test]
    fn handle_thinking_command_off() {
        let cli = crate::cli::Cli::parse_from(["openbatrangs"]);
        let mut app = App::new(&cli, Arc::new(Mutex::new(None)));
        app.log.clear();
        app.handle_thinking_command("off");
        assert!(!app.run_config.show_thinking);
        assert!(app.log.text().contains("Thinking display: OFF"));
    }

    #[test]
    fn handle_thinking_command_invalid_shows_current_state() {
        let cli = crate::cli::Cli::parse_from(["openbatrangs"]);
        let mut app = App::new(&cli, Arc::new(Mutex::new(None)));
        app.log.clear();
        app.handle_thinking_command("maybe");
        assert!(app.log.text().contains("Usage: /thinking on|off"));
    }

    #[test]
    fn handle_mouse_command_on() {
        let cli = crate::cli::Cli::parse_from(["openbatrangs"]);
        let mut app = App::new(&cli, Arc::new(Mutex::new(None)));
        app.log.clear();
        app.handle_mouse_command("on");
        assert!(app.mouse_capture);
        assert!(app.log.text().contains("Mouse mode: ON"));
    }

    #[test]
    fn handle_mouse_command_off() {
        let cli = crate::cli::Cli::parse_from(["openbatrangs"]);
        let mut app = App::new(&cli, Arc::new(Mutex::new(None)));
        app.log.clear();
        app.handle_mouse_command("off");
        assert!(!app.mouse_capture);
        assert!(app.log.text().contains("Mouse mode: OFF"));
    }

    #[test]
    fn handle_yolo_disables_confirmation() {
        let cli = crate::cli::Cli::parse_from(["openbatrangs"]);
        let mut app = App::new(&cli, Arc::new(Mutex::new(None)));
        app.run_config.should_confirm = true;
        app.log.clear();
        app.handle_yolo_command();
        assert!(!app.run_config.should_confirm);
        assert!(app.log.text().contains("YOLO mode ON"));
    }

    #[test]
    fn log_unknown_command_shows_message() {
        let cli = crate::cli::Cli::parse_from(["openbatrangs"]);
        let mut app = App::new(&cli, Arc::new(Mutex::new(None)));
        app.log.clear();
        app.log_unknown_command("foobar");
        assert!(app.log.text().contains("Unknown command: /foobar"));
        assert!(app.log.text().contains("/help"));
    }

    #[test]
    fn handle_cwd_command_with_empty_arg_shows_current() {
        let cli = crate::cli::Cli::parse_from(["openbatrangs"]);
        let mut app = App::new(&cli, Arc::new(Mutex::new(None)));
        let original_cwd = app.run_config.cwd.clone();
        app.log.clear();
        app.handle_cwd_command("");
        assert!(app.log.text().contains(&format!("Workspace: {}", original_cwd.display())));
    }

    #[test]
    fn handle_cwd_command_with_path_changes_workspace() {
        let cli = crate::cli::Cli::parse_from(["openbatrangs"]);
        let mut app = App::new(&cli, Arc::new(Mutex::new(None)));
        app.log.clear();
        app.handle_cwd_command("/tmp/new_workspace");
        assert_eq!(app.run_config.cwd, std::path::PathBuf::from("/tmp/new_workspace"));
        assert!(app.log.text().contains("Workspace set to /tmp/new_workspace"));
    }

    #[test]
    fn clear_chat_resets_all_state() {
        let cli = crate::cli::Cli::parse_from(["openbatrangs"]);
        let mut app = App::new(&cli, Arc::new(Mutex::new(None)));
        app.live = "streaming text".to_string();
        app.chat_scroll_offset = 42;
        app.scroll_mode = ScrollMode::Manual;
        app.clear_chat();
        assert_eq!(app.live, "");
        assert_eq!(app.chat_scroll_offset, 0);
        assert_eq!(app.scroll_mode, ScrollMode::Follow);
    }

    #[test]
    fn toggle_perf_flips_display() {
        let cli = crate::cli::Cli::parse_from(["openbatrangs"]);
        let mut app = App::new(&cli, Arc::new(Mutex::new(None)));
        let initial = app.show_perf;
        app.toggle_perf();
        assert_eq!(app.show_perf, !initial);
        assert!(app.log.text().contains("Perf panel:"));
    }
}
