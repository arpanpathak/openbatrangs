//! TUI slash-command handlers.
//!
//! Keeping command parsing and state mutations separate from the main event
//! loop keeps `App` focused on navigation/editing while this module owns the
//! `/command` behaviors.

use super::app::{App, PickerState};
use super::split_command;
use crate::cli::AgentMode;
use crate::ollama::{OllamaClient, OllamaModel};
use std::path::PathBuf;

/// What `/model <name>` should do after checking installed tags.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ModelCommandAction {
    /// The model is already installed; just switch to it.
    Activate,
    /// The model is missing; pull it before switching.
    Pull,
}

/// Decide whether to activate or pull a requested model (pure and testable).
fn model_action(tags: &[OllamaModel], requested: &str) -> ModelCommandAction {
    if tags.iter().any(|model| model.name == requested) {
        ModelCommandAction::Activate
    } else {
        ModelCommandAction::Pull
    }
}

impl App {
    pub(super) async fn run_slash_command(
        &mut self,
        client: &OllamaClient,
        line: &str,
    ) -> anyhow::Result<()> {
        let (name, arg) = split_command(line);
        match name {
            "help" | "h" => self.log_help(),
            "exit" | "quit" => self.should_quit = true,
            "models" => self.show_model_picker(client).await?,
            "model" => self.handle_model_command(client, arg).await?,
            "read-only" => self.toggle_read_only(),
            "confirm" => self.toggle_confirm(),
            "steps" => self.handle_steps_command(arg),
            "cwd" => self.handle_cwd_command(arg),
            "doctor" => self.handle_doctor_command(client).await?,
            "setup" => self.handle_setup_command(client).await?,
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

    fn log_help(&mut self) {
        self.log.push("Commands:".to_string());
        self.log
            .push("  /help, /exit, /quit, /setup, /models".to_string());
        self.log.push(
            "  /model <tag> (auto-pulls if missing), /read-only, /confirm, /perf".to_string(),
        );
        self.log
            .push("  /mode agent|plan|chat, /thinking on|off, /steps <n>, /cwd <path>".to_string());
        self.log.push(
            "  /doctor, /clear, /mouse on|off · Ctrl+C cancel · PgUp/PgDn scroll".to_string(),
        );
        self.log.push(
            "  Mouse select/copy always on by default · /mouse on for wheel/click".to_string(),
        );
        self.log.push(
            "  Confirmations ON by default · /yolo to skip them · /confirm to re-enable"
                .to_string(),
        );
        self.log
            .push("  Shift+Enter / Ctrl+J = new line · Enter = send".to_string());
    }

    async fn show_model_picker(&mut self, client: &OllamaClient) -> anyhow::Result<()> {
        let tags = client.tags().await?;
        if tags.is_empty() {
            self.log
                .push("No models installed. Run /setup.".to_string());
        } else {
            self.picker = Some(PickerState {
                models: tags.into_iter().map(|model| model.name).collect(),
                selected: 0,
            });
            self.status = "select model".to_string();
        }
        Ok(())
    }

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
        match model_action(&tags, arg) {
            ModelCommandAction::Activate => self.activate_model(client, arg).await,
            ModelCommandAction::Pull => self.pull_and_activate_model(client, arg).await,
        }
        Ok(())
    }

    fn log_current_model(&mut self) {
        match &self.model {
            Some(model) => self.log.push(format!("Current model: {model}")),
            None => self
                .log
                .push("Auto mode — best model will be selected on first task.".to_string()),
        }
    }

    async fn activate_model(&mut self, client: &OllamaClient, name: &str) {
        self.model = Some(name.to_string());
        self.log.push(format!("✅ Model set to {name}"));
        self.refresh_model_info(client).await;
    }

    async fn pull_and_activate_model(&mut self, client: &OllamaClient, name: &str) {
        self.log.push(format!("⬇️  Pulling model '{name}'..."));
        let messages = std::sync::Mutex::new(Vec::new());
        let result = client
            .pull(name, &|msg| {
                if let Ok(mut messages) = messages.lock() {
                    messages.push(msg.to_string());
                }
            })
            .await;
        let messages = messages.into_inner().unwrap_or_default();
        self.log.extend(messages);

        match result {
            Ok(()) => {
                self.log.push(format!("✅ Model ready: {name}"));
                self.activate_model(client, name).await;
            }
            Err(error) => self
                .log
                .push(format!("❌ Failed to pull model '{name}': {error:#}")),
        }
    }

    fn toggle_read_only(&mut self) {
        self.run_config.is_read_only = !self.run_config.is_read_only;
        let state = if self.run_config.is_read_only {
            "ON"
        } else {
            "OFF"
        };
        self.log.push(format!("Read-only mode: {state}"));
    }

    fn toggle_confirm(&mut self) {
        self.run_config.should_confirm = !self.run_config.should_confirm;
        let state = if self.run_config.should_confirm {
            "ON"
        } else {
            "OFF"
        };
        self.log.push(format!("Confirm mode: {state}"));
    }

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
                    "Mouse mode: OFF (native select/copy always works, scroll with PgUp/PgDn)"
                        .to_string(),
                );
            }
            _ => {
                let state = if self.mouse_capture { "ON" } else { "OFF" };
                self.log
                    .push(format!("Mouse mode: {state} · Usage: /mouse on|off"));
            }
        }
    }

    fn handle_yolo_command(&mut self) {
        self.run_config.should_confirm = false;
        self.log.push(
            "⚠️  YOLO mode ON — agent writes/commands run without confirmation. Be careful!"
                .to_string(),
        );
    }

    fn handle_steps_command(&mut self, arg: &str) {
        match arg.parse::<usize>() {
            Ok(steps) if steps > 0 => {
                self.run_config.max_steps = steps;
                self.log.push(format!("Max steps set to {steps}"));
            }
            _ => self.log.push("Usage: /steps <positive number>".to_string()),
        }
    }

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

    async fn handle_doctor_command(&mut self, client: &OllamaClient) -> anyhow::Result<()> {
        for line in crate::commands::doctor_lines(client, self.min_context).await? {
            self.log.push(line);
        }
        Ok(())
    }

    async fn handle_setup_command(&mut self, client: &OllamaClient) -> anyhow::Result<()> {
        self.log.push("Running setup...".to_string());
        let messages = std::sync::Mutex::new(Vec::new());
        crate::commands::setup_with_status(client, &|msg| {
            if let Ok(mut messages) = messages.lock() {
                messages.push(msg.to_string());
            }
        })
        .await?;
        let messages = messages.into_inner().unwrap_or_default();
        self.log.extend(messages);
        self.log.push("✅ Setup finished.".to_string());
        Ok(())
    }

    pub(super) fn clear_chat(&mut self) {
        self.log.clear();
        self.live.clear();
        self.current_prompt = None;
        self.chat_history.clear();
    }

    fn toggle_perf(&mut self) {
        self.show_perf = !self.show_perf;
        self.log.push(format!(
            "Perf panel: {}",
            if self.show_perf { "ON" } else { "OFF" }
        ));
    }

    fn log_unknown_command(&mut self, name: &str) {
        self.log
            .push(format!("Unknown command: /{name}. Try /help"));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn installed_model(name: &str) -> OllamaModel {
        serde_json::from_value(serde_json::json!({
            "name": name,
            "size": 123,
        }))
        .expect("valid model JSON")
    }

    #[test]
    fn model_action_activates_installed_models() {
        let tags = vec![installed_model("qwen2.5-coder:7b")];
        assert_eq!(
            model_action(&tags, "qwen2.5-coder:7b"),
            ModelCommandAction::Activate
        );
    }

    #[test]
    fn model_action_pulls_missing_models() {
        let tags = vec![installed_model("qwen2.5-coder:7b")];
        assert_eq!(model_action(&tags, "llama3.2:3b"), ModelCommandAction::Pull);
    }

    #[test]
    fn model_action_pulls_when_no_models_installed() {
        assert_eq!(
            model_action(&[], "qwen2.5-coder:7b"),
            ModelCommandAction::Pull
        );
    }
}
