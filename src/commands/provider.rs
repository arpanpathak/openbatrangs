//! `openbatrangs provider`: list, add and remove the providers in the
//! config file, for scripts and for setup without the TUI.

use crate::cli::{ConfigFile, ProviderAction, add_provider, check_new_provider, remove_provider};
use anyhow::Result;
use std::io::{BufRead, IsTerminal};
use std::path::Path;

/// Run `action` on the config file at `path`.
pub(crate) fn provider(action: &ProviderAction, path: &Path) -> Result<()> {
    match action {
        ProviderAction::List => list(path),
        ProviderAction::Add { name, url } => add(path, name, url.as_deref()),
        ProviderAction::Remove { name } => remove(path, name),
    }
}

/// Print the providers in the file, without their keys.
fn list(path: &Path) -> Result<()> {
    let config = ConfigFile::load(path)?;

    println!("{}", config.lines().join("\n"));

    Ok(())
}

/// Read a key for `name` and write its table into the file.
fn add(path: &Path, name: &str, url: Option<&str>) -> Result<()> {
    check_new_provider(name, url)?;

    let terminal = std::io::stdin().is_terminal();
    let key = read_api_key(name, terminal, &mut std::io::stdin().lock())?;
    add_provider(path, name, url, key.as_deref())?;

    println!("Added {name} to {}", path.display());

    Ok(())
}

/// Remove the table of `name` from the file.
fn remove(path: &Path, name: &str) -> Result<()> {
    remove_provider(path, name)?;

    println!("Removed {name} from {}", path.display());

    Ok(())
}

/// The API key for `name`: typed on the terminal without echo, or the first
/// line of `input` when standard input is not a terminal. An empty answer
/// means no key.
fn read_api_key(name: &str, terminal: bool, input: &mut impl BufRead) -> Result<Option<String>> {
    let key = match terminal {
        true => rpassword::prompt_password(format!("API key for {name} (Enter for none): "))?,
        false => {
            let mut line = String::new();
            input.read_line(&mut line)?;
            line
        }
    };
    let key = key.trim();

    Ok((!key.is_empty()).then(|| key.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::config_file_with;

    #[test]
    fn a_piped_key_is_the_first_line_and_an_empty_line_is_no_key() -> Result<()> {
        let key = read_api_key("deepseek", false, &mut "  sk-test \nnext\n".as_bytes())?;
        let empty = read_api_key("home-gpu", false, &mut "\n".as_bytes())?;

        assert_eq!(key.as_deref(), Some("sk-test"));
        assert_eq!(empty, None);
        Ok(())
    }

    #[test]
    fn list_and_remove_work_on_the_file() -> Result<()> {
        let path = config_file_with("[deepseek]\napi_key = \"k\"\n[openai]\n")?;

        provider(&ProviderAction::List, &path)?;
        provider(
            &ProviderAction::Remove {
                name: "openai".to_string(),
            },
            &path,
        )?;

        let config = ConfigFile::load(&path)?;
        assert!(config.providers.contains_key("deepseek"));
        assert!(!config.providers.contains_key("openai"));
        Ok(())
    }

    #[test]
    fn adding_checks_the_name_before_asking_for_a_key() -> Result<()> {
        let path = config_file_with("")?;
        let action = ProviderAction::Add {
            name: "home-gpu".to_string(),
            url: None,
        };

        let outcome = provider(&action, &path);

        assert!(format!("{:?}", outcome.err()).contains("needs a URL"));
        Ok(())
    }
}
