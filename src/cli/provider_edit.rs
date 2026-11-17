//! Changes to `config.toml` that keep the rest of the file as written.
//!
//! Every change parses the file with `toml_edit`, changes one table, and
//! writes the document back, so comments, order and formatting stay. The
//! file holds API keys, so it is written readable by its owner only.
//!
//! ## References
//!
//! - `toml_edit`: <https://docs.rs/toml_edit/latest/toml_edit/>

use super::providers::{built_in_url, read_config_text};
use crate::constants::cli::SEARCH_ENGINE_URL_KEY;
use anyhow::{Context, Result, anyhow, bail};
use std::fs::{OpenOptions, Permissions};
use std::io::Write;
use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};
use std::path::Path;
use toml_edit::{DocumentMut, Item, Table, value};

/// File mode of the config file: read and write for its owner only.
const PRIVATE_MODE: u32 = 0o600;

/// Check that `name` can name a new provider: it is not empty or a
/// top-level setting, and it has `url` unless the name is built in.
pub(crate) fn check_new_provider(name: &str, url: Option<&str>) -> Result<()> {
    if name.is_empty() || name == SEARCH_ENGINE_URL_KEY {
        bail!("`{name}` cannot name a provider");
    }

    if url.is_none() && built_in_url(name).is_none() {
        bail!("provider `{name}` needs a URL");
    }

    Ok(())
}

/// Add the provider `name` to the file at `path`, or update its table. The
/// URL and the key are written when given; every other setting stays.
pub(crate) fn add_provider(
    path: &Path,
    name: &str,
    url: Option<&str>,
    api_key: Option<&str>,
) -> Result<()> {
    check_new_provider(name, url)?;

    edit_config(path, |document| {
        let block = document
            .entry(name)
            .or_insert(Item::Table(Table::new()))
            .as_table_like_mut()
            .ok_or_else(|| anyhow!("`{name}` in {} is not a table", path.display()))?;

        if let Some(url) = url {
            block.insert("url", value(url));
        }

        if let Some(api_key) = api_key {
            block.insert("api_key", value(api_key));
        }

        Ok(())
    })
}

/// Remove the provider `name` from the file at `path`.
pub(crate) fn remove_provider(path: &Path, name: &str) -> Result<()> {
    edit_config(path, |document| {
        let is_provider = document.get(name).is_some_and(Item::is_table_like);

        if !is_provider || name == SEARCH_ENGINE_URL_KEY {
            bail!("no provider named `{name}` in {}", path.display());
        }

        document.remove(name);

        Ok(())
    })
}

/// Save the thinking and temperature settings of the provider `name` in the
/// file at `path`. A `temperature` of `None` is written as `"auto"`.
pub(crate) fn save_settings(
    path: &Path,
    name: &str,
    think: bool,
    temperature: Option<f64>,
) -> Result<()> {
    let temperature = match temperature {
        Some(number) => value(number),
        None => value("auto"),
    };

    edit_config(path, |document| {
        let block = document
            .get_mut(name)
            .and_then(Item::as_table_like_mut)
            .ok_or_else(|| anyhow!("no provider named `{name}` in {}", path.display()))?;

        block.insert("think", value(think));
        block.insert("temperature", temperature);

        Ok(())
    })
}

/// Parse the file at `path` (empty when missing), apply `edit`, and write
/// the document back, readable by its owner only.
fn edit_config(path: &Path, edit: impl FnOnce(&mut DocumentMut) -> Result<()>) -> Result<()> {
    let text = read_config_text(path)?;
    let mut document: DocumentMut = text
        .parse()
        .with_context(|| format!("{} is not a valid config file", path.display()))?;

    edit(&mut document)?;

    let folder = path.parent().unwrap_or(Path::new("."));
    std::fs::create_dir_all(folder)
        .with_context(|| format!("cannot create {}", folder.display()))?;

    let mut file = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .mode(PRIVATE_MODE)
        .open(path)
        .with_context(|| format!("cannot write {}", path.display()))?;

    file.set_permissions(Permissions::from_mode(PRIVATE_MODE))?;
    file.write_all(document.to_string().as_bytes())?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli::ConfigFile;
    use crate::test_support::{config_file_with, unique_temp_dir};

    /// A file with a comment, a search setting and one provider.
    const COMMENTED: &str = "# my servers\nsearch_engine_url = \"http://localhost:8888\"\n\n\
        [home-gpu] # the desk machine\nurl = \"http://192.168.1.20:8000/v1\"\nthink = false\n";

    /// The error of `outcome`, as text.
    fn message(outcome: Result<()>) -> String {
        format!("{:?}", outcome.err())
    }

    /// The permission bits of the file at `path`.
    fn mode(path: &Path) -> Result<u32> {
        Ok(std::fs::metadata(path)?.permissions().mode() & 0o777)
    }

    #[test]
    fn a_new_file_gets_the_provider_and_owner_only_access() -> Result<()> {
        let path = unique_temp_dir("openbatrangs-edit").join("openbatrangs/config.toml");

        add_provider(&path, "deepseek", None, Some("sk-test"))?;

        let file = ConfigFile::load(&path)?;
        assert_eq!(
            file.providers["deepseek"].url,
            "https://api.deepseek.com/v1"
        );
        assert_eq!(
            file.providers["deepseek"].api_key.as_deref(),
            Some("sk-test")
        );
        assert_eq!(mode(&path)?, PRIVATE_MODE);
        Ok(())
    }

    #[test]
    fn adding_keeps_comments_order_and_other_settings() -> Result<()> {
        let path = config_file_with(COMMENTED)?;
        std::fs::set_permissions(&path, Permissions::from_mode(0o644))?;

        add_provider(&path, "colab", Some("http://localhost:8000/v1"), None)?;
        add_provider(
            &path,
            "home-gpu",
            Some("http://192.168.1.21:8000/v1"),
            Some("new-key"),
        )?;

        let text = std::fs::read_to_string(&path)?;
        assert!(text.starts_with("# my servers\nsearch_engine_url"));
        assert!(text.contains("[home-gpu] # the desk machine"));
        assert!(text.find("[home-gpu]") < text.find("[colab]"));
        assert!(text.contains("think = false"));
        assert!(text.contains("url = \"http://192.168.1.21:8000/v1\""));
        assert!(text.contains("api_key = \"new-key\""));
        assert!(text.contains("[colab]\nurl = \"http://localhost:8000/v1\""));
        assert_eq!(mode(&path)?, PRIVATE_MODE);
        Ok(())
    }

    #[test]
    fn a_new_provider_needs_a_usable_name_and_a_url_unless_built_in() -> Result<()> {
        let path = config_file_with("")?;

        let without_url = message(add_provider(&path, "home-gpu", None, None));
        let empty = message(add_provider(&path, "", Some("http://box/v1"), None));
        let reserved = message(add_provider(
            &path,
            "search_engine_url",
            Some("http://box/v1"),
            None,
        ));

        assert!(without_url.contains("provider `home-gpu` needs a URL"));
        assert!(empty.contains("cannot name a provider"));
        assert!(reserved.contains("`search_engine_url` cannot name a provider"));
        assert_eq!(std::fs::read_to_string(&path)?, "");
        Ok(())
    }

    #[test]
    fn a_key_that_is_not_a_table_cannot_become_a_provider() -> Result<()> {
        let path = config_file_with("box = 1\n")?;

        let outcome = message(add_provider(&path, "box", Some("http://box/v1"), None));

        assert!(outcome.contains("`box` in"));
        assert!(outcome.contains("is not a table"));
        Ok(())
    }

    #[test]
    fn removing_keeps_the_rest_of_the_file() -> Result<()> {
        let path = config_file_with(&format!("{COMMENTED}\n[deepseek]\napi_key = \"k\"\n"))?;

        remove_provider(&path, "deepseek")?;

        let text = std::fs::read_to_string(&path)?;
        assert!(!text.contains("deepseek"));
        assert!(text.contains("[home-gpu] # the desk machine"));
        assert!(text.starts_with("# my servers"));
        Ok(())
    }

    #[test]
    fn removing_a_missing_provider_or_a_setting_is_an_error() -> Result<()> {
        let path = config_file_with(COMMENTED)?;

        let missing = message(remove_provider(&path, "nope"));
        let setting = message(remove_provider(&path, "search_engine_url"));

        assert!(missing.contains("no provider named `nope`"));
        assert!(setting.contains("no provider named `search_engine_url`"));
        assert_eq!(std::fs::read_to_string(&path)?, COMMENTED);
        Ok(())
    }

    #[test]
    fn settings_are_saved_to_the_provider_table() -> Result<()> {
        let path = config_file_with(COMMENTED)?;

        save_settings(&path, "home-gpu", true, Some(0.3))?;
        let numbered = ConfigFile::load(&path)?;
        save_settings(&path, "home-gpu", false, None)?;
        let automatic = std::fs::read_to_string(&path)?;

        assert_eq!(numbered.providers["home-gpu"].think, Some(true));
        assert_eq!(numbered.providers["home-gpu"].temperature, Some(0.3));
        assert!(automatic.contains("think = false"));
        assert!(automatic.contains("temperature = \"auto\""));
        assert!(automatic.contains("# the desk machine"));
        Ok(())
    }

    #[test]
    fn saving_settings_needs_the_provider_and_a_valid_file() -> Result<()> {
        let path = config_file_with(COMMENTED)?;
        let malformed = config_file_with("[home-gpu\n")?;

        let missing = message(save_settings(&path, "nope", true, None));
        let invalid = message(save_settings(&malformed, "home-gpu", true, None));

        assert!(missing.contains("no provider named `nope`"));
        assert!(invalid.contains("is not a valid config file"));
        Ok(())
    }

    #[test]
    fn a_read_only_file_is_an_error_that_names_it() -> Result<()> {
        let path = config_file_with("")?;
        std::fs::set_permissions(&path, Permissions::from_mode(0o400))?;

        let outcome = message(add_provider(&path, "deepseek", None, None));

        assert!(outcome.contains(&format!("cannot write {}", path.display())));
        Ok(())
    }

    #[test]
    fn an_unwritable_folder_is_an_error_that_names_it() -> Result<()> {
        let blocker = config_file_with("")?;
        let path = blocker.join("config.toml");

        let outcome = message(add_provider(&path, "deepseek", None, None));

        assert!(outcome.contains(&format!("cannot create {}", blocker.display())));
        Ok(())
    }
}
