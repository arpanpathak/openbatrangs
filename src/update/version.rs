//! Release versions, and the comparison that decides whether a release is
//! newer than the running binary.

use std::fmt;

/// A release version, compared by major, then minor, then patch number.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct Version {
    /// The first number, 0 in `0.3.1`.
    major: u64,
    /// The second number, 3 in `0.3.1`.
    minor: u64,
    /// The third number, 1 in `0.3.1`.
    patch: u64,
}

impl Version {
    /// Read a tag such as `v0.3.0` or `0.3.0`.
    ///
    /// A tag with a pre-release suffix such as `0.3.0-rc1`, or with other
    /// than three numbers, gives `None`.
    pub(crate) fn parse(tag: &str) -> Option<Self> {
        let digits = tag.strip_prefix('v').unwrap_or(tag);
        let numbers = digits
            .split('.')
            .map(str::parse)
            .collect::<Result<Vec<u64>, _>>();

        match numbers.as_deref() {
            Ok(&[major, minor, patch]) => Some(Self {
                major,
                minor,
                patch,
            }),
            _ => None,
        }
    }
}

impl fmt::Display for Version {
    /// Write the version as `major.minor.patch`, without a `v`.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}

/// A release that is newer than the running binary.
#[derive(Debug, PartialEq)]
pub(crate) struct NewRelease {
    /// Version of the newer release.
    pub(crate) latest: Version,
    /// Version of the running binary, as `Cargo.toml` gives it.
    pub(crate) current: String,
}

impl NewRelease {
    /// `openbatrangs 0.3.0 is available (you have 0.2.2).`
    pub(crate) fn available_line(&self) -> String {
        format!(
            "openbatrangs {} is available (you have {}).",
            self.latest, self.current
        )
    }

    /// The line printed after a one-shot task.
    pub(crate) fn one_shot_line(&self) -> String {
        format!("{} Run: openbatrangs update", self.available_line())
    }

    /// The question the TUI asks before it installs the release.
    pub(crate) fn install_question(&self) -> String {
        format!("{} Install it now? y/n", self.available_line())
    }

    /// The line shown when the installer starts.
    pub(crate) fn installing_line(&self) -> String {
        format!("Installing openbatrangs {}.", self.latest)
    }

    /// The line shown after the installer succeeds.
    pub(crate) fn restart_line(&self) -> String {
        format!("Restart openbatrangs to use {}.", self.latest)
    }
}

/// The release that `latest_tag` names when it is strictly newer than
/// `current`. A tag or version that does not parse gives `None`.
pub(crate) fn newer_release(latest_tag: &str, current: &str) -> Option<NewRelease> {
    let versions = Version::parse(latest_tag).zip(Version::parse(current));

    versions
        .filter(|(latest, running)| latest > running)
        .map(|(latest, _running)| NewRelease {
            latest,
            current: current.to_string(),
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Whether `latest_tag` counts as newer than `current`.
    fn is_newer(latest_tag: &str, current: &str) -> bool {
        newer_release(latest_tag, current).is_some()
    }

    #[test]
    fn a_tag_parses_with_or_without_the_v() {
        assert_eq!(Version::parse("v0.3.0"), Version::parse("0.3.0"));
        assert_eq!(
            Version::parse("v1.20.3").map(|version| version.to_string()),
            Some("1.20.3".to_string())
        );
    }

    #[test]
    fn a_tag_that_is_not_three_numbers_does_not_parse() {
        for tag in ["", "v", "0.3", "0.3.0.1", "0.3.0-rc1", "0.3.x", "latest"] {
            assert_eq!(Version::parse(tag), None, "{tag}");
        }
    }

    #[test]
    fn only_a_strictly_newer_release_counts() {
        assert!(is_newer("v0.3.0", "0.2.2"));
        assert!(is_newer("v0.2.10", "0.2.9"));
        assert!(is_newer("v1.0.0", "0.99.99"));

        assert!(!is_newer("v0.2.2", "0.2.2"));
        assert!(!is_newer("v0.2.1", "0.2.2"));
        assert!(!is_newer("v0.1.9", "0.2.0"));
    }

    #[test]
    fn a_pre_release_or_unparsable_tag_is_not_newer() {
        assert!(!is_newer("v0.3.0-rc1", "0.2.2"));
        assert!(!is_newer("nightly", "0.2.2"));
        assert!(!is_newer("v0.3.0", "0.2.2-dev"));
    }

    #[test]
    fn the_lines_name_both_versions() -> anyhow::Result<()> {
        let release = newer_release("v0.3.0", "0.2.2").ok_or(anyhow::anyhow!("not newer"))?;

        assert_eq!(
            release.one_shot_line(),
            "openbatrangs 0.3.0 is available (you have 0.2.2). Run: openbatrangs update"
        );
        assert_eq!(
            release.install_question(),
            "openbatrangs 0.3.0 is available (you have 0.2.2). Install it now? y/n"
        );
        assert_eq!(release.installing_line(), "Installing openbatrangs 0.3.0.");
        assert_eq!(release.restart_line(), "Restart openbatrangs to use 0.3.0.");
        Ok(())
    }
}
