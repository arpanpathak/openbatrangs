//! Terminal banner: colorful Batman ASCII art with a random quote on each open.
//!
//! The banner uses ANSI colors on stdout. The TUI strips ANSI codes before
//! rendering so the ASCII art itself stays aligned inside ratatui.

use crate::constants::ansi::{COLOR_BOLD, COLOR_CYAN, COLOR_MAGENTA, COLOR_RESET};
use crate::constants::banner::{
    BATMAN_ART, BATMAN_QUOTES, DEFAULT_QUOTE, OPENBATARANGS_ART, QUOTE_COLORS,
};

/// Returns the startup banner as a string.
///
/// # Returns
/// Multi-line string containing the Batman logo and a random Batman quote.
pub fn banner_text() -> String {
    let quote_index = random_quote_index();
    let quote = BATMAN_QUOTES
        .get(quote_index)
        .copied()
        .unwrap_or(DEFAULT_QUOTE);
    let quote_color = QUOTE_COLORS[quote_index % QUOTE_COLORS.len()];

    format!(
        "{COLOR_CYAN}{COLOR_BOLD}{OPENBATARANGS_ART}{COLOR_RESET}\n\
         {COLOR_MAGENTA}{COLOR_BOLD}{BATMAN_ART}{COLOR_RESET}\n\
         {quote_color}🦇 {quote}{COLOR_RESET}\n"
    )
}

/// Pick a quote index for this process using the current time as entropy.
fn random_quote_index() -> usize {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.subsec_nanos() as usize)
        .unwrap_or(0);
    nanos % BATMAN_QUOTES.len()
}

/// Print the startup banner to stdout.
pub fn print_banner() {
    print!("{}", banner_text());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn has_non_empty_batman_quotes() {
        assert!(!BATMAN_QUOTES.is_empty());
        assert!(BATMAN_QUOTES.iter().all(|quote| !quote.is_empty()));
    }

    #[test]
    fn banner_contains_art_and_quote() {
        let banner = banner_text();
        assert!(banner.contains('⣿') || banner.contains('█'));
        assert!(banner.contains('🦇'));
    }

    #[test]
    fn quote_index_is_always_in_range() {
        for _ in 0..100 {
            assert!(random_quote_index() < BATMAN_QUOTES.len());
        }
    }
}
