//! Terminal banner: colorful Batman ASCII art with a random quote on each open.
//!
//! The banner uses ANSI colors on stdout. The TUI strips ANSI codes before
//! rendering so the ASCII art itself stays aligned inside ratatui.

const COLOR_CYAN: &str = "\x1b[36m";
const COLOR_PURPLE: &str = "\x1b[35m";
const COLOR_YELLOW: &str = "\x1b[33m";
const COLOR_GREEN: &str = "\x1b[32m";
const COLOR_BOLD: &str = "\x1b[1m";
const COLOR_RESET: &str = "\x1b[0m";

/// Colors cycled through for the random quote, one per launch.
const QUOTE_COLORS: &[&str] = &[COLOR_YELLOW, COLOR_CYAN, COLOR_GREEN, COLOR_PURPLE];

/// Pixel-art wordmark for `openBatarangs`.
const OPENBATARANGS_ART: &str = r#" █  ██  ███ █ █ ██   █  ███  █  ██   █  █ █  ██  ██
█ █ █ █ █   ███ █ █ █ █  █  █ █ █ █ █ █ ███ █   █  
█ █ ██  ██  █ █ ██  ███  █  ███ ██  ███ █ █ █ █  █ 
█ █ █   █   █ █ █ █ █ █  █  █ █ █ █ █ █ █ █ █ █   █
 █  █   ███ █ █ ██  █ █  █  █ █ █ █ █ █ █ █  ██ ██ "#;

/// Random Batman quotes shown at startup.
const BATMAN_QUOTES: &[&str] = &[
    "I am vengeance. I am the night. I am Batman!",
    "It's not who I am underneath, but what I do that defines me.",
    "The night is darkest just before the dawn.",
    "Why do we fall, sir? So that we can learn to pick ourselves up.",
    "Our greatest glory is not in never falling, but in rising every time we fall.",
    "You either die a hero, or you live long enough to see yourself become the villain.",
    "A hero can be anyone, even a man doing something as simple and reassuring as putting a coat around a young boy's shoulders.",
    "I wear a mask. And that mask is not to hide who I am, but to create what I am.",
    "Criminals are a superstitious, cowardly lot.",
    "Endure, Master Wayne. Take it. They'll hate you for it, but that's the point of Batman.",
];

/// Returns the startup banner as a string.
///
/// # Returns
/// Multi-line string containing the Batman logo and a random Batman quote.
pub fn banner_text() -> String {
    let batman = r#"          ______
       _-'      '-_
     -'            '-
    /   |      |    \
   |  __|  __  |__   |
   | |  | |  | |  |  |
   | |  | |  | |  |  |
   | |__| |__| |__|  |
    \                /
     '-.__________.-'
"#;

    let quote_index = random_quote_index();
    let quote = BATMAN_QUOTES
        .get(quote_index)
        .copied()
        .unwrap_or("I'm Batman.");
    let quote_color = QUOTE_COLORS[quote_index % QUOTE_COLORS.len()];

    format!(
        "{COLOR_CYAN}{COLOR_BOLD}{OPENBATARANGS_ART}{COLOR_RESET}\n\
         {COLOR_PURPLE}{COLOR_BOLD}{batman}{COLOR_RESET}\n\
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
        assert!(banner.contains('█'));
        assert!(banner.contains('_'));
    }

    #[test]
    fn quote_index_is_always_in_range() {
        for _ in 0..100 {
            assert!(random_quote_index() < BATMAN_QUOTES.len());
        }
    }
}
