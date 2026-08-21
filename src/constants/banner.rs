//! Banner display constants: wordmark, ASCII art, and startup quotes.

use crate::constants::ansi::{COLOR_CYAN, COLOR_GREEN, COLOR_MAGENTA, COLOR_YELLOW};

/// Colors cycled through for the random quote, one per launch.
pub const QUOTE_COLORS: &[&str] = &[COLOR_YELLOW, COLOR_CYAN, COLOR_GREEN, COLOR_MAGENTA];

/// Pixel-art wordmark for `openBatarangs` (compact title line).
pub const OPENBATARANGS_ART: &str = "🦇 OPEN-BATARANGS!";

/// Compact Batman pixel art used in the startup banner.
pub const BATMAN_ART: &str = r#"⣿⣿⣟⣛⠛⠛⠛⠛⠛⠛⠛⣿⣿⣿⢿⣿⣿⣿⡟⠛⠛⠛⠛⠛⠛⢛⣛⣿⣿⣿
⣿⣿⣿⣿⣷⣦⠀⠀⠀⠀⠀⠙⠻⠟⠈⠙⠿⠛⠁⠀⠀⠀⠀⢠⣶⣿⣿⣿⣿⣿
⣿⣿⣿⣿⣿⣿⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣿⣿⣿⣿⣿⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣶⣦⣀⠀⢀⣴⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣆⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
"#;

/// Random Batman quotes shown at startup.
pub const BATMAN_QUOTES: &[&str] = &[
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

/// Quote shown when the randomized index is somehow out of range.
pub const DEFAULT_QUOTE: &str = "I'm Batman.";
