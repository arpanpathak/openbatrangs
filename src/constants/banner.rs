//! Startup banner art: a batarang over the wordmark, and the colors of its gradient.

/// Left half of the batarang, 20 columns per row. Each row is mirrored to draw
/// the right half, so the shape stays symmetric.
pub const BATARANG_LEFT_HALF: &[&str] = &[
    "▄▄▄▄▄               ",
    " ▀██████▄▄        ▄ ",
    "    ▀▀█████████▄▄███",
    "         ▀▀▀████████",
    "              ▀▀▀▀▀▀",
];

/// The name in block letters, three rows tall.
pub const WORDMARK: &[&str] = &[
    " ▄▀▀▄ █▀▀▄ █▀▀ █▄ █   █▀▀▄ ▄▀▀▄ ▀█▀ ▄▀▀▄ █▀▀▄ ▄▀▀▄ █▄ █ ▄▀▀▀ ▄▀▀▀",
    " █  █ █▀▀  █▀▀ █ ▀█   █▀▀▄ █▀▀█  █  █▀▀█ █▀▀▄ █▀▀█ █ ▀█ █ ▀█  ▀▀▄",
    "  ▀▀  ▀    ▀▀▀ ▀  ▀   ▀▀▀  ▀  ▀  ▀  ▀  ▀ ▀  ▀ ▀  ▀ ▀  ▀  ▀▀  ▀▀▀",
];

/// Gradient colors from the left edge to the right edge: cyan, violet, pink.
pub const GRADIENT_STOPS: [[u8; 3]; 3] = [[0, 220, 255], [150, 100, 255], [255, 80, 180]];

/// Text under the wordmark, followed by the version.
pub const TAGLINE: &str = "local coding agent";
