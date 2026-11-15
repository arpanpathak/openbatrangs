//! Startup banner art: the logo over the wordmark, and the colors of its gradient.

/// The logo from `assets/logo.svg` drawn in quarter-block characters, and
/// where it sits on the SVG canvas, so each column takes the SVG's color.
///
/// `scripts/logo_cells.py` draws the rows; rerun it after the logo changes.
pub struct LogoArt {
    /// Width of the whole SVG canvas, in columns.
    pub canvas_columns: usize,
    /// The canvas column of the rows' first character.
    pub first_column: usize,
    /// The rows from top to bottom, all of the same width.
    pub rows: &'static [&'static str],
}

/// The logo above the wordmark on the start screen: `logo_cells.py logo.svg 60`.
pub const LOGO: LogoArt = LogoArt {
    canvas_columns: 60,
    first_column: 5,
    rows: &[
        "                                             ▗▄▄",
        "                          ▗▖            ▗▄▄██▛▘ ",
        "                   ▙      ██▖      ▗▄▄██████▛   ",
        "                   ██▖  ▄▄█████████████████▛    ",
        "                  ▄████████████████████▘        ",
        "          ▄▄▄▄▄▄███████▛▀▀▜███████████▛         ",
        "▄▟█████████████████████▌   ███████▌             ",
        "   ▀▜████▛▜█████████████▄▄▟███▛▜██              ",
        "      ▀▘   ▝▜█▛▀▀████████████▀                  ",
        "                  ▀█▘  ▝▀▜██▘                   ",
        "                          ▝▘                    ",
    ],
};

/// The logo in the corner of the stats panel: `logo_cells.py logo.svg 28`.
pub const SMALL_LOGO: LogoArt = LogoArt {
    canvas_columns: 28,
    first_column: 2,
    rows: &[
        "                    ▗▄",
        "         ▄  ▐▙▄▄▄▄███▘",
        "       ▄▄███▜█████▌   ",
        "▝▀▜████████▄▄███▘     ",
        "      ▀ ▀▛▀▀█▛        ",
    ],
};

/// The name in block letters, three rows tall.
pub const WORDMARK: &[&str] = &[
    " ▄▀▀▄ █▀▀▄ █▀▀ █▄ █   █▀▀▄ ▄▀▀▄ ▀█▀ ▄▀▀▄ █▀▀▄ ▄▀▀▄ █▄ █ ▄▀▀▀ ▄▀▀▀",
    " █  █ █▀▀  █▀▀ █ ▀█   █▀▀▄ █▀▀█  █  █▀▀█ █▀▀▄ █▀▀█ █ ▀█ █ ▀█  ▀▀▄",
    "  ▀▀  ▀    ▀▀▀ ▀  ▀   ▀▀▀  ▀  ▀  ▀  ▀  ▀ ▀  ▀ ▀  ▀ ▀  ▀  ▀▀  ▀▀▀",
];

/// Gradient colors from the left edge to the right edge: cyan, violet, pink,
/// the stops of the blade's gradient in `assets/logo.svg`.
pub const GRADIENT_STOPS: [[u8; 3]; 3] = [[0, 220, 255], [150, 100, 255], [255, 80, 180]];

/// Text under the wordmark, followed by the version.
pub const TAGLINE: &str = "local coding agent";
