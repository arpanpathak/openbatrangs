"""Draw assets/logo.svg as rows of quarter-block characters.

Each terminal cell holds 2 x 2 pixels, drawn with one of the 16 quarter-block
characters. A cell is about twice as tall as it is wide, so the canvas is
rendered at half its height: the pixels then look square on screen. The blade
is rendered without its background, its alpha is cut at one half, and the rows
are cropped to the blade.

Usage: python3 scripts/logo_cells.py assets/logo.svg COLUMNS
Prints the first canvas column of the art, then one row per line.
"""

import io
import re
import sys
from dataclasses import dataclass

import cairosvg  # type: ignore[import-untyped]
from PIL import Image

CANVAS = 512

# Quarter blocks by the pixels they fill, written as four bits in the order
# top-left, top-right, bottom-left, bottom-right.
QUARTERS = {
    "0000": " ",
    "1000": "▘",
    "0100": "▝",
    "1100": "▀",
    "0010": "▖",
    "1010": "▌",
    "0110": "▞",
    "1110": "▛",
    "0001": "▗",
    "1001": "▚",
    "0101": "▐",
    "1101": "▜",
    "0011": "▄",
    "1011": "▙",
    "0111": "▟",
    "1111": "█",
}

BACKGROUND = re.compile(r"<rect [^>]*/>")


@dataclass(frozen=True)
class Art:
    """Rows of quarter blocks and the canvas column where they start."""

    first_column: int
    rows: list[str]


def blade_pixels(svg: str, columns: int) -> Image.Image:
    """The blade alone, as a black-and-white image of 2 x 2 pixels per cell,
    squashed to half height so that a cell's pixels look square."""
    without_background = BACKGROUND.sub("", svg).replace(
        'viewBox="0 0 512 512"', 'viewBox="0 0 512 512" preserveAspectRatio="none"'
    )
    png = cairosvg.svg2png(
        bytestring=without_background.encode(),
        output_width=columns * 2,
        output_height=columns,
    )
    alpha = Image.open(io.BytesIO(png)).getchannel("A")

    return alpha.point(lambda value: 255 if value >= 128 else 0)


def pixel_bit(pixels: Image.Image, corner: tuple[int, int]) -> str:
    """"1" where the blade covers the pixel, "0" elsewhere."""
    return "1" if pixels.getpixel(corner) == 255 else "0"


def cell(pixels: Image.Image, column: int, row: int) -> str:
    """The quarter block for the 2 x 2 pixels of one cell."""
    x = column * 2
    y = row * 2
    corners = [(x, y), (x + 1, y), (x, y + 1), (x + 1, y + 1)]
    filled = "".join(map(lambda corner: pixel_bit(pixels, corner), corners))

    return QUARTERS[filled]


def all_rows(pixels: Image.Image, columns: int) -> list[str]:
    """Every row of cells across the whole canvas."""
    return [
        "".join(cell(pixels, column, row) for column in range(columns))
        for row in range(columns // 2)
    ]


def holds_blade(row: str) -> bool:
    """Whether a row holds part of the blade."""
    return row.strip() != ""


def leading_spaces(row: str) -> int:
    """The number of blank cells before the blade starts in a row."""
    return len(row) - len(row.lstrip())


def drawn_width(row: str) -> int:
    """The width of a row up to its last drawn cell."""
    return len(row.rstrip())


def art(svg: str, columns: int) -> Art:
    """The blade's rows, cropped to the blade on every side; every row has
    the same width."""
    rows = list(filter(holds_blade, all_rows(blade_pixels(svg, columns), columns)))
    first = min(map(leading_spaces, rows))
    last = max(map(drawn_width, rows))

    return Art(first_column=first, rows=[row[first:last] for row in rows])


def main() -> None:
    """Print the art for the logo and width given on the command line."""
    with open(sys.argv[1], encoding="utf-8") as logo:
        svg = logo.read()
    drawn = art(svg, int(sys.argv[2]))

    print(drawn.first_column)
    print("\n".join(drawn.rows))


if __name__ == "__main__":
    main()
