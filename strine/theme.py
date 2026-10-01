from __future__ import annotations

from typing import List, Tuple

from rich.console import Console

CYAN = "#7dcfff"
BLUE = "#7aa2f7"
MAGENTA = "#bb9af7"
GREEN = "#9ece6a"
YELLOW = "#e0af68"
RED = "#f7768e"
ORANGE = "#ff9e64"
DIM = "#565f89"
FG = "#c0caf5"

_ASCII_MARK = [
    ("   ╭─╮ ╭─╮ ╭─╮", CYAN),
    ("   │ ╰─╯ ╰─╯ │", CYAN),
    ("   │  ●   ●  │", BLUE),
    ("   ╰────┬────╯", BLUE),
    ("      ╭─┴─╮", MAGENTA),
    ("      │   │", MAGENTA),
    ("      ╰───╯", MAGENTA),
]

_SWATCHES = [RED, ORANGE, YELLOW, GREEN, CYAN, BLUE, MAGENTA, FG]


def render_banner(console: Console, stats: List[Tuple[str, str]]) -> None:
    """Prints a neofetch-style banner: ascii mark + key/value stats + swatches.

    `stats` is a list of (label, value) pairs; value may already carry
    rich markup (callers color the number themselves).
    """
    ascii_lines = [f"[{color}]{line}[/{color}]" for line, color in _ASCII_MARK]
    stat_lines = [f"[{ORANGE}]{label:<10}[/{ORANGE}]{value}" for label, value in stats]

    height = max(len(ascii_lines), len(stat_lines))
    ascii_lines += [""] * (height - len(ascii_lines))
    stat_lines += [""] * (height - len(stat_lines))

    console.print()
    for ascii_line, stat_line in zip(ascii_lines, stat_lines):
        console.print(f"{ascii_line}   {stat_line}")

    swatch_row = "  ".join(f"[{color}]███[/{color}]" for color in _SWATCHES)
    console.print(f"\n{swatch_row}\n")
