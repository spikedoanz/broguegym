"""ANSI rendering helpers for Gym observations."""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray

from bruhogue.brogue import ObservationDict

type TerminalCharset = Literal["ascii", "unicode"]

ASCII_GLYPHS: dict[int, str] = {
    128: "^",
    129: "v",
    196: ".",
    197: ":",
    198: "%",
    199: "^",
    200: "&",
    201: ",",
    202: "?",
    203: "=",
    204: "(",
    205: "*",
    206: "0",
    207: "$",
    208: "+",
    209: "<",
    210: "7",
    217: "5",
    218: "5",
    219: "o",
    225: "&",
    226: ".",
    227: "U",
    228: "+",
    231: ".",
    238: "5",
    239: "5",
    245: ":",
    248: "$",
    258: "<",
}


def observation_to_ansi(
    observation: ObservationDict,
    *,
    charset: TerminalCharset = "ascii",
) -> str:
    glyphs = observation["glyphs"]
    chars = observation["chars"]
    colors_fg = observation["colors_fg"]
    colors_bg = observation["colors_bg"]
    if not bool(np.any(colors_fg)) and not bool(np.any(colors_bg)):
        return _chars_to_text(glyphs, chars, charset)
    return _chars_to_ansi(glyphs, chars, colors_fg, colors_bg, charset)


def _terminal_char(glyph: int, codepoint: int, *, charset: TerminalCharset) -> str:
    if charset == "unicode":
        return chr(codepoint) if codepoint else " "
    if glyph in ASCII_GLYPHS:
        return ASCII_GLYPHS[glyph]
    if codepoint == 0:
        return " "
    if 0x20 <= codepoint <= 0x7E:
        return chr(codepoint)
    return "?"


def _chars_to_text(
    glyphs: NDArray[np.generic],
    chars: NDArray[np.generic],
    charset: TerminalCharset,
) -> str:
    rows: list[str] = []
    for row_index, row in enumerate(chars):
        glyph_row = glyphs[row_index].astype(int)
        rows.append(
            "".join(
                _terminal_char(int(glyph_row[col_index]), int(value), charset=charset)
                for col_index, value in enumerate(row.astype(int))
            )
        )
    return "\n".join(rows)


def _chars_to_ansi(
    glyphs: NDArray[np.generic],
    chars: NDArray[np.generic],
    colors_fg: NDArray[np.generic],
    colors_bg: NDArray[np.generic],
    charset: TerminalCharset,
) -> str:
    zero = (0, 0, 0)
    rows: list[str] = []
    for row_index, row in enumerate(chars):
        active = False
        current_fg = zero
        current_bg = zero
        pieces: list[str] = []
        glyph_row = glyphs[row_index].astype(int)
        for col_index, value in enumerate(row.astype(int)):
            fg = _rgb_at(colors_fg, row_index, col_index)
            bg = _rgb_at(colors_bg, row_index, col_index)
            colored = fg != zero or bg != zero
            if colored and (not active or fg != current_fg or bg != current_bg):
                pieces.append(_ansi_color(fg, bg))
                active = True
                current_fg = fg
                current_bg = bg
            elif not colored and active:
                pieces.append("\x1b[0m")
                active = False
                current_fg = zero
                current_bg = zero
            pieces.append(_terminal_char(int(glyph_row[col_index]), int(value), charset=charset))
        if active:
            pieces.append("\x1b[0m")
        rows.append("".join(pieces))
    return "\n".join(rows)


def _rgb_at(
    colors: NDArray[np.generic],
    row_index: int,
    col_index: int,
) -> tuple[int, int, int]:
    return (
        int(colors[row_index, col_index, 0]),
        int(colors[row_index, col_index, 1]),
        int(colors[row_index, col_index, 2]),
    )


def _ansi_color(fg: tuple[int, int, int], bg: tuple[int, int, int]) -> str:
    return (
        f"\x1b[38;2;{fg[0]};{fg[1]};{fg[2]};"
        f"48;2;{bg[0]};{bg[1]};{bg[2]}m"
    )
