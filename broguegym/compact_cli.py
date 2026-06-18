"""Interactive compact-observation viewer for the Brogue bridge."""

from __future__ import annotations

import argparse
import ctypes
import curses
import secrets
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_BRIDGE_ABI_VERSION = 11
_COMPACT_COLS = 79
_COMPACT_ROWS = 29
_COMPACT_CELLS = _COMPACT_COLS * _COMPACT_ROWS
_BLSTATS_SIZE = 21
_PROGRAM_STATE_SIZE = 8
_COMPACT_AGENT_BYTES = _COMPACT_CELLS + _BLSTATS_SIZE * 4 + _PROGRAM_STATE_SIZE * 4
_GYM_ZERO_BRIDGE_SEED = 0x9E3779B97F4A7C15

_PACKAGE_ROOT = Path(__file__).resolve().parent
_PACKAGED_DATA_DIR = _PACKAGE_ROOT / "_native" / "bin"
_SOURCE_DATA_DIR = _PACKAGE_ROOT.parent / "BrogueCE" / "bin"

_BLSTAT_NAMES = (
    "x",
    "y",
    "str",
    "hp",
    "max_hp",
    "depth",
    "gold",
    "turn",
    "abs_turn",
    "stealth",
    "nutrition",
)
_PROGRAM_STATE_NAMES = (
    "turn",
    "terminated",
    "won",
    "depth",
    "seed",
    "gold",
    "score",
    "in_progress",
)


class _CCompactObservation(ctypes.Structure):
    _fields_ = [
        ("chars", ctypes.c_uint8 * _COMPACT_CELLS),
        ("blstats", ctypes.c_int32 * _BLSTATS_SIZE),
        ("program_state", ctypes.c_int32 * _PROGRAM_STATE_SIZE),
    ]


@dataclass
class _StepResult:
    invalid: bool = False
    key: str | None = None


class _CompactBrogue:
    def __init__(self, *, library_path: Path, data_dir: Path) -> None:
        self.library_path = library_path
        self.data_dir = data_dir
        self.observation = _CCompactObservation()
        self._library: Any | None = None

    def __enter__(self) -> _CompactBrogue:
        self.load()
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def load(self) -> None:
        if self._library is not None:
            return
        if not self.library_path.is_file():
            msg = f"Brogue bridge library does not exist: {self.library_path}"
            raise RuntimeError(msg)
        if not self.data_dir.is_dir():
            msg = f"Brogue data directory does not exist: {self.data_dir}"
            raise RuntimeError(msg)

        library = ctypes.CDLL(str(self.library_path))
        _configure_library(library)
        _validate_bridge(library)
        library.brh_set_data_dir(str(self.data_dir).encode("utf-8"))
        self._library = library

    def reset(self, seed: int | None) -> None:
        rc = self._require_library().brh_reset_compact(
            ctypes.c_uint64(_bridge_seed(seed)),
            ctypes.byref(self.observation),
        )
        self._raise_if_failed(rc, "reset")

    def step(self, key: str, *, control: bool = False, shift: bool = False) -> _StepResult:
        rc = self._require_library().brh_step_compact(
            _bridge_key(key),
            int(control),
            int(shift),
            ctypes.byref(self.observation),
        )
        if rc < 0:
            self._raise_if_failed(rc, "step")
        return _StepResult(invalid=rc == 1, key=key)

    def close(self) -> None:
        if self._library is not None:
            self._library.brh_close()

    def _require_library(self) -> Any:
        if self._library is None:
            msg = "Brogue compact bridge is not loaded"
            raise RuntimeError(msg)
        return self._library

    def _raise_if_failed(self, rc: int, operation: str) -> None:
        if rc == 0:
            return
        raw_error = self._require_library().brh_last_error()
        detail = raw_error.decode("utf-8", errors="replace") if raw_error else "unknown error"
        msg = f"Brogue compact bridge {operation} failed: {detail}"
        raise RuntimeError(msg)


def _configure_library(library: Any) -> None:
    library.brh_abi_version.argtypes = []
    library.brh_abi_version.restype = ctypes.c_uint32
    library.brh_compact_observation_size.argtypes = []
    library.brh_compact_observation_size.restype = ctypes.c_size_t
    library.brh_set_data_dir.argtypes = [ctypes.c_char_p]
    library.brh_set_data_dir.restype = None
    library.brh_reset_compact.argtypes = [ctypes.c_uint64, ctypes.POINTER(_CCompactObservation)]
    library.brh_reset_compact.restype = ctypes.c_int
    library.brh_step_compact.argtypes = [
        ctypes.c_long,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(_CCompactObservation),
    ]
    library.brh_step_compact.restype = ctypes.c_int
    library.brh_close.argtypes = []
    library.brh_close.restype = None
    library.brh_last_error.argtypes = []
    library.brh_last_error.restype = ctypes.c_char_p


def _validate_bridge(library: Any) -> None:
    abi_version = int(library.brh_abi_version())
    compact_size = int(library.brh_compact_observation_size())
    expected_size = ctypes.sizeof(_CCompactObservation)
    if abi_version != _BRIDGE_ABI_VERSION or compact_size != expected_size:
        msg = (
            "Brogue compact bridge ABI mismatch: "
            f"abi={abi_version} expected={_BRIDGE_ABI_VERSION}, "
            f"compact_size={compact_size} expected={expected_size}"
        )
        raise RuntimeError(msg)


def _bridge_library_name() -> str:
    suffix = ".dylib" if sys.platform == "darwin" else ".so"
    return f"libbruhogue_brogue{suffix}"


def _default_data_dir() -> Path:
    packaged_library = _PACKAGED_DATA_DIR / _bridge_library_name()
    if packaged_library.is_file():
        return _PACKAGED_DATA_DIR
    source_library = _SOURCE_DATA_DIR / _bridge_library_name()
    if source_library.is_file():
        return _SOURCE_DATA_DIR
    return _PACKAGED_DATA_DIR


def _default_library_path() -> Path:
    return _default_data_dir() / _bridge_library_name()


def _bridge_seed(seed: int | None) -> int:
    if seed is None:
        return secrets.randbits(64) or 1
    if seed < 0 or seed > 2**64 - 1:
        msg = "Brogue seed must be an unsigned 64-bit integer"
        raise ValueError(msg)
    if seed == 0:
        return _GYM_ZERO_BRIDGE_SEED
    return seed


def _bridge_key(key: str) -> int:
    encoded = key.encode("latin-1")
    if len(encoded) != 1:
        msg = f"Brogue compact bridge only supports one-byte keys: {key!r}"
        raise ValueError(msg)
    return encoded[0]


def _compact_map_lines(observation: _CCompactObservation) -> list[str]:
    chars = bytes(observation.chars)
    lines = []
    for row in range(_COMPACT_ROWS):
        start = row * _COMPACT_COLS
        raw = chars[start : start + _COMPACT_COLS]
        lines.append("".join(_display_char(value) for value in raw))
    return lines


def _display_char(value: int) -> str:
    if value == 0:
        return " "
    if 32 <= value <= 126:
        return chr(value)
    return "?"


def _named_values(names: Sequence[str], values: Sequence[int]) -> list[str]:
    rows = [f"{name}={values[index]}" for index, name in enumerate(names)]
    for index in range(len(names), len(values)):
        rows.append(f"{index}={values[index]}")
    return rows


def _compact_text(observation: _CCompactObservation, status: str = "") -> str:
    lines = _compact_map_lines(observation)
    blstats = list(observation.blstats)
    program_state = list(observation.program_state)
    lines.append("")
    lines.append("blstats: " + " ".join(_named_values(_BLSTAT_NAMES, blstats)))
    lines.append("program_state: " + " ".join(_named_values(_PROGRAM_STATE_NAMES, program_state)))
    if status:
        lines.append(status)
    return "\n".join(lines)


def _curses_key_to_brogue(key: int) -> str | None:
    match key:
        case curses.KEY_UP:
            return "k"
        case curses.KEY_DOWN:
            return "j"
        case curses.KEY_LEFT:
            return "h"
        case curses.KEY_RIGHT:
            return "l"
        case curses.KEY_HOME:
            return "y"
        case curses.KEY_PPAGE:
            return "u"
        case curses.KEY_END:
            return "b"
        case curses.KEY_NPAGE:
            return "n"
        case curses.KEY_ENTER:
            return "\n"
        case 10 | 13:
            return "\n"
        case _:
            if 0 <= key <= 255:
                return chr(key)
            return None


def _draw(stdscr: Any, observation: _CCompactObservation, status: str) -> None:
    stdscr.erase()
    height, width = stdscr.getmaxyx()
    map_lines = _compact_map_lines(observation)
    for row, line in enumerate(map_lines[:height]):
        stdscr.addnstr(row, 0, line, max(0, width - 1))

    side_x = _COMPACT_COLS + 2
    info_rows = [
        "compact observation",
        "chars: 79x29 uint8",
        f"agent bytes: {_COMPACT_AGENT_BYTES}",
        f"bridge struct: {ctypes.sizeof(_CCompactObservation)}",
        "colors: not included",
        "Ctrl-C exits",
        "",
        "blstats",
        *_named_values(_BLSTAT_NAMES, list(observation.blstats)),
        "",
        "program_state",
        *_named_values(_PROGRAM_STATE_NAMES, list(observation.program_state)),
    ]
    if width > side_x + 8:
        for row, line in enumerate(info_rows[:height]):
            stdscr.addnstr(row, side_x, line, max(0, width - side_x - 1))
    else:
        start = min(_COMPACT_ROWS + 1, max(0, height - 5))
        for offset, line in enumerate(info_rows[: max(0, height - start - 1)]):
            stdscr.addnstr(start + offset, 0, line, max(0, width - 1))

    if status and height > 0:
        stdscr.addnstr(height - 1, 0, status, max(0, width - 1))
    stdscr.refresh()


def _play_curses(stdscr: Any, game: _CompactBrogue) -> None:
    curses.curs_set(0)
    curses.noecho()
    curses.cbreak()
    stdscr.keypad(True)
    status = ""
    while True:
        _draw(stdscr, game.observation, status)
        key = stdscr.getch()
        brogue_key = _curses_key_to_brogue(key)
        if brogue_key is None:
            status = f"unhandled terminal key: {key}"
            continue
        result = game.step(brogue_key)
        status = f"sent {brogue_key!r}"
        if result.invalid:
            status += " (invalid in current Brogue state)"
        if game.observation.program_state[1]:
            status += " terminated"


def _run_script(game: _CompactBrogue, script: str, *, print_each: bool) -> None:
    status = "reset"
    if print_each:
        print(_compact_text(game.observation, status))
    for key in script:
        result = game.step(key)
        status = f"sent {key!r}"
        if result.invalid:
            status += " (invalid in current Brogue state)"
        if print_each:
            print("\n" + "=" * _COMPACT_COLS)
            print(_compact_text(game.observation, status))
    if not print_each:
        print(_compact_text(game.observation, status))


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Play Brogue through the compact char-only observation used by Puffer agents.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Brogue seed; 0 uses the Gym fixed seed")
    parser.add_argument("--library", type=Path, default=_default_library_path())
    parser.add_argument("--data-dir", type=Path, default=_default_data_dir())
    parser.add_argument(
        "--script",
        default=None,
        help="Run these literal Brogue keys non-interactively and print the compact observation.",
    )
    parser.add_argument(
        "--print-each",
        action="store_true",
        help="With --script, print the compact observation after reset and after every key.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    with _CompactBrogue(library_path=args.library, data_dir=args.data_dir) as game:
        game.reset(args.seed)
        if args.script is not None:
            _run_script(game, args.script, print_each=bool(args.print_each))
            return 0
        try:
            curses.wrapper(_play_curses, game)
        except KeyboardInterrupt:
            return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
