"""Interactive compact-observation viewer for the Brogue bridge."""

from __future__ import annotations

import argparse
import ctypes
import select
import secrets
import sys
import termios
import tty
from collections.abc import Generator, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import msgspec

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
    "flags",
)
_MISSING_COMPACT_FIELDS = (
    "inventory",
    "message_log",
    "colors",
    "item_ids",
    "monster_ids",
    "map_flags",
    "terrain_layers",
)

type KeyEvent = tuple[str, bool, bool]


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


def _state_dict(observation: _CCompactObservation) -> dict[str, object]:
    blstats = [int(value) for value in observation.blstats]
    program = [int(value) for value in observation.program_state]
    flags = program[7]
    return {
        "x": blstats[0],
        "y": blstats[1],
        "strength": blstats[2],
        "hp": blstats[3],
        "max_hp": blstats[4],
        "depth": blstats[5],
        "gold": blstats[6],
        "turn": program[0],
        "absolute_turn": blstats[8],
        "stealth": blstats[9],
        "nutrition": blstats[10],
        "seed": program[4],
        "score": program[6],
        "terminated": bool(program[1]),
        "won": bool(program[2]),
        "in_progress": bool(flags & 1),
        "quit": bool(flags & 2),
        "disturbed": bool(flags & 4),
        "flags": flags,
    }


def _state_line(observation: _CCompactObservation) -> str:
    state = _state_dict(observation)
    return (
        f"state: depth={state['depth']} turn={state['turn']} "
        f"pos=({state['x']},{state['y']}) hp={state['hp']}/{state['max_hp']} "
        f"str={state['strength']} gold={state['gold']} score={state['score']} "
        f"nutrition={state['nutrition']} stealth={state['stealth']} "
        f"seed={state['seed']} in_progress={state['in_progress']} "
        f"disturbed={state['disturbed']} terminated={state['terminated']} won={state['won']}"
    )


def _compact_text(observation: _CCompactObservation) -> str:
    lines = _compact_map_lines(observation)
    blstats = list(observation.blstats)
    program_state = list(observation.program_state)
    lines.append("")
    lines.append(_state_line(observation))
    lines.append("blstats: " + " ".join(_named_values(_BLSTAT_NAMES, blstats)))
    lines.append("program_state: " + " ".join(_named_values(_PROGRAM_STATE_NAMES, program_state)))
    lines.append("inventory: not_observed")
    lines.append("message_log: not_observed")
    lines.append(f"compact_agent_bytes={_COMPACT_AGENT_BYTES} bridge_struct_bytes={ctypes.sizeof(_CCompactObservation)} colors=none")
    lines.append("not_observed: " + ", ".join(_MISSING_COMPACT_FIELDS))
    return "\n".join(lines)


def _step(game: _CompactBrogue, step_count: int, event: KeyEvent) -> int:
    key, control, shift = event
    try:
        result = game.step(key, control=control, shift=shift and not key.isupper())
    except (RuntimeError, ValueError) as exc:
        _emit("error", step_count + 1, game.observation, event, {"error": str(exc)})
        return step_count + 1

    info: dict[str, object] = {}
    if result.invalid:
        info["error_code"] = "key_invalid"
        info["error"] = f"Brogue rejected key {key!r} in the current state"
    _emit("error" if result.invalid else "step", step_count + 1, game.observation, event, info)
    if game.observation.program_state[1]:
        raise SystemExit(0)
    return step_count + 1


def _emit(
    event_name: str,
    step_count: int,
    observation: _CCompactObservation,
    event: KeyEvent | None = None,
    info: dict[str, object] | None = None,
) -> None:
    program = [int(value) for value in observation.program_state]
    state = _state_dict(observation)
    key, control, shift = (None, False, False) if event is None else event
    payload = {
        "event": event_name,
        "step": step_count,
        "key": key,
        "control": control,
        "shift": shift,
        "turn": program[0],
        "terminated": bool(program[1]),
        "won": bool(program[2]),
        "depth": program[3],
        "seed": program[4],
        "gold": program[5],
        "score": program[6],
        "state": state,
        "compact_agent_bytes": _COMPACT_AGENT_BYTES,
        "bridge_struct_bytes": ctypes.sizeof(_CCompactObservation),
        "colors": None,
        "inventory": None,
        "message_log": None,
        "not_observed": list(_MISSING_COMPACT_FIELDS),
        "blstats": [int(value) for value in observation.blstats],
        "program_state": program,
        "info": {} if info is None else info,
    }
    sys.stdout.write(msgspec.json.encode(payload).decode() + "\n")
    sys.stdout.write(_compact_text(observation) + "\n")
    sys.stdout.flush()


def _scripted(actions: str | None) -> Iterator[KeyEvent] | None:
    return None if actions is None else (_event(bytes((ord(char),))) for char in actions)


def _read_event() -> KeyEvent | None:
    data = sys.stdin.buffer.read(1)
    if data in (b"", b"\x04"):
        return None
    if data == b"\x1b":
        return _arrow_event() or ("\x1b", False, False)
    return _event(data)


def _arrow_event() -> KeyEvent | None:
    if not select.select([sys.stdin], [], [], 0.01)[0]:
        return None
    arrows: dict[bytes, KeyEvent] = {
        b"[A": ("k", False, False),
        b"[B": ("j", False, False),
        b"[C": ("l", False, False),
        b"[D": ("h", False, False),
    }
    return arrows.get(sys.stdin.buffer.read(2))


def _event(data: bytes) -> KeyEvent:
    if data == b"\r":
        return ("\n", False, False)
    if 1 <= data[0] <= 26 and data not in (b"\t", b"\n"):
        return (chr(ord("a") + data[0] - 1), True, False)
    key = data.decode("latin-1")
    return (key, False, key.isalpha() and key.isupper())


@contextmanager
def _stdin_mode() -> Generator[None, None, None]:
    if not sys.stdin.isatty():
        yield
        return
    fd, previous = sys.stdin.fileno(), termios.tcgetattr(sys.stdin.fileno())
    try:
        tty.setcbreak(fd)
        attrs = termios.tcgetattr(fd)
        attrs[0] = int(attrs[0]) & ~termios.IXON
        termios.tcsetattr(fd, termios.TCSADRAIN, attrs)
        yield
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, previous)


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Play Brogue through the compact char-only observation used by Puffer agents.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Optional Brogue seed; 0 uses the Gym fixed seed.")
    parser.add_argument("--actions", default=None, help="Replay raw keys instead of reading stdin.")
    parser.add_argument("--library", type=Path, default=_default_library_path())
    parser.add_argument("--data-dir", type=Path, default=_default_data_dir())
    parser.add_argument(
        "--script",
        default=None,
        help=argparse.SUPPRESS,
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    with _CompactBrogue(library_path=args.library, data_dir=args.data_dir) as game:
        game.reset(args.seed)
        step_count = 0
        _emit("reset", step_count, game.observation)
        actions = args.actions if args.actions is not None else args.script
        events = _scripted(actions)
        if events is not None:
            for event in events:
                step_count = _step(game, step_count, event)
        else:
            with _stdin_mode():
                while event := _read_event():
                    step_count = _step(game, step_count, event)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
