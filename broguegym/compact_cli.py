"""Interactive observation viewer for the Brogue bridge."""

from __future__ import annotations

# pyright: reportPrivateUsage=false

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

from broguegym.brogue import (
    _BRIDGE_ABI_VERSION,
    _CObservation,
    _GYM_ZERO_BRIDGE_SEED,
    _INVENTORY_SIZE,
    _INVENTORY_STR_LENGTH,
    _MAP_COLS,
    _MAP_ROWS,
    _SCREEN_COLS,
    _SCREEN_ROWS,
    _TERRAIN_LAYERS,
)

_AGENT_OBSERVATION_BYTES = ctypes.sizeof(_CObservation)
_MAP_CELLS = _MAP_COLS * _MAP_ROWS
_UNKNOWN_SHORT = -(2**15)
_SUMMARY_LIMIT = 16

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
type KeyEvent = tuple[str, bool, bool]


@dataclass
class _StepResult:
    invalid: bool = False
    key: str | None = None


class _CompactBrogue:
    def __init__(self, *, library_path: Path, data_dir: Path) -> None:
        self.library_path = library_path
        self.data_dir = data_dir
        self.observation = _CObservation()
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
        rc = self._require_library().brh_reset(
            ctypes.c_uint64(_bridge_seed(seed)),
            ctypes.byref(self.observation),
        )
        self._raise_if_failed(rc, "reset")

    def step(self, key: str, *, control: bool = False, shift: bool = False) -> _StepResult:
        rc = self._require_library().brh_step(
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
            msg = "Brogue bridge is not loaded"
            raise RuntimeError(msg)
        return self._library

    def _raise_if_failed(self, rc: int, operation: str) -> None:
        if rc == 0:
            return
        raw_error = self._require_library().brh_last_error()
        detail = raw_error.decode("utf-8", errors="replace") if raw_error else "unknown error"
        msg = f"Brogue bridge {operation} failed: {detail}"
        raise RuntimeError(msg)


def _configure_library(library: Any) -> None:
    library.brh_abi_version.argtypes = []
    library.brh_abi_version.restype = ctypes.c_uint32
    library.brh_observation_size.argtypes = []
    library.brh_observation_size.restype = ctypes.c_size_t
    library.brh_screen_cols.argtypes = []
    library.brh_screen_cols.restype = ctypes.c_int
    library.brh_screen_rows.argtypes = []
    library.brh_screen_rows.restype = ctypes.c_int
    library.brh_map_cols.argtypes = []
    library.brh_map_cols.restype = ctypes.c_int
    library.brh_map_rows.argtypes = []
    library.brh_map_rows.restype = ctypes.c_int
    library.brh_set_data_dir.argtypes = [ctypes.c_char_p]
    library.brh_set_data_dir.restype = None
    library.brh_reset.argtypes = [ctypes.c_uint64, ctypes.POINTER(_CObservation)]
    library.brh_reset.restype = ctypes.c_int
    library.brh_step.argtypes = [
        ctypes.c_long,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(_CObservation),
    ]
    library.brh_step.restype = ctypes.c_int
    library.brh_close.argtypes = []
    library.brh_close.restype = None
    library.brh_last_error.argtypes = []
    library.brh_last_error.restype = ctypes.c_char_p


def _validate_bridge(library: Any) -> None:
    abi_version = int(library.brh_abi_version())
    observation_size = int(library.brh_observation_size())
    screen_cols = int(library.brh_screen_cols())
    screen_rows = int(library.brh_screen_rows())
    map_cols = int(library.brh_map_cols())
    map_rows = int(library.brh_map_rows())
    expected_size = ctypes.sizeof(_CObservation)
    if (
        abi_version != _BRIDGE_ABI_VERSION
        or observation_size != expected_size
        or screen_cols != _SCREEN_COLS
        or screen_rows != _SCREEN_ROWS
        or map_cols != _MAP_COLS
        or map_rows != _MAP_ROWS
    ):
        msg = (
            "Brogue bridge ABI mismatch: "
            f"abi={abi_version} expected={_BRIDGE_ABI_VERSION}, "
            f"observation_size={observation_size} expected={expected_size}, "
            f"screen={screen_cols}x{screen_rows} expected={_SCREEN_COLS}x{_SCREEN_ROWS}, "
            f"map={map_cols}x{map_rows} expected={_MAP_COLS}x{_MAP_ROWS}"
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
        msg = f"Brogue bridge only supports one-byte keys: {key!r}"
        raise ValueError(msg)
    return encoded[0]


def _compact_map_lines(observation: _CObservation) -> list[str]:
    chars: list[int] = [int(value) for value in observation.chars]
    lines: list[str] = []
    for row in range(_SCREEN_ROWS):
        start = row * _SCREEN_COLS
        raw = chars[start : start + _SCREEN_COLS]
        lines.append("".join(_display_char(value) for value in raw))
    return lines


def _display_char(value: int) -> str:
    if value == 0:
        return " "
    try:
        char = chr(value)
    except ValueError:
        return "?"
    if char.isprintable():
        return char
    return "?"


def _named_values(names: Sequence[str], values: Sequence[int]) -> list[str]:
    rows = [f"{name}={values[index]}" for index, name in enumerate(names)]
    for index in range(len(names), len(values)):
        rows.append(f"{index}={values[index]}")
    return rows


def _state_dict(observation: _CObservation) -> dict[str, object]:
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


def _state_line(observation: _CObservation) -> str:
    state = _state_dict(observation)
    return (
        f"state: depth={state['depth']} turn={state['turn']} "
        f"pos=({state['x']},{state['y']}) hp={state['hp']}/{state['max_hp']} "
        f"str={state['strength']} gold={state['gold']} score={state['score']} "
        f"nutrition={state['nutrition']} stealth={state['stealth']} "
        f"seed={state['seed']} in_progress={state['in_progress']} "
        f"disturbed={state['disturbed']} terminated={state['terminated']} won={state['won']}"
    )


def _decode_bytes(values: Sequence[int]) -> str:
    return bytes(int(value) for value in values).split(b"\0", 1)[0].decode("utf-8", "replace")


def _unknown_or_int(value: int) -> int | str:
    return "unknown" if value == _UNKNOWN_SHORT else value


def _inventory_entries(observation: _CObservation) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    for slot in range(_INVENTORY_SIZE):
        if not int(observation.inventory_present[slot]):
            continue
        letter_value = int(observation.inventory_letters[slot])
        start = slot * _INVENTORY_STR_LENGTH
        name = _decode_bytes(observation.inventory_strs[start : start + _INVENTORY_STR_LENGTH])
        entries.append(
            {
                "slot": slot,
                "letter": chr(letter_value) if letter_value else "",
                "name": name,
                "category": int(observation.inventory_category[slot]),
                "kind": _unknown_or_int(int(observation.inventory_kind[slot])),
                "quantity": _unknown_or_int(int(observation.inventory_quantity[slot])),
                "flags": int(observation.inventory_flags[slot]),
                "enchant1": _unknown_or_int(int(observation.inventory_enchant1[slot])),
                "enchant2": _unknown_or_int(int(observation.inventory_enchant2[slot])),
                "charges": _unknown_or_int(int(observation.inventory_charges[slot])),
            }
        )
    return entries


def _message_log(observation: _CObservation) -> str:
    return _decode_bytes(observation.message).rstrip("\n")


def _rgb_at(values: Sequence[int], row: int, col: int) -> list[int]:
    if row < 0 or row >= _SCREEN_ROWS or col < 0 or col >= _SCREEN_COLS:
        return []
    offset = (row * _SCREEN_COLS + col) * 3
    return [int(values[offset]), int(values[offset + 1]), int(values[offset + 2])]


def _colors_summary(observation: _CObservation) -> dict[str, object]:
    player_x = int(observation.blstats[0])
    player_y = int(observation.blstats[1])
    return {
        "shape": [_SCREEN_ROWS, _SCREEN_COLS, 3],
        "fg_nonzero_components": sum(1 for value in observation.colors_fg if int(value)),
        "bg_nonzero_components": sum(1 for value in observation.colors_bg if int(value)),
        "player_fg": _rgb_at(observation.colors_fg, player_y, player_x),
        "player_bg": _rgb_at(observation.colors_bg, player_y, player_x),
    }


def _map_flag_summary(observation: _CObservation) -> dict[str, object]:
    sample: list[dict[str, int]] = []
    nonzero = 0
    for index, raw_value in enumerate(observation.map_flags):
        value = int(raw_value)
        if value == 0:
            continue
        nonzero += 1
        if len(sample) < _SUMMARY_LIMIT:
            sample.append({"x": index % _MAP_COLS, "y": index // _MAP_COLS, "flags": value})
    return {"shape": [_MAP_ROWS, _MAP_COLS], "nonzero": nonzero, "sample": sample}


def _terrain_layers_summary(observation: _CObservation) -> dict[str, object]:
    nonzero_by_layer = [0] * _TERRAIN_LAYERS
    sample: list[dict[str, int]] = []
    for cell in range(_MAP_CELLS):
        x = cell % _MAP_COLS
        y = cell // _MAP_COLS
        layer_offset = cell * _TERRAIN_LAYERS
        for layer in range(_TERRAIN_LAYERS):
            value = int(observation.map_layers[layer_offset + layer])
            if value == 0:
                continue
            nonzero_by_layer[layer] += 1
            if len(sample) < _SUMMARY_LIMIT:
                sample.append({"x": x, "y": y, "layer": layer, "terrain": value})
    return {
        "shape": [_MAP_ROWS, _MAP_COLS, _TERRAIN_LAYERS],
        "nonzero_by_layer": nonzero_by_layer,
        "sample": sample,
    }


def _item_entries(observation: _CObservation) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    for index in range(_MAP_CELLS):
        if not int(observation.map_has_item[index]):
            continue
        entries.append(
            {
                "x": index % _MAP_COLS,
                "y": index // _MAP_COLS,
                "id": _unknown_or_int(int(observation.map_item_kind[index])),
                "category": int(observation.map_item_category[index]),
                "quantity": _unknown_or_int(int(observation.map_item_quantity[index])),
                "flags": int(observation.map_item_flags[index]),
            }
        )
    return entries


def _monster_entries(observation: _CObservation) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    for index in range(_MAP_CELLS):
        if not int(observation.map_has_monster[index]):
            continue
        entries.append(
            {
                "x": index % _MAP_COLS,
                "y": index // _MAP_COLS,
                "id": _unknown_or_int(int(observation.map_monster_kind[index])),
                "hp": _unknown_or_int(int(observation.map_monster_hp[index])),
                "state": _unknown_or_int(int(observation.map_monster_state[index])),
            }
        )
    return entries


def _append_block(lines: list[str], label: str, text: str) -> None:
    if not text:
        lines.append(f"{label}: <empty>")
        return
    lines.append(f"{label}:")
    lines.extend(f"  {line}" for line in text.splitlines())


def _compact_text(observation: _CObservation) -> str:
    lines = _compact_map_lines(observation)
    blstats = list(observation.blstats)
    program_state = list(observation.program_state)
    inventory = _inventory_entries(observation)
    items = _item_entries(observation)
    monsters = _monster_entries(observation)
    colors = _colors_summary(observation)
    map_flags = _map_flag_summary(observation)
    terrain_layers = _terrain_layers_summary(observation)
    lines.append("")
    lines.append(_state_line(observation))
    lines.append("blstats: " + " ".join(_named_values(_BLSTAT_NAMES, blstats)))
    lines.append("program_state: " + " ".join(_named_values(_PROGRAM_STATE_NAMES, program_state)))
    _append_block(lines, "message_log", _message_log(observation))
    if inventory:
        lines.append("inventory:")
        for entry in inventory:
            letter = f"{entry['letter']}) " if entry["letter"] else ""
            lines.append(
                "  "
                f"{entry['slot']}: {letter}{entry['name']} "
                f"category={entry['category']} kind={entry['kind']} qty={entry['quantity']} "
                f"flags={entry['flags']} enchant=({entry['enchant1']},{entry['enchant2']}) charges={entry['charges']}"
            )
    else:
        lines.append("inventory: empty")
    lines.append(
        "colors: "
        f"shape={colors['shape']} fg_nonzero_components={colors['fg_nonzero_components']} "
        f"bg_nonzero_components={colors['bg_nonzero_components']} "
        f"player_fg={colors['player_fg']} player_bg={colors['player_bg']}"
    )
    lines.append(
        "item_ids: "
        f"count={len(items)} sample={items[:_SUMMARY_LIMIT]}"
    )
    lines.append(
        "monster_ids: "
        f"count={len(monsters)} sample={monsters[:_SUMMARY_LIMIT]}"
    )
    lines.append(
        "map_flags: "
        f"shape={map_flags['shape']} nonzero={map_flags['nonzero']} sample={map_flags['sample']}"
    )
    lines.append(
        "terrain_layers: "
        f"shape={terrain_layers['shape']} nonzero_by_layer={terrain_layers['nonzero_by_layer']} "
        f"sample={terrain_layers['sample']}"
    )
    lines.append(f"agent_observation_bytes={_AGENT_OBSERVATION_BYTES} bridge_struct_bytes={ctypes.sizeof(_CObservation)}")
    lines.append("not_observed: none")
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
    observation: _CObservation,
    event: KeyEvent | None = None,
    info: dict[str, object] | None = None,
) -> None:
    program = [int(value) for value in observation.program_state]
    state = _state_dict(observation)
    key, control, shift = (None, False, False) if event is None else event
    colors = _colors_summary(observation)
    inventory = _inventory_entries(observation)
    message_log = _message_log(observation)
    items = _item_entries(observation)
    monsters = _monster_entries(observation)
    map_flags = _map_flag_summary(observation)
    terrain_layers = _terrain_layers_summary(observation)
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
        "agent_observation_bytes": _AGENT_OBSERVATION_BYTES,
        "bridge_struct_bytes": ctypes.sizeof(_CObservation),
        "screen_shape": [_SCREEN_ROWS, _SCREEN_COLS],
        "map_shape": [_MAP_ROWS, _MAP_COLS],
        "colors": colors,
        "inventory": inventory,
        "message_log": message_log,
        "item_ids": {"count": len(items), "sample": items[:_SUMMARY_LIMIT]},
        "monster_ids": {"count": len(monsters), "sample": monsters[:_SUMMARY_LIMIT]},
        "map_flags": map_flags,
        "terrain_layers": terrain_layers,
        "not_observed": [],
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
        description="Play Brogue through the full observation used by Puffer agents.",
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
