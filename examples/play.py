"""Interactive stdin/stdout shim for one in-process Brogue session."""

from __future__ import annotations

import argparse
import select
import sys
import termios
import tty
from collections.abc import Generator, Iterator, Sequence
from contextlib import contextmanager
from enum import Enum
from typing import TYPE_CHECKING

import msgspec

from broguegym.actions import Action, BrogueInput
from broguegym.brogue import (
    BackendErrorCode,
    BackendInfoKey,
    BackendUnavailableError,
    ObservationDict,
)

if TYPE_CHECKING:
    from broguegym.gym import BrogueEnv

type KeyEvent = tuple[str, bool, bool]

TENSOR_PREFIXES = ("map_", "inventory_")


def main() -> None:
    parser = argparse.ArgumentParser(description="Play Brogue through the in-process bridge.")
    parser.add_argument("--seed", type=int, default=None, help="Optional Brogue-job sampler seed.")
    parser.add_argument("--game-seed", type=int, default=None, help="Optional concrete Brogue game seed override.")
    parser.add_argument("--actions", default=None, help="Replay raw keys instead of reading stdin.")
    parser.add_argument("--tensors", action="store_true", help="Print privileged tensor reprs.")
    args = parser.parse_args()
    from broguegym.gym import BrogueEnv

    env = BrogueEnv(
        render_mode="ansi",
        observation_mode="privileged" if args.tensors else "player",
        actions="full",
    )
    step_count = 0
    try:
        reset_options: dict[str, object] | None = None
        if args.game_seed is not None:
            reset_options = {"game_seed": args.game_seed}
        last_observation, info = env.reset(seed=args.seed, options=reset_options)
        _emit("reset", step_count, last_observation, info, env.render(), tensors=args.tensors)
        action_lookup = _action_lookup(env.actions)
        events = _scripted(args.actions)
        if events is not None:
            for event in events:
                step_count, last_observation = _step(env, action_lookup, step_count, event, last_observation, tensors=args.tensors)
        else:
            with _stdin_mode():
                while event := _read_event():
                    step_count, last_observation = _step(env, action_lookup, step_count, event, last_observation, tensors=args.tensors)
    except (BackendUnavailableError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    finally:
        env.close()


def _step(
    env: BrogueEnv,
    action_lookup: dict[BrogueInput, int],
    step_count: int,
    event: KeyEvent,
    last_observation: ObservationDict,
    *,
    tensors: bool = False,
) -> tuple[int, ObservationDict]:
    brogue_input = _brogue_input(event)
    action_index = action_lookup.get(brogue_input)
    if action_index is None:
        info: dict[str, object] = {
            BackendInfoKey.ERROR_CODE: BackendErrorCode.ACTION_UNENCODABLE,
            BackendInfoKey.ERROR: f"CLI input {brogue_input!r} is not in the Gym action table",
        }
        _emit("error", step_count + 1, last_observation, info, env.render(), event, tensors=tensors)
        return step_count + 1, last_observation

    try:
        observation, _reward, terminated, truncated, info = env.step(action_index)
    except BackendUnavailableError as exc:
        info = {BackendInfoKey.ERROR_CODE: exc.code, BackendInfoKey.ERROR: str(exc)}
        _emit("error", step_count + 1, last_observation, info, env.render(), event, tensors=tensors)
        return step_count + 1, last_observation

    event_name = "error" if BackendInfoKey.ERROR_CODE in info else "step"
    _emit(event_name, step_count + 1, observation, info, env.render(), event, tensors=tensors)
    if terminated or truncated:
        raise SystemExit(0)
    return step_count + 1, observation


def _emit(
    event_name: str,
    step_count: int,
    observation: ObservationDict,
    info: dict[str, object],
    rendered: str | None,
    event: KeyEvent | None = None,
    *,
    tensors: bool = False,
) -> None:
    program = [int(value) for value in observation["program_state"]]
    message = bytes(observation["message"]).split(b"\0", 1)[0].decode("utf-8", "replace")
    key, control, shift = (None, False, False) if event is None else event
    payload = {
        "event": event_name, "step": step_count, "key": key, "control": control, "shift": shift,
        "turn": program[0], "terminated": bool(program[1]), "won": bool(program[2]),
        "depth": program[3], "seed": program[4], "gold": program[5], "score": program[6], "message": message,
        "blstats": [int(value) for value in observation["blstats"]], "info": {str(k): v.value if isinstance(v, Enum) else v for k, v in info.items()},
    }
    error_code, error = info.get(BackendInfoKey.ERROR_CODE), info.get(BackendInfoKey.ERROR)
    if isinstance(error_code, Enum): payload["error_code"] = error_code.value
    if isinstance(error, str): payload["error"] = error
    sys.stdout.write(msgspec.json.encode(payload).decode() + "\n")
    if tensors:
        sys.stdout.write(repr({k: v for k, v in observation.items() if k.startswith(TENSOR_PREFIXES)}) + "\n")
    sys.stdout.write((rendered or "") + "\n")
    sys.stdout.flush()


def _action_lookup(actions: Sequence[Action]) -> dict[BrogueInput, int]:
    return {action.brogue_input(): index for index, action in enumerate(actions)}


def _brogue_input(event: KeyEvent) -> BrogueInput:
    key, control, shift = event
    return BrogueInput(key=key, control=control, shift=shift and not key.isupper())


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
    arrows: dict[bytes, KeyEvent] = {b"[A": ("k", False, False), b"[B": ("j", False, False), b"[C": ("l", False, False), b"[D": ("h", False, False)}
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


if __name__ == "__main__":
    main()
