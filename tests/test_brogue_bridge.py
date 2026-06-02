from __future__ import annotations

# pyright: reportPrivateUsage=false

import ctypes
from pathlib import Path

import pytest

from brogue_gym.actions import Action, ActionKind, Direction
from brogue_gym.brogue import (
    BackendErrorCode,
    BackendInfoKey,
    _InProcessBrogue,
    _CObservation,
    _default_library_path,
    _observation_from_c,
)
from brogue_gym.gym import BrogueEnv
from brogue_gym.spaces import ACTIONS, INVENTORY_STR_LENGTH, SCREEN_COLS, SCREEN_ROWS

G_TRAP = 198
G_BLOODWORT_STALK = 225
U_DIAMOND = 0x25C7
U_ARIES = 0x2648


def test_brogue_bridge_converts_c_observation() -> None:
    c_observation = _CObservation()
    c_observation.glyphs[0] = 64
    c_observation.chars[1] = U_DIAMOND
    c_observation.colors_fg[2] = 66
    c_observation.colors_bg[3] = 67
    c_observation.specials[4] = 68
    c_observation.map_has_item[3] = 1
    c_observation.map_has_monster[4] = 1
    c_observation.inventory_present[5] = 1
    c_observation.inventory_strs[5 * INVENTORY_STR_LENGTH] = ord("x")
    c_observation.blstats[5] = 69
    c_observation.message[6] = 70
    c_observation.program_state[7] = 71

    observation = _observation_from_c(c_observation)

    assert observation["glyphs"].shape == (SCREEN_ROWS, SCREEN_COLS)
    assert observation["chars"].shape == (SCREEN_ROWS, SCREEN_COLS)
    assert observation["colors_fg"].shape == (SCREEN_ROWS, SCREEN_COLS, 3)
    assert observation["colors_bg"].shape == (SCREEN_ROWS, SCREEN_COLS, 3)
    assert observation["glyphs"][0, 0] == 64
    assert observation["chars"][0, 1] == U_DIAMOND
    assert observation["colors_fg"][0, 0, 2] == 66
    assert observation["colors_bg"][0, 1, 0] == 67
    assert observation["specials"].reshape(-1)[4] == 68
    assert observation["map_has_item"][0, 3] == 1
    assert observation["map_has_monster"][0, 4] == 1
    assert observation["inventory_present"][5] == 1
    assert observation["inventory_strs"][5, 0] == ord("x")
    assert observation["blstats"][5] == 69
    assert observation["message"][6] == 70
    assert observation["program_state"][7] == 71


def test_bridge_exports_unicode_codepoints_in_chars_if_built() -> None:
    library_path = _default_library_path()
    if not library_path.exists():
        pytest.skip("run through uv so package sync builds the bridge")

    library = ctypes.CDLL(str(library_path))
    library.glyphToUnicode.argtypes = [ctypes.c_int]
    library.glyphToUnicode.restype = ctypes.c_uint

    assert library.glyphToUnicode(G_TRAP) == U_DIAMOND
    assert library.glyphToUnicode(G_BLOODWORT_STALK) == U_ARIES


def test_brogue_bridge_drives_one_bridge_session_if_built() -> None:
    library_path = _default_library_path()
    if not library_path.exists():
        pytest.skip("run through uv so package sync builds the bridge")

    backend = _InProcessBrogue()
    env = BrogueEnv(backend=backend, render_mode="ansi")

    try:
        observation, info = env.reset(seed=1)
        assert env.observation_space.contains(observation)
        assert "inventory_strs" in observation
        assert info == {BackendInfoKey.SEED: int(observation["program_state"][4])}
        assert info[BackendInfoKey.SEED] == 1
        message = bytes(observation["message"]).split(b"\0", 1)[0]
        assert b"\x19" not in message
        assert b"Hello and welcome" in message

        rest_index = ACTIONS.index(Action(kind=ActionKind.REST))
        observation, reward, terminated, truncated, info = env.step(rest_index)

        assert env.observation_space.contains(observation)
        assert reward == 0.0
        assert terminated is False
        assert truncated is False
        assert info[BackendInfoKey.KEY] == "z"
    finally:
        env.close()


def test_brogue_bridge_reports_invalid_noop_key_without_closing_if_built() -> None:
    library_path = _default_library_path()
    if not library_path.exists():
        pytest.skip("run through uv so package sync builds the bridge")

    backend = _InProcessBrogue()

    try:
        backend.reset(seed=1)
        rejected = backend.step(Action.keypress("!"))
        assert rejected.info[BackendInfoKey.ERROR_CODE] is BackendErrorCode.KEY_INVALID
        assert rejected.info[BackendInfoKey.ERROR] == "Brogue rejected key '!' in the current state"
        assert rejected.terminated is False

        result = backend.step(Action(kind=ActionKind.REST))
        assert result.info[BackendInfoKey.KEY] == "z"
        assert result.terminated is False
    finally:
        backend.close()


def test_brogue_bridge_can_close_inventory_if_built() -> None:
    library_path = _default_library_path()
    if not library_path.exists():
        pytest.skip("run through uv so package sync builds the bridge")

    backend = _InProcessBrogue()

    try:
        backend.reset(seed=1)
        opened = backend.step(Action(kind=ActionKind.INVENTORY))
        assert opened.terminated is False

        rejected = backend.step(Action.keypress("!"))
        assert rejected.info[BackendInfoKey.ERROR_CODE] is BackendErrorCode.KEY_INVALID

        closed = backend.step(Action(kind=ActionKind.ESCAPE))
        assert closed.terminated is False

        result = backend.step(Action(kind=ActionKind.REST))
        assert result.info[BackendInfoKey.KEY] == "z"
    finally:
        backend.close()


def test_brogue_bridge_accepts_cursor_keys_with_button_overlay_if_built() -> None:
    library_path = _default_library_path()
    if not library_path.exists():
        pytest.skip("run through uv so package sync builds the bridge")

    backend = _InProcessBrogue()

    try:
        backend.reset(seed=1)
        cursor = backend.step(Action(kind=ActionKind.CURSOR))
        assert BackendInfoKey.ERROR_CODE not in cursor.info

        moved_cursor = backend.step(Action.move(Direction.WEST))
        assert BackendInfoKey.ERROR_CODE not in moved_cursor.info

        closed = backend.step(Action(kind=ActionKind.ESCAPE))
        assert BackendInfoKey.ERROR_CODE not in closed.info
    finally:
        backend.close()


def test_brogue_bridge_close_does_not_write_last_game_if_built(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    library_path = _default_library_path()
    if not library_path.exists():
        pytest.skip("run through uv so package sync builds the bridge")

    monkeypatch.chdir(tmp_path)
    backend = _InProcessBrogue()
    env = BrogueEnv(backend=backend)

    env.reset(seed=1)
    env.close()

    assert list(tmp_path.glob("LastGame*.broguesave")) == []


def test_brogue_bridge_passes_modifier_actions_if_built() -> None:
    library_path = _default_library_path()
    if not library_path.exists():
        pytest.skip("run through uv so package sync builds the bridge")

    backend = _InProcessBrogue()

    try:
        backend.reset(seed=1)
        result = backend.step(Action(kind=ActionKind.LONG_SEARCH))
        assert result.info[BackendInfoKey.KEY] == "s"
        assert result.info[BackendInfoKey.CONTROL] is True
        assert result.info[BackendInfoKey.SHIFT] is False
    finally:
        backend.close()
