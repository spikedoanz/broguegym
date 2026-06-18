from __future__ import annotations

import ctypes

from broguegym.compact_cli import (
    _BLSTATS_SIZE,
    _COMPACT_AGENT_BYTES,
    _COMPACT_CELLS,
    _PROGRAM_STATE_SIZE,
    _CCompactObservation,
    _compact_text,
    _compact_map_lines,
    _state_dict,
)


def test_compact_observation_layout_size() -> None:
    assert _COMPACT_AGENT_BYTES == _COMPACT_CELLS + _BLSTATS_SIZE * 4 + _PROGRAM_STATE_SIZE * 4

    assert ctypes.sizeof(_CCompactObservation) == _COMPACT_AGENT_BYTES + 1


def test_compact_map_lines_decode_ascii_chars() -> None:
    observation = _CCompactObservation()
    observation.chars[0] = ord("@")
    observation.chars[1] = ord(".")
    observation.chars[2] = 0

    lines = _compact_map_lines(observation)

    assert lines[0].startswith("@. ")
    assert len(lines) == 29
    assert all(len(line) == 79 for line in lines)


def test_compact_text_decodes_state_and_missing_fields() -> None:
    observation = _CCompactObservation()
    observation.blstats[0] = 11
    observation.blstats[1] = 12
    observation.blstats[2] = 13
    observation.blstats[3] = 14
    observation.blstats[4] = 20
    observation.blstats[5] = 3
    observation.blstats[6] = 99
    observation.blstats[8] = 1234
    observation.blstats[9] = 5
    observation.blstats[10] = 750
    observation.program_state[0] = 42
    observation.program_state[3] = 3
    observation.program_state[4] = 12345
    observation.program_state[6] = 99
    observation.program_state[7] = 5

    state = _state_dict(observation)
    text = _compact_text(observation)

    assert state["x"] == 11
    assert state["in_progress"] is True
    assert state["disturbed"] is True
    assert state["quit"] is False
    assert "state: depth=3 turn=42 pos=(11,12) hp=14/20" in text
    assert "inventory: not_observed" in text
    assert "colors=none" in text
