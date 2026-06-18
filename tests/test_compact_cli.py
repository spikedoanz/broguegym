from __future__ import annotations

import ctypes

from broguegym.compact_cli import (
    _BLSTATS_SIZE,
    _COMPACT_AGENT_BYTES,
    _COMPACT_CELLS,
    _PROGRAM_STATE_SIZE,
    _CCompactObservation,
    _compact_map_lines,
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
