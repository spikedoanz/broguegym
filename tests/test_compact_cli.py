from __future__ import annotations

# pyright: reportPrivateUsage=false

import ctypes

from broguegym.compact_cli import (
    _AGENT_OBSERVATION_BYTES,
    _compact_text,
    _compact_map_lines,
    _state_dict,
)
from broguegym.brogue import _CObservation


def test_cli_observation_layout_uses_full_bridge_observation() -> None:
    assert _AGENT_OBSERVATION_BYTES == ctypes.sizeof(_CObservation)
    assert _AGENT_OBSERVATION_BYTES == 155032


def test_compact_map_lines_decode_ascii_chars_from_full_screen() -> None:
    observation = _CObservation()
    observation.chars[0] = ord("@")
    observation.chars[1] = ord(".")
    observation.chars[2] = 0

    lines = _compact_map_lines(observation)

    assert lines[0].startswith("@. ")
    assert len(lines) == 34
    assert all(len(line) == 100 for line in lines)


def test_compact_text_decodes_state_and_full_observation_fields() -> None:
    observation = _CObservation()
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
    observation.message[0] = ord("h")
    observation.message[1] = ord("i")
    observation.inventory_present[0] = 1
    observation.inventory_letters[0] = ord("a")
    observation.inventory_strs[0] = ord("d")
    observation.inventory_strs[1] = ord("a")
    observation.inventory_strs[2] = ord("g")
    observation.inventory_strs[3] = ord("g")
    observation.inventory_strs[4] = ord("e")
    observation.inventory_strs[5] = ord("r")
    observation.inventory_category[0] = 2
    observation.inventory_kind[0] = 7
    observation.inventory_quantity[0] = 1
    observation.colors_fg[0] = 10
    observation.colors_bg[1] = 20
    observation.map_flags[0] = 123
    observation.map_layers[0] = 4
    observation.map_has_item[0] = 1
    observation.map_item_kind[0] = 55
    observation.map_has_monster[1] = 1
    observation.map_monster_kind[1] = 66

    state = _state_dict(observation)
    text = _compact_text(observation)

    assert state["x"] == 11
    assert state["in_progress"] is True
    assert state["disturbed"] is True
    assert state["quit"] is False
    assert "state: depth=3 turn=42 pos=(11,12) hp=14/20" in text
    assert "message_log:" in text
    assert "hi" in text
    assert "inventory:" in text
    assert "a) dagger" in text
    assert "colors: shape=[34, 100, 3]" in text
    assert "item_ids: count=1" in text
    assert "monster_ids: count=1" in text
    assert "map_flags: shape=[29, 79]" in text
    assert "terrain_layers: shape=[29, 79, 4]" in text
    assert "not_observed: none" in text
