from __future__ import annotations

from bruhogue.actions import ActionKind, BrogueInput
from bruhogue.spaces import (
    ACTION_DESCRIPTIONS,
    ACTION_INPUTS,
    ACTION_KEYS,
    ACTIONS,
    BASE_OBSERVATION_KEYS,
    BRIDGE_OBSERVATION_KEYS,
    EXCLUDED_PLAYER_COMMANDS,
    FULL_ACTION_DESCRIPTIONS,
    FULL_ACTION_INPUTS,
    FULL_ACTION_KEYS,
    PRIVILEGED_OBSERVATION_KEYS,
    MODAL_INPUTS,
    empty_observation,
    INVENTORY_OBSERVATION_KEYS,
    observation_space,
)


def test_action_table_exposes_stable_brogue_keys() -> None:
    assert tuple(action.brogue_key() for action in ACTIONS) == ACTION_KEYS
    assert ACTION_KEYS[:8] == ("k", "j", "h", "l", "y", "u", "b", "n")
    assert ACTION_INPUTS[8].control is True
    assert ActionKind.INVENTORY.value in ACTION_DESCRIPTIONS
    assert ActionKind.EXPLORE.value in ACTION_DESCRIPTIONS
    assert ActionKind.FAST_EXPLORE.value in ACTION_DESCRIPTIONS
    assert "A" not in ACTION_KEYS
    assert all(command not in ACTION_INPUTS for command in EXCLUDED_PLAYER_COMMANDS)
    assert len(ACTION_INPUTS) == len(set(ACTION_INPUTS))


def test_action_table_includes_modal_selection_keys() -> None:
    for key in "fgmopqv0123456789":
        assert BrogueInput(key=key) in MODAL_INPUTS
        assert BrogueInput(key=key) in ACTION_INPUTS

    modal_index = ACTION_INPUTS.index(BrogueInput(key="f"))
    assert ACTION_DESCRIPTIONS[modal_index] == "key:f"


def test_full_action_table_includes_raw_keyboard_inputs() -> None:
    for key in "ANSQG~&,+-'\"$":
        assert BrogueInput(key=key) in FULL_ACTION_INPUTS
        assert key in FULL_ACTION_KEYS

    raw_index = FULL_ACTION_INPUTS.index(BrogueInput(key="+"))
    assert FULL_ACTION_DESCRIPTIONS[raw_index] == "key:+"


def test_empty_observation_matches_declared_space() -> None:
    assert observation_space().contains(empty_observation())


def test_base_observation_space_includes_inventory() -> None:
    observation = empty_observation()

    assert set(INVENTORY_OBSERVATION_KEYS).issubset(observation_space().spaces)
    assert set(INVENTORY_OBSERVATION_KEYS).issubset(observation)
    assert "inventory_strs" in observation
    assert "map_layers" not in observation
    assert observation_space().contains(observation)


def test_privileged_observation_space_includes_map_and_entity_presence_masks() -> None:
    space = observation_space(observation_mode="privileged")
    observation = empty_observation(observation_mode="privileged")

    assert set(observation_space().spaces) == set(BASE_OBSERVATION_KEYS)
    assert set(space.spaces) == set(BRIDGE_OBSERVATION_KEYS)
    assert set(PRIVILEGED_OBSERVATION_KEYS).issubset(space.spaces)
    assert space.contains(observation)
    assert "map_has_item" in observation
    assert "map_has_monster" in observation
    assert "inventory_present" in observation
