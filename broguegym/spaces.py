"""Gymnasium action and observation spaces for Brogue."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final, Literal

import numpy as np
from gymnasium import spaces

from broguegym.actions import Action, ActionKind, BrogueInput, Direction
from broguegym.brogue import ObservationDict

SCREEN_COLS: Final = 100
SCREEN_ROWS: Final = 34
MAP_COLS: Final = 79
MAP_ROWS: Final = 29
TERRAIN_LAYERS: Final = 4
BLSTATS_SIZE: Final = 21
MESSAGE_SIZE: Final = 256
PROGRAM_STATE_SIZE: Final = 8
INVENTORY_SIZE: Final = 26
INVENTORY_STR_LENGTH: Final = 80
UNKNOWN_SHORT: Final = np.iinfo(np.int16).min

PROGRAM_TURN_INDEX: Final = 0
PROGRAM_TERMINATED_INDEX: Final = 1
PROGRAM_WON_INDEX: Final = 2
PROGRAM_DEPTH_INDEX: Final = 3
PROGRAM_SEED_INDEX: Final = 4
PROGRAM_GOLD_INDEX: Final = 5
PROGRAM_SCORE_INDEX: Final = 6
PROGRAM_FLAGS_INDEX: Final = 7
CORE_OBSERVATION_KEYS: Final = (
    "glyphs",
    "chars",
    "colors_fg",
    "colors_bg",
    "specials",
    "blstats",
    "message",
    "program_state",
)
INVENTORY_OBSERVATION_KEYS: Final = (
    "inventory_present",
    "inventory_letters",
    "inventory_strs",
    "inventory_category",
    "inventory_kind",
    "inventory_quantity",
    "inventory_flags",
    "inventory_enchant1",
    "inventory_enchant2",
    "inventory_charges",
)
BASE_OBSERVATION_KEYS: Final = CORE_OBSERVATION_KEYS + INVENTORY_OBSERVATION_KEYS
PRIVILEGED_OBSERVATION_KEYS: Final = (
    "map_layers",
    "map_flags",
    "map_volume",
    "map_machine",
    "map_light",
    "map_has_item",
    "map_item_category",
    "map_item_kind",
    "map_item_quantity",
    "map_item_flags",
    "map_has_monster",
    "map_monster_kind",
    "map_monster_hp",
    "map_monster_state",
)
SEMANTIC_OBSERVATION_KEYS: Final = PRIVILEGED_OBSERVATION_KEYS
BRIDGE_OBSERVATION_KEYS: Final = BASE_OBSERVATION_KEYS + PRIVILEGED_OBSERVATION_KEYS
type ObservationMode = Literal["player", "privileged"]

MOVE_ACTIONS: Final = (
    Action.move(Direction.NORTH),
    Action.move(Direction.SOUTH),
    Action.move(Direction.WEST),
    Action.move(Direction.EAST),
    Action.move(Direction.NORTHWEST),
    Action.move(Direction.NORTHEAST),
    Action.move(Direction.SOUTHWEST),
    Action.move(Direction.SOUTHEAST),
)

RUN_ACTIONS: Final = tuple(Action.move(direction, run=True) for direction in Direction)

EXCLUDED_PLAYER_COMMANDS: Final[tuple[BrogueInput, ...]] = (
    BrogueInput(key="A"),
    BrogueInput(key="S"),
    BrogueInput(key="N"),
    BrogueInput(key="Q"),
    BrogueInput(key="G"),
    BrogueInput(key="~"),
    BrogueInput(key="&"),
    BrogueInput(key=","),
)
_MODAL_INPUT_CHARS: Final = "abcdefghijklmnopqrstuvwxyz0123456789"
_FULL_INPUT_CHARS: Final = "\t\n\x1b" + "".join(
    chr(codepoint)
    for codepoint in range(0x20, 0x7F)
)
type ActionSet = Literal["default", "full"]


def _append_missing_actions(
    base_actions: Sequence[Action],
    extra_actions: Sequence[Action],
) -> tuple[Action, ...]:
    actions = list(base_actions)
    seen = {action.brogue_input() for action in actions}
    for action in extra_actions:
        brogue_input = action.brogue_input()
        if brogue_input not in seen:
            actions.append(action)
            seen.add(brogue_input)
    return tuple(actions)

# This is source-derived from Rogue.h's keyboard command constants and IO.c's main
# executeKeystroke switch. It is still a training-oriented default: save/new/quit, debug commands,
# screenshot, graphics toggles, full autopilot, and load/view-recording commands are deliberately
# excluded. Modal selection keys are appended below only when they do not duplicate an existing
# command key event. Coordinate inspection is represented by Action.inspect(...) rather than a
# discrete key; keyboard cursor mode is included because Brogue exposes it through Return.
COMMAND_ACTIONS: Final = (
    *MOVE_ACTIONS,
    *RUN_ACTIONS,
    Action(kind=ActionKind.REST),
    Action(kind=ActionKind.AUTO_REST),
    Action(kind=ActionKind.SEARCH),
    Action(kind=ActionKind.LONG_SEARCH),
    Action(kind=ActionKind.ASCEND),
    Action(kind=ActionKind.DESCEND),
    Action(kind=ActionKind.INVENTORY),
    Action(kind=ActionKind.EQUIP),
    Action(kind=ActionKind.UNEQUIP),
    Action(kind=ActionKind.APPLY),
    Action(kind=ActionKind.THROW),
    Action(kind=ActionKind.RETHROW),
    Action(kind=ActionKind.DROP),
    Action(kind=ActionKind.CALL),
    Action(kind=ActionKind.RELABEL),
    Action(kind=ActionKind.SWAP),
    Action(kind=ActionKind.EXPLORE),
    Action(kind=ActionKind.FAST_EXPLORE),
    Action(kind=ActionKind.MESSAGE_ARCHIVE),
    Action(kind=ActionKind.DISCOVERIES),
    Action(kind=ActionKind.FEATS),
    Action(kind=ActionKind.HELP),
    Action(kind=ActionKind.TRUE_COLORS),
    Action(kind=ActionKind.STEALTH_RANGE),
    Action(kind=ActionKind.CURSOR),
    Action(kind=ActionKind.ACKNOWLEDGE),
    Action(kind=ActionKind.ESCAPE),
)

_COMMAND_INPUTS: Final = tuple(action.brogue_input() for action in COMMAND_ACTIONS)
MODAL_INPUT_ACTIONS: Final = tuple(
    Action.keypress(key)
    for key in _MODAL_INPUT_CHARS
    if BrogueInput(key=key) not in _COMMAND_INPUTS
    and BrogueInput(key=key) not in EXCLUDED_PLAYER_COMMANDS
)
MODAL_INPUTS: Final = tuple(action.brogue_input() for action in MODAL_INPUT_ACTIONS)
ACTIONS: Final = (*COMMAND_ACTIONS, *MODAL_INPUT_ACTIONS)
ACTION_INPUTS: Final = tuple(action.brogue_input() for action in ACTIONS)
ACTION_KEYS: Final = tuple(action.brogue_key() for action in ACTIONS)

FULL_TEXT_INPUT_ACTIONS: Final = tuple(Action.keypress(key) for key in _FULL_INPUT_CHARS)
FULL_ACTIONS: Final = _append_missing_actions(COMMAND_ACTIONS, FULL_TEXT_INPUT_ACTIONS)
FULL_ACTION_INPUTS: Final = tuple(action.brogue_input() for action in FULL_ACTIONS)
FULL_ACTION_KEYS: Final = tuple(action.brogue_key() for action in FULL_ACTIONS)


def action_description(action: Action) -> str:
    if action.kind is not ActionKind.KEY:
        return action.kind.value
    brogue_input = action.brogue_input()
    modifiers = ""
    if brogue_input.control:
        modifiers += "ctrl+"
    if brogue_input.shift:
        modifiers += "shift+"
    return f"key:{modifiers}{_key_label(brogue_input.key)}"


def _key_label(key: str) -> str:
    if key == " ":
        return "space"
    if key == "\n":
        return "return"
    if key == "\x1b":
        return "escape"
    return key


ACTION_DESCRIPTIONS: Final = tuple(
    action_description(action)
    for action in ACTIONS
)
FULL_ACTION_DESCRIPTIONS: Final = tuple(
    action_description(action)
    for action in FULL_ACTIONS
)


def resolve_actions(actions: Sequence[Action] | ActionSet | None = None) -> tuple[Action, ...]:
    """Return the action table represented by a named or explicit action set."""

    if actions is None:
        return ACTIONS
    if isinstance(actions, str):
        if actions == "default":
            return ACTIONS
        if actions == "full":
            return FULL_ACTIONS
        msg = f"unknown Brogue action set: {actions}"
        raise ValueError(msg)
    resolved = tuple(actions)
    if not resolved:
        msg = "Brogue action table must not be empty"
        raise ValueError(msg)
    if len({action.brogue_input() for action in resolved}) != len(resolved):
        msg = "Brogue action table must not contain duplicate key events"
        raise ValueError(msg)
    return resolved


def resolve_observation_mode(
    observation_mode: ObservationMode = "player",
    *,
    include_semantic_tensors: bool | None = None,
) -> ObservationMode:
    """Return the effective Gym policy-observation mode."""

    if observation_mode not in ("player", "privileged"):
        msg = f"unknown Brogue observation mode: {observation_mode}"
        raise ValueError(msg)
    if include_semantic_tensors is None:
        return observation_mode
    legacy_mode: ObservationMode = "privileged" if include_semantic_tensors else "player"
    if observation_mode != "player" and observation_mode != legacy_mode:
        msg = "include_semantic_tensors conflicts with observation_mode"
        raise ValueError(msg)
    return legacy_mode if observation_mode == "player" else observation_mode


def observation_space(
    *,
    observation_mode: ObservationMode = "player",
    include_semantic_tensors: bool | None = None,
) -> spaces.Dict:
    """Return the fixed tensor observation space for the direct Brogue backend."""

    int64_info = np.iinfo(np.int64)
    uint64_info = np.iinfo(np.uint64)
    int16_info = np.iinfo(np.int16)
    uint16_info = np.iinfo(np.uint16)
    all_spaces = {
            "glyphs": spaces.Box(
                low=0,
                high=np.iinfo(np.int16).max,
                shape=(SCREEN_ROWS, SCREEN_COLS),
                dtype=np.int16,
            ),
            "chars": spaces.Box(
                low=0,
                high=np.iinfo(np.uint32).max,
                shape=(SCREEN_ROWS, SCREEN_COLS),
                dtype=np.uint32,
            ),
            "colors_fg": spaces.Box(
                low=0,
                high=255,
                shape=(SCREEN_ROWS, SCREEN_COLS, 3),
                dtype=np.uint8,
            ),
            "colors_bg": spaces.Box(
                low=0,
                high=255,
                shape=(SCREEN_ROWS, SCREEN_COLS, 3),
                dtype=np.uint8,
            ),
            "specials": spaces.Box(
                low=0,
                high=255,
                shape=(SCREEN_ROWS, SCREEN_COLS),
                dtype=np.uint8,
            ),
            "map_layers": spaces.Box(
                low=0,
                high=uint16_info.max,
                shape=(MAP_ROWS, MAP_COLS, TERRAIN_LAYERS),
                dtype=np.uint16,
            ),
            "map_flags": spaces.Box(
                low=uint64_info.min,
                high=uint64_info.max,
                shape=(MAP_ROWS, MAP_COLS),
                dtype=np.uint64,
            ),
            "map_volume": spaces.Box(
                low=0,
                high=uint16_info.max,
                shape=(MAP_ROWS, MAP_COLS),
                dtype=np.uint16,
            ),
            "map_machine": spaces.Box(
                low=0,
                high=255,
                shape=(MAP_ROWS, MAP_COLS),
                dtype=np.uint8,
            ),
            "map_light": spaces.Box(
                low=int16_info.min,
                high=int16_info.max,
                shape=(MAP_ROWS, MAP_COLS, 3),
                dtype=np.int16,
            ),
            "map_has_item": spaces.Box(
                low=0,
                high=1,
                shape=(MAP_ROWS, MAP_COLS),
                dtype=np.uint8,
            ),
            "map_item_category": spaces.Box(
                low=0,
                high=uint16_info.max,
                shape=(MAP_ROWS, MAP_COLS),
                dtype=np.uint16,
            ),
            "map_item_kind": spaces.Box(
                low=int16_info.min,
                high=int16_info.max,
                shape=(MAP_ROWS, MAP_COLS),
                dtype=np.int16,
            ),
            "map_item_quantity": spaces.Box(
                low=int16_info.min,
                high=int16_info.max,
                shape=(MAP_ROWS, MAP_COLS),
                dtype=np.int16,
            ),
            "map_item_flags": spaces.Box(
                low=uint64_info.min,
                high=uint64_info.max,
                shape=(MAP_ROWS, MAP_COLS),
                dtype=np.uint64,
            ),
            "map_has_monster": spaces.Box(
                low=0,
                high=1,
                shape=(MAP_ROWS, MAP_COLS),
                dtype=np.uint8,
            ),
            "map_monster_kind": spaces.Box(
                low=int16_info.min,
                high=int16_info.max,
                shape=(MAP_ROWS, MAP_COLS),
                dtype=np.int16,
            ),
            "map_monster_hp": spaces.Box(
                low=int16_info.min,
                high=int16_info.max,
                shape=(MAP_ROWS, MAP_COLS),
                dtype=np.int16,
            ),
            "map_monster_state": spaces.Box(
                low=int16_info.min,
                high=int16_info.max,
                shape=(MAP_ROWS, MAP_COLS),
                dtype=np.int16,
            ),
            "inventory_present": spaces.Box(
                low=0,
                high=1,
                shape=(INVENTORY_SIZE,),
                dtype=np.uint8,
            ),
            "inventory_letters": spaces.Box(
                low=0,
                high=255,
                shape=(INVENTORY_SIZE,),
                dtype=np.uint8,
            ),
            "inventory_strs": spaces.Box(
                low=0,
                high=255,
                shape=(INVENTORY_SIZE, INVENTORY_STR_LENGTH),
                dtype=np.uint8,
            ),
            "inventory_category": spaces.Box(
                low=0,
                high=uint16_info.max,
                shape=(INVENTORY_SIZE,),
                dtype=np.uint16,
            ),
            "inventory_kind": spaces.Box(
                low=int16_info.min,
                high=int16_info.max,
                shape=(INVENTORY_SIZE,),
                dtype=np.int16,
            ),
            "inventory_quantity": spaces.Box(
                low=int16_info.min,
                high=int16_info.max,
                shape=(INVENTORY_SIZE,),
                dtype=np.int16,
            ),
            "inventory_flags": spaces.Box(
                low=uint64_info.min,
                high=uint64_info.max,
                shape=(INVENTORY_SIZE,),
                dtype=np.uint64,
            ),
            "inventory_enchant1": spaces.Box(
                low=int16_info.min,
                high=int16_info.max,
                shape=(INVENTORY_SIZE,),
                dtype=np.int16,
            ),
            "inventory_enchant2": spaces.Box(
                low=int16_info.min,
                high=int16_info.max,
                shape=(INVENTORY_SIZE,),
                dtype=np.int16,
            ),
            "inventory_charges": spaces.Box(
                low=int16_info.min,
                high=int16_info.max,
                shape=(INVENTORY_SIZE,),
                dtype=np.int16,
            ),
            "blstats": spaces.Box(
                low=int64_info.min,
                high=int64_info.max,
                shape=(BLSTATS_SIZE,),
                dtype=np.int64,
            ),
            "message": spaces.Box(
                low=0,
                high=255,
                shape=(MESSAGE_SIZE,),
                dtype=np.uint8,
            ),
            "program_state": spaces.Box(
                low=uint64_info.min,
                high=uint64_info.max,
                shape=(PROGRAM_STATE_SIZE,),
                dtype=np.uint64,
            ),
        }
    mode = resolve_observation_mode(
        observation_mode,
        include_semantic_tensors=include_semantic_tensors,
    )
    keys = BRIDGE_OBSERVATION_KEYS if mode == "privileged" else BASE_OBSERVATION_KEYS
    return spaces.Dict({key: all_spaces[key] for key in keys})


def empty_observation(
    *,
    observation_mode: ObservationMode = "player",
    include_semantic_tensors: bool | None = None,
) -> ObservationDict:
    """Build an all-zero observation with the right shapes and dtypes for tests and fakes."""

    map_unknown = np.full((MAP_ROWS, MAP_COLS), UNKNOWN_SHORT, dtype=np.int16)
    inventory_unknown = np.full((INVENTORY_SIZE,), UNKNOWN_SHORT, dtype=np.int16)
    observation: ObservationDict = {
        "glyphs": np.zeros((SCREEN_ROWS, SCREEN_COLS), dtype=np.int16),
        "chars": np.zeros((SCREEN_ROWS, SCREEN_COLS), dtype=np.uint32),
        "colors_fg": np.zeros((SCREEN_ROWS, SCREEN_COLS, 3), dtype=np.uint8),
        "colors_bg": np.zeros((SCREEN_ROWS, SCREEN_COLS, 3), dtype=np.uint8),
        "specials": np.zeros((SCREEN_ROWS, SCREEN_COLS), dtype=np.uint8),
        "map_layers": np.zeros((MAP_ROWS, MAP_COLS, TERRAIN_LAYERS), dtype=np.uint16),
        "map_flags": np.zeros((MAP_ROWS, MAP_COLS), dtype=np.uint64),
        "map_volume": np.zeros((MAP_ROWS, MAP_COLS), dtype=np.uint16),
        "map_machine": np.zeros((MAP_ROWS, MAP_COLS), dtype=np.uint8),
        "map_light": np.zeros((MAP_ROWS, MAP_COLS, 3), dtype=np.int16),
        "map_has_item": np.zeros((MAP_ROWS, MAP_COLS), dtype=np.uint8),
        "map_item_category": np.zeros((MAP_ROWS, MAP_COLS), dtype=np.uint16),
        "map_item_kind": map_unknown.copy(),
        "map_item_quantity": map_unknown.copy(),
        "map_item_flags": np.zeros((MAP_ROWS, MAP_COLS), dtype=np.uint64),
        "map_has_monster": np.zeros((MAP_ROWS, MAP_COLS), dtype=np.uint8),
        "map_monster_kind": map_unknown.copy(),
        "map_monster_hp": map_unknown.copy(),
        "map_monster_state": map_unknown.copy(),
        "inventory_present": np.zeros((INVENTORY_SIZE,), dtype=np.uint8),
        "inventory_letters": np.zeros((INVENTORY_SIZE,), dtype=np.uint8),
        "inventory_strs": np.zeros((INVENTORY_SIZE, INVENTORY_STR_LENGTH), dtype=np.uint8),
        "inventory_category": np.zeros((INVENTORY_SIZE,), dtype=np.uint16),
        "inventory_kind": inventory_unknown.copy(),
        "inventory_quantity": inventory_unknown.copy(),
        "inventory_flags": np.zeros((INVENTORY_SIZE,), dtype=np.uint64),
        "inventory_enchant1": inventory_unknown.copy(),
        "inventory_enchant2": inventory_unknown.copy(),
        "inventory_charges": inventory_unknown.copy(),
        "blstats": np.zeros((BLSTATS_SIZE,), dtype=np.int64),
        "message": np.zeros((MESSAGE_SIZE,), dtype=np.uint8),
        "program_state": np.zeros((PROGRAM_STATE_SIZE,), dtype=np.uint64),
    }
    return filter_observation(
        observation,
        observation_mode=observation_mode,
        include_semantic_tensors=include_semantic_tensors,
    )


def filter_observation(
    observation: ObservationDict,
    *,
    observation_mode: ObservationMode = "player",
    include_semantic_tensors: bool | None = None,
) -> ObservationDict:
    """Return a Gym-facing observation subset from a full bridge observation."""

    mode = resolve_observation_mode(
        observation_mode,
        include_semantic_tensors=include_semantic_tensors,
    )
    keys = BRIDGE_OBSERVATION_KEYS if mode == "privileged" else BASE_OBSERVATION_KEYS
    return {key: observation[key] for key in keys}
