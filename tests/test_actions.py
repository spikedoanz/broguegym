from __future__ import annotations

from typing import Any, cast

import pytest

from bruhogue.actions import Action, ActionKind, BrogueInput, Direction


def test_direction_actions_encode_to_brogue_keys() -> None:
    assert Action.move(Direction.NORTH).brogue_key() == "k"
    assert Action.move(Direction.SOUTH).brogue_key() == "j"
    assert Action.move(Direction.WEST).brogue_key() == "h"
    assert Action.move(Direction.EAST).brogue_key() == "l"
    assert Action.move(Direction.NORTHWEST).brogue_key() == "y"
    assert Action.move(Direction.NORTHEAST).brogue_key() == "u"
    assert Action.move(Direction.SOUTHWEST).brogue_key() == "b"
    assert Action.move(Direction.SOUTHEAST).brogue_key() == "n"


def test_run_actions_encode_to_modified_movement_keys() -> None:
    assert Action.move(Direction.NORTH, run=True).brogue_input() == BrogueInput(
        key="k",
        control=True,
    )


def test_common_actions_encode_to_brogue_keys() -> None:
    assert Action(kind=ActionKind.REST).brogue_key() == "z"
    assert Action(kind=ActionKind.AUTO_REST).brogue_key() == "Z"
    assert Action(kind=ActionKind.SEARCH).brogue_key() == "s"
    assert Action(kind=ActionKind.LONG_SEARCH).brogue_input() == BrogueInput(
        key="s",
        control=True,
    )
    assert Action(kind=ActionKind.ASCEND).brogue_key() == "<"
    assert Action(kind=ActionKind.DESCEND).brogue_key() == ">"
    assert Action(kind=ActionKind.INVENTORY).brogue_key() == "i"
    assert Action(kind=ActionKind.APPLY).brogue_key() == "a"
    assert Action(kind=ActionKind.THROW).brogue_key() == "t"
    assert Action(kind=ActionKind.EXPLORE).brogue_key() == "x"
    assert Action(kind=ActionKind.FAST_EXPLORE).brogue_input() == BrogueInput(
        key="x",
        control=True,
    )
    assert Action(kind=ActionKind.ACKNOWLEDGE).brogue_key() == " "
    assert Action(kind=ActionKind.ESCAPE).brogue_key() == "\x1b"


def test_inspect_is_not_a_keypress() -> None:
    with pytest.raises(ValueError, match="inspect actions"):
        Action.inspect(10, 20).brogue_key()


def test_raw_keypress_must_be_one_character() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        Action.keypress("too long")


def test_action_records_require_total_keyword_construction() -> None:
    dynamic_action = cast(Any, Action)
    dynamic_input = cast(Any, BrogueInput)

    with pytest.raises(TypeError, match="Extra positional"):
        dynamic_action(ActionKind.REST)
    with pytest.raises(TypeError, match="Missing required argument 'kind'"):
        dynamic_action()
    with pytest.raises(TypeError, match="Unexpected keyword argument 'nope'"):
        dynamic_action(kind=ActionKind.REST, nope=True)
    with pytest.raises(TypeError, match="Extra positional"):
        dynamic_input("x")
