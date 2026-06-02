"""Action types shared by agents, tests, and the future Brogue backend."""

from __future__ import annotations

from enum import StrEnum

import msgspec


class Direction(StrEnum):
    """The eight movement directions supported by Brogue's vi-style controls."""

    NORTH = "north"
    SOUTH = "south"
    WEST = "west"
    EAST = "east"
    NORTHWEST = "northwest"
    NORTHEAST = "northeast"
    SOUTHWEST = "southwest"
    SOUTHEAST = "southeast"


# These key values mirror Brogue CE's definitions in src/brogue/Rogue.h.
_DIRECTION_KEYS: dict[Direction, str] = {
    Direction.NORTH: "k",
    Direction.SOUTH: "j",
    Direction.WEST: "h",
    Direction.EAST: "l",
    Direction.NORTHWEST: "y",
    Direction.NORTHEAST: "u",
    Direction.SOUTHWEST: "b",
    Direction.SOUTHEAST: "n",
}


class ActionKind(StrEnum):
    """Protocol-level actions the Python harness knows how to encode."""

    KEY = "key"
    MOVE = "move"
    RUN = "run"
    REST = "rest"
    AUTO_REST = "auto_rest"
    SEARCH = "search"
    LONG_SEARCH = "long_search"
    ASCEND = "ascend"
    DESCEND = "descend"
    INVENTORY = "inventory"
    EQUIP = "equip"
    UNEQUIP = "unequip"
    APPLY = "apply"
    THROW = "throw"
    RETHROW = "rethrow"
    DROP = "drop"
    CALL = "call"
    RELABEL = "relabel"
    SWAP = "swap"
    EXPLORE = "explore"
    FAST_EXPLORE = "fast_explore"
    MESSAGE_ARCHIVE = "message_archive"
    DISCOVERIES = "discoveries"
    FEATS = "feats"
    HELP = "help"
    TRUE_COLORS = "true_colors"
    STEALTH_RANGE = "stealth_range"
    CURSOR = "cursor"
    ACKNOWLEDGE = "acknowledge"
    ESCAPE = "escape"
    INSPECT = "inspect"


class BrogueInput(msgspec.Struct, frozen=True, kw_only=True, forbid_unknown_fields=True):
    """One key event as Brogue's input loop sees it."""

    key: str
    control: bool = False
    shift: bool = False

    def __post_init__(self) -> None:
        if len(self.key) != 1:
            msg = "Brogue inputs must contain exactly one character"
            raise ValueError(msg)


class Action(msgspec.Struct, frozen=True, kw_only=True, forbid_unknown_fields=True):
    """An immutable action sent to the Brogue backend.

    The class intentionally models low-level Brogue inputs first. Higher-level RL helpers can build
    on top of these primitives without changing the backend ABI.
    """

    kind: ActionKind
    key: str | None = None
    direction: Direction | None = None
    x: int | None = None
    y: int | None = None
    control: bool = False
    shift: bool = False

    @classmethod
    def keypress(cls, key: str, *, control: bool = False, shift: bool = False) -> Action:
        """Create a raw key action, validating that it is a single logical key."""

        if len(key) != 1:
            msg = "keypress actions must contain exactly one character"
            raise ValueError(msg)
        return cls(kind=ActionKind.KEY, key=key, control=control, shift=shift)

    @classmethod
    def move(cls, direction: Direction, *, run: bool = False) -> Action:
        """Create a movement action in one of Brogue's eight directions."""

        return cls(kind=ActionKind.RUN if run else ActionKind.MOVE, direction=direction)

    @classmethod
    def inspect(cls, x: int, y: int) -> Action:
        """Create a non-turn-taking request for a description of a map cell."""

        return cls(kind=ActionKind.INSPECT, x=x, y=y)

    def brogue_key(self) -> str:
        """Return the Brogue keyboard command represented by this action.

        Inspect actions are intentionally excluded because they should be serviced by the
        backend API directly, not translated into cursor movement.
        """

        return self.brogue_input().key

    def brogue_input(self) -> BrogueInput:
        """Return the Brogue key event represented by this action."""

        match self.kind:
            case ActionKind.KEY:
                if self.key is None:
                    msg = "raw key action is missing its key"
                    raise ValueError(msg)
                return BrogueInput(key=self.key, control=self.control, shift=self.shift)
            case ActionKind.MOVE:
                if self.direction is None:
                    msg = "move action is missing its direction"
                    raise ValueError(msg)
                return BrogueInput(key=_DIRECTION_KEYS[self.direction])
            case ActionKind.RUN:
                if self.direction is None:
                    msg = "run action is missing its direction"
                    raise ValueError(msg)
                return BrogueInput(key=_DIRECTION_KEYS[self.direction], control=True)
            case ActionKind.REST:
                return BrogueInput(key="z")
            case ActionKind.AUTO_REST:
                return BrogueInput(key="Z")
            case ActionKind.SEARCH:
                return BrogueInput(key="s")
            case ActionKind.LONG_SEARCH:
                return BrogueInput(key="s", control=True)
            case ActionKind.ASCEND:
                return BrogueInput(key="<")
            case ActionKind.DESCEND:
                return BrogueInput(key=">")
            case ActionKind.INVENTORY:
                return BrogueInput(key="i")
            case ActionKind.EQUIP:
                return BrogueInput(key="e")
            case ActionKind.UNEQUIP:
                return BrogueInput(key="r")
            case ActionKind.APPLY:
                return BrogueInput(key="a")
            case ActionKind.THROW:
                return BrogueInput(key="t")
            case ActionKind.RETHROW:
                return BrogueInput(key="T")
            case ActionKind.DROP:
                return BrogueInput(key="d")
            case ActionKind.CALL:
                return BrogueInput(key="c")
            case ActionKind.RELABEL:
                return BrogueInput(key="R")
            case ActionKind.SWAP:
                return BrogueInput(key="w")
            case ActionKind.EXPLORE:
                return BrogueInput(key="x")
            case ActionKind.FAST_EXPLORE:
                return BrogueInput(key="x", control=True)
            case ActionKind.MESSAGE_ARCHIVE:
                return BrogueInput(key="M")
            case ActionKind.DISCOVERIES:
                return BrogueInput(key="D")
            case ActionKind.FEATS:
                return BrogueInput(key="F")
            case ActionKind.HELP:
                return BrogueInput(key="?")
            case ActionKind.TRUE_COLORS:
                return BrogueInput(key="\\")
            case ActionKind.STEALTH_RANGE:
                return BrogueInput(key="]")
            case ActionKind.CURSOR:
                return BrogueInput(key="\n")
            case ActionKind.ACKNOWLEDGE:
                return BrogueInput(key=" ")
            case ActionKind.ESCAPE:
                return BrogueInput(key="\x1b")
            case ActionKind.INSPECT:
                msg = "inspect actions do not map to a single Brogue key"
                raise ValueError(msg)
