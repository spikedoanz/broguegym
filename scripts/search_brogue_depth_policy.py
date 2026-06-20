#!/usr/bin/env python3
"""Search for simple Brogue depth policies over concrete dungeon seeds.

This script is intentionally not an RL trainer. It evaluates a deterministic
hand policy against the same Python bridge observations that the Gym wrapper
exposes, with exact Brogue seed overrides for reproducibility.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

import numpy as np

from broguegym.actions import Action
from broguegym.brogue import BackendStep, BrogueBackend, ObservationDict
from broguegym.spaces import (
    PROGRAM_DEPTH_INDEX,
    PROGRAM_GOLD_INDEX,
    PROGRAM_SCORE_INDEX,
    PROGRAM_SEED_INDEX,
    PROGRAM_TERMINATED_INDEX,
    PROGRAM_TURN_INDEX,
)

MAP_COLS: Final = 79
MAP_ROWS: Final = 29
SCREEN_MAP_X0: Final = 21
SCREEN_MAP_Y0: Final = 3

DISCOVERED: Final = 1 << 0
VISIBLE: Final = 1 << 1
HAS_PLAYER: Final = 1 << 2
MAGIC_MAPPED: Final = 1 << 11
CLAIRVOYANT_VISIBLE: Final = 1 << 13
TELEPATHIC_VISIBLE: Final = 1 << 29
KNOWN_MASK: Final = (
    DISCOVERED | VISIBLE | MAGIC_MAPPED | CLAIRVOYANT_VISIBLE | TELEPATHIC_VISIBLE
)

NOTHING: Final = 0
GRANITE: Final = 1
FLOOR: Final = 2
WALL: Final = 6
DOOR: Final = 7
OPEN_DOOR: Final = 8
SECRET_DOOR: Final = 9
LOCKED_DOOR: Final = 10
DOWN_STAIRS: Final = 12
UP_STAIRS: Final = 13
TORCH_WALL: Final = 16
CRYSTAL_WALL: Final = 17
PORTCULLIS_CLOSED: Final = 18
PORTCULLIS_DORMANT: Final = 19
WOODEN_BARRICADE: Final = 20
WALL_LEVER_HIDDEN: Final = 26
WALL_LEVER: Final = 27
WALL_LEVER_PULLED: Final = 28
WALL_LEVER_HIDDEN_DORMANT: Final = 29
STATUE_INERT: Final = 30
STATUE_DORMANT: Final = 31
PORTAL: Final = 34
WALL_MONSTER_DORMANT: Final = 36
TRAP_DOOR: Final = 57
DEEP_WATER: Final = 86
SHALLOW_WATER: Final = 87
CHASM: Final = 89
LAVA: Final = 93
LAVA_RETRACTABLE: Final = 94
LAVA_RETRACTING: Final = 95
BRIDGE: Final = 101
STONE_BRIDGE: Final = 104
HOLE: Final = 112
SPIDERWEB: Final = 140
NETTING: Final = 141
FORCEFIELD: Final = 147
PLAIN_FIRE: Final = 160
BRIMSTONE_FIRE: Final = 161
FLAMEDANCER_FIRE: Final = 162
GAS_FIRE: Final = 163
GAS_EXPLOSION: Final = 164
DART_EXPLOSION: Final = 165
ITEM_FIRE: Final = 166
CREATURE_FIRE: Final = 167
POISON_GAS: Final = 168
CONFUSION_GAS: Final = 169
ROT_GAS: Final = 170
STENCH_SMOKE_GAS: Final = 171
PARALYSIS_GAS: Final = 172
METHANE_GAS: Final = 173
STEAM: Final = 174
DARKNESS_CLOUD: Final = 175
HEALING_CLOUD: Final = 176
SACRIFICE_LAVA: Final = 195
STATUE_INERT_DOORWAY: Final = 198
STATUE_DORMANT_DOORWAY: Final = 199
CHASM_WITH_HIDDEN_BRIDGE: Final = 200
CHASM_WITH_HIDDEN_BRIDGE_ACTIVE: Final = 201
RAT_TRAP_WALL_DORMANT: Final = 203
RAT_TRAP_WALL_CRACKING: Final = 204
WORM_TUNNEL_OUTER_WALL: Final = 210
MUD_WALL: Final = 213

BLOCKING_DUNGEON_TILES: Final = {
    NOTHING,
    GRANITE,
    WALL,
    SECRET_DOOR,
    TORCH_WALL,
    CRYSTAL_WALL,
    PORTCULLIS_CLOSED,
    PORTCULLIS_DORMANT,
    WOODEN_BARRICADE,
    WALL_LEVER_HIDDEN,
    WALL_LEVER,
    WALL_LEVER_PULLED,
    WALL_LEVER_HIDDEN_DORMANT,
    STATUE_INERT,
    STATUE_DORMANT,
    WALL_MONSTER_DORMANT,
    FORCEFIELD,
    RAT_TRAP_WALL_DORMANT,
    RAT_TRAP_WALL_CRACKING,
    WORM_TUNNEL_OUTER_WALL,
    MUD_WALL,
}
HAZARD_TILES: Final = {
    DEEP_WATER,
    CHASM,
    LAVA,
    LAVA_RETRACTABLE,
    LAVA_RETRACTING,
    HOLE,
    SPIDERWEB,
    NETTING,
    PLAIN_FIRE,
    CREATURE_FIRE,
    SACRIFICE_LAVA,
}
ACTIVE_HAZARD_TILES: Final = HAZARD_TILES | {
    BRIMSTONE_FIRE,
    FLAMEDANCER_FIRE,
    GAS_FIRE,
    GAS_EXPLOSION,
    DART_EXPLOSION,
    ITEM_FIRE,
    POISON_GAS,
    CONFUSION_GAS,
    ROT_GAS,
    STENCH_SMOKE_GAS,
    PARALYSIS_GAS,
    METHANE_GAS,
    STEAM,
}
SAFE_BRIDGE_TILES: Final = {
    BRIDGE,
    STONE_BRIDGE,
    CHASM_WITH_HIDDEN_BRIDGE_ACTIVE,
}

FOOD_CATEGORY: Final = 1 << 0
WEAPON_CATEGORY: Final = 1 << 1
ARMOR_CATEGORY: Final = 1 << 2
POTION_CATEGORY: Final = 1 << 3
SCROLL_CATEGORY: Final = 1 << 4
STAFF_CATEGORY: Final = 1 << 5
WAND_CATEGORY: Final = 1 << 6
RING_CATEGORY: Final = 1 << 7
CHARM_CATEGORY: Final = 1 << 8

ITEM_EQUIPPED: Final = 1 << 1
ITEM_CAN_BE_IDENTIFIED: Final = 1 << 8
UNKNOWN_SHORT: Final = np.iinfo(np.int16).min
STRENGTH_RE: Final = re.compile(r"<(\d+)>")

DIRECTIONS: Final = (
    (-1, 0, "h"),
    (1, 0, "l"),
    (0, -1, "k"),
    (0, 1, "j"),
    (-1, -1, "y"),
    (1, -1, "u"),
    (-1, 1, "b"),
    (1, 1, "n"),
)
CARDINALS: Final = DIRECTIONS[:4]
MOVE_KEYS: Final = {direction[2] for direction in DIRECTIONS}

KEY_ACTIONS: Final = {
    "h": Action.keypress("h"),
    "j": Action.keypress("j"),
    "k": Action.keypress("k"),
    "l": Action.keypress("l"),
    "y": Action.keypress("y"),
    "u": Action.keypress("u"),
    "b": Action.keypress("b"),
    "n": Action.keypress("n"),
    ">": Action.keypress(">"),
    "s": Action.keypress("s"),
    "Z": Action.keypress("Z"),
    " ": Action.keypress(" "),
    ".": Action.keypress("."),
    "\n": Action.keypress("\n"),
    "\x1b": Action.keypress("\x1b"),
    "x": Action.keypress("x", control=True),
    "z": Action.keypress("z"),
    "a": Action.keypress("a"),
    "e": Action.keypress("e"),
}


@dataclass
class PolicyState:
    action_queue: list[str] = field(default_factory=list)
    pending_path: list[str] = field(default_factory=list)
    path_target: tuple[int, int] | None = None
    path_mode: str = ""
    auto_failures: int = 0
    search_count: int = 0
    blocked_targets: set[tuple[int, int]] = field(default_factory=set)
    blocked_cells: set[tuple[int, int]] = field(default_factory=set)
    tried_items: set[tuple[int, str]] = field(default_factory=set)
    searched_secret_targets: set[tuple[int, int]] = field(default_factory=set)
    secret_search_target: tuple[int, int] | None = None
    secret_search_count: int = 0
    last_action: str = ""
    no_progress_count: int = 0
    pending_item_prompt: str = ""
    trace: list[str] = field(default_factory=list)

    def clear_path(self) -> None:
        self.pending_path.clear()
        self.path_target = None
        self.path_mode = ""

    def reset_level_state(self) -> None:
        self.clear_path()
        self.pending_item_prompt = ""
        self.auto_failures = 0
        self.search_count = 0
        self.blocked_targets.clear()
        self.blocked_cells.clear()
        self.searched_secret_targets.clear()
        self.secret_search_target = None
        self.secret_search_count = 0


@dataclass
class EpisodeState:
    seed: int
    policy: PolicyState = field(default_factory=PolicyState)
    steps: int = 0
    done: bool = False
    max_depth: int = 1
    final_depth: int = 1
    score: int = 0
    gold: int = 0
    turn: int = 0
    terminated: bool = False
    last_message: str = ""
    no_change_steps: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a scripted Brogue depth policy over exact dungeon seeds.",
    )
    parser.add_argument("--start-seed", type=int, default=1)
    parser.add_argument("--episodes", type=int, default=64)
    parser.add_argument("--parallel-envs", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=4096)
    parser.add_argument("--target-depth", type=int, default=10)
    parser.add_argument("--auto-retries", type=int, default=2)
    parser.add_argument("--max-no-change-steps", type=int, default=64)
    parser.add_argument("--progress-interval", type=int, default=64)
    parser.add_argument("--rest-hp-frac", type=float, default=0.0)
    parser.add_argument("--rest-min-depth", type=int, default=1)
    parser.add_argument("--equip-items", action="store_true")
    parser.add_argument("--panic-items", action="store_true")
    parser.add_argument("--panic-min-depth", type=int, default=1)
    parser.add_argument("--panic-scrolls", action="store_true")
    parser.add_argument("--panic-scrolls-min-depth", type=int, default=8)
    parser.add_argument("--panic-zaps", action="store_true")
    parser.add_argument("--panic-zaps-min-depth", type=int, default=8)
    parser.add_argument("--panic-near-radius", type=int, default=-1)
    parser.add_argument("--escape-active-hazards", action="store_true")
    parser.add_argument("--avoid-monsters-min-depth", type=int, default=0)
    parser.add_argument("--adjacent-attack-min-hp-frac", type=float, default=0.45)
    parser.add_argument("--secret-search-max", type=int, default=0)
    parser.add_argument("--stuck-dive-min-depth", type=int, default=0)
    parser.add_argument("--stuck-dive-min-hp-frac", type=float, default=0.6)
    parser.add_argument("--native-stairs", action="store_true")
    parser.add_argument("--trace-dir", type=Path)
    parser.add_argument("--stop-on-success", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def message_text(obs: ObservationDict) -> str:
    raw = bytes(obs["message"].tolist()).split(b"\0", 1)[0]
    return raw.decode("latin-1", errors="replace")


def stats_from_obs(obs: ObservationDict) -> tuple[int, int, int, int, bool]:
    program_state = obs["program_state"]
    depth = int(program_state[PROGRAM_DEPTH_INDEX])
    turn = int(program_state[PROGRAM_TURN_INDEX])
    gold = int(program_state[PROGRAM_GOLD_INDEX])
    score = int(program_state[PROGRAM_SCORE_INDEX])
    terminated = bool(int(program_state[PROGRAM_TERMINATED_INDEX]))
    return depth, turn, gold, score, terminated


def player_position(obs: ObservationDict) -> tuple[int, int]:
    blstats = obs["blstats"]
    return int(blstats[0]), int(blstats[1])


def in_bounds(x: int, y: int) -> bool:
    return 0 <= x < MAP_COLS and 0 <= y < MAP_ROWS


def known(obs: ObservationDict, x: int, y: int) -> bool:
    if not in_bounds(x, y):
        return False
    return bool(int(obs["map_flags"][y, x]) & KNOWN_MASK)


def visible(obs: ObservationDict, x: int, y: int) -> bool:
    return in_bounds(x, y) and bool(int(obs["map_flags"][y, x]) & VISIBLE)


def inventory_name(obs: ObservationDict, slot: int) -> str:
    raw = bytes(obs["inventory_strs"][slot].tolist()).split(b"\0", 1)[0]
    return raw.decode("latin-1", errors="replace")


def item_key(obs: ObservationDict, slot: int) -> tuple[int, str]:
    return int(obs["inventory_letters"][slot]), inventory_name(obs, slot)


def inventory_slots(obs: ObservationDict, category_mask: int | None = None) -> list[int]:
    slots: list[int] = []
    for slot in range(len(obs["inventory_present"])):
        if not int(obs["inventory_present"][slot]):
            continue
        category = int(obs["inventory_category"][slot])
        if category_mask is not None and not (category & category_mask):
            continue
        slots.append(slot)
    return slots


def item_letter(obs: ObservationDict, slot: int) -> str:
    letter = int(obs["inventory_letters"][slot])
    if 0 < letter < 256:
        return chr(letter)
    return ""


def strength_requirement(name: str) -> int | None:
    match = STRENGTH_RE.search(name)
    if match is None:
        return None
    return int(match.group(1))


def hp_fraction(obs: ObservationDict) -> float:
    hp = max(0, int(obs["blstats"][3]))
    max_hp = max(1, int(obs["blstats"][4]))
    return hp / max_hp


def screen_char(obs: ObservationDict, x: int, y: int) -> int:
    if not in_bounds(x, y):
        return ord(" ")
    return int(obs["chars"][SCREEN_MAP_Y0 + y, SCREEN_MAP_X0 + x])


def terrain_layers(obs: ObservationDict, x: int, y: int) -> tuple[int, int, int, int]:
    layers = obs["map_layers"][y, x]
    return int(layers[0]), int(layers[1]), int(layers[2]), int(layers[3])


def has_monster(obs: ObservationDict, x: int, y: int) -> bool:
    if not in_bounds(x, y):
        return False
    return bool(int(obs["map_has_monster"][y, x]))


def is_hazardous(obs: ObservationDict, x: int, y: int) -> bool:
    dungeon, liquid, gas, surface = terrain_layers(obs, x, y)
    if dungeon in SAFE_BRIDGE_TILES or surface in SAFE_BRIDGE_TILES:
        return False
    return (
        dungeon in HAZARD_TILES
        or liquid in HAZARD_TILES
        or surface in HAZARD_TILES
        or gas in HAZARD_TILES
    )


def is_active_hazardous(obs: ObservationDict, x: int, y: int) -> bool:
    dungeon, liquid, gas, surface = terrain_layers(obs, x, y)
    if dungeon in SAFE_BRIDGE_TILES or surface in SAFE_BRIDGE_TILES:
        return False
    return (
        dungeon in ACTIVE_HAZARD_TILES
        or liquid in ACTIVE_HAZARD_TILES
        or surface in ACTIVE_HAZARD_TILES
        or gas in ACTIVE_HAZARD_TILES
    )


def is_passable(obs: ObservationDict, x: int, y: int, *, avoid_hazards: bool = True) -> bool:
    if not known(obs, x, y):
        return False
    dungeon, liquid, _gas, surface = terrain_layers(obs, x, y)
    if dungeon in {DOWN_STAIRS, UP_STAIRS, FLOOR, DOOR, OPEN_DOOR, LOCKED_DOOR, PORTAL}:
        pass
    elif dungeon in {STATUE_INERT_DOORWAY, STATUE_DORMANT_DOORWAY}:
        pass
    elif dungeon in BLOCKING_DUNGEON_TILES:
        return False
    if avoid_hazards and is_hazardous(obs, x, y):
        return False
    if liquid == DEEP_WATER and avoid_hazards:
        return False
    if surface in {FORCEFIELD, PLAIN_FIRE, CREATURE_FIRE}:
        return False
    return True


def dangerous_cells(obs: ObservationDict) -> set[tuple[int, int]]:
    player = player_position(obs)
    danger: set[tuple[int, int]] = set()
    monster_cells = np.argwhere(obs["map_has_monster"] != 0)
    for y_np, x_np in monster_cells:
        x = int(x_np)
        y = int(y_np)
        if (x, y) == player or not visible(obs, x, y):
            continue
        danger.add((x, y))
        for dx, dy, _key in DIRECTIONS:
            nx = x + dx
            ny = y + dy
            if in_bounds(nx, ny):
                danger.add((nx, ny))
    danger.discard(player)
    return danger


def visible_monsters(obs: ObservationDict) -> list[tuple[int, int]]:
    monsters: list[tuple[int, int]] = []
    player = player_position(obs)
    monster_cells = np.argwhere(obs["map_has_monster"] != 0)
    for y_np, x_np in monster_cells:
        x = int(x_np)
        y = int(y_np)
        if (x, y) != player and visible(obs, x, y):
            monsters.append((x, y))
    return monsters


def adjacent_monster_count(obs: ObservationDict) -> int:
    px, py = player_position(obs)
    count = 0
    for dx, dy, _key in DIRECTIONS:
        nx = px + dx
        ny = py + dy
        if in_bounds(nx, ny) and has_monster(obs, nx, ny) and visible(obs, nx, ny):
            count += 1
    return count


def nearest_visible_monster_distance(obs: ObservationDict) -> int | None:
    player = player_position(obs)
    monsters = visible_monsters(obs)
    if not monsters:
        return None
    px, py = player
    return min(max(abs(px - mx), abs(py - my)) for mx, my in monsters)


def down_stair_targets(obs: ObservationDict) -> set[tuple[int, int]]:
    targets: set[tuple[int, int]] = set()
    ys, xs = np.where(obs["map_layers"][:, :, 0] == DOWN_STAIRS)
    for y_np, x_np in zip(ys, xs, strict=True):
        x = int(x_np)
        y = int(y_np)
        if known(obs, x, y):
            targets.add((x, y))
    screen = obs["chars"][SCREEN_MAP_Y0 : SCREEN_MAP_Y0 + MAP_ROWS, SCREEN_MAP_X0 : SCREEN_MAP_X0 + MAP_COLS]
    ys, xs = np.where(screen == ord(">"))
    for y_np, x_np in zip(ys, xs, strict=True):
        x = int(x_np)
        y = int(y_np)
        if known(obs, x, y):
            targets.add((x, y))
    return targets


def frontier_targets(obs: ObservationDict, blocked: set[tuple[int, int]]) -> set[tuple[int, int]]:
    targets: set[tuple[int, int]] = set()
    for y in range(MAP_ROWS):
        for x in range(MAP_COLS):
            if (x, y) in blocked or not is_passable(obs, x, y):
                continue
            if has_monster(obs, x, y):
                continue
            for dx, dy, _key in CARDINALS:
                nx = x + dx
                ny = y + dy
                if in_bounds(nx, ny) and not known(obs, nx, ny):
                    targets.add((x, y))
                    break
    return targets


def secret_search_targets(obs: ObservationDict, searched: set[tuple[int, int]]) -> set[tuple[int, int]]:
    targets: set[tuple[int, int]] = set()
    for y in range(MAP_ROWS):
        for x in range(MAP_COLS):
            if (x, y) in searched or not is_passable(obs, x, y):
                continue
            passable_neighbors = 0
            blocked_neighbors = 0
            for dx, dy, _key in CARDINALS:
                nx = x + dx
                ny = y + dy
                if not in_bounds(nx, ny):
                    continue
                if is_passable(obs, nx, ny):
                    passable_neighbors += 1
                elif known(obs, nx, ny):
                    blocked_neighbors += 1
            if blocked_neighbors >= 2 and passable_neighbors <= 2:
                targets.add((x, y))
    return targets


def descent_hazard_targets(obs: ObservationDict) -> set[tuple[int, int]]:
    targets: set[tuple[int, int]] = set()
    descent_tiles = {CHASM, HOLE, TRAP_DOOR}
    for y in range(MAP_ROWS):
        for x in range(MAP_COLS):
            if not known(obs, x, y) or has_monster(obs, x, y):
                continue
            layers = terrain_layers(obs, x, y)
            if any(tile in descent_tiles for tile in layers):
                targets.add((x, y))
    return targets


def bfs_path(
    obs: ObservationDict,
    targets: set[tuple[int, int]],
    *,
    avoid_danger: bool,
    avoid_hazards: bool = True,
    blocked_cells: set[tuple[int, int]] | None = None,
) -> list[str] | None:
    start = player_position(obs)
    blocked = blocked_cells or set()
    if start in targets:
        return []

    danger = dangerous_cells(obs) if avoid_danger else set()
    queue: deque[tuple[int, int]] = deque([start])
    parent: dict[tuple[int, int], tuple[tuple[int, int], str]] = {}
    seen = {start}

    while queue:
        x, y = queue.popleft()
        for dx, dy, key in DIRECTIONS:
            nx = x + dx
            ny = y + dy
            pos = (nx, ny)
            if pos in seen or not in_bounds(nx, ny):
                continue
            if pos in blocked and pos not in targets:
                continue
            if avoid_danger and pos in danger and pos not in targets:
                continue
            if has_monster(obs, nx, ny) and pos not in targets:
                continue
            if not is_passable(obs, nx, ny, avoid_hazards=avoid_hazards):
                continue
            if dx and dy:
                if (nx, y) in blocked or (x, ny) in blocked:
                    continue
                if not is_passable(obs, nx, y, avoid_hazards=avoid_hazards):
                    continue
                if not is_passable(obs, x, ny, avoid_hazards=avoid_hazards):
                    continue
            seen.add(pos)
            parent[pos] = ((x, y), key)
            if pos in targets:
                return reconstruct_path(parent, start, pos)
            queue.append(pos)
    return None


def reconstruct_path(
    parent: dict[tuple[int, int], tuple[tuple[int, int], str]],
    start: tuple[int, int],
    end: tuple[int, int],
) -> list[str]:
    keys: list[str] = []
    current = end
    while current != start:
        previous, key = parent[current]
        keys.append(key)
        current = previous
    keys.reverse()
    return keys


def choose_action(
    obs: ObservationDict,
    state: PolicyState,
    *,
    auto_retries: int,
    rest_hp_frac: float,
    rest_min_depth: int,
    equip_items: bool,
    panic_items: bool,
    panic_min_depth: int,
    panic_scrolls: bool,
    panic_scrolls_min_depth: int,
    panic_zaps: bool,
    panic_zaps_min_depth: int,
    panic_near_radius: int,
    escape_active_hazards: bool,
    avoid_monsters_min_depth: int,
    adjacent_attack_min_hp_frac: float,
    secret_search_max: int,
    stuck_dive_min_depth: int,
    stuck_dive_min_hp_frac: float,
    native_stairs: bool,
) -> str:
    if state.action_queue:
        return state.action_queue.pop(0)

    msg = message_text(obs).lower()
    if "press space" in msg or "--more--" in msg:
        state.clear_path()
        return " "
    if "game over" in msg or "press any key" in msg and "stop" not in msg:
        state.clear_path()
        return " "
    if state.pending_item_prompt in {"scroll", "enchant"} and (
        "scroll of enchanting" in msg or "can't enchant that" in msg
    ):
        if state.pending_item_prompt != "enchant":
            state.pending_item_prompt = "enchant"
            return " "
        enchant_letter = choose_enchant_target(obs)
        if enchant_letter:
            state.action_queue.clear()
            state.clear_path()
            state.pending_item_prompt = ""
            return enchant_letter
    if state.pending_item_prompt in {"scroll", "identify"} and (
        "scroll of identify" in msg or "identify what" in msg
    ):
        if state.pending_item_prompt != "identify":
            state.pending_item_prompt = "identify"
            return " "
        identify_letter = choose_identify_target(obs)
        if identify_letter:
            state.action_queue.clear()
            state.clear_path()
            state.pending_item_prompt = ""
            return identify_letter
    if state.no_progress_count >= 3:
        state.action_queue.clear()
        state.clear_path()
        state.pending_item_prompt = ""
        return "\x1b" if state.no_progress_count % 2 else " "

    px, py = player_position(obs)
    dungeon, _liquid, _gas, _surface = terrain_layers(obs, px, py)
    if dungeon == DOWN_STAIRS:
        state.clear_path()
        return ">"

    survival_action = choose_survival_action(
        obs,
        state,
        rest_hp_frac=rest_hp_frac,
        rest_min_depth=rest_min_depth,
        equip_items=equip_items,
        panic_items=panic_items,
        panic_min_depth=panic_min_depth,
        panic_scrolls=panic_scrolls,
        panic_scrolls_min_depth=panic_scrolls_min_depth,
        panic_zaps=panic_zaps,
        panic_zaps_min_depth=panic_zaps_min_depth,
        panic_near_radius=panic_near_radius,
        escape_active_hazards=escape_active_hazards,
    )
    if survival_action is not None:
        return survival_action

    if state.pending_path and state.path_mode == "stairs":
        return state.pending_path.pop(0)

    stairs = down_stair_targets(obs)
    if native_stairs and stairs:
        state.clear_path()
        state.action_queue.append(">")
        return ">"

    if stairs:
        path = bfs_path(obs, stairs, avoid_danger=True, blocked_cells=state.blocked_cells)
        if path is None:
            path = bfs_path(obs, stairs, avoid_danger=False, blocked_cells=state.blocked_cells)
        if path == []:
            state.clear_path()
            return ">"
        if path:
            state.pending_path = path[1:]
            state.path_mode = "stairs"
            state.path_target = None
            return path[0]

    avoid_action = choose_avoidance_action(
        obs,
        state,
        avoid_monsters_min_depth=avoid_monsters_min_depth,
        adjacent_attack_min_hp_frac=adjacent_attack_min_hp_frac,
    )
    if avoid_action is not None:
        return avoid_action

    if state.pending_path:
        return state.pending_path.pop(0)

    if state.auto_failures < auto_retries:
        state.path_mode = "auto"
        return "x"

    frontiers = frontier_targets(obs, state.blocked_targets)
    if frontiers:
        path = bfs_path(obs, frontiers, avoid_danger=True, blocked_cells=state.blocked_cells)
        if path is None:
            path = bfs_path(obs, frontiers, avoid_danger=False, blocked_cells=state.blocked_cells)
        if path:
            state.pending_path = path[1:]
            state.path_mode = "frontier"
            state.path_target = path_endpoint(player_position(obs), path)
            return path[0]
        if path == []:
            state.search_count += 1
            return "s" if state.search_count <= 4 else "x"

    if stuck_dive_min_depth > 0:
        depth = int(obs["program_state"][PROGRAM_DEPTH_INDEX])
        if depth >= stuck_dive_min_depth and hp_fraction(obs) >= stuck_dive_min_hp_frac:
            dive_targets = descent_hazard_targets(obs)
            if dive_targets:
                path = bfs_path(
                    obs,
                    dive_targets,
                    avoid_danger=True,
                    avoid_hazards=False,
                    blocked_cells=state.blocked_cells,
                )
                if path is None:
                    path = bfs_path(
                        obs,
                        dive_targets,
                        avoid_danger=False,
                        avoid_hazards=False,
                        blocked_cells=state.blocked_cells,
                    )
                if path == []:
                    state.clear_path()
                    return "."
                if path:
                    state.pending_path = path[1:]
                    state.path_mode = "dive"
                    state.path_target = path_endpoint(player_position(obs), path)
                    return path[0]

    secret_action = choose_secret_search_action(
        obs,
        state,
        search_max=secret_search_max,
    )
    if secret_action is not None:
        return secret_action

    state.search_count += 1
    if state.search_count <= 6:
        return "s"
    state.search_count = 0
    state.auto_failures = 0
    return "x"


def choose_survival_action(
    obs: ObservationDict,
    state: PolicyState,
    *,
    rest_hp_frac: float,
    rest_min_depth: int,
    equip_items: bool,
    panic_items: bool,
    panic_min_depth: int,
    panic_scrolls: bool,
    panic_scrolls_min_depth: int,
    panic_zaps: bool,
    panic_zaps_min_depth: int,
    panic_near_radius: int,
    escape_active_hazards: bool,
) -> str | None:
    hp_frac = hp_fraction(obs)
    depth = int(obs["program_state"][PROGRAM_DEPTH_INDEX])
    visible_count = len(visible_monsters(obs))
    adjacent_count = adjacent_monster_count(obs)
    danger = visible_count > 0 or adjacent_count > 0

    if escape_active_hazards:
        hazard_action = escape_hazard_action(obs, state.blocked_cells)
        if hazard_action is not None:
            state.clear_path()
            return hazard_action

    if rest_hp_frac > 0 and depth >= rest_min_depth and hp_frac < rest_hp_frac and not danger:
        state.clear_path()
        return "Z"

    if equip_items:
        equip_action = choose_equipment_action(obs, state)
        if equip_action is not None and not danger:
            return equip_action

    if not panic_items or depth < panic_min_depth:
        return None

    if panic_near_radius >= 0:
        nearest = nearest_visible_monster_distance(obs)
        should_panic = hp_frac <= 0.45 or adjacent_count > 0 or (
            nearest is not None and nearest <= panic_near_radius
        )
    else:
        should_panic = not (hp_frac > 0.45 and adjacent_count == 0 and visible_count < 2)
    if not should_panic:
        return None

    panic_action = choose_panic_item_action(
        obs,
        state,
        include_scrolls=panic_scrolls and depth >= panic_scrolls_min_depth,
        include_zaps=panic_zaps and depth >= panic_zaps_min_depth,
    )
    if panic_action is not None:
        return panic_action
    return None


def choose_avoidance_action(
    obs: ObservationDict,
    state: PolicyState,
    *,
    avoid_monsters_min_depth: int,
    adjacent_attack_min_hp_frac: float,
) -> str | None:
    if avoid_monsters_min_depth <= 0:
        return None
    depth = int(obs["program_state"][PROGRAM_DEPTH_INDEX])
    if depth < avoid_monsters_min_depth:
        return None

    if state.pending_path and state.path_mode == "avoid":
        return state.pending_path.pop(0)

    nearest_monster = nearest_visible_monster_distance(obs)
    if nearest_monster is not None and nearest_monster <= 2:
        adjacent_attack = adjacent_attack_direction(obs)
        if adjacent_attack is not None and hp_fraction(obs) >= adjacent_attack_min_hp_frac:
            state.clear_path()
            return adjacent_attack

        flee_key = flee_direction(obs, state.blocked_cells)
        if flee_key is not None:
            state.clear_path()
            return flee_key
        if adjacent_attack is not None:
            state.clear_path()
            return adjacent_attack

    frontiers = frontier_targets(obs, state.blocked_targets)
    if frontiers:
        path = bfs_path(obs, frontiers, avoid_danger=True, blocked_cells=state.blocked_cells)
        if path:
            state.pending_path = path[1:]
            state.path_mode = "avoid"
            state.path_target = path_endpoint(player_position(obs), path)
            return path[0]

    return None


def choose_secret_search_action(
    obs: ObservationDict,
    state: PolicyState,
    *,
    search_max: int,
) -> str | None:
    if search_max <= 0:
        return None

    player = player_position(obs)
    if state.secret_search_target == player:
        if state.secret_search_count < search_max:
            state.secret_search_count += 1
            state.clear_path()
            return "s"
        state.searched_secret_targets.add(player)
        state.secret_search_target = None
        state.secret_search_count = 0

    targets = secret_search_targets(obs, state.searched_secret_targets)
    if not targets:
        return None

    path = bfs_path(obs, targets, avoid_danger=True, blocked_cells=state.blocked_cells)
    if path is None:
        path = bfs_path(obs, targets, avoid_danger=False, blocked_cells=state.blocked_cells)
    if path is None:
        state.searched_secret_targets.update(targets)
        return None
    if path == []:
        state.secret_search_target = player
        state.secret_search_count = 1
        state.clear_path()
        return "s"

    state.pending_path = path[1:]
    state.path_mode = "secret"
    state.path_target = path_endpoint(player, path)
    state.secret_search_target = state.path_target
    state.secret_search_count = 0
    return path[0]


def adjacent_attack_direction(obs: ObservationDict) -> str | None:
    px, py = player_position(obs)
    best: tuple[int, str] | None = None
    for dx, dy, key in DIRECTIONS:
        nx = px + dx
        ny = py + dy
        if not in_bounds(nx, ny) or not has_monster(obs, nx, ny) or not visible(obs, nx, ny):
            continue
        hp = int(obs["map_monster_hp"][ny, nx])
        if hp < 0:
            hp = 9999
        if best is None or hp < best[0]:
            best = (hp, key)
    return None if best is None else best[1]


def flee_direction(obs: ObservationDict, blocked_cells: set[tuple[int, int]]) -> str | None:
    px, py = player_position(obs)
    monsters = visible_monsters(obs)
    if not monsters:
        return None
    danger = dangerous_cells(obs)
    best: tuple[int, str] | None = None
    for dx, dy, key in DIRECTIONS:
        nx = px + dx
        ny = py + dy
        if not in_bounds(nx, ny):
            continue
        if (nx, ny) in blocked_cells:
            continue
        if (nx, ny) in danger or has_monster(obs, nx, ny):
            continue
        if not is_passable(obs, nx, ny):
            continue
        score = min((nx - mx) * (nx - mx) + (ny - my) * (ny - my) for mx, my in monsters)
        if best is None or score > best[0]:
            best = (score, key)
    return None if best is None else best[1]


def escape_hazard_action(obs: ObservationDict, blocked_cells: set[tuple[int, int]]) -> str | None:
    px, py = player_position(obs)
    if not is_active_hazardous(obs, px, py):
        return None

    monsters = visible_monsters(obs)
    danger = dangerous_cells(obs)
    best: tuple[int, str] | None = None
    for dx, dy, key in DIRECTIONS:
        nx = px + dx
        ny = py + dy
        if not in_bounds(nx, ny):
            continue
        if (nx, ny) in blocked_cells or (nx, ny) in danger or has_monster(obs, nx, ny):
            continue
        if not is_passable(obs, nx, ny, avoid_hazards=True):
            continue
        if is_active_hazardous(obs, nx, ny):
            continue
        if dx and dy:
            if (nx, py) in blocked_cells or (px, ny) in blocked_cells:
                continue
            if not is_passable(obs, nx, py, avoid_hazards=True):
                continue
            if not is_passable(obs, px, ny, avoid_hazards=True):
                continue

        safe_neighbors = 0
        for ndx, ndy, _nkey in DIRECTIONS:
            sx = nx + ndx
            sy = ny + ndy
            if (
                in_bounds(sx, sy)
                and is_passable(obs, sx, sy, avoid_hazards=True)
                and not is_active_hazardous(obs, sx, sy)
            ):
                safe_neighbors += 1
        monster_dist = 0
        if monsters:
            monster_dist = min((nx - mx) * (nx - mx) + (ny - my) * (ny - my) for mx, my in monsters)
        score = safe_neighbors * 100 + monster_dist
        candidate = (score, key)
        if best is None or candidate > best:
            best = candidate
    return None if best is None else best[1]


def choose_equipment_action(obs: ObservationDict, state: PolicyState) -> str | None:
    strength = int(obs["blstats"][2])
    candidates: list[tuple[int, int, int]] = []
    for slot in inventory_slots(obs, WEAPON_CATEGORY | ARMOR_CATEGORY):
        flags = int(obs["inventory_flags"][slot])
        if flags & ITEM_EQUIPPED:
            continue
        letter = item_letter(obs, slot)
        if not letter:
            continue
        key = item_key(obs, slot)
        if key in state.tried_items:
            continue
        name = inventory_name(obs, slot)
        required = strength_requirement(name)
        if required is not None and required > strength:
            continue
        category = int(obs["inventory_category"][slot])
        kind = int(obs["inventory_kind"][slot])
        if kind == UNKNOWN_SHORT:
            kind = 0
        category_bonus = 100 if category & ARMOR_CATEGORY else 0
        candidates.append((category_bonus + kind, slot, ord(letter)))

    if not candidates:
        return None
    _score, slot, _letter_ord = max(candidates)
    letter = item_letter(obs, slot)
    state.tried_items.add(item_key(obs, slot))
    state.clear_path()
    state.action_queue.extend([letter, "y", "\x1b"])
    return "e"


def choose_panic_item_action(
    obs: ObservationDict,
    state: PolicyState,
    *,
    include_scrolls: bool,
    include_zaps: bool,
) -> str | None:
    categories = [CHARM_CATEGORY, POTION_CATEGORY]
    if include_zaps and visible_monsters(obs):
        categories.append(STAFF_CATEGORY | WAND_CATEGORY)
    if include_scrolls:
        categories.append(SCROLL_CATEGORY)
    for category in categories:
        for slot in inventory_slots(obs, category):
            letter = item_letter(obs, slot)
            if not letter:
                continue
            key = item_key(obs, slot)
            if key in state.tried_items:
                continue
            state.tried_items.add(key)
            state.clear_path()
            if category == SCROLL_CATEGORY:
                state.pending_item_prompt = "scroll"
                state.action_queue.append(letter)
            elif category & (STAFF_CATEGORY | WAND_CATEGORY):
                state.action_queue.extend([letter, "\n"])
            else:
                state.action_queue.extend([letter, "\x1b"])
            return "a"
    return None


def choose_enchant_target(obs: ObservationDict) -> str | None:
    best: tuple[int, str] | None = None
    for slot in inventory_slots(
        obs,
        WEAPON_CATEGORY
        | ARMOR_CATEGORY
        | RING_CATEGORY
        | STAFF_CATEGORY
        | WAND_CATEGORY
        | CHARM_CATEGORY,
    ):
        letter = item_letter(obs, slot)
        if not letter:
            continue
        category = int(obs["inventory_category"][slot])
        flags = int(obs["inventory_flags"][slot])
        equipped = bool(flags & ITEM_EQUIPPED)
        if category & ARMOR_CATEGORY:
            score = 1000 if equipped else 350
        elif category & WEAPON_CATEGORY:
            score = 900 if equipped else 300
        elif category & STAFF_CATEGORY:
            score = 800
        elif category & CHARM_CATEGORY:
            score = 700
        elif category & WAND_CATEGORY:
            score = 500
        else:
            score = 250
        enchant = int(obs["inventory_enchant1"][slot])
        if enchant != UNKNOWN_SHORT:
            score += max(-50, min(50, enchant))
        candidate = (score, letter)
        if best is None or candidate > best:
            best = candidate
    return None if best is None else best[1]


def choose_identify_target(obs: ObservationDict) -> str | None:
    best: tuple[int, str] | None = None
    for slot in inventory_slots(obs):
        flags = int(obs["inventory_flags"][slot])
        if not flags & ITEM_CAN_BE_IDENTIFIED:
            continue
        letter = item_letter(obs, slot)
        if not letter:
            continue
        category = int(obs["inventory_category"][slot])
        if category & STAFF_CATEGORY:
            score = 1000
        elif category & WAND_CATEGORY:
            score = 950
        elif category & SCROLL_CATEGORY:
            score = 850
        elif category & POTION_CATEGORY:
            score = 800
        elif category & RING_CATEGORY:
            score = 650
        elif category & ARMOR_CATEGORY:
            score = 550
        elif category & WEAPON_CATEGORY:
            score = 500
        else:
            score = 100
        candidate = (score, letter)
        if best is None or candidate > best:
            best = candidate
    return None if best is None else best[1]


def path_endpoint(start: tuple[int, int], path: list[str]) -> tuple[int, int]:
    x, y = start
    deltas = {key: (dx, dy) for dx, dy, key in DIRECTIONS}
    for key in path:
        dx, dy = deltas[key]
        x += dx
        y += dy
    return x, y


def update_policy_state(
    state: EpisodeState,
    before: ObservationDict,
    after: ObservationDict,
    action_key: str,
) -> None:
    policy = state.policy
    before_depth, before_turn, _before_gold, _before_score, _before_terminal = stats_from_obs(before)
    after_depth, after_turn, after_gold, after_score, after_terminal = stats_from_obs(after)
    before_pos = player_position(before)
    after_pos = player_position(after)

    state.max_depth = max(state.max_depth, after_depth)
    state.final_depth = after_depth
    state.gold = after_gold
    state.score = after_score
    state.turn = after_turn
    state.terminated = after_terminal
    state.last_message = message_text(after)
    if after_depth == before_depth and after_turn == before_turn and after_pos == before_pos:
        state.no_change_steps += 1
        policy.no_progress_count += 1
    else:
        state.no_change_steps = 0
        policy.no_progress_count = 0
        policy.pending_item_prompt = ""

    if after_depth != before_depth:
        policy.reset_level_state()
        return

    if action_key == "x":
        if after_turn == before_turn and after_pos == before_pos:
            policy.auto_failures += 1
        else:
            policy.auto_failures = 0
        return

    if action_key in MOVE_KEYS:
        if after_pos == before_pos:
            attempted = move_destination(before_pos, action_key)
            if attempted is not None:
                policy.blocked_cells.add(attempted)
            if policy.path_target is not None:
                policy.blocked_targets.add(policy.path_target)
            policy.clear_path()
        elif not policy.pending_path and policy.path_mode == "frontier":
            policy.auto_failures = 0
            policy.clear_path()
        return

    if action_key == ">":
        if after_depth == before_depth:
            policy.clear_path()
            policy.auto_failures = max(policy.auto_failures, 1)
        return


def move_destination(start: tuple[int, int], key: str) -> tuple[int, int] | None:
    for dx, dy, direction_key in DIRECTIONS:
        if key == direction_key:
            return start[0] + dx, start[1] + dy
    return None


def action_for_key(key: str) -> Action:
    action = KEY_ACTIONS.get(key)
    if action is not None:
        return action
    return Action.keypress(key)


def reset_batch(backend: BrogueBackend, seeds: list[int]) -> list[ObservationDict]:
    seed_array = np.array(seeds, dtype=np.uint64)
    return [result.observation for result in backend.reset_many(seed=seed_array)]


def reset_one(backend: BrogueBackend, env_id: int, seed: int) -> ObservationDict:
    result = backend.reset(env_id, seed=np.array([seed], dtype=np.uint64))
    return result.observation


def save_trace(trace_dir: Path, episode: EpisodeState) -> None:
    trace_dir.mkdir(parents=True, exist_ok=True)
    path = trace_dir / f"seed-{episode.seed}-depth-{episode.max_depth}.json"
    payload = {
        "seed": episode.seed,
        "max_depth": episode.max_depth,
        "final_depth": episode.final_depth,
        "turn": episode.turn,
        "steps": episode.steps,
        "score": episode.score,
        "gold": episode.gold,
        "terminated": episode.terminated,
        "actions": episode.policy.trace,
        "last_message": episode.last_message,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_search(args: argparse.Namespace) -> list[EpisodeState]:
    total = args.episodes
    parallel_envs = max(1, min(args.parallel_envs, total))
    next_seed = args.start_seed
    next_dummy_seed = 2**63
    launched = 0
    completed: list[EpisodeState] = []

    with BrogueBackend(parallel_envs) as backend:
        active: list[EpisodeState | None] = []
        initial_seeds: list[int] = []
        for _ in range(parallel_envs):
            initial_seeds.append(next_seed)
            active.append(EpisodeState(seed=next_seed))
            next_seed += 1
            launched += 1

        observations: list[ObservationDict | None] = reset_batch(backend, initial_seeds)
        last_progress_completed = 0
        for env_id, obs in enumerate(observations):
            assert obs is not None
            seed = int(obs["program_state"][PROGRAM_SEED_INDEX])
            if active[env_id] is not None:
                active[env_id].seed = seed

        while len(completed) < total:
            actions: list[Action] = []
            action_keys: list[str] = []
            before_observations: list[ObservationDict | None] = list(observations)

            for env_id, episode in enumerate(active):
                obs = observations[env_id]
                if episode is None or obs is None:
                    actions.append(action_for_key("z"))
                    action_keys.append("z")
                    continue
                key = choose_action(
                    obs,
                    episode.policy,
                    auto_retries=args.auto_retries,
                    rest_hp_frac=args.rest_hp_frac,
                    rest_min_depth=args.rest_min_depth,
                    equip_items=args.equip_items,
                    panic_items=args.panic_items,
                    panic_min_depth=args.panic_min_depth,
                    panic_scrolls=args.panic_scrolls,
                    panic_scrolls_min_depth=args.panic_scrolls_min_depth,
                    panic_zaps=args.panic_zaps,
                    panic_zaps_min_depth=args.panic_zaps_min_depth,
                    panic_near_radius=args.panic_near_radius,
                    escape_active_hazards=args.escape_active_hazards,
                    avoid_monsters_min_depth=args.avoid_monsters_min_depth,
                    adjacent_attack_min_hp_frac=args.adjacent_attack_min_hp_frac,
                    secret_search_max=args.secret_search_max,
                    stuck_dive_min_depth=args.stuck_dive_min_depth,
                    stuck_dive_min_hp_frac=args.stuck_dive_min_hp_frac,
                    native_stairs=args.native_stairs,
                )
                episode.policy.last_action = key
                episode.policy.trace.append(key)
                actions.append(action_for_key(key))
                action_keys.append(key)

            results: list[BackendStep] = backend.step_many(actions)
            for env_id, result in enumerate(results):
                episode = active[env_id]
                before = before_observations[env_id]
                if episode is None or before is None:
                    observations[env_id] = result.observation
                    if result.terminated:
                        observations[env_id] = reset_one(backend, env_id, next_dummy_seed)
                        next_dummy_seed += 1
                    continue

                episode.steps += 1
                observations[env_id] = result.observation
                update_policy_state(episode, before, result.observation, action_keys[env_id])

                reached_target = episode.max_depth >= args.target_depth
                timed_out = episode.steps >= args.max_steps
                no_change_stalled = episode.no_change_steps >= args.max_no_change_steps
                terminal = bool(result.terminated) or episode.terminated
                if terminal or timed_out or no_change_stalled or reached_target:
                    episode.done = True
                    completed.append(episode)
                    if args.trace_dir is not None and (reached_target or args.verbose):
                        save_trace(args.trace_dir, episode)
                    if args.verbose or reached_target:
                        print_episode("done", episode)
                    if args.stop_on_success and reached_target:
                        return completed

                    if launched < total:
                        seed = next_seed
                        next_seed += 1
                        launched += 1
                        active[env_id] = EpisodeState(seed=seed)
                        observations[env_id] = reset_one(backend, env_id, seed)
                        actual_seed = int(observations[env_id]["program_state"][PROGRAM_SEED_INDEX])
                        active[env_id].seed = actual_seed
                    else:
                        active[env_id] = None
                        observations[env_id] = reset_one(backend, env_id, next_dummy_seed)
                        next_dummy_seed += 1

            if (
                args.progress_interval > 0
                and len(completed) - last_progress_completed >= args.progress_interval
            ):
                last_progress_completed = len(completed)
                print_progress(completed, launched, total)

            if all(episode is None for episode in active):
                break

    return completed


def print_episode(prefix: str, episode: EpisodeState) -> None:
    clean_message = " | ".join(
        line.strip() for line in episode.last_message.splitlines() if line.strip()
    )
    print(
        f"{prefix} seed={episode.seed} max_depth={episode.max_depth} "
        f"final_depth={episode.final_depth} score={episode.score} gold={episode.gold} "
        f"turn={episode.turn} steps={episode.steps} terminated={episode.terminated} "
        f"last={clean_message[:180]}",
        flush=True,
    )


def print_summary(episodes: list[EpisodeState], elapsed: float, target_depth: int) -> None:
    if not episodes:
        print("No episodes completed.")
        return
    hist = Counter(episode.max_depth for episode in episodes)
    best = max(episodes, key=lambda episode: (episode.max_depth, episode.score, -episode.steps))
    successes = [episode for episode in episodes if episode.max_depth >= target_depth]
    print_episode("best", best)
    print(f"episodes={len(episodes)} elapsed_sec={elapsed:.2f}")
    print(f"depth_hist={dict(sorted(hist.items()))}")
    print(
        f"target_depth={target_depth} successes={len(successes)} "
        f"success_rate={len(successes) / len(episodes):.4f}"
    )
    if successes:
        print("success_seeds=" + ",".join(str(episode.seed) for episode in successes[:20]))


def print_progress(episodes: list[EpisodeState], launched: int, total: int) -> None:
    if not episodes:
        return
    best = max(episode.max_depth for episode in episodes)
    hist = Counter(episode.max_depth for episode in episodes)
    print(
        f"progress completed={len(episodes)}/{total} launched={launched} "
        f"best_depth={best} depth_hist={dict(sorted(hist.items()))}",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    start = time.perf_counter()
    episodes = run_search(args)
    elapsed = time.perf_counter() - start
    print_summary(episodes, elapsed, args.target_depth)


if __name__ == "__main__":
    main()
