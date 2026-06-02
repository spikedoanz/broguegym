# NLE-style process model

This branch replaces the earlier HTTP/screen-dump prototype with a Gymnasium-first design. New
training work should build against the API described here.

## Goals

- Make `BrogueEnv` a normal Gymnasium environment, usable by standard training loops and vector
  environment wrappers.
- Avoid HTTP, polling, terminal scraping, and long-running manager processes in the training path.
- Keep Brogue's global-state C implementation isolated by process: one rollout worker owns one game
  instance. Vectorization should use multiple OS processes, not multiple live games in one C global
  address space.
- Preserve a narrow low-level API that can later be backed by a Python extension, `ctypes`, or
  another direct dynamic-library bridge.
- Treat full-game snapshots as a core requirement rather than a debugging add-on.

## Python boundary

The Python package is split into three layers:

- `BrogueEnv`: the Gymnasium wrapper, registered as `Bruhogue-v0`. It owns Gym reset/step/render
  semantics, action-space validation, truncation, observation filtering, and user-facing snapshot
  helpers.
- `BrogueBackend`: the process-isolated Brogue boundary. It owns reset, step, snapshot, restore,
  and close.
- `BrogueSnapshot`: an opaque versioned byte payload with enough metadata to reject incompatible
  files before restore.

The default backend is the process-isolated `BrogueBackend`: `num_envs=1` is the scalar Gym
path, and larger values route commands to one Brogue worker process per environment. Each worker
owns a private in-process bridge binding, which loads Brogue's bridge shared library through
`ctypes`. If the native library has not been built, construction raises `BackendUnavailableError`
with the build command. Tests can still use small fake backends without spawning Brogue.

## C bridge target

The first Brogue-side bridge exposes a small singleton C API:

```c
int brh_reset(uint64_t seed, struct brh_observation *out);
int brh_step(long key, int control, int shift, struct brh_observation *out);
uint32_t brh_abi_version(void);
size_t brh_observation_size(void);
int brh_screen_cols(void);
int brh_screen_rows(void);
void brh_set_data_dir(const char *path);
const char *brh_last_error(void);
void brh_close(void);
```

Python validates the ABI version, observation struct size, and screen dimensions before the first
session starts. A mismatch is treated as a typed backend availability error, not as a best-effort
decode.

Snapshot support is still the next native API extension:

```c
int brh_snapshot(uint8_t **payload, size_t *payload_len, struct brh_snapshot_meta *meta);
int brh_restore(const uint8_t *payload, size_t payload_len, struct brh_observation *out);
```

Brogue CE is mostly global-state C code, so the first implementation can be a singleton inside one
Python process. Parallel rollout should come from process-level vector environments. If we later
make Brogue re-entrant, this API can grow handles without changing the Gym-facing contract.

## Observation contract

The default Gym observation space is fixed-size and NLE-like:

- `glyphs`: stable semantic IDs for rendered cells.
- `chars`: Unicode terminal codepoints for the full 100x34 Brogue screen.
- `colors_fg` and `colors_bg`: RGB terminal colors.
- `specials`: per-cell flags for bridge-specific state.
- `blstats`: a compact numeric status vector.
- `message`: the latest message as fixed-size bytes.
- `program_state`: turn/depth/seed/gold/done-style bookkeeping.
- `inventory_present`: explicit fixed-slot inventory presence mask.
- `inventory_letters`: fixed 26-slot inventory letters.
- `inventory_strs`: fixed 26x80 item-name bytes, using an NLE-style fixed-width string layout.
- `inventory_category`, `inventory_kind`, `inventory_quantity`, `inventory_flags`,
  `inventory_enchant1`, `inventory_enchant2`, and `inventory_charges`: known public inventory
  metadata.

The in-process bridge also exposes privileged map/entity tensors. They are not part of the default
policy observation because they may include hidden or full-info Brogue state useful for reward
calculation, diagnostics, and curriculum code. `BrogueEnv(include_privileged_info=True)` attaches
that full bridge view to Gym `info` while keeping the policy observation player-facing.
`BrogueEnv(observation_mode="privileged")` opts into using the full-info tensors as the policy
observation. The older `include_semantic_tensors=True` name is retained as a compatibility alias for
privileged observation mode.

- `map_layers`, `map_flags`, `map_volume`, `map_machine`, and `map_light`: Brogue-native dungeon
  tensors for the 79x29 map area.
- `map_has_item` and `map_has_monster`: explicit per-cell presence masks for sparse item and
  creature tensors.
- `map_item_*`: per-cell item category, known kind, quantity, and public flags.
- `map_monster_*`: per-cell visible monster kind, HP, and behavior state.

Gym `render("ansi")` is intentionally geometry-safe: it renders from `glyphs` through a fixed ASCII
compatibility table and applies the screen colors. The raw Unicode `chars` tensor remains available
for export inspection, but it is not the default TTY rendering contract because terminal Unicode
width and emoji presentation can corrupt grid geometry.

The first C bridge fills screen tensors from Brogue's render buffers, copies displayed messages,
exports a compact status vector, and exports inventory strings through Brogue's player-facing
`itemName` path. Numeric map, entity, and inventory tensors stay semantic rather than stringly typed.
The bridge intentionally preserves privileged Brogue map/entity state; the Gym layer owns the policy
observation filter. Unknown item kind, enchantment, and charge fields use the signed 16-bit minimum
sentinel; hidden item kinds and private item flags are not exported as known facts. Presence masks
distinguish "no entity in this slot/cell" from "the entity exists but a scalar field is unknown or
private."

## Action contract

The default discrete action table is derived from Brogue's `Rogue.h` keyboard constants and the
main `executeKeystroke` switch in `IO.c`. It includes movement, directional running, rest/search
variants, stairs, inventory/item commands, auto-explore, message/discovery/feat/help screens,
display toggles that affect observation, cursor mode, acknowledge, and escape.

The table intentionally excludes save/new/quit, screenshot, graphics mode cycling, full autopilot,
debug-only commands, and load/view-recording commands. Those are real end-player commands, but they
are poor default training actions. Coordinate inspection remains a backend query
(`Action.inspect(x, y)`) instead of a discrete key action because Brogue exposes inspection through
mouse/cursor interaction rather than one stateless keypress.

The discrete table also appends non-duplicated modal input keys for lowercase item/menu letters and
digits. For example, the inventory command opens the inventory screen, and subsequent modal key
actions can select slots such as `f`, `g`, `m`, `o`, `p`, `q`, or `v`. Existing movement and command
actions already cover their own raw key values inside modal prompts, so duplicate key events are not
added just to give them modal-specific labels.

For parity experiments that need NLE challenge-style full keyboard control, `BrogueEnv` also accepts
`actions="full"`. That profile keeps the source-derived command actions first, then appends
non-duplicated raw key events for tab, return, escape, and printable ASCII. It is opt-in because it
contains commands the default training table deliberately avoids, including save/new/quit and other
out-of-band player commands.

## Snapshot contract

`BrogueSnapshot` is intentionally opaque to Python. Python can hold it in memory, pass it back to
`restore`, or write it to disk with a small versioned wrapper. The payload itself belongs to the C
bridge.

The Brogue payload must be a complete game-state snapshot. At minimum, restore must reproduce:

- the full `rogue` global state and RNG state
- all level, terrain, item, monster, inventory, and player structures
- message/archive state and any pending prompt/input state
- display buffers needed to emit the same next observation after restore
- recording/logging counters that affect gameplay or observation output
- bridge-side metadata that affects reward, termination, or truncation handling

Snapshots written to disk should include the snapshot format version, Brogue CE build identity, and
bridge ABI version. `restore` should reject incompatible snapshots loudly instead of attempting a
best-effort load.

If direct memory serialization is too brittle, the fallback should use Brogue's native save/load
machinery plus supplemental bridge metadata. Either way, the Python API remains the same:

```python
snapshot = env.snapshot()
env.restore(snapshot)
env.save_snapshot("state.brhsnap")
env.load_snapshot("state.brhsnap")
```
