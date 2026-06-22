# Brogue Puffer/NLE Performance Plan

Goal: make Brogue comparable to the Puffer NetHack/NLE environment in the
benchmark lane Puffer actually uses: `StaticVec` plus `profile envspeed`, not a
standalone Python loop.

The target comparison is:

```sh
cd broguegym-dev
bash build.sh nethack --profile
NETHACKDIR=$PWD/vendor/nle/src/build/dat \
  ./profile envspeed --total-agents 64 --buffers 1 --threads 1 --horizon 256
```

The Brogue endpoint should be the same command shape:

```sh
cd broguegym-dev
bash build.sh brogue --profile
BROGUE_DATA_DIR=$PWD/../BrogueCE/bin \
BROGUE_LIB=$PWD/../BrogueCE/bin/libbruhogue_brogue.so \
  ./profile envspeed --total-agents 64 --buffers 1 --threads 1 --horizon 256
```

Report both aggregate SPS and `SPS / num_threads`. A Brogue implementation is
"up to snuff" when it is within 2x of NetHack on the same machine in the
Puffer `envspeed` lane, with compact policy observations enabled.

For CPU simulation speed, use Puffer's `EVAL_ENV_STEP` timing as the primary
number. Total wall SPS can be GPU/copy limited for fast envs like NLE and can
hide the true per-core simulation gap.

## Current Implementation Status

Implemented:

- `ocean/brogue` Puffer env, `config/brogue.ini`, and `build.sh brogue`
  support.
- full Brogue Puffer observation: `155,032` bytes, including full terminal
  state, colors, semantic map tensors, inventory, messages, blstats, and
  program state.
- compact Brogue Puffer observation: `2407` bytes, available only as an
  explicit `build.sh brogue --compact-obs` ablation.
- private shared-library isolation, so `total_agents = 64` works despite
  Brogue's global singleton game state.
- same-thread `ucontext` bridge coroutine replacing the old pthread handoff.
- reward/log fields for score, depth, valid moves, illegal actions, new tiles,
  returns, and resets.
- Puffer `profile envspeed` phase output for `gpu/copy` and `env_step`.
- optional `BROGUE_PROFILE=1` wrapper/bridge counters.
- full bridge observation API (`brh_reset`, `brh_step`) used by the Puffer
  default Brogue env.
- optional map-only compact observations: `79x29` chars plus `int32` blstats
  and program state.
- sidebar refresh is skipped in compact/no-observation bridge mode.
- bridge/server fast-input path skips cursor pathing, snap maps, button state,
  and mouse-target prep for direct keyboard actions.
- compact map observations are now backed by a dirty-cell semantic cache:
  Brogue updates the compact map cache from `refreshDungeonCell()` in compact
  mode, and observation fill is a steady-state memcpy plus stat packing.
- bridge profiling now splits `updateVision`, `updateLighting`, and
  `updateEnvironment` into sub-zones.
- compact mode skips terminal display-buffer refresh work in the Puffer path.
- lossy compact simulation shortcuts were tested for benchmarking, but are now
  gated behind `BROGUE_COMPACT_SIMULATION_SHORTCUTS=1` and are not the default
  playable environment.
- explicit `profile envspeed` action modes:
  `--action-mode zero|fixed|random`, `--action-id`, and `--action-limit`.

Latest same-harness zero-action result at `total_agents=64, buffers=1,
threads=1, horizon=256`:

- Brogue wall SPS with full observation: `~7.0k`
- Brogue CPU env-step SPS/core from `EVAL_ENV_STEP`: `~7.6k`
- NLE CPU simulation SPS/core from `EVAL_ENV_STEP`: `~847k`
- zero-action CPU/env-step gap: `~111x`

Additional Brogue traces:

- deterministic random action ids `[0,8)` with full observation and faithful
  simulation: `~2.5k`
  env-step SPS/core
- deterministic random action ids `[0,8)` with compact observation and faithful
  simulation: `~6.0k` env-step SPS/core
- deterministic random action ids `[0,8)` with
  `BROGUE_COMPACT_SIMULATION_SHORTCUTS=1`: `~42.6k` env-step SPS/core
- NLE deterministic random action ids `[0,8)`: `~68.6k` env-step SPS/core
- deterministic movement-random gap with full observation and faithful
  simulation: `~27.1x`
- deterministic movement-random gap with compact observation and faithful
  simulation: `~11.4x`
- deterministic movement-random gap with lossy shortcuts: `~1.6x`
- unrestricted random over all 60 Brogue actions can enter prompt/menu paths
  and did not complete promptly; keep it out of headline benchmark numbers
  until prompt auto-dismiss/classification is implemented.

Latest Brogue profile after restoring full observations as default, with
compact rows retained as ablations:

- zero-action full-observation wrapper `c_step`: `~132 us/step`
- fixed search wrapper `c_step`: `~190 us/step`
- bounded random movement wrapper `c_step` with full observation and faithful
  simulation: `~396 us/step`
- bounded random movement wrapper `c_step` with compact observation and faithful
  simulation:
  `~167 us/step`
- full bridge observation fill and pack are back on the default Puffer path.
- compact bridge observation fill remains sub-microsecond in the compact
  ablation.
- terminal `commitDraws`: reduced from hundreds of us/step in rest and
  movement traces to about `0-2 us/step` in server mode
- `refresh_sidebar`: reduced from `~70-100 us/step` to about `0.0-0.1 us/step`
  in the compact benchmark path
- `input_loop_prep`: reduced from `~30-50 us/step` to about `0.0 us/step`
  in the compact benchmark path

Observation conclusion:

- the full bridge observation is about `93x` NLE's current `1659` byte
  observation, and it is now the default again.
- the compact Brogue observation is only `1.45x` NLE's observation, but it is an
  explicit ablation, not the playable default.
- current Brogue speed is limited by both faithful Brogue turn simulation and
  full observation export/copy. The shortcut benchmark reaches the target by
  skipping monster AI scheduling, but that path is opt-in only and should not be
  used as the playable environment.

Current hot zones:

- Zero-action: bridge resume, compact observation fill, and keystroke dispatch
  dominate; the zero-action gap is much smaller than before.
- Fixed search: many pause callbacks remain, but terminal flush cost is gone.
- Bounded movement-random: with faithful full-observation turn scheduling,
  `playerTurnEnded` still dominates. The remaining work is to replace the
  shortcut result with exact optimizations such as active-monster scheduling and
  incremental scent/vision updates, while also reducing full-observation copy
  overhead without removing fields.

## Concrete Next Implementation Plan

Priority order from the current profile:

1. Keep full observation as the default playable Puffer environment.
2. Keep compact map-only observation only as an explicit ablation for measuring
   observation payload cost.
3. Keep faithful simulation as the default. Compact fast turns are an explicit
   diagnostic shortcut only. To meet the benchmark target for the real
   environment, replace the shortcut with exact optimizations such as an
   active-monster scheduler and incremental scent updates.
4. Replace private shared-library isolation with a true per-env game context
   after the scalar hot path is no longer dominated by easy UI work. This is
   required for NLE-like memory footprint and reset architecture, but it is not
   the next single-core SPS bottleneck.

Near-term acceptance gates:

- full observation runs through `profile envspeed`. Done.
- compact map-derived observation runs through `profile envspeed` as an
  explicit ablation. Done.
- bounded movement-random reaches at least half of NLE's same-harness
  env-step SPS/core without changing game semantics or removing observation
  fields. Not done; current full-observation faithful result is `~2.5k` versus
  NLE `~68.6k`.
- every benchmark row is reported from the same Puffer `profile envspeed`
  command shape as NLE.

## Current Facts

- Brogue Puffer full-observation movement steps are around `2.5k` env-step
  SPS/core with faithful simulation. Compact-observation faithful movement is
  around `6.0k` and is now an ablation only. The old `~40k` result requires
  opt-in lossy compact simulation shortcuts.
- Brogue full observation is 155,032 bytes and is now intentionally back on the
  Puffer default path.
- NetHack Puffer chars-only observation is 1,659 bytes. In the same
  `profile envspeed` harness, its CPU env-step bucket reaches hundreds of
  thousands of simulation steps/s/core.
- Current Brogue still has singleton game state inside one loaded library
  image. Puffer multi-env currently works by loading one private shared-library
  image per env. This is a benchmark-enabling shim, not the final architecture.
- Puffer's authoritative env benchmark is `tests/profile_kernels.cu`:
  `create_static_vec`, `create_static_threads`, `static_vec_reset`, repeated
  `static_vec_omp_step`, and CUDA event timing.

## Phase 1: Add A Puffer Brogue Baseline

Purpose: get Brogue measured inside Puffer, even if only `total_agents = 1`.
This gives a direct Puffer overhead baseline and prevents future work from
optimizing against the wrong harness.

Add:

- `broguegym-dev/ocean/brogue/brogue.h`
- `broguegym-dev/ocean/brogue/binding.c`
- `broguegym-dev/ocean/brogue/brogue.c` or standalone debug harness
- `broguegym-dev/config/brogue.ini`
- `build.sh` support for `ENV=brogue`

Initial `Brogue` env shape:

```c
typedef struct Log {
    float perf;
    float score;
    float depth;
    float episode_return;
    float episode_length;
    float invalid_actions;
    float n;
} Log;

typedef struct Brogue {
    Log log;
    unsigned char *observations;
    float *actions;
    float *rewards;
    float *terminals;
    int num_agents;
    unsigned int rng;

    brh_env *env;
    brh_observation *full_obs;
    long key_storage[1];
    unsigned char control_storage[1];
    unsigned char shift_storage[1];

    int episode_length;
    float episode_return;
} Brogue;
```

`binding.c` should mirror NetHack:

```c
#include "brogue.h"
#define OBS_SIZE BROGUE_OBS_SIZE
#define NUM_ATNS 1
#define ACT_SIZES {BROGUE_NUM_ACTIONS}
#define OBS_TENSOR_T ByteTensor
#define Env Brogue
#include "vecenv.h"
```

Acceptance:

- `bash build.sh brogue --cpu` builds a CPU `_C` module.
- `bash build.sh brogue --profile` builds `./profile` on a CUDA host.
- `./profile envspeed --total-agents 1 --buffers 1 --threads 1 --horizon 256`
  runs and reports throughput.
- historical note: `total_agents = 2` failed before library-copy isolation.
  Current Brogue should run `total_agents = 64` in the Puffer harness.

## Phase 2: Match NetHack's Benchmark Surface

Purpose: make the Brogue benchmark apples-to-apples with NLE.

Keep the full `brh_observation` as Puffer's default policy observation. Compact
profiles are useful only as ablations to quantify observation payload cost.

Start with:

```c
#define BROGUE_ROWS 34
#define BROGUE_COLS 100
#define BROGUE_SCREEN_CELLS (BROGUE_ROWS * BROGUE_COLS)
#define BROGUE_BLSTATS_SIZE 16
#define BROGUE_PROGRAM_STATE_SIZE 8
#define BROGUE_OBS_SIZE \
    (BROGUE_SCREEN_CELLS + BROGUE_BLSTATS_SIZE * 4 + BROGUE_PROGRAM_STATE_SIZE * 4)
```

Optional compact packing rules:

- `screen chars`: one byte per displayed cell, lossy ASCII/small-id is fine for
  the first policy benchmark.
- `blstats`: packed `int32`.
- `program_state`: packed `int32`.
- no RGB, no full semantic map, no inventory strings, no floor item metadata,
  no monster metadata in the compact ablation.

Keep compact observation available behind an explicit build flag such as
`build.sh brogue --compact-obs`.

Acceptance:

- default `profile envspeed` prints `obs_size = 155032`.
- a compact profile can still be built for performance ablations.
- Direct C benchmark keeps separate rows for `none`, `compact`, and `full`.

## Phase 3: Instrument The Hot Path In Puffer

Purpose: stop guessing about where Brogue spends time after it is in Puffer.

Add optional counters around the Brogue `c_step` path:

- action decode
- bridge step call
- game coroutine/thread resume cost
- compact observation pack
- reward/log update
- reset

Expose these in `my_log` or a Brogue-specific profile dump. Keep the existing
direct C bridge benchmark, but treat `profile envspeed` as the primary number.

Acceptance:

- `./profile envspeed` can be paired with a Brogue profile summary.
- `SPS/core` and per-phase timing are captured for at least:
  - `wait/rest`
  - `search`
  - `explore`
  - random action trace
  - reset-heavy trace

## Phase 4: Remove The Singleton Barrier

Purpose: Puffer performance requires many envs in one process. The current
Brogue bridge has both bridge-control singleton state and Brogue game singleton
state.

Do this in two steps.

First, move bridge-control singleton state into `brh_env`:

- mutex/condvar/thread
- pending event
- waiting/action flags
- close/exited flags
- last observation
- step status
- capture-observation flag
- end score/win flag

Use a thread-local active bridge pointer for platform callbacks:

```c
static _Thread_local brh_env *currentBridgeEnv;
```

Second, choose the game-state isolation route:

- Short-term route: library-copy isolation. Each Puffer env copies/dlopens a
  unique `libbruhogue_brogue.so`, resolves `brh_env_*`, and creates one scalar
  env inside that library image. This proves vector benchmark plumbing without
  migrating all Brogue globals.
- Long-term route: NLE-style context migration. Move mutable Brogue globals
  into a `BrogueCtx` owned by `brh_env`, and route accesses through an active
  context pointer. This is the real end state for memory and scaling.

Acceptance:

- Native C test creates two live `brh_env` instances, resets both, steps both,
  and closes both in one process.
- Puffer `total_agents = 2, buffers = 1, threads = 1` runs.
- Then `total_agents = 64, buffers = 1, threads = 1` runs.

## Phase 5: Replace Pthread Handoff

Purpose: the no-observation benchmark shows most legal-step cost remains after
observation export is removed. The background pthread plus condition-variable
handoff is still unnecessary overhead, but benchmarking shows it is not the
dominant legal-step cost.

Replace the bridge thread wait/resume path with a same-thread stackful
coroutine, like NLE's direct step/yield structure:

- `brh_env_reset` initializes the game coroutine.
- Brogue's `nextKeyOrMouseEvent` callback yields to the caller when an action is
  needed.
- `brh_env_step` stores the action and resumes the coroutine until the next
  input boundary or terminal state.
- No pthread, no condvar, no cross-thread callback state on the hot path.

Use Puffer's vendored `deboost.context` or the same context backend NLE uses in
`vendor/nle/src/third_party/deboost.context`.

Implemented state:

- The bridge now uses a same-thread `ucontext` coroutine behind the same public
  `brh_*` ABI.
- Direct scalar valid-action speed improved only modestly; invalid/no-op paths
  improved more.
- Puffer `profile envspeed` at `total_agents = 64, threads = 1` improved from
  about `4092` SPS/core to about `4641` SPS/core.

Acceptance:

- `brh_env_step_no_observation` improves from the current 1.5k-2.8k SPS/core
  range without regressing correctness.
- Reset remains correct and deterministic.
- Puffer `profile envspeed --threads 1` shows the same improvement.

Follow-up:

- The remaining gap is not the scheduler alone. The next optimization target is
  the legal-action turn loop and Brogue's global-state layout.

## Phase 6: Reward, Termination, And Logs

Purpose: make the Puffer env trainable, not just fast.

Implement reward shaping in the Brogue binding:

- score/gold delta
- new depth bonus
- exploration/new-tile bonus if cheap under compact profile
- invalid action penalty
- terminal win/loss score

Implement logs:

- `perf`
- `score`
- `depth`
- `episode_return`
- `episode_length`
- `invalid_actions`
- `new_tiles` if tracked
- `n`

Acceptance:

- Puffer logs match NetHack-style log aggregation.
- `profile envspeed` does not regress meaningfully from logging.
- Short CPU rollout produces nonzero `n` after terminations.

## Phase 7: Benchmark Matrix And Gates

Run this matrix for NetHack and Brogue from the same Puffer checkout:

```sh
for env in nethack brogue; do
  bash build.sh $env --profile
  for threads in 1 2 4 8 16; do
    ./profile envspeed \
      --total-agents 64 \
      --buffers 1 \
      --threads $threads \
      --horizon 256
  done
done
```

Also run a larger occupancy point:

```sh
./profile envspeed --total-agents 1024 --buffers 2 --threads 16 --horizon 256
```

Report:

- aggregate SPS
- SPS/core
- rollout ms
- obs size
- reset rate
- terminations per second
- per-phase Brogue profile counters

Performance gates:

- Completed baseline: correct `total_agents = 64` Puffer run.
- Completed full default observation restore: Puffer `obs_size = 155032`.
- Completed compact obs ablation: Puffer `obs_size = 3516`.
- Completed coroutine: pthread handoff removed, but only reaches `~4.8k`
  env-step SPS/core.
- Completed compact bridge observation: full bridge observation fill removed
  from the Puffer default path, zero-action env-step throughput is now
  `~6.3k` SPS/core.
- Completed map-only compact observation and sidebar elision: Puffer
  `obs_size = 2407`; zero-action env-step throughput is now `~13.2k`
  SPS/core.
- Completed bridge fast input: input-loop prep is near zero in bridge mode;
  zero-action env-step throughput is now `~62.7k` SPS/core.
- Completed dirty-cell compact map cache: compact observation fill is
  sub-microsecond in steady state; zero-action env-step throughput is now
  `~87.8k` SPS/core and bounded random movement is now `~3.81k` SPS/core.
- Completed compact-mode profile split and observation/display cleanup. Bounded
  random movement with compact observation and faithful simulation is about
  `~6.0k` env-step SPS/core; with full observation restored as default it is
  about `~2.5k` env-step SPS/core.
- Next gate, turn-loop optimization: named Brogue-internal timers explain
  `api.step`, then bounded random movement moves materially beyond the current
  `~2.5k` env-step SPS/core without changing game semantics or removing fields.
- Diagnostic only: `BROGUE_COMPACT_SIMULATION_SHORTCUTS=1` can reach the
  `~40k` env-step SPS/core range by changing game semantics.
- Final target: Brogue is within 2x of NLE's `EVAL_ENV_STEP` SPS/core in the
  same `profile envspeed` matrix with faithful simulation.

## Work Order

Completed:

1. Add Puffer `ocean/brogue` skeleton and `config/brogue.ini`.
2. Add compact observation packer and action table.
3. Build and run `profile envspeed`.
4. Prototype library-copy isolation to get `total_agents = 64` benchmark data.
5. Replace pthread handoff with same-thread coroutine.
6. Add reward/logging.
7. Add same-harness Puffer phase timing and Brogue-specific profile counters.
8. Add compact bridge observation capture and wire Puffer Brogue to use it.
9. Add explicit envspeed action modes and bounded random action support.
10. Add dirty-cell semantic compact map caching so compact observation fill no
    longer rescans all map cells or depends on the terminal display buffer.
11. Add expanded turn-loop profile zones and compact-mode lighting filtering
    for non-observed glowing terrain/display-detail work.
12. Move lossy compact simulation shortcuts behind
    `BROGUE_COMPACT_SIMULATION_SHORTCUTS=1`; faithful simulation is the default.

Next concrete work:

1. Optimize true time-advancing movement from the current `BROGUE_PROFILE=1`
   timers. Random `[0,8)` now reaches `~2.5k` env-step SPS/core with full
   observation and is
   dominated by `playerTurnEnded`, `updateVision`, and `updateEnvironment`;
   the largest named sub-zones are now `env_gas`, `lighting_tile_glow`,
   `env_promotions`, and `lighting_miner`.
   Acceptance: bounded random movement moves materially beyond `2.5k`
   env-step SPS/core with all fields retained.
2. Classify the zero-action path. If repeated `k` is mostly wall-bump/no-turn,
   add a cheap prompt/no-advance path comparable to NLE's auto-dismiss/no-advance
   handling. Acceptance: zero-action `profile envspeed` improves materially
   without changing time-advancing legal moves.
3. Add a reduced "policy-random" trace or recorded policy trace that avoids
   full prompt/menu action space while still advancing real gameplay.
   Acceptance: the trace completes promptly and is stable enough for regression
   benchmarking.
4. Continue true time-advancing movement optimization. Likely
   targets are display refresh work that is unnecessary for compact obs,
   repeated map/lighting recomputation, full-observation packing/copy, and
   exact active-set approaches for monster/environment updates.
   Acceptance: fixed legal movement/search/rest traces improve, not just the
   zero-action benchmark.
5. Replace private shared-library isolation with an NLE-style `BrogueCtx`.
   Move mutable Brogue globals and bridge-control state behind a per-env context
   and route callbacks through a thread-local active context. Acceptance:
   multiple `brh_env` instances live in one library image; Puffer no longer
   copies/dlopens one library per env; memory use and multi-thread scaling
   improve.
6. Run the full NetHack-vs-Brogue Puffer benchmark matrix and report both wall
   SPS and `EVAL_ENV_STEP` SPS/core. The final gate remains within 2x of NLE's
   CPU simulation SPS/core in the same harness with faithful simulation.
