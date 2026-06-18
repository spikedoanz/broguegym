# Scalar BrogueEnv Contract

This branch introduces a scalar C environment contract as the bridge target for
Python and for a future Puffer Ocean binding. It is deliberately not a vector
manager.

The first real performance step is not the Puffer binding, and it is not
multi-env correctness yet. A thin Puffer Ocean binding is mostly plumbing, and
multi-env correctness only matters once the scalar env is fast enough to be worth
batching. The next implementation milestone is getting `N=1` scalar throughput
to an acceptable level.

## C Surface

`BrogueCE/src/platform/bridge-env.h` exposes:

- `brh_env_create(const brh_env_buffers *buffers)`
- `brh_env_reset(brh_env *env, uint64_t seed)`
- `brh_env_step(brh_env *env, long key, int control, int shift)`
- `brh_env_step_no_observation(brh_env *env, long key, int control, int shift)`
- `brh_env_step_from_buffers(brh_env *env)`
- `brh_env_num_agents(const brh_env *env)`
- `brh_env_close(brh_env *env)`

`brh_env_buffers` is caller-owned:

- `observations`: one `brh_observation` slot
- `actions`: one keycode slot, used by `brh_env_step_from_buffers`
- `controls`: one boolean byte slot
- `shifts`: one boolean byte slot
- `rewards`: one float slot
- `terminals`: one float slot

The current implementation hard-rejects a second live `brh_env` in the same
process. That guard should stay until Brogue mutable globals are moved behind a
per-env state handle or another isolation mechanism exists.

## Puffer Shape

A future Puffer binding should be thin:

```c
typedef struct Brogue {
    unsigned char *observations;
    float *actions;
    float *rewards;
    float *terminals;
    int num_agents;
    unsigned int rng;
    brh_env *env;
    brh_observation *obs_storage;
    long key_storage[1];
    uint8_t control_storage[1];
    uint8_t shift_storage[1];
} Brogue;
```

`c_reset(Brogue*)` should call `brh_env_reset`. `c_step(Brogue*)` should decode
Puffer's action into key/control/shift and call `brh_env_step`. Puffer should own
batching, OpenMP scheduling, and contiguous rollout buffers.

This binding should not be treated as the source of truth for batching. It should
be a thin adapter over a scalar `brh_env` after scalar throughput is acceptable
and `brh_env` can safely coexist with other `brh_env` instances in the same
process.

## Remaining Blocker

The current scalar C path is only around 2k valid gameplay steps/sec for one
instance. That is not good enough to justify spending the next work block on
Puffer or `N > 1` scheduling.

The immediate blocker is scalar step cost:

- Brogue game work
- pthread/condition-variable bridge handoff
- full `brh_observation` fill
- semantic map/item/monster/inventory observation fill
- full observation export/copy
- Python-side conversion for the ctypes backend

Until `N=1` is acceptable, a C vector manager or Puffer Ocean binding would only
batch a slow scalar kernel.

## Current N=1 Throughput

Last measured: 2026-06-17 22:40 EDT.

Command:

```sh
uv run python scripts/bench_bridge_c.py --profile standard --trace-repeats 3 --seed-count 1
```

Commit flags:

```text
broguegym=feature/scalar-brogue-env@7e10b2e[dirty:1T/1U] BrogueCE=feature/scalar-brogue-env@2c294f1[dirty:0T/1U] broguegym-dev=4.0@e90b58ed[clean]
```

Dirty state at measurement time was expected: the parent saw the BrogueCE
submodule's untracked generated dylib plus the local `NOTES` file; BrogueCE saw
only `bin/libbruhogue_brogue.dylib`.

Observation size: 155,032 bytes. Direct C scalar ABI, no Python step loop.

| Case | Median us/step | Median steps/s | Note |
| --- | ---: | ---: | --- |
| copy-only | 2.35 | 426,158 | memcpy floor for the full observation size |
| invalid-key | 89.76 | 11,140 | invalid input loop with full observation export |
| invalid-no-obs | 60.43 | 16,548 | invalid input loop without full observation export |
| rest | 510.86 | 1,957 | valid Brogue step with full observation export |
| rest-no-obs | 494.07 | 2,024 | valid Brogue step without full observation export |
| search | 515.17 | 1,941 | valid Brogue step with full observation export |
| search-no-obs | 488.34 | 2,048 | valid Brogue step without full observation export |
| explore | 514.62 | 1,943 | valid Brogue step with full observation export |
| explore-no-obs | 451.23 | 2,216 | valid Brogue step without full observation export |
| fast-explore | 422.56 | 2,367 | valid Brogue step with full observation export |
| fastx-no-obs | 422.74 | 2,366 | valid Brogue step without full observation export |
| reset | 19,459.18 | 51 | reset cost |

The no-observation ABI is useful, but it did not reveal a huge hidden memcpy or
semantic-fill tax on ordinary valid actions. `rest` improved from 1,957 to 2,024
steps/s, `search` from 1,941 to 2,048 steps/s, and `explore` from 1,943 to 2,216
steps/s. Most of the `N=1` cost is still in Brogue's step/event/render path and
the pthread handoff, not in Python or the final 155 KiB observation copy.

## Puffer NetHack Scalar Reference

Last measured: 2026-06-17 EDT.

This is a reference point for the question "what does a fast scalar roguelike C
env look like?" It is not a vectorized `StaticVec` measurement. The binary is
Puffer's standalone `ocean/nethack/nethack.c` harness with one `Nethack env` in
one process and thread-count environment variables pinned to one.

The local macOS run required a throwaway modified-NLE checkout at commit
`188b523` plus local build-system workarounds for Linux linker flags and
Darwin/glibc allocator differences. Those source compatibility changes were not
saved to this branch; use Linux for authoritative Puffer/NLE comparisons.

Command shape:

```sh
/usr/bin/time -l env \
  OMP_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 \
  VECLIB_MAXIMUM_THREADS=1 \
  MKL_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 \
  NETHACKDIR=$PWD/vendor/nle/src/build/dat \
  ./nethack bench 1000000 random
```

| Env | Policy | Steps | Steps/s | Max RSS | Peak memory footprint |
| --- | --- | ---: | ---: | ---: | ---: |
| Puffer NetHack scalar | wait | 10,000 | 82,921 | not measured | not measured |
| Puffer NetHack scalar | wait | 1,000,000 | 81,301 | 16,285,696 bytes | 9,013,120 bytes |
| Puffer NetHack scalar | random | 1,000,000 | 113,459 | 19,152,896 bytes | 11,683,776 bytes |

Against the current Brogue direct C valid-step lane, this scalar NetHack path is
roughly 34x to 58x faster depending on which Brogue action profile is used for
the denominator. That comparison should be rerun on Linux before treating the
ratio as a stable target.

## Next Milestone: N=1 Scalar Throughput

The concrete target is to make this path fast before changing batching:

```c
brh_env *env = brh_env_create(&buffers);
brh_env_reset(env, 1);
for (int i = 0; i < steps; i++) {
    brh_env_step(env, 'z', 0, 0);
}
brh_env_close(env);
```

This should have stable, reproducible timing for fixed action traces and should
support measuring smaller observation profiles than the current full
`brh_observation`.

## Concrete N=1 Next Steps

1. **Make the benchmark phase-aware**

   Extend the direct C benchmark so a scalar step can be broken into:

   - action handoff / wakeup
   - Brogue turn processing
   - screen observation fill
   - semantic map/item/monster/inventory fill
   - final observation copy or pack

   The current benchmark reports only end-to-end step cases plus copy-only. It
   does not tell us which part of the valid-action path is responsible for the
   ~500 us step time.

2. **Use the no-observation step mode as a guardrail**

   ABI 9 adds a debug/performance function that advances Brogue without filling
   a full observation. This gives a lower bound for game/input overhead:

   ```c
   int brh_env_step_no_observation(brh_env *env, long key, int control, int shift);
   ```

   Current measurements show this is still slow for valid actions, so the next
   target is the pthread bridge/game/render loop rather than Python interop.

3. **Add compact observation profiles**

   Do not make the default policy path fill the 155 KiB full observation. Add
   explicit profiles, for example:

   - `none`: step only
   - `screen`: screen chars/glyph ids only
   - `core`: screen + `blstats` + message + program state
   - `full`: current raw `brh_observation`

   The C API should make the selected profile explicit at env creation or step
   time. Puffer should start on `screen` or `core`, not `full`.

4. **Port registered/caller-owned buffer filling onto the scalar ABI**

   The scalar ABI already has caller-owned buffers, but the Python path still
   materializes a full `_CObservation` and then converts it into owned NumPy
   arrays. Reintroduce direct writes into registered field buffers or packed
   policy buffers on top of `brh_env`, so Python does not allocate/copy the full
   dict on every step when it is not needed.

5. **Skip expensive semantic fills unless requested**

   Split `bridge_fill_observation` so map semantics, inventory strings, item
   metadata, monster metadata, and color tensors are not filled for compact
   policy observations. These fields are useful for debugging and privileged
   training, but they should not be on the hot path by default.

6. **Measure Python separately from C**

   Keep three benchmark lanes:

   - direct C scalar ABI
   - Python `_InProcessBrogue` scalar backend
   - process-backed `BrogueBackend`

   Only the first lane answers whether the C bridge is fast enough. The process
   backend numbers should not be used to judge the scalar C kernel.

7. **Only then consider replacing pthread handoff**

   If compact/no-observation stepping is still too slow, the pthread bridge
   architecture is likely the bottleneck. At that point evaluate a same-thread
   coroutine/fcontext-style bridge, NLE-style, so each input wait yields directly
   to the caller instead of using a background pthread and condition variables.

## N=1 Acceptance Criteria

- Direct C benchmark reports separate timings for no-observation, screen/core,
  and full observation profiles.
- Compact policy observation is materially faster than full observation.
- Python scalar backend can use the compact/direct-buffer path without per-step
  full-dict allocation.
- We have a defensible scalar throughput number before starting `N > 1`.
- If the compact path still cannot meet the target, the benchmark identifies
  whether the remaining cost is game simulation or bridge scheduling.

## Following Milestone: Multiple Envs Per Process

Puffer's `vecenv.h` allocates an array of `Env` structs and calls `c_reset` and
`c_step` on each one, including through OpenMP. For Brogue, that requires two
independent `brh_env` handles in one process to be able to reset, step, close,
and re-reset without corrupting each other.

The concrete target is:

```c
brh_env *a = brh_env_create(&buffers_a);
brh_env *b = brh_env_create(&buffers_b);

brh_env_reset(a, 1);
brh_env_reset(b, 2);
brh_env_step(a, 'z', 0, 0);
brh_env_step(b, 'z', 0, 0);
brh_env_close(a);
brh_env_close(b);
```

That should pass under ASan/UBSan and produce stable, independent observations.
Threaded stepping can come after that; first prove sequential multi-instance
correctness.

## Concrete Next Steps

These steps should wait until the `N=1` scalar path is acceptable.

1. **Add a failing multi-env C test**

   Add a small native test or benchmark harness that tries to create two
   `brh_env` handles in one process. It should assert the current failure caused
   by the single-live-env guard. This locks the expected blocker in place before
   we start weakening the guard.

2. **Classify bridge globals**

   In `bridge-platform.c`, separate state into:

   - per-env bridge control state: mutex, condvar, thread, pending event,
     waiting/action flags, close/exited flags, last observation, step status,
     end score, end won
   - truly process-global platform state: `currentConsole`, `dataDirectory`,
     `serverMode`, `nonInteractivePlayback`, `hasGraphics`, `graphicsMode`
   - Brogue game globals reached through `Globals.c`, `GlobalsBase.c`, variant
     globals, `displayBuffer`, `rogue`, `player`, `pmap`, `tmap`, item lists,
     monster lists, message buffers, RNG state

3. **Move bridge control state into `brh_env`**

   Replace the singleton bridge fields with fields on `brh_env`, and pass the
   env pointer into the bridge thread. The bridge thread should not read or write
   `bridgeLastObservation`, `bridgePendingEvent`, `bridgeWaitingForAction`, etc.
   as file-static globals.

   This is necessary but not sufficient; it removes the bridge's own singleton
   shape without solving Brogue game globals.

4. **Make platform callbacks env-aware**

   Current Brogue platform callbacks have signatures like
   `nextKeyOrMouseEvent(rogueEvent*)`, so they do not receive `brh_env*`.
   Introduce a thread-local pointer such as:

   ```c
   static _Thread_local brh_env *currentBridgeEnv;
   ```

   Set it at the start of each bridge thread. Every bridge callback should
   resolve `brh_env *env = currentBridgeEnv` and operate on that env's bridge
   control state.

5. **Decide the Brogue game-state isolation strategy**

   There are two viable routes:

   - **Context migration:** move mutable Brogue globals into a `BrogueGameState`
     / `BrogueCtx` and route accesses through the active env. This is the clean
     Puffer/NLE-style end state, but it is invasive.
   - **Library-copy isolation:** load separate copies of the Brogue bridge shared
     library so each env gets its own `.data` globals. This is less invasive but
     more delicate to build, unload, and package.

   The current scalar ABI supports either route. Do not add OpenMP or Puffer
   multi-env support until one route proves sequential `N=2` correctness.

6. **Prototype the cheaper route first**

   For a quick proof, prototype library-copy isolation in a separate branch:

   - copy `libbruhogue_brogue.{so,dylib}` to unique temp paths
   - `dlopen` one copy per env
   - look up `brh_env_*` symbols per copy
   - create one scalar env inside each library image
   - run the `N=2` sequential test

   If this works, it gives an interim way to batch inside one host process while
   deferring the full global migration. If it is too brittle on macOS/Linux, fall
   back to context migration.

7. **Only then add the Puffer binding**

   Once `N=2` sequential multi-env correctness is proven, add:

   - `broguegym-dev/ocean/brogue/brogue.h`
   - `broguegym-dev/ocean/brogue/binding.c`
   - `broguegym-dev/config/brogue.ini`
   - `build.sh` handling for `ENV=brogue`

   Start with `total_agents = 1`, then run `total_agents = 2` in sequential CPU
   mode, then raise `num_threads` after correctness is stable.

8. **Add packed observations**

   Do not feed Puffer the full `brh_observation` as the default policy input.
   Add C packers for compact `ByteTensor` observations, initially something like:

   - screen chars or glyph ids
   - basic stats
   - terminal/program state fields needed by logging

   Keep the full raw observation available for tests/debugging, not as the first
   training surface.

9. **Add reward/log accessors**

   The bridge currently reports zero reward. Before training, add either bridge
   reward fields or Puffer-side reward shaping from `program_state`, depth, gold,
   exploration, invalid key status, and terminal outcome.

## Multi-Env Acceptance Criteria

- `brh_env_create` can create at least two live envs in one process.
- Sequential `N=2` reset/step/close passes repeatedly under sanitizers.
- Same-seed single-env behavior remains unchanged.
- Different-seed envs maintain independent observations and terminal state.
- A Puffer `brogue` binding can run `total_agents = 2`, `num_threads = 1`.
- Only after that: OpenMP stepping with `num_threads > 1`.
