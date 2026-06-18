# Scalar BrogueEnv Contract

This branch introduces a scalar C environment contract as the bridge target for
Python and for a future Puffer Ocean binding. It is deliberately not a vector
manager.

The first real performance step is not the Puffer binding. A thin Puffer Ocean
binding is mostly plumbing, but it will only be useful at `total_agents = 1`
while Brogue has process-global mutable game state. The next implementation
milestone is making multiple Brogue environments correct in one process.

## C Surface

`BrogueCE/src/platform/bridge-env.h` exposes:

- `brh_env_create(const brh_env_buffers *buffers)`
- `brh_env_reset(brh_env *env, uint64_t seed)`
- `brh_env_step(brh_env *env, long key, int control, int shift)`
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
be a thin adapter over a scalar `brh_env` once `brh_env` can safely coexist with
other `brh_env` instances in the same process.

## Remaining Blocker

This is still single-instance-safe only. The next performance milestone is a
state-isolation audit of reset, step, and observation fill paths. Until that is
done, a C vector manager or Puffer Ocean binding must not instantiate multiple
Brogue envs in one process.

## Next Milestone: Multiple Envs Per Process

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

## Acceptance Criteria

- `brh_env_create` can create at least two live envs in one process.
- Sequential `N=2` reset/step/close passes repeatedly under sanitizers.
- Same-seed single-env behavior remains unchanged.
- Different-seed envs maintain independent observations and terminal state.
- A Puffer `brogue` binding can run `total_agents = 2`, `num_threads = 1`.
- Only after that: OpenMP stepping with `num_threads > 1`.
