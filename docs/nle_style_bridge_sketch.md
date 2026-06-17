# NLE-Style Brogue Bridge Sketch

This sketch is for a future high-throughput Brogue runtime. The current bridge
is a compatibility bridge: one Brogue game runs behind a pthread, the caller
submits one action, and the bridge rebuilds/copies a full observation at every
input boundary. That is robust, but it is not shaped like NLE or a native VecEnv.

## Performance Target

The target shape is a native stepper with stable, pre-registered observation
buffers:

```c
brh_env_step(env, action, obs_views);
```

The hot path should avoid:

- per-step pthread/condition-variable handoff
- allocating during step
- formatting strings unless the requested observation requires them
- rebuilding unchanged observation fields
- copying through an intermediate full-observation struct

## What NLE Uses

NLE's core contracts are:

- Python allocates NumPy arrays and passes their pointers to native code once.
- Native code stores those pointers in an observation struct of field pointers.
- NetHack runs on a user-space coroutine stack via `fcontext`.
- At each input boundary, the window-port fills requested observation buffers
  and yields to the caller.
- Individual observation fields can be omitted by passing null buffers.
- Multiple NetHack instances are isolated by loading separate copies of the
  NetHack shared object, because NetHack still has global state.

The important distinction is that NLE is not "just faster C". It is a different
communication contract: direct buffers, explicit yielded control, optional
observation fields, and no OS-thread scheduler rendezvous per step.

## Proposed Communication Contracts

### 1. ABI Versioning

Every native API must expose:

```c
uint32_t brh_abi_version(void);
uint32_t brh_feature_flags(void);
const char *brh_build_id(void);
```

Python must fail closed on an ABI mismatch. A minor-compatible ABI can add fields
only behind feature flags; removing or changing field layout increments the ABI.

### 2. Environment Ownership

The caller owns opaque environment handles:

```c
typedef struct brh_env brh_env;

int brh_env_create(const brh_config *config, brh_env **out);
int brh_env_reset(brh_env *env, uint64_t seed);
int brh_env_step(brh_env *env, const brh_action *action, brh_step_result *out);
void brh_env_destroy(brh_env *env);
```

The handle is the unit of isolation. If Brogue globals remain, the first
implementation may still map one handle to one dynamically loaded Brogue image or
one worker. A true VecEnv later should move mutable game state into per-env
storage.

### 3. Buffer Registration

Python allocates observation arrays once and registers pointers:

```c
int brh_env_set_buffers(brh_env *env, const brh_observation_buffers *buffers);
```

Rules:

- The caller owns memory and must keep it alive until unregister/destroy.
- The native bridge writes only during `reset` and `step`.
- Null field pointers mean "do not populate this observation field".
- Shape, dtype, stride, and alignment are validated at registration.
- Hot step code assumes registered buffers are valid and does not revalidate.

### 4. Observation Profiles

Observation should be requested by profiles or field masks:

- `screen`: glyph/char/color/special/message/blstats/program_state
- `semantic`: visible remembered map/item/monster semantics
- `privileged`: full map and hidden state useful for debugging/curriculum
- `inventory_text`: item names and inventory strings
- `terminal`: tty-like terminal cells, if supported

Default training should not pay for privileged or text fields.

### 5. Step Semantics

One `step` consumes one caller action and returns when Brogue reaches the next
agent decision boundary.

The result contract should distinguish:

- `BRH_STEP_OK`: action accepted; game may or may not have consumed a turn
- `BRH_STEP_INVALID`: key was rejected in the current mode
- `BRH_STEP_MODAL`: game is waiting in a menu/text/more prompt
- `BRH_STEP_TERMINATED`: game reached terminal state
- `BRH_STEP_ERROR`: native failure; inspect `brh_last_error`

The contract must state whether "invalid" returns an updated observation and
whether invalid keys count as environment steps. For RL, returning a fresh
observation is preferable, but the caller should be able to filter invalid steps.

### 6. Modal/Input Boundaries

Brogue has normal movement/action input plus modal prompts. The native bridge
must expose enough mode information to avoid agents blindly sending keys:

```c
typedef enum {
    BRH_INPUT_GAME,
    BRH_INPUT_MORE,
    BRH_INPUT_MENU,
    BRH_INPUT_TEXT,
    BRH_INPUT_DIRECTION,
    BRH_INPUT_CONFIRM,
} brh_input_mode;
```

This belongs in `program_state` or a small typed state struct.

### 7. Determinism

Reset must specify all seed inputs that affect gameplay and display randomness.
If Brogue has separate substantive/cosmetic RNG streams, the contract should
either seed both explicitly or document which stream is observable.

### 8. Lifecycle and Errors

The native bridge must define:

- whether `reset` can be called after terminal
- whether `reset` reuses allocations
- whether `destroy` is safe from any state
- whether `close` can interrupt a blocked modal
- whether errors poison the env handle

For high-throughput batching, failed env handles should be destroyable without
tearing down the whole vector.

### 9. Thread Safety

A single env handle is not thread-safe unless documented otherwise. A VecEnv API
can be thread-safe at the vector level only if each env state is independent.

Global-state Brogue builds must explicitly document that only one active env may
exist per loaded image.

### 10. Batch API

The PufferLib-style surface should eventually be:

```c
int brh_vec_create(size_t num_envs, const brh_config *config, brh_vec **out);
int brh_vec_set_buffers(brh_vec *vec, const brh_vec_observation_buffers *buffers);
int brh_vec_reset(brh_vec *vec, const uint64_t *seeds, uint8_t *reset_mask);
int brh_vec_step(brh_vec *vec, const brh_action *actions, brh_step_result *results);
void brh_vec_destroy(brh_vec *vec);
```

The batch layout should be structure-of-arrays with leading env dimension:

```text
glyphs[num_envs][rows][cols]
colors[num_envs][rows][cols][3]
program_state[num_envs][state_fields]
```

This lets Python, PufferLib, and learner code consume contiguous slabs without
per-env dictionary allocation in the hot path.

## Implementation Options

### Option A: Direct Buffers on Current Pthread Bridge

Keep the pthread bridge but register output buffers directly. This removes
caller-side observation copies and allows optional fields. It is lower risk and
should be the first measurable improvement.

Expected benefit: moderate. It does not remove scheduler handoff.

### Option B: Smaller Stack and Tightened Pthread Path

Measure actual stack use and reduce `BRH_THREAD_STACK_SIZE` if safe. Also avoid
holding the bridge mutex while copying large observations.

Expected benefit: memory and some latency. Still not NLE-like.

### Option C: Coroutine/Fiber Bridge

Replace the game thread with a user-space context switch similar to NLE. Brogue
runs until input is requested, yields to the caller, and resumes with one action.

Expected benefit: removes OS scheduler from per-step path. Risk is high:
custom stacks, abnormal exits, sanitizer/debugger friction, and portability.

### Option D: Real Per-Env State Refactor

Move Brogue mutable globals into explicit state and implement a direct stepper.

Expected benefit: best long-term architecture and true C VecEnv. Cost is very
high and invasive.

## Recommended Sequence

Phase 1 on `sketch/nle-style-bridge` implements Option A without changing the
singleton ownership model:

- BrogueCE ABI 8 adds registered observation buffers and group masks.
- Legacy `brh_reset`/`brh_step` remain available and continue returning full
  copied observations.
- The Python process backend registers each worker's shared-memory observation
  slot, then skips the intermediate C-struct-to-NumPy copy when ABI 8 symbols are
  available.
- The benchmark can compare `--api legacy` vs `--api registered` and full vs
  screen-only observation profiles.

The remaining gap versus NLE is still the control-transfer model: the current
path uses pthread/condition-variable handoff at each input boundary, while NLE
uses a user-space context switch.

1. Add direct buffer registration to the existing bridge.
2. Add observation field masks/profiles and benchmark screen-only vs full.
3. Add fill-only and game-only benchmarks to locate remaining cost.
4. Reduce stack size based on measured stack usage.
5. Prototype a coroutine bridge behind an experimental build flag.
6. Only consider a full state refactor after the cheaper steps are measured.
