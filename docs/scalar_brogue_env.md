# Scalar BrogueEnv Contract

This branch introduces a scalar C environment contract as the bridge target for
Python and for a future Puffer Ocean binding. It is deliberately not a vector
manager.

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

## Remaining Blocker

This is still single-instance-safe only. The next performance milestone is a
state-isolation audit of reset, step, and observation fill paths. Until that is
done, a C vector manager or Puffer Ocean binding must not instantiate multiple
Brogue envs in one process.
