# Puffer Brogue Benchmark Results

These results use Puffer's native `profile envspeed` path:

- `create_static_vec`
- `create_static_threads`
- `static_vec_reset`
- timed `static_vec_omp_step`

The Brogue Puffer env uses a compact byte observation by default:

- map chars: `79 * 29 = 2291` bytes
- blstats: `21 * int32 = 84` bytes
- program state: `8 * int32 = 32` bytes
- total: `2407` bytes

The full bridge observation is still `155,032` bytes. That is `93.4x` the
current NLE chars-only observation (`1659` bytes), which is the source of the
"100x bigger" number. The Puffer policy observation is not using that full
payload by default; its Brogue observation is now only `1.45x` NLE.

Fidelity correction: compact observation no longer implies compact simulation
shortcuts. The default `brh_step_compact` path now keeps full Brogue turn,
monster, lighting, and environment simulation. The earlier compact fast-turn
path is available only as an explicit lossy benchmark mode by setting
`BROGUE_COMPACT_SIMULATION_SHORTCUTS=1`; it should not be treated as the
playable agent environment.

## Latest Same-Harness Results

These rows were run with `profile envspeed` phase timing, explicit benchmark
action modes, optional `BROGUE_PROFILE=1` Brogue counters, the compact bridge
observation path, map-only compact observations, compact mode sidebar elision,
the bridge/server fast-input path, and the dirty-cell compact map cache. The
most relevant column for simulation speed is `Env-step SPS/core`: it is
computed from Puffer's `EVAL_ENV_STEP` timer, so GPU/copy time is excluded.

Command shape:

```sh
./profile envspeed --total-agents 64 --buffers 1 --threads T --horizon 256 --action-mode zero
```

For NLE:

```sh
NETHACKDIR=$PWD/vendor/nle/src/build/dat \
LD_LIBRARY_PATH=$PWD/vendor/nle/src/build:$LD_LIBRARY_PATH \
  ./profile envspeed --total-agents 64 --buffers 1 --threads T --horizon 256
```

| Env | Threads | Obs bytes | Rollout ms | Puffer gpu/copy ms | Puffer env-step ms | Wall SPS | Env-step SPS/core |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Brogue zero action | 1 | 2407 | 149.90 | 31.01 | 116.42 | 109301 | 140730 |
| NLE zero action | 1 | 1659 | 62.80 | 41.73 | 19.35 | 260879 | 846718 |

Additional Brogue action modes:

| Env/action stream | Threads | Obs bytes | Rollout ms | Puffer gpu/copy ms | Puffer env-step ms | Wall SPS | Env-step SPS/core |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Brogue fixed action 18 (`s` search) | 1 | 2407 | 3063.86 | 295.48 | 2763.41 | 5347 | 5929 |
| Brogue random action ids `[0,8)`, faithful simulation | 1 | 2407 | 2746.14 | 16.49 | 2726.04 | 5966 | 6008 |
| Brogue random action ids `[0,8)`, `BROGUE_COMPACT_SIMULATION_SHORTCUTS=1` | 1 | 2407 | 396.66 | 10.01 | 384.55 | 41304 | 42604 |
| NLE random action ids `[0,8)` | 1 | 1659 | 272.94 | 31.48 | 238.86 | 60027 | 68591 |

Single-core simulation gap at this shape:

- Brogue zero-action headline: `~141k` env-step SPS/core
- NLE zero-action headline: `~847k` env-step SPS/core
- zero-action gap: `~6.0x`
- Brogue fixed search trace: `~5.9k` env-step SPS/core
- Brogue bounded movement-random trace with faithful simulation: `~6.0k`
  env-step SPS/core
- Brogue bounded movement-random trace with explicit lossy compact simulation
  shortcuts: `~42.6k` env-step SPS/core
- NLE bounded random trace: `~68.6k` env-step SPS/core
- faithful movement-random gap: `~11.4x`
- lossy shortcut movement-random gap: `~1.6x`

For NLE, total wall throughput is GPU/copy limited at this small 64-agent
shape. For Brogue, total wall throughput is CPU simulation limited.

## Brogue Profile Breakdown

Profile command:

```sh
BROGUE_PROFILE=1 ./profile envspeed --total-agents 64 --buffers 1 --threads 1 --horizon 256
```

Observed Brogue timed-loop counters after compact bridge capture, the
server-mode terminal-flush bypass, map-only compact observations, compact mode
sidebar elision, bridge fast input, the dirty-cell compact map cache, expanded
turn-loop profile zones, compact terrain-glow and environment shortcuts,
compact miner-light reuse, compact visibility transitions, cached
gas-obstruction terrain flags, active gas-cell diffusion, and cached light
distances:

| Zero-action phase | Time |
| --- | ---: |
| wrapper `c_step` total | 11.95 us/step |
| bridge `api.step` | 11.75 us/step |
| reward/bookkeeping | 0.03 us/step |
| compact Puffer pack | 0.06 us/step |
| bridge `brh_step` | 11.59 us/step |
| coroutine resume/game work | 11.38 us/step |
| full bridge observation fill | 0.00 us/step |
| compact bridge observation fill | 0.17 us/step |
| `execute_event` inclusive | 7.77 us/step |
| `refresh_sidebar` inclusive | 0.04 us/step |
| input-loop prep inclusive | 0.02 us/step |
| `commitDraws` | 0.04 us/step |

Fixed search profile:

| Search phase | Time |
| --- | ---: |
| wrapper `c_step` total | 189.87 us/step |
| compact bridge observation fill | 0.27 us/step |
| pause callbacks | 83.80 calls/step |
| `commitDraws` | 1.99 us/step |
| `manual_search` inclusive | 38.03 us/step |
| `playerTurnEnded` inclusive | 37.57 us/step |
| `updateVision` inclusive | 16.51 us/step |
| `updateEnvironment` inclusive | 10.60 us/step |
| input-loop prep inclusive | 0.00 us/step |

Pre-fast-turn bounded movement-random profile:

| Movement-random phase | Time |
| --- | ---: |
| wrapper `c_step` total | 71.32 us/step |
| compact bridge observation fill | 0.31 us/step |
| `execute_event` inclusive | 24.67 us/step |
| `playerTurnEnded` inclusive | 22.55 us/step |
| `updateVision` inclusive | 10.16 us/step |
| `updateEnvironment` inclusive | 0.16 us/step |
| `refresh_dungeon_cell` inclusive | 1.09 us/step |
| `refresh_sidebar` inclusive | 0.04 us/step |
| input-loop prep inclusive | 0.02 us/step |
| `commitDraws` | 0.06 us/step |

Additional movement-random sub-zones:

| Sub-zone | Time |
| --- | ---: |
| `vision_fov` inclusive | 3.13 us/step |
| `vision_lighting` inclusive | 4.08 us/step |
| `vision_display` inclusive | 2.16 us/step |
| `lighting_tile_glow` inclusive | 0.26 us/step |
| `lighting_miner` inclusive | 1.75 us/step |
| `turn_update_scent` inclusive | 3.44 us/step |
| `turn_schedule_loop` inclusive | 6.04 us/step |
| `monsters_turn` inclusive | 3.51 us/step |
| `env_bookkeeping` inclusive | 0.03 us/step |

Conclusion: the full observation explains the apparent "100x bigger" number,
but it is not the current default-path bottleneck. Compact observation fill is
now sub-microsecond in steady state, and sidebar refresh has been removed from
the compact benchmark path. Bridge fast input removes input-loop prep from the
benchmark path. With faithful simulation, legal movement is still dominated by
Brogue turn simulation, especially vision/FOV/light bookkeeping, scent, monster
scheduling, environment updates, and remaining bridge coroutine/control-flow
overhead. The compact fast-turn shortcut can move bounded movement-random into
the same range as NLE, but it skips core mechanics and is now explicitly
opt-in only.

The first targeted optimization was to skip terminal flushes in bridge/server
mode. `commitDraws()` only pushes `displayBuffer` to the terminal backend; the
Puffer observation reads `displayBuffer` directly. That changed fixed rest from
`~1.45k` wall SPS to `~5.21k` wall SPS and bounded random movement from
`~1.53k` to `~2.14k` wall SPS. The second targeted optimization moved compact
observations from full-screen chars to map-only chars and skipped sidebar
refresh in compact/no-observation bridge mode. That changed zero action from
`~6.38k` to `~13.2k` env-step SPS/core and bounded random movement from
`~2.15k` to `~2.81k` env-step SPS/core. The third targeted optimization added
a bridge fast-input path, changing zero action from `~13.2k` to `~62.7k`
env-step SPS/core and bounded random movement from `~2.81k` to `~3.19k`
env-step SPS/core. The fourth targeted optimization replaced full compact-map
rescans with a dirty-cell semantic map cache, changing zero action to
`~87.8k` env-step SPS/core and bounded random movement to `~3.81k` env-step
SPS/core. The fifth targeted optimization added expanded profile zones and a
compact-mode lighting filter, changing bounded random movement to `~4.16k`
env-step SPS/core. The sixth set of targeted optimizations reused current FOV
for compact miner light, localized gas diffusion to active gas bounds, replaced
several full-map environment phases with tracked cell lists, and added a compact
visibility transition path. That changed bounded random movement to a clean
`~5.9k` env-step SPS/core sample, with subsequent noisy samples in the
`~5.4-5.9k` range. The seventh targeted set added cached gas-obstruction
terrain flags, an active gas-cell candidate list, cached light distances, and
promotion neighbor terrain-flag caching. That moved the clean bounded
movement-random sample to `~6.2k` env-step SPS/core and reduced the profiled
movement-random sub-zones to `env_gas ~7.7 us/step`, `lighting_miner
~1.3 us/step`, and `env_promotions ~3.7 us/step`.
The eighth targeted set skipped compact display-buffer work, skipped compact
rolling waypoint refreshes, skipped compact terrain-glow painting, skipped
compact gas/promotion/fire environment simulation, and used compact visibility
transition lists. That moved the clean bounded movement-random sample to
`~14.5k` env-step SPS/core. A direct compact `executeEvent()` path was tested
and rejected because it segfaulted under the Puffer multi-env shared-library
setup; the coroutine path remains the stable bridge.
The ninth targeted set added compact fast turns and demonstrated that skipping
monster scheduling could move bounded movement-random to the `~40k` env-step
SPS/core range. That is no longer the default environment because it changes
game semantics. It is gated behind `BROGUE_COMPACT_SIMULATION_SHORTCUTS=1` for
diagnostic benchmarking only.

## Build Commands

The local machine has GCC/libgomp and CUDA wheel NCCL/CuDNN libraries, so the
working build command is:

```sh
cd broguegym-dev
PATH=/home/spike/.venv/bin:$PATH \
PYTHONPATH=/home/spike/r/broguegym/.venv/lib/python3.13/site-packages:$PYTHONPATH \
CC=gcc CXX=g++ OMP_LIB=-lgomp \
  bash build.sh brogue --profile
```

For NLE:

```sh
cd broguegym-dev
PATH=/home/spike/.venv/bin:$PATH \
PYTHONPATH=/home/spike/r/broguegym/.venv/lib/python3.13/site-packages:$PYTHONPATH \
CC=gcc CXX=g++ OMP_LIB=-lgomp \
  bash build.sh nethack --profile
```

Runtime library path:

```sh
export LD_LIBRARY_PATH=$PWD/vendor/nle/src/build:/home/spike/.venv/lib/python3.13/site-packages/nvidia/nccl/lib:/home/spike/.venv/lib/python3.13/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH
```

The historical snapshots below predate the explicit Puffer `gpu/copy` versus
`env_step` phase print. Treat their SPS/core values as wall-throughput
normalization, not CPU-only simulation rate.

## Brogue Results: Private-Library Pthread Bridge

Command shape:

```sh
./profile envspeed --total-agents N --buffers 1 --threads T --horizon 256
```

| Env | Agents | Threads | Obs bytes | Rollout ms | Steps/s | Steps/s/core |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Brogue | 1 | 1 | 3516 | 78.37 | 3267 | 3267 |
| Brogue | 2 | 1 | 3516 | 126.60 | 4044 | 4044 |
| Brogue | 8 | 1 | 3516 | 461.23 | 4440 | 4440 |
| Brogue | 64 | 1 | 3516 | 4004.04 | 4092 | 4092 |
| Brogue | 64 | 2 | 3516 | 2131.05 | 7688 | 3844 |
| Brogue | 64 | 4 | 3516 | 1224.14 | 13384 | 3346 |
| Brogue | 64 | 8 | 3516 | 725.95 | 22569 | 2821 |

## Brogue Results: Private-Library Coroutine Bridge

This replaces the bridge pthread/condition-variable handoff with a same-thread
`ucontext` coroutine. The public bridge ABI is unchanged.

Direct scalar C benchmark deltas were modest:

| Case | Pthread median us | Coroutine median us | Coroutine steps/s |
| --- | ---: | ---: | ---: |
| invalid-key | 94.95 | 82.87 | 12068 |
| invalid-no-obs | 64.40 | 53.22 | 18792 |
| rest | 686.83 | 669.54 | 1494 |
| rest-no-obs | 653.20 | 643.75 | 1553 |
| search | 676.27 | 673.08 | 1486 |
| search-no-obs | 650.38 | 649.85 | 1539 |
| explore | 462.81 | 449.97 | 2222 |
| explore-no-obs | 408.95 | 398.62 | 2509 |
| fast-explore | 392.74 | 382.98 | 2611 |
| fastx-no-obs | 368.76 | 356.33 | 2806 |

Puffer `profile envspeed` results:

| Env | Agents | Threads | Obs bytes | Rollout ms | Steps/s | Steps/s/core |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Brogue | 64 | 1 | 3516 | 3530.54 | 4641 | 4641 |
| Brogue | 64 | 2 | 3516 | 2160.61 | 7583 | 3792 |
| Brogue | 64 | 4 | 3516 | 1036.34 | 15809 | 3952 |
| Brogue | 64 | 8 | 3516 | 457.62 | 35803 | 4475 |

## NLE Results

Command shape:

```sh
NETHACKDIR=$PWD/vendor/nle/src/build/dat \
  ./profile envspeed --total-agents 64 --buffers 1 --threads T --horizon 256
```

| Env | Agents | Threads | Obs bytes | Rollout ms | Steps/s | Steps/s/core |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| NLE | 64 | 1 | 1659 | 60.15 | 272397 | 272397 |
| NLE | 64 | 4 | 1659 | 58.19 | 281581 | 70395 |
| NLE | 64 | 8 | 1659 | 58.31 | 280960 | 35120 |

## Interpretation

Brogue now benchmarks inside Puffer with the same `envspeed` harness as NLE and
supports many envs in one process through private shared-library copies.

The compact Brogue observation is `2407` bytes, about `1.45x` NLE's `1659` byte
chars-only observation. The full bridge observation is about `93x` NLE's
observation, but that full payload is not the Puffer default.

At `total_agents=64, threads=1`, the CPU simulation rate from Puffer's
`EVAL_ENV_STEP` bucket is about `84.5k` SPS/core for Brogue zero-action and
`878k` SPS/core for NLE zero-action. The remaining single-core zero-action
simulation gap is about `10.4x`. Brogue fixed search is now about `5.44k`
SPS/core; bounded movement-random is about `4.16k` SPS/core. NLE bounded
movement-random is about `70.4k` SPS/core, a `~16.9x` gap.

The bridge handoff and observation copy are not the dominant legal-step costs.
After terminal flushes, sidebar refresh, input-loop prep, compact observation
rescans, and compact-only lighting work were removed from compact bridge mode,
movement is dominated by Brogue game logic: `playerTurnEnded`,
`updateEnvironment`, and the remaining pieces of `updateVision`. The next
performance targets are gas/environment bookkeeping and incremental
vision/environment updates.
