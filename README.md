# bruhogue

`bruhogue` is a Gymnasium-compatible Python harness for turning Brogue CE into a
reinforcement-learning environment.

The design follows the NLE (NetHack Learning Environment) in-process model: training code imports
the environment directly, each rollout process owns one Brogue game instance, and the game boundary
is a narrow in-process bridge API. See `docs/nle-process-model.md` for implementation details.

## Repository layout

- `BrogueCE/` — Brogue CE source, vendored as a submodule so bridge changes can be reviewed and
  evolved alongside the Python harness.
- `bruhogue/` — typed Python package with the Gymnasium wrapper, action table, Brogue backend
  contract, observation spaces, and snapshot file format.
- `examples/` — local scripts for interactive play.
- `docs/nle-process-model.md` — implementation notes for the in-process Brogue bridge.
- `tests/` — unit tests for the Gym wrapper, snapshot, observation, and action contracts.

## Installation

With [uv](https://github.com/astral-sh/uv):

```bash
uv sync
```

Or from source with pip:

```bash
pip install .
```

## Quick start

```python
import bruhogue
import gymnasium as gym

env = gym.make(bruhogue.BROGUE_ENV_ID)
observation, info = env.reset(seed=1)

action = env.action_space.sample()
observation, reward, terminated, truncated, info = env.step(action)

env.close()
```

Gym `seed` values seed the Brogue-job sampler. The returned `info["seed"]` is the sampled concrete
Brogue seed for that episode.

`BrogueEnv` defaults to an NLE-like, player-facing observation subset: rendered screen tensors,
fixed-slot inventory tensors, status, latest message bytes, and program bookkeeping. Pass
`include_privileged_info=True` to attach the full bridge view to Gym `info` without changing the
policy observation, or `observation_mode="privileged"` to make it the policy observation
explicitly.

The discrete action table includes the primary player commands plus non-duplicated modal keys for
item/menu selection letters and digits. Pass `actions="full"` to `BrogueEnv` or `gym.make(...)`
for a larger raw-key profile with tab, return, escape, and printable ASCII inputs, closer to NLE
challenge-style full keyboard control.

`render("ansi")` uses a fixed-width ASCII compatibility table keyed by Brogue glyph IDs so terminal
geometry is stable. The `chars` observation still carries Brogue's Unicode codepoints; pass
`render_charset="unicode"` to `BrogueEnv` only when inspecting that raw export.

## Interactive play

```bash
uv run python examples/play.py --seed 1
```

The shim prints one JSON state line and then the rendered board after reset and after each key.
Rejected keys are reported as JSON `event: "error"` records with an `error_code`, followed by the
last known board. With `--actions`, it replays raw keys non-interactively and exits after the
replayed input. Add `--tensors` to print a Python/NumPy repr of the privileged map and inventory
tensors before each board render.

As a quick smoke check:

```bash
uv run python examples/play.py --seed 1 --actions ' z'
```

## Development

```bash
uv run pytest
uv run pyright
```

`uv run` automatically syncs the project before invoking commands. The project cache key includes
the vendored Brogue build inputs, so the bridge in `BrogueCE/bin/` is rebuilt when those inputs
change.

The bridge build produces `BrogueCE/bin/libbruhogue_brogue.dylib` on macOS and
`BrogueCE/bin/libbruhogue_brogue.so` elsewhere. Python loads that library directly through
`ctypes` inside isolated worker processes.

## License

AGPL-3.0-or-later. See [LICENSE](LICENSE).
