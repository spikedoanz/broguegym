# brogue-gym

Gymnasium environment for [Brogue CE](https://github.com/tmewett/BrogueCE).

## Install

```bash
uv sync
```

## Usage

Single environment (Gymnasium API):

```python
import brogue_gym
import gymnasium as gym

env = gym.make(brogue_gym.BROGUE_ENV_ID)
obs, info = env.reset(seed=42)
obs, reward, term, trunc, info = env.step(env.action_space.sample())
env.close()
```

Batched (for training):

```python
from brogue_gym import BrogueBackend

backend = BrogueBackend(num_envs=64)
resets = backend.reset_many(seed=42)
steps = backend.step_many(actions)
backend.close()
```

## Development

```bash
uv run pytest
uv run pyright
```

## License

AGPL-3.0-or-later
