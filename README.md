# broguegym

Gymnasium environment for [Brogue CE](https://github.com/tmewett/BrogueCE).

## Install

```bash
git clone --recurse-submodules git@github.com:spikedoanz/broguegym.git
cd broguegym
uv sync
```

If you already cloned without submodules:

```bash
git submodule update --init --recursive
```

## Usage

Single environment (Gymnasium API):

```python
import broguegym
import gymnasium as gym

env = gym.make(broguegym.BROGUE_ENV_ID)
obs, info = env.reset(seed=42)
obs, reward, term, trunc, info = env.step(env.action_space.sample())
env.close()
```

Batched (for training):

```python
from broguegym import BrogueBackend

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

MIT
