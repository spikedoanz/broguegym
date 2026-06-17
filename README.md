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
obs, info = env.reset(seed=42)  # samples a Brogue game seed from sampler seed 42
obs, info = env.reset(seed=42, options={"game_seed": 123})
obs, reward, term, trunc, info = env.step(env.action_space.sample())
env.close()
```

Batched (for training):

```python
from broguegym import BrogueVectorEnv

envs = BrogueVectorEnv(num_envs=64)
resets = envs.reset(seed=42)  # samples 64 Brogue seeds from sampler seed 42
resets = envs.reset(seed=42, game_seed=list(range(1, 65)))
steps = envs.step(actions)
envs.close()
```

## Development

```bash
uv run pytest
uv run pyright
```

## License

MIT
