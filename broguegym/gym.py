"""Gymnasium environment wrapper for Brogue."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Final, Literal, Protocol, Self, cast

import gymnasium as gym
from gymnasium.envs.registration import register, registry  # pyright: ignore[reportUnknownVariableType]

from broguegym._render import TerminalCharset, observation_to_ansi
from broguegym.actions import Action
from broguegym.brogue import (
    BackendInfoKey,
    BackendReset,
    BackendStep,
    BrogueGameSeedSpec,
    BrogueVectorEnv,
    ObservationDict,
)
from broguegym.snapshot import BrogueSnapshot
from broguegym.spaces import (
    ActionSet,
    ObservationMode,
    action_description,
    filter_observation,
    observation_space,
    resolve_observation_mode,
    resolve_actions,
)

BROGUE_ENV_ID: Final = "Bruhogue-v0"


class _BrogueVectorGame(Protocol):
    num_envs: int

    def reset(
        self,
        *,
        seed: int | None = None,
        game_seed: BrogueGameSeedSpec | None = None,
    ) -> list[BackendReset]: ...

    def step(self, actions: Sequence[Action]) -> list[BackendStep]: ...

    def snapshot(self) -> BrogueSnapshot: ...

    def restore(self, snapshot: BrogueSnapshot) -> BackendReset: ...

    def close(self) -> None: ...


def register_envs() -> None:
    """Register Bruhogue environments with Gymnasium if needed."""

    if BROGUE_ENV_ID in registry:
        return
    register(
        id=BROGUE_ENV_ID,
        entry_point="broguegym.gym:BrogueEnv",
    )


def _reset_game_seed(options: dict[str, object] | None) -> BrogueGameSeedSpec | None:
    if not options:
        return None
    if set(options) != {"game_seed"}:
        msg = "BrogueEnv reset options support only 'game_seed'"
        raise ValueError(msg)
    return cast(BrogueGameSeedSpec, options["game_seed"])


class BrogueEnv(gym.Env[ObservationDict, int]):
    """Scalar Gymnasium wrapper around a one-environment BrogueVectorEnv.

    The reset ``seed`` is a sampler seed. Reset option ``{"game_seed": ...}``
    overrides the concrete Brogue seed sampled from it.
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        vector_env: _BrogueVectorGame | None = None,
        *,
        render_mode: str | None = None,
        render_charset: TerminalCharset = "ascii",
        max_episode_steps: int | None = None,
        observation_mode: ObservationMode = "player",
        include_privileged_info: bool = False,
        include_semantic_tensors: bool | None = None,
        actions: Sequence[Action] | ActionSet | None = None,
    ) -> None:
        if render_mode not in (None, "ansi"):
            msg = f"unsupported render mode: {render_mode}"
            raise ValueError(msg)
        if render_charset not in ("ascii", "unicode"):
            msg = f"unsupported render charset: {render_charset}"
            raise ValueError(msg)
        if max_episode_steps is not None and max_episode_steps <= 0:
            msg = "max_episode_steps must be positive"
            raise ValueError(msg)

        self.vector_env = vector_env if vector_env is not None else BrogueVectorEnv(1)
        if self.vector_env.num_envs != 1:
            msg = "BrogueEnv requires a vector_env with num_envs == 1"
            raise ValueError(msg)
        self.render_mode = render_mode
        self.render_charset: TerminalCharset = render_charset
        self.max_episode_steps = max_episode_steps
        self.observation_mode: ObservationMode = resolve_observation_mode(
            observation_mode,
            include_semantic_tensors=include_semantic_tensors,
        )
        self.include_semantic_tensors = self.observation_mode == "privileged"
        self.include_privileged_info = include_privileged_info
        self.actions = resolve_actions(actions)
        self.action_descriptions = tuple(action_description(action) for action in self.actions)
        self.action_space = cast(gym.Space[int], gym.spaces.Discrete(len(self.actions)))
        self.observation_space = observation_space(
            observation_mode=self.observation_mode,
        )
        self._elapsed_steps = 0
        self._last_observation: ObservationDict | None = None

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, object] | None = None,
    ) -> tuple[ObservationDict, dict[str, object]]:
        """Reset the wrapped Brogue game and return Gymnasium's `(observation, info)` pair."""

        game_seed = _reset_game_seed(options)
        super().reset(seed=seed)
        result = self.vector_env.reset(seed=seed, game_seed=game_seed)[0]
        observation = self._gym_observation(result.observation)
        self._elapsed_steps = 0
        self._last_observation = observation
        return observation, self._gym_info(result.info, result.observation)

    def step(self, action: int) -> tuple[ObservationDict, float, bool, bool, dict[str, object]]:
        """Apply one discrete action index."""

        if not self.action_space.contains(action):
            msg = f"invalid Brogue action index: {action}"
            raise ValueError(msg)

        result = self.vector_env.step([self.actions[int(action)]])[0]
        observation = self._gym_observation(result.observation)
        self._elapsed_steps += 1
        truncated = (
            self.max_episode_steps is not None
            and self._elapsed_steps >= self.max_episode_steps
            and not result.terminated
        )
        info = dict(result.info)
        if truncated:
            info["TimeLimit.truncated"] = True

        self._last_observation = observation
        info = self._gym_info(info, result.observation)
        return observation, result.reward, result.terminated, truncated, info

    def render(self) -> str | None:
        """Render the latest character observation as ANSI terminal text."""

        if self.render_mode != "ansi":
            return None
        if self._last_observation is None:
            return ""
        return observation_to_ansi(self._last_observation, charset=self.render_charset)

    def snapshot(self) -> BrogueSnapshot:
        """Return an in-memory full-state snapshot from the backend."""

        return self.vector_env.snapshot()

    def restore(self, snapshot: BrogueSnapshot) -> tuple[ObservationDict, dict[str, object]]:
        """Restore an in-memory snapshot and return the restored observation."""

        result = self.vector_env.restore(snapshot)
        observation = self._gym_observation(result.observation)
        self._elapsed_steps = 0
        self._last_observation = observation
        return observation, self._gym_info(result.info, result.observation)

    def save_snapshot(self, path: Path | str) -> BrogueSnapshot:
        """Capture the current game state and write it to disk."""

        snapshot = self.snapshot()
        snapshot.write_to(path)
        return snapshot

    def load_snapshot(self, path: Path | str) -> tuple[ObservationDict, dict[str, object]]:
        """Load a snapshot from disk, restore it, and return the restored observation."""

        return self.restore(BrogueSnapshot.read_from(path))

    def close(self) -> None:
        """Release backend resources."""

        self.vector_env.close()

    def __enter__(self) -> Self:
        """Allow `with BrogueEnv(...) as env:` usage in training scripts."""

        return self

    def __exit__(self, *args: object) -> Literal[False]:
        """Close the backend when leaving a context manager."""

        _ = args
        self.close()
        return False

    def _gym_observation(self, observation: ObservationDict) -> ObservationDict:
        filtered = filter_observation(
            observation,
            observation_mode=self.observation_mode,
        )
        return {key: value.copy() for key, value in filtered.items()}

    def _gym_info(
        self,
        info: dict[str, object],
        bridge_observation: ObservationDict,
    ) -> dict[str, object]:
        gym_info = dict(info)
        if self.include_privileged_info:
            privileged = filter_observation(
                bridge_observation,
                observation_mode="privileged",
            )
            gym_info[BackendInfoKey.PRIVILEGED_OBSERVATION] = {
                key: value.copy() for key, value in privileged.items()
            }
        return gym_info
