from __future__ import annotations

# pyright: reportPrivateUsage=false

from typing import Any, cast

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env  # pyright: ignore[reportUnknownVariableType]

from bruhogue import BROGUE_ENV_ID
from bruhogue.actions import Action, BrogueInput
from bruhogue.brogue import BackendInfoKey, BackendReset, BackendStep, ObservationDict, _default_library_path
from bruhogue.gym import BrogueEnv
from bruhogue.snapshot import BrogueSnapshot
from bruhogue.spaces import ACTIONS, FULL_ACTIONS, empty_observation
from bruhogue.brogue import BrogueBackend


class FakeBackend:
    def __init__(self, *, include_semantic_tensors: bool = False) -> None:
        self.actions: list[Action] = []
        self.closed = False
        self.include_semantic_tensors = include_semantic_tensors
        self.restored: list[BrogueSnapshot] = []
        self.reset_seed: int | None = None

    def reset(
        self,
        *,
        seed: int | None = None,
    ) -> BackendReset:
        self.actions.clear()
        self.reset_seed = seed
        return BackendReset(
            observation=empty_observation(include_semantic_tensors=self.include_semantic_tensors),
            info={"seed": seed},
        )

    def step(self, action: Action) -> BackendStep:
        self.actions.append(action)
        observation = empty_observation(include_semantic_tensors=self.include_semantic_tensors)
        observation["program_state"][0] = len(self.actions)
        return BackendStep(
            observation=observation,
            reward=1.25,
            terminated=False,
            info={"key": action.brogue_key()},
        )

    def snapshot(self) -> BrogueSnapshot:
        return BrogueSnapshot(payload=b"complete-state", metadata={"seed": 123})

    def restore(self, snapshot: BrogueSnapshot) -> BackendReset:
        self.restored.append(snapshot)
        observation = empty_observation(include_semantic_tensors=self.include_semantic_tensors)
        observation["program_state"][0] = 99
        return BackendReset(observation=observation, info={"restored": True})

    def close(self) -> None:
        self.closed = True


class ReusingFakeBackend(FakeBackend):
    def __init__(self) -> None:
        super().__init__()
        self.observation = empty_observation()

    def reset(
        self,
        *,
        seed: int | None = None,
    ) -> BackendReset:
        self.reset_seed = seed
        self.observation["program_state"][0] = 0
        return BackendReset(observation=self.observation, info={"seed": seed})

    def step(self, action: Action) -> BackendStep:
        self.actions.append(action)
        self.observation["program_state"][0] += 1
        return BackendStep(
            observation=self.observation,
            reward=0.0,
            terminated=False,
            info={},
        )


def test_env_reset_and_step_dispatch_discrete_actions() -> None:
    backend = FakeBackend()
    env = BrogueEnv(backend=backend, max_episode_steps=2)

    observation, info = env.reset(seed=7)
    assert env.observation_space.contains(observation)
    assert info == {"seed": 7}
    assert backend.reset_seed == 7

    observation, reward, terminated, truncated, info = env.step(0)
    assert env.observation_space.contains(observation)
    assert reward == 1.25
    assert terminated is False
    assert truncated is False
    assert info == {"key": ACTIONS[0].brogue_key()}
    assert backend.actions == [ACTIONS[0]]

    _, _, _, truncated, info = env.step(1)
    assert truncated is True
    assert info["TimeLimit.truncated"] is True


def test_env_observations_are_stable_after_backend_reuses_buffers() -> None:
    backend = ReusingFakeBackend()
    env = BrogueEnv(backend=backend)

    reset_observation, _ = env.reset()
    step_observation, *_ = env.step(0)

    assert int(reset_observation["program_state"][0]) == 0
    assert int(step_observation["program_state"][0]) == 1
    assert not np.shares_memory(
        reset_observation["program_state"],
        backend.observation["program_state"],
    )
    assert not np.shares_memory(
        step_observation["program_state"],
        backend.observation["program_state"],
    )


def test_env_rejects_invalid_action_index() -> None:
    env = BrogueEnv(backend=FakeBackend())

    with pytest.raises(ValueError, match="invalid Brogue action"):
        env.step(len(ACTIONS))


def test_env_defaults_to_player_observation_and_can_opt_into_privileged_mode() -> None:
    base_env = BrogueEnv(backend=FakeBackend(include_semantic_tensors=True))
    base_observation, _ = base_env.reset()

    assert "glyphs" in base_observation
    assert "inventory_letters" in base_observation
    assert "inventory_strs" in base_observation
    assert "map_layers" not in base_observation
    assert base_env.observation_space.contains(base_observation)

    privileged_env = BrogueEnv(
        backend=FakeBackend(include_semantic_tensors=True),
        observation_mode="privileged",
    )
    privileged_observation, _ = privileged_env.reset()

    assert "map_has_item" in privileged_observation
    assert "map_has_monster" in privileged_observation
    assert "inventory_present" in privileged_observation
    assert privileged_env.observation_space.contains(privileged_observation)


def test_env_can_attach_privileged_info_without_policy_observation() -> None:
    backend = FakeBackend(include_semantic_tensors=True)
    env = BrogueEnv(backend=backend, include_privileged_info=True)

    observation, info = env.reset()

    privileged = cast(ObservationDict, info[BackendInfoKey.PRIVILEGED_OBSERVATION])
    assert "map_layers" not in observation
    assert "map_layers" in privileged
    assert "map_has_item" in privileged
    assert env.observation_space.contains(observation)


def test_env_can_use_full_action_profile() -> None:
    backend = FakeBackend()
    env = BrogueEnv(backend=backend, actions="full")
    full_action_index = tuple(action.brogue_input() for action in FULL_ACTIONS).index(
        BrogueInput(key="+"),
    )

    env.reset()
    _, _, _, _, info = env.step(full_action_index)

    assert len(env.action_descriptions) == len(FULL_ACTIONS)
    assert info == {"key": "+"}
    assert backend.actions == [Action.keypress("+")]


def test_env_registers_with_gymnasium_make() -> None:
    make_env = gym.make  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    env = cast(
        Any,
        make_env(
            BROGUE_ENV_ID,
            backend=FakeBackend(),
            disable_env_checker=True,
        ),
    )

    try:
        observation, info = env.reset(seed=11)

        assert info == {"seed": 11}
        assert env.observation_space.contains(observation)
    finally:
        env.close()


def test_env_default_backend_uses_process_backend_path_if_built() -> None:
    if not _default_library_path().exists():
        pytest.skip("run through uv so package sync builds the bridge")
    env = BrogueEnv()
    try:
        assert isinstance(env.backend, BrogueBackend)
    finally:
        env.close()


def test_env_render_includes_ansi_colors_when_present() -> None:
    env = BrogueEnv(backend=FakeBackend(), render_mode="ansi")

    observation, _ = env.reset()
    observation["chars"][0, 0] = ord("@")
    observation["colors_fg"][0, 0] = [255, 128, 0]
    observation["colors_bg"][0, 0] = [0, 0, 64]

    rendered = env.render()

    assert rendered is not None
    assert "\x1b[38;2;255;128;0;48;2;0;0;64m@" in rendered


def test_env_render_defaults_to_ascii_charset_for_unicode_glyphs() -> None:
    env = BrogueEnv(backend=FakeBackend(), render_mode="ansi")

    observation, _ = env.reset()
    observation["glyphs"][0, 0] = 198
    observation["chars"][0, 0] = 0x25C7
    observation["glyphs"][0, 1] = 225
    observation["chars"][0, 1] = 0x2648

    rendered = env.render()

    assert rendered is not None
    assert rendered.splitlines()[0].startswith("%&")


def test_env_passes_gymnasium_checker() -> None:
    check_env(BrogueEnv(backend=FakeBackend()), skip_render_check=True)
