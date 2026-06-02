from __future__ import annotations

# pyright: reportPrivateUsage=false

from collections.abc import Sequence
from typing import Any, cast

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env  # pyright: ignore[reportUnknownVariableType]

from broguegym import BROGUE_ENV_ID
from broguegym.actions import Action, BrogueInput
from broguegym.brogue import (
    BackendInfoKey,
    BackendReset,
    BackendStep,
    BrogueGameSeedSpec,
    BrogueVectorEnv,
    ObservationDict,
    _default_library_path,
    _sample_brogue_seeds,
)
from broguegym.gym import BrogueEnv
from broguegym.snapshot import BrogueSnapshot
from broguegym.spaces import ACTIONS, FULL_ACTIONS, empty_observation


def _fake_reset_info_seed(
    seed: int | None,
    game_seed: BrogueGameSeedSpec | None,
) -> object:
    if game_seed is None:
        return seed
    if isinstance(game_seed, int):
        return game_seed
    return list(game_seed)[0]


class FakeVectorEnv:
    def __init__(self, *, include_semantic_tensors: bool = False) -> None:
        self.num_envs = 1
        self.actions: list[Action] = []
        self.closed = False
        self.include_semantic_tensors = include_semantic_tensors
        self.restored: list[BrogueSnapshot] = []
        self.reset_seed: int | None = None
        self.reset_game_seed: BrogueGameSeedSpec | None = None

    def reset(
        self,
        *,
        seed: int | None = None,
        game_seed: BrogueGameSeedSpec | None = None,
    ) -> list[BackendReset]:
        self.actions.clear()
        self.reset_seed = seed
        self.reset_game_seed = game_seed
        return [
            BackendReset(
                observation=empty_observation(include_semantic_tensors=self.include_semantic_tensors),
                info={"seed": _fake_reset_info_seed(seed, game_seed)},
            ),
        ]

    def step(self, actions: Sequence[Action]) -> list[BackendStep]:
        action = actions[0]
        self.actions.append(action)
        observation = empty_observation(include_semantic_tensors=self.include_semantic_tensors)
        observation["program_state"][0] = len(self.actions)
        return [
            BackendStep(
                observation=observation,
                reward=1.25,
                terminated=False,
                info={"key": action.brogue_key()},
            ),
        ]

    def snapshot(self) -> BrogueSnapshot:
        return BrogueSnapshot(payload=b"complete-state", metadata={"seed": 123})

    def restore(self, snapshot: BrogueSnapshot) -> BackendReset:
        self.restored.append(snapshot)
        observation = empty_observation(include_semantic_tensors=self.include_semantic_tensors)
        observation["program_state"][0] = 99
        return BackendReset(observation=observation, info={"restored": True})

    def close(self) -> None:
        self.closed = True


class ReusingFakeVectorEnv(FakeVectorEnv):
    def __init__(self) -> None:
        super().__init__()
        self.observation = empty_observation()

    def reset(
        self,
        *,
        seed: int | None = None,
        game_seed: BrogueGameSeedSpec | None = None,
    ) -> list[BackendReset]:
        self.reset_seed = seed
        self.reset_game_seed = game_seed
        self.observation["program_state"][0] = 0
        return [
            BackendReset(
                observation=self.observation,
                info={"seed": _fake_reset_info_seed(seed, game_seed)},
            ),
        ]

    def step(self, actions: Sequence[Action]) -> list[BackendStep]:
        action = actions[0]
        self.actions.append(action)
        self.observation["program_state"][0] += 1
        return [
            BackendStep(
                observation=self.observation,
                reward=0.0,
                terminated=False,
                info={},
            ),
        ]


def test_env_reset_and_step_dispatch_discrete_actions() -> None:
    vector_env = FakeVectorEnv()
    env = BrogueEnv(vector_env=vector_env, max_episode_steps=2)

    observation, info = env.reset(seed=7)
    assert env.observation_space.contains(observation)
    assert info == {"seed": 7}
    assert vector_env.reset_seed == 7

    observation, reward, terminated, truncated, info = env.step(0)
    assert env.observation_space.contains(observation)
    assert reward == 1.25
    assert terminated is False
    assert truncated is False
    assert info == {"key": ACTIONS[0].brogue_key()}
    assert vector_env.actions == [ACTIONS[0]]

    _, _, _, truncated, info = env.step(1)
    assert truncated is True
    assert info["TimeLimit.truncated"] is True


def test_env_reset_routes_game_seed_option() -> None:
    vector_env = FakeVectorEnv()
    env = BrogueEnv(vector_env=vector_env)

    observation, info = env.reset(seed=11, options={"game_seed": 7})

    assert env.observation_space.contains(observation)
    assert info == {"seed": 7}
    assert vector_env.reset_seed == 11
    assert vector_env.reset_game_seed == 7


def test_env_reset_rejects_unknown_options() -> None:
    env = BrogueEnv(vector_env=FakeVectorEnv())

    with pytest.raises(ValueError, match="reset options"):
        env.reset(options={"unknown": 7})


def test_env_observations_are_stable_after_backend_reuses_buffers() -> None:
    vector_env = ReusingFakeVectorEnv()
    env = BrogueEnv(vector_env=vector_env)

    reset_observation, _ = env.reset()
    step_observation, *_ = env.step(0)

    assert int(reset_observation["program_state"][0]) == 0
    assert int(step_observation["program_state"][0]) == 1
    assert not np.shares_memory(
        reset_observation["program_state"],
        vector_env.observation["program_state"],
    )
    assert not np.shares_memory(
        step_observation["program_state"],
        vector_env.observation["program_state"],
    )


def test_env_rejects_invalid_action_index() -> None:
    env = BrogueEnv(vector_env=FakeVectorEnv())

    with pytest.raises(ValueError, match="invalid Brogue action"):
        env.step(len(ACTIONS))


def test_env_defaults_to_player_observation_and_can_opt_into_privileged_mode() -> None:
    base_env = BrogueEnv(vector_env=FakeVectorEnv(include_semantic_tensors=True))
    base_observation, _ = base_env.reset()

    assert "glyphs" in base_observation
    assert "inventory_letters" in base_observation
    assert "inventory_strs" in base_observation
    assert "map_layers" not in base_observation
    assert base_env.observation_space.contains(base_observation)

    privileged_env = BrogueEnv(
        vector_env=FakeVectorEnv(include_semantic_tensors=True),
        observation_mode="privileged",
    )
    privileged_observation, _ = privileged_env.reset()

    assert "map_has_item" in privileged_observation
    assert "map_has_monster" in privileged_observation
    assert "inventory_present" in privileged_observation
    assert privileged_env.observation_space.contains(privileged_observation)


def test_env_can_attach_privileged_info_without_policy_observation() -> None:
    vector_env = FakeVectorEnv(include_semantic_tensors=True)
    env = BrogueEnv(vector_env=vector_env, include_privileged_info=True)

    observation, info = env.reset()

    privileged = cast(ObservationDict, info[BackendInfoKey.PRIVILEGED_OBSERVATION])
    assert "map_layers" not in observation
    assert "map_layers" in privileged
    assert "map_has_item" in privileged
    assert env.observation_space.contains(observation)


def test_env_can_use_full_action_profile() -> None:
    vector_env = FakeVectorEnv()
    env = BrogueEnv(vector_env=vector_env, actions="full")
    full_action_index = tuple(action.brogue_input() for action in FULL_ACTIONS).index(
        BrogueInput(key="+"),
    )

    env.reset()
    _, _, _, _, info = env.step(full_action_index)

    assert len(env.action_descriptions) == len(FULL_ACTIONS)
    assert info == {"key": "+"}
    assert vector_env.actions == [Action.keypress("+")]


def test_env_registers_with_gymnasium_make() -> None:
    make_env = gym.make  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    env = cast(
        Any,
        make_env(
            BROGUE_ENV_ID,
            vector_env=FakeVectorEnv(),
            disable_env_checker=True,
        ),
    )

    try:
        observation, info = env.reset(seed=11)

        assert info == {"seed": 11}
        assert env.observation_space.contains(observation)
    finally:
        env.close()


def test_env_default_vector_env_uses_process_runner_if_built() -> None:
    if not _default_library_path().exists():
        pytest.skip("run through uv so package sync builds the bridge")
    env = BrogueEnv()
    try:
        assert isinstance(env.vector_env, BrogueVectorEnv)
    finally:
        env.close()


def test_env_default_backend_reset_seed_is_sampler_seed_if_built() -> None:
    if not _default_library_path().exists():
        pytest.skip("run through uv so package sync builds the bridge")
    env = BrogueEnv()
    try:
        expected_seed = _sample_brogue_seeds(42, 1)[0]
        observation, info = env.reset(seed=42)

        assert info[BackendInfoKey.SEED] == expected_seed
        assert int(observation["program_state"][4]) == expected_seed
        assert expected_seed != 42
    finally:
        env.close()


def test_env_default_backend_reset_game_seed_override_if_built() -> None:
    if not _default_library_path().exists():
        pytest.skip("run through uv so package sync builds the bridge")
    env = BrogueEnv()
    try:
        observation, info = env.reset(seed=11, options={"game_seed": 42})

        assert info[BackendInfoKey.SEED] == 42
        assert int(observation["program_state"][4]) == 42
    finally:
        env.close()


def test_env_render_includes_ansi_colors_when_present() -> None:
    env = BrogueEnv(vector_env=FakeVectorEnv(), render_mode="ansi")

    observation, _ = env.reset()
    observation["chars"][0, 0] = ord("@")
    observation["colors_fg"][0, 0] = [255, 128, 0]
    observation["colors_bg"][0, 0] = [0, 0, 64]

    rendered = env.render()

    assert rendered is not None
    assert "\x1b[38;2;255;128;0;48;2;0;0;64m@" in rendered


def test_env_render_defaults_to_ascii_charset_for_unicode_glyphs() -> None:
    env = BrogueEnv(vector_env=FakeVectorEnv(), render_mode="ansi")

    observation, _ = env.reset()
    observation["glyphs"][0, 0] = 198
    observation["chars"][0, 0] = 0x25C7
    observation["glyphs"][0, 1] = 225
    observation["chars"][0, 1] = 0x2648

    rendered = env.render()

    assert rendered is not None
    assert rendered.splitlines()[0].startswith("%&")


def test_env_passes_gymnasium_checker() -> None:
    check_env(BrogueEnv(vector_env=FakeVectorEnv()), skip_render_check=True)
