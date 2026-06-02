"""Tests for BrogueVectorEnv."""

from __future__ import annotations

# pyright: reportPrivateUsage=false

import ctypes
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

import broguegym.brogue as brogue_backend_module
from broguegym.actions import Action, ActionKind
from broguegym.brogue import (
    BackendErrorCode,
    BackendInfoKey,
    BackendReset,
    BackendUnavailableError,
    _default_library_path,
)
from broguegym.brogue import BrogueVectorEnv
from broguegym.spaces import SCREEN_COLS, SCREEN_ROWS

_REST = Action(kind=ActionKind.REST)
_EXPLORE = Action(kind=ActionKind.EXPLORE)


def _skip_if_no_library() -> None:
    if not _default_library_path().exists():
        pytest.skip("run through uv so package sync builds the bridge")


def _game_seeds(*seeds: int) -> list[int]:
    return list(seeds)


def _info_seed(result: BackendReset) -> int:
    return cast(int, result.info[BackendInfoKey.SEED])


class _UnresponsiveConnection:
    def __init__(self) -> None:
        self.sent: list[tuple[str, ...]] = []
        self.closed = False
        self.polled_timeout: float | None = None
        self.recv_called = False

    def send(self, message: tuple[str, ...]) -> None:
        self.sent.append(message)

    def poll(self, timeout: float) -> bool:
        self.polled_timeout = timeout
        return False

    def recv(self) -> tuple[str, ...]:
        self.recv_called = True
        raise AssertionError("close() must not call recv() without poll() readiness")

    def close(self) -> None:
        self.closed = True


class _IgnoringProcess:
    def __init__(self) -> None:
        self.terminated = False
        self.killed = False
        self.join_timeouts: list[float | None] = []

    def is_alive(self) -> bool:
        return not self.killed

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.killed = True

    def join(self, timeout: float | None = None) -> None:
        self.join_timeouts.append(timeout)


class _QueuedConnection:
    def __init__(
        self,
        responses: list[tuple[Any, ...]],
        *,
        fail_send: bool = False,
    ) -> None:
        self.responses = responses
        self.fail_send = fail_send
        self.sent: list[tuple[object, ...]] = []

    def send(self, message: tuple[object, ...]) -> None:
        if self.fail_send:
            raise BrokenPipeError
        self.sent.append(message)

    def recv(self) -> tuple[Any, ...]:
        if not self.responses:
            raise AssertionError("expected no additional recv() calls")
        return self.responses.pop(0)


class _ClosingConnection:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _StartFailingProcess:
    def start(self) -> None:
        raise OSError("spawn failed")


class _StartFailingContext:
    def __init__(self) -> None:
        self.parent_connection = _ClosingConnection()
        self.child_connection = _ClosingConnection()

    def Pipe(self) -> tuple[_ClosingConnection, _ClosingConnection]:  # noqa: N802
        return self.parent_connection, self.child_connection

    def Process(self, **kwargs: object) -> _StartFailingProcess:  # noqa: N802
        _ = kwargs
        return _StartFailingProcess()


def _backend_with_queued_workers(
    responses: list[list[tuple[Any, ...]]],
) -> tuple[BrogueVectorEnv, list[_QueuedConnection]]:
    backend = BrogueVectorEnv.__new__(BrogueVectorEnv)
    connections = [_QueuedConnection(response_queue) for response_queue in responses]
    backend.num_envs = len(connections)
    backend._workers = [
        brogue_backend_module._ProcessWorker(
            process=object(),
            connection=cast(Connection, connection),
        )
        for connection in connections
    ]
    backend._shared_memory = None
    backend._shared_obs = {}
    backend._obs_views = [
        {"glyphs": np.array([env_id], dtype=np.int16)}
        for env_id in range(len(connections))
    ]
    backend._states = [
        brogue_backend_module.BackendSessionState.CLOSED
        for _ in connections
    ]
    return backend, connections


def test_brogue_backend_close_terminates_and_kills_unresponsive_worker() -> None:
    backend = BrogueVectorEnv.__new__(BrogueVectorEnv)
    connection = _UnresponsiveConnection()
    process = _IgnoringProcess()
    backend.num_envs = 1
    backend._workers = [
        brogue_backend_module._ProcessWorker(
            process=process,
            connection=cast(Connection, connection),
        ),
    ]
    backend._shared_memory = None
    backend._shared_obs = {}
    backend._obs_views = []
    backend._states = [brogue_backend_module.BackendSessionState.RUNNING]

    backend.close()

    assert connection.sent == [("close",)]
    assert connection.polled_timeout == brogue_backend_module._WORKER_CLOSE_TIMEOUT_SECONDS
    assert connection.recv_called is False
    assert connection.closed is True
    assert process.terminated is True
    assert process.killed is True
    assert process.join_timeouts == [
        brogue_backend_module._WORKER_TERMINATE_TIMEOUT_SECONDS,
        brogue_backend_module._WORKER_TERMINATE_TIMEOUT_SECONDS,
        brogue_backend_module._WORKER_TERMINATE_TIMEOUT_SECONDS,
    ]
    assert backend._workers == []
    assert backend._states == [brogue_backend_module.BackendSessionState.CLOSED]


def test_brogue_backend_init_workers_closes_pipe_if_process_start_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _StartFailingContext()
    library_path = tmp_path / "libbruhogue_brogue.so"
    library_path.write_bytes(b"")
    backend = BrogueVectorEnv.__new__(BrogueVectorEnv)
    backend.num_envs = 1
    backend.library_path = library_path
    backend.data_dir = tmp_path
    backend._states = [brogue_backend_module.BackendSessionState.CLOSED]
    backend._workers = []
    backend._shared_memory = None
    backend._obs_size = ctypes.sizeof(brogue_backend_module._CObservation)
    backend._shared_obs = {}
    backend._obs_views = []

    def fake_get_context(method: str) -> _StartFailingContext:
        _ = method
        return context

    monkeypatch.setattr(brogue_backend_module.mp, "get_context", fake_get_context)

    with pytest.raises(OSError, match="spawn failed"):
        backend._init_workers()

    assert context.parent_connection.closed is True
    assert context.child_connection.closed is True
    assert backend._workers == []
    assert backend._shared_memory is None


def test_brogue_backend_reset_drains_peer_responses_after_worker_error() -> None:
    backend, connections = _backend_with_queued_workers(
        [
            [("backend_error", BackendErrorCode.BRIDGE_OPERATION_FAILED.value, "boom")],
            [("reset", {BackendInfoKey.SEED: 2})],
        ],
    )

    with pytest.raises(BackendUnavailableError, match="boom"):
        backend.reset(game_seed=_game_seeds(1, 2))

    assert connections[1].responses == []
    assert backend._states == [
        brogue_backend_module.BackendSessionState.CLOSED,
        brogue_backend_module.BackendSessionState.CLOSED,
    ]

    connections[0].responses.append(("reset", {BackendInfoKey.SEED: 1}))
    connections[1].responses.append(("reset", {BackendInfoKey.SEED: 99}))

    results = backend.reset(game_seed=_game_seeds(1, 99))

    assert [_info_seed(result) for result in results] == [1, 99]


def test_brogue_backend_reset_drains_sent_responses_after_send_error() -> None:
    backend, connections = _backend_with_queued_workers(
        [
            [("reset", {BackendInfoKey.SEED: 1})],
            [],
            [],
        ],
    )
    connections[1].fail_send = True

    with pytest.raises(BackendUnavailableError, match="not available"):
        backend.reset(game_seed=_game_seeds(1, 2, 3))

    assert connections[0].responses == []
    assert connections[2].sent == []
    assert backend._states == [
        brogue_backend_module.BackendSessionState.CLOSED,
        brogue_backend_module.BackendSessionState.CLOSED,
        brogue_backend_module.BackendSessionState.CLOSED,
    ]

    connections[1].fail_send = False
    connections[0].responses.append(("reset", {BackendInfoKey.SEED: 10}))
    connections[1].responses.append(("reset", {BackendInfoKey.SEED: 20}))
    connections[2].responses.append(("reset", {BackendInfoKey.SEED: 30}))

    results = backend.reset(game_seed=_game_seeds(10, 20, 30))

    assert [_info_seed(result) for result in results] == [10, 20, 30]


def test_brogue_backend_step_drains_peer_responses_after_worker_error() -> None:
    backend, connections = _backend_with_queued_workers(
        [
            [("backend_error", BackendErrorCode.BRIDGE_OPERATION_FAILED.value, "boom")],
            [("step", 0.0, False, {})],
        ],
    )
    backend._states = [
        brogue_backend_module.BackendSessionState.RUNNING,
        brogue_backend_module.BackendSessionState.RUNNING,
    ]

    with pytest.raises(BackendUnavailableError, match="boom"):
        backend.step([_REST, _REST])

    assert connections[1].responses == []
    assert backend._states == [
        brogue_backend_module.BackendSessionState.CLOSED,
        brogue_backend_module.BackendSessionState.CLOSED,
    ]


def test_brogue_backend_step_drains_sent_responses_after_send_error() -> None:
    backend, connections = _backend_with_queued_workers(
        [
            [("step", 10.0, False, {})],
            [],
            [],
        ],
    )
    connections[1].fail_send = True
    backend._states = [
        brogue_backend_module.BackendSessionState.RUNNING,
        brogue_backend_module.BackendSessionState.RUNNING,
        brogue_backend_module.BackendSessionState.RUNNING,
    ]

    with pytest.raises(BackendUnavailableError, match="not available"):
        backend.step([_REST, _REST, _REST])

    assert connections[0].responses == []
    assert connections[2].sent == []
    assert backend._states == [
        brogue_backend_module.BackendSessionState.CLOSED,
        brogue_backend_module.BackendSessionState.CLOSED,
        brogue_backend_module.BackendSessionState.CLOSED,
    ]

    connections[1].fail_send = False
    connections[0].responses.append(("step", 1.0, False, {}))
    connections[1].responses.append(("step", 2.0, False, {}))
    connections[2].responses.append(("step", 3.0, False, {}))
    backend._states = [
        brogue_backend_module.BackendSessionState.RUNNING,
        brogue_backend_module.BackendSessionState.RUNNING,
        brogue_backend_module.BackendSessionState.RUNNING,
    ]

    results = backend.step([_REST, _REST, _REST])

    assert [result.reward for result in results] == [1.0, 2.0, 3.0]


def test_brogue_backend_rejects_zero_num_envs() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        BrogueVectorEnv(0)


def test_brogue_backend_rejects_negative_num_envs() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        BrogueVectorEnv(-3)


def test_brogue_backend_reports_missing_bridge_library(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    missing_library = tmp_path / "missing.so"
    monkeypatch.setattr(brogue_backend_module, "_default_library_path", lambda: missing_library)

    with pytest.raises(BackendUnavailableError, match="package sync builds") as exc_info:
        BrogueVectorEnv(1)

    assert exc_info.value.code is BackendErrorCode.BRIDGE_LIBRARY_MISSING


def test_brogue_backend_reports_missing_data_dir_if_built(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _skip_if_no_library()
    missing_data_dir = tmp_path / "no-such-dir"
    monkeypatch.setattr(brogue_backend_module, "_default_data_dir", lambda: missing_data_dir)

    with pytest.raises(BackendUnavailableError, match="data directory") as exc_info:
        BrogueVectorEnv(1)

    assert exc_info.value.code is BackendErrorCode.BRIDGE_DATA_DIR_INVALID


def test_brogue_backend_step_before_reset_raises() -> None:
    _skip_if_no_library()
    with BrogueVectorEnv(1) as backend:
        with pytest.raises(BackendUnavailableError, match="call reset") as exc_info:
            backend.step([_REST])

    assert exc_info.value.code is BackendErrorCode.BRIDGE_NOT_RUNNING


def test_brogue_backend_step_rejects_wrong_action_count() -> None:
    _skip_if_no_library()
    with BrogueVectorEnv(2) as backend:
        backend.reset(game_seed=_game_seeds(1, 2))
        with pytest.raises(ValueError, match="expected 2 actions"):
            backend.step([_REST])


def test_brogue_backend_scalar_seed_uses_sampler_and_step() -> None:
    _skip_if_no_library()
    with BrogueVectorEnv(1) as backend:
        expected_seed = brogue_backend_module._sample_brogue_seeds(42, 1)[0]
        reset = backend.reset(seed=42)[0]
        step = backend.step([_EXPLORE])[0]

        assert _info_seed(reset) == expected_seed
        assert expected_seed != 42
        assert step.info[BackendInfoKey.KEY] == "x"
        assert step.reward == 0.0
        assert step.terminated is False


def test_brogue_backend_game_seed_override_accepts_single_int() -> None:
    _skip_if_no_library()
    with BrogueVectorEnv(1) as backend:
        reset = backend.reset(game_seed=99)[0]

        assert _info_seed(reset) == 99


def test_brogue_backend_game_seed_int_broadcasts() -> None:
    _skip_if_no_library()
    with BrogueVectorEnv(2) as backend:
        resets = backend.reset(seed=123, game_seed=99)

        assert [_info_seed(result) for result in resets] == [99, 99]


def test_brogue_backend_returned_observations_survive_close() -> None:
    _skip_if_no_library()
    with BrogueVectorEnv(1) as backend:
        reset = backend.reset(game_seed=42)[0]
        step = backend.step([_EXPLORE])[0]

    assert reset.observation["glyphs"].shape == (SCREEN_ROWS, SCREEN_COLS)
    assert step.observation["glyphs"].shape == (SCREEN_ROWS, SCREEN_COLS)
    assert isinstance(int(reset.observation["glyphs"][0, 0]), int)
    assert isinstance(int(step.observation["glyphs"][0, 0]), int)


def test_brogue_backend_does_not_expose_shared_observation_views() -> None:
    assert not hasattr(BrogueVectorEnv, "shared_obs")


def test_brogue_backend_default_seed_is_deterministic_sampler() -> None:
    _skip_if_no_library()
    with BrogueVectorEnv(2) as backend:
        first = backend.reset()
        second = backend.reset(seed=None)

        assert [_info_seed(result) for result in first] == [
            _info_seed(result) for result in second
        ]


def test_brogue_backend_sampler_seed_samples_brogue_seed_range() -> None:
    _skip_if_no_library()
    with BrogueVectorEnv(4) as backend:
        resets = backend.reset(seed=123)

        seeds = [_info_seed(result) for result in resets]
        assert len(set(seeds)) == 4
        assert seeds != [123, 124, 125, 126]
        assert all(1 <= seed <= 2**64 - 1 for seed in seeds)


def test_brogue_backend_seed_rejects_manual_seed_list() -> None:
    _skip_if_no_library()
    with BrogueVectorEnv(2) as backend:
        with pytest.raises(TypeError, match="game_seed"):
            backend.reset(seed=[1, 2])  # pyright: ignore[reportArgumentType]


def test_brogue_backend_game_seed_list_requires_one_seed_per_env() -> None:
    _skip_if_no_library()
    with BrogueVectorEnv(2) as backend:
        with pytest.raises(ValueError, match="expected 2 game_seed overrides"):
            backend.reset(game_seed=_game_seeds(1, 2, 3))


def test_brogue_backend_game_seed_rejects_zero() -> None:
    _skip_if_no_library()
    with BrogueVectorEnv(1) as backend:
        with pytest.raises(ValueError, match=r"\[1, 2\*\*64 - 1\]"):
            backend.reset(game_seed=0)


def test_brogue_backend_parallel_reset_and_step() -> None:
    _skip_if_no_library()
    with BrogueVectorEnv(4) as backend:
        resets = backend.reset(game_seed=_game_seeds(1, 2, 3, 4))
        steps = backend.step([_EXPLORE] * 4)

        assert [_info_seed(result) for result in resets] == [1, 2, 3, 4]
        assert len(steps) == 4
        for step in steps:
            assert step.info[BackendInfoKey.KEY] == "x"
            assert step.observation["glyphs"].shape == (SCREEN_ROWS, SCREEN_COLS)


def test_brogue_backend_same_manual_seed_gives_same_observation() -> None:
    _skip_if_no_library()
    with BrogueVectorEnv(2) as backend:
        resets = backend.reset(game_seed=99)

        np.testing.assert_array_equal(
            resets[0].observation["glyphs"],
            resets[1].observation["glyphs"],
        )


def test_brogue_backend_different_manual_seeds_give_different_observations() -> None:
    _skip_if_no_library()
    with BrogueVectorEnv(2) as backend:
        resets = backend.reset(game_seed=_game_seeds(1, 2))

        assert not np.array_equal(
            resets[0].observation["glyphs"],
            resets[1].observation["glyphs"],
        )
