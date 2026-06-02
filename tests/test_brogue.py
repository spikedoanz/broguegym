"""Tests for BrogueBackend."""

from __future__ import annotations

# pyright: reportPrivateUsage=false

from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from numpy.typing import NDArray

import broguegym.brogue as brogue_backend_module
from broguegym.actions import Action, ActionKind
from broguegym.brogue import (
    BackendErrorCode,
    BackendInfoKey,
    BackendReset,
    BackendUnavailableError,
    _bridge_library_name,
    _default_data_dir,
    _default_library_path,
)
from broguegym.brogue import BrogueBackend
from broguegym.spaces import SCREEN_COLS, SCREEN_ROWS

_REST = Action(kind=ActionKind.REST)
_EXPLORE = Action(kind=ActionKind.EXPLORE)


def _skip_if_no_library() -> None:
    if not _default_library_path().exists():
        pytest.skip("run through uv so package sync builds the bridge")


def _seeds(*seeds: int) -> NDArray[np.uint64]:
    return np.array(seeds, dtype=np.uint64)


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


def _backend_with_queued_workers(
    responses: list[list[tuple[Any, ...]]],
) -> tuple[BrogueBackend, list[_QueuedConnection]]:
    backend = BrogueBackend.__new__(BrogueBackend)
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
    backend = BrogueBackend.__new__(BrogueBackend)
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


def test_brogue_backend_reset_many_drains_peer_responses_after_worker_error() -> None:
    backend, connections = _backend_with_queued_workers(
        [
            [("backend_error", BackendErrorCode.BRIDGE_OPERATION_FAILED.value, "boom")],
            [("reset", {BackendInfoKey.SEED: 2})],
        ],
    )

    with pytest.raises(BackendUnavailableError, match="boom"):
        backend.reset_many(seed=_seeds(1, 2))

    assert connections[1].responses == []
    assert backend._states == [
        brogue_backend_module.BackendSessionState.CLOSED,
        brogue_backend_module.BackendSessionState.CLOSED,
    ]

    connections[0].responses.append(("reset", {BackendInfoKey.SEED: 1}))
    connections[1].responses.append(("reset", {BackendInfoKey.SEED: 99}))

    results = backend.reset_many(seed=_seeds(1, 99))

    assert [_info_seed(result) for result in results] == [1, 99]


def test_brogue_backend_reset_many_drains_sent_responses_after_send_error() -> None:
    backend, connections = _backend_with_queued_workers(
        [
            [("reset", {BackendInfoKey.SEED: 1})],
            [],
            [],
        ],
    )
    connections[1].fail_send = True

    with pytest.raises(BackendUnavailableError, match="not available"):
        backend.reset_many(seed=_seeds(1, 2, 3))

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

    results = backend.reset_many(seed=_seeds(10, 20, 30))

    assert [_info_seed(result) for result in results] == [10, 20, 30]


def test_brogue_backend_step_many_drains_peer_responses_after_worker_error() -> None:
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
        backend.step_many([_REST, _REST])

    assert connections[1].responses == []
    assert backend._states == [
        brogue_backend_module.BackendSessionState.CLOSED,
        brogue_backend_module.BackendSessionState.CLOSED,
    ]


def test_brogue_backend_step_many_drains_sent_responses_after_send_error() -> None:
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
        backend.step_many([_REST, _REST, _REST])

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

    results = backend.step_many([_REST, _REST, _REST])

    assert [result.reward for result in results] == [1.0, 2.0, 3.0]


def test_brogue_backend_rejects_zero_num_envs() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        BrogueBackend(0)


def test_brogue_backend_rejects_negative_num_envs() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        BrogueBackend(-3)


def test_brogue_backend_reports_missing_bridge_library(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    missing_library = tmp_path / "missing.so"
    monkeypatch.setattr(brogue_backend_module, "_default_library_path", lambda: missing_library)

    with pytest.raises(BackendUnavailableError, match="package sync builds") as exc_info:
        BrogueBackend(1)

    assert exc_info.value.code is BackendErrorCode.BRIDGE_LIBRARY_MISSING


def test_default_runtime_paths_prefer_packaged_bridge(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package_root = tmp_path / "site-packages" / "broguegym"
    packaged_bin = package_root / "_native" / "bin"
    packaged_bin.mkdir(parents=True)
    bridge_library = packaged_bin / _bridge_library_name()
    bridge_library.touch()
    source_root = tmp_path / "source"
    monkeypatch.setattr(brogue_backend_module, "_package_root", lambda: package_root)
    monkeypatch.setattr(brogue_backend_module, "_repo_root", lambda: source_root)

    assert _default_data_dir() == packaged_bin
    assert _default_library_path() == bridge_library


def test_default_runtime_paths_fall_back_to_source_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package_root = tmp_path / "site-packages" / "broguegym"
    source_root = tmp_path / "source"
    monkeypatch.setattr(brogue_backend_module, "_package_root", lambda: package_root)
    monkeypatch.setattr(brogue_backend_module, "_repo_root", lambda: source_root)

    assert _default_data_dir() == source_root / "BrogueCE" / "bin"
    assert _default_library_path() == source_root / "BrogueCE" / "bin" / _bridge_library_name()


def test_brogue_backend_reports_missing_data_dir_if_built(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _skip_if_no_library()
    missing_data_dir = tmp_path / "no-such-dir"
    monkeypatch.setattr(brogue_backend_module, "_default_data_dir", lambda: missing_data_dir)

    with pytest.raises(BackendUnavailableError, match="data directory") as exc_info:
        BrogueBackend(1)

    assert exc_info.value.code is BackendErrorCode.BRIDGE_DATA_DIR_INVALID


def test_brogue_backend_step_many_before_reset_raises() -> None:
    _skip_if_no_library()
    with BrogueBackend(1) as backend:
        with pytest.raises(BackendUnavailableError, match="call reset") as exc_info:
            backend.step_many([_REST])

    assert exc_info.value.code is BackendErrorCode.BRIDGE_NOT_RUNNING


def test_brogue_backend_step_many_rejects_wrong_action_count() -> None:
    _skip_if_no_library()
    with BrogueBackend(2) as backend:
        backend.reset_many(seed=_seeds(1, 2))
        with pytest.raises(ValueError, match="expected 2 actions"):
            backend.step_many([_REST])


def test_brogue_backend_check_env_id_out_of_range() -> None:
    _skip_if_no_library()
    with BrogueBackend(2) as backend:
        with pytest.raises(IndexError, match="out of range"):
            backend.reset(2, seed=_seeds(1))


def test_brogue_backend_check_env_id_negative() -> None:
    _skip_if_no_library()
    with BrogueBackend(2) as backend:
        with pytest.raises(IndexError, match="out of range"):
            backend.reset(-1, seed=_seeds(1))


def test_brogue_backend_manual_scalar_seed_and_step() -> None:
    _skip_if_no_library()
    with BrogueBackend(1) as backend:
        reset = backend.reset(seed=_seeds(42))
        step = backend.step(_EXPLORE)

        assert _info_seed(reset) == 42
        assert step.info[BackendInfoKey.KEY] == "x"
        assert step.reward == 0.0
        assert step.terminated is False


def test_brogue_backend_returned_observations_survive_close() -> None:
    _skip_if_no_library()
    with BrogueBackend(1) as backend:
        reset = backend.reset(seed=_seeds(42))
        step = backend.step(_EXPLORE)

    assert reset.observation["glyphs"].shape == (SCREEN_ROWS, SCREEN_COLS)
    assert step.observation["glyphs"].shape == (SCREEN_ROWS, SCREEN_COLS)
    assert isinstance(int(reset.observation["glyphs"][0, 0]), int)
    assert isinstance(int(step.observation["glyphs"][0, 0]), int)


def test_brogue_backend_does_not_expose_shared_observation_views() -> None:
    assert not hasattr(BrogueBackend, "shared_obs")


def test_brogue_backend_default_seed_is_deterministic_sampler() -> None:
    _skip_if_no_library()
    with BrogueBackend(2) as backend:
        first = backend.reset_many()
        second = backend.reset_many(seed=None)

        assert [_info_seed(result) for result in first] == [
            _info_seed(result) for result in second
        ]


def test_brogue_backend_sampler_seed_samples_brogue_seed_range() -> None:
    _skip_if_no_library()
    with BrogueBackend(4) as backend:
        resets = backend.reset_many(seed=123)

        seeds = [_info_seed(result) for result in resets]
        assert len(set(seeds)) == 4
        assert seeds != [123, 124, 125, 126]
        assert all(1 <= seed <= 2**64 - 1 for seed in seeds)


def test_brogue_backend_manual_seed_overrides_require_ndarray() -> None:
    _skip_if_no_library()
    with BrogueBackend(2) as backend:
        with pytest.raises(TypeError, match="integer ndarray"):
            backend.reset_many(seed=[1, 2])  # pyright: ignore[reportArgumentType]


def test_brogue_backend_manual_seed_overrides_reject_wrong_shape() -> None:
    _skip_if_no_library()
    with BrogueBackend(2) as backend:
        with pytest.raises(ValueError, match=r"shape \(2,\)"):
            backend.reset_many(seed=_seeds(1, 2, 3))


def test_brogue_backend_manual_seed_overrides_reject_zero() -> None:
    _skip_if_no_library()
    with BrogueBackend(1) as backend:
        with pytest.raises(ValueError, match=r"\[1, 2\*\*64 - 1\]"):
            backend.reset(seed=_seeds(0))


def test_brogue_backend_parallel_reset_and_step_many() -> None:
    _skip_if_no_library()
    with BrogueBackend(4) as backend:
        resets = backend.reset_many(seed=_seeds(1, 2, 3, 4))
        steps = backend.step_many([_EXPLORE] * 4)

        assert [_info_seed(result) for result in resets] == [1, 2, 3, 4]
        assert len(steps) == 4
        for step in steps:
            assert step.info[BackendInfoKey.KEY] == "x"
            assert step.observation["glyphs"].shape == (SCREEN_ROWS, SCREEN_COLS)


def test_brogue_backend_same_manual_seed_gives_same_observation() -> None:
    _skip_if_no_library()
    with BrogueBackend(2) as backend:
        resets = backend.reset_many(seed=_seeds(99, 99))

        np.testing.assert_array_equal(
            resets[0].observation["glyphs"],
            resets[1].observation["glyphs"],
        )


def test_brogue_backend_different_manual_seeds_give_different_observations() -> None:
    _skip_if_no_library()
    with BrogueBackend(2) as backend:
        resets = backend.reset_many(seed=_seeds(1, 2))

        assert not np.array_equal(
            resets[0].observation["glyphs"],
            resets[1].observation["glyphs"],
        )
