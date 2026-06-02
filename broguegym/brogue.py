"""Brogue bridge integration and process-isolated backend."""

from __future__ import annotations

# pyright: reportPrivateUsage=false

import ctypes
import multiprocessing as mp
import secrets
import sys
import traceback
from collections.abc import Sequence
from enum import IntEnum, StrEnum
from pathlib import Path
from typing import Any, cast

import msgspec
import numpy as np
from multiprocessing.connection import Connection
from multiprocessing.shared_memory import SharedMemory
from numpy.typing import NDArray

from broguegym.actions import Action
from broguegym.snapshot import BrogueSnapshot

__all__ = [
    "BackendErrorCode",
    "BackendInfoKey",
    "BackendReset",
    "BackendStep",
    "BackendUnavailableError",
    "BrogueBackend",
    "ObservationDict",
]

type ObservationDict = dict[str, NDArray[np.generic]]
type InfoDict = dict[str, object]


class BackendInfoKey(StrEnum):
    """Known Gym `info` metadata keys emitted by bundled backends."""

    CONTROL = "control"
    ERROR = "error"
    ERROR_CODE = "error_code"
    KEY = "key"
    PRIVILEGED_OBSERVATION = "privileged_observation"
    SEED = "seed"
    SHIFT = "shift"


class BackendErrorCode(StrEnum):
    """Stable reason codes for backend failures and rejected actions."""

    ACTION_UNENCODABLE = "action_unencodable"
    BRIDGE_ABI_MISMATCH = "bridge_abi_mismatch"
    BRIDGE_DATA_DIR_INVALID = "bridge_data_dir_invalid"
    BRIDGE_LIBRARY_LOAD_FAILED = "bridge_library_load_failed"
    BRIDGE_LIBRARY_MISSING = "bridge_library_missing"
    BRIDGE_NOT_RUNNING = "bridge_not_running"
    BRIDGE_OPERATION_FAILED = "bridge_operation_failed"
    KEY_INVALID = "key_invalid"
    KEY_UNSUPPORTED = "key_unsupported"
    RESTORE_UNSUPPORTED = "restore_unsupported"
    SNAPSHOT_UNSUPPORTED = "snapshot_unsupported"


class BackendSessionState(StrEnum):
    """Lifecycle states for a singleton backend session."""

    CLOSED = "closed"
    RUNNING = "running"


def _empty_info() -> InfoDict:
    return {}


class BackendReset(msgspec.Struct, frozen=True, kw_only=True, forbid_unknown_fields=True):
    """Result returned by a backend reset or restore."""

    observation: ObservationDict
    info: InfoDict = msgspec.field(default_factory=_empty_info)


class BackendStep(msgspec.Struct, frozen=True, kw_only=True, forbid_unknown_fields=True):
    """Result returned by one backend action."""

    observation: ObservationDict
    reward: float = 0.0
    terminated: bool = False
    info: InfoDict = msgspec.field(default_factory=_empty_info)


class BackendUnavailableError(RuntimeError):
    """Raised when Brogue cannot be started or advanced."""

    def __init__(self, code: BackendErrorCode, message: str) -> None:
        super().__init__(message)
        self.code = code


class _InProcessBrogue:
    """Drive one Brogue game through the in-process C bridge shared library.

    This is private to the worker process. Use one instance per process or
    serialize access externally.
    """

    def __init__(self) -> None:
        self.library_path = _default_library_path()
        self.data_dir = _default_data_dir()
        self._library: Any | None = None
        self._state = BackendSessionState.CLOSED

    def reset(
        self,
        *,
        seed: int | None = None,
    ) -> BackendReset:
        """Start a new game through the C bridge."""

        seed_value = _bridge_seed(seed)
        self._validate_data_dir()
        if self._state is BackendSessionState.RUNNING:
            self.close()
        observation = _CObservation()
        library = self._load_library()
        library.brh_set_data_dir(str(self.data_dir).encode("utf-8"))
        rc = library.brh_reset(seed_value, ctypes.byref(observation))
        try:
            self._raise_if_failed(rc, "reset")
        except BackendUnavailableError:
            self._state = BackendSessionState.CLOSED
            raise
        self._state = BackendSessionState.RUNNING
        observation_dict = _observation_from_c(observation)
        actual_seed = int(observation_dict["program_state"][_PROGRAM_SEED_INDEX])
        return BackendReset(
            observation=observation_dict,
            info={
                BackendInfoKey.SEED: actual_seed,
            },
        )

    def step(self, action: Action) -> BackendStep:
        """Apply one action through the C bridge."""

        try:
            brogue_input = action.brogue_input()
        except ValueError as exc:
            msg = f"in-process Brogue bridge cannot encode action {action.kind.value}"
            raise BackendUnavailableError(BackendErrorCode.ACTION_UNENCODABLE, msg) from exc

        if self._state is not BackendSessionState.RUNNING:
            msg = "Brogue bridge is not running; call reset() first"
            raise BackendUnavailableError(BackendErrorCode.BRIDGE_NOT_RUNNING, msg)

        key = _bridge_key(brogue_input.key)
        observation = _CObservation()
        library = self._require_library()
        rc = library.brh_step(
            key,
            int(brogue_input.control),
            int(brogue_input.shift),
            ctypes.byref(observation),
        )
        if rc < 0:
            self._raise_if_failed(rc, "step")
        status = _bridge_step_status(rc)
        observation_dict = _observation_from_c(observation)
        terminated = bool(observation_dict["program_state"][_PROGRAM_TERMINATED_INDEX])
        info: InfoDict = {
            BackendInfoKey.KEY: brogue_input.key,
            BackendInfoKey.CONTROL: brogue_input.control,
            BackendInfoKey.SHIFT: brogue_input.shift,
        }
        if status is _BridgeStepStatus.INVALID_KEY:
            info[BackendInfoKey.ERROR_CODE] = BackendErrorCode.KEY_INVALID
            info[BackendInfoKey.ERROR] = (
                f"Brogue rejected key {brogue_input.key!r} in the current state"
            )
        return BackendStep(
            observation=observation_dict,
            reward=0.0,
            terminated=terminated,
            info=info,
        )

    def snapshot(self) -> BrogueSnapshot:
        """Capture a complete game-state snapshot through the C bridge."""

        msg = "in-process Brogue bridge does not expose full-state snapshots yet"
        raise BackendUnavailableError(BackendErrorCode.SNAPSHOT_UNSUPPORTED, msg)

    def restore(self, snapshot: BrogueSnapshot) -> BackendReset:
        """Restore a game-state snapshot through the C bridge."""

        _ = snapshot
        msg = "in-process Brogue bridge does not expose full-state restore yet"
        raise BackendUnavailableError(BackendErrorCode.RESTORE_UNSUPPORTED, msg)

    def close(self) -> None:
        """Stop the current Brogue session, if one is running."""

        if self._state is BackendSessionState.RUNNING:
            self._require_library().brh_close()
            self._state = BackendSessionState.CLOSED

    def __enter__(self) -> _InProcessBrogue:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def _load_library(self) -> Any:
        if self._library is not None:
            return self._library
        if not self.library_path.is_file():
            msg = (
                f"Brogue bridge library does not exist: {self.library_path}. "
                "Run through `uv run` from the project root so package sync builds it."
            )
            raise BackendUnavailableError(BackendErrorCode.BRIDGE_LIBRARY_MISSING, msg)
        try:
            library = ctypes.CDLL(str(self.library_path))
        except OSError as exc:
            msg = f"failed to load Brogue bridge library {self.library_path}: {exc}"
            raise BackendUnavailableError(BackendErrorCode.BRIDGE_LIBRARY_LOAD_FAILED, msg) from exc

        try:
            _configure_library(library)
            _validate_bridge_abi(library)
        except AttributeError as exc:
            msg = f"Brogue bridge ABI mismatch: missing ABI symbol ({exc})"
            raise BackendUnavailableError(BackendErrorCode.BRIDGE_ABI_MISMATCH, msg) from exc
        self._library = library
        return library

    def _require_library(self) -> Any:
        if self._library is None:
            msg = "Brogue bridge library is not loaded despite a running session"
            raise RuntimeError(msg)
        return self._library

    def _validate_data_dir(self) -> None:
        if not self.data_dir.is_dir():
            msg = f"Brogue data directory does not exist: {self.data_dir}"
            raise BackendUnavailableError(BackendErrorCode.BRIDGE_DATA_DIR_INVALID, msg)

    def _raise_if_failed(self, rc: int, operation: str) -> None:
        if rc == 0:
            return
        library = self._require_library()
        raw_error = cast(bytes | None, library.brh_last_error())
        detail = raw_error.decode("utf-8", errors="replace") if raw_error else "unknown error"
        msg = f"Brogue bridge {operation} failed: {detail}"
        raise BackendUnavailableError(BackendErrorCode.BRIDGE_OPERATION_FAILED, msg)


_BRIDGE_ABI_VERSION = 7
_SCREEN_COLS = 100
_SCREEN_ROWS = 34
_MAP_COLS = 79
_MAP_ROWS = 29
_TERRAIN_LAYERS = 4
_OBS_CELLS = _SCREEN_COLS * _SCREEN_ROWS
_MAP_CELLS = _MAP_COLS * _MAP_ROWS
_MAP_LAYER_CELLS = _MAP_CELLS * _TERRAIN_LAYERS
_COLOR_CELLS = _OBS_CELLS * 3
_MAP_COLOR_CELLS = _MAP_CELLS * 3
_BLSTATS_SIZE = 21
_MESSAGE_SIZE = 256
_PROGRAM_STATE_SIZE = 8
_INVENTORY_SIZE = 26
_INVENTORY_STR_LENGTH = 80
_INVENTORY_STR_CELLS = _INVENTORY_SIZE * _INVENTORY_STR_LENGTH
_PROGRAM_TERMINATED_INDEX = 1
_PROGRAM_SEED_INDEX = 4
_GYM_ZERO_BRIDGE_SEED = 0x9E3779B97F4A7C15
_PACKAGE_ROOT = Path(__file__).resolve().parent
_PACKAGED_DATA_DIR = _PACKAGE_ROOT / "_native" / "bin"


class _BridgeStepStatus(IntEnum):
    OK = 0
    INVALID_KEY = 1


class _CObservation(ctypes.Structure):
    _fields_ = [
        ("glyphs", ctypes.c_int16 * _OBS_CELLS),
        ("chars", ctypes.c_uint32 * _OBS_CELLS),
        ("colors_fg", ctypes.c_uint8 * _COLOR_CELLS),
        ("colors_bg", ctypes.c_uint8 * _COLOR_CELLS),
        ("specials", ctypes.c_uint8 * _OBS_CELLS),
        ("map_layers", ctypes.c_uint16 * _MAP_LAYER_CELLS),
        ("map_flags", ctypes.c_uint64 * _MAP_CELLS),
        ("map_volume", ctypes.c_uint16 * _MAP_CELLS),
        ("map_machine", ctypes.c_uint8 * _MAP_CELLS),
        ("map_light", ctypes.c_int16 * _MAP_COLOR_CELLS),
        ("map_has_item", ctypes.c_uint8 * _MAP_CELLS),
        ("map_item_category", ctypes.c_uint16 * _MAP_CELLS),
        ("map_item_kind", ctypes.c_int16 * _MAP_CELLS),
        ("map_item_quantity", ctypes.c_int16 * _MAP_CELLS),
        ("map_item_flags", ctypes.c_uint64 * _MAP_CELLS),
        ("map_has_monster", ctypes.c_uint8 * _MAP_CELLS),
        ("map_monster_kind", ctypes.c_int16 * _MAP_CELLS),
        ("map_monster_hp", ctypes.c_int16 * _MAP_CELLS),
        ("map_monster_state", ctypes.c_int16 * _MAP_CELLS),
        ("inventory_present", ctypes.c_uint8 * _INVENTORY_SIZE),
        ("inventory_letters", ctypes.c_uint8 * _INVENTORY_SIZE),
        ("inventory_strs", ctypes.c_uint8 * _INVENTORY_STR_CELLS),
        ("inventory_category", ctypes.c_uint16 * _INVENTORY_SIZE),
        ("inventory_kind", ctypes.c_int16 * _INVENTORY_SIZE),
        ("inventory_quantity", ctypes.c_int16 * _INVENTORY_SIZE),
        ("inventory_flags", ctypes.c_uint64 * _INVENTORY_SIZE),
        ("inventory_enchant1", ctypes.c_int16 * _INVENTORY_SIZE),
        ("inventory_enchant2", ctypes.c_int16 * _INVENTORY_SIZE),
        ("inventory_charges", ctypes.c_int16 * _INVENTORY_SIZE),
        ("blstats", ctypes.c_int64 * _BLSTATS_SIZE),
        ("message", ctypes.c_uint8 * _MESSAGE_SIZE),
        ("program_state", ctypes.c_uint64 * _PROGRAM_STATE_SIZE),
    ]


class _ObservationField(msgspec.Struct, frozen=True, kw_only=True, forbid_unknown_fields=True):
    name: str
    dtype: Any
    shape: tuple[int, ...]


# Field specification for building zero-copy strided numpy views.
# Used by process-backed BrogueBackend to alias the raw C observation buffer directly.
_OBS_FIELD_SPECS: tuple[_ObservationField, ...] = (
    _ObservationField(name="glyphs", dtype=np.int16, shape=(_SCREEN_ROWS, _SCREEN_COLS)),
    _ObservationField(name="chars", dtype=np.uint32, shape=(_SCREEN_ROWS, _SCREEN_COLS)),
    _ObservationField(name="colors_fg", dtype=np.uint8, shape=(_SCREEN_ROWS, _SCREEN_COLS, 3)),
    _ObservationField(name="colors_bg", dtype=np.uint8, shape=(_SCREEN_ROWS, _SCREEN_COLS, 3)),
    _ObservationField(name="specials", dtype=np.uint8, shape=(_SCREEN_ROWS, _SCREEN_COLS)),
    _ObservationField(name="map_layers", dtype=np.uint16, shape=(_MAP_ROWS, _MAP_COLS, _TERRAIN_LAYERS)),
    _ObservationField(name="map_flags", dtype=np.uint64, shape=(_MAP_ROWS, _MAP_COLS)),
    _ObservationField(name="map_volume", dtype=np.uint16, shape=(_MAP_ROWS, _MAP_COLS)),
    _ObservationField(name="map_machine", dtype=np.uint8, shape=(_MAP_ROWS, _MAP_COLS)),
    _ObservationField(name="map_light", dtype=np.int16, shape=(_MAP_ROWS, _MAP_COLS, 3)),
    _ObservationField(name="map_has_item", dtype=np.uint8, shape=(_MAP_ROWS, _MAP_COLS)),
    _ObservationField(name="map_item_category", dtype=np.uint16, shape=(_MAP_ROWS, _MAP_COLS)),
    _ObservationField(name="map_item_kind", dtype=np.int16, shape=(_MAP_ROWS, _MAP_COLS)),
    _ObservationField(name="map_item_quantity", dtype=np.int16, shape=(_MAP_ROWS, _MAP_COLS)),
    _ObservationField(name="map_item_flags", dtype=np.uint64, shape=(_MAP_ROWS, _MAP_COLS)),
    _ObservationField(name="map_has_monster", dtype=np.uint8, shape=(_MAP_ROWS, _MAP_COLS)),
    _ObservationField(name="map_monster_kind", dtype=np.int16, shape=(_MAP_ROWS, _MAP_COLS)),
    _ObservationField(name="map_monster_hp", dtype=np.int16, shape=(_MAP_ROWS, _MAP_COLS)),
    _ObservationField(name="map_monster_state", dtype=np.int16, shape=(_MAP_ROWS, _MAP_COLS)),
    _ObservationField(name="inventory_present", dtype=np.uint8, shape=(_INVENTORY_SIZE,)),
    _ObservationField(name="inventory_letters", dtype=np.uint8, shape=(_INVENTORY_SIZE,)),
    _ObservationField(name="inventory_strs", dtype=np.uint8, shape=(_INVENTORY_SIZE, _INVENTORY_STR_LENGTH)),
    _ObservationField(name="inventory_category", dtype=np.uint16, shape=(_INVENTORY_SIZE,)),
    _ObservationField(name="inventory_kind", dtype=np.int16, shape=(_INVENTORY_SIZE,)),
    _ObservationField(name="inventory_quantity", dtype=np.int16, shape=(_INVENTORY_SIZE,)),
    _ObservationField(name="inventory_flags", dtype=np.uint64, shape=(_INVENTORY_SIZE,)),
    _ObservationField(name="inventory_enchant1", dtype=np.int16, shape=(_INVENTORY_SIZE,)),
    _ObservationField(name="inventory_enchant2", dtype=np.int16, shape=(_INVENTORY_SIZE,)),
    _ObservationField(name="inventory_charges", dtype=np.int16, shape=(_INVENTORY_SIZE,)),
    _ObservationField(name="blstats", dtype=np.int64, shape=(_BLSTATS_SIZE,)),
    _ObservationField(name="message", dtype=np.uint8, shape=(_MESSAGE_SIZE,)),
    _ObservationField(name="program_state", dtype=np.uint64, shape=(_PROGRAM_STATE_SIZE,)),
)


def _default_library_path() -> Path:
    return _default_data_dir() / _bridge_library_name()


def _default_data_dir() -> Path:
    return _PACKAGED_DATA_DIR


def _bridge_library_name() -> str:
    suffix = ".dylib" if sys.platform == "darwin" else ".so"
    return f"libbruhogue_brogue{suffix}"


def _configure_library(library: Any) -> None:
    library.brh_abi_version.argtypes = []
    library.brh_abi_version.restype = ctypes.c_uint32
    library.brh_observation_size.argtypes = []
    library.brh_observation_size.restype = ctypes.c_size_t
    library.brh_screen_cols.argtypes = []
    library.brh_screen_cols.restype = ctypes.c_int
    library.brh_screen_rows.argtypes = []
    library.brh_screen_rows.restype = ctypes.c_int
    library.brh_map_cols.argtypes = []
    library.brh_map_cols.restype = ctypes.c_int
    library.brh_map_rows.argtypes = []
    library.brh_map_rows.restype = ctypes.c_int
    library.brh_inventory_size.argtypes = []
    library.brh_inventory_size.restype = ctypes.c_int
    library.brh_inventory_str_length.argtypes = []
    library.brh_inventory_str_length.restype = ctypes.c_int
    library.brh_reset.argtypes = [ctypes.c_uint64, ctypes.POINTER(_CObservation)]
    library.brh_reset.restype = ctypes.c_int
    library.brh_step.argtypes = [
        ctypes.c_long,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(_CObservation),
    ]
    library.brh_step.restype = ctypes.c_int
    library.brh_close.argtypes = []
    library.brh_close.restype = None
    library.brh_set_data_dir.argtypes = [ctypes.c_char_p]
    library.brh_set_data_dir.restype = None
    library.brh_last_error.argtypes = []
    library.brh_last_error.restype = ctypes.c_char_p


def _validate_bridge_abi(library: Any) -> None:
    abi_version = int(library.brh_abi_version())
    observation_size = int(library.brh_observation_size())
    screen_cols = int(library.brh_screen_cols())
    screen_rows = int(library.brh_screen_rows())
    map_cols = int(library.brh_map_cols())
    map_rows = int(library.brh_map_rows())
    inventory_size = int(library.brh_inventory_size())
    inventory_str_length = int(library.brh_inventory_str_length())
    expected_size = ctypes.sizeof(_CObservation)

    if (
        abi_version != _BRIDGE_ABI_VERSION
        or observation_size != expected_size
        or screen_cols != _SCREEN_COLS
        or screen_rows != _SCREEN_ROWS
        or map_cols != _MAP_COLS
        or map_rows != _MAP_ROWS
        or inventory_size != _INVENTORY_SIZE
        or inventory_str_length != _INVENTORY_STR_LENGTH
    ):
        msg = (
            "Brogue bridge ABI mismatch: "
            f"abi={abi_version} expected={_BRIDGE_ABI_VERSION}, "
            f"observation_size={observation_size} expected={expected_size}, "
            f"screen={screen_cols}x{screen_rows} expected={_SCREEN_COLS}x{_SCREEN_ROWS}, "
            f"map={map_cols}x{map_rows} expected={_MAP_COLS}x{_MAP_ROWS}, "
            f"inventory={inventory_size} expected={_INVENTORY_SIZE}, "
            f"inventory_str_length={inventory_str_length} expected={_INVENTORY_STR_LENGTH}"
        )
        raise BackendUnavailableError(BackendErrorCode.BRIDGE_ABI_MISMATCH, msg)


def _observation_from_c(observation: _CObservation) -> ObservationDict:
    return {
        field.name: _copy_array(getattr(observation, field.name), field.shape)
        for field in _OBS_FIELD_SPECS
    }


def _copy_array(source: Any, shape: tuple[int, ...]) -> NDArray[np.generic]:
    array = np.ctypeslib.as_array(source).copy()
    return cast(NDArray[np.generic], array.reshape(shape))


def _allocate_observation_buffers() -> ObservationDict:  # pyright: ignore[reportUnusedFunction]
    """Pre-allocate numpy arrays matching the observation layout."""
    return {
        field.name: np.zeros(field.shape, dtype=field.dtype)
        for field in _OBS_FIELD_SPECS
    }


def _observation_into(observation: _CObservation, buffers: ObservationDict) -> None:  # pyright: ignore[reportUnusedFunction]
    """Copy C observation data into pre-allocated numpy buffers (zero allocation)."""
    for field in _OBS_FIELD_SPECS:
        _copy_into(getattr(observation, field.name), buffers[field.name])


def _allocate_shared_observation_buffers(num_envs: int) -> ObservationDict:  # pyright: ignore[reportUnusedFunction]
    """Pre-allocate shared numpy arrays with a leading num_envs dimension (PufferLib-style).

    Each field has shape ``(num_envs, *field_shape)`` so that a single contiguous
    allocation backs all environment slots.  Per-env views are obtained by indexing
    the first axis (e.g. ``shared["glyphs"][i]``), which yields a zero-copy view.
    """
    return {
        field.name: np.zeros((num_envs, *field.shape), dtype=field.dtype)
        for field in _OBS_FIELD_SPECS
    }


def _copy_into(source: Any, dest: NDArray[np.generic]) -> None:
    """Copy ctypes array data into a pre-allocated numpy array without allocation."""
    flat = np.ctypeslib.as_array(source)
    np.copyto(dest.ravel(), flat)


def _bridge_seed(seed: int | None) -> int:
    if seed is None:
        return secrets.randbits(64) or 1  # avoid 0 which triggers time-based seed on C side
    if seed < 0 or seed > 2**64 - 1:
        msg = "Brogue seed must be an unsigned 64-bit integer"
        raise ValueError(msg)
    if seed == 0:
        return _GYM_ZERO_BRIDGE_SEED
    return seed


def _bridge_key(key: str) -> int:
    encoded = key.encode("latin-1")
    if len(encoded) != 1:
        msg = f"in-process Brogue bridge only supports one-byte keys: {key!r}"
        raise BackendUnavailableError(BackendErrorCode.KEY_UNSUPPORTED, msg)
    return encoded[0]


def _bridge_step_status(rc: int) -> _BridgeStepStatus:
    try:
        return _BridgeStepStatus(rc)
    except ValueError as exc:
        msg = f"Brogue bridge returned unknown step status {rc}"
        raise BackendUnavailableError(BackendErrorCode.BRIDGE_OPERATION_FAILED, msg) from exc


_DEFAULT_BROGUE_SAMPLER_SEED = 0
_BROGUE_SEED_MIN = 1
_BROGUE_SEED_MAX = 2**64 - 1
_WORKER_CLOSE_TIMEOUT_SECONDS = 1.0
_WORKER_TERMINATE_TIMEOUT_SECONDS = 5.0

type BrogueSeedSpec = int | NDArray[np.integer[Any]] | None


class _ProcessWorker(msgspec.Struct, frozen=True, kw_only=True, forbid_unknown_fields=True):
    process: Any
    connection: Connection


def _field_inner_strides(dtype: Any, shape: tuple[int, ...]) -> tuple[int, ...]:
    itemsize = np.dtype(dtype).itemsize
    strides: list[int] = []
    stride = itemsize
    for dim in reversed(shape):
        strides.insert(0, stride)
        stride *= dim
    return tuple(strides)


def _shared_observation_views(buffer: Any, num_envs: int) -> tuple[ObservationDict, list[ObservationDict]]:
    obs_size = ctypes.sizeof(_CObservation)
    shared_obs: ObservationDict = {}
    for field in _OBS_FIELD_SPECS:
        field_offset: int = getattr(_CObservation, field.name).offset
        shared_obs[field.name] = np.ndarray(
            shape=(num_envs, *field.shape),
            dtype=field.dtype,
            buffer=buffer,
            offset=field_offset,
            strides=(obs_size, *_field_inner_strides(field.dtype, field.shape)),
        )
    obs_views = [
        {key: value[index] for key, value in shared_obs.items()}
        for index in range(num_envs)
    ]
    return shared_obs, obs_views


def _single_observation_view(buffer: Any, env_id: int) -> ObservationDict:
    obs_size = ctypes.sizeof(_CObservation)
    observation: ObservationDict = {}
    base_offset = env_id * obs_size
    for field in _OBS_FIELD_SPECS:
        field_offset: int = getattr(_CObservation, field.name).offset
        observation[field.name] = np.ndarray(
            shape=field.shape,
            dtype=field.dtype,
            buffer=buffer,
            offset=base_offset + field_offset,
            strides=_field_inner_strides(field.dtype, field.shape),
        )
    return observation


def _copy_observation(source: ObservationDict, destination: ObservationDict) -> None:
    for key, value in source.items():
        np.copyto(destination[key], value, casting="no")


def _owned_observation(source: ObservationDict) -> ObservationDict:
    return {key: value.copy() for key, value in source.items()}


def _process_worker_main(connection: Connection, shm_name: str, env_id: int) -> None:
    shared_memory = SharedMemory(name=shm_name)
    observation = _single_observation_view(shared_memory.buf, env_id)
    backend = _InProcessBrogue()
    try:
        while True:
            command = cast(tuple[Any, ...], connection.recv())
            operation = command[0]
            try:
                if operation == "close":
                    connection.send(("closed",))
                    return
                if operation == "reset":
                    seed = cast(int, command[1])
                    result = backend.reset(seed=seed)
                    _copy_observation(result.observation, observation)
                    connection.send(("reset", result.info))
                    continue
                if operation == "step":
                    action = cast(Action, command[1])
                    result = backend.step(action)
                    _copy_observation(result.observation, observation)
                    connection.send(("step", result.reward, result.terminated, result.info))
                    continue
                msg = f"unknown worker command: {operation!r}"
                connection.send(("error", "RuntimeError", msg, ""))
            except BackendUnavailableError as exc:
                connection.send(("backend_error", exc.code.value, str(exc)))
            except Exception as exc:  # noqa: BLE001
                connection.send(("error", type(exc).__name__, str(exc), traceback.format_exc()))
    finally:
        backend.close()
        shared_memory.close()
        connection.close()


def _worker_start_method() -> str:
    main_file = getattr(sys.modules.get("__main__"), "__file__", None)
    if isinstance(main_file, str) and main_file and not main_file.startswith("<"):
        return "spawn"
    if "fork" in mp.get_all_start_methods():
        return "fork"
    return "spawn"


def _validate_sampler_seed(seed: int | None) -> int:
    if seed is None:
        return _DEFAULT_BROGUE_SAMPLER_SEED
    if seed < 0 or seed > _BROGUE_SEED_MAX:
        msg = "Brogue sampler seed must be an unsigned 64-bit integer"
        raise ValueError(msg)
    return seed


def _manual_brogue_seeds(seed_array: NDArray[np.integer[Any]], count: int) -> list[int]:
    if seed_array.shape != (count,):
        msg = f"expected manual Brogue seed override array with shape ({count},), got {seed_array.shape}"
        raise ValueError(msg)
    if not np.issubdtype(seed_array.dtype, np.integer):
        msg = "manual Brogue seed overrides must be an integer ndarray"
        raise TypeError(msg)

    seeds = [int(seed) for seed in seed_array.tolist()]
    for seed in seeds:
        if seed < _BROGUE_SEED_MIN or seed > _BROGUE_SEED_MAX:
            msg = "manual Brogue seed overrides must be in [1, 2**64 - 1]"
            raise ValueError(msg)
    return seeds


def _sample_brogue_seeds(seed: int | None, count: int) -> list[int]:
    sampler_seed = _validate_sampler_seed(seed)
    rng = np.random.default_rng(sampler_seed)
    seeds = rng.integers(
        _BROGUE_SEED_MIN,
        _BROGUE_SEED_MAX,
        size=count,
        dtype=np.uint64,
        endpoint=True,
    )
    return [int(seed) for seed in seeds.tolist()]


def _resolve_brogue_seeds(seed: object, count: int) -> list[int]:
    if isinstance(seed, np.ndarray):
        return _manual_brogue_seeds(cast(NDArray[np.integer[Any]], seed), count)
    if seed is None or isinstance(seed, int):
        return _sample_brogue_seeds(seed, count)
    msg = "seed must be None, an integer sampler seed, or an integer ndarray of Brogue seed overrides"
    raise TypeError(msg)


class BrogueBackend:
    """Run Brogue instances in isolated worker processes.

    Each worker process owns one normal in-process Brogue bridge instance. The
    parent process routes reset/step commands over pipes and uses shared memory
    as the worker-to-parent observation transport. Public reset/step results are
    owned arrays so callers cannot retain views into unmapped memory after
    close(). This keeps the Python experiment path single and makes OS
    processes the isolation boundary between Brogue global-state instances.
    """

    def __init__(self, num_envs: int) -> None:
        if num_envs < 1:
            msg = "num_envs must be at least 1"
            raise ValueError(msg)

        self.num_envs = num_envs
        self.library_path = _default_library_path()
        self.data_dir = _default_data_dir()
        self._states: list[BackendSessionState] = [
            BackendSessionState.CLOSED for _ in range(num_envs)
        ]
        self._workers: list[_ProcessWorker] = []
        self._shared_memory: SharedMemory | None = None
        self._obs_size = ctypes.sizeof(_CObservation)
        self._shared_obs: ObservationDict = {}
        self._obs_views: list[ObservationDict] = []

        self._init_workers()

    def reset(
        self,
        env_id: int | None = None,
        *,
        seed: BrogueSeedSpec = None,
    ) -> BackendReset:
        """Reset one environment.

        Without *env_id*, this is the scalar ``BrogueBackend`` reset path and is
        only valid when ``num_envs == 1``. Integer seeds seed the Brogue-job
        sampler; pass a shape ``(1,)`` integer ndarray to override the concrete
        Brogue seed manually.
        """

        resolved_env_id = self._scalar_env_id() if env_id is None else env_id
        self._check_env_id(resolved_env_id)
        return self._reset_envs([resolved_env_id], _resolve_brogue_seeds(seed, 1))[0]

    def step(self, action: Action) -> BackendStep:
        """Apply one scalar action for ``num_envs == 1``."""

        env_id = self._scalar_env_id()
        return self.step_many([action])[env_id]

    def step_many(self, actions: Sequence[Action]) -> list[BackendStep]:
        """Step all environments via one command fan-out to worker processes."""

        if len(actions) != self.num_envs:
            msg = f"expected {self.num_envs} actions, got {len(actions)}"
            raise ValueError(msg)

        not_running = [
            index
            for index, state in enumerate(self._states)
            if state is not BackendSessionState.RUNNING
        ]
        if not_running:
            ids = ", ".join(str(index) for index in not_running)
            msg = f"env(s) {ids}: Brogue bridge is not running; call reset() first"
            raise BackendUnavailableError(BackendErrorCode.BRIDGE_NOT_RUNNING, msg)

        # Validate action encodability before sending any worker command, so a
        # bad action cannot partially advance the batch.
        for action in actions:
            try:
                _bridge_key(action.brogue_input().key)
            except ValueError as exc:
                msg = f"BrogueBackend cannot encode action {action.kind.value}"
                raise BackendUnavailableError(BackendErrorCode.ACTION_UNENCODABLE, msg) from exc

        env_ids = range(self.num_envs)
        sent_env_ids: list[int] = []
        try:
            for env_id, worker, action in zip(env_ids, self._workers, actions, strict=True):
                self._send(worker, ("step", action))
                sent_env_ids.append(env_id)
        except BackendUnavailableError:
            self._discard_batch_responses(
                sent_env_ids,
                operation="step",
                expected_status="step",
            )
            for env_id in env_ids:
                self._states[env_id] = BackendSessionState.CLOSED
            raise

        results: list[BackendStep] = []
        responses = self._recv_batch_responses(
            env_ids,
            operation="step",
            expected_status="step",
        )
        for env_id, response in enumerate(responses):
            reward = cast(float, response[1])
            terminated = cast(bool, response[2])
            info = cast(InfoDict, response[3])
            if terminated:
                self._states[env_id] = BackendSessionState.CLOSED
            results.append(
                BackendStep(
                    observation=_owned_observation(self._obs_views[env_id]),
                    reward=reward,
                    terminated=terminated,
                    info=info,
                ),
            )
        return results

    def reset_many(self, *, seed: BrogueSeedSpec = None) -> list[BackendReset]:
        """Reset all environments via one command fan-out to worker processes.

        ``seed=None`` uses a fixed sampler seed. ``seed=<int>`` uses that integer
        to sample concrete Brogue seeds for all environments. Pass a shape
        ``(num_envs,)`` integer ndarray to manually override the concrete Brogue
        seeds.
        """

        return self._reset_envs(
            list(range(self.num_envs)),
            _resolve_brogue_seeds(seed, self.num_envs),
        )

    def snapshot(self) -> BrogueSnapshot:
        msg = "process Brogue backend does not expose full-state snapshots yet"
        raise BackendUnavailableError(BackendErrorCode.SNAPSHOT_UNSUPPORTED, msg)

    def restore(self, snapshot: BrogueSnapshot) -> BackendReset:
        _ = snapshot
        msg = "process Brogue backend does not expose full-state restore yet"
        raise BackendUnavailableError(BackendErrorCode.RESTORE_UNSUPPORTED, msg)

    def close(self) -> None:
        """Close workers and release shared memory."""

        workers, self._workers = self._workers, []
        for worker in workers:
            if worker.process.is_alive():
                try:
                    worker.connection.send(("close",))
                except (BrokenPipeError, EOFError, OSError):
                    pass
        for worker in workers:
            if worker.process.is_alive():
                try:
                    if worker.connection.poll(_WORKER_CLOSE_TIMEOUT_SECONDS):
                        response = cast(tuple[Any, ...], worker.connection.recv())
                        if response[0] != "closed":
                            worker.process.terminate()
                    else:
                        worker.process.terminate()
                except (BrokenPipeError, EOFError, OSError):
                    worker.process.terminate()
            worker.process.join(timeout=_WORKER_TERMINATE_TIMEOUT_SECONDS)
            if worker.process.is_alive():
                worker.process.terminate()
                worker.process.join(timeout=_WORKER_TERMINATE_TIMEOUT_SECONDS)
            if worker.process.is_alive():
                worker.process.kill()
                worker.process.join(timeout=_WORKER_TERMINATE_TIMEOUT_SECONDS)
            worker.connection.close()

        if self._shared_memory is not None:
            self._shared_memory.close()
            self._shared_memory.unlink()
            self._shared_memory = None
        self._shared_obs = {}
        self._obs_views = []
        self._states = [BackendSessionState.CLOSED for _ in range(self.num_envs)]

    def __enter__(self) -> BrogueBackend:
        return self

    def __exit__(self, *args: object) -> None:
        _ = args
        self.close()

    def _init_workers(self) -> None:
        if not self.library_path.is_file():
            msg = (
                f"Brogue bridge library does not exist: {self.library_path}. "
                "Run through `uv run` from the project root so package sync builds it."
            )
            raise BackendUnavailableError(BackendErrorCode.BRIDGE_LIBRARY_MISSING, msg)
        if not self.data_dir.is_dir():
            msg = f"Brogue data directory does not exist: {self.data_dir}"
            raise BackendUnavailableError(BackendErrorCode.BRIDGE_DATA_DIR_INVALID, msg)

        shared_memory = SharedMemory(create=True, size=self.num_envs * self._obs_size)
        self._shared_memory = shared_memory
        np.ndarray(
            (self.num_envs * self._obs_size,),
            dtype=np.uint8,
            buffer=shared_memory.buf,
        ).fill(0)
        self._shared_obs, self._obs_views = _shared_observation_views(
            shared_memory.buf,
            self.num_envs,
        )

        ctx = cast(Any, mp.get_context(_worker_start_method()))
        try:
            for env_id in range(self.num_envs):
                parent_conn, child_conn = ctx.Pipe()
                process = ctx.Process(
                    target=_process_worker_main,
                    args=(child_conn, shared_memory.name, env_id),
                    daemon=True,
                )
                process.start()
                child_conn.close()
                self._workers.append(_ProcessWorker(process=process, connection=parent_conn))
        except Exception:
            self.close()
            raise

    def _reset_envs(
        self,
        env_ids: Sequence[int],
        seeds: Sequence[int],
    ) -> list[BackendReset]:
        if len(env_ids) != len(seeds):
            msg = "env_ids and seeds must have the same length"
            raise ValueError(msg)
        for seed in seeds:
            _bridge_seed(seed)
        for env_id in env_ids:
            self._check_env_id(env_id)
            self._states[env_id] = BackendSessionState.CLOSED
        sent_env_ids: list[int] = []
        try:
            for env_id, seed in zip(env_ids, seeds, strict=True):
                self._send(self._workers[env_id], ("reset", seed))
                sent_env_ids.append(env_id)
        except BackendUnavailableError:
            self._discard_batch_responses(
                sent_env_ids,
                operation="reset",
                expected_status="reset",
            )
            for env_id in env_ids:
                self._states[env_id] = BackendSessionState.CLOSED
            raise

        results: list[BackendReset] = []
        responses = self._recv_batch_responses(
            env_ids,
            operation="reset",
            expected_status="reset",
        )
        for env_id, response in zip(env_ids, responses, strict=True):
            self._states[env_id] = BackendSessionState.RUNNING
            results.append(
                BackendReset(
                    observation=_owned_observation(self._obs_views[env_id]),
                    info=cast(InfoDict, response[1]),
                ),
            )
        return results

    def _scalar_env_id(self) -> int:
        if self.num_envs != 1:
            msg = "scalar reset/step is only available when num_envs == 1"
            raise ValueError(msg)
        return 0

    def _check_env_id(self, env_id: int) -> None:
        if not (0 <= env_id < self.num_envs):
            msg = f"env_id {env_id} is out of range [0, {self.num_envs})"
            raise IndexError(msg)

    def _recv_batch_responses(
        self,
        env_ids: Sequence[int],
        *,
        operation: str,
        expected_status: str,
    ) -> list[tuple[Any, ...]]:
        responses: list[tuple[Any, ...] | None] = []
        first_error: BackendUnavailableError | None = None

        for env_id in env_ids:
            try:
                response = self._recv(self._workers[env_id], operation, env_id)
                if response[0] != expected_status:
                    msg = f"env {env_id}: expected {expected_status} response, got {response[0]!r}"
                    raise BackendUnavailableError(
                        BackendErrorCode.BRIDGE_OPERATION_FAILED,
                        msg,
                    )
            except BackendUnavailableError as exc:
                if first_error is None:
                    first_error = exc
                responses.append(None)
            else:
                responses.append(response)

        if first_error is not None:
            for env_id in env_ids:
                self._states[env_id] = BackendSessionState.CLOSED
            raise first_error

        return [cast(tuple[Any, ...], response) for response in responses]

    def _discard_batch_responses(
        self,
        env_ids: Sequence[int],
        *,
        operation: str,
        expected_status: str,
    ) -> None:
        try:
            self._recv_batch_responses(
                env_ids,
                operation=operation,
                expected_status=expected_status,
            )
        except BackendUnavailableError:
            pass

    @staticmethod
    def _send(worker: _ProcessWorker, message: tuple[object, ...]) -> None:
        try:
            worker.connection.send(message)
        except (BrokenPipeError, EOFError, OSError) as exc:
            msg = "Brogue worker process is not available"
            raise BackendUnavailableError(BackendErrorCode.BRIDGE_OPERATION_FAILED, msg) from exc

    @staticmethod
    def _recv(worker: _ProcessWorker, operation: str, env_id: int) -> tuple[Any, ...]:
        try:
            response = cast(tuple[Any, ...], worker.connection.recv())
        except (BrokenPipeError, EOFError, OSError) as exc:
            msg = f"env {env_id}: Brogue worker process stopped during {operation}"
            raise BackendUnavailableError(BackendErrorCode.BRIDGE_OPERATION_FAILED, msg) from exc

        status = response[0]
        if status == "backend_error":
            code = BackendErrorCode(cast(str, response[1]))
            detail = cast(str, response[2])
            msg = f"env {env_id}: {detail}"
            raise BackendUnavailableError(code, msg)
        if status == "error":
            error_type = cast(str, response[1])
            detail = cast(str, response[2])
            traceback_text = cast(str, response[3])
            msg = f"env {env_id}: Brogue worker {operation} failed: {error_type}: {detail}\n{traceback_text}"
            raise BackendUnavailableError(BackendErrorCode.BRIDGE_OPERATION_FAILED, msg)
        return response
