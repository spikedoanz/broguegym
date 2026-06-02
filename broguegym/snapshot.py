"""Versioned snapshot container for complete Brogue game state."""

from __future__ import annotations

from pathlib import Path
from typing import Final, Self

import msgspec

type SnapshotMetadataValue = str | int | float | bool | None
type SnapshotMetadata = dict[str, SnapshotMetadataValue]

SNAPSHOT_FORMAT_VERSION: Final = 1
SNAPSHOT_MAGIC: Final = b"BRUHOGUE-SNAPSHOT\x00"
_HEADER_LEN_BYTES: Final = 4


def _empty_metadata() -> SnapshotMetadata:
    return {}


class _SnapshotHeader(msgspec.Struct, frozen=True, kw_only=True, forbid_unknown_fields=True):
    format: str
    version: int
    metadata: SnapshotMetadata = msgspec.field(default_factory=_empty_metadata)


class BrogueSnapshot(msgspec.Struct, frozen=True, kw_only=True, forbid_unknown_fields=True):
    """Opaque complete Brogue game state, optionally annotated with compatibility metadata."""

    payload: bytes
    metadata: SnapshotMetadata = msgspec.field(default_factory=_empty_metadata)

    def __post_init__(self) -> None:
        if not self.payload:
            msg = "snapshot payload must not be empty"
            raise ValueError(msg)

    def to_bytes(self) -> bytes:
        """Serialize this snapshot to the bruhogue file wrapper."""

        header = _SnapshotHeader(
            format="bruhogue.snapshot",
            version=SNAPSHOT_FORMAT_VERSION,
            metadata=dict(self.metadata),
        )
        header_bytes = msgspec.json.encode(header, order="deterministic")
        max_header_len = (1 << (_HEADER_LEN_BYTES * 8)) - 1
        if len(header_bytes) > max_header_len:
            msg = "snapshot metadata is too large"
            raise ValueError(msg)
        return (
            SNAPSHOT_MAGIC
            + len(header_bytes).to_bytes(_HEADER_LEN_BYTES, "big")
            + header_bytes
            + self.payload
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> Self:
        """Deserialize a bruhogue snapshot file wrapper."""

        if not data.startswith(SNAPSHOT_MAGIC):
            msg = "not a bruhogue snapshot"
            raise ValueError(msg)

        offset = len(SNAPSHOT_MAGIC)
        header_end = offset + _HEADER_LEN_BYTES
        if len(data) < header_end:
            msg = "truncated snapshot header"
            raise ValueError(msg)

        header_len = int.from_bytes(data[offset:header_end], "big")
        payload_start = header_end + header_len
        if len(data) <= payload_start:
            msg = "truncated snapshot payload"
            raise ValueError(msg)

        try:
            header = msgspec.json.decode(
                data[header_end:payload_start],
                type=_SnapshotHeader,
            )
        except msgspec.DecodeError as exc:
            msg = f"invalid snapshot header: {exc}"
            raise ValueError(msg) from exc

        if header.format != "bruhogue.snapshot":
            msg = "snapshot has an unsupported format marker"
            raise ValueError(msg)
        if header.version != SNAPSHOT_FORMAT_VERSION:
            msg = "snapshot has an unsupported format version"
            raise ValueError(msg)

        return cls(payload=data[payload_start:], metadata=header.metadata)

    def write_to(self, path: Path | str) -> None:
        """Write this snapshot to a file."""

        Path(path).write_bytes(self.to_bytes())

    @classmethod
    def read_from(cls, path: Path | str) -> Self:
        """Read a snapshot from a file."""

        return cls.from_bytes(Path(path).read_bytes())
