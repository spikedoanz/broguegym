from __future__ import annotations

from pathlib import Path

import pytest

from broguegym.snapshot import BrogueSnapshot


def test_snapshot_round_trips_through_memory_and_file(tmp_path: Path) -> None:
    snapshot = BrogueSnapshot(
        payload=b"state-bytes",
        metadata={"seed": 12, "build": "abc123"},
    )

    assert BrogueSnapshot.from_bytes(snapshot.to_bytes()) == snapshot

    path = tmp_path / "state.brhsnap"
    snapshot.write_to(path)
    loaded = BrogueSnapshot.read_from(path)

    assert loaded.payload == b"state-bytes"
    assert loaded.metadata == {"seed": 12, "build": "abc123"}


def test_snapshot_rejects_empty_payload() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        BrogueSnapshot(payload=b"")


def test_snapshot_rejects_unknown_bytes() -> None:
    with pytest.raises(ValueError, match="not a bruhogue snapshot"):
        BrogueSnapshot.from_bytes(b"not-a-snapshot")
