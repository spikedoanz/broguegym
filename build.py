"""PEP 517 build backend for bruhogue's Python package and native bridge."""

from __future__ import annotations

import subprocess
from importlib import import_module
from pathlib import Path
from typing import Any, cast

_ROOT = Path(__file__).resolve().parent


def _build_bridge() -> None:
    subprocess.run(
        ["make", "-C", "BrogueCE", "bridge"],
        cwd=_ROOT,
        check=True,
    )


def build_editable(*args: Any, **kwargs: Any) -> str:
    _build_bridge()
    hatchling = import_module("hatchling.build")
    return cast(str, hatchling.build_editable(*args, **kwargs))


def build_wheel(*args: Any, **kwargs: Any) -> str:
    _build_bridge()
    hatchling = import_module("hatchling.build")
    return cast(str, hatchling.build_wheel(*args, **kwargs))


def build_sdist(*args: Any, **kwargs: Any) -> str:
    hatchling = import_module("hatchling.build")
    return cast(str, hatchling.build_sdist(*args, **kwargs))
