"""PEP 517 build backend for broguegym's Python package and native bridge."""

from __future__ import annotations

from importlib import import_module
from typing import Any, cast


def build_editable(*args: Any, **kwargs: Any) -> str:
    hatchling = import_module("hatchling.build")
    return cast(str, hatchling.build_editable(*args, **kwargs))


def build_wheel(*args: Any, **kwargs: Any) -> str:
    hatchling = import_module("hatchling.build")
    return cast(str, hatchling.build_wheel(*args, **kwargs))


def build_sdist(*args: Any, **kwargs: Any) -> str:
    hatchling = import_module("hatchling.build")
    return cast(str, hatchling.build_sdist(*args, **kwargs))
