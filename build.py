"""PEP 517 build backend for broguegym's Python package and native bridge."""

from __future__ import annotations

import platform
import re
import subprocess
import sys
from importlib import import_module
from pathlib import Path
from typing import Any, cast

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class CustomBuildHook(BuildHookInterface[Any]):
    """Build and package the platform-specific Brogue bridge library."""

    def initialize(self, version: str, build_data: dict[str, Any]) -> None:
        root = Path(self.root)
        self._build_bridge(root)

        if version != "standard":
            return

        bin_dir = root / "BrogueCE" / "bin"
        native_bin = "broguegym/_native/bin"
        bridge_library = bin_dir / _bridge_library_name()
        if not bridge_library.is_file():
            msg = f"Brogue bridge build did not produce {bridge_library}"
            raise FileNotFoundError(msg)

        build_data["tag"] = _platform_wheel_tag(bridge_library)
        build_data["pure_python"] = False
        build_data["force_include"].update(
            {
                str(bridge_library): f"{native_bin}/{bridge_library.name}",
                str(bin_dir / "keymap.txt"): f"{native_bin}/keymap.txt",
                str(bin_dir / "assets"): f"{native_bin}/assets",
                str(root / "BrogueCE" / "LICENSE.txt"): "broguegym/_native/LICENSE.BrogueCE.txt",
            },
        )

    @staticmethod
    def _build_bridge(root: Path) -> None:
        subprocess.run(
            ["make", "-C", str(root / "BrogueCE"), "bridge"],
            cwd=root,
            check=True,
        )


def build_editable(*args: Any, **kwargs: Any) -> str:
    hatchling = import_module("hatchling.build")
    return cast(str, hatchling.build_editable(*args, **kwargs))


def build_wheel(*args: Any, **kwargs: Any) -> str:
    hatchling = import_module("hatchling.build")
    return cast(str, hatchling.build_wheel(*args, **kwargs))


def build_sdist(*args: Any, **kwargs: Any) -> str:
    hatchling = import_module("hatchling.build")
    return cast(str, hatchling.build_sdist(*args, **kwargs))


def _bridge_library_name() -> str:
    suffix = ".dylib" if sys.platform == "darwin" else ".so"
    return f"libbruhogue_brogue{suffix}"


def _platform_wheel_tag(bridge_library: Path) -> str:
    system = platform.system()
    machine = _normalized_machine()
    if system == "Darwin":
        platform_tag = f"macosx_{_macos_deployment_target(bridge_library)}_{machine}"
    elif system == "Linux":
        platform_tag = f"linux_{machine}"
    else:
        msg = f"unsupported Brogue bridge wheel platform: {system} {machine}"
        raise RuntimeError(msg)
    return f"py3-none-{platform_tag}"


def _normalized_machine() -> str:
    machine = platform.machine().lower().replace("-", "_")
    return {
        "amd64": "x86_64",
        "aarch64": "arm64" if platform.system() == "Darwin" else "aarch64",
    }.get(machine, machine)


def _macos_deployment_target(bridge_library: Path) -> str:
    output = subprocess.check_output(
        ["otool", "-l", str(bridge_library)],
        text=True,
    )
    if match := re.search(r"\bminos\s+(\d+)\.(\d+)", output):
        return f"{match.group(1)}_{match.group(2)}"
    if match := re.search(r"\bversion\s+(\d+)\.(\d+)", output):
        return f"{match.group(1)}_{match.group(2)}"
    msg = f"could not detect macOS deployment target for {bridge_library}"
    raise RuntimeError(msg)
