"""Hatch build hook for bundling Brogue's native bridge runtime."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

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

        build_data["tag"] = _platform_wheel_tag()
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


def _bridge_library_name() -> str:
    suffix = ".dylib" if sys.platform == "darwin" else ".so"
    return f"libbruhogue_brogue{suffix}"


def _platform_wheel_tag() -> str:
    from hatchling.builders.macos import process_macos_plat_tag
    from packaging.tags import sys_tags

    tag = next(
        iter(
            tag
            for tag in sys_tags()
            if "manylinux" not in tag.platform and "musllinux" not in tag.platform
        ),
    )
    platform = tag.platform
    if sys.platform == "darwin":
        platform = process_macos_plat_tag(platform, compat=True)
    return f"py3-none-{platform}"
