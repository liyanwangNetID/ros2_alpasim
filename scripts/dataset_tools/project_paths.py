#!/usr/bin/env python3
"""Shared machine-local paths for AlpaSim dataset tools.

Path resolution priority:
1. Process environment variable.
2. Repository-local config/local_paths.env.
3. Repository-derived default, where applicable.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
LOCAL_CONFIG_PATH = REPOSITORY_ROOT / "config" / "local_paths.env"


def read_local_path_config(
    path: Path = LOCAL_CONFIG_PATH,
) -> dict[str, str]:
    """Read simple KEY=VALUE entries from the local path configuration."""
    values: dict[str, str] = {}

    if not path.is_file():
        return values

    with path.open("r", encoding="utf-8") as file:
        for line_number, raw_line in enumerate(file, start=1):
            line = raw_line.strip()

            if not line or line.startswith("#"):
                continue

            if "=" not in line:
                raise ValueError(
                    f"Invalid path configuration at {path}:{line_number}"
                )

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")

            if not key or not value:
                raise ValueError(
                    f"Empty key or value at {path}:{line_number}"
                )

            values[key] = value

    return values


def configured_path(
    name: str,
    *,
    config: Mapping[str, str],
    default: Path | None = None,
) -> Path:
    """Resolve a path using an environment override and local config."""
    raw_value = os.environ.get(name) or config.get(name)

    if raw_value:
        return Path(raw_value).expanduser().resolve()

    if default is not None:
        return default.expanduser().resolve()

    raise RuntimeError(
        f"{name} is not configured. Create {LOCAL_CONFIG_PATH} "
        "from config/local_paths.env.example."
    )


_LOCAL_CONFIG = read_local_path_config()

ALPASIM_ROS2_WS = configured_path(
    "ALPASIM_ROS2_WS",
    config=_LOCAL_CONFIG,
    default=REPOSITORY_ROOT,
)

ALPASIM_DATA_ROOT = configured_path(
    "ALPASIM_DATA_ROOT",
    config=_LOCAL_CONFIG,
)

ALPASIM_ROOT = configured_path(
    "ALPASIM_ROOT",
    config=_LOCAL_CONFIG,
)

ANNOTATION_ROOT = ALPASIM_DATA_ROOT / "annotations" / "v0.1-draft"
INTERMEDIATE_ROOT = ANNOTATION_ROOT / "intermediate"
REPORT_ROOT = ALPASIM_DATA_ROOT / "reports"
MANIFEST_ROOT = ALPASIM_DATA_ROOT / "manifests"
