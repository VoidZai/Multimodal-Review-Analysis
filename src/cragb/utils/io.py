"""Small filesystem/config helpers shared across the pipeline.

Kept deliberately minimal in M1: a YAML config loader and a path resolver
that anchors relative paths (as written in configs/*.yaml) to the repo
root, regardless of the current working directory a script/notebook is
launched from.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

# repo root = three levels up from this file (src/cragb/utils/io.py -> repo/)
REPO_ROOT = Path(__file__).resolve().parents[3]


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load a YAML config file into a dict.

    Args:
        config_path: path to a `.yaml` file, absolute or relative to the
            repo root (e.g. "configs/data.yaml").

    Returns:
        Parsed config as a dict.

    Raises:
        FileNotFoundError: if the resolved path does not exist.
    """
    path = resolve_path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(path: str | Path) -> Path:
    """Resolve a path relative to the repo root if it isn't already absolute.

    Args:
        path: an absolute path, or one relative to the repo root.

    Returns:
        An absolute `Path`.
    """
    p = Path(path)
    return p if p.is_absolute() else (REPO_ROOT / p)
