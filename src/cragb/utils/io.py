"""Small filesystem/config helpers shared across the pipeline.

Kept deliberately minimal in M1: a YAML config loader and a path resolver
that anchors relative paths (as written in configs/*.yaml) to the repo
root, regardless of the current working directory a script/notebook is
launched from.
"""

from __future__ import annotations

import hashlib
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


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Compute the SHA-256 hex digest of a file via a streamed read.

    Used for manifest generation (e.g. `corpus_v1_manifest.json`), so a
    frozen artifact's integrity can be independently re-verified later
    without ever holding the whole file in memory.

    Args:
        path: file to hash, absolute or relative to the repo root.
        chunk_size: bytes read per iteration.

    Returns:
        Lowercase hex-encoded SHA-256 digest.
    """
    resolved = resolve_path(path)
    digest = hashlib.sha256()
    with resolved.open("rb") as f:
        for block in iter(lambda: f.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()
