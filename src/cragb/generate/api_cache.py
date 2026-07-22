"""Disk cache for LLM API calls (T2.2; PLAN.md §5, §1.4 bottleneck #3).

Every API call in this project is cached to disk, keyed on the full
request payload (model, messages, temperature, ...). Two things this
buys, per PLAN.md §5's "no reproducibility without it" principle:

- **Quota safety.** Groq/Google AI Studio free tiers rate-limit; a re-run
  after a crash, or a re-run that only adds one more taxonomy category,
  should not re-spend quota on prompts already answered.
- **Reproducibility.** A cached response is the literal bytes returned
  for a given prompt+params, so results are re-derivable offline without
  depending on model availability or non-determinism at temperature > 0.

The cache key deliberately excludes the API key and any other
credential — only the request *content* (model, messages, temperature,
etc.) goes into the hash, so cache files never contain secrets and are
safe to commit if ever desired (they currently are not — see
`.gitignore`).
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from cragb.utils.io import resolve_path

logger = logging.getLogger(__name__)


def _cache_key(payload: dict[str, Any]) -> str:
    """Stable SHA-256 hash of a JSON-serializable request payload.

    `sort_keys=True` and fixed separators make the encoding
    order-independent and whitespace-independent, so semantically
    identical payloads always hash identically regardless of how the
    caller happened to construct the dict.
    """
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass
class DiskCache:
    """A flat directory of `<sha256>.json` files, one per cached call."""

    cache_dir: str | Path
    _dir: Path = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._dir = resolve_path(self.cache_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    def _path_for(self, key: str) -> Path:
        return self._dir / f"{key}.json"

    def get(self, payload: dict[str, Any]) -> Any | None:
        """Return the cached response for `payload`, or `None` on a miss."""
        path = self._path_for(_cache_key(payload))
        if not path.is_file():
            return None
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)["response"]

    def set(self, payload: dict[str, Any], response: Any) -> None:
        """Store `response` for `payload`, alongside the payload itself for auditability."""
        path = self._path_for(_cache_key(payload))
        with path.open("w", encoding="utf-8") as f:
            json.dump({"request": payload, "response": response}, f, indent=2, sort_keys=True)
            f.write("\n")

    def call(self, payload: dict[str, Any], fn: Callable[[], Any]) -> Any:
        """Return the cached response for `payload`, calling `fn()` only on a miss.

        Args:
            payload: the JSON-serializable request, used as the cache key.
            fn: zero-argument callable that performs the actual API call;
                only invoked when `payload` has never been seen before.

        Returns:
            The cached or freshly-fetched response.
        """
        cached = self.get(payload)
        if cached is not None:
            logger.debug("cache hit for key %s", _cache_key(payload))
            return cached
        response = fn()
        self.set(payload, response)
        return response
