"""Groq chat-completions client (T2.2; PLAN.md §1.3 hardware, §5).

A thin wrapper over Groq's OpenAI-compatible REST API, built on
`requests` (already a project dependency, see T1.2's `download.py`)
rather than adding the `groq` SDK as a new dependency for one call
shape. Every completion is cached to disk via `cragb.generate.api_cache`
and requests retry with backoff on transient errors, mirroring
`download.py`'s `_build_session` pattern.

Groq is one of the two free-tier APIs named in PLAN.md §1.3
(Groq, Google AI Studio); a second provider client can be added later
behind the same `complete(messages) -> str` shape without changing any
caller in `cragb.generate` or `cragb.bench`.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from cragb.generate.api_cache import DiskCache

logger = logging.getLogger(__name__)

CONNECT_TIMEOUT_S = 10


class MissingAPIKeyError(RuntimeError):
    """Raised when the configured API-key environment variable is unset."""


def _build_session(max_retries: int) -> requests.Session:
    """A requests Session with retry/backoff for transient API errors."""
    session = requests.Session()
    retry = Retry(
        total=max_retries,
        backoff_factor=2.0,  # 2s, 4s, 8s, 16s, 32s
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("POST",),
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


@dataclass
class GroqClient:
    """Chat-completions client for Groq's OpenAI-compatible API.

    Args:
        model: Groq model id (e.g. "llama-3.1-8b-instant").
        api_base: API base URL.
        api_key_env: name of the environment variable holding the API
            key (read lazily, per call, so importing/constructing this
            class never requires the key to already be set).
        temperature: sampling temperature.
        max_tokens: max completion tokens.
        timeout_s: read timeout in seconds (connect timeout is fixed at
            `CONNECT_TIMEOUT_S`).
        max_retries: retry attempts on 429/5xx before giving up.
        cache_dir: directory for the disk-cached responses.
    """

    model: str
    api_base: str = "https://api.groq.com/openai/v1"
    api_key_env: str = "GROQ_API_KEY"
    temperature: float = 0.7
    max_tokens: int = 1500
    timeout_s: int = 30
    max_retries: int = 5
    cache_dir: str = "results/cache/api"
    _cache: DiskCache = field(init=False, repr=False)
    _session: requests.Session = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._cache = DiskCache(self.cache_dir)
        self._session = _build_session(self.max_retries)

    def _api_key(self) -> str:
        key = os.environ.get(self.api_key_env)
        if not key:
            raise MissingAPIKeyError(
                f"Environment variable {self.api_key_env!r} is not set. "
                f"Copy .env.example to .env, add your Groq key "
                f"(https://console.groq.com/keys), and make sure it's "
                f"loaded (e.g. via python-dotenv) before running."
            )
        return key

    def complete(self, messages: list[dict[str, str]]) -> str:
        """Return the assistant's completion text for `messages`.

        Args:
            messages: OpenAI-style chat messages, e.g.
                `[{"role": "user", "content": "..."}]`.

        Returns:
            The completion text content of the first choice.

        Raises:
            MissingAPIKeyError: if the API key env var is unset.
            requests.HTTPError: on a non-2xx response after retries.
        """
        # The cache key intentionally excludes the API key: only request
        # content determines whether two calls are "the same call".
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        def _call() -> str:
            resp = self._session.post(
                f"{self.api_base}/chat/completions",
                json=payload,
                headers={"Authorization": f"Bearer {self._api_key()}"},
                timeout=(CONNECT_TIMEOUT_S, self.timeout_s),
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]

        return self._cache.call(payload, _call)
