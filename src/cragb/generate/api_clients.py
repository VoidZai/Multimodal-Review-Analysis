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

**Usage telemetry (T5.2; PLAN.md §3 E6, §8 G4).** `complete()` returns
only the completion text and always has — every caller written against
it (T4a, T4b.1-4) keeps working unchanged. `complete_with_usage()` is an
additive sibling for M5's cost/latency work: it returns a
`CompletionResult` carrying token counts, measured latency, and whether
the call was a cache hit, via `DiskCache.call_with_meta`'s metadata
sidecar. Both methods build their request payload through the same
`_build_payload` so a call already cached by `complete()` is a cache hit
for `complete_with_usage()` too (and vice versa) — this is what lets
T5.4 recover token counts for the 180 T4b.2 transcripts for free, by
re-issuing them through `complete_with_usage` and hitting the existing
cache rather than re-spending quota.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from cragb.generate.api_cache import DiskCache
from cragb.utils.io import resolve_path
from cragb.utils.timing import Timer

logger = logging.getLogger(__name__)

CONNECT_TIMEOUT_S = 10


@dataclass(frozen=True)
class CompletionResult:
    """Return value of `GroqClient.complete_with_usage`.

    Attributes:
        text: the completion text (identical to what `complete()` returns
            for the same messages).
        prompt_tokens: input token count, or `None` if unrecoverable —
            either the API omitted `usage` on a fresh call, or this was a
            cache hit on an entry written before T5.2 (no metadata
            sidecar exists to recover it from).
        completion_tokens: output token count, or `None` for the same
            reasons as `prompt_tokens`.
        latency_s: measured wall-clock round-trip time for a fresh call,
            or `None` for a cache hit. A hit is served from disk in
            microseconds regardless of how slow the original call was;
            reporting that as "latency" would be meaningless, so a hit
            always reports `None` here even if a sidecar happens to hold
            a stored value from the original call.
        cached: whether this call was served from the disk cache.
        model: the Groq model id that produced `text`.
    """

    text: str
    prompt_tokens: int | None
    completion_tokens: int | None
    latency_s: float | None
    cached: bool
    model: str


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
        reasoning_effort: optional reasoning-effort hint (e.g. `"none"`,
            `"low"`), forwarded to the API only when set. Added for
            T4b.4's judge model (`qwen/qwen3.6-27b`), which otherwise
            emits visible `<think>...</think>` reasoning inline in its
            completion (PLAN.md §14.4) — `"none"` suppresses that.
            Left as `None` by default so every existing config
            (`grounded_qa.yaml`, `closed_book_qa.yaml`, ...) that never
            sets this keeps sending the exact same request payload as
            before, byte for byte — see `complete`'s docstring for why
            that matters for the disk cache.
        timeout_s: read timeout in seconds (connect timeout is fixed at
            `CONNECT_TIMEOUT_S`).
        max_retries: retry attempts on 429/5xx before giving up.
        cache_dir: directory for the disk-cached responses.
        call_log_path: append-only JSONL log of every `complete_with_usage`
            call (T5.2; PLAN.md §3 E6) — one row per call with timestamp,
            model, token counts, latency, and cache-hit status, the raw
            material T5.4's per-arm $/query accounting reads. `complete()`
            never writes to this log, so it stays untouched for every
            existing caller. Relative to the repo root, like `cache_dir`.
    """

    model: str
    api_base: str = "https://api.groq.com/openai/v1"
    api_key_env: str = "GROQ_API_KEY"
    temperature: float = 0.7
    max_tokens: int = 1500
    reasoning_effort: str | None = None
    timeout_s: int = 30
    max_retries: int = 5
    cache_dir: str = "results/cache/api"
    call_log_path: str = "results/cache/api_calls_v1.jsonl"
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

    def _build_payload(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        """Build the request body for `messages` — the sole source of the cache key.

        Shared by `complete()` and `complete_with_usage()` so a call made
        through either method is a cache hit for the other: same messages,
        same config, byte-identical payload, byte-identical
        `cragb.generate.api_cache._cache_key`.
        """
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        # Only included when set, and deliberately not `"reasoning_effort": None`
        # otherwise -- every payload built before this field existed had no such key at
        # all, and this keeps it that way for any caller that leaves it unset, so their
        # cache keys (a hash of the full payload, cragb.generate.api_cache._cache_key)
        # don't change and every already-cached response stays a hit.
        if self.reasoning_effort is not None:
            payload["reasoning_effort"] = self.reasoning_effort
        return payload

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        """POST `payload` to the chat-completions endpoint and return the parsed JSON body."""
        resp = self._session.post(
            f"{self.api_base}/chat/completions",
            json=payload,
            headers={"Authorization": f"Bearer {self._api_key()}"},
            timeout=(CONNECT_TIMEOUT_S, self.timeout_s),
        )
        resp.raise_for_status()
        return resp.json()

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
        payload = self._build_payload(messages)

        def _call() -> str:
            data = self._post(payload)
            return data["choices"][0]["message"]["content"]

        return self._cache.call(payload, _call)

    def complete_with_usage(
        self, messages: list[dict[str, str]], bypass_cache: bool = False
    ) -> CompletionResult:
        """Like `complete`, but also returns token usage, latency, and cache status.

        Uses the exact same request payload as `complete()` (via
        `_build_payload`), so this method and `complete()` share one cache
        entry per distinct `messages`/config — a question already answered
        via `complete()` (as every T4a/T4b transcript was) is a cache hit
        here, with `prompt_tokens`/`completion_tokens`/`latency_s` all
        `None` since no sidecar was ever recorded for it (see
        `CompletionResult`). Every call through this method, hit or miss,
        appends one row to `call_log_path`.

        Args:
            messages: OpenAI-style chat messages, e.g.
                `[{"role": "user", "content": "..."}]`.
            bypass_cache: if `True` (T5.5's end-to-end latency harness),
                always issues a live call and never touches the disk cache
                — no read, no write. A cache hit's ~0ms latency would make
                every "how long does a real question take" measurement
                meaningless, so bypassing the read is the whole point; the
                response is deliberately not written back either, since a
                live sample at temperature > 0 can return a different
                completion than the one already cached from the original
                generation run, and silently overwriting that canonical
                transcript would be worse than this call simply not
                caching at all. Default `False` preserves this method's
                exact prior behaviour for every existing caller.

        Returns:
            A `CompletionResult`. `cached` is always `False` when
            `bypass_cache=True`.

        Raises:
            MissingAPIKeyError: if the API key env var is unset.
            requests.HTTPError: on a non-2xx response after retries.
        """
        payload = self._build_payload(messages)

        def _call() -> tuple[str, dict[str, Any]]:
            with Timer() as t:
                data = self._post(payload)
            assert t.elapsed_s is not None
            text = data["choices"][0]["message"]["content"]
            usage = data.get("usage") or {}
            meta = {
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "latency_s": t.elapsed_s,
                "model": data.get("model", self.model),
            }
            return text, meta

        if bypass_cache:
            text, meta = _call()
            cached = False
        else:
            text, meta, cached = self._cache.call_with_meta(payload, _call)

        result = CompletionResult(
            text=text,
            prompt_tokens=(meta.get("prompt_tokens") if meta else None),
            completion_tokens=(meta.get("completion_tokens") if meta else None),
            # A hit is served from disk in microseconds; the original call's
            # latency (even if a sidecar happens to hold it) does not
            # describe *this* call, so a hit always reports None here.
            latency_s=(None if cached else (meta.get("latency_s") if meta else None)),
            cached=cached,
            model=(meta.get("model") if meta else None) or self.model,
        )
        self._log_call(result)
        return result

    def _log_call(self, result: CompletionResult) -> None:
        """Append one JSONL row for `result` to `call_log_path`."""
        path = resolve_path(self.call_log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": result.model,
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "latency_s": result.latency_s,
            "cached": result.cached,
        }
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, sort_keys=True))
            f.write("\n")
