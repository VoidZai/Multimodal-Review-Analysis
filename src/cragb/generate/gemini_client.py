"""Google AI Studio (Gemini) multimodal client (T6.2; PLAN.md §3 E7, M6.md T6.2).

`GroqClient` (T2.2, T5.2) cannot serve E7's vision judge: PLAN.md §14.4's live
Groq catalog snapshot (2026-08-19, 13 models) has no vision-capable model at
all, and `GroqClient.complete()` types `messages` as `list[dict[str, str]]`
— there is no way to attach image bytes to a Groq call. Google AI Studio is
the second free-tier provider PLAN.md §1.3 already names, and its REST API
accepts inline image bytes directly in a `generateContent` request.

**This is an additive provider, not a replacement.** `GroqClient` is
untouched by this module — not imported for anything but its two most
generic, config-agnostic pieces (`CompletionResult`, the dataclass every
future provider's `complete_with_usage` should return, and
`MissingAPIKeyError`, so a missing key raises the same exception type
regardless of which provider is missing it). Every existing Groq-backed
result (M1-M5, T6.1) is exactly as it was before this file existed.

**Message shape is provider-neutral, not Gemini's wire format.** Callers
pass a flat list of *parts* for one user turn — `{"type": "text", "text":
...}` and `{"type": "image", "photo_id": ..., "mime": ..., "data_b64":
...}` (the exact shape `cragb.multimodal.photo_store.PhotoStore.to_data_part`
produces) — and `_build_payload` translates that into Gemini's
`contents[].parts[]` `inline_data` shape. This keeps T6.4's vision-judge
module from ever importing a Gemini-specific dict, so a third provider
could be added later behind the same shape without touching T6.4. Note
`photo_id` never reaches the actual HTTP payload (and therefore never
enters the cache key) — it exists only for the caller's own bookkeeping;
what Gemini receives, and what the cache hashes, is exactly `mime` +
`data_b64`. This is also what makes swapping one image for another (same
question text) a cache **miss**: the base64 image bytes are part of the
hashed payload, with nothing extra required to force that.

**Model choice (`gemini-3.6-flash`, `configs/vision_judge.yaml`):**
Google's pricing/model docs listed `gemini-2.5-flash` as current on
2026-08-24, and it still appears in the live `ListModels` response — but a
live `generateContent` call against it returns HTTP 404: "This model
models/gemini-2.5-flash is no longer available to new users. Please
update your code to use models/gemini-3.6-flash." A model that's
listed but not callable is exactly the failure mode PLAN.md §14.4's
"confirmed live" discipline exists to catch for Groq's catalog — the same
discipline applies here: a docs page or a `ListModels` entry is not
verification, only an actual `generateContent` call is. `gemini-3.6-flash`
(Google's own named replacement) is confirmed live, multimodal, and
free-tier eligible. If Google deprecates it before this project's final
report, only `configs/vision_judge.yaml` needs to change — no code here is
version-pinned. See that config's comments for a live gotcha worth
knowing before T6.4: Gemini 3.x's hidden "thinking" tokens are drawn from
the same `maxOutputTokens` budget as the visible answer, so a budget too
small for question + thinking + a short JSON verdict returns an empty
candidate, not an error.

Usage:
    client = GeminiClient(model="gemini-3.6-flash", cache_dir="results/cache/api")
    text = client.complete([{"type": "text", "text": "Describe this photo."}, image_part])
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
from cragb.generate.api_clients import CONNECT_TIMEOUT_S, CompletionResult, MissingAPIKeyError
from cragb.utils.io import resolve_path
from cragb.utils.timing import Timer

logger = logging.getLogger(__name__)


def _build_session(max_retries: int) -> requests.Session:
    """A `requests.Session` with retry/backoff for transient network errors.

    Duplicated from (not imported from) `cragb.generate.api_clients` /
    `cragb.data.download` deliberately: each API-calling module in this
    project owns its own copy rather than importing another module's
    leading-underscore internal, so a future change to Groq's retry policy
    can't silently change Gemini's (or vice versa).
    """
    session = requests.Session()
    retry = Retry(
        total=max_retries,
        backoff_factor=2.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("POST",),
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


class GeminiResponseError(RuntimeError):
    """Raised when a Gemini response has no usable candidate text.

    Covers both a safety-blocked prompt (`promptFeedback.blockReason` set,
    `candidates` empty) and a candidate that stopped for a reason other than
    normal completion with no text (e.g. `finishReason="SAFETY"` on the
    candidate itself) — a real, first-class outcome for a vision judge shown
    an arbitrary review photo, not something to paper over with an empty
    string that would silently fail `parse_vision_response` (T6.4) instead.
    """


def _build_payload(
    parts: list[dict[str, Any]], *, temperature: float, max_tokens: int
) -> dict[str, Any]:
    """Translate provider-neutral `parts` into a Gemini `generateContent` request body.

    The sole source of the disk-cache key (mirrors `GroqClient._build_payload`'s
    role): two calls with the same parts and generation config produce a
    byte-identical payload and therefore the same cache key, regardless of
    which method (`complete` / `complete_with_usage`) built it.

    Args:
        parts: `{"type": "text", "text": str}` or
            `{"type": "image", "mime": str, "data_b64": str}` entries (an
            image part's `photo_id`, if present, is ignored here — it never
            reaches Gemini or the cache key).
        temperature: sampling temperature.
        max_tokens: max output tokens.

    Returns:
        A `generateContent` request body (`contents` + `generationConfig`).

    Raises:
        ValueError: an unknown part `"type"`, or an image part missing
            `"mime"`/`"data_b64"`.
    """
    content_parts: list[dict[str, Any]] = []
    for part in parts:
        kind = part.get("type")
        if kind == "text":
            content_parts.append({"text": part["text"]})
        elif kind == "image":
            mime = part.get("mime")
            data_b64 = part.get("data_b64")
            if not mime or not data_b64:
                raise ValueError(f"image part missing 'mime' or 'data_b64': {part!r}")
            content_parts.append({"inline_data": {"mime_type": mime, "data": data_b64}})
        else:
            raise ValueError(f"unknown part type {kind!r}; expected 'text' or 'image'")

    return {
        "contents": [{"role": "user", "parts": content_parts}],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
        },
    }


def _extract_text(data: dict[str, Any]) -> str:
    """Pull the completion text out of a `generateContent` response body.

    Raises:
        GeminiResponseError: no candidates (prompt blocked) or the first
            candidate carries no text (e.g. blocked/truncated by safety
            filters) — a typed, catchable outcome rather than a bare
            `KeyError`/`IndexError` from indexing an empty list.
    """
    candidates = data.get("candidates") or []
    if not candidates:
        reason = (data.get("promptFeedback") or {}).get("blockReason", "no candidates returned")
        raise GeminiResponseError(f"Gemini returned no candidates (reason={reason})")

    content = candidates[0].get("content") or {}
    text_parts = [p.get("text", "") for p in content.get("parts") or []]
    text = "".join(text_parts)
    if not text:
        finish_reason = candidates[0].get("finishReason", "unknown")
        raise GeminiResponseError(f"Gemini candidate had no text (finishReason={finish_reason})")
    return text


@dataclass
class GeminiClient:
    """Chat-completions client for Google AI Studio's Gemini `generateContent` API.

    Mirrors `GroqClient`'s field shape and exposes the same two methods
    (`complete`, `complete_with_usage`) with the same disk-cache and
    call-log discipline, so `cragb.eval.cost_model`'s G4 accounting (T5.4)
    reads Gemini calls exactly the way it reads Groq calls — same
    `results/cache/api_calls_v1.jsonl`, same row shape.

    Args:
        model: Gemini model id (e.g. `"gemini-3.6-flash"`).
        api_base: API base URL.
        api_key_env: name of the environment variable holding the API key
            (read lazily, per call, so constructing this class never
            requires the key to already be set — matches
            `GroqClient._api_key`).
        temperature: sampling temperature.
        max_tokens: max output tokens (`generationConfig.maxOutputTokens`).
        timeout_s: read timeout in seconds (connect timeout is fixed at
            `cragb.generate.api_clients.CONNECT_TIMEOUT_S`).
        max_retries: retry attempts on 429/5xx before giving up.
        cache_dir: directory for the disk-cached responses. Shares
            `GroqClient`'s default (`results/cache/api`) but writes under a
            different cache key (the payload shapes are entirely
            different), so the two providers never collide in one
            directory.
        call_log_path: append-only JSONL log of every `complete_with_usage`
            call — shared with `GroqClient` by default so T5.4's cost
            accounting sees every provider's calls in one place.
    """

    model: str
    api_base: str = "https://generativelanguage.googleapis.com/v1beta"
    api_key_env: str = "GOOGLE_API_KEY"
    temperature: float = 0.7
    max_tokens: int = 1500
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
                f"Copy .env.example to .env, add your Google AI Studio key "
                f"(https://aistudio.google.com/apikey), and make sure it's "
                f"loaded (e.g. via python-dotenv) before running."
            )
        return key

    def _build_payload(self, parts: list[dict[str, Any]]) -> dict[str, Any]:
        """Build the request body for `parts` — shared by `complete()` and
        `complete_with_usage()` so a call made through either is a cache hit
        for the other, exactly mirroring `GroqClient._build_payload`."""
        return _build_payload(parts, temperature=self.temperature, max_tokens=self.max_tokens)

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        """POST `payload` to `generateContent` and return the parsed JSON body.

        The API key is sent as the `x-goog-api-key` header, not a URL query
        parameter, so it never ends up in request-URL logs (`requests`
        session/retry logging, proxies, etc.).
        """
        url = f"{self.api_base}/models/{self.model}:generateContent"
        resp = self._session.post(
            url,
            json=payload,
            headers={"x-goog-api-key": self._api_key()},
            timeout=(CONNECT_TIMEOUT_S, self.timeout_s),
        )
        resp.raise_for_status()
        return resp.json()

    def complete(self, parts: list[dict[str, Any]]) -> str:
        """Return the completion text for one user turn made of `parts`.

        Args:
            parts: provider-neutral text/image parts (see module docstring).

        Returns:
            The completion text.

        Raises:
            MissingAPIKeyError: if the API key env var is unset.
            GeminiResponseError: the response has no usable candidate text
                (e.g. safety-blocked).
            requests.HTTPError: on a non-2xx response after retries.
        """
        payload = self._build_payload(parts)

        def _call() -> str:
            data = self._post(payload)
            return _extract_text(data)

        return self._cache.call(payload, _call)

    def complete_with_usage(
        self, parts: list[dict[str, Any]], bypass_cache: bool = False
    ) -> CompletionResult:
        """Like `complete`, but also returns token usage, latency, and cache status.

        Args:
            parts: provider-neutral text/image parts (see module docstring).
            bypass_cache: if `True`, always issues a live call and never
                touches the disk cache (read or write) — see
                `GroqClient.complete_with_usage`'s docstring for why a
                latency measurement needs this. Default `False`.

        Returns:
            A `CompletionResult` (shared with `GroqClient` — same shape,
            same semantics: `cached=True` implies `latency_s is None`).

        Raises:
            MissingAPIKeyError: if the API key env var is unset.
            GeminiResponseError: the response has no usable candidate text.
            requests.HTTPError: on a non-2xx response after retries.
        """
        payload = self._build_payload(parts)

        def _call() -> tuple[str, dict[str, Any]]:
            with Timer() as t:
                data = self._post(payload)
            assert t.elapsed_s is not None
            text = _extract_text(data)
            usage = data.get("usageMetadata") or {}
            meta = {
                "prompt_tokens": usage.get("promptTokenCount"),
                "completion_tokens": usage.get("candidatesTokenCount"),
                # Gemini 3.x's hidden "thinking" tokens (module docstring; confirmed
                # live, T6.2) -- billed at the output rate but never appear in `text`,
                # so a cost calculation using `completion_tokens` alone understates the
                # real cost. Captured from the first call site rather than reconstructed
                # later, per PLAN.md §14.5's lesson from Groq's original usage-discarding
                # bug.
                "thinking_tokens": usage.get("thoughtsTokenCount"),
                "latency_s": t.elapsed_s,
                "model": data.get("modelVersion") or self.model,
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
            latency_s=(None if cached else (meta.get("latency_s") if meta else None)),
            cached=cached,
            model=(meta.get("model") if meta else None) or self.model,
            thinking_tokens=(meta.get("thinking_tokens") if meta else None),
        )
        self._log_call(result)
        return result

    def _log_call(self, result: CompletionResult) -> None:
        """Append one JSONL row for `result` to `call_log_path` (same shape and
        file `GroqClient` writes to, so T5.4's cost accounting reads both)."""
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
