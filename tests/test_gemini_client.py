"""Unit tests for cragb.generate.gemini_client.GeminiClient (T6.2; M6.md T6.2).

No real network access: `GeminiClient._session.post` is monkeypatched on the
client's own `_session` instance to a fake that records the JSON payload it
was called with and returns a canned Gemini-shaped response — the same
pattern `test_api_clients.py` uses for `GroqClient`.

Four things this file exists to lock down, per M6.md T6.2's validation
checks:

1. The Groq cache is untouched by this module's existence: constructing and
   using a `GeminiClient` never writes under a `GroqClient`'s cache
   namespace, and a known `GroqClient` payload's cache key is still the
   exact hash `test_api_clients.py` already hard-codes.
2. A multimodal payload's cache key includes the image bytes, so swapping
   the image (same question text) is a cache **miss** — otherwise T6.4's
   order-swap judge calls would collide.
3. A missing `GOOGLE_API_KEY` raises the same `MissingAPIKeyError` as a
   missing `GROQ_API_KEY`, with an actionable message.
4. A response with no `usageMetadata` yields `None` token counts rather
   than crashing (the M5 §14.5 lesson, applied here from the start).
"""

from __future__ import annotations

import json

import pytest

from cragb.generate.api_cache import _cache_key
from cragb.generate.api_clients import GroqClient, MissingAPIKeyError
from cragb.generate.gemini_client import GeminiClient, GeminiResponseError, _build_payload


class FakeResponse:
    def __init__(self, text: str | None, status_code: int = 200, usage: dict | None = None, model_version: str | None = None):
        self._text = text
        self.status_code = status_code
        self._usage = usage
        self._model_version = model_version

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self) -> dict:
        body: dict = {}
        if self._text is not None:
            body["candidates"] = [
                {"content": {"parts": [{"text": self._text}], "role": "model"}, "finishReason": "STOP"}
            ]
        else:
            body["candidates"] = []
            body["promptFeedback"] = {"blockReason": "SAFETY"}
        if self._usage is not None:
            body["usageMetadata"] = self._usage
        if self._model_version is not None:
            body["modelVersion"] = self._model_version
        return body


def make_client(tmp_path, **overrides) -> GeminiClient:
    kwargs = dict(
        model="gemini-3.6-flash",
        cache_dir=str(tmp_path / "cache"),
        call_log_path=str(tmp_path / "api_calls_v1.jsonl"),
    )
    kwargs.update(overrides)
    return GeminiClient(**kwargs)


TEXT_PART = {"type": "text", "text": "Which photo better shows a torn seam?"}


def image_part(data_b64: str = "AAAA", mime: str = "image/jpeg", photo_id: str = "abc123") -> dict:
    return {"type": "image", "photo_id": photo_id, "mime": mime, "data_b64": data_b64}


class TestPayloadConstruction:
    def test_text_only_payload_shape(self):
        payload = _build_payload([TEXT_PART], temperature=0.0, max_tokens=300)
        assert payload == {
            "contents": [{"role": "user", "parts": [{"text": TEXT_PART["text"]}]}],
            "generationConfig": {"temperature": 0.0, "maxOutputTokens": 300},
        }

    def test_image_part_becomes_inline_data_and_drops_photo_id(self):
        payload = _build_payload([TEXT_PART, image_part()], temperature=0.0, max_tokens=300)
        parts = payload["contents"][0]["parts"]
        assert parts[1] == {"inline_data": {"mime_type": "image/jpeg", "data": "AAAA"}}
        # photo_id must never reach the wire payload (and therefore never
        # enters the cache key) -- it's caller-side bookkeeping only.
        assert "photo_id" not in json.dumps(payload)

    def test_unknown_part_type_raises(self):
        with pytest.raises(ValueError, match="unknown part type"):
            _build_payload([{"type": "audio"}], temperature=0.0, max_tokens=300)

    def test_image_part_missing_data_raises(self):
        with pytest.raises(ValueError, match="missing"):
            _build_payload([{"type": "image", "mime": "image/jpeg"}], temperature=0.0, max_tokens=300)


class TestComplete:
    def test_returns_text(self, tmp_path, monkeypatch):
        client = make_client(tmp_path)
        monkeypatch.setattr(client._session, "post", lambda url, **kw: FakeResponse("A torn seam is visible in photo B."))
        monkeypatch.setenv("GOOGLE_API_KEY", "test-key")

        text = client.complete([TEXT_PART, image_part()])

        assert text == "A torn seam is visible in photo B."

    def test_second_identical_call_is_a_cache_hit(self, tmp_path, monkeypatch):
        client = make_client(tmp_path)
        calls = []
        monkeypatch.setattr(client._session, "post", lambda url, **kw: calls.append(1) or FakeResponse("ok"))
        monkeypatch.setenv("GOOGLE_API_KEY", "test-key")

        parts = [TEXT_PART, image_part()]
        client.complete(parts)
        client.complete(parts)

        assert len(calls) == 1

    def test_swapping_the_image_is_a_cache_miss(self, tmp_path, monkeypatch):
        # Same question text, different image bytes -- must NOT collide,
        # or T6.4's order-swap (surfaced-as-A vs surfaced-as-B) calls would
        # return each other's cached answers.
        client = make_client(tmp_path)
        calls = []
        monkeypatch.setattr(client._session, "post", lambda url, **kw: calls.append(1) or FakeResponse("ok"))
        monkeypatch.setenv("GOOGLE_API_KEY", "test-key")

        client.complete([TEXT_PART, image_part(data_b64="AAAA")])
        client.complete([TEXT_PART, image_part(data_b64="BBBB")])

        assert len(calls) == 2

    def test_blocked_prompt_raises_gemini_response_error(self, tmp_path, monkeypatch):
        client = make_client(tmp_path)
        monkeypatch.setattr(client._session, "post", lambda url, **kw: FakeResponse(None))
        monkeypatch.setenv("GOOGLE_API_KEY", "test-key")

        with pytest.raises(GeminiResponseError, match="SAFETY"):
            client.complete([TEXT_PART])

    def test_missing_api_key_raises_missing_api_key_error(self, tmp_path, monkeypatch):
        client = make_client(tmp_path)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

        with pytest.raises(MissingAPIKeyError, match="GOOGLE_API_KEY"):
            client.complete([TEXT_PART])


class TestCompleteWithUsage:
    def test_fresh_call_returns_tokens_latency_and_model(self, tmp_path, monkeypatch):
        client = make_client(tmp_path)
        monkeypatch.setattr(
            client._session,
            "post",
            lambda url, **kw: FakeResponse(
                "An answer.", usage={"promptTokenCount": 42, "candidatesTokenCount": 7}, model_version="gemini-3.6-flash-001"
            ),
        )
        monkeypatch.setenv("GOOGLE_API_KEY", "test-key")

        result = client.complete_with_usage([TEXT_PART, image_part()])

        assert result.text == "An answer."
        assert result.prompt_tokens == 42
        assert result.completion_tokens == 7
        assert result.latency_s is not None and result.latency_s >= 0
        assert result.cached is False
        assert result.model == "gemini-3.6-flash-001"

    def test_missing_usage_metadata_reports_none_tokens_not_a_crash(self, tmp_path, monkeypatch):
        client = make_client(tmp_path)
        monkeypatch.setattr(client._session, "post", lambda url, **kw: FakeResponse("ok"))
        monkeypatch.setenv("GOOGLE_API_KEY", "test-key")

        result = client.complete_with_usage([TEXT_PART])

        assert result.prompt_tokens is None
        assert result.completion_tokens is None

    def test_second_call_is_a_cache_hit_with_no_latency(self, tmp_path, monkeypatch):
        client = make_client(tmp_path)
        call_count = 0

        def fake_post(url, **kw):
            nonlocal call_count
            call_count += 1
            return FakeResponse("Cached.", usage={"promptTokenCount": 10, "candidatesTokenCount": 3})

        monkeypatch.setattr(client._session, "post", fake_post)
        monkeypatch.setenv("GOOGLE_API_KEY", "test-key")

        parts = [TEXT_PART, image_part()]
        first = client.complete_with_usage(parts)
        second = client.complete_with_usage(parts)

        assert call_count == 1
        assert second.cached is True
        assert second.text == first.text == "Cached."
        assert second.prompt_tokens == 10
        assert second.latency_s is None

    def test_shares_a_cache_entry_with_complete(self, tmp_path, monkeypatch):
        client = make_client(tmp_path)
        call_count = 0

        def fake_post(url, **kw):
            nonlocal call_count
            call_count += 1
            return FakeResponse("Shared.", usage={"promptTokenCount": 1, "candidatesTokenCount": 1})

        monkeypatch.setattr(client._session, "post", fake_post)
        monkeypatch.setenv("GOOGLE_API_KEY", "test-key")

        parts = [TEXT_PART]
        client.complete(parts)
        result = client.complete_with_usage(parts)

        assert call_count == 1
        assert result.cached is True
        assert result.text == "Shared."

    def test_bypass_cache_never_reads_or_writes(self, tmp_path, monkeypatch):
        client = make_client(tmp_path)
        call_count = 0

        def fake_post(url, **kw):
            nonlocal call_count
            call_count += 1
            return FakeResponse("Live.", usage={"promptTokenCount": 1, "candidatesTokenCount": 1})

        monkeypatch.setattr(client._session, "post", fake_post)
        monkeypatch.setenv("GOOGLE_API_KEY", "test-key")

        parts = [TEXT_PART]
        first = client.complete_with_usage(parts, bypass_cache=True)
        second = client.complete_with_usage(parts, bypass_cache=True)

        assert call_count == 2
        assert first.cached is False and second.cached is False
        assert first.latency_s is not None and second.latency_s is not None
        # Never written to the cache either -- a normal (non-bypass) call
        # afterwards must still be a live call, not a stale hit.
        assert client._cache.get(client._build_payload(parts)) is None

    def test_blocked_prompt_raises_gemini_response_error(self, tmp_path, monkeypatch):
        client = make_client(tmp_path)
        monkeypatch.setattr(client._session, "post", lambda url, **kw: FakeResponse(None))
        monkeypatch.setenv("GOOGLE_API_KEY", "test-key")

        with pytest.raises(GeminiResponseError, match="SAFETY"):
            client.complete_with_usage([TEXT_PART])

    def test_appends_one_row_per_call_to_call_log(self, tmp_path, monkeypatch):
        client = make_client(tmp_path)
        monkeypatch.setattr(client._session, "post", lambda url, **kw: FakeResponse("ok"))
        monkeypatch.setenv("GOOGLE_API_KEY", "test-key")

        client.complete_with_usage([TEXT_PART])
        client.complete_with_usage([{"type": "text", "text": "different question"}])

        log_path = tmp_path / "api_calls_v1.jsonl"
        rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
        assert len(rows) == 2
        assert all(r["model"] == "gemini-3.6-flash" for r in rows)


class TestGroqCacheUntouched:
    def test_groq_hardcoded_cache_key_hash_is_unaffected(self, tmp_path):
        # Exact hash test_api_clients.py already hard-codes for GroqClient's
        # default complete() payload (T5.2's regression guard) -- this
        # module must not change it. Constructing a GeminiClient (even
        # pointed at the same cache_dir) must not perturb GroqClient's own
        # payload construction or cache key derivation in any way.
        expected_hash = "d83909556c4eb0319ac15394f140dcbef1a05dce20367a2b52802d4958240403"
        shared_cache_dir = str(tmp_path / "cache")

        groq_client = GroqClient(
            model="openai/gpt-oss-20b",
            cache_dir=shared_cache_dir,
            call_log_path=str(tmp_path / "groq_calls.jsonl"),
        )
        GeminiClient(model="gemini-3.6-flash", cache_dir=shared_cache_dir, call_log_path=str(tmp_path / "gemini_calls.jsonl"))

        payload = groq_client._build_payload([{"role": "user", "content": "Hi"}])
        assert _cache_key(payload) == expected_hash

    def test_gemini_and_groq_never_collide_in_a_shared_cache_dir(self, tmp_path, monkeypatch):
        shared_cache_dir = str(tmp_path / "cache")

        groq_client = GroqClient(
            model="openai/gpt-oss-20b", cache_dir=shared_cache_dir, call_log_path=str(tmp_path / "groq_calls.jsonl")
        )
        gemini_client = GeminiClient(
            model="gemini-3.6-flash", cache_dir=shared_cache_dir, call_log_path=str(tmp_path / "gemini_calls.jsonl")
        )

        class FakeGroqResponse:
            status_code = 200

            def raise_for_status(self):
                return None

            def json(self):
                return {"choices": [{"message": {"content": "groq answer"}}]}

        monkeypatch.setattr(groq_client._session, "post", lambda url, **kw: FakeGroqResponse())
        monkeypatch.setattr(gemini_client._session, "post", lambda url, **kw: FakeResponse("gemini answer"))
        monkeypatch.setenv("GROQ_API_KEY", "test-key")
        monkeypatch.setenv("GOOGLE_API_KEY", "test-key")

        assert groq_client.complete([{"role": "user", "content": "Hi"}]) == "groq answer"
        assert gemini_client.complete([TEXT_PART]) == "gemini answer"
