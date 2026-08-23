"""Unit tests for cragb.generate.api_clients.GroqClient (T4b.4; M4b.md T4b.4).

No real network access: `requests.Session.post` is monkeypatched on the client's own
`_session` instance to a fake that records the JSON payload it was called with and
returns a canned successful response. This is the first test file for `GroqClient`
itself -- added alongside T4b.4's `reasoning_effort` field specifically to lock down a
guarantee the judge config leans on: adding that field must not change the request
payload (and therefore the disk-cache key, `cragb.generate.api_cache._cache_key`) for any
existing caller that never sets it, or every response already cached under T4a/T4b.1-3
silently stops being a cache hit.

`TestCompletePayloadIsUnchangedByT5_2` and `TestCompleteWithUsage` (T5.2; M5.md T5.2)
cover the usage/latency telemetry addition: the cache key for `complete()`'s payload is
byte-identical to a hard-coded hash captured before T5.2 (the actual regression that
would silently re-spend free-tier quota); `complete_with_usage` reads real token counts
and latency on a fresh call, reports `latency_s=None` and recovered token counts on a
cache hit, and shares one cache entry with `complete()` for the same messages/config.
"""

from __future__ import annotations

import json

import pytest

from cragb.generate.api_cache import DiskCache, _cache_key
from cragb.generate.api_clients import GroqClient, MissingAPIKeyError


class FakeResponse:
    def __init__(self, content: str, status_code: int = 200, usage: dict | None = None, model: str | None = None):
        self._content = content
        self.status_code = status_code
        self._usage = usage
        self._model = model

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self) -> dict:
        body: dict = {"choices": [{"message": {"content": self._content}}]}
        if self._usage is not None:
            body["usage"] = self._usage
        if self._model is not None:
            body["model"] = self._model
        return body


def make_client(tmp_path, **overrides) -> GroqClient:
    # call_log_path defaults inside tmp_path too -- complete_with_usage() writes to it
    # unconditionally, and the default "results/cache/api_calls_v1.jsonl" resolves
    # against the real repo root (cragb.utils.io.REPO_ROOT), not tmp_path, so any test
    # that forgot to override it would otherwise leak rows into the actual project.
    kwargs = dict(
        model="openai/gpt-oss-20b",
        cache_dir=str(tmp_path / "cache"),
        call_log_path=str(tmp_path / "api_calls_v1.jsonl"),
    )
    kwargs.update(overrides)
    return GroqClient(**kwargs)


class TestPayloadConstruction:
    def test_default_payload_has_no_reasoning_effort_key(self, tmp_path, monkeypatch):
        client = make_client(tmp_path)
        captured = {}

        def fake_post(url, json, headers, timeout):
            captured.update(json)
            return FakeResponse("An answer.")

        monkeypatch.setattr(client._session, "post", fake_post)
        monkeypatch.setenv("GROQ_API_KEY", "test-key")

        client.complete([{"role": "user", "content": "Hi"}])

        assert "reasoning_effort" not in captured
        assert captured == {
            "model": "openai/gpt-oss-20b",
            "messages": [{"role": "user", "content": "Hi"}],
            "temperature": 0.7,
            "max_tokens": 1500,
        }

    def test_reasoning_effort_none_default_produces_identical_payload_to_pre_t4b4_client(
        self, tmp_path, monkeypatch
    ):
        # Explicitly constructing without reasoning_effort (as every config predating
        # T4b.4 does) must be indistinguishable, at the payload level, from a client
        # built before this field existed -- this is the actual cache-compatibility
        # guarantee, phrased as a test rather than just a docstring claim.
        client = GroqClient(model="openai/gpt-oss-20b", cache_dir=str(tmp_path / "cache"))
        captured = {}

        def fake_post(url, **kw):
            captured.update(kw["json"])
            return FakeResponse("ok")

        monkeypatch.setattr(client._session, "post", fake_post)
        monkeypatch.setenv("GROQ_API_KEY", "test-key")

        client.complete([{"role": "user", "content": "Hi"}])

        assert "reasoning_effort" not in captured

    def test_reasoning_effort_set_is_included_in_payload(self, tmp_path, monkeypatch):
        client = make_client(tmp_path, model="qwen/qwen3.6-27b", reasoning_effort="none")
        captured = {}

        def fake_post(url, json, headers, timeout):
            captured.update(json)
            return FakeResponse('{"correctness": 5}')

        monkeypatch.setattr(client._session, "post", fake_post)
        monkeypatch.setenv("GROQ_API_KEY", "test-key")

        client.complete([{"role": "user", "content": "Hi"}])

        assert captured["reasoning_effort"] == "none"


class TestComplete:
    def test_returns_message_content(self, tmp_path, monkeypatch):
        client = make_client(tmp_path)
        monkeypatch.setattr(client._session, "post", lambda url, **kw: FakeResponse("The answer."))
        monkeypatch.setenv("GROQ_API_KEY", "test-key")

        assert client.complete([{"role": "user", "content": "Hi"}]) == "The answer."

    def test_missing_api_key_raises_before_any_network_call(self, tmp_path, monkeypatch):
        client = make_client(tmp_path)
        calls = []
        monkeypatch.setattr(client._session, "post", lambda url, **kw: calls.append(kw) or FakeResponse("x"))
        monkeypatch.delenv("GROQ_API_KEY", raising=False)

        with pytest.raises(MissingAPIKeyError):
            client.complete([{"role": "user", "content": "Hi"}])
        assert calls == []

    def test_second_identical_call_is_served_from_cache_not_network(self, tmp_path, monkeypatch):
        client = make_client(tmp_path)
        call_count = 0

        def fake_post(url, json, headers, timeout):
            nonlocal call_count
            call_count += 1
            return FakeResponse("Cached answer.")

        monkeypatch.setattr(client._session, "post", fake_post)
        monkeypatch.setenv("GROQ_API_KEY", "test-key")

        messages = [{"role": "user", "content": "Hi"}]
        first = client.complete(messages)
        second = client.complete(messages)

        assert first == second == "Cached answer."
        assert call_count == 1

    def test_different_reasoning_effort_is_a_different_cache_key(self, tmp_path, monkeypatch):
        # Same model/messages, only reasoning_effort differs -- must not collide in the
        # disk cache (they are genuinely different requests to the API).
        call_count = 0

        def fake_post(url, json, headers, timeout):
            nonlocal call_count
            call_count += 1
            return FakeResponse("ok")

        cache_dir = str(tmp_path / "cache")
        client_a = GroqClient(model="qwen/qwen3.6-27b", cache_dir=cache_dir, reasoning_effort=None)
        client_b = GroqClient(model="qwen/qwen3.6-27b", cache_dir=cache_dir, reasoning_effort="none")
        monkeypatch.setattr(client_a._session, "post", fake_post)
        monkeypatch.setattr(client_b._session, "post", fake_post)
        monkeypatch.setenv("GROQ_API_KEY", "test-key")

        messages = [{"role": "user", "content": "Hi"}]
        client_a.complete(messages)
        client_b.complete(messages)

        assert call_count == 2


class TestCompletePayloadIsUnchangedByT5_2:
    def test_default_payload_cache_key_matches_pre_t5_2_hash(self, tmp_path, monkeypatch):
        # Captured by hashing `complete()`'s exact payload dict before T5.2 touched this
        # file (model/messages/temperature/max_tokens, no reasoning_effort key). If this
        # ever changes, every response cached under T2.2-T4b.4 silently stops being a
        # cache hit -- the same hazard T4b.4's own tests already guard for
        # reasoning_effort, extended here to guard the T5.2 refactor of payload
        # construction into `_build_payload`.
        expected_hash = "d83909556c4eb0319ac15394f140dcbef1a05dce20367a2b52802d4958240403"
        client = make_client(tmp_path, model="openai/gpt-oss-20b")
        monkeypatch.setattr(client._session, "post", lambda url, **kw: FakeResponse("ok"))
        monkeypatch.setenv("GROQ_API_KEY", "test-key")

        payload = client._build_payload([{"role": "user", "content": "Hi"}])

        assert _cache_key(payload) == expected_hash


class TestCompleteWithUsage:
    def test_fresh_call_returns_tokens_latency_and_model(self, tmp_path, monkeypatch):
        client = make_client(tmp_path)
        monkeypatch.setattr(
            client._session,
            "post",
            lambda url, **kw: FakeResponse(
                "An answer.", usage={"prompt_tokens": 42, "completion_tokens": 7}, model="openai/gpt-oss-20b"
            ),
        )
        monkeypatch.setenv("GROQ_API_KEY", "test-key")

        result = client.complete_with_usage([{"role": "user", "content": "Hi"}])

        assert result.text == "An answer."
        assert result.prompt_tokens == 42
        assert result.completion_tokens == 7
        assert result.latency_s is not None and result.latency_s >= 0
        assert result.cached is False
        assert result.model == "openai/gpt-oss-20b"

    def test_missing_usage_block_reports_none_tokens_not_a_crash(self, tmp_path, monkeypatch):
        client = make_client(tmp_path)
        monkeypatch.setattr(client._session, "post", lambda url, **kw: FakeResponse("ok"))
        monkeypatch.setenv("GROQ_API_KEY", "test-key")

        result = client.complete_with_usage([{"role": "user", "content": "Hi"}])

        assert result.prompt_tokens is None
        assert result.completion_tokens is None

    def test_second_call_is_a_cache_hit_with_recovered_tokens_and_no_latency(self, tmp_path, monkeypatch):
        client = make_client(tmp_path)
        call_count = 0

        def fake_post(url, **kw):
            nonlocal call_count
            call_count += 1
            return FakeResponse("Cached.", usage={"prompt_tokens": 42, "completion_tokens": 7})

        monkeypatch.setattr(client._session, "post", fake_post)
        monkeypatch.setenv("GROQ_API_KEY", "test-key")

        messages = [{"role": "user", "content": "Hi"}]
        first = client.complete_with_usage(messages)
        second = client.complete_with_usage(messages)

        assert call_count == 1
        assert second.cached is True
        assert second.text == first.text == "Cached."
        assert second.prompt_tokens == 42
        assert second.completion_tokens == 7
        assert second.latency_s is None

    def test_shares_a_cache_entry_with_complete(self, tmp_path, monkeypatch):
        client = make_client(tmp_path)
        call_count = 0

        def fake_post(url, **kw):
            nonlocal call_count
            call_count += 1
            return FakeResponse("Shared answer.", usage={"prompt_tokens": 10, "completion_tokens": 3})

        monkeypatch.setattr(client._session, "post", fake_post)
        monkeypatch.setenv("GROQ_API_KEY", "test-key")

        messages = [{"role": "user", "content": "Hi"}]
        plain = client.complete(messages)
        via_usage = client.complete_with_usage(messages)

        assert call_count == 1
        assert via_usage.cached is True
        assert via_usage.text == plain == "Shared answer."

    def test_hit_on_response_cached_before_t5_2_reports_none_tokens_and_latency(self, tmp_path, monkeypatch):
        # A response written by plain complete()/DiskCache.call() (no sidecar) must
        # still be usable through complete_with_usage -- this is the exact scenario
        # T5.4 relies on for the 180 pre-T5.2 T4b.2 transcripts.
        client = make_client(tmp_path)
        monkeypatch.setenv("GROQ_API_KEY", "test-key")
        messages = [{"role": "user", "content": "Hi"}]

        monkeypatch.setattr(client._session, "post", lambda url, **kw: FakeResponse("Old response."))
        client.complete(messages)  # writes only the response file, no *.meta.json

        def fn_should_not_run(url, **kw):
            raise AssertionError("a cache hit must not re-issue the network call")

        monkeypatch.setattr(client._session, "post", fn_should_not_run)

        result = client.complete_with_usage(messages)

        assert result.text == "Old response."
        assert result.cached is True
        assert result.prompt_tokens is None
        assert result.completion_tokens is None
        assert result.latency_s is None

    def test_appends_one_call_log_row_per_call(self, tmp_path, monkeypatch):
        client = make_client(tmp_path, call_log_path=str(tmp_path / "calls.jsonl"))
        monkeypatch.setattr(
            client._session,
            "post",
            lambda url, **kw: FakeResponse("ok", usage={"prompt_tokens": 5, "completion_tokens": 2}),
        )
        monkeypatch.setenv("GROQ_API_KEY", "test-key")

        client.complete_with_usage([{"role": "user", "content": "Hi"}])
        client.complete_with_usage([{"role": "user", "content": "Hi"}])  # cache hit
        client.complete_with_usage([{"role": "user", "content": "Bye"}])  # distinct call

        log_path = tmp_path / "calls.jsonl"
        rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]

        assert len(rows) == 3
        assert [r["cached"] for r in rows] == [False, True, False]
        assert all(r["model"] == "openai/gpt-oss-20b" for r in rows)

    def test_missing_api_key_raises_before_any_network_call(self, tmp_path, monkeypatch):
        client = make_client(tmp_path)
        calls = []
        monkeypatch.setattr(client._session, "post", lambda url, **kw: calls.append(kw) or FakeResponse("x"))
        monkeypatch.delenv("GROQ_API_KEY", raising=False)

        with pytest.raises(MissingAPIKeyError):
            client.complete_with_usage([{"role": "user", "content": "Hi"}])
        assert calls == []


class TestCompleteWithUsageBypassCache:
    def test_bypassed_call_is_never_cached_true(self, tmp_path, monkeypatch):
        client = make_client(tmp_path)
        monkeypatch.setattr(
            client._session,
            "post",
            lambda url, **kw: FakeResponse("live answer", usage={"prompt_tokens": 5, "completion_tokens": 2}),
        )
        monkeypatch.setenv("GROQ_API_KEY", "test-key")

        result = client.complete_with_usage([{"role": "user", "content": "Hi"}], bypass_cache=True)

        assert result.cached is False
        assert result.text == "live answer"
        assert result.latency_s is not None and result.latency_s >= 0

    def test_repeated_bypassed_calls_always_hit_the_network(self, tmp_path, monkeypatch):
        call_count = 0

        def fake_post(url, **kw):
            nonlocal call_count
            call_count += 1
            return FakeResponse("live answer", usage={"prompt_tokens": 5, "completion_tokens": 2})

        client = make_client(tmp_path)
        monkeypatch.setattr(client._session, "post", fake_post)
        monkeypatch.setenv("GROQ_API_KEY", "test-key")

        messages = [{"role": "user", "content": "Hi"}]
        client.complete_with_usage(messages, bypass_cache=True)
        client.complete_with_usage(messages, bypass_cache=True)

        assert call_count == 2

    def test_bypassed_call_writes_nothing_to_the_disk_cache(self, tmp_path, monkeypatch):
        cache_dir = tmp_path / "cache"
        client = make_client(tmp_path, cache_dir=str(cache_dir))
        monkeypatch.setattr(
            client._session,
            "post",
            lambda url, **kw: FakeResponse("live answer", usage={"prompt_tokens": 5, "completion_tokens": 2}),
        )
        monkeypatch.setenv("GROQ_API_KEY", "test-key")

        client.complete_with_usage([{"role": "user", "content": "Hi"}], bypass_cache=True)

        response_files = list(cache_dir.glob("*.json")) if cache_dir.is_dir() else []
        assert response_files == []

    def test_bypass_does_not_read_an_existing_cache_entry(self, tmp_path, monkeypatch):
        # A question already cached via a normal (non-bypassed) call must still be
        # re-fetched live when bypass_cache=True -- that is the entire point.
        client = make_client(tmp_path)
        monkeypatch.setenv("GROQ_API_KEY", "test-key")
        messages = [{"role": "user", "content": "Hi"}]

        monkeypatch.setattr(
            client._session, "post", lambda url, **kw: FakeResponse("cached answer", usage={"prompt_tokens": 1, "completion_tokens": 1})
        )
        client.complete_with_usage(messages)  # populates the cache

        call_count = 0

        def fake_post(url, **kw):
            nonlocal call_count
            call_count += 1
            return FakeResponse("fresh live answer", usage={"prompt_tokens": 9, "completion_tokens": 9})

        monkeypatch.setattr(client._session, "post", fake_post)
        result = client.complete_with_usage(messages, bypass_cache=True)

        assert call_count == 1
        assert result.cached is False
        assert result.text == "fresh live answer"
