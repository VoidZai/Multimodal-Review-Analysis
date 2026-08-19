"""Unit tests for cragb.generate.api_clients.GroqClient (T4b.4; M4b.md T4b.4).

No real network access: `requests.Session.post` is monkeypatched on the client's own
`_session` instance to a fake that records the JSON payload it was called with and
returns a canned successful response. This is the first test file for `GroqClient`
itself -- added alongside T4b.4's `reasoning_effort` field specifically to lock down a
guarantee the judge config leans on: adding that field must not change the request
payload (and therefore the disk-cache key, `cragb.generate.api_cache._cache_key`) for any
existing caller that never sets it, or every response already cached under T4a/T4b.1-3
silently stops being a cache hit.
"""

from __future__ import annotations

import pytest

from cragb.generate.api_clients import GroqClient, MissingAPIKeyError


class FakeResponse:
    def __init__(self, content: str, status_code: int = 200):
        self._content = content
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self) -> dict:
        return {"choices": [{"message": {"content": self._content}}]}


def make_client(tmp_path, **overrides) -> GroqClient:
    kwargs = dict(model="openai/gpt-oss-20b", cache_dir=str(tmp_path / "cache"))
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
