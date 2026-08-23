"""Unit tests for cragb.generate.api_cache.

Covers: a cache miss calls through to the underlying function and
persists the result; a cache hit returns the stored response without
re-invoking the function (the whole point of disk-caching API calls,
per PLAN.md §5 / §1.4 bottleneck #3); distinct payloads get distinct
entries; the cache survives across separate `DiskCache` instances
pointed at the same directory, since a real run and a later re-run are
different Python processes.

`TestCallWithMeta` (T5.2; M5.md T5.2) covers the metadata-sidecar
extension: a miss persists both response and meta and invokes `fn` once;
a hit returns the stored meta without calling `fn`; a response cached via
plain `call()`/`complete()` (no sidecar ever written) is a hit through
`call_with_meta` too, with `meta=None` rather than an error — the
compatibility guarantee T5.4 relies on to recover historical token counts.
"""

from __future__ import annotations

from cragb.generate.api_cache import DiskCache


class TestDiskCache:
    def test_miss_then_hit_calls_underlying_fn_once(self, tmp_path):
        cache = DiskCache(tmp_path)
        calls = []

        def fn():
            calls.append(1)
            return "response-a"

        first = cache.call({"prompt": "x"}, fn)
        second = cache.call({"prompt": "x"}, fn)

        assert first == "response-a"
        assert second == "response-a"
        assert len(calls) == 1

    def test_different_payloads_get_different_entries(self, tmp_path):
        cache = DiskCache(tmp_path)
        cache.call({"prompt": "a"}, lambda: "resp-a")
        cache.call({"prompt": "b"}, lambda: "resp-b")

        assert cache.get({"prompt": "a"}) == "resp-a"
        assert cache.get({"prompt": "b"}) == "resp-b"

    def test_cache_persists_across_instances(self, tmp_path):
        DiskCache(tmp_path).call({"prompt": "x"}, lambda: "resp")
        reloaded = DiskCache(tmp_path)
        assert reloaded.get({"prompt": "x"}) == "resp"

    def test_get_on_missing_key_returns_none(self, tmp_path):
        cache = DiskCache(tmp_path)
        assert cache.get({"prompt": "never called"}) is None

    def test_key_is_order_independent(self, tmp_path):
        cache = DiskCache(tmp_path)
        cache.call({"a": 1, "b": 2}, lambda: "resp")
        assert cache.get({"b": 2, "a": 1}) == "resp"


class TestCallWithMeta:
    def test_miss_persists_response_and_meta_and_calls_fn_once(self, tmp_path):
        cache = DiskCache(tmp_path)
        calls = []

        def fn():
            calls.append(1)
            return "response-a", {"prompt_tokens": 10, "completion_tokens": 5}

        response, meta, cached = cache.call_with_meta({"prompt": "x"}, fn)

        assert response == "response-a"
        assert meta == {"prompt_tokens": 10, "completion_tokens": 5}
        assert cached is False
        assert len(calls) == 1

    def test_hit_returns_stored_meta_without_calling_fn(self, tmp_path):
        cache = DiskCache(tmp_path)
        cache.call_with_meta(
            {"prompt": "x"}, lambda: ("response-a", {"prompt_tokens": 10, "completion_tokens": 5})
        )

        calls = []

        def fn_should_not_run():
            calls.append(1)
            return "should-not-happen", {}

        response, meta, cached = cache.call_with_meta({"prompt": "x"}, fn_should_not_run)

        assert response == "response-a"
        assert meta == {"prompt_tokens": 10, "completion_tokens": 5}
        assert cached is True
        assert calls == []

    def test_hit_on_entry_with_no_sidecar_reports_meta_none(self, tmp_path):
        # Simulates a response cached before T5.2 (via plain call()/complete()),
        # which never wrote a *.meta.json sidecar.
        cache = DiskCache(tmp_path)
        cache.call({"prompt": "x"}, lambda: "pre-existing-response")

        response, meta, cached = cache.call_with_meta(
            {"prompt": "x"}, lambda: (_ for _ in ()).throw(AssertionError("fn must not run on a hit"))
        )

        assert response == "pre-existing-response"
        assert meta is None
        assert cached is True

    def test_get_meta_on_missing_key_returns_none(self, tmp_path):
        cache = DiskCache(tmp_path)
        assert cache.get_meta({"prompt": "never called"}) is None
