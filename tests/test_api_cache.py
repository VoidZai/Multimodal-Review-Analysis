"""Unit tests for cragb.generate.api_cache.

Covers: a cache miss calls through to the underlying function and
persists the result; a cache hit returns the stored response without
re-invoking the function (the whole point of disk-caching API calls,
per PLAN.md §5 / §1.4 bottleneck #3); distinct payloads get distinct
entries; the cache survives across separate `DiskCache` instances
pointed at the same directory, since a real run and a later re-run are
different Python processes.
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
