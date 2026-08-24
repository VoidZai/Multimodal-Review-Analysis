"""Unit tests for cragb.multimodal.photo_store (T6.1; M6.md T6.1).

No real network access: `PhotoStore._session.get` is monkeypatched to a
fake context-manager response, exactly like `test_api_clients.py` fakes
`GroqClient._session.post`. Covers: content-addressing, the four failure
statuses (`http_error`, `timeout`, `not_an_image`, `too_large`), the
cache-hit path (assert exactly one HTTP call for a repeated URL), manifest
aggregation, `load_photo_bytes`/`to_data_part` round-tripping, and
`collect_candidate_urls`'s filtering over a small fixture (image-target
questions only, has-image docs only, first URL per doc, deduplicated).
"""

from __future__ import annotations

import base64
import io
import json

import pandas as pd
import pytest
import requests
from PIL import Image

from cragb.multimodal.photo_store import (
    PhotoStore,
    build_manifest,
    collect_candidate_urls,
    load_manifest,
    photo_id,
    save_manifest,
)


def _jpeg_bytes(size: tuple[int, int] = (8, 6), color: str = "red") -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color=color).save(buf, format="JPEG")
    return buf.getvalue()


class FakeResponse:
    """Minimal stand-in for `requests.Response` used as a context manager."""

    def __init__(self, content: bytes, status_code: int = 200):
        self._content = content
        self.status_code = status_code

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def iter_content(self, chunk_size: int = 65536):
        for i in range(0, len(self._content), chunk_size):
            yield self._content[i : i + chunk_size]


def make_store(tmp_path, **overrides) -> PhotoStore:
    kwargs = dict(photos_dir=str(tmp_path / "photos"))
    kwargs.update(overrides)
    return PhotoStore(**kwargs)


class TestPhotoId:
    def test_deterministic(self):
        assert photo_id("https://x/1.jpg") == photo_id("https://x/1.jpg")

    def test_distinct_urls_differ(self):
        assert photo_id("https://x/1.jpg") != photo_id("https://x/2.jpg")

    def test_is_16_hex_chars(self):
        pid = photo_id("https://x/1.jpg")
        assert len(pid) == 16
        int(pid, 16)  # raises ValueError if not hex


class TestFetchPhotoOk:
    def test_valid_jpeg_is_cached_and_recorded(self, tmp_path, monkeypatch):
        store = make_store(tmp_path)
        content = _jpeg_bytes((10, 5))
        monkeypatch.setattr(store._session, "get", lambda url, **kw: FakeResponse(content))

        record = store.fetch_photo("https://example.com/a.jpg")

        assert record.status == "ok"
        assert record.photo_id == photo_id("https://example.com/a.jpg")
        assert record.mime == "image/jpeg"
        assert record.width == 10
        assert record.height == 5
        assert record.bytes_len == len(content)
        cached = list((tmp_path / "photos").glob("*.jpg"))
        assert len(cached) == 1
        assert cached[0].read_bytes() == content

    def test_second_call_is_a_cache_hit_no_new_request(self, tmp_path, monkeypatch):
        store = make_store(tmp_path)
        content = _jpeg_bytes()
        calls = []

        def fake_get(url, **kw):
            calls.append(url)
            return FakeResponse(content)

        monkeypatch.setattr(store._session, "get", fake_get)

        first = store.fetch_photo("https://example.com/a.jpg")
        second = store.fetch_photo("https://example.com/a.jpg")

        assert len(calls) == 1
        assert first.status == second.status == "ok"
        assert first.photo_id == second.photo_id

    def test_force_redownloads(self, tmp_path, monkeypatch):
        store = make_store(tmp_path)
        content = _jpeg_bytes()
        calls = []
        monkeypatch.setattr(store._session, "get", lambda url, **kw: calls.append(1) or FakeResponse(content))

        store.fetch_photo("https://example.com/a.jpg")
        store.fetch_photo("https://example.com/a.jpg", force=True)

        assert len(calls) == 2


class TestFetchPhotoFailureModes:
    def test_http_error_status_no_file_written(self, tmp_path, monkeypatch):
        store = make_store(tmp_path)
        monkeypatch.setattr(store._session, "get", lambda url, **kw: FakeResponse(b"not found", status_code=404))

        record = store.fetch_photo("https://example.com/missing.jpg")

        assert record.status == "http_error"
        assert record.error == "HTTP 404"
        assert record.path is None
        assert list((tmp_path / "photos").glob("*")) == []

    def test_html_body_returned_as_200_is_not_an_image(self, tmp_path, monkeypatch):
        store = make_store(tmp_path)
        html = b"<html><body>rate limited</body></html>"
        monkeypatch.setattr(store._session, "get", lambda url, **kw: FakeResponse(html, status_code=200))

        record = store.fetch_photo("https://example.com/fake.jpg")

        assert record.status == "not_an_image"
        assert record.path is None
        assert list((tmp_path / "photos").glob("*")) == []

    def test_body_over_max_bytes_is_too_large_without_full_buffering(self, tmp_path, monkeypatch):
        store = make_store(tmp_path, max_bytes=100)
        big = _jpeg_bytes((200, 200))  # comfortably over 100 bytes
        assert len(big) > 100
        monkeypatch.setattr(store._session, "get", lambda url, **kw: FakeResponse(big))

        record = store.fetch_photo("https://example.com/huge.jpg")

        assert record.status == "too_large"
        assert record.path is None
        assert list((tmp_path / "photos").glob("*")) == []

    def test_timeout_is_recorded_not_raised(self, tmp_path, monkeypatch):
        store = make_store(tmp_path)

        def raise_timeout(url, **kw):
            raise requests.Timeout("read timed out")

        monkeypatch.setattr(store._session, "get", raise_timeout)

        record = store.fetch_photo("https://example.com/slow.jpg")

        assert record.status == "timeout"
        assert record.path is None

    def test_connection_error_is_http_error_not_raised(self, tmp_path, monkeypatch):
        store = make_store(tmp_path)

        def raise_conn_error(url, **kw):
            raise requests.ConnectionError("refused")

        monkeypatch.setattr(store._session, "get", raise_conn_error)

        record = store.fetch_photo("https://example.com/down.jpg")

        assert record.status == "http_error"


class TestFetchMany:
    def test_deduplicates_and_returns_one_record_per_unique_url(self, tmp_path, monkeypatch):
        store = make_store(tmp_path, request_delay_s=0.0)
        content = _jpeg_bytes()
        monkeypatch.setattr(store._session, "get", lambda url, **kw: FakeResponse(content))

        urls = ["https://x/a.jpg", "https://x/b.jpg", "https://x/a.jpg"]
        records = store.fetch_many(urls)

        assert len(records) == 2
        assert {r.url for r in records} == {"https://x/a.jpg", "https://x/b.jpg"}

    def test_mixed_outcomes_all_reported(self, tmp_path, monkeypatch):
        store = make_store(tmp_path, request_delay_s=0.0)

        def fake_get(url, **kw):
            if "ok" in url:
                return FakeResponse(_jpeg_bytes())
            return FakeResponse(b"nope", status_code=404)

        monkeypatch.setattr(store._session, "get", fake_get)

        records = store.fetch_many(["https://x/ok1.jpg", "https://x/bad1.jpg", "https://x/ok2.jpg"])

        statuses = sorted(r.status for r in records)
        assert statuses == ["http_error", "ok", "ok"]


class TestLoadPhotoBytesAndDataPart:
    def test_round_trip(self, tmp_path, monkeypatch):
        store = make_store(tmp_path)
        content = _jpeg_bytes()
        monkeypatch.setattr(store._session, "get", lambda url, **kw: FakeResponse(content))
        record = store.fetch_photo("https://example.com/a.jpg")

        loaded = store.load_photo_bytes(record.photo_id)
        assert loaded == content

        part = store.to_data_part(record.photo_id)
        assert part["type"] == "image"
        assert part["photo_id"] == record.photo_id
        assert part["mime"] == "image/jpeg"
        assert base64.b64decode(part["data_b64"]) == content

    def test_missing_photo_id_raises(self, tmp_path):
        store = make_store(tmp_path)
        with pytest.raises(FileNotFoundError):
            store.load_photo_bytes("deadbeefdeadbeef")
        with pytest.raises(FileNotFoundError):
            store.to_data_part("deadbeefdeadbeef")


class TestManifest:
    def test_counts_match_records(self, tmp_path, monkeypatch):
        store = make_store(tmp_path, request_delay_s=0.0)

        def fake_get(url, **kw):
            if "ok" in url:
                return FakeResponse(_jpeg_bytes())
            if "404" in url:
                return FakeResponse(b"x", status_code=404)
            return FakeResponse(b"<html>nope</html>")

        monkeypatch.setattr(store._session, "get", fake_get)
        records = store.fetch_many(
            ["https://x/ok1.jpg", "https://x/ok2.jpg", "https://x/404.jpg", "https://x/junk.jpg"]
        )

        manifest = build_manifest(records, config_path="configs/photo_store.yaml")

        assert manifest["n_attempted"] == 4
        assert manifest["n_ok"] == 2
        assert manifest["n_failed_by_status"] == {"http_error": 1, "not_an_image": 1}
        assert len(manifest["entries"]) == 4

        # n_ok must equal what's actually on disk -- a manifest that
        # disagrees with the filesystem is worse than no manifest.
        files_on_disk = list((tmp_path / "photos").glob("*"))
        assert len(files_on_disk) == manifest["n_ok"]

    def test_save_and_load_round_trip(self, tmp_path):
        manifest = build_manifest(
            [], config_path="configs/photo_store.yaml"
        )
        out = tmp_path / "manifest.json"
        save_manifest(out, manifest)
        loaded = load_manifest(out)
        assert loaded["n_attempted"] == 0
        assert loaded["config_path"] == "configs/photo_store.yaml"


class TestCollectCandidateUrls:
    def _write_jsonl(self, path, rows):
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row))
                f.write("\n")

    def test_filters_to_image_target_pool_has_image_first_url(self, tmp_path):
        questions_path = tmp_path / "questions.jsonl"
        pools_path = tmp_path / "pools.jsonl"
        corpus_path = tmp_path / "corpus.parquet"

        self._write_jsonl(
            questions_path,
            [
                {"id": "q_img", "image_target": True},
                {"id": "q_text", "image_target": False},
            ],
        )
        self._write_jsonl(
            pools_path,
            [
                {"question_id": "q_img", "doc_ids": ["0", "1", "2"]},
                # q_text is not image_target -- its pool must be excluded
                # even though it references doc 3 (which has an image).
                {"question_id": "q_text", "doc_ids": ["3"]},
            ],
        )
        corpus = pd.DataFrame(
            {
                "has_image": [True, False, True, True],
                "image_urls": [
                    ["https://x/0-a.jpg", "https://x/0-b.jpg"],
                    [],
                    ["https://x/2-a.jpg"],
                    ["https://x/3-a.jpg"],
                ],
            }
        )
        corpus.to_parquet(corpus_path)

        urls = collect_candidate_urls(str(questions_path), str(pools_path), str(corpus_path))

        # doc 0: has_image, first URL only. doc 1: not has_image, excluded.
        # doc 2: has_image, included. doc 3: only reachable via q_text's
        # pool (not image_target), excluded.
        assert urls == sorted(["https://x/0-a.jpg", "https://x/2-a.jpg"])

    def test_dedupes_urls_shared_across_docs(self, tmp_path):
        questions_path = tmp_path / "questions.jsonl"
        pools_path = tmp_path / "pools.jsonl"
        corpus_path = tmp_path / "corpus.parquet"

        self._write_jsonl(questions_path, [{"id": "q1", "image_target": True}])
        self._write_jsonl(pools_path, [{"question_id": "q1", "doc_ids": ["0", "1"]}])
        corpus = pd.DataFrame(
            {
                "has_image": [True, True],
                "image_urls": [["https://x/same.jpg"], ["https://x/same.jpg"]],
            }
        )
        corpus.to_parquet(corpus_path)

        urls = collect_candidate_urls(str(questions_path), str(pools_path), str(corpus_path))

        assert urls == ["https://x/same.jpg"]
