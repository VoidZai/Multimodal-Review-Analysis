"""Unit tests for cragb.multimodal.photo_link (T6.3; M6.md T6.3).

No real network access: `PhotoStore._session.get` is monkeypatched to a fake
that resolves URLs containing `"good"` to a valid JPEG and everything else
to a 404 -- the same fake-transport pattern `test_photo_store.py` and
`test_api_clients.py` both use, letting `surfaced_photo`/`control_photo`
exercise `PhotoStore`'s real fetch/cache logic without hitting the network.

Covers, per M6.md T6.3's validation checks: the control photo is never
drawn from `relevant_ids` or the retrieved context; the same seed
reproduces byte-identical control draws across independent runs; a
question with no photo-bearing retrieved doc produces a row with
`drop_reason` set rather than being silently dropped; and the coverage
funnel's stage counts are monotonically non-increasing with the final
stage equal to the number of rows `write_pairs_jsonl` actually writes.
"""

from __future__ import annotations

import io
import json

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from cragb.multimodal.photo_link import (
    DROP_REASON_CONTROL_EXHAUSTED,
    DROP_REASON_NO_PHOTO_IN_CONTEXT,
    DROP_REASON_NO_TRANSCRIPT,
    DROP_REASON_SURFACED_UNFETCHABLE,
    CragbQuestion,
    RagContext,
    build_coverage_funnel,
    build_pairs,
    control_photo,
    load_cragb_questions,
    load_rag_small_context,
    surfaced_photo,
    write_pairs_jsonl,
)
from cragb.multimodal.photo_store import PhotoStore, photo_id


def _jpeg_bytes(size: tuple[int, int] = (6, 4), color: str = "blue") -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color=color).save(buf, format="JPEG")
    return buf.getvalue()


class FakeResponse:
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
    kwargs = dict(photos_dir=str(tmp_path / "photos"), request_delay_s=0.0)
    kwargs.update(overrides)
    return PhotoStore(**kwargs)


def patch_good_bad_urls(monkeypatch, store: PhotoStore) -> None:
    """URLs containing "good" resolve to a real JPEG; everything else 404s."""

    def fake_get(url, **kw):
        if "good" in url:
            return FakeResponse(_jpeg_bytes())
        return FakeResponse(b"not found", status_code=404)

    monkeypatch.setattr(store._session, "get", fake_get)


def make_corpus(rows: dict[int, dict]) -> pd.DataFrame:
    """Build a minimal corpus_v1-shaped DataFrame indexed by int doc_id."""
    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = None
    return df.sort_index()


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


class TestLoadCragbQuestions:
    def test_parses_fields(self, tmp_path):
        path = tmp_path / "cragb_v1.jsonl"
        path.write_text(
            json.dumps(
                {
                    "id": "fit_sizing_000",
                    "type": "fit_sizing",
                    "question": "Do these run true to size?",
                    "relevant_ids": ["1", "2", "3"],
                    "image_target": True,
                }
            )
            + "\n",
            encoding="utf-8",
        )

        questions = load_cragb_questions(path)

        assert len(questions) == 1
        q = questions[0]
        assert q.id == "fit_sizing_000"
        assert q.relevant_ids == frozenset({"1", "2", "3"})
        assert q.image_target is True

    def test_skips_blank_lines(self, tmp_path):
        path = tmp_path / "cragb_v1.jsonl"
        path.write_text(
            "\n"
            + json.dumps(
                {"id": "q1", "type": "t", "question": "?", "relevant_ids": [], "image_target": False}
            )
            + "\n\n",
            encoding="utf-8",
        )
        assert len(load_cragb_questions(path)) == 1


class TestLoadRagSmallContext:
    def test_parses_context_doc_ids_and_flags(self, tmp_path):
        path = tmp_path / "transcripts.jsonl"
        path.write_text(
            json.dumps(
                {
                    "question_id": "q1",
                    "context_doc_ids": ["10", "20", "30"],
                    "context_photo_flags": {"10": False, "20": True, "30": False},
                }
            )
            + "\n",
            encoding="utf-8",
        )

        contexts = load_rag_small_context(path)

        assert "q1" in contexts
        assert contexts["q1"].context_doc_ids == ("10", "20", "30")
        assert contexts["q1"].context_photo_flags == {"10": False, "20": True, "30": False}


# --------------------------------------------------------------------------
# surfaced_photo
# --------------------------------------------------------------------------


class TestSurfacedPhoto:
    def test_takes_first_photo_flagged_doc_in_rank_order(self, tmp_path, monkeypatch):
        store = make_store(tmp_path)
        patch_good_bad_urls(monkeypatch, store)
        corpus = make_corpus(
            {
                1: {"has_image": False, "image_urls": []},
                2: {"has_image": True, "image_urls": ["https://x/good-2.jpg"]},
                3: {"has_image": True, "image_urls": ["https://x/good-3.jpg"]},
            }
        )
        context = RagContext(
            context_doc_ids=("1", "2", "3"),
            context_photo_flags={"1": False, "2": True, "3": True},
        )

        result = surfaced_photo(context, corpus, store)

        assert result.status == "ok"
        assert result.doc_id == "2"
        assert result.photo_id == photo_id("https://x/good-2.jpg")

    def test_skips_unfetchable_candidate_for_next_photo_bearing_doc(self, tmp_path, monkeypatch):
        store = make_store(tmp_path)
        patch_good_bad_urls(monkeypatch, store)
        corpus = make_corpus(
            {
                2: {"has_image": True, "image_urls": ["https://x/bad-2.jpg"]},
                3: {"has_image": True, "image_urls": ["https://x/good-3.jpg"]},
            }
        )
        context = RagContext(
            context_doc_ids=("2", "3"),
            context_photo_flags={"2": True, "3": True},
        )

        result = surfaced_photo(context, corpus, store)

        assert result.status == "ok"
        assert result.doc_id == "3"

    def test_all_photo_candidates_unfetchable(self, tmp_path, monkeypatch):
        store = make_store(tmp_path)
        patch_good_bad_urls(monkeypatch, store)
        corpus = make_corpus({2: {"has_image": True, "image_urls": ["https://x/bad-2.jpg"]}})
        context = RagContext(context_doc_ids=("2",), context_photo_flags={"2": True})

        result = surfaced_photo(context, corpus, store)

        assert result.status == "unfetchable"
        assert result.doc_id is None

    def test_no_photo_in_context(self, tmp_path, monkeypatch):
        store = make_store(tmp_path)
        patch_good_bad_urls(monkeypatch, store)
        corpus = make_corpus({1: {"has_image": False, "image_urls": []}})
        context = RagContext(context_doc_ids=("1",), context_photo_flags={"1": False})

        result = surfaced_photo(context, corpus, store)

        assert result.status == "no_photo_in_context"

    def test_top_k_caps_scan_depth(self, tmp_path, monkeypatch):
        store = make_store(tmp_path)
        patch_good_bad_urls(monkeypatch, store)
        corpus = make_corpus({2: {"has_image": True, "image_urls": ["https://x/good-2.jpg"]}})
        context = RagContext(context_doc_ids=("1", "2"), context_photo_flags={"1": False, "2": True})

        capped = surfaced_photo(context, corpus, store, top_k=1)
        uncapped = surfaced_photo(context, corpus, store, top_k=None)

        assert capped.status == "no_photo_in_context"
        assert uncapped.status == "ok"


# --------------------------------------------------------------------------
# control_photo
# --------------------------------------------------------------------------


class TestControlPhoto:
    def _corpus(self, n: int = 20) -> pd.DataFrame:
        return make_corpus(
            {i: {"has_image": True, "image_urls": [f"https://x/good-{i}.jpg"]} for i in range(n)}
        )

    def test_never_draws_from_excluded_ids(self, tmp_path, monkeypatch):
        store = make_store(tmp_path)
        patch_good_bad_urls(monkeypatch, store)
        corpus = self._corpus(20)
        exclude = frozenset(str(i) for i in range(15))  # only 15-19 remain eligible

        for seed in range(10):
            rng = np.random.default_rng(seed)
            result = control_photo(exclude, corpus, store, rng)
            assert result.status == "ok"
            assert result.doc_id not in exclude

    def test_same_seed_reproduces_identical_control(self, tmp_path, monkeypatch):
        store = make_store(tmp_path)
        patch_good_bad_urls(monkeypatch, store)
        corpus = self._corpus(30)
        exclude = frozenset({"0", "1"})

        first = control_photo(exclude, corpus, store, np.random.default_rng(42))
        second = control_photo(exclude, corpus, store, np.random.default_rng(42))

        assert first.status == second.status == "ok"
        assert first.doc_id == second.doc_id
        assert first.photo_id == second.photo_id

    def test_different_seeds_can_diverge(self, tmp_path, monkeypatch):
        store = make_store(tmp_path)
        patch_good_bad_urls(monkeypatch, store)
        corpus = self._corpus(30)
        exclude = frozenset()

        results = {control_photo(exclude, corpus, store, np.random.default_rng(s)).doc_id for s in range(5)}
        assert len(results) > 1

    def test_exhausted_when_every_candidate_unfetchable(self, tmp_path, monkeypatch):
        store = make_store(tmp_path)

        def all_bad(url, **kw):
            return FakeResponse(b"nope", status_code=404)

        monkeypatch.setattr(store._session, "get", all_bad)
        corpus = self._corpus(5)

        result = control_photo(frozenset(), corpus, store, np.random.default_rng(1), max_attempts=5)

        assert result.status == "exhausted"
        assert result.doc_id is None

    def test_no_eligible_candidates(self, tmp_path, monkeypatch):
        store = make_store(tmp_path)
        patch_good_bad_urls(monkeypatch, store)
        corpus = self._corpus(3)
        exclude = frozenset({"0", "1", "2"})

        result = control_photo(exclude, corpus, store, np.random.default_rng(1))

        assert result.status == "no_eligible_candidates"


# --------------------------------------------------------------------------
# build_pairs / build_coverage_funnel / write_pairs_jsonl
# --------------------------------------------------------------------------


def _question(id_, type_="fit_sizing", relevant_ids=(), image_target=True) -> CragbQuestion:
    return CragbQuestion(
        id=id_, type=type_, question=f"question {id_}?", relevant_ids=frozenset(relevant_ids), image_target=image_target
    )


class TestBuildPairs:
    def test_non_image_target_questions_produce_no_row(self, tmp_path, monkeypatch):
        store = make_store(tmp_path)
        patch_good_bad_urls(monkeypatch, store)
        corpus = make_corpus({1: {"has_image": True, "image_urls": ["https://x/good-1.jpg"]}})
        questions = [_question("q1", image_target=False)]

        pairs = build_pairs(questions, {}, corpus, store, np.random.default_rng(1))

        assert len(pairs) == 0

    def test_missing_transcript_is_a_reported_drop_not_a_crash(self, tmp_path, monkeypatch):
        store = make_store(tmp_path)
        patch_good_bad_urls(monkeypatch, store)
        corpus = make_corpus({1: {"has_image": True, "image_urls": ["https://x/good-1.jpg"]}})
        questions = [_question("q1")]

        pairs = build_pairs(questions, {}, corpus, store, np.random.default_rng(1))

        assert len(pairs) == 1
        assert pairs.iloc[0]["drop_reason"] == DROP_REASON_NO_TRANSCRIPT

    def test_no_photo_in_context_is_a_reported_drop(self, tmp_path, monkeypatch):
        store = make_store(tmp_path)
        patch_good_bad_urls(monkeypatch, store)
        corpus = make_corpus({1: {"has_image": False, "image_urls": []}})
        questions = [_question("q1")]
        contexts = {"q1": RagContext(context_doc_ids=("1",), context_photo_flags={"1": False})}

        pairs = build_pairs(questions, contexts, corpus, store, np.random.default_rng(1))

        assert pairs.iloc[0]["drop_reason"] == DROP_REASON_NO_PHOTO_IN_CONTEXT
        assert pairs.iloc[0]["surfaced_photo_id"] is None

    def test_surfaced_ok_but_control_exhausted_is_a_reported_drop(self, tmp_path, monkeypatch):
        store = make_store(tmp_path)

        def only_doc2_good(url, **kw):
            if "good-2" in url:
                return FakeResponse(_jpeg_bytes())
            return FakeResponse(b"nope", status_code=404)

        monkeypatch.setattr(store._session, "get", only_doc2_good)
        corpus = make_corpus(
            {
                2: {"has_image": True, "image_urls": ["https://x/good-2.jpg"]},
                3: {"has_image": True, "image_urls": ["https://x/bad-3.jpg"]},
            }
        )
        questions = [_question("q1")]
        contexts = {"q1": RagContext(context_doc_ids=("2",), context_photo_flags={"2": True})}

        pairs = build_pairs(
            questions, contexts, corpus, store, np.random.default_rng(1), max_control_attempts=5
        )

        row = pairs.iloc[0]
        assert row["drop_reason"] == DROP_REASON_CONTROL_EXHAUSTED
        assert row["surfaced_photo_id"] is not None  # surfaced still resolved
        assert row["control_photo_id"] is None

    def test_usable_row_has_no_drop_reason_and_both_photos_set(self, tmp_path, monkeypatch):
        store = make_store(tmp_path)
        patch_good_bad_urls(monkeypatch, store)
        corpus = make_corpus(
            {
                2: {"has_image": True, "image_urls": ["https://x/good-2.jpg"]},
                9: {"has_image": True, "image_urls": ["https://x/good-9.jpg"]},
            }
        )
        questions = [_question("q1", relevant_ids=("2",))]
        contexts = {"q1": RagContext(context_doc_ids=("2",), context_photo_flags={"2": True})}

        pairs = build_pairs(questions, contexts, corpus, store, np.random.default_rng(1))

        row = pairs.iloc[0]
        assert row["drop_reason"] is None
        assert row["surfaced_photo_id"] is not None
        assert row["control_photo_id"] is not None
        assert row["control_doc_id"] != "2"  # excluded via relevant_ids

    def test_control_never_drawn_from_relevant_ids_or_context_across_many_questions(self, tmp_path, monkeypatch):
        store = make_store(tmp_path)
        patch_good_bad_urls(monkeypatch, store)
        # 50 image-bearing docs; each question excludes a different half.
        corpus = make_corpus(
            {i: {"has_image": True, "image_urls": [f"https://x/good-{i}.jpg"]} for i in range(50)}
        )
        questions = []
        contexts = {}
        for qi in range(10):
            qid = f"q{qi}"
            relevant = {str(qi)}
            context_docs = tuple(str((qi + 10 + j) % 50) for j in range(3))
            questions.append(_question(qid, relevant_ids=relevant))
            contexts[qid] = RagContext(
                context_doc_ids=context_docs, context_photo_flags={d: (d == context_docs[0]) for d in context_docs}
            )

        pairs = build_pairs(questions, contexts, corpus, store, np.random.default_rng(7))

        usable = pairs[pairs["drop_reason"].isna()]
        assert len(usable) == 10  # every question should resolve given ample eligible docs
        for _, row in usable.iterrows():
            qi = int(row["question_id"][1:])
            excluded = {str(qi)} | {str((qi + 10 + j) % 50) for j in range(3)}
            assert row["control_doc_id"] not in excluded

    def test_same_seed_reproduces_byte_identical_pairs(self, tmp_path, monkeypatch):
        store = make_store(tmp_path)
        patch_good_bad_urls(monkeypatch, store)
        corpus = make_corpus(
            {i: {"has_image": True, "image_urls": [f"https://x/good-{i}.jpg"]} for i in range(20)}
        )
        questions = [_question("q1", relevant_ids=("0",)), _question("q2", relevant_ids=("1",))]
        contexts = {
            "q1": RagContext(context_doc_ids=("2",), context_photo_flags={"2": True}),
            "q2": RagContext(context_doc_ids=("3",), context_photo_flags={"3": True}),
        }

        first = build_pairs(questions, contexts, corpus, store, np.random.default_rng(99))
        second = build_pairs(questions, contexts, corpus, store, np.random.default_rng(99))

        pd.testing.assert_frame_equal(first, second)


class TestBuildCoverageFunnel:
    def test_stage_counts_monotonic_and_final_matches_pairs_written(self, tmp_path, monkeypatch):
        store = make_store(tmp_path)
        patch_good_bad_urls(monkeypatch, store)
        corpus = make_corpus(
            {
                1: {"has_image": False, "image_urls": []},
                2: {"has_image": True, "image_urls": ["https://x/good-2.jpg"]},
                9: {"has_image": True, "image_urls": ["https://x/good-9.jpg"]},
            }
        )
        all_questions = [
            _question("q1", image_target=False),  # not image_target -- excluded before stage 2
            _question("q2"),  # no transcript -> excluded at stage 3
            _question("q3"),  # no photo in context -> excluded at stage 3
            _question("q4", relevant_ids=("2",)),  # usable
        ]
        contexts = {
            "q3": RagContext(context_doc_ids=("1",), context_photo_flags={"1": False}),
            "q4": RagContext(context_doc_ids=("2",), context_photo_flags={"2": True}),
        }

        pairs = build_pairs(all_questions, contexts, corpus, store, np.random.default_rng(3))
        funnel = build_coverage_funnel(all_questions, pairs)

        counts = dict(zip(funnel["stage"], funnel["count"]))
        assert counts["total_questions"] == 4
        assert counts["image_target"] == 3
        assert counts["photo_in_context"] == 1
        assert counts["fetchable_bytes"] == 1
        assert counts["usable_pairs"] == 1

        # Monotonically non-increasing.
        values = list(funnel["count"])
        assert all(values[i] >= values[i + 1] for i in range(len(values) - 1))

        # Final stage equals what write_pairs_jsonl actually writes.
        out_path = tmp_path / "mm_pairs_v1.jsonl"
        n_written = write_pairs_jsonl(pairs, out_path)
        assert n_written == counts["usable_pairs"]


class TestWritePairsJsonl:
    def test_writes_only_usable_rows_without_drop_reason_column(self, tmp_path, monkeypatch):
        store = make_store(tmp_path)
        patch_good_bad_urls(monkeypatch, store)
        corpus = make_corpus(
            {
                1: {"has_image": False, "image_urls": []},
                2: {"has_image": True, "image_urls": ["https://x/good-2.jpg"]},
                9: {"has_image": True, "image_urls": ["https://x/good-9.jpg"]},
            }
        )
        questions = [_question("q1"), _question("q2", relevant_ids=("2",))]
        contexts = {
            "q1": RagContext(context_doc_ids=("1",), context_photo_flags={"1": False}),
            "q2": RagContext(context_doc_ids=("2",), context_photo_flags={"2": True}),
        }

        pairs = build_pairs(questions, contexts, corpus, store, np.random.default_rng(5))
        out_path = tmp_path / "mm_pairs_v1.jsonl"
        n_written = write_pairs_jsonl(pairs, out_path)

        assert n_written == 1
        lines = out_path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1
        row = json.loads(lines[0])
        assert "drop_reason" not in row
        assert row["question_id"] == "q2"
        assert row["control_photo_id"] is not None
