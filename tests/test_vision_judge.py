"""Unit tests for cragb.multimodal.vision_judge (T6.4; M6.md T6.4).

No real network access anywhere in this file: `judge_pair` takes an injected
`chat_fn` (mirrors `cragb.eval.judge.score_answer`'s pattern), and the one
place real photo bytes are needed (`PhotoStore.to_data_part`), a real
`PhotoStore` is fetched once via a monkeypatched `_session.get` (the same
fake-transport pattern `test_photo_store.py` uses).

Covers, per M6.md T6.4's validation checks: the parser accepts a bare JSON
object, a fenced ```json block, and a `<think>...</think>`-prefixed
response; rejects two JSON objects, prose-only text, and a response missing
`winner`, each with a typed `VisionJudgeParseError`; `judge_pair` issues
exactly two calls with the photo positions genuinely swapped (asserted on
the recorded payloads' image parts, not just the call count); and a stubbed
judge that always answers `"A"` yields `order_agreement=False` and a tie
for every pair.
"""

from __future__ import annotations

import io
import json
from string import Template

import pytest
from PIL import Image

from cragb.multimodal.photo_store import PhotoStore
from cragb.multimodal.vision_judge import (
    PairVerdict,
    VisionJudgeParseError,
    VisionVerdict,
    build_vision_prompt,
    judge_pair,
    parse_vision_response,
)

TEMPLATE = Template(
    "Question: $question\n\nPhoto A:\n[[PHOTO_A]]\n\nPhoto B:\n[[PHOTO_B]]\n\nRespond with JSON."
)


def _jpeg_bytes(color: str = "green") -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (5, 5), color=color).save(buf, format="JPEG")
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


def make_store_with_two_photos(tmp_path, monkeypatch) -> tuple[PhotoStore, str, str]:
    """A real PhotoStore with two distinct cached photos (network mocked)."""
    store = PhotoStore(photos_dir=str(tmp_path / "photos"), request_delay_s=0.0)

    def fake_get(url, **kw):
        color = "green" if "surfaced" in url else "orange"
        return FakeResponse(_jpeg_bytes(color))

    monkeypatch.setattr(store._session, "get", fake_get)
    surfaced = store.fetch_photo("https://x/surfaced.jpg")
    control = store.fetch_photo("https://x/control.jpg")
    assert surfaced.status == "ok" and control.status == "ok"
    return store, surfaced.photo_id, control.photo_id


# --------------------------------------------------------------------------
# build_vision_prompt
# --------------------------------------------------------------------------


class TestBuildVisionPrompt:
    def test_interleaves_text_and_photo_parts_in_order(self):
        photo_a = {"type": "image", "mime": "image/jpeg", "data_b64": "AAAA"}
        photo_b = {"type": "image", "mime": "image/jpeg", "data_b64": "BBBB"}

        parts = build_vision_prompt("Is it warm?", photo_a, photo_b, TEMPLATE)

        kinds = [p["type"] for p in parts]
        assert kinds == ["text", "image", "text", "image", "text"]
        assert parts[1] == photo_a
        assert parts[3] == photo_b
        assert "Is it warm?" in parts[0]["text"]

    def test_missing_marker_raises(self):
        bad_template = Template("Question: $question\n\nPhoto A:\n[[PHOTO_A]]\n(no B marker)")
        photo = {"type": "image", "mime": "image/jpeg", "data_b64": "AAAA"}
        with pytest.raises(ValueError, match=r"\[\[PHOTO_B\]\]"):
            build_vision_prompt("q", photo, photo, bad_template)


# --------------------------------------------------------------------------
# parse_vision_response
# --------------------------------------------------------------------------


class TestParseVisionResponse:
    def test_bare_json_object(self):
        raw = '{"winner": "A", "confidence": 4, "rationale": "Photo A shows the seam clearly."}'
        verdict = parse_vision_response(raw)
        assert verdict == VisionVerdict(winner="A", confidence=4, rationale="Photo A shows the seam clearly.")

    def test_fenced_json_block(self):
        raw = '```json\n{"winner": "B", "confidence": 3, "rationale": "B is closer to the claim."}\n```'
        verdict = parse_vision_response(raw)
        assert verdict.winner == "B"
        assert verdict.confidence == 3

    def test_think_prefixed_response_is_stripped(self):
        raw = (
            "<think>Photo A looks irrelevant, photo B shows the defect.</think>\n"
            '{"winner": "B", "confidence": 5, "rationale": "B shows the defect."}'
        )
        verdict = parse_vision_response(raw)
        assert verdict.winner == "B"
        assert verdict.confidence == 5

    def test_tie_is_a_valid_winner(self):
        raw = '{"winner": "tie", "confidence": 2, "rationale": "Neither photo is relevant."}'
        verdict = parse_vision_response(raw)
        assert verdict.winner == "tie"

    def test_two_json_objects_rejected(self):
        raw = '{"winner": "A", "confidence": 1, "rationale": "x"} {"winner": "B", "confidence": 2, "rationale": "y"}'
        with pytest.raises(VisionJudgeParseError):
            parse_vision_response(raw)

    def test_prose_only_rejected(self):
        with pytest.raises(VisionJudgeParseError):
            parse_vision_response("Photo A is clearly better evidence here.")

    def test_missing_winner_key_rejected(self):
        raw = '{"confidence": 3, "rationale": "no winner field"}'
        with pytest.raises(VisionJudgeParseError, match="winner"):
            parse_vision_response(raw)

    def test_invalid_winner_value_rejected(self):
        raw = '{"winner": "C", "confidence": 3, "rationale": "not A/B/tie"}'
        with pytest.raises(VisionJudgeParseError, match="winner"):
            parse_vision_response(raw)

    def test_confidence_out_of_range_rejected(self):
        raw = '{"winner": "A", "confidence": 9, "rationale": "too confident"}'
        with pytest.raises(VisionJudgeParseError, match="confidence"):
            parse_vision_response(raw)

    def test_confidence_bool_rejected(self):
        # bool is a subclass of int in Python -- must not silently pass as a score.
        raw = '{"winner": "A", "confidence": true, "rationale": "x"}'
        with pytest.raises(VisionJudgeParseError, match="confidence"):
            parse_vision_response(raw)

    def test_empty_rationale_rejected(self):
        raw = '{"winner": "A", "confidence": 3, "rationale": "   "}'
        with pytest.raises(VisionJudgeParseError, match="rationale"):
            parse_vision_response(raw)


# --------------------------------------------------------------------------
# judge_pair
# --------------------------------------------------------------------------


class TestJudgePair:
    def test_issues_exactly_two_calls_with_positions_genuinely_swapped(self, tmp_path, monkeypatch):
        store, surfaced_id, control_id = make_store_with_two_photos(tmp_path, monkeypatch)
        recorded_calls: list[list[dict]] = []

        def chat_fn(parts):
            recorded_calls.append(parts)
            return '{"winner": "A", "confidence": 3, "rationale": "ok"}'

        judge_pair("Is it warm?", surfaced_id, control_id, store, TEMPLATE, chat_fn)

        assert len(recorded_calls) == 2
        images_call_1 = [p for p in recorded_calls[0] if p["type"] == "image"]
        images_call_2 = [p for p in recorded_calls[1] if p["type"] == "image"]
        assert images_call_1[0]["data_b64"] == store.to_data_part(surfaced_id)["data_b64"]
        assert images_call_1[1]["data_b64"] == store.to_data_part(control_id)["data_b64"]
        # Second call: positions genuinely swapped, not just re-sent identically.
        assert images_call_2[0]["data_b64"] == store.to_data_part(control_id)["data_b64"]
        assert images_call_2[1]["data_b64"] == store.to_data_part(surfaced_id)["data_b64"]

    def test_surfaced_wins_both_orders_is_a_surfaced_win(self, tmp_path, monkeypatch):
        store, surfaced_id, control_id = make_store_with_two_photos(tmp_path, monkeypatch)
        responses = iter(
            [
                '{"winner": "A", "confidence": 4, "rationale": "surfaced (A) wins"}',  # call1: A=surfaced
                '{"winner": "B", "confidence": 4, "rationale": "surfaced (B) wins"}',  # call2: B=surfaced
            ]
        )

        result = judge_pair("q", surfaced_id, control_id, store, TEMPLATE, lambda parts: next(responses))

        assert result.outcome == "surfaced_win"
        assert result.order_agreement is True

    def test_control_wins_both_orders_is_a_control_win(self, tmp_path, monkeypatch):
        store, surfaced_id, control_id = make_store_with_two_photos(tmp_path, monkeypatch)
        responses = iter(
            [
                '{"winner": "B", "confidence": 4, "rationale": "control (B) wins"}',  # call1: B=control
                '{"winner": "A", "confidence": 4, "rationale": "control (A) wins"}',  # call2: A=control
            ]
        )

        result = judge_pair("q", surfaced_id, control_id, store, TEMPLATE, lambda parts: next(responses))

        assert result.outcome == "control_win"
        assert result.order_agreement is True

    def test_always_answers_a_yields_order_disagreement_and_tie(self, tmp_path, monkeypatch):
        # The sanity check M6.md's T6.4 spec calls for explicitly: a judge with a pure
        # positional bias (always picks whichever photo is in position A) must not be
        # able to manufacture a surfaced-photo win.
        store, surfaced_id, control_id = make_store_with_two_photos(tmp_path, monkeypatch)
        chat_fn = lambda parts: '{"winner": "A", "confidence": 5, "rationale": "always A"}'  # noqa: E731

        result = judge_pair("q", surfaced_id, control_id, store, TEMPLATE, chat_fn)

        assert result.order_agreement is False
        assert result.outcome == "tie"

    def test_explicit_tie_in_either_order_is_a_tie_outcome(self, tmp_path, monkeypatch):
        store, surfaced_id, control_id = make_store_with_two_photos(tmp_path, monkeypatch)
        responses = iter(
            [
                '{"winner": "A", "confidence": 3, "rationale": "surfaced (A) wins"}',
                '{"winner": "tie", "confidence": 1, "rationale": "genuinely tied"}',
            ]
        )

        result = judge_pair("q", surfaced_id, control_id, store, TEMPLATE, lambda parts: next(responses))

        assert result.outcome == "tie"

    def test_returns_both_raw_verdicts(self, tmp_path, monkeypatch):
        store, surfaced_id, control_id = make_store_with_two_photos(tmp_path, monkeypatch)
        responses = iter(
            [
                '{"winner": "A", "confidence": 4, "rationale": "first"}',
                '{"winner": "B", "confidence": 2, "rationale": "second"}',
            ]
        )

        result = judge_pair("q", surfaced_id, control_id, store, TEMPLATE, lambda parts: next(responses))

        assert isinstance(result, PairVerdict)
        assert result.verdict_surfaced_as_a.rationale == "first"
        assert result.verdict_surfaced_as_b.rationale == "second"
