"""Unit tests for cragb.multimodal.vision_spotcheck (T6.6; M6.md T6.6), plus
cragb.eval.run_multimodal_pilot.plot_winrate (figure #6, colocated with the
win-rate data it visualizes -- see that module's docstring).

No real network access: photo files are fetched into a real `PhotoStore`
via a monkeypatched `_session.get` (the same fake-transport pattern every
other multimodal test file in this project uses), so `build_sample`'s
`PhotoStore.photo_path` calls resolve to genuine on-disk files without ever
touching the network.

Covers, per M6.md T6.6's validation checks: the worksheet round-trips
exactly (including a `notes:` value containing a colon, which a naive
"split on the first colon" parser would truncate); an unfilled
`my_verdict:` is flagged, not silently dropped; Cohen's kappa is 1.0 on a
hand-built perfect-agreement fixture and exactly 0.0 on a fixture
constructed so observed agreement equals chance-expected agreement; and
`plot_winrate`'s figure never recomputes what it plots -- it accepts (and
faithfully renders) numbers that wouldn't survive independent
recomputation, proving it only reads the DataFrame it's given.
"""

from __future__ import annotations

import io
import json

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from cragb.eval.run_multimodal_pilot import plot_winrate
from cragb.multimodal.photo_store import PhotoStore
from cragb.multimodal.vision_spotcheck import (
    SpotcheckRow,
    _apportion_quotas,
    build_paired_scores,
    build_sample,
    compute_agreement,
    export_worksheet,
    parse_worksheet,
    render_worksheet,
    score_worksheet,
)


def _jpeg_bytes(color: str = "navy") -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (4, 4), color=color).save(buf, format="JPEG")
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


def make_store_with_cached_photos(tmp_path, monkeypatch, n: int) -> tuple[PhotoStore, list[str]]:
    """A real PhotoStore with `n` distinct cached photos; returns their photo_ids."""
    store = PhotoStore(photos_dir=str(tmp_path / "photos"), request_delay_s=0.0)
    monkeypatch.setattr(store._session, "get", lambda url, **kw: FakeResponse(_jpeg_bytes()))
    photo_ids = [store.fetch_photo(f"https://x/p{i}.jpg").photo_id for i in range(n)]
    return store, photo_ids


def make_verdicts_pairs(n: int, photo_ids: list[str], outcomes: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    verdicts = pd.DataFrame(
        {
            "question_id": [f"q{i}" for i in range(n)],
            "type": ["fit_sizing" if i % 2 == 0 else "colour_appearance" for i in range(n)],
            "outcome": outcomes,
        }
    )
    pairs = pd.DataFrame(
        {
            "question_id": [f"q{i}" for i in range(n)],
            "question": [f"question {i}?" for i in range(n)],
            "surfaced_photo_id": [photo_ids[2 * i] for i in range(n)],
            "control_photo_id": [photo_ids[2 * i + 1] for i in range(n)],
        }
    )
    return verdicts, pairs


# --------------------------------------------------------------------------
# _apportion_quotas
# --------------------------------------------------------------------------


class TestApportionQuotas:
    def test_matches_hand_computed_largest_remainder(self):
        # total=25, n=15: exact = {sw:7.2, cw:1.8, tie:6.0} -> floor {7,1,6}=14,
        # remainder 1 goes to the largest fractional remainder (control_win, 0.8).
        quotas = _apportion_quotas(15, {"surfaced_win": 12, "control_win": 3, "tie": 10})
        assert quotas == {"surfaced_win": 7, "control_win": 2, "tie": 6}
        assert sum(quotas.values()) == 15

    def test_never_exceeds_a_groups_own_pool(self):
        counts = {"a": 1, "b": 2, "c": 20}
        quotas = _apportion_quotas(15, counts)
        for k, q in quotas.items():
            assert q <= counts[k]
        assert sum(quotas.values()) == 15

    def test_raises_when_n_exceeds_total_pool(self):
        with pytest.raises(ValueError, match="exceeds"):
            _apportion_quotas(10, {"a": 3, "b": 3})

    def test_equal_n_and_total_allocates_everything(self):
        quotas = _apportion_quotas(6, {"a": 2, "b": 4})
        assert quotas == {"a": 2, "b": 4}


# --------------------------------------------------------------------------
# build_sample
# --------------------------------------------------------------------------


class TestBuildSample:
    def test_sample_size_and_stratification(self, tmp_path, monkeypatch):
        n = 12
        store, photo_ids = make_store_with_cached_photos(tmp_path, monkeypatch, 2 * n)
        outcomes = ["surfaced_win"] * 8 + ["control_win"] * 2 + ["tie"] * 2
        verdicts, pairs = make_verdicts_pairs(n, photo_ids, outcomes)

        rows = build_sample(verdicts, pairs, 6, seed=1, photos_dir=str(tmp_path / "photos"))

        assert len(rows) == 6
        assert all(isinstance(r, SpotcheckRow) for r in rows)
        assert {r.row_id for r in rows} == {f"R{i:02d}" for i in range(1, 7)}

    def test_raises_on_empty_verdicts(self, tmp_path):
        empty = pd.DataFrame(columns=["question_id", "type", "outcome"])
        pairs = pd.DataFrame(columns=["question_id", "question", "surfaced_photo_id", "control_photo_id"])
        with pytest.raises(ValueError, match="empty"):
            build_sample(empty, pairs, 5, seed=1)

    def test_raises_on_unmatched_question_id(self, tmp_path, monkeypatch):
        store, photo_ids = make_store_with_cached_photos(tmp_path, monkeypatch, 4)
        verdicts = pd.DataFrame({"question_id": ["q0", "qX"], "type": ["t", "t"], "outcome": ["tie", "tie"]})
        pairs = pd.DataFrame(
            {
                "question_id": ["q0"],
                "question": ["?"],
                "surfaced_photo_id": [photo_ids[0]],
                "control_photo_id": [photo_ids[1]],
            }
        )
        with pytest.raises(ValueError, match="no matching pair"):
            build_sample(verdicts, pairs, 1, seed=1, photos_dir=str(tmp_path / "photos"))

    def test_raises_file_not_found_when_photo_not_cached(self, tmp_path):
        verdicts = pd.DataFrame({"question_id": ["q0"], "type": ["t"], "outcome": ["tie"]})
        pairs = pd.DataFrame(
            {
                "question_id": ["q0"],
                "question": ["?"],
                "surfaced_photo_id": ["deadbeefdeadbeef"],
                "control_photo_id": ["deadbeefdeadbeee"],
            }
        )
        with pytest.raises(FileNotFoundError):
            build_sample(verdicts, pairs, 1, seed=1, photos_dir=str(tmp_path / "photos"))

    def test_same_seed_reproduces_identical_sample(self, tmp_path, monkeypatch):
        n = 10
        store, photo_ids = make_store_with_cached_photos(tmp_path, monkeypatch, 2 * n)
        outcomes = ["surfaced_win"] * 5 + ["control_win"] * 2 + ["tie"] * 3
        verdicts, pairs = make_verdicts_pairs(n, photo_ids, outcomes)

        first = build_sample(verdicts, pairs, 6, seed=9, photos_dir=str(tmp_path / "photos"))
        second = build_sample(verdicts, pairs, 6, seed=9, photos_dir=str(tmp_path / "photos"))

        assert [r.question_id for r in first] == [r.question_id for r in second]


# --------------------------------------------------------------------------
# render_worksheet / parse_worksheet round trip
# --------------------------------------------------------------------------


def _sample_rows() -> list[SpotcheckRow]:
    return [
        SpotcheckRow("R01", "q0", "fit_sizing", "Does it run small?", "photos/a.jpg", "photos/b.jpg", "A"),
        SpotcheckRow("R02", "q1", "colour_appearance", "Is it as pictured?", "photos/c.jpg", "photos/d.jpg", "tie"),
    ]


class TestWorksheetRoundTrip:
    def test_round_trips_verdict_and_notes_with_colon(self):
        rows = _sample_rows()
        rendered = render_worksheet(rows)

        # Fill in: R01 -> "A" with a note containing a colon (the exact case
        # a naive "split on first colon" parser would truncate).
        filled = rendered.replace(
            "## R01\n\n**Question:** Does it run small?\n\n**Photo A:** `photos/a.jpg`\n"
            "**Photo B:** `photos/b.jpg`\n\n**Your verdict:**\n- my_verdict: \n- notes: \n",
            "## R01\n\n**Question:** Does it run small?\n\n**Photo A:** `photos/a.jpg`\n"
            "**Photo B:** `photos/b.jpg`\n\n**Your verdict:**\n- my_verdict: A\n"
            "- notes: A shows fit: clearer than B\n",
        )
        filled = filled.replace("- my_verdict: \n- notes: \n\n", "- my_verdict: tie\n- notes: \n\n", 1)

        parsed = parse_worksheet(filled)

        assert parsed["R01"] == {"my_verdict": "A", "notes": "A shows fit: clearer than B"}
        assert parsed["R02"] == {"my_verdict": "tie", "notes": ""}

    def test_lowercase_and_mixed_case_are_normalized(self):
        rendered = render_worksheet(_sample_rows())
        filled = rendered.replace("- my_verdict: \n", "- my_verdict: a\n", 1)
        filled = filled.replace("- my_verdict: \n", "- my_verdict: TIE\n", 1)
        parsed = parse_worksheet(filled)
        assert parsed["R01"]["my_verdict"] == "A"
        assert parsed["R02"]["my_verdict"] == "tie"

    def test_unfilled_verdict_raises(self):
        rendered = render_worksheet(_sample_rows())
        with pytest.raises(ValueError, match="blank"):
            parse_worksheet(rendered)  # no substitutions made -- both rows unfilled

    def test_missing_my_verdict_line_raises(self):
        rendered = render_worksheet(_sample_rows())
        broken = rendered.replace("- my_verdict: \n", "")  # strip the line from R01 only (first match)
        with pytest.raises(ValueError, match="no 'my_verdict:' line"):
            parse_worksheet(broken)

    def test_invalid_verdict_value_raises(self):
        rendered = render_worksheet(_sample_rows())
        filled = rendered.replace("- my_verdict: \n", "- my_verdict: C\n", 1)
        filled = filled.replace("- my_verdict: \n", "- my_verdict: tie\n", 1)
        with pytest.raises(ValueError, match="not one of"):
            parse_worksheet(filled)


# --------------------------------------------------------------------------
# compute_agreement
# --------------------------------------------------------------------------


class TestComputeAgreement:
    def test_perfect_agreement_kappa_is_one(self):
        paired = pd.DataFrame(
            {
                "judge_verdict": ["A", "B", "tie", "A", "B", "tie"],
                "human_verdict": ["A", "B", "tie", "A", "B", "tie"],
            }
        )
        result = compute_agreement(paired)
        assert result.iloc[0]["cohens_kappa"] == pytest.approx(1.0)
        assert result.iloc[0]["raw_agreement"] == pytest.approx(1.0)
        assert result.iloc[0]["n"] == 6
        assert result.iloc[0]["n_agree"] == 6

    def test_chance_level_agreement_kappa_is_zero(self):
        # Balanced 3x3 marginals (3 A/3 B/3 tie each), observed matches = 3
        # -- exactly the expected-under-independence count, by construction
        # (see test file docstring / development notes): kappa = 0 exactly.
        judge = ["A", "A", "A", "B", "B", "B", "tie", "tie", "tie"]
        human = ["B", "tie", "A", "tie", "A", "B", "A", "B", "tie"]
        paired = pd.DataFrame({"judge_verdict": judge, "human_verdict": human})

        result = compute_agreement(paired)

        assert result.iloc[0]["cohens_kappa"] == pytest.approx(0.0, abs=1e-9)
        assert result.iloc[0]["n_agree"] == 3

    def test_raises_on_empty(self):
        empty = pd.DataFrame(columns=["judge_verdict", "human_verdict"])
        with pytest.raises(ValueError, match="empty"):
            compute_agreement(empty)


# --------------------------------------------------------------------------
# export_worksheet / build_paired_scores / score_worksheet end-to-end
# --------------------------------------------------------------------------


class TestEndToEndWorkflow:
    def test_perfect_agreement_workflow(self, tmp_path, monkeypatch):
        n = 6
        store, photo_ids = make_store_with_cached_photos(tmp_path, monkeypatch, 2 * n)
        outcomes = ["surfaced_win", "surfaced_win", "control_win", "control_win", "tie", "tie"]
        verdicts, pairs = make_verdicts_pairs(n, photo_ids, outcomes)
        verdicts_path = tmp_path / "mm_verdicts_v1.jsonl"
        pairs_path = tmp_path / "mm_pairs_v1.jsonl"
        verdicts_path.write_text("\n".join(json.dumps(r) for r in verdicts.to_dict(orient="records")) + "\n", encoding="utf-8")
        pairs_path.write_text("\n".join(json.dumps(r) for r in pairs.to_dict(orient="records")) + "\n", encoding="utf-8")

        worksheet_path, rows = export_worksheet(
            verdicts_path, pairs_path, 6, seed=3, out_path=tmp_path / "worksheet.md", photos_dir=str(tmp_path / "photos")
        )
        assert worksheet_path.is_file()

        # A human who agrees with the judge on every row -- fill each blank
        # my_verdict with that row's own (hidden-from-worksheet) judge_verdict.
        text = worksheet_path.read_text(encoding="utf-8")
        for row in rows:
            text = text.replace("- my_verdict: \n", f"- my_verdict: {row.judge_verdict}\n", 1)
        worksheet_path.write_text(text, encoding="utf-8")

        summary = score_worksheet(
            worksheet_path, verdicts_path, pairs_path, 6, seed=3, photos_dir=str(tmp_path / "photos")
        )

        assert summary.iloc[0]["cohens_kappa"] == pytest.approx(1.0)
        assert summary.iloc[0]["n"] == 6

    def test_mismatched_worksheet_raises_actionable_error(self, tmp_path, monkeypatch):
        n = 4
        store, photo_ids = make_store_with_cached_photos(tmp_path, monkeypatch, 2 * n)
        outcomes = ["surfaced_win", "control_win", "tie", "tie"]
        verdicts, pairs = make_verdicts_pairs(n, photo_ids, outcomes)
        verdicts_path = tmp_path / "mm_verdicts_v1.jsonl"
        pairs_path = tmp_path / "mm_pairs_v1.jsonl"
        verdicts_path.write_text("\n".join(json.dumps(r) for r in verdicts.to_dict(orient="records")) + "\n", encoding="utf-8")
        pairs_path.write_text("\n".join(json.dumps(r) for r in pairs.to_dict(orient="records")) + "\n", encoding="utf-8")

        worksheet_path, rows = export_worksheet(
            verdicts_path, pairs_path, 3, seed=5, out_path=tmp_path / "worksheet.md", photos_dir=str(tmp_path / "photos")
        )
        text = worksheet_path.read_text(encoding="utf-8")
        for row in rows:
            text = text.replace("- my_verdict: \n", f"- my_verdict: {row.judge_verdict}\n", 1)
        # Corrupt a row id so the re-derived sample no longer matches.
        text = text.replace("## R01", "## R99")
        worksheet_path.write_text(text, encoding="utf-8")

        with pytest.raises(ValueError, match="do not match"):
            build_paired_scores(worksheet_path, verdicts_path, pairs_path, 3, seed=5, photos_dir=str(tmp_path / "photos"))


# --------------------------------------------------------------------------
# plot_winrate (figure #6)
# --------------------------------------------------------------------------


def _winrate_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"group": "overall", "n_pairs": 25, "win_rate": 0.48, "ci_lo": 0.28, "ci_hi": 0.68},
            {"group": "fit_sizing", "n_pairs": 2, "win_rate": 0.5, "ci_lo": 0.0, "ci_hi": 1.0},
            {"group": "colour_appearance", "n_pairs": 4, "win_rate": 0.75, "ci_lo": 0.25, "ci_hi": 1.0},
        ]
    )


class TestPlotWinrate:
    def test_writes_a_nonempty_png(self, tmp_path):
        out_path = plot_winrate(_winrate_df(), tmp_path / "winrate.png")
        assert out_path.is_file()
        assert out_path.stat().st_size > 0

    def test_creates_parent_directories(self, tmp_path):
        out_path = plot_winrate(_winrate_df(), tmp_path / "nested" / "dir" / "winrate.png")
        assert out_path.is_file()

    def test_deterministic_across_calls(self, tmp_path):
        first = plot_winrate(_winrate_df(), tmp_path / "a.png")
        second = plot_winrate(_winrate_df(), tmp_path / "b.png")
        assert first.read_bytes() == second.read_bytes()

    def test_raises_without_overall_row(self, tmp_path):
        df = _winrate_df()
        df = df[df["group"] != "overall"]
        with pytest.raises(ValueError, match="overall"):
            plot_winrate(df, tmp_path / "winrate.png")

    def test_only_overall_row_still_plots(self, tmp_path):
        df = _winrate_df()
        df = df[df["group"] == "overall"]
        out_path = plot_winrate(df, tmp_path / "winrate.png")
        assert out_path.is_file()
        assert out_path.stat().st_size > 0

    def test_plots_whatever_is_passed_without_recomputing(self, tmp_path):
        # A win_rate/CI that no honest recomputation from other columns
        # could produce (impossible values) -- if plot_winrate independently
        # recomputed anything from e.g. n_surfaced_win, this would raise or
        # silently diverge. It doesn't: it just renders exactly what it's given.
        df = pd.DataFrame([{"group": "overall", "n_pairs": 5, "win_rate": 0.91, "ci_lo": 0.91, "ci_hi": 0.91}])
        out_path = plot_winrate(df, tmp_path / "winrate.png")
        assert out_path.is_file()
