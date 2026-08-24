"""Unit tests for cragb.finetune.datasheet_stats (T7.10; M7.md T7.10).

Covers: `_over_abstention_breakdown`'s split of a single `abstention_accuracy` figure
into false-abstention-on-answerable vs. missed-true-abstention counts (the distinction
`reports/finetune_plan_v1.md`'s go/no-go rule is written against), `_abstention_method_counts`
and `_category_counts`'s tallying, and `build_stats` end-to-end against a minimal set of
fixture files shaped exactly like T7.2-T7.9's real committed artifacts -- so a change to
any of those on-disk shapes breaks this test before it breaks the datasheet's numbers.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from cragb.finetune.datasheet_stats import (
    _abstention_method_counts,
    _category_counts,
    _over_abstention_breakdown,
    build_stats,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _write_csv(path: Path, rows: list[dict]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_over_abstention_breakdown_separates_the_two_failure_directions():
    transcripts = [
        {"question_id": "q_answerable_wrongly_abstained", "source": "cragb_60", "abstained": True},
        {"question_id": "q_answerable_correctly_answered", "source": "cragb_60", "abstained": False},
        {"question_id": "q_gold_abstention_correctly_abstained", "source": "cragb_60", "abstained": True},
        {"question_id": "q_gold_abstention_missed", "source": "cragb_60", "abstained": False},
        {"question_id": "q_other_source_ignored", "source": "probe", "abstained": True},
    ]
    gold_abstention_ids = {"q_gold_abstention_correctly_abstained", "q_gold_abstention_missed"}

    result = _over_abstention_breakdown(transcripts, gold_abstention_ids, source="cragb_60")

    assert result["n_total"] == 4
    assert result["n_answerable"] == 2
    assert result["n_false_abstention_on_answerable"] == 1
    assert result["false_abstention_rate"] == 0.5
    assert result["n_gold_abstention"] == 2
    assert result["n_missed_true_abstention"] == 1
    assert result["true_abstention_recall"] == 0.5


def test_over_abstention_breakdown_perfect_recall_and_zero_false_abstention():
    transcripts = [
        {"question_id": "a", "source": "probe", "abstained": False},
        {"question_id": "b", "source": "probe", "abstained": True},
    ]
    result = _over_abstention_breakdown(transcripts, gold_abstention_ids={"b"}, source="probe")
    assert result["false_abstention_rate"] == 0.0
    assert result["true_abstention_recall"] == 1.0


def test_over_abstention_breakdown_handles_no_gold_abstentions_in_slice():
    transcripts = [{"question_id": "a", "source": "probe", "abstained": False}]
    result = _over_abstention_breakdown(transcripts, gold_abstention_ids=set(), source="probe")
    assert result["n_gold_abstention"] == 0
    assert result["true_abstention_recall"] is None


def test_abstention_method_counts_tallies_provenance_method():
    rows = [
        {"provenance": {"method": "transplant"}},
        {"provenance": {"method": "transplant"}},
        {"provenance": {"method": "evidence_stripped"}},
    ]
    assert _abstention_method_counts(rows) == {"evidence_stripped": 1, "transplant": 2}


def test_category_counts_tallies_category_field():
    rows = [{"category": "fit_sizing"}, {"category": "value"}, {"category": "fit_sizing"}]
    assert _category_counts(rows) == {"fit_sizing": 2, "value": 1}


def test_build_stats_end_to_end_against_minimal_fixture_files(tmp_path: Path):
    root = tmp_path

    _write_jsonl(
        root / "data/finetune/filtered_pairs_v1.jsonl",
        [
            {"category": "fit_sizing", "is_abstention": False},
            {"category": "value", "is_abstention": True},
        ],
    )
    _write_jsonl(
        root / "data/finetune/abstentions_v1.jsonl",
        [{"provenance": {"method": "transplant"}}],
    )
    _write_jsonl(root / "data/finetune/raw_pairs_v1.jsonl", [{"a": 1}, {"a": 2}])
    _write_jsonl(root / "data/finetune/raw_pairs_v1_progress.jsonl", [{"a": 1}])
    (root / "data/finetune/contexts_v1_manifest.json").parent.mkdir(parents=True, exist_ok=True)
    (root / "data/finetune/contexts_v1_manifest.json").write_text(
        json.dumps(
            {
                "n_groups": 5,
                "per_category_counts": {"fit_sizing": 5},
                "n_docs_excluded_as_cragb_evidence": 1,
                "n_parent_asins_excluded": 1,
                "photo_bearing_share": 0.5,
                "category_quota_report": {},
            }
        ),
        encoding="utf-8",
    )
    (root / "data/finetune/split_manifest_v1.json").write_text(
        json.dumps(
            {
                "n_input": 2,
                "n_dropped_exact_leak": 0,
                "n_dropped_near_duplicate": 0,
                "near_duplicate_matches": [],
                "n_kept": 2,
                "train": {"n": 0},
                "val": {"n": 0},
                "probe": {"n": 2, "n_answerable": 1, "n_abstention": 1, "per_category": {}},
                "parent_asin_disjointness": {"train_val_overlap": 0, "train_probe_overlap": 0, "val_probe_overlap": 0},
                "embedding_backstop_used": False,
                "config": {},
            }
        ),
        encoding="utf-8",
    )
    _write_csv(
        root / "results/tables/ft_generation_cost_v1.csv",
        [{"model": "m", "n_calls": "1", "total_usd": "0.01"}],
    )
    _write_csv(
        root / "results/tables/ft_filter_v1.csv",
        [{"slice": "overall", "n_raw": "2", "n_accepted": "2"}],
    )
    _write_csv(root / "results/tables/ft_model_probe_v1.csv", [{"model": "m", "fits": "yes"}])
    _write_csv(
        root / "results/tables/ft_qlora_probe_v1.csv",
        [{"model": "m", "oom": "False", "error": "", "extrapolated_minutes_per_epoch": "3.0"}],
    )
    _write_csv(root / "results/tables/ft_prompt_length_stats_v1.csv", [{"model": "m", "p50": "800"}])
    _write_csv(
        root / "results/tables/ft_base_baseline_v1.csv",
        [
            {"source": "cragb_60", "model": "m", "abstention_accuracy": "0.5"},
            {"source": "probe", "model": "m", "abstention_accuracy": "0.4"},
        ],
    )
    _write_jsonl(
        root / "results/tables/ft_base_baseline_transcripts_v1.jsonl",
        [
            {"question_id": "q1", "source": "cragb_60", "abstained": False},
            {"question_id": "q2", "source": "probe", "abstained": True},
        ],
    )
    _write_jsonl(root / "benchmark/cragb_v1.jsonl", [{"id": "q1", "is_abstention": False}])
    _write_jsonl(root / "data/finetune/probe.jsonl", [{"example_id": "q2", "is_abstention": True}])
    (root / "configs").mkdir(parents=True, exist_ok=True)
    (root / "configs/finetune.yaml").write_text(
        yaml.safe_dump({"sampling": {"n_contexts": 500}}), encoding="utf-8"
    )

    from cragb.finetune.datasheet_stats import DEFAULT_PATHS

    stats = build_stats(DEFAULT_PATHS, root)

    assert stats["scale"]["n_contexts_target_full_sweep"] == 500
    assert stats["scale"]["n_contexts_sampled_and_available"] == 5
    assert stats["scale"]["n_contexts_attempted_this_pilot"] == 1
    assert stats["scale"]["n_filtered_accepted"] == 2
    assert stats["filtered_pairs_category_counts"] == {"fit_sizing": 1, "value": 1}
    assert stats["abstention_method_counts"] == {"transplant": 1}
    assert stats["baseline_abstention_breakdown"]["cragb_60"]["n_false_abstention_on_answerable"] == 0
    assert stats["baseline_abstention_breakdown"]["probe"]["n_missed_true_abstention"] == 0
