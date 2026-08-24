"""Computed numbers for the T7.10 datasheet and go/no-go plan (M7.md T7.10).

`data/finetune/finetune_data_datasheet.md` and `reports/finetune_plan_v1.md` are hand-written
prose, but every *number* either document quotes must trace to a committed manifest or results
CSV rather than be retyped by hand (T7.10's own validation check). This module is that single
generation step: it re-reads T7.2-T7.9's committed artifacts and writes one flat JSON of the
derived figures the two documents cite, so a future re-run of the M7 pipeline (a full-scale
T7.3 sweep, say) can regenerate the numbers without anyone hand-editing prose to match.

Deliberately not a notebook cell (T7.11 owns the notebook, out of scope here) and deliberately
not folded into any of T7.2-T7.9's own modules (each of those already writes its own
manifest/CSV for its own task; this module only re-reads what already exists, computing nothing
those tasks didn't already compute).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

DEFAULT_PATHS: dict[str, str] = {
    "contexts_manifest": "data/finetune/contexts_v1_manifest.json",
    "split_manifest": "data/finetune/split_manifest_v1.json",
    "filtered_pairs": "data/finetune/filtered_pairs_v1.jsonl",
    "abstentions": "data/finetune/abstentions_v1.jsonl",
    "raw_pairs_progress": "data/finetune/raw_pairs_v1_progress.jsonl",
    "generation_cost": "results/tables/ft_generation_cost_v1.csv",
    "filter_report": "results/tables/ft_filter_v1.csv",
    "model_probe": "results/tables/ft_model_probe_v1.csv",
    "qlora_probe": "results/tables/ft_qlora_probe_v1.csv",
    "prompt_length_stats": "results/tables/ft_prompt_length_stats_v1.csv",
    "baseline": "results/tables/ft_base_baseline_v1.csv",
    "baseline_transcripts": "results/tables/ft_base_baseline_transcripts_v1.jsonl",
    "cragb_questions": "benchmark/cragb_v1.jsonl",
    "probe_examples": "data/finetune/probe.jsonl",
    "finetune_config": "configs/finetune.yaml",
}

DEFAULT_STATS_OUT = "results/tables/ft_datasheet_stats_v1.json"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    import csv

    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _abstention_method_counts(abstentions: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in abstentions:
        method = row.get("provenance", {}).get("method", "unknown")
        counts[method] += 1
    return dict(sorted(counts.items()))


def _category_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter(row.get("category", "unknown") for row in rows)
    return dict(sorted(counts.items()))


def _over_abstention_breakdown(
    transcripts: list[dict[str, Any]],
    gold_abstention_ids: set[str],
    source: str,
) -> dict[str, Any]:
    """False-abstention and missed-true-abstention counts for one baseline slice.

    T7.8's headline CSV reports a single `abstention_accuracy` figure, which conflates
    two very different failure directions (wrongly refusing an answerable question vs.
    failing to refuse an unanswerable one). The go/no-go plan needs them separated,
    because the transcripts show the local base model's failure is almost entirely one
    of those two directions, not the other (see `reports/finetune_plan_v1.md`).
    """
    rows = [r for r in transcripts if r.get("source") == source]
    n_answerable = 0
    n_false_abstention = 0
    n_gold_abstention = 0
    n_missed_true_abstention = 0
    for row in rows:
        qid = row["question_id"]
        if qid in gold_abstention_ids:
            n_gold_abstention += 1
            if not row["abstained"]:
                n_missed_true_abstention += 1
            continue
        n_answerable += 1
        if row["abstained"]:
            n_false_abstention += 1
    return {
        "n_total": len(rows),
        "n_answerable": n_answerable,
        "n_false_abstention_on_answerable": n_false_abstention,
        "false_abstention_rate": round(n_false_abstention / n_answerable, 4) if n_answerable else None,
        "n_gold_abstention": n_gold_abstention,
        "n_missed_true_abstention": n_missed_true_abstention,
        "true_abstention_recall": (
            round((n_gold_abstention - n_missed_true_abstention) / n_gold_abstention, 4)
            if n_gold_abstention
            else None
        ),
    }


def build_stats(paths: dict[str, str], root: Path) -> dict[str, Any]:
    contexts_manifest = json.loads((root / paths["contexts_manifest"]).read_text(encoding="utf-8"))
    split_manifest = json.loads((root / paths["split_manifest"]).read_text(encoding="utf-8"))
    filtered_pairs = _read_jsonl(root / paths["filtered_pairs"])
    abstentions = _read_jsonl(root / paths["abstentions"])
    raw_progress = _read_jsonl(root / paths["raw_pairs_progress"])
    generation_cost = _read_csv_rows(root / paths["generation_cost"])
    filter_report = _read_csv_rows(root / paths["filter_report"])
    model_probe = _read_csv_rows(root / paths["model_probe"])
    qlora_probe = _read_csv_rows(root / paths["qlora_probe"])
    prompt_len = _read_csv_rows(root / paths["prompt_length_stats"])
    baseline = _read_csv_rows(root / paths["baseline"])

    import yaml

    finetune_cfg = yaml.safe_load((root / paths["finetune_config"]).read_text(encoding="utf-8"))

    n_contexts_target = finetune_cfg["sampling"]["n_contexts"]
    n_contexts_sampled = contexts_manifest["n_groups"]
    n_contexts_attempted = len(raw_progress)

    n_filtered = len(filtered_pairs)
    n_abstentions = len(abstentions)

    baseline_by_source = {row["source"]: row for row in baseline}
    qlora_fits = [row for row in qlora_probe if row.get("oom") == "False" and row.get("error", "") == ""]
    qlora_best = min(qlora_fits, key=lambda r: float(r["extrapolated_minutes_per_epoch"])) if qlora_fits else None

    baseline_transcripts = _read_jsonl(root / paths["baseline_transcripts"])
    cragb_questions = _read_jsonl(root / paths["cragb_questions"])
    probe_examples = _read_jsonl(root / paths["probe_examples"])
    cragb_gold_abstention_ids = {q["id"] for q in cragb_questions if q.get("is_abstention")}
    probe_gold_abstention_ids = {e["example_id"] for e in probe_examples if e.get("is_abstention")}
    abstention_breakdown = {
        "cragb_60": _over_abstention_breakdown(baseline_transcripts, cragb_gold_abstention_ids, "cragb_60"),
        "probe": _over_abstention_breakdown(baseline_transcripts, probe_gold_abstention_ids, "probe"),
    }

    stats: dict[str, Any] = {
        "generated_from": paths,
        "scale": {
            "n_contexts_target_full_sweep": n_contexts_target,
            "n_contexts_sampled_and_available": n_contexts_sampled,
            "n_contexts_attempted_this_pilot": n_contexts_attempted,
            "pilot_fraction_of_target": round(n_contexts_attempted / n_contexts_target, 4),
            "n_raw_pairs_pilot": sum(1 for _ in _read_jsonl(root / "data/finetune/raw_pairs_v1.jsonl")),
            "n_filtered_accepted": n_filtered,
            "n_abstentions_constructed": n_abstentions,
            "n_train": split_manifest["train"]["n"],
            "n_val": split_manifest["val"]["n"],
            "n_probe": split_manifest["probe"]["n"],
        },
        "context_sampling": {
            "per_category_counts": contexts_manifest["per_category_counts"],
            "n_docs_excluded_as_cragb_evidence": contexts_manifest["n_docs_excluded_as_cragb_evidence"],
            "n_parent_asins_excluded": contexts_manifest["n_parent_asins_excluded"],
            "photo_bearing_share": contexts_manifest["photo_bearing_share"],
            "category_quota_report": contexts_manifest["category_quota_report"],
        },
        "generation_cost": generation_cost[0] if generation_cost else None,
        "filter_funnel": next((r for r in filter_report if r["slice"] == "overall"), None),
        "filtered_pairs_category_counts": _category_counts(filtered_pairs),
        "abstention_method_counts": _abstention_method_counts(abstentions),
        "leakage_and_split": {
            "n_input": split_manifest["n_input"],
            "n_dropped_exact_leak": split_manifest["n_dropped_exact_leak"],
            "n_dropped_near_duplicate": split_manifest["n_dropped_near_duplicate"],
            "near_duplicate_matches": split_manifest["near_duplicate_matches"],
            "n_kept": split_manifest["n_kept"],
            "probe": split_manifest["probe"],
            "parent_asin_disjointness": split_manifest["parent_asin_disjointness"],
            "embedding_backstop_used": split_manifest["embedding_backstop_used"],
            "config": split_manifest["config"],
        },
        "model_probe": model_probe,
        "qlora_probe": qlora_probe,
        "qlora_best_fitting_config": qlora_best,
        "prompt_length_stats": prompt_len,
        "baseline_cragb_60": baseline_by_source.get("cragb_60"),
        "baseline_probe": baseline_by_source.get("probe"),
        "baseline_abstention_breakdown": abstention_breakdown,
    }
    return stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".", help="Project root (default: cwd).")
    parser.add_argument("--out", default=DEFAULT_STATS_OUT, help="Output JSON path, relative to --root.")
    args = parser.parse_args(argv)

    root = Path(args.root)
    stats = build_stats(DEFAULT_PATHS, root)

    out_path = root / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
