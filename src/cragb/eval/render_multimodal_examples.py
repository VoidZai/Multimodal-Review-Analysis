"""Multimodal photo-evidence appendix examples (T6.7; PLAN.md §7, M6.md T6.7).

Renders a hand-picked slice of T6.5's judged pairs into report-ready markdown:
`reports/multimodal_examples_v1.md`, the "3-4 photo-evidence examples" PLAN.md
§7 lists as mid-progress report appendix material for RQ4's multimodal
pilot. Mirrors `cragb.eval.render_grounded_qa_appendix`'s pattern exactly:
`APPENDIX_ENTRIES`' `note` field is a human-authored editorial judgment made
after reading the real output, never generated; every fact the renderer
prints (question, both photos, both order-swapped verdicts, confidences,
rationales) is read straight from `mm_pairs_v1.jsonl`/`mm_verdicts_v1.jsonl`,
never re-derived or reworded.

**Photos are linked, not embedded.** Each example's two photos are plain
markdown image references (`![...](../data/photos/<id>.jpg)`) resolved
relative to the output file's own directory via `PhotoStore.photo_path`, not
base64-inlined -- consistent with `data/photos/` being git-ignored (T6.1)
the same way `data/raw/` and `data/processed/` already are: the manifest is
the audit trail, the bytes are regenerable. "Working image references"
(M6.md T6.7's own validation wording) means the referenced path resolves to
a real file on disk at render time, which `_relative_photo_link` verifies by
construction -- `PhotoStore.photo_path` returns `None` for anything not
cached, and that's a raised, not silently skipped, error here.

Usage:
    python -m cragb.eval.render_multimodal_examples --config configs/multimodal.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cragb.eval.run_multimodal_pilot import load_pairs
from cragb.multimodal.photo_store import PhotoStore
from cragb.utils.io import load_config, resolve_path

logger = logging.getLogger(__name__)

_OUTCOME_LABEL: dict[str, str] = {
    "surfaced_win": "surfaced photo wins",
    "control_win": "control photo wins",
    "tie": "tie",
}


@dataclass(frozen=True)
class AppendixEntry:
    """One curated pair, plus the editorial label/note explaining why it's here."""

    question_id: str
    label: str
    note: str


# The 4-example appendix slice (M6.md T6.7: "at least one clear win, one
# loss, and one tie"), hand-picked from T6.5's live 25-pair run
# (`results/tables/mm_verdicts_v1.jsonl`) after reading every candidate's
# actual photos and rationale.
APPENDIX_ENTRIES: tuple[AppendixEntry, ...] = (
    AppendixEntry(
        question_id="colour_appearance_001",
        label="Clean win: surfaced photo is obviously the right evidence",
        note=(
            "The surfaced photo is a genuine, well-lit product photo of the purple "
            "shoes the question asks about; the control is a photo of unrelated "
            "anime pin badges. Both order-swapped calls agree at confidence 5/5, and "
            "the human spot-check (T6.6, worksheet row R08) independently agreed too "
            "-- the one clear case in this pilot where every signal lines up."
        ),
    ),
    AppendixEntry(
        question_id="fabric_quality_005",
        label="Honest loss: the control photo really is better evidence here",
        note=(
            "T6.3's control is drawn at random from outside the question's retrieved "
            "context, so it can occasionally land on genuinely more relevant evidence "
            "than what the pipeline actually surfaced -- this is exactly that case. "
            "The surfaced photo shows thick knit socks; the control happens to show a "
            "mesh, breathable shoe upper, which is what a breathability/moisture-"
            "wicking question actually needs. The judge is not gaming this: both "
            "order-swapped calls independently identify the mesh photo as better "
            "evidence, not just defaulting to whichever position looks first."
        ),
    ),
    AppendixEntry(
        question_id="value_004",
        label="Principled tie: neither photo is evidence at all",
        note=(
            "'Do customers feel this is a good investment?' is a claim about opinion, "
            "not something a photo can show either way -- an intact-earbuds-in-a-case "
            "photo and a shoe-lacing close-up are equally uninformative here. Both "
            "order-swapped calls independently say tie at confidence 5/5: this is the "
            "judge correctly recognising a question type photos structurally cannot "
            "answer, not indecision."
        ),
    ),
    AppendixEntry(
        question_id="fit_sizing_010",
        label="Defensible but contested: where the human spot-check disagreed",
        note=(
            "The judge picks the control (a watch worn on a wrist, showing real "
            "on-body fit) over the surfaced photo (a necklace still boxed, showing no "
            "fit information at all) -- a coherent, evidence-grounded reason, agreeing "
            "with itself in both photo orders. But T6.6's human spot-check (worksheet "
            "row R07) picked the opposite: the necklace, on the reasoning that it is "
            "at least the product this specific review is actually about. Included "
            "here precisely because it is one of only two disagreements the spot-check "
            "surfaced (PLAN.md §14.6) that reads as genuine reasonable disagreement "
            "rather than a judge error -- reliability limitations should be shown, not "
            "only described in the abstract."
        ),
    ),
)


def load_verdicts(path: str | Path) -> dict[str, dict[str, Any]]:
    """Load `mm_verdicts_v1.jsonl` into `{question_id: row}`."""
    verdicts: dict[str, dict[str, Any]] = {}
    with resolve_path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                row = json.loads(line)
                verdicts[row["question_id"]] = row
    return verdicts


def _relative_photo_link(store: PhotoStore, photo_id: str, from_dir: Path) -> str:
    """Path to `photo_id`'s cached file, relative to `from_dir`, forward-slashed.

    Raises:
        FileNotFoundError: `photo_id` has no cached file -- an appendix
            example must never reference a photo that doesn't exist.
    """
    path = store.photo_path(photo_id)
    if path is None:
        raise FileNotFoundError(f"photo not cached: {photo_id!r}")
    return os.path.relpath(path, start=from_dir).replace("\\", "/")


def render_example_markdown(
    entry: AppendixEntry,
    pair: dict[str, Any],
    verdict: dict[str, Any],
    store: PhotoStore,
    reports_dir: Path,
) -> str:
    """Render one curated pair + its T6.5 verdict as a markdown section."""
    surfaced_link = _relative_photo_link(store, pair["surfaced_photo_id"], reports_dir)
    control_link = _relative_photo_link(store, pair["control_photo_id"], reports_dir)
    outcome_label = _OUTCOME_LABEL[verdict["outcome"]]

    return (
        f"## {entry.label}: `{entry.question_id}` ({outcome_label})\n\n"
        f"*{entry.note}*\n\n"
        f"**Question:** {pair['question']}\n\n"
        f"**Photo A (the pipeline's surfaced photo):**\n\n"
        f"![surfaced photo]({surfaced_link})\n\n"
        f"**Photo B (T6.3's random control):**\n\n"
        f"![control photo]({control_link})\n\n"
        f"**Judge verdict, surfaced shown as A:** {verdict['winner_surfaced_as_a']} "
        f"(confidence {verdict['confidence_surfaced_as_a']}/5) -- "
        f"{verdict['rationale_surfaced_as_a']}\n\n"
        f"**Judge verdict, surfaced shown as B (order swapped):** {verdict['winner_surfaced_as_b']} "
        f"(confidence {verdict['confidence_surfaced_as_b']}/5) -- "
        f"{verdict['rationale_surfaced_as_b']}\n\n"
        f"**Order agreement:** {verdict['order_agreement']} -> **outcome: {verdict['outcome']}**\n"
    )


def render_examples_markdown(
    entries: list[AppendixEntry],
    pairs_by_id: dict[str, dict[str, Any]],
    verdicts_by_id: dict[str, dict[str, Any]],
    store: PhotoStore,
    reports_dir: Path,
) -> str:
    """Assemble the full appendix document from `entries`, in order.

    Raises:
        KeyError: an entry's `question_id` is missing from `pairs_by_id` or
            `verdicts_by_id` -- the appendix must never silently drop a
            curated example.
        FileNotFoundError: propagated from `_relative_photo_link` if an
            entry's photo isn't cached.
    """
    missing_pairs = [e.question_id for e in entries if e.question_id not in pairs_by_id]
    if missing_pairs:
        raise KeyError(f"question id(s) not found in mm_pairs_v1.jsonl: {missing_pairs}")
    missing_verdicts = [e.question_id for e in entries if e.question_id not in verdicts_by_id]
    if missing_verdicts:
        raise KeyError(f"question id(s) not found in mm_verdicts_v1.jsonl: {missing_verdicts}")

    header = (
        "# Multimodal photo-evidence examples (T6.7, RQ4)\n\n"
        "Four pairs hand-picked from T6.5's live 25-pair pilot run (PLAN.md §3 E7, "
        "§7 appendix material): a clean win, an honest loss (the random control "
        "genuinely was better evidence), a principled tie (the question isn't "
        "something a photo can answer), and one case T6.6's human spot-check "
        "disagreed with. Both photos, both order-swapped judge verdicts, and every "
        "rationale below are exactly what T6.4/T6.5 produced -- nothing has been "
        "edited.\n\n"
    )
    sections = [
        render_example_markdown(e, pairs_by_id[e.question_id], verdicts_by_id[e.question_id], store, reports_dir)
        for e in entries
    ]
    return header + "\n---\n\n".join(sections)


def build_examples_markdown(cfg: dict, out_path: str | Path) -> str:
    """Load T6.3/T6.5's real output and render the curated appendix.

    Args:
        cfg: a loaded `configs/multimodal.yaml`-shaped config dict.
        out_path: where the caller intends to write the result -- its
            parent directory is what photo links are resolved relative to.

    Returns:
        Complete Markdown document text.
    """
    paths = cfg["paths"]
    pairs_df = load_pairs(paths["pairs_out"])
    pairs_by_id = {row["question_id"]: row for row in pairs_df.to_dict(orient="records")}
    verdicts_by_id = load_verdicts(paths["verdicts_in"])

    store_cfg = cfg["photo_store"]
    store = PhotoStore(
        photos_dir=store_cfg["photos_dir"],
        max_bytes=store_cfg["max_bytes"],
        timeout_s=store_cfg["timeout_s"],
        max_retries=store_cfg["max_retries"],
        request_delay_s=store_cfg["request_delay_s"],
        allowed_mime=tuple(store_cfg["allowed_mime"]),
    )
    reports_dir = resolve_path(out_path).parent

    return render_examples_markdown(list(APPENDIX_ENTRIES), pairs_by_id, verdicts_by_id, store, reports_dir)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/multimodal.yaml", help="Path to multimodal config YAML.")
    parser.add_argument("--out", default=None, help="Output markdown path (default: cfg.paths.examples_out).")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    cfg = load_config(args.config)
    out_path = args.out or cfg["paths"]["examples_out"]
    markdown = build_examples_markdown(cfg, out_path)

    resolved = resolve_path(out_path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(markdown, encoding="utf-8")
    logger.info("Wrote %d-example appendix to %s", len(APPENDIX_ENTRIES), resolved)

    return 0


if __name__ == "__main__":
    sys.exit(main())
