"""Leakage guard + grouped train/val/probe split (T7.6; PLAN.md §1.4 risk F, §3 E8, M7.md T7.6).

Proves no CRAGB evaluation question -- or a paraphrase of one -- ended up as fine-tuning
*training* data, then splits what survives so that no product's reviews appear in more
than one of train/val/probe. Every RQ3 number the second half of the project produces
rests on this task: a training set that has memorised CRAGB's own evidence would inflate
every downstream comparison without anyone being able to tell from the numbers alone.

**Three leakage layers, each catching what the previous one structurally cannot:**

1. **Exact hash** (`cragb.bench.assemble.compute_question_hash`/`check_no_leakage`, T2.10
   -- called here, not reimplemented). Catches a CRAGB question copied verbatim, or
   differing only in case/whitespace, into a training question.
2. **difflib near-duplicate.** A `SequenceMatcher` ratio against all 60 CRAGB questions
   catches a close paraphrase that shares most of its surface form -- exactly the layer
   `tests/test_no_leakage.py`'s own docstring names as the guard's known limitation
   ("catches verbatim/near-verbatim copies, not semantic paraphrase detection").
3. **Embedding cosine backstop.** For a paraphrase that shares *no* surface form at all
   (different words, same meaning), `BAAI/bge-small-en-v1.5` -- the same model
   `cragb.retrieval.dense.DenseRetriever` already uses -- embeds every surviving
   candidate and every CRAGB question, and anything above
   `embedding_similarity_threshold` cosine similarity to any CRAGB question is dropped
   too. Requires `sentence-transformers`, installed in `C:\\venv\\cragb` (PLAN.md §14.1)
   but not the conda environment this project's other tests run under. **Its absence is
   not an error**: `guard_leakage` degrades to difflib-only, logs a warning, and records
   `embedding_backstop_used=False` in the returned report (and, via
   `build_split_manifest`, in the committed manifest) -- the spec's own words are "degrade
   gracefully... with the degradation recorded... rather than silently skipped."

All three layers **drop**, never warn-and-keep -- a false negative here (a leaked question
that slips through) is a silent validity threat to every RQ3 result; a false positive
(a legitimately different question dropped for looking similar) just costs one training
example out of hundreds.

**Grouping for the split is by `source_parent_asins[0]`, not by which example "belongs
to" which context group.** Every `TrainingExample` this project constructs
(T7.3's teacher generation, and all three of T7.4's abstention methods) carries exactly
one source parent_asin -- even a `transplant` abstention, whose *question* came from a
different product's positive, is grouped by the product whose *review text it actually
shows in `context_text`*, because that is the leakage-relevant fact: training on this
example exposes that product's reviews to the model, regardless of whose question is
attached to them. `_group_by_parent_asin` raises if that single-parent_asin invariant
ever stops holding (a future construction method producing a multi-product example would
need this module updated deliberately, not silently mis-split).

Usage:
    python -m cragb.finetune.split --config configs/finetune.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path

from cragb.bench.assemble import check_no_leakage, compute_question_hash, normalize_question_text
from cragb.finetune.schema import TrainingExample, load_training_examples_jsonl, write_training_examples_jsonl
from cragb.utils.io import load_config, resolve_path
from cragb.utils.seeds import set_global_seed

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# CRAGB reference data
# --------------------------------------------------------------------------


def load_cragb_question_hashes(path: str | Path = "benchmark/cragb_v1_leakage_manifest.json") -> dict[str, str]:
    """`question_id -> sha256`, from T2.10's committed manifest."""
    obj = json.loads(resolve_path(path).read_text(encoding="utf-8"))
    return obj["question_hashes"]


def load_cragb_questions(path: str | Path = "benchmark/cragb_v1.jsonl") -> dict[str, str]:
    """`question_id -> question text`, from the frozen benchmark."""
    questions: dict[str, str] = {}
    with resolve_path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            questions[obj["id"]] = obj["question"]
    return questions


# --------------------------------------------------------------------------
# Embedding backstop (optional dependency)
# --------------------------------------------------------------------------


def _embed_texts(texts: list[str], model_name: str):
    """Load `model_name` and return L2-normalized embeddings for `texts`.

    Imports `sentence_transformers` locally (not at module level) so this module stays
    importable in an environment that doesn't have it -- only calling this function
    requires the dependency, matching the graceful-degradation contract `guard_leakage`
    implements around it.
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    return model.encode(texts, normalize_embeddings=True)


def embedding_near_duplicates(
    candidate_questions: dict[str, str],
    cragb_questions: dict[str, str],
    threshold: float,
    model_name: str,
) -> tuple[dict[str, tuple[str, float]], bool]:
    """Cosine-similarity semantic backstop: each candidate against every CRAGB question.

    Args:
        candidate_questions: `example_id -> question text` to check.
        cragb_questions: `question_id -> question text` (the 60 CRAGB questions).
        threshold: cosine similarity at or above which a candidate is flagged.
        model_name: sentence-transformers model id (`BAAI/bge-small-en-v1.5`, matching
            `cragb.retrieval.dense.DenseRetriever`'s default).

    Returns:
        `(matches, available)`. `matches` is `{example_id: (cragb_question_id,
        similarity)}` for the single highest-similarity CRAGB question per flagged
        candidate, above `threshold`; empty if nothing was flagged. `available` is
        `False` (and `matches` is always `{}`) iff `sentence-transformers` isn't
        importable, or model loading/encoding raised for any other reason (e.g. no
        network to fetch the model on first use) -- either way this degrades to "backstop
        not run", never a hard failure, per this module's graceful-degradation contract.
    """
    if not candidate_questions:
        return {}, True

    try:
        import numpy as np

        cragb_ids = list(cragb_questions)
        cragb_embeddings = _embed_texts([cragb_questions[qid] for qid in cragb_ids], model_name)

        candidate_ids = list(candidate_questions)
        candidate_embeddings = _embed_texts([candidate_questions[eid] for eid in candidate_ids], model_name)
    except Exception:
        logger.warning(
            "Embedding backstop unavailable (sentence-transformers not importable, or "
            "model load/encode failed); degrading to difflib-only near-duplicate detection.",
            exc_info=True,
        )
        return {}, False

    similarities = candidate_embeddings @ cragb_embeddings.T  # cosine, since both L2-normalized
    matches: dict[str, tuple[str, float]] = {}
    for i, example_id in enumerate(candidate_ids):
        best_j = int(np.argmax(similarities[i]))
        best_score = float(similarities[i, best_j])
        if best_score >= threshold:
            matches[example_id] = (cragb_ids[best_j], best_score)
    return matches, True


# --------------------------------------------------------------------------
# guard_leakage
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class NearDuplicateMatch:
    example_id: str
    cragb_question_id: str
    method: str  # "difflib" or "embedding"
    score: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class LeakageReport:
    """T7.6's own leakage report -- distinct from (and not to be confused with)
    `cragb.bench.assemble.LeakageReport`, which `guard_leakage` calls internally for the
    exact-hash layer's pass/fail determination but whose narrower shape (a single
    CRAGB-side ok/leaked-ids verdict) can't carry the per-example, multi-layer detail
    this module's callers need.
    """

    ok: bool  # True iff zero exact leaks (near-duplicates are dropped but don't flip this)
    n_checked: int
    n_exact_leak: int
    n_near_duplicate: int
    exact_leak_example_ids: tuple[str, ...]
    near_duplicate_matches: tuple[NearDuplicateMatch, ...]
    embedding_backstop_used: bool
    kept_example_ids: tuple[str, ...]


def guard_leakage(
    examples: list[TrainingExample],
    cragb_question_hashes: dict[str, str],
    cragb_questions: dict[str, str],
    *,
    near_duplicate_threshold: float,
    embedding_similarity_threshold: float,
    embedding_model_name: str,
) -> LeakageReport:
    """Run all three leakage layers over `examples`, in order, dropping at each.

    Args:
        examples: candidate training examples (e.g. T7.5's `filtered_pairs_v1.jsonl`).
        cragb_question_hashes: from `load_cragb_question_hashes`.
        cragb_questions: from `load_cragb_questions`.
        near_duplicate_threshold: difflib `SequenceMatcher` ratio at or above which a
            candidate is dropped as a near-duplicate of some CRAGB question.
        embedding_similarity_threshold: cosine similarity at or above which a candidate
            is dropped by the embedding backstop.
        embedding_model_name: forwarded to `embedding_near_duplicates`.

    Returns:
        A `LeakageReport`.
    """
    cragb_hash_set = set(cragb_question_hashes.values())

    # Layer 1: exact hash. `check_no_leakage` (T2.10, called not reimplemented) gives the
    # overall CRAGB-side verdict; the per-example hash recomputation below is what
    # actually determines *which* examples to drop, and its own `ok` is cross-checked
    # against check_no_leakage's for a free consistency guard between the two.
    overall = check_no_leakage(cragb_question_hashes, [e.question for e in examples])
    exact_leak_ids = tuple(
        e.example_id for e in examples if compute_question_hash(e.question) in cragb_hash_set
    )
    assert overall.ok == (not exact_leak_ids), (
        "check_no_leakage and this module's own per-example hash check disagree -- "
        f"overall.ok={overall.ok}, exact_leak_ids={exact_leak_ids}"
    )
    exact_leak_id_set = set(exact_leak_ids)
    survivors = [e for e in examples if e.example_id not in exact_leak_id_set]

    # Layer 2: difflib near-duplicate.
    cragb_normalized = {qid: normalize_question_text(text) for qid, text in cragb_questions.items()}
    near_duplicate_matches: list[NearDuplicateMatch] = []
    still_surviving: list[TrainingExample] = []
    for example in survivors:
        normalized_question = normalize_question_text(example.question)
        best_qid, best_ratio = None, 0.0
        for qid, normalized_cragb_question in cragb_normalized.items():
            ratio = SequenceMatcher(None, normalized_question, normalized_cragb_question).ratio()
            if ratio > best_ratio:
                best_qid, best_ratio = qid, ratio
        if best_qid is not None and best_ratio >= near_duplicate_threshold:
            near_duplicate_matches.append(
                NearDuplicateMatch(example.example_id, best_qid, "difflib", best_ratio)
            )
        else:
            still_surviving.append(example)

    # Layer 3: embedding backstop.
    embedding_matches, embedding_backstop_used = embedding_near_duplicates(
        {e.example_id: e.question for e in still_surviving},
        cragb_questions,
        embedding_similarity_threshold,
        embedding_model_name,
    )
    final_survivors: list[TrainingExample] = []
    for example in still_surviving:
        if example.example_id in embedding_matches:
            cragb_qid, score = embedding_matches[example.example_id]
            near_duplicate_matches.append(
                NearDuplicateMatch(example.example_id, cragb_qid, "embedding", score)
            )
        else:
            final_survivors.append(example)

    return LeakageReport(
        ok=overall.ok,
        n_checked=len(examples),
        n_exact_leak=len(exact_leak_ids),
        n_near_duplicate=len(near_duplicate_matches),
        exact_leak_example_ids=exact_leak_ids,
        near_duplicate_matches=tuple(near_duplicate_matches),
        embedding_backstop_used=embedding_backstop_used,
        kept_example_ids=tuple(e.example_id for e in final_survivors),
    )


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SplitConfig:
    """Resolved, validated `configs/finetune.yaml`'s `split:` block."""

    seed: int
    val_fraction: float
    probe_answerable_count: int
    probe_abstention_count: int
    near_duplicate_threshold: float
    embedding_similarity_threshold: float
    embedding_model: str

    def __post_init__(self) -> None:
        if not 0.0 < self.val_fraction < 1.0:
            raise ValueError(f"val_fraction must be in (0, 1), got {self.val_fraction}")
        if self.probe_answerable_count < 0 or self.probe_abstention_count < 0:
            raise ValueError("probe_answerable_count/probe_abstention_count must be non-negative")
        if not 0.0 <= self.near_duplicate_threshold <= 1.0:
            raise ValueError(f"near_duplicate_threshold must be in [0, 1], got {self.near_duplicate_threshold}")
        if not 0.0 <= self.embedding_similarity_threshold <= 1.0:
            raise ValueError(
                f"embedding_similarity_threshold must be in [0, 1], got {self.embedding_similarity_threshold}"
            )


def load_split_config(config_path: str | Path = "configs/finetune.yaml") -> SplitConfig:
    cfg = load_config(config_path)
    s = cfg["split"]
    return SplitConfig(
        seed=int(cfg["seed"]),
        val_fraction=float(s["val_fraction"]),
        probe_answerable_count=int(s["probe_answerable_count"]),
        probe_abstention_count=int(s["probe_abstention_count"]),
        near_duplicate_threshold=float(s["near_duplicate_threshold"]),
        embedding_similarity_threshold=float(s["embedding_similarity_threshold"]),
        embedding_model=str(s["embedding_model"]),
    )


# --------------------------------------------------------------------------
# Grouping + selection
# --------------------------------------------------------------------------


def _group_by_parent_asin(examples: list[TrainingExample]) -> dict[str, list[TrainingExample]]:
    """`parent_asin -> examples`, preserving `examples`' relative order within each group.

    Raises:
        ValueError: if any example doesn't carry exactly one `source_parent_asins` entry
            -- every construction method in this project (T7.3, T7.4's three methods)
            always produces exactly one; a different shape is a caller bug, not a
            legitimate case this function should guess how to handle.
    """
    groups: dict[str, list[TrainingExample]] = {}
    for example in examples:
        if len(example.source_parent_asins) != 1:
            raise ValueError(
                f"{example.example_id}: expected exactly one source_parent_asin for "
                f"grouping, got {example.source_parent_asins}"
            )
        groups.setdefault(example.source_parent_asins[0], []).append(example)
    return groups


def _select_probe_groups(
    groups: dict[str, list[TrainingExample]],
    target_answerable: int,
    target_abstention: int,
    rng: random.Random,
) -> set[str]:
    """Greedily select whole product groups for the probe until both targets are met.

    Groups are shuffled once under `rng`, then added in that order as long as *either*
    target hasn't yet been reached -- a group can (and typically does) contribute to only
    one of the two counts, but a mixed group (e.g. a product whose own positive sits
    alongside a `transplant` abstention targeting it) contributes to both at once. This is
    a simple greedy pass, not an exact-target optimizer: with product groups this small
    (usually 1-3 examples), it lands within a few examples of target in practice, and
    `build_split_manifest` reports the *actual* achieved balance rather than assuming the
    target was hit.
    """
    group_ids = list(groups)
    rng.shuffle(group_ids)

    selected: set[str] = set()
    n_answerable = 0
    n_abstention = 0
    for parent_asin in group_ids:
        if n_answerable >= target_answerable and n_abstention >= target_abstention:
            break
        group_examples = groups[parent_asin]
        if n_answerable < target_answerable or n_abstention < target_abstention:
            selected.add(parent_asin)
            n_answerable += sum(1 for e in group_examples if not e.is_abstention)
            n_abstention += sum(1 for e in group_examples if e.is_abstention)

    return selected


def _select_val_groups(
    groups: dict[str, list[TrainingExample]],
    val_fraction: float,
    rng: random.Random,
) -> set[str]:
    """Greedily select whole product groups for val until its example count reaches
    `round(val_fraction * total_examples_in_groups)`.
    """
    group_ids = list(groups)
    rng.shuffle(group_ids)
    total_examples = sum(len(v) for v in groups.values())
    target_val = round(val_fraction * total_examples)

    selected: set[str] = set()
    n_val = 0
    for parent_asin in group_ids:
        if n_val >= target_val:
            break
        selected.add(parent_asin)
        n_val += len(groups[parent_asin])

    return selected


# --------------------------------------------------------------------------
# split_examples
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SplitResult:
    train: list[TrainingExample]
    val: list[TrainingExample]
    probe: list[TrainingExample]
    leakage_report: LeakageReport


def split_examples(
    examples: list[TrainingExample],
    config: SplitConfig,
    cragb_question_hashes: dict[str, str],
    cragb_questions: dict[str, str],
) -> SplitResult:
    """Guard against leakage, then split survivors into train/val/probe, grouped by product.

    Order: leakage guard first (a leaked or near-duplicate example must never appear in
    *any* split, including train), then probe selection (held out entirely), then val
    selection from whatever remains -- train is everything left over.

    Args:
        examples: candidate training examples (T7.5's filtered output).
        config: `SplitConfig`.
        cragb_question_hashes: from `load_cragb_question_hashes`.
        cragb_questions: from `load_cragb_questions`.

    Returns:
        A `SplitResult`. `train`/`val`/`probe` partition the leakage guard's survivors
        exactly (`len(train) + len(val) + len(probe) == len(leakage_report.kept_example_ids)`).
    """
    leakage_report = guard_leakage(
        examples,
        cragb_question_hashes,
        cragb_questions,
        near_duplicate_threshold=config.near_duplicate_threshold,
        embedding_similarity_threshold=config.embedding_similarity_threshold,
        embedding_model_name=config.embedding_model,
    )
    kept_ids = set(leakage_report.kept_example_ids)
    survivors = [e for e in examples if e.example_id in kept_ids]

    groups = _group_by_parent_asin(survivors)

    rng = random.Random(config.seed)
    probe_parent_asins = _select_probe_groups(
        groups, config.probe_answerable_count, config.probe_abstention_count, rng
    )
    if len(probe_parent_asins) == len(groups):
        logger.warning(
            "Probe selection consumed every available product group (%d) without "
            "reaching its target -- val/train will be empty.",
            len(groups),
        )

    remaining_groups = {pa: exs for pa, exs in groups.items() if pa not in probe_parent_asins}
    val_parent_asins = _select_val_groups(remaining_groups, config.val_fraction, rng)

    train: list[TrainingExample] = []
    val: list[TrainingExample] = []
    probe: list[TrainingExample] = []
    for parent_asin, group_examples in groups.items():
        if parent_asin in probe_parent_asins:
            probe.extend(group_examples)
        elif parent_asin in val_parent_asins:
            val.extend(group_examples)
        else:
            train.extend(group_examples)

    return SplitResult(train=train, val=val, probe=probe, leakage_report=leakage_report)


# --------------------------------------------------------------------------
# Manifest
# --------------------------------------------------------------------------


def _composition(examples: list[TrainingExample]) -> dict:
    return {
        "n": len(examples),
        "n_answerable": sum(1 for e in examples if not e.is_abstention),
        "n_abstention": sum(1 for e in examples if e.is_abstention),
        "per_category": dict(sorted(Counter(e.category for e in examples).items())),
    }


def build_split_manifest(result: SplitResult, config: SplitConfig) -> dict:
    """Assemble `split_manifest_v1.json`'s contents from an already-computed `SplitResult`."""
    train_parent_asins = {e.source_parent_asins[0] for e in result.train}
    val_parent_asins = {e.source_parent_asins[0] for e in result.val}
    probe_parent_asins = {e.source_parent_asins[0] for e in result.probe}

    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": config.seed,
        "n_input": result.leakage_report.n_checked,
        "n_dropped_exact_leak": result.leakage_report.n_exact_leak,
        "exact_leak_example_ids": list(result.leakage_report.exact_leak_example_ids),
        "n_dropped_near_duplicate": result.leakage_report.n_near_duplicate,
        "near_duplicate_matches": [m.to_dict() for m in result.leakage_report.near_duplicate_matches],
        "embedding_backstop_used": result.leakage_report.embedding_backstop_used,
        "n_kept": len(result.leakage_report.kept_example_ids),
        "train": _composition(result.train),
        "val": _composition(result.val),
        "probe": _composition(result.probe),
        "parent_asin_disjointness": {
            "train_val_overlap": len(train_parent_asins & val_parent_asins),
            "train_probe_overlap": len(train_parent_asins & probe_parent_asins),
            "val_probe_overlap": len(val_parent_asins & probe_parent_asins),
        },
        "config": {
            "val_fraction": config.val_fraction,
            "probe_answerable_count": config.probe_answerable_count,
            "probe_abstention_count": config.probe_abstention_count,
            "near_duplicate_threshold": config.near_duplicate_threshold,
            "embedding_similarity_threshold": config.embedding_similarity_threshold,
            "embedding_model": config.embedding_model,
        },
    }


def write_split_manifest(manifest: dict, out_path: str | Path) -> Path:
    resolved = resolve_path(out_path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return resolved


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/finetune.yaml", help="Path to fine-tuning config YAML.")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    raw_cfg = load_config(args.config)
    config = load_split_config(args.config)
    set_global_seed(config.seed)

    examples = load_training_examples_jsonl(raw_cfg["paths"]["filtered_pairs_out"])
    cragb_question_hashes = load_cragb_question_hashes(raw_cfg["paths"]["cragb_leakage_manifest"])
    cragb_questions = load_cragb_questions(raw_cfg["paths"]["cragb_questions_in"])

    result = split_examples(examples, config, cragb_question_hashes, cragb_questions)

    write_training_examples_jsonl(result.train, raw_cfg["paths"]["train_out"])
    write_training_examples_jsonl(result.val, raw_cfg["paths"]["val_out"])
    write_training_examples_jsonl(result.probe, raw_cfg["paths"]["probe_out"])

    manifest = build_split_manifest(result, config)
    manifest_path = write_split_manifest(manifest, raw_cfg["paths"]["split_manifest_out"])

    logger.info("exact leaks: %d", result.leakage_report.n_exact_leak)
    logger.info(
        "near-duplicates dropped: %d (embedding backstop used: %s)",
        result.leakage_report.n_near_duplicate,
        result.leakage_report.embedding_backstop_used,
    )
    logger.info(
        "train=%d val=%d probe=%d (probe: %d answerable, %d abstention)",
        len(result.train),
        len(result.val),
        len(result.probe),
        sum(1 for e in result.probe if not e.is_abstention),
        sum(1 for e in result.probe if e.is_abstention),
    )
    logger.info("Wrote manifest to %s", manifest_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
