"""Training-context sampler, disjoint from CRAGB's gold evidence (T7.2; PLAN.md §3 E8, M7.md T7.2).

Chooses which corpus documents the fine-tuning training set is allowed to be built from.
"Disjoint from CRAGB questions" (PLAN.md §3 E8) is necessary but not sufficient: if a
training context is built from the *same reviews* CRAGB's `relevant_ids` point at, the
tuned model has memorised the eval's evidence even when the training question is worded
differently from any CRAGB question. This module closes that gap at the *product* level,
not just the *document* level.

**Exclusion is at the `parent_asin` level, chosen deliberately over the weaker
per-document alternative.** `cragb_evidence_doc_ids` collects every doc id CRAGB's ground
truth touches (pooled `relevant_ids`, reference-answer `cited_doc_ids`, and — for
completeness — every id in `benchmark/pools_v1.jsonl`, since the pools are what T2.7's
relevance labels were drawn from). `excluded_parent_asins` then maps those doc ids to the
*products* they belong to, and `sample_contexts` removes every review of those products
from the candidate pool entirely — not just the specific evidence reviews. `corpus_v1` has
139,448 unique `parent_asin`s against at most ~1,200 pooled evidence documents, so this
costs essentially nothing in candidate-pool size and buys the stronger guarantee outright:
the intersection of sampled `parent_asin`s with CRAGB-evidence `parent_asin`s is empty by
construction, not just the intersection of sampled *doc ids*. (See M7.md T7.2's validation
checks: "empty, or the count is reported explicitly as a knowingly accepted weaker
guarantee — pick one and write down which." This module picks the stronger guarantee.)

**Category assignment reuses the taxonomy's own keyword vocabulary**
(`cragb.bench.taxonomy.TaxonomySpec.keyword_lists`, via `cragb.data.vocab_check
.matches_any_keyword` — the same public reuse point `cragb.bench.curate` and
`cragb.generate.draft_questions` already use for the identical "does this review mention
this category's vocabulary?" check) rather than a second classification scheme. A
context group's category is whichever of the 7 categories the group's deduplicated
candidate reviews match most often; a group matching none of them is dropped rather than
force-assigned, since a category label with zero evidence behind it would be worse than
no label.

**Group construction, in one pass, is deliberately vectorized rather than a per-group
Python loop** — a per-group `DataFrame.groupby` `.sum()` and one seeded global shuffle
(`numpy.random.default_rng(seed).permutation`) does the work that a naive implementation
would otherwise do with a regex call and a dedup pass per candidate product, which does
not scale past a few thousand `parent_asin`s. The one global shuffle also does double
duty: because pandas' `sort=False` groupby preserves each group's first-appearance order,
shuffling the whole candidate frame once, up front, makes every later per-group operation
(the deduplication tie-break, the category-pool ordering, the photo-priority selection)
already randomized without a second RNG call anywhere else in the pipeline.

Usage:
    python -m cragb.finetune.sample_contexts --config configs/finetune.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from cragb.bench.taxonomy import TaxonomySpec, load_taxonomy
from cragb.data.vocab_check import matches_any_keyword
from cragb.finetune.schema import VALID_CATEGORIES
from cragb.generate.context_builder import render_excerpt
from cragb.utils.io import load_config, resolve_path, sha256_file
from cragb.utils.seeds import set_global_seed

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SamplingConfig:
    """Resolved, validated `configs/finetune.yaml`'s `sampling:` block (T7.1)."""

    seed: int
    n_contexts: int
    k: int
    max_review_chars: int
    category_quotas: dict[str, int]
    photo_bearing_target_share: float

    def __post_init__(self) -> None:
        if self.k <= 0:
            raise ValueError(f"k must be positive, got {self.k}")
        if self.max_review_chars <= 0:
            raise ValueError(f"max_review_chars must be positive, got {self.max_review_chars}")
        if not 0.0 <= self.photo_bearing_target_share <= 1.0:
            raise ValueError(
                f"photo_bearing_target_share must be in [0, 1], got {self.photo_bearing_target_share}"
            )
        if not self.category_quotas:
            raise ValueError("category_quotas must not be empty")
        unknown = set(self.category_quotas) - set(VALID_CATEGORIES)
        if unknown:
            raise ValueError(f"category_quotas has unknown categories: {sorted(unknown)}")
        for name, quota in self.category_quotas.items():
            if quota < 0:
                raise ValueError(f"category_quotas[{name!r}] must be non-negative, got {quota}")
        quota_total = sum(self.category_quotas.values())
        if self.n_contexts != quota_total:
            raise ValueError(
                f"n_contexts ({self.n_contexts}) must equal sum(category_quotas.values()) "
                f"({quota_total}) -- configs/finetune.yaml's sampling block is out of sync"
            )


def load_sampling_config(config_path: str | Path = "configs/finetune.yaml") -> SamplingConfig:
    """Load and validate the `seed` + `sampling:` block of a fine-tuning config YAML."""
    cfg = load_config(config_path)
    sampling = cfg["sampling"]
    return SamplingConfig(
        seed=int(cfg["seed"]),
        n_contexts=int(sampling["n_contexts"]),
        k=int(sampling["k"]),
        max_review_chars=int(sampling["max_review_chars"]),
        category_quotas={str(name): int(q) for name, q in sampling["category_quotas"].items()},
        photo_bearing_target_share=float(sampling["photo_bearing_target_share"]),
    )


# --------------------------------------------------------------------------
# CRAGB evidence
# --------------------------------------------------------------------------


def load_cragb_entries(path: str | Path = "benchmark/cragb_v1.jsonl") -> list[dict]:
    """Load `cragb_v1.jsonl` as raw parsed dicts.

    Deliberately not `cragb.bench.assemble.CragbEntry` (coupled to that module's own
    write-side fields) or `cragb.eval.cragb_questions.RetrievalQuestion` (doesn't carry
    `cited_doc_ids`, which this module also needs) — mirrors
    `cragb.eval.cragb_questions`'s own stated reasoning for defining a narrow, local,
    read-only view of the finished artifact rather than importing across modules for a
    shape that doesn't quite fit.
    """
    entries: list[dict] = []
    with resolve_path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entries.append(json.loads(line))
    return entries


def load_pool_doc_ids(path: str | Path = "benchmark/pools_v1.jsonl") -> set[str]:
    """Union of every `doc_ids` entry across all pooled questions in `pools_v1.jsonl`."""
    doc_ids: set[str] = set()
    with resolve_path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            doc_ids.update(str(d) for d in obj.get("doc_ids", ()))
    return doc_ids


def cragb_evidence_doc_ids(entries: list[dict], pool_doc_ids: Iterable[str] = ()) -> set[str]:
    """Every doc id CRAGB's ground truth touches: `relevant_ids` + `cited_doc_ids` + pools.

    Args:
        entries: parsed `cragb_v1.jsonl` rows (e.g. from `load_cragb_entries`).
        pool_doc_ids: doc ids from `benchmark/pools_v1.jsonl` (e.g. from
            `load_pool_doc_ids`) — included "for completeness" per M7.md T7.2, since the
            pools are the superset T2.7's relevance labels were drawn from and may contain
            reviews a human labeled *irrelevant* but which are still, in a strict sense,
            evidence CRAGB's construction process looked at. Defaults to empty so this
            function is independently testable against just `relevant_ids`/`cited_doc_ids`.

    Returns:
        The union, as strings.
    """
    evidence: set[str] = set()
    for entry in entries:
        evidence.update(str(d) for d in entry.get("relevant_ids", ()))
        evidence.update(str(d) for d in entry.get("cited_doc_ids", ()))
    evidence.update(str(d) for d in pool_doc_ids)
    return evidence


def excluded_parent_asins(corpus: pd.DataFrame, exclude_doc_ids: set[str]) -> set[str]:
    """`parent_asin`s of every corpus row whose doc id (`corpus.index.astype(str)`) is in
    `exclude_doc_ids` -- the products this module's stronger, product-level exclusion
    guarantee removes entirely from the candidate pool (see module docstring).
    """
    doc_ids = corpus.index.astype(str)
    mask = doc_ids.isin(exclude_doc_ids)
    return set(corpus.loc[mask, "parent_asin"].astype(str))


# --------------------------------------------------------------------------
# ContextGroup
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ContextGroup:
    """One sampled training context: `k` reviews of one product, one taxonomy category.

    Attributes:
        group_id: stable id, `f"ctx_{category}_{i:04d}"`.
        category: one of `cragb.finetune.schema.VALID_CATEGORIES`.
        parent_asin: the product every review in this group belongs to.
        doc_ids: the `k` selected doc ids, in the same order they appear in
            `context_text` (photo-bearing reviews first, within a category's photo-priority
            selection; see `sample_contexts`).
        context_text: the fully rendered `$context_block` value -- built from
            `cragb.generate.context_builder.render_excerpt`, the exact function
            `build_context` uses for a real retrieved context, so
            `cragb.finetune.schema.render_training_prompt` reproduces the same prompt
            shape for a sampler-built group as it would for a real one.
        photo_bearing: whether at least one of `doc_ids` has a photo.
    """

    group_id: str
    category: str
    parent_asin: str
    doc_ids: tuple[str, ...]
    context_text: str
    photo_bearing: bool

    def __post_init__(self) -> None:
        if self.category not in VALID_CATEGORIES:
            raise ValueError(
                f"{self.group_id}: unknown category {self.category!r}; "
                f"must be one of {sorted(VALID_CATEGORIES)}"
            )
        if not self.doc_ids:
            raise ValueError(f"{self.group_id}: doc_ids must not be empty")
        if len(set(self.doc_ids)) != len(self.doc_ids):
            raise ValueError(f"{self.group_id}: doc_ids must not contain duplicates: {self.doc_ids}")

    def to_dict(self) -> dict:
        return {
            "group_id": self.group_id,
            "category": self.category,
            "parent_asin": self.parent_asin,
            "doc_ids": list(self.doc_ids),
            "context_text": self.context_text,
            "photo_bearing": self.photo_bearing,
        }

    @classmethod
    def from_dict(cls, obj: dict) -> "ContextGroup":
        return cls(
            group_id=obj["group_id"],
            category=obj["category"],
            parent_asin=obj["parent_asin"],
            doc_ids=tuple(obj["doc_ids"]),
            context_text=obj["context_text"],
            photo_bearing=obj["photo_bearing"],
        )


def write_contexts_jsonl(groups: list[ContextGroup], out_path: str | Path) -> Path:
    """Write `groups` as newline-delimited JSON, one object per line."""
    resolved = resolve_path(out_path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("w", encoding="utf-8") as f:
        for g in groups:
            f.write(json.dumps(g.to_dict(), ensure_ascii=False))
            f.write("\n")
    return resolved


def load_contexts_jsonl(path: str | Path) -> list[ContextGroup]:
    """Load context groups written by `write_contexts_jsonl` (T7.3's input)."""
    groups: list[ContextGroup] = []
    with resolve_path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            groups.append(ContextGroup.from_dict(json.loads(line)))
    return groups


# --------------------------------------------------------------------------
# Sampling
# --------------------------------------------------------------------------


def _build_doc_frame(corpus: pd.DataFrame) -> pd.DataFrame:
    """One row per non-empty-text review: `doc_id`, `parent_asin`, `user_id`, `text`, `has_image`.

    `doc_id` is `corpus.index.astype(str)` -- the same default id
    `cragb.generate.context_builder.build_corpus_lookup`/`cragb.retrieval.base.Retriever
    .index` use, so a `doc_id` here resolves the same way everywhere else in the project.
    """
    frame = pd.DataFrame(
        {
            "doc_id": corpus.index.astype(str),
            "parent_asin": corpus["parent_asin"].astype(str),
            "user_id": corpus["user_id"].astype(str),
            "text": corpus["text"].fillna("").astype(str),
            "has_image": corpus["has_image"].fillna(False).astype(bool),
        }
    )
    frame = frame[frame["text"].str.strip() != ""]
    return frame.reset_index(drop=True)


def sample_contexts(
    corpus: pd.DataFrame,
    taxonomy: TaxonomySpec,
    *,
    exclude_doc_ids: set[str],
    config: SamplingConfig,
) -> list[ContextGroup]:
    """Sample `config.n_contexts` training contexts, stratified by category and disjoint
    from CRAGB's evidence products.

    Pipeline (see module docstring for why each step is shaped this way):
      1. Build a one-row-per-review candidate frame, dropping every review of a product
         (`parent_asin`) that owns any doc id in `exclude_doc_ids`.
      2. Shuffle the whole candidate frame once under `config.seed`.
      3. Deduplicate to one review per `(parent_asin, user_id)` pair (a shopper reviewing
         the same product twice is not two independent pieces of evidence), keeping the
         first (post-shuffle, i.e. random) occurrence.
      4. Keep only products with >= `config.k` deduplicated candidate reviews.
      5. Assign each product a taxonomy category by keyword-match count (ties broken by
         `taxonomy`'s declared category order); products matching no category's keywords
         at all are dropped.
      6. Per category, fill the configured quota, deliberately over-selecting
         photo-bearing products up to `config.photo_bearing_target_share` before filling
         the remainder, falling back to whatever is available (and logging the shortfall)
         if a category's pool is smaller than its quota.
      7. For each selected product, take its `k` highest-`has_image`-priority deduplicated
         reviews (stable sort, so ties keep the step-2 shuffled order) and render them into
         one `ContextGroup`.

    Raises:
        ValueError: if `config.category_quotas`'s categories don't exactly match
            `taxonomy`'s declared categories.
    """
    category_names = [c.name for c in taxonomy.categories]
    if set(config.category_quotas) != set(category_names):
        raise ValueError(
            "config.category_quotas categories do not match taxonomy categories: "
            f"{sorted(config.category_quotas)} vs {sorted(category_names)}"
        )

    doc_frame = _build_doc_frame(corpus)
    excluded_pas = excluded_parent_asins(corpus, exclude_doc_ids)
    candidates = doc_frame[~doc_frame["parent_asin"].isin(excluded_pas)]
    if candidates.empty:
        logger.warning("No candidate reviews remain after excluding CRAGB-evidence products.")
        return []

    rng = np.random.default_rng(config.seed)
    shuffled_order = rng.permutation(len(candidates))
    candidates = candidates.iloc[shuffled_order].reset_index(drop=True)

    candidates = candidates.drop_duplicates(subset=["parent_asin", "user_id"], keep="first")

    group_sizes = candidates.groupby("parent_asin", sort=False).size()
    viable_pas = group_sizes[group_sizes >= config.k].index
    candidates = candidates[candidates["parent_asin"].isin(viable_pas)].reset_index(drop=True)
    if candidates.empty:
        logger.warning(
            "No parent_asin has >= k=%d distinct-reviewer candidate reviews after exclusion.",
            config.k,
        )
        return []

    match_matrix = pd.DataFrame(
        {name: matches_any_keyword(candidates["text"], keywords) for name, keywords in taxonomy.keyword_lists.items()},
        index=candidates.index,
    )[category_names]
    cat_counts = match_matrix.groupby(candidates["parent_asin"], sort=False).sum()
    has_any_match = cat_counts.sum(axis=1) > 0
    assigned_category = cat_counts.loc[has_any_match].idxmax(axis=1)
    n_dropped_no_match = int((~has_any_match).sum())
    if n_dropped_no_match:
        logger.info(
            "%d/%d viable parent_asin groups matched no taxonomy category's keywords and were dropped.",
            n_dropped_no_match,
            len(cat_counts),
        )

    photo_capable = candidates.groupby("parent_asin", sort=False)["has_image"].any()

    groups_by_category: dict[str, list[str]] = {name: [] for name in category_names}
    for parent_asin, category in assigned_category.items():
        groups_by_category[category].append(parent_asin)

    selected: list[tuple[str, str]] = []  # (category, parent_asin)
    for category_name in category_names:
        quota = config.category_quotas[category_name]
        pool = groups_by_category[category_name]
        photo_pool = [pa for pa in pool if photo_capable.get(pa, False)]
        nonphoto_pool = [pa for pa in pool if not photo_capable.get(pa, False)]

        target_photo_n = round(quota * config.photo_bearing_target_share)
        chosen_photo = photo_pool[:target_photo_n]
        remaining_needed = quota - len(chosen_photo)
        remaining_pool = photo_pool[len(chosen_photo):] + nonphoto_pool
        chosen = chosen_photo + remaining_pool[:remaining_needed]

        if len(chosen) < quota:
            logger.warning(
                "Category %r: only %d/%d groups available (pool size %d) -- shortfall %d.",
                category_name,
                len(chosen),
                quota,
                len(pool),
                quota - len(chosen),
            )
        selected.extend((category_name, pa) for pa in chosen)

    grouped_candidates = candidates.groupby("parent_asin", sort=False)
    per_category_index: dict[str, int] = {name: 0 for name in category_names}
    groups: list[ContextGroup] = []
    for category_name, parent_asin in selected:
        rows = grouped_candidates.get_group(parent_asin)
        rows = rows.sort_values("has_image", ascending=False, kind="stable").head(config.k)

        excerpts = [
            render_excerpt(row.doc_id, row.text, row.has_image, config.max_review_chars)
            for row in rows.itertuples(index=False)
        ]
        doc_ids = tuple(rows["doc_id"])
        context_text = "\n\n".join(excerpts)
        photo_bearing = bool(rows["has_image"].any())

        idx = per_category_index[category_name]
        per_category_index[category_name] = idx + 1
        groups.append(
            ContextGroup(
                group_id=f"ctx_{category_name}_{idx:04d}",
                category=category_name,
                parent_asin=parent_asin,
                doc_ids=doc_ids,
                context_text=context_text,
                photo_bearing=photo_bearing,
            )
        )

    return groups


# --------------------------------------------------------------------------
# Manifest
# --------------------------------------------------------------------------


def build_contexts_manifest(
    groups: list[ContextGroup],
    *,
    seed: int,
    corpus_sha256: str,
    config_hash: str,
    n_docs_excluded_as_cragb_evidence: int,
    n_parent_asins_excluded: int,
    category_quotas: dict[str, int],
    photo_bearing_target_share: float,
) -> dict:
    """Assemble `contexts_v1_manifest.json`'s contents from an already-sampled `groups` list.

    A pure function over `groups` plus the scalar exclusion/config stats
    `sample_contexts` doesn't itself return (its signature is fixed to
    `list[ContextGroup]` per M7.md T7.2) -- kept separate so it's testable against a small
    hand-built `groups` list with no corpus or file I/O.
    """
    per_category_counts = Counter(g.category for g in groups)
    category_quota_report = {
        name: {
            "target": quota,
            "actual": per_category_counts.get(name, 0),
            "shortfall": max(0, quota - per_category_counts.get(name, 0)),
        }
        for name, quota in category_quotas.items()
    }
    photo_bearing_share = (sum(1 for g in groups if g.photo_bearing) / len(groups)) if groups else 0.0

    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "corpus_sha256": corpus_sha256,
        "config_hash": config_hash,
        "n_groups": len(groups),
        "per_category_counts": dict(sorted(per_category_counts.items())),
        "n_docs_excluded_as_cragb_evidence": n_docs_excluded_as_cragb_evidence,
        "n_parent_asins_excluded": n_parent_asins_excluded,
        "photo_bearing_share": photo_bearing_share,
        "photo_bearing_target_share": photo_bearing_target_share,
        "category_quota_report": category_quota_report,
    }


def write_contexts_manifest(manifest: dict, out_path: str | Path) -> Path:
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
    sampling_cfg = load_sampling_config(args.config)
    set_global_seed(sampling_cfg.seed)

    taxonomy = load_taxonomy(raw_cfg["paths"]["taxonomy_config"])
    corpus_path = resolve_path(raw_cfg["paths"]["corpus_in"])
    corpus = pd.read_parquet(corpus_path)

    entries = load_cragb_entries(raw_cfg["paths"]["cragb_questions_in"])
    pool_doc_ids = load_pool_doc_ids(raw_cfg["paths"]["cragb_pools_in"])
    evidence_doc_ids = cragb_evidence_doc_ids(entries, pool_doc_ids)
    # Recomputed here (cheap: one pass over the corpus) purely for the manifest's own
    # stats -- sample_contexts's return type is fixed to list[ContextGroup] per M7.md
    # T7.2, so the exclusion-stage counts aren't otherwise recoverable from its output.
    excluded_pas = excluded_parent_asins(corpus, evidence_doc_ids)

    groups = sample_contexts(corpus, taxonomy, exclude_doc_ids=evidence_doc_ids, config=sampling_cfg)

    contexts_path = write_contexts_jsonl(groups, raw_cfg["paths"]["contexts_out"])
    manifest = build_contexts_manifest(
        groups,
        seed=sampling_cfg.seed,
        corpus_sha256=sha256_file(corpus_path),
        config_hash=sha256_file(args.config),
        n_docs_excluded_as_cragb_evidence=len(evidence_doc_ids),
        n_parent_asins_excluded=len(excluded_pas),
        category_quotas=sampling_cfg.category_quotas,
        photo_bearing_target_share=sampling_cfg.photo_bearing_target_share,
    )
    manifest_path = write_contexts_manifest(manifest, raw_cfg["paths"]["contexts_manifest_out"])

    logger.info("Wrote %d context groups to %s", len(groups), contexts_path)
    logger.info("Wrote manifest to %s", manifest_path)
    logger.info(
        "Excluded %d CRAGB-evidence doc id(s) across %d parent_asin(s).",
        len(evidence_doc_ids),
        len(excluded_pas),
    )
    logger.info("%-20s %8s %8s %10s", "category", "target", "actual", "shortfall")
    for name, report in manifest["category_quota_report"].items():
        logger.info("%-20s %8d %8d %10d", name, report["target"], report["actual"], report["shortfall"])
    logger.info(
        "Photo-bearing share: %.1f%% (target >= %.0f%%)",
        manifest["photo_bearing_share"] * 100,
        manifest["photo_bearing_target_share"] * 100,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
