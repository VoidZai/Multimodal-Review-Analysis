"""Constructed correct-abstention training examples (T7.4; PLAN.md §3 E8, §10, M7.md T7.4).

Builds the abstention half of the fine-tuning training set *deliberately*, rather than by
hand-authoring unanswerable-looking questions the way T2.3 originally did for CRAGB. PLAN.md
§14.2 already tested that assumption against real data and it mostly failed: 9 of 11
hand-authored "negative" questions turned out to have genuine relevant evidence once
pooled and labeled. Construction beats authoring here because the ground truth is
*structural*, not a guess about what a shopper wouldn't be able to find out: if a context
provably cannot support an answer -- by removing its evidence, by pairing it with a question
about something else entirely, or by asking for information no customer review anywhere
could ever contain -- the correct answer is exactly `ABSTENTION_TEXT`, full stop, and that
correctness doesn't depend on guessing what real shoppers happen to ask.

Three independent construction methods, each tagged in `provenance["method"]`:

- **`transplant`** pairs a real question (from a `TrainingExample` T7.3's teacher already
  answered) with a *different* context -- different product, different taxonomy category --
  guarded by `overlap_ratio`: if too many of the question's distinctive words show up in the
  new context's text, the pairing is rejected rather than risk an accidentally-answerable
  transplant becoming a mislabelled "must abstain".
- **`categorical_absence`** asks for information no customer review, for any product, could
  ever report -- internal QA rates, lab measurements, wholesale cost, dye-lot numbers. PLAN.md
  §14.2 identified exactly which shape of question actually survived contact with real
  pooled data (`fabric_quality_neg_000`'s exact-thread-count question,
  `defects_neg_000`'s internal-defect-rate question) -- `CATEGORICAL_ABSENCE_QUESTIONS`
  is built from that shape, not from the 9 negatives that turned out answerable.
- **`evidence_stripped`** takes a real (question, answer) pair, removes the specific reviews
  the answer cited from its context, and keeps the rest -- the hardest and most valuable
  case, because the remaining context still *looks* relevant (same product, same topic).
  This module goes one step further than the base method PLAN.md §3 E8 describes: even
  after removing the cited evidence, the *other*, never-cited reviews in a five-review
  context could coincidentally still answer the same question. `overlap_ratio` is applied
  a second time, against the stripped context, to catch that residual-leak risk before it
  becomes a mislabelled abstention -- exactly the failure mode this module's own
  verification step ("hand-read 20 examples, try to answer them myself") exists to catch.

Every produced `TrainingExample` is `answer=ABSTENTION_TEXT` exactly, `cited_doc_ids=()`,
`is_abstention=True` -- `TrainingExample.__post_init__` (T7.1) already enforces the
containment/consistency invariant between these three fields, so nothing here re-validates
it; a bug in this module's construction would surface as a `TrainingExample` constructor
`ValueError`, not a silent bad record.

Usage:
    python -m cragb.finetune.abstentions --config configs/finetune.yaml
"""

from __future__ import annotations

import argparse
import logging
import random
import re
import sys
from collections import Counter
from dataclasses import dataclass

from cragb.bench.reference_answers import ABSTENTION_TEXT
from cragb.finetune.sample_contexts import ContextGroup, load_contexts_jsonl
from cragb.finetune.schema import (
    VALID_CATEGORIES,
    TrainingExample,
    load_training_examples_jsonl,
    write_training_examples_jsonl,
)
from cragb.utils.io import load_config
from cragb.utils.seeds import set_global_seed

logger = logging.getLogger(__name__)

VALID_METHODS = frozenset({"transplant", "categorical_absence", "evidence_stripped"})

# Curated per PLAN.md §14.2: information categorically absent from *every* customer
# review, regardless of product -- internal manufacturer/seller data, lab-controlled
# measurements, or figures no shopper could observe. `fabric_quality`'s and `defects`'
# first entries are the exact shape §14.2 confirmed against real pooled data
# (`fabric_quality_neg_000`, `defects_neg_000`); the rest extend the same shape --
# "internal", "manufacturer's", "lab", "wholesale" -- to the remaining 5 categories.
CATEGORICAL_ABSENCE_QUESTIONS: dict[str, tuple[str, ...]] = {
    "fit_sizing": (
        "What are the manufacturer's exact garment measurements, in centimeters, for each size?",
        "What cutting tolerance, in millimeters, does the factory allow for each size?",
    ),
    "colour_appearance": (
        "What is the exact Pantone or RAL colour code used in manufacturing this item?",
        "What dye-lot number was used to produce this batch?",
    ),
    "fabric_quality": (
        "What is the exact thread count of the fabric?",
        "What tensile strength, in pounds per square inch, does the fabric mill's spec sheet report?",
    ),
    "durability": (
        "How many wash cycles did the manufacturer's internal durability testing show before failure?",
        "What abrasion-resistance rating, in Martindale cycles, did the factory's lab testing record?",
    ),
    "defects": (
        "What percentage of units shipped had a manufacturing defect, according to internal QA data?",
        "How many units were rejected at the factory's quality-control stage before shipping?",
    ),
    "occasion": (
        "What internal market-research data did the brand use to decide which occasions to market this for?",
        "What percentage of buyers, per the seller's internal analytics, purchased this for a wedding?",
    ),
    "value": (
        "What is the manufacturer's wholesale cost per unit?",
        "What profit margin does the seller make on each unit sold?",
    ),
}
assert set(CATEGORICAL_ABSENCE_QUESTIONS) == set(VALID_CATEGORIES), (
    "CATEGORICAL_ABSENCE_QUESTIONS must cover exactly the 7 taxonomy categories"
)

# A small, curated function-word list -- the same "small hand-picked list, reviewable as
# code" precedent `cragb.data.vocab_check.ATTRIBUTE_KEYWORDS` and
# `cragb.bench.taxonomy.CATEGORY_KEYWORDS` already set -- used only to strip words that
# carry no topical signal before computing `overlap_ratio`.
_STOPWORDS = frozenset(
    {
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
        "do", "does", "did", "doing", "will", "would", "can", "could", "should",
        "shall", "may", "might", "must",
        "this", "that", "these", "those", "it", "its", "they", "them", "their",
        "you", "your", "yours", "i", "my", "we", "our", "he", "she", "his", "her",
        "for", "of", "to", "in", "on", "at", "with", "by", "from", "as", "if",
        "and", "or", "but", "not", "no", "nor", "so", "than", "then", "there", "here",
        "what", "when", "where", "why", "how", "who", "which", "whose",
        "one", "some", "any", "all", "each", "every", "such",
        "have", "has", "had", "about", "into", "out", "up", "down", "over",
        "just", "really", "very", "much", "more", "most", "like",
    }
)

_WORD_RE = re.compile(r"[a-z']+")


def _content_words(text: str) -> set[str]:
    """Lowercased words with length > 2, minus `_STOPWORDS` -- the topical signal of `text`."""
    return {w for w in _WORD_RE.findall(text.lower()) if len(w) > 2 and w not in _STOPWORDS}


def overlap_ratio(question: str, text: str) -> float:
    """Fraction of `question`'s content words that appear (word-boundary, case-insensitive) in `text`.

    `1.0` means every distinctive word in the question shows up somewhere in `text`
    (strong topical overlap -- likely still answerable); `0.0` means none do. A question
    with no content words at all (rare -- e.g. after stripping, nothing but stopwords
    remain) returns `0.0` rather than dividing by zero, since there is nothing left that
    *could* overlap.
    """
    words = _content_words(question)
    if not words:
        return 0.0
    text_lower = text.lower()
    matched = sum(1 for w in words if re.search(rf"\b{re.escape(w)}\b", text_lower))
    return matched / len(words)


def _split_context_blocks(context_text: str, doc_ids: tuple[str, ...]) -> dict[str, str]:
    """Recover each doc's individually-rendered excerpt block from a joined `context_text`.

    `cragb.finetune.sample_contexts.sample_contexts` joins per-doc excerpts
    (`cragb.generate.context_builder.render_excerpt`'s `"[doc_id] has_photo:
    yes/no\\n<snippet>"` shape) with `"\\n\\n"`. Splitting on that separator naively would
    be wrong here: a review's own raw text can itself contain a blank line, which would
    misattribute part of one review's snippet into a neighbouring block. Locating each
    known `"[doc_id] has_photo:"` marker explicitly, in `doc_ids`' order, is immune to
    that -- it never depends on where a blank line happens to fall inside a snippet.

    Args:
        context_text: a `ContextGroup.context_text` value.
        doc_ids: that same group's `doc_ids`, in the order they were rendered.

    Returns:
        `{doc_id: its excerpt block, trailing separator newlines stripped}`.

    Raises:
        ValueError: if a doc id's marker can't be found in `context_text` at or after the
            previous doc id's marker -- indicates `context_text`/`doc_ids` weren't built
            together by `sample_contexts` (a caller bug, not a data quality issue).
    """
    positions: list[int] = []
    search_from = 0
    for doc_id in doc_ids:
        marker = f"[{doc_id}] has_photo:"
        idx = context_text.find(marker, search_from)
        if idx == -1:
            raise ValueError(
                f"Marker {marker!r} not found in context_text at or after position "
                f"{search_from}; context_text and doc_ids must come from the same "
                "ContextGroup."
            )
        positions.append(idx)
        search_from = idx + len(marker)

    blocks: dict[str, str] = {}
    for i, doc_id in enumerate(doc_ids):
        start = positions[i]
        end = positions[i + 1] if i + 1 < len(positions) else len(context_text)
        blocks[doc_id] = context_text[start:end].rstrip("\n")
    return blocks


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class AbstentionConfig:
    """Resolved, validated `configs/finetune.yaml`'s `abstention:` block (T7.1)."""

    seed: int
    target_share: float
    methods: tuple[str, ...]
    transplant_overlap_threshold: float
    evidence_stripped_min_remaining_docs: int

    def __post_init__(self) -> None:
        if not 0.0 <= self.target_share < 1.0:
            raise ValueError(f"target_share must be in [0, 1), got {self.target_share}")
        if not self.methods:
            raise ValueError("methods must not be empty")
        unknown = set(self.methods) - VALID_METHODS
        if unknown:
            raise ValueError(f"Unknown abstention method(s) {sorted(unknown)}; must be one of {sorted(VALID_METHODS)}")
        if len(set(self.methods)) != len(self.methods):
            raise ValueError(f"methods must not contain duplicates: {self.methods}")
        if not 0.0 <= self.transplant_overlap_threshold <= 1.0:
            raise ValueError(
                f"transplant_overlap_threshold must be in [0, 1], got {self.transplant_overlap_threshold}"
            )
        if self.evidence_stripped_min_remaining_docs < 0:
            raise ValueError(
                "evidence_stripped_min_remaining_docs must be non-negative, got "
                f"{self.evidence_stripped_min_remaining_docs}"
            )


def load_abstention_config(config_path: str = "configs/finetune.yaml") -> AbstentionConfig:
    """Load and validate the `seed` + `abstention:` block of a fine-tuning config YAML."""
    cfg = load_config(config_path)
    a = cfg["abstention"]
    return AbstentionConfig(
        seed=int(cfg["seed"]),
        target_share=float(a["target_share"]),
        methods=tuple(a["methods"]),
        transplant_overlap_threshold=float(a["transplant_overlap_threshold"]),
        evidence_stripped_min_remaining_docs=int(a["evidence_stripped_min_remaining_docs"]),
    )


# --------------------------------------------------------------------------
# Method 1: transplant
# --------------------------------------------------------------------------


def _build_transplant(
    positives: list[TrainingExample],
    contexts: list[ContextGroup],
    context_by_id: dict[str, ContextGroup],
    n_target: int,
    overlap_threshold: float,
    rng: random.Random,
) -> list[TrainingExample]:
    """Pair each of (up to `n_target`) shuffled positives' questions with an unrelated,
    overlap-guarded context. One accepted transplant per source positive at most.
    """
    shuffled_positives = list(positives)
    rng.shuffle(shuffled_positives)

    accepted: list[TrainingExample] = []
    for p in shuffled_positives:
        if len(accepted) >= n_target:
            break
        source_group_id = p.provenance.get("context_group_id")
        if source_group_id is None or source_group_id not in context_by_id:
            continue
        source_context = context_by_id[source_group_id]

        candidates = [
            c
            for c in contexts
            if c.parent_asin != source_context.parent_asin and c.category != source_context.category
        ]
        rng.shuffle(candidates)

        for target_context in candidates:
            ratio = overlap_ratio(p.question, target_context.context_text)
            if ratio > overlap_threshold:
                continue
            accepted.append(
                TrainingExample(
                    example_id=f"abst_transplant_{len(accepted):04d}",
                    category=p.category,
                    source_doc_ids=target_context.doc_ids,
                    source_parent_asins=(target_context.parent_asin,),
                    question=p.question,
                    context_text=target_context.context_text,
                    answer=ABSTENTION_TEXT,
                    cited_doc_ids=(),
                    is_abstention=True,
                    provenance={
                        "method": "transplant",
                        "source_positive_example_id": p.example_id,
                        "source_context_group_id": source_context.group_id,
                        "target_context_group_id": target_context.group_id,
                        "overlap_ratio": ratio,
                    },
                )
            )
            break  # one safe transplant per source positive; move on

    return accepted


# --------------------------------------------------------------------------
# Method 2: categorical_absence
# --------------------------------------------------------------------------


def _build_categorical_absence(
    contexts: list[ContextGroup],
    n_target: int,
    rng: random.Random,
) -> list[TrainingExample]:
    """Pair every (context, categorical-absence question) combination in the same
    category, shuffle, and take the first `n_target` -- no overlap guard needed, since
    these questions are unanswerable from customer review text by construction (PLAN.md
    §14.2), not merely by chance, for any product in any context.
    """
    by_category: dict[str, list[ContextGroup]] = {}
    for c in contexts:
        by_category.setdefault(c.category, []).append(c)

    candidates: list[tuple[ContextGroup, str]] = [
        (context, question)
        for category, questions in CATEGORICAL_ABSENCE_QUESTIONS.items()
        for context in by_category.get(category, [])
        for question in questions
    ]
    rng.shuffle(candidates)

    accepted: list[TrainingExample] = []
    for context, question in candidates[:n_target]:
        accepted.append(
            TrainingExample(
                example_id=f"abst_categorical_absence_{len(accepted):04d}",
                category=context.category,
                source_doc_ids=context.doc_ids,
                source_parent_asins=(context.parent_asin,),
                question=question,
                context_text=context.context_text,
                answer=ABSTENTION_TEXT,
                cited_doc_ids=(),
                is_abstention=True,
                provenance={"method": "categorical_absence", "context_group_id": context.group_id},
            )
        )
    return accepted


# --------------------------------------------------------------------------
# Method 3: evidence_stripped
# --------------------------------------------------------------------------


def _build_evidence_stripped(
    positives: list[TrainingExample],
    context_by_id: dict[str, ContextGroup],
    n_target: int,
    min_remaining_docs: int,
    overlap_threshold: float,
    rng: random.Random,
) -> list[TrainingExample]:
    """Strip each (up to `n_target`) shuffled positive's cited evidence from its own
    context, keeping the rest -- guarded on both ends: `min_remaining_docs` (a context
    stripped to zero documents is a different, degenerate case) and a residual
    `overlap_ratio` check on what's *left* (the other, never-cited reviews in the same
    context could coincidentally still answer the question even after the cited ones are
    gone -- see module docstring).
    """
    shuffled_positives = list(positives)
    rng.shuffle(shuffled_positives)

    accepted: list[TrainingExample] = []
    for p in shuffled_positives:
        if len(accepted) >= n_target:
            break
        if not p.cited_doc_ids:
            continue
        group_id = p.provenance.get("context_group_id")
        if group_id is None or group_id not in context_by_id:
            continue
        context = context_by_id[group_id]

        cited = set(p.cited_doc_ids)
        remaining_doc_ids = tuple(d for d in context.doc_ids if d not in cited)
        if len(remaining_doc_ids) < min_remaining_docs:
            continue

        blocks = _split_context_blocks(context.context_text, context.doc_ids)
        stripped_text = "\n\n".join(blocks[d] for d in remaining_doc_ids)

        ratio = overlap_ratio(p.question, stripped_text)
        if ratio > overlap_threshold:
            continue

        accepted.append(
            TrainingExample(
                example_id=f"abst_evidence_stripped_{len(accepted):04d}",
                category=context.category,
                source_doc_ids=remaining_doc_ids,
                source_parent_asins=(context.parent_asin,),
                question=p.question,
                context_text=stripped_text,
                answer=ABSTENTION_TEXT,
                cited_doc_ids=(),
                is_abstention=True,
                provenance={
                    "method": "evidence_stripped",
                    "source_positive_example_id": p.example_id,
                    "source_context_group_id": context.group_id,
                    "n_docs_removed": len(context.doc_ids) - len(remaining_doc_ids),
                    "overlap_ratio": ratio,
                },
            )
        )

    return accepted


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------


def build_abstentions(
    positives: list[TrainingExample],
    contexts: list[ContextGroup],
    config: AbstentionConfig,
) -> list[TrainingExample]:
    """Build a stratified mix of constructed-abstention `TrainingExample`s.

    The overall target is sized so abstentions make up `config.target_share` of the
    *combined* (positives + abstentions) set: solving `n / (n_positives + n) ==
    target_share` for `n` gives `n = target_share * n_positives / (1 - target_share)`.
    That total is then split evenly (remainder distributed to the first methods in
    `config.methods`' order) across `config.methods`; a method whose pool is smaller than
    its share is logged as a shortfall and simply yields fewer, mirroring
    `cragb.finetune.sample_contexts.sample_contexts`'s quota-shortfall handling -- no
    cross-method redistribution, so the achieved mix is always visible from the returned
    examples' `provenance["method"]`, not silently smoothed over.

    Args:
        positives: T7.3's accepted grounded, cited examples (e.g.
            `cragb.finetune.schema.load_training_examples_jsonl` on `raw_pairs_v1.jsonl`).
            Each must carry `provenance["context_group_id"]` pointing back to the
            `ContextGroup` it was generated from (T7.3's `generate_pairs.parse_generated_pairs`
            always sets this) -- a positive missing it is skipped rather than raising,
            since a hand-built or otherwise-sourced positive without that pointer simply
            can't be traced back to its context.
        contexts: T7.2's sampled context groups (e.g. `sample_contexts.load_contexts_jsonl`
            on `contexts_v1.jsonl`).
        config: `AbstentionConfig`.

    Returns:
        Accepted abstention `TrainingExample`s across all configured methods, grouped by
        method in `config.methods`' order (not interleaved) -- `example_id`s are
        `f"abst_{method}_{i:04d}"`, independently numbered per method.
    """
    if not positives:
        logger.warning("No positives given; returning zero abstentions.")
        return []

    context_by_id = {c.group_id: c for c in contexts}
    n_total_target = round(config.target_share * len(positives) / (1 - config.target_share))

    n_methods = len(config.methods)
    base = n_total_target // n_methods
    remainder = n_total_target % n_methods
    per_method_target = {
        method: base + (1 if i < remainder else 0) for i, method in enumerate(config.methods)
    }

    rng = random.Random(config.seed)

    all_abstentions: list[TrainingExample] = []
    for method in config.methods:
        n_target = per_method_target[method]
        if method == "transplant":
            built = _build_transplant(
                positives, contexts, context_by_id, n_target, config.transplant_overlap_threshold, rng
            )
        elif method == "categorical_absence":
            built = _build_categorical_absence(contexts, n_target, rng)
        else:  # evidence_stripped
            built = _build_evidence_stripped(
                positives,
                context_by_id,
                n_target,
                config.evidence_stripped_min_remaining_docs,
                config.transplant_overlap_threshold,
                rng,
            )

        if len(built) < n_target:
            logger.warning(
                "Method %r: only %d/%d abstentions built (pool exhausted).",
                method,
                len(built),
                n_target,
            )
        all_abstentions.extend(built)

    return all_abstentions


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
    config = load_abstention_config(args.config)
    set_global_seed(config.seed)

    positives = load_training_examples_jsonl(raw_cfg["paths"]["raw_pairs_out"])
    contexts = load_contexts_jsonl(raw_cfg["paths"]["contexts_out"])

    abstentions = build_abstentions(positives, contexts, config)
    out_path = write_training_examples_jsonl(abstentions, raw_cfg["paths"]["abstentions_out"])

    counts = Counter(ex.provenance.get("method", "unknown") for ex in abstentions)
    logger.info("Loaded %d positives, %d context groups", len(positives), len(contexts))
    logger.info("Wrote %d abstention examples to %s", len(abstentions), out_path)
    for method in config.methods:
        logger.info("  %s: %d", method, counts.get(method, 0))

    return 0


if __name__ == "__main__":
    sys.exit(main())
