"""LLM-drafted CRAGB candidate questions (T2.2; PLAN.md §3 E1, §9).

Drafts candidate benchmark questions *from sampled reviews* rather than
from imagination, then over-generates per category so T2.3's human
curation has real material to select and rebalance from — this is the
fix PLAN.md §9 names for the "student only writes answerable questions"
selection bias: an LLM draft-then-human-curate loop, with humans always
in the loop, never auto-accepted.

Per category (from T2.1's `TaxonomySpec`):
  1. sample real corpus reviews that mention the category's vocabulary,
     as grounding context (`sample_context_reviews`);
  2. render the versioned prompt template with that context
     (`render_prompt`);
  3. call the LLM and parse its JSON-array response into `DraftQuestion`s
     (`generate_candidates_for_category`, `parse_llm_questions`);
  4. drop near-duplicate questions within the category (`dedup_questions`).

The LLM call is injected as a plain `chat_fn` callable everywhere in
this module (never imported directly), so every function above can be
unit-tested without network access or an API key — `main()` is the only
place a real `GroqClient` is constructed.

Usage:
    python -m cragb.generate.draft_questions --config configs/generate.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from pathlib import Path
from string import Template
from typing import Callable

import pandas as pd
from dotenv import load_dotenv

from cragb.bench.taxonomy import TaxonomyCategory, TaxonomySpec, load_taxonomy
from cragb.generate.api_clients import GroqClient
from cragb.utils.io import load_config, resolve_path
from cragb.utils.seeds import set_global_seed

logger = logging.getLogger(__name__)

# A chat function takes OpenAI-style messages and returns completion text.
# `GroqClient.complete` matches this shape; tests inject a plain function.
ChatFn = Callable[[list[dict[str, str]]], str]


@dataclass(frozen=True)
class DraftQuestion:
    id: str
    category: str
    question: str
    rationale: str


def load_prompt_template(path: str | Path) -> Template:
    """Load the versioned prompt template as a `string.Template`.

    `string.Template`'s `$name` placeholders are used instead of
    `str.format`'s `{name}` deliberately: the prompt instructs the model
    to emit JSON, and JSON is full of literal curly braces that
    `str.format` would try to interpret as fields.
    """
    text = resolve_path(path).read_text(encoding="utf-8")
    return Template(text)


def _matches_category(text: pd.Series, keywords: list[str]) -> pd.Series:
    """Boolean mask of `text` entries containing any of `keywords` (word-boundary, case-insensitive).

    A small, self-contained duplicate of the matching logic in
    `cragb.data.vocab_check._compile_category_pattern` — kept local
    rather than importing that private helper, so this module doesn't
    reach into another module's internals for four lines of regex.
    """
    escaped = sorted((re.escape(k) for k in keywords), key=len, reverse=True)
    pattern = r"\b(?:" + "|".join(escaped) + r")\b"
    return text.str.contains(pattern, regex=True, case=False, na=False)


def sample_context_reviews(
    corpus: pd.DataFrame,
    category: TaxonomyCategory,
    n: int,
    seed: int,
    text_col: str = "text",
    max_chars: int = 400,
) -> list[str]:
    """Sample `n` review snippets that mention this category's vocabulary.

    Grounding the LLM in real, category-relevant review language (rather
    than instructions alone) keeps drafted questions answerable from
    actual corpus content, guarding against the opposite of the "student
    only writes answerable questions" bias (PLAN.md §9): unanswerable-by
    -corpus questions that would only surface as empty pools much later,
    during CRAGB pooling (T2.6).

    Args:
        corpus: reviews frame with `text_col`.
        category: the taxonomy category to match against
            (`category.keywords` supplies the match vocabulary).
        n: number of review snippets to sample.
        seed: RNG seed for `pandas.DataFrame.sample`, for reproducibility.
        text_col: column containing review text.
        max_chars: truncate each snippet to this length, so the prompt
            (and token cost) stays small.

    Returns:
        Up to `n` truncated review-text snippets.

    Raises:
        ValueError: if no corpus rows mention this category's keywords
            at all — a signal to re-run T2.1's coverage check before
            drafting, not to silently draft ungrounded questions.
    """
    text = corpus[text_col].fillna("").astype(str)
    matched = corpus.loc[_matches_category(text, category.keywords)]
    if matched.empty:
        raise ValueError(
            f"No corpus reviews match category {category.name!r} keywords; "
            f"re-run T2.1's coverage check (cragb.bench.taxonomy) before drafting."
        )
    sample_n = min(n, len(matched))
    sampled = matched[text_col].sample(n=sample_n, random_state=seed)
    return [s[:max_chars] for s in sampled]


def render_prompt(
    template: Template,
    category: TaxonomyCategory,
    example_reviews: list[str],
    n_questions: int,
) -> str:
    """Fill the prompt template for one category's drafting call."""
    reviews_block = (
        "\n".join(f"- {r}" for r in example_reviews)
        if example_reviews
        else "(no example reviews available)"
    )
    return template.substitute(
        category_name=category.name,
        category_description=category.description,
        example_reviews=reviews_block,
        n_questions=n_questions,
    )


_JSON_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)
_CODE_FENCE_LANG_RE = re.compile(r"^json\s*", re.IGNORECASE)


def parse_llm_questions(raw_response: str, category: str) -> list[DraftQuestion]:
    """Parse the model's JSON-array response into `DraftQuestion`s.

    Tolerant of an accidental markdown code fence around the JSON (models
    frequently add one despite being told not to). Anything else that
    fails to parse raises with the raw response text attached, so a bad
    batch is visible and re-runnable rather than silently dropped or
    partially accepted.

    Args:
        raw_response: the LLM's raw completion text.
        category: category name, used for `id` prefixes and error context.

    Returns:
        Parsed `DraftQuestion`s, ids assigned as `"{category}_{i:03d}"`.

    Raises:
        ValueError: if no valid JSON array of well-formed items can be
            extracted from `raw_response`.
    """
    text = raw_response.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = _CODE_FENCE_LANG_RE.sub("", text)

    match = _JSON_ARRAY_RE.search(text)
    candidate_text = match.group(0) if match else text

    try:
        items = json.loads(candidate_text)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Could not parse JSON array from LLM response for category "
            f"{category!r}: {e}\n--- raw response ---\n{raw_response}"
        ) from e

    if not isinstance(items, list):
        raise ValueError(
            f"Expected a JSON array for category {category!r}, got {type(items).__name__}: {items!r}"
        )

    questions: list[DraftQuestion] = []
    for i, item in enumerate(items):
        if not isinstance(item, dict) or "question" not in item:
            raise ValueError(f"Malformed item {i} for category {category!r}: {item!r}")
        questions.append(
            DraftQuestion(
                id=f"{category}_{i:03d}",
                category=category,
                question=str(item["question"]).strip(),
                rationale=str(item.get("rationale", "")).strip(),
            )
        )
    return questions


def generate_candidates_for_category(
    category: TaxonomyCategory,
    example_reviews: list[str],
    template: Template,
    chat_fn: ChatFn,
    n_questions: int,
) -> list[DraftQuestion]:
    """Draft `n_questions` candidate questions for one taxonomy category."""
    prompt = render_prompt(template, category, example_reviews, n_questions)
    raw = chat_fn([{"role": "user", "content": prompt}])
    return parse_llm_questions(raw, category.name)


def dedup_questions(
    questions: list[DraftQuestion], threshold: float = 0.9
) -> tuple[list[DraftQuestion], int]:
    """Drop near-duplicate questions within a list (normalized-text + fuzzy ratio).

    Two checks, in order: an exact match on normalized (lowercased,
    whitespace-collapsed) text is a duplicate outright; otherwise a
    `difflib.SequenceMatcher` ratio at or above `threshold` against every
    already-kept question is also treated as a duplicate. This is O(n^2)
    in the number of questions, which is fine at CRAGB's per-category
    draft scale (single-digit to low tens).

    Args:
        questions: candidate questions, in original order.
        threshold: SequenceMatcher ratio (0-1) at or above which two
            questions are considered near-duplicates.

    Returns:
        `(kept, n_dropped)` — `kept` preserves original relative order.
    """
    kept: list[DraftQuestion] = []
    seen_normalized: set[str] = set()
    n_dropped = 0

    for q in questions:
        normalized = " ".join(q.question.lower().split())
        if normalized in seen_normalized:
            n_dropped += 1
            continue
        is_near_dup = any(
            SequenceMatcher(None, normalized, " ".join(k.question.lower().split())).ratio() >= threshold
            for k in kept
        )
        if is_near_dup:
            n_dropped += 1
            continue
        kept.append(q)
        seen_normalized.add(normalized)

    return kept, n_dropped


def draft_questions_for_taxonomy(
    corpus: pd.DataFrame,
    spec: TaxonomySpec,
    template: Template,
    chat_fn: ChatFn,
    overgenerate_factor: float,
    reviews_per_category: int,
    max_review_chars: int,
    dedup_threshold: float,
    seed: int,
) -> tuple[list[DraftQuestion], dict[str, int]]:
    """Draft, tag, and dedup candidate questions across every taxonomy category.

    Returns:
        `(questions, counts)` — `questions` is the full deduped candidate
        list across all categories; `counts` maps category name to how
        many candidates survived, for the "taxonomy balance close to
        T2.1 targets" check M2.md T2.2 requires.
    """
    all_questions: list[DraftQuestion] = []
    for category in spec.categories:
        n_target = max(1, round(category.target_count * overgenerate_factor))
        reviews = sample_context_reviews(
            corpus, category, n=reviews_per_category, seed=seed, max_chars=max_review_chars
        )
        candidates = generate_candidates_for_category(
            category, reviews, template, chat_fn, n_questions=n_target
        )
        deduped, n_dropped = dedup_questions(candidates, threshold=dedup_threshold)
        if n_dropped:
            logger.info("category %s: dropped %d near-duplicate draft(s)", category.name, n_dropped)
        all_questions.extend(deduped)

    counts = {c.name: sum(1 for q in all_questions if q.category == c.name) for c in spec.categories}
    return all_questions, counts


def write_drafts_jsonl(questions: list[DraftQuestion], out_path: str | Path) -> Path:
    """Write `questions` as newline-delimited JSON, one object per line."""
    resolved = resolve_path(out_path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("w", encoding="utf-8") as f:
        for q in questions:
            f.write(json.dumps(asdict(q), ensure_ascii=False))
            f.write("\n")
    return resolved


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/generate.yaml", help="Path to generate config YAML.")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    # Loads GROQ_API_KEY (and anything else) from a local .env if present;
    # a no-op if the file doesn't exist, so this is safe in CI too.
    load_dotenv()

    cfg = load_config(args.config)
    set_global_seed(cfg["seed"])

    spec = load_taxonomy(cfg["paths"]["taxonomy_config"])
    corpus = pd.read_parquet(resolve_path(cfg["paths"]["corpus_in"]))
    template = load_prompt_template(cfg["paths"]["prompt_template"])

    provider_cfg = cfg["provider"]
    client = GroqClient(
        model=provider_cfg["model"],
        api_base=provider_cfg["api_base"],
        api_key_env=provider_cfg["api_key_env"],
        temperature=provider_cfg["temperature"],
        max_tokens=provider_cfg["max_tokens"],
        timeout_s=provider_cfg["timeout_s"],
        max_retries=provider_cfg["max_retries"],
        cache_dir=cfg["paths"]["cache_dir"],
    )

    sampling_cfg = cfg["sampling"]
    questions, counts = draft_questions_for_taxonomy(
        corpus=corpus,
        spec=spec,
        template=template,
        chat_fn=client.complete,
        overgenerate_factor=sampling_cfg["overgenerate_factor"],
        reviews_per_category=sampling_cfg["reviews_per_category"],
        max_review_chars=sampling_cfg["max_review_chars"],
        dedup_threshold=cfg["dedup"]["near_duplicate_threshold"],
        seed=cfg["seed"],
    )

    out_path = write_drafts_jsonl(questions, cfg["paths"]["draft_out"])
    logger.info("Wrote %d draft questions to %s", len(questions), out_path)
    for name, n in counts.items():
        logger.info("  %s: %d", name, n)

    return 0


if __name__ == "__main__":
    sys.exit(main())
