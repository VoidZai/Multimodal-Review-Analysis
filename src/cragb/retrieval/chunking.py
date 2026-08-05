"""Document chunking schemes for retrieval (T3.1; PLAN.md §3 E2, M3.md T3.1).

E2's question is which unit of text a retriever should index: a whole
review, a fixed-size window of words, or a window of sentences. This
module implements all three as pure functions with an identical output
shape, so the chunking *study* (T3.4) can swap schemes without touching
any retriever or eval code — the same "single swappable shape" principle
`cragb.retrieval.base.Retriever` already applies to retrieval method.

Every chunker returns a `DataFrame` with exactly three columns:
    - `chunk_id`      (str, unique)      — what a `Retriever` indexes on.
    - `parent_doc_id` (str)              — the review this chunk came from;
                                            how a chunk-level hit is scored
                                            against CRAGB's review-level
                                            `relevant_ids` (T3.4/T3.6).
    - `text`          (str, non-empty)   — the chunk's searchable text.

For `whole_review`, `chunk_id == parent_doc_id` by construction (one
chunk per review, so the two ids denote the same thing). For
`fixed_token`/`sentence_window`, a review that splits into more than one
chunk gets ids `f"{parent_doc_id}::0"`, `f"{parent_doc_id}::1"`, ...

Rows whose text is empty/whitespace-only (after `fillna("")`) are
dropped rather than emitted as empty chunks — an empty chunk is
unsearchable and only exists to pad a row count. `corpus_v1` is already
filtered upstream to `min_text_chars: 5` (configs/data.yaml), so in
practice this never fires on the real corpus; it exists so this module
is safe to call on arbitrary/test DataFrames too.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

from cragb.utils.io import load_config

ALLOWED_SCHEMES = ("whole_review", "fixed_token", "sentence_window")

# Sentence boundary = punctuation in .!? followed by whitespace. A
# lookbehind keeps the punctuation attached to the sentence it ends,
# rather than being consumed by the split. Deliberately simple (no
# abbreviation/decimal handling) — review text is short and informal,
# and the sentence-window scheme only needs an approximate boundary to
# be a meaningfully different unit from fixed-token, not a linguistically
# exact sentence splitter.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class ChunkingConfig:
    """Resolved, validated chunking parameters (mirrors `configs/chunking.yaml`)."""

    scheme: str
    fixed_token_size: int = 256
    sentence_window_size: int = 3

    def __post_init__(self) -> None:
        if self.scheme not in ALLOWED_SCHEMES:
            raise ValueError(
                f"Unknown chunking scheme {self.scheme!r}; must be one of {ALLOWED_SCHEMES}"
            )
        if self.fixed_token_size <= 0:
            raise ValueError(f"fixed_token_size must be positive, got {self.fixed_token_size}")
        if self.sentence_window_size <= 0:
            raise ValueError(
                f"sentence_window_size must be positive, got {self.sentence_window_size}"
            )


def load_chunking_config(config_path: str = "configs/chunking.yaml") -> ChunkingConfig:
    """Load and validate the `chunking:` block of a chunking config YAML.

    Args:
        config_path: path to a YAML file with a top-level `chunking:` key
            (see `configs/chunking.yaml`), absolute or relative to the
            repo root.

    Returns:
        A validated `ChunkingConfig`.

    Raises:
        FileNotFoundError: if `config_path` does not exist.
        KeyError: if the config has no `chunking:` block.
        ValueError: if `scheme`/`fixed_token_size`/`sentence_window_size`
            are missing or invalid (see `ChunkingConfig.__post_init__`).
    """
    raw = load_config(config_path)
    chunking = raw["chunking"]
    return ChunkingConfig(
        scheme=chunking["scheme"],
        fixed_token_size=chunking.get("fixed_token_size", 256),
        sentence_window_size=chunking.get("sentence_window_size", 3),
    )


def _resolve_parent_ids(corpus: pd.DataFrame, id_col: str | None) -> list[str]:
    """Parent (review) ids as strings, from `id_col` or the DataFrame index.

    Mirrors the id-resolution contract `BM25Retriever`/`DenseRetriever`
    already use (`cragb.retrieval.base.Retriever.index`), so a chunked
    corpus and a whole-review corpus expose ids the same way.
    """
    parent_ids = (
        corpus[id_col].astype(str).tolist()
        if id_col is not None
        else corpus.index.astype(str).tolist()
    )
    if len(set(parent_ids)) != len(parent_ids):
        raise ValueError(
            "Parent document ids are not unique; pass a unique `id_col` or "
            "reset the corpus index before chunking."
        )
    return parent_ids


def _build_chunk_frame(rows: list[tuple[str, str, str]], scheme: str) -> pd.DataFrame:
    """Assemble + validate the standard `[chunk_id, parent_doc_id, text]` output."""
    if not rows:
        raise ValueError(
            f"No non-empty text remained after chunking (scheme={scheme!r}); "
            "check text_col and the input corpus."
        )
    df = pd.DataFrame(rows, columns=["chunk_id", "parent_doc_id", "text"])
    if df["chunk_id"].duplicated().any():
        # Not expected to be reachable given each chunker constructs ids
        # from unique parent ids + a per-parent-local index, but this is
        # the invariant every downstream consumer (a `Retriever.index()`
        # call) relies on, so it is checked explicitly rather than
        # assumed.
        dupes = df.loc[df["chunk_id"].duplicated(), "chunk_id"].unique().tolist()
        raise ValueError(f"Duplicate chunk ids produced by scheme={scheme!r}: {dupes[:5]}")
    return df


def chunk_whole_review(
    corpus: pd.DataFrame,
    text_col: str = "text",
    id_col: str | None = None,
) -> pd.DataFrame:
    """One chunk per review — the corpus's natural, un-split unit.

    Args:
        corpus: one row per review.
        text_col: column containing the review text.
        id_col: column to use as each review's id; the DataFrame index
            is used if `None`.

    Returns:
        `[chunk_id, parent_doc_id, text]`, one row per non-empty review.
        `chunk_id == parent_doc_id` for every row.

    Raises:
        ValueError: if `corpus` is empty, ids are not unique, or every
            row's text is empty.
    """
    if corpus.empty:
        raise ValueError("Cannot chunk an empty corpus.")

    parent_ids = _resolve_parent_ids(corpus, id_col)
    texts = corpus[text_col].fillna("").astype(str)

    rows = [
        (parent_id, parent_id, text.strip())
        for parent_id, text in zip(parent_ids, texts)
        if text.strip()
    ]
    return _build_chunk_frame(rows, scheme="whole_review")


def chunk_fixed_token(
    corpus: pd.DataFrame,
    text_col: str = "text",
    id_col: str | None = None,
    token_size: int = 256,
) -> pd.DataFrame:
    """Split each review into fixed-size, non-overlapping windows of words.

    "Token" here means whitespace-delimited word, not a model-specific
    subword token (same approximation `cragb.data.features.compute_token_length`
    documents using for the same reason: a cheap, dependency-light,
    reproducible unit). Plain whitespace split-and-rejoin (rather than
    `compute_token_length`'s punctuation-splitting regex) is used
    deliberately here, because chunk text must round-trip back into
    readable text via `" ".join(...)`; splitting punctuation into its
    own tokens would reintroduce as stray spaces before commas/periods.

    Args:
        corpus: one row per review.
        text_col: column containing the review text.
        id_col: column to use as each review's id; the DataFrame index
            is used if `None`.
        token_size: max words per chunk (must be positive).

    Returns:
        `[chunk_id, parent_doc_id, text]`. Reviews with `token_size` or
        fewer words produce exactly one chunk; longer reviews produce
        `ceil(n_words / token_size)` chunks. `chunk_id` is
        `f"{parent_doc_id}::{i}"`, `i` 0-indexed per parent review.

    Raises:
        ValueError: if `corpus` is empty, `token_size` is not positive,
            ids are not unique, or every row's text is empty.
    """
    if corpus.empty:
        raise ValueError("Cannot chunk an empty corpus.")
    if token_size <= 0:
        raise ValueError(f"token_size must be positive, got {token_size}")

    parent_ids = _resolve_parent_ids(corpus, id_col)
    texts = corpus[text_col].fillna("").astype(str)

    rows: list[tuple[str, str, str]] = []
    for parent_id, text in zip(parent_ids, texts):
        words = text.split()
        if not words:
            continue
        for i, start in enumerate(range(0, len(words), token_size)):
            chunk_words = words[start : start + token_size]
            rows.append((f"{parent_id}::{i}", parent_id, " ".join(chunk_words)))

    return _build_chunk_frame(rows, scheme="fixed_token")


def _split_sentences(text: str) -> list[str]:
    """Approximate sentence split on `.`/`!`/`?` + whitespace; `[]` for blank input."""
    stripped = text.strip()
    if not stripped:
        return []
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(stripped) if s.strip()]


def chunk_sentence_window(
    corpus: pd.DataFrame,
    text_col: str = "text",
    id_col: str | None = None,
    window_size: int = 3,
) -> pd.DataFrame:
    """Split each review into non-overlapping windows of `window_size` sentences.

    Windows are non-overlapping (stride == `window_size`), not a sliding
    window: a sliding window would duplicate sentences across multiple
    chunks, inflating the index and double-counting the same evidence at
    retrieval time — a confound this project has no reason to introduce
    when comparing against the other two (non-duplicating) schemes.

    Args:
        corpus: one row per review.
        text_col: column containing the review text.
        id_col: column to use as each review's id; the DataFrame index
            is used if `None`.
        window_size: max sentences per chunk (must be positive).

    Returns:
        `[chunk_id, parent_doc_id, text]`. Reviews with `window_size` or
        fewer sentences produce exactly one chunk. `chunk_id` is
        `f"{parent_doc_id}::{i}"`, `i` 0-indexed per parent review.

    Raises:
        ValueError: if `corpus` is empty, `window_size` is not positive,
            ids are not unique, or every row's text is empty.
    """
    if corpus.empty:
        raise ValueError("Cannot chunk an empty corpus.")
    if window_size <= 0:
        raise ValueError(f"window_size must be positive, got {window_size}")

    parent_ids = _resolve_parent_ids(corpus, id_col)
    texts = corpus[text_col].fillna("").astype(str)

    rows: list[tuple[str, str, str]] = []
    for parent_id, text in zip(parent_ids, texts):
        sentences = _split_sentences(text)
        if not sentences:
            continue
        for i, start in enumerate(range(0, len(sentences), window_size)):
            window = sentences[start : start + window_size]
            rows.append((f"{parent_id}::{i}", parent_id, " ".join(window)))

    return _build_chunk_frame(rows, scheme="sentence_window")


def chunk_corpus(
    corpus: pd.DataFrame,
    config: ChunkingConfig,
    text_col: str = "text",
    id_col: str | None = None,
) -> pd.DataFrame:
    """Dispatch to the chunker named by `config.scheme`.

    The single entry point the chunking study (T3.4) and the retrieval
    eval harness (T3.5+) call — neither needs to know which of the three
    scheme functions exists, only that `config.scheme` selects one.

    Args:
        corpus: one row per review.
        config: a validated `ChunkingConfig` (see `load_chunking_config`).
        text_col: column containing the review text.
        id_col: column to use as each review's id; the DataFrame index
            is used if `None`.

    Returns:
        `[chunk_id, parent_doc_id, text]`, produced by the scheme named
        in `config.scheme`.

    Raises:
        ValueError: propagated from the selected scheme function (empty
            corpus, non-positive size parameter, non-unique ids, or no
            non-empty text).
    """
    if config.scheme == "whole_review":
        return chunk_whole_review(corpus, text_col=text_col, id_col=id_col)
    if config.scheme == "fixed_token":
        return chunk_fixed_token(
            corpus, text_col=text_col, id_col=id_col, token_size=config.fixed_token_size
        )
    return chunk_sentence_window(
        corpus, text_col=text_col, id_col=id_col, window_size=config.sentence_window_size
    )
