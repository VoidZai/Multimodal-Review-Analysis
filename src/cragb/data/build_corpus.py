"""End-to-end corpus build: raw files -> frozen `corpus_v1.parquet` (T1.8).

Wires together every prior M1 task in dependency order:

    T1.3 load (stream)  -> T1.4 clean  -> T1.5 join
        -> T1.6 features -> T1.7 stratified sample -> frozen corpus + manifest

The one genuinely new engineering decision at this stage is *how* to reach
a ~200k-row stratified sample out of a multi-million-row source file
without either (a) cleaning and language-detecting the entire raw file
(prohibitively slow — `langdetect` runs at only ~450 rows/sec, so tens of
millions of rows would take many hours) or (b) reading the whole file into
memory. The approach:

1. **Collect a bounded candidate pool per stratum** (`collect_candidate_pool`):
   stream the raw file in chunks, and for each (rating_bucket, has_image)
   cell, buffer up to `overcollect_factor x` its final target row count —
   enough headroom to absorb the small drop rate from cleaning — then stop
   reading a cell once its buffer is full. Streaming stops entirely once
   every cell has reached its cap (or the file is exhausted). This step
   only does cheap parsing/bucketing, so it can afford to scan far more
   raw rows than it keeps, purely to find enough of the *rare* strata
   (empirically, "1-2 star with a photo" is under 1% of raw rows).
2. **Clean and enrich only that bounded pool** (`clean_and_enrich_pool`):
   this is where the expensive language-detection step runs — but over
   ~1.5x the final target size, not over the whole corpus.
3. **Join** the cleaned pool to a lean metadata lookup (`build_metadata_lookup`,
   itself a single streamed pass over the metadata file, keeping only the
   scalar/categorical fields corpus_v1 actually uses — not the heavy
   `images`/`videos`/`description` blobs).
4. **Final stratified draw** down to the exact configured `target_size`,
   with any shortfall reported rather than hidden.
5. Write `corpus_v1.parquet` and a manifest (row count, SHA-256, seed,
   schema, and every intermediate report) for reproducibility.

Usage:
    python -m cragb.data.build_corpus --config configs/data.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import pandas as pd

from cragb.data.clean import dedup, drop_empty_text, filter_language, normalize_whitespace
from cragb.data.features import add_review_features, has_image
from cragb.data.join import join_reviews_metadata
from cragb.data.load import stream_records
from cragb.data.sample import add_rating_bucket, bucket_rating, compute_joint_targets, stratified_sample
from cragb.utils.io import load_config, resolve_path, sha256_file

logger = logging.getLogger(__name__)


def _chunked(records: Iterator[dict], chunk_size: int) -> Iterator[list[dict]]:
    """Group a record stream into fixed-size lists (last one may be smaller)."""
    chunk: list[dict] = []
    for rec in records:
        chunk.append(rec)
        if len(chunk) >= chunk_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def collect_candidate_pool(cfg: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Stream raw reviews, keeping up to a capped number of candidates per
    (rating_bucket, has_image) stratum, stopping once every stratum is full.

    Args:
        cfg: the loaded `configs/data.yaml`.

    Returns:
        `(pool_df, report)` — `pool_df` is the concatenated, uncleaned
        candidate rows (raw fields only, plus `rating_bucket`/`has_image`
        already attached); `report` records how many raw rows were
        scanned and each stratum's cap vs. how many were actually found.
    """
    strata_weights = {
        "rating_bucket": cfg["sampling"]["strata"]["rating"],
        "has_image": cfg["sampling"]["strata"]["has_image"],
    }
    target_size = cfg["sampling"]["target_size"]
    overcollect = cfg["sampling"]["overcollect_factor"]
    chunk_size = cfg["sampling"]["chunk_size"]

    targets = compute_joint_targets(strata_weights, target_size)
    caps = {key: max(1, math.ceil(t * overcollect)) for key, t in targets.items()}
    buffers: dict[tuple, list[dict]] = {key: [] for key in caps}

    scanned = 0
    start = time.monotonic()
    for chunk in _chunked(stream_records(cfg["paths"]["reviews_raw"]), chunk_size):
        scanned += len(chunk)
        for rec in chunk:
            key = (bucket_rating(rec.get("rating")), has_image(rec.get("images")))
            if key not in buffers:
                continue  # invalid/unmapped rating -> not a stratum we sample from
            if len(buffers[key]) < caps[key]:
                buffers[key].append(rec)

        if all(len(buf) >= caps[key] for key, buf in buffers.items()):
            logger.info("collect_candidate_pool: all strata capped after scanning %d rows", scanned)
            break
    else:
        logger.warning(
            "collect_candidate_pool: reached end of file after scanning %d rows "
            "without filling every stratum cap — some strata will be short.",
            scanned,
        )

    elapsed = time.monotonic() - start
    collected = {key: len(buf) for key, buf in buffers.items()}
    logger.info("collect_candidate_pool: scanned %d rows in %.1fs; collected=%s", scanned, elapsed, collected)

    all_rows = [rec for buf in buffers.values() for rec in buf]
    pool = pd.json_normalize(all_rows, max_level=0)
    pool = add_rating_bucket(pool)
    pool["has_image"] = pool["images"].map(has_image)

    report = {
        "rows_scanned": scanned,
        "scan_seconds": round(elapsed, 1),
        "targets": {"|".join(map(str, k)): v for k, v in targets.items()},
        "caps": {"|".join(map(str, k)): v for k, v in caps.items()},
        "collected": {"|".join(map(str, k)): v for k, v in collected.items()},
    }
    return pool, report


def clean_and_enrich_pool(pool: pd.DataFrame, cfg: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply T1.4 cleaning filters and T1.6 feature computation to a candidate pool.

    Args:
        pool: output of `collect_candidate_pool`.
        cfg: the loaded config.

    Returns:
        `(cleaned_df, report)` — `report` records the row count after each
        filter step, so drop rates are auditable rather than only visible
        in a single before/after total.
    """
    n0 = len(pool)
    df = normalize_whitespace(pool, columns=["title", "text"])
    df = drop_empty_text(df, text_col="text", min_chars=cfg["filters"]["min_text_chars"])
    n1 = len(df)
    df = dedup(df, subset=cfg["filters"]["dedup_on"])
    n2 = len(df)
    df = filter_language(df, text_col="text", keep_languages=[cfg["filters"]["language"]])
    n3 = len(df)
    df = add_review_features(df)
    df = add_rating_bucket(df)  # re-attach post-filter (row subset changed, column semantics unchanged)

    report = {
        "rows_before": n0,
        "rows_after_empty_text_filter": n1,
        "rows_after_dedup": n2,
        "rows_after_language_filter": n3,
    }
    logger.info("clean_and_enrich_pool: %s", report)
    return df, report


def build_metadata_lookup(cfg: dict[str, Any]) -> pd.DataFrame:
    """Stream the full metadata file, keeping only `corpus.metadata_columns`.

    A full pass is required (the needed `parent_asin`s aren't known in
    advance, and metadata order doesn't correlate with review order), but
    keeping only lean scalar/categorical fields — not `images`, `videos`,
    `description`, `details`, `bought_together` — keeps the resulting
    DataFrame's memory footprint manageable for a multi-million-row file.

    Args:
        cfg: the loaded config.

    Returns:
        A DataFrame with exactly `corpus.metadata_columns` as columns.
    """
    fields = cfg["corpus"]["metadata_columns"]
    start = time.monotonic()
    rows = [{k: rec.get(k) for k in fields} for rec in stream_records(cfg["paths"]["metadata_raw"])]
    elapsed = time.monotonic() - start
    df = pd.DataFrame(rows, columns=fields)
    # Raw metadata mixes Python `None` and JSON-numeric values for these
    # fields; coerce explicitly so the merged corpus gets a clean float64
    # column (NaN for missing) instead of an `object` dtype.
    for numeric_col in ("average_rating", "price"):
        if numeric_col in df.columns:
            df[numeric_col] = pd.to_numeric(df[numeric_col], errors="coerce")
    logger.info("build_metadata_lookup: streamed %d metadata rows in %.1fs", len(df), elapsed)
    return df


def build_corpus(config_path: str | Path) -> dict[str, Any]:
    """Run the full T1.8 pipeline and write `corpus_v1.parquet` + manifest.

    Args:
        config_path: path to `configs/data.yaml`.

    Returns:
        The manifest dict that was also written to
        `cfg["paths"]["manifest_out"]`.
    """
    cfg = load_config(config_path)
    seed = cfg["seed"]

    pool, collect_report = collect_candidate_pool(cfg)
    cleaned, clean_report = clean_and_enrich_pool(pool, cfg)

    metadata_lookup = build_metadata_lookup(cfg)
    merged, join_report = join_reviews_metadata(cleaned, metadata_lookup)

    strata_weights = {
        "rating_bucket": cfg["sampling"]["strata"]["rating"],
        "has_image": cfg["sampling"]["strata"]["has_image"],
    }
    sampled, sampling_report = stratified_sample(
        merged, strata_weights, target_size=cfg["sampling"]["target_size"], seed=seed
    )

    review_cols = cfg["corpus"]["review_columns"]
    metadata_cols_renamed = [
        ("product_title" if c == "title" else c) for c in cfg["corpus"]["metadata_columns"] if c != "parent_asin"
    ]
    final_cols = review_cols + metadata_cols_renamed
    missing = [c for c in final_cols if c not in sampled.columns]
    if missing:
        raise KeyError(f"corpus_v1 is missing configured output column(s): {missing}")
    corpus = sampled[final_cols].reset_index(drop=True)

    corpus_out = resolve_path(cfg["paths"]["corpus_out"])
    corpus_out.parent.mkdir(parents=True, exist_ok=True)
    corpus.to_parquet(corpus_out, index=False)

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "row_count": len(corpus),
        "sha256": sha256_file(corpus_out),
        "columns": {col: str(dtype) for col, dtype in corpus.dtypes.items()},
        "config_path": str(config_path),
        "reports": {
            "collect_candidate_pool": collect_report,
            "clean_and_enrich_pool": clean_report,
            "join_reviews_metadata": join_report.as_dict(),
            "stratified_sample": sampling_report.as_dict(),
        },
    }
    manifest_out = resolve_path(cfg["paths"]["manifest_out"])
    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    with manifest_out.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=False)
        f.write("\n")

    logger.info(
        "build_corpus: wrote %d rows to %s (sha256=%s), manifest at %s",
        manifest["row_count"],
        corpus_out,
        manifest["sha256"][:12],
        manifest_out,
    )
    return manifest


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data.yaml", help="Path to data config YAML.")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    build_corpus(args.config)
    return 0


if __name__ == "__main__":
    sys.exit(main())
