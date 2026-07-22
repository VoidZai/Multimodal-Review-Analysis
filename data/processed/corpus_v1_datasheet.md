# Datasheet for `corpus_v1`

Following the structure of Gebru et al., *Datasheets for Datasets* (2021,
CACM). This datasheet describes the dataset artifact at
`data/processed/corpus_v1.parquet`, built by [`src/cragb/data/build_corpus.py`](../../src/cragb/data/build_corpus.py)
(T1.8) and profiled in [`notebooks/01_eda.ipynb`](../../notebooks/01_eda.ipynb) (T1.9–T1.11).

- **Row count:** 200,000
- **SHA-256:** `180c2f102922854374eebfa1e5df0977520435bafa6c89f29f67d3bd10fe91f4`
- **Built:** 2026-07-21, seed `42`, config [`configs/data.yaml`](../../configs/data.yaml)
- **Machine-readable manifest:** [`corpus_v1_manifest.json`](corpus_v1_manifest.json) (this file is the human-readable companion — numbers here are pulled directly from that manifest and the executed EDA notebook, not re-derived by hand)

---

## Motivation

**For what purpose was the dataset created?**
`corpus_v1` is the working review corpus for CRAGB — a retrieval-augmented
QA system over Amazon Clothing/Shoes/Jewelry reviews (see [`PLAN.md`](../../PLAN.md)
§1). It exists to support: (1) building a retrieval index (BM25/dense) for
review evidence, (2) pooling relevance labels for the CRAGB benchmark, and
(3) grounded-QA generation and evaluation. It is a curated, stratified
*subset* of the source dataset, not a general-purpose release.

**Who created it, and on behalf of whom?**
Mohammed Zaid Shaikh, as coursework for ECS-8060 (AI Engineering). Built
entirely from a public academic dataset (see Composition/provenance
below) — no first-party data collection was performed.

**Who funded it?**
No external funding; a university module project.

---

## Composition

**What do the instances represent?**
Each row is one Amazon product review (English, Clothing/Shoes/Jewelry
category) joined with lean product-metadata fields. One review = one row;
one product (`parent_asin`) can appear in multiple rows (139,448 unique
products across 200,000 reviews; median 1 review/product, max 843 — see
Collection process for why this long tail exists).

**How many instances, and is this all the source data or a sample?**
200,000 rows — a **stratified sample**, not the full source file. The raw
`Clothing_Shoes_and_Jewelry` review file is 7.1 GB (source row count in
the tens of millions per the dataset card); `corpus_v1` is deliberately a
"low hundreds of thousands" working subset per PLAN.md §1.3, sized for
local indexing/fine-tuning under an 8 GB VRAM budget rather than for
exhaustive coverage.

**What data does each instance consist of?**
Raw text (review title/body), structured review metadata (rating,
verified-purchase flag, helpful-vote count, timestamp, user/product IDs),
derived features (token length, parsed image URLs, image count/flag,
rating bucket), and joined product metadata (product title, category
breadcrumb, store, average rating, price). Full column list and dtypes
are in the manifest's `columns` field; see also `configs/data.yaml`'s
`corpus.review_columns` / `corpus.metadata_columns`.

**Is any information missing from individual instances?**
Yes, and by design where the source data is missing it — `price` is null
for 37.4% of rows (Amazon product listings frequently omit price in the
metadata dump; this is a genuine gap in the upstream data, not a defect
introduced by this pipeline), `main_category` is null for ~2.9%, `store`
for ~0.3%. No nulls in `text`, `parent_asin`, or `product_title` — every
row has a matched product (100% join match rate; see Collection
process).

**Are relationships between instances made explicit?**
Yes — `parent_asin` links reviews to the same product; `user_id` links
reviews from the same reviewer. No review-to-review relationship (e.g.
replies) exists in the source data.

**Are there recommended data splits?**
None yet. `corpus_v1` is the retrieval/generation corpus; CRAGB benchmark
questions (a separate, much smaller artifact built later in E1) will be
held out from any fine-tuning data by construction (PLAN.md §3 E8) —
that split does not exist at this milestone.

**Are there errors, noise, or redundancies?**
- **Near-duplicate reviews**: exact duplicates on `(user_id, parent_asin,
  text)` were removed (326 dropped from the 298,015-row candidate pool
  during cleaning); near-duplicates with minor text variation are not
  detected and may remain.
- **Non-English residue**: language filtering (`langdetect`) is
  probabilistic, not perfect — a small fraction of borderline/short
  reviews may be misclassified in either direction. 9,406 of 297,689
  candidate rows (3.2%) were dropped as non-English.
- **`categories` breadcrumb noise**: the department-level field
  (`categories[1]`) has 371 unique values including inconsistent/noisy
  entries (e.g. a store name, `"Westlake"`, appears as if it were a
  department) — see EDA Figure 1e. The top 15 values still cover 99% of
  rows, so this noise is confined to a small tail.

**Is the dataset self-contained, or does it rely on external resources?**
Self-contained for text/tabular fields. `image_urls` contains external
`https://m.media-amazon.com/...` URLs (Amazon's CDN) rather than the
images themselves — no image bytes are stored in this artifact.

---

## Collection process

**How was the data acquired?**
Downloaded (T1.2) from the McAuley Lab's public Amazon Reviews 2023
mirror: `review_categories/Clothing_Shoes_and_Jewelry.jsonl.gz` (7.1 GB,
SHA-256 `ae07b625...`) and `meta_categories/meta_Clothing_Shoes_and_Jewelry.jsonl.gz`
(4.05 GB, SHA-256 `df6ac054...`), both recorded in
[`data/raw/checksums.json`](../raw/checksums.json). Citation: Hou, Y. et
al. (2024), *Bridging Language and Items for Retrieval and Recommendation*,
arXiv:2403.03952.

**Sampling strategy (why `corpus_v1` looks the way it does):**
A plain random sample of the source would be ~82% five-star and ~9%
image-bearing (measured directly from the raw stream during T1.7
design), which would starve the corpus of defect/fit/negative evidence
and visual-evidence questions — PLAN.md's named "Risk C". Instead,
`cragb.data.sample.stratified_sample` (T1.7) draws a **stratified**
sample forcing:
- rating: 20% 1–2★ / 15% 3★ / 65% 4–5★ (vs. a natural ~82% 4–5★)
- image presence: 15% has-image / 85% no-image

Both targets were hit **exactly** (0 shortfall in any stratum — see
manifest `reports.stratified_sample`), with comfortable headroom
(available candidates were 1.3–1.5× the target in every stratum). This
means `corpus_v1`'s rating/image distribution is an intentionally
engineered property of this dataset, not a natural artifact — analyses
that need the *natural* distribution (e.g. estimating true population
prevalence of complaints) must not use `corpus_v1` directly.

**Pipeline stages and yield at each step** (from the manifest, this exact
build):

| Stage | Rows in → out | Notes |
|---|---|---|
| Raw stream scan | 1,560,000 raw rows scanned | To fill per-stratum candidate buffers at 1.5× target (see `cragb.data.build_corpus.collect_candidate_pool`); stopped early once every stratum cap was reached — the full multi-million-row file was never fully scanned or cleaned |
| Empty-text filter | 300,000 → 298,015 | Dropped reviews with <5 stripped characters |
| Deduplication | 298,015 → 297,689 | Exact duplicates on `(user_id, parent_asin, text)` |
| Language filter | 297,689 → 288,283 | English-only (`langdetect`, seeded for reproducibility) |
| Metadata join | 288,283 reviews × 7,218,481 metadata rows → 100% matched | Left join on `parent_asin`; 0 metadata duplicate-key rows found |
| Stratified draw | 288,283 candidates → 200,000 final | Exact target hit in every stratum, 0 shortfall |

**Who was involved in the collection process?**
Automated pipeline only (T1.2–T1.8); no manual labelling or curation at
this milestone (that begins with CRAGB benchmark construction, E1).

**Over what time frame was the data collected?**
Source data downloaded 2026-07-21 (see `checksums.json` timestamps); the
underlying reviews/metadata span whatever date range the Amazon Reviews
2023 dataset itself covers (not independently re-verified here — see the
dataset's own documentation for collection dates).

---

## Preprocessing / cleaning / labeling

**What preprocessing was done?**
1. Whitespace normalization (collapse runs of whitespace, strip ends) —
   `cragb.data.clean.normalize_whitespace` (T1.4).
2. Empty/near-empty text removal (<5 chars) — `drop_empty_text`.
3. Exact-duplicate removal on `(user_id, parent_asin, text)` — `dedup`.
4. English-language filtering — `filter_language` (`langdetect`, seeded).
5. Feature derivation: token length (regex word/punctuation count, *not*
   an LLM tokenizer — see `cragb.data.features` docstring for why that
   distinction is deliberate at this stage), image URL parsing, image
   count/presence flag — `cragb.data.features.add_review_features` (T1.6).
6. Left-join to a lean metadata projection (product title, category
   breadcrumb, store, average rating, price only — heavy nested fields
   like `videos`/`description`/`details` were dropped as noise for this
   milestone) — `cragb.data.join.join_reviews_metadata` (T1.5).
7. Stratified sampling to the final 200,000 rows — `cragb.data.sample.stratified_sample` (T1.7).

**Is the raw data also available?**
Yes — `data/raw/reviews_clothing.jsonl.gz` and `data/raw/meta_clothing.jsonl.gz`
(git-ignored due to size; re-downloadable via `python -m cragb.data.download`,
integrity-checked against `data/raw/checksums.json`).

**Is the preprocessing software available?**
Yes, this entire repository: `src/cragb/data/{load,clean,join,features,sample,build_corpus}.py`,
each independently unit-tested (`tests/test_clean.py`, `tests/test_join.py`,
`tests/test_sample.py`).

---

## Uses

**What tasks has the dataset been used for, or could it be used for?**
Intended: building sparse/dense retrieval indexes (E3), CRAGB benchmark
pooling (E1), grounded-QA generation and evaluation (E4/E5), and
synthetic fine-tuning data preparation (E8) — all within the CRAGB
project. Not intended, and not validated, for any use outside that
project scope.

**Are there risks or limitations future users should know about?**
- **Not a natural sample** — see Collection process. Do not use
  `corpus_v1`'s rating/image proportions to estimate true population
  statistics about the source category.
- **English-only** — non-English reviews were filtered out entirely; the
  corpus says nothing about non-English-speaking shoppers.
- **No image bytes** — `image_urls` are external CDN links; they may
  break or change over time, and no snapshot/caching of the actual images
  has been done at this milestone.
- **Attribute-vocabulary coverage is uneven** (T1.11 finding): size
  31.1%, fit 47.1%, colour 16.6%, fabric 33.5% of reviews contain at
  least one relevant keyword. All four clear the >1% "is there any
  signal at all" bar comfortably, but colour-related questions will have
  the thinnest textual evidence of the four — worth extra attention when
  drafting the CRAGB taxonomy (E1) so colour questions aren't
  over-represented relative to available evidence.
- **Long-tail product concentration**: while median reviews-per-product
  is 1, the max is 843 — a small number of popular products contribute
  disproportionately many reviews, which could bias retrieval evaluation
  toward those products unless accounted for.

**Are there tasks for which the dataset should not be used?**
Any claim of "visual grounding" that implies a text-only generator
reasons over review photos — per PLAN.md's own flagged risk (§1.4-D),
this corpus supports *surfacing* a cited photo as evidence, not
multimodal reasoning over pixels.

---

## Distribution

Not distributed externally. Internal project artifact only,
version-controlled by manifest hash rather than by committing the
(60 MB) parquet file itself to git.

---

## Maintenance

**Who maintains the dataset?**
Mohammed Zaid Shaikh, for the duration of the ECS-8060 project.

**Will it be updated?**
`corpus_v1` is frozen (the "v1" in its name, and its SHA-256 in the
manifest, are the contract). A future re-sample or expanded corpus would
be released as `corpus_v2` with its own manifest and datasheet, not as a
silent overwrite of this file.

---

## Summary statistics (for quick reference — see `notebooks/01_eda.ipynb` for full figures)

| Metric | Value |
|---|---|
| Rows | 200,000 |
| Unique products (`parent_asin`) | 139,448 |
| Rating distribution | 1★ 21,407 · 2★ 18,593 · 3★ 30,000 · 4★ 29,830 · 5★ 100,170 (mean 3.844) |
| Image coverage | 15.0% (30,000 rows) |
| Review length (tokens) | median 41, mean 64.6, p95 195, max 1,914 |
| Reviews per product | median 1, max 843 |
| Unique departments (`categories[1]`) | 371 (top 5 = 86.8% of rows) |
| Attribute-vocabulary coverage | size 31.1% · fit 47.1% · colour 16.6% · fabric 33.5% |
| `price` null rate | 37.4% (genuine source-data gap) |
| Join match rate | 100% (0 unmatched reviews) |
