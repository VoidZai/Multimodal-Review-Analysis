# Datasheet for `finetune_v1` (training / val / probe splits)

Following the structure of Gebru et al., *Datasheets for Datasets* (2021, CACM), and the
project's own precedent in [`data/processed/corpus_v1_datasheet.md`](../processed/corpus_v1_datasheet.md).
This datasheet describes the synthetic fine-tuning dataset built by
[`src/cragb/finetune/{sample_contexts,generate_pairs,abstentions,filter_pairs,split}.py`](../../src/cragb/finetune/)
(T7.2–T7.6; PLAN.md §3 E8, §10).

- **Built:** 2026-08-24, seed `42`, config [`configs/finetune.yaml`](../../configs/finetune.yaml)
- **Machine-readable manifests:** [`contexts_v1_manifest.json`](contexts_v1_manifest.json),
  [`split_manifest_v1.json`](split_manifest_v1.json); cost/funnel tables
  [`ft_generation_cost_v1.csv`](../../results/tables/ft_generation_cost_v1.csv),
  [`ft_filter_v1.csv`](../../results/tables/ft_filter_v1.csv). Every number below is pulled
  directly from these files, or from
  [`ft_datasheet_stats_v1.json`](../../results/tables/ft_datasheet_stats_v1.json) — the flat
  re-derived summary [`src/cragb/finetune/datasheet_stats.py`](../../src/cragb/finetune/datasheet_stats.py)
  generates from them — never retyped by hand.

**Read this first — current scale is a pilot, not the full sweep.** `configs/finetune.yaml`'s
`sampling.n_contexts` targets **500** context groups (M7.md's ~1,200–1,600-example plan). This
build has **377 groups sampled** (available, disjoint from CRAGB evidence) but only **10
attempted** through T7.3's teacher-generation stage — `data/finetune/raw_pairs_v1_progress.jsonl`
has exactly 10 rows, and `train.jsonl` / `val.jsonl` are empty (every one of the 32 examples
that survived filtering and leakage-screening landed in the probe set, since the probe's 40+40
target absorbs the entire pool at this size). This is a **deliberate, resumable pilot run**
(`generate_all`'s incremental append-and-skip design, T7.3) that exercises T7.2→T7.6's full
chain end-to-end and produces real numbers for T7.7–T7.9's independent chain, not a partial
failure — `configs/finetune_train.yaml`'s own header says as much ("T7.3's full 377-context
generation sweep hasn't run yet"). Scaling `raw_pairs_v1.jsonl` from 10 to 500 contexts before
M8 is **resuming the same command**, not new engineering; see
[`reports/finetune_plan_v1.md`](../../reports/finetune_plan_v1.md) §3 for the scale-up note and
PLAN.md §14.7 for the finding logged against it.

---

## Motivation

**For what purpose was the dataset created?**
`finetune_v1` is the synthetic supervised-fine-tuning dataset for E8/RQ3 — training a small
local model (T7.7) to match the RQ1 "large" teacher's grounding and citation discipline via
QLoRA (PLAN.md §10). It exists to support M8's fine-tuning run and the pre-declared go/no-go
evaluation in `reports/finetune_plan_v1.md`. It is *not* CRAGB — CRAGB (`benchmark/cragb_v1.jsonl`)
remains the frozen evaluation set this training data is built to be disjoint from.

**Who created it, and on behalf of whom?**
Mohammed Zaid Shaikh, coursework for ECS-8060 (AI Engineering). Every question/answer pair is
synthetic: distilled from `openai/gpt-oss-120b` (Groq) over real Amazon review text from
`corpus_v1`, not authored by a human and not drawn from any external QA dataset.

**Who funded it?**
No external funding. Generation cost real money only in the trivial sense of Groq API usage —
see Collection process below ($0.007 for this pilot's 10 calls).

---

## Composition

**What do the instances represent?**
Each row (`TrainingExample`, [`src/cragb/finetune/schema.py`](../../src/cragb/finetune/schema.py))
is one (context, question, grounded-and-cited answer) triple, plus a boolean `is_abstention` flag
and a `provenance` dict recording how it was made. A **positive** example's context is 5 reviews
of one product (`parent_asin`), rendered through the identical `render_prompt` /
`build_context` path used at inference time (T7.1's parity guarantee), with a teacher-generated
answer citing specific reviews by id. An **abstention** example pairs a context with a question
the context cannot answer, with the canonical `ABSTENTION_TEXT` as the answer and no citations.

**How many instances, and is this the full target or a sample?**
A **pilot sample**, not the target volume — see the callout above. This build's numbers:

| Stage | Count | Source |
|---|---|---|
| Context groups sampled (disjoint from CRAGB evidence) | 377 of 500 targeted | `contexts_v1_manifest.json` |
| Context groups attempted through teacher generation | 10 | `raw_pairs_v1_progress.jsonl` |
| Raw positive pairs generated | 29 | `raw_pairs_v1.jsonl` |
| Constructed abstention examples | 10 | `abstentions_v1.jsonl` |
| Accepted after T7.5's filter (citation validity + faithfulness ≥ 4) | 39 (39 raw → 0 dropped) | `ft_filter_v1.csv` |
| Dropped as an exact or near-duplicate of a CRAGB question (T7.6) | 7 | `split_manifest_v1.json` |
| Final pool after leakage screening | 32 | `split_manifest_v1.json` |
| → train | 0 | `train.jsonl` |
| → val | 0 | `val.jsonl` |
| → probe (held out entirely) | 32 (24 answerable / 8 abstention) | `probe.jsonl` |

**Why the filter's accept rate is 100% at this scale, and why that's not yet meaningful.**
T7.5's validation checklist explicitly flags a >95% accept rate as a signal to check whether
the threshold is doing anything — at n=39 all 39 raw pairs cleared both the deterministic
citation check and the faithfulness≥4 judge bar. That is plausible at this scale (a capable
120B teacher, closely following a prompt that restates the citation rules) but the sample is
too small to conclude the filter is inert; re-check this figure once the full 500-context sweep
runs.

**Category composition (accepted pairs, this pilot).** Skewed toward `fit_sizing` because T7.3's
resumable generator processes `contexts_v1.jsonl` in file order, and that file (like the target
quotas) lists `fit_sizing` first:

| Category | Accepted (n=39) |
|---|---|
| fit_sizing | 36 |
| fabric_quality | 2 |
| value | 1 |
| colour_appearance / durability / defects / occasion | 0 (not yet reached) |

The **target** category mix (`contexts_v1_manifest.json`'s quotas, which the full sweep will
approximate): fit_sizing 100, fabric_quality 83, colour_appearance 67, occasion 52, value 49,
durability 15, defects 11 — durability and defects fell short of their nominal 67-each target
(shortfall 52 and 56 respectively) because too few corpus reviews carry that content after
excluding CRAGB's evidence; the sampler correctly reported this rather than silently
backfilling from a different category (`sample_contexts.py`'s quota-report contract).

**Abstention construction method mix (n=10, this pilot):** `transplant` 4, `categorical_absence`
3, `evidence_stripped` 3 — roughly even across T7.4's three methods, as designed; too small a
sample to read a firm proportion into yet.

**Is any information missing from individual instances?**
No — every field in `TrainingExample` is required by the schema (`to_dict`/`from_dict`
round-trips losslessly, T7.1) and `__post_init__` raises on an unknown `category` or a
self-contradictory abstention record (§14.3's containment-check lesson), so a malformed record
cannot exist in the committed files.

**Are there recommended data splits?**
Yes — `train` / `val` / `probe`, grouped by `parent_asin` so no product straddles a split
(`parent_asin_disjointness`: 0 overlap in every pairwise check, `split_manifest_v1.json`). At
this pilot's scale the entire surviving pool (32 examples) was absorbed by the probe set's
40-answerable/40-abstention target, leaving train and val empty — **this is not a bug**;
`split_examples` fills probe first because M7.md fact #5 makes the probe the only set with
enough abstention examples to support a go/no-go threshold, and correctly reports empty
train/val rather than under-filling probe to leave some behind. Both will populate once the
full sweep supplies enough volume.

**Is the dataset disjoint from CRAGB's evaluation set?**
Yes, at three layers (T7.2, T7.6):
1. **Document-level exclusion at sampling time** — every doc id in any CRAGB `relevant_ids`,
   `cited_doc_ids`, or `benchmark/pools_v1.jsonl` entry (1,097 docs across 1,070 `parent_asin`s)
   is excluded before a single context is sampled (`contexts_v1_manifest.json`).
2. **Exact-hash question screening** — `cragb.bench.assemble.check_no_leakage` (T2.10, reused
   not reimplemented) against `benchmark/cragb_v1_leakage_manifest.json`'s 60 hashes.
   **0 exact leaks found** in this pool.
3. **Near-duplicate screening** — difflib ratio (threshold 0.85) plus an embedding-cosine
   backstop (`BAAI/bge-small-en-v1.5`, threshold 0.80, calibrated against real CRAGB-question
   score gaps — see `configs/finetune.yaml`'s `split.embedding_similarity_threshold` comment)
   against all 60 CRAGB questions. **7 of 39 candidates (18%) were dropped** as near-duplicates,
   all caught by the embedding layer, all against `fit_sizing_000`/`fit_sizing_001` — expected,
   since `fit_sizing` is this pilot's dominant category and CRAGB's own `fit_sizing` questions
   ("Do these run true to size?") are a natural paraphrase target for a teacher drafting
   fit-sizing questions from similar review text. Every dropped example is listed by id in
   `split_manifest_v1.json`'s `near_duplicate_matches`.

**Are there errors, noise, or redundancies?**
Not observed at this scale, beyond the category skew already noted. The embedding-backstop
finding above (near-duplicates cluster in exactly one category, the one CRAGB itself has
questions in) is worth re-checking once other categories are populated — it may be a
category-specific effect (small, templated CRAGB question phrasing in `fit_sizing`) rather than
a general 18% duplicate rate across the whole dataset.

---

## Collection process

**How was the data acquired?**
1. **T7.2 sampling** — `sample_contexts.py` groups `corpus_v1.parquet` reviews by `parent_asin`
   (k=5 per group), excludes CRAGB-evidence documents, and stratifies to the taxonomy category
   mix.
2. **T7.3 teacher generation** — each context group is sent to `openai/gpt-oss-120b` via Groq
   (`src/cragb/generate/prompts/finetune_gen_v1.md`, temperature 0.6), asking for 3 diverse
   (question, grounded-and-cited answer) pairs per context, returned as a JSON array. Cost this
   pilot: **10 calls, $0.007, 125.3s wall-clock, 11,907 prompt + 9,112 completion tokens**
   (`ft_generation_cost_v1.csv`) — extrapolating to 500 contexts is ~$0.36 and ~1 hour, matching
   M7.md fact #6's estimate.
3. **T7.4 constructed abstentions** — three deliberate methods (`transplant`,
   `categorical_absence`, `evidence_stripped`; see Composition above), *not* hand-authored
   negatives, because PLAN.md §14.2 already showed hand-authored negatives mostly fail (9 of 11
   turned out answerable).
4. **T7.5 filter** — deterministic citation-validity + format check (free, always runs) then a
   judged faithfulness≥4 threshold (`qwen/qwen3.6-27b`, the same distinct-family judge T4b.4
   uses). Cost this pilot: 29 judge calls, $0.025 (`ft_filter_v1.csv`).
5. **T7.6 leakage + split** — see disjointness section above.

**Who was involved?**
Automated pipeline only (`openai/gpt-oss-120b` as teacher, `qwen/qwen3.6-27b` as filter judge);
no human authored or hand-curated any question or answer in this dataset — a deliberate
difference from CRAGB itself, which used human curation on teacher-drafted candidates (T2.3).

**Over what time frame?**
This pilot: 2026-08-24. Full-scale generation is a resume of the same command, not a new run.

---

## Preprocessing / cleaning / filtering

**What filtering was applied, and why only faithfulness?**
`results/tables/judge_validation_v1.csv` (T4b.6) measured four judge criteria against human
labels: faithfulness κ=0.597, conciseness κ=0.321, completeness κ=0.243, correctness
κ=−0.151 (worse than chance agreement). Only faithfulness clears the project's 0.4
usability bar. T7.5's filter therefore scores **faithfulness only** — filtering training data
on correctness, completeness, or conciseness would dress up instruments known to be
unreliable as quality control. Citation validity is checked **deterministically**
(`cragb.eval.citation_validity`), not by an LLM judge at all, for the same reason: it doesn't
need to be — it's checkable by exact string matching.

**Is the raw (pre-filter) data available?**
Yes, `data/finetune/raw_pairs_v1.jsonl` (git-ignored; regenerable via
`python -m cragb.finetune.generate_pairs`, resumable, cache-backed).

**Is the pipeline software available?**
Yes: `src/cragb/finetune/{sample_contexts,generate_pairs,abstentions,filter_pairs,split}.py`,
each independently unit-tested
(`tests/test_sample_contexts.py`, `tests/test_generate_pairs.py`, `tests/test_abstentions.py`,
`tests/test_filter_pairs.py`, `tests/test_ft_split.py`).

---

## Uses

**What has this dataset been used for, or is it intended for?**
Intended: QLoRA supervised fine-tuning of a local base model (M8), with `train`/`val` for
training/early-stopping and `probe` reserved entirely for the go/no-go behaviour evaluation in
`reports/finetune_plan_v1.md`. Not validated, and not intended, for any use outside this
project's M8 fine-tuning run.

**Are there risks or limitations future users should know about?**
- **Pilot scale** — see the callout at the top of this document. `train`/`val` are currently
  empty; do not attempt M8 training against this exact committed state without first resuming
  T7.3's generation to the configured 500-context target.
- **Category imbalance in the current pool** — see Composition; `durability` and `defects` are
  under-represented relative to their target quota even at the *sampled-context* stage (before
  any generation), because too little disjoint corpus evidence exists for them once CRAGB's
  own evidence is excluded.
- **Single teacher, single filter judge** — every positive example's grounding discipline is
  only as good as `openai/gpt-oss-120b`'s, and every accept/reject decision only as reliable as
  `qwen/qwen3.6-27b`'s faithfulness scoring (κ=0.597 — good, not perfect). A systematic teacher
  error would propagate uncorrected into every accepted training example.
- **Near-duplicate drop rate concentrated in one category** — see Collection process; a
  category-specific rather than dataset-wide effect, worth re-verifying once the full sweep
  populates the other six categories.

**Are there tasks this dataset should not be used for?**
Any claim about CRAGB performance should never be validated *against this dataset's own probe
split as if it were CRAGB* — the probe set exists precisely because CRAGB's own abstention
n=2 cannot support a go/no-go rule (M7.md fact #5), not as a replacement benchmark. CRAGB
remains the sole held-out evaluation set for RQ0–RQ2; this dataset (`train`/`val`/`probe`
alike) is fine-tuning material and behaviour-probe material only.

---

## Distribution

Not distributed externally — internal project artifact. `train.jsonl`/`val.jsonl`/`probe.jsonl`
and the two manifests are the versioned contract; `raw_pairs_v1.jsonl` (pre-filter) is
git-ignored and regenerable.

---

## Maintenance

**Who maintains this dataset?**
Mohammed Zaid Shaikh, for the duration of the ECS-8060 project.

**Will it be updated?**
Yes, deliberately — unlike `corpus_v1` (frozen), `finetune_v1` is expected to grow from this
pilot to the full ~500-context / ~1,200–1,600-example target before M8 runs. Re-running
`sample_contexts.py` → `generate_pairs.py` → `abstentions.py` → `filter_pairs.py` → `split.py`
in sequence with the same seed (42) and config is the intended path to that scale-up; the
manifests' `config_hash`/seed fields make any future re-run's provenance checkable against
this one. A future structural change (new abstention method, different filter threshold) would
be released as `finetune_v2` with its own datasheet, not a silent overwrite.
