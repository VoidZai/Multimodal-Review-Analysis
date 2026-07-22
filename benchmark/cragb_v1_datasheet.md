# CRAGB v1 Datasheet

Gebru-style datasheet for `benchmark/cragb_v1.jsonl` (T2.10; PLAN.md §3 E1, §5). All numbers below are computed directly from the assembled entries, not hand-maintained.

## Motivation
CRAGB evaluates whether strictly-grounded review-RAG can answer open-ended shopper questions about Amazon Clothing/Shoes/Jewelry products without hallucinating (RQ0), compare retrieval methods (RQ2), and surface photo evidence honestly (RQ4). Built per PLAN.md §3 E1 as a pooled, human-labeled benchmark — not scraped or auto-generated without human review at any stage.

## Composition
**60 questions** across 7 taxonomy categories (T2.1), of which **11 are deliberately unanswerable negatives**.

| Category | Total | Negative |
|---|---:|---:|
| `colour_appearance` | 8 | 1 |
| `defects` | 8 | 2 |
| `durability` | 8 | 2 |
| `fabric_quality` | 10 | 2 |
| `fit_sizing` | 12 | 2 |
| `occasion` | 8 | 1 |
| `value` | 6 | 1 |

Relevant-document count per question ranges 0–20 (mean 13.82), from T2.7's pooled relevance labeling. **2 questions correctly abstain** ("not enough information") — evidence-driven, not taken from the original taxonomy negative flag; see the Known limitations section.

**Image evidence (RQ4):** 49 questions were flagged image-target by T2.3 (categories with real image-bearing corpus coverage). Of those, **37/49 (75.51%) got a genuine best-evidence photo** from T2.9 — the rest had relevant reviews, but none of *those specific* reviews carried an image. This is the honest, unpadded number; report it as-is (PLAN.md risk D).

Grounded (non-abstention) answers cite a mean of 8.79 reviews each.

## Collection process
T2.1 taxonomy → T2.2 LLM-drafted candidates (Groq) → T2.3 human curation + authored negatives → T2.4/T2.5 BM25 + dense retrievers → T2.6 TREC-style pooling (union of top-k) → T2.7 human relevance labeling over every pooled pair → T2.8 human reference answers, grounded and citation-checked against T2.7 → T2.9 best-evidence photo selection, scoped to T2.7-relevant image-bearing reviews → T2.10 (this file): assembly, cross-task re-validation, datasheet, leakage guard.

## Known limitations
- **Single annotator.** All human judgments (curation, relevance, reference answers, photo selection) were made by one person; no second-annotator agreement (κ) has been computed for this benchmark itself (distinct from the answer-quality *judge* validation planned in E5).
- **Several authored negatives turned out answerable.** T2.7's labeling found real relevant evidence for 9 of the 11 originally-authored negative questions (PLAN.md §14.2) — abstention in this file is evidence-driven (zero relevant docs), not taken from the original taxonomy intent, so it reflects what's actually true of the corpus rather than what T2.3 assumed.
- **Image coverage is real, not padded.** T2.3's `image_target` flag is a category-level estimate; T2.9's actual per-question coverage (75.51%) is lower, because a category having decent image coverage overall doesn't guarantee the specific relevant reviews for every question are photographed.
- **Pooled, not exhaustive, relevance.** Recall/Hit@k computed against this file are correct *with respect to the retrievers pooled in T2.6* (BM25 + dense), per PLAN.md risk A — not against the full corpus, which was never exhaustively labeled.

## Held-out status
**CRAGB v1 must never appear in fine-tuning training data.** A leakage-guard manifest (`benchmark/cragb_v1_leakage_manifest.json`) records a SHA-256 hash of every question's normalized text; `cragb.bench.assemble.check_no_leakage` checks candidate fine-tune training text against it before E8 accepts any synthetic example.

## Provenance
Built from `corpus_v1.parquet` (seed 42, 200000 rows, sha256 `180c2f1029228543…`; see `data/processed/corpus_v1_manifest.json`).

