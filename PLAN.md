# CRAGB — Project Plan & Research Roadmap
### Multimodal (photo-grounded) Retrieval-Augmented QA over Amazon Clothing Reviews
**Module:** ECS-8060 AI Engineering · **Author:** Mohammed Zaid Shaikh (40507362)
**Status:** Living document — this is the source of truth for scope, experiments, and sequencing. Update it as results land; commit changes.

> **How to read this.** Sections 1–4, 7 and 13 are the operational core (what unblocks the first half + the mid-progress report). Sections 5–6 and 8–11 are frameworks. Section 10 (fine-tuning) deliberately fixes *decision rules*, not final hyperparameters — those are a second-half deliverable once baselines exist. Section 12 is what to deliberately *not* build.

> **Alignment anchors used throughout:**
> - **Course (Lec 3):** AI Application Life Cycle = *Understand problem → Select model → Build evaluation → Change model behaviour*. Eval design = *pick axis → how to measure (functional correctness / MCQ / similarity / AI-as-judge) → find or build a benchmark → measure cost & latency*.
> - **Course (Lec 5):** RAG = chunk → index → retrieve (sparse TF-IDF/BM25 vs dense embeddings, cosine) → generate. RAG eval goals = *(1) retrieve correct docs (Recall), (2) better answers, (3) cost (QPS / build time / index size)*.
> - **Amazon (Rufus / Alexa-for-Shopping JDs):** benchmark creation for GenAI, LLM-as-a-judge + rubric design + dataset curation, retrieval augmentation & context enrichment, SLM post-training under latency/cost limits, multimodal understanding, statistical rigor, dashboards & reporting.

---

## SECTION 1 — Understanding the proposal (supervisor's read)

### 1.1 What the project is
A **retrieval-augmented QA system over Amazon *Clothing/Shoes/Jewelry* reviews** that answers open-ended shopper questions **strictly grounded** in retrieved review text, and **surfaces the photo attached to a cited review as visual evidence**. It is evaluated on a purpose-built benchmark, **CRAGB**, along four goals: retrieval quality, answer quality, visual-evidence relevance, cost & latency. Three experimental axes are studied: **LLM scale (RQ1), retrieval method (RQ2), small-scale fine-tuning (RQ3)**.

### 1.2 Research problem, motivation, novelty, gap
- **Problem:** shoppers ask open-ended questions ("does this run small?", "is the colour as pictured?") that a single product description can't answer but the *review corpus* can — if you can retrieve the right reviews and answer without hallucinating.
- **Motivation:** a claim backed by a *cited customer photo* ("buyers say the colour runs darker" + the photo showing it) is more trustworthy than text alone.
- **Novelty / gap:** text-only review RAG is well-trodden; **full cross-modal fusion** is heavy and infeasible on 8 GB VRAM. The proposal occupies a deliberate middle ground: **retrieve photos *with* their review and cite them, rather than fusing them into a joint embedding.** That is a defensible, well-scoped novelty — *not* a weakness, provided the claims stay honest (see 1.4-D).

### 1.3 Datasets, benchmark, evaluation, components
- **Corpus:** Amazon Reviews 2023, Clothing/Shoes/Jewelry (proposal cites 66.0 M reviews / 7.2 M products, joined on `parent_asin`; ~5.86% image-bearing; 63.65% five-star, 14.26% one–two-star). *Verify these figures against the dataset card before quoting them in the report.* Sampled to a working subset (low hundreds of thousands of reviews).
- **Benchmark — CRAGB:** 50–100 hand-labelled category-wide questions, each with (1) relevant review IDs, (2) a human reference answer, (3) a best-evidence photo for the image-bearing subset.
- **Evaluation goals:** G1 retrieval (Recall, Hit@k) · G2 answer quality (similarity, AI-judge) · G3 visual-evidence relevance (pairwise AI-judge) · G4 cost & latency (QPS, build time, index size).
- **Components:** (1) prompt engineering / context construction [core]; (2) multimodal retrieval & evidence grounding [core]; (3) fine-tuning [primary + fallback]; (4) local inference [core]; (5) lightweight demo front end.
- **Hardware:** RTX 4060, 8 GB VRAM; local embedding/indexing/BM25 + fine-tuned SLM; two comparison LLMs via free-tier APIs (Groq, Google AI Studio).

### 1.4 Strengths → Weaknesses → Risks → Bottlenecks

**Strengths**
- **Course-aligned to the point of being the reference example** (CRAGB *is* Lec 3's "build your own benchmark"; the four goals *are* Lec 5's eval goals + one).
- **Amazon-aligned to the specific JDs** — benchmark creation, LLM-as-judge, RAG, SLM fine-tuning under cost/latency, multimodal — including a *preferred qualification* that appears twice.
- **Good engineering judgment:** the cross-modal-fusion → photo-with-review simplification is exactly the "simplicity beats complexity" call a senior reviewer wants to see.
- **Mature risk posture:** a *pre-declared go/no-go* fallback for fine-tuning.

**Weaknesses / risks (ranked by how badly they can hurt you)**

| # | Risk | Why it matters | Fix (where addressed) |
|---|------|----------------|------------------------|
| A | **Incomplete relevance judgments** | Hand-labelling relevant IDs over 200k reviews is impossible to do completely → Recall/Hit@k count unlabelled true-positives as misses → RQ2 uninterpretable. | **Pooling** (§3-E1, §8) |
| B | **RQ1 confounds scale** with family + API + quantization | Can't attribute a quality gap to "scale". | Same-family size pair over one API (§2-RQ1, §3-E-scale) |
| C | **Random sampling starves negative questions** | 63.65% 5-star → defect/fit questions may lack evidence in the index → low recall blamed on retriever, not corpus. | Stratified / query-aware sampling (§3-E0/E1) |
| D | **"Visual grounding" over-claim** | Text-RAG generator never sees pixels; the UI displays the photo. Claiming the model "reasons over" images is false. | Reframe as *evidence surfacing*; vision-LLM judge (§2-RQ4, §9) |
| E | **Un-validated answer metric + judge** | Embedding-similarity to one reference is weak; LLM judges have position/verbosity/self-preference bias. | Judge-vs-human agreement (κ) before trusting it (§5? no →§8) |
| F | **Fine-tune data separation / volume** | 50–100 Qs can't train a model, and training on "CRAGB-style" data risks eval contamination. | Separate synthetic train set + strict split (§10) |
| G | **8 GB VRAM misattribution** | Retrieval is *not* the VRAM bottleneck (index sits in RAM; encoding is quick); the *LLM fine-tune/generation* is. Plan compute accordingly. | §10, §4 |

**Bottlenecks (schedule-critical)**
1. **CRAGB construction** — the whole project gates on it. Start it early; build it *pooled* so it improves as retrievers improve.
2. **Human labelling time** — relevance + reference answers + best-evidence photos is the scarcest resource. Budget it explicitly; keep the question count at the lower end (≈60) done *well* rather than 100 done *shallowly*.
3. **API rate limits / drift** — free-tier Groq/Google AI Studio quotas and model deprecations. Cache all API responses to disk (also makes results reproducible).

---

## SECTION 2 — Research questions → hypotheses → experiments (mapping)

The proposal is one umbrella RQ with three axes. Decompose into five testable questions. **RQ0** is the feasibility claim; **RQ1–3** are the proposal's axes (de-confounded); **RQ4** is the multimodal contribution stated honestly.

| RQ | Question | Independent var | Dependent var(s) | Controls held fixed |
|----|----------|-----------------|------------------|----------------------|
| RQ0 | Can strictly-grounded review-RAG answer category-wide questions without hallucinating? | pipeline present vs closed-book LLM | answer quality, grounding/faithfulness, abstention correctness | retriever, generator, prompt |
| RQ1 | Does **LLM scale** improve answer quality (given the same retrieved context)? | model **size within one family, same API** | answer quality (judge + similarity), faithfulness, latency, $/query | retriever, k, prompt |
| RQ2 | **Sparse (BM25) vs dense** retrieval — which retrieves better evidence? | retriever type | Recall@k, Hit@k, nDCG@k, MRR | corpus, chunking, generator, k |
| RQ3 | Does **LoRA/QLoRA fine-tuning** of a small local model beat the un-tuned baseline? | base vs fine-tuned SLM | answer quality, faithfulness, format/citation validity, latency | retriever, prompt, eval set |
| RQ4 | Does surfacing the **cited photo** add evidential value, and how often is the surfaced photo actually relevant? | photo shown vs text-only | visual-evidence relevance (pairwise vision-judge), human preference | question set, retriever |

**Hypotheses (directional, falsifiable):**
- H1: larger same-family model → higher answer-quality judge score, at higher latency/cost (expect diminishing returns once context is good — retrieval quality may dominate scale).
- H2: dense ≥ BM25 on *paraphrased/semantic* questions; **BM25 competitive or better on exact-attribute/lexical** questions (size, colour names, brand). Expect a **query-type interaction**, not a global winner — that interaction is a headline result.
- H3: LoRA fine-tune improves *format/citation compliance and abstention* more than raw factuality; watch for regression on out-of-distribution questions.
- H4: surfaced photo is judged relevant well above a random-photo baseline, but only on the image-bearing subset — report coverage honestly.

**Metrics / outputs / visualisations / stats per RQ:** see §8 for the metric definitions; each RQ produces (i) a results table with bootstrap 95% CIs, (ii) at least one figure, (iii) a significance test appropriate to the design (paired bootstrap or Wilcoxon signed-rank across the shared question set — questions are the paired unit).

### Dependency graph (RQ → experiment → evaluation → conclusion)
```
                         ┌────────────────────────── E0 Data + EDA + sampling ─────────────────────────┐
                         │                                                                              │
                         ▼                                                                              ▼
              E1 CRAGB construction (POOLED)  ◄──────────── pools come from ───────────  E3 Retrieval (BM25 vs dense)
                 │   │        │                                                                │
   relevant IDs  │   │ ref answers │ best-evidence photos                                     │ Recall/Hit@k/nDCG/MRR
                 ▼   ▼        ▼                                                                ▼
        ┌── RQ2 eval ──┐   E4 Grounded-QA prompt + citations (Component 1)  ──►  E5 Answer-quality eval + JUDGE VALIDATION
        │  (retrieval) │        │                                                    │      (RQ0 baseline, RQ1 scale)
        ▼              │        ▼                                                    ▼
   Conclusion RQ2      │   E6 Cost & latency (G4)  ──►  Conclusion RQ1 / RQ0    E7 Multimodal grounding pilot (RQ4)
                       │                                                             │
                       └───────────────────►  E8 Fine-tune data prep + PLAN  ──► (2nd half) RQ3 execution ──► Conclusion RQ3
```
Read top-to-bottom: **data feeds the benchmark; the benchmark + retrievers co-produce the pooled labels; retrievers answer RQ2; the grounded-QA prompt + validated judge answer RQ0/RQ1; the photo pilot answers RQ4; everything feeds the fine-tuning plan for RQ3.**

---

## SECTION 3 — Experiment designs (everything due before mid-progress)

Format per experiment: **Objective · RQ · Input · Preprocessing · Baseline · Models · Pipeline · Metrics · Outputs · Failure modes · Interpretation · Industry relevance.**

### E0 — Data acquisition, EDA, and corpus construction
- **Objective:** turn 66 M raw reviews into a documented, reproducible working subset whose composition is *justified*, not arbitrary.
- **RQ:** enables all; directly de-risks C (sampling) and G (compute).
- **Input:** Amazon Reviews 2023 Clothing/Shoes/Jewelry (reviews + item metadata), joined on `parent_asin`.
- **Preprocessing:** dedup, language filter (English), drop empty/near-empty text, normalise whitespace, parse image URLs, compute per-review token length; join product title/category/attributes.
- **Baseline:** none (descriptive).
- **Models:** none.
- **Pipeline:** streamed/chunked read (never load 66 M in RAM) → filters → **stratified sample** across rating (force adequate 1–2★ mass), presence-of-image, and product sub-category → freeze `corpus_v1` with a fixed seed and a manifest (row count, hash, column schema).
- **Metrics/EDA outputs:** rating distribution; review-length distribution; image-coverage %; reviews-per-product; sub-category coverage; vocabulary/coverage of attribute terms (size, fit, colour, fabric).
- **Outputs:** `data/processed/corpus_v1.parquet`, `reports/figures/eda_*.png`, a **datasheet** (Gebru et al. style) documenting provenance, filters, seed, and known biases.
- **Failure modes:** silent join loss on `parent_asin`; sampling that still under-represents negatives; memory blow-up.
- **Interpretation:** the EDA *is* your "Understand the problem" stage (Lec 3) and directly motivates why CRAGB includes negative questions.
- **Industry relevance:** Amazon DS JDs = "hands-on analysis of enormous multimodal datasets," "ETL pipelines," "structured + unstructured signals." A datasheet + stratified sampling is exactly the rigor they screen for.

### E1 — CRAGB construction (the spine) — build it **pooled**
- **Objective:** a defensible benchmark of ~60 questions with trustworthy relevance labels, reference answers, and best-evidence photos.
- **RQ:** underpins RQ0–RQ4; fixes risk A.
- **Input:** `corpus_v1`; optionally real question signals (mine "Questions & Answers" style phrasings, or LLM-draft then human-curate to avoid the student-writes-only-answerable-questions bias).
- **Preprocessing:** define a **question taxonomy** (fit/sizing, colour/appearance, fabric/quality, durability, defects, occasion, value) with target counts per type incl. negatives.
- **Pooling protocol (the key move):** for each question, run **every retriever under study** (BM25 + dense, multiple k) → **pool the union of top-k** → a human judges only the pool for relevance (binary or graded). Labels are then complete *with respect to the systems being compared* — the standard TREC remedy for incomplete judgments.
- **Reference answers:** human-written from the pooled relevant reviews (this makes them grounded, not imagined).
- **Best-evidence photo:** for image-bearing questions, human selects the single most probative photo among pooled relevant reviews.
- **Metrics of the benchmark itself:** inter-item balance across taxonomy; % questions with ≥1 relevant doc; % with a usable photo; (optional) second-annotator agreement on a subset.
- **Outputs:** `benchmark/cragb_v1.jsonl` (question, type, relevant_ids[], reference_answer, best_photo_id), a **held-out split** never used for fine-tuning, and a benchmark datasheet.
- **Failure modes:** pool too shallow (raise k); taxonomy imbalance; leakage of eval questions into any training set.
- **Interpretation:** now Recall@k means what it says.
- **Industry relevance:** "defining and creating benchmarks for assessing GenAI performance" — the literal JD line. Pooling is the detail that signals you actually know IR evaluation.

### E2 — Chunking study (small, decisive)
- **Objective:** pick a chunking scheme with evidence, not vibe.
- **RQ:** control for RQ2/RQ0.
- **Pipeline:** compare {whole-review} vs {fixed 128/256-token chunks} vs {sentence-window}; measure downstream Recall@k on a dev slice of CRAGB.
- **Outputs:** one table + one figure; a fixed `chunking.yaml`.
- **Interpretation:** reviews are short → *whole-review-as-chunk* is often best and simplest; prove it rather than assume. (Simplicity, if the data supports it.)
- **Industry relevance:** "context engineering / context enrichment" (Applied Scientist JD).

### E3 — Retrieval baselines: BM25 vs dense (**RQ2**)
- **Objective:** clean sparse-vs-dense comparison.
- **Input:** `corpus_v1` index; CRAGB questions + pooled labels.
- **Baseline:** **BM25** (`rank_bm25`/Pyserini) — the industrial sparse standard; TF-IDF only as an optional weaker reference (Lec 5 taught TF/IDF conceptually — use BM25 in practice).
- **Models:** dense = `bge-small-en-v1.5` and/or `all-MiniLM-L6-v2`, cosine over a FAISS **flat** index (exact; no ANN tuning needed at this scale — see §12).
- **Pipeline:** identical chunking + identical k for both; vary k ∈ {1,3,5,10}.
- **Metrics:** Recall@k, Hit@k, nDCG@k, MRR; **paired bootstrap CIs** across questions; **breakdown by question type** (H2 interaction).
- **Outputs:** results table; Recall@k-vs-k curves (both retrievers); a per-type bar chart; 3–5 qualitative win/loss examples each way.
- **Failure modes:** unfair k/chunking; embedding model domain mismatch; ANN recall loss (avoided by flat index).
- **Interpretation:** expect *no global winner* → the query-type interaction is the finding.
- **Industry relevance:** "information retrieval, ranking, relevance" (all three shopping JDs).

### E4 — Grounded-QA prompt + citations (**Component 1**, RQ0 substrate)
- **Objective:** a prompt that answers *only* from retrieved chunks + attached photo, cites text (and photo) evidence, and says **"not enough information"** when unsupported.
- **Pipeline:** retrieved chunks → structured context with IDs → strict-grounding instruction → answer with inline citations `[R#]` and, where present, `[photo of R#]`.
- **Metrics:** citation validity (do cited IDs exist & support the claim?), abstention correctness on deliberately unanswerable questions, format compliance.
- **Outputs:** the prompt (versioned), 5 worked transcripts, a citation-validity rate.
- **Failure modes:** model ignores grounding (hallucinates), over-abstains, fabricates citations.
- **Interpretation:** this is "Change model behaviour" via prompting (Lec 3) *before* touching weights.
- **Industry relevance:** "instruction design, contextual grounding, context engineering" (Applied Scientist JD) — verbatim.

### E5 — Answer-quality evaluation + **judge validation** (RQ0, RQ1)
- **Objective:** trustworthy answer-quality measurement; then use it to compare closed-book vs RAG (RQ0) and model sizes (RQ1).
- **Baseline:** closed-book LLM (no retrieval) — isolates the value of retrieval (RQ0).
- **Models (RQ1):** **same family, two sizes, same API** (e.g. Llama-3.1-8B vs Llama-3.3-70B on Groq; *or* Gemini Flash vs Pro). This is the de-confounded RQ1.
- **Pipeline:** similarity metric (BERTScore/embedding-cos to reference) **plus** rubric AI-judge (correctness, grounding/faithfulness, completeness, conciseness). **Validate the judge:** hand-score 30–50 answers, compute judge-vs-human **Cohen's κ / correlation**, only then trust it at scale; mitigate position bias by randomising order and (for pairwise) running both orders.
- **Metrics:** per-criterion judge scores + similarity, with bootstrap CIs; judge-human agreement reported.
- **Outputs:** RQ0 table (RAG vs closed-book), RQ1 table (small vs large), judge-reliability table, error examples.
- **Failure modes:** judge bias unmeasured; single-reference similarity punishing valid paraphrases (that's *why* you pair it with the judge).
- **Industry relevance:** "LLM-as-a-judge methodologies, rubric design, dataset curation" (Applied Scientist JD) — verbatim; measuring judge reliability is the senior move.

### E6 — Cost & latency (**G4**)
- **Objective:** quantify the operational cost of each configuration.
- **Metrics:** index **build time** & **size**; **QPS** / p50–p95 retrieval latency; end-to-end latency; tokens & **$/query** for API models; local VRAM/RAM.
- **Outputs:** a cost/latency table joined to quality → a **quality-vs-cost scatter** (the Pareto view).
- **Interpretation:** directly Lec 5 Goal 3 + Lec 3 "Cost and Latency"; grounds the "is the big model worth it?" decision.
- **Industry relevance:** SLM post-training exists *because* of "latency and cost constraints" (Applied Scientist JD). Showing the trade-off curve is the whole point.

### E7 — Multimodal grounding pilot (**RQ4**, Component 2 visual half)
- **Objective:** measure whether the surfaced photo is actually relevant evidence, honestly scoped to the image-bearing subset.
- **Pipeline:** for image-bearing questions, take the top retrieved review's photo → **vision-LLM pairwise judge**: is the surfaced photo more relevant to the question/claim than a random-photo control? (This is your CADB visual-relevance judge.)
- **Metrics:** win-rate vs random-photo baseline; coverage (% questions with a usable photo); human spot-check.
- **Failure modes:** too few image-bearing questions → underpowered (mitigate in E1 by targeting image coverage).
- **Interpretation:** frame as *evidence surfacing*, not image-conditioned generation (fixes risk D).
- **Industry relevance:** "multimodal understanding," "answers that combine text, image, video" (Rufus/AfS JDs).

### E8 — Fine-tuning data prep + go/no-go plan (feeds RQ3; execution is 2nd half)
- **Objective:** produce a *separate, larger* synthetic grounded-QA training set and lock the decision rules — **not** run the fine-tune yet.
- **Pipeline:** generate ~1–5k (context → grounded, cited answer / correct abstention) pairs from `corpus_v1`, **disjoint from CRAGB questions**; filter with the validated judge; split train/val.
- **Outputs:** `data/finetune/{train,val}.jsonl`, a data datasheet, and the §10 go/no-go criteria written down.
- **Failure modes:** eval leakage (guard with hashing), low-quality synthetic labels capping the ceiling.
- **Industry relevance:** "dataset curation," "supervised fine-tuning… for low-latency experiences" (Applied Scientist JD).

---

## SECTION 4 — Roadmap & milestones
*(Durations are placeholders on a 12-week semester, mid-progress ≈ week 6. Replace with your real dates.)*

| Stage | Purpose | Key deliverables | Est. | Depends on | Git commits (examples) | Artifacts |
|------|---------|------------------|------|-----------|------------------------|-----------|
| M0 Repo & env | Reproducible scaffold | repo tree, `env.yml`, seed util, `configs/`, CI lint | 0.5 wk | — | `chore: scaffold repo + conda env` | `environment.yml` |
| M1 Data + EDA (E0) | Understand + build corpus | `corpus_v1`, EDA figures, datasheet | 1 wk | M0 | `feat(data): stratified corpus_v1 + EDA` | parquet, figs |
| M2 CRAGB v1 (E1) | The benchmark, pooled | `cragb_v1.jsonl`, taxonomy, datasheet | 1.5 wk | M1, M4a | `feat(bench): pooled CRAGB v1 (60 Qs)` | jsonl |
| M3 Retrieval (E2,E3) | RQ2 answered | BM25+dense indexes, retrieval tables/figs | 1.5 wk | M1 | `feat(retrieval): bm25+dense + eval` | indexes, results |
| M4a Grounded prompt (E4) | Component 1 core | prompt v1, transcripts | 0.5 wk | M1 | `feat(prompt): grounded QA + citations` | prompt file |
| M4b Answer eval (E5) | RQ0/RQ1 + judge | judge, validation table, RQ0/RQ1 tables | 1.5 wk | M2,M3,M4a | `feat(eval): rubric judge + human κ` | results |
| M5 Cost/latency (E6) | G4 | cost table, Pareto scatter | 0.5 wk | M3,M4b | `feat(eval): cost & latency harness` | results |
| M6 Visual pilot (E7) ✅ | RQ4 | photo store + pairing + order-swap vision judge (Gemini) + win-rate CI + human κ spot-check + figure #6 + appendix | 1 wk | M2,M3 | `feat(mm): photo evidence pilot` | `mm_winrate_v1.csv`, `mm_spotcheck_v1.csv`, `mm_photo_winrate_v1.png`, `multimodal_examples_v1.md` |
| M7 FT prep + plan (E8) | RQ3 setup | train/val sets, go/no-go doc | 1 wk | M2,M4b | `feat(ft): synthetic data + FT plan` | jsonl, doc |
| **MID-PROGRESS REPORT** | Write-up | report + appendix | 1 wk | M1–M7 | `docs: mid-progress report` | PDF |
| M8 (2nd half) FT exec | RQ3 result | LoRA runs, RQ3 tables | — | M7 | `feat(ft): qlora runs + eval` | adapters |
| M9 Demo | Component 5 | Streamlit/Gradio demo | — | all | `feat(demo): grounded QA UI` | app |

**Critical path:** M1 → M3 → M2 (pooling needs retrievers) → M4b. Start M4a in parallel with M3.

---

## SECTION 5 — Code architecture (industry-standard, config-driven)

```
cragb/
├── README.md
├── PLAN.md                      # this file
├── environment.yml / pyproject.toml
├── Makefile                     # `make data`, `make index`, `make eval`
├── configs/                     # YAML — all knobs live here, zero magic numbers in code
│   ├── data.yaml   retrieval.yaml   generate.yaml   eval.yaml   finetune.yaml
├── src/cragb/                   # importable package (logic lives here, NOT in notebooks)
│   ├── data/        # load, clean, sample, datasheet
│   ├── retrieval/   # bm25.py, dense.py, index.py, base.py (Retriever interface)
│   ├── generate/    # prompt templates, grounded_qa.py, api_clients.py (+ disk cache)
│   ├── eval/        # metrics_retrieval.py, metrics_answer.py, judge.py, bootstrap.py
│   ├── multimodal/  # photo_link.py, vision_judge.py
│   ├── finetune/    # dataset.py, train_lora.py, gonogo.py
│   └── utils/       # seeds.py, io.py, logging.py, timing.py
├── notebooks/                   # THIN: call src/, make figures, tell the story (§6)
├── benchmark/                   # cragb_v1.jsonl + datasheet
├── data/  {raw, processed, finetune}   # git-ignored; tracked by hash/DVC
├── results/ {tables, cache}     # versioned result artifacts
├── reports/ {figures, mid_progress}
└── tests/                       # pytest: metrics correctness, retriever interface, prompt I/O
```
**Principles:** single `Retriever` interface (`.index()`, `.search(q,k)`) so BM25/dense are swappable; **all randomness seeded**; **every API call cached to disk** (reproducibility + quota safety); **no magic numbers** (config only); metrics have **unit tests** (a wrong Recall implementation invalidates everything). This mirrors "automated eval pipelines / ETL / experimentation frameworks" in the JDs.

---

## SECTION 6 — Notebook plan (thin notebooks over a fat `src/`)

Notebooks *tell the story and make figures*; they import from `src/` and contain no reusable logic. This is the antidote to "notebook spaghetti."

| Notebook | Produces (report material) |
|----------|----------------------------|
| `01_eda.ipynb` | rating/length/image-coverage figures, datasheet numbers |
| `02_corpus_build.ipynb` | stratified `corpus_v1` + manifest |
| `03_chunking_study.ipynb` | chunking table + chosen scheme |
| `04_retrieval_bm25_vs_dense.ipynb` | Recall@k curves, per-type bars, win/loss examples (RQ2) |
| `05_cragb_pooling.ipynb` | pooled labels, benchmark stats |
| `06_grounded_qa.ipynb` | prompt transcripts, citation-validity |
| `07_answer_eval_judge.ipynb` | judge validation (κ), RQ0/RQ1 tables |
| `08_cost_latency.ipynb` | cost table, quality-vs-cost Pareto |
| `09_multimodal_pilot.ipynb` | photo-evidence win-rate, examples (RQ4) |
| `10_finetune_dataprep.ipynb` | train/val stats, leakage check |

---

## SECTION 7 — Exactly what to have ready before the mid-progress report

The report needs: **investigation of RQs, progress on the first two components (prompt engineering + multimodal retrieval), a fine-tuning plan, appendix.** Have these concrete artefacts in hand so writing is assembly, not discovery:

**Figures**
1. Rating + review-length distributions; image-coverage bar (motivates negatives & multimodality).
2. Recall@k vs k, BM25 vs dense (RQ2 headline).
3. Per-question-type retrieval bar chart (the interaction).
4. Quality-vs-cost Pareto scatter (RQ1/G4).
5. Judge-vs-human agreement plot (reliability).
6. Multimodal: photo-evidence win-rate vs random baseline.
7. Architecture diagram (pipeline; you already have one — keep it consistent with Lec 5's block diagram).

**Tables**
- Corpus/benchmark composition (with taxonomy counts).
- Retrieval metrics (Recall/Hit@/nDCG/MRR) + bootstrap CIs.
- RQ0 (RAG vs closed-book) and RQ1 (small vs large) answer-quality.
- Cost & latency (build time, index size, QPS, $/query).
- Judge validation (κ / correlation).

**Metrics/numbers ready:** Recall@{1,3,5,10}, Hit@k, nDCG, MRR, citation-validity rate, abstention accuracy, per-criterion judge means + CIs, QPS/latency/cost, photo win-rate, image coverage %.

**Examples / qualitative (appendix gold):** 3–5 retrieval win/loss pairs each way; 5 grounded-QA transcripts with citations; 2–3 correct-abstention cases; 3–4 photo-evidence examples; a first-pass **error taxonomy** (retrieval miss / grounding failure / hallucinated citation / over-abstain / photo-irrelevant).

**Prompt examples:** grounded-QA prompt, answer-quality judge prompt, pairwise vision-judge prompt, benchmark-generation prompt (all versioned).

**Fine-tuning plan section:** base model choice + justification, LoRA/QLoRA approach, the *separate* training set, and the **go/no-go rule** — plan only, no runs.

---

## SECTION 8 — Evaluation framework (the metric taxonomy)

Organised by Lec 5's goals, extended with the rigor the JDs expect.

- **G1 Retrieval:** Recall@k, Hit@k, **nDCG@k** (graded relevance), **MRR**. Report with **paired bootstrap 95% CIs**; significance via paired bootstrap / Wilcoxon signed-rank over the shared question set.
- **G2 Answer quality:** similarity (BERTScore/embedding-cos to reference) **+** rubric AI-judge (correctness, **faithfulness/grounding**, completeness, conciseness) **+ judge validation (Cohen's κ vs human)**. Add **abstention accuracy** and **citation validity** (a grounded system must cite real, supporting evidence).
- **G3 Visual-evidence relevance:** pairwise vision-judge win-rate vs random-photo baseline; coverage %.
- **G4 Cost & latency:** build time, index size, QPS, p50/p95 latency, tokens & $/query, VRAM/RAM.
- **Cross-cutting rigor:** bootstrap CIs everywhere; **error taxonomy + per-slice breakdown** (by question type); calibration of abstention; a **single results schema** so every experiment writes comparable rows → feeds one summary dashboard (§11).

*Hallucination/grounding note:* measure **faithfulness** as "every claim traceable to a cited chunk" (judge-scored on a 3-point scale) rather than a vague "hallucination rate" — it's operational and defensible.

---

## SECTION 9 — Prompt engineering strategy (and *why* each is shaped so)

- **Grounded-QA prompt:** explicit context block with per-chunk IDs; a hard rule "answer *only* from the context; if insufficient, reply *not enough information*"; required inline citations. *Why:* Lec 5's whole motivation is stopping hallucination via external context — the prompt operationalises that and makes faithfulness measurable.
- **Citation formatting:** deterministic `[R#]` / `[photo of R#]` scheme. *Why:* machine-checkable citation validity → a metric, not a vibe.
- **Answer-quality judge prompt:** specify **task, criteria, scoring scale** (exactly Lec 3's AI-judge structure); force JSON with per-criterion scores + rationale. *Why:* structure reduces judge variance and enables instruction-following verification.
- **Pairwise vision-judge:** present question + two photos (surfaced vs control), ask which better evidences the claim; **run both orders** to cancel position bias. *Why:* pairwise is more reliable than absolute scoring for relevance, and order-swap controls a known bias.
- **Benchmark-generation prompt:** draft candidate questions *from sampled reviews*, tagged by taxonomy type, for **human curation** (never auto-accepted). *Why:* reduces the "student only writes answerable questions" selection bias while keeping humans in the loop.

*Common mistakes to avoid:* leaking the reference answer into the judge when you want a reference-free score; letting the judge see which system produced an answer (enables self-preference); vague criteria; unversioned prompts (kills reproducibility).

---

## SECTION 10 — Fine-tuning plan (decision rules now; hyperparameters after baselines)

> **M7 correction (2026-08-24, §14.7 / `reports/finetune_plan_v1.md`):** this section's "fits
> 8 GB" assumption was wrong — the real dev machine has **4 GB VRAM**
> (RTX 3050 Laptop, `nvidia-smi`-confirmed). Measured against the real hardware: Qwen2.5-3B-Instruct
> *inference* fits at 4-bit with headroom (T7.7), but every QLoRA *training* configuration
> measured requires partial CPU offload to fit at all (T7.9) — Llama-3.2-3B-Instruct is dropped
> (gated HF repo, no access requested). The base model, method, and go/no-go rule below are
> superseded by the fully-worked, numbers-backed version in `reports/finetune_plan_v1.md` — this
> section is kept as the original a-priori plan for the record, not updated in place.

> I'm deliberately **not** inventing final hyperparameters here. Locking a LoRA config before you know where the baseline fails is false precision. The plan below fixes the *method and the go/no-go*; the exact rank/lr/epochs are set once E5 shows the failure modes.

- **Base model:** small local, 4-bit, fits 8 GB — **Qwen2.5-3B-Instruct** or **Llama-3.2-3B-Instruct** (sanity-check for a newer small model at build time; the choice is stable enough that it won't change the design).
- **Method:** **QLoRA** (4-bit base + LoRA adapters) — the only realistic route on 8 GB. Expect batch size 1 + gradient accumulation, gradient checkpointing, short max-seq consistent with your retrieved-context length.
- **Data (from E8):** ~1–5k synthetic *grounded, cited* QA pairs (and correct-abstention examples), **strictly disjoint from CRAGB** (hash-guard), judge-filtered, train/val split.
- **Target behaviours to improve (hypothesis-led):** citation/format compliance, correct abstention, grounding faithfulness — *not* raw world knowledge (a 3B won't gain facts; it can gain discipline). **Superseded by T7.8's measured baseline (§14.7): citation/format compliance and faithfulness are already near-ceiling untuned; the real target is severe over-abstention (53–75% false-abstention on answerable questions, 100% true-abstention recall) — see `reports/finetune_plan_v1.md` §4.**
- **Validation:** track val loss + a small held-out grounded set during training; early stop on val.
- **Evaluation of the tuned model:** re-run E5/E6 on CRAGB held-out — answer quality, faithfulness, citation validity, **latency** (does it stay fast?), with bootstrap CIs and per-type slices.
- **Go/no-go (pre-declared):** proceed to full runs only if a pilot LoRA shows ≥ *X* improvement on citation/abstention with no quality regression and no latency blow-up within *T* hours of GPU time; **else trigger the proposal's fallback** — fine-tune the **embedding/retriever** instead (contrastive on CRAGB-style pairs), reusing the same data. **X and T are now concrete numbers, not placeholders — `reports/finetune_plan_v1.md` §6.**
- **"Did it actually help?"** Not a single aggregate win — require **per-slice improvement + significance vs the un-tuned baseline on the shared held-out set**, and check it didn't regress OOD questions.
- **Industry relevance:** "supervised fine-tuning… distillation… for low-latency conversational experiences," "post-training of SLMs where latency and cost demand it" (Applied Scientist JD, verbatim).

---

## SECTION 11 — Amazon-level upgrades (marks *and* recruiter signal)

Each is low-effort-high-signal given the pipeline already exists:

1. **One results schema → a small eval dashboard** (even a static HTML/Streamlit table): every experiment writes comparable rows; one view compares all configs. *JD:* "metrics, dashboards, reporting frameworks."
2. **Bootstrap CIs + significance tests** on every headline number. *JD:* "statistical rigor and actionable insights." Separates you from students who report point estimates.
3. **Judge reliability (κ vs human)** before trusting the judge. *JD:* "LLM-as-a-judge methodologies, rubric design."
4. **Error taxonomy + per-slice analysis** (by question type). *JD:* "deep-dive analyses to identify opportunities." This is how Applied Scientists turn a metric into a roadmap.
5. **Quality-vs-cost Pareto** as a decision artefact ("the 70B is +6% quality at 12× cost"). *JD:* the entire rationale for SLM post-training.
6. **Retriever & prompt ablations** (k, chunking, grounding-strictness on/off) reported as a table. *JD:* "context enrichment, prompt decomposition."
7. **Model/benchmark datasheets + reproducibility** (seeds, cached API calls, hashes). *JD:* "automated processes, experimentation frameworks."
8. **A 1-page "working-backwards"-style summary** framing customer problem → metric → result. Amazon's cultural tell; also a great report abstract.
9. **(Framed, not built) A/B-test design paragraph:** how you'd online-test the winning config (metric, unit of randomisation, power). *JD:* "design and analyse A/B tests." Costs a paragraph, signals seniority.

---

## SECTION 12 — What NOT to build (protect the timeline)

- **Full cross-modal fusion / joint image-text embeddings** — the proposal already, correctly, scoped this out. Don't drift back into it.
- **ANN index tuning (HNSW/IVF/PQ)** — at hundreds-of-thousands scale a **FAISS flat** exact index is fast and removes a confound (ANN recall loss). No vector-DB service needed.
- **Indexing the full 66 M corpus** — no marginal insight, large cost. The stratified subset is the scientifically *better* choice, not just the cheaper one.
- **RLHF / DPO / preference optimisation** — out of scope for 8 GB and a mini-project; SFT via QLoRA is the right ceiling. (Mention DPO as future work in one line.)
- **A production/persistent front end, auth, deployment, MLOps orchestration** — the proposal caps the demo correctly; keep it a thin Streamlit/Gradio.
- **Agentic / multi-tool orchestration** — tempting given the JDs, but it dilutes the RQs. One clean pipeline studied rigorously beats a flashy agent studied shallowly.
- **A second dataset / cross-domain generalisation** — scope creep; note as future work.

---

## SECTION 13 — Immediate next steps (next ~4 working sessions)

| # | Task | Why | Expected output | Est. | Git commit |
|---|------|-----|-----------------|------|------------|
| 1 | Scaffold repo + conda env + `configs/` + seed util + pytest stub | Reproducibility before science; nothing else is trustworthy without it | running `make` skeleton, `environment.yml` | 2–3 h | `chore: scaffold repo, env, configs, seeds` |
| 2 | Streamed data load + filters + **stratified sample** → `corpus_v1` + manifest | Fixes risk C; unblocks everything downstream | `corpus_v1.parquet` + hash manifest | 4–6 h | `feat(data): stratified corpus_v1 + manifest` |
| 3 | EDA notebook → the 3 core figures + datasheet | This is your "Understand the problem" evidence and motivates CRAGB negatives | `01_eda.ipynb`, `reports/figures/eda_*.png`, datasheet | 3–4 h | `feat(eda): distributions + datasheet` |
| 4 | `Retriever` interface + **BM25** + **dense (bge-small)** over `corpus_v1`, top-k search working | You need retrievers *before* you can pool CRAGB labels | `src/cragb/retrieval/*`, smoke-test search | 4–6 h | `feat(retrieval): bm25 + dense + interface` |
| 5 | Define CRAGB **taxonomy** + draft ~20 questions (LLM-draft → you curate) | De-risks the benchmark bottleneck early; taxonomy first prevents imbalance | `benchmark/taxonomy.md`, `cragb_draft.jsonl` | 3–4 h | `feat(bench): taxonomy + draft questions` |

**Do not** start prompt-tuning or fine-tuning until tasks 1–4 exist. The order is deliberate: **infrastructure → data → retrievers → benchmark → evaluation → generation → fine-tuning.** Retrievers come before the benchmark specifically because the benchmark's relevance labels must be *pooled* from those retrievers.

---

## SECTION 14 — Build/engineering findings log (things worth knowing before they bite twice)

A running log of concrete gotchas hit during implementation — environment quirks and benchmark-construction surprises — kept here so they're not silently re-discovered in a later session or lost once the task that surfaced them is done.

### 14.1 — Windows MAX_PATH blocks a full local install of the dense-retrieval stack (T2.5)

Installing `sentence-transformers` (and, on first attempt, `torch`'s default CUDA/ROCm wheel) into this machine's actual Python environment hits a **Windows MAX_PATH error**: both packages bundle files nested deep enough (`torch`'s third-party license trees; `transformers/models/deprecated/trajectory_transformer/...`) that the full install path exceeds 260 characters under this Python's per-user site-packages location (`...\WindowsApps\PythonSoftwareFoundation.Python.3.12_...\LocalCache\local-packages\Python312\site-packages\...`).

**Current state (verified 2026-07-22):**
- Main conda/project env: `torch` (CPU build) and `faiss` install fine and are present; **`sentence-transformers` does not install** — this is the actual blocker. `cragb.retrieval.dense` therefore fails to import here, and its 12 tests correctly skip (not fail) in this environment.
- Workaround in use: a virtual environment at a short path, `C:\venv\cragb` (e.g. `C:\venv\cragb\Scripts\python.exe`), which keeps the full install path short enough to avoid the limit entirely. This venv has the full stack working, including a CUDA build of `torch` (`2.6.0+cu124`) that correctly detects the RTX 3050 (`torch.cuda.is_available() == True`).

**Not done, and won't be without explicit sign-off:** enabling Windows long-path support system-wide (`HKLM` registry edit, needs admin) — this is a system-settings change, out of scope for an agent to make unilaterally. If preferred over the venv workaround, it needs an explicit go-ahead first.

**Practical implication:** any task needing the dense retriever (E3 retrieval eval, E5 answer generation if it embeds context, etc.) must run via `C:\venv\cragb\Scripts\python.exe`, not the main environment's `python`. `environment.yml` documents both the CPU default and the CUDA install command for this hardware.

### 14.2 — Several of CRAGB v1's authored "negative" questions turned out to be answerable (Mileston2 T2.7)

T2.3 authored 11 deliberately-unanswerable negative questions (one per category's quota) by hand, reasoning that they asked for a level of detail — exact measurements, lab data, internal QA/manufacturing figures, certifications, warranty terms — that customer review text doesn't typically carry. T2.7's actual relevance labeling (reading all 1,176 pooled review candidates against their questions) tested that assumption directly, and it didn't fully hold up.

**Result:** 9 of the 11 negatives unexpectedly have at least one genuinely relevant pooled document:
`fit_sizing_neg_000`, `fit_sizing_neg_001`, `colour_appearance_neg_000`, `fabric_quality_neg_001`, `durability_neg_000`, `durability_neg_001`, `defects_neg_001`, `occasion_neg_000`, `value_neg_000`.

Only `fabric_quality_neg_000` ("exact thread count") and `defects_neg_000` ("% units with a manufacturing defect per internal QA data") came back genuinely empty, exactly as designed — nobody's review reports a thread count or cites internal QA statistics.

The most striking miss is **`fit_sizing_neg_001`** — "Does the manufacturer's official size chart match what buyers experience?" — where **17 of 19** pooled reviews were genuinely on-topic. Reviewers discuss size-chart mismatches constantly (returns, wrong measurements, chart-vs-garment discrepancies), so this reads as a real, well-evidenced category of shopper question, not a good negative.

**Why this isn't a labeling bug:** `cragb.bench.label_relevance.validate_labels` computes exactly this check (`unexpected_positive_negatives`) as a designed flag, not a hard failure — the point of building it was to catch precisely this kind of authoring mistake by testing it against real data instead of assuming it. It worked.

**Not fixed now, deliberately:** re-authoring these negatives is out of scope for T2.7 (labeling existing pools, not redesigning the taxonomy) and would need a fresh pooling + labeling pass anyway. Worth revisiting in a future CRAGB revision — likely by choosing negatives that ask for information categorically absent from review text everywhere (internal seller data, lab-controlled measurements) rather than plausible-sounding "shopper" phrasings that happen to overlap with common review vocabulary.

### 14.3 — An exact-equality invariant check silently became dead code (Milestone 2 T2.8)

`cragb.bench.reference_answers.make_reference_answer` needed a guard against an authoring mistake: an abstention answer ("not enough information...") that also carries a citation marker, which is self-contradictory. The first version detected abstention via **exact string equality** to the canonical abstention phrase, then checked `if is_abstention and citations_found: raise`.

That guard could never fire. If a citation marker is appended to the phrase, the text is no longer *exactly equal* to the canonical string — so `is_abstention` is `False` before the citation is even inspected, and the contradiction check is unreachable by construction. The bug was caught by the unit test written specifically to prove the guard worked (`test_abstention_text_with_citations_raises`) — it failed with "did not raise," which is what surfaced the dead code.

**Fix:** detect abstention by *containment* (`ABSTENTION_TEXT in answer_text`) rather than exact equality, so appending a stray citation still matches "this is an abstention" and the contradiction guard becomes reachable.

**General lesson for future validation/guard code:** when a check's trigger condition is defined by exact equality to a fixed constant, and the thing you're guarding against is "constant + extra content," the guard is structurally unreachable — equality can never hold once anything is appended. Prefer containment, a prefix/suffix check, or restructuring the invariant so the failure mode you're testing for is actually representable by the detection condition. Write the test for the guard *before* trusting it, the way this one was — that's what caught it here, before it ever mattered in production data.

### 14.4 — Groq catalog snapshot for RQ1's "large" model and the answer-quality judge model (M4b T4b.2)

`GET /openai/v1/models` on Groq (2026-08-19, T4b.2 build time) returned only 13 models
total, most of them not general chat-instruct models at all:

```
allam-2-7b                          SDAIA        ctx 4096   (Arabic/English, small ctx)
canopylabs/orpheus-arabic-saudi     Canopy Labs  ctx 4000   (TTS)
canopylabs/orpheus-v1-english       Canopy Labs  ctx 4000   (TTS)
groq/compound                       Groq         ctx 131072 (agentic/tool-use compound system)
groq/compound-mini                  Groq         ctx 131072 (agentic/tool-use compound system)
meta-llama/llama-prompt-guard-2-22m Meta         ctx 512    (classifier, not chat)
meta-llama/llama-prompt-guard-2-86m Meta         ctx 512    (classifier, not chat)
openai/gpt-oss-120b                 OpenAI       ctx 131072
openai/gpt-oss-20b                  OpenAI       ctx 131072
openai/gpt-oss-safeguard-20b        OpenAI       ctx 131072 (safety-tuned variant)
qwen/qwen3.6-27b                    Alibaba      ctx 131072
whisper-large-v3(-turbo)            OpenAI       (speech-to-text)
```

**RQ1 "large" model:** `openai/gpt-oss-120b` — same `gpt-oss` family as
`configs/grounded_qa.yaml`'s `openai/gpt-oss-20b` (20B vs 120B), confirmed live with a
real completion call. No other same-family size pair exists in this catalog at all,
which narrows RQ1's model choice to this one rather than being a preference among
options — `configs/grounded_qa_large.yaml` locks it in.

**Answer-quality judge model (earmarked here for T4b.4, not yet built):** of the
remaining models, `groq/compound`/`compound-mini` are Groq's own agentic/tool-orchestration
systems, not plain instruct models — an unpredictable fit for a call that must return one
deterministic JSON object, not a tool-use trace. `allam-2-7b`'s 4096-token context window
is too small for a judge prompt carrying the question, context block, candidate answer,
and reference answer together. That leaves **`qwen/qwen3.6-27b`** — a distinct family
from `gpt-oss`, so the judge never scores its own family's output on either the RQ0 or
RQ1 comparison (PLAN.md §9's self-preference-bias caution). One gotcha found testing it:
by default it emits visible `<think>...</think>` reasoning inline in `content` (unlike
`gpt-oss`, which hides its reasoning in a separate `usage.completion_tokens_details`
field and keeps `content` clean) — passing `"reasoning_effort": "none"` in the request
body suppresses this and returns a bare answer, confirmed with a live call. T4b.4 must
either set that flag or strip `<think>...</think>` before parsing the judge's JSON, or
every judge call will fail to parse.

### 14.5 — Cost had to be partly reconstructed after the fact (M5 T5.2/T5.4/T5.5)

M5 (cost & latency, E6/G4) hit two gotchas that both trace back to the same root cause:
`GroqClient.complete()` (T2.2, unchanged through T4a/T4b) returns only the completion
*string* — it reads `data["choices"][0]["message"]["content"]` and throws the rest of the
response body away, `usage` block included. `DiskCache` then stores only that string.
Neither of these was a problem until M5 needed to cost and time calls that had already
happened.

**Token counts for the 180 T4b.2 transcripts don't exist anywhere.** Closed-book,
RAG-small, and RAG-large were all generated before T5.2 added usage telemetry
(`complete_with_usage`, a metadata sidecar on `DiskCache`), so none of those 180 cached
responses carry a token count — and a cache **hit** can never recover one retroactively,
because the original API response that had it is gone; only the completion text survived.
T5.4's fix was to re-issue every transcript's *exact original request* through
`complete_with_usage` (free — every one is a cache hit, since the payload is rebuilt
byte-for-byte from what each transcript already saved: `context_text` for the RAG arms,
run back through the same `render_prompt` the original generation call used) and accept
that the sidecar will be empty for all of them, falling back to a documented
`len(text) / 4` chars-per-token estimate flagged via an `is_estimated` column. The
lesson, not just for this project: **log `usage` from the first API-calling line of
code**, because a cache that stores only the answer makes the question "what did that
already-answered call actually cost" unanswerable after the fact, not just inconvenient.

**A cache hit's ~0ms latency would have made every G4 timing number fiction.** T5.5's
end-to-end latency harness needs genuine wall-clock numbers, and the disk cache exists
specifically to make repeat calls near-instant — exactly the opposite of what a latency
measurement wants. `complete_with_usage` gained a `bypass_cache` flag that skips the
cache on *both* the read and the write side when timing for real: not reading avoids the
fake-latency problem; not writing avoids a live temperature>0 sample silently overwriting
the canonical T4b.2 transcript with different (but not wrong) text. The live 45-call run
(15 stratified questions × 3 arms) took ~5.5 real minutes end-to-end on Groq's free tier
— slow enough to notice, but no rate-limit errors surfaced at this volume; a much larger
sweep (e.g. a full 60-question run per arm, or repeats for a tighter CI) would be worth
budgeting extra wall-clock for and watching for 429s, per PLAN.md §1.4 bottleneck #3.

---

### 14.6 — Multimodal pilot (M6/E7): provider switches, real coverage, and an honest null result

M6 (the multimodal grounding pilot, RQ4/G3) was the milestone with the most live-call
surprises of the project so far — three of its six tasks (T6.2, T6.4, T6.5) each hit a
real, undiscoverable-except-by-calling-it blocker. None of them were coding bugs in the
usual sense; all were the gap between what a docs page/catalog claims and what a live
API call actually does, the same class of lesson §14.4 already logged for Groq.

**Why Gemini exists in this codebase at all.** T6.2's live check of Groq's catalog
(§14.4's 2026-08-19 snapshot, 13 models) confirmed no model on it accepts image input —
`gpt-oss-{20b,120b}` and `qwen/qwen3.6-27b` are text-only, and the rest are TTS/ASR/
classifiers. PLAN.md §1.3 already named Google AI Studio as the project's second
free-tier provider, so `cragb.generate.gemini_client.GeminiClient` was added
(T6.2) behind the exact same `complete`/`complete_with_usage` shape as `GroqClient` —
additive, not a replacement; every Groq-backed M1–M5 result is untouched. This is the
reason `GOOGLE_API_KEY` is now in `.env.example`: nothing else in the project needs it,
only the vision judge.

**Model choice took three live attempts, not one.** `gemini-2.5-flash` (T6.2's first
choice, reasoned from Google's pricing docs) returned HTTP 404 "no longer available to
new users" on an actual call, despite still appearing in `ListModels` — a catalog listing
is not verification, exactly §14.4's Groq lesson recurring on a different provider.
`gemini-3.6-flash` (T6.2/T6.4, Google's own named replacement) worked correctly through
all of T6.4's validation, but T6.5's live batch run died on HTTP 429: its free tier caps
at **20 `generateContent` requests per day per project per model** (read directly from
the 429 error body's `QuotaFailure` detail — Google publishes no public per-model RPD
table, so this is only discoverable by exhausting it), nowhere near the ~50 calls a
25-pair pilot needs. `gemini-3.5-flash-lite` (final choice) has worked without a quota
failure since. **Separately**, `gemini-3.6-flash`'s hidden "thinking" tokens
(`usageMetadata.thoughtsTokenCount`, billed at the output rate, never in `text`)
silently truncated a real multimodal judge call mid-JSON at `max_tokens=300` — not the
clean empty-candidate case a smoke test at T6.2 had already handled, but a *different*
failure shape (a parseable-looking but cut-off JSON object) that slips past that
check entirely. Fixed by raising to 700, confirmed live. `gemini-3.5-flash-lite`
carries no `thoughtsTokenCount` field at all, removing this risk category for the model
actually in use.

**Amazon's CDN was healthy.** T6.1's real fetch over the 123 candidate URLs (first
image of every has-image doc in the retrieval pool of an image-target question) got
**122 `ok`, 1 `http_error`** (99.2%) — the one failure a dead
`images-na.ssl-images-amazon.com` legacy host, not a rate limit or a systemic problem.

**The real coverage funnel came in well under the naive per-question-image-coverage
estimate, and the drop happened at a different stage than expected.** T6.3's live run:

```
60 total questions → 49 image_target → 25 photo_in_context → 25 fetchable_bytes → 25 usable_pairs
```

All 24 dropped questions fell out at the *same* stage — RAG-small's actual retrieved
context (BM25, k=5, `configs/grounded_qa.yaml`) simply didn't happen to include an
image-bearing doc, even though the broader 20-doc pool used for CRAGB pooling often did.
This is a property of the pipeline as built (a narrow top-5 context has a much lower
chance of catching the ~13.5% image-bearing minority than pooling's top-20 union did),
not a fetch failure or a labelling gap — and it is exactly the risk PLAN.md §3 E7 named
in advance ("too few image-bearing questions → underpowered"), just materialising via
retrieval breadth rather than raw image scarcity.

**Order-swap did catch real position bias, not zero.** `order_agreement_rate` across the
25 judged pairs was **0.88** (22/25) — 3 pairs where the two photo orders disagreed with
each other, each correctly folded into `tie` rather than credited as a win either
direction (`cragb.multimodal.vision_judge.judge_pair`). Not a large fraction, but a
real, live-confirmed instance of the exact failure mode the control exists to prevent —
without it, up to 3 of the 12 raw single-order "wins" could have been position artifacts
rather than genuine preference.

**RQ4's headline is an honest null result, not a finding to spin.** n=25 usable pairs
(below the ~30 threshold flagged before any judge quota was spent) gave **win-rate 0.48,
95% CI [0.28, 0.68], p=0.655 (one-sided vs 0.5)** — the CI brackets the null, so this is
underpowered at this sample size, reported as such rather than fished for a
per-question-type subgroup that clears significance. The human spot-check (T6.6, n=15)
adds a second, independent reliability caveat: **Cohen's κ=0.20**, below the 0.4
usability bar. Reading the actual disagreements (not just the number) matters here: every
row where the human and the judge both said "A" agreed (7/7); nearly every disagreement
is the judge calling `tie` where the human saw a marginal winner, plus one case
(`fit_sizing_010`, kept as a worked example in `reports/multimodal_examples_v1.md`) where
the judge's reasoning (a worn watch shows real on-body fit; a boxed necklace shows none)
is arguably *more* defensible than the human's first instinct (favouring the necklace
because it's the actual reviewed product) — genuine reasonable disagreement on a
subjective task, at a very small sample, not an obviously broken judge. Both the win-rate
and the κ belong in the report exactly as measured; a photo-evidence result with n=25 and
a validated instrument would look different, and that difference is the honest scope of
what a pilot at this scale can claim.

### 14.7 — Fine-tuning data prep (M7): the 8 GB assumption was wrong, and the untuned baseline's real failure is over-abstention, not hallucination

**§10's 8 GB assumption corrected with a measured number.** PLAN.md §10 (written before this
project had touched a GPU) proposed a 3B-class model "fits 8 GB." The actual dev machine is an
**RTX 3050 Laptop, 4096 MiB VRAM** (`nvidia-smi`, confirmed at T7.7 build time). T7.7's measured
inference probe (`results/tables/ft_model_probe_v1.csv`) found both Qwen2.5-1.5B-Instruct
(1.2 GB peak) and Qwen2.5-3B-Instruct (2.1 GB peak) fit comfortably at 4-bit for *inference* —
the 8 GB-vs-4 GB gap turned out not to bind there. It does bind for **training**: T7.9's QLoRA
probe (`ft_qlora_probe_v1.csv`) shows every configuration that completed without an outright OOM
still reports peak VRAM above the card's physical 4,096 MB, meaning every one of them ran partly
via `bitsandbytes`/`accelerate`'s automatic CPU offload, not purely on-GPU — a second, separate
correction from the inference-only finding. Full reasoning and the chosen configuration
(Qwen2.5-1.5B, LoRA rank 16, seq 1,152) are in `reports/finetune_plan_v1.md` §§1–2.

**Two Python environments, cleanly separated.** `peft`, `bitsandbytes`, `accelerate`, `datasets`,
`trl` were installed into `C:\venv\cragb` only (§14.1's existing MAX_PATH-workaround venv), never
into the main conda env — matching M7.md fact #2's constraint. `bitsandbytes` installed and
loaded correctly on Windows without incident (the Windows-wheel risk M7.md flagged as a possible
finding did not materialise).

**The untuned baseline's real failure, found by decomposing one aggregate metric.** T7.8's
headline `abstention_accuracy` column reads as mediocre-but-unremarkable (48.3% CRAGB, 43.8%
probe) until split by *direction* — `results/tables/ft_datasheet_stats_v1.json`'s
`baseline_abstention_breakdown`, computed from the raw transcripts against each question's
actual gold label:

```
CRAGB (58 answerable / 2 gold-abstention):  false-abstention 31/58 = 53.5%   true-abstention recall 2/2 = 100%
Probe (24 answerable / 8 gold-abstention):  false-abstention 18/24 = 75.0%   true-abstention recall 8/8 = 100%
```

The untuned `Qwen2.5-3B-Instruct` local model **never once fails to abstain when it should**
(perfect recall on both slices) — its entire abstention failure is refusing to answer 53–75% of
questions its own retrieved context could actually answer. Combined with citation validity
already at 100% and faithfulness already at ~4.6–4.7/5 on the much smaller set of questions it
does answer, this is not the "hallucinates and drops citations" failure mode a first guess might
expect from a small local model — it is a **severely over-cautious refusal policy**. This
reshapes M8's target: the fine-tune's job is to make the model *less* conservative about
answering, not more careful about grounding (it is already careful). Full baseline table and the
go/no-go thresholds built from this finding are in `reports/finetune_plan_v1.md` §§4, 6.

**Consequence for abstention evaluation, connecting back to §14.2.** §14.2 already found CRAGB
retains only 2 genuine abstention questions after 9 of 11 authored negatives turned out
answerable — too small an n to support any threshold on its own, let alone detect a 53.5%-vs-
75.0% gap between two evaluation sets the way the probe set (deliberately built with 8 balanced
abstention examples, T7.6) did here. The probe set's existence is what let this finding surface
at all; a go/no-go rule written against CRAGB's n=2 alone would have reported "100% recall,
nothing to fix" and missed the dominant failure entirely.

---

### Open input needed from you
Your **real mid-progress and final deadlines** (§4 durations are on a placeholder 12-week / week-6 schedule). Give me the dates and I'll re-flow the milestones and tell you where to cut if time is tight (first cut: drop CRAGB to ~50 questions and defer the demo, never the pooling or the judge validation).
