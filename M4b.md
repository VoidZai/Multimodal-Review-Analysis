# Milestone 4b — Answer-quality evaluation + judge validation (E5, RQ0/RQ1)

**Status:** M1, M2, M3, M4a complete. This milestone covers PLAN.md §3 E5 — trustworthy
answer-quality measurement, then using it to compare closed-book vs RAG (**RQ0**) and
model sizes within one family (**RQ1**) — plus the judge-reliability table, RQ0/RQ1
tables, and judge-vs-human agreement figure required for the mid-progress report (§7).
Cost/latency (E6) and the multimodal pilot (E7) are separate milestones (M5, M6) and are
out of scope here even where a figure (the quality-vs-cost Pareto scatter) needs this
milestone's numbers as an input.

Tasks are mostly linear — each consumes the previous task's artifact — except T4b.3 and
T4b.4, which can be built in parallel (neither depends on the other; both only need
T4b.2's transcripts to be *run at scale*, not to be written).

Code is not written yet; this file is the task breakdown only.

---

## T4b.1 — Closed-book baseline: prompt + generation pipeline (RQ0 arm)

**Objective:** Build the no-retrieval baseline RQ0 needs to isolate what grounding is
actually worth. Same question, no retrieved review context. To keep the RQ0 comparison
controlled, this arm uses the **same generator model** as the RAG-small arm
(`openai/gpt-oss-20b`, per `configs/grounded_qa.yaml`) — the only thing that should
differ between the two arms is presence of context, not which model answered. The prompt
reuses the existing `ABSTENTION_TEXT` convention (`cragb.bench.reference_answers`) so
abstention accuracy stays comparable across arms, even though there is nothing to cite
here.

**Artifact:**
- `src/cragb/generate/prompts/closed_book_qa_v1.md` (instructs: answer from your own
  knowledge only; if you cannot know this without seeing actual customer reviews, reply
  with the exact `ABSTENTION_TEXT` phrase; no citation format, since there is no context
  to cite)
- `configs/closed_book_qa.yaml` (paths + provider block, model pinned to match RAG-small)
- `src/cragb/generate/closed_book_qa.py` + `tests/test_closed_book_qa.py` —
  `generate_closed_book_answer(question_id, question, template, chat_fn) ->
  ClosedBookTranscript(answer_text, abstained)`, `run_closed_book_qa(questions, ...)`,
  `write_transcripts_jsonl`/`load_transcripts_jsonl` (mirrors `grounded_qa.py`'s shape
  minus context/citations)

**Expected outputs:** module importable; one real completion for a hand-picked question,
confirmed to *not* fabricate a specific product fact it can't know.

**Python packages:** none new.

**Validation checks:** unit tests — abstention detected via **containment, not equality**
(same class of bug as PLAN.md §14.3 — test it explicitly again here since it's a new
parser, not a reuse of `grounded_qa.parse_completion`); transcript round-trips through
JSONL.

**How to verify it worked:** `pytest tests/test_closed_book_qa.py -q`; one real API call,
inspect the new entry in `results/cache/api`.

**Git commit message:** `feat(eval): closed-book baseline prompt + generation pipeline (RQ0 arm)`

**Mid-progress report:** Indirect (infrastructure for T4b.2).

---

## T4b.2 — Full-benchmark generation runs: closed-book + RAG-small + RAG-large

**Objective:** Produce complete, all-60-question transcripts for the three arms RQ0/RQ1
need. M4a's grounded run only covered a ~10-12 question pilot slice; there is no
closed-book data yet, and no second (larger) model in the same family. Before locking the
RQ1 "large" config, **check Groq's current model catalog** (`GET /openai/v1/models`,
same check `configs/grounded_qa.yaml`'s comment already records doing once — Groq
deprecated `llama-3.1-8b-instant` between M2 and M4a) and confirm `openai/gpt-oss-120b`
is live — same family as the already-validated `gpt-oss-20b`, so RQ1 stays de-confounded
(PLAN.md risk B). While checking the catalog, also earmark a **third, distinct model**
(different family from `gpt-oss`, e.g. a current Llama model) to use as the judge in
T4b.4 — using `gpt-oss-20b` or `-120b` as judge would let a model score its own output,
a self-preference risk PLAN.md §9 flags by name, and it would specifically undermine the
RQ1 comparison (small vs large) if the large model judged itself.

**Artifact:**
- `configs/grounded_qa_large.yaml` (copy of `grounded_qa.yaml`, model swapped to the
  verified larger same-family model)
- `src/cragb/eval/run_answer_generation.py` — thin batch-driver CLI, `--arm
  {closed_book,rag_small,rag_large}`, all 60 CRAGB questions, reuses
  `run_closed_book_qa`/`run_grounded_qa` unchanged
- `results/tables/answer_gen_closed_book_v1.jsonl`,
  `results/tables/answer_gen_rag_small_v1.jsonl`,
  `results/tables/answer_gen_rag_large_v1.jsonl` (60 rows each)

**Expected outputs:** 180 total transcripts across the three files, one row per
(question, arm).

**Python packages:** none new.

**Validation checks:** row count == 60 per file; every row has non-null `answer_text`;
re-running the script produces zero new cache entries (proves caching, protects
free-tier quota per PLAN.md §1.4 bottleneck #3).

**How to verify it worked:** row-count check on all three files; spot-check 3 random rows
per arm by hand. Note: wall-clock for the live run may exceed the 30-90 min engineering
estimate if Groq rate-limits kick in (180 calls across three configs) — the disk cache
makes it safe to stop and resume, so budget the *active* work at ~60-90 min and let the
run itself continue/retry in the background if needed.

**Git commit message:** `feat(eval): full-benchmark answer generation — closed-book, RAG-small, RAG-large`

**Mid-progress report:** Yes — raw material behind the RQ0/RQ1 tables (§7).

---

## T4b.3 — Answer-quality similarity metric (embedding-cosine to reference)

**Objective:** A reference-based similarity score per answer. Uses the dense-embedding
model already installed and validated for retrieval (`BAAI/bge-small-en-v1.5`, via the
`C:\venv\cragb` short-path venv, PLAN.md §14.1) rather than adding `bert-score` as a new
dependency — one fewer Windows MAX_PATH risk, and it reuses infrastructure already proven
to work on this machine.

**Artifact:** `src/cragb/eval/metrics_answer.py` + `tests/test_metrics_answer.py` —
`embedding_similarity(answer_text, reference_text, model) -> float` (cosine of two
`SentenceTransformer.encode()` calls), `score_arm(transcripts, references) ->
pd.DataFrame` (`question_id`, `similarity`).

**Expected outputs:** importable module; a similarity score in `[-1, 1]` for every
transcript/reference pair.

**Python packages:** none new (`sentence-transformers` already in `environment.yml`,
venv-only).

**Validation checks:** unit tests — identical strings → similarity ≈ 1.0; unrelated
strings → low similarity; two abstention texts (candidate abstained, reference is the
canonical `ABSTENTION_TEXT`) → similarity ≈ 1.0, confirming abstention-correctness and
similarity agree on the easy case.

**How to verify it worked:** `C:\venv\cragb\Scripts\python.exe -m pytest
tests/test_metrics_answer.py -q`; run on 3 real transcript/reference pairs and confirm
the ranking matches intuition (a genuine paraphrase scores higher than an off-topic
answer).

**Git commit message:** `feat(eval): embedding-cosine answer-similarity metric`

**Mid-progress report:** Indirect (feeds the RQ0/RQ1 tables, T4b.7).

---

## T4b.4 — Rubric AI-judge: prompt + JSON-scoring module

**Objective:** The four-criterion rubric judge PLAN.md §8/§9 specifies — **correctness,
faithfulness/grounding, completeness, conciseness**, each 1-5 — forced into
machine-parseable JSON (reduces judge variance, Lec 3's task/criteria/scale structure).
Design constraints to bake into the prompt, not left implicit:
- The judge is given the question, the context shown to the generator (or an explicit
  "no context available (closed-book)" marker), the candidate answer, and the CRAGB
  reference answer — but **never which system/model produced the candidate** (§9: avoid
  self-preference bias).
- Uses the **third, distinct model** earmarked in T4b.2 — never `gpt-oss-20b`/`-120b` —
  so the judge is never scoring its own family's output.
- Explicit rule for the abstention case: if the reference is itself the abstention text
  and the candidate correctly abstains, that is full marks on correctness — not "no
  answer to grade."
- Same `max_tokens`-vs-hidden-reasoning gotcha `configs/grounded_qa.yaml`'s comment
  already documents for reasoning models on Groq applies here too if the judge model is a
  reasoning model — size `max_tokens` generously and verify against a real call before
  batch-running T4b.5.

**Artifact:** `src/cragb/generate/prompts/answer_judge_v1.md`, `configs/judge.yaml`,
`src/cragb/eval/judge.py` + `tests/test_judge.py` — `build_judge_prompt(question,
context_text_or_none, candidate_answer, reference_answer) -> str`,
`parse_judge_response(raw_json) -> JudgeScore(correctness, faithfulness, completeness,
conciseness, rationale)`, `score_answer(..., chat_fn) -> JudgeScore`.

**Expected outputs:** importable module; one real scored example.

**Python packages:** none new.

**Validation checks:** unit tests — well-formed JSON parses to four in-range (1-5) ints +
a rationale string; malformed/truncated JSON raises a clear error rather than silently
returning a placeholder score (mirrors `citation_validity.score_transcripts`'s
fail-loudly-on-missing-data convention); a regression test asserting the rendered prompt
never contains a model/system name or the words "closed-book"/"RAG".

**How to verify it worked:** `pytest tests/test_judge.py -q`; one real API call scoring a
hand-picked clearly-good answer against a clearly-bad one, confirm the good one scores
higher on every criterion.

**Git commit message:** `feat(eval): rubric answer-quality judge — prompt + JSON scorer`

**Mid-progress report:** **Yes** — the judge prompt is a required §7 versioned-prompt
artifact.

---

## T4b.5 — Batch judge run over all three arms

**Objective:** Score every generated answer (3 arms × 60 questions = 180 candidates)
against its CRAGB reference with T4b.4's judge — the per-question score table the RQ0/RQ1
tables are built from.

**Artifact:** `src/cragb/eval/run_judge_eval.py`, `results/tables/judge_scores_v1.csv`
(`arm`, `question_id`, `correctness`, `faithfulness`, `completeness`, `conciseness`,
`rationale`).

**Expected outputs:** 180 scored rows.

**Python packages:** none new.

**Validation checks:** row count == 180; no null scores; every score in `[1, 5]`;
re-running the script is a full cache-hit (no new API spend).

**How to verify it worked:** shape/null check on the CSV; read 3 rationale strings and
confirm they plausibly justify their scores against the actual answer text.

**Git commit message:** `feat(eval): batch judge run over closed-book/RAG-small/RAG-large`

**Mid-progress report:** Yes — raw scores behind the RQ0/RQ1 tables.

---

## T4b.6 — Judge validation: human-scoring worksheet + Cohen's κ

**Objective:** PLAN.md risk E — don't trust the judge at scale until it's measured
against a human. Hand-score a 30-50 answer subset on the identical rubric and compute
judge-vs-human agreement, mirroring T2.7's worksheet-based human-labeling pattern
(`benchmark/relevance_worksheet_v1.md`).

**Artifact:**
- `configs/judge_validation.yaml` (sample size, seed, paths — mirrors `labeling.yaml`)
- `src/cragb/eval/judge_validation.py` + `tests/test_judge_validation.py` — a seeded
  stratified sample (across the 3 arms, not just easy cases) of T4b.5's 180 scored rows,
  rendered as `results/worksheets/judge_validation_worksheet_v1.md` (question, candidate
  answer, reference answer, **judge's own score hidden**, blank columns for a human
  score); a scorer that reads the filled-in worksheet back and computes per-criterion
  **Cohen's κ (quadratic-weighted, since the rubric is ordinal 1-5)** and Spearman
  correlation between judge and human scores.

**Expected outputs:** worksheet with 30-50 rows ready for hand-scoring; once filled,
`results/tables/judge_validation_v1.csv` (per-criterion κ, correlation, n).

**Python packages:** **`scikit-learn`** (new — `cohen_kappa_score(weights="quadratic")`;
add to `environment.yml`'s pip block).

**Validation checks:** unit tests on synthetic data — identical judge/human scores → κ ≈
1.0; independently-random scores → κ ≈ 0; the scorer refuses to run (raises) if any
worksheet row is left unscored, rather than silently dropping it.

**How to verify it worked:** after hand-scoring the worksheet, run the scorer and sanity
check the printed κ against the % of rows where judge and human differ by ≤1 point. Note
explicitly (as PLAN.md §1.4 bottleneck #2 already does for relevance labeling): the
*coding* here is a 60-90 min task, but hand-scoring 30-50 answers on a 4-criterion rubric
is separate manual labor — budget it as extra time outside the engineering estimate and
log time spent, same as T2.7.

**Git commit message:** `feat(eval): judge validation worksheet + Cohen's kappa`

**Mid-progress report:** **Yes** — the judge-reliability table §7 explicitly requires,
and the §11 point-3 "senior move" (measure the judge before trusting it).

---

## T4b.7 — RQ0 + RQ1 comparison tables (bootstrap CIs + significance)

**Objective:** Turn T4b.3's similarity scores and T4b.5's judge scores into the two
headline result tables — **RQ0**: RAG-small vs closed-book; **RQ1**: RAG-small vs
RAG-large — reusing `cragb.eval.bootstrap` (`bootstrap_ci`, `paired_significance`)
**unchanged**, exactly as T3.6 already validated the same pattern for RQ2. Pairing is by
`question_id` across arms, same reasoning `bootstrap.py`'s docstring already gives for
why the pairing carries real statistical information here too.

**Artifact:** `src/cragb/eval/run_answer_quality_eval.py` + `tests/`,
`results/tables/rq0_answer_quality_v1.csv`, `results/tables/rq1_answer_quality_v1.csv`.
Each table: per-arm mean similarity + mean per-criterion judge score with bootstrap 95%
CI, plus a paired Wilcoxon p-value between the two arms being compared.

**Expected outputs:** two small CSVs, one row per arm plus a significance column.

**Python packages:** none new.

**Validation checks:** every CI satisfies `lo <= mean <= hi`; every p-value in `[0, 1]`;
unit test on constructed scores confirming the RQ0 table correctly flags a
large/obvious difference as significant.

**How to verify it worked:** run the script; hand-check the direction of the RQ0 result
against 3 real transcripts (RAG should beat closed-book on faithfulness at minimum, since
closed-book has nothing to cite by construction).

**Git commit message:** `feat(eval): RQ0 (RAG vs closed-book) + RQ1 (small vs large) answer-quality tables`

**Mid-progress report:** **Yes** — the two headline tables §7 explicitly lists.

---

## T4b.8 — Judge-agreement figure + notebook + appendix examples

**Objective:** Package T4b.6's validation and T4b.7's tables into report-ready form: the
notebook §6 names (`07_answer_eval_judge.ipynb`) and the judge-vs-human agreement figure
§7 lists.

**Artifact:**
- `reports/figures/judge_human_agreement_v1.png` (judge score vs human score per
  criterion, e.g. a jittered scatter with the κ annotated)
- `notebooks/07_answer_eval_judge.ipynb` (imports from `src/` only, no logic, produces the
  figure + prints the RQ0/RQ1/judge-validation tables)
- `reports/answer_quality_examples_v1.md` (3-5 qualitative examples: a clear RAG-wins
  case, a closed-book hallucination, a judge/human disagreement — seeds the §7 error
  taxonomy: grounding failure, hallucinated fact, over-abstain)

**Expected outputs:** figure file; notebook that runs top-to-bottom; a short markdown
appendix.

**Python packages:** none new (`matplotlib` already present).

**Validation checks:** notebook executes with zero errors (Run All); figure renders with
legible axes/labels; every example in the appendix cites its actual `question_id`.

**How to verify it worked:** open the PNG; run the notebook fresh from a clean kernel.

**Git commit message:** `feat(eval): judge-agreement figure + answer-eval notebook + appendix examples (M4b)`

**Mid-progress report:** **Yes** — literal §7 figure, §6 notebook, and appendix
deliverables.

---

## Notes

- Sequencing: T4b.1 → T4b.2 → {T4b.3, T4b.4 in parallel} → T4b.5 → T4b.6 → T4b.7 → T4b.8.
  T4b.3 and T4b.4 don't depend on each other's code, only on T4b.2's transcripts existing
  before either is run at full scale.
- Total estimate: roughly 8-11 hours of engineering time against PLAN.md's 1.5-week
  placeholder for M4b, **plus** separate hand-scoring time for T4b.6 (not counted in the
  30-90 min task estimates, same treatment as T2.7's relevance labeling).
- New dependency this milestone: `scikit-learn` (T4b.6, Cohen's κ). Everything else reuses
  what M1-M4a already installed.
- `embedding_similarity` (T4b.3) needs the `C:\venv\cragb` short-path venv (PLAN.md
  §14.1); every other task in this milestone runs fine in the main environment.
- The RQ1 "large" model and the judge model both depend on a live check of Groq's model
  catalog at T4b.2 build time — do not hardcode `gpt-oss-120b` (or any specific judge
  model) into this plan as final; verify availability the same way `configs/
  grounded_qa.yaml`'s comment already records doing once before.
- Cost/latency (E6, M5) and the multimodal pilot (E7, M6) are explicitly out of scope
  here, including the quality-vs-cost Pareto scatter — this milestone produces the
  "quality" half of that figure; M5 adds the "cost" half.
