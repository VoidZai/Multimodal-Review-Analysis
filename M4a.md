# Milestone 4a — Grounded-QA prompt + citations (E4, Component 1 core)

**Status:** M1, M2, M3 complete. This milestone covers PLAN.md §3 E4 — a prompt that answers
strictly from retrieved chunks, cites evidence, and abstains when unsupported — plus the
5 worked transcripts and citation-validity rate required for the mid-progress report (§7).

Tasks are sequenced linearly: each one consumes the previous task's artifact.

---

## T4a.1 — Design & version the grounded-QA prompt

**Objective:** Write the actual prompt that enforces strict grounding, the `[R#]` /
`[photo of R#]` citation scheme, and the exact abstention phrase — this *is* the E4
deliverable, not scaffolding for it.

**Artifact:**
- `src/cragb/generate/prompts/grounded_qa_v1.md` (template with `{question}` /
  `{context_block}` placeholders)
- `configs/grounded_qa.yaml` (provider block mirroring `configs/generate.yaml`, low
  temperature for grounding strictness, paths)

**Expected outputs:** prompt file with role instruction, context-block schema description,
citation format spec, hard "answer only from context" rule, and the literal abstention
string.

**Python packages:** none new.

**Validation checks:** placeholders present in the template; `configs/grounded_qa.yaml`
round-trips through `yaml.safe_load`.

**How to verify it worked:** read the file back; manually check it against E4's checklist
(grounding, citations, abstention).

**Git commit message:** `feat(prompt): grounded-QA prompt v1 + config`

**Mid-progress report:** **Yes** — required as a versioned prompt example in §7.

---

## T4a.2 — Context-block builder

**Objective:** Turn a question + retrieved chunks into the structured, ID-labelled context
block the prompt expects.

**Artifact:** `src/cragb/generate/context_builder.py` + `tests/test_context_builder.py`.
Reuses `cragb.eval.run_retrieval_eval.build_retriever` (BM25 — no GPU/venv dependency) over
`corpus_v1`.

**Expected outputs:** `build_context(question, corpus, retriever, k) -> ContextBlock(text,
id_map, photo_flags)`; smoke print for one real CRAGB question.

**Python packages:** none new.

**Validation checks:** unit tests — R# labels assigned in rank order, `id_map` keys match
tokens in the text, `photo_flags` reflects `has_image`, k-larger-than-corpus edge case.

**How to verify it worked:** `pytest tests/test_context_builder.py -q`.

**Git commit message:** `feat(prompt): context-block builder for grounded QA`

**Mid-progress report:** Indirect (infrastructure, not itself an appendix artifact).

---

## T4a.3 — Generation pipeline + citation parsing

**Objective:** Render the prompt, call the (already-cached) `GroqClient`, and parse the
completion into a structured answer.

**Artifact:** `src/cragb/generate/grounded_qa.py` + `tests/test_grounded_qa.py`.

**Expected outputs:** `generate_answer(question_id, context, config) ->
GroundedAnswer(answer_text, cited_rs, abstained)`.

**Python packages:** none new.

**Validation checks:** parser tests (citation extraction, abstention via **containment not
equality** — same class of bug fixed in PLAN.md §14.3, so test that explicitly), malformed
citation `[R99]` handling; one real API call gated behind `GROQ_API_KEY` presence, mirroring
existing test patterns.

**How to verify it worked:** `pytest tests/test_grounded_qa.py -q`; one manual real run,
inspect the new file in `results/cache/api`.

**Git commit message:** `feat(prompt): grounded-QA generation pipeline + citation parsing`

**Mid-progress report:** Indirect (building block for T4a.5's transcripts).

---

## T4a.4 — Citation-validity & abstention-correctness scorer

**Objective:** Compute the three metrics E4 requires: format compliance, citation validity,
abstention correctness.

**Artifact:** `src/cragb/eval/citation_validity.py` + `tests/test_citation_validity.py`.

**Expected outputs:** `score_transcript(...)` (per-answer booleans/floats) +
`summarize(transcripts) -> DataFrame`.

**Python packages:** none new.

**Validation checks:** constructed cases — all-valid, fabricated citation,
abstention-with-citation (self-contradiction), false-negative abstention.

**How to verify it worked:** `pytest tests/test_citation_validity.py -q`; hand-check
aggregate numbers against 2–3 real transcripts.

**Git commit message:** `feat(eval): citation-validity + abstention-correctness checker`

**Mid-progress report:** **Yes** — produces the citation-validity rate and abstention
accuracy numbers §7 lists.

---

## T4a.5 — Pilot run: transcripts + validity table

**Objective:** Run T4a.2–T4a.4 end-to-end over a deliberately chosen ~10–12 question slice
(spread across taxonomy types, both real `is_abstention=True` cases, and 1–2 of the
"surprisingly answerable negative" questions from §14.2 as failure-mode material).

**Artifact:** `results/tables/grounded_qa_transcripts_v1.jsonl`,
`results/tables/grounded_qa_validity_v1.csv`.

**Expected outputs:** ≥10 full transcripts (question, context, raw completion, parsed
answer, scores); one aggregate + per-question validity table.

**Python packages:** none new.

**Validation checks:** row count matches selected ids; every answer non-null; validity
numbers match manual spot-checks.

**How to verify it worked:** re-run the script — confirm no new cache files appear (proves
caching) and outputs are stable.

**Git commit message:** `feat(prompt): grounded-QA pilot run — transcripts + citation-validity table`

**Mid-progress report:** **Yes** — the required 5+ transcripts and citation-validity rate.

---

## T4a.6 — Appendix transcripts + notebook assembly

**Objective:** Hand-pick and format 5 transcripts (2–3 clean grounded answers, 1 abstention,
1 failure mode) for the report; assemble the thin notebook.

**Artifact:** `reports/grounded_qa_transcripts_v1.md`, `notebooks/06_grounded_qa.ipynb`.

**Expected outputs:** report-ready markdown; notebook that imports from `src/` only (no
logic) and runs top-to-bottom.

**Python packages:** none new (existing `jupyter`/`nbformat`).

**Validation checks:** citations render cleanly in markdown; notebook executes with zero
errors.

**How to verify it worked:** Run All in the notebook; visually check the `.md` file.

**Git commit message:** `feat(prompt): grounded-QA transcripts appendix + notebook (M4a)`

**Mid-progress report:** **Yes** — this is the literal §7 appendix artifact ("5 grounded-QA
transcripts with citations") plus the §6 notebook deliverable.

---

## Notes

- Sequencing is linear (each task consumes the previous one's artifact).
- Total estimate: roughly 5–7 hours against PLAN.md's 0.5-week placeholder for M4a.
- No fine-tuning, no dense-retrieval/venv dependency required for this milestone — BM25
  (already validated in M3, no GPU needed) is sufficient for context retrieval here.
- Code is not written yet; this file is the task breakdown only.
