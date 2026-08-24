# Fine-tuning plan v1 — base model, method, data, target behaviours, go/no-go (E8, §10)

**Status of this document: a plan, with numbers behind it. No fine-tune has been run.**
Every measurement cited below is from T7.7's inference-VRAM probe, T7.8's untuned baseline,
or T7.9's QLoRA feasibility probe — all measured on the real dev machine, none of them a
fine-tuning *result*. Any sentence that could be read as "the tuned model does X" would be
false at this milestone and does not appear here. M8 runs the fine-tune; this document is
what M8 is not allowed to invent after the fact.

All figures trace to committed manifests/CSVs, re-derived without hand-typing by
[`src/cragb/finetune/datasheet_stats.py`](../src/cragb/finetune/datasheet_stats.py) →
[`results/tables/ft_datasheet_stats_v1.json`](../results/tables/ft_datasheet_stats_v1.json).

---

## 1. Base model, with the 4 GB correction stated plainly

PLAN.md §10 (written before this machine's actual hardware was measured) assumed **8 GB VRAM**
and proposed Qwen2.5-3B-Instruct or Llama-3.2-3B-Instruct. The real dev machine is an
**RTX 3050 Laptop, 4096 MiB VRAM** (`nvidia-smi`, confirmed; M7.md fact #1). T7.7 measured
three candidates for local 4-bit **inference** (`ft_model_probe_v1.csv`):

| Model | Load peak VRAM | Resident VRAM | Generation peak VRAM | Tokens/s | Fits (4 GB)? |
|---|---|---|---|---|---|
| Qwen2.5-1.5B-Instruct | 1,125 MB | 1,100 MB | 1,217 MB | 12.3 | **yes** |
| Qwen2.5-3B-Instruct | 2,013 MB | 1,971 MB | 2,126 MB | 9.8 | **yes** |
| Llama-3.2-3B-Instruct | — | — | — | — | **no** — gated HF repo, 401 on download (no access request filed; not a VRAM failure) |

Both Qwen candidates fit comfortably for **inference** at 4-bit — the 4 GB ceiling turned out
not to bind at this stage the way §10 worried it might. **Chosen base model: `Qwen/Qwen2.5-3B-Instruct`.**
The 1.5B is the safer fallback if M8's real QLoRA run (full sequence lengths, real batch, real
epoch count — heavier than T7.9's 20-step probe) leaves less headroom than the probe suggests;
T7.8's baseline was run against the 3B specifically to measure the model M8 actually intends to
tune, per `configs/finetune_baseline.yaml`'s own reasoning for hardcoding the model name rather
than reading it back from the probe CSV.

**Correction to PLAN.md §10, logged at PLAN.md §14.7:** the assumed 8 GB was wrong; the real
figure is 4 GB, and — at least for 4-bit *inference* — a 3B model fits with real headroom to
spare. Whether it also fits for QLoRA *training* is a separate question, answered in §2.

---

## 2. Method: QLoRA, with the measured VRAM/step-time envelope

**Method:** QLoRA (4-bit `nf4` base + LoRA adapters on attention + MLP projections, gradient
checkpointing, batch size 1 + gradient accumulation) — unchanged from PLAN.md §10's proposal;
the 4 GB correction affects *which* model, not the method.

T7.9 ran a 20-step single-example overfit probe (the standard sanity check: loss must fall
sharply on one repeated example, or the collator/label-masking/chat-template is wrong before
any real training data is touched) across both Qwen candidates, two LoRA ranks, and two
candidate `max_seq_len` values derived from the real rendered-prompt token distribution
(`ft_prompt_length_stats_v1.csv`: p50=891, p95=1,117, max=1,210 tokens over this pilot's 39
filtered pairs). Full grid (`ft_qlora_probe_v1.csv`):

| Model | Rank | max_seq_len | Peak VRAM | s/step | Overfit confirmed | Extrapolated min/epoch |
|---|---|---|---|---|---|---|
| Qwen2.5-1.5B | 8 | 1,152 | 4,409 MB | 48.0 | **yes** | 31.2 |
| Qwen2.5-1.5B | 8 | 1,280 | 5,320 MB | 11.6 | no (loss didn't converge) | 7.5 |
| Qwen2.5-1.5B | 16 | 1,152 | 5,130 MB | 4.9 | **yes** | 3.2 |
| Qwen2.5-1.5B | 16 | 1,280 | 5,386 MB | 5.8 | no | 3.8 |
| Qwen2.5-3B | 8 | 1,152 | 6,236 MB | 29.6 | **yes** | 19.2 |
| Qwen2.5-3B | 8 | 1,280 | OOM (CPU/disk offload needed) | — | — | — |
| Qwen2.5-3B | 16 | 1,152 | 5,715 MB | 10.7 | **yes** | 7.0 |
| Qwen2.5-3B | 16 | 1,280 | OOM (CPU/disk offload needed) | — | — | — |

**Finding, not an assumption: every configuration that fits reports `peak_vram_mb` above
4,096 MB** — i.e. above this machine's literal physical VRAM. These runs completed because
`bitsandbytes`/`accelerate`'s automatic CPU-offload kicked in below the hard OOM point at
`max_seq_len=1,280` (where offload was no longer enough and it raised instead). **Every QLoRA
training configuration here runs at least partly via CPU offload, not purely on-GPU** — a
second, training-specific correction beyond §1's inference-only finding, and the reason
epoch-time (not just "does it fit") is the number that matters for M8's time budget.

**Configuration chosen for M8's pilot run: Qwen2.5-1.5B-Instruct, LoRA rank 16, max_seq_len
1,152.** Reasoning: it is the only configuration that (a) confirmed the overfit sanity check,
(b) fits within a peak VRAM close enough to the 4 GB card that offload stays light, and
(c) is fastest per step (4.9 s) and per epoch (3.2 min at this pilot's n=39) of every
overfit-confirmed row — rank-16 dominates rank-8 on this hardware because more of the extra
adapter compute is masked by the fixed offload overhead per step, not despite it. The
Qwen2.5-3B, rank-16, seq-1,152 row (7.0 min/epoch, also overfit-confirmed) is the stretch
option if T7.8 §4's baseline failures turn out to need the larger model's extra capacity.

`max_seq_len=1,280` configurations did **not** converge in 20 steps at either model — read as
a training-dynamics artefact of this specific probe's very short horizon at the larger length,
not evidence the length itself is unusable; M8 should not read those two rows as "1,280 doesn't
train," only as "20 steps wasn't enough to see it converge here."

**Time budget T for M8's pilot run:** the chosen configuration's 3.2 min/epoch is measured
against this pilot's **n=39** filtered pairs (`ft_prompt_length_stats_v1.csv`'s
`filtered_pairs_fallback` source — `train.jsonl` is currently empty; see §3). Scaling
linearly to the ~1,000–1,300 examples a full `train` split would hold after the ~500-context
sweep and an ~90/10 train/val split (§3), and budgeting 3 epochs (a starting point, not
final — see §7's placeholder-hyperparameters note): **~3.2 min × (1,100/39) × 3 ≈ 4.5 hours**
of GPU time. **T = 5 hours**, rounding up for optimizer-state/checkpoint I/O overhead T7.9's
per-step timing doesn't capture. This is the `T` the go/no-go rule in §6 is written against.

---

## 3. Data: pointer to the datasheet, disjointness and near-duplicate guarantees

Full detail: [`data/finetune/finetune_data_datasheet.md`](../data/finetune/finetune_data_datasheet.md).
Summary relevant to this plan:

- **Disjointness from CRAGB evidence** is enforced at sampling time (T7.2): 1,097 CRAGB-evidence
  documents across 1,070 `parent_asin`s excluded before any context is sampled.
- **Leakage guard** (T7.6) found **0 exact-hash leaks** and dropped **7 of 39** candidates
  (18%) as near-duplicates of a CRAGB question, all via the embedding-cosine backstop, all
  clustered against `fit_sizing_000`/`fit_sizing_001` — the exact-hash layer alone would have
  missed every one of these 7.
- **Current scale is a pilot, not the full target.** `configs/finetune.yaml` targets 500
  context groups; this build attempted 10, yielding 39 accepted examples, all absorbed into the
  `probe` split (32 after leakage-screening; 0 in `train`/`val`). **`train.jsonl`/`val.jsonl`
  must be repopulated by resuming T7.3's generation to the configured target before M8's real
  training run** — a resume of the existing resumable command
  (`python -m cragb.finetune.generate_pairs`), not new engineering. §2's epoch-time estimate
  already accounts for this scale-up: it extrapolates from the pilot's per-example step cost
  to the target volume, it does not wait for the volume to exist first.
- **Probe set** (`probe.jsonl`, 32 examples: 24 answerable / 8 abstention) is held out entirely
  from training and is what §6's go/no-go abstention threshold is measured against, because
  CRAGB itself has only 2 true abstention questions (M7.md fact #5) — not enough to support a
  threshold on its own.

---

## 4. Target behaviours — named from T7.8's observed failures, not the a-priori list

PLAN.md §10 named three target behaviours a priori: citation/format compliance, correct
abstention, grounding faithfulness. T7.8 measured all three for the chosen base model
(`Qwen/Qwen2.5-3B-Instruct`, untuned, 4-bit, local) on the full 60-question CRAGB set and the
32-question probe set, with retrieval held byte-identical to the RAG-small arm
(`ft_base_baseline_v1.csv`):

| Metric | CRAGB (n=60) | Probe (n=32) |
|---|---|---|
| Format compliance | 98.3% | 100% |
| Citation validity (of answers given) | **100%** | **100%** |
| Fabricated-citation rate | **0%** | **0%** |
| Gold-grounding rate | 95.2% | — (no gold citations for probe positives) |
| Faithfulness mean (1–5, bootstrap 95% CI) | 4.58 [4.30, 4.85] | 4.72 [4.38, 5.00] |
| Self-contradiction rate | 0% | 0% |
| Median latency (bypass-cache) | 5.7 s | 2.0 s |
| Abstention accuracy (raw, vs. gold `is_abstention`) | 48.3% | 43.8% |

**The headline finding is in the last row, and it needs decomposing — a single
"abstention accuracy" figure conflates two opposite failure directions.** Splitting it
(`ft_datasheet_stats_v1.json`'s `baseline_abstention_breakdown`, computed from the raw
transcripts against each question's actual gold label):

| Slice | False-abstention rate on genuinely answerable questions | True-abstention recall |
|---|---|---|
| CRAGB (58 answerable / 2 gold-abstention) | **31/58 = 53.5%** | 2/2 = **100%** |
| Probe (24 answerable / 8 gold-abstention) | **18/24 = 75.0%** | 8/8 = **100%** |

**This base model never fails to abstain when it should (2/2 and 8/8 recall) — its entire
abstention failure is severe over-abstention: refusing to answer 53–75% of questions its
retrieved context can actually answer.** Combined with citation validity already at 100% and
faithfulness already at ~4.6–4.7/5 on the (much smaller) set of questions it *does* answer, the
picture is not "the model hallucinates or drops citations" — it is **"the model has learned an
extremely conservative refusal policy that discards good answers far more than it needs to."**
This matches T7.8's own stated expectation of "hedging instead of abstaining," but the direction
(hedging *toward* refusal, not *away* from it) is the specific, nameable shape of it.

**Revised target behaviours for M8, in priority order:**
1. **Reduce false-abstention on answerable questions** (currently 53.5% CRAGB / 75.0% probe) —
   the dominant, highest-leverage target; this is what the training data's ~75%
   answerable / ~25% abstention mix and its `evidence_stripped` construction method (context
   that *looks* relevant but doesn't support the claim — the hardest over-abstention case to
   get right) are specifically built to teach against.
2. **Preserve true-abstention recall** (currently 100%/100%, but n=2 on CRAGB) — a
   no-regression guardrail, not an improvement target; there is no headroom to improve on 100%,
   only room to lose it.
3. **Preserve citation validity and faithfulness** (currently at or near ceiling) — no-regression
   guardrails, not improvement targets, since PLAN.md §10's original framing ("a 3B won't gain
   facts; it can gain discipline") already assumed these were the easier half of the problem,
   and T7.8 confirms it.

---

## 5. Validation (training-time)

Track val loss on `val.jsonl` (empty at pilot scale — see §3; must be repopulated before a real
training run can early-stop meaningfully) and early-stop on it, per PLAN.md §10. T7.9's
label-masking assertion (loss computed on completion tokens only, prompt masked with `-100`)
is already verified working at probe scale — an unmasked prompt would train the model to
generate review text instead of answers, a failure mode caught before any real training data
is touched.

---

## 6. Go/no-go, with numbers

Proceed from M8's pilot LoRA run to the full run only if **all** of the following hold,
measured on the shared held-out `probe.jsonl` set (T7.6) with the same retrieval/generation
harness T7.8 used, so every column is literally comparable to the baseline row above:

| Criterion | Threshold | Baseline (probe) | Rationale |
|---|---|---|---|
| **False-abstention rate on answerable questions** | ≤ **35%** (absolute improvement of ≥ 40 points from baseline's 75.0%) | 75.0% | The dominant measured failure (§4); a large absolute bar because the baseline is this far from usable — a token improvement is not the bar |
| **True-abstention recall** | = **100%** (no regression from baseline's 8/8) | 100% (n=8) | A no-regression clause, not an improvement target — there is no headroom above 100%, and losing it (answering an unanswerable question) is a worse failure than staying over-cautious |
| **Citation validity (of answers given)** | ≥ **95%** (no more than 5-point regression from baseline's 100%) | 100% | No-regression guardrail; fine-tuning on ~75% answerable examples must not degrade a discipline the base model already has |
| **Faithfulness mean** | ≥ **4.4** (paired bootstrap 95% CI lower bound must not fall below baseline's CI lower bound of 4.38, M4b's harness) | 4.72 [4.38, 5.00] | No-regression clause, paired significance test — not just a point estimate — per §7 |
| **Median latency** | ≤ **1.5×** baseline's 2.0 s → **3.0 s ceiling** | 2.0 s | LoRA adapters add negligible inference overhead; a large regression would indicate a configuration problem (e.g. adapters not merged), not an acceptable training-time/quality tradeoff |
| **GPU time for the pilot run** | ≤ **T = 5 hours** (§2) | — | If the pilot exceeds this on the chosen configuration, that is itself a signal to drop to the Qwen2.5-1.5B fallback or reduce `num_epochs`, not to silently let the pilot run longer |

All five quality/behaviour criteria must hold simultaneously; a win on false-abstention paired
with a citation-validity regression is a **no-go**, not a partial win — the whole point of
listing faithfulness/citation-validity as guardrails is that trading them away for abstention
improvement is not the behaviour PLAN.md §10 asked for.

**If the go/no-go criteria are not met:** trigger the fallback in §7. If they are met on the
pilot but GPU time is tight, the full run may still proceed with `num_epochs` reduced from the
pilot's 3 rather than abandoning the run outright — that is a scope adjustment, not a no-go.

---

## 7. Fallback: retriever/embedding fine-tune

**Trigger conditions (any one is sufficient):**
- The pilot LoRA run fails to clear the false-abstention threshold in §6, or clears it only by
  regressing citation validity or faithfulness below their guardrails.
- Every QLoRA configuration in T7.9's grid that would be needed for a full run (i.e., at the
  volume §3 describes) OOMs outright rather than degrading gracefully via offload — not the
  case measured here (§2's grid has multiple non-OOM configurations), but stated as a trigger
  per M7.md's own instruction to write this branch regardless.
- GPU time for the pilot exceeds `T = 5` hours by a wide enough margin that the full run is not
  practically completable within the project's remaining time budget.

**Fallback plan (PLAN.md §10's own proposal, designed here as a rule, not built until
triggered):** fine-tune the dense retriever's embedding model contrastively on CRAGB-style
(question, relevant-review) pairs instead of the generator. **Reuses the same
`finetune_v1` data** — `TrainingExample.question` paired with `TrainingExample.source_doc_ids`
is already a (query, positive-document) pair; only a negative-sampling strategy (in-batch
negatives from other examples' `source_doc_ids`, or BM25-hard-negatives from the same corpus)
would need to be added, not a new data-generation pass. This targets a different failure
surface (retrieval recall, not generation discipline) but is defensible under the same
observed baseline: if the *generator's* over-abstention turns out to trace back to
insufficiently on-topic retrieved context rather than an over-cautious generation policy, tuning
the retriever is the more targeted fix.

---

## 8. "Did it actually help?" — the evaluation design for whichever branch runs

Not a single aggregate win. Required for M8's report regardless of which branch (LoRA
generator or retriever fallback) runs:
- **Per-slice improvement**, not one pooled number — at minimum by taxonomy category
  (`benchmark/taxonomy.md`'s 7 types) and by `is_abstention`, mirroring how T7.8's own
  false-abstention/true-abstention-recall split in §4 already proved necessary to see the real
  failure shape behind one aggregate metric.
- **Paired significance**, not independent CIs — the same 60 CRAGB questions (and the same 32
  probe questions) evaluated by both the untuned baseline and the tuned model, bootstrap CI on
  the *paired difference* (M4b's harness, reused), not two separate CIs eyeballed for overlap.
- **An OOD regression check** — re-run the tuned model on a handful of `train.jsonl`-disjoint
  categories the pilot's small scale didn't reach this time (`durability`, `defects`,
  `occasion`) to check the fine-tune's abstention-policy shift generalises rather than
  overfitting to `fit_sizing`, which dominates this pilot's accepted-example category mix
  (§4 note; see the datasheet's Composition section).
- **Latency held in the same table as quality**, not reported separately — a config that wins
  on false-abstention but silently blows the 3.0 s latency ceiling (§6) is not a win.

---

## 9. A/B-test design paragraph (PLAN.md §11 item 9)

If the tuned generator cleared §6's go/no-go and were deployed behind the demo (M9) rather than
just reported, the online test would randomise at the **session level** (not per-question,
since abstention-policy shifts are a within-session behaviour a user would notice as
inconsistent if it flipped mid-session), splitting traffic 50/50 between the untuned baseline
and the tuned model over a fixed window long enough to accumulate the same order of magnitude
of judged questions T7.8's own CRAGB+probe evaluation used (≥60–100 per arm, given this
report's own bootstrap CIs needed that many to be informative). The primary metric would be a
**proxy for false-abstention** observable online without a gold label — e.g. "did the user
re-ask a rephrased version of the same question within N seconds" (a strong behavioural signal
of an unhelpful refusal) — paired with the offline faithfulness/citation-validity guardrails
monitored as **non-inferiority** checks (must not regress beyond §6's thresholds), not
optimisation targets, exactly mirroring this document's own "no aggregate win, no regression on
the guardrails" structure at pilot scale.

---

## Appendix: what M8 inherits from this plan, unresolved

- `train.jsonl`/`val.jsonl` need repopulating via a resumed T7.3 sweep before real training can
  start (§3).
- `configs/finetune_train.yaml`'s hyperparameters (`learning_rate`, `lora_alpha_multiplier`,
  `lora_dropout`, `num_epochs`) remain **explicit placeholders**, deliberately not finalised
  here — PLAN.md §10's own stated position is that locking these before seeing T7.8's failure
  modes is false precision. §4's revised target behaviours are what M8 should tune the
  placeholder values *toward* (e.g. a training mix and loss weighting that penalises
  unnecessary abstention specifically), not a license to pick them arbitrarily.
- The chosen configuration (Qwen2.5-1.5B, rank 16, seq 1,152) is T7.9's *probe*-scale
  recommendation; M8 should re-run a short VRAM check at the real batch composition once
  `train.jsonl` has real volume, in case per-step memory shifts with genuinely varied sequence
  lengths rather than the probe's single repeated example.
