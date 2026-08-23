# Answer-quality qualitative examples (T4b.8)

Four worked examples pulled from the real M4b run — T4b.2's generated transcripts,
T4b.5's judge scores, and T4b.6's human-scored validation worksheet — chosen to
illustrate the error taxonomy PLAN.md §7 asks for (grounding failure, hallucinated
fact, over-abstain) alongside one clean win for the pipeline. Every example below cites
its real `question_id` from `benchmark/cragb_v1.jsonl`; nothing here is paraphrased or
invented.

---

## 1. RAG clearly wins — `fit_sizing_000`

**Question:** "Do these run true to size?"

**Context shown to `rag_small`** (5 retrieved reviews, abbreviated):

> `[128775]` These do not run true to size. I am a very true Euro 39, 8.5, but these were too short.
> `[132660]` Nice looking design and pattern, however they do not run true to size.
> `[185658]` Very light material, they do not run true to size and are too short. Sent them back.
> `[177777]` I found these pants to run not true to size.
> `[24440]` ...They run true to size and are plain enough that they do not scream 'look at me'...

**Closed-book candidate:** "Not enough information in the available reviews to answer this question."

**`rag_small` candidate:** "Several reviewers report that the items do not run true to size, noting they were too short or not true to size [128775][132660][185658][177777]. However, one reviewer states that the shoes run true to size [24440]."

**Reference answer:** "Buyers are split. A majority of reviews report these run small or not true to size — several recommend sizing up or note returns due to a tight fit [128775][168968][6434][151873] — while other buyers say they fit true to size with no issues [174175][114606][181526][24440]. One review reports the opposite, running slightly big [116140]."

**Judge scores:**

| arm | correctness | faithfulness | completeness | conciseness |
|---|---|---|---|---|
| closed_book | 1 | 5 | 1 | 5 |
| rag_small | 5 | 5 | 4 | 5 |

**Why this matters:** with no product context at all, closed-book has nothing to work with and correctly refuses to guess — but that refusal is scored as *incorrect and incomplete* here, not just "honest," because the reference itself demonstrates the question was answerable from real evidence. `rag_small`, given the same evidence, reconstructs the same "mostly no, some say yes" split the reference gives, with real citations. This is the RQ0 headline result (`results/tables/rq0_answer_quality_v1.csv`) in miniature: grounding is the difference between "can't say" and a correct, evidenced answer.

---

## 2. Closed-book hallucination — `fabric_quality_006`

**Question:** "How does the color of the fabric hold up to repeated washing and wear?"

**Closed-book candidate:** "Colors on most fabrics tend to stay true for a good number of washes when the garment is cared for properly. Natural fibers like cotton can fade after several cycles, especially if washed in hot water or exposed to direct sunlight, whereas synthetic fibers such as polyester or nylon hold their dye better and resist fading. Blended fabrics may behave somewhere in between... To preserve color, wash inside-out on a gentle cycle in cold water, use a mild detergent, avoid bleach, and dry on low heat or air-dry..."

**Reference answer:** "Mostly well, with some exceptions: several reviews confirm the colour holds up after multiple washes without fading or bleeding [27549][148516][103690][93013][25284][10808], but a notable minority report fading, bleeding, or colour degradation after washing [70775][172916][91178][114985][38714]."

**Judge scores:** correctness=2, **faithfulness=1**, completeness=2, conciseness=2.

**Why this matters:** this is the most instructive failure mode in the whole batch. The candidate never cites anything (it *can't* — closed-book has no context), yet it reads as confident, specific, and reasonable — generic textile-care knowledge dressed up as an answer about *this* product. It says nothing false in the abstract, but it answers a question about *this specific item's actual reviews* with material that could apply to any garment, missing the reference's real, evidenced finding (mostly fine, but a real minority report fading). The judge caught this correctly (faithfulness=1); a first-pass human read of the same answer scored it 5/5/5/5 — exactly the "fluent text reads as correct on a skim" trap discussed live while building T4b.6's validation worksheet.

---

## 3. Over-abstain — `value_000`

**Question:** "Is this worth the extra cost?"

**Closed-book candidate:** "Not enough information in the available reviews to answer this question."

**Reference answer:** "Opinions are mixed: several reviews say the item is worth the cost [153666][47243][156672][110348][192891][105771], while a comparable number say it is not worth the price, citing thin material or poor quality relative to cost [106164][48724][187084][95228][124878]."

**Judge scores:** correctness=1, faithfulness=5, completeness=1, conciseness=5.

**Why this matters:** T4b.4's abstention special-rule only grants full marks when the *reference* also abstains. Here it doesn't — the reference shows real, substantial (if mixed) evidence exists. A candidate that abstains anyway isn't being appropriately cautious; it's failing to cover ground it could have covered, closed-book's structural limitation (no context at all) producing a real coverage gap. This distinction — "abstained" is not automatically "correct" — was the specific thing an early pass of T4b.6's human scoring missed on several rows before a second, more careful pass caught it.

---

## 4. Grounding failure & a real judge/human disagreement — `occasion_003`

**Question:** "Can I wear this in cold weather?"

**`rag_small` candidate:** "Yes, customers say it is warm and suitable for cold weather [106096][9657][123694][60436][150049]."

**Reference answer:** "Mostly yes: the large majority of pooled reviews confirm the item is warm and suitable for cold weather [9657][123694][29096][150049][53093][117346][95620][26993], though a couple of reviews report it is not warm enough or lets cold in [169199][27429][47856]."

**Judge scores vs. this project's own second human pass:**

| | correctness | faithfulness | completeness | conciseness |
|---|---|---|---|---|
| judge | 4 | 5 | 3 | 5 |
| human | 1 | 3 | 5 | 3 |

**Why this matters, two ways at once:**
- **Grounding failure:** the candidate's citations are all real (drawn from the context it was shown), so nothing is fabricated — but it drops the reference's caveat entirely, flattening "mostly yes, with a couple of exceptions" into a flat "Yes." The citations are technically valid while the *claim* they support is subtly overconfident — a grounding failure that citation-validity checking alone (T4a.4's mechanism) cannot catch, since every cited id is genuine.
- **Judge/human disagreement:** this is the single largest judge-vs-human gap in T4b.6's 40-row validation sample. The human scored it much harder on correctness (1 vs. the judge's 4) — reading the dropped caveat as a real error — while the judge treated "mostly right" as good enough. Whether the judge is too lenient here or the human too strict is a legitimate open question, not a settled one; it's exactly the kind of disagreement `results/tables/judge_validation_v1.csv`'s correctness κ (which stayed negative across every human-scoring pass) is picking up on at scale, not just in this one row.

---

## Error taxonomy summary

| Category | Example | Symptom |
|---|---|---|
| Hallucinated fact | `fabric_quality_006` | Confident, plausible-sounding claim with zero grounding (closed-book has no context to cite) |
| Over-abstain | `value_000` | Abstains despite the reference showing the question was answerable from real evidence |
| Grounding failure | `occasion_003` | Citations are genuine, but the claim they're used to support drops a real caveat present in the evidence |
| Judge/human disagreement | `occasion_003` | Judge and independent human scorer diverge sharply on correctness for the same answer |
