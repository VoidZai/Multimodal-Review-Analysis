# Grounded-QA worked transcripts (T4a.6)

Five transcripts hand-picked from T4a.5's 11-question pilot run over CRAGB v1 (PLAN.md §3 E4, §7 appendix material): three clean grounded answers spanning different taxonomy types and edge cases, one correct abstention, and one documented failure mode. Every citation and every word of the model's answer below is exactly what it produced — nothing has been edited.

## Clean grounded answer: `fit_sizing_000`

*A standard case: five reviews retrieved, the model reports the majority/minority split in what buyers actually say rather than picking one side, and cites every claim to a real review id.*

**Question:** Do these run true to size?

**Reviews retrieved (k=5):**
- `128775` (has_photo: no)
- `132660` (has_photo: no)
- `185658` (has_photo: no)
- `177777` (has_photo: no)
- `24440` (has_photo: no)

**Model's answer:**

> Several reviewers report that the items do not run true to size, noting they were too short or not true to size [128775][132660][185658][177777]. However, one reviewer states that the shoes run true to size [24440].

**Scoring:** abstained=False (not expected), format_compliant=True, 5 citation(s), all valid.

---

## Clean grounded answer (second taxonomy type): `durability_000`

*Same pattern as fit_sizing_000, on a different taxonomy category (durability, not fit).*

**Question:** Does this fall apart easily after repeated wear?

**Reviews retrieved (k=5):**
- `19886` (has_photo: no)
- `94200` (has_photo: no)
- `44922` (has_photo: no)
- `197055` (has_photo: no)
- `5104` (has_photo: no)

**Model's answer:**

> Some reviewers say the item falls apart easily after repeated wear, citing [19886] and [5104], whereas others report that it remains durable and does not fall apart after washing or repeated use, citing [94200], [44922], and [197055].

**Scoring:** abstained=False (not expected), format_compliant=True, 5 citation(s), all valid.

---

## Clean grounded answer on a tricky edge case: `fit_sizing_neg_001`

*Authored as a taxonomy negative (T2.3), but T2.7's pooling found real relevant evidence for it (PLAN.md §14.2: 17 of 19 pooled reviews were on-topic — the most striking miss of the 9 negatives that turned out answerable). CRAGB v1's ground truth is evidence-driven, not taxonomy-driven (T2.8), so this question's `is_abstention` is `False` — and the model correctly answers it instead of abstaining, matching that evidence-driven ground truth rather than the original taxonomy label.*

**Question:** Does the manufacturer's official size chart match what buyers experience?

**Reviews retrieved (k=5):**
- `73770` (has_photo: no)
- `184290` (has_photo: no)
- `37676` (has_photo: no)
- `140670` (has_photo: no)
- `3905` (has_photo: no)

**Model's answer:**

> Several reviewers report that the size chart does not match the actual fit, describing items as too small or ill‑fitting and noting that the chart is illegible or misleading [73770][184290][140670]. One reviewer, however, found the chart to be accurate for their size, noting that the bra fit as expected based on the chart [3905]. Thus, buyers have mixed experiences with the size chart, and it does not consistently match what they receive.

**Scoring:** abstained=False (not expected), format_compliant=True, 4 citation(s), all valid.

---

## Correct abstention: `fabric_quality_neg_000`

*One of CRAGB v1's only two genuine ground-truth abstentions (PLAN.md §14.2): no review reports an exact thread-count measurement, and the model correctly says so instead of guessing or fabricating a number.*

**Question:** What is the exact thread count of the fabric used?

**Reviews retrieved (k=5):**
- `141566` (has_photo: no)
- `185360` (has_photo: no)
- `27044` (has_photo: yes)
- `63886` (has_photo: no)
- `43800` (has_photo: no)

**Model's answer:**

> Not enough information in the available reviews to answer this question.

**Scoring:** abstained=True (expected), format_compliant=True, no citations (abstained).

---

## Failure mode: unused photo-citation affordance: `colour_appearance_009`

*Two of the three retrieved reviews had a photo attached (`has_photo: yes` in the context shown to the model), and the prompt (T4a.1, rule 4) explicitly permits citing `[photo of doc_id]` when a photo is the best evidence for a colour/appearance claim — exactly this question's type. The model never reaches for it here: it cites review text only. This is not a scored failure (every text citation below is valid and gold-grounded) but a real, measured limitation worth flagging for E7's multimodal pilot — the model under-uses photo evidence even when the prompt explicitly offers it.*

**Question:** Do buyers describe the colour as looking cheaper or different in person than in photos?

**Reviews retrieved (k=5):**
- `105003` (has_photo: yes)
- `182740` (has_photo: yes)
- `119326` (has_photo: yes)
- `51764` (has_photo: no)
- `174822` (has_photo: no)

**Model's answer:**

> Some buyers note that the colour they see in person or in their own photos differs from what is shown in the product images. For example, one reviewer said the colour in their picture was different from the stock photo, though the shoe itself matched the listed colour [105003], and another said the wallet looked different than in the photos [182740]. Additionally, one reviewer felt the item looked cheaper in person than in the pictures [51764].

**Scoring:** abstained=False (not expected), format_compliant=True, 3 citation(s), all valid.
