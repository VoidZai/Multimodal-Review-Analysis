# Multimodal photo-evidence examples (T6.7, RQ4)

Four pairs hand-picked from T6.5's live 25-pair pilot run (PLAN.md §3 E7, §7 appendix material): a clean win, an honest loss (the random control genuinely was better evidence), a principled tie (the question isn't something a photo can answer), and one case T6.6's human spot-check disagreed with. Both photos, both order-swapped judge verdicts, and every rationale below are exactly what T6.4/T6.5 produced -- nothing has been edited.

## Clean win: surfaced photo is obviously the right evidence: `colour_appearance_001` (surfaced photo wins)

*The surfaced photo is a genuine, well-lit product photo of the purple shoes the question asks about; the control is a photo of unrelated anime pin badges. Both order-swapped calls agree at confidence 5/5, and the human spot-check (T6.6, worksheet row R08) independently agreed too -- the one clear case in this pilot where every signal lines up.*

**Question:** Is the colour as pictured on the listing?

**Photo A (the pipeline's surfaced photo):**

![surfaced photo](../data/photos/64434d2927e4745d.jpg)

**Photo B (T6.3's random control):**

![control photo](../data/photos/b625d2f96026a504.jpg)

**Judge verdict, surfaced shown as A:** A (confidence 5/5) -- Photo A clearly displays the actual purple color of the shoes in real-life lighting, directly helping to verify color accuracy.

**Judge verdict, surfaced shown as B (order swapped):** B (confidence 5/5) -- Photo B clearly displays the vibrant purple color and material texture of the shoes in real lighting, making it ideal for verifying color accuracy.

**Order agreement:** True -> **outcome: surfaced_win**

---

## Honest loss: the control photo really is better evidence here: `fabric_quality_005` (control photo wins)

*T6.3's control is drawn at random from outside the question's retrieved context, so it can occasionally land on genuinely more relevant evidence than what the pipeline actually surfaced -- this is exactly that case. The surfaced photo shows thick knit socks; the control happens to show a mesh, breathable shoe upper, which is what a breathability/moisture-wicking question actually needs. The judge is not gaming this: both order-swapped calls independently identify the mesh photo as better evidence, not just defaulting to whichever position looks first.*

**Question:** Is the fabric breathable and moisture-wicking?

**Photo A (the pipeline's surfaced photo):**

![surfaced photo](../data/photos/a1501451138db73d.jpg)

**Photo B (T6.3's random control):**

![control photo](../data/photos/3b87dd7e3816be92.jpg)

**Judge verdict, surfaced shown as A:** B (confidence 4/5) -- Photo B clearly displays the textured, breathable mesh upper of the athletic shoe, which is relevant to moisture-wicking properties, whereas Photo A shows thick knit socks.

**Judge verdict, surfaced shown as B (order swapped):** A (confidence 4/5) -- Photo A clearly displays the mesh and textile construction of the shoe upper, which helps evaluate its breathability and material texture.

**Order agreement:** True -> **outcome: control_win**

---

## Principled tie: neither photo is evidence at all: `value_004` (tie)

*'Do customers feel this is a good investment?' is a claim about opinion, not something a photo can show either way -- an intact-earbuds-in-a-case photo and a shoe-lacing close-up are equally uninformative here. Both order-swapped calls independently say tie at confidence 5/5: this is the judge correctly recognising a question type photos structurally cannot answer, not indecision.*

**Question:** Do customers feel like this product is a good investment?

**Photo A (the pipeline's surfaced photo):**

![surfaced photo](../data/photos/06ae5347e7fc8bf4.jpg)

**Photo B (T6.3's random control):**

![control photo](../data/photos/ff0159030f68977d.jpg)

**Judge verdict, surfaced shown as A:** tie (confidence 5/5) -- Neither photo contains any information about whether customers feel the product is a good investment or value for money.

**Judge verdict, surfaced shown as B (order swapped):** tie (confidence 5/5) -- Neither photo provides any information regarding whether customers feel the product is a good investment.

**Order agreement:** True -> **outcome: tie**

---

## Defensible but contested: where the human spot-check disagreed: `fit_sizing_010` (control photo wins)

*The judge picks the control (a watch worn on a wrist, showing real on-body fit) over the surfaced photo (a necklace still boxed, showing no fit information at all) -- a coherent, evidence-grounded reason, agreeing with itself in both photo orders. But T6.6's human spot-check (worksheet row R07) picked the opposite: the necklace, on the reasoning that it is at least the product this specific review is actually about. Included here precisely because it is one of only two disagreements the spot-check surfaced (PLAN.md §14.6) that reads as genuine reasonable disagreement rather than a judge error -- reliability limitations should be shown, not only described in the abstract.*

**Question:** Do buyers report any noticeable flaws in how it fits?

**Photo A (the pipeline's surfaced photo):**

![surfaced photo](../data/photos/9c90ad68ff92bc53.jpg)

**Photo B (T6.3's random control):**

![control photo](../data/photos/081eb97a787d378e.jpg)

**Judge verdict, surfaced shown as A:** B (confidence 5/5) -- Photo B shows a watch being worn on a wrist, providing direct visual evidence of fit, whereas Photo A shows a packaged necklace not being worn.

**Judge verdict, surfaced shown as B (order swapped):** A (confidence 5/5) -- Photo A shows the watch being worn on a wrist to demonstrate real-world fit, whereas Photo B only shows a packaged necklace inside a box.

**Order agreement:** True -> **outcome: control_win**
