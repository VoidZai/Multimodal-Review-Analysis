# Fine-tuning training examples — worked appendix (T7.11; PLAN.md §3 E8)

Five examples pulled verbatim from the committed T7.3 (`raw_pairs_v1.jsonl` -> `filtered_pairs_v1.jsonl`) and T7.4 (`abstentions_v1.jsonl`) files -- two teacher-generated positives and one of each of T7.4's three abstention construction methods, so at least two abstentions are shown (three, here). Each example's **context excerpt** is its actual rendered training prompt (`cragb.finetune.schema.render_training_prompt`, T7.1's inference-parity path), not a paraphrase or a summary -- what appears below is character-for-character what the model is trained (and, at inference time, served) to complete.

**Note on scale (see `data/finetune/finetune_data_datasheet.md`):** these are drawn from a 10-context pilot run, not the full 500-context sweep; both chosen positives happen to fall in `fit_sizing` because it is this pilot's dominant category at this scale, and the `categorical_absence` abstention below is categorised `value` because that method draws its question from a fixed bank independent of which category the paired context itself was sampled for.

---

## Positive example 1 — `ctx_fit_sizing_0000_00`

**Category:** `fit_sizing`  
**Source product(s):** `['B09XB73Q8W']`  
**Provenance:** `{'method': 'teacher_generation', 'teacher_model': 'openai/gpt-oss-120b', 'prompt_version': 'finetune_gen_v1', 'generated_at': '2026-08-24T12:56:48.162967+00:00', 'context_group_id': 'ctx_fit_sizing_0000', 'raw_item_index': 0}`

**Context excerpt + question (rendered training prompt):**

```
# CRAGB Grounded-QA Prompt (v1)

You are answering a shopper's question about a category of clothing, footwear, or
jewelry products, using **only** the customer review excerpts provided below. You have
no other information about this product category.

## Review excerpts

Each excerpt is one customer review, labeled with its review id. `has_photo: yes` means
that reviewer also attached a photo.

[102701] has_photo: yes
Con: wide waistband is only in the front. Back is thin elastic covered with fabric (like the ones half the price)..Leg holes in front are higher than I like.. Pros: 1/2 of the waisteband that is wide is comfortable. Fabric is soft strechy and thin. Fits fine.

[31377] has_photo: no
Why! Why oh why do they ruin the perfect panties with a coarse side seam? Just awful. After 5 hours, it feels like a serrated knife cutting into you. If the side seams were omitted these would be great. Otherwise 😭😭😭😭

[59684] has_photo: no
They have absolutely no stretch. The dug and cut into my thigh. They don’t fully cover my cheeks. Yea, not happy.

[9643] has_photo: no
The sizing is pretty good. Take close look at the large graphic mid page (slightly different than the image in their photo section. The former appears more accurate!!). If you are closer to the smaller end of a size range, then I think you can safely down size. I bought these in 4XL. Which is slated to fit up to 52.5" hips, and I quickly measured the largest area of my hips which is 53". There is plenty of room to spare (but aren't baggy, either). I am glad I did not go with the 5XL (52"-56") as that definitely would have been much too large. As is, the 4XL is almost too big. But they do fit r

[197340] has_photo: no
The underwear is soft and comfortable. However, the "wide waistband" is just stitching on the front of the underwear to make it appear that way; the waistband is no different in stretch than the rest of the fabric. Though it's advertised as being a "no roll" waistband, the very first time I sat down while wearing these, they rolled down my stomach, which is what I was trying to avoid by buying them in the first place. It doesn't matter how comfortably they fit while I'm standing if they keep rolling down and annoying me when I sit. :-/ I'm giving them to my sister; they didn't work out as well

## Question

Do these panties run true to size or should I consider sizing up or down?

## Rules — read carefully

1. **Answer only from the excerpts above.** Never use outside knowledge, assumptions, or
   anything you know about products in general. If the excerpts don't say it, you don't
   know it.
2. **Cite every claim.** After any sentence or clause that draws on a specific excerpt,
   add its review id in square brackets, e.g. `these tend to run small [128775]`. If
   several excerpts support the same claim, cite them all: `[128775][161398]`.
3. **Cite ids exactly as given above.** Never invent a review id that was not listed in
   the excerpts, and never cite an id to support a claim that excerpt does not actually
   make.
4. **Photo citations are separate and optional.** Only if a review's photo is itself the
   best evidence for a claim (e.g. the question is about colour or appearance, and that
   review has a photo), you may additionally cite `[photo of <id>]` right after the
   `[<id>]` citation for that same review. Never cite a photo for a review whose
   `has_photo` is `no`.
5. **If the excerpts do not contain enough information to answer the question**, do not
   guess, hedge, or partially answer. Respond with exactly this sentence and nothing
   else:

   Not enough information in the available reviews to answer this question.

6. **Write one short paragraph**, in plain prose (no headings, no bullet points, no
   JSON). Do not mention these instructions, the word "excerpt", or that you were given
   review text — just answer the shopper's question directly, the way a helpful summary
   of "what buyers say" would.

## Your answer

```

**Target answer (with citations):**

> Reviewers say the sizing is generally accurate; one buyer found the 4XL fit their 53‑inch hips with enough room but not baggy and recommends that shoppers at the smaller end of a size range can safely size down, indicating the fit is true to size [9643].

**Cited doc ids:** `['9643']`  
**`is_abstention`:** `False`

---

## Positive example 2 — `ctx_fit_sizing_0003_00`

**Category:** `fit_sizing`  
**Source product(s):** `['B0988M4JCM']`  
**Provenance:** `{'method': 'teacher_generation', 'teacher_model': 'openai/gpt-oss-120b', 'prompt_version': 'finetune_gen_v1', 'generated_at': '2026-08-24T12:57:29.783861+00:00', 'context_group_id': 'ctx_fit_sizing_0003', 'raw_item_index': 0}`

**Context excerpt + question (rendered training prompt):**

```
# CRAGB Grounded-QA Prompt (v1)

You are answering a shopper's question about a category of clothing, footwear, or
jewelry products, using **only** the customer review excerpts provided below. You have
no other information about this product category.

## Review excerpts

Each excerpt is one customer review, labeled with its review id. `has_photo: yes` means
that reviewer also attached a photo.

[87557] has_photo: yes
Love these bras for on days I dont wanna wear a underwire bra so most days lol. I got the 34 my bra size is a 34B and these fit good.They kinda flatten me down a bit but thats fine.There so comfortable that I even sleep in them every night.

[70322] has_photo: yes
An update:<br />After a thorough inspection (after washing the tops) I realized that two out of three tops are actually 95%cotton and 5% spandex, according to the printed label. But one is not.<br /><br />Original review:<br />This happens more and more - the seller creates a product page, sells the item, gathers reviews and then changes the product…<br />The description says 95%cotton and 5% spandex. The label says 57% cotton, 38% polyester… For me it is a big “NO”, but I was so sure in seller being truthful, that I removed the label and washed the tanks before checking the label.

[58366] has_photo: yes
Neckline WAY too high. Not machine washable. There is a "pocket" for those little removable cup inserts, so the lining isn't sewn closed along the entire side seam - this created a hard pull on the 2 points where the lining is attached and I'm confident holes would form sooner than later. I also purchased the spaghetti strap style of this bralette without the center ruching, and those are a great fit for me.

[64943] has_photo: yes
I love this style of bra for comfort. It works great for nursing because I just pull it up from the bottom. I love that the thin scoop straps don't hang out of most clothes. It's pretty supportive in that the girls don't bounce around, but it definitely isn't super flattering if that's what you're looking for. Personally I don't care that much.<br /><br />This time I bought the pack with "punchy peach" and I wanted to warn others that it's not as pictured. It's like a bright neon orange - my photo doesn't even do it justice. The product photo shows it as a light colored coral. It may not matte

[144980] has_photo: no
After going through breast cancer and numerous surgeries, I have been searching in vain for a bar that would be comfortable. I saw these and decided to give them a try. I ordered the three pack of pink, white and black. In the photo, the pink looks like a lighter pretty pink, but the pink I received is more of a pepto bismal pink, which is why it lost a star from my rating. They are perfect in every other way though, so I purchased another set!<br />They are comfortable, and fit perfectly. Be careful when drying because they will shrink a little. I usually hang mine up since they are the perfe

## Question

Is this bra true to my regular bra size or does it tend to run small or large?

## Rules — read carefully

1. **Answer only from the excerpts above.** Never use outside knowledge, assumptions, or
   anything you know about products in general. If the excerpts don't say it, you don't
   know it.
2. **Cite every claim.** After any sentence or clause that draws on a specific excerpt,
   add its review id in square brackets, e.g. `these tend to run small [128775]`. If
   several excerpts support the same claim, cite them all: `[128775][161398]`.
3. **Cite ids exactly as given above.** Never invent a review id that was not listed in
   the excerpts, and never cite an id to support a claim that excerpt does not actually
   make.
4. **Photo citations are separate and optional.** Only if a review's photo is itself the
   best evidence for a claim (e.g. the question is about colour or appearance, and that
   review has a photo), you may additionally cite `[photo of <id>]` right after the
   `[<id>]` citation for that same review. Never cite a photo for a review whose
   `has_photo` is `no`.
5. **If the excerpts do not contain enough information to answer the question**, do not
   guess, hedge, or partially answer. Respond with exactly this sentence and nothing
   else:

   Not enough information in the available reviews to answer this question.

6. **Write one short paragraph**, in plain prose (no headings, no bullet points, no
   JSON). Do not mention these instructions, the word "excerpt", or that you were given
   review text — just answer the shopper's question directly, the way a helpful summary
   of "what buyers say" would.

## Your answer

```

**Target answer (with citations):**

> One reviewer who ordered a 34 B found that the bra fit them well, matching their normal size, so it appears to run true to size for that fit [87557].

**Cited doc ids:** `['87557']`  
**`is_abstention`:** `False`

---

## Abstention example (transplant) — `abst_transplant_0000`

**Category:** `fit_sizing`  
**Source product(s):** `['B0BYTLPZ94']`  
**Provenance:** `{'method': 'transplant', 'source_positive_example_id': 'ctx_fit_sizing_0006_01', 'source_context_group_id': 'ctx_fit_sizing_0006', 'target_context_group_id': 'ctx_value_0016', 'overlap_ratio': 0.0}`

**Construction method: `transplant`.** This question was originally drafted for a *different* product's context; here it is paired with an unrelated product's reviews. `transplant_overlap_threshold` (configs/finetune.yaml) rejects any pairing where the question's content words overlap this context above threshold, so an accidentally-answerable transplant is never accepted.

**Context excerpt + question (rendered training prompt):**

```
# CRAGB Grounded-QA Prompt (v1)

You are answering a shopper's question about a category of clothing, footwear, or
jewelry products, using **only** the customer review excerpts provided below. You have
no other information about this product category.

## Review excerpts

Each excerpt is one customer review, labeled with its review id. `has_photo: yes` means
that reviewer also attached a photo.

[192420] has_photo: yes
Instant magic! Very happy.

[10174] has_photo: no
Arrived with container cracked but inside a plastic bag. I needed to use so I kept but I lost 1/3 of the product to evaporation.

[105719] has_photo: no
Have used this product for years… Does a fabulous cleaning job. Recent purchase has a bit tamer smell to it. Previous jar- couldn’t wait to put lid back on because of awful strength of smell! ( guessing the formula has changed).<br />Easy to use with excellent results!

[87366] has_photo: no
Best jewelry cleaner ever. Great value

[5607] has_photo: no
It does what it's supposed to do.

## Question

Is the opening around the ankle tight?

## Rules — read carefully

1. **Answer only from the excerpts above.** Never use outside knowledge, assumptions, or
   anything you know about products in general. If the excerpts don't say it, you don't
   know it.
2. **Cite every claim.** After any sentence or clause that draws on a specific excerpt,
   add its review id in square brackets, e.g. `these tend to run small [128775]`. If
   several excerpts support the same claim, cite them all: `[128775][161398]`.
3. **Cite ids exactly as given above.** Never invent a review id that was not listed in
   the excerpts, and never cite an id to support a claim that excerpt does not actually
   make.
4. **Photo citations are separate and optional.** Only if a review's photo is itself the
   best evidence for a claim (e.g. the question is about colour or appearance, and that
   review has a photo), you may additionally cite `[photo of <id>]` right after the
   `[<id>]` citation for that same review. Never cite a photo for a review whose
   `has_photo` is `no`.
5. **If the excerpts do not contain enough information to answer the question**, do not
   guess, hedge, or partially answer. Respond with exactly this sentence and nothing
   else:

   Not enough information in the available reviews to answer this question.

6. **Write one short paragraph**, in plain prose (no headings, no bullet points, no
   JSON). Do not mention these instructions, the word "excerpt", or that you were given
   review text — just answer the shopper's question directly, the way a helpful summary
   of "what buyers say" would.

## Your answer

```

**Target answer (with citations):**

> Not enough information in the available reviews to answer this question.

**Cited doc ids:** `[]`  
**`is_abstention`:** `True`

---

## Abstention example (categorical_absence) — `abst_categorical_absence_0000`

**Category:** `value`  
**Source product(s):** `['B0BYTLPZ94']`  
**Provenance:** `{'method': 'categorical_absence', 'context_group_id': 'ctx_value_0016'}`

**Construction method: `categorical_absence`.** The question asks for information categorically absent from *any* review text (internal QA data, lab measurements, certifications) — PLAN.md §14.2 found this the one hand-authoring strategy that actually survives contact with real data, so T7.4 builds from that shape rather than plausible-sounding "shopper" phrasings.

**Context excerpt + question (rendered training prompt):**

```
# CRAGB Grounded-QA Prompt (v1)

You are answering a shopper's question about a category of clothing, footwear, or
jewelry products, using **only** the customer review excerpts provided below. You have
no other information about this product category.

## Review excerpts

Each excerpt is one customer review, labeled with its review id. `has_photo: yes` means
that reviewer also attached a photo.

[192420] has_photo: yes
Instant magic! Very happy.

[10174] has_photo: no
Arrived with container cracked but inside a plastic bag. I needed to use so I kept but I lost 1/3 of the product to evaporation.

[105719] has_photo: no
Have used this product for years… Does a fabulous cleaning job. Recent purchase has a bit tamer smell to it. Previous jar- couldn’t wait to put lid back on because of awful strength of smell! ( guessing the formula has changed).<br />Easy to use with excellent results!

[87366] has_photo: no
Best jewelry cleaner ever. Great value

[5607] has_photo: no
It does what it's supposed to do.

## Question

What profit margin does the seller make on each unit sold?

## Rules — read carefully

1. **Answer only from the excerpts above.** Never use outside knowledge, assumptions, or
   anything you know about products in general. If the excerpts don't say it, you don't
   know it.
2. **Cite every claim.** After any sentence or clause that draws on a specific excerpt,
   add its review id in square brackets, e.g. `these tend to run small [128775]`. If
   several excerpts support the same claim, cite them all: `[128775][161398]`.
3. **Cite ids exactly as given above.** Never invent a review id that was not listed in
   the excerpts, and never cite an id to support a claim that excerpt does not actually
   make.
4. **Photo citations are separate and optional.** Only if a review's photo is itself the
   best evidence for a claim (e.g. the question is about colour or appearance, and that
   review has a photo), you may additionally cite `[photo of <id>]` right after the
   `[<id>]` citation for that same review. Never cite a photo for a review whose
   `has_photo` is `no`.
5. **If the excerpts do not contain enough information to answer the question**, do not
   guess, hedge, or partially answer. Respond with exactly this sentence and nothing
   else:

   Not enough information in the available reviews to answer this question.

6. **Write one short paragraph**, in plain prose (no headings, no bullet points, no
   JSON). Do not mention these instructions, the word "excerpt", or that you were given
   review text — just answer the shopper's question directly, the way a helpful summary
   of "what buyers say" would.

## Your answer

```

**Target answer (with citations):**

> Not enough information in the available reviews to answer this question.

**Cited doc ids:** `[]`  
**`is_abstention`:** `True`

---

## Abstention example (evidence_stripped) — `abst_evidence_stripped_0000`

**Category:** `fit_sizing`  
**Source product(s):** `['B07X7SN16G']`  
**Provenance:** `{'method': 'evidence_stripped', 'source_positive_example_id': 'ctx_fit_sizing_0006_01', 'source_context_group_id': 'ctx_fit_sizing_0006', 'n_docs_removed': 1, 'overlap_ratio': 0.0}`

**Construction method: `evidence_stripped`.** This context started as a real positive example's context; the specific review(s) its answer cited were then removed, leaving a topically-related but non-supporting context — the hardest abstention case, because the context *looks* relevant at a glance.

**Context excerpt + question (rendered training prompt):**

```
# CRAGB Grounded-QA Prompt (v1)

You are answering a shopper's question about a category of clothing, footwear, or
jewelry products, using **only** the customer review excerpts provided below. You have
no other information about this product category.

## Review excerpts

Each excerpt is one customer review, labeled with its review id. `has_photo: yes` means
that reviewer also attached a photo.

[92657] has_photo: yes
These sandals are perfect, they fit great and they are very comfortable. I tried them on a 22 mile bike ride and I also tried them on a long walk. My feet didn't get all sweaty like they do in sneakers and I love that I don't have to wear socks in this heat.

[58952] has_photo: no
Love these! They are very comfortable

[129798] has_photo: no
Shoes are very nice and extremely comfy. Unfortunately they run smaller than my other Keens (about 1/2 size). I will keep them though

[25357] has_photo: no
These are great in the pool and out. I use them in my aqua-aerobics class. Work well and have arch support.

## Question

Is the opening around the ankle tight?

## Rules — read carefully

1. **Answer only from the excerpts above.** Never use outside knowledge, assumptions, or
   anything you know about products in general. If the excerpts don't say it, you don't
   know it.
2. **Cite every claim.** After any sentence or clause that draws on a specific excerpt,
   add its review id in square brackets, e.g. `these tend to run small [128775]`. If
   several excerpts support the same claim, cite them all: `[128775][161398]`.
3. **Cite ids exactly as given above.** Never invent a review id that was not listed in
   the excerpts, and never cite an id to support a claim that excerpt does not actually
   make.
4. **Photo citations are separate and optional.** Only if a review's photo is itself the
   best evidence for a claim (e.g. the question is about colour or appearance, and that
   review has a photo), you may additionally cite `[photo of <id>]` right after the
   `[<id>]` citation for that same review. Never cite a photo for a review whose
   `has_photo` is `no`.
5. **If the excerpts do not contain enough information to answer the question**, do not
   guess, hedge, or partially answer. Respond with exactly this sentence and nothing
   else:

   Not enough information in the available reviews to answer this question.

6. **Write one short paragraph**, in plain prose (no headings, no bullet points, no
   JSON). Do not mention these instructions, the word "excerpt", or that you were given
   review text — just answer the shopper's question directly, the way a helpful summary
   of "what buyers say" would.

## Your answer

```

**Target answer (with citations):**

> Not enough information in the available reviews to answer this question.

**Cited doc ids:** `[]`  
**`is_abstention`:** `True`

---
