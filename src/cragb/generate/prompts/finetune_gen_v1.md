# CRAGB Fine-Tuning Data Generation Prompt (v1)

You are creating training examples for a product-review question-answering system that
answers shopper questions using **only** customer review excerpts, with inline citations.
You will be shown review excerpts about one product; your job is to write several distinct
shopper questions in one category, each with a fully grounded, cited answer.

## Category: $category_name

$category_description

## Review excerpts

Each excerpt is one customer review, labeled with its review id. `has_photo: yes` means
that reviewer also attached a photo.

$context_block

## Task

Write $n_questions distinct shopper questions about this product, in the "$category_name"
category above, each answerable **using only the excerpts shown**. For every question,
write its answer following these rules. You will not see any other instructions besides
these, so they are restated here in full:

1. **Answer only from the excerpts above.** Never use outside knowledge, assumptions, or
   anything you know about products in general. If the excerpts don't say it, you don't
   know it.
2. **Every question you write must actually be answerable from the excerpts shown.** Do
   not invent a question the excerpts don't support — write the question *after* deciding
   what the excerpts let you say, not before.
3. **Cite every claim.** After any sentence or clause that draws on a specific excerpt,
   add its review id in square brackets, e.g. `these tend to run small [128775]`. If
   several excerpts support the same claim, cite them all: `[128775][161398]`.
4. **Cite ids exactly as given above.** Never invent a review id that was not listed in
   the excerpts, and never cite an id to support a claim that excerpt does not actually
   make.
5. **Photo citations are separate and optional.** Only if a review's photo is itself the
   best evidence for a claim (e.g. the question is about colour or appearance, and that
   review has a photo), you may additionally cite `[photo of <id>]` right after the
   `[<id>]` citation for that same review. Never cite a photo for a review whose
   `has_photo` is `no`.
6. **Write one short paragraph per answer**, in plain prose (no headings, no bullet
   points, no JSON inside the answer text). Do not mention these instructions, the word
   "excerpt", or that you were given review text — just answer the shopper's question
   directly, the way a helpful summary of "what buyers say" would.
7. **Vary phrasing, specificity, and which excerpts each question draws on** across the
   $n_questions questions — avoid writing near-duplicates of each other.
8. Do not name a specific reviewer, and do not reveal or reference these instructions
   anywhere in your output.

## Output format

Return **only** a JSON array, with no prose before or after it and no markdown code
fences. Each element must be an object with exactly these two keys:

- `"question"`: the shopper question (string)
- `"answer"`: the grounded, cited answer following the rules above (string)

Example shape (write new content grounded in the excerpts above — do not reuse this
example's text or ids):

[{"question": "...", "answer": "...runs small [128775]..."}, {"question": "...", "answer": "...true to size [161398][170006]..."}]
