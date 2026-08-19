# CRAGB Grounded-QA Prompt (v1)

You are answering a shopper's question about a category of clothing, footwear, or
jewelry products, using **only** the customer review excerpts provided below. You have
no other information about this product category.

## Review excerpts

Each excerpt is one customer review, labeled with its review id. `has_photo: yes` means
that reviewer also attached a photo.

$context_block

## Question

$question

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
