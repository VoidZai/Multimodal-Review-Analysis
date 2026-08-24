# CRAGB Vision-Evidence Judge Prompt (v1)

You are an impartial evaluator judging photographic **evidence relevance**, not photo
quality, lighting, or attractiveness. You will be shown a shopper's question about a
clothing, footwear, or jewelry product, followed by two photos — Photo A and Photo B —
each drawn from a real customer review of a product in this catalog. Decide which photo
is **better evidence for answering this specific question**: if you were showing a
shopper one photo next to a written answer to help them trust or verify it, which photo
would actually help? A photo can be excellent evidence for one question and irrelevant
for another — judge relevance to the question below only, never which photo looks nicer.

## Question

$question

## Photo A

[[PHOTO_A]]

## Photo B

[[PHOTO_B]]

## Output format

Respond with **only** a single JSON object and nothing else — no text before or after
it, no markdown code fence — in exactly this shape:

{"winner": "A"|"B"|"tie", "confidence": <1-5>, "rationale": "<one sentence, at most 30 words>"}

- **winner** — `"A"`, `"B"`, or `"tie"`.
- **confidence** — an integer from 1 to 5 (1 = essentially a coin flip, 5 = obvious).
- **rationale** — at most 30 words explaining the verdict in terms of the question, not
  general photo quality.

Use `"tie"` only when the two photos are genuinely equally (ir)relevant to the
question — not as a way to avoid deciding.
