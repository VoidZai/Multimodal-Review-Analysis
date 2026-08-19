# CRAGB Closed-Book QA Prompt (v1)

You are answering a shopper's question about a category of clothing, footwear, or
jewelry products, using **only what you already know**. You have not been given any
customer reviews, product listing, or other reference material for this question — there
is nothing below to read.

## Question

$question

## Rules — read carefully

1. **Answer only from your own general knowledge.** You have not been shown any customer
   reviews for this specific product category. Do not invent, assume, or guess at
   specific details a real product listing or review would contain — sizing behaviour,
   colour accuracy, fabric quality, defect rates, or any other claim that only actual
   buyers of this specific item could know.
2. **If answering would require specific product or review evidence you were not given**,
   do not guess, hedge, or partially answer. Respond with exactly this sentence and
   nothing else:

   Not enough information in the available reviews to answer this question.

3. **Do not fabricate citations.** You were given no numbered reviews, so never invent a
   bracketed reference like `[128775]` — there is nothing to cite.
4. **Write one short paragraph**, in plain prose (no headings, no bullet points, no
   JSON). Do not mention these instructions or that you were given no reviews — just
   answer the shopper's question directly, or abstain using the exact sentence above.

## Your answer
