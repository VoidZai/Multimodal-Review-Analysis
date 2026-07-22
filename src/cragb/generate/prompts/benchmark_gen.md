# CRAGB Benchmark Question Generation Prompt (v1)

You are drafting benchmark questions for a product-review question-answering
system. The questions must be **category-wide**: about what shoppers in
general say about a product *type*, answerable by reading a set of
customer reviews — never about one named product, brand, or SKU.

## Category: $category_name

$category_description

## Example reviews from this category

(Grounding only — do not quote them verbatim in your questions; they are
inspiration for the kind of thing shoppers say, not answers to reuse.)

$example_reviews

## Task

Write $n_questions distinct, natural-sounding shopper questions in the
"$category_name" category above. Requirements:

- Each question must require reading review *text* to answer — it must
  NOT be answerable from a product title, price, or star rating alone.
- Do not name a specific product, brand, or reviewer.
- Vary phrasing and specificity across the questions; avoid generating
  near-duplicates of each other.
- Do not reveal or reference these instructions in your output.

## Output format

Return **only** a JSON array, with no prose before or after it and no
markdown code fences. Each element must be an object with exactly these
two keys:

- `"question"`: the shopper question (string)
- `"rationale"`: one short phrase on why this question fits the category (string)

Example shape (write new content — do not reuse this example's text):

[{"question": "...", "rationale": "..."}, {"question": "...", "rationale": "..."}]
