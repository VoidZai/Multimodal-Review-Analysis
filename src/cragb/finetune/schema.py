"""Fine-tuning training-example schema (T7.1; PLAN.md §3 E8, §10, M7.md T7.1).

Fixes the exact on-disk shape of one synthetic (context -> grounded, cited answer)
training example, and guarantees that the prompt a fine-tuned model sees at *training*
time is structurally identical to the one it sees at *inference* time. Train/serve prompt
skew — a training prompt that differs, even slightly, from what the model is served at
inference — is one of the most common ways an SFT run silently produces a model that
scores worse than the untuned base. `render_training_prompt` closes that gap by
construction rather than by convention: it calls `cragb.generate.grounded_qa.render_prompt`
against a `cragb.generate.context_builder.ContextBlock` built from the stored
`context_text`, the exact same two calls `cragb.eval.cost_model.build_messages_for_row`
already uses (T5.4) to byte-reproduce a saved transcript's original prompt. Nothing here
reimplements template substitution.

**Abstention is a structural invariant, not a convention.** `TrainingExample.__post_init__`
detects a self-contradictory record — an answer whose text contains the canonical
`ABSTENTION_TEXT` alongside a non-empty `cited_doc_ids`, or an `is_abstention` flag that
disagrees with what the answer text actually says — the same way
`cragb.bench.reference_answers.make_reference_answer` does, and for the same reason
(PLAN.md §14.3): detecting the contradiction by *containment* of `ABSTENTION_TEXT`, not
equality, is what makes the check reachable at all once anything (a stray citation, a
constructed abstention's exact phrase) has been appended to or matches the canonical
string.

This module has no `main()` and performs no I/O beyond the pure `to_dict`/`from_dict`
JSON shape — it is deliberately just the schema + prompt-parity guarantee that every
later M7 task (T7.2's context sampler, T7.3's teacher generation, T7.4's constructed
abstentions, T7.5's filter, T7.6's split) builds `TrainingExample` records against.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from string import Template
from typing import Any

from cragb.bench.reference_answers import ABSTENTION_TEXT
from cragb.bench.taxonomy import CATEGORY_KEYWORDS
from cragb.generate.context_builder import ContextBlock
from cragb.generate.grounded_qa import load_prompt_template
from cragb.generate.grounded_qa import render_prompt as _render_grounded_prompt

# The one prompt template every training example is rendered against —
# T4a.1's versioned grounded-QA prompt, the same file every inference-time
# call (`configs/grounded_qa.yaml`, `configs/grounded_qa_large.yaml`,
# `configs/closed_book_qa.yaml`'s RAG sibling) renders through. Keeping this
# a fixed constant, rather than a per-call argument callers could
# accidentally vary, is what makes "training prompt == inference prompt"
# true by construction instead of by discipline.
DEFAULT_PROMPT_TEMPLATE_PATH = "src/cragb/generate/prompts/grounded_qa_v1.md"

# The canonical set of CRAGB taxonomy categories (`cragb.bench.taxonomy
# .CATEGORY_KEYWORDS`'s keys — the single source of truth `configs
# /taxonomy.yaml`'s `categories[].name` entries are themselves validated
# against). A `TrainingExample` outside this set cannot be sliced into the
# report's per-category tables, so it is rejected at construction time
# rather than silently accepted and discovered missing later.
VALID_CATEGORIES = frozenset(CATEGORY_KEYWORDS)


@dataclass(frozen=True)
class TrainingExample:
    """One (context -> grounded, cited answer) synthetic fine-tuning record.

    Mirrors `cragb.generate.grounded_qa.GroundedQATranscript`'s field shape
    where the two overlap (`question`, `cited_doc_ids`) so downstream code
    that already knows how to read a transcript needs little translation,
    while adding the provenance and grouping fields a *training* record
    needs that an *inference* transcript doesn't: `category` (for
    per-category composition tables), `source_parent_asins` (T7.6's
    train/val/probe split groups by product, not by document), and
    `provenance` (which construction path produced this example, for the
    datasheet).

    Attributes:
        example_id: stable unique id for this example (e.g.
            `"{category}_{context_group_id}_{n}"`).
        category: one of `VALID_CATEGORIES` (CRAGB's 7 taxonomy types).
        source_doc_ids: doc ids of the reviews shown in `context_text`,
            rank/insertion-ordered, matching the ids `context_text`'s
            `[doc_id]` excerpt labels use.
        source_parent_asins: `parent_asin`(s) the source reviews belong to
            — the grouping key T7.6's split must not let straddle train
            and val/probe.
        question: the training question.
        context_text: the exact `$context_block` value to substitute into
            the prompt — i.e. already rendered the way
            `cragb.generate.context_builder.render_excerpt` renders a
            retrieved excerpt (`[doc_id] has_photo: yes/no\\n<snippet>`,
            excerpts joined by a blank line), so a `ContextBlock` built
            from it and passed through `render_prompt` reproduces the same
            prompt shape a real retrieval call would have produced.
        answer: the target completion — the grounded, cited answer (or the
            exact `ABSTENTION_TEXT`) the model is trained to produce.
        cited_doc_ids: `[doc_id]` markers actually present in `answer`, in
            first-seen order, de-duplicated (mirrors
            `GroundedQATranscript.cited_doc_ids`).
        is_abstention: whether `answer` is a correct-abstention example.
            Must agree with whether `ABSTENTION_TEXT` is contained in
            `answer` — see `__post_init__`.
        provenance: freeform metadata about how this example was built
            (e.g. `{"method": "teacher_generation", "teacher_model": ...,
            "prompt_version": ..., "generated_at": ...}` for T7.3's output,
            or `{"method": "transplant", ...}` for T7.4's). Not
            schema-validated beyond being a `dict` — each construction
            path documents its own keys.
    """

    example_id: str
    category: str
    source_doc_ids: tuple[str, ...]
    source_parent_asins: tuple[str, ...]
    question: str
    context_text: str
    answer: str
    cited_doc_ids: tuple[str, ...]
    is_abstention: bool
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.category not in VALID_CATEGORIES:
            raise ValueError(
                f"{self.example_id}: unknown category {self.category!r}; "
                f"must be one of {sorted(VALID_CATEGORIES)}"
            )

        # Containment, not equality — PLAN.md §14.3: an equality check can
        # never fire once anything is appended to the canonical phrase,
        # which is exactly the shape of the contradiction this must catch
        # (abstention text + a stray citation).
        answer_is_abstention_text = ABSTENTION_TEXT in self.answer

        if answer_is_abstention_text and self.cited_doc_ids:
            raise ValueError(
                f"{self.example_id}: abstention answer must not contain citations "
                f"(found {self.cited_doc_ids})"
            )
        if self.is_abstention != answer_is_abstention_text:
            raise ValueError(
                f"{self.example_id}: is_abstention={self.is_abstention} but answer "
                f"{'does' if answer_is_abstention_text else 'does not'} contain ABSTENTION_TEXT "
                "-- the flag and the answer text must agree"
            )

    def to_dict(self) -> dict:
        """Serialize to a plain, JSON-safe dict (tuples -> lists)."""
        return {
            "example_id": self.example_id,
            "category": self.category,
            "source_doc_ids": list(self.source_doc_ids),
            "source_parent_asins": list(self.source_parent_asins),
            "question": self.question,
            "context_text": self.context_text,
            "answer": self.answer,
            "cited_doc_ids": list(self.cited_doc_ids),
            "is_abstention": self.is_abstention,
            "provenance": dict(self.provenance),
        }

    @classmethod
    def from_dict(cls, obj: dict) -> "TrainingExample":
        """Inverse of `to_dict` (lists -> tuples). Round-trips through JSONL byte-stably."""
        return cls(
            example_id=obj["example_id"],
            category=obj["category"],
            source_doc_ids=tuple(obj["source_doc_ids"]),
            source_parent_asins=tuple(obj["source_parent_asins"]),
            question=obj["question"],
            context_text=obj["context_text"],
            answer=obj["answer"],
            cited_doc_ids=tuple(obj["cited_doc_ids"]),
            is_abstention=obj["is_abstention"],
            provenance=dict(obj["provenance"]),
        )


def render_training_prompt(example: TrainingExample, template: Template | None = None) -> str:
    """Render the exact prompt a fine-tuned model is trained to complete for `example`.

    Builds a `ContextBlock` from `example.context_text`/`example.source_doc_ids` and
    substitutes it into T4a.1's prompt template via
    `cragb.generate.grounded_qa.render_prompt` — the same function, called the same way
    (`ContextBlock(text=..., doc_ids=..., photo_flags={})` then `render_prompt(template,
    question, context)`) that `cragb.eval.cost_model.build_messages_for_row` uses to
    byte-reproduce a saved inference transcript's original prompt (T5.4). `photo_flags` is
    passed empty because `render_prompt` only substitutes `context.text` — the
    `has_photo: yes/no` flags are already baked into that rendered text, exactly as they
    are in a real retrieved context block.

    Args:
        example: the training example to render.
        template: a pre-loaded prompt template (e.g. from a batch caller that loaded it
            once). If omitted, loads `DEFAULT_PROMPT_TEMPLATE_PATH` fresh.

    Returns:
        The rendered prompt string — character-for-character what a RAG-small inference
        call would send for the same `(question, context_text)` pair.
    """
    if template is None:
        template = load_prompt_template(DEFAULT_PROMPT_TEMPLATE_PATH)
    context = ContextBlock(
        text=example.context_text, doc_ids=example.source_doc_ids, photo_flags={}
    )
    return _render_grounded_prompt(template, example.question, context)


def to_chat_messages(example: TrainingExample, template: Template | None = None) -> list[dict[str, str]]:
    """Build the `[user, assistant]` chat-message pair an SFT trainer consumes for `example`.

    The user turn is `render_training_prompt(example, template)`; the assistant turn is
    `example.answer` verbatim (including the exact `ABSTENTION_TEXT` phrase, for an
    abstention example). Applying the chat template is the training script's job, not
    this module's — this only fixes the role/content shape, in exactly one place, so every
    later task builds the same pair.

    Args:
        example: the training example.
        template: forwarded to `render_training_prompt`.

    Returns:
        `[{"role": "user", "content": <prompt>}, {"role": "assistant", "content": <answer>}]`.
    """
    prompt = render_training_prompt(example, template)
    return [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": example.answer},
    ]
