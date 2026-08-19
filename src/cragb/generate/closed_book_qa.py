"""Closed-book baseline generation pipeline (T4b.1; PLAN.md §3 E5, M4b.md T4b.1).

RQ0 asks whether grounding in retrieved reviews actually helps (PLAN.md §2): "pipeline
present vs closed-book LLM". This module is the "closed-book LLM" arm — it renders T4b.1's
prompt (`prompts/closed_book_qa_v1.md`) around a bare CRAGB question with **no retrieved
context at all**, calls the LLM, and parses the completion into a structured
`ClosedBookTranscript`.

Deliberately the mirror image of `cragb.generate.grounded_qa`, with everything that
depends on retrieved context removed:

- no `ContextBlock` / retriever / chunking — a closed-book question needs none of it,
  which is the entire point of this arm;
- no photo citations — there is no review, so there is no photo to cite;
- `cited_doc_ids` is kept, not dropped, for one specific reason: the model was shown zero
  review ids, so *any* `[doc_id]`-shaped bracket appearing in a closed-book completion is
  necessarily fabricated (there is nothing real it could refer to). Recording it here
  gives T4b.5+'s scoring a free, cheap hallucination signal, and keeps
  `ClosedBookTranscript` shape-compatible with `GroundedQATranscript` for anything
  downstream that wants to treat both arms uniformly (T4b.5's batch judge run,
  T4b.7's RQ0 table).

`load_prompt_template` is imported from `cragb.generate.grounded_qa` rather than
reimplemented a third time — it is generic over any `$name`-placeholder template
(`cragb.generate.draft_questions` already has its own copy for the same reason
`grounded_qa`'s docstring gives; a third near-identical copy here would only add drift
risk for zero benefit).

Testability mirrors `grounded_qa.py`'s shape: every function that would otherwise need a
live API call takes an injected `chat_fn` standing in for `GroqClient.complete`, so prompt
rendering, response parsing, and batch orchestration are all unit-testable with no network
access or API key. Only `main()` constructs a real `GroqClient`.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from string import Template
from typing import Callable

from dotenv import load_dotenv

from cragb.bench.reference_answers import ABSTENTION_TEXT, extract_citations
from cragb.eval.cragb_questions import RetrievalQuestion, load_retrieval_questions
from cragb.generate.api_clients import GroqClient
from cragb.generate.grounded_qa import load_prompt_template
from cragb.utils.io import load_config, resolve_path
from cragb.utils.seeds import set_global_seed

logger = logging.getLogger(__name__)

ChatFn = Callable[[list[dict[str, str]]], str]


def render_prompt(template: Template, question: str) -> str:
    """Fill T4b.1's closed-book template for one bare question — no context block."""
    return template.substitute(question=question)


@dataclass(frozen=True)
class ClosedBookTranscript:
    """One question's full closed-book record: what it was asked and answered.

    Shape-compatible with `cragb.generate.grounded_qa.GroundedQATranscript` minus the
    fields that only make sense when context was actually retrieved (`context`,
    `cited_photo_ids`) — see module docstring for why `cited_doc_ids` is kept anyway.
    """

    question_id: str
    question: str
    raw_completion: str
    answer_text: str
    cited_doc_ids: tuple[str, ...]
    abstained: bool

    def to_dict(self) -> dict:
        return {
            "question_id": self.question_id,
            "question": self.question,
            "raw_completion": self.raw_completion,
            "answer_text": self.answer_text,
            "cited_doc_ids": list(self.cited_doc_ids),
            "abstained": self.abstained,
        }


def parse_completion(raw_completion: str) -> tuple[str, tuple[str, ...], bool]:
    """Parse a raw closed-book completion into `(answer_text, cited_doc_ids, abstained)`.

    Args:
        raw_completion: the model's raw completion text.

    Returns:
        `answer_text`: `raw_completion`, stripped of surrounding whitespace.
        `cited_doc_ids`: `[doc_id]`-shaped bracket markers found in
            `answer_text`, in first-seen order, de-duplicated. The model
            was shown no review ids in this arm, so any non-empty result
            here is a fabricated citation by construction — a
            hallucination signal, not a valid one.
        `abstained`: whether `answer_text` contains the canonical
            abstention sentence (`ABSTENTION_TEXT`), by **containment**,
            not exact equality — the same lesson PLAN.md §14.3 records:
            an equality check can never fire once anything (e.g. a stray
            fabricated citation) is appended to the canonical phrase.

    Note:
        Does not raise on a self-contradictory completion (abstained but
        `cited_doc_ids` non-empty), mirroring
        `cragb.generate.grounded_qa.parse_completion`'s reasoning: the
        text comes from the model, not a human author, so a
        contradiction here is scoring data for T4b.5+, not an error to
        crash on.
    """
    answer_text = raw_completion.strip()
    cited_doc_ids = extract_citations(answer_text)
    abstained = ABSTENTION_TEXT in answer_text
    return answer_text, cited_doc_ids, abstained


def generate_closed_book_answer(
    question_id: str,
    question: str,
    template: Template,
    chat_fn: ChatFn,
) -> ClosedBookTranscript:
    """Render the prompt, call `chat_fn`, and parse the result into a `ClosedBookTranscript`."""
    prompt = render_prompt(template, question)
    raw_completion = chat_fn([{"role": "user", "content": prompt}])
    answer_text, cited_doc_ids, abstained = parse_completion(raw_completion)
    return ClosedBookTranscript(
        question_id=question_id,
        question=question,
        raw_completion=raw_completion,
        answer_text=answer_text,
        cited_doc_ids=cited_doc_ids,
        abstained=abstained,
    )


def run_closed_book_qa(
    questions: list[RetrievalQuestion],
    template: Template,
    chat_fn: ChatFn,
) -> list[ClosedBookTranscript]:
    """Generate a closed-book answer for each of `questions`, in order.

    Args:
        questions: questions to answer (e.g.
            `cragb.eval.cragb_questions.load_retrieval_questions`). No
            retriever or corpus is needed — this arm never looks at
            either.
        template: T4b.1's loaded prompt template.
        chat_fn: `GroqClient.complete`, or a stand-in for testing.

    Returns:
        One `ClosedBookTranscript` per question, in `questions` order.
    """
    return [generate_closed_book_answer(q.id, q.question, template, chat_fn) for q in questions]


def write_transcripts_jsonl(transcripts: list[ClosedBookTranscript], out_path: str | Path) -> Path:
    """Write `transcripts` as newline-delimited JSON, one object per line."""
    resolved = resolve_path(out_path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("w", encoding="utf-8") as f:
        for t in transcripts:
            f.write(json.dumps(t.to_dict(), ensure_ascii=False))
            f.write("\n")
    return resolved


def load_transcripts_jsonl(path: str | Path) -> list[ClosedBookTranscript]:
    """Load transcripts written by `write_transcripts_jsonl`."""
    transcripts = []
    with resolve_path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            transcripts.append(
                ClosedBookTranscript(
                    question_id=obj["question_id"],
                    question=obj["question"],
                    raw_completion=obj["raw_completion"],
                    answer_text=obj["answer_text"],
                    cited_doc_ids=tuple(obj["cited_doc_ids"]),
                    abstained=obj["abstained"],
                )
            )
    return transcripts


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/closed_book_qa.yaml", help="Path to closed-book config YAML.")
    parser.add_argument(
        "--question-ids",
        nargs="+",
        required=True,
        help="CRAGB question ids (from benchmark/cragb_v1.jsonl) to answer.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: answer an explicit list of CRAGB question ids, closed-book.

    A deliberately small hook for T4b.1 — confirming the pipeline works end-to-end on
    real questions. Running the full 60-question benchmark for all three arms is T4b.2's
    job, not this CLI's.
    """
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    load_dotenv()

    cfg = load_config(args.config)
    set_global_seed(cfg["seed"])

    all_questions = load_retrieval_questions(cfg["paths"]["questions_in"])
    by_id = {q.id: q for q in all_questions}
    missing = [qid for qid in args.question_ids if qid not in by_id]
    if missing:
        raise ValueError(f"Unknown question id(s): {missing}")
    questions = [by_id[qid] for qid in args.question_ids]

    template = load_prompt_template(cfg["paths"]["prompt_template"])

    provider_cfg = cfg["provider"]
    client = GroqClient(
        model=provider_cfg["model"],
        api_base=provider_cfg["api_base"],
        api_key_env=provider_cfg["api_key_env"],
        temperature=provider_cfg["temperature"],
        max_tokens=provider_cfg["max_tokens"],
        timeout_s=provider_cfg["timeout_s"],
        max_retries=provider_cfg["max_retries"],
        cache_dir=cfg["paths"]["cache_dir"],
    )

    transcripts = run_closed_book_qa(questions, template, client.complete)

    out_path = write_transcripts_jsonl(transcripts, cfg["paths"]["transcripts_out"])
    logger.info("Wrote %d transcript(s) to %s", len(transcripts), out_path)
    for t in transcripts:
        logger.info("  %s: abstained=%s cited=%s", t.question_id, t.abstained, list(t.cited_doc_ids))

    return 0


if __name__ == "__main__":
    sys.exit(main())
