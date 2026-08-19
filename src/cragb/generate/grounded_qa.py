"""Grounded-QA generation pipeline (T4a.3; PLAN.md §3 E4, M4a.md T4a.3).

Renders T4a.1's prompt (`prompts/grounded_qa_v1.md`) around a question and a T4a.2
`ContextBlock`, calls the LLM, and parses the completion into a structured
`GroundedQATranscript` — question, context, raw completion, and the citations/abstention
signal T4a.4's validity checker will score.

Mirrors T2.2's testability shape (`cragb.generate.draft_questions`): every function that
would otherwise need a live API call takes an injected `chat_fn` — a plain callable
standing in for `GroqClient.complete` — so prompt rendering, response parsing, and batch
orchestration are all unit-testable with no network access or API key. Only `main()`
constructs a real `GroqClient`.

**Citation parsing reuses `cragb.bench.reference_answers` rather than duplicating it**
(T4a.1's config comment: this prompt cites `[doc_id]` directly, the same convention the
human reference answers already use, specifically so this reuse is possible):
`extract_citations` and `ABSTENTION_TEXT` are imported, not reimplemented.

One deliberate divergence from how that module uses them: `cragb.bench.reference_answers
.make_reference_answer` *raises* if a human-authored answer is abstention text that also
carries a citation, because there a contradiction is an authoring bug to catch
immediately. Here the text comes from the model, not a human author, and the same
contradiction is itself one of the failure modes E4 exists to measure (PLAN.md §3 E4:
"model ignores grounding... over-abstains... fabricates citations"). So `parse_completion`
never raises on a self-contradictory completion — it records exactly what the model said,
including its mistakes, and leaves scoring them to T4a.4.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from string import Template
from typing import Callable

from dotenv import load_dotenv

from cragb.bench.reference_answers import ABSTENTION_TEXT, extract_citations
from cragb.eval.cragb_questions import RetrievalQuestion, load_retrieval_questions
from cragb.generate.api_clients import GroqClient
from cragb.generate.context_builder import (
    ContextBlock,
    CorpusLookup,
    build_context,
    build_corpus_lookup,
    index_bm25_retriever,
)
from cragb.retrieval.base import Retriever
from cragb.retrieval.chunking import load_chunking_config
from cragb.utils.io import load_config, resolve_path
from cragb.utils.seeds import set_global_seed

logger = logging.getLogger(__name__)

ChatFn = Callable[[list[dict[str, str]]], str]

_PHOTO_CITATION_RE = re.compile(r"\[photo of (\w+)\]")


def load_prompt_template(path: str | Path) -> Template:
    """Load T4a.1's versioned grounded-QA prompt as a `string.Template`.

    `$name` placeholders, not `str.format`'s `{name}` — identical
    rationale to `cragb.generate.draft_questions.load_prompt_template`:
    the model's own completion (and, upstream, this template's citation
    examples) contains literal square/curly-adjacent punctuation that
    `str.format` has no reason to be exposed to.
    """
    text = resolve_path(path).read_text(encoding="utf-8")
    return Template(text)


def render_prompt(template: Template, question: str, context: ContextBlock) -> str:
    """Fill T4a.1's template for one question against its retrieved context."""
    return template.substitute(question=question, context_block=context.text)


def extract_photo_citations(answer_text: str) -> tuple[str, ...]:
    """Pull `[photo of doc_id]` markers out of `answer_text`, first-seen order, de-duplicated.

    Separate from `extract_citations` (which only matches `[doc_id]`) by
    construction: `\\w+` cannot match the spaces inside `[photo of ...]`,
    so the two regexes never collide — a photo citation is only ever
    recognised here, and the plain `[doc_id]` citation T4a.1's rule 4
    requires alongside it is only ever recognised by `extract_citations`.
    """
    seen: dict[str, None] = {}
    for match in _PHOTO_CITATION_RE.finditer(answer_text):
        seen.setdefault(match.group(1), None)
    return tuple(seen)


@dataclass(frozen=True)
class GroundedQATranscript:
    """One question's full grounded-QA record: what it was asked, shown, and answered."""

    question_id: str
    question: str
    context: ContextBlock
    raw_completion: str
    answer_text: str
    cited_doc_ids: tuple[str, ...]
    cited_photo_ids: tuple[str, ...]
    abstained: bool

    def to_dict(self) -> dict:
        return {
            "question_id": self.question_id,
            "question": self.question,
            "context_doc_ids": list(self.context.doc_ids),
            "context_text": self.context.text,
            "context_photo_flags": self.context.photo_flags,
            "raw_completion": self.raw_completion,
            "answer_text": self.answer_text,
            "cited_doc_ids": list(self.cited_doc_ids),
            "cited_photo_ids": list(self.cited_photo_ids),
            "abstained": self.abstained,
        }


def parse_completion(raw_completion: str) -> tuple[str, tuple[str, ...], tuple[str, ...], bool]:
    """Parse a raw LLM completion into `(answer_text, cited_doc_ids, cited_photo_ids, abstained)`.

    Args:
        raw_completion: the model's raw completion text.

    Returns:
        `answer_text`: `raw_completion`, stripped of surrounding whitespace.
        `cited_doc_ids`: `[doc_id]` markers found in `answer_text`, in
            first-seen order, de-duplicated.
        `cited_photo_ids`: `[photo of doc_id]` markers found in
            `answer_text`, same ordering rule.
        `abstained`: whether `answer_text` contains T4a.1's canonical
            abstention sentence (`ABSTENTION_TEXT`), by *containment* —
            not exact equality, per the lesson PLAN.md §14.3 records: an
            equality check can never fire once anything is appended to
            the canonical phrase, which is exactly the failure mode
            (abstention text + a stray citation) this needs to detect.

    Note:
        Does not raise on a self-contradictory completion (abstained but
        `cited_doc_ids` non-empty) — see module docstring. That
        contradiction is data for T4a.4 to score, not an error here.
    """
    answer_text = raw_completion.strip()
    cited_doc_ids = extract_citations(answer_text)
    cited_photo_ids = extract_photo_citations(answer_text)
    abstained = ABSTENTION_TEXT in answer_text
    return answer_text, cited_doc_ids, cited_photo_ids, abstained


def generate_answer(
    question_id: str,
    question: str,
    context: ContextBlock,
    template: Template,
    chat_fn: ChatFn,
) -> GroundedQATranscript:
    """Render the prompt, call `chat_fn`, and parse the result into a `GroundedQATranscript`."""
    prompt = render_prompt(template, question, context)
    raw_completion = chat_fn([{"role": "user", "content": prompt}])
    answer_text, cited_doc_ids, cited_photo_ids, abstained = parse_completion(raw_completion)
    return GroundedQATranscript(
        question_id=question_id,
        question=question,
        context=context,
        raw_completion=raw_completion,
        answer_text=answer_text,
        cited_doc_ids=cited_doc_ids,
        cited_photo_ids=cited_photo_ids,
        abstained=abstained,
    )


def run_grounded_qa(
    questions: list[RetrievalQuestion],
    retriever: Retriever,
    chunk_to_parent: dict[str, str],
    lookup: CorpusLookup,
    template: Template,
    chat_fn: ChatFn,
    k: int,
) -> list[GroundedQATranscript]:
    """Build context and generate an answer for each of `questions`, in order.

    The end-to-end pipeline T4a.2 (retrieval -> context) and T4a.3
    (prompt -> completion -> parse) exist to support: one call per
    question, in the shape T4a.5's pilot run and any later batch run
    both need.

    Args:
        questions: questions to answer (e.g. a slice of
            `cragb.eval.cragb_questions.load_retrieval_questions`).
        retriever: an already-indexed `Retriever` (e.g. from
            `cragb.generate.context_builder.index_bm25_retriever`).
        chunk_to_parent: as required by `build_context`.
        lookup: a `CorpusLookup` over the same corpus the retriever was
            indexed on.
        template: T4a.1's loaded prompt template.
        chat_fn: `GroqClient.complete`, or a stand-in for testing.
        k: number of reviews to retrieve per question.

    Returns:
        One `GroundedQATranscript` per question, in `questions` order.
    """
    transcripts = []
    for q in questions:
        context = build_context(q.question, retriever, chunk_to_parent, lookup, k=k)
        transcripts.append(generate_answer(q.id, q.question, context, template, chat_fn))
    return transcripts


def write_transcripts_jsonl(transcripts: list[GroundedQATranscript], out_path: str | Path) -> Path:
    """Write `transcripts` as newline-delimited JSON, one object per line."""
    resolved = resolve_path(out_path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("w", encoding="utf-8") as f:
        for t in transcripts:
            f.write(json.dumps(t.to_dict(), ensure_ascii=False))
            f.write("\n")
    return resolved


def load_transcripts_jsonl(path: str | Path) -> list[GroundedQATranscript]:
    """Load transcripts written by `write_transcripts_jsonl`, reconstructing each `ContextBlock`.

    The inverse of `GroundedQATranscript.to_dict`/`write_transcripts_jsonl` — used by
    T4a.6's appendix renderer to read T4a.5's pilot output back without re-running any
    retrieval or generation.

    Args:
        path: a transcripts JSONL file (e.g.
            `results/tables/grounded_qa_transcripts_v1.jsonl`).

    Returns:
        One `GroundedQATranscript` per line, in file order.
    """
    transcripts = []
    with resolve_path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            context = ContextBlock(
                text=obj["context_text"],
                doc_ids=tuple(obj["context_doc_ids"]),
                photo_flags=obj["context_photo_flags"],
            )
            transcripts.append(
                GroundedQATranscript(
                    question_id=obj["question_id"],
                    question=obj["question"],
                    context=context,
                    raw_completion=obj["raw_completion"],
                    answer_text=obj["answer_text"],
                    cited_doc_ids=tuple(obj["cited_doc_ids"]),
                    cited_photo_ids=tuple(obj["cited_photo_ids"]),
                    abstained=obj["abstained"],
                )
            )
    return transcripts


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/grounded_qa.yaml", help="Path to grounded-QA config YAML.")
    parser.add_argument(
        "--question-ids",
        nargs="+",
        required=True,
        help="CRAGB question ids (from benchmark/cragb_v1.jsonl) to answer.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: answer an explicit list of CRAGB question ids.

    A deliberately small hook for T4a.3 — confirming the pipeline works
    end-to-end on real questions. Curating *which* questions go into the
    mid-progress report's worked-transcript slice, and scoring citation
    validity/abstention correctness over them, is T4a.5/T4a.4's job, not
    this CLI's.
    """
    import pandas as pd

    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    load_dotenv()

    cfg = load_config(args.config)
    set_global_seed(cfg["seed"])

    corpus = pd.read_parquet(resolve_path(cfg["paths"]["corpus_in"]))
    all_questions = load_retrieval_questions(cfg["paths"]["questions_in"])
    by_id = {q.id: q for q in all_questions}
    missing = [qid for qid in args.question_ids if qid not in by_id]
    if missing:
        raise ValueError(f"Unknown question id(s): {missing}")
    questions = [by_id[qid] for qid in args.question_ids]

    template = load_prompt_template(cfg["paths"]["prompt_template"])
    chunking_config = load_chunking_config(cfg["retrieval"]["chunking_config"])
    retriever, chunk_to_parent = index_bm25_retriever(corpus, chunking_config)
    lookup = build_corpus_lookup(corpus)

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

    transcripts = run_grounded_qa(
        questions, retriever, chunk_to_parent, lookup, template, client.complete, k=cfg["retrieval"]["k"]
    )

    out_path = write_transcripts_jsonl(transcripts, cfg["paths"]["transcripts_out"])
    logger.info("Wrote %d transcript(s) to %s", len(transcripts), out_path)
    for t in transcripts:
        logger.info(
            "  %s: abstained=%s cited=%s photo_cited=%s",
            t.question_id,
            t.abstained,
            list(t.cited_doc_ids),
            list(t.cited_photo_ids),
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
