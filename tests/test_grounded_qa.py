"""Unit tests for cragb.generate.grounded_qa (T4a.3; M4a.md T4a.3).

Every function that would otherwise call a live LLM takes an injected `chat_fn` — a
plain callable standing in for `GroqClient.complete` — so prompt rendering, completion
parsing, and batch orchestration are all tested with no network access and no API key,
mirroring `tests/test_draft_questions.py`'s pattern for T2.2.
"""

from __future__ import annotations

from string import Template

import pandas as pd
import pytest

from cragb.eval.cragb_questions import RetrievalQuestion
from cragb.generate.context_builder import ContextBlock, build_corpus_lookup, index_bm25_retriever
from cragb.generate.grounded_qa import (
    ABSTENTION_TEXT,
    extract_photo_citations,
    generate_answer,
    parse_completion,
    render_prompt,
    run_grounded_qa,
    write_transcripts_jsonl,
)
from cragb.retrieval.chunking import ChunkingConfig

TEMPLATE = Template("Q: $question\n---\n$context_block\n---")


def make_context() -> ContextBlock:
    return ContextBlock(
        text="[101] has_photo: no\nRuns small, size up.\n\n[202] has_photo: yes\nTrue to size.",
        doc_ids=("101", "202"),
        photo_flags={"101": False, "202": True},
    )


# --------------------------------------------------------------------------
# render_prompt
# --------------------------------------------------------------------------


class TestRenderPrompt:
    def test_fills_question_and_context(self):
        prompt = render_prompt(TEMPLATE, "Do these run small?", make_context())
        assert "Q: Do these run small?" in prompt
        assert "[101] has_photo: no" in prompt
        assert "[202] has_photo: yes" in prompt


# --------------------------------------------------------------------------
# extract_photo_citations
# --------------------------------------------------------------------------


class TestExtractPhotoCitations:
    def test_extracts_single_photo_citation(self):
        assert extract_photo_citations("The colour matches [128775][photo of 128775].") == ("128775",)

    def test_no_photo_citations_returns_empty(self):
        assert extract_photo_citations("These run small [128775].") == ()

    def test_deduplicates_preserving_first_seen_order(self):
        text = "[photo of 202] then again [photo of 101] then [photo of 202]"
        assert extract_photo_citations(text) == ("202", "101")

    def test_does_not_confuse_plain_doc_citation_with_photo_citation(self):
        # `[doc_id]` alone must never be picked up by the photo-citation
        # regex — it requires the literal "photo of " prefix.
        assert extract_photo_citations("[128775]") == ()


# --------------------------------------------------------------------------
# parse_completion
# --------------------------------------------------------------------------


class TestParseCompletion:
    def test_parses_clean_grounded_answer(self):
        raw = "  These run small according to most buyers [128775][161398].  "
        answer_text, cited, photo_cited, abstained = parse_completion(raw)
        assert answer_text == "These run small according to most buyers [128775][161398]."
        assert cited == ("128775", "161398")
        assert photo_cited == ()
        assert abstained is False

    def test_parses_photo_citation_alongside_doc_citation(self):
        raw = "The colour looks darker in person [40507][photo of 40507]."
        _, cited, photo_cited, _ = parse_completion(raw)
        assert cited == ("40507",)
        assert photo_cited == ("40507",)

    def test_parses_clean_abstention(self):
        answer_text, cited, photo_cited, abstained = parse_completion(ABSTENTION_TEXT)
        assert abstained is True
        assert cited == ()
        assert photo_cited == ()
        assert answer_text == ABSTENTION_TEXT

    def test_does_not_raise_on_self_contradictory_completion(self):
        # A model abstaining but still citing is exactly the failure mode
        # E4 exists to measure (PLAN.md §3 E4) — parse_completion must
        # surface it as data, not raise (unlike
        # cragb.bench.reference_answers.make_reference_answer, which
        # raises on the same shape because there it's a human-authoring
        # bug to catch immediately, not a model failure to score).
        raw = f"{ABSTENTION_TEXT} [128775]"
        answer_text, cited, photo_cited, abstained = parse_completion(raw)
        assert abstained is True
        assert cited == ("128775",)

    def test_malformed_citation_not_extracted(self):
        # Extra characters inside the brackets (a citation the model
        # fabricated in a non-conforming shape) simply isn't recognised
        # as a citation at all — it is neither silently accepted nor
        # specially flagged here; T4a.4 scores what *is* extracted
        # against what doc_ids were actually shown to the model.
        answer_text, cited, _, _ = parse_completion("These run small [see review 128775].")
        assert cited == ()


# --------------------------------------------------------------------------
# generate_answer
# --------------------------------------------------------------------------


class TestGenerateAnswer:
    def test_builds_transcript_from_chat_fn_response(self):
        captured_messages = []

        def fake_chat_fn(messages):
            captured_messages.append(messages)
            return "Runs small, most buyers size up [101]."

        transcript = generate_answer("q1", "Do these run small?", make_context(), TEMPLATE, fake_chat_fn)

        assert transcript.question_id == "q1"
        assert transcript.question == "Do these run small?"
        assert transcript.context.doc_ids == ("101", "202")
        assert transcript.answer_text == "Runs small, most buyers size up [101]."
        assert transcript.cited_doc_ids == ("101",)
        assert transcript.abstained is False
        # Exactly one chat call, with the rendered prompt as a single user message.
        assert len(captured_messages) == 1
        assert captured_messages[0] == [
            {"role": "user", "content": render_prompt(TEMPLATE, "Do these run small?", make_context())}
        ]

    def test_abstention_response_produces_abstained_transcript(self):
        transcript = generate_answer(
            "q2", "What is the exact thread count?", make_context(), TEMPLATE, lambda messages: ABSTENTION_TEXT
        )
        assert transcript.abstained is True
        assert transcript.cited_doc_ids == ()

    def test_to_dict_round_trips_core_fields(self):
        transcript = generate_answer("q1", "Q?", make_context(), TEMPLATE, lambda m: "Answer [101].")
        d = transcript.to_dict()
        assert d["question_id"] == "q1"
        assert d["cited_doc_ids"] == ["101"]
        assert d["context_doc_ids"] == ["101", "202"]
        assert d["context_photo_flags"] == {"101": False, "202": True}
        assert d["abstained"] is False


# --------------------------------------------------------------------------
# run_grounded_qa (batch orchestration, real BM25 retriever + whole_review chunking)
# --------------------------------------------------------------------------


class TestRunGroundedQa:
    def _corpus(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "text": [
                    "These run small, I sized up two sizes for a good fit.",
                    "Colour matched the photo exactly, very happy.",
                ],
                "has_image": [False, True],
            },
            index=pd.Index(["101", "202"]),
        )

    def test_generates_one_transcript_per_question_in_order(self):
        corpus = self._corpus()
        retriever, chunk_to_parent = index_bm25_retriever(corpus, ChunkingConfig(scheme="whole_review"))
        lookup = build_corpus_lookup(corpus)

        questions = [
            RetrievalQuestion(
                id="fit_000", type="fit_sizing", question="Do these run small?",
                is_negative=False, relevant_ids=frozenset({"101"}),
            ),
            RetrievalQuestion(
                id="colour_000", type="colour_appearance", question="Does the colour match the photo?",
                is_negative=False, relevant_ids=frozenset({"202"}),
            ),
        ]

        def fake_chat_fn(messages):
            prompt = messages[0]["content"]
            return "Not enough information in the available reviews to answer this question." \
                if "colour" not in prompt.lower() else "Yes, matches [202]."

        transcripts = run_grounded_qa(
            questions, retriever, chunk_to_parent, lookup, TEMPLATE, fake_chat_fn, k=1
        )

        assert [t.question_id for t in transcripts] == ["fit_000", "colour_000"]
        assert transcripts[1].cited_doc_ids == ("202",)

    def test_write_transcripts_jsonl_writes_one_line_per_transcript(self, tmp_path):
        transcript = generate_answer("q1", "Q?", make_context(), TEMPLATE, lambda m: "Answer [101].")
        out_path = write_transcripts_jsonl([transcript], tmp_path / "transcripts.jsonl")
        lines = out_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        assert '"question_id": "q1"' in lines[0]
