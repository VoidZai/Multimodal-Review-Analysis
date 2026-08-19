"""Unit tests for cragb.generate.closed_book_qa (T4b.1; M4b.md T4b.1).

Every function that would otherwise call a live LLM takes an injected `chat_fn` — a plain
callable standing in for `GroqClient.complete` — so prompt rendering, completion parsing,
and batch orchestration are all tested with no network access and no API key, mirroring
`tests/test_grounded_qa.py`'s pattern for T4a.3.
"""

from __future__ import annotations

from string import Template

from cragb.eval.cragb_questions import RetrievalQuestion
from cragb.generate.closed_book_qa import (
    ABSTENTION_TEXT,
    generate_closed_book_answer,
    parse_completion,
    render_prompt,
    run_closed_book_qa,
    write_transcripts_jsonl,
)

TEMPLATE = Template("Q: $question\n---")


# --------------------------------------------------------------------------
# render_prompt
# --------------------------------------------------------------------------


class TestRenderPrompt:
    def test_fills_question_only(self):
        prompt = render_prompt(TEMPLATE, "Do these run small?")
        assert prompt == "Q: Do these run small?\n---"

    def test_no_context_block_placeholder_required(self):
        # Unlike grounded_qa's render_prompt, this template has no
        # $context_block placeholder at all — substitute() must not
        # require or choke on its absence.
        template = Template("Question only: $question")
        assert render_prompt(template, "Does it run small?") == "Question only: Does it run small?"


# --------------------------------------------------------------------------
# parse_completion
# --------------------------------------------------------------------------


class TestParseCompletion:
    def test_parses_clean_closed_book_answer(self):
        raw = "  Sizing varies a lot by brand, so it's hard to say in general.  "
        answer_text, cited, abstained = parse_completion(raw)
        assert answer_text == "Sizing varies a lot by brand, so it's hard to say in general."
        assert cited == ()
        assert abstained is False

    def test_parses_clean_abstention(self):
        answer_text, cited, abstained = parse_completion(ABSTENTION_TEXT)
        assert abstained is True
        assert cited == ()
        assert answer_text == ABSTENTION_TEXT

    def test_bracketed_token_in_closed_book_answer_is_a_fabricated_citation(self):
        # The model was shown zero review ids in this arm, so any
        # [doc_id]-shaped bracket it produces is necessarily invented —
        # parse_completion still extracts it (as a hallucination signal
        # for later scoring), it just can never be legitimate here.
        raw = "These tend to run small [128775]."
        _, cited, _ = parse_completion(raw)
        assert cited == ("128775",)

    def test_does_not_raise_on_self_contradictory_completion(self):
        # Same class of model failure grounded_qa.parse_completion
        # deliberately does not raise on: abstaining but also citing.
        # containment-based abstention detection is what makes this
        # representable at all (PLAN.md §14.3) — an equality check could
        # never fire once a citation is appended to the phrase.
        raw = f"{ABSTENTION_TEXT} [128775]"
        answer_text, cited, abstained = parse_completion(raw)
        assert abstained is True
        assert cited == ("128775",)


# --------------------------------------------------------------------------
# generate_closed_book_answer
# --------------------------------------------------------------------------


class TestGenerateClosedBookAnswer:
    def test_builds_transcript_from_chat_fn_response(self):
        captured_messages = []

        def fake_chat_fn(messages):
            captured_messages.append(messages)
            return "Sizing varies by brand, hard to say in general without seeing reviews."

        transcript = generate_closed_book_answer("q1", "Do these run small?", TEMPLATE, fake_chat_fn)

        assert transcript.question_id == "q1"
        assert transcript.question == "Do these run small?"
        assert transcript.answer_text == "Sizing varies by brand, hard to say in general without seeing reviews."
        assert transcript.cited_doc_ids == ()
        assert transcript.abstained is False
        # Exactly one chat call, with the rendered prompt as a single user message.
        assert len(captured_messages) == 1
        assert captured_messages[0] == [{"role": "user", "content": render_prompt(TEMPLATE, "Do these run small?")}]

    def test_abstention_response_produces_abstained_transcript(self):
        transcript = generate_closed_book_answer(
            "q2", "What is the exact thread count?", TEMPLATE, lambda messages: ABSTENTION_TEXT
        )
        assert transcript.abstained is True
        assert transcript.cited_doc_ids == ()

    def test_to_dict_round_trips_core_fields(self):
        transcript = generate_closed_book_answer("q1", "Q?", TEMPLATE, lambda m: "An answer [999].")
        d = transcript.to_dict()
        assert d["question_id"] == "q1"
        assert d["question"] == "Q?"
        assert d["cited_doc_ids"] == ["999"]
        assert d["abstained"] is False
        assert "context" not in d
        assert "cited_photo_ids" not in d


# --------------------------------------------------------------------------
# run_closed_book_qa / write_transcripts_jsonl / load_transcripts_jsonl
# --------------------------------------------------------------------------


class TestRunClosedBookQa:
    def _questions(self) -> list[RetrievalQuestion]:
        return [
            RetrievalQuestion(
                id="fit_000",
                type="fit_sizing",
                question="Do these run small?",
                is_negative=False,
                relevant_ids=frozenset({"101"}),
            ),
            RetrievalQuestion(
                id="fabric_000",
                type="fabric_quality",
                question="What is the exact thread count?",
                is_negative=True,
                relevant_ids=frozenset(),
            ),
        ]

    def test_generates_one_transcript_per_question_in_order_with_no_retriever_or_corpus(self):
        def fake_chat_fn(messages):
            prompt = messages[0]["content"]
            return ABSTENTION_TEXT if "thread count" in prompt else "Runs small, most buyers size up."

        transcripts = run_closed_book_qa(self._questions(), TEMPLATE, fake_chat_fn)

        assert [t.question_id for t in transcripts] == ["fit_000", "fabric_000"]
        assert transcripts[0].abstained is False
        assert transcripts[1].abstained is True

    def test_write_transcripts_jsonl_writes_one_line_per_transcript(self, tmp_path):
        transcript = generate_closed_book_answer("q1", "Q?", TEMPLATE, lambda m: "Answer.")
        out_path = write_transcripts_jsonl([transcript], tmp_path / "transcripts.jsonl")
        lines = out_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        assert '"question_id": "q1"' in lines[0]

    def test_load_transcripts_jsonl_round_trips_write(self, tmp_path):
        from cragb.generate.closed_book_qa import load_transcripts_jsonl

        original = [
            generate_closed_book_answer("q1", "Do these run small?", TEMPLATE, lambda m: "Runs small [101]."),
            generate_closed_book_answer("q2", "Exact thread count?", TEMPLATE, lambda m: ABSTENTION_TEXT),
        ]
        out_path = write_transcripts_jsonl(original, tmp_path / "transcripts.jsonl")
        loaded = load_transcripts_jsonl(out_path)

        assert loaded == original
