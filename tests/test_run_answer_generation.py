"""Unit tests for cragb.eval.run_answer_generation (T4b.2; M4b.md T4b.2).

Two layers, mirroring `tests/test_run_grounded_qa_pilot.py`'s scope discipline for the
same kind of module (a thin batch driver over already-tested pipeline pieces):

- `validate_full_run` and the `ARMS`/`_ARM_DEFAULT_*` wiring get thorough pure-logic
  unit tests.
- `run_arm` gets an end-to-end test per arm against a tiny real corpus/questions/config
  fixture, with `GroqClient.complete` monkeypatched so nothing touches the network or
  needs an API key -- proving the actual wiring (config -> client -> pipeline -> file)
  works, not just each piece in isolation.

`main()` itself (argument parsing plus the same real I/O `run_arm` already covers) is not
separately unit-tested, following `cragb.eval.run_grounded_qa_pilot`'s own precedent.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest
import yaml

from cragb.eval.run_answer_generation import (
    ARMS,
    _ARM_DEFAULT_CONFIG,
    _ARM_DEFAULT_OUT,
    main,
    run_arm,
    validate_full_run,
)
from cragb.generate.api_clients import GroqClient
from cragb.generate.closed_book_qa import ClosedBookTranscript
from cragb.generate.context_builder import ContextBlock
from cragb.generate.grounded_qa import GroundedQATranscript


def make_closed_book_transcript(qid: str, answer_text: str = "An answer.") -> ClosedBookTranscript:
    return ClosedBookTranscript(
        question_id=qid, question="q?", raw_completion=answer_text,
        answer_text=answer_text, cited_doc_ids=(), abstained=False,
    )


def make_grounded_transcript(qid: str, answer_text: str = "Runs small [101].") -> GroundedQATranscript:
    context = ContextBlock(text="ctx", doc_ids=("101",), photo_flags={"101": False})
    return GroundedQATranscript(
        question_id=qid, question="q?", context=context, raw_completion=answer_text,
        answer_text=answer_text, cited_doc_ids=("101",) if "[101]" in answer_text else (),
        cited_photo_ids=(), abstained=False,
    )


# --------------------------------------------------------------------------
# ARMS / _ARM_DEFAULT_CONFIG / _ARM_DEFAULT_OUT wiring
# --------------------------------------------------------------------------


class TestArmWiring:
    def test_every_arm_has_a_default_config(self):
        assert set(_ARM_DEFAULT_CONFIG) == set(ARMS)

    def test_every_arm_has_a_default_output_path(self):
        assert set(_ARM_DEFAULT_OUT) == set(ARMS)

    def test_default_output_paths_are_distinct(self):
        # A collision here would mean two arms silently overwrite each other's transcripts.
        paths = list(_ARM_DEFAULT_OUT.values())
        assert len(paths) == len(set(paths))

    def test_rag_small_config_is_not_rag_large_config(self):
        # RQ1 needs these to actually differ (different model); a copy-paste slip
        # pointing both at the same file would silently collapse RQ1 to a no-op.
        assert _ARM_DEFAULT_CONFIG["rag_small"] != _ARM_DEFAULT_CONFIG["rag_large"]


# --------------------------------------------------------------------------
# validate_full_run
# --------------------------------------------------------------------------


class TestValidateFullRun:
    def test_passes_on_matching_nonempty_closed_book_transcripts(self):
        transcripts = [make_closed_book_transcript("q1"), make_closed_book_transcript("q2")]
        validate_full_run(transcripts, expected_question_ids=("q1", "q2"))  # no raise

    def test_passes_on_matching_nonempty_grounded_transcripts(self):
        transcripts = [make_grounded_transcript("q1"), make_grounded_transcript("q2")]
        validate_full_run(transcripts, expected_question_ids=("q1", "q2"))  # no raise

    def test_raises_on_missing_transcript(self):
        transcripts = [make_closed_book_transcript("q1")]
        with pytest.raises(ValueError, match="missing="):
            validate_full_run(transcripts, expected_question_ids=("q1", "q2"))

    def test_raises_on_unexpected_extra_transcript(self):
        transcripts = [make_closed_book_transcript("q1"), make_closed_book_transcript("q2")]
        with pytest.raises(ValueError, match="extra="):
            validate_full_run(transcripts, expected_question_ids=("q1",))

    def test_raises_on_wrong_order(self):
        transcripts = [make_closed_book_transcript("q1"), make_closed_book_transcript("q2")]
        with pytest.raises(ValueError, match="do not match"):
            validate_full_run(transcripts, expected_question_ids=("q2", "q1"))

    def test_raises_on_empty_answer_text(self):
        transcripts = [make_closed_book_transcript("q1"), make_closed_book_transcript("q2", answer_text="")]
        with pytest.raises(ValueError, match="empty answer_text"):
            validate_full_run(transcripts, expected_question_ids=("q1", "q2"))

    def test_raises_on_whitespace_only_answer_text(self):
        transcripts = [make_closed_book_transcript("q1", answer_text="   \n  ")]
        with pytest.raises(ValueError, match="empty answer_text"):
            validate_full_run(transcripts, expected_question_ids=("q1",))


# --------------------------------------------------------------------------
# main() argument validation (no network/file writes reached before the raise)
# --------------------------------------------------------------------------


class TestMainArgValidation:
    def test_arm_all_with_config_override_raises(self):
        with pytest.raises(ValueError, match="--config/--out only apply"):
            main(["--arm", "all", "--config", "configs/grounded_qa.yaml"])

    def test_arm_all_with_out_override_raises(self):
        with pytest.raises(ValueError, match="--config/--out only apply"):
            main(["--arm", "all", "--out", "somewhere.jsonl"])

    def test_unknown_question_id_raises(self, tmp_path):
        questions_path = tmp_path / "questions.jsonl"
        questions_path.write_text(
            json.dumps(
                {"id": "q1", "type": "fit_sizing", "question": "Q?", "is_negative": False, "relevant_ids": ["101"]}
            )
            + "\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="Unknown question id"):
            main(
                [
                    "--arm",
                    "closed_book",
                    "--questions-in",
                    str(questions_path),
                    "--question-ids",
                    "does_not_exist",
                ]
            )


# --------------------------------------------------------------------------
# run_arm -- end-to-end against tiny real fixtures, network mocked out
# --------------------------------------------------------------------------


def _write_yaml(path, data: dict) -> None:
    path.write_text(yaml.safe_dump(data), encoding="utf-8")


@pytest.fixture
def fake_groq_complete(monkeypatch):
    """Replace GroqClient.complete with a deterministic, offline stand-in."""
    calls = []

    def _fake_complete(self, messages):
        calls.append(messages)
        return "A short honest answer."

    monkeypatch.setattr(GroqClient, "complete", _fake_complete)
    return calls


class TestRunArmClosedBook:
    def test_writes_one_transcript_per_question(self, tmp_path, fake_groq_complete):
        from cragb.eval.cragb_questions import RetrievalQuestion

        prompt_path = tmp_path / "closed_book_qa_v1.md"
        prompt_path.write_text("Question: $question", encoding="utf-8")

        config_path = tmp_path / "closed_book_qa.yaml"
        _write_yaml(
            config_path,
            {
                "seed": 42,
                "paths": {
                    "prompt_template": str(prompt_path),
                    "cache_dir": str(tmp_path / "cache"),
                },
                "provider": {
                    "model": "openai/gpt-oss-20b",
                    "api_base": "https://api.groq.com/openai/v1",
                    "api_key_env": "GROQ_API_KEY",
                    "temperature": 0.2,
                    "max_tokens": 1200,
                    "timeout_s": 30,
                    "max_retries": 5,
                },
            },
        )

        questions = [
            RetrievalQuestion(id="q1", type="fit_sizing", question="Do these run small?",
                               is_negative=False, relevant_ids=frozenset({"101"})),
            RetrievalQuestion(id="q2", type="fabric_quality", question="What is the thread count?",
                               is_negative=True, relevant_ids=frozenset()),
        ]

        out_path = run_arm("closed_book", str(config_path), str(tmp_path / "out.jsonl"), questions, {})

        assert out_path.is_file()
        lines = out_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2
        assert [json.loads(line)["question_id"] for line in lines] == ["q1", "q2"]
        assert len(fake_groq_complete) == 2  # one completion call per question


class TestRunArmRag:
    def test_writes_one_transcript_per_question_using_real_bm25_index(self, tmp_path, fake_groq_complete):
        from cragb.eval.cragb_questions import RetrievalQuestion

        corpus = pd.DataFrame(
            {
                "text": [
                    "These run small, I sized up two sizes for a good fit.",
                    "Colour matched the photo exactly, very happy.",
                ],
                "has_image": [False, True],
            },
            index=pd.Index(["101", "202"]),
        )
        corpus_path = tmp_path / "corpus.parquet"
        corpus.to_parquet(corpus_path)

        chunking_path = tmp_path / "chunking.yaml"
        _write_yaml(chunking_path, {"chunking": {"scheme": "whole_review"}})

        prompt_path = tmp_path / "grounded_qa_v1.md"
        prompt_path.write_text("Q: $question\n$context_block", encoding="utf-8")

        config_path = tmp_path / "grounded_qa.yaml"
        _write_yaml(
            config_path,
            {
                "seed": 42,
                "paths": {
                    "corpus_in": str(corpus_path),
                    "prompt_template": str(prompt_path),
                    "cache_dir": str(tmp_path / "cache"),
                },
                "provider": {
                    "model": "openai/gpt-oss-20b",
                    "api_base": "https://api.groq.com/openai/v1",
                    "api_key_env": "GROQ_API_KEY",
                    "temperature": 0.2,
                    "max_tokens": 1200,
                    "timeout_s": 30,
                    "max_retries": 5,
                },
                "retrieval": {"retriever": "bm25", "k": 1, "chunking_config": str(chunking_path)},
            },
        )

        questions = [
            RetrievalQuestion(id="fit_000", type="fit_sizing", question="Do these run small?",
                               is_negative=False, relevant_ids=frozenset({"101"})),
        ]

        rag_index_cache: dict = {}
        out_path = run_arm("rag_small", str(config_path), str(tmp_path / "out.jsonl"), questions, rag_index_cache)

        assert out_path.is_file()
        lines = out_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        row = json.loads(lines[0])
        assert row["question_id"] == "fit_000"
        assert row["context_doc_ids"]  # real BM25 retrieval actually ran
        # The index was built and cached for reuse by a second RAG arm sharing the corpus.
        assert len(rag_index_cache) == 1

    def test_unknown_arm_raises(self, tmp_path):
        with pytest.raises(ValueError, match="Unknown arm"):
            run_arm("not_a_real_arm", "irrelevant.yaml", "out.jsonl", [], {})
