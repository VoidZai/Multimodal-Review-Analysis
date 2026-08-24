"""Unit tests for cragb.finetune.baseline_eval (T7.8; PLAN.md §3 E8, §10, M7.md T7.8).

This module is deliberately thin plumbing over existing, already-tested pipeline pieces
(`cragb.eval.citation_validity`, `cragb.eval.judge`, `cragb.eval.run_answer_generation
.validate_full_run`) -- these tests cover the *plumbing itself*: the retrieval-parity
guard (the spec's own "write it first" emphasis), probe-set judge scoring against each
example's own stored answer as reference (not CRAGB reference answers, and not a
self-referential trick), and the summary-row/table builders. No real model, retriever, or
API call anywhere here -- `GroundedQATranscript`/`TrainingExample` fixtures are hand-built,
and `score_probe_transcripts` is driven through a plain stub `chat_fn`, matching this
project's established testability shape throughout `cragb.eval`/`cragb.generate`.
"""

from __future__ import annotations

import json
from string import Template

import numpy as np
import pytest

from cragb.eval.citation_validity import score_transcripts
from cragb.eval.judge import JudgeScore
from cragb.finetune.baseline_eval import (
    CRAGB_SOURCE,
    PROBE_SOURCE,
    SourceEvalResult,
    assert_retrieval_matches_rag_small,
    build_baseline_row,
    build_baseline_table,
    score_probe_transcript,
    score_probe_transcripts,
    write_baseline_transcripts_jsonl,
)
from cragb.finetune.schema import TrainingExample
from cragb.generate.context_builder import ContextBlock
from cragb.generate.grounded_qa import GroundedQATranscript, write_transcripts_jsonl

JUDGE_TEMPLATE = Template(
    "Q:$question\nCTX:$context_block\nCAND:$candidate_answer\nREF:$reference_answer"
)


def make_transcript(
    question_id: str = "q1",
    question: str = "Does this run small?",
    doc_ids: tuple[str, ...] = ("1", "2"),
    context_text: str = "some context",
    answer_text: str = "Yes, it runs small [1].",
    cited_doc_ids: tuple[str, ...] = ("1",),
    abstained: bool = False,
) -> GroundedQATranscript:
    return GroundedQATranscript(
        question_id=question_id,
        question=question,
        context=ContextBlock(text=context_text, doc_ids=doc_ids, photo_flags={}),
        raw_completion=answer_text,
        answer_text=answer_text,
        cited_doc_ids=cited_doc_ids,
        cited_photo_ids=(),
        abstained=abstained,
    )


def make_example(
    example_id: str = "q1",
    question: str = "Does this run small?",
    answer: str = "Teacher answer, runs small [1].",
    category: str = "fit_sizing",
    is_abstention: bool = False,
    cited_doc_ids: tuple[str, ...] = ("1",),
) -> TrainingExample:
    return TrainingExample(
        example_id=example_id,
        category=category,
        source_doc_ids=("1", "2"),
        source_parent_asins=("P1",),
        question=question,
        context_text="[1] has_photo: no\nsome context",
        answer=answer,
        cited_doc_ids=cited_doc_ids,
        is_abstention=is_abstention,
        provenance={"method": "test"},
    )


def make_judge_response(faithfulness: int) -> str:
    return json.dumps(
        {"correctness": 3, "faithfulness": faithfulness, "completeness": 3, "conciseness": 3, "rationale": "r"}
    )


# --------------------------------------------------------------------------
# assert_retrieval_matches_rag_small
# --------------------------------------------------------------------------


class TestAssertRetrievalMatchesRagSmall:
    def test_matching_retrieval_does_not_raise(self, tmp_path):
        rag_small_transcript = make_transcript(doc_ids=("1", "2", "3"))
        rag_small_path = write_transcripts_jsonl([rag_small_transcript], tmp_path / "rag_small.jsonl")

        baseline_transcript = make_transcript(doc_ids=("1", "2", "3"))
        assert_retrieval_matches_rag_small([baseline_transcript], rag_small_path)  # no raise

    def test_mismatched_doc_ids_raises(self, tmp_path):
        rag_small_transcript = make_transcript(doc_ids=("1", "2", "3"))
        rag_small_path = write_transcripts_jsonl([rag_small_transcript], tmp_path / "rag_small.jsonl")

        baseline_transcript = make_transcript(doc_ids=("9", "8", "7"))
        with pytest.raises(ValueError, match="Retrieval mismatch"):
            assert_retrieval_matches_rag_small([baseline_transcript], rag_small_path)

    def test_mismatched_doc_order_raises(self, tmp_path):
        # Same set, different rank order -- still a real retrieval difference.
        rag_small_transcript = make_transcript(doc_ids=("1", "2", "3"))
        rag_small_path = write_transcripts_jsonl([rag_small_transcript], tmp_path / "rag_small.jsonl")

        baseline_transcript = make_transcript(doc_ids=("3", "2", "1"))
        with pytest.raises(ValueError, match="Retrieval mismatch"):
            assert_retrieval_matches_rag_small([baseline_transcript], rag_small_path)

    def test_no_shared_question_ids_raises(self, tmp_path):
        rag_small_transcript = make_transcript(question_id="other_question")
        rag_small_path = write_transcripts_jsonl([rag_small_transcript], tmp_path / "rag_small.jsonl")

        baseline_transcript = make_transcript(question_id="q1")
        with pytest.raises(ValueError, match="No question ids in common"):
            assert_retrieval_matches_rag_small([baseline_transcript], rag_small_path)

    def test_reports_every_mismatch_not_just_the_first(self, tmp_path):
        rag_small_transcripts = [
            make_transcript(question_id="q1", doc_ids=("1",)),
            make_transcript(question_id="q2", doc_ids=("2",)),
        ]
        rag_small_path = write_transcripts_jsonl(rag_small_transcripts, tmp_path / "rag_small.jsonl")

        baseline_transcripts = [
            make_transcript(question_id="q1", doc_ids=("9",)),
            make_transcript(question_id="q2", doc_ids=("8",)),
        ]
        with pytest.raises(ValueError, match=r"2/2 shared question"):
            assert_retrieval_matches_rag_small(baseline_transcripts, rag_small_path)

    def test_only_checks_shared_ids_not_full_set_equality(self, tmp_path):
        # baseline ran a subset of what RAG-small covers -- that's fine, only the
        # overlap needs to agree.
        rag_small_transcripts = [
            make_transcript(question_id="q1", doc_ids=("1",)),
            make_transcript(question_id="q2", doc_ids=("2",)),
        ]
        rag_small_path = write_transcripts_jsonl(rag_small_transcripts, tmp_path / "rag_small.jsonl")

        baseline_transcripts = [make_transcript(question_id="q1", doc_ids=("1",))]
        assert_retrieval_matches_rag_small(baseline_transcripts, rag_small_path)  # no raise


# --------------------------------------------------------------------------
# score_probe_transcript(s)
# --------------------------------------------------------------------------


class TestScoreProbeTranscripts:
    def test_reference_is_the_examples_own_answer_not_candidate_itself(self):
        example = make_example(answer="Teacher's original answer [1].")
        transcript = make_transcript(answer_text="Local model's independent answer [1].")

        calls = []

        def fake_chat_fn(messages):
            calls.append(messages)
            return make_judge_response(faithfulness=4)

        score_probe_transcript(transcript, example, JUDGE_TEMPLATE, fake_chat_fn)

        sent_prompt = calls[0][0]["content"]
        assert "REF:Teacher's original answer [1]." in sent_prompt
        assert "CAND:Local model's independent answer [1]." in sent_prompt
        # The two are genuinely different strings in the prompt -- not a self-comparison.
        assert "Teacher's original answer" != "Local model's independent answer"

    def test_returns_parsed_judge_score(self):
        example = make_example()
        transcript = make_transcript()
        score = score_probe_transcript(
            transcript, example, JUDGE_TEMPLATE, lambda m: make_judge_response(faithfulness=5)
        )
        assert isinstance(score, JudgeScore)
        assert score.faithfulness == 5

    def test_abstention_examples_are_still_judged_not_skipped(self):
        # Unlike T7.5's filter (which skips abstentions for stage 2), T7.8 judges every
        # probe transcript, abstentions included -- the judge's own abstention special
        # rule is what surfaces "hedged instead of abstaining" as a low faithfulness score.
        example = make_example(is_abstention=True, answer="Not enough information in the available reviews to answer this question.", cited_doc_ids=())
        transcript = make_transcript(
            answer_text="Not enough information in the available reviews to answer this question.",
            cited_doc_ids=(),
            abstained=True,
        )
        calls = []

        def fake_chat_fn(messages):
            calls.append(messages)
            return make_judge_response(faithfulness=5)

        score_probe_transcript(transcript, example, JUDGE_TEMPLATE, fake_chat_fn)
        assert len(calls) == 1  # actually called, not skipped

    def test_multiple_transcripts_scored_in_order(self):
        examples_by_id = {"q1": make_example(example_id="q1"), "q2": make_example(example_id="q2")}
        transcripts = [make_transcript(question_id="q1"), make_transcript(question_id="q2")]
        responses = iter([make_judge_response(3), make_judge_response(5)])
        scores = score_probe_transcripts(transcripts, examples_by_id, JUDGE_TEMPLATE, lambda m: next(responses))
        assert [s.faithfulness for s in scores] == [3, 5]

    def test_missing_example_raises(self):
        transcripts = [make_transcript(question_id="unknown_id")]
        with pytest.raises(KeyError, match="unknown_id"):
            score_probe_transcripts(transcripts, {}, JUDGE_TEMPLATE, lambda m: make_judge_response(5))


# --------------------------------------------------------------------------
# build_baseline_row / build_baseline_table
# --------------------------------------------------------------------------


def make_result(
    source: str = PROBE_SOURCE,
    model: str = "fake-model",
    transcripts=None,
    faithfulness_scores=(4, 5, 3),
    latency_seconds=(1.0, 2.0, 1.5),
    expected_abstentions=None,
) -> SourceEvalResult:
    transcripts = transcripts or [make_transcript()]
    if expected_abstentions is None:
        expected_abstentions = {t.question_id: t.abstained for t in transcripts}
    citation_scores = score_transcripts(transcripts, expected_abstentions, gold_relevant_ids=None)
    return SourceEvalResult(
        source=source,
        model=model,
        transcripts=transcripts,
        citation_scores=citation_scores,
        faithfulness_scores=list(faithfulness_scores),
        latency_seconds=list(latency_seconds),
    )


class TestBuildBaselineRow:
    def test_columns_match_citation_validity_summarize_plus_faithfulness_and_latency(self):
        row = build_baseline_row(make_result(), n_boot=100, alpha=0.05, rng=np.random.default_rng(1))
        expected_keys = {
            "source", "model", "n_questions", "format_compliance_rate", "citation_validity_rate",
            "n_fabricated_citations", "fabricated_citation_rate", "gold_grounding_rate",
            "abstention_accuracy", "self_contradiction_rate", "ungrounded_answer_rate",
            "faithfulness_mean", "faithfulness_ci_lo", "faithfulness_ci_hi",
            "n_latency_questions", "median_latency_s",
        }
        assert set(row) == expected_keys

    def test_faithfulness_mean_matches_hand_computed_mean(self):
        row = build_baseline_row(
            make_result(faithfulness_scores=(4, 5, 3)), n_boot=100, alpha=0.05, rng=np.random.default_rng(1)
        )
        assert row["faithfulness_mean"] == pytest.approx(4.0)

    def test_median_latency_matches_hand_computed_median(self):
        row = build_baseline_row(
            make_result(latency_seconds=(1.0, 2.0, 1.5)), n_boot=100, alpha=0.05, rng=np.random.default_rng(1)
        )
        assert row["median_latency_s"] == pytest.approx(1.5)

    def test_fabricated_citation_rate_is_derived_correctly(self):
        fabricated = make_transcript(cited_doc_ids=("999",))  # not in doc_ids ("1","2")
        result = make_result(transcripts=[fabricated], expected_abstentions={"q1": False})
        row = build_baseline_row(result, n_boot=100, alpha=0.05, rng=np.random.default_rng(1))
        assert row["n_fabricated_citations"] == 1
        assert row["fabricated_citation_rate"] == pytest.approx(1.0)

    def test_zero_citations_gives_none_fabrication_rate_not_division_error(self):
        no_citations = make_transcript(cited_doc_ids=(), answer_text="No citations here.")
        result = make_result(transcripts=[no_citations], expected_abstentions={"q1": False})
        row = build_baseline_row(result, n_boot=100, alpha=0.05, rng=np.random.default_rng(1))
        assert row["fabricated_citation_rate"] is None

    def test_no_latency_data_gives_none_not_an_error(self):
        result = make_result(latency_seconds=())
        row = build_baseline_row(result, n_boot=100, alpha=0.05, rng=np.random.default_rng(1))
        assert row["median_latency_s"] is None
        assert row["n_latency_questions"] == 0

    def test_source_and_model_are_carried_through(self):
        row = build_baseline_row(
            make_result(source=CRAGB_SOURCE, model="Qwen/Qwen2.5-3B-Instruct"),
            n_boot=100, alpha=0.05, rng=np.random.default_rng(1),
        )
        assert row["source"] == CRAGB_SOURCE
        assert row["model"] == "Qwen/Qwen2.5-3B-Instruct"


class TestBuildBaselineTable:
    def test_one_row_per_result_in_order(self):
        table = build_baseline_table(
            [make_result(source=CRAGB_SOURCE), make_result(source=PROBE_SOURCE)],
            n_boot=100, alpha=0.05, rng=np.random.default_rng(1),
        )
        assert list(table["source"]) == [CRAGB_SOURCE, PROBE_SOURCE]


# --------------------------------------------------------------------------
# write_baseline_transcripts_jsonl
# --------------------------------------------------------------------------


class TestWriteBaselineTranscriptsJsonl:
    def test_tags_each_row_with_its_source(self, tmp_path):
        cragb_result = make_result(source=CRAGB_SOURCE, transcripts=[make_transcript(question_id="c1")])
        probe_result = make_result(source=PROBE_SOURCE, transcripts=[make_transcript(question_id="p1")])

        out_path = write_baseline_transcripts_jsonl([cragb_result, probe_result], tmp_path / "transcripts.jsonl")
        rows = [json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines()]

        assert rows[0]["question_id"] == "c1"
        assert rows[0]["source"] == CRAGB_SOURCE
        assert rows[1]["question_id"] == "p1"
        assert rows[1]["source"] == PROBE_SOURCE

    def test_creates_parent_directories(self, tmp_path):
        result = make_result()
        out_path = write_baseline_transcripts_jsonl([result], tmp_path / "nested" / "dir" / "transcripts.jsonl")
        assert out_path.is_file()

    def test_every_transcripts_field_survives(self, tmp_path):
        result = make_result(transcripts=[make_transcript(answer_text="A specific answer [1].")])
        out_path = write_baseline_transcripts_jsonl([result], tmp_path / "transcripts.jsonl")
        row = json.loads(out_path.read_text(encoding="utf-8").splitlines()[0])
        assert row["answer_text"] == "A specific answer [1]."
        assert row["cited_doc_ids"] == ["1"]
