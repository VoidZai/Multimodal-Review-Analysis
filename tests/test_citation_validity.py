"""Unit tests for cragb.eval.citation_validity (T4a.4; M4a.md T4a.4).

Constructs `GroundedQATranscript`s directly (no LLM/network involved) to cover the four
cases M4a.md calls out explicitly — all-valid, a fabricated citation, an abstention that
also carries a citation (self-contradiction), and a false-negative abstention — plus the
supporting checks (format compliance, gold-grounding, aggregation).
"""

from __future__ import annotations

import pandas as pd
import pytest

from cragb.eval.cragb_questions import RetrievalQuestion
from cragb.eval.citation_validity import (
    find_malformed_citations,
    gold_relevant_ids_by_question,
    per_question_dataframe,
    score_transcript,
    score_transcripts,
    summarize,
)
from cragb.generate.context_builder import ContextBlock
from cragb.generate.grounded_qa import GroundedQATranscript


def make_transcript(
    question_id: str = "q1",
    context_doc_ids: tuple[str, ...] = ("101", "202", "303"),
    answer_text: str = "These run small [101][202].",
    abstained: bool | None = None,
) -> GroundedQATranscript:
    context = ContextBlock(
        text="(irrelevant for these tests)",
        doc_ids=context_doc_ids,
        photo_flags={d: False for d in context_doc_ids},
    )
    from cragb.bench.reference_answers import extract_citations
    from cragb.generate.grounded_qa import ABSTENTION_TEXT, extract_photo_citations

    cited = extract_citations(answer_text)
    photo_cited = extract_photo_citations(answer_text)
    resolved_abstained = (ABSTENTION_TEXT in answer_text) if abstained is None else abstained
    return GroundedQATranscript(
        question_id=question_id,
        question="does it run small?",
        context=context,
        raw_completion=answer_text,
        answer_text=answer_text,
        cited_doc_ids=cited,
        cited_photo_ids=photo_cited,
        abstained=resolved_abstained,
    )


# --------------------------------------------------------------------------
# find_malformed_citations
# --------------------------------------------------------------------------


class TestFindMalformedCitations:
    def test_well_formed_doc_citation_is_not_malformed(self):
        assert find_malformed_citations("These run small [128775].") == ()

    def test_well_formed_photo_citation_is_not_malformed(self):
        assert find_malformed_citations("Colour differs [128775][photo of 128775].") == ()

    def test_non_conforming_bracket_shape_is_malformed(self):
        assert find_malformed_citations("These run small [see review 128775].") == (
            "[see review 128775]",
        )

    def test_empty_brackets_are_malformed(self):
        assert find_malformed_citations("Odd citation here [].") == ("[]",)

    def test_no_brackets_at_all_is_compliant(self):
        assert find_malformed_citations("Not enough information in the available reviews.") == ()


# --------------------------------------------------------------------------
# score_transcript — the four cases M4a.md calls out, plus supporting checks
# --------------------------------------------------------------------------


class TestScoreTranscriptAllValid:
    def test_all_citations_exist_in_context(self):
        t = make_transcript(context_doc_ids=("101", "202", "303"), answer_text="Runs small [101][202].")
        score = score_transcript(t, expected_abstained=False)
        assert score.format_compliant is True
        assert score.n_citations == 2
        assert score.fabricated_citations == ()
        assert score.citation_validity_rate == 1.0
        assert score.abstention_correct is True
        assert score.self_contradiction is False
        assert score.ungrounded_answer is False

    def test_gold_grounding_when_all_cited_ids_are_gold_relevant(self):
        t = make_transcript(context_doc_ids=("101", "202"), answer_text="Runs small [101][202].")
        score = score_transcript(t, expected_abstained=False, gold_relevant_ids=frozenset({"101", "202"}))
        assert score.n_grounded_in_gold == 2
        assert score.ungrounded_in_gold == ()


class TestScoreTranscriptFabricatedCitation:
    def test_citation_not_in_context_is_fabricated(self):
        # "999" was never shown to the model (context is 101/202/303).
        t = make_transcript(context_doc_ids=("101", "202", "303"), answer_text="Runs small [101][999].")
        score = score_transcript(t, expected_abstained=False)
        assert score.fabricated_citations == ("999",)
        assert score.citation_validity_rate == pytest.approx(0.5)
        assert score.format_compliant is True  # well-formed shape, just not grounded

    def test_gold_grounding_flags_cited_but_not_gold_relevant(self):
        t = make_transcript(context_doc_ids=("101", "202"), answer_text="Runs small [101][202].")
        # 202 is in context (so not "fabricated") but not in the gold pool.
        score = score_transcript(t, expected_abstained=False, gold_relevant_ids=frozenset({"101"}))
        assert score.fabricated_citations == ()
        assert score.n_grounded_in_gold == 1
        assert score.ungrounded_in_gold == ("202",)


class TestScoreTranscriptSelfContradiction:
    def test_abstention_with_citation_flagged_not_raised(self):
        from cragb.generate.grounded_qa import ABSTENTION_TEXT

        t = make_transcript(answer_text=f"{ABSTENTION_TEXT} [101]", context_doc_ids=("101",))
        score = score_transcript(t, expected_abstained=True)
        assert score.self_contradiction is True
        assert score.n_citations == 1
        # abstention_correct only compares the abstention *flag*, not the
        # (separately-flagged) contradiction in the same answer.
        assert score.abstention_correct is True


class TestScoreTranscriptFalseNegativeAbstention:
    def test_model_abstains_when_it_should_have_answered(self):
        from cragb.generate.grounded_qa import ABSTENTION_TEXT

        t = make_transcript(answer_text=ABSTENTION_TEXT, context_doc_ids=("101", "202"))
        # Ground truth says this question is answerable (expected_abstained=False),
        # but the model abstained anyway.
        score = score_transcript(t, expected_abstained=False)
        assert score.predicted_abstained is True
        assert score.expected_abstained is False
        assert score.abstention_correct is False

    def test_model_answers_when_it_should_have_abstained(self):
        t = make_transcript(answer_text="Runs small [101].", context_doc_ids=("101",))
        score = score_transcript(t, expected_abstained=True)
        assert score.abstention_correct is False


class TestScoreTranscriptUngroundedAnswer:
    def test_non_abstention_with_zero_citations_is_ungrounded(self):
        t = make_transcript(answer_text="These probably run small.", context_doc_ids=("101",))
        score = score_transcript(t, expected_abstained=False)
        assert score.ungrounded_answer is True
        assert score.citation_validity_rate is None
        assert score.n_citations == 0


class TestScoreTranscriptMalformedCitation:
    def test_non_conforming_bracket_marks_format_non_compliant(self):
        t = make_transcript(answer_text="Runs small [see 101].", context_doc_ids=("101",))
        score = score_transcript(t, expected_abstained=False)
        assert score.format_compliant is False
        assert score.malformed_citations == ("[see 101]",)


# --------------------------------------------------------------------------
# score_transcripts / batch loading
# --------------------------------------------------------------------------


class TestScoreTranscripts:
    def test_scores_every_transcript_in_order(self):
        transcripts = [
            make_transcript("q1", answer_text="Runs small [101]."),
            make_transcript("q2", answer_text="Runs true to size [202]."),
        ]
        scores = score_transcripts(transcripts, expected_abstentions={"q1": False, "q2": False})
        assert [s.question_id for s in scores] == ["q1", "q2"]

    def test_missing_ground_truth_label_raises(self):
        transcripts = [make_transcript("unknown_q")]
        with pytest.raises(KeyError):
            score_transcripts(transcripts, expected_abstentions={"other_q": False})

    def test_gold_relevant_ids_looked_up_per_question(self):
        transcripts = [make_transcript("q1", context_doc_ids=("101",), answer_text="Runs small [101].")]
        gold = {"q1": frozenset({"101"})}
        scores = score_transcripts(transcripts, expected_abstentions={"q1": False}, gold_relevant_ids=gold)
        assert scores[0].n_grounded_in_gold == 1


class TestGoldRelevantIdsByQuestion:
    def test_builds_mapping_from_retrieval_questions(self):
        questions = [
            RetrievalQuestion(
                id="q1", type="fit_sizing", question="q?", is_negative=False,
                relevant_ids=frozenset({"101", "202"}),
            )
        ]
        mapping = gold_relevant_ids_by_question(questions)
        assert mapping == {"q1": frozenset({"101", "202"})}


# --------------------------------------------------------------------------
# per_question_dataframe / summarize
# --------------------------------------------------------------------------


class TestPerQuestionDataframe:
    def test_one_row_per_score_tuple_fields_as_lists(self):
        scores = [score_transcript(make_transcript("q1", answer_text="Runs small [101][999]."), False)]
        df = per_question_dataframe(scores)
        assert len(df) == 1
        assert df.iloc[0]["fabricated_citations"] == ["999"]
        assert isinstance(df.iloc[0]["fabricated_citations"], list)


class TestSummarize:
    def test_raises_on_empty_scores(self):
        with pytest.raises(ValueError, match="empty"):
            summarize([])

    def test_micro_averaged_citation_validity_rate(self):
        # q1: 2 citations, both valid. q2: 1 citation, fabricated.
        # Micro-average: (2 + 0) valid / (2 + 1) total = 2/3, not the mean
        # of per-question rates (1.0 and 0.0 -> 0.5), which would weight
        # a 1-citation question the same as a 2-citation one.
        scores = [
            score_transcript(make_transcript("q1", context_doc_ids=("101", "202"), answer_text="a [101][202]."), False),
            score_transcript(make_transcript("q2", context_doc_ids=("101",), answer_text="b [999]."), False),
        ]
        summary = summarize(scores)
        assert summary.loc[0, "citation_validity_rate"] == pytest.approx(2 / 3)

    def test_questions_with_zero_citations_excluded_from_citation_validity_rate(self):
        from cragb.generate.grounded_qa import ABSTENTION_TEXT

        scores = [
            score_transcript(make_transcript("q1", answer_text="a [101]."), False),
            score_transcript(make_transcript("q2", answer_text=ABSTENTION_TEXT), True),
        ]
        summary = summarize(scores)
        # q2 contributes 0 citations; the rate is 1/1, not diluted to 1/2.
        assert summary.loc[0, "citation_validity_rate"] == 1.0
        assert summary.loc[0, "n_total_citations"] == 1

    def test_gold_grounding_rate_none_when_never_evaluated(self):
        scores = [score_transcript(make_transcript("q1", answer_text="a [101]."), False)]
        summary = summarize(scores)
        assert summary.loc[0, "gold_grounding_rate"] is None

    def test_aggregate_counts_and_rates(self):
        scores = [
            score_transcript(make_transcript("q1", answer_text="a [101]."), expected_abstained=False),
            score_transcript(make_transcript("q2", answer_text="b [999]."), expected_abstained=True),
        ]
        summary = summarize(scores)
        assert summary.loc[0, "n_questions"] == 2
        assert summary.loc[0, "abstention_accuracy"] == 0.5  # q1 correct, q2 wrong
        assert summary.loc[0, "n_total_citations"] == 2
        assert summary.loc[0, "n_fabricated_citations"] == 1
        assert isinstance(summary, pd.DataFrame)
