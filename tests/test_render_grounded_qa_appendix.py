"""Unit tests for cragb.eval.render_grounded_qa_appendix and
cragb.generate.grounded_qa.load_transcripts_jsonl (T4a.6; M4a.md T4a.6).
"""

from __future__ import annotations

import pytest

from cragb.eval.citation_validity import score_transcript
from cragb.eval.render_grounded_qa_appendix import (
    APPENDIX_ENTRIES,
    AppendixEntry,
    render_appendix_markdown,
    render_transcript_markdown,
)
from cragb.generate.context_builder import ContextBlock
from cragb.generate.grounded_qa import (
    ABSTENTION_TEXT,
    GroundedQATranscript,
    load_transcripts_jsonl,
    parse_completion,
    write_transcripts_jsonl,
)


def make_transcript(
    question_id: str = "q1",
    answer_text: str = "Runs small [101][202].",
    doc_ids: tuple[str, ...] = ("101", "202", "303"),
    photo_flags: dict[str, bool] | None = None,
) -> GroundedQATranscript:
    context = ContextBlock(
        text="ctx text",
        doc_ids=doc_ids,
        photo_flags=photo_flags or {d: False for d in doc_ids},
    )
    answer_text, cited, photo_cited, abstained = parse_completion(answer_text)
    return GroundedQATranscript(
        question_id=question_id, question="Do these run small?", context=context,
        raw_completion=answer_text, answer_text=answer_text, cited_doc_ids=cited,
        cited_photo_ids=photo_cited, abstained=abstained,
    )


# --------------------------------------------------------------------------
# load_transcripts_jsonl (round-trips write_transcripts_jsonl)
# --------------------------------------------------------------------------


class TestLoadTranscriptsJsonl:
    def test_round_trips_a_written_transcript(self, tmp_path):
        original = make_transcript(
            "q1", photo_flags={"101": True, "202": False, "303": False}
        )
        path = write_transcripts_jsonl([original], tmp_path / "t.jsonl")
        loaded = load_transcripts_jsonl(path)

        assert len(loaded) == 1
        rt = loaded[0]
        assert rt.question_id == original.question_id
        assert rt.question == original.question
        assert rt.context.doc_ids == original.context.doc_ids
        assert rt.context.photo_flags == original.context.photo_flags
        assert rt.answer_text == original.answer_text
        assert rt.cited_doc_ids == original.cited_doc_ids
        assert rt.abstained == original.abstained

    def test_round_trips_multiple_transcripts_in_order(self, tmp_path):
        originals = [make_transcript("q1"), make_transcript("q2", answer_text=ABSTENTION_TEXT)]
        path = write_transcripts_jsonl(originals, tmp_path / "t.jsonl")
        loaded = load_transcripts_jsonl(path)
        assert [t.question_id for t in loaded] == ["q1", "q2"]
        assert loaded[1].abstained is True

    def test_skips_blank_lines(self, tmp_path):
        path = tmp_path / "t.jsonl"
        original = make_transcript("q1")
        write_transcripts_jsonl([original], path)
        path.write_text(path.read_text(encoding="utf-8") + "\n\n", encoding="utf-8")
        loaded = load_transcripts_jsonl(path)
        assert len(loaded) == 1


# --------------------------------------------------------------------------
# render_transcript_markdown
# --------------------------------------------------------------------------


class TestRenderTranscriptMarkdown:
    def test_includes_question_answer_and_context_ids(self):
        t = make_transcript("fit_sizing_000")
        score = score_transcript(t, expected_abstained=False)
        entry = AppendixEntry(question_id="fit_sizing_000", label="Clean grounded answer", note="A note.")
        md = render_transcript_markdown(t, score, entry)
        assert "`fit_sizing_000`" in md
        assert "Do these run small?" in md
        assert "Runs small [101][202]." in md
        assert "`101`" in md and "`202`" in md and "`303`" in md
        assert "A note." in md

    def test_photo_flag_rendered_per_review(self):
        t = make_transcript("q1", photo_flags={"101": True, "202": False, "303": False})
        score = score_transcript(t, expected_abstained=False)
        entry = AppendixEntry(question_id="q1", label="x", note="n")
        md = render_transcript_markdown(t, score, entry)
        assert "`101` (has_photo: yes)" in md
        assert "`202` (has_photo: no)" in md

    def test_abstention_shows_no_citations_note(self):
        t = make_transcript("q1", answer_text=ABSTENTION_TEXT)
        score = score_transcript(t, expected_abstained=True)
        entry = AppendixEntry(question_id="q1", label="Correct abstention", note="n")
        md = render_transcript_markdown(t, score, entry)
        assert "no citations (abstained)" in md
        assert "abstained=True (expected)" in md

    def test_fabricated_citation_shown_in_scoring_line(self):
        t = make_transcript("q1", answer_text="Runs small [999].", doc_ids=("101",))
        score = score_transcript(t, expected_abstained=False)
        entry = AppendixEntry(question_id="q1", label="x", note="n")
        md = render_transcript_markdown(t, score, entry)
        assert "1 fabricated" in md


# --------------------------------------------------------------------------
# render_appendix_markdown
# --------------------------------------------------------------------------


class TestRenderAppendixMarkdown:
    def test_assembles_sections_in_entry_order(self):
        transcripts_by_id = {"q1": make_transcript("q1"), "q2": make_transcript("q2")}
        scores_by_id = {
            "q1": score_transcript(transcripts_by_id["q1"], expected_abstained=False),
            "q2": score_transcript(transcripts_by_id["q2"], expected_abstained=False),
        }
        entries = [
            AppendixEntry(question_id="q2", label="Second", note="n2"),
            AppendixEntry(question_id="q1", label="First", note="n1"),
        ]
        md = render_appendix_markdown(entries, transcripts_by_id, scores_by_id)
        assert md.index("`q2`") < md.index("`q1`")
        assert md.startswith("# Grounded-QA worked transcripts")

    def test_missing_transcript_raises_keyerror(self):
        entries = [AppendixEntry(question_id="missing_q", label="x", note="n")]
        with pytest.raises(KeyError, match="not found in transcripts"):
            render_appendix_markdown(entries, transcripts_by_id={}, scores_by_id={})

    def test_missing_score_raises_keyerror(self):
        entries = [AppendixEntry(question_id="q1", label="x", note="n")]
        transcripts_by_id = {"q1": make_transcript("q1")}
        with pytest.raises(KeyError, match="not found in scores"):
            render_appendix_markdown(entries, transcripts_by_id, scores_by_id={})


# --------------------------------------------------------------------------
# APPENDIX_ENTRIES itself
# --------------------------------------------------------------------------


class TestAppendixEntries:
    def test_exactly_five_entries(self):
        assert len(APPENDIX_ENTRIES) == 5

    def test_no_duplicate_question_ids(self):
        ids = [e.question_id for e in APPENDIX_ENTRIES]
        assert len(ids) == len(set(ids))

    def test_every_entry_has_a_nonempty_note(self):
        assert all(e.note.strip() for e in APPENDIX_ENTRIES)
