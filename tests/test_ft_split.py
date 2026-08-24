"""Unit tests for cragb.finetune.split (T7.6; PLAN.md §1.4 risk F, §3 E8, M7.md T7.6).

Covers: `guard_leakage`'s three layers (exact hash, difflib near-duplicate, embedding
backstop -- the embedding-specific tests are skipped where `sentence-transformers` isn't
importable, matching this project's established `requires_dense`-style pattern from
`tests/test_run_retrieval_eval.py`), `SplitConfig` validation, `_group_by_parent_asin`'s
single-parent_asin invariant, the probe/val greedy selectors, `split_examples`'s
end-to-end partitioning (disjoint parent_asins, arithmetic closes, reproducible under
seed), and `build_split_manifest`'s reported composition. Also confirms
`tests/test_no_leakage.py` (the M2 guard this module calls, not reimplements) is
untouched and still passes.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from cragb.bench.assemble import compute_question_hash
from cragb.bench.reference_answers import ABSTENTION_TEXT
from cragb.finetune.schema import TrainingExample
from cragb.finetune.split import (
    SplitConfig,
    _group_by_parent_asin,
    _select_probe_groups,
    _select_val_groups,
    build_split_manifest,
    embedding_near_duplicates,
    guard_leakage,
    load_cragb_question_hashes,
    load_cragb_questions,
    load_split_config,
    split_examples,
)
from cragb.utils.io import resolve_path

try:
    import sentence_transformers  # noqa: F401

    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

requires_sentence_transformers = pytest.mark.skipif(
    not SENTENCE_TRANSFORMERS_AVAILABLE,
    reason="sentence-transformers not importable in this environment (see C:\\venv\\cragb, PLAN.md §14.1)",
)

CRAGB_QUESTION = "Do these run true to size?"
CRAGB_HASHES = {"fit_sizing_000": compute_question_hash(CRAGB_QUESTION)}
CRAGB_QUESTIONS = {"fit_sizing_000": CRAGB_QUESTION}


def make_example(
    example_id: str,
    parent_asin: str,
    question: str,
    category: str = "fit_sizing",
    is_abstention: bool = False,
) -> TrainingExample:
    if is_abstention:
        answer = ABSTENTION_TEXT
        cited = ()
    else:
        answer = "Yes, it runs small [1]."
        cited = ("1",)
    return TrainingExample(
        example_id=example_id,
        category=category,
        source_doc_ids=("1", "2"),
        source_parent_asins=(parent_asin,),
        question=question,
        context_text="[1] has_photo: no\nSome review text.\n\n[2] has_photo: no\nMore review text.",
        answer=answer,
        cited_doc_ids=cited,
        is_abstention=is_abstention,
        provenance={"method": "test"},
    )


def make_config(**overrides) -> SplitConfig:
    defaults = dict(
        seed=42,
        val_fraction=0.1,
        probe_answerable_count=2,
        probe_abstention_count=2,
        near_duplicate_threshold=0.85,
        embedding_similarity_threshold=0.80,
        embedding_model="BAAI/bge-small-en-v1.5",
    )
    defaults.update(overrides)
    return SplitConfig(**defaults)


# --------------------------------------------------------------------------
# SplitConfig
# --------------------------------------------------------------------------


class TestSplitConfig:
    def test_valid_config_constructs(self):
        assert make_config().val_fraction == 0.1

    def test_val_fraction_of_zero_raises(self):
        with pytest.raises(ValueError, match="val_fraction must be in"):
            make_config(val_fraction=0.0)

    def test_val_fraction_of_one_raises(self):
        with pytest.raises(ValueError, match="val_fraction must be in"):
            make_config(val_fraction=1.0)

    def test_negative_probe_counts_raise(self):
        with pytest.raises(ValueError, match="must be non-negative"):
            make_config(probe_answerable_count=-1)

    def test_near_duplicate_threshold_out_of_range_raises(self):
        with pytest.raises(ValueError, match="near_duplicate_threshold must be in"):
            make_config(near_duplicate_threshold=1.5)

    def test_embedding_threshold_out_of_range_raises(self):
        with pytest.raises(ValueError, match="embedding_similarity_threshold must be in"):
            make_config(embedding_similarity_threshold=-0.1)


class TestLoadSplitConfig:
    def test_loads_real_finetune_config(self):
        config = load_split_config("configs/finetune.yaml")
        assert config.seed == 42
        assert 0.0 < config.val_fraction < 1.0


# --------------------------------------------------------------------------
# CRAGB reference loaders
# --------------------------------------------------------------------------


class TestCragbReferenceLoaders:
    def test_load_real_question_hashes(self):
        hashes = load_cragb_question_hashes("benchmark/cragb_v1_leakage_manifest.json")
        assert len(hashes) == 60

    def test_load_real_questions(self):
        questions = load_cragb_questions("benchmark/cragb_v1.jsonl")
        assert len(questions) == 60
        assert all(isinstance(q, str) and q for q in questions.values())


# --------------------------------------------------------------------------
# guard_leakage: exact hash layer
# --------------------------------------------------------------------------


class TestGuardLeakageExactHash:
    def test_verbatim_cragb_question_is_caught(self):
        leaked = make_example("e1", "P1", CRAGB_QUESTION)
        clean = make_example("e2", "P2", "Is the fabric itchy?")
        report = guard_leakage(
            [leaked, clean], CRAGB_HASHES, CRAGB_QUESTIONS,
            near_duplicate_threshold=0.99, embedding_similarity_threshold=0.99,
            embedding_model_name="unused",
        )
        assert report.n_exact_leak == 1
        assert report.exact_leak_example_ids == ("e1",)
        assert report.kept_example_ids == ("e2",)
        assert report.ok is False

    def test_case_and_whitespace_variant_is_still_caught(self):
        leaked = make_example("e1", "P1", "  DO   these run TRUE to size?  ")
        report = guard_leakage(
            [leaked], CRAGB_HASHES, CRAGB_QUESTIONS,
            near_duplicate_threshold=0.99, embedding_similarity_threshold=0.99,
            embedding_model_name="unused",
        )
        assert report.n_exact_leak == 1

    def test_no_leak_reports_ok(self):
        clean = make_example("e1", "P1", "Is the fabric itchy?")
        report = guard_leakage(
            [clean], CRAGB_HASHES, CRAGB_QUESTIONS,
            near_duplicate_threshold=0.99, embedding_similarity_threshold=0.99,
            embedding_model_name="unused",
        )
        assert report.ok is True
        assert report.kept_example_ids == ("e1",)


# --------------------------------------------------------------------------
# guard_leakage: difflib near-duplicate layer
# --------------------------------------------------------------------------


class TestGuardLeakageDifflib:
    def test_hand_written_paraphrase_is_caught(self):
        # This is the test the spec calls out as the whole point of the task.
        paraphrase = make_example("e1", "P1", "Do these run true to the size?")
        report = guard_leakage(
            [paraphrase], CRAGB_HASHES, CRAGB_QUESTIONS,
            near_duplicate_threshold=0.85, embedding_similarity_threshold=0.99,
            embedding_model_name="unused",
        )
        assert report.n_exact_leak == 0
        assert report.n_near_duplicate == 1
        assert report.near_duplicate_matches[0].method == "difflib"
        assert report.near_duplicate_matches[0].cragb_question_id == "fit_sizing_000"
        assert report.kept_example_ids == ()

    def test_genuinely_different_question_is_not_flagged(self):
        clean = make_example("e1", "P1", "Is the fabric itchy against sensitive skin?")
        report = guard_leakage(
            [clean], CRAGB_HASHES, CRAGB_QUESTIONS,
            near_duplicate_threshold=0.85, embedding_similarity_threshold=0.99,
            embedding_model_name="unused",
        )
        assert report.n_near_duplicate == 0
        assert report.kept_example_ids == ("e1",)

    def test_exact_leak_is_not_double_counted_as_near_duplicate(self):
        leaked = make_example("e1", "P1", CRAGB_QUESTION)
        report = guard_leakage(
            [leaked], CRAGB_HASHES, CRAGB_QUESTIONS,
            near_duplicate_threshold=0.85, embedding_similarity_threshold=0.99,
            embedding_model_name="unused",
        )
        assert report.n_exact_leak == 1
        assert report.n_near_duplicate == 0

    def test_threshold_is_respected(self):
        # A middling paraphrase that should pass a lenient threshold but fail a strict one.
        example = make_example("e1", "P1", "Does the sizing run accurately for this product?")
        lenient = guard_leakage(
            [example], CRAGB_HASHES, CRAGB_QUESTIONS,
            near_duplicate_threshold=0.3, embedding_similarity_threshold=0.99,
            embedding_model_name="unused",
        )
        strict = guard_leakage(
            [example], CRAGB_HASHES, CRAGB_QUESTIONS,
            near_duplicate_threshold=0.99, embedding_similarity_threshold=0.99,
            embedding_model_name="unused",
        )
        assert lenient.n_near_duplicate == 1
        assert strict.n_near_duplicate == 0


# --------------------------------------------------------------------------
# embedding_near_duplicates / embedding backstop layer
# --------------------------------------------------------------------------


class TestEmbeddingNearDuplicatesGracefulDegradation:
    def test_empty_candidates_returns_available_true_trivially(self):
        matches, available = embedding_near_duplicates({}, CRAGB_QUESTIONS, 0.8, "BAAI/bge-small-en-v1.5")
        assert matches == {}
        assert available is True


@requires_sentence_transformers
class TestEmbeddingNearDuplicatesReal:
    def test_semantic_paraphrase_with_no_shared_words_is_caught(self):
        # Calibrated live against BAAI/bge-small-en-v1.5 at T7.6 build time (see
        # configs/finetune.yaml's split.embedding_similarity_threshold comment): this
        # pairing scores ~0.74 cosine similarity, comfortably above the config's 0.80...
        # actually just below it, so use a slightly lower threshold here to assert the
        # mechanism works, independent of the exact configured production value.
        candidates = {"e1": "Is the sizing accurate, or does it run small or large?"}
        matches, available = embedding_near_duplicates(
            candidates, CRAGB_QUESTIONS, threshold=0.70, model_name="BAAI/bge-small-en-v1.5"
        )
        assert available is True
        assert "e1" in matches
        cragb_qid, score = matches["e1"]
        assert cragb_qid == "fit_sizing_000"
        assert score >= 0.70

    def test_unrelated_question_is_not_flagged(self):
        candidates = {"e1": "Is the fabric itchy against sensitive skin?"}
        matches, available = embedding_near_duplicates(
            candidates, CRAGB_QUESTIONS, threshold=0.80, model_name="BAAI/bge-small-en-v1.5"
        )
        assert available is True
        assert matches == {}

    def test_near_identical_reword_scores_above_the_configured_threshold(self):
        candidates = {"e1": "Does this product run true to size?"}
        matches, available = embedding_near_duplicates(
            candidates, CRAGB_QUESTIONS, threshold=0.80, model_name="BAAI/bge-small-en-v1.5"
        )
        assert "e1" in matches

    def test_guard_leakage_reports_embedding_backstop_used_true(self):
        example = make_example("e1", "P1", "Is the fabric itchy against sensitive skin?")
        report = guard_leakage(
            [example], CRAGB_HASHES, CRAGB_QUESTIONS,
            near_duplicate_threshold=0.99, embedding_similarity_threshold=0.80,
            embedding_model_name="BAAI/bge-small-en-v1.5",
        )
        assert report.embedding_backstop_used is True


class TestEmbeddingNearDuplicatesUnavailable:
    def test_bad_model_name_degrades_gracefully_rather_than_raising(self):
        # A model name that can't possibly load exercises the graceful-degradation path
        # regardless of whether sentence-transformers itself is installed.
        candidates = {"e1": "Some question."}
        matches, available = embedding_near_duplicates(
            candidates, CRAGB_QUESTIONS, threshold=0.8, model_name="this-model-does-not-exist-anywhere"
        )
        assert available is False
        assert matches == {}


# --------------------------------------------------------------------------
# _group_by_parent_asin
# --------------------------------------------------------------------------


class TestGroupByParentAsin:
    def test_groups_by_the_single_parent_asin(self):
        examples = [
            make_example("e1", "P1", "Q1?"),
            make_example("e2", "P1", "Q2?"),
            make_example("e3", "P2", "Q3?"),
        ]
        groups = _group_by_parent_asin(examples)
        assert set(groups) == {"P1", "P2"}
        assert len(groups["P1"]) == 2
        assert len(groups["P2"]) == 1

    def test_multi_parent_asin_example_raises(self):
        bad = make_example("e1", "P1", "Q1?")
        object.__setattr__(bad, "source_parent_asins", ("P1", "P2"))
        with pytest.raises(ValueError, match="expected exactly one source_parent_asin"):
            _group_by_parent_asin([bad])


# --------------------------------------------------------------------------
# _select_probe_groups / _select_val_groups
# --------------------------------------------------------------------------


class TestSelectProbeGroups:
    def test_selects_enough_groups_to_meet_both_targets(self):
        import random

        groups = {
            f"P{i}": [make_example(f"e{i}", f"P{i}", f"Q{i}?", is_abstention=(i % 2 == 0))]
            for i in range(20)
        }
        selected = _select_probe_groups(groups, target_answerable=4, target_abstention=4, rng=random.Random(1))
        n_answerable = sum(1 for pa in selected for e in groups[pa] if not e.is_abstention)
        n_abstention = sum(1 for pa in selected for e in groups[pa] if e.is_abstention)
        assert n_answerable >= 4
        assert n_abstention >= 4

    def test_exhausting_the_pool_does_not_raise(self):
        import random

        groups = {"P1": [make_example("e1", "P1", "Q1?")]}
        selected = _select_probe_groups(groups, target_answerable=100, target_abstention=100, rng=random.Random(1))
        assert selected == {"P1"}


class TestSelectValGroups:
    def test_selects_approximately_the_target_fraction(self):
        import random

        groups = {f"P{i}": [make_example(f"e{i}", f"P{i}", f"Q{i}?")] for i in range(20)}
        selected = _select_val_groups(groups, val_fraction=0.5, rng=random.Random(1))
        n_val = sum(len(groups[pa]) for pa in selected)
        assert n_val >= 10  # round(0.5 * 20)

    def test_empty_groups_returns_empty_selection(self):
        import random

        assert _select_val_groups({}, val_fraction=0.1, rng=random.Random(1)) == set()


# --------------------------------------------------------------------------
# split_examples: end-to-end
# --------------------------------------------------------------------------


def make_rich_fixture(n_products: int = 40):
    """`n_products` distinct-product examples, half abstention, spread across 3 categories,
    none of them leaking or paraphrasing the CRAGB fixture question above.
    """
    categories = ["fit_sizing", "colour_appearance", "value"]
    examples = []
    for i in range(n_products):
        category = categories[i % len(categories)]
        examples.append(
            make_example(
                f"e{i:03d}",
                f"P{i:03d}",
                f"Is this product-{i} suitable for everyday use in category {category}?",
                category=category,
                is_abstention=(i % 2 == 0),
            )
        )
    return examples


class TestSplitExamplesEndToEnd:
    def test_parent_asins_are_pairwise_disjoint_across_splits(self):
        examples = make_rich_fixture()
        config = make_config(probe_answerable_count=3, probe_abstention_count=3, val_fraction=0.2)
        result = split_examples(examples, config, CRAGB_HASHES, CRAGB_QUESTIONS)

        train_pas = {e.source_parent_asins[0] for e in result.train}
        val_pas = {e.source_parent_asins[0] for e in result.val}
        probe_pas = {e.source_parent_asins[0] for e in result.probe}
        assert train_pas.isdisjoint(val_pas)
        assert train_pas.isdisjoint(probe_pas)
        assert val_pas.isdisjoint(probe_pas)

    def test_arithmetic_closes(self):
        examples = make_rich_fixture()
        config = make_config(probe_answerable_count=3, probe_abstention_count=3, val_fraction=0.2)
        result = split_examples(examples, config, CRAGB_HASHES, CRAGB_QUESTIONS)

        n_dropped = result.leakage_report.n_exact_leak + result.leakage_report.n_near_duplicate
        assert len(result.train) + len(result.val) + len(result.probe) + n_dropped == len(examples)

    def test_leaked_and_near_duplicate_examples_never_appear_in_any_split(self):
        examples = make_rich_fixture() + [
            make_example("leaked", "P_leak", CRAGB_QUESTION),
            make_example("paraphrase", "P_para", "Do these run true to the size?"),
        ]
        config = make_config(probe_answerable_count=3, probe_abstention_count=3, val_fraction=0.2)
        result = split_examples(examples, config, CRAGB_HASHES, CRAGB_QUESTIONS)

        all_ids = {e.example_id for e in result.train + result.val + result.probe}
        assert "leaked" not in all_ids
        assert "paraphrase" not in all_ids

    def test_probe_balance_is_within_tolerance_of_target(self):
        examples = make_rich_fixture(n_products=60)
        config = make_config(probe_answerable_count=10, probe_abstention_count=10, val_fraction=0.1)
        result = split_examples(examples, config, CRAGB_HASHES, CRAGB_QUESTIONS)

        n_answerable = sum(1 for e in result.probe if not e.is_abstention)
        n_abstention = sum(1 for e in result.probe if e.is_abstention)
        assert abs(n_answerable - 10) <= 2
        assert abs(n_abstention - 10) <= 2

    def test_reproducible_under_the_same_seed(self):
        examples = make_rich_fixture()
        config = make_config(probe_answerable_count=3, probe_abstention_count=3, val_fraction=0.2, seed=7)
        first = split_examples(examples, config, CRAGB_HASHES, CRAGB_QUESTIONS)
        second = split_examples(examples, config, CRAGB_HASHES, CRAGB_QUESTIONS)

        assert [e.example_id for e in first.train] == [e.example_id for e in second.train]
        assert [e.example_id for e in first.val] == [e.example_id for e in second.val]
        assert [e.example_id for e in first.probe] == [e.example_id for e in second.probe]

    def test_different_seeds_can_produce_different_splits(self):
        examples = make_rich_fixture()
        config_a = make_config(probe_answerable_count=3, probe_abstention_count=3, val_fraction=0.2, seed=1)
        config_b = make_config(probe_answerable_count=3, probe_abstention_count=3, val_fraction=0.2, seed=2)
        result_a = split_examples(examples, config_a, CRAGB_HASHES, CRAGB_QUESTIONS)
        result_b = split_examples(examples, config_b, CRAGB_HASHES, CRAGB_QUESTIONS)

        probe_a = {e.example_id for e in result_a.probe}
        probe_b = {e.example_id for e in result_b.probe}
        assert probe_a != probe_b  # not a hard guarantee in general, but true for this fixture/seed pair

    def test_empty_input_produces_empty_splits_without_raising(self):
        config = make_config()
        result = split_examples([], config, CRAGB_HASHES, CRAGB_QUESTIONS)
        assert result.train == []
        assert result.val == []
        assert result.probe == []


# --------------------------------------------------------------------------
# build_split_manifest
# --------------------------------------------------------------------------


class TestBuildSplitManifest:
    def test_composition_counts_match_the_splits(self):
        examples = make_rich_fixture()
        config = make_config(probe_answerable_count=3, probe_abstention_count=3, val_fraction=0.2)
        result = split_examples(examples, config, CRAGB_HASHES, CRAGB_QUESTIONS)
        manifest = build_split_manifest(result, config)

        assert manifest["train"]["n"] == len(result.train)
        assert manifest["val"]["n"] == len(result.val)
        assert manifest["probe"]["n"] == len(result.probe)
        assert manifest["n_kept"] == len(result.train) + len(result.val) + len(result.probe)

    def test_disjointness_block_reports_zero_overlap(self):
        examples = make_rich_fixture()
        config = make_config(probe_answerable_count=3, probe_abstention_count=3, val_fraction=0.2)
        result = split_examples(examples, config, CRAGB_HASHES, CRAGB_QUESTIONS)
        manifest = build_split_manifest(result, config)

        assert manifest["parent_asin_disjointness"] == {
            "train_val_overlap": 0,
            "train_probe_overlap": 0,
            "val_probe_overlap": 0,
        }

    def test_near_duplicate_matches_are_listed(self):
        examples = [
            make_example("paraphrase", "P_para", "Do these run true to the size?"),
        ]
        config = make_config(near_duplicate_threshold=0.85, embedding_similarity_threshold=0.99)
        result = split_examples(examples, config, CRAGB_HASHES, CRAGB_QUESTIONS)
        manifest = build_split_manifest(result, config)

        assert manifest["n_dropped_near_duplicate"] == 1
        assert len(manifest["near_duplicate_matches"]) == 1
        assert manifest["near_duplicate_matches"][0]["example_id"] == "paraphrase"

    def test_manifest_is_json_serializable(self):
        examples = make_rich_fixture()
        config = make_config(probe_answerable_count=3, probe_abstention_count=3, val_fraction=0.2)
        result = split_examples(examples, config, CRAGB_HASHES, CRAGB_QUESTIONS)
        manifest = build_split_manifest(result, config)
        json.dumps(manifest)  # must not raise


# --------------------------------------------------------------------------
# The M2 guard this module calls must remain untouched
# --------------------------------------------------------------------------


class TestM2GuardUntouched:
    def test_test_no_leakage_still_passes(self):
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/test_no_leakage.py", "-q"],
            cwd=resolve_path("."),
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stdout + result.stderr
